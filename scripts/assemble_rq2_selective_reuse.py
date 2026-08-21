#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from configfuzz.experiment import (
    ExecutionMilestone,
    ExperimentMethod,
    ExperimentOutcome,
    ExperimentRunRecord,
)
from configfuzz.experiment_campaign import CampaignWorkload, load_campaign_workloads
from configfuzz.rq2_gpu_executor import (
    RUNTIME_INSTRUMENTATION_VERSION,
    _modification_distance,
    _run_id,
    materialize_profile,
)


# Runtime equivalence is defined by the code actually launched on the
# accelerator.  The outer campaign executor is intentionally excluded: full
# materialized configs are compared byte-for-byte before reuse, and executor
# changes such as master-port allocation do not alter workload behavior.
DEFAULT_HARNESS_PATHS = (
    "experiments/gpu/rq2_family_runner.py",
    "experiments/gpu/rq2_megatron_runner.py",
    "experiments/gpu/rq2_family_factory.py",
    "experiments/gpu/runtime_events.py",
    "experiments/gpu/launch_rq2_pytorch.sh",
    "experiments/gpu/launch_rq2_deepspeed.sh",
    "experiments/gpu/launch_rq2_accelerate.sh",
    "experiments/gpu/launch_rq2_megatron.sh",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if isinstance(value, dict):
                yield value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_assignments(value: Mapping[str, Any]) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _source_key(
    record: Mapping[str, Any],
    workloads: Mapping[str, CampaignWorkload],
) -> tuple[str, str] | None:
    workload_id = str(record.get("workload_id", ""))
    workload = workloads.get(workload_id)
    assignments = record.get("solver_modifications")
    if (
        workload is None
        or str(record.get("baseline_id", "")) != workload.baseline_id
        or not isinstance(assignments, Mapping)
    ):
        return None
    try:
        profile = materialize_profile(workload, {"assignments": assignments})
    except (KeyError, TypeError, ValueError):
        return None
    return workload_id, _stable_assignments(profile)


def _case_key(
    case: Mapping[str, Any],
    workloads: Mapping[str, CampaignWorkload],
) -> tuple[str, str]:
    workload_id = str(case["workload_id"])
    workload = workloads.get(workload_id)
    if workload is None:
        raise KeyError(f"plan references unknown workload: {workload_id}")
    profile = materialize_profile(workload, case)
    return workload_id, _stable_assignments(profile)


def _git_revision() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _harness_unchanged(
    revision: str,
    harness_paths: Sequence[str],
    cache: dict[str, bool],
) -> bool:
    if revision in cache:
        return cache[revision]
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        committed = subprocess.run(
            ["git", "diff", "--quiet", f"{revision}..HEAD", "--", *harness_paths],
            check=False,
        )
        unstaged = subprocess.run(
            ["git", "diff", "--quiet", "--", *harness_paths],
            check=False,
        )
        staged = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", *harness_paths],
            check=False,
        )
        value = committed.returncode == unstaged.returncode == staged.returncode == 0
    except OSError:
        value = False
    cache[revision] = value
    return value


def _harness_hashes(harness_paths: Sequence[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_path in harness_paths:
        path = Path(raw_path)
        if path.is_file():
            hashes[str(path)] = _sha256(path)
    return hashes


def _provenance(
    *,
    framework: str,
    plan: Path,
    workloads: Path,
    launcher: Path,
    harness_paths: Sequence[str],
    accelerator_kind: str | None,
) -> dict[str, Any]:
    return {
        "framework_id": framework,
        "runner_revision": _git_revision(),
        "plan_sha256": _sha256(plan),
        "workload_registry_sha256": _sha256(workloads),
        "launcher_sha256": _sha256(launcher),
        "runtime_instrumentation": RUNTIME_INSTRUMENTATION_VERSION,
        "runtime_harness_sha256": _harness_hashes(harness_paths),
        **({"accelerator_kind": accelerator_kind} if accelerator_kind else {}),
    }


def _eligible_source(
    record: Mapping[str, Any],
    *,
    framework: str,
    seed: int,
    launcher_sha256: str,
    harness_paths: Sequence[str],
    current_harness_hashes: Mapping[str, str],
    harness_cache: dict[str, bool],
) -> bool:
    if not bool(record.get("generated")):
        return False
    if str(record.get("outcome")) == ExperimentOutcome.INFRASTRUCTURE_FAILURE.value:
        return False
    if record.get("seed") != seed:
        return False
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        return False
    if str(metadata.get("framework_id")) != framework:
        return False
    if str(metadata.get("launcher_sha256")) != launcher_sha256:
        return False
    if str(metadata.get("runtime_instrumentation", "")) != RUNTIME_INSTRUMENTATION_VERSION:
        return False
    source_harness_hashes = metadata.get("runtime_harness_sha256")
    if isinstance(source_harness_hashes, Mapping):
        return dict(source_harness_hashes) == dict(current_harness_hashes)
    revision = str(metadata.get("runner_revision", ""))
    return bool(revision) and _harness_unchanged(revision, harness_paths, harness_cache)


def _planner_only_record(
    case: Mapping[str, Any],
    *,
    framework: str,
    seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(case.get("status", "unknown"))
    outcome = (
        ExperimentOutcome.EXPECTED_REJECTION
        if status in {"filtered", "unsat"}
        else ExperimentOutcome.UNKNOWN
    )
    record = ExperimentRunRecord(
        run_id=_run_id(framework, case, seed),
        rq="rq2",
        method=ExperimentMethod(str(case["method"])),
        workload_id=str(case["workload_id"]),
        baseline_id=str(case["baseline_id"]),
        intent_id=str(case["intent_id"]),
        intent_pool=str(case.get("intent_pool", "method_independent")),
        seed=seed,
        generated=False,
        target_value_preserved=bool(case.get("target_value_preserved", False)),
        coordinated_parameters=tuple(str(x) for x in case.get("coordinated_parameters", ())),
        modification_distance=_modification_distance(case),
        solver_seconds=float(case.get("solver_seconds", 0.0)),
        deepest_milestone=ExecutionMilestone.CONFIG_GENERATION,
        outcome=outcome,
        duration_seconds=0.0,
        gpu_seconds=0.0,
        peak_memory_mib=None,
        timed_out=False,
        constraints_exercised=tuple(str(x) for x in case.get("violated_constraints", ())),
        boundaries_exercised=(f"{case.get('target_parameter')}={case.get('target_value')!r}",),
        affected_region=tuple(str(x) for x in case.get("coordinated_parameters", ())),
        solver_modifications=(
            dict(case.get("assignments", {}))
            if isinstance(case.get("assignments"), Mapping)
            else {}
        ),
        metadata={
            **dict(provenance),
            "planner_status": status,
            "preflight": case.get("preflight"),
            "target_parameter": case.get("target_parameter"),
            "target_value": case.get("target_value"),
            "case_metadata": (
                dict(case.get("metadata", {}))
                if isinstance(case.get("metadata"), Mapping)
                else {}
            ),
            "execution_provenance": "planner_only_no_accelerator",
        },
    )
    return record.to_dict()


def _reuse_record(
    source: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    source_path: Path,
    framework: str,
    seed: int,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = copy.deepcopy(dict(source))
    source_metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
    payload.update(
        {
            "run_id": _run_id(framework, case, seed),
            "rq": "rq2",
            "method": str(case["method"]),
            "workload_id": str(case["workload_id"]),
            "baseline_id": str(case["baseline_id"]),
            "intent_id": str(case["intent_id"]),
            "intent_pool": str(case.get("intent_pool", "method_independent")),
            "seed": seed,
            "target_value_preserved": bool(case.get("target_value_preserved", False)),
            "coordinated_parameters": list(case.get("coordinated_parameters", ())),
            "modification_distance": _modification_distance(case),
            "solver_seconds": float(case.get("solver_seconds", 0.0)),
            "constraints_exercised": list(case.get("violated_constraints", ())),
            "boundaries_exercised": [
                f"{case.get('target_parameter')}={case.get('target_value')!r}"
            ],
            "affected_region": list(case.get("coordinated_parameters", ())),
            "solver_modifications": dict(case.get("assignments", {})),
            "active_constraint_ids": [],
            "constraint_provenance_ids": [],
            "metadata": {
                **dict(provenance),
                "planner_status": str(case.get("status", "unknown")),
                "preflight": case.get("preflight"),
                "target_parameter": case.get("target_parameter"),
                "target_value": case.get("target_value"),
                "case_metadata": (
                    dict(case.get("metadata", {}))
                    if isinstance(case.get("metadata"), Mapping)
                    else {}
                ),
                "runtime_reuse": {
                    "source_result": str(source_path),
                    "source_result_sha256": _sha256(source_path),
                    "source_run_id": source.get("run_id"),
                    "source_method": source.get("method"),
                    "source_intent_id": source.get("intent_id"),
                    "source_runner_revision": source_metadata.get("runner_revision"),
                    "source_config_path": source_metadata.get("config_path"),
                    "source_log_path": source_metadata.get("log_path"),
                    "rationale": (
                        "byte-equivalent canonical materialized configuration for the same "
                        "workload; same launcher, seed, framework, and unchanged execution "
                        "harness; intent labels are not consumed by the runtime runner"
                    ),
                },
            },
        }
    )
    payload.pop("campaign_test_index", None)
    payload.pop("campaign_elapsed_seconds", None)
    payload.pop("campaign_gpu_seconds", None)
    payload.pop("campaign_accelerator_seconds", None)
    ExperimentRunRecord.from_dict(payload)
    return payload


def _runtime_source_key(row: Mapping[str, Any]) -> tuple[str, str]:
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        reuse = metadata.get("runtime_reuse")
        if isinstance(reuse, Mapping):
            source_sha = str(reuse.get("source_result_sha256", ""))
            source_run = str(reuse.get("source_run_id", ""))
            if source_sha and source_run:
                return source_sha, source_run
    return "direct", str(row.get("run_id", ""))


def _refresh_campaign_fields(rows: list[dict[str, Any]]) -> None:
    # Expanded logical records keep the observed per-config runtime, while
    # cumulative campaign cost counts each physical runtime execution once.
    cumulative_accelerator = 0.0
    cumulative_elapsed = 0.0
    seen_runtime_sources: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, 1):
        if bool(row.get("generated")):
            source_key = _runtime_source_key(row)
            if source_key not in seen_runtime_sources:
                seen_runtime_sources.add(source_key)
                cumulative_elapsed += float(row.get("duration_seconds", 0.0))
                cumulative_accelerator += float(
                    row.get("accelerator_seconds", row.get("gpu_seconds", 0.0))
                )
        row["campaign_test_index"] = index
        row["campaign_elapsed_seconds"] = cumulative_elapsed
        row["campaign_gpu_seconds"] = cumulative_accelerator
        row["campaign_accelerator_seconds"] = cumulative_accelerator


def assemble(
    *,
    framework: str,
    plan_path: Path,
    workload_registry: Path,
    launcher: Path,
    source_paths: list[Path],
    output: Path,
    missing_plan: Path,
    manifest: Path,
    seed: int,
    harness_paths: Sequence[str | Path] = DEFAULT_HARNESS_PATHS,
    accelerator_kind: str | None = None,
) -> dict[str, Any]:
    plan = _read_json(plan_path)
    normalized_harness_paths = tuple(str(Path(path)) for path in harness_paths)
    missing_harness_paths = [
        path for path in normalized_harness_paths if not Path(path).is_file()
    ]
    if missing_harness_paths:
        raise FileNotFoundError(
            "runtime harness path does not exist: " + ", ".join(missing_harness_paths)
        )
    cases = [case for case in plan.get("cases", ()) if isinstance(case, Mapping)]
    workloads = load_campaign_workloads(workload_registry)
    provenance = _provenance(
        framework=framework,
        plan=plan_path,
        workloads=workload_registry,
        launcher=launcher,
        harness_paths=normalized_harness_paths,
        accelerator_kind=accelerator_kind,
    )
    launcher_sha256 = str(provenance["launcher_sha256"])
    current_harness_hashes = dict(provenance["runtime_harness_sha256"])
    harness_cache: dict[str, bool] = {}
    indexed: dict[tuple[str, str], list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    rejected_sources: Counter[str] = Counter()
    for source_path in source_paths:
        for record in _iter_jsonl(source_path):
            key = _source_key(record, workloads)
            if key is None:
                continue
            if not _eligible_source(
                record,
                framework=framework,
                seed=seed,
                launcher_sha256=launcher_sha256,
                harness_paths=normalized_harness_paths,
                current_harness_hashes=current_harness_hashes,
                harness_cache=harness_cache,
            ):
                rejected_sources[str(source_path)] += 1
                continue
            indexed[key].append((record, source_path))
            source_counts[str(source_path)] += 1

    assembled: list[dict[str, Any]] = []
    missing_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    reuse_methods: Counter[str] = Counter()
    planner_statuses: Counter[str] = Counter()
    for case in cases:
        status = str(case.get("status", "unknown"))
        planner_statuses[status] += 1
        if status != "ready":
            assembled.append(
                _planner_only_record(
                    case,
                    framework=framework,
                    seed=seed,
                    provenance=provenance,
                )
            )
            continue
        key = _case_key(case, workloads)
        matches = indexed.get(key, ())
        if matches:
            source, source_path = matches[0]
            assembled.append(
                _reuse_record(
                    source,
                    case,
                    source_path=source_path,
                    framework=framework,
                    seed=seed,
                    provenance=provenance,
                )
            )
            reuse_methods[str(source.get("method"))] += 1
        else:
            missing_by_key.setdefault(key, case)

    representative_cases = list(missing_by_key.values())
    missing_payload = copy.deepcopy(plan)
    missing_payload["cases"] = representative_cases
    missing_payload["case_count"] = len(representative_cases)
    missing_payload["intent_count"] = len({str(c["intent_id"]) for c in representative_cases})
    missing_payload["method_counts"] = dict(
        sorted(Counter(str(c["method"]) for c in representative_cases).items())
    )
    missing_payload["status_counts"] = dict(
        sorted(Counter(str(c.get("status", "unknown")) for c in representative_cases).items())
    )
    missing_payload["selective_runtime_reuse"] = {
        "source_plan_sha256": _sha256(plan_path),
        "unique_missing_runtime_configurations": len(representative_cases),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    missing_plan.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    missing_plan.write_text(
        json.dumps(missing_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    complete = len(representative_cases) == 0
    if complete:
        if len(assembled) != len(cases):
            raise AssertionError(f"assembled {len(assembled)} rows for {len(cases)} planned cases")
        _refresh_campaign_fields(assembled)
        with output.open("w", encoding="utf-8") as handle:
            for row in assembled:
                ExperimentRunRecord.from_dict(row)
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    payload = {
        "schema_version": 1,
        "name": "rq2-selective-runtime-reuse",
        "framework_id": framework,
        "complete": complete,
        "planned_cases": len(cases),
        "assembled_cases": len(assembled),
        "planner_statuses": dict(sorted(planner_statuses.items())),
        "runtime_reused_cases": sum(reuse_methods.values()),
        "reuse_source_methods": dict(sorted(reuse_methods.items())),
        "unique_missing_runtime_configurations": len(representative_cases),
        "missing_method_counts": missing_payload["method_counts"],
        "source_record_counts": dict(sorted(source_counts.items())),
        "rejected_source_records": dict(sorted(rejected_sources.items())),
        "harness_equivalence": dict(sorted(harness_cache.items())),
        "harness_paths": list(normalized_harness_paths),
        "campaign_cost_accounting": "unique_physical_runtime_execution",
        "provenance": provenance,
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "workload_registry": str(workload_registry),
        "workload_registry_sha256": _sha256(workload_registry),
        "launcher": str(launcher),
        "launcher_sha256": _sha256(launcher),
        "sources": [
            {"path": str(path), "sha256": _sha256(path)} for path in source_paths
        ],
        "missing_plan": str(missing_plan),
        "output": str(output) if complete else None,
    }
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble an RQ2 plan from provenance-checked identical runtime results and "
            "emit one representative case for each runtime configuration that still needs execution."
        )
    )
    parser.add_argument("--framework", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workloads", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--source-result", action="append", type=Path, default=[])
    parser.add_argument(
        "--harness-path",
        action="append",
        default=[],
        help="runtime harness path used to validate cross-revision equivalence; repeatable",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--missing-plan", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--accelerator-kind")
    args = parser.parse_args()
    payload = assemble(
        framework=args.framework,
        plan_path=args.plan,
        workload_registry=args.workloads,
        launcher=args.launcher.resolve(),
        source_paths=args.source_result,
        output=args.output,
        missing_plan=args.missing_plan,
        manifest=args.manifest,
        seed=args.seed,
        harness_paths=tuple(args.harness_path) or DEFAULT_HARNESS_PATHS,
        accelerator_kind=args.accelerator_kind,
    )
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
