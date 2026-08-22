#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from rq1_pairs_26_1 import (
    CASES,
    PairCase,
    ROOT,
    deepest_milestone,
    failure_message,
    first_failure_milestone,
    interpretable,
    peak_memory_mib,
)


LAUNCHER = ROOT / "experiments/runtime/mindspeed_msa_26_1_baseline.sh"
DEFAULT_RUN_ROOT = Path("/model/cyh/configfuzz-runs/rq1-msa-26.1-dual-npu")


def _constraint_rejection(text: str, message: str | None, case: PairCase) -> bool:
    if interpretable(message, case.participants):
        return True
    lowered = text.lower()
    constraint_tokens = {
        "lmsv.task1.swiglu-ffn-divisibility": ("swiglu", "size of tensor", "ffn"),
        "lmsv.task1.moe-topk-range": ("topk", "top k", "selected index k", "num_experts"),
        "lmsv.task1.train-iters-after-warmup": ("warmup", "train_iters", "train iters"),
        "lmsv.task1.warmup-before-decay": ("warmup", "decay"),
        "lmsv.task1.cosine-decay-iters": ("decay", "lr_decay"),
        "lmsv.task1.sequence-within-position": ("position", "seq_length", "seq length"),
        "lmsv.task1.micro-batch-positive": ("micro_batch", "micro batch"),
    }
    return any(token in lowered for token in constraint_tokens.get(case.constraint_id, ()))


def _infrastructure_failure(text: str, message: str | None) -> bool:
    lowered = (text + "\n" + (message or "")).lower()
    tokens = (
        "hccp_process_initialization_failure",
        "failed to init communicator",
        "communicator of group hccl_world_group inited: failed",
        "address already been bound",
        "failed to enable listening for the host network adapter socket",
        "open tsd error",
        "maybe the last training process is running",
    )
    return any(token in lowered for token in tokens)


def run_member(
    case: PairCase,
    role: str,
    *,
    devices: tuple[int, int],
    port: int,
    run_root: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    overrides = case.satisfying_args if role == "satisfying" else case.violating_args
    train_iters = case.satisfying_train_iters if role == "satisfying" else case.violating_train_iters
    member_root = run_root / case.constraint_id / role
    member_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": ",".join(str(device) for device in devices),
            "NPROC": "2",
            "TP": "1",
            "PP": "1",
            "MASTER_PORT": str(port),
            "OUT_ROOT": str(member_root),
            "INITIAL_TRAIN_ITERS": str(train_iters),
            "PROFILE": case.profile,
        }
    )
    command = ["bash", str(LAUNCHER), *overrides]
    started = time.monotonic()
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        output = completed.stdout or ""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        returncode = 124

    duration = time.monotonic() - started
    (member_root / "runner.log").write_text(output, encoding="utf-8")
    deepest = deepest_milestone(output, returncode)
    failed_at = first_failure_milestone(output, deepest, returncode)
    message = failure_message(output)

    if timed_out:
        outcome = "infrastructure_failure"
        failure_mode = "timeout"
    elif returncode == 0:
        outcome = "valid"
        failure_mode = "none"
    elif _infrastructure_failure(output, message):
        outcome = "infrastructure_failure"
        failure_mode = "distributed_runtime"
    elif role == "violating" and _constraint_rejection(output, message, case):
        outcome = "expected_rejection"
        failure_mode = "explicit_rejection" if message else "constraint_failure"
    else:
        outcome = "unexplained_failure"
        failure_mode = "crash" if message is None else "unattributed_rejection"

    record: dict[str, object] = {
        "run_id": f"rq1-msa-26.1-{case.constraint_id}-{role}",
        "rq": "rq1",
        "framework_id": "msa",
        "method": "raw_mutation",
        "workload_id": case.workload_id,
        "baseline_id": f"{case.workload_id}-msa-26.1-dual-npu",
        "intent_id": None,
        "seed": 42,
        "generated": True,
        "target_value_preserved": True,
        "coordinated_parameters": list(case.participants),
        "modification_distance": None,
        "solver_seconds": 0.0,
        "deepest_milestone": deepest,
        "outcome": outcome,
        "duration_seconds": duration,
        "gpu_seconds": duration * len(devices),
        "peak_memory_mib": peak_memory_mib(output),
        "timed_out": timed_out,
        "constraint_id": case.constraint_id,
        "pair_role": role,
        "first_failure_milestone": failed_at,
        "failure_mode": failure_mode,
        "error_message_interpretable": interpretable(message, case.participants),
        "constraints_exercised": [case.constraint_id],
        "boundaries_exercised": [],
        "guard_transitions": [],
        "topologies": ["world=2,tp=1,pp=1,cp=1,ep=1"],
        "feature_interactions": [],
        "backend_paths": ["mindspore", case.profile],
        "metadata": {
            "framework_line": "MSA / MindSpore 2.10.0 / MSAdapter 0.7.0 / MindSpeed-LLM v26.1.0 / CANN 9.1.0",
            "devices": list(devices),
            "device_count": len(devices),
            "command": command,
            "participant_overrides": list(overrides),
            "returncode": returncode,
            "error_message": message,
            "run_root": str(member_root),
        },
    }
    (member_root / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled two-NPU RQ1 pairs on the MindSpore/MSAdapter backend.")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/rq1.msa.26.1.dual-npu.jsonl")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--devices", default="0,1", help="exactly two comma-separated NPU device IDs")
    parser.add_argument("--base-port", type=int, default=6710)
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--constraint", action="append", dest="constraints")
    parser.add_argument(
        "--cooldown-seconds",
        type=float,
        default=15.0,
        help="delay between msrun members so Ascend HCCP/TSD state can be released",
    )
    args = parser.parse_args()

    devices = tuple(int(item.strip()) for item in args.devices.split(",") if item.strip())
    if len(devices) != 2 or len(set(devices)) != 2:
        parser.error("--devices must name exactly two distinct NPU device IDs")

    selected = [case for case in CASES if not args.constraints or case.constraint_id in set(args.constraints)]
    if not selected:
        raise SystemExit("no matching RQ1 cases")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    members = [(case, role) for case in selected for role in ("satisfying", "violating")]
    port = args.base_port
    for member_index, (case, role) in enumerate(members):
        print(f"RUN framework=msa {case.constraint_id} {role}", flush=True)
        record = run_member(
            case,
            role,
            devices=devices,
            port=port,
            run_root=args.run_root,
            timeout_seconds=args.timeout_seconds,
        )
        records.append(record)
        print(
            f"  outcome={record['outcome']} deepest={record['deepest_milestone']} "
            f"failure={record['first_failure_milestone']} duration={record['duration_seconds']:.3f}s",
            flush=True,
        )
        port += 1
        if member_index + 1 < len(members) and args.cooldown_seconds > 0:
            print(f"  cooldown={args.cooldown_seconds:.1f}s", flush=True)
            time.sleep(args.cooldown_seconds)

    args.output.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    print(f"WROTE {len(records)} records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
