#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence


def _split_control_and_megatron_argv(argv: Sequence[str]) -> tuple[list[str], list[str]]:
    try:
        sep = argv.index("--")
    except ValueError as exc:
        raise SystemExit("persistent worker requires '--' before the MindSpeed/Megatron arguments") from exc
    return list(argv[:sep]), list(argv[sep + 1 :])


def _parse_control(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute many same-topology MindSpeed cases inside one Python/NPU process so imports, "
            "CANN initialization, and the distributed process group are amortized across cases."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-log-root", type=Path, required=True)
    parser.add_argument("--warmup-train-iters", type=int, default=1)
    parser.add_argument(
        "--allow-multi-rank-experimental",
        action="store_true",
        help="Allow WORLD_SIZE>1. Fault isolation for rank-asymmetric NPU failures is still experimental.",
    )
    return parser.parse_args(argv)


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(cases, list):
        raise ValueError("persistent-worker manifest must contain a cases array")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(cases):
        if not isinstance(raw, dict):
            raise ValueError(f"manifest case {index} must be an object")
        case_id = str(raw.get("case_id") or f"case-{index:05d}")
        overrides = raw.get("args", [])
        if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
            raise ValueError(f"manifest case {case_id}: args must be a string array")
        train_iters = int(raw.get("train_iters", 1))
        normalized.append({"case_id": case_id, "args": list(overrides), "train_iters": train_iters})
    return normalized


@contextlib.contextmanager
def _temporary_argv(args: Sequence[str]):
    old = sys.argv
    sys.argv = [old[0], *args]
    try:
        yield
    finally:
        sys.argv = old


def _topology_signature(args: Any) -> tuple[int, ...]:
    return (
        int(args.world_size),
        int(args.tensor_model_parallel_size),
        int(args.pipeline_model_parallel_size),
        int(getattr(args, "context_parallel_size", 1)),
        int(getattr(args, "expert_model_parallel_size", 1)),
        int(getattr(args, "encoder_tensor_model_parallel_size", 0) or 0),
        int(getattr(args, "encoder_pipeline_model_parallel_size", 0) or 0),
    )


def _exception_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


def _likely_poisoned_runtime(text: str) -> bool:
    lowered = text.lower()
    tokens = (
        "acl_error",
        "hccl",
        "hccp",
        "device error",
        "npu out of memory",
        "alloc device memory failed",
        "rtdevice",
        "ez9999",
        "force stop",
        "connection reset",
    )
    return any(token in lowered for token in tokens)


def main() -> int:
    control_argv, base_megatron_args = _split_control_and_megatron_argv(sys.argv[1:])
    control = _parse_control(control_argv)
    cases = _load_manifest(control.manifest)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size > 1 and not control.allow_multi_rank_experimental:
        raise SystemExit(
            "persistent worker currently defaults to WORLD_SIZE=1 for fault isolation; "
            "pass --allow-multi-rank-experimental only for controlled same-topology pilots"
        )

    # Import the full stack exactly once. pretrain_gpt imports the MindSpeed adaptor,
    # which installs the same Megatron patches as the normal launcher.
    import torch
    import torch_npu
    import pretrain_gpt
    import megatron.training.arguments as megatron_arguments
    from megatron.core.enums import ModelType
    from megatron.training import get_args, get_timers
    from megatron.training.global_vars import set_global_variables, unset_global_variables
    from megatron.training.initialize import _set_random_seed
    from mindspeed_llm.training.training import build_train_args, set_jit_fusion_options, train

    # pretrain_gpt imported megatron_adaptor above, so Megatron's parse_args is
    # already decorated with MindSpeed-LLM feature arguments. Decorating it a
    # second time creates duplicate argparse options.
    parse_args = megatron_arguments.parse_args
    validate_args = megatron_arguments.validate_args

    # One known-good initialization establishes CANN, the device context, global
    # Megatron state, and the distributed/HCCL process group. No model is built here.
    warm_cli = [*base_megatron_args, "--train-iters", str(control.warmup_train_iters)]
    warm_started = time.monotonic()
    with _temporary_argv(warm_cli):
        from megatron.training.initialize import initialize_megatron

        initialize_megatron()
    set_jit_fusion_options()
    torch.distributed.barrier()
    warm_seconds = time.monotonic() - warm_started
    warm_args = get_args()
    warm_topology = _topology_signature(warm_args)

    pretrain_gpt.train_valid_test_datasets_provider.is_distributed = True
    control.case_log_root.mkdir(parents=True, exist_ok=True)
    control.output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    worker_poisoned = False

    for index, case in enumerate(cases, 1):
        case_id = case["case_id"]
        case_log = control.case_log_root / f"{index:05d}-{case_id}.log"
        started = time.monotonic()
        deepest = "argument_parsing"
        outcome = "unknown"
        returncode = 0
        error: str | None = None
        topology: tuple[int, ...] | None = None
        stream = io.StringIO()
        train_args = None
        model = None
        optimizer = None
        opt_param_scheduler = None
        train_data_iterator = None
        valid_data_iterator = None
        config = None

        try:
            case_cli = [
                *base_megatron_args,
                "--train-iters",
                str(case["train_iters"]),
                *case["args"],
            ]
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                with _temporary_argv(case_cli):
                    fresh_args = parse_args(None, False)
                deepest = "config_validation"
                fresh_args = validate_args(fresh_args, {})
                topology = _topology_signature(fresh_args)
                if topology != warm_topology:
                    raise RuntimeError(
                        "PERSISTENT_WORKER_TOPOLOGY_CHANGE: "
                        f"warm={warm_topology}, case={topology}; start a separate worker for this topology"
                    )

                # Recreate the ordinary per-run Megatron globals (args, tokenizer,
                # timers, microbatch calculator) while deliberately retaining the
                # expensive distributed/HCCL and model-parallel process groups.
                unset_global_variables()
                set_global_variables(fresh_args)
                _set_random_seed(fresh_args.seed, fresh_args.data_parallel_random_init)

                # The persistent launcher deliberately omits --save, so ordinary
                # milestone-validation cases do not pay per-case checkpoint I/O. If a
                # future test explicitly supplies --save/--load, preserve that intent.

                if hasattr(torch_npu.npu, "reset_peak_memory_stats"):
                    torch_npu.npu.reset_peak_memory_stats()

                deepest = "model_construction"
                timers = get_timers()
                app_metrics: dict[str, Any] = {}
                train_args, _ = build_train_args(
                    fresh_args,
                    timers,
                    pretrain_gpt.train_valid_test_datasets_provider,
                    pretrain_gpt.model_provider,
                    ModelType.encoder_or_decoder,
                    pretrain_gpt.forward_step,
                    None,
                    app_metrics,
                )
                (
                    forward_step_func,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    train_data_iterator,
                    valid_data_iterator,
                    process_non_loss_data_func,
                    config,
                ) = train_args

                if not fresh_args.do_train or fresh_args.train_iters <= 0:
                    raise RuntimeError("persistent worker case produced no training data")

                deepest = "forward"
                iteration, _ = train(
                    forward_step_func,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    train_data_iterator,
                    valid_data_iterator,
                    process_non_loss_data_func,
                    config,
                )
                if iteration >= fresh_args.train_iters:
                    deepest = "optimizer_step"
                outcome = "valid"
                torch_npu.npu.synchronize()

        except BaseException as exc:  # The worker must record malformed fuzz cases, not crash silently.
            returncode = 1
            error = _exception_text(exc)
            stream.write(error)
            if "PERSISTENT_WORKER_TOPOLOGY_CHANGE" in error:
                outcome = "requires_worker_restart"
            else:
                outcome = "rejected_or_failed"
            worker_poisoned = _likely_poisoned_runtime(error)
        finally:
            # Drop model/optimizer/data references before the next case. The NPU device
            # context and process group intentionally stay alive.
            train_args = None
            model = None
            optimizer = None
            opt_param_scheduler = None
            train_data_iterator = None
            valid_data_iterator = None
            config = None
            gc.collect()
            if not worker_poisoned:
                try:
                    torch_npu.npu.empty_cache()
                except Exception:
                    pass

        duration = time.monotonic() - started
        log_text = stream.getvalue()
        if rank == 0:
            case_log.write_text(log_text, encoding="utf-8", errors="replace")
            records.append(
                {
                    "case_id": case_id,
                    "case_index": index,
                    "duration_seconds": duration,
                    "deepest_milestone": deepest,
                    "outcome": outcome,
                    "returncode": returncode,
                    "error": error.splitlines()[-1] if error else None,
                    "topology": list(topology) if topology is not None else None,
                    "worker_warm_seconds": warm_seconds,
                    "worker_reused": True,
                    "worker_poisoned": worker_poisoned,
                }
            )
            control.output.write_text(
                "".join(json.dumps(item, sort_keys=True) + "\n" for item in records),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "case_id": case_id,
                        "duration_seconds": round(duration, 4),
                        "deepest_milestone": deepest,
                        "outcome": outcome,
                        "worker_poisoned": worker_poisoned,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if worker_poisoned:
            # A device/HCCL failure may leave the context unsafe for subsequent cases.
            # Let an outer batch supervisor restart this worker; successful earlier
            # cases still benefited from the amortized startup.
            break

    if rank == 0:
        print(
            json.dumps(
                {
                    "worker_warm_seconds": warm_seconds,
                    "cases_requested": len(cases),
                    "cases_completed": len(records),
                    "worker_poisoned": worker_poisoned,
                    "output": str(control.output),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 70 if worker_poisoned else 0


if __name__ == "__main__":
    raise SystemExit(main())
