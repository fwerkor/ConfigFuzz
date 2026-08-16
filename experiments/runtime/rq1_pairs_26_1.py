#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
QWEN_LAUNCHER = ROOT / "experiments/runtime/qwen2_26_1_baseline.sh"
TEXT_LAUNCHER = ROOT / "experiments/runtime/text_26_1_baseline.sh"
DEFAULT_RUN_ROOT = Path("/model/cyh/configfuzz-runs/rq1-pairs-26.1")


@dataclass(frozen=True)
class PairCase:
    constraint_id: str
    workload_id: str
    profile: str
    participants: tuple[str, ...]
    satisfying_args: tuple[str, ...]
    violating_args: tuple[str, ...]
    satisfying_train_iters: int = 1
    violating_train_iters: int = 1


CASES: tuple[PairCase, ...] = (
    PairCase(
        constraint_id="lmsv.task1.micro-batch-positive",
        workload_id="qwen2-text-train",
        profile="qwen2",
        participants=("training.micro_batch_size",),
        satisfying_args=("--micro-batch-size", "1"),
        violating_args=("--micro-batch-size", "0"),
    ),
    PairCase(
        constraint_id="lmsv.task1.sequence-within-position",
        workload_id="qwen2-text-train",
        profile="qwen2",
        participants=("model.seq_length", "model.max_position_embeddings"),
        satisfying_args=("--seq-length", "128", "--max-position-embeddings", "128"),
        violating_args=("--seq-length", "129", "--max-position-embeddings", "128"),
    ),
    PairCase(
        constraint_id="lmsv.task1.train-iters-after-warmup",
        workload_id="qwen2-text-train",
        profile="qwen2",
        participants=("training.train_iters", "training.lr_warmup_iters"),
        satisfying_args=("--lr-warmup-iters", "1"),
        violating_args=("--lr-warmup-iters", "1"),
        satisfying_train_iters=2,
        violating_train_iters=1,
    ),
    PairCase(
        constraint_id="lmsv.task1.warmup-before-decay",
        workload_id="qwen2-text-train",
        profile="qwen2",
        participants=("training.lr_warmup_iters", "training.train_iters", "training.lr_decay_iters"),
        satisfying_args=(
            "--lr-decay-style", "cosine", "--lr-decay-iters", "2", "--lr-warmup-iters", "1"
        ),
        violating_args=(
            "--lr-decay-style", "cosine", "--lr-decay-iters", "1", "--lr-warmup-iters", "1"
        ),
        satisfying_train_iters=2,
        violating_train_iters=2,
    ),
    PairCase(
        constraint_id="lmsv.task1.vocab-divisibility",
        workload_id="qwen2-text-train",
        profile="qwen2",
        participants=("model.vocab_size", "model.make_vocab_size_divisible_by"),
        satisfying_args=("--vocab-size", "4096", "--make-vocab-size-divisible-by", "128"),
        violating_args=("--vocab-size", "4097", "--make-vocab-size-divisible-by", "128"),
    ),
    PairCase(
        constraint_id="lmsv.task1.moe-topk-range",
        workload_id="mixtral-text-train",
        profile="mixtral",
        participants=("moe.num_experts", "moe.moe_router_topk"),
        satisfying_args=("--num-experts", "4", "--moe-router-topk", "2"),
        violating_args=("--num-experts", "4", "--moe-router-topk", "5"),
    ),
    PairCase(
        constraint_id="lmsv.task1.cosine-decay-iters",
        workload_id="qwen2-text-train",
        profile="qwen2",
        participants=("training.lr_decay_style", "training.lr_decay_iters"),
        satisfying_args=("--lr-decay-style", "cosine", "--lr-decay-iters", "2"),
        violating_args=("--lr-decay-style", "cosine", "--lr-decay-iters", "0"),
        satisfying_train_iters=2,
        violating_train_iters=2,
    ),
    PairCase(
        constraint_id="lmsv.task1.ffn-hidden-size-cap",
        workload_id="qwen2-text-train",
        profile="qwen2",
        participants=("model.hidden_size", "model.ffn_hidden_size"),
        satisfying_args=("--hidden-size", "512", "--ffn-hidden-size", "3072"),
        violating_args=("--hidden-size", "512", "--ffn-hidden-size", "4096"),
    ),
    PairCase(
        constraint_id="lmsv.task1.swiglu-ffn-divisibility",
        workload_id="qwen2-text-train",
        profile="qwen2",
        participants=("model.swiglu", "model.ffn_hidden_size", "parallel.tensor_model_parallel_size"),
        satisfying_args=("--ffn-hidden-size", "1024"),
        violating_args=("--ffn-hidden-size", "1025"),
    ),
    PairCase(
        constraint_id="lmsv.task1.moe-topk-one-pre-softmax",
        workload_id="mixtral-text-train",
        profile="mixtral",
        participants=("moe.moe_router_topk", "moe.moe_router_pre_softmax"),
        satisfying_args=("--moe-router-topk", "1", "--moe-router-pre-softmax"),
        violating_args=("--moe-router-topk", "1"),
    ),
)


ERROR_PATTERNS = (
    re.compile(r"AssertionError:\s*(.+)"),
    re.compile(r"ValueError:\s*(.+)"),
    re.compile(r"RuntimeError:\s*(.+)"),
)


def deepest_milestone(text: str, returncode: int) -> str:
    lowered = text.lower()
    if re.search(r"iteration\s+\d+/\s*\d+", text):
        return "optimizer_step"
    # Optimizer/scheduler construction happens after model construction but before the
    # first forward. Generic log text contains many "forward" tokens, so recognize
    # this assertion before the forward-path heuristics below.
    if "lr_warmup_steps < self.lr_decay_steps" in text or "assert self.lr_decay_steps > 0" in text:
        return "model_construction"
    if "bias_swiglu" in lowered and "must match the size of tensor" in lowered:
        return "model_construction"
    if any(token in lowered for token in ("aclnntopk", "torch.topk", "selected index k out of range")):
        return "forward"
    if "building gpt model" in lowered or "number of parameters" in lowered:
        return "model_construction"
    if "initialize distributed" in lowered or "world size:" in lowered:
        return "process_group_initialization"
    if "arguments" in lowered or returncode != 0:
        return "config_validation"
    return "argument_parsing"


def first_failure_milestone(text: str, deepest: str, returncode: int) -> str | None:
    if returncode == 0:
        return None
    lowered = text.lower()
    if "lr_warmup_steps < self.lr_decay_steps" in text or "assert self.lr_decay_steps > 0" in text:
        return "model_construction"
    if "bias_swiglu" in lowered and "must match the size of tensor" in lowered:
        return "model_construction"
    if any(token in lowered for token in ("aclnntopk", "torch.topk", "selected index k out of range")):
        return "forward"
    # These are direct native argument checks even though this stack initializes
    # distributed state before finishing argument validation.
    if "assert args.micro_batch_size > 0" in text or "max_position_embeddings" in lowered:
        return "config_validation"
    if "building gpt model" in lowered or "number of parameters" in lowered:
        return "model_construction"
    if "initialize distributed" in lowered or "world size:" in lowered:
        return "process_group_initialization"
    return "config_validation"


def failure_message(text: str) -> str | None:
    matches: list[str] = []
    for pattern in ERROR_PATTERNS:
        matches.extend(m.group(1).strip() for m in pattern.finditer(text) if m.group(1).strip())
    if matches:
        return matches[-1]
    if "assert args.micro_batch_size > 0" in text:
        return "micro_batch_size must be greater than 0"
    if "lr_warmup_steps < self.lr_decay_steps" in text:
        return "lr_warmup_steps must be smaller than lr_decay_steps"
    if "assert self.lr_decay_steps > 0" in text:
        return "lr_decay_steps must be greater than 0"
    return None


def interpretable(message: str | None, participants: Sequence[str]) -> bool | None:
    if message is None:
        return None
    normalized = message.lower().replace("-", "_")
    leaves = [p.rsplit(".", 1)[-1].lower() for p in participants]
    if any(leaf in normalized for leaf in leaves):
        return True
    return any(token in normalized for token in ("topk", "warmup", "decay", "position", "micro batch", "micro_batch"))


def peak_memory_mib(text: str) -> float | None:
    values = [
        float(m.group(1))
        for m in re.finditer(r"max memory allocated:\s*([0-9.]+)\s*MB", text, re.IGNORECASE)
    ]
    return max(values) if values else None


def run_member(
    case: PairCase,
    role: str,
    *,
    device: int,
    port: int,
    run_root: Path,
    timeout_seconds: float,
) -> dict[str, object]:
    args = case.satisfying_args if role == "satisfying" else case.violating_args
    train_iters = case.satisfying_train_iters if role == "satisfying" else case.violating_train_iters
    member_root = run_root / case.constraint_id / role
    member_root.mkdir(parents=True, exist_ok=True)
    launcher = QWEN_LAUNCHER if case.profile == "qwen2" else TEXT_LAUNCHER
    env = os.environ.copy()
    env.update(
        {
            "ASCEND_RT_VISIBLE_DEVICES": str(device),
            "NPROC": "1",
            "TP": "1",
            "PP": "1",
            "MASTER_PORT": str(port),
            "OUT_ROOT": str(member_root),
            "INITIAL_TRAIN_ITERS": str(train_iters),
            "SKIP_RELOAD": "1",
            "SAVE_CHECKPOINTS": "0",
            "REUSE_EXISTING": "0",
        }
    )
    if case.profile != "qwen2":
        env["PROFILE"] = case.profile
    command = ["bash", str(launcher), *args]
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
    else:
        outcome = "expected_rejection" if role == "violating" else "unexplained_failure"
        failure_mode = "explicit_rejection" if message else "crash"
    record = {
        "run_id": f"rq1-26.1-{case.constraint_id}-{role}",
        "rq": "rq1",
        "method": "raw_mutation",
        "workload_id": case.workload_id,
        "baseline_id": f"{case.workload_id}-26.1-single-npu",
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
        "gpu_seconds": duration,
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
        "topologies": ["single_npu"],
        "feature_interactions": [],
        "backend_paths": [case.profile],
        "metadata": {
            "framework_line": "MindSpeed-LLM v26.1.0 / MindSpeed 26.1.0_core_r0.12.1 / MCore v0.12.1",
            "device": device,
            "command": command,
            "participant_overrides": list(args),
            "returncode": returncode,
            "error_message": message,
            "run_root": str(member_root),
        },
    }
    (member_root / "record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run controlled single-NPU RQ1 constraint pairs on the stable 26.1 stack.")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/rq1.26.1.single-npu.jsonl")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--base-port", type=int, default=6100)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--constraint", action="append", dest="constraints")
    args = parser.parse_args()

    selected = [c for c in CASES if not args.constraints or c.constraint_id in set(args.constraints)]
    if not selected:
        raise SystemExit("no matching RQ1 cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    port = args.base_port
    for case in selected:
        for role in ("satisfying", "violating"):
            print(f"RUN {case.constraint_id} {role}", flush=True)
            record = run_member(
                case,
                role,
                device=args.device,
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
    args.output.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    print(f"WROTE {len(records)} records -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
