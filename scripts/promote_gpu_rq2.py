#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from configfuzz.dependencies import DependencyGraph
from configfuzz.rq2_graph import merge_dependency_graphs, rename_dependency_graph_parameters


SUBJECTS = {
    "pytorch-cuda": {
        "rq1_subject": "pytorch-native",
        "rq1_file": "pytorch-native.json",
        "static_artifact": "artifacts/frameworks/pytorch_v2.13.0.json",
        "qualification_prefix": "pytorch-cuda-",
        "launcher": "experiments/gpu/launch_rq2_pytorch.sh",
        "core_aliases": {},
        "uses_hf_model_graph": True,
    },
    "deepspeed": {
        "rq1_subject": "deepspeed",
        "rq1_file": "deepspeed.json",
        "static_artifact": "artifacts/frameworks/deepspeed_v0.19.1.json",
        "qualification_prefix": "deepspeed-",
        "launcher": "experiments/gpu/launch_rq2_deepspeed.sh",
        "core_aliases": {
            "train_micro_batch_size_per_gpu": "training.micro_batch_size",
            "gradient_accumulation_steps": "training.gradient_accumulation_steps",
            "bf16": "precision.bf16",
            "fp16": "precision.fp16",
        },
        "uses_hf_model_graph": True,
    },
    "transformers-accelerate": {
        "rq1_subject": "transformers-accelerate",
        "rq1_file": "transformers-accelerate.json",
        "static_artifact": "artifacts/frameworks/transformers_v5.9.0_accelerate_v1.14.0.json",
        "qualification_prefix": "transformers-accelerate-",
        "launcher": "experiments/gpu/launch_rq2_accelerate.sh",
        "core_aliases": {
            "gradient_accumulation_steps": "training.gradient_accumulation_steps",
            "bf16": "precision.bf16",
            "fp16": "precision.fp16",
        },
        "uses_hf_model_graph": True,
    },
    "megatron-core": {
        "rq1_subject": "megatron-core",
        "rq1_file": "megatron-core.json",
        "static_artifact": "artifacts/frameworks/megatron_core_v0.18.2.json",
        "qualification_prefix": "megatron-core-",
        "launcher": "experiments/gpu/launch_rq2_megatron.sh",
        "core_aliases": {
            "hidden_size": "model.hidden_size",
            "ffn_hidden_size": "model.ffn_hidden_size",
            "num_layers": "model.num_layers",
            "num_attention_heads": "model.num_attention_heads",
            "num_query_groups": "model.num_query_groups",
            "num_moe_experts": "moe.num_experts",
            "micro_batch_size": "training.micro_batch_size",
            "global_batch_size": "training.global_batch_size",
            "bf16": "precision.bf16",
            "fp16": "precision.fp16",
            "tensor_model_parallel_size": "parallel.tensor_model_parallel_size",
            "pipeline_model_parallel_size": "parallel.pipeline_model_parallel_size",
            "context_parallel_size": "parallel.context_parallel_size",
            "expert_model_parallel_size": "parallel.expert_model_parallel_size",
            "sequence_parallel": "parallel.sequence_parallel",
            "recompute_granularity": "training.recompute_granularity",
        },
        "uses_hf_model_graph": False,
    },
}

COMMON_TEXT_ALIASES = {
    "hidden_size": "model.hidden_size",
    "intermediate_size": "model.ffn_hidden_size",
    "num_hidden_layers": "model.num_layers",
    "num_attention_heads": "model.num_attention_heads",
    "num_key_value_heads": "model.num_query_groups",
    "max_position_embeddings": "model.max_position_embeddings",
    "vocab_size": "model.vocab_size",
    "attention_dropout": "model.attention_dropout",
}

MODEL_GRAPHS: dict[str, tuple[tuple[str, Mapping[str, str]], ...]] = {
    "qwen2-train": (("qwen2.json", COMMON_TEXT_ALIASES),),
    "llama2-train": (("llama2.json", COMMON_TEXT_ALIASES),),
    "chatglm3-train": (("chatglm3.json", COMMON_TEXT_ALIASES),),
    "mixtral-train": (
        (
            "mixtral.json",
            {
                **COMMON_TEXT_ALIASES,
                "num_local_experts": "moe.num_experts",
                "num_experts_per_tok": "moe.moe_router_topk",
            },
        ),
    ),
    "deepseekv3-train": (
        (
            "deepseekv3.json",
            {
                **COMMON_TEXT_ALIASES,
                "n_routed_experts": "moe.num_experts",
                "n_shared_experts": "moe.n_shared_experts",
                "n_group": "moe.moe_router_num_groups",
                "topk_group": "moe.topk_group",
                "num_experts_per_tok": "moe.moe_router_topk",
                "moe_intermediate_size": "moe.moe_intermediate_size",
                "kv_lora_rank": "mla.kv_lora_rank",
                "q_lora_rank": "mla.q_lora_rank",
                "qk_rope_head_dim": "mla.qk_rope_head_dim",
                "qk_nope_head_dim": "mla.qk_nope_head_dim",
                "v_head_dim": "mla.v_head_dim",
            },
        ),
    ),
    "internvl3-train": (
        ("internvl3-text.json", COMMON_TEXT_ALIASES),
        (
            "internvl3-vision.json",
            {
                "hidden_size": "multimodal.vision_hidden_size",
                "intermediate_size": "multimodal.vision_ffn_hidden_size",
                "num_hidden_layers": "multimodal.vision_num_layers",
                "num_attention_heads": "multimodal.vision_num_attention_heads",
                "image_size": "multimodal.image_size",
                "patch_size": "multimodal.patch_size",
                "num_channels": "multimodal.num_channels",
                "image_seq_length": "multimodal.image_seq_length",
                "downsample_ratio": "multimodal.downsample_ratio",
            },
        ),
    ),
    "cogvideox-train": (
        (
            "cogvideox.json",
            {
                "time_embed_dim": "model.hidden_size",
                "num_attention_heads": "model.num_attention_heads",
                "num_layers": "model.num_layers",
                "attention_head_dim": "video.attention_head_dim",
                "in_channels": "video.in_channels",
                "out_channels": "video.out_channels",
                "sample_width": "video.sample_width",
                "sample_height": "video.sample_height",
                "sample_frames": "video.frames",
                "patch_size": "video.patch_size",
                "temporal_compression_ratio": "video.temporal_compression_ratio",
                "max_text_seq_length": "video.max_text_seq_length",
                "text_embed_dim": "video.text_embed_dim",
            },
        ),
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_mapping(path: Path) -> Mapping[str, Any]:
    if path.suffix in {".yaml", ".yml"}:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"object required: {path}")
    return value


def _latest_qualification(root: Path, prefix: str) -> Path:
    matches = sorted(path for path in root.glob(f"{prefix}*") if path.is_dir())
    if not matches:
        raise FileNotFoundError(f"no qualification directory matching {prefix!r} in {root}")
    return matches[-1]


def _read_qualification(path: Path) -> list[dict[str, Any]]:
    summary = path / "summary.tsv"
    if not summary.is_file():
        raise FileNotFoundError(summary)
    rows: list[dict[str, Any]] = []
    for line in summary.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        workload, status, duration, milestone = line.split("\t")
        rows.append(
            {
                "workload_id": workload,
                "status": status,
                "duration_seconds": float(duration),
                "deepest_milestone": milestone,
            }
        )
    return rows


def _rel(path: Path, base: Path) -> str:
    return Path(os.path.relpath(path, base)).as_posix()


def _graph_from_payload(payload: Mapping[str, Any]) -> DependencyGraph:
    graph = payload.get("dependency_graph", payload)
    if not isinstance(graph, Mapping):
        raise ValueError("dependency graph object required")
    return DependencyGraph.from_dict(graph)


def _model_graph(repo: Path, workload_id: str) -> DependencyGraph:
    pieces: list[DependencyGraph] = []
    for index, (filename, aliases) in enumerate(MODEL_GRAPHS[workload_id], 1):
        source = repo / "artifacts/rq2_models" / filename
        payload = _load_mapping(source)
        pieces.append(
            rename_dependency_graph_parameters(
                _graph_from_payload(payload),
                aliases,
                edge_id_prefix=f"model:{workload_id}:{index}",
                metadata={"workload_id": workload_id, "source_artifact": str(source.relative_to(repo))},
            )
        )
    return merge_dependency_graphs(*pieces, metadata={"workload_id": workload_id, "source": "workload_model_stack"})


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote qualified GPU RQ2 subjects using framework and workload-scoped graphs.")
    parser.add_argument("--rq1-results-dir", type=Path, required=True)
    parser.add_argument("--qualification-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, default=Path("experiments/rq2/framework_workload_matrix.yaml"))
    parser.add_argument("--workloads", type=Path, default=Path("experiments/rq2/workloads.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("experiments/rq2/promoted/gpu"))
    args = parser.parse_args()

    repo = Path.cwd().resolve()
    matrix_path = (repo / args.matrix).resolve()
    workloads_path = (repo / args.workloads).resolve()
    matrix = _load_mapping(matrix_path)
    workload_template = _load_mapping(workloads_path)
    template_by_id = {
        str(item["workload_id"]): item
        for item in workload_template.get("workloads", ())
        if isinstance(item, Mapping)
    }
    supported: dict[str, list[str]] = {}
    for binding in matrix.get("bindings", ()):
        if not isinstance(binding, Mapping) or not bool(binding.get("formal_rq2")):
            continue
        framework = str(binding["framework_id"])
        if framework in SUBJECTS:
            supported.setdefault(framework, []).append(str(binding["workload_id"]))

    output_root = (repo / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    rq1_root = (repo / args.rq1_results_dir).resolve()
    qualification_root = (repo / args.qualification_root).resolve()
    summary_subjects: list[dict[str, Any]] = []

    for framework, spec in SUBJECTS.items():
        expected = sorted(supported.get(framework, ()))
        if not expected:
            raise ValueError(f"matrix has no formal workloads for {framework}")
        rq1_path = rq1_root / str(spec["rq1_file"])
        static_path = repo / str(spec["static_artifact"])
        rq1 = _load_mapping(rq1_path)
        static_artifact = _load_mapping(static_path)
        if str(rq1.get("subject")) != spec["rq1_subject"]:
            raise ValueError(f"RQ1 subject mismatch for {framework}")
        static_core = rename_dependency_graph_parameters(
            _graph_from_payload(static_artifact),
            spec["core_aliases"],
            metadata={"framework_id": framework, "validation_state": "pre_execution"},
        )
        validated_core = rename_dependency_graph_parameters(
            _graph_from_payload(rq1),
            spec["core_aliases"],
            metadata={"framework_id": framework, "validation_state": "rq1_feedback_applied"},
        )

        qualification_dir = _latest_qualification(qualification_root, str(spec["qualification_prefix"]))
        qualification = _read_qualification(qualification_dir)
        observed = sorted(row["workload_id"] for row in qualification if row["status"] == "PASS" and row["deepest_milestone"] == "completed")
        if observed != expected:
            raise ValueError(f"qualification mismatch for {framework}: expected {expected}, got {observed}")

        subject_root = output_root / framework
        graph_root = subject_root / "graphs"
        graph_root.mkdir(parents=True, exist_ok=True)
        validator_out = subject_root / "native_validator.json"
        validator_out.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "framework_id": framework,
                    "mode": "embedded_framework_runtime",
                    "launcher": _rel(repo / str(spec["launcher"]), subject_root),
                    "qualification_milestone": "checkpoint_save_load",
                },
                indent=2,
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )

        workload_rows: list[dict[str, Any]] = []
        graph_summaries: list[dict[str, Any]] = []
        for workload_id in expected:
            model_graph = _model_graph(repo, workload_id) if spec["uses_hf_model_graph"] else DependencyGraph()
            static_graph = merge_dependency_graphs(
                static_core,
                model_graph,
                metadata={"framework_id": framework, "workload_id": workload_id, "validation_state": "pre_execution"},
            )
            validated_graph = merge_dependency_graphs(
                validated_core,
                model_graph,
                metadata={"framework_id": framework, "workload_id": workload_id, "validation_state": "rq1_feedback_applied"},
            )
            static_out = graph_root / f"{workload_id}.static.json"
            validated_out = graph_root / f"{workload_id}.validated.json"
            static_out.write_text(json.dumps(static_graph.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
            validated_out.write_text(json.dumps(validated_graph.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

            template = dict(template_by_id[workload_id])
            baseline = (workloads_path.parent / str(template["baseline_config"])).resolve()
            template.update(
                {
                    "status": "gpu_accelerator_qualified",
                    "baseline_config": _rel(baseline, subject_root),
                    "dependency_graph": _rel(validated_out, subject_root),
                    "static_dependency_graph": _rel(static_out, subject_root),
                    "native_validator_manifest": "native_validator.json",
                    "command_manifest": _rel(repo / str(spec["launcher"]), subject_root),
                }
            )
            workload_rows.append(template)
            graph_summaries.append(
                {
                    "workload_id": workload_id,
                    "static_edges": len(static_graph.edges),
                    "validated_edges": len(validated_graph.edges),
                    "model_stack_edges": len(model_graph.edges),
                    "static_graph_sha256": _sha256(static_out),
                    "validated_graph_sha256": _sha256(validated_out),
                }
            )

        registry = {
            "schema_version": 1,
            "name": f"rq2-{framework}-promoted-workloads",
            "metadata": {
                "status": "gpu_accelerator_qualified",
                "framework_id": framework,
                "rq1_result_sha256": _sha256(rq1_path),
                "static_artifact_sha256": _sha256(static_path),
                "qualification_summary_sha256": _sha256(qualification_dir / "summary.tsv"),
            },
            "workloads": workload_rows,
        }
        registry_path = subject_root / "workloads.yaml"
        registry_path.write_text(yaml.safe_dump(registry, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
        summary_subjects.append(
            {
                "framework_id": framework,
                "workload_count": len(expected),
                "workloads": expected,
                "rq1_result": _rel(rq1_path, output_root),
                "rq1_result_sha256": _sha256(rq1_path),
                "static_artifact": _rel(static_path, output_root),
                "static_artifact_sha256": _sha256(static_path),
                "qualification_dir": _rel(qualification_dir, output_root),
                "qualification": qualification,
                "graphs": graph_summaries,
            }
        )

    summary = {
        "schema_version": 2,
        "name": "rq2-gpu-promotion",
        "status": "gpu_ready_for_formal_rq2",
        "graph_scope": "framework_x_workload",
        "subjects": summary_subjects,
    }
    summary_path = output_root / "promotion.yaml"
    summary_path.write_text(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "subjects": len(summary_subjects), "output": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
