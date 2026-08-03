import subprocess
import tempfile
import unittest
from pathlib import Path

from utils.runtime import cluster_runtime
from utils.task import task2, task3
from utils.task.task1 import (
    align_bias_linear_flags,
    apply_deepseekv3_unified_low_memory_profile,
    apply_multinode_script_settings,
    apply_script_constraints,
    _contains_mf_fatal_text,
    ensure_distributed_optimizer_shared_param_compatible,
    sanitize_moe_expert_bias_aux_loss,
    sanitize_swiglu_fusion_script,
    sanitize_task1_mutation_runtime_flags,
)


class Task1ScriptSanitizerTests(unittest.TestCase):
    def test_task1_mutation_runtime_flags_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --use-flash-attn \\",
                        "    --overlap-grad-reduce \\",
                        "    --overlap-param-gather \\",
                        "    --train-iters 10 \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(sanitize_task1_mutation_runtime_flags(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertNotIn("--use-flash-attn", content)
            self.assertNotIn("--overlap-grad-reduce", content)
            self.assertNotIn("--overlap-param-gather", content)
            self.assertIn("--train-iters 10", content)

    def test_moe_expert_bias_forces_positive_aux_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --moe-router-enable-expert-bias \\",
                        "    --moe-aux-loss-coeff 0.0 \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(sanitize_moe_expert_bias_aux_loss(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--moe-router-enable-expert-bias", content)
            self.assertIn("--moe-aux-loss-coeff 0.01", content)
            self.assertNotIn("--moe-aux-loss-coeff 0.0 \\", content)

    def test_shared_param_optimizer_cleanup_tolerates_absent_overlap_flags(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "pretrain_mutated_pangu-2.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'OPTIMIZER_ARGS="',
                        "    --use-distributed-optimizer \\",
                        "    --train-iters 1 \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(ensure_distributed_optimizer_shared_param_compatible(script_path))
            self.assertTrue(apply_script_constraints(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertNotIn("--use-distributed-optimizer", content)
            self.assertIn("--train-iters 1", content)

    def test_multinode_task1_only_updates_torchrun_topology(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        "NPUS_PER_NODE=8",
                        "MASTER_ADDR=127.0.0.1",
                        "MASTER_PORT=29500",
                        "NNODES=1",
                        "NODE_RANK=0",
                        "WORLD_SIZE=8",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                apply_multinode_script_settings(
                    script_path,
                    local_workers=8,
                    total_workers=16,
                    nnodes=2,
                    node_rank=1,
                    master_addr="10.0.0.1",
                    master_port=29501,
                    enable_pta_env=True,
                )
            )

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("NPUS_PER_NODE=8", content)
            self.assertIn("MASTER_ADDR=10.0.0.1", content)
            self.assertIn("MASTER_PORT=29501", content)
            self.assertIn("NNODES=2", content)
            self.assertIn("NODE_RANK=1", content)
            self.assertIn("WORLD_SIZE=16", content)
            self.assertNotIn("GLOO_SOCKET_IFNAME", content)
            self.assertNotIn("TP_SOCKET_IFNAME", content)
            self.assertNotIn("HCCL_SOCKET_IFNAME", content)
            self.assertNotIn("HCCL_IF_IP", content)
            self.assertNotIn("HCCL_WHITELIST_DISABLE", content)
            self.assertNotIn("unset RANK_TABLE_FILE", content)
            self.assertNotIn("unset LOCAL_RANK", content)
            self.assertNotIn("unset TORCHELASTIC_RUN_ID", content)

    def test_multinode_task1_ignores_explicit_hccl_if_ip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        "NPUS_PER_NODE=8",
                        "MASTER_ADDR=127.0.0.1",
                        "MASTER_PORT=29500",
                        "NNODES=1",
                        "NODE_RANK=0",
                        "WORLD_SIZE=8",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                apply_multinode_script_settings(
                    script_path,
                    local_workers=8,
                    total_workers=16,
                    nnodes=2,
                    node_rank=1,
                    master_addr="10.0.0.11",
                    master_port=29501,
                    hccl_if_ip="10.0.0.12",
                    enable_pta_env=True,
                )
            )

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("MASTER_ADDR=10.0.0.11", content)
            self.assertIn("NNODES=2", content)
            self.assertIn("NODE_RANK=1", content)
            self.assertNotIn("HCCL_IF_IP", content)
            self.assertNotIn("HCCL_WHITELIST_DISABLE", content)
            self.assertNotIn("unset HCCL_DETERMINISTIC", content)

    def test_task23_multinode_commands_do_not_inject_hccl_if_ip(self) -> None:
        for module in (task2, task3):
            block = module._build_multinode_hccl_if_ip_env_block(
                {"nnodes": 2, "master_addr": "10.0.0.11", "hccl_if_ip": "10.0.0.12"}
            )
            self.assertEqual(block, "")

    def test_task23_multinode_uses_minimal_env_block(self) -> None:
        for module in (task2, task3):
            multinode_block = module._build_distributed_deterministic_env_block({"nnodes": 2})
            single_node_block = module._build_distributed_deterministic_env_block({"nnodes": 1})
            self.assertIn("export CUDA_DEVICE_MAX_CONNECTIONS=1", multinode_block)
            self.assertNotIn("unset HCCL_DETERMINISTIC", multinode_block)
            self.assertNotIn("unset NCCL_DETERMINISTIC", multinode_block)
            self.assertNotIn("ASCEND_LAUNCH_BLOCKING", multinode_block)
            self.assertNotIn("PYTORCH_NPU_ALLOC_CONF", multinode_block)
            self.assertNotIn("export HCCL_DETERMINISTIC=true", multinode_block)
            self.assertIn("export HCCL_DETERMINISTIC=true", single_node_block)

    def test_task23_multinode_commands_do_not_touch_hccl_deterministic(self) -> None:
        for module in (task2, task3):
            original = {
                "TARGET_NNODES": module.Config.TARGET_NNODES,
                "TARGET_WORLD_SIZE": module.Config.TARGET_WORLD_SIZE,
                "TARGET_NPUS_PER_NODE": module.Config.TARGET_NPUS_PER_NODE,
                "TARGET_MASTER_ADDR": module.Config.TARGET_MASTER_ADDR,
                "TARGET_HCCL_IF_IP": module.Config.TARGET_HCCL_IF_IP,
            }
            try:
                module.Config.TARGET_NNODES = 2
                module.Config.TARGET_WORLD_SIZE = 16
                module.Config.TARGET_NPUS_PER_NODE = 8
                module.Config.TARGET_MASTER_ADDR = "10.0.0.11"
                module.Config.TARGET_HCCL_IF_IP = "10.0.0.12"
                if module is task2:
                    commands = [
                        module.build_pta_verify_stage_cmd(1, "", "pta_env", "/opt/pta", "/tmp/shared", "save", 1),
                        module.build_msa_verify_load_cmd(1, "", "msa_env", "/opt/msa", "/tmp/shared", 1),
                    ]
                else:
                    commands = [
                        module.build_pta_verify_stage_cmd(1, "", "/tmp/load", "pta_env", "/opt/pta", "/tmp/shared", "save", 1),
                        module.build_msa_verify_load_cmd(1, "", "/tmp/load", "msa_env", "/opt/msa", "/tmp/shared", 1),
                ]
                for command in commands:
                    self.assertNotIn("unset HCCL_DETERMINISTIC", command)
                    self.assertNotIn("unset NCCL_DETERMINISTIC", command)
                    self.assertNotIn("export HCCL_DETERMINISTIC=true", command)
            finally:
                for key, value in original.items():
                    setattr(module.Config, key, value)

    def test_task123_commands_export_path_aliases_like_task6(self) -> None:
        runtime_settings = {
            "PTA_NAME": "pta_env",
            "MSA_NAME": "msa_env",
            "MF_NAME": "mf_env",
            "PTA_PATH": "/opt/pta",
            "MSA_PATH": "/opt/msa",
        }
        task1_pta = cluster_runtime._build_task1_run_script_command(
            {"script_rel": "pta/run.sh", "env_type": 1},
            runtime_settings,
        )
        task1_msa = cluster_runtime._build_task1_run_script_command(
            {"script_rel": "msa/run.sh", "env_type": 2},
            runtime_settings,
        )
        self.assertIn("export PTA_PATH=/opt/pta", task1_pta)
        self.assertIn("export PTAPATH=/opt/pta", task1_pta)
        self.assertIn('LMSV_TASK_SCRIPT="$LMSV_JOB_REPO_ROOT"/pta/run.sh', task1_pta)
        self.assertIn('bash -x -o pipefail "$LMSV_TASK_SCRIPT"', task1_pta)
        self.assertIn("export MSA_PATH=/opt/msa", task1_msa)
        self.assertIn("export MSAPATH=/opt/msa", task1_msa)
        self.assertIn('LMSV_TASK_SCRIPT="$LMSV_JOB_REPO_ROOT"/msa/run.sh', task1_msa)
        self.assertIn('bash -x -o pipefail "$LMSV_TASK_SCRIPT"', task1_msa)

        task2_pta = task2.build_pta_verify_stage_cmd(1, "", "pta_env", "/opt/pta", "/tmp/shared", "save", 1)
        task2_msa = task2.build_msa_verify_load_cmd(1, "", "msa_env", "/opt/msa", "/tmp/shared", 1)
        task3_pta = task3.build_pta_verify_stage_cmd(1, "", "/tmp/load", "pta_env", "/opt/pta", "/tmp/shared", "save", 1)
        task3_msa = task3.build_msa_verify_load_cmd(1, "", "/tmp/load", "msa_env", "/opt/msa", "/tmp/shared", 1)

        self.assertIn("export PTA_PATH=/opt/pta", task2_pta)
        self.assertIn("export PTAPATH=/opt/pta", task2_pta)
        self.assertIn("export MSA_PATH=/opt/msa", task2_msa)
        self.assertIn("export MSAPATH=/opt/msa", task2_msa)
        self.assertIn("export PTA_PATH=/opt/pta", task3_pta)
        self.assertIn("export PTAPATH=/opt/pta", task3_pta)
        self.assertIn("export MSA_PATH=/opt/msa", task3_msa)
        self.assertIn("export MSAPATH=/opt/msa", task3_msa)

    def test_task1_remote_mf_training_exports_hccl_if_ip(self) -> None:
        command = cluster_runtime._build_task1_mf_training_command(
            {
                "yaml_rel": "mf/qwen.yaml",
                "local_workers": 8,
                "total_workers": 16,
                "nnodes": 2,
                "master_addr": "10.0.0.11",
                "master_port": 6123,
                "node_rank": 1,
                "hccl_if_ip": "10.0.0.12",
            },
            {
                "MF_NAME": "mf_env",
                "PTA_NAME": "pta_env",
                "MSA_NAME": "msa_env",
                "PTA_PATH": "/opt/pta",
                "MSA_PATH": "/opt/msa",
            },
        )

        self.assertIn("export LMSV_MF_MASTER_ADDR=10.0.0.11", command)
        self.assertIn("export MASTER_ADDR=10.0.0.11", command)
        self.assertIn("export NNODES=2", command)
        self.assertIn("HCCL_IF_IP=10.0.0.12", command)
        self.assertIn("export HCCL_WHITELIST_DISABLE=1", command)
        self.assertIn('export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1200}"', command)
        self.assertIn("unset HCCL_DETERMINISTIC", command)

    def test_multinode_task1_rewrite_survives_empty_master_addr(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        "NPUS_PER_NODE=8",
                        "MASTER_ADDR=",
                        "MASTER_PORT=29500",
                        "NNODES=2",
                        "NODE_RANK=1",
                        "WORLD_SIZE=16",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                apply_multinode_script_settings(
                    script_path,
                    local_workers=8,
                    total_workers=16,
                    nnodes=2,
                    node_rank=1,
                    master_addr="",
                    master_port=29501,
                    enable_pta_env=True,
                )
            )

            result = subprocess.run(
                ["bash", "-lc", f"set -e -o pipefail; source {script_path}"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr or result.stdout)

    def test_task23_socket_ifname_blocks_are_disabled(self) -> None:
        for module in (task2, task3):
            block = module._build_multinode_pta_socket_ifname_env_block({"nnodes": 2})
            self.assertEqual(block, "")

    def test_single_node_pta_env_does_not_inject_extra_network_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "task1.sh"
            script_path.write_text(
                "\n".join(
                    [
                        "NPUS_PER_NODE=8",
                        "MASTER_ADDR=127.0.0.1",
                        "MASTER_PORT=29500",
                        "NNODES=1",
                        "NODE_RANK=0",
                        "WORLD_SIZE=8",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(
                apply_multinode_script_settings(
                    script_path,
                    local_workers=8,
                    total_workers=8,
                    nnodes=1,
                    node_rank=0,
                    master_addr="127.0.0.1",
                    master_port=29500,
                    enable_pta_env=True,
                )
            )

            content = script_path.read_text(encoding="utf-8")
            self.assertNotIn("GLOO_SOCKET_IFNAME", content)
            self.assertNotIn("TP_SOCKET_IFNAME", content)
            self.assertNotIn("HCCL_SOCKET_IFNAME", content)
            self.assertNotIn("unset RANK_TABLE_FILE", content)
            self.assertNotIn("unset LOCAL_RANK", content)
            self.assertNotIn("unset LOCAL_WORLD_SIZE", content)

    def test_gqa_tensor_parallel_is_reduced_to_query_group_divisor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "qwen3.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --num-attention-heads 8 \\",
                        "    --num-query-groups 2 \\",
                        "    --tensor-model-parallel-size 4 \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(apply_script_constraints(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--tensor-model-parallel-size 2", content)

    def test_attention_heads_parallel_product_reduces_context_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            for model_name in ("grok1", "chatglm3"):
                with self.subTest(model_name=model_name):
                    script_path = Path(tmpdir) / f"{model_name}.sh"
                    script_path.write_text(
                        "\n".join(
                            [
                                'GPT_ARGS="',
                                "    --num-attention-heads 4 \\",
                                "    --tensor-model-parallel-size 2 \\",
                                "    --context-parallel-size 4 \\",
                                '"',
                                "",
                            ]
                        ),
                        encoding="utf-8",
                    )

                    self.assertTrue(apply_script_constraints(script_path))

                    content = script_path.read_text(encoding="utf-8")
                    self.assertIn("--num-attention-heads 4", content)
                    self.assertIn("--tensor-model-parallel-size 2", content)
                    self.assertIn("--context-parallel-size 2", content)

    def test_first_k_dense_replace_does_not_fill_pipeline_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "deepseekv3.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --num-layers 4 \\",
                        "    --pipeline-model-parallel-size 4 \\",
                        '"',
                        'MOE_ARGS="',
                        "    --first-k-dense-replace 1 \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(apply_script_constraints(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--num-layers 4", content)
            self.assertIn("--pipeline-model-parallel-size 4", content)
            self.assertIn("--first-k-dense-replace 0", content)

    def test_moe_seq_aux_forces_positive_aux_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "deepseekv3.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'MOE_ARGS="',
                        "    --moe-aux-loss-coeff 0.0 \\",
                        "    --moe-router-load-balancing-type group_limited_greedy \\",
                        "    --moe-comm-aux-loss-coeff 0.02 \\",
                        "    --moe-device-level-aux-loss-coeff 0.05 \\",
                        "    --seq-aux \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(apply_script_constraints(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--moe-aux-loss-coeff 0.01", content)
            self.assertIn("--moe-router-load-balancing-type aux_loss", content)
            self.assertNotIn("--moe-router-load-balancing-type group_limited_greedy", content)

    def test_distributed_optimizer_removed_for_tied_shared_params(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "pangu.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --num-layers 4 \\",
                        "    --use-distributed-optimizer \\",
                        "    --overlap-param-gather \\",
                        "    --overlap-grad-reduce \\",
                        '"',
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(apply_script_constraints(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertNotIn("--use-distributed-optimizer", content)
            self.assertNotIn("--overlap-param-gather", content)
            self.assertNotIn("--overlap-grad-reduce", content)
            self.assertIn("--num-layers 4", content)

    def test_msa_bias_linear_flags_follow_pta_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pta_script = Path(tmpdir) / "pta.sh"
            msa_script = Path(tmpdir) / "msa.sh"
            pta_script.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --train-iters 10 \\",
                        "    --disable-bias-linear \\",
                        '"',
                    ]
                ),
                encoding="utf-8",
            )
            msa_script.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --train-iters 10 \\",
                        '"',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(align_bias_linear_flags(pta_script, msa_script))

            content = msa_script.read_text(encoding="utf-8")
            self.assertIn("--disable-bias-linear", content)
            self.assertNotIn("--add-bias-linear", content)

    def test_num_moe_experts_triggers_disable_bias_linear(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "moe.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --train-iters 10 \\",
                        "    --num-moe-experts 4 \\",
                        '"',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(sanitize_swiglu_fusion_script(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--disable-bias-linear", content)
            self.assertIn("--no-bias-swiglu-fusion", content)

    def test_deepseek_low_memory_profile_reduces_large_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = Path(tmpdir) / "deepseekv3.sh"
            script_path.write_text(
                "\n".join(
                    [
                        'GPT_ARGS="',
                        "    --num-layers 16 \\",
                        "    --hidden-size 5120 \\",
                        "    --ffn-hidden-size 12288 \\",
                        "    --num-attention-heads 128 \\",
                        "    --seq-length 4096 \\",
                        "    --max-position-embeddings 4096 \\",
                        "    --micro-batch-size 1 \\",
                        "    --global-batch-size 256 \\",
                        "    --pipeline-model-parallel-size 1 \\",
                        "    --tensor-model-parallel-size 1 \\",
                        "    --train-iters 1 \\",
                        '"',
                        'MLA_ARGS="',
                        "    --q-lora-rank 1536 \\",
                        "    --kv-lora-rank 512 \\",
                        '"',
                        'MOE_ARGS="',
                        "    --first-k-dense-replace 1 \\",
                        "    --moe-intermediate-size 1536 \\",
                        "    --moe-layer-freq 1 \\",
                        "    --moe-aux-loss-coeff 0.0 \\",
                        "    --moe-router-load-balancing-type group_limited_greedy \\",
                        "    --moe-router-topk 6 \\",
                        "    --n-shared-experts 2 \\",
                        "    --num-experts 32 \\",
                        '"',
                    ]
                ),
                encoding="utf-8",
            )

            self.assertTrue(apply_deepseekv3_unified_low_memory_profile(script_path))

            content = script_path.read_text(encoding="utf-8")
            self.assertIn("--num-layers 8", content)
            self.assertIn("--hidden-size 1024", content)
            self.assertIn("--ffn-hidden-size 2048", content)
            self.assertIn("--num-attention-heads 16", content)
            self.assertIn("--q-lora-rank 192", content)
            self.assertIn("--kv-lora-rank 64", content)
            self.assertIn("--seq-length 1024", content)
            self.assertIn("--global-batch-size 8", content)
            self.assertIn("--moe-aux-loss-coeff 0.01", content)
            self.assertIn("--moe-router-load-balancing-type aux_loss", content)
            self.assertIn("--num-experts 16", content)
            self.assertIn("--moe-router-topk 2", content)
            self.assertIn("--first-k-dense-replace 7", content)

    def test_mf_worker_ignores_nfs_temp_cleanup_traceback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "worker_0.log"
            log_path.write_text(
                "\n".join(
                    [
                        "Traceback (most recent call last):",
                        '  File "/root/miniconda3/envs/mindformers/lib/python3.11/multiprocessing/process.py", line 314, in _bootstrap',
                        "    self.run()",
                        "SystemExit: 0",
                        "",
                        "During handling of the above exception, another exception occurred:",
                        "",
                        "Traceback (most recent call last):",
                        '  File "/root/miniconda3/envs/mindformers/lib/python3.11/multiprocessing/util.py", line 303, in _run_finalizers',
                        "    finalizer()",
                        '  File "/root/miniconda3/envs/mindformers/lib/python3.11/multiprocessing/util.py", line 136, in _remove_temp_dir',
                        "    rmtree(tempdir, onerror=onerror)",
                        "OSError: [Errno 16] Device or resource busy: '.nfs0000000008a6084a00000089'",
                        "2026-04-25 22:13:34,321 - INFO - Safetensors Convert Complete",
                    ]
                ),
                encoding="utf-8",
            )

            self.assertIsNone(_contains_mf_fatal_text(log_path, ["Traceback (most recent call last)"]))


if __name__ == "__main__":
    unittest.main()
