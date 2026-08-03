import builtins
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from utils.runtime import cluster_runtime
from utils.task import task2, task3


_ORIGINAL_IMPORT = builtins.__import__


def _block_torch_import(name, *args, **kwargs):
    if name == "torch":
        raise ImportError("torch blocked for deterministic device detection test")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


class RuntimeDeviceDetectionTests(unittest.TestCase):
    def test_cluster_health_ignores_distributed_world_size_env(self) -> None:
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}, clear=True),
            patch("builtins.__import__", side_effect=_block_torch_import),
            patch("utils.runtime.cluster_runtime.subprocess.run", side_effect=OSError),
        ):
            self.assertEqual(cluster_runtime.detect_visible_device_count(), 1)

    def test_task2_visible_device_inference_ignores_distributed_world_size_env(self) -> None:
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}, clear=True),
            patch("builtins.__import__", side_effect=_block_torch_import),
            patch("utils.task.task2.subprocess.run", side_effect=OSError),
        ):
            self.assertEqual(task2._infer_visible_device_count(), 1)

    def test_task3_visible_device_inference_ignores_distributed_world_size_env(self) -> None:
        with (
            patch.dict(os.environ, {"WORLD_SIZE": "16", "LOCAL_WORLD_SIZE": "8"}, clear=True),
            patch("builtins.__import__", side_effect=_block_torch_import),
            patch("utils.task.task3.subprocess.run", side_effect=OSError),
        ):
            self.assertEqual(task3._infer_visible_device_count(), 1)

    def test_task123_cluster_config_prefers_multinode_ssh_backend(self) -> None:
        cfg = cluster_runtime.parse_task123_cluster_config(
            {
                "MULTI_NODE": {
                    "ENABLED": True,
                    "MASTER_ADDR": "10.0.0.1",
                    "MASTER_PORT": 6123,
                    "HCCL_IF_IP": "10.0.0.1",
                    "LOCAL_NPUS_PER_NODE": 8,
                    "OTHER_NODES": [
                        {
                            "HOST": "worker@10.0.0.2",
                            "HCCL_IF_IP": "10.0.0.2",
                            "SSH_PORT": 2222,
                            "LMSV_PATH": "/data/lm-sv",
                            "PTA_NAME": "pta_env",
                            "MSA_NAME": "msa_env",
                            "MF_NAME": "mf_env",
                            "PTA_PATH": "/opt/pta",
                            "MSA_PATH": "/opt/msa",
                            "NPUS_PER_NODE": 4,
                        }
                    ],
                },
                "CLUSTER": {"ENABLED": True, "SLAVES": [{"HOST": "legacy", "PORT": 19001}]},
            }
        )

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.backend, "ssh")
        self.assertEqual(cfg.master_addr, "10.0.0.1")
        self.assertEqual(cfg.master_port, 6123)
        self.assertEqual(cfg.local_hccl_if_ip, "10.0.0.1")
        self.assertEqual(cfg.local_npus_per_node, 8)
        self.assertEqual(cfg.slaves[0].host, "worker@10.0.0.2")
        self.assertEqual(cfg.slaves[0].hccl_if_ip, "10.0.0.2")
        self.assertEqual(cfg.slaves[0].ssh_port, 2222)
        self.assertEqual(cfg.slaves[0].npus_per_node, 4)
        self.assertEqual(cfg.slaves[0].mf_name, "mf_env")

    def test_task123_cluster_config_falls_back_to_legacy_cluster(self) -> None:
        cfg = cluster_runtime.parse_task123_cluster_config(
            {
                "MULTI_NODE": {"ENABLED": False},
                "CLUSTER": {
                    "ENABLED": True,
                    "MASTER_ADDR": "192.168.1.10",
                    "SLAVES": [{"ENDPOINT": "192.168.1.11:19002"}],
                },
            }
        )

        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.backend, "http")
        self.assertEqual(cfg.master_addr, "192.168.1.10")
        self.assertEqual(cfg.slaves[0].endpoint, "192.168.1.11:19002")

    def test_ssh_remote_repo_root_accepts_parent_or_repo_path(self) -> None:
        master = cluster_runtime.ClusterMaster(
            cluster_runtime.ClusterConfig(enabled=True, backend="ssh"),
            lambda _message: None,
            lambda _message: None,
            lambda _message: None,
        )

        parent_node = cluster_runtime.ClusterNode(
            endpoint="worker",
            host="worker",
            port=0,
            node_rank=1,
            lmsv_path="/data/lm-sv",
        )
        repo_node = cluster_runtime.ClusterNode(
            endpoint="worker",
            host="worker",
            port=0,
            node_rank=1,
            lmsv_path="/data/lm-sv/lmsv_rec",
        )

        self.assertEqual(master._remote_repo_root(parent_node), "/data/lm-sv/lmsv_rec")
        self.assertEqual(master._remote_repo_root(repo_node), "/data/lm-sv/lmsv_rec")

    def test_container_upload_copies_bundle_into_container_repo(self) -> None:
        master = cluster_runtime.ClusterMaster(
            cluster_runtime.ClusterConfig(
                enabled=True,
                backend="ssh",
                request_timeout=7,
                ssh_bin="ssh",
                rsync_bin="rsync",
            ),
            lambda _message: None,
            lambda _message: None,
            lambda _message: None,
        )
        node = cluster_runtime.ClusterNode(
            endpoint="worker",
            host="worker",
            port=0,
            node_rank=1,
            ssh_port=2222,
            lmsv_path="/remote/lmsv_rec",
            has_container=True,
            container_name="lmsv-worker",
        )
        calls = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "pta.sh"
            src.write_text("echo ok\n", encoding="utf-8")
            with patch("utils.runtime.cluster_runtime.subprocess.run", side_effect=fake_run):
                master._upload_paths_ssh(node, "session-1", [(src, "pta/qwen3/pta.sh")])

        rendered = [" ".join(map(str, call)) if isinstance(call, list) else str(call) for call in calls]
        self.assertTrue(any("rsync" in call and "bundle.tar.gz" in call for call in rendered))
        self.assertTrue(any("docker cp" in call and "lmsv-worker:/tmp/lmsv_cluster_upload_" in call for call in rendered))
        self.assertTrue(any("docker exec" in call and "tar -xzf" in call and " -C /" in call for call in rendered))

    def test_ssh_runtime_upload_targets_direct_job_workspace(self) -> None:
        master = cluster_runtime.ClusterMaster(
            cluster_runtime.ClusterConfig(enabled=True, backend="ssh"),
            lambda _message: None,
            lambda _message: None,
            lambda _message: None,
        )
        node = cluster_runtime.ClusterNode(
            endpoint="worker",
            host="worker",
            port=0,
            node_rank=1,
            lmsv_path="/remote/lm-sv",
        )

        self.assertEqual(
            master._remote_upload_path(node, "session-1", "pta/qwen3/pta.sh"),
            "/remote/lm-sv/lmsv_rec/tmp/ssh_direct/session-1/runtime_workspace/pta/qwen3/pta.sh",
        )
        self.assertEqual(
            master._remote_upload_path(node, "session-1", "dataset/train.bin"),
            "/remote/lm-sv/lmsv_rec/dataset/train.bin",
        )

    def test_ssh_stale_training_cleanup_covers_task6_risks(self) -> None:
        shell = cluster_runtime._build_stale_training_cleanup_shell(master_port=6123)

        self.assertIn("[m]srun", shell)
        self.assertIn("[t]orchrun", shell)
        self.assertIn("[p]retrain_gpt", shell)
        self.assertIn("fuser -k", shell)
        self.assertIn("6000", shell)
        self.assertIn("6123", shell)
        self.assertIn("60000", shell)
        self.assertIn("60031", shell)
        self.assertIn("/dev/shm/hccl_*", shell)
        self.assertIn("/dev/shm/psm_*", shell)
        self.assertIn("npu-smi", shell)

    def test_ssh_direct_job_bootstraps_target_conda_env(self) -> None:
        settings = {"PTA_NAME": "pta_env", "MSA_NAME": "msa_env", "MF_NAME": "mf_env"}

        self.assertEqual(
            cluster_runtime._remote_bootstrap_env_name("task1_run_script", {"env_type": 1}, settings),
            "pta_env",
        )
        self.assertEqual(
            cluster_runtime._remote_bootstrap_env_name("task1_run_script", {"env_type": 2}, settings),
            "msa_env",
        )
        self.assertEqual(
            cluster_runtime._remote_bootstrap_env_name("task2_mf_verify", {}, settings),
            "mf_env",
        )


if __name__ == "__main__":
    unittest.main()
