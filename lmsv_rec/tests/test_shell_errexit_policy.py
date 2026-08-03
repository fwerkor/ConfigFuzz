import subprocess
import tempfile
import unittest
from pathlib import Path

from utils.task import runtime_helpers


class ShellErrexitPolicyTests(unittest.TestCase):
    def test_run_shell_to_file_does_not_exit_on_failed_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "run.log"
            result = runtime_helpers.run_shell_to_file(
                "OPTIONAL_VALUE=$(false)\necho still-running",
                log_file,
                Path(tmp_dir),
                lambda _message: None,
                check=True,
            )

            self.assertIsNotNone(result)
            self.assertEqual(result.returncode, 0)
            self.assertIn("still-running", log_file.read_text(encoding="utf-8"))

    def test_run_shell_to_file_keeps_final_failure_return_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "run.log"
            result = runtime_helpers.run_shell_to_file(
                "OPTIONAL_VALUE=$(false)\nfalse",
                log_file,
                Path(tmp_dir),
                lambda _message: None,
                check=False,
            )

            self.assertEqual(result.returncode, 1)

    def test_written_runtime_script_does_not_exit_on_failed_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = Path(tmp_dir) / "runtime.sh"
            runtime_helpers.write_runtime_script(
                script_path,
                "OPTIONAL_VALUE=$(false)\necho script-still-running",
            )

            result = subprocess.run(
                ["bash", str(script_path)],
                cwd=tmp_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0)
            self.assertIn("script-still-running", result.stdout)


if __name__ == "__main__":
    unittest.main()
