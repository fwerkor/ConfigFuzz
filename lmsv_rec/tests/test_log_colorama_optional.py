import builtins
import importlib
import sys
import unittest
from unittest.mock import patch


_ORIGINAL_IMPORT = builtins.__import__


def _block_colorama_import(name, *args, **kwargs):
    if name == "colorama" or name.startswith("colorama."):
        raise ImportError("colorama blocked for optional dependency test")
    return _ORIGINAL_IMPORT(name, *args, **kwargs)


class LogColoramaOptionalTests(unittest.TestCase):
    def test_log_module_imports_without_colorama(self) -> None:
        sys.modules.pop("utils.log.write", None)
        sys.modules.pop("colorama", None)
        try:
            with patch("builtins.__import__", side_effect=_block_colorama_import):
                module = importlib.import_module("utils.log.write")
                self.assertEqual(module.Fore.RED, "")
                self.assertEqual(module.Style.RESET_ALL, "")
                module.info("plain log ok")
        finally:
            sys.modules.pop("utils.log.write", None)
            importlib.import_module("utils.log.write")


if __name__ == "__main__":
    unittest.main()
