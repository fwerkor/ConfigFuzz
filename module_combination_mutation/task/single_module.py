#!/usr/bin/env python3
"""
外部脚本控制：单模块变异 + 跑测（save / load）

流程：每轮执行 single_module_mutate -> single_module_test --save-ckpt -> single_module_test --load-ckpt
参考：module_combine.py（模块间组合变异 + PTA 跑测）
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# 与 module_combine.py 一致：task 目录作为 package 运行时，需能 import 到 module_combination_mutation 下的 utils
try:
    import utils as mm_utils
except ImportError:
    import sys
    _MODULE_DIR = Path(__file__).resolve().parents[1]
    if str(_MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(_MODULE_DIR))
    import utils as mm_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "module_combination_mutation"


def _build_ms_pythonpath_block(ms_path: str) -> str:
    if not ms_path:
        return ""
    base = shlex.quote(ms_path)
    parts = [
        base,
        f"{base}/MSAdapter",
        f"{base}/MSAdapter/msa_thirdparty",
        f"{base}/Megatron-LM",
        f"{base}/MindSpeed",
        f"{base}/MindSpeed-MM",
    ]
    pythonpath_expr = ":".join(parts) + ":$PYTHONPATH"
    lines = [
        f"export MindSpeed_Core_MS_PATH={base}",
        f"export PYTHONPATH={pythonpath_expr}",
    ]
    return "\n".join(lines)


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _abs(p: str | Path, *, base: Path) -> Path:
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _run_bash(
    cmd: str,
    *,
    cwd: Path,
    check: bool = True,
    dry_run: bool = False,
    capture_output: bool = True,
    env: Optional[Dict[str, str]] = None,
) -> subprocess.CompletedProcess[str]:
    if dry_run:
        mm_utils.log_step("DRY-RUN", f"(cwd={cwd}) 即将执行命令：", indent=1)
        mm_utils.log_bullet(cmd, indent=2)
        return subprocess.CompletedProcess(args=["bash", "-lc", cmd], returncode=0, stdout="", stderr="")
    if not capture_output:
        return subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(cwd),
            check=check,
            text=True,
            capture_output=False,
            env=env,
        )
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )
    if check and proc.returncode != 0:
        mm_utils.log_step("ERROR", f"命令执行失败 (cwd={cwd})，返回码={proc.returncode}")
        mm_utils.log_bullet(cmd, indent=1)
        if proc.stdout:
            mm_utils.log_step("STDOUT", "子进程 stdout（截断前几行）", indent=1)
            for line in proc.stdout.splitlines()[:50]:
                mm_utils.log_bullet(line, indent=2)
        if proc.stderr:
            mm_utils.log_step("STDERR", "子进程 stderr（截断前几行）", indent=1)
            for line in proc.stderr.splitlines()[:50]:
                mm_utils.log_bullet(line, indent=2)
        raise subprocess.CalledProcessError(
            returncode=proc.returncode,
            cmd=proc.args,
            output=proc.stdout,
            stderr=proc.stderr,
        )
    return proc


def _build_conda_activate_block(env_name: str) -> str:
    lines = [
        "CONDA_PATH=$(conda info --base 2>/dev/null)",
        'if [ -z "$CONDA_PATH" ]; then',
        '  echo "ERROR: conda base path not found" >&2',
        "  exit 1",
        "fi",
        'source "$CONDA_PATH/etc/profile.d/conda.sh"',
    ]
    if env_name:
        lines.append(f"conda activate {shlex.quote(env_name)}")
    return "\n".join(lines)


@dataclass(frozen=True)
class GlobalConfig:
    results_root: str = "./module_combination_mutation/task/results_single_module"
    conda_env_pta: str = "mindspeed-mm"
    dry_run: bool = False
    rounds: int = 1
    capture_child_output: bool = True
    MindSpeed_Core_MS_PATH: str = ""


@dataclass(frozen=True)
class MutateConfig:
    rounds: int = 10
    iterations: int = 10
    extra_args: str = ""
    output_dir: str = ""


@dataclass(frozen=True)
class TestSaveConfig:
    iterations: int = 2
    save_ckpt: bool = True
    results_dir: str = ""
    ckpt_dir: str = ""
    extra_args: str = ""


@dataclass(frozen=True)
class TestLoadConfig:
    iterations: int = 2
    results_dir: str = ""
    load_ckpt_dir: str = ""
    extra_args: str = ""


@dataclass(frozen=True)
class Config:
    global_cfg: GlobalConfig
    mutate: MutateConfig
    test_save: TestSaveConfig
    test_load: TestLoadConfig


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping, got: {type(data)}")
    return data


def _dc_from_dict(dc_type, d: Dict[str, Any]):
    fields = {f.name for f in dataclasses.fields(dc_type)}
    kwargs = {k: v for k, v in (d or {}).items() if k in fields}
    return dc_type(**kwargs)


def load_config(yaml_path: Path) -> Config:
    raw = _load_yaml(yaml_path)
    global_cfg = _dc_from_dict(GlobalConfig, raw.get("global", {}))
    mutate = _dc_from_dict(MutateConfig, raw.get("mutate", {}))
    test_save = _dc_from_dict(TestSaveConfig, raw.get("test_save", {}))
    test_load = _dc_from_dict(TestLoadConfig, raw.get("test_load", {}))
    return Config(global_cfg=global_cfg, mutate=mutate, test_save=test_save, test_load=test_load)


def _select_config_with_max_iteration(configs_dir: Path) -> Path:
    """
    从 configs 目录中选出 iteration 数最大的配置。
    文件名格式为 {round}-{iteration}-{module_type}.json，取 iteration 最大者（因非每次变异都成功）。
    """
    jsons = list(configs_dir.glob("*.json"))
    if not jsons:
        raise FileNotFoundError(f"no json config under: {configs_dir}")

    def parse_iteration(p: Path) -> int:
        # 1-7-text_decoder.json -> 7
        stem = p.stem
        parts = stem.split("-", 2)
        if len(parts) >= 2 and parts[1].isdigit():
            return int(parts[1])
        return -1

    selected = max(jsons, key=parse_iteration)
    return selected.resolve()


def mutate_one(cfg: Config, run_root: Path) -> tuple[Path, Path]:
    """
    运行 single_module_mutate.sh，返回 (configs_dir, selected_config_path)。
    selected_config_path 为 iteration 数最大的那个配置，用于后续 save/load。
    """
    if cfg.mutate.output_dir:
        out_dir = _abs(cfg.mutate.output_dir, base=REPO_ROOT)
        use_subdir = False
    else:
        out_dir = run_root / "mutate"
        use_subdir = True
    out_dir.mkdir(parents=True, exist_ok=True)

    export_lines = [
        f"export RESULTS_DIR={shlex.quote(str(out_dir))}",
        f"export ROUNDS={int(cfg.mutate.rounds)}",
        f"export ITERATIONS={int(cfg.mutate.iterations)}",
    ]
    args = []
    if cfg.mutate.extra_args:
        args.append(cfg.mutate.extra_args)
    bash_cmd = "\n".join(
        [
            _build_conda_activate_block(cfg.global_cfg.conda_env_pta),
            _build_ms_pythonpath_block(cfg.global_cfg.MindSpeed_Core_MS_PATH),
            "\n".join(export_lines),
            f"bash single_module_mutate.sh {' '.join(args)}",
        ]
    )

    mm_utils.log_step("MUTATE", "开始运行 single_module_mutate 脚本...", indent=1)
    mm_utils.log_bullet(f"run_root = {run_root}", indent=2)
    mm_utils.log_bullet(f"输出基目录 (RESULTS_DIR): {out_dir}", indent=2)
    _run_bash(
        bash_cmd,
        cwd=MODULE_DIR,
        dry_run=cfg.global_cfg.dry_run,
        capture_output=cfg.global_cfg.capture_child_output,
    )
    mm_utils.log_step("MUTATE", "single_module_mutate 阶段完成。", indent=1)

    if use_subdir:
        run_dirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith("mutate_single_")]
        if not run_dirs:
            raise FileNotFoundError(f"no mutate_single_* run dir produced under: {out_dir}")
        run_dir = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        mm_utils.log_bullet(f"本次 mutate 运行目录: {run_dir}", indent=2)
    else:
        run_dir = out_dir
    configs_dir = (run_dir / "configs").resolve()

    if not cfg.global_cfg.dry_run:
        if not configs_dir.is_dir():
            raise FileNotFoundError(f"mutate configs dir not found: {configs_dir}")
        selected_config = _select_config_with_max_iteration(configs_dir)
        mm_utils.log_bullet(f"configs_dir: {configs_dir}，选用最大 iteration 配置: {selected_config.name}", indent=2)
        return (configs_dir, selected_config)
    # dry_run: 伪造一个选中配置路径
    dummy_config = configs_dir / "1-1-text_decoder.json"
    return (configs_dir, dummy_config.resolve())


def _find_ckpt_for_config_path(ckpt_dir: Path, config_path: Path) -> Path:
    """根据配置路径在 ckpt_dir 中查找对应权重（文件名含 config stem）。"""
    stem = config_path.stem  # e.g. 1-7-text_decoder
    candidates = list(ckpt_dir.glob(f"*_{stem}.pt")) + list(ckpt_dir.glob(f"*_*_{stem}.pt"))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint for config {config_path.name} under {ckpt_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def test_save_ckpt(
    cfg: Config,
    selected_config_path: Path,
    run_root: Path,
) -> Path:
    """仅对选中的配置跑 single_module_test 并保存权重，返回该配置对应的 ckpt 文件路径。"""
    if cfg.test_save.results_dir:
        results_dir = _abs(cfg.test_save.results_dir, base=REPO_ROOT)
    else:
        results_dir = run_root / "test_save"
    if cfg.test_save.ckpt_dir:
        ckpt_dir = _abs(cfg.test_save.ckpt_dir, base=REPO_ROOT)
    else:
        ckpt_dir = results_dir / "ckpts"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    extra = [
        f"--config {shlex.quote(str(selected_config_path))}",
        f"--iterations {int(cfg.test_save.iterations)}",
        f"--results-dir {shlex.quote(str(results_dir))}",
        f"--ckpt-dir {shlex.quote(str(ckpt_dir))}",
    ]
    if cfg.test_save.save_ckpt:
        extra.append("--save-ckpt")
    if cfg.test_save.extra_args:
        extra.append(cfg.test_save.extra_args)

    bash_cmd = "\n".join(
        [
            _build_conda_activate_block(cfg.global_cfg.conda_env_pta),
            _build_ms_pythonpath_block(cfg.global_cfg.MindSpeed_Core_MS_PATH),
            f"bash single_module_test.sh {' '.join(extra)}",
        ]
    )
    mm_utils.log_step("TEST-SAVE", "开始运行 single_module_test (save)，仅选中的配置...", indent=1)
    mm_utils.log_bullet(f"selected_config: {selected_config_path}", indent=2)
    mm_utils.log_bullet(f"ckpt_dir: {ckpt_dir}", indent=2)
    _run_bash(
        bash_cmd,
        cwd=MODULE_DIR,
        dry_run=cfg.global_cfg.dry_run,
        capture_output=cfg.global_cfg.capture_child_output,
    )
    mm_utils.log_step("TEST-SAVE", "test save 阶段完成。", indent=1)

    if cfg.global_cfg.dry_run:
        dummy_ckpt = ckpt_dir / f"dummy_{selected_config_path.stem}.pt"
        mm_utils.log_bullet(f"[DRY-RUN] 预期 ckpt: {dummy_ckpt}", indent=2)
        return dummy_ckpt.resolve()
    return _find_ckpt_for_config_path(ckpt_dir, selected_config_path).resolve()


def test_load_ckpt(
    cfg: Config,
    selected_config_path: Path,
    selected_ckpt_path: Path,
    run_root: Path,
) -> None:
    """对选中的配置跑 single_module_test，加载选中的权重文件。"""
    if cfg.test_load.results_dir:
        results_dir = _abs(cfg.test_load.results_dir, base=REPO_ROOT)
    else:
        results_dir = run_root / "test_load"
    results_dir.mkdir(parents=True, exist_ok=True)

    extra = [
        f"--config {shlex.quote(str(selected_config_path))}",
        f"--iterations {int(cfg.test_load.iterations)}",
        f"--results-dir {shlex.quote(str(results_dir))}",
        f"--ckpt {shlex.quote(str(selected_ckpt_path))}",
    ]
    if cfg.test_load.extra_args:
        extra.append(cfg.test_load.extra_args)

    bash_cmd = "\n".join(
        [
            _build_conda_activate_block(cfg.global_cfg.conda_env_pta),
            _build_ms_pythonpath_block(cfg.global_cfg.MindSpeed_Core_MS_PATH),
            f"bash single_module_test.sh {' '.join(extra)}",
        ]
    )
    mm_utils.log_step("TEST-LOAD", "开始运行 single_module_test (load)，选中的配置+权重...", indent=1)
    mm_utils.log_bullet(f"selected_config: {selected_config_path}", indent=2)
    mm_utils.log_bullet(f"ckpt: {selected_ckpt_path}", indent=2)
    _run_bash(
        bash_cmd,
        cwd=MODULE_DIR,
        dry_run=cfg.global_cfg.dry_run,
        capture_output=cfg.global_cfg.capture_child_output,
    )
    mm_utils.log_step("TEST-LOAD", "test load 阶段完成。", indent=1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="单模块变异 + 跑测全流程：mutate -> test_save -> test_load",
    )
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=str(MODULE_DIR / "task" / "single_module_config.yaml"),
        help="配置文件路径（YAML）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的命令，不实际运行")
    args = parser.parse_args()

    cfg = load_config(Path(args.config_yaml).resolve())
    if args.dry_run:
        cfg = dataclasses.replace(
            cfg,
            global_cfg=dataclasses.replace(cfg.global_cfg, dry_run=True),
        )

    results_root = _abs(cfg.global_cfg.results_root, base=REPO_ROOT)
    run_root = results_root / f"run_{_now_tag()}"
    run_root.mkdir(parents=True, exist_ok=True)
    mm_utils.log_section("单模块变异 + 跑测全流程")
    mm_utils.log_step("MAIN", "初始化运行配置")
    mm_utils.log_bullet(f"run_root: {run_root}")

    total_rounds = max(1, int(cfg.global_cfg.rounds))
    mm_utils.log_bullet(f"总轮次: {total_rounds}")

    successful_rounds: list[int] = []
    failed_rounds: list[int] = []

    for r in range(1, total_rounds + 1):
        round_root = run_root / f"round_{r}"
        round_root.mkdir(parents=True, exist_ok=True)
        mm_utils.log_section(f"Round {r}/{total_rounds}")
        mm_utils.log_step("ROUND", f"开始 round {r}/{total_rounds}")
        mm_utils.log_bullet(f"round_root: {round_root}", indent=1)

        try:
            configs_dir, selected_config_path = mutate_one(cfg, round_root)
        except FileNotFoundError as e:
            mm_utils.log_step(
                "ROUND",
                f"round {r} 变异失败（未找到生成的 configs），本 round 作废，跳过后续阶段",
                indent=1,
            )
            mm_utils.log_bullet(str(e), indent=2)
            failed_rounds.append(r)
            continue

        selected_ckpt_path = test_save_ckpt(cfg, selected_config_path, round_root)
        test_load_ckpt(cfg, selected_config_path, selected_ckpt_path, round_root)

        mm_utils.log_step("ROUND", f"round {r}/{total_rounds} 全部阶段完成")
        successful_rounds.append(r)

    mm_utils.log_section("Round 汇总")
    mm_utils.log_step("MAIN", f"成功执行的 round: {successful_rounds}")
    mm_utils.log_step("MAIN", f"失败的 round: {failed_rounds}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
