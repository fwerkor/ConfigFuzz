#!/usr/bin/env python3
"""
外部脚本控制：模块组合变异 + PTA 跑测（save / load）

依据开发指南：module_combination_mutation/dev_doc/external_script.md
参考实现：lmsv_rec/utils/task/task1.py（仅借鉴运行命令/封装风格）
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

import utils as mm_utils


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = REPO_ROOT / "module_combination_mutation"

# 注意：ptaa 环境名作为全局变量，便于后续统一修改
PTA_CONDA_ENV_NAME = "ptaa"


def _build_ms_pythonpath_block(ms_path: str) -> str:
    """
    根据 MindSpeed_Core_MS_PATH 构造 PYTHONPATH 相关的 shell 片段。
    若 ms_path 为空字符串，则返回空字符串，不做任何设置。
    """
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
        # 直接将子进程 stdout/stderr 打到当前控制台（与原行为一致）
        return subprocess.run(
            ["bash", "-lc", cmd],
            cwd=str(cwd),
            check=check,
            text=True,
            capture_output=False,
            env=env,
        )
    # 捕获输出，仅在出错时打印关键信息并抛异常。
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
    """
    构造一个可复用的 shell 片段，用于通过 conda 激活指定环境。
    参考 lmsv_rec/utils/task/runtime_helpers.py::build_conda_activate_block 的实现。
    """
    lines = [
        "CONDA_PATH=$(conda info --base 2>/dev/null)",
        'if [ -z "$CONDA_PATH" ]; then',
        '  echo "ERROR: conda base path not found" >&2',
        "  exit 1",
        "fi",
        'source \"$CONDA_PATH/etc/profile.d/conda.sh\"',
    ]
    if env_name:
        lines.append(f"conda activate {shlex.quote(env_name)}")
    return "\n".join(lines)


@dataclass(frozen=True)
class GlobalConfig:
    results_root: str = "./module_combination_mutation/results"
    conda_env_pta: str = PTA_CONDA_ENV_NAME
    dry_run: bool = False
    # 整体流程重复轮数（每轮执行 mutate + PTA save + PTA load）
    rounds: int = 1
    # 是否捕获子脚本输出而不直接打印到当前控制台；True=静默（仅出错时打印摘要）
    capture_child_output: bool = True
    # MindSpeed 核心工程路径，用于在子命令前设置 PYTHONPATH
    MindSpeed_Core_MS_PATH: str = ""


@dataclass(frozen=True)
class MutateConfig:
    rounds: int = 1
    extra_args: str = ""
    output_dir: str = ""  # 若为空则自动生成到 results_root/run_<ts>/mutate


@dataclass(frozen=True)
class PtaSaveConfig:
    iterations: int = 2
    save_ckpt: bool = True
    results_dir: str = ""  # 若为空则自动生成到 results_root/run_<ts>/pta_save
    ckpt_dir: str = ""  # 若为空则使用 results_dir/ckpts
    extra_args: str = ""


@dataclass(frozen=True)
class PtaLoadConfig:
    iterations: int = 2
    results_dir: str = ""  # 若为空则自动生成到 results_root/run_<ts>/pta_load
    ckpt: str = ""  # 若为空则从 save 阶段产生的 ckpt_dir 里推断最新 .pt
    extra_args: str = ""


@dataclass(frozen=True)
class MsaLoadConfig:
    enabled: bool = False


@dataclass(frozen=True)
class Config:
    global_cfg: GlobalConfig
    mutate: MutateConfig
    pta_save: PtaSaveConfig
    pta_load: PtaLoadConfig
    msa_load: MsaLoadConfig


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
    pta_save = _dc_from_dict(PtaSaveConfig, raw.get("pta_save", {}))
    pta_load = _dc_from_dict(PtaLoadConfig, raw.get("pta_load", {}))
    msa_load = _dc_from_dict(MsaLoadConfig, raw.get("msa_load", {}))
    return Config(global_cfg=global_cfg, mutate=mutate, pta_save=pta_save, pta_load=pta_load, msa_load=msa_load)


def mutate_one(cfg: Config, run_root: Path) -> Path:
    if cfg.mutate.output_dir:
        out_dir = _abs(cfg.mutate.output_dir, base=REPO_ROOT)
        use_subdir = False
    else:
        # 统一本次 run 的 mutate 根目录：results_root/run_xxx/mutate
        # 真实运行目录由 mutate.py 在此目录下再创建 mutate_<ts> 子目录
        out_dir = run_root / "mutate"
        use_subdir = True
    out_dir.mkdir(parents=True, exist_ok=True)

    args = [f"--rounds {int(cfg.mutate.rounds)}", f"--results-dir {shlex.quote(str(out_dir))}"]
    if cfg.mutate.extra_args:
        args.append(cfg.mutate.extra_args)
    bash_cmd = "\n".join(
        [
            _build_conda_activate_block(cfg.global_cfg.conda_env_pta),
            _build_ms_pythonpath_block(cfg.global_cfg.MindSpeed_Core_MS_PATH),
            f"bash mm_mutate.sh {' '.join(args)}",
        ]
    )

    mm_utils.log_step("MUTATE", "开始运行 mutate 脚本...", indent=1)
    mm_utils.log_bullet(f"run_root = {run_root}", indent=2)
    mm_utils.log_bullet(f"输出基目录: {out_dir}", indent=2)
    _run_bash(
        bash_cmd,
        cwd=MODULE_DIR,
        dry_run=cfg.global_cfg.dry_run,
        capture_output=cfg.global_cfg.capture_child_output,
    )
    mm_utils.log_step("MUTATE", "mutate 阶段完成。", indent=1)

    # mutate.py 会在 out_dir 下再创建 mutate_<ts> 子目录，configs 存在于该子目录下
    if use_subdir:
        run_dirs = [p for p in out_dir.iterdir() if p.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"no mutate run dir produced under: {out_dir}")
        # 选最近修改的一个作为本次运行目录
        run_dir = sorted(run_dirs, key=lambda p: p.stat().st_mtime, reverse=True)[0]
        mm_utils.log_bullet(f"本次 mutate 运行目录: {run_dir}", indent=2)
    else:
        run_dir = out_dir
    configs_dir = run_dir / "configs"

    if not cfg.global_cfg.dry_run:
        if not configs_dir.is_dir():
            raise FileNotFoundError(f"mutate configs dir not found: {configs_dir}")
        
        jsons = sorted(configs_dir.glob("*.json"))
        if not jsons:
            raise FileNotFoundError(f"no json config generated under: {configs_dir}")
        # rounds=1 期望 round_0.json；保险起见取最小文件名
        config_path = jsons[0].resolve()
        mm_utils.log_bullet(f"组合生成的模型配置 config_json: {config_path}", indent=2)

         # 打印 mutate 阶段生成的 DOT 文件路径（若存在）
        dot_path = (run_dir / "dots" / "graph_round0.dot").resolve()
        if dot_path.is_file():
            mm_utils.log_bullet(f"mutate 生成的 DOT 文件: {dot_path}", indent=2)
        
        return config_path
    return (configs_dir / "round_0.json").resolve()


def pta_save_ckpt(cfg: Config, config_json: Path, run_root: Path) -> Path:
    if cfg.pta_save.results_dir:
        results_dir = _abs(cfg.pta_save.results_dir, base=REPO_ROOT)
    else:
        results_dir = run_root / "pta_save"
    if cfg.pta_save.ckpt_dir:
        ckpt_dir = _abs(cfg.pta_save.ckpt_dir, base=REPO_ROOT)
    else:
        ckpt_dir = results_dir / "ckpts"
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    extra = []
    extra.append(f"--config {shlex.quote(str(config_json))}")
    extra.append(f"--iterations {int(cfg.pta_save.iterations)}")
    extra.append(f"--results-dir {shlex.quote(str(results_dir))}")
    extra.append(f"--ckpt-dir {shlex.quote(str(ckpt_dir))}")
    if cfg.pta_save.save_ckpt:
        extra.append("--save-ckpt")
    if cfg.pta_save.extra_args:
        extra.append(cfg.pta_save.extra_args)

    bash_cmd = "\n".join(
        [
            _build_conda_activate_block(cfg.global_cfg.conda_env_pta),
            _build_ms_pythonpath_block(cfg.global_cfg.MindSpeed_Core_MS_PATH),
            f"bash mm_test.sh {' '.join(extra)}",
        ]
    )
    mm_utils.log_step("PTA-SAVE", "开始运行 PTA save 脚本...", indent=1)
    mm_utils.log_bullet(f"结果目录: {results_dir}", indent=2)
    mm_utils.log_bullet(f"ckpt 目录: {ckpt_dir}", indent=2)
    mm_utils.log_bullet(f"config_json: {config_json}", indent=2)
    _run_bash(
        bash_cmd,
        cwd=MODULE_DIR,
        dry_run=cfg.global_cfg.dry_run,
        capture_output=cfg.global_cfg.capture_child_output,
    )
    mm_utils.log_step("PTA-SAVE", "PTA save 阶段完成。", indent=1)

    if cfg.global_cfg.dry_run:
        dummy_ckpt = (ckpt_dir / "DUMMY.pt").resolve()
        mm_utils.log_bullet(f"[DRY-RUN] 伪造 ckpt 路径: {dummy_ckpt}", indent=2)
        return dummy_ckpt

    pts = sorted(ckpt_dir.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not pts:
        raise FileNotFoundError(f"no checkpoint produced under: {ckpt_dir}")
    latest_ckpt = pts[0].resolve()
    mm_utils.log_bullet(f"选取的 ckpt: {latest_ckpt}", indent=2)
    return latest_ckpt


def pta_load_ckpt(cfg: Config, config_json: Path, ckpt_path: Path, run_root: Path) -> None:
    if cfg.pta_load.results_dir:
        results_dir = _abs(cfg.pta_load.results_dir, base=REPO_ROOT)
    else:
        results_dir = run_root / "pta_load"
    results_dir.mkdir(parents=True, exist_ok=True)

    extra = []
    extra.append(f"--config {shlex.quote(str(config_json))}")
    extra.append(f"--iterations {int(cfg.pta_load.iterations)}")
    extra.append(f"--results-dir {shlex.quote(str(results_dir))}")
    extra.append("--load-ckpt")
    extra.append(f"--ckpt {shlex.quote(str(ckpt_path))}")
    if cfg.pta_load.extra_args:
        extra.append(cfg.pta_load.extra_args)

    bash_cmd = "\n".join(
        [
            _build_conda_activate_block(cfg.global_cfg.conda_env_pta),
            _build_ms_pythonpath_block(cfg.global_cfg.MindSpeed_Core_MS_PATH),
            f"bash mm_test.sh {' '.join(extra)}",
        ]
    )
    mm_utils.log_step("PTA-LOAD", "开始运行 PTA load 脚本...", indent=1)
    mm_utils.log_bullet(f"结果目录: {results_dir}", indent=2)
    mm_utils.log_bullet(f"config_json: {config_json}", indent=2)
    mm_utils.log_bullet(f"ckpt: {ckpt_path}", indent=2)
    _run_bash(
        bash_cmd,
        cwd=MODULE_DIR,
        dry_run=cfg.global_cfg.dry_run,
        capture_output=cfg.global_cfg.capture_child_output,
    )
    mm_utils.log_step("PTA-LOAD", "PTA load 阶段完成。", indent=1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=str(MODULE_DIR / "task" / "module_combine_config.yaml"),
        help="配置文件路径（YAML）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的命令，不实际运行")
    args = parser.parse_args()

    cfg = load_config(_abs(args.config_yaml, base=REPO_ROOT))
    if args.dry_run:
        cfg = dataclasses.replace(cfg, global_cfg=dataclasses.replace(cfg.global_cfg, dry_run=True))

    results_root = _abs(cfg.global_cfg.results_root, base=REPO_ROOT)
    run_root = results_root / f"run_{_now_tag()}"
    run_root.mkdir(parents=True, exist_ok=True)
    mm_utils.log_section("Module Combination Mutate + PTA 全流程")
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
            config_json = mutate_one(cfg, round_root)
        except FileNotFoundError as e:
            mm_utils.log_step("ROUND", f"round {r} 变异失败（未找到生成的 config 文件），本 round 作废，跳过后续阶段", indent=1)
            mm_utils.log_bullet(str(e), indent=2)
            failed_rounds.append(r)
            continue

        ckpt = pta_save_ckpt(cfg, config_json, round_root)
        # 若 YAML 显式指定了 ckpt，则覆盖自动推断
        ckpt_to_load = _abs(cfg.pta_load.ckpt, base=REPO_ROOT) if cfg.pta_load.ckpt else ckpt
        pta_load_ckpt(cfg, config_json, ckpt_to_load, round_root)

        mm_utils.log_step("ROUND", f"round {r}/{total_rounds} 全部阶段完成")
        successful_rounds.append(r)

    mm_utils.log_section("Round 汇总")
    mm_utils.log_step("MAIN", f"成功执行的 round: {successful_rounds}")
    mm_utils.log_step("MAIN", f"失败的 round: {failed_rounds}")

    if cfg.msa_load.enabled:
        raise NotImplementedError("MSA load 阶段暂未实现（指南要求可省略）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

