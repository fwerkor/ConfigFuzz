#!/bin/bash
# Task6 环境搭建与定制化修改一键应用脚本
# =============================================================================
# 功能：对"裸"conda 环境完成以下操作：
#   1. 从原始 tar.gz 环境包中补充安装当前环境缺失的包
#   2. 应用源码级别的定制化修改（transformers 补丁（双环境）、msadapter bfloat16 fallback 等）
#   3. 部署环境脚本到 lmsv_rec/scripts/envset/
#
# 前置条件（起点）：
#   两个 conda 环境已创建，且仅安装了 requirements.txt 中声明的标准库：
#     - mindspeed  (PTA 环境)
#     - msadapter  (MSA 环境)
#   这些"裸"环境可从 ../standard_env/ 中通过 yml 还原
#
# 本脚本不做的事：
#   - 不创建 conda 环境
#   - 不降级/替换标准环境中已有的包（以标准环境版本为准）
#
# 用法：
#   cd task6_conda_envs_export/automated_setup
#   bash setup_task6_envs.sh
#
# 可选环境变量：
#   PTA_NAME       PTA 环境名，默认 mindspeed
#   MSA_NAME       MSA 环境名，默认 msadapter
#   MM_WORKSPACE   mm-new 工作区绝对路径，默认自动推断
#   LMSV_REC       lmsv_rec 项目绝对路径，默认自动推断
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCHES_DIR="${SCRIPT_DIR}/patches"

# 配置
MSA_NAME="${MSA_NAME:-msadapter}"
PTA_NAME="${PTA_NAME:-mindspeed}"

# MindSpeed-MM 工作区路径（优先从环境变量获取，否则自动推断）
MM_WORKSPACE="${MM_WORKSPACE:-}"

# apex 本地编译包路径（已从 tar.gz 提取到 patches 目录）
APEX_PATCH_DIR="${SCRIPT_DIR}/patches/apex/lib/python3.10/site-packages"

# ========================================================================
# 预定义：从 tar.gz 分析得出的缺失包列表
# 这些包在标准环境（requirements.txt）中缺失，但在原始 tar.gz 环境包中存在
# 安装原则：标准环境中已有的包保持版本不变，只补充安装缺失项
# ========================================================================

# mindspeed (PTA) 环境需补充安装的 PyPI 包
PTA_EXTRA_PACKAGES=(
    "annotated_types==0.7.0"
    "bitsandbytes_npu_beta==0.45.3"
    "click==8.3.1"
    "imageio==2.37.2"
    "imageio_ffmpeg==0.6.0"
    "jaxtyping==0.3.5"
    "jsonschema==4.26.0"
    "jsonschema_specifications==2025.9.1"
    "nvidia_ml_py==13.590.48"
    "prettytable==3.17.0"
    "pydantic==2.12.5"
    "pydantic_core==2.41.5"
    "pyecharts==2.0.9"
    "pyvers==0.2.2"
    "qwen_vl_utils==0.0.14"
    "ray==2.10.0"
    "referencing==0.37.0"
    "rpds_py==0.30.0"
    "simplejson==3.20.2"
    "swanlab==0.7.8"
    "typing_inspection==0.4.2"
    "wadler_lindig==0.1.7"
    "wrapt==2.1.1"
)

# msadapter (MSA) 环境需补充安装的 PyPI 包
MSA_EXTRA_PACKAGES=(
    "annotated_doc==0.0.4"
    "annotated_types==0.7.0"
    "click==8.3.1"
    "imageio==2.37.3"
    "jaxtyping==0.3.5"
    "nvidia_ml_py==13.590.48"
    "prettytable==3.17.0"
    "pydantic==2.12.5"
    "pydantic_core==2.41.5"
    "pyecharts==2.0.9"
    "qwen_vl_utils==0.0.14"
    "shellingham==1.5.4"
    "simplejson==3.20.2"
    "swanlab==0.7.8"
    "tornado==6.5.4"
    "typer==0.24.1"
    "typing_inspection==0.4.2"
    "wadler_lindig==0.1.7"
    "wrapt==2.1.1"
)

# 日志
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/patch_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $*"
    exit 1
}

warn() {
    log "WARNING: $*"
}

# ========================================================================
# 阶段 1：检查 conda 环境存在
# ========================================================================
check_conda_envs() {
    log "=== 阶段 1：检查 conda 环境和前置条件 ==="
    CONDA_BASE=$(conda info --base 2>/dev/null) || error "未找到 conda，请确认已安装"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    log "Conda 根目录: $CONDA_BASE"

    for env in "$PTA_NAME" "$MSA_NAME"; do
        if ! conda env list | grep -q "^${env} "; then
            error "conda 环境 '${env}' 不存在。请先创建裸环境：\n" \
                  "  conda env create -f ../standard_env/${env}_bare.yml -n ${env}\n" \
                  "或从 ../standard_env/README.md 中选择其他方式创建。"
        fi
        log "环境 '${env}' 存在"
    done

    # 检查关键包是否已安装
    for env in "$PTA_NAME" "$MSA_NAME"; do
        conda activate "$env"
        python -c "import transformers" 2>/dev/null || warn "环境 ${env} 未安装 transformers"
        python -c "import torch" 2>/dev/null || warn "环境 ${env} 未安装 torch"
    done

    log "前置条件检查通过"
}

# ========================================================================
# 阶段 2：推断 MindSpeed-MM 工作区路径
# ========================================================================
infer_mm_workspace() {
    log "=== 阶段 2：推断 MindSpeed-MM 工作区路径 ==="

    if [ -n "$MM_WORKSPACE" ]; then
        if [ -d "${MM_WORKSPACE}/MindSpeed" ] && [ -d "${MM_WORKSPACE}/Megatron-LM" ]; then
            log "使用指定的 MM_WORKSPACE: $MM_WORKSPACE"
            return
        else
            warn "指定的 MM_WORKSPACE (${MM_WORKSPACE}) 缺少 MindSpeed 或 Megatron-LM 子目录"
        fi
    fi

    # 尝试从脚本位置推断（automated_setup 位于 task6_conda_envs_export/）
    INFERRED="$(cd "${SCRIPT_DIR}/../.." && pwd)/mm-new"
    if [ -d "${INFERRED}/MindSpeed" ] && [ -d "${INFERRED}/Megatron-LM" ]; then
        MM_WORKSPACE="$INFERRED"
        log "自动推断 MM_WORKSPACE: $MM_WORKSPACE"
        return
    fi

    # 尝试从 MINDSPEED_MM_PATH 环境变量推断
    if [ -n "$MINDSPEED_MM_PATH" ]; then
        mm_path="$MINDSPEED_MM_PATH"
        # 如果指向 MindSpeed-MM 子目录，取父目录作为 workspace
        if [ -f "${mm_path}/pretrain_vlm.py" ] || [ -f "${mm_path}/pretrain_sora.py" ]; then
            mm_path=$(dirname "$mm_path")
        fi
        if [ -d "${mm_path}/MindSpeed" ] && [ -d "${mm_path}/Megatron-LM" ]; then
            MM_WORKSPACE="$mm_path"
            log "自动推断 MM_WORKSPACE: $MM_WORKSPACE (来自 MINDSPEED_MM_PATH)"
            return
        fi
    fi

    warn "未找到 MindSpeed-MM 工作区，msadapter bfloat16 补丁将跳过"
    MM_WORKSPACE=""
}

# ========================================================================}
# 阶段 2.5：安装缺失的包（直接安装预定义列表）
# ========================================================================
# 修改说明：
#   直接安装预定义的缺失包列表。这些包通过分析原始 tar.gz 与标准环境
#   的差异得出，已硬编码在脚本中。不需要在运行时对比 tar.gz。
#   安装原则：标准环境中已有的包保持版本不变，只补充安装缺失项。
# ========================================================================
install_missing_packages() {
    local env_name="$1"

    log "=== 阶段 2.5：在 ${env_name} 中安装缺失的包 ==="

    conda activate "$env_name"
    local python_cmd
    python_cmd="$(which python)"

    # 选择对应环境的包列表
    local -a packages
    if [[ "$env_name" == "$PTA_NAME" ]]; then
        packages=("${PTA_EXTRA_PACKAGES[@]}")
    else
        packages=("${MSA_EXTRA_PACKAGES[@]}")
    fi

    log "准备安装 ${#packages[@]} 个缺失包到 ${env_name}..."

    # 分批安装（每批 10 个，避免命令行过长）
    local batch_size=10
    local total=${#packages[@]}
    local installed=0
    local i=0

    while [[ $i -lt $total ]]; do
        local batch=()
        local j=0
        while [[ $j -lt $batch_size && $((i + j)) -lt $total ]]; do
            batch+=("${packages[$((i + j))]}")
            j=$((j + 1))
        done

        log "安装批次 $((i / batch_size + 1)): ${batch[*]}"
        local pip_log="${LOG_DIR}/pip_${env_name}_batch_$((i / batch_size + 1)).log"
        if "$python_cmd" -m pip install "${batch[@]}" > "$pip_log" 2>&1; then
            installed=$((installed + ${#batch[@]}))
            log "  批次 $((i / batch_size + 1)) 安装成功"
        else
            warn "批次 $((i / batch_size + 1)) 安装可能部分失败，日志: ${pip_log}"
            # 输出错误日志的最后 5 行
            tail -5 "$pip_log" | while IFS= read -r line; do
                log "  PIP: $line"
            done
        fi

        i=$((i + batch_size))
    done

    log "${env_name} PyPI 包安装完成: ${installed}/${total}"

    # 特殊包：msadapter 环境安装 torch_npu
    if [[ "$env_name" == "$MSA_NAME" ]]; then
        log "安装 torch_npu 到 ${env_name}..."
        local torch_npu_url="https://gitcode.com/Ascend/pytorch/releases/download/v7.3.0-pytorch2.7.1/torch_npu-2.7.1.post2-cp310-cp310-manylinux_2_28_aarch64.whl"
        local pip_log="${LOG_DIR}/pip_${env_name}_torch_npu.log"
        if "$python_cmd" -m pip install "$torch_npu_url" > "$pip_log" 2>&1; then
            log "torch_npu 安装成功"
        else
            warn "torch_npu 安装可能失败，日志: ${pip_log}"
        fi
    fi

    # 特殊包：mindspeed 环境从 patches 目录复制 apex
    if [[ "$env_name" == "$PTA_NAME" ]]; then
        if [[ -d "$APEX_PATCH_DIR/apex" ]]; then
            log "从 patches 目录复制 apex 到 ${env_name}..."
            install_apex_from_patches "$env_name"
        else
            warn "apex patches 目录不存在，跳过 apex 安装: $APEX_PATCH_DIR"
        fi
    fi

    log "${env_name} 缺失包安装阶段完成"
}

# ========================================================================
# 从 patches 目录复制 apex 到指定 conda 环境
# ========================================================================
install_apex_from_patches() {
    local env_name="$1"

    conda activate "$env_name"
    local site_packages
    site_packages=$(python -c "import site; print(site.getsitepackages()[0])")

    log "复制 apex 到: $site_packages"

    # 复制 apex 目录和 dist-info
    if [[ -d "$APEX_PATCH_DIR/apex" ]]; then
        cp -r "$APEX_PATCH_DIR/apex" "$site_packages/"
        log "apex 目录复制完成"
    fi
    if [[ -d "$APEX_PATCH_DIR/apex-0.1+ascend.dist-info" ]]; then
        cp -r "$APEX_PATCH_DIR/apex-0.1+ascend.dist-info" "$site_packages/"
        log "apex dist-info 复制完成"
    fi

    log "apex 安装完成"
}

# ========================================================================
# 阶段 3：应用 transformers 兼容性补丁（双环境）
# ========================================================================
# 修改说明：
#   transformers modeling_utils.py 的 _load_state_dict_into_meta_model 调用中，
#   expected_keys 和 reverse_key_renaming_mapping 使用位置参数传参。
#   在 msadapter 环境下，其 decorator 会与这种传参方式冲突，导致
#   "got multiple values for argument 'device_map'" 错误。
#   需将第 4、5 个位置参数改为关键字参数。
#   注：实测 transformers 4.55.2 仍需此补丁。
# ========================================================================
apply_transformers_patch() {
    local env_name="$1"
    log "=== 阶段 3：在 ${env_name} 中应用 transformers 兼容性补丁 ==="

    conda activate "$env_name"
    local site_packages
    site_packages=$(python -c "import site; print(site.getsitepackages()[0])")

    local modeling_utils="${site_packages}/transformers/modeling_utils.py"
    if [[ ! -f "$modeling_utils" ]]; then
        warn "未找到 transformers modeling_utils.py，跳过"
        return 1
    fi

    # 检查 transformers 版本
    local tf_version
    tf_version=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
    log "检测到 transformers 版本: $tf_version"

    # 检查是否已打过补丁（关键字参数形式）
    if grep -q "expected_keys=expected_keys" "$modeling_utils" 2>/dev/null; then
        log "transformers 补丁已存在（关键字参数形式），跳过"
        return 0
    fi

    # 检查是否已经是新版函数签名（4.55.2+），无需补丁
    if grep -q "def _load_state_dict_into_meta_model" "$modeling_utils" 2>/dev/null; then
        local call_line
        call_line=$(grep -n "_load_state_dict_into_meta_model" "$modeling_utils" | grep -v "def " | head -1 || true)
        if [[ -n "$call_line" ]] && echo "$call_line" | grep -q "expected_keys=expected_keys"; then
            log "transformers ${tf_version} 已使用关键字参数，无需补丁"
            return 0
        fi
    fi

    # 使用 patch 命令应用补丁（精确匹配，避免 sed 误伤函数定义）
    local patch_file="${PATCHES_DIR}/transformers/transformers_4.55.2_modeling_utils.patch"
    if [[ -f "$patch_file" ]]; then
        cd "$site_packages" || return 1
        if patch --dry-run -p1 < "$patch_file" >/dev/null 2>&1; then
            patch -p1 < "$patch_file" >/dev/null 2>&1
            log "已应用 transformers 补丁（patch 命令），版本 ${tf_version}"
        else
            warn "patch 命令 dry-run 失败，尝试 sed fallback"
            # 精确 fallback：只替换 _load_state_dict_into_meta_model( 调用后的参数
            python3 -c "
import re, sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
# 只替换函数调用块内的 expected_keys, 和 reverse_key_renaming_mapping,
pattern = r'(disk_offload_index, cpu_offload_index = _load_state_dict_into_meta_model\(\n            model_to_load,\n            state_dict,\n            shard_file,\n)(            expected_keys,\n)(            reverse_key_renaming_mapping,\n)(            device_map=device_map,)'
repl = r'\1            expected_keys=expected_keys,\n            reverse_renaming_mapping=reverse_key_renaming_mapping,\n\4'
new_text = re.sub(pattern, repl, text)
if new_text != text:
    with open(path, 'w') as f:
        f.write(new_text)
    print('sed fallback applied')
else:
    print('sed fallback: pattern not found')
" "$modeling_utils"
        fi
    else
        warn "未找到 patch 文件 ${patch_file}"
        return 1
    fi

    # 验证
    if grep -q "expected_keys=expected_keys" "$modeling_utils" 2>/dev/null; then
        log "transformers 补丁验证通过"
        return 0
    else
        warn "transformers 补丁可能未完全应用，请手动检查 ${modeling_utils}"
        return 1
    fi
}

# ========================================================================
# 阶段 4：应用 msadapter bfloat16 fallback 补丁
# ========================================================================
# 修改说明：
#   MindSpore 内置的 np_dtype.bfloat16 在某些版本/设备上不可用。
#   当 MSA 侧加载包含 bfloat16 权重的模型（如 InternVL3）时，会因找不到
#   bfloat16 类型而崩溃。本补丁在 _utils.py 和 serialization.py 中增加
#   fallback 逻辑：当 MindSpore 原生 bfloat16 不可用时，自动回退到
#   ml_dtypes.bfloat16。
# ========================================================================
apply_msadapter_bfloat16_patch() {
    log "=== 阶段 4：应用 msadapter bfloat16 fallback 补丁 ==="

    if [[ -z "$MM_WORKSPACE" ]]; then
        warn "MM_WORKSPACE 未设置，跳过 msadapter bfloat16 补丁"
        return 1
    fi

    local target_utils="${MM_WORKSPACE}/msadapter/msadapter/_utils.py"
    local target_serialization="${MM_WORKSPACE}/msadapter/msadapter/serialization.py"
    local patch_utils="${PATCHES_DIR}/msadapter/_utils.py"
    local patch_serialization="${PATCHES_DIR}/msadapter/serialization.py"

    if [[ ! -f "$target_utils" ]]; then
        warn "未找到 ${target_utils}，跳过 bfloat16 补丁"
        return 1
    fi
    if [[ ! -f "$target_serialization" ]]; then
        warn "未找到 ${target_serialization}，跳过 bfloat16 补丁"
        return 1
    fi

    # 备份原文件（如未备份过）
    if [[ ! -f "${target_utils}.bak" ]]; then
        cp "$target_utils" "${target_utils}.bak"
        log "已备份 ${target_utils} -> ${target_utils}.bak"
    fi
    if [[ ! -f "${target_serialization}.bak" ]]; then
        cp "$target_serialization" "${target_serialization}.bak"
        log "已备份 ${target_serialization} -> ${target_serialization}.bak"
    fi

    # 应用补丁（直接覆盖为修改后的版本）
    cp "$patch_utils" "$target_utils"
    log "已覆盖 _utils.py（增加 ml_dtypes.bfloat16 fallback）"

    cp "$patch_serialization" "$target_serialization"
    log "已覆盖 serialization.py（增加 ml_dtypes.bfloat16 fallback）"

    # 验证
    local ok=true
    if grep -q "ml_dtypes.bfloat16" "$target_utils" 2>/dev/null; then
        log "[PASS] _utils.py 包含 ml_dtypes.bfloat16 fallback"
    else
        warn "[FAIL] _utils.py 未找到 ml_dtypes.bfloat16 fallback"
        ok=false
    fi
    if grep -q "ml_dtypes.bfloat16" "$target_serialization" 2>/dev/null; then
        log "[PASS] serialization.py 包含 ml_dtypes.bfloat16 fallback"
    else
        warn "[FAIL] serialization.py 未找到 ml_dtypes.bfloat16 fallback"
        ok=false
    fi

    $ok && return 0 || return 1
}

# ========================================================================
# 阶段 5：部署环境脚本
# ========================================================================
# 修改说明：
#   将包含运行时补丁的环境脚本部署到 lmsv_rec/scripts/envset/
#   - mm-pta-task6.sh：包含 decord 运行时补丁
#   - mm-msa-task6.sh：包含 libstdc++ 兼容性修复
# ========================================================================
deploy_env_scripts() {
    log "=== 阶段 5：部署环境脚本 ==="

    local lmsv_rec="${LMSV_REC:-}"

    # 自动推断 lmsv_rec 路径
    if [[ -z "$lmsv_rec" ]]; then
        local inferred="$(cd "${SCRIPT_DIR}/../../lmsv_rec" 2>/dev/null && pwd)"
        if [[ -d "${inferred}/scripts/envset" ]]; then
            lmsv_rec="$inferred"
        fi
    fi

    if [[ -n "$lmsv_rec" ]] && [[ -d "${lmsv_rec}/scripts/envset" ]]; then
        cp "${PATCHES_DIR}/envset/mm-pta-task6.sh" "${lmsv_rec}/scripts/envset/mm-pta-task6.sh"
        cp "${PATCHES_DIR}/envset/mm-msa-task6.sh" "${lmsv_rec}/scripts/envset/mm-msa-task6.sh"
        chmod +x "${lmsv_rec}/scripts/envset/mm-pta-task6.sh"
        chmod +x "${lmsv_rec}/scripts/envset/mm-msa-task6.sh"
        log "环境脚本已部署到 ${lmsv_rec}/scripts/envset/"
    else
        warn "未找到 lmsv_rec/scripts/envset/ 目录"
        log "环境脚本保留在 ${PATCHES_DIR}/envset/，请手动复制到正确位置"
        chmod +x "${PATCHES_DIR}/envset/mm-pta-task6.sh"
        chmod +x "${PATCHES_DIR}/envset/mm-msa-task6.sh"
    fi
}

# ========================================================================
# 阶段 6：验证所有修改
# ========================================================================
verify_all() {
    local env_name="$1"
    log "=== 阶段 6：验证 ${env_name} 环境修改 ==="

    conda activate "$env_name"
    local site_packages
    site_packages=$(python -c "import site; print(site.getsitepackages()[0])")

    local ok=true

    # 验证 transformers 补丁
    local modeling_utils="${site_packages}/transformers/modeling_utils.py"
    local tf_version
    tf_version=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null || echo "unknown")
    if [[ -f "$modeling_utils" ]]; then
        if grep -q "expected_keys=expected_keys" "$modeling_utils" 2>/dev/null && \
             grep -q "reverse_renaming_mapping=reverse_key_renaming_mapping" "$modeling_utils" 2>/dev/null; then
            log "[PASS] transformers ${tf_version} 兼容性补丁已应用"
        else
            warn "[FAIL] transformers 兼容性补丁未找到"
            ok=false
        fi
    else
        warn "[SKIP] transformers 未安装"
    fi

    # 验证 msadapter bfloat16 补丁
    if [[ -n "$MM_WORKSPACE" ]]; then
        local target_utils="${MM_WORKSPACE}/msadapter/msadapter/_utils.py"
        local target_serialization="${MM_WORKSPACE}/msadapter/msadapter/serialization.py"
        if [[ -f "$target_utils" ]] && grep -q "ml_dtypes.bfloat16" "$target_utils" 2>/dev/null; then
            log "[PASS] msadapter _utils.py bfloat16 fallback 已应用"
        else
            warn "[FAIL] msadapter _utils.py bfloat16 fallback 未找到"
            ok=false
        fi
        if [[ -f "$target_serialization" ]] && grep -q "ml_dtypes.bfloat16" "$target_serialization" 2>/dev/null; then
            log "[PASS] msadapter serialization.py bfloat16 fallback 已应用"
        else
            warn "[FAIL] msadapter serialization.py bfloat16 fallback 未找到"
            ok=false
        fi
    else
        log "[SKIP] msadapter bfloat16 补丁（MM_WORKSPACE 未设置）"
    fi

    # 验证环境脚本
    local env_script
    if [[ "$env_name" == "$PTA_NAME" ]]; then
        env_script="${PATCHES_DIR}/envset/mm-pta-task6.sh"
    else
        env_script="${PATCHES_DIR}/envset/mm-msa-task6.sh"
    fi
    if [[ -f "$env_script" ]]; then
        log "[PASS] 环境脚本存在: $(basename "$env_script")"
    else
        warn "[FAIL] 环境脚本缺失: $(basename "$env_script")"
        ok=false
    fi

    if $ok; then
        log "${env_name} 验证全部通过"
        return 0
    else
        warn "${env_name} 验证存在失败项，请检查日志"
        return 1
    fi
}

# ========================================================================
# 主流程
# ========================================================================
main() {
    log "========================================"
    log "Task6 定制化修改一键应用开始"
    log "========================================"
    log "PTA 环境: $PTA_NAME"
    log "MSA 环境: $MSA_NAME"
    log "PATCHES_DIR: $PATCHES_DIR"
    log ""
    log "前置条件："
    log "  1. 两个 conda 环境已创建（仅含 requirements.txt 中的标准库）"
    log "  2. 可从 ../standard_env/ 还原裸环境"
    log ""

    check_conda_envs
    infer_mm_workspace

    # MSA 环境：安装缺失包 + transformers 补丁
    log "----------------------------------------"
    log "处理 MSA 环境: $MSA_NAME"
    log "----------------------------------------"
    install_missing_packages "$MSA_NAME"
    apply_transformers_patch "$MSA_NAME"
    verify_all "$MSA_NAME"

    # msadapter bfloat16 补丁（作用于 mm-new 源码，两个环境共用）
    apply_msadapter_bfloat16_patch

    # 部署环境脚本（两个环境共用）
    deploy_env_scripts

    # PTA 环境：安装缺失包 + transformers 补丁 + 验证
    log "----------------------------------------"
    log "处理 PTA 环境: $PTA_NAME"
    log "----------------------------------------"
    install_missing_packages "$PTA_NAME"
    apply_transformers_patch "$PTA_NAME"
    verify_all "$PTA_NAME"

    log "========================================"
    log "Task6 环境搭建与定制化修改一键应用完成"
    log "日志文件: $LOG_FILE"
    log "========================================"
    log ""
    log "使用说明："
    log "  1. PTA 环境执行前: source <envset>/mm-pta-task6.sh"
    log "  2. MSA 环境执行前: source <envset>/mm-msa-task6.sh"
    log ""
    log "注意："
    log "  - decord 运行时补丁和 libstdc++ 兼容性修复在 source 环境脚本时自动生效"
    log "  - 如使用非标准路径，请设置 MM_WORKSPACE、LMSV_REC 环境变量"
    log "  - 脚本仅从预定义列表安装缺失包，不会降级已有包"
}

main "$@"
