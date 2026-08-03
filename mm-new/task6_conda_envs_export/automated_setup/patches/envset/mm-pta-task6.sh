#!/bin/bash
# Task6 PTA环境变量设置脚本
# 在使用mm-pta conda环境前source此脚本

# 动态定位 conda 并激活环境
CONDA_BASE=$(conda info --base 2>/dev/null)
if [ -z "${CONDA_BASE}" ]; then
    echo "错误: 未找到 conda 安装路径，请确认 conda 已安装并在 PATH 中"
    return 1
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${PTA_NAME:-mindspeed}"

# 设置CANN环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

# 设置基本环境变量
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=2
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ACLNN_CACHE_LIMIT=100000

# 修复 decord 兼容性 issues
export DECORD_CPU=0

# 创建 decord 修复补丁目录
DECORD_PATCH_DIR="/tmp/decord_patch"
mkdir -p $DECORD_PATCH_DIR

# 创建 decord 修复模块
cat > $DECORD_PATCH_DIR/decord_fix.py << 'EOF'
import sys
import warnings

# 尝试导入 decord 并修复
try:
    import decord
    # 如果 decord 没有 cpu 属性，添加一个空实现
    if not hasattr(decord, 'cpu'):
        decord.cpu = lambda: None
        warnings.warn("Patched decord.cpu() as empty function")
    # 确保 decord.cpu 可以被调用
    original_init = decord.__init__ if hasattr(decord, '__init__') else None
except Exception as e:
    warnings.warn(f"Failed to patch decord: {e}")
EOF

# 设置 PYTHONPATH 使修复模块优先加载
export PYTHONPATH="${DECORD_PATCH_DIR}:${PYTHONPATH}"

# 设置Python路径（必须从环境变量获取）
# MINDSPEED_MM_PATH 可为 workspace root 或 MindSpeed-MM 代码目录
if [ -n "$MINDSPEED_MM_PATH" ]; then
    # 自动判断：如果包含 pretrain_*.py 则为代码目录，否则为 workspace root
    if [ -f "${MINDSPEED_MM_PATH}/pretrain_vlm.py" ] || [ -f "${MINDSPEED_MM_PATH}/pretrain_sora.py" ]; then
        MM_PATH="$MINDSPEED_MM_PATH"
        WORKSPACE_ROOT=$(dirname "$MINDSPEED_MM_PATH")
    else
        WORKSPACE_ROOT="$MINDSPEED_MM_PATH"
        MM_PATH="${MINDSPEED_MM_PATH}/MindSpeed-MM"
    fi
elif [ -n "$PTAPATH" ]; then
    MM_PATH="$PTAPATH"
    WORKSPACE_ROOT="$PTAPATH"
elif [ -n "$PTA_PATH" ]; then
    MM_PATH="$PTA_PATH"
    WORKSPACE_ROOT="$PTA_PATH"
else
    echo "错误: 未设置 MINDSPEED_MM_PATH 或 PTA_PATH 环境变量"
    return 1
fi

# 推断 lm-sv 根目录（与 workspace root 同级的 lm-sv 目录）
PARENT=$(dirname "${WORKSPACE_ROOT}")
LM_SV_ROOT="${PARENT}/lm-sv"

# 检查 Megatron-LM（必须在 workspace root 下）
if [ ! -d "${WORKSPACE_ROOT}/Megatron-LM" ]; then
    echo "错误: WORKSPACE_ROOT (${WORKSPACE_ROOT}) 下缺少 Megatron-LM 目录"
    echo "请确认 config.json 中 MINDSPEED_MM_PATH 指向正确的工作区路径"
    return 1
fi

# MindSpeed 路径：优先从 lm-sv 下找，否则 fallback 到 workspace root 或 PTA_PATH
if [ -d "${LM_SV_ROOT}/mm-new/MindSpeed" ]; then
    MINDSPEED_PATH="${LM_SV_ROOT}/mm-new/MindSpeed"
elif [ -d "${WORKSPACE_ROOT}/MindSpeed" ]; then
    MINDSPEED_PATH="${WORKSPACE_ROOT}/MindSpeed"
elif [ -n "${PTA_PATH}" ] && [ -d "${PTA_PATH}/MindSpeed" ]; then
    MINDSPEED_PATH="${PTA_PATH}/MindSpeed"
elif [ -n "${PTAPATH}" ] && [ -d "${PTAPATH}/MindSpeed" ]; then
    MINDSPEED_PATH="${PTAPATH}/MindSpeed"
fi

# 设置 ffmpeg 库路径（decord 依赖）
if [ -n "${CONDA_PREFIX}" ] && [ -f "${CONDA_PREFIX}/lib/libavcodec.so" ]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
fi

# 设置 PYTHONPATH
PYTHONPATH_ENTRIES="${WORKSPACE_ROOT}/Megatron-LM"
if [ -n "$MINDSPEED_PATH" ]; then
    PYTHONPATH_ENTRIES="${PYTHONPATH_ENTRIES}:${MINDSPEED_PATH}"
fi
PYTHONPATH_ENTRIES="${PYTHONPATH_ENTRIES}:${MM_PATH}:${MM_PATH}/tmp/decord/python"
export PYTHONPATH=${PYTHONPATH_ENTRIES}:$PYTHONPATH

echo "PTA环境变量设置完成"
echo "MM_PATH: $MM_PATH"
echo "WORKSPACE_ROOT: $WORKSPACE_ROOT"
