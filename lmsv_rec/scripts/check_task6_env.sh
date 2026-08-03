#!/bin/bash
#
# Task6 环境对齐检查脚本
# 在 Worker（新机器）上运行，检查与 Master（老机器）的环境差异
#
# 用法:
#   ./check_task6_env.sh <MASTER_IP> [MASTER_USER] [SSH_PORT] [CONFIG_JSON]
#
# 示例:
#   ./check_task6_env.sh 192.168.0.170
#   ./check_task6_env.sh 192.168.0.170 root 22 /data2/lm-sv/lmsv_rec/config.json
#

set -e

MASTER_IP="${1}"
MASTER_USER="${2:-root}"
SSH_PORT="${3:-2222}"
CONFIG_JSON="${4:-/data2/lm-sv/lmsv_rec/config.json}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 帮助信息
if [ -z "$MASTER_IP" ]; then
    echo "=========================================="
    echo "Task6 环境对齐检查脚本"
    echo "=========================================="
    echo ""
    echo "用法: $0 <老机器IP> [用户名] [SSH端口] [config.json路径]"
    echo ""
    echo "示例:"
    echo "  $0 192.168.0.170"
    echo "  $0 192.168.0.170 root 22"
    echo "  $0 192.168.0.170 root 22 /data2/lm-sv/lmsv_rec/config.json"
    echo ""
    echo "前提条件:"
    echo "  1. Worker（新机器）能免密 SSH 到 Master（老机器）"
    echo "  2. 如果提供了 config.json，脚本会从中读取路径配置"
    echo ""
    echo "检查项:"
    echo "  - /data2/lm-sv/lmsv_rec/  (git commit 对比)"
    echo "  - /data2/dataset/          (目录存在性)"
    echo "  - /shared/mindspeed-mm/    (symlink 检查)"
    echo "  - Python / CANN / NPU 版本 (版本一致性)"
    echo ""
    exit 1
fi

# 从 config.json 读取路径
if [ -f "$CONFIG_JSON" ]; then
    echo "[配置] 从 $CONFIG_JSON 读取路径配置"
    LMSV_PATH=$(python3 -c "import json; d=json.load(open('$CONFIG_JSON')); nodes=d.get('tasks',{}).get('6',{}).get('MULTI_NODE',{}).get('OTHER_NODES',[]); print(nodes[0].get('LMSV_PATH', '/data2/lm-sv') if nodes else '/data2/lm-sv')" 2>/dev/null || echo "/data2/lm-sv")
    MINDSPEED_MM_PATH=$(python3 -c "import json; d=json.load(open('$CONFIG_JSON')); print(d.get('MINDSPEED_MM_PATH', '/shared/mindspeed-mm'))" 2>/dev/null || echo "/shared/mindspeed-mm")
    DATASET_ROOT=$(python3 -c "import json; d=json.load(open('$CONFIG_JSON')); print(d.get('DATASET_ROOT', '/data2/dataset'))" 2>/dev/null || echo "/data2/dataset")
else
    echo "[配置] 未找到 config.json，使用默认路径"
    LMSV_PATH="/data2/lm-sv"
    MINDSPEED_MM_PATH="/shared/mindspeed-mm"
    DATASET_ROOT="/data2/dataset"
fi

# 推断 lmsv_rec 路径
if [[ "$LMSV_PATH" == */lmsv_rec ]]; then
    LMSV_REC_PATH="$LMSV_PATH"
else
    LMSV_REC_PATH="$LMSV_PATH/lmsv_rec"
fi

# 推断 MindSpeed 路径
MM_PARENT=$(dirname "$MINDSPEED_MM_PATH")
MINDSPEED_PATH1="$MINDSPEED_MM_PATH/MindSpeed"
MINDSPEED_PATH2="$MM_PARENT/mm-new/MindSpeed"

echo ""
echo "=========================================="
echo "Task6 环境对齐检查"
echo "=========================================="
printf "老机器: ${BLUE}%s@%s:%s\e[0m\n" "$MASTER_USER" "$MASTER_IP" "$SSH_PORT"
printf "配置源: ${BLUE}%s\e[0m\n" "$CONFIG_JSON"
echo ""
echo "检查路径:"
printf "  LMSV_REC:     %s\n" "$LMSV_REC_PATH"
printf "  DATASET_ROOT: %s\n" "$DATASET_ROOT"
printf "  MINDSPEED_MM: %s\n" "$MINDSPEED_MM_PATH"
echo ""

# 1. 检查 SSH 连接
echo "[1/6] 检查 SSH 免密登录..."
if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p "${SSH_PORT}" "${MASTER_USER}@${MASTER_IP}" "echo OK" >/dev/null 2>&1; then
    printf "  \e[31m✗ 失败\e[0m: 无法免密 SSH 到老机器\n"
    echo ""
    echo "  请先在老机器上执行以下命令配置免密登录:"
    echo "    ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N \"\""
    echo "    ssh-copy-id -p ${SSH_PORT} ${MASTER_USER}@${MASTER_IP}"
    echo ""
    exit 1
fi
printf "  \e[32m✓ 通过\e[0m\n"

# 辅助函数: 检查老机器上的目录类型
check_remote_dir() {
    local path="$1"
    ssh -p "${SSH_PORT}" "${MASTER_USER}@${MASTER_IP}" "
        if [ -L '$path' ]; then
            echo 'SYMLINK'
        elif [ -d '$path' ]; then
            echo 'DIR'
        elif [ -e '$path' ]; then
            echo 'FILE'
        else
            echo 'NOT_FOUND'
        fi
    " 2>/dev/null
}

get_remote_git_commit() {
    local path="$1"
    ssh -p "${SSH_PORT}" "${MASTER_USER}@${MASTER_IP}" "cd '$path' 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo 'NO_GIT'" 2>/dev/null
}

get_remote_python_version() {
    ssh -p "${SSH_PORT}" "${MASTER_USER}@${MASTER_IP}" "python3 --version 2>&1 || echo 'NOT_FOUND'" 2>/dev/null
}

get_remote_cann_version() {
    ssh -p "${SSH_PORT}" "${MASTER_USER}@${MASTER_IP}" "cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | grep 'runtime_running_version' | head -1 || echo 'NOT_FOUND'" 2>/dev/null
}

get_remote_npu_driver() {
    ssh -p "${SSH_PORT}" "${MASTER_USER}@${MASTER_IP}" "npu-smi info 2>/dev/null | grep 'Version:' | head -1 || echo 'NOT_FOUND'" 2>/dev/null
}

# 辅助函数: 检查本地目录
check_local_dir() {
    local path="$1"
    if [ -L "$path" ]; then
        local target=$(readlink -f "$path" 2>/dev/null || readlink "$path" 2>/dev/null)
        if [ -e "$path" ]; then
            echo "SYMLINK_OK:$target"
        else
            echo "SYMLINK_BROKEN:$target"
        fi
    elif [ -d "$path" ]; then
        echo "DIR"
    elif [ -e "$path" ]; then
        echo "FILE"
    else
        echo "NOT_FOUND"
    fi
}

get_local_git_commit() {
    local path="$1"
    cd "$path" 2>/dev/null && git rev-parse --short HEAD 2>/dev/null || echo "NO_GIT"
}

get_local_python_version() {
    python3 --version 2>&1 || echo "NOT_FOUND"
}

get_local_cann_version() {
    cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | grep "runtime_running_version" | head -1 || echo "NOT_FOUND"
}

get_local_npu_driver() {
    npu-smi info 2>/dev/null | grep "Version:" | head -1 || echo "NOT_FOUND"
}

# 2. 检查 /data2/lm-sv/lmsv_rec/
echo ""
echo "[2/6] 检查 LMSV 代码仓库..."
LOCAL_LMSV=$(check_local_dir "$LMSV_REC_PATH")
REMOTE_LMSV=$(check_remote_dir "$LMSV_REC_PATH")

if [ "$LOCAL_LMSV" = "NOT_FOUND" ]; then
    printf "  \e[31m✗\e[0m 本地: %s 不存在\n" "$LMSV_REC_PATH"
    printf "  \e[32m✓\e[0m 老机器: 存在\n"
    LMSV_SYNC="rsync -avz --progress -e \"ssh -p ${SSH_PORT}\" ${MASTER_USER}@${MASTER_IP}:${LMSV_REC_PATH}/ ${LMSV_REC_PATH}/"
elif [ "$LOCAL_LMSV" = "DIR" ] && [ "$REMOTE_LMSV" = "DIR" ]; then
    LOCAL_COMMIT=$(get_local_git_commit "$LMSV_REC_PATH")
    REMOTE_COMMIT=$(get_remote_git_commit "$LMSV_REC_PATH")
    if [ "$LOCAL_COMMIT" = "$REMOTE_COMMIT" ]; then
        printf "  \e[32m✓\e[0m 本地: %s (commit: %s)\n" "$LMSV_REC_PATH" "$LOCAL_COMMIT"
        printf "  \e[32m✓\e[0m 老机器: %s (commit: %s)\n" "$LMSV_REC_PATH" "$REMOTE_COMMIT"
        printf "  \e[32m✓\e[0m git commit 一致\n"
        LMSV_SYNC=""
    else
        printf "  \e[33m!\e[0m 本地: %s (commit: %s)\n" "$LMSV_REC_PATH" "$LOCAL_COMMIT"
        printf "  \e[33m!\e[0m 老机器: %s (commit: %s)\n" "$LMSV_REC_PATH" "$REMOTE_COMMIT"
        printf "  \e[33m!\e[0m git commit 不一致\n"
        LMSV_SYNC="cd ${LMSV_REC_PATH} && git fetch origin && git reset --hard ${REMOTE_COMMIT}"
    fi
else
    printf "  \e[33m!\e[0m 本地: %s (类型: %s)\n" "$LMSV_REC_PATH" "$LOCAL_LMSV"
    printf "  \e[32m✓\e[0m 老机器: %s (类型: %s)\n" "$LMSV_REC_PATH" "$REMOTE_LMSV"
    LMSV_SYNC="rsync -avz --progress -e \"ssh -p ${SSH_PORT}\" ${MASTER_USER}@${MASTER_IP}:${LMSV_REC_PATH}/ ${LMSV_REC_PATH}/"
fi

# 3. 检查 /data2/dataset/
echo ""
echo "[3/6] 检查数据集目录..."
LOCAL_DATASET=$(check_local_dir "$DATASET_ROOT")
REMOTE_DATASET=$(check_remote_dir "$DATASET_ROOT")

if [ "$LOCAL_DATASET" = "NOT_FOUND" ]; then
    printf "  \e[31m✗\e[0m 本地: %s 不存在\n" "$DATASET_ROOT"
    printf "  \e[32m✓\e[0m 老机器: 存在\n"
    DATASET_SYNC="rsync -avz --progress -e \"ssh -p ${SSH_PORT}\" ${MASTER_USER}@${MASTER_IP}:${DATASET_ROOT}/ ${DATASET_ROOT}/"
elif [ "$LOCAL_DATASET" = "DIR" ] && [ "$REMOTE_DATASET" = "DIR" ]; then
    printf "  \e[32m✓\e[0m 本地: %s 存在\n" "$DATASET_ROOT"
    printf "  \e[32m✓\e[0m 老机器: %s 存在\n" "$DATASET_ROOT"
    DATASET_SYNC=""
else
    printf "  \e[33m!\e[0m 本地: %s (类型: %s)\n" "$DATASET_ROOT" "$LOCAL_DATASET"
    printf "  \e[32m✓\e[0m 老机器: %s (类型: %s)\n" "$DATASET_ROOT" "$REMOTE_DATASET"
    DATASET_SYNC="rsync -avz --progress -e \"ssh -p ${SSH_PORT}\" ${MASTER_USER}@${MASTER_IP}:${DATASET_ROOT}/ ${DATASET_ROOT}/"
fi

# 4. 检查 /shared/mindspeed-mm/
echo ""
echo "[4/6] 检查 MindSpeed-MM 相关目录..."

MM_ITEMS=("Megatron-LM" "MindSpeed-MM" "MindSpeed")
MM_SYNC_CMDS=""

for item in "${MM_ITEMS[@]}"; do
    local_path="${MINDSPEED_MM_PATH}/${item}"
    local_status=$(check_local_dir "$local_path")
    remote_status=$(check_remote_dir "$local_path")

    if [ "$local_status" = "NOT_FOUND" ]; then
        if [ "$remote_status" = "DIR" ]; then
            printf "  \e[31m✗\e[0m %s: 本地不存在，老机器存在\n" "$item"
            MM_SYNC_CMDS="${MM_SYNC_CMDS}
# 同步 ${item}
rsync -avz --progress -e \"ssh -p ${SSH_PORT}\" ${MASTER_USER}@${MASTER_IP}:${local_path}/ ${local_path}/"
        else
            printf "  \e[33m!\e[0m %s: 本地不存在，老机器也不存在（可能不需要）\n" "$item"
        fi
    elif [[ "$local_status" == SYMLINK_OK:* ]]; then
        target="${local_status#SYMLINK_OK:}"
        printf "  \e[33m!\e[0m %s: symlink -> %s\n" "$item" "$target"
        if [ "$remote_status" = "DIR" ]; then
            MM_SYNC_CMDS="${MM_SYNC_CMDS}
# 同步 ${item} (替换 symlink)
rm -f ${local_path}
mkdir -p ${local_path}
rsync -avz --progress -e \"ssh -p ${SSH_PORT}\" ${MASTER_USER}@${MASTER_IP}:${local_path}/ ${local_path}/"
        fi
    elif [[ "$local_status" == SYMLINK_BROKEN:* ]]; then
        target="${local_status#SYMLINK_BROKEN:}"
        printf "  \e[31m✗\e[0m %s: broken symlink -> %s\n" "$item" "$target"
        if [ "$remote_status" = "DIR" ]; then
            MM_SYNC_CMDS="${MM_SYNC_CMDS}
# 同步 ${item} (修复 broken symlink)
rm -f ${local_path}
mkdir -p ${local_path}
rsync -avz --progress -e \"ssh -p ${SSH_PORT}\" ${MASTER_USER}@${MASTER_IP}:${local_path}/ ${local_path}/"
        fi
    elif [ "$local_status" = "DIR" ] && [ "$remote_status" = "DIR" ]; then
        printf "  \e[32m✓\e[0m %s: 本地和老机器都是真实目录\n" "$item"
    elif [ "$local_status" = "DIR" ] && [ "$remote_status" = "NOT_FOUND" ]; then
        printf "  \e[32m✓\e[0m %s: 本地存在，老机器不存在（可能不需要）\n" "$item"
    else
        printf "  \e[33m!\e[0m %s: 本地(%s) vs 老机器(%s)\n" "$item" "$local_status" "$remote_status"
    fi
done

# 检查 /shared/mm-new/MindSpeed (如果 MINDSPEED_MM_PATH 不是 /shared/mindspeed-mm)
if [ "$MM_PARENT" != "$MINDSPEED_MM_PATH" ]; then
    alt_mindspeed="$MINDSPEED_PATH2"
    if [ -e "$alt_mindspeed" ] || [ -e "$(dirname "$MINDSPEED_MM_PATH")/mm-new" ]; then
        echo ""
        echo "  检查替代 MindSpeed 路径: $alt_mindspeed"
        local_status=$(check_local_dir "$alt_mindspeed")
        remote_status=$(check_remote_dir "$alt_mindspeed")
        if [[ "$local_status" == SYMLINK* ]] || [ "$local_status" = "NOT_FOUND" ]; then
            if [ "$remote_status" = "DIR" ]; then
                printf "  \e[31m✗\e[0m %s: 需要同步\n" "$alt_mindspeed"
                MM_SYNC_CMDS="${MM_SYNC_CMDS}
# 同步 MindSpeed (替代路径)
mkdir -p $(dirname "$alt_mindspeed")
rsync -avz --progress -e \"ssh -p ${SSH_PORT}\" ${MASTER_USER}@${MASTER_IP}:${alt_mindspeed}/ ${alt_mindspeed}/"
            fi
        fi
    fi
fi

# 5. 检查软件版本
echo ""
echo "[5/6] 检查软件版本..."

LOCAL_PY=$(get_local_python_version)
REMOTE_PY=$(get_remote_python_version)
printf "  Python:     本地=%s  老机器=%s\n" "$LOCAL_PY" "$REMOTE_PY"

LOCAL_CANN=$(get_local_cann_version)
REMOTE_CANN=$(get_remote_cann_version)
printf "  CANN:       本地=%s  老机器=%s\n" "$LOCAL_CANN" "$REMOTE_CANN"

LOCAL_NPU=$(get_local_npu_driver)
REMOTE_NPU=$(get_remote_npu_driver)
printf "  NPU Driver: 本地=%s  老机器=%s\n" "$LOCAL_NPU" "$REMOTE_NPU"

# 版本对比
VERSION_WARN=""
if [ "$LOCAL_PY" != "$REMOTE_PY" ]; then
    VERSION_WARN="${VERSION_WARN}
# Python 版本不一致: 本地=${LOCAL_PY}, 老机器=${REMOTE_PY}"
fi
if [ "$LOCAL_CANN" != "$REMOTE_CANN" ]; then
    VERSION_WARN="${VERSION_WARN}
# CANN 版本不一致: 本地=${LOCAL_CANN}, 老机器=${REMOTE_CANN}"
fi

# 6. 输出修复命令
echo ""
echo "=========================================="
echo "检查完成 - 修复命令"
echo "=========================================="

if [ -z "$LMSV_SYNC" ] && [ -z "$DATASET_SYNC" ] && [ -z "$MM_SYNC_CMDS" ] && [ -z "$VERSION_WARN" ]; then
    echo ""
    printf "\e[32m所有检查项通过！环境已对齐。\e[0m\n"
    echo ""
    exit 0
fi

echo ""
echo "# 请复制以下命令并在 Worker 上执行"
echo "# ============================================================"
echo ""

if [ -n "$LMSV_SYNC" ]; then
    echo "# 1. 同步 LMSV 代码仓库"
    echo "$LMSV_SYNC"
    echo ""
fi

if [ -n "$DATASET_SYNC" ]; then
    echo "# 2. 同步数据集"
    echo "$DATASET_SYNC"
    echo ""
fi

if [ -n "$MM_SYNC_CMDS" ]; then
    echo "# 3. 同步 MindSpeed-MM 相关目录"
    echo "$MM_SYNC_CMDS"
    echo ""
fi

if [ -n "$VERSION_WARN" ]; then
    echo "# 4. 版本警告（请手动处理）"
    echo "$VERSION_WARN"
    echo ""
fi

echo "# ============================================================"
echo "# 注意:"
echo "# 1. rsync 命令不会自动执行，请手动复制粘贴并确认"
echo "# 2. 数据集同步可能耗时较长（数十GB），建议在非业务时段执行"
echo "# 3. 同步前请确保 Worker 磁盘空间充足"
echo "# 4. Python/CANN 版本不一致需要手动安装对齐"
echo "# ============================================================"
echo ""
