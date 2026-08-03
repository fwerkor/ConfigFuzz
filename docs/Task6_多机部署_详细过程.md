# Task6 多机部署详细过程文档

> **创建时间**: 2026-04-25
> **更新时间**: 2026-04-29
> **状态**: 4模型跨机验证通过（cogvideox / internvl3 / opensora / qwenvl）
> **机器**: old-server (192.168.0.170) + new-server (192.168.0.203)

---

## 一、当前整体进展

| 模型 | 类型 | PTA跨机 | MSA跨机 | 状态 |
|------|------|---------|---------|------|
| CogVideoX | 训练 | 通过 | 通过 | 精度diff ~1.5% |
| InternVL3 | 训练 | 通过 | 通过 | 精度diff ~20% |
| OpenSora | 推理 | 通过 | 预期失败(UntypedStorage) | 已复现已知bug |
| QwenVL | 推理 | 通过 | 预期失败(InnerInplaceIndexPut) | 已复现已知bug |

---

## 二、架构设计

### 2.1 多机部署模式

Task6采用 **SSH + rsync + ThreadPoolExecutor** 并发启动本地/远程任务:

```
old-server (192.168.0.170, MASTER, NODE_RANK=0)
  ├─ PTA: torchrun --nnodes 2 --node_rank 0
  └─ MSA: msrun --worker_num 16 --local_worker_num 8 --node_rank 0

new-server (192.168.0.203, NODE_RANK=1)
  ├─ PTA: torchrun --nnodes 2 --node_rank 1
  └─ MSA: msrun --worker_num 16 --local_worker_num 8 --node_rank 1
```

### 2.2 统一入口

所有 Task 统一通过 `./lmsv` 脚本执行：

```bash
./lmsv conf    # 交互式生成/修改 config.json
./lmsv do      # 执行当前 config.json 中 task_type 对应的任务
./lmsv help    # 查看帮助
```

`lmsv` 调用 `lmsv.py`，`lmsv.py` 调用 `do.py` 执行任务。

### 2.3 路径映射

| 本地路径 | 远程映射路径 |
|---------|------------|
| `/data2/lm-sv/lmsv_rec` | `/zyl/lm-sv/lmsv_rec` |
| `/data2/dataset` | `/zyl/dataset` |
| `/shared/mindspeed-mm` | `/zyl/mindspeed-mm` |

路径映射通过 `config.json` 中 Master/Worker 路径对动态推导，**不硬编码 `/zyl`**。

---

## 三、核心修改文件

```
/data2/lm-sv/lmsv_rec/utils/task/task6.py              # 多机支持核心实现
/data2/lm-sv/lmsv_rec/utils/analyze/task6_result.py    # 报告生成（精度/显存/性能判定）
/data2/lm-sv/lmsv_rec/scripts/runtime/mm_pta_*.sh      # PTA 多机wrapper脚本
/data2/lm-sv/lmsv_rec/scripts/runtime/mm_msa_*.sh      # MSA 多机wrapper脚本
/data2/lm-sv/lmsv_rec/scripts/runtime/pta_*_real.sh    # PTA 实际执行脚本
/data2/lm-sv/lmsv_rec/scripts/runtime/msa_*_real.sh    # MSA 实际执行脚本
```

---

## 四、已修复的关键问题

### 4.1 SSH_PORT 未使用（2026-04-29）

**问题**: `config.json` 中配置了 `SSH_PORT`，但代码中 SSH/rsync 命令未使用 `-p` 参数。

**修复**: `task6.py` 中所有 SSH/rsync 命令均读取 `node.get('SSH_PORT', 22)`：
- `_run_remote_shell()`: SSH 命令添加 `-p {port}`
- `sync_iteration_to_remote_nodes()`: mkdir/rsync 命令均支持端口
- `_clean_ports_and_processes()`: 远程清理命令支持端口

### 4.2 TASK6_ITER_OUTPUT_DIR 路径错误（2026-04-29）

**问题**: 推理模型样本被保存到 `iter_N/weights/samples/` 而非 `iter_N/samples/`。

**修复**: `task6.py` 中 `TASK6_ITER_OUTPUT_DIR` 改为 `.parent.parent`，远程函数也新增该环境变量传递。

### 4.3 MSA wrapper sed 修改 JSON 不可靠（2026-04-29）

**问题**: `mm_msa_opensora.sh` 使用 sed 修改 `save_path`，可能破坏 JSON 格式。

**修复**: 统一使用 Python `json.load`/`json.dump` 修改配置。

### 4.4 推理模型 PTA 被误判失败（2026-04-29）

**问题**: `pta_opensora_real.sh` 检查 `[ -z "$LOSS" ]`，但推理模型无 loss。

**修复**: 改为仅检查 `EXIT_CODE`。

### 4.5 MINDSPEED_MM_PATH 未传递（2026-04-28）

**问题**: `_init_config()` 未将 `MINDSPEED_MM_PATH` 从 `raw_node` 复制到 `normalized_nodes`。

**修复**: `_init_config()` 和 `_build_path_prefix_mappings()` 均添加 `MINDSPEED_MM_PATH` 处理。

### 4.6 精度/显存/性能判定错误归类（2026-04-29）

**问题**: MSA 执行失败（无 loss 数据）时被错误归类为"精度问题"。

**修复**:
- `task6.py`: 一方无数据时，不将 `accuracy_ok` 设为 False
- `task6_result.py`: 报告生成层也做同样修正

### 4.7 HCCL_DETERMINISTIC 多机死锁（已修复）

**修复**: 仅在单机模式设置 `HCCL_DETERMINISTIC=true`。

### 4.8 MSA 日志选择 bug（已修复）

**修复**: 选择包含最多 loss 行的 worker 日志，而非固定 worker_0。

### 4.9 NFS 竞争清除 msrun_log（已修复）

**修复**: 仅在 master 节点（NODE_RANK=0）清空 msrun_log。

---


### 4.11 网络接口名硬编码（2026-04-29）

**问题**: `GLOO_SOCKET_IFNAME` 和 `HCCL_SOCKET_IFNAME` 在代码中硬编码为 `enp67s0f5`，客户环境网卡名不同时出现 `RuntimeError: Unable to find address for: enp67s0f5`。

**修复文件**: `utils/task/task6.py`
- 新增 `Config.GLOO_SOCKET_IFNAME` 和 `Config.HCCL_SOCKET_IFNAME`，默认值为 `"enp67s0f5"`
- `_init_config()`: 从 `config.json` 的 `MULTI_NODE.GLOO_SOCKET_IFNAME` 和 `MULTI_NODE.HCCL_SOCKET_IFNAME` 读取
- 替换代码中 6 处硬编码的 `enp67s0f5` 为 Config 变量

**客户配置**: 在 `config.json` 的 `MULTI_NODE` 段添加：
```json
"GLOO_SOCKET_IFNAME": "enp67s0f5",
"HCCL_SOCKET_IFNAME": "enp67s0f5"
```

### 4.12 pgrep 自匹配 Bug（2026-04-29）

**问题**: `msa_cogvideox_real.sh` 中 `pgrep -f "pretrain_sora.py"` 在 SSH/bash wrapper 执行时匹配到自身进程，导致无限等待。

**修复文件**: `scripts/runtime/msa_cogvideox_real.sh`
- 替换为 `ps aux | grep "[p]retrain_sora.py"`


## 五、环境配置备忘

### 5.1 本地节点 (192.168.0.170)

```bash
export HCCL_IF_IP=192.168.0.170
export HCCL_SOCKET_IFNAME=enp67s0f5
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_IF_BASE_PORT=61000
export HCCL_CONNECT_TIMEOUT=1200
export ENABLE_OVERLAP=""  # 多机必须关闭
```

### 5.2 远程节点 (192.168.0.203)

```bash
export HCCL_IF_IP=192.168.0.203
export HCCL_SOCKET_IFNAME=enp67s0f5
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_IF_BASE_PORT=61000
export HCCL_CONNECT_TIMEOUT=1200
export ENABLE_OVERLAP=""
```

### 5.3 端口预留

```bash
sysctl -w net.ipv4.ip_local_reserved_ports="61000-61015"
```

---

## 六、日志路径

| 日志 | 路径 |
|-----|------|
| PTA运行日志 | `output/YYYYMMDD_HHMMSS/iters/iter_N/runtime_logs/pta_verify_iterN.log` |
| MSA运行日志 | `output/YYYYMMDD_HHMMSS/iters/iter_N/runtime_logs/msa_verify_iterN.log` |
| msrun worker日志 | `output/YYYYMMDD_HHMMSS/iters/iter_N/msrun_log/worker_*.log` |
| 任务汇总报告 | `output/YYYYMMDD_HHMMSS/analysis/summary.md` |

---

## 七、注意事项

1. **不要同时运行多个 Task6 进程**
2. **每次启动前 kill 残留进程** (已实现到代码中)
3. **多机模式下必须关闭 ENABLE_OVERLAP**，否则 HCCL 死锁
4. **不要设置 ASCEND_LAUNCH_BLOCKING=1**，会导致多机分布式 hang
5. **定期清理 `/dev/shm/psm_*`**，避免 NPU 内存分配失败
6. **远程节点的环境变更必须同步** (conda 包、sysctl 配置等)

---

## 七、新老机器环境差异对比（必须同步检查）

> **客户要求**：主从机文件结构必须完全相同。以下是在测试环境(old-server 192.168.0.170 + new-server 192.168.0.203)中发现的实际差异，客户部署前必须逐项确认。

### 7.1 `/shared/mindspeed-mm/` 目录结构差异（关键）

| 项目 | 老机器 (192.168.0.170) | 新机器 (192.168.0.203) | 影响 |
|------|----------------------|----------------------|------|
| Megatron-LM | 真实目录 | symlink → `/zyl/lm-sv/mm-new/Megatron-LM` | 客户环境需为真实目录或相同symlink |
| MindSpeed-MM | 真实目录 | symlink → `/zyl/mindspeed-mm/MindSpeed-MM` | 同上 |
| MindSpeed | 不存在 | symlink → `/zyl/lm-sv/mm-new/MindSpeed` | 训练/推理均需此目录 |

**客户行动项**：确保新机器 `/shared/mindspeed-mm/` 下三个目录（Megatron-LM、MindSpeed、MindSpeed-MM）与老机器完全一致。如果是symlink，目标路径也必须存在且内容相同。

### 7.2 `/shared/` 根目录差异

| 项目 | 老机器 | 新机器 |
|------|--------|--------|
| lm-sv | symlink → `/data2/lm-sv` | 真实目录 |
| mm-new | 真实目录 | symlink → `/zyl/lm-sv/mm-new` |
| 其他目录 | .claude, gpt_dataset, lmsv, offload, results, tmp 等 | 不存在 |

**说明**：Task6 核心只依赖 `mindspeed-mm` 和 `lm-sv/lmsv_rec`，其他目录不影响。但客户应保证两边一致。

### 7.3 Python 版本差异

| 项目 | 老机器 | 新机器 |
|------|--------|--------|
| Python | 3.13.9 | 3.10.20 |

**说明**：当前测试跨机cogvideox能跑通，但客户环境建议保持Python版本一致，避免潜在兼容性问题。

### 7.4 NPU 驱动版本差异

| 项目 | 老机器 | 新机器 |
|------|--------|--------|
| npu-smi | 24.1.0.3 | 23.0.6 |
| Driver Version | 24.1.0.3 | 24.1.0.3 |

**说明**：npu-smi工具版本不同但Driver Version相同，不影响功能。客户环境建议统一。

### 7.5 CANN 版本（一致）

| 项目 | 老机器 | 新机器 |
|------|--------|--------|
| CANN | 8.3.0.1.200:8.3.RC1 | 8.3.0.1.200:8.3.RC1 |

### 7.6 `/data2/dataset/`（完全一致）

两边目录结构、文件大小、修改时间完全相同。无需处理。

### 7.7 `/data2/lm-sv/lmsv_rec/`（完全一致）

两边为相同git仓库，相同commit。无需处理。

---

## 八、必须向客户说明的部署要点

1. **主从机 `/shared/mindspeed-mm/` 目录结构必须相同**，包含 Megatron-LM、MindSpeed、MindSpeed-MM 三个子目录
2. **主从机 `/data2/lm-sv/lmsv_rec/` 必须为相同git仓库的相同commit**
3. **主从机 `/data2/dataset/` 必须完全相同**
4. **Python版本建议一致**（当前测试环境老机器3.13.9、新机器3.10.20，能跑通但不推荐）
5. **NPU驱动版本建议一致**（npu-smi工具版本可不同，但Driver Version必须相同）
6. **CANN版本必须一致**
7. **不要同时运行多个 Task6 进程**
8. **每次启动前 kill 残留进程** (已实现到代码中)
9. **多机模式下必须关闭 ENABLE_OVERLAP**，否则 HCCL 死锁
10. **不要设置 ASCEND_LAUNCH_BLOCKING=1**，会导致多机分布式 hang
11. **定期清理 `/dev/shm/psm_*`**，避免 NPU 内存分配失败

---

## 九、环境对齐一键检查脚本

### 9.1 脚本说明

项目已提供 `scripts/check_task6_env.sh` 脚本，在 **Worker（新机器）** 上运行，自动完成以下工作：

1. **路径智能读取**：从 `config.json` 读取 `LMSV_PATH`、`MINDSPEED_MM_PATH`、`DATASET_ROOT`
2. **目录对比**：Worker 本地 vs Master 远程，检查真实目录 / symlink / broken symlink / 缺失
3. **Git commit 对比**：确保 `lmsv_rec` 代码仓库两边 commit 一致
4. **版本对比**：Python、CANN、NPU Driver
5. **生成修复命令**：输出具体的 `rsync` 和 `git` 命令供手动执行

### 9.2 使用方法

```bash
# 在 Worker（新机器）上执行
cd /data2/lm-sv/lmsv_rec
./scripts/check_task6_env.sh <MASTER_IP> [MASTER_USER] [SSH_PORT] [CONFIG_JSON]
```

**完整示例**：
```bash
# 步骤1: 确保 Worker 能免密 SSH 到 Master
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
ssh-copy-id -p 22 root@192.168.0.170

# 步骤2: 运行检查脚本
./scripts/check_task6_env.sh 192.168.0.170 root 22

# 步骤3: 根据输出结果，手动执行生成的 rsync 命令
```

### 9.3 检查结果解读

| 标记 | 含义 |
|------|------|
| ✓ | 本地和老机器一致，无需处理 |
| ✗ | 本地缺失或 broken symlink，需要从老机器同步 |
| ! | 存在差异但可能不影响（如 git commit 不一致、版本差异） |

### 9.4 实际运行示例

在测试环境（old-server 192.168.0.170 + new-server 192.168.0.203）中运行结果：

```
[2/6] 检查 LMSV 代码仓库...
  ✓ 本地: /data2/lm-sv/lmsv_rec (commit: 15f86c2ca)
  ✓ 老机器: /data2/lm-sv/lmsv_rec (commit: 15f86c2ca)
  ✓ git commit 一致

[3/6] 检查数据集目录...
  ✓ 本地: /data2/dataset 存在
  ✓ 老机器: /data2/dataset 存在

[4/6] 检查 MindSpeed-MM 相关目录...
  ! Megatron-LM: symlink -> /zyl/lm-sv/mm-new/Megatron-LM
  ! MindSpeed-MM: symlink -> /zyl/mindspeed-mm/MindSpeed-MM
  ! MindSpeed: 本地不存在，老机器也不存在（可能不需要）

[5/6] 检查软件版本...
  Python:     本地=Python 3.13.9  老机器=Python 3.10.20
  CANN:       本地=runtime_running_version=[8.3.0.1.200:8.3.RC1]  老机器=runtime_running_version=[8.3.0.1.200:8.3.RC1]
  NPU Driver: 本地=Version: 24.1.0.3  老机器=Version: 24.1.0.3
```

**生成的修复命令**：
```bash
# 同步 Megatron-LM (替换 symlink)
rm -f /shared/mindspeed-mm/Megatron-LM
mkdir -p /shared/mindspeed-mm/Megatron-LM
rsync -avz --progress -e "ssh -p 22" root@192.168.0.170:/shared/mindspeed-mm/Megatron-LM/ /shared/mindspeed-mm/Megatron-LM/

# 同步 MindSpeed-MM (替换 symlink)
rm -f /shared/mindspeed-mm/MindSpeed-MM
mkdir -p /shared/mindspeed-mm/MindSpeed-MM
rsync -avz --progress -e "ssh -p 22" root@192.168.0.170:/shared/mindspeed-mm/MindSpeed-MM/ /shared/mindspeed-mm/MindSpeed-MM/
```

### 9.5 注意事项

1. **脚本不会自动执行 rsync**：所有修复命令只输出到终端，需手动复制执行
2. **数据集可能很大**：/data2/dataset/ 可能有数十GB，同步时间较长
3. **磁盘空间**：同步前确保 Worker 磁盘空间充足
4. **版本差异**：Python 版本不一致（如 3.13.9 vs 3.10.20）需要手动安装对齐
5. **CANN 版本必须一致**：如果 CANN 不一致，必须先对齐再运行 Task6


### 4.10 变异配置文件未同步到远程节点（2026-04-29）

**问题**: 多机模式下，Master 节点执行变异后生成的 `mutation_gen*.json`、`data_config_*.json`、`model_config_*.json` 位于 `tmp/task6/` 下，但从未同步到 Worker 节点。远程节点的训练进程找不到变异后的配置文件，导致执行失败。

**根因**: `sync_iteration_to_remote_nodes()` 只同步了 `output/.../iter_N/`（权重和日志目录），没有同步 `tmp/task6/mutation_results/` 和 `tmp/task6/` 下的配置文件。

**修复**: `task6.py` 新增 `sync_mutation_configs_to_remote_nodes()` 函数：
1. 同步 `tmp/task6/mutation_results/{model_name}/` 目录（包含 `mutation_gen*.json`）
2. 同步 `tmp/task6/` 下的所有 `*.json` 文件（包含 `data_config_*.json` 和 `model_config_*.json`）
3. 在 `run_pta_verify_multinode()` 和 `run_msa_verify_multinode()` 中，启动远程任务前调用该函数

**客户影响**: 客户不需要任何操作，代码会自动处理配置同步。

