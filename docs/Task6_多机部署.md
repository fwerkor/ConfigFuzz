# Task6 多机部署文档

> **目标**: 将 Task6 从单机部署拓展到多机器部署
> **原则**: 最小必要修改，所有路径通过 `config.json` 配置动态推导

---

## 1. 机器信息示例

| 属性 | Master | Worker |
|------|--------|--------|
| 内网 IP | 192.168.0.170 | 192.168.0.203 |
| NPU | 8x Ascend 910B | 8x Ascend 910B |
| 角色 | 协调+变异+执行 | 执行 |

---

## 2. 新机器最小配置

### 2.1 SSH 免密登录

Master 节点必须能免密 SSH 到所有 Worker 节点：

```bash
# Master 上执行
ssh-copy-id root@192.168.0.203
ssh root@192.168.0.203 echo "OK"  # 验证无需密码
```

### 2.2 代码与数据路径

Worker 节点上 `lmsv_rec` 代码仓库的位置通过 `config.json` 的 `MULTI_NODE.OTHER_NODES[].LMSV_PATH` 指定。

**推荐方案（主从机路径相同）**:
- Master: `/data2/lm-sv/lmsv_rec`
- Worker: `/data2/lm-sv/lmsv_rec`（与老机器完全相同）

**独立部署方案**:
- Worker 将老机器的代码完整复制到相同路径，在 `config.json` 中配置 `LMSV_PATH`（通常与老机器相同）

### 2.3 Conda 环境

Worker 节点需要与 Master 相同的 conda 环境：
- `PTA_NAME`: 对应 Master 的 `PTA_NAME`（如 `mindspeed`）
- `MSA_NAME`: 对应 Master 的 `MSA_NAME`（如 `msadapter`）

### 2.4 MindSpeed-MM 与依赖仓库

Worker 节点需要以下目录（路径任意，通过 `config.json` 映射）：

| 目录 | 说明 |
|------|------|
| `Megatron-LM` | Megatron 框架 |
| `MindSpeed` | MindSpeed 加速库 |
| `MindSpeed-MM` | MindSpeed-MM 多模态框架 |
| `msadapter` | MSA 适配层（仅 MSA 环境需要） |

### 2.5 数据集

Worker 节点需要能访问 `DATASET_ROOT` 指定的数据集目录。推荐通过 NFS 共享。

---

## 3. 常见问题排查

### 3.1 SSH 免密登录失败

**现象**：
```
jenkins@xxx_node1_ip: Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)
```

**根因**：Master 节点无法通过 SSH 密钥认证连接到 Worker 节点，导致 rsync 同步和远程进程启动均失败。

**解决步骤**：

1. **生成 SSH 密钥**（Master 节点）：
```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
```

2. **复制公钥到 Worker**：
```bash
ssh-copy-id -p 22 user@worker_ip
```

3. **验证免密登录**：
```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p 22 user@worker_ip "echo OK"
# 必须输出 OK，不提示密码
```

4. **检查远程 SSH 配置**（Worker 节点）：
确保 `/etc/ssh/sshd_config` 包含：
```
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
```
修改后执行 `sudo systemctl restart sshd`

### 3.2 快速诊断脚本

```bash
NODE_IP="192.168.0.203"
NODE_PORT="22"
NODE_USER="root"

echo "=== 测试 SSH 连接 ==="
ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p $NODE_PORT $NODE_USER@$NODE_IP "echo 'SSH_OK'" 2>&1

echo "=== 测试远程目录创建 ==="
ssh -p $NODE_PORT $NODE_USER@$NODE_IP "mkdir -p /tmp/test_lm_sv && echo 'MKDIR_OK'" 2>&1

echo "=== 测试 rsync ==="
echo "test" > /tmp/test_sync.txt
rsync -az -e "ssh -p $NODE_PORT" /tmp/test_sync.txt $NODE_USER@$NODE_IP:/tmp/test_lm_sv/ 2>&1 && echo "RSYNC_OK"
```

## 4. 唯一配置：config.json

多机部署**只需要通过 `./lmsv conf` 交互式生成 `config.json`**，不需要修改任何代码。

### 3.1 配置示例

```json
{
  "task_type": 6,
  "PTA_NAME": "mindspeed",
  "MINDSPEED_MM_PATH": "/shared/mindspeed-mm",
  "MSA_NAME": "msadapter",
  "DATASET_ROOT": "/data2/dataset",
  "SAVE_ABNORMAL_WEIGHTS": false,
  "tasks": {
    "6": {
      "MODEL_NAME": "cogvideox",
      "TOTAL_ITER": 2,
      "MUTNM": 2,
      "COMPARE_MODE": "pta_msa",
      "TRAIN_ITER": 2,
      "BASE_SEED": 42,
      "MULTI_NODE": {
        "ENABLED": true,
        "MASTER_ADDR": "192.168.0.170",
        "MASTER_PORT": 29505,
        "NNODES": 2,
        "OTHER_NODES": [
          {
            "HOST": "192.168.0.203",
            "SSH_PORT": 22,
            "LMSV_PATH": "/data2/lm-sv",
            "PTA_NAME": "mindspeed",
            "MSA_NAME": "msadapter",
            "PTA_PATH": "/shared/mindspeed-mm",
            "MSA_PATH": "/shared/mindspeed-mm",
            "HAS_CONTAINER": false,
            "MINDSPEED_MM_PATH": "/shared/mindspeed-mm/MindSpeed-MM"
          }
        ]
      },
      "PTA_MAX_RUNTIME": 1800,
      "MSA_MAX_RUNTIME": 1800
    }
  }
}
```

### 3.2 配置字段说明

#### 全局路径（Master 节点本地路径）

| 字段 | 说明 | 示例 |
|------|------|------|
| `PTA_PATH` | Master 上 PTA workspace 根目录 | `/shared/lm-sv/mm-new` |
| `MSA_PATH` | Master 上 MSA workspace 根目录 | `/shared/lm-sv/mm-new` |
| `MINDSPEED_MM_PATH` | Master 上 MindSpeed-MM 路径 | `/shared/mindspeed-mm` |
| `DATASET_ROOT` | Master 上数据集根目录 | `/data2/dataset` |

#### MULTI_NODE 配置

| 字段 | 说明 |
|------|------|
| `ENABLED` | 是否启用多机模式 |
| `MASTER_ADDR` | Master 节点 IP |
| `MASTER_PORT` | 分布式训练主端口（默认 29505）|
| `NNODES` | 总节点数 |
| `OTHER_NODES[].HOST` | Worker 节点 IP |
| `OTHER_NODES[].SSH_PORT` | Worker SSH 端口（默认 22）|
| `OTHER_NODES[].LMSV_PATH` | Worker 上 `lmsv_rec` 代码路径 |
| `OTHER_NODES[].PTA_PATH` | Worker 上 PTA workspace 路径 |
| `OTHER_NODES[].MSA_PATH` | Worker 上 MSA workspace 路径 |
| `OTHER_NODES[].PTA_NAME` | Worker 上 PTA conda 环境名 |
| `OTHER_NODES[].MSA_NAME` | Worker 上 MSA conda 环境名 |
| `OTHER_NODES[].MINDSPEED_MM_PATH` | Worker 上 MindSpeed-MM 路径 |

### 3.3 路径映射原理

Task6 使用 `_build_path_prefix_mappings()` 动态推导本地→远程路径映射，**无需硬编码**。

路径映射基于共同后缀推导。例如：
- Master `MINDSPEED_MM_PATH` = `/shared/mindspeed-mm/MindSpeed-MM`
- Worker `MINDSPEED_MM_PATH` = `/zyl/mindspeed-mm/MindSpeed-MM`
- 共同后缀：`mindspeed-mm/MindSpeed-MM`
- 推导映射：`/shared` → `/zyl`

**WORKSPACE_ROOT / MINDSPEED_PATH 特殊处理**：
- `MINDSPEED_MM_PATH`、`MM_MODEL`、`SAVE_PATH`、`DATASET_ROOT` 等通过前缀映射自动转换
- `WORKSPACE_ROOT` 和 `MINDSPEED_PATH` 直接使用远程节点配置的 `PTA_PATH` / `MSA_PATH`，不做前缀映射

所有路径均通过 `config.json` 配置，**代码中不硬编码任何路径假设**。

---

## 4. 运行

配置完成后，直接执行：

```bash
cd /data2/lm-sv/lmsv_rec
./lmsv do
```

或通过交互式向导生成配置后执行：

```bash
./lmsv conf   # 选择 Task6，按提示配置
./lmsv do
```

---

## 5. 报告格式

每个迭代轮次在报告中展示：

| 列 | 说明 |
|----|------|
| 精度情况 | `OK`=Loss相对差异<5%, `NG`=差异>=5%, `N/A`=无数据 |
| 显存情况 | `OK`=MSA峰值<=PTA峰值, `NG`=MSA>PTA, `N/A`=无数据 |
| 性能情况 | `OK`=MSA时间<=PTA时间, `NG`=MSA>PTA, `N/A`=无数据 |

报告末尾包含：
- **问题归类**：按精度/显存/性能分类汇总
- **问题汇总**：所有问题的完整列表

---


## 5.1 跨机验证成功案例

### CogVideoX 跨机 10 轮成功（2026-04-25）

- **机器**: Master(192.168.0.170) + Worker(192.168.0.203)
- **配置**: 2机16卡，torchrun/msrun 分布式
- **结果**: PTA 100% (10/10)，MSA 100% (10/10)
- **Loss 差异**: 1.52% - 1.56%（正常范围）
- **输出**: `/data2/lm-sv/lmsv_rec/output/跨机cogvideoxTask6第一次成功/20260425_183829/`

### CogVideoX 跨机验证（2026-04-29）

- **目的**: 验证网卡配置修复和变异配置同步修复
- **结果**: PTA 通过，MSA 通过，Loss diff=4.58%
- **输出**: `/data2/lm-sv/lmsv_rec/output/2026-04-29-18-02-19/20260429_180253/`

### 四模型跨机验证总结

| 模型 | 类型 | PTA跨机 | MSA跨机 | 状态 |
|------|------|---------|---------|------|
| CogVideoX | 训练 | 通过 | 通过 | 精度diff ~1.5% |
| InternVL3 | 训练 | 通过 | 通过 | 精度diff ~1.6% |
| OpenSora | 推理 | 通过 | 预期失败 | 已复现UntypedStorage bug |
| QwenVL | 推理 | 通过 | 预期失败 | 已复现aclnn tensor dim>8 bug |

## 6. 常见问题

### 6.1 HCCL 初始化超时

**排查**:
1. 检查 `MASTER_ADDR` 是否为 Master 的内网 IP
2. 检查 Worker 能否 `ping MASTER_ADDR`
3. 检查 `HCCL_IF_BASE_PORT` 范围是否被占用
4. 检查防火墙是否放行了相关端口

### 6.2 路径映射失败

**排查**:
1. 确认 `config.json` 中 `OTHER_NODES[].LMSV_PATH` 指向 Worker 上实际存在的路径
2. 确认 `PTA_PATH`/`MSA_PATH` 在 Worker 上存在
3. 确认远程路径存在：`ssh root@worker_ip "ls -la /data2/lm-sv/lmsv_rec"`

### 6.3 远程 conda 环境找不到

**排查**:
1. 在 Worker 上手动执行 `conda activate mindspeed`，确认环境存在
2. 检查 `OTHER_NODES[].PTA_NAME` 与 Worker 上实际环境名一致

---

## 7. 约束

- 不要同时运行多个 Task6 进程
- 所有路径配置集中在 `config.json`，代码中不硬编码路径
- 定期清理 `/dev/shm/psm_*` 和 `/tmp` 残留文件，避免 NPU 内存分配失败

---

## 8. 环境版本要求（客户部署参考）

### 8.1 硬件与驱动

| 项目 | 要求 | 当前测试环境 |
|------|------|------------|
| NPU | Ascend 910B | 老机器 4卡 / 新机器 4卡 |
| Driver Version | 必须一致 | 24.1.0.3（两边相同） |
| npu-smi | 建议一致 | 24.1.0.3 / 23.0.6（不同但能用） |
| CANN | 必须一致 | 8.3.0.1.200:8.3.RC1 |

### 8.2 软件环境

| 项目 | 要求 | 当前测试环境 |
|------|------|------------|
| Python | 建议一致 | 3.13.9 / 3.10.20（不同但能用） |
| Conda环境 | 主从机同名 | mindspeed / msadapter |
| MindSpeed-MM | 必须一致 | symlink vs 真实目录（**必须统一**） |

### 8.3 关键目录一致性检查清单

客户部署前，必须逐项确认主从机以下目录完全相同：

| 目录 | 检查方式 |
|------|---------|
| `/data2/lm-sv/lmsv_rec/` | `git status` 两边输出相同commit |
| `/data2/dataset/` | `ls -laR` 对比文件列表和大小 |
| `/shared/mindspeed-mm/Megatron-LM/` | 目录存在且内容相同 |
| `/shared/mindspeed-mm/MindSpeed/` | 目录存在且内容相同 |
| `/shared/mindspeed-mm/MindSpeed-MM/` | 目录存在且内容相同 |

**注意**：客户环境中不要依赖任何NFS挂载或symlink机制。如果必须用symlink，目标路径在两边也必须存在且内容相同。


---

## 9. 环境对齐一键检查脚本

项目已提供 `scripts/check_task6_env.sh` 脚本，在 **Worker（新机器）** 上运行，自动检查与 Master（老机器）的环境差异并生成修复命令。

### 9.1 脚本位置

```
lmsv_rec/scripts/check_task6_env.sh
```

### 9.2 用法

在 **Worker（新机器）** 上执行：

```bash
cd /data2/lm-sv/lmsv_rec
./scripts/check_task6_env.sh <MASTER_IP> [MASTER_USER] [SSH_PORT] [CONFIG_JSON]
```

示例：
```bash
# 基本用法（使用默认 config.json 路径）
./scripts/check_task6_env.sh 192.168.0.170

# 指定用户名和 SSH 端口
./scripts/check_task6_env.sh 192.168.0.170 root 22

# 指定自定义 config.json 路径
./scripts/check_task6_env.sh 192.168.0.170 root 22 /data2/lm-sv/lmsv_rec/config.json
```

### 9.3 前提条件

1. **Worker 能免密 SSH 到 Master**
   ```bash
   # 在 Worker 上执行
   ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""
   ssh-copy-id -p 22 root@192.168.0.170
   ```

2. **Worker 上已存在 config.json**（脚本会从中读取路径配置）

### 9.4 检查项

| 检查项 | 说明 |
|--------|------|
| SSH 免密登录 | 验证 Worker 到 Master 的 SSH 连接 |
| LMSV 代码仓库 | 检查 git commit 是否一致 |
| 数据集目录 | 检查目录存在性 |
| MindSpeed-MM 相关目录 | 检查 symlink / broken symlink / 缺失目录 |
| Python 版本 | 对比两边 Python 版本 |
| CANN 版本 | 对比两边 CANN 版本 |
| NPU Driver | 对比两边 NPU 驱动版本 |

### 9.5 输出示例

脚本会输出检查结果，如果有差异，会生成具体的 `rsync` 修复命令：

```
==========================================
检查完成 - 修复命令
==========================================

# 请复制以下命令并在 Worker 上执行
# ============================================================

# 1. 同步 LMSV 代码仓库
cd /data2/lm-sv/lmsv_rec && git fetch origin && git reset --hard abc1234

# 2. 同步数据集
rsync -avz --progress -e "ssh -p 22" root@192.168.0.170:/data2/dataset/ /data2/dataset/

# 3. 同步 MindSpeed-MM 相关目录
# 同步 Megatron-LM (替换 symlink)
rm -f /shared/mindspeed-mm/Megatron-LM
mkdir -p /shared/mindspeed-mm/Megatron-LM
rsync -avz --progress -e "ssh -p 22" root@192.168.0.170:/shared/mindspeed-mm/Megatron-LM/ /shared/mindspeed-mm/Megatron-LM/

# 4. 版本警告（请手动处理）
# Python 版本不一致: 本地=Python 3.10.20, 老机器=Python 3.13.9
```

**注意**：
- rsync 命令不会自动执行，请手动复制粘贴并确认
- 数据集同步可能耗时较长（数十GB），建议在非业务时段执行
- 同步前请确保 Worker 磁盘空间充足
- Python/CANN 版本不一致需要手动安装对齐

