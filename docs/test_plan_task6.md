# Task6 完整测试计划

> **日期**: 2026-04-29
> **目标**: 验证 Task6 在多机模式下对 4 个模型的测试正确性

---

## 一、模型验证状态

| 模型 | 类型 | PTA跨机 | MSA跨机 | 已知问题 |
|------|------|---------|---------|----------|
| CogVideoX | 训练 | 通过 | 通过 | 精度diff ~1.5%（正常范围）|
| InternVL3 | 训练 | 通过 | 通过 | 精度diff ~20%（历史已知）|
| OpenSora | 推理 | 通过 | **预期失败** | TypeError: 'UntypedStorage' object is not callable |
| QwenVL | 推理 | 通过 | **预期失败** | RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed, AclNN tensor dim > 8 |

OpenSora 和 QwenVL 的 MSA 失败是 **已知的 detected_bugs**，Task6 的任务是复现这些 bug，不是修复它们。

**注意**：历史记录中 QwenVL 的 bug 曾标记为 `ValueError: For 'InnerInplaceIndexPut', shape mismatch`，但 2026-04-29 Task6 实际复现的是 `AclNN_Parameter_Error(EZ1001)`（tensor 维度 > 8）。两个都是有效的 MSA 兼容性问题，后者的触发路径更靠前。

## 一.1 验证成功案例详情

### 跨机 CogVideoX（2026-04-25，10轮）

| 轮次 | PTA Loss | PTA 显存(MB) | PTA 时间(ms) | MSA Loss | MSA 显存(MB) | MSA 时间(ms) | Loss 差异 |
|------|----------|-------------|-------------|----------|-------------|-------------|----------|
| 1 | 0.703366 | 34126.06 | 10857.20 | 0.692646 | 31336.93 | 12488.10 | 1.52% |
| 2 | 0.703312 | 34126.06 | 10851.40 | 0.692644 | 31336.93 | 12453.80 | 1.52% |
| 3 | 0.703306 | 34126.06 | 10850.90 | 0.692640 | 31354.34 | 12269.30 | 1.52% |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 10 | 0.703490 | 34126.06 | 10849.50 | 0.692571 | 31354.34 | 12546.60 | 1.55% |

- **PTA 成功率**: 100% (10/10)
- **MSA 成功率**: 100% (10/10)
- **输出目录**: `/data2/lm-sv/lmsv_rec/output/跨机cogvideoxTask6第一次成功/20260425_183829/`

### 跨机 CogVideoX 验证（2026-04-29，修复后1轮）

| 指标 | PTA | MSA | 判定 |
|------|-----|-----|------|
| Loss | 1.010575 | 0.964249 | diff=4.58% (<5%, OK) |
| 显存(MB) | 34126.06 | 31336.93 | MSA<PTA (OK) |
| 时间(ms) | 10875.80 | 11496.30 | MSA>PTA (NG) |

- **输出目录**: `/data2/lm-sv/lmsv_rec/output/2026-04-29-18-02-19/20260429_180253/`

### 单机 CogVideoX（2026-04-28，5轮）

| 指标 | 数值 |
|------|------|
| PTA 成功率 | 100% (5/5) |
| MSA 成功率 | 100% (5/5) |
| 平均 Loss 差异 | 1.73% - 1.79% |

- **输出目录**: `/data2/lm-sv/lmsv_rec/output/cogvideox成功/20260428_122156/`

### 单机 InternVL3（2026-04-28，5轮）

| 指标 | 数值 |
|------|------|
| PTA 成功率 | 100% (5/5) |
| MSA 成功率 | 100% (5/5) |
| 平均 Loss 差异 | 1.62% - 1.94% |

- **输出目录**: `/data2/lm-sv/lmsv_rec/output/internvl3成功/20260428_130816/`

### 单机 OpenSora（2026-04-29，2轮）

- PTA: 100% (2/2)，显存=21223MB，时间=169395ms/152636ms
- MSA: 0% (0/2)，错误=`TypeError: 'UntypedStorage' object is not callable`
- **输出目录**: `/data2/lm-sv/lmsv_rec/output/opensora成功/20260429_003245/`

### 单机 QwenVL（2026-04-28，5轮）

- PTA: 100% (5/5)，显存=19360MB，时间=77877-80308ms
- MSA: 0% (0/5)，错误=`RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed!`
- **输出目录**: `/data2/lm-sv/lmsv_rec/output/qwenvl成功/20260428_031317/`



---

## 二、报告格式验证清单

- [x] 每个迭代有"精度情况/显存情况/性能情况"列（HTML 显示为数值差异）
- [x] 精度判定：Loss 相对差异 < 5% = OK，>= 5% = NG
- [x] 显存判定：MSA 峰值 <= PTA 峰值 = OK
- [x] 性能判定：MSA 时间 <= PTA 时间 = OK
- [x] MSA 执行失败时显示 N/A，不归入精度问题
- [x] 报告末尾有"问题归类"汇总
- [x] 报告末尾有"问题汇总"列表
- [x] HTML 报告：精度/显存/性能列显示数值差异（带颜色），而非纯 OK/NG 标签

---

## 三、关键修复记录

### 2026-04-29 HTML 报告数值差异显示

**问题**: HTML report.html 的精度/显存/性能列只显示 OK/NG 标签，用户要求显示具体数值差异。

**修复文件**: `utils/analyze/task6_result.py`
- `generate_html_report()`: 精度列显示相对差异百分比（如 `4.58%`）
- 显存列显示差异 MB（如 `-2771.72MB`）
- 性能列显示差异 ms（如 `+1198.40ms`）
- 颜色区分：绿色 = OK，红色 = NG

### 2026-04-29 PTA time 提取修复

**问题**: InternVL3 等模型的 PTA 日志输出格式为 `elapsed time per iteration: 384 ms`（无括号），原有正则无法匹配，导致 `pta_time` 为 null。

**修复文件**: `utils/task/task6.py`
- `_parse_pta_log()`: 新增对 `elapsed time per iteration:\s*([\d.]+)\s*ms` 格式的匹配
- `_parse_msa_log()`: 同步增加相同格式支持

### 2026-04-29 QwenVL 实际 bug 确认

**问题**: Task6 实际复现的 QwenVL MSA 错误与历史记录不一致。

**实际错误**: `RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed!` + `AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.`

**触发位置**: `transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py:177`，`image_grid_thw[index].prod()`

**新 detected_bugs 记录**:
- 文件夹: `detected_bugs/qwenvl_msa_aclnn_tensor_dims_task6/`
- 报告: `detected_bugs/bug_report_qwenvl_msa_aclnn_tensor_dims_task6_huawei.md`

### 2026-04-29 SSH_PORT 支持

**问题**: `config.json` 中配置了 `SSH_PORT: 22`，但代码中 SSH/rsync 命令未使用 `-p` 参数，非标准端口时无法连接。

**修复文件**: `utils/task/task6.py`
- `_run_remote_shell()`: SSH 命令读取 `node['SSH_PORT']` 添加 `-p {port}`
- `sync_iteration_to_remote_nodes()`: mkdir 命令和 rsync 的 `-e` 参数均支持端口
- `_clean_ports_and_processes()`: 远程清理 SSH 命令支持端口

### 2026-04-29 lmsv.py 统一入口

**问题**: Task6 之前通过 `do.py` 直接执行，与其他 Task 不统一。

**修复**: 创建 `lmsv.py` 作为统一 Python 入口，`lmsv` bash 脚本调用 `lmsv.py`。所有 Task 统一使用 `./lmsv do` 执行。

### 2026-04-29 精度/显存/性能判定修正

**问题**: MSA 执行失败（无 loss/显存/时间数据）时，被错误归类为"精度问题"。

**修复**:
- `task6.py`: 一方无数据时，不将 `accuracy_ok`/`memory_ok`/`perf_ok` 设为 False
- `utils/analyze/task6_result.py`: 报告生成层对缺失数据的情况也做修正

### 2026-04-29 TASK6_ITER_OUTPUT_DIR 路径修复

**问题**: 推理模型样本保存到 `iter_N/weights/samples/` 而非 `iter_N/samples/`。

**修复**:
- `task6.py`: `.parent` 改为 `.parent.parent`
- 远程 `run_remote_pta_verify()` / `run_remote_msa_verify()` 新增 `TASK6_ITER_OUTPUT_DIR` 传递

### 2026-04-29 MSA wrapper JSON 修改修复

**问题**: `mm_msa_opensora.sh` 使用 sed 修改 `save_path`，可能破坏 JSON 格式。

**修复**: 统一使用 Python `json.load`/`json.dump` 修改配置。

### 2026-04-29 OpenSora PTA 推理模型 metric 提取修复

**问题**: `pta_opensora_real.sh` 检查 `[ -z "$LOSS" ]`，推理模型无 loss 导致误判失败。

**修复**: 改为仅检查 `EXIT_CODE`。

### 2026-04-28 MINDSPEED_MM_PATH 远程映射修复

**问题**: `_init_config()` 未传递 `MINDSPEED_MM_PATH`，远程节点路径映射错误。

**修复**: `_init_config()` 和 `_build_path_prefix_mappings()` 均添加 `MINDSPEED_MM_PATH` 处理。

### 2026-04-29 网络接口名可配置

**问题**: `GLOO_SOCKET_IFNAME` 和 `HCCL_SOCKET_IFNAME` 在代码中硬编码为 `enp67s0f5`，客户环境网卡名不同时出现 `RuntimeError: Unable to find address for: enp67s0f5`。

**修复文件**: `utils/task/task6.py`
- 新增 `Config.GLOO_SOCKET_IFNAME` 和 `Config.HCCL_SOCKET_IFNAME`，默认值为 `"enp67s0f5"`
- `_init_config()`: 从 `config.json` 的 `MULTI_NODE.GLOO_SOCKET_IFNAME` 和 `MULTI_NODE.HCCL_SOCKET_IFNAME` 读取
- 替换代码中 6 处硬编码的 `enp67s0f5` 为 `Config.GLOO_SOCKET_IFNAME` / `Config.HCCL_SOCKET_IFNAME`

### 2026-04-29 变异配置文件同步到远程节点

**问题**: 多机模式下，Master 执行变异后生成的 `mutation_gen*.json`、`data_config_*.json`、`model_config_*.json` 位于 `tmp/task6/` 下，从未同步到 Worker。远程节点找不到变异后的配置文件，导致训练失败。

**根因**: `sync_iteration_to_remote_nodes()` 只同步了 `output/.../iter_N/`，没有同步 `tmp/task6/mutation_results/` 和 `tmp/task6/*.json`。

**修复文件**: `utils/task/task6.py`
- 新增 `sync_mutation_configs_to_remote_nodes()` 函数，同步 `mutation_results/{model_name}/` 和 `tmp/task6/*.json`
- 在 `run_pta_verify_multinode()` 和 `run_msa_verify_multinode()` 中，启动远程任务前调用

### 2026-04-29 pgrep 自匹配 Bug

**问题**: `msa_cogvideox_real.sh` 中 `pgrep -f "pretrain_sora.py"` 在 SSH/bash wrapper 执行时匹配到自身进程，导致无限等待。

**修复文件**: `scripts/runtime/msa_cogvideox_real.sh`
- 替换为 `ps aux | grep "[p]retrain_sora.py"`


### 历史修复（2026-04-25 及之前）

- **HCCL_DETERMINISTIC 多机死锁**: 仅在单机模式设置
- **MSA 日志选择 bug**: 选择包含最多 loss 行的 worker 日志
- **NFS 竞争清除 msrun_log**: 仅在 NODE_RANK=0 清空
- **awk 除零错误**: `pta_cogvideox_real.sh` 中 STEP_TIME="N/A" 时保护
- **Simplejson ImportError**: 远程 msadapter 环境安装 simplejson
- **PIPE 缓冲 hang**: 重定向 stdout 到日志文件
- **HCCL 端口冲突**: `HCCL_IF_BASE_PORT=61000` + 预留端口
- **僵尸进程**: `kill_pretraingpt()` 增加 pta_memory_wrapper kill

---

## 四、运行方式

```bash
cd /data2/lm-sv/lmsv_rec

# 生成/修改配置
./lmsv conf

# 执行 Task6
./lmsv do

# 查看帮助
./lmsv help
```

---

## 五、定期维护

- **清理 `/dev/shm/psm_*`**: 避免 NPU 内存分配失败
- **清理 `/tmp` 残留**: 历史测试文件可能占用大量空间
- **同步脚本到远程**: `rsync -avz scripts/ remote:scripts/`

---

## 六、多机部署常见问题

### SSH 免密登录失败

**现象**: `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)`

**根因**: Master 节点无法通过 SSH 密钥认证连接到 Worker 节点。

**解决**:
```bash
# 1. Master 生成密钥（如没有）
ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""

# 2. 复制公钥到 Worker
ssh-copy-id -p 22 user@worker_ip

# 3. 验证
ssh -o BatchMode=yes -p 22 user@worker_ip "echo OK"
```

---

*文档维护人：邹英龙 | 最后更新：2026-04-29*

### 2026-04-29 LMSV_PATH PROJECT_ROOT 兼容性

**问题**: 客户要求 `LMSV_PATH` 指向 `PROJECT_ROOT`（如 `/home/jenkins/workspace/TDT_deployment/zjc/lm-sv`），而非 `lmsv_rec` 本身。

**修复**: `task6.py` 新增 `_get_remote_lmsv_root()` 和 `_get_remote_project_root()`：
- 如果 `LMSV_PATH` 以 `lmsv_rec` 结尾 → 直接返回
- 如果 `LMSV_PATH` 指向 `PROJECT_ROOT` → 自动追加 `/lmsv_rec`

**影响**: `_local_to_remote_path()`、`run_remote_pta_verify()`、`run_remote_msa_verify()` 的 `cd` 目标。

### 2026-04-29 pgrep 自匹配 Bug

**问题**: `msa_cogvideox_real.sh` 中 `pgrep -f "pretrain_sora.py"` 在 SSH/bash wrapper 执行时匹配到自身进程，导致无限等待。

**修复**: 替换为 `ps aux | grep "[p]retrain_sora.py"`。

