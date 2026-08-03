# Task6 开发经验与避坑指南

> **作者**: 邹英龙
> **更新日期**: 2026-04-29（最终版，所有修复已验证）
>
> 本文档记录Task6开发过程中踩过的所有坑、经验、原则和上下文。仅供开发参考，不属于最终交付内容。
> **适用范围**: Task6 多模态整网变异和验证（单机 + 多机）

---

## 核心原则（不可违背）

### 原则0：CogVideoX / InternVL3 跑通后作为基准

**CogVideoX 和 InternVL3 已成功跑通（单机 + 多机），证明Task6框架、环境配置、日志捕获逻辑、多机部署均正确。**

**在调试其他模型时，以下部分禁止修改：**

1. **Task6核心框架**（`utils/task/task6.py`）：
   - 环境变量读取逻辑（`_init_config`）
   - PTA/MSA执行流程（`_run_pta_verify`、`_run_msa_verify`）
   - 多机并发启动逻辑（`ThreadPoolExecutor` + SSH/rsync）
   - 日志解析正则表达式
   - 成功/失败判定逻辑
   - 精度/显存/性能判定标准

2. **CogVideoX / InternVL3 相关脚本**：
   - `scripts/runtime/mm_pta_cogvideox.sh`、`mm_msa_cogvideox.sh`
   - `scripts/runtime/mm_pta_internvl3.sh`、`mm_msa_internvl3.sh`
   - `scripts/runtime/pta_cogvideox_real.sh`、`msa_cogvideox_real.sh`
   - `scripts/runtime/pta_internvl3_8B_real.sh`、`msa_internvl3_8B_real.sh`

3. **环境配置**：
   - MindSpeed-MM 路径（`MINDSPEED_MM_PATH`，兼容旧版 `PTA_PATH`/`MSA_PATH`）
   - Conda环境名称（`PTA_NAME`、`MSA_NAME`）
   - CANN环境设置方式
   - 多机环境变量（HCCL_IF_IP、HCCL_SOCKET_IFNAME等）

**调试其他模型的正确方式**：
- 如果其他模型执行失败，应修改该模型**专属的脚本**（如`mm_pta_opensora.sh`）
- 不应为了修复某个模型而回退Task6框架的修复
- 各模型脚本需独立适配，但框架层逻辑保持一致

---

### 原则1：PTA和日志分析代码不能随意修改

**四个模型（InternVL3、QwenVL2.5、OpenSora1.2、CogVideoX）的PTA执行和日志分析部分已经验证是正确的。**

| 模型 | 模式 | 单机PTA | 单机MSA | 多机PTA | 多机MSA | 验证数据 |
|------|------|---------|---------|---------|---------|---------|
| InternVL3 | 训练 | 正常 | 正常 | 正常 | 正常 | 5轮，loss diff 1.6%-1.9% |
| QwenVL2.5 | 推理 | 正常 | 预期失败(已知bug) | 正常 | 预期失败(已知bug) | 5轮，PTA通过，MSA复现aclnn tensor dim>8 |
| OpenSora1.2 | 推理 | 正常 | 预期失败(已知bug) | 正常 | 预期失败(已知bug) | 2轮，PTA通过，MSA复现UntypedStorage |
| CogVideoX | 训练 | 正常 | 正常 | 正常 | 正常 | 10轮跨机+5轮单机，loss diff 1.5%-1.8% |

**风险**：修改这些已验证的代码可能导致回归问题，使原本正常的模型执行失败。

---

### 原则2：环境必须与Task1和docs保持一致

| 项目 | 要求 |
|------|------|
| PTA环境 | `mindspeed` conda环境 |
| MSA环境 | `msadapter` conda环境 |
| 路径配置 | 参考 `docs/environment-modifications.md` 一键安装指南 |
| 环境变量 | MINDSPEED_MM_PATH、LMSV_OUTPATH 必须正确设置（兼容 PTA_PATH/MSA_PATH） |
| CANN | `/usr/local/Ascend/ascend-toolkit/latest` |

**风险**：环境不一致会导致各种难以排查的错误。

---

### 原则3：配置统一走 `config.json`，禁止硬编码（基本原则，不可违反）

**Task6 的配置方式已与 Task1-5 完全一致：所有参数通过 `config.json` 的 `tasks["6"]` 字典传入，不再使用 `TASK6_*` 环境变量。**

**实现方式**：
1. **`task6.py` 配置读取**（`_init_config` 函数）：
   - 与 Task1-5 一致，所有任务参数从 `params` 字典读取
   - `params` 由 `./lmsv do` 从 `config.json` 的 `tasks["6"]` 中提取并传入
   - 不再读取 `TASK6_*` 环境变量
   - 支持的参数字段：
     - `MODE` - 运行模式
     - `TOTAL_ITER` - 最大迭代次数
     - `MUTNM` - 每轮变异参数个数
     - `TRAIN_ITER` - 每轮训练/推理步数（兼容旧版 `SAVE_STEPS` / `TRAIN_ITERS`）
     - `COMPARE_MODE` - 对比模式
     - `MODEL_NAME` - 模型名称
     - `BASE_SEED` - 基础随机种子
     - `MULTI_NODE` - 多机配置

2. **全局路径/环境名**：
   - `PTA_NAME`、`MSA_NAME`、`MINDSPEED_MM_PATH` 等仍在 `config.json` 根级别配置
   - `do.py`（由 `./lmsv do` 调用）会将这些字段导出为环境变量供子脚本使用
   - Task6 统一使用 `MINDSPEED_MM_PATH`，不再拆分 `PTA_PATH` / `MSA_PATH`

3. **shell 脚本配置读取**：
   - 使用 `${VAR:-default}` 语法支持环境变量覆盖
   - 示例：`export LOAD_PATH="${LOAD_PATH:-${DATASET_ROOT}/model/ckpt}"`

---

### 原则4：无效突变处理

**无效突变定义**：PTA执行时日志中出现真正的error（Traceback、RuntimeError等，Warning不算）的突变。

**处理原则**：
1. **不计入轮次**：无效突变不计入总迭代次数，该轮次需要重新执行
2. **清理策略**：
   - 保留：`tmp/task6/pta_verify_iterX.log` —— 用于调试分析
   - 删除：`tmp/task6/mutation_results/`中的`mutation_genX.json`
   - 删除：`tmp/task6/`中的临时配置文件
   - 删除：`results/iters/iter_X/`归档目录（无效突变不归档）
3. **有效突变**：即使MSA失败也是有效突变，所有日志和配置都要保留

**关键区分**：
- **PTA失败** = 无效突变（崩溃/出错，无法验证）→ 清理 → 不计入轮次
- **MSA失败** = 有效突变（PTA正常执行，发现框架问题）→ 全部保留 → 计入轮次

---

### 原则5：成功状态判断标准

**绝对禁止**：通过"有没有loss"、"有没有metrics"、"返回码是否为0"来判断PTA/MSA是否成功。

**唯一正确方式**：
1. **执行时判断**：检查日志中是否有真正的error（Traceback、RuntimeError、OSError等，Warning不算）
2. **报告生成时**：从`status.json`的`PTA_VERIFY`和`MSA_VERIFY`字段读取状态

---


## 跨机验证成果

Task6 跨机（多节点）部署已全部验证通过，以下是完整的验证记录：

### 四模型跨机验证状态

| 模型 | 类型 | PTA跨机 | MSA跨机 | 验证日期 | 输出目录 |
|------|------|---------|---------|----------|----------|
| CogVideoX | 训练 | 通过 | 通过 | 2026-04-25 | `output/跨机cogvideoxTask6第一次成功/20260425_183829/` |
| CogVideoX | 训练 | 通过 | 通过 | 2026-04-29 | `output/2026-04-29-18-02-19/20260429_180253/` |
| InternVL3 | 训练 | 通过 | 通过 | 2026-04-28 | `output/internvl3成功/20260428_130816/` |
| OpenSora | 推理 | 通过 | 预期失败 | 2026-04-29 | `output/opensora成功/20260429_003245/` |
| QwenVL | 推理 | 通过 | 预期失败 | 2026-04-28 | `output/qwenvl成功/20260428_031317/` |

### CogVideoX 跨机 10 轮详细数据

| 轮次 | PTA Loss | MSA Loss | Loss 差异 | PTA 显存 | MSA 显存 | PTA 时间 | MSA 时间 |
|------|----------|----------|----------|----------|----------|----------|----------|
| 1 | 0.703366 | 0.692646 | 1.52% | 34126 | 31337 | 10857 | 12488 |
| 2 | 0.703312 | 0.692644 | 1.52% | 34126 | 31337 | 10851 | 12454 |
| 3 | 0.703306 | 0.692640 | 1.52% | 34126 | 31354 | 10851 | 12269 |
| ... | ... | ... | ... | ... | ... | ... | ... |
| 10 | 0.703490 | 0.692571 | 1.55% | 34126 | 31354 | 10850 | 12547 |

**统计**: PTA 成功率 100% (10/10)，MSA 成功率 100% (10/10)，Loss 差异稳定保持在 1.52%-1.56%。

### 关键修复验证

以下修复均已在跨机 CogVideoX 验证中得到确认：

| 修复项 | 验证状态 | 验证日期 |
|--------|----------|----------|
| SSH_PORT 支持 | 验证通过 | 2026-04-29 |
| 网络接口名可配置 | 验证通过 | 2026-04-29 |
| 变异配置文件同步到远程 | 验证通过 | 2026-04-29 |
| pgrep 自匹配修复 | 验证通过 | 2026-04-29 |
| MINDSPEED_MM_PATH 远程映射 | 验证通过 | 2026-04-28 |
| LMSV_PATH PROJECT_ROOT 兼容 | 验证通过 | 2026-04-29 |
| HCCL_DETERMINISTIC 多机保护 | 验证通过 | 2026-04-25 |
| MSA 日志选择（最多loss行） | 验证通过 | 2026-04-25 |
| NFS 竞争清除 msrun_log | 验证通过 | 2026-04-25 |


## 模型状态总览（2026-04-29）

### 四模型完整状态

| 模型 | 类型 | PTA | MSA | 精度diff | 已知问题 |
|------|------|-----|-----|---------|---------|
| **InternVL3** | 训练 | 单机/多机均正常 | 单机/多机均正常 | ~1.6% (对齐通过) | 无 |
| **CogVideoX** | 训练 | 单机/多机均正常 | 单机/多机均正常 | ~1.7%-4.6% (对齐通过) | 无 |
| **QwenVL2.5** | 推理 | 单机/多机均正常 | **预期失败** | N/A (推理无loss) | `AclNN_Parameter_Error(EZ1001)` tensor dim > 8 |
| **OpenSora1.2** | 推理 | 单机/多机均正常 | **预期失败** | N/A (推理无loss) | `TypeError: 'UntypedStorage' object is not callable` |

### 训练模型精度对齐结论

**CogVideoX**：
- 历史 10 轮测试（2026-04-13）：Loss 相对差异 **1.72%-1.79%**，标准差仅 0.025%
- 最新 2 轮测试（2026-04-29）：Loss 相对差异 **4.58%**（不同数据集/配置导致基线不同）
- **结论**：差异极其稳定，不受突变配置影响，属于框架级系统性偏差
- **判定**：对齐通过（<5%阈值）
- **详细分析**：见 `detected_bugs/bug_report_cogvideox_msa_precision_misalignment_task6_huawei.md`

**InternVL3**：
- 5 轮测试（2026-04-28）：Loss 相对差异 **1.62%-1.94%**
- 历史（FP16/BF16混用时期）：差异 ~20%，修正后降至 ~1.6%
- **结论**：权重初始化一致性对多模态模型至关重要（FP16/BF16混用→20%→统一后1.6%）
- **判定**：对齐通过（<5%阈值）
- **详细分析**：见 `detected_bugs/bug_report_internvl3_msa_precision_misalignment_task6_huawei.md`

### 推理模型MSA失败（预期行为）

**QwenVL**：
- 实际错误：`RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed!` + `AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.`
- 触发位置：`transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py:177`，`image_grid_thw[index].prod()`
- 历史错误（2026-04-14前）：`ValueError: For 'InnerInplaceIndexPut', shape mismatch`（被新错误提前拦截）
- **报告**：`detected_bugs/bug_report_qwenvl_msa_aclnn_tensor_dims_task6_huawei.md`

**OpenSora**：
- 实际错误：`TypeError: 'UntypedStorage' object is not callable`
- 根因：msadapter 对 PyTorch `torch.UntypedStorage` 的 patch 导致 safetensors 库底层调用异常
- **报告**：`detected_bugs/bug_report_opensora_msa_huawei.md`
- **注意**：FakeUntypedStorage patch 无法修复 Rust 扩展层面的问题，Task6 的任务是复现/检测此bug

---

## 精度 / 显存 / 性能判定标准（核心）

### 判定逻辑

Task6 报告对每个 mutation iteration 进行三方面判定：

| 维度 | 判定标准 | OK 条件 | NG 条件 | N/A 条件 |
|------|---------|---------|---------|---------|
| **精度** | Loss 相对差异 | \|msa-pta\|/pta < 5% | \|msa-pta\|/pta >= 5% | 任一方无 loss 数据 |
| **显存** | 峰值显存对比 | MSA <= PTA | MSA > PTA | 任一方无显存数据 |
| **性能** | 执行时间对比 | MSA <= PTA | MSA > PTA | 任一方无时间数据 |

**关键规则**：
- **推理模型**（QwenVL、OpenSora）无 loss 输出，精度列始终显示 N/A，这不是异常
- **MSA 执行失败**时（无 loss/显存/时间数据），该维度显示 N/A，**不归入精度/显存/性能问题**
- **一方无数据时**，`accuracy_ok`/`memory_ok`/`perf_ok` 保持为 `True`（由报告层根据 None 显示 N/A）

### 报告显示格式

**Markdown 报告**（`summary.md`）：
```
| 轮次 | 精度情况 | 显存情况 | 性能情况 | 问题 |
|------|----------|----------|----------|------|
| 1    | OK (diff=1.73%) | OK (diff=-2771.72MB) | NG (diff=+1198.40ms) | 性能异常... |
```

**HTML 报告**（`report.html`）：
- 精度列显示相对差异百分比（如 `4.58%`）
- 显存列显示差异 MB（如 `-2771.72MB`）
- 性能列显示差异 ms（如 `+1198.40ms`）
- 颜色区分：绿色 = OK，红色 = NG

**问题归类节**（报告末尾）：
```markdown
## 问题归类
- **精度问题** (0轮): 无
- **显存问题** (0轮): 无
- **性能问题** (2轮): Iter1, Iter2
```

### 实现位置

- **判定逻辑**：`utils/task/task6.py` 中 `analyze_results()` 函数
- **报告生成**：`utils/analyze/task6_result.py` 中 `generate_markdown_report()` / `generate_html_report()`
- **状态字段**：`metrics.json` 的 `analysis` 字段包含 `accuracy_ok`、`memory_ok`、`perf_ok`

---

## 多机部署核心经验

### 架构设计

Task6 采用 **SSH + rsync + ThreadPoolExecutor** 并发启动本地/远程任务：

```
old-server (192.168.0.170, MASTER, NODE_RANK=0)
  ├─ PTA: torchrun --nnodes 2 --node_rank 0
  └─ MSA: msrun --worker_num 16 --local_worker_num 8 --node_rank 0

new-server (192.168.0.203, NODE_RANK=1)
  ├─ PTA: torchrun --nnodes 2 --node_rank 1
  └─ MSA: msrun --worker_num 16 --local_worker_num 8 --node_rank 1
```

### SSH_PORT 配置与修复（2026-04-29）

**问题**：`config.json` 中配置了 `SSH_PORT`，但代码中 SSH/rsync 命令未使用 `-p` 参数，非标准端口时无法连接。

**根因**：`_init_config()` 构建 `node_info` 时未将 `raw_node.get("SSH_PORT", 22)` 复制到 `normalized_nodes`，导致所有 `node.get('SSH_PORT', 22)` 调用均返回 22。

**修复**（`task6.py`）：
1. `_init_config()`: 添加 `ssh_port = int(raw_node.get("SSH_PORT", 22))` 并传入 `node_info`
2. `_run_remote_shell()`: SSH 命令格式为 `-p {port} -o BatchMode=yes -o StrictHostKeyChecking=no`（`-p` 必须在 `-o` 之前）
3. `sync_iteration_to_remote_nodes()`: mkdir 和 rsync 命令均支持端口
4. `_clean_ports_and_processes()`: 远程清理 SSH 命令支持端口

**rsync `-e` 参数格式**（关键）：
```python
rsync_ssh_port_arg = f"-p {ssh_port} " if ssh_port != 22 else ""
rsync_cmd = (
    f"rsync -az -e '{Config.SSH_BIN} {rsync_ssh_port_arg}-o BatchMode=yes ...' "
    f"{src} {node['HOST']}:{dst}"
)
```

### 路径映射原理

Task6 使用 `_build_path_prefix_mappings()` 动态推导本地→远程路径映射，**不硬编码任何路径**。

路径映射基于共同后缀推导。例如：
- Master `MINDSPEED_MM_PATH` = `/shared/mindspeed-mm/MindSpeed-MM`
- Worker `MINDSPEED_MM_PATH` = `/zyl/mindspeed-mm/MindSpeed-MM`
- 共同后缀：`mindspeed-mm/MindSpeed-MM`
- 推导映射：`/shared` → `/zyl`

**关键配置字段**（`config.json` 中 `MULTI_NODE.OTHER_NODES[]`）：
| 字段 | 说明 | 必填 |
|------|------|------|
| `HOST` | Worker IP | 是 |
| `SSH_PORT` | SSH 端口（默认22）| 是 |
| `LMSV_PATH` | Worker 上 lmsv_rec 路径 | 是 |
| `PTA_PATH` / `MSA_PATH` | Worker 上 workspace 路径 | 是 |
| `MINDSPEED_MM_PATH` | Worker 上 MindSpeed-MM 路径 | 是 |
| `PTA_NAME` / `MSA_NAME` | Worker conda 环境名 | 是 |

**WORKSPACE_ROOT / MINDSPEED_PATH 特殊处理**：
- `run_remote_pta_verify()` / `run_remote_msa_verify()` 中，`WORKSPACE_ROOT` / `MINDSPEED_PATH` 直接使用远程 `node["PTA_PATH"]` / `node["MSA_PATH"]`，不做前缀映射

### 环境变量传递

多机模式下，以下环境变量必须正确传递到远程节点：

| 环境变量 | 传递方式 | 说明 |
|---------|---------|------|
| `MASTER_ADDR` | SSH 命令 env | 分布式训练主节点IP |
| `MASTER_PORT` | SSH 命令 env | 分布式训练端口 |
| `NNODES` | SSH 命令 env | 总节点数 |
| `NODE_RANK` | SSH 命令 env | 当前节点rank |
| `MINDSPEED_MM_PATH` | SSH 命令 env | MindSpeed-MM路径（已做远程映射） |
| `TASK6_ITER_OUTPUT_DIR` | SSH 命令 env | 推理模型样本输出目录 |
| `HCCL_IF_IP` | 远程节点本地设置 | 各节点自己的IP |
| `HCCL_SOCKET_IFNAME` | 远程节点本地设置 | 网卡名 |
| `HCCL_IF_BASE_PORT` | 远程节点本地设置 | HCCL基础端口 |
| `ENABLE_OVERLAP` | 远程节点本地设置 | **多机必须设为空字符串** |

**多机关键约束**：
- `ENABLE_OVERLAP=""`（空字符串）：多机模式下必须关闭，否则 HCCL 死锁
- **不要设置 `ASCEND_LAUNCH_BLOCKING=1`**：会导致多机分布式 hang
- `HCCL_IF_BASE_PORT=61000` + `sysctl -w net.ipv4.ip_local_reserved_ports="61000-61015"`：避免端口冲突

### MSA 多机日志同步与 fallback（2026-04-29）

**问题**：多机模式下，本地节点的 `exec_log_file` 可能缺少 loss（因为 `print_rank_last` 输出到 worker 日志而非主日志）。

**解决**：`run_msa_verify_multinode()` 在本地执行完成后：
1. 通过 `scp` 从远程节点同步 `worker_*.log` 到本地 `msrun_log/`
2. 从 worker 日志中补充缺失的 loss / memory / time 指标
3. 选择包含最多有效指标的 worker 日志作为 fallback

**关键代码**（只在没有 `error_info` 时才设置 `local_ok=True`，避免掩盖真实错误）：
```python
if not local_metrics.get("error_info"):
    local_ok = True
    all_ok = True
```

### 推理模型多机特殊处理

**PTA 推理模型成功判定**：
- 推理模型（QwenVL、OpenSora）无 loss
- 成功条件：`returncode == 0` 且（`memory is not None` 或 `time is not None`）
- 远程节点失败（如HCCL连接超时）不阻断本地成功判定

**MSA 推理模型成功判定**：
- 同样不需要 loss，只要有显存或时间指标即可
- `run_msa_verify_multinode()` 中：`if local_metrics.get('memory') is not None or local_metrics.get('time') is not None`

### HCCL 配置备忘

**本地节点 (192.168.0.170)**：
```bash
export HCCL_IF_IP=192.168.0.170
export HCCL_SOCKET_IFNAME=enp67s0f5
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_IF_BASE_PORT=61000
export HCCL_CONNECT_TIMEOUT=1200
export ENABLE_OVERLAP=""
```

**远程节点 (192.168.0.203)**：
```bash
export HCCL_IF_IP=192.168.0.203
export HCCL_SOCKET_IFNAME=enp67s0f5
export GLOO_SOCKET_IFNAME=enp67s0f5
export HCCL_IF_BASE_PORT=61000
export HCCL_CONNECT_TIMEOUT=1200
export ENABLE_OVERLAP=""
```

**端口预留**：
```bash
sysctl -w net.ipv4.ip_local_reserved_ports="61000-61015"
```

---

## 多机部署已修复问题清单（按时间倒序）

### 2026-04-29 SSH_PORT 未使用
- **根因**：`_init_config()` 未将 `SSH_PORT` 复制到 `normalized_nodes`
- **修复**：`task6.py` 中所有 SSH/rsync 命令均读取 `node.get('SSH_PORT', 22)`，`-p` 放在 `-o` 之前

### 2026-04-29 推理模型 PTA 被误判失败
- **根因**：`_check_log_has_real_error()` 检测到 torchrun 初始连接 Traceback，标记 PTA 失败，但推理模型 returncode=0 且已完成
- **修复**：推理模型先检查 `returncode == 0`，通过则直接视为成功

### 2026-04-29 MSA fallback 掩盖错误
- **根因**：`run_msa_verify_multinode()` 同步 worker 日志后无条件设置 `local_ok = True`，即使 `error_info` 存在
- **修复**：仅在 `not local_metrics.get("error_info")` 时才设置 `local_ok = True`

### 2026-04-29 精度/显存/性能判定错误归类
- **根因**：MSA 执行失败（无 loss/显存/时间数据）时被错误归类为"精度问题"
- **修复**：`analyze_results()` 中一方无数据时不将 `accuracy_ok`/`memory_ok`/`perf_ok` 设为 False

### 2026-04-29 TASK6_ITER_OUTPUT_DIR 路径错误
- **根因**：`.parent` 只能回到 `weights/`，需 `.parent.parent` 回到 `iter_N/`
- **修复**：本地和远程函数均改为 `.parent.parent`

### 2026-04-29 MSA wrapper sed 修改 JSON 不可靠
- **根因**：`mm_msa_opensora.sh` 使用 sed 改 `save_path`，破坏 JSON 格式
- **修复**：统一使用 Python `json.load`/`json.dump` 修改配置

### 2026-04-29 PTA time 提取修复
- **根因**：InternVL3 日志格式为 `elapsed time per iteration: 384 ms`（无括号），原有正则无法匹配
- **修复**：`_parse_pta_log()` / `_parse_msa_log()` 增加 `elapsed time per iteration:\s*([\d.]+)\s*ms` 格式支持

### 2026-04-29 OpenSora PTA 推理模型 metric 提取修复
- **根因**：`pta_opensora_real.sh` 检查 `[ -z "$LOSS" ]`，推理模型无 loss 导致误判失败
- **修复**：改为仅检查 `EXIT_CODE`

### 2026-04-28 MINDSPEED_MM_PATH 远程映射修复
- **根因**：`_init_config()` 未传递 `MINDSPEED_MM_PATH`
- **修复**：`_init_config()` 和 `_build_path_prefix_mappings()` 均添加 `MINDSPEED_MM_PATH` 处理

### 2026-04-28 多机推理模型Bug修复
- **问题1**: `cannot access local variable 're'` — 删除函数内部条件分支中的 `import re`
- **问题2**: MSA bug信息被"读取日志异常"掩盖 — 优先使用 `msa_metrics.get("error_info")`
- **问题3**: 推理模型错误显示loss=0.0 — 返回前将 `metrics["loss"]` 重置为 `None`
- **问题4**: qwen MSA失败后时间/显存N/A — `set +e` / `set -e` 保护
- **问题5**: 多机推理模型PTA被误判为失败 — 推理模型只需 memory/time 不为 None

### 2026-04-26 HCCL_DETERMINISTIC 多机死锁
- **根因**：多机模式下设置 `HCCL_DETERMINISTIC=true` 导致死锁
- **修复**：仅在单机模式设置 `HCCL_DETERMINISTIC=true`

### 2026-04-25 MSA 日志选择 bug
- **根因**：固定读取 worker_0.log，但 PP>1 时 loss 只在最后一个 worker 输出
- **修复**：选择包含最多 loss 行的 worker 日志

### 2026-04-25 NFS 竞争清除 msrun_log
- **根因**：所有节点同时清空 msrun_log 导致竞争
- **修复**：仅在 NODE_RANK=0 清空

### 2026-04-25 awk 除零错误
- **根因**：`pta_cogvideox_real.sh` 中 STEP_TIME="N/A" 时 awk 除零
- **修复**：添加保护逻辑

### 2026-04-25 Simplejson ImportError
- **根因**：远程 msadapter 环境缺少 simplejson
- **修复**：在远程节点安装 simplejson

### 2026-04-25 PIPE 缓冲 hang
- **根因**：PIPE 缓冲导致输出堵塞
- **修复**：stdout 重定向到日志文件

### 2026-04-25 HCCL 端口冲突
- **根因**：默认端口被占用
- **修复**：`HCCL_IF_BASE_PORT=61000` + 预留端口

### 2026-04-25 僵尸进程
- **根因**：残留进程占用 NPU
- **修复**：`kill_pretraingpt()` 增加 pta_memory_wrapper kill

---

## 已知 Bug 汇总

### 1. OpenSora MSA safetensors 模型加载兼容性

**错误**：`TypeError: 'UntypedStorage' object is not callable`

**根因**：msadapter 对 PyTorch 的 `torch.UntypedStorage` 进行了 patch，导致 safetensors 库底层调用的存储分配逻辑发生变化。

**状态**：已记录为 detected_bug，Task6 的任务是复现/检测，不是修复。

**报告**：`detected_bugs/bug_report_opensora_msa_huawei.md`

**注意**：FakeUntypedStorage patch（`sitecustomize.py`）不能修复 Rust 扩展层面的问题。

### 2. QwenVL MSA tensor 维度限制

**错误**：`AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.`

**触发位置**：`transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py:177`，`image_grid_thw[index].prod()`

**状态**：已记录为 detected_bug。

**报告**：`detected_bugs/bug_report_qwenvl_msa_aclnn_tensor_dims_task6_huawei.md`

### 3. CogVideoX MSA 精度系统性偏差

**现象**：10 轮有效突变下，Loss 相对差异稳定在 **1.72%-1.79%**（历史）或 **4.58%**（最新），标准差极小。

**根因**：PyTorch NPU 与 MindSpore 在 BF16 累加语义、Flash Attention 实现、分布式优化器、图模式 vs Eager 执行等方面的框架级差异。

**状态**：已记录为 detected_bug，对齐通过（<5%）。

**报告**：`detected_bugs/bug_report_cogvideox_msa_precision_misalignment_task6_huawei.md`

### 4. InternVL3 MSA 精度系统性偏差

**现象**：5 轮有效突变下，Loss 相对差异 **1.62%-1.94%**。

**根因**：同 CogVideoX，框架级系统性偏差。FP16/BF16 混用修正后从 20% 降至 1.6%。

**状态**：已记录为 detected_bug，对齐通过（<5%）。

**报告**：`detected_bugs/bug_report_internvl3_msa_precision_misalignment_task6_huawei.md`

---

## 关键避坑点

### 1. 路径配置

**原则**: Task6 默认使用相对于代码目录 `lmsv_rec/` 的相对路径，便于在不同环境部署。

**默认路径**:
- `MINDSPEED_MM_PATH` 默认: `<lm-sv-root>/../mindspeed-mm`（指向 workspace root，自动推导 MindSpeed-MM 子目录）
- `LMSV_OUTPATH` 默认: `output`

**解决方案**:
```bash
# 使用绝对路径（推荐，以实际部署路径为准）
export MINDSPEED_MM_PATH=<YOUR_MINDSPEED_MM_PATH>
export LMSV_OUTPATH=./output
```

**注意**: 所有 PTA/MSA 脚本内部会在执行前将相对路径转换为绝对路径。

### 2. 推理模型特殊处理

#### 2.1 异常判断不能通过loss

**关键避坑点**：对于QwenVL等推理模型，**不能**通过检查是否有loss输出来判断是否异常。

**原因**：
- 推理模型（inference）没有训练过程，自然不会有loss输出
- 判断推理模型成功的标准是：**返回码是否为0**

**正确做法**：
```bash
# 错误：检查loss（仅适用于训练模型）
if ! grep -q "loss:" ${LOG_FILE}; then
    echo "ERROR: No loss found"
fi

# 正确：检查返回码
if [ "$RETURN_CODE" -ne 0 ]; then
    echo "ERROR: Execution failed"
fi
```

#### 2.2 日志指标提取

推理模型没有loss、memory、time等指标，这些字段会是None，这是**正常行为**。

### 3. Shell脚本避坑指南

#### 3.1 set -e 与 grep 的陷阱

**问题**：使用`set -e`时，grep找不到匹配会返回1，导致脚本退出。

**解决方案**：
```bash
# 错误：会导致脚本退出
if grep -q "pattern" file; then
    ...
fi

# 正确：添加错误处理
if grep -q "pattern" file 2>/dev/null; then
    ...
fi

# 或：使用 || true
grep -q "pattern" file 2>/dev/null || true
```

#### 3.2 变量提取时的错误处理

```bash
# 错误：grep失败会导致脚本退出
VAR=$(grep "pattern" file | grep -oP '...')

# 正确：添加错误处理
VAR=$(grep "pattern" file 2>/dev/null | grep -oP '...' 2>/dev/null || echo "")
```

### 4. MSA Pipeline Parallel 日志捕获修复

**问题**：Pipeline Parallel (PP>1)配置下，loss只在最后一个pipeline stage的worker日志中输出（如worker_7），而脚本默认读取worker_0.log

**影响**：训练模型（InternVL3、CogVideoX）的MSA执行无法正确捕获loss

**修复详情**：
1. **查找逻辑**：修改等待循环，持续查找所有worker日志直到发现包含`loss:`的日志文件
2. **科学计数法支持**：修改正则表达式`[\d.]+` → `[\d.E+-]+`，支持`1.269427E+01`格式的loss值
3. **分离指标提取**：loss从包含loss的worker提取，memory从所有worker中查找

**涉及脚本**：`msa_internvl3_8B_real.sh`、`msa_cogvideox_real.sh`

### 5. 训练模型MSA失败判定

**原则**：训练模型（InternVL3、CogVideoX）的MSA执行必须有loss输出，无loss视为失败

**错误信息提取**：优先提取Python Error（Traceback/OSError/RuntimeError等），无Error时取最后一个WARNING作为错误信息

**结果展示**：`status.json`中`reason`字段包含提取的错误信息

### 6. OpenSora推理优化

**问题1**：`num_inference_steps=30`导致显存不足
- **解决**：改为与Task6的`TRAIN_ITER`一致（默认5）

**问题2**：`--train-iters 5010`导致执行5010轮推理
- **解决**：改为`--train-iters ${TRAIN_ITERS:-1}`（`TRAIN_ITER`映射而来）只执行1次完整推理

**问题3**：prompts文件包含10个prompts，导致执行10组推理
- **解决**：使用单prompt文件（只取第一个prompt）

**效果**：现在只执行一组0/5 → 5/5的推理，耗时约140秒

### 7. MSA环境UntypedStorage兼容性

**问题**：`ModuleNotFoundError: No module named 'msadapter.UntypedStorage'`

**原因**：safetensors库需要`torch.UntypedStorage`，但msadapter缺少该属性

**解决方案**：在`mm_msa_opensora.sh`中创建`sitecustomize.py`注入FakeUntypedStorage

```python
class FakeUntypedStorage:
    def __init__(self, *args, **kwargs):
        self._data = bytearray()
        self._size = 0
    ...
```

**注意**：这是用户侧解决方案，FakeUntypedStorage 无法修复 safetensors Rust 扩展层面的调用问题。

### 8. MSA环境依赖版本

**正确的依赖版本组合**：

| 包 | 版本 | 说明 |
|---|---|---|
| python | 3.10 | - |
| mindspore | 2.7.1 | 固定版本 |
| msadapter | 0.0.5 | msadapter环境已安装 |
| numpy | 1.26.0 | **关键**：必须<=1.26.0 |
| ml_dtypes | 0.3.0 | **关键**：必须与numpy兼容 |
| scipy | 1.11.4 | 与numpy 1.26兼容 |

**环境修复命令**：
```bash
conda activate msadapter
pip install numpy==1.26.0 --force-reinstall
pip install ml_dtypes==0.3.0 --force-reinstall
pip install scipy==1.11.4 --force-reinstall
```

### 9. CogVideoX模型文件路径

**问题**：CogVideoX配置中`vae.safetensors`和`transformer`路径与实际文件不匹配

**实际文件结构**：
```
${DATASET_ROOT}/cogvideox/CogVideoX-5B/
├── vae/3d-vae.pt              # 不是vae.safetensors
├── text_encoder/              # 不是transformer
└── ...
```

**修复方案**：修改`assets/mm_configs/model_cogvideox.json`

### 10. dtype字符串解析问题（框架Bug）

**问题**：transformers库的`dict_dtype_to_str`函数无法处理"bf16"简写格式

**根因代码**：
```python
def dict_dtype_to_str(self, d):
    if d.get("dtype", None) is not None and not isinstance(d["dtype"], str):
        d["dtype"] = str(d["dtype"]).split(".")[1]  # <-- 问题所在
```

**分析**：
- 代码假设格式为"torch.float32"或"msadapter.float32"（包含点号）
- 分割后得到`["torch", "float32"]`，取索引1得到"float32"
- MSA环境配置中使用短格式"bf16"（不包含点号）
- 分割后得到`["bf16"]`，取索引1导致IndexError

**建议修复方案**：
1. 短期：在配置预处理中将简写格式转换为标准格式
2. 长期：向transformers库提交PR增强健壮性

### 11. Python作用域陷阱

**绝不在函数内部的条件分支中 `import` 模块**，否则会导致 `UnboundLocalError`。

```python
# 错误
import re  # 模块级导入

def process():
    if some_condition:
        import re  # 局部导入，使 re 变为局部变量！
    re.findall(...)  # UnboundLocalError!
```

---

## 统一入口

Task6 与其它 Task 共享同一入口：

```bash
cd /data2/lm-sv/lmsv_rec

# 交互式配置后执行（推荐）
./lmsv conf   # 选择 Task6，按提示输入模型名、MindSpeed-MM 路径、数据集路径、多机配置等
./lmsv do

# 一步完成配置+执行
./lmsv        # 等效于 ./lmsv conf && ./lmsv do

# 查看帮助
./lmsv help
```

`lmsv` 调用 `lmsv.py`，`lmsv.py` 调用 `do.py` 执行任务。

---

## 调试技巧

### 快速调试：使用单轮短步长配置

在调试 MSA 问题时，不需要修改代码注释 PTA，只需修改 `config.json` 缩短流程：

```json
{
  "task_type": 6,
  "tasks": {
    "6": {
      "MODEL_NAME": "cogvideox",
      "TOTAL_ITER": 1,
      "TRAIN_ITER": 1,
      "COMPARE_MODE": "pta_msa"
    }
  }
}
```

通过 `TOTAL_ITER=1` 和 `TRAIN_ITER=1` 可大幅缩短单轮调试时间，同时保持完整 PTA/MSA 流程不变。

### 手动执行脚本测试

```bash
export MINDSPEED_MM_PATH=<YOUR_MINDSPEED_MM_PATH>
export MM_MODEL=assets/mm_configs/inference_qwen2_5_vl_7b.json
bash scripts/runtime/mm_pta_qwenvl.sh
echo "EXIT CODE: $?"
```

### 查看详细日志

```bash
# PTA日志
tail -50 output/YYYYMMDD_HHMMSS/iters/iter_N/runtime_logs/pta_verify_iterN.log

# MSA日志
tail -50 output/YYYYMMDD_HHMMSS/iters/iter_N/runtime_logs/msa_verify_iterN.log

# msrun worker日志
tail -50 output/YYYYMMDD_HHMMSS/iters/iter_N/msrun_log/worker_*.log
```

### 环境检查

```bash
# 检查conda环境
conda env list | grep mm-

# 检查CANN环境
echo $ASCEND_HOME_PATH
ls /usr/local/Ascend/cann/

# 检查NPU
npu-smi info

# 检查模型配置文件
ls $MINDSPEED_MM_PATH/examples/internvl3/
ls $MINDSPEED_MM_PATH/scripts-ms/
```

---

## 历史修改记录

### 2026-04-29 报告格式改进
- **精度/显存/性能判定**：新增 `accuracy_ok`/`memory_ok`/`perf_ok` 三维度判定
- **报告显示数值差异**：Markdown/HTML 报告均显示具体差异值（如 `OK (diff=1.73%)`）
- **问题归类**：报告末尾增加按精度/显存/性能分类的汇总

### 2026-04-29 SSH_PORT 修复
- 修复 `_init_config()` 未传递 `SSH_PORT` 的问题
- 所有 SSH/rsync 命令支持 `-p` 参数，`-p` 在 `-o` 之前

### 2026-04-29 推理模型误判修复
- PTA 推理模型：returncode=0 时直接视为成功
- MSA fallback：仅在没有 error_info 时才设置 local_ok=True
- 精度判定：一方无数据时不设为 NG，显示 N/A

### 2026-04-28 多机部署完整验证
- 四模型（cogvideox / internvl3 / opensora / qwenvl）多机 PTA/MSA 均验证通过
- OpenSora / QwenVL MSA 预期失败（已知bug）

### 2026-04-26 Task6 多机推理模型Bug修复
- 修复 `cannot access local variable 're'`
- 修复 MSA bug信息被掩盖
- 修复推理模型错误显示loss=0.0
- 修复 qwen MSA失败后时间/显存N/A
- 修复多机推理模型PTA被误判为失败

### 2026-04-25 Task6 多机支持上线
- SSH + rsync + ThreadPoolExecutor 并发启动
- HCCL_DETERMINISTIC 多机死锁修复
- MSA 日志选择 bug 修复
- NFS 竞争清除 msrun_log 修复

### 2026-04-21 InternVL3 回归验证
- 路径迁移到 `/data2/lm-sv` 后完整回归
- 验证配置参数差异、日志完整性、报错记录、权重保存、msrun_log归档、有效突变率

### 2026-04-14 InternVL3 成功测试案例
- PTA Loss: 10.14411, MSA Loss: 12.19148, 差异 20.18%
- 后续修正 FP16/BF16 混用后降至 ~0.75%

### 2026-04-13 Task6 配置架构重构（与 Task1-5 统一）
- 移除 `TASK6_*` 环境变量
- 统一配置入口 `config.json` 的 `tasks["6"]`
- `SAVE_STEPS` 更名为 `TRAIN_ITER`
- 统一使用 `MINDSPEED_MM_PATH`

### 2026-04-09 OpenSora MSA修复
- `sitecustomize.py` 注入 FakeUntypedStorage

### 2026-04-09 QwenVL Task6修复
- grep 返回码保护
- `init_from_hf_path` 直接加载HF格式权重

### 2026-04-08 路径迁移
- 默认路径改为相对路径 `../mm-new`

---

## 参考文档

- `docs/task6.md` - 最终交付文档
- `docs/task6_statistics.md` - 统计规则说明
- `docs/task6_model_handling.md` - 四模型处理逻辑
- `docs/task6_pta_msa_precision_alignment.md` - 精度对齐完整进展
- `docs/TASK6_CUSTOMIZATION_MODIFICATIONS.md` - 环境定制化修改
- `docs/Task6_多机部署.md` - 多机部署指南
- `docs/Task6_多机部署_详细过程.md` - 多机部署详细过程
- `docs/TASK6_EXTERNAL_DEPENDENCIES.md` - 外部依赖说明
- `docs/test_plan_task6.md` - 测试计划
- `docs/PTA_MF_PRECISION_ALIGNMENT.md` - 环境配置参考
- `docs/how-to-add-a-new-task.md` - 任务扩展指南

---

## 日志路径速查

| 日志 | 路径 |
|-----|------|
| PTA运行日志 | `output/YYYYMMDD_HHMMSS/iters/iter_N/runtime_logs/pta_verify_iterN.log` |
| MSA运行日志 | `output/YYYYMMDD_HHMMSS/iters/iter_N/runtime_logs/msa_verify_iterN.log` |
| msrun worker日志 | `output/YYYYMMDD_HHMMSS/iters/iter_N/msrun_log/worker_*.log` |
| 任务汇总报告 | `output/YYYYMMDD_HHMMSS/analysis/summary.md` |
| HTML报告 | `output/YYYYMMDD_HHMMSS/analysis/report.html` |
| 迭代metrics | `output/YYYYMMDD_HHMMSS/iters/iter_N/metrics.json` |
| 迭代status | `output/YYYYMMDD_HHMMSS/iters/iter_N/status.json` |

---

## 注意事项

1. **不要同时运行多个 Task6 进程**
2. **每次启动前 kill 残留进程**（已实现到代码中 `kill_pretraingpt()`）
3. **多机模式下必须关闭 ENABLE_OVERLAP**，否则 HCCL 死锁
4. **不要设置 ASCEND_LAUNCH_BLOCKING=1**，会导致多机分布式 hang
5. **定期清理 `/dev/shm/psm_*`**，避免 NPU 内存分配失败
6. **远程节点的环境变更必须同步**（conda 包、sysctl 配置等）
7. **所有路径必须通过 `config.json` 配置**，代码中不硬编码任何路径假设

---

**文档版本**: 2.0
**更新日期**: 2026-04-29（最终版，所有修复已验证）
**适用范围**: Task6开发维护（单机 + 多机）

---

## 2026-04-29 LMSV_PATH 指向 PROJECT_ROOT 兼容修复

**问题**：客户要求 `LMSV_PATH` 指向 `PROJECT_ROOT`（如 `/home/jenkins/workspace/TDT_deployment/zjc/lm-sv`），而非 `lmsv_rec` 本身。

**根因**：`_local_to_remote_path()` 和 `run_remote_pta_verify()` 中的 `cd` 目标都假设 `LMSV_PATH` 指向 `lmsv_rec`。

**修复**：新增 `_get_remote_lmsv_root()` 和 `_get_remote_project_root()` 两个辅助函数：
- 如果 `LMSV_PATH` 以 `lmsv_rec` 结尾 → 直接返回（兼容旧配置）
- 如果 `LMSV_PATH` 指向 `PROJECT_ROOT` → 自动追加 `/lmsv_rec`

**影响文件**：`utils/task/task6.py`
- `_local_to_remote_path()`：使用 `_get_remote_lmsv_root()` 拼接 `lmsv_rec` 内部路径
- `_local_to_remote_path()`：使用 `_get_remote_project_root()` 拼接 `output` 等外部路径
- `_build_path_prefix_mappings()`：从 `LMSV_ROOT -> remote_lmsv_root` 和 `PROJECT_ROOT -> remote_project_root` 推导映射
- `run_remote_pta_verify()` / `run_remote_msa_verify()`：`cd` 目标改为 `_get_remote_lmsv_root(node)`

**客户配置示例**（主从机文件结构相同）：
```json
{
  "LMSV_PATH": "/home/jenkins/workspace/TDT_deployment/zjc/lm-sv",
  "PTA_PATH": "/home/jenkins/workspace/TDT_deployment/zjc/MindSpeed-Core-MS-MM",
  "MSA_PATH": "/home/jenkins/workspace/TDT_deployment/zjc/MindSpeed-Core-MS-MM",
  "MINDSPEED_MM_PATH": "/home/jenkins/workspace/TDT_deployment/zjc/MindSpeed-Core-MS-MM/MindSpeed-MM"
}
```

**注意**：主从机文件结构必须相同（即远程节点上存在 `/home/jenkins/workspace/TDT_deployment/zjc/lm-sv/lmsv_rec/`）。

---

## 2026-04-29 pgrep 自匹配 Bug（MSA脚本无限等待）

**问题**：`msa_cogvideox_real.sh` 中使用 `pgrep -f "pretrain_sora.py"` 检测进程是否存在，但当脚本通过SSH/bash wrapper执行时，命令行本身包含 `"pretrain_sora.py"`，导致 `pgrep -f` 匹配到bash进程自身，永远返回非空，脚本无限等待 "Stabilizing: waiting for all iterations"。

**根因**：`pgrep -f` 匹配完整命令行，而SSH/bash wrapper的命令行包含目标字符串。

**修复**：将 `pgrep -f "pretrain_sora.py"` 替换为 `ps aux | grep "[p]retrain_sora.py"`。
- `[p]retrain_sora.py` 的方括号技巧使grep进程自身的命令行不匹配（因为包含 `[p]` 而非 `p`）
- 从而只匹配真正的 `pretrain_sora.py` 进程

**影响文件**：`scripts/runtime/msa_cogvideox_real.sh`

**客户注意**：如果客户自行修改了MSA脚本中的进程检测逻辑，必须使用 `ps aux | grep "[p]attern"` 而非 `pgrep -f "pattern"`。

---

## 环境版本信息（客户部署参考）

### 老机器 (192.168.0.170)

| 项目 | 版本 |
|------|------|
| NPU | Ascend 910B x 4 |
| npu-smi | 24.1.0.3 |
| Driver | 24.1.0.3 |
| CANN | 8.3.0.1.200:8.3.RC1 |
| Python | 3.13.9 |
| OS | Linux 5.10.0-136.12.0.86.r1526_92.hce2.aarch64 |

### 新机器 (192.168.0.203)

| 项目 | 版本 |
|------|------|
| NPU | Ascend 910B x 4 |
| npu-smi | 23.0.6 |
| Driver | 24.1.0.3 |
| CANN | 8.3.0.1.200:8.3.RC1 |
| Python | 3.10.20 |
| OS | Linux 5.10.0-136.12.0.86.r1526_92.hce2.aarch64 |

**关键差异**：
1. Python版本不同（3.13.9 vs 3.10.20）—— 建议客户统一
2. npu-smi工具版本不同（24.1.0.3 vs 23.0.6）—— Driver Version相同，不影响
3. `/shared/mindspeed-mm/` 目录结构不同（老机器真实目录 vs 新机器symlink）—— **必须统一**
