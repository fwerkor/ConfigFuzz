# 多模态模块变异工具说明

## 1 简介

本目录下提供了一套**多模态模块变异工具链**，包括两类能力：

- **模块级组合变异（module combination mutation）**：从已有多模态模型中抽取可复用模块（如 text decoder / image encoder），按预置模板范式进行**跨模型组合与重连**；
- **模块内细粒度变异（single module mutation）**：在固定单个模块（text decoder / image encoder）的前提下，对其配置进行**小步参数/结构扰动**并回归验证。

整体目标是：

- 快速发现框架 / 算子 / 配置在“非常规模块组合或极端配置”下的潜在问题；
- 产出一批**可复现的 JSON 配置文件**，用于后续回归与最小化复现；
- 支撑 MindSpeed-MM + Ascend NPU 场景下，对多模态系统做系统化“压力测试 + Fuzzing”。

### 1.1 目录结构（概览）

```
.
├── mm_mutate.sh                     # 模块组合变异脚本（单卡，产出多组组合配置）
├── mm_test.sh                       # 组合配置回归测试脚本（多卡，输入为 round_*.json 目录）
├── mutate.py                        # 模块组合变异主程序
├── mm_test.py                       # 根据组合变异配置构建/测试模型
├── single_module_mutate.sh          # 单模块细粒度变异启动脚本
├── single_module_mutate.py          # 单模块细粒度变异主程序
├── single_module_mutate_dict.json   # 单模块变异 schema / 搜索空间定义
├── modules.json                     # 可复用模块池（text/image encoder/decoder 配置）
├── common.py                        # 全局公用变量/常量等（如并行 size 等）
├── results/
│   ├── mutate_YYYYMMDD_HHMMSS/             # 组合变异单次运行目录（自动生成）
│   │   ├── train.log                       # 训练/测试日志
│   │   ├── profile/                        # 性能分析信息
│   │   └── configs/                        # 该轮所有模块组合配置 round_*.json
│   └── _single_module/
│       └── mutate_single_YYYYMMDD_HHMMSS/  # 单模块变异单次运行目录（自动生成）
│           ├── *.json                      # 每轮/每次迭代的单模块配置快照
│           └── profile/                    # 若开启 profile，将写入该目录
├── modules/
│   ├── pool.py                        # 模块池、注册相关
│   ├── text_decoder.py                # 文本解码器实现
│   ├── image_encoder.py               # 图像编码器实现
│   └── ...
├── templates/                         # 组合范式模板
│   ├── template.py
│   ├── single_text_decoder_template.py
│   ├── single_image_encoder_template.py
│   └── ...
├── it_combine/                        # 图文组合策略实现
│   └── ...
├── utils.py                           # 通用工具函数（如路径、日志、序列化等）
└── README.md                          # 本说明文档
```




## 2 功能支持（目前）

- **支持范式**：文本-图像（Text-Image / VL）
- **支持模块类型**：
  - 组合变异：来自 `deepseekv3_vl`、`deepseekvl2`、`llava1.5` 等模型的 `text decoder` / `image encoder` 组合；
  - 单模块变异：对上述池内单个 `text decoder` 或 `image encoder` 的配置做细粒度扰动。
- **支持图文组合策略**：插入（`interleave`）
- **运行模式**：
  - **模块组合变异**：多轮随机组合 + 每轮一次最小训练闭环（前向 → loss → backward）；
  - **模块内变异**：在固定模块上多次迭代变异，并对每次变异后的配置做最小闭环验证。

## 3 工具使用方法

### 3.1 依赖与前置条件

本目录下脚本默认运行在 MindSpeed-MM + Ascend NPU 环境中。

- **Ascend 环境**：确保已安装 Ascend Toolkit，且可以执行 `source /usr/local/Ascend/ascend-toolkit/set_env.sh`
- **MindSpeed相关依赖**：设置mindspeed相关的PYTHONPATH，示例如下：
```
/path/to/MindSpeed_Core_MS/MSAdapter:/path/to/MindSpeed_Core_MS/MSAdapter/msa_thirdparty:/path/to/MindSpeed_Core_MS/Megatron-LM:/path/to/MindSpeed_Core_MS/MindSpeed:/path/to/MindSpeed_Core_MS/MindSpeed-MM
```

### 3.2 自动化脚本

自动化脚本负责组织整个测试流程，包括：变异 -> 配置验证 -> 权重保存 -> PTA训练 -> MSA训练 -> 差分验证。
如需单独执行其中的变异或者跑测，参考3.3-3.5。

#### 3.2.1 模块间组合变异
**前置条件**：修改 `./task/module_combine_config.yaml`中的`global.conda_env_pta`和`global.MindSpeed_Core_MS_PATH`为自己的PTA conda环境名与mindspeed_core_ms源码路径；
同时，用户也可以根据自身需求，自定义调整 `module_combine_config.yaml` 中的其他配置项（如文件输出路径、组合变异次数等），整体配置说明可参考文件内注释。

**运行:**
```
python task/module_combine.py
```

#### 3.2.2 模块内组合变异
**前置条件**：修改 `./task/single_module_config.yaml`中的`global.conda_env_pta`和`global.MindSpeed_Core_MS_PATH`为自己的PTA conda环境名与mindspeed_core_ms源码路径；
同时，用户也可以根据自身需求，自定义调整 `single_module_config.yaml` 中的其他配置项（如文件输出路径、组合变异次数等），整体配置说明可参考文件内注释。

**运行:**
```
python task/single_module.py
```

### 3.3 模块间组合变异：生成组合配置文件（单卡）

```bash
bash ./mm_mutate.sh
```

运行后将进行多轮（默认 10 轮）**模块级随机组合**，每一轮会做一次最小训练闭环（前向 -> loss -> backward），并在结果目录中落盘该轮的组合配置文件。

### 3.4 基于组合配置实例化模型进行训练测试（多卡）

```bash
bash mm_test.sh ./results/mutate_xxx/configs/
bash mm_test.sh ./results/mutate_xxx/configs/ --no-mm-test-optimizer
bash mm_test.sh ./results//mutate_20260311_084708/configs/ --save-ckpt
bash mm_test.sh ./results//mutate_20260311_084708/configs/ --load-ckpt --load-ckpt-dir ./results/test_20260316_101436/ckpts/
```

其中：

- `./results/mutate_xxx/configs/`：为本次变异生成的配置目录（每轮一个 `round_*.json`）
- `--no-mm-test-optimizer`：仅做前向/反向或跳过优化器相关（具体以 `mm_test.sh` 实现为准）

### 3.5 模块内细粒度变异（单模块变异，单卡）

除“模块级组合变异”外，本工具还支持**在单一模块内部做细粒度参数/结构变异**，用于：

- 探索某个 `text decoder` 或 `image encoder` 的鲁棒性边界与潜在问题；
- 针对单模块做更细粒度的 Fuzzing 与兼容性测试；
- 为后续手工调参或架构改动提供“问题种子配置”。

**核心脚本与流程：**

- 核心脚本：`single_module_mutate.py`；
- 每一轮随机选择一个模块类型（`TEXT_DECODER` 或 `IMAGE_ENCODER`）及其具体实现（由模板完成选择）；
- `iteration=0` 使用原始配置，其后在上一轮配置基础上做多次小步变异；
- 每次变异后都会执行一次最小闭环（前向 → Loss → 反向），
  - 若成功：将该轮配置落盘为 JSON，并作为下一次迭代的“基线配置”；
  - 若失败：自动回退到变异前配置，继续后续迭代。

推荐使用提供的 `single_module_mutate.sh` 脚本启动（已封装 Ascend 环境变量、分布式参数与日志目录）：

```bash
cd ./module_combination_mutation
bash single_module_mutate.sh
```

其中：

- `--all-modules`：模块池 JSON 文件路径，与组合变异共用，默认 `./modules.json`
- `--rounds`：变异轮数，默认 `10`。每一轮固定一个模块，再在其配置空间内做多次迭代变异
- `--iterations`：**每轮的变异迭代次数**，默认 `10`
- `--results-dir`：结果目录，默认 `./results/_single_module`

运行结束后，每个通过最小闭环验证的配置会以 JSON 形式保存在 `--results-dir` 对应运行子目录下，命名形如：

- `round-iteration-module_type.json`，例如：`3-1-image_encoder.json`、`9-8-text_decoder.json`

这些单模块配置可以：

- 单独用于复现某个算子/配置问题；
- 作为后续组合变异或手工建模的“种子配置”进一步扩展。

## 4 常用参数

### 4.1 组合变异相关（`mm_mutate.sh` / `mutate.py`）

`mm_mutate.sh`（内部使用 `torchrun mutate.py ...`），核心参数通常包括：

- **`--rounds`**：组合变异轮数，默认 `10`
- **`--all-modules`**：模块池 JSON 路径，默认 `./modules.json`
- **`--results-dir`**：结果目录（本次运行的根目录），通常为 `./results`

### 4.2 组合变异测试相关（`mm_test.sh` / `mm_test.py`）

`mm_test.sh`（内部使用 `torchrun mm_test.py ...`），核心参数如下：

- **`<config_dir>`**：必需参数，配置目录，含多轮 `round_*.json`（即组合变异阶段产出的每轮模块组合配置）
- **`--no-mm-test-optimizer`**：（可选）仅做前向/反向，跳过 optimizer 相关步骤
- **`RESULTS_DIR`**：环境变量（可选），自定义结果输出基目录

### 4.3 单模块变异相关（`single_module_mutate.py`）

单模块变异通常通过 `single_module_mutate.sh` 启动，核心配置以**环境变量**形式传入：

- **`ALL_MODULES`**：模块池 JSON 文件路径，对应 `--all-modules`，默认 `./modules.json`
- **`ROUNDS`**：单模块变异轮数，对应 `--rounds`，默认 `10`
- **`ITERATIONS`**：每轮的变异迭代次数，对应 `--iterations`，默认 `10`
- **`RESULTS_DIR`**：单模块变异结果基目录，对应 `--results-dir` 的上级目录，默认 `./results/_single_module`

示例：

```bash
cd ./module_combination_mutation
RESULTS_DIR=/path/to/my_single_results \
ROUNDS=5 ITERATIONS=8 \
bash single_module_mutate.sh
```

更多分布式与 Ascend 相关参数（如 `NPUS_PER_NODE`、`MASTER_PORT` 等）可参考脚本内注释进行调整。

其他参数可参考各脚本内注释及命令行 `--help`。



## 5 结果目录与产物

### 5.1 组合变异结果（`results/mutate_*`）

每次运行组合变异，会在 `results`（或你指定的 `--results-dir`/`RESULTS_DIR`）下创建一个独立文件夹：

- `results/mutate_YYYYMMDD_HHMMSS/`
  - `train.log`：该次运行的完整日志（`mm_mutate.sh` 里 `tee` 输出）
  - `configs/round_0.json`、`configs/round_1.json` ...：每一轮生成的模块组合配置（可用于复现/回归）
  - `profile/`：性能分析输出（若开启 profile 相关参数，会被重定向到该目录）

### 5.2 单模块变异结果（`results/_single_module/mutate_single_*`）

每次运行单模块变异（`single_module_mutate.py`），会在 `--results-dir`（默认 `./results/_single_module`）下创建一个独立子目录：

- `results/_single_module/mutate_single_YYYYMMDD_HHMMSS/`
  - `*.json`：每轮/每次迭代的单模块配置快照，命名格式为 `round-iteration-module_type.json`
    - 示例：`1-1-text_decoder.json`、`3-1-image_encoder.json`、`9-8-text_decoder.json`
  - `profile/`：若开启 profile 相关参数，会将 profile 输出重定向到该目录

这些 JSON 文件可以直接被后续脚本读取，或手工拷贝/修改后再作为新的实验起点。

### 5.3 指定自定义结果目录

- **组合变异**：在 `mm_mutate.sh` 中通过 `--results-dir ${RUN_DIR}` 传入；
- **单模块变异**：通过 `single_module_mutate.py` 的 `--results-dir` 指定；
- **测试阶段（推荐）**：通过环境变量 `RESULTS_DIR` 指定 results 基目录（每次仍会在其下创建带时间戳的子目录）。

示例：

```bash
RESULTS_DIR=/path/to/my_results bash mm_mutate.sh
```



## Troubleshooting

- **卡在 PyTorch extensions lock**：`mm_mutate.sh` 已尝试清理 `${TORCH_EXTENSIONS_DIR:-~/.cache/torch_extensions}` 下的陈旧锁文件；如仍卡住，检查缓存目录权限或手动清理。
- **端口冲突 `Bind_Failed`**：`mm_mutate.sh` 默认随机选择 `27000–27999` 的 `MASTER_PORT`；必要时可 `MASTER_PORT=27001 bash mm_mutate.sh` 固定端口。
- **路径不匹配/ImportError**：优先检查 `mutate.py` 顶部的 MindSpeed 相关 `sys.path` 是否与你环境一致，以及 `MM_DATA/MM_MODEL/MM_TOOL` 是否存在且可读。