# Task6 标准化裸环境 (Standard Environment)

本目录存放 Task6 的**裸 conda 环境**定义，即仅包含 `requirements.txt` 中声明的标准库、**不含任何定制化修改**的 conda 环境。

## 用途

这是自动化脚本 `../automated_setup/setup_task6_envs.sh` 的**起点**。工作流程如下：

```
standard_env (裸环境)
    |
    v
加载/还原裸环境到 conda
    |
    v
运行 setup_task6_envs.sh (一键定制化修改)
    |
    v
得到完整的 Task6 运行环境
```

## 环境文件

| 文件 | 说明 | 对应 conda 环境名 |
|------|------|------------------|
| `mindspeed_bare.yml` | PTA 侧裸环境定义 (conda env export) | `mindspeed` |
| `msadapter_bare.yml` | MSA 侧裸环境定义 (conda env export) | `msadapter` |

## 从裸环境还原

### 方式一：通过 yml 文件创建新环境

```bash
# PTA 环境
conda env create -f mindspeed_bare.yml -n mindspeed

# MSA 环境
conda env create -f msadapter_bare.yml -n msadapter
```

### 方式二：通过 requirements.txt 安装

```bash
# 1. 创建基础环境
conda create -n mindspeed python=3.10 -y
conda create -n msadapter python=3.10 -y

# 2. 激活并安装依赖
conda activate mindspeed
pip install -r ../automated_setup/requirements.txt

conda activate msadapter
pip install -r ../automated_setup/requirements.txt
```

### 方式三：通过 conda-pack 导出/导入（推荐用于跨机器部署）

```bash
# 导出（在一台已配置好的机器上执行）
conda install -c conda-forge conda-pack -y
conda pack -n mindspeed -o mindspeed_bare.tar.gz
conda pack -n msadapter -o msadapter_bare.tar.gz

# 导入（在新机器上执行）
mkdir -p ~/conda_envs/mindspeed ~/conda_envs/msadapter
tar -xzf mindspeed_bare.tar.gz -C ~/conda_envs/mindspeed
tar -xzf msadapter_bare.tar.gz -C ~/conda_envs/msadapter

# 注册到 conda
conda info --base  # 获取 conda 根目录
# 在 conda 根目录的 envs/ 下创建软链接
ln -s ~/conda_envs/mindspeed $(conda info --base)/envs/mindspeed
ln -s ~/conda_envs/msadapter $(conda info --base)/envs/msadapter
```

## 验证是否为裸环境

裸环境**不应包含**任何以下定制化修改：

1. **transformers 补丁**：`modeling_utils.py` 中不应有 `expected_keys=expected_keys` 等关键字参数修改
2. **msadapter bfloat16 fallback**：`mm-new/msadapter/msadapter/_utils.py` 和 `serialization.py` 应为原始版本（无 `ml_dtypes.bfloat16` fallback）
3. **decord 运行时补丁**：不应存在 `/tmp/decord_patch/decord_fix.py`
4. **libstdc++ 兼容性修复**：`LD_LIBRARY_PATH` 不应包含 conda 的 lib 路径

验证脚本：

```bash
# 验证 transformers 未打补丁
conda activate msadapter
python -c "
import transformers, inspect
src = inspect.getsourcefile(transformers.modeling_utils)
with open(src) as f:
    content = f.read()
has_patch = 'expected_keys=expected_keys' in content
print(f'transformers {transformers.__version__}: patch applied = {has_patch}')
assert not has_patch, 'ERROR: transformers is patched!'
print('PASS: transformers is bare')
"
```

## 与定制化环境的关系

| 特性 | 裸环境 (standard_env) | 定制化环境 (运行 setup_task6_envs.sh 后) |
|------|----------------------|----------------------------------------|
| transformers | 原始版本 (4.55.2) | 已应用签名补丁（所有版本均需，因 msadapter 装饰器会改变参数传递方式） |
| msadapter 源码 | 原始版本 | `_utils.py` 和 `serialization.py` 增加 bfloat16 fallback |
| decord | 原始版本 | 运行时通过 `mm-pta-task6.sh` 自动 patch |
| libstdc++ | 使用系统默认 | 运行时通过 `mm-msa-task6.sh` 优先使用 conda 版本 |

## 注意事项

1. **环境隔离**：裸环境和定制化环境使用**相同的环境名**（`mindspeed` 和 `msadapter`），只是状态不同
2. **快速回滚**：如需回滚到裸状态，可删除当前环境并从 yml 重新创建
3. **版本锁定**：yml 文件使用 `--no-builds` 导出，跨平台兼容性更好

---

*文档更新时间：2026-04-20*
