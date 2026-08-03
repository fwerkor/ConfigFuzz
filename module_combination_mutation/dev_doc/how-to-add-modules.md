# 多模态模块内模块间泛化变异测试扩展

本文面向后续开发者，说明如何扩展本工具中的模块与模板能力，并让新增能力同时接入模块间变异和模块内变异流程。

本工具包含两条主流程：

- **模块间组合变异**（`mutate.py`）：将不同模块类型按模板范式组合，形成新的多模态模型，并执行前向/反向验证；
- **模块内变异**（`single_module_mutate.py`）：固定单一模块类型（如 `text_decoder`/`image_encoder`），在配置空间内连续扰动并逐轮回归验证。

两条流程都会从 `modules.json` 读取模块池并构建模块实例，验证通过后保留可用配置，用于后续 MSA/PTA 环境的差分测试。

因此，扩展能力可分为四层：

1. **扩展一个新的模块类型**（例如新增 `image_encoder` 这一类能力，或类比新增 `ae`）；
2. **在某个模块类型下扩展一个具体模块**（例如在 `image_encoder` 下新增 `llava` 对应的 image encoder）；
3. **扩展一个新的模板**（定义新的模块组合路径、调用顺序和前向验证逻辑）；
4. **将新增模块接入模块内变异**（配置可变参数范围，并增加对应单模块模板）。

---

## 一、如何扩展一个新的模块类型（以 `image_encoder` 为例）

这一节讲“类型级接入”要改哪些地方。你可以把它当成一个通用模板：后续新增 `video_encoder`、`ae` 等类型时，按同样路径走。

### 步骤 1：新增模块类型实现文件，定义基类与注册表

在 `modules/` 下新增对应文件（`image_encoder.py` 已是现成示例），至少要包含：

- 类型基类（如 `ImageEncoder`）；
- 基类接口建议：提供 `_build()` 用于实例化内部模块对象（例如 image encoder 的`VisionModel`等、text decoder 的`MOEModel`和`GPTModel`等）；同时暴露该类型的核心能力接口（例如 text decoder 的 `embedding()`、`decode()`，image encoder 的 `encode()`），便于模板层统一调用；
- 构建器字典（如 `IMAGE_ENCODER_BUILDERS`）；
- 注册装饰器（如 `@register_image_encoder(...)`）。

示例（节选自 `modules/image_encoder.py`）：

```python
IMAGE_ENCODER_BUILDERS = {}

def register_image_encoder(*names: str):
    def decorator(cls):
        for name in names:
            IMAGE_ENCODER_BUILDERS[name] = cls
        return cls
    return decorator

class ImageEncoder:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config

    def _build(self):
        pass

    def encode(self, images):
        pass
```

### 步骤 2：在 `modules/pool.py` 增加该类型的 DictPool 与 build 逻辑

模块池职责是：

- 存放该类型所有候选配置；
- 支持随机选取（`random_choice`）；
- 支持按名字构建实例（`build`）。

示例（节选）：

```python
class ImageEncoderDictPool:
    def __init__(self):
        self.image_encoders = {}

    def register_one(self, name: str, config: dict):
        self.image_encoders[name] = config

    def random_choice(self) -> dict:
        name = secrets.choice(list(self.image_encoders.keys()))
        return {name: self.image_encoders[name]}

    def build(self, name: str, config: dict) -> ImageEncoder:
        EncoderClass = IMAGE_ENCODER_BUILDERS[name]
        return EncoderClass.build(name, config)
```

> 如果你新增的是 `ae` 这类类型，需要按同样方式新增 `AEDictPool`，并提供 `AE_BUILDERS` 与 `register_ae`。

### 步骤 3：在 `modules.json` 中为新类型建立顶层键

`mutate.py` 会从 `modules.json` 读取模块池，所以必须有清晰的顶层结构。  
`image_encoder` 对应的是顶层键 `image_encoders`：

```json
{
  "image_encoders": {
    "llava": {
      "vision_encoder": { "...": "..." },
      "vision_projector": { "...": "..." }
    }
  }
}
```

后续若扩展新类型（如 `ae`），建议同样采用复数键（如 `aes`），并保持“名称 -> 配置”结构一致。

### 步骤 4：在主流程 `mutate.py` 中注册该类型

关键是把该类型从 `all_modules` 取出，并注入对应 DictPool。

示例（当前 `image_encoder` 逻辑）：

```python
def register_all_modules(all_modules, args):
    text_decoders = all_modules["text_decoders"]
    image_encoders = all_modules["image_encoders"]

    for name, config in text_decoders.items():
        TEXT_DECODER_DICT_POOL.register_one(name, config)
    for name, config in image_encoders.items():
        IMAGE_ENCODER_DICT_POOL.register_one(name, config)
```

如果新增类型（例如 `ae`），这里要继续补充：

```python
aes = all_modules["aes"]
for name, config in aes.items():
    AE_DICT_POOL.register_one(name, config)
```

### 步骤 5：更新模板，让新类型真正参与组合

仅注册到池子还不够，模板里也要“选中 + 构建 + 使用”。

`templates/image_model_template.py` 中已有 `image_encoder` 的标准用法：

- `select_modules()`：`IMAGE_ENCODER_DICT_POOL.random_choice()`
- `instantiate()`：`IMAGE_ENCODER_DICT_POOL.build(...)`
- `forward()`：调用 `instance.image_encoder.encode(images)`

新增模块类型时，建议在模板中按这 3 个阶段补齐。

---

## 二、如何在模块类型下扩展一个具体模块（以 `llava` 的 `image_encoder` 为例）

这一节讲“类型内扩点”：即 `image_encoder` 类型已经存在，现在再加一个具体实现。

### 步骤 1：在 `modules/image_encoder.py` 中实现子类并注册名字

通过装饰器把字符串名字绑定到实现类。

示例（`llava`）：

```python
@register_image_encoder("llava")
class LLavaImageEncoder(ImageEncoder):
    @classmethod
    def build(cls, name: str, config: dict) -> "ImageEncoder":
        return cls(name, config)

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        # 这里完成模型配置装配和子模块构建
        self._build()

    def encode(self, images):
        return self.encoder(images)
```

要点：

- 装饰器中的名字（这里是 `"llava"`）必须和 `modules.json` 中键名完全一致；
- `build` 建议保持统一签名，便于池子统一调用；
- `encode` 输出 shape 需满足下游 `combine strategy` 期望。

### 步骤 2：在 `modules.json` 的 `image_encoders` 下新增该模块配置

`llava` 的图像编码器通常含两部分：`vision_encoder` 与 `vision_projector`。  
示例（简化版，字段请按真实需求补齐）：

```json
{
  "image_encoders": {
    "llava": {
      "vision_encoder": {
        "model_id": "clip",
        "num_layers": 23,
        "hidden_size": 1024,
        "image_size": 336,
        "params_dtype": "fp32"
      },
      "vision_projector": {
        "model_id": "mlp",
        "num_layers": 2,
        "input_size": 1024,
        "hidden_size": 4096,
        "params_dtype": "fp32"
      }
    }
  }
}
```

### 步骤 3：确认配置预处理规则不会破坏该模块

`mutate.py` 中 `config_preprocess()` 会递归处理：

- `params_dtype` 字符串转 `torch.dtype`；
- `vision_encoder` / `vision_projector` 的嵌套字段；
- 自动对齐 `vision_projector.input_size = vision_encoder.hidden_size`。

因此你需要重点检查：

- 新模块是否依赖特殊字段（避免被默认裁剪逻辑影响）；
- `hidden_size`、`num_attention_heads`、`num_query_groups` 等约束是否兼容。

### 步骤 4：运行主脚本验证是否能被随机选中并成功前向

常用命令（在 `module_combination_mutation` 目录）：

```bash
python mutate.py
```

验证点：

- 日志中出现 `image encoder: llava`；
- 能完成 `forward -> loss -> backward`；
- 结果目录下生成对应 `configs/round_*.json` 和 `dots/graph_round*.dot`。

---

## 三、如何扩展模板（Template）

这一节讲“组合范式扩展”：当现有模板（如 `image_model`）不能覆盖你的新测试场景时，如何新增一个模板并接入主流程。

### 步骤 1：在 `templates/` 下新增模板类文件

建议参考 `templates/image_model_template.py` 的结构，新建例如 `templates/your_template.py`，并继承 `Template`。

最少要实现的方法：

- `select_modules()`：定义如何从模块池选择本模板需要的模块；
- `instantiate()`：将模块配置构建为 `MMInstance`；
- `forward()`：定义模块连接顺序与前向路径；
- `dump_graph()`：导出该模板的组合图（便于排查结构问题）。

示例骨架：

```python
from templates.template import Template, MMInstance
from modules.pool import TEXT_DECODER_DICT_POOL, IMAGE_ENCODER_DICT_POOL

class YourTemplate(Template):
    def __init__(self):
        super().__init__("your_template")
        self.text_decoder_name = None
        self.text_decoder_config = None
        self.image_encoder_name = None
        self.image_encoder_config = None

    def select_modules(self):
        td = TEXT_DECODER_DICT_POOL.random_choice()
        self.text_decoder_name = list(td.keys())[0]
        self.text_decoder_config = td[self.text_decoder_name]

        ie = IMAGE_ENCODER_DICT_POOL.random_choice()
        self.image_encoder_name = list(ie.keys())[0]
        self.image_encoder_config = ie[self.image_encoder_name]

    def instantiate(self) -> MMInstance:
        instance = MMInstance(name=self.name, config={}, template=self)
        # 根据模板需要 set/add 模块
        return instance

    def forward(self, instance: MMInstance, *args, **kwargs):
        # 定义组合后的前向逻辑
        return ...
```

### 步骤 2：在模板中明确“模块接口契约”

模板应只依赖模块基类暴露的稳定接口，而不是具体实现细节。  
例如：

- `text decoder`：优先调用 `embedding()`、`decode()`；
- `image encoder`：调用 `encode()`。

这样做的好处是：后续新增具体模块（比如新的 decoder/encoder）时，只要满足接口契约，就可以直接被该模板复用。

### 步骤 3：在主流程注册新模板

在 `mutate.py` 的 `register_all_templates()` 中注册新模板实例：

```python
from templates.image_model_template import ImageModelTemplate
from templates.your_template import YourTemplate

def register_all_templates():
    TEMPLATE_REGISTRY.register(ImageModelTemplate())
    TEMPLATE_REGISTRY.register(YourTemplate())
```

注册后，主循环中的 `TEMPLATE_REGISTRY.random_choice()` 就会随机抽到你的模板。

### 步骤 4：验证模板是否真正生效

执行：

```bash
python mutate.py
```

重点观察：

- 日志里 `Template` 字段是否出现新模板名；
- 新模板是否能走通 `select -> instantiate -> forward -> backward`；
- 是否正常产出 `configs/` 与 `dots/` 文件。

如果只想验证该模板，可临时让 `register_all_templates()` 只注册它一个，先通过功能回归后再恢复随机组合。

---

## 四、如何将新增模块用于模块内变异

本节对应模块内变异主流程 `single_module_mutate.py`。  
目标是：让你新增的模块不仅能参与“模块间组合变异”（`mutate.py`），也能参与“单模块连续扰动与回归验证”。

至少需要完成两步：**设置变异参数及范围**、**增加对应的单模块模板**。

### 步骤 1：设置变异参数及范围（`single_module_mutate_dict.json`）

`single_module_mutate.py` 会通过 `SingleModuleMutator(schema_path=...)` 读取 `single_module_mutate_dict.json`，并在每轮迭代中调用：

```python
mutated_config = single_mutator.mutate(
    module_type=module_type,
    base_config=current_config,
    mutation_num=3,
)
```

因此，你需要在 schema 中为目标模块类型配置可变字段和范围。常见形式：

- 连续/整数扰动：`min_factor`、`max_factor`、`min_val`、`max_val`
- 枚举扰动：`enums`
- 临时禁用字段：`enabled: false`

示例（节选）：

```json
{
  "text_decoder": {
    "num_layers": {
      "min_factor": 0.5,
      "max_factor": 2.0,
      "min_val": 4,
      "max_val": 128
    },
    "attention_dropout": {
      "enums": [0.0, 0.1, 0.2, 0.3]
    }
  },
  "image_encoder": {
    "hidden_dropout": {
      "enums": [0.0, 0.05, 0.1]
    },
    "add_qkv_bias": {
      "enums": [true, false]
    }
  }
}
```

如果你新增的是新模块类型（如 `ae`），需要在 schema 顶层增加对应键（如 `"ae"`），否则变异器无法命中该类型字段。

### 步骤 2：增加对应的单模块模板并注册

模块内变异依赖单模块模板来完成三件事：

1. 从模块池选择一个基线模块（`select_modules`）；
2. 支持外部读写“当前被变异模块”的配置（`get_module_config` / `set_module_config`）；
3. 用最小前向路径验证该模块在变异后仍可运行（`forward`）。

可以参考已有模板：

- `templates/single_text_decoder_template.py`
- `templates/single_image_encoder_template.py`

单模块模板建议最少实现以下接口：

```python
class SingleXXXTemplate(Template):
    def get_module_type(self) -> SupportedModules: ...
    def get_module_name(self) -> str: ...
    def get_module_config(self) -> dict: ...
    def set_module_config(self, module_config: dict): ...
```

完成模板后，在 `single_module_mutate.py` 的 `register_single_templates()` 中注册：

```python
def register_single_templates():
    TEMPLATE_REGISTRY.register(SingleTextDecoderTemplate())
    TEMPLATE_REGISTRY.register(SingleImageEncoderTemplate())
    # TEMPLATE_REGISTRY.register(SingleAETemplate())  # 新增类型时补充
```

如果需要通过 `--type` 精确筛选类型，还要确保：

- `get_module_type()` 返回值与 `SupportedModules` 枚举一致；
- `_parse_type_to_supported_module()` 能解析你的类型别名。

### 可选步骤：运行验证与结果检查

执行示例：

```bash
python single_module_mutate.py --type image_encoder --rounds 3 --iterations 20
```

关注以下结果：

- 日志中是否出现你新增模块类型/模板；
- 每轮迭代是否能完成 `forward -> loss -> backward`；
- `results/_single_module/.../configs/` 下是否生成 `round-iteration-module_type.json` 快照。

---

## 常见问题与排查建议

1. **报错 `Invalid image encoder: xxx`**  
   说明实现类未注册或名字不一致。优先检查 `@register_image_encoder("xxx")` 与 `modules.json` 键名。

2. **报 dtype 不匹配（Conv/MatMul）**  
   检查 `params_dtype` 与输入图片 dtype 是否一致；当前模板会按 `vision_encoder.params_dtype` 生成 `images`。

3. **组合后 shape 不匹配**  
   检查 `vision_projector.input_size` 与 `vision_encoder.hidden_size`，以及 `encode` 输出是否符合融合策略输入要求。

4. **模块加了但从不被选中**  
   检查是否被正确写入 `modules.json` 对应顶层键，并确认 `mutate.py` 的 `register_all_modules()` 已注册该类型。

---

按上面的“模块类型扩展 + 具体模块扩展 + 模板扩展 + 模块内变异接入”四个层次推进，可以比较稳定地把新能力接入当前泛化变异测试框架。