# Task6 数据集文件结构与还原指南

本文档详细说明 `/data2/dataset` 的层级文件结构、`/data2/data2_zips` 中的打包方式，以及使用 zips 中的包完全还原 Task6 四个模型所需数据的完整过程。

---

## 1. 总体概览

```
/data2/dataset/                    # 数据集根目录，总计约 570GB
├── cogvideox/                     # CogVideoX 模型数据 (71GB) — Task6
├── cogvideox_data.zip             # cogvideox 小数据压缩包 (48MB)
├── data/                          # 共享数据目录 (101GB)
├── data.zip                       # data/ 目录压缩包 (96GB)
├── internvl3/                     # InternVL3 模型数据 (281GB) — Task6
├── llava1.5/                      # LLaVA 1.5 数据 (28GB)
├── opensora1.2/                   # OpenSora 1.2 模型数据 (28GB) — Task6
├── qwen2.5vl/                     # Qwen2.5VL 模型数据 (72GB) — Task6
├── tokenizer.json                 # 共享 tokenizer (7.5MB)
├── wiki_4096.json                 # Wiki 文本数据 (491MB)
├── wiki_4096_text_document/       # Megatron 格式文本数据 (125MB)
├── wiki_4096_text_document.bin    # (421MB)
└── wiki_4096_text_document.idx    # (528KB)

/data2/data2_zips/                 # 分割 tar 归档存储目录，总计约 440GB
├── cogvideox_01                   # CogVideoX tar 分卷 1 (41GB)
├── cogvideox_02                   # CogVideoX tar 分卷 2 (31GB)
├── internvl3_1 ~ internvl3_7      # InternVL3 tar 分卷 1~7 (共 285GB)
├── opensora1.2_1                  # OpenSora tar 单卷 (28GB)
├── qwen2.5vl_1                    # Qwen2.5VL tar 分卷 1 (41GB)
└── qwen2.5vl_2                    # Qwen2.5VL tar 分卷 2 (32GB)
```

---

## 2. Task6 四个模型详细文件结构

### 2.1 CogVideoX (`/data2/dataset/cogvideox/` — 71GB)

Task6 CogVideoX 使用此目录作为 `DATASET_PATH` 和 `LOAD_PATH`。

```
cogvideox/
├── astronaut.jpg                           # 示例图片 (936KB)
├── data.json                               # 数据集元数据 (4KB)
├── data.jsonl                              # 训练数据列表 (40KB)
├── labels/                                 # 视频标注目录 (280KB)
│   ├── 0288f3d69c08e816d81b014da620db49.txt
│   ├── 05a234b0164d015d468f2f53e771b4cf.txt
│   └── ... (共约 70 个 txt 标注文件)
├── videos/                                 # 训练视频目录 (25MB)
│   ├── 0288f3d69c08e816d81b014da620db49.mp4
│   ├── 05a234b0164d015d468f2f53e771b4cf.mp4
│   └── ... (共 69 个 mp4 视频文件)
├── CogVideoX-5B/                           # CogVideoX-5B 模型权重 (59GB)
│   ├── model_index.json
│   ├── README.md / README_zh.md
│   ├── scheduler/                          # 调度器配置 (16KB)
│   ├── text_encoder/                       # T5 文本编码器 (18GB)
│   │   ├── config.json
│   │   ├── model-00001-of-00002.bin        # 4.7GB
│   │   ├── model-00001-of-00002.safetensors # 4.7GB
│   │   ├── model-00002-of-00002.bin        # 4.3GB
│   │   ├── model-00002-of-00002.safetensors # 4.3GB
│   │   ├── model.safetensors.index.json
│   │   └── pytorch_model.bin.index.json
│   ├── tokenizer/                          # T5 Tokenizer (848KB)
│   │   ├── added_tokens.json
│   │   ├── spiece.model
│   │   ├── tokenizer_config.json
│   │   └── special_tokens_map.json
│   ├── transformer/                        # Transformer 权重 (15GB)
│   │   └── 1000/
│   │       └── mp_rank_00_model_states.pt
│   ├── transformer_5B/                     # 5B Transformer (11GB)
│   │   ├── config.json
│   │   ├── diffusion_pytorch_model-00001-of-00002.safetensors
│   │   ├── diffusion_pytorch_model-00002-of-00002.safetensors
│   │   └── diffusion_pytorch_model.safetensors.index.json
│   ├── transformer_t2v/                    # T2V Transformer (15GB)
│   │   └── 1000/
│   │       └── mp_rank_00_model_states.pt
│   ├── vae/                                # VAE 权重 (1.1GB)
│   └── vae_old/                            # 旧版 VAE (823MB)
├── transformer/                            # 独立 transformer 目录 (10GB)
│   ├── t5-v1_1-xxl/                        # T5-XXL 文本编码器 (8.9GB)
│   └── vae/                                # VAE (1.1GB)
├── vae/                                    # 独立 VAE 目录 (1.1GB)
├── i2v_1.0/                                # I2V 1.0 配置 (44KB)
├── i2v_1.5/                                # I2V 1.5 配置及测试 (687MB)
│   └── test/
│       └── iter_0000002/
│           └── mp_rank_00/
├── t2v_1.0/                                # T2V 1.0 配置 (40KB)
├── t2v_1.5/                                # T2V 1.5 配置 (64KB)
├── cogvideox_hf_convert_to_mm_ckpt.py      # 转换脚本
├── cogvideox_sat_convert_to_mm_ckpt.py
├── cogvideox_lora_dataset_convert.py
├── test.py
├── README.md
└── .ipynb_checkpoints/                     # Jupyter checkpoint 文件
```

**关键说明：**
- `CogVideoX-5B/` 是主模型目录，包含完整的模型权重和配置
- `data.jsonl` 格式示例：`{"video": "videos/xxx.mp4", "caption": "..."}`
- `labels/` 和 `videos/` 一一对应，构成训练数据对

---

### 2.2 OpenSora 1.2 (`/data2/dataset/opensora1.2/` — 28GB)

Task6 OpenSora 是推理模式，使用此目录加载模型权重。

```
opensora1.2/
├── data.json                               # 数据配置 (4KB)
├── model.json                              # 模型配置 (4KB)
├── README.md
├── inference_model_102x720x1280.json       # 推理配置 (4KB)
├── pretrain_opensora1_2.sh                 # 预训练脚本 (4KB)
├── inference_opensora1_2.sh                # 推理脚本 (4KB)
├── t5-v1_1-xxl/                            # T5-XXL 文本编码器 (18GB)
│   ├── config.json
│   ├── pytorch_model-00001-of-00002.bin    # ~9GB
│   ├── pytorch_model-00002-of-00002.bin    # ~9GB
│   ├── pytorch_model.bin.index.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   └── spiece.model
├── OpenSora-STDiT-v3/                      # STDiT-v3 模型权重 (4.6GB)
│   ├── config.json
│   ├── model.safetensors                   # 主权重文件
│   └── README.md
├── OpenSora-VAE-v1.2/                      # OpenSora VAE (1.5GB)
│   ├── config.json
│   └── model.safetensors
├── PixArt-alpha/                           # PixArt 初始化权重 (2.3GB)
│   └── PixArt-XL-2-512x512.pth
├── sd-vae-ft-ema/                          # SD VAE EMA 版本 (639MB)
│   ├── config.json
│   ├── diffusion_pytorch_model.bin
│   └── diffusion_pytorch_model.safetensors
├── sd-vae-ft-mse/                          # SD VAE MSE 版本 (639MB)
│   ├── config.json
│   └── ...
└── msr-vtt/                                # MSR-VTT 测试视频 (288KB)
    └── *.mp4 (约 70 个短视频片段)
```

**关键说明：**
- `t5-v1_1-xxl/` 是最大的子目录，占 18GB，为文本编码器
- `OpenSora-STDiT-v3/model.safetensors` 是扩散模型主权重 (4.6GB)
- OpenSora 在 Task6 中运行在推理模式，不直接使用训练数据

---

### 2.3 Qwen2.5VL (`/data2/dataset/qwen2.5vl/` — 72GB)

Task6 Qwen2.5VL 是推理模式，使用此目录加载模型权重和数据。

```
qwen2.5vl/
├── ckpt/
│   ├── Qwen2.5-VL-7B-Instruct/             # HuggingFace 格式权重 (16GB)
│   │   ├── config.json
│   │   ├── generation_config.json
│   │   ├── chat_template.json
│   │   ├── tokenizer_config.json
│   │   ├── tokenizer.json
│   │   ├── vocab.json
│   │   ├── merges.txt
│   │   ├── preprocessor_config.json
│   │   ├── model.safetensors.index.json
│   │   ├── model-00001-of-00005.safetensors  # ~3.2GB
│   │   ├── model-00002-of-00005.safetensors
│   │   ├── model-00003-of-00005.safetensors
│   │   ├── model-00004-of-00005.safetensors
│   │   └── model-00005-of-00005.safetensors
│   └── Qwen2.5-VL-7B-Instruct-MM/          # Megatron-MM 格式权重 (16GB)
│       └── release/
│           └── mp_rank_00_000/
│               └── model_optim_rng.pt
├── data/                                   # 训练/推理数据 (41GB)
│   ├── COCO2017/                           # COCO 图片 (19GB)
│   │   └── train2017/
│   │       └── 000000*.jpg (~118349 张)
│   ├── cache_dir/                          # 数据集缓存 (1.5GB)
│   │   └── json/default-*/0.0.0/... (HuggingFace datasets 缓存)
│   ├── llava_instruct_150k.json            # LLaVA 指令数据 (219MB)
│   ├── mllm_format_llava_instruct_data.json # MLLM 格式数据 (210MB)
│   ├── train2017.zip                       # COCO 训练集 zip (19GB)
│   └── .locks/                             # HuggingFace lock 文件
├── data_7b.json / data_3b.json / data_32b.json / data_72b.json  # 各尺寸数据配置
├── model_7b.json / model_3b.json / model_32b.json / model_72b.json # 各尺寸模型配置
├── finetune_qwen2_5_vl_7b.sh              # 微调脚本
├── inference_qwen2_5_vl_7b.sh             # 推理脚本
├── evaluate_qwen2_5_vl_7b.sh
├── README.md
└── .ipynb_checkpoints/
```

**关键说明：**
- `ckpt/Qwen2.5-VL-7B-Instruct/` 为 HuggingFace 格式权重，共 5 个 safetensors 分片
- `ckpt/Qwen2.5-VL-7B-Instruct-MM/` 为 MindSpeed-MM 格式权重 (model_optim_rng.pt)
- `data/COCO2017/train2017/` 包含约 11.8 万张图片
- Task6 默认使用 7B 配置

---

### 2.4 InternVL3 (`/data2/dataset/internvl3/` — 281GB)

Task6 InternVL3 使用此目录作为 `DATASET_PATH` 和 `LOAD_PATH`。

```
internvl3/
├── pretrained/
│   └── InternVL3-8B/                       # MindSpeed-MM 预训练权重 (15GB)
│       ├── latest_checkpointed_iteration.txt
│       └── release/
│           └── mp_rank_00/
│               └── model_optim_rng.pt        # 主权重文件
├── raw_ckpt/
│   └── InternVL3-8B/                       # HuggingFace 原始权重 (15GB)
│       ├── config.json
│       ├── generation_config.json
│       ├── tokenizer_config.json
│       ├── tokenizer.json
│       ├── vocab.json
│       ├── merges.txt
│       ├── preprocessor_config.json
│       ├── special_tokens_map.json
│       ├── model.safetensors.index.json
│       ├── model-00001-of-00004.safetensors  # ~3.8GB
│       ├── model-00002-of-00004.safetensors
│       ├── model-00003-of-00004.safetensors
│       ├── model-00004-of-00004.safetensors
│       ├── modeling_intern_vit.py
│       ├── modeling_internvl_chat.py
│       ├── configuration_intern_vit.py
│       ├── configuration_internvl_chat.py
│       ├── conversation.py
│       └── examples/                       # 示例文件
│           ├── image1.jpg
│           ├── image2.jpg
│           └── red-panda.mp4
└── playground/                             # 训练数据目录 (251GB)
    ├── data/                               # 图片数据集 (250GB)
    │   ├── ai2d/                           # AI2D 图表数据 (1.6GB)
    │   │   ├── images/
    │   │   ├── annotations/
    │   │   └── ...
    │   ├── chartqa/                        # ChartQA 数据
    │   │   ├── train/png/ / test/png/ / val/png/
    │   │   └── tables/
    │   ├── coco/                           # COCO 数据 (19GB)
    │   │   └── train2017/
    │   ├── docvqa/                         # DocVQA 文档问答 (17GB)
    │   │   ├── train/documents/ / test/documents/ / val/documents/
    │   │   └── ocr_results/
    │   ├── dvqa/                           # DVQA 数据 (8GB)
    │   │   └── images/
    │   ├── geoqa+/                         # GeoQA+ 几何问答
    │   │   └── images/ / json-image_files/
    │   ├── gqa/                            # GQA 视觉问答 (21GB)
    │   │   └── images/
    │   ├── llava/                          # LLaVA 数据 (27GB)
    │   │   └── llava_pretrain/
    │   ├── ocr_vqa/                        # OCR-VQA (9GB)
    │   │   └── images/
    │   ├── sam/                            # SAM 分割数据 (9.3GB)
    │   │   └── images/
    │   ├── share_textvqa/
    │   ├── synthdog-en/                    # SynthDog 合成文档 (2.4GB)
    │   ├── textvqa/                        # TextVQA (6.7GB)
    │   ├── vg/                             # Visual Genome (20GB)
    │   ├── web-celebrity/
    │   ├── web-landmark/
    │   └── wikiart/
    ├── opensource/                         # 开源指令数据集 (1.5GB)
    │   ├── sharegpt4v_instruct_gpt4-vision_cap100k.jsonl
    │   ├── sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.jsonl
    │   ├── llava_instruct_150k_zh.jsonl
    │   ├── ai2d_train_12k.jsonl
    │   ├── chartqa_train_18k.jsonl
    │   ├── docvqa_train_10k.jsonl
    │   ├── dvqa_train_200k.jsonl
    │   ├── geoqa+.jsonl
    │   └── synthdog_en.jsonl
    └── README.md
```

**关键说明：**
- `pretrained/` 为 MindSpeed-MM 训练用格式 (model_optim_rng.pt)
- `raw_ckpt/` 为 HuggingFace 原始格式 (safetensors 分片)
- `playground/data/` 是主要训练图片数据，约 250GB，包含 15+ 个视觉数据集
- `playground/opensource/` 为指令微调用的 JSONL 格式数据
- Task6 data config 引用：`{{DATASET_PATH}}/playground/opensource/sharegpt4v_instruct_gpt4-vision_cap100k.jsonl`

---

## 3. 共享数据目录 (`/data2/dataset/data/` — 101GB)

此目录存放 Task6 各模型共享或历史使用的训练数据。

```
data/
├── cogvideox/                              # CogVideoX 训练数据副本 (49MB)
│   ├── cogvideox-disney.tar.gz
│   ├── data.jsonl
│   ├── labels/                             # 与 /data2/dataset/cogvideox/labels 内容相同
│   └── videos/                             # 与 /data2/dataset/cogvideox/videos 内容相同
├── coco-data-sampled/                      # COCO 采样数据 (16MB)
│   └── COCO2017/train2017/
├── LLaVA-Instruct-150K/                    # LLaVA 指令数据 (1.6GB)
│   └── .cache/
├── LLaVA-Pretrain/                         # LLaVA 预训练数据 (53GB)
│   └── images/
├── msr-vtt/                                # MSR-VTT 视频数据 (4.2GB)
│   ├── clips/                              # 视频片段 (2.1GB)
│   ├── video/                              # 原始视频 (2.1GB)
│   ├── raw_data/                           # 原始元数据 (48MB)
│   ├── parts/                              # 分割数据 (2.2MB)
│   ├── clips_meta.csv / meta.csv           # 元数据 CSV
│   ├── msrvtt_train_7k.json / msrvtt_train_9k.json / msrvtt_test_1k.json
│   └── data.json / data_sampling.py
├── Open-Sora-1.2.0/                        # OpenSora 源码副本 (8.6MB)
│   ├── opensora/ / configs/ / scripts/ / tools/ ...
│   └── pretrained_models/
├── open-sora-pexels-45k/                   # OpenSora Pexels 数据 (空目录)
├── qwen2vl/                                # Qwen2VL 训练数据 (39GB)
│   ├── data/
│   │   ├── COCO2017/                       # COCO 图片
│   │   ├── llava_instruct_150k.json
│   │   └── mllm_format_llava_instruct_data.json
│   ├── train2017.zip                       # COCO zip (19GB)
│   └── cache_dir/                          # 数据集缓存 (1.5GB)
└── kernel_meta/                            # 昇腾算子缓存
```

---

## 4. `/data2/data2_zips` 打包方式说明

### 4.1 打包原理

`data2_zips` 中的文件是使用 **tar + split** 方式创建的归档分卷：

1. 先用 `tar cvf - <目录>` 将整个模型目录打包为 tar 流
2. 再用 `split -b 40G` 将 tar 流分割为每卷约 40GB 的文件
3. 结果：每个模型产生一个或多个无扩展名的分卷文件

**打包特征：**
- 文件类型：POSIX tar archive (GNU)
- 分卷大小：约 40GB/卷（最后一个分卷通常不足 40GB）
- 必须按顺序 `cat` 合并所有分卷后才能解压

### 4.2 各模型 tar 包详情

| 模型 | 分卷文件 | 总大小 | tar 内路径前缀 | 条目数 |
|------|----------|--------|----------------|--------|
| **CogVideoX** | `cogvideox_01` + `cogvideox_02` | 72GB | `./` (相对路径) | 411 |
| **OpenSora** | `opensora1.2_1` (单卷) | 28GB | `data2/dataset/opensora1.2/` | 196 |
| **Qwen2.5VL** | `qwen2.5vl_1` + `qwen2.5vl_2` | 73GB | `data2/dataset/qwen2.5vl/` | 146,436 |
| **InternVL3** | `internvl3_1` ~ `internvl3_7` | 285GB | `data2/dataset/internvl3/` | 1,685,341 |

**路径前缀差异说明：**
- **CogVideoX** 使用 `./` 相对路径打包，解压时需在目标目录内执行（或在 tar 命令中用 `-C` 指定 `/data2/dataset`）
- **OpenSora / Qwen2.5VL / InternVL3** 使用绝对路径 `data2/dataset/...` 打包，解压时需在根目录 `/` 执行（或在 tar 命令中用 `-C /`）

### 4.3 额外 zip 包

除 tar 分卷外，dataset 根目录还有两个独立 zip：

| 文件 | 大小 | 内容 | 解压前缀 |
|------|------|------|----------|
| `data.zip` | 96GB | `data/` 目录全部内容 | `data/` |
| `cogvideox_data.zip` | 48MB | `cogvideox/` 的 labels + videos | `cogvideox/` |

---

## 5. 完整还原步骤与具体指令

以下指令演示如何在一个全新环境中，从 `data2_zips` 完全还原 Task6 四个模型的全部数据到 `/data2/dataset`。

### 5.1 前提准备

```bash
# 1. 确保目标目录存在
sudo mkdir -p /data2/dataset

# 2. 确保磁盘空间充足（四个模型+共享数据约需 570GB）
df -h /data2

# 3. 确保有 zips 目录的读取权限
ls -la /data2/data2_zips/
```

### 5.2 还原 CogVideoX (71GB)

```bash
# 方式一：cd 到目标目录后解压（推荐，因 tar 内路径为 ./）
cd /data2/dataset
cat /data2/data2_zips/cogvideox_01 /data2/data2_zips/cogvideox_02 | tar -xvf -

# 方式二：使用 -C 指定解压目录
cat /data2/data2_zips/cogvideox_01 /data2/data2_zips/cogvideox_02 | tar -xvf - -C /data2/dataset
```

**验证：**
```bash
du -sh /data2/dataset/cogvideox          # 应显示约 71GB
ls /data2/dataset/cogvideox/CogVideoX-5B # 应存在
ls /data2/dataset/cogvideox/videos       # 应存在 69 个 mp4
```

### 5.3 还原 OpenSora 1.2 (28GB)

```bash
# tar 内路径为 data2/dataset/opensora1.2/，需从根目录解压
cd /
cat /data2/data2_zips/opensora1.2_1 | tar -xvf -

# 或等价方式：
cat /data2/data2_zips/opensora1.2_1 | tar -xvf - -C /
```

**验证：**
```bash
du -sh /data2/dataset/opensora1.2              # 应显示约 28GB
ls /data2/dataset/opensora1.2/OpenSora-STDiT-v3 # 应存在 model.safetensors
```

### 5.4 还原 Qwen2.5VL (72GB)

```bash
# tar 内路径为 data2/dataset/qwen2.5vl/，需从根目录解压
cd /
cat /data2/data2_zips/qwen2.5vl_1 /data2/data2_zips/qwen2.5vl_2 | tar -xvf -

# 或等价方式：
cat /data2/data2_zips/qwen2.5vl_1 /data2/data2_zips/qwen2.5vl_2 | tar -xvf - -C /
```

**验证：**
```bash
du -sh /data2/dataset/qwen2.5vl                       # 应显示约 72GB
ls /data2/dataset/qwen2.5vl/ckpt/Qwen2.5-VL-7B-Instruct  # 应存在 5 个 safetensors
ls /data2/dataset/qwen2.5vl/data/COCO2017/train2017   # 应存在大量 jpg
```

### 5.5 还原 InternVL3 (281GB)

```bash
# tar 内路径为 data2/dataset/internvl3/，需从根目录解压
cd /
cat /data2/data2_zips/internvl3_1 \
    /data2/data2_zips/internvl3_2 \
    /data2/data2_zips/internvl3_3 \
    /data2/data2_zips/internvl3_4 \
    /data2/data2_zips/internvl3_5 \
    /data2/data2_zips/internvl3_6 \
    /data2/data2_zips/internvl3_7 | tar -xvf -

# 或等价方式：
cat /data2/data2_zips/internvl3_* | tar -xvf - -C /
```

**验证：**
```bash
du -sh /data2/dataset/internvl3                        # 应显示约 281GB
ls /data2/dataset/internvl3/pretrained/InternVL3-8B    # 应存在
ls /data2/dataset/internvl3/playground/data/coco       # 应存在
ls /data2/dataset/internvl3/playground/opensource      # 应存在多个 jsonl
```

### 5.6 还原共享数据 (可选)

如果 `/data2/dataset/data.zip` 和 `/data2/dataset/cogvideox_data.zip` 可用，可直接在 dataset 根目录解压：

```bash
cd /data2/dataset

# 还原 data/ 目录（96GB，包含 LLaVA、MSR-VTT、Qwen2VL 数据等）
unzip -o data.zip

# 还原 cogvideox_data/（48MB，labels 和 videos 的小数据包）
unzip -o cogvideox_data.zip
```

**注意：**
- `data.zip` 解压后内容位于 `/data2/dataset/data/`
- `cogvideox_data.zip` 解压后内容位于 `/data2/dataset/cogvideox/`
- 若已用 tar 还原了完整 cogvideox，则 `cogvideox_data.zip` 的内容会被覆盖但内容一致（labels 和 videos 完全相同）

---

## 6. 一键还原脚本

将以下内容保存为 `restore_task6_datasets.sh`：

```bash
#!/bin/bash
set -e

ZIPS_DIR="/data2/data2_zips"
DATASET_DIR="/data2/dataset"

echo "========================================"
echo "Task6 Dataset Restore Script"
echo "========================================"

# 检查空间
echo "[1/5] 检查磁盘空间..."
AVAILABLE=$(df -BG "$DATASET_DIR" | awk 'NR==2 {print $4}' | tr -d 'G')
if [ "$AVAILABLE" -lt 600 ]; then
    echo "ERROR: 磁盘空间不足，需要至少 600GB，当前可用 ${AVAILABLE}GB"
    exit 1
fi

# 创建目录
mkdir -p "$DATASET_DIR"

# 还原 CogVideoX
echo "[2/5] 还原 CogVideoX (约 72GB)..."
cat "$ZIPS_DIR"/cogvideox_01 "$ZIPS_DIR"/cogvideox_02 | tar -xf - -C "$DATASET_DIR"

# 还原 OpenSora
echo "[3/5] 还原 OpenSora 1.2 (约 28GB)..."
cat "$ZIPS_DIR"/opensora1.2_1 | tar -xf - -C /

# 还原 Qwen2.5VL
echo "[4/5] 还原 Qwen2.5VL (约 72GB)..."
cat "$ZIPS_DIR"/qwen2.5vl_1 "$ZIPS_DIR"/qwen2.5vl_2 | tar -xf - -C /

# 还原 InternVL3
echo "[5/5] 还原 InternVL3 (约 281GB)..."
cat "$ZIPS_DIR"/internvl3_1 "$ZIPS_DIR"/internvl3_2 "$ZIPS_DIR"/internvl3_3 \
    "$ZIPS_DIR"/internvl3_4 "$ZIPS_DIR"/internvl3_5 "$ZIPS_DIR"/internvl3_6 \
    "$ZIPS_DIR"/internvl3_7 | tar -xf - -C /

# 还原共享数据（可选）
if [ -f "$DATASET_DIR/data.zip" ]; then
    echo "[可选] 还原共享数据 data.zip (约 96GB)..."
    cd "$DATASET_DIR" && unzip -qo data.zip
fi

if [ -f "$DATASET_DIR/cogvideox_data.zip" ]; then
    echo "[可选] 还原 cogvideox_data.zip (约 48MB)..."
    cd "$DATASET_DIR" && unzip -qo cogvideox_data.zip
fi

echo "========================================"
echo "还原完成，校验中..."
echo "========================================"

du -sh "$DATASET_DIR"/cogvideox
ls "$DATASET_DIR"/cogvideox/CogVideoX-5B > /dev/null 2>&1 && echo "  [OK] CogVideoX-5B 存在" || echo "  [FAIL] CogVideoX-5B 缺失"

du -sh "$DATASET_DIR"/opensora1.2
ls "$DATASET_DIR"/opensora1.2/OpenSora-STDiT-v3 > /dev/null 2>&1 && echo "  [OK] OpenSora-STDiT-v3 存在" || echo "  [FAIL] OpenSora-STDiT-v3 缺失"

du -sh "$DATASET_DIR"/qwen2.5vl
ls "$DATASET_DIR"/qwen2.5vl/ckpt/Qwen2.5-VL-7B-Instruct > /dev/null 2>&1 && echo "  [OK] Qwen2.5-VL-7B-Instruct 存在" || echo "  [FAIL] Qwen2.5-VL-7B-Instruct 缺失"

du -sh "$DATASET_DIR"/internvl3
ls "$DATASET_DIR"/internvl3/pretrained/InternVL3-8B > /dev/null 2>&1 && echo "  [OK] InternVL3-8B 存在" || echo "  [FAIL] InternVL3-8B 缺失"

echo "========================================"
echo "Task6 数据集还原完成！"
echo "========================================"
```

**执行：**
```bash
chmod +x restore_task6_datasets.sh
./restore_task6_datasets.sh
```

---

## 7. 路径映射（多机部署场景）

在多机部署时，远程节点的数据集可以通过 NFS 或本地同步方式准备。Task6 通过 `DATASET_ROOT` 环境变量读取数据路径：

| 节点 | 实际路径 | 映射后路径 |
|------|----------|------------|
| Master | `/data2/dataset` | `/data2/dataset` |
| Worker | `<WORKER_DATASET_ROOT>` | `<WORKER_DATASET_ROOT>` |

配置方式：
```bash
# 在 config.json 中设置
{
  "DATASET_ROOT": "/data2/dataset"
}

# 或在环境变量中设置
export DATASET_ROOT=/data2/dataset
```

Task6 启动时会读取 `DATASET_ROOT` 并自动替换配置文件中的 `{{DATASET_PATH}}` 和 `{{LOAD_PATH}}` 占位符。

---

## 8. 常见问题

### Q1: tar 解压时报 "Unexpected EOF"
**原因：** 分卷未按正确顺序合并，或某个分卷损坏。
**解决：** 确保 `cat` 时按编号顺序拼接所有分卷（如 `internvl3_1` ~ `internvl3_7`）。

### Q2: 解压后文件不在预期位置
**原因：** CogVideoX tar 路径前缀为 `./`，其余模型为 `data2/dataset/...`，解压目录不同。
**解决：** CogVideoX 用 `-C /data2/dataset`；其余模型用 `-C /`。

### Q3: 解压时权限不足
**原因：** `/data2` 目录需要 root 权限。
**解决：** `sudo mkdir -p /data2/dataset` 并确保当前用户有写入权限。

### Q4: 是否需要同时解压 data.zip？
**答：** 若已完整还原四个模型的 tar 包，则 `data.zip` 中的大部分内容（如 `data/qwen2vl/`）可能被模型自身目录覆盖或重复。建议先还原 tar 包，再按需解压 `data.zip` 补充缺失内容。

---

*文档版本：2026-04-29*
*适用项目：lm-sv Task6 多模态整网变异对齐*
