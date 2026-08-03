# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
"""Pretrain LLaVA."""
from copy import deepcopy
import dataclasses
import time
import torch
import torch.nn.functional as F
import mindspeed.megatron_adaptor
import json
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.transformer import TransformerConfig
from megatron.training import get_args, print_rank_0, get_timers
from megatron.training.utils import average_losses_across_data_parallel_group
from mindspeed_mm.configs.config import mm_extra_args_provider
from mindspeed_mm.models.vl_model import VLModel
from mindspeed_mm.training import pretrain
from mindspeed_mm.configs.config import MMConfig
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.utils.transformer_model_config import get_model_config
from mindspeed_mm.models.common.module_spec.llava_layer_spec import get_layer_spec, get_mlp_module_spec
from datetime import datetime

npu_mem = torch.npu
import os
import time
import pandas as pd
# from demo import *
# args = get_args()

import random


def model_provider(pre_process=True, post_process=True):
    """Builds the model."""
    args = get_args()
    vlm_config = deepcopy(args.mm.model)
    print_rank_0("building LLaVA model ...")
    vlm_config.text_decoder = get_model_config(vlm_config.text_decoder)
    vlm_config.text_decoder.language_tansformer_layer_spec = get_layer_spec(is_vit=False)
    vlm_config.image_encoder.vision_encoder = get_model_config(vlm_config.image_encoder.vision_encoder)
    vlm_config.image_encoder.vision_encoder.vision_transformer_layer_spec = get_layer_spec(is_vit=True)
    vlm_config.image_encoder.vision_projector = get_model_config(vlm_config.image_encoder.vision_projector)
    vlm_config.image_encoder.vision_projector.vision_projection_layer_spec = get_mlp_module_spec(use_te=False).submodules
    vlm_config.pre_process = pre_process
    vlm_config.post_process = post_process
    model = VLModel(vlm_config)
    model_text_decoder = model.get_GPTModel()
    model_type_textdecoder = "textdecoder"

    model_text_encoder = model.TextEncoder_init()
    model_type_textencoder = "textencoder"
    root = "/data/mm/MindSpeed-Core-MS/MindSpeed-MM/mm_mutation_results/mm_mutation_results_all_20260129_211225/"
    for i in range(1,100):
        json_path = root + "mutation_gen" + str(i) + ".json"
        graph_init_2(model_text_decoder,model_type_textdecoder,model_text_encoder,model_type_textencoder,json_path)
    exit()
    model.freeze(vlm_config.text_decoder.freeze, vlm_config.image_encoder.vision_encoder.freeze, vlm_config.image_encoder.vision_projector.freeze)
    
    
    return model


def add_custom_node(graph, node_id, model, model_type='custom'):
    node = MMNode(config=graph.predict_config, index=node_id, model_type=model_type)
    node.set_mm_model(model, state='normal', str_op=model_type)
    graph.nodes[node_id] = node
    return node


def append_log_to_excel(
    jsonpath: str,
    flag: str,
    loss: float = None,
    elapsed_ms: float = None,
    peak_mb: float = None,
    curr_mb: float = None,
    log_path: str = "training_logs.xlsx"
):
    # 构造一行日志数据
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "json_path": jsonpath,
        "status": flag,
    }
    # 如果是完整模式则加上这些字段
    if loss is not None:
        row.update({
            "loss": f"{loss:.6f}",
            "duration_ms": f"{elapsed_ms:.2f}",
            "npu_peak_mem_mb": f"{peak_mb:.1f}",
            "npu_cur_mem_mb": f"{curr_mb:.1f}"
        })

    # 将单行转换为 DataFrame
    df_row = pd.DataFrame([row])

    # 如果文件已存在，则追加，否则新建
    if os.path.exists(log_path):
        # 注意：openpyxl 引擎支持 Excel 追加写入
        with pd.ExcelWriter(log_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
            # 找到已有的行数，然后从下一行开始写
            book = writer.book
            ws = book.active
            start_row = ws.max_row
            df_row.to_excel(writer, index=False, header=False, startrow=start_row)
    else:
        # 首次写入，保留表头
        df_row.to_excel(log_path, index=False)

    # 控制台输出
    print("✓ 前向+反向+优化 完成")
    for k, v in row.items():
        print(f"    {k.ljust(15)}: {v}")


def get_batch(data_iterator):
    """Generate a batch."""
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    images = data["pixel_values"].to(dtype=torch.bfloat16, device=torch.cuda.current_device())
    input_ids = data["input_ids"].to(device=torch.cuda.current_device())
    labels = data["labels"].to(device=torch.cuda.current_device())
    attention_mask = data["attention_mask"].to(device=torch.cuda.current_device())

    return images, input_ids, labels, attention_mask


def loss_func(output_tensor):
    """Loss function."""
    averaged_loss = average_losses_across_data_parallel_group([output_tensor])
    loss = output_tensor.unsqueeze(0).clone()
    return loss, {"loss": averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    timers = get_timers()
    images, input_ids, labels, attention_mask = get_batch(data_iterator)
    timers("batch-generator").stop()
    position_ids = None
    output_tensor = model(
        images,
        input_ids,
        position_ids,
        attention_mask,
        labels
    )

    # print(output_tensor)
    # print("yessssssssssssssssssss")
    # exit()
    return output_tensor, loss_func

import sys

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)  # 输出到控制台
        self.log.write(message)       # 写入日志文件

    def flush(self):
        self.terminal.flush()
        self.log.flush()



def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_dataset = build_mm_dataset(args.mm.data.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
        consumed_samples=args.consumed_train_samples,
    )
    return iter(train_dataloader), None, None

import os
import copy


def mutate_json_all(r,n,mutnm,file_path):
    mutator = MMConfigMutator()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
    # 输入输出路径
    
    output_dir = f"/shared/mm/MindSpeed-Core-MS/MindSpeed-MM/mm_mutation_results/mm_mutation_results_all_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始配置（多模型）
    with open(file_path, 'r', encoding='utf-8') as f:
        base_config = json.load(f)

    # 准备当前配置副本，用于迭代变异
    current_configs = {model_key: copy.deepcopy(cfg)
                       for model_key, cfg in base_config.items()}

    # 设置变异率
    mutation_rate = 1

    # 迭代多代变异
    num_generations = r
    for gen in range(1, num_generations + 1):
        # 对每个模型配置执行变异

        for model_key, cfg in current_configs.items():
            if isinstance(cfg, int) or isinstance(cfg, str) or isinstance(cfg, list):
                continue
            mutated_cfg = mutator.mutate_predict_config(
                base_config=cfg,
                mutation_rate=mutation_rate,
                model_type=model_key,  # 假设 model_key 对应 mutator 支持的类型
                model_num=mutnm,
            )
            current_configs[model_key] = mutated_cfg

        # 将所有变异后的模型配置组合成新的配置
        new_config = {}
        for model_key in base_config.keys():
            new_config[model_key] = current_configs[model_key]

        # 保存结果
        output_path = os.path.join(output_dir, f"mutation_gen{gen}.json")
        with open(output_path, 'w', encoding='utf-8') as out_f:
            json.dump(new_config, out_f, ensure_ascii=False, indent=2)

        print(f"Generation {gen} saved to {output_path}")

        
from mm_mutation_system import MMConfigMutator
if __name__ == "__main__":
    train_valid_test_datasets_provider.is_distributed = True
    # 应放在创建 res_dir 之后、所有 print 之前
    # log_file_path = "/mnt/fangcr/ascendc-api-adv/examples/pipeline/vector_chain/Mindspeed-mm/MindSpeed-MM/model_config_mm/llava.txt"
    # sys.stdout = Logger(log_file_path)
    mutate_json_all(100,1,2,
                    "/shared/mm-new/pretrain_examples/cogvideox/i2v_1.5/model_cogvideox_i2v_1.5.json")
    # exit()
    # pretrain(
    #     train_valid_test_datasets_provider,
    #     model_provider,
    #     ModelType.encoder_or_decoder,
    #     forward_step,
    #     extra_args_provider=mm_extra_args_provider,
    #     args_defaults={"dataloader_type": "external", "vision_pretraining": False},
    # )

