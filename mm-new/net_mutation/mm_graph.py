#!/usr/bin/env python3
"""
多模态图结构模块
参考mutate_and_forward/graph.py，但针对多模态模型进行适配
支持STDiT3多模态模型的图结构构建和前向传播，所有模型类型统一使用STDiT3架构
"""
import copy
import sys
import torch
import json
from ruamel.yaml import YAML
from typing import Dict, Any, List, Optional, Tuple
import torch.nn as nn
import mindspeed.megatron_adaptor
with open("./demo.json", "r") as f:
    CONFIG = json.load(f)
# NPU设备检测和全局设置
try:
    import torch_npu
    if torch.npu.is_available():
        # 设置默认NPU设备
        torch.npu.set_device(0)
        GLOBAL_DEVICE = torch.device('npu:0')
        print(f"✓ MMGraph: 检测到NPU设备: {GLOBAL_DEVICE}")
    else:
        GLOBAL_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"✓ MMGraph: 使用设备: {GLOBAL_DEVICE}")
except ImportError:
    GLOBAL_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"✓ MMGraph: torch_npu未安装，使用设备: {GLOBAL_DEVICE}")

# 导入多模态模型相关模块
from mm_mutation_system import MMConfigMutator, MutatedMMNode


class MMNode:
    """多模态节点类"""
    
    def __init__(self, config: Dict[str, Any], index: int = -1, model_type: str = 'stdit3'):
        super().__init__()
        self.from_nodes = []
        self.to_nodes = []
        self.layer_limits = []
        self.op = None
        self.id = index
        self.origin_id = -1
        self.state = 'none'
        self.is_des = False
        self.is_src = False
        self.visit_count = 0
        self.succ_count = 0
        self.str_op = 'empty'
        self.params = {}
        self.input_shape = []
        self.output_shape = []
        self.in_degree = len(self.from_nodes)
        self.out_degree = len(self.to_nodes)
        self.config = config
        self.model_type = model_type
        self.mm_model = None  # 多模态模型实例
        self.mutated_node = None  # 变异节点实例

    def add_from(self, node):
        """添加前驱节点"""
        if isinstance(node, MMNode):
            self.from_nodes.append(node.id)
        else:
            self.from_nodes.append(node)

    def get_from(self, index=-1):
        """获取前驱节点"""
        if index == -1:
            return self.from_nodes
        return self.from_nodes[index]

    def del_from(self, index):
        """删除前驱节点"""
        if index in self.from_nodes:
            self.from_nodes.remove(index)

    def add_to(self, node):
        """添加后继节点"""
        if isinstance(node, MMNode):
            self.to_nodes.append(node.id)
        else:
            self.to_nodes.append(node)

    def get_to(self, index=-1):
        """获取后继节点"""
        if index == -1:
            return self.to_nodes
        return self.to_nodes[index]

    def del_to(self, index):
        """删除后继节点"""
        if index in self.to_nodes:
            self.to_nodes.remove(index)

    def set_mm_model(self, model, state='none', str_op='mm_model'):
        """设置多模态模型"""
        self.mm_model = model
        self.str_op = str_op
        self.state = state
        if self.mm_model is not None:
            model_name = str(type(self.mm_model))
            model_name = model_name[:-2]
            dot_ip = model_name.rindex('.')
            model_name = model_name[dot_ip + 1:]
            self.str_op = model_name.lower()

    def set_mutated_node(self, mutated_node: MutatedMMNode):
        """设置变异节点"""
        self.mutated_node = mutated_node
        self.mm_model = mutated_node.model
        self.str_op = f"mutated_{mutated_node.model_type}"

    def run_mm(self, video: torch.Tensor, timestep: torch.Tensor, 
              prompt: torch.Tensor,model_type, **kwargs) -> torch.Tensor:
        """运行多模态模型"""
        if self.mutated_node is not None:
            return self.mutated_node.forward(video, timestep, prompt,self.model_type, **kwargs)
        elif self.mm_model is not None:
            print(self.model_type)
            if self.model_type == "stdit3":
                # 为STDiT3模型准备必要的参数
                model_type_mapping = {
                    'sora': 'stdit3',
                    'cogvideox': 'stdit3'
                }
                internal_model_type = model_type_mapping.get(self.model_type, self.model_type)
                
                # 确保输入数据类型正确
                if hasattr(self.mm_model, 'x_embedder') and hasattr(self.mm_model.x_embedder, 'proj'):
                    # 从模型的第一个层获取期望的数据类型
                    expected_dtype = self.mm_model.x_embedder.proj.weight.dtype
                    if video.dtype != expected_dtype:
                        old_dtype = video.dtype
                        video = video.to(expected_dtype)
                        print(f"    在run_mm中转换video数据类型: {old_dtype} -> {expected_dtype}")
                    if timestep.dtype != expected_dtype:
                        timestep = timestep.to(expected_dtype)
                    if prompt.dtype != expected_dtype:
                        prompt = prompt.to(expected_dtype)
                
                if internal_model_type == 'stdit3':
                    # 从视频张量中获取空间维度
                    _, _, _, H, W = video.shape
                    if 'height' not in kwargs:
                        kwargs['height'] = torch.tensor([H], dtype=torch.long, device=video.device)
                    if 'width' not in kwargs:
                        kwargs['width'] = torch.tensor([W], dtype=torch.long, device=video.device)
                    if 'fps' not in kwargs:
                        # 使用默认fps
                        fps_dtype = torch.float16 if video.dtype == torch.float16 else torch.float32
                        kwargs['fps'] = torch.tensor([24.0], dtype=fps_dtype, device=video.device)
                    if 'prompt_mask' not in kwargs:
                        # 创建prompt掩码
                        B, _, N_token, _ = prompt.shape
                        kwargs['prompt_mask'] = torch.ones(B, N_token, dtype=torch.bool, device=video.device)
                        print(f"    自动创建prompt_mask: {kwargs['prompt_mask'].shape}")
                
                return self.mm_model(video=video, timestep=timestep, prompt=prompt, **kwargs)
            elif self.model_type == "text_encoder":
                device = torch.device('npu:0')
                text_encoder_config = {
                    "hub_backend": "hf",
                    "model_id": "T5",
                    "from_pretrained": "/mnt/fangcr/ascendc-api-adv/examples/pipeline/vector_chain/Mindspeed-mm/MindSpeed-MM/examples/opensora1.2/t5-v1_1-xxl",
                    "dtype": "fp32",
                    "low_cpu_mem_usage": True
                }

                input_ids = torch.randint(0, 1000, (1, 5)).to(device)   # 更小的词汇表和序列长度
                attention_mask = torch.ones_like(input_ids)  # 创建全 1 的注意力掩码
                outputs, masks = self.mm_model.encode(input_ids, attention_mask)
                return outputs
            elif self.model_type == "projector":
                device = torch.device('npu:0')
                predict_config = {
                    "dtype": "fp16",  # LLaVA使用fp16
                    "num_layers": 2,  # 减少层数以避免内存问题（原配置6层）
                    "num_heads": 8,   # 保持LLaVA的注意力头数
                    "hidden_size": 512,  # LLaVA的隐藏层大小
                    "input_size": [1, 32, 32],  # 简化的输入尺寸（原配置[1, 224, 224]）
                    "patch_size": [1, 16, 16],  # 保持LLaVA的patch大小
                    "in_channels": 4,  # 默认潜在表示通道数
                    "caption_channels": 512,  # LLaVA的文本特征通道数
                    "model_max_length": 64,  # 简化的文本长度（原配置512）
                    "mlp_ratio": 4.0,
                    "class_dropout_prob": 0.1,
                    "space_scale": 1.0,
                    "time_scale": 1.0,
                    "enable_flashattn": False,  # 在NPU上可能有兼容性问题
                    "enable_sequence_parallelism": False,
                    "pred_sigma": True,
                    "drop_path": 0.0,
                    "qk_norm": False,  # 避免NPU兼容性问题
                }
                batch_size = 1
                # 基于 predict_config 的配置构造输入，使用fp16
                frames, height, width = predict_config["input_size"]  # [1, 32, 32]
                # 创建5D视频张量 [B, C, T, H, W] - 对于图像，T=1
                in_channels = predict_config["in_channels"]  # 4
                video = torch.randn(batch_size, in_channels, frames, height, width, dtype=torch.float16).to(device)
                
                # 创建时间步数据
                timestep = torch.tensor([500], dtype=torch.long).to(device)
                
                # 创建文本条件数据 (LLaVA的CLIP文本特征)
                text_seq_len = predict_config["model_max_length"]  # 64
                text_hidden_size = predict_config["caption_channels"]  # 512
                text_condition = torch.randn(batch_size, 1, text_seq_len, text_hidden_size, dtype=torch.float16).to(device)
                
                # 创建掩码
                text_mask = torch.ones(batch_size, text_seq_len, dtype=torch.bool).to(device)
                
                # 创建视频掩码
                total_spatial_temporal = frames * height * width
                video_mask = torch.ones(batch_size, total_spatial_temporal, dtype=torch.bool).to(device)
                
                # 创建STDiT3需要的额外参数
                fps = torch.tensor([1.0], dtype=torch.float16).to(device)  # 图像帧率为1
                height_tensor = torch.tensor([height], dtype=torch.long).to(device)
                width_tensor = torch.tensor([width], dtype=torch.long).to(device)
                
                print(f"输入形状: video={video.shape}, timestep={timestep.shape}")
                print(f"文本条件形状: {text_condition.shape}")
                print(f"配置检查: model_max_length={predict_config['model_max_length']}, caption_channels={predict_config['caption_channels']}")
                print(f"输入数据类型检查:")
                print(f"  video dtype: {video.dtype}")
                print(f"  timestep dtype: {timestep.dtype}")
                print(f"  text_condition dtype: {text_condition.dtype}")
                print(f"  fps dtype: {fps.dtype}")
                print("正在进行前向传播...")
                
                # 调用 forward 方法
                output = self.mm_model(
                    video=video,
                    timestep=timestep,
                    prompt=text_condition,
                    prompt_mask=text_mask,
                    fps=fps,
                    height=height_tensor,
                    width=width_tensor
                )
                return output
            elif self.model_type == "text_decoder":
                text_decoder_config = {
                    "num_layers": 4,
                    "hidden_size": 512,
                    "num_attention_heads": 32,
                    "num_query_groups": 32,
                    "ffn_hidden_size": 1376,
                    "add_bias_linear": False,
                    "bias_activation_fusion": False,
                    "gated_linear_unit": True,
                    "apply_query_key_layer_scaling":False,
                    "layernorm_zero_centered_gamma": False,
                    "max_position_embeddings": 2048,
                    "bias_dropout_fusion":False,
                    "apply_rope_fusion": False,
                    "attention_softmax_in_fp32": True,
                    "attention_dropout": 0.0,
                    "hidden_dropout": 0.0,
                    "bf16": True,
                    "params_dtype": "bf16",
                    "deallocate_pipeline_outputs": True,
                    "persist_layer_norm": True,
                    "activation_func": "silu",
                    "normalization": "RMSNorm",
                    "language_vocab_size": 32000,
                    "language_max_sequence_length": 2048,
                    "lm_position_embedding_type": "rope",
                    "recompute_granularity": "full",
                    "recompute_method": "block",
                    "recompute_num_layers": 4,
                    "freeze": True,
                    "ckpt_path": None
                }
                print("✅ TextDecoder instantiated.")

                # 测一次前向
                print("\n=== Running a forward pass ===")

                device = torch.device('npu:0')

                # --- 3. 生成随机输入数据 ---
                # 设置 batch_size 和 seq_len
                batch_size = 2
                seq_len = 16
                vocab_size = 32000

                # 随机生成 input_ids: [seq_len, batch_size]
                input_ids = torch.randint(
                    low=0,
                    high=vocab_size,
                    size=(seq_len, batch_size),
                    dtype=torch.long,
                ).to(device)
                # 生成 position_ids: [seq_len, batch_size]
                position_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(1).expand(-1, batch_size).to(device)
                # attention_mask: 全部可关注
                attention_mask = torch.ones((seq_len, batch_size), dtype=torch.long).to(device)


                return self.mm_model(input_ids=input_ids,
                        position_ids=position_ids,
                        attention_mask=attention_mask)

            elif self.model_type == "word_embeddings":
                device = torch.device('npu:0')
                cfg = CONFIG["word_embeddings"]
                vocab_size = cfg["vocab_size"]
                hidden_size = cfg["hidden_size"]
                batch_size = cfg["batch_size"]
                seq_len = cfg["seq_len"]
                torch.manual_seed(42)

                # word_embeddings = nn.Embedding(
                #     num_embeddings=vocab_size,
                #     embedding_dim=hidden_size
                # )
                # nn.init.constant_(word_embeddings.weight, 0.5)
                
                tokens = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)  # [B, S]
                tokens = tokens % vocab_size
                tokens = tokens.to(device)
                # we = word_embeddings(tokens)  # [batch_size, seq_len, hidden_size]
                # print("success word_embeddings, output shape:", we.shape)

                # # 3）用普通 Adam 优化器
                # optimizer = optim.Adam(word_embeddings.parameters(), lr=1e-4, weight_decay=1e-2)

                # # 4）定义一个简单的“loss”：这里用 Frobenius 范数
                # loss = we.norm()
                # print("word_embeddings Initial loss:", loss.item())

                # # 5）反向 + 更新
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()

                # # 6）再算一次 loss 看下变化（选做）
                # with torch.no_grad():
                #     we2 = word_embeddings(tokens)
                #     loss2 = we2.norm()
                # print("word_embeddings Post-update loss:", loss2.item())

                return self.mm_model(tokens)

            elif self.model_type == "rotary_pos_emb":
                return self.mm_model(2)
            elif self.model_type == "position_embeddings":
                device = torch.device('npu:0')
                cfg = CONFIG["position_embeddings"]
                hidden_size = cfg["hidden_size"]
                max_sequence_length = cfg["max_sequence_length"]
                batch_size = cfg["batch_size"]
                seq_len = cfg["seq_len"]

                # position_embeddings = nn.Embedding(
                #     num_embeddings=max_sequence_length,
                #     embedding_dim=hidden_size
                # )
                positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)
                # pe = position_embeddings(positions)  # [batch_size, seq_len, hidden_size]
                # print("success position_embeddings, output shape:", pe.shape)

                return self.mm_model(positions)

            elif self.model_type == "tokentype_embeddings":
                device = torch.device('npu:0')
                cfg = CONFIG["tokentype_embeddings"]
                hidden_size = cfg["hidden_size"]
                num_tokentypes = cfg["num_tokentypes"]
                batch_size = cfg["batch_size"]
                seq_len = cfg["seq_len"]

                # tokentype_embeddings = nn.Embedding(
                #     num_embeddings=num_tokentypes,
                #     embedding_dim=hidden_size
                # )
                tokentypes = torch.randint(0, num_tokentypes, (batch_size, seq_len)).to(device)
                # te = tokentype_embeddings(tokentypes)  # [batch_size, seq_len, hidden_size]
                # print("success tokentype_embeddings, output shape:", te.shape)

                return self.mm_model(tokentypes)
            elif self.model_type == "embeddings":
                device = torch.device('npu:0')
                print("\n=== Running a forward pass ===")
                batch_size = 2
                seq_len = 16
                input_ids = torch.full((batch_size, seq_len),
                                    fill_value=42,            # 固定 token ID
                                    dtype=torch.long,
                                    device=device)
                position_ids = torch.arange(seq_len,
                                            dtype=torch.long,
                                            device=device
                                            ).unsqueeze(0).expand(batch_size, -1)
                return self.mm_model(input_ids,position_ids)
            elif self.model_type == "embedding_dropout":
                device = torch.device('npu:0')
                cfg = CONFIG["embedding_dropout"]
                hidden_size = cfg["hidden_size"]
                dropout_prob = cfg["dropout_prob"]
                batch_size = cfg["batch_size"]
                seq_len = cfg["seq_len"]

                # embedding_dropout = nn.Dropout(p=dropout_prob)
                dummy = torch.randn(batch_size, seq_len, hidden_size).to(device)

                # out = embedding_dropout(dummy)

                return self.mm_model(dummy)

            elif self.model_type == "conv1":
                device = torch.device('npu:0')
                cfg = CONFIG["conv1"]
                batch_size = cfg["batch_size"]
                in_channels = cfg["in_channels"]
                image_size = cfg["image_size"]
                patch_size = cfg["patch_size"]
                hidden_size = cfg["hidden_size"]

                # conv1 = nn.Conv2d(
                #     in_channels=in_channels,
                #     out_channels=hidden_size,
                #     kernel_size=patch_size,
                #     stride=patch_size,
                #     bias=False
                # )

                x = torch.randn(batch_size, in_channels, image_size, image_size).to(device)
                # out = conv1(x)  # [B, hidden_size, H//patch, W//patch]

                return self.mm_model(x)

            elif self.model_type == "class_token":
                cfg = CONFIG["class_token"]
                batch_size = cfg["batch_size"]
                class_token_len = cfg["class_token_len"]
                hidden_size = cfg["hidden_size"]

                class_token = nn.Parameter(
                    torch.randn(1, class_token_len, hidden_size)
                )
                out = class_token.expand(batch_size, -1, -1)

                return self.mm_model.expand(batch_size, -1, -1)


        else:
            raise RuntimeError(f"节点 {self.id} 没有可用的多模态模型")

    def get_id(self):
        """获取节点ID"""
        return self.id

    def set_id(self, index):
        """设置节点ID"""
        self.id = index

    def set_state(self, state):
        """设置节点状态"""
        self.state = state
        if state == 'des':
            self.is_des = True
        if state == 'src':
            self.is_src = True

    def set_input_shape(self, input_shape=None):
        """设置输入形状"""
        if input_shape is None:
            input_shape = [None, None]
        self.input_shape = input_shape

    def get_input_shape(self):
        """获取输入形状"""
        return self.input_shape

    def set_output_shape(self, output_shape):
        """设置输出形状"""
        self.output_shape = output_shape

    def get_output_shape(self):
        """获取输出形状"""
        return self.output_shape

    def __str__(self) -> str:
        return f'id:{self.id},model_type:{self.model_type},op:{self.str_op},to:{self.to_nodes},' \
               f'from:{self.from_nodes},state:{self.state},in_degree:{self.in_degree},' \
               f'out_degree:{self.out_degree},input_shape:{self.input_shape},' \
               f'output_shape:{self.output_shape},params:{self.params},({self.origin_id})'

    def __hash__(self) -> int:
        return hash(self.str_op)

    def __eq__(self, o) -> bool:
        return hash(self.str_op) == hash(o.str_op)


class MMGraph:
    """多模态图类"""

    def __init__(self, config_path: str = None, config_dict: dict = None, 
                 nums: list = None, mutated_nodes: dict = None, 
                 node_model_mapping: dict = None,model_type:str = None):
        """
        初始化多模态图
        
        Args:
            config_path: 配置文件路径
            config_dict: 配置字典
            nums: 节点编号列表
            mutated_nodes: 变异节点字典
            node_model_mapping: 节点模型类型映射 {node_id: model_type}
        """
        # 支持两种初始化方式：配置文件路径或配置字典
        if config_dict is not None:
            # 使用配置字典初始化
            model_config = config_dict
        elif config_path is not None:
            # 使用配置文件路径初始化
            yaml = YAML()
            with open(config_path, 'r', encoding='utf-8') as file:
                model_config = yaml.load(file)
        else:
            # 使用默认配置
            model_config = {
                'predict_config': {
                    'model_id': 'stdit3',
                    'dtype': 'fp16',
                    'num_layers': 4,
                    'hidden_size': 512,
                    'num_heads': 8,
                    'input_size': [4, 16, 16],
                    'patch_size': [1, 2, 2],
                    'in_channels': 4,
                    'caption_channels': 512,
                    'model_max_length': 64,
                    'mlp_ratio': 4.0,
                    'class_dropout_prob': 0.1,
                    'enable_flashattn': False,
                    'enable_sequence_parallelism': False,
                }
            }
        
        self.total_config = model_config
        self.predict_config = model_config.get('predict_config', model_config)
        
        # 默认节点编号
        if nums is None:
            nums = [1, 2]
        
        # 创建节点
        self.nodes = {}
        for node_id in nums:
            if model_type == "stdit3":
                node = MMNode(config=self.predict_config, index=node_id, 
                            model_type=self.predict_config.get('model_id', 'stdit3'))
            else:
                node = MMNode(config=self.predict_config, index=node_id, 
                            model_type=model_type)

            self.nodes[node_id] = node
        
        # 保存变异节点信息
        self.mutated_nodes = mutated_nodes if mutated_nodes is not None else {}
        
        # 保存节点模型类型映射
        self.node_model_mapping = node_model_mapping if node_model_mapping is not None else {}
        
        # 初始化组件
        self._initialize_components()
        
        print(f"✓ 多模态图初始化完成，包含 {len(self.nodes)} 个节点")

    def _initialize_components(self):
        """初始化图组件"""
        try:
            # 创建默认的多模态模型配置器
            self.mm_mutator = MMConfigMutator()
            
            # 为每个节点设置模型
            for node_id, node in self.nodes.items():
                if node_id in self.mutated_nodes:
                    # 使用变异节点
                    mutated_node = self.mutated_nodes[node_id]
                    node.set_mutated_node(mutated_node)
                    print(f"  节点 {node_id} 设置为变异节点 ({mutated_node.model_type})")
                else:
                    # 创建默认模型，使用节点预分配的模型类型
                    try:

                        node_model_type = self.node_model_mapping.get(node_id, 'stdit3')
                        print(f"  为节点 {node_id} 创建默认模型 (架构: {node_model_type})...")
                        
                        # 根据节点的模型类型调整配置
                        node_config = self.predict_config.copy()
                        node_config['model_id'] = node_model_type
                        
                        model = self.mm_mutator.create_mutated_model(node_config)
                        
                        # 额外的设备同步检查（特别针对NPU）
                        from mindspeed_mm.models.predictor.dits.stdit3 import STDiT3
                        
                        # 获取全局设备
                        try:
                            import torch_npu
                            if torch.npu.is_available():
                                target_device = torch.device('npu:0')
                            else:
                                target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                        except ImportError:
                            target_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                        
                        # 确保模型在正确的设备上
                        if isinstance(model, STDiT3):
                            model = model.to(target_device)
                            
                            # 对NPU设备进行额外的同步检查
                            if 'npu' in str(target_device):
                                print(f"    NPU设备检查: 确保所有参数都在 {target_device}")
                                for name, param in model.named_parameters():
                                    if param.device != target_device:
                                        param.data = param.data.to(target_device)
                                        print(f"      参数 {name} 移动到 {target_device}")
                                
                                for name, buffer in model.named_buffers():
                                    if buffer is not None and buffer.device != target_device:
                                        buffer.data = buffer.data.to(target_device)
                                        print(f"      缓冲区 {name} 移动到 {target_device}")
                        
                        node.set_mm_model(model, state='normal')
                        print(f"  节点 {node_id} 设置为默认模型 ({node_model_type})")
                    except Exception as e:
                        print(f"  ✗ 节点 {node_id} 模型创建失败: {e}")
                        import traceback
                        traceback.print_exc()
                        node.mm_model = None
            
        except Exception as e:
            print(f"✗ 初始化组件失败: {e}")
            import traceback
            traceback.print_exc()

    def forward(self, video: torch.Tensor, timestep: torch.Tensor, 
                prompt: torch.Tensor, debug: bool = True, **kwargs) -> torch.Tensor:
        """
        多模态图前向传播
        
        Args:
            video: 视频张量 [B, C, T, H, W]
            timestep: 时间步张量 [B]
            prompt: 提示张量 [B, 1, N_token, C]
            debug: 是否打印调试信息
            **kwargs: 其他参数
            
        Returns:
            torch.Tensor: 输出张量
        """
        if debug:
            print(f"=== 多模态图前向传播开始 ===")
            print(f"输入形状: video={video.shape}, timestep={timestep.shape}, prompt={prompt.shape}")
        
        # 获取源节点和目标节点
        src_nodes = self.get_src()
        des_nodes = self.get_des()
        
        if not src_nodes:
            # 如果没有明确的源节点，使用第一个节点
            src_nodes = [min(self.nodes.keys())]
            
        if not des_nodes:
            # 如果没有明确的目标节点，使用最后一个节点
            des_nodes = [max(self.nodes.keys())]
        
        if debug:
            print(f"源节点: {src_nodes}, 目标节点: {des_nodes}")
        
        # 简单的串行前向传播（后续可以改为更复杂的图遍历）
        current_output = video
        
        # 按节点ID顺序执行
        sorted_node_ids = sorted(self.nodes.keys())
        
        for node_id in sorted_node_ids:
            node = self.nodes[node_id]
            
            if debug:
                print(f"  执行节点 {node_id} ({node.str_op})")
            
            try:
                # 检查节点是否有可用的模型
                has_model = False
                if node.mutated_node is not None and node.mutated_node.model is not None:
                    has_model = True
                elif node.mm_model is not None:
                    has_model = True
                
                if has_model:
                    # 为每个节点准备合适的输入
                    if node_id == sorted_node_ids[0]:
                        # 第一个节点使用原始输入
                        node_output = node.run_mm(video, timestep, prompt,node.model_type, **kwargs)
                    else:
                        # 后续节点使用前一个节点的输出作为video输入
                        # 但需要确保数据类型正确
                        input_video = current_output
                        
                        # 检查并修正数据类型 - 从模型权重推断期望的数据类型
                        target_dtype = None
                        if hasattr(node, 'mutated_node') and node.mutated_node is not None and node.mutated_node.model is not None:
                            # 从变异节点的模型权重推断数据类型
                            if hasattr(node.mutated_node.model, 'x_embedder') and hasattr(node.mutated_node.model.x_embedder, 'proj'):
                                target_dtype = node.mutated_node.model.x_embedder.proj.weight.dtype
                        elif hasattr(node, 'mm_model') and node.mm_model is not None:
                            # 从节点模型权重推断数据类型
                            if hasattr(node.mm_model, 'x_embedder') and hasattr(node.mm_model.x_embedder, 'proj'):
                                target_dtype = node.mm_model.x_embedder.proj.weight.dtype
                            elif hasattr(node.mm_model, 'module') and hasattr(node.mm_model.module, 'x_embedder'):
                                target_dtype = node.mm_model.module.x_embedder.proj.weight.dtype
                        
                        # 如果无法从模型推断，使用配置或默认值
                        if target_dtype is None:
                            if hasattr(node, 'mutated_node') and node.mutated_node is not None:
                                expected_dtype = node.mutated_node.mutated_config.get('dtype', 'fp16')
                            elif hasattr(self, 'predict_config'):
                                expected_dtype = self.predict_config.get('dtype', 'fp16')
                            else:
                                expected_dtype = 'fp16'
                            
                            if expected_dtype == 'fp16':
                                target_dtype = torch.float16
                            elif expected_dtype == 'bf16':
                                target_dtype = torch.bfloat16
                            else:
                                target_dtype = torch.float16
                        
                        if input_video.dtype != target_dtype:
                            input_video = input_video.to(target_dtype)
                            if debug:
                                print(f"    转换video数据类型: {current_output.dtype} -> {target_dtype}")
                        
                        # 同样也要确保timestep和prompt的数据类型正确
                        if timestep.dtype != target_dtype:
                            timestep = timestep.to(target_dtype)
                        if prompt.dtype != target_dtype:
                            prompt = prompt.to(target_dtype)
                        
                        # 转换kwargs中的张量参数
                        converted_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor) and v.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                                converted_kwargs[k] = v.to(target_dtype)
                            else:
                                converted_kwargs[k] = v
                        
                        node_output = node.run_mm(input_video, timestep, prompt,node.model_type, **converted_kwargs)
                    
                    current_output = node_output
                    
                    if debug:
                        if hasattr(node_output, 'shape'):
                            print(f"    输出形状: {node_output.shape}, 数据类型: {node_output.dtype}")
                        else:
                            print(f"    输出类型: {type(node_output)}")
                
                else:
                    if debug:
                        print(f"    跳过节点 {node_id} (无可用模型)")
                    # 如果是第一个节点且无模型，使用原始输入作为输出
                    if node_id == sorted_node_ids[0]:
                        current_output = video
                    # 否则保持current_output不变，相当于直通
                    
            except Exception as e:
                print(f"  ✗ 节点 {node_id} 执行失败: {e}")
                if debug:
                    import traceback
                    traceback.print_exc()
                
                # 失败时的回退策略：如果是第一个节点失败，使用原始输入
                if node_id == sorted_node_ids[0]:
                    current_output = video
                    print(f"    回退: 使用原始输入作为节点 {node_id} 的输出")
                else:
                    print(f"    回退: 保持前一个节点的输出，跳过节点 {node_id}")
                # 继续执行其他节点
                return False,1
                continue
        
        if debug:
            print(f"=== 多模态图前向传播完成 ===")
            if hasattr(current_output, 'shape'):
                print(f"最终输出形状: {current_output.shape}")
        
        return True, current_output

    def set_mutated_nodes(self, mutated_nodes: dict):
        """设置变异节点"""
        self.mutated_nodes = mutated_nodes
        
        # 更新现有节点
        for node_id, mutated_node in mutated_nodes.items():
            if node_id in self.nodes:
                self.nodes[node_id].set_mutated_node(mutated_node)
                print(f"✓ 节点 {node_id} 更新为变异节点")

    def get_mutated_nodes(self):
        """获取变异节点"""
        return self.mutated_nodes

    def get_node(self, index):
        """获取节点"""
        return self.nodes.get(index)

    def add_edge(self, src: int, des: int):
        """添加边"""
        if src in self.nodes and des in self.nodes:
            self.nodes[src].add_to(des)
            self.nodes[des].add_from(src)
            print(f"✓ 添加边: {src} -> {des}")

    def del_edge(self, src: int, des: int):
        """删除边"""
        if src in self.nodes and des in self.nodes:
            self.nodes[src].del_to(des)
            self.nodes[des].del_from(src)
            print(f"✓ 删除边: {src} -> {des}")

    def add_node(self, node_id: int, config: Dict[str, Any] = None, 
                 model_type: str = 'stdit3'):
        """添加节点"""
        if config is None:
            config = self.predict_config
            
        new_node = MMNode(config=config, index=node_id, model_type=model_type)
        self.nodes[node_id] = new_node
        
        # 为新节点创建模型
        try:
            model = self.mm_mutator.create_mutated_model(config)
            new_node.set_mm_model(model, state='normal')
            print(f"✓ 添加节点 {node_id} ({model_type})")
        except Exception as e:
            print(f"✗ 添加节点 {node_id} 失败: {e}")

    def get_src(self):
        """获取源节点"""
        src_nodes = []
        for node_id, node in self.nodes.items():
            if node.is_src or len(node.from_nodes) == 0:
                src_nodes.append(node_id)
        return src_nodes

    def get_des(self):
        """获取目标节点"""
        des_nodes = []
        for node_id, node in self.nodes.items():
            if node.is_des or len(node.to_nodes) == 0:
                des_nodes.append(node_id)
        return des_nodes

    def display(self):
        """显示图结构"""
        print("=== 多模态图结构 ===")
        for node_id, node in self.nodes.items():
            print(f"节点 {node_id}: {node}")
        print("==================")

    def get_graph_info(self) -> Dict[str, Any]:
        """获取图信息"""
        total_params = 0
        node_info = {}
        
        for node_id, node in self.nodes.items():
            if node.mutated_node is not None:
                info = node.mutated_node.get_model_info()
                total_params += info.get('total_parameters', 0)
                node_info[node_id] = info
            elif node.mm_model is not None:
                params = sum(p.numel() for p in node.mm_model.parameters())
                total_params += params
                node_info[node_id] = {
                    'node_id': node_id,
                    'model_type': node.model_type,
                    'total_parameters': params,
                    'model_initialized': True
                }
            else:
                node_info[node_id] = {
                    'node_id': node_id,
                    'model_type': node.model_type,
                    'total_parameters': 0,
                    'model_initialized': False
                }
        
        return {
            'total_nodes': len(self.nodes),
            'total_parameters': total_params,
            'total_size_mb': total_params * 4 / (1024 * 1024),
            'predict_config': self.predict_config,
            'node_info': node_info
        }

    def __len__(self):
        """获取节点数量"""
        return len(self.nodes)


def demo_mm_graph():
    """演示多模态图"""
    print("=== 多模态图演示 ===")
    
    # 创建图
    graph = MMGraph(nums=[1, 2, 3])
    
    # 添加边
    graph.add_edge(1, 2)
    graph.add_edge(2, 3)
    
    # 设置节点状态
    graph.nodes[1].set_state('src')
    graph.nodes[3].set_state('des')
    
    # 显示图结构
    graph.display()
    
    # 获取图信息
    info = graph.get_graph_info()
    print("图信息:", info)
    
    # 测试前向传播（使用虚拟数据）
    try:
        print("\n--- 测试前向传播 ---")
        batch_size = 1
        video = torch.randn(batch_size, 4, 4, 16, 16, dtype=torch.float16)
        timestep = torch.tensor([500], dtype=torch.long)
        prompt = torch.randn(batch_size, 1, 64, 512, dtype=torch.float16)
        
        output = graph.forward(video, timestep, prompt, debug=True)
        print(f"✓ 前向传播成功，输出形状: {output.shape}")
        
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_mm_graph() 