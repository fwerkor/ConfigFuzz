#!/usr/bin/env python3
"""
多模态变异系统模块
支持STDiT3多模态模型的predict_config变异，所有模型类型统一使用STDiT3架构
参考mutate_and_forward/mutation_system.py，但针对多模态模型进行适配
"""

import os
import yaml
import random
import torch
import numpy as np
import math
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import mindspeed.megatron_adaptor

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# 设置全局设备
try:
    import torch_npu

    if torch.npu.is_available():
        GLOBAL_DEVICE = torch.device('npu:0')
        print(f"使用NPU设备: {GLOBAL_DEVICE}")
    else:
        GLOBAL_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {GLOBAL_DEVICE}")
except ImportError:
    GLOBAL_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {GLOBAL_DEVICE}")

# 导入多模态模型相关模块

class MMConfigMutator:
    """多模态配置变异器"""

    def __init__(self,
                 structure_config_path: str = "/workspace/mm/MindSpeed-Core-MS/MindSpeed-MM/Megatron_model_config/model_scripts_mm/configs/mm_structure_config.yaml",
                 template_config_path: str = "/workspace/mm/MindSpeed-Core-MS/MindSpeed-MM/Megatron_model_config/model_scripts_mm/configs/mm_template_config.yaml",
                 output_dir: str = "./mutated_mm_configs",
                 config_dir: str = "./model_config_mm"):
        """
        初始化多模态配置变异器

        Args:
            structure_config_path: 结构配置文件路径
            template_config_path: 模板配置文件路径 
            output_dir: 输出目录
            config_dir: 配置文件目录
        """
        self.structure_config_path = structure_config_path
        self.template_config_path = template_config_path
        self.output_dir = output_dir
        self.config_dir = config_dir

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 加载配置
        self.structure_config = self._load_yaml(structure_config_path)
        self.template_config = self._load_yaml(template_config_path)

        # 合并配置作为基础配置
        self.merged_config = self._merge_configs()

        # 加载变异参数池配置
        self.mutable_params_pool = self._load_mutable_params_pool()

        # 增量变异状态管理
        self.incremental_configs = {}  # 存储每个节点的增量变异配置 {node_id: config}
        self.mutation_history = {}  # 存储每个节点的变异历史 {node_id: [mutation_records]}
        self.current_round = 0  # 当前变异轮次

        # 基础STDiT3配置（所有模型类型统一使用）
        self.base_stdit3_config = {
            'model_id': 'stdit3',
            'dtype': 'bf16',
            'input_sq_size': 512,
            'in_channels': 4,
            'patch_size': [1, 2, 2],
            'hidden_size': 1152,
            'num_layers': 28,
            'num_heads': 16,
            'mlp_ratio': 4.0,
            'class_dropout_prob': 0.1,
            'pred_sigma': True,
            'drop_path': 0.0,
            'caption_channels': 4096,
            'model_max_length': 300,
            'qk_norm': True,
            'enable_flashattn': True,
            'enable_sequence_parallelism': False,
            'only_train_temporal': False,
            'freeze_y_embedder': True,
            'skip_y_embedder': False,
            'input_size': [16, 32, 32],
        }

        print(f"✓ MM配置变异器初始化完成")
        print(f"  结构配置: {structure_config_path}")
        print(f"  模板配置: {template_config_path}")
        print(f"  输出目录: {output_dir}")
        print(f"  配置目录: {config_dir}")
        print(f"  增量变异支持: 启用")

    def _load_yaml(self, file_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"⚠️  配置文件不存在: {file_path}")
            return {}
        except Exception as e:
            print(f"⚠️  加载配置文件失败: {file_path}, 错误: {e}")
            return {}

    def _load_mutable_params_pool(self) -> Dict[str, Any]:
        """加载变异参数池配置"""
        # 获取当前文件所在目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        pool_config_path = os.path.join(current_dir, 'mutable_params_pool.yaml')

        pool_config = self._load_yaml(pool_config_path)

        if not pool_config:
            print(f"⚠️  变异参数池配置加载失败，使用空配置: {pool_config_path}")
            return {}

        # 将YAML配置转换为内部格式
        mutable_params = {}

        # 处理数值型参数
        numeric_params = pool_config.get('numeric_params', {})
        for param_name, config in numeric_params.items():
            mutable_params[param_name] = {
                'min_val': config.get('min_val'),
                'max_val': config.get('max_val'),
                'min_factor': config.get('min_factor', 0.5),
                'max_factor': config.get('max_factor', 2.0)
            }

        # 处理枚举型参数
        enum_params = pool_config.get('enum_params', {})
        for param_name, values in enum_params.items():
            mutable_params[param_name] = {'enum': values}

        print(f"✓ 变异参数池配置加载成功: {len(mutable_params)} 个参数")
        return mutable_params

    def _merge_configs(self) -> Dict[str, Any]:
        """合并结构配置和模板配置"""
        merged = {}
        merged.update(self.structure_config)
        merged.update(self.template_config)
        return merged

    def reset_incremental_state(self):
        """重置增量变异状态，开始新的变异序列"""
        self.incremental_configs.clear()
        self.mutation_history.clear()
        self.current_round = 0
        print("✓ 增量变异状态已重置")

    def set_base_config_for_node(self, node_id: int, base_config: Dict[str, Any]):
        """
        为指定节点设置基础配置（用于初始轮次）

        Args:
            node_id: 节点ID
            base_config: 基础配置
        """
        self.incremental_configs[node_id] = base_config.copy()
        if node_id not in self.mutation_history:
            self.mutation_history[node_id] = []
        print(f"✓ 为节点 {node_id} 设置基础配置")

    def get_incremental_config(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        获取指定节点的增量变异配置

        Args:
            node_id: 节点ID

        Returns:
            增量的配置，如果不存在则返回None
        """
        return self.incremental_configs.get(node_id)

    def get_mutation_history(self, node_id: int) -> List[Dict[str, Any]]:
        """
        获取指定节点的变异历史

        Args:
            node_id: 节点ID

        Returns:
            变异历史列表
        """
        return self.mutation_history.get(node_id, [])

    def mutate_predict_config(self, base_config: Dict[str, Any] = None,
                              mutation_rate: float = 1,
                              model_type: str = 'stdit3',
                              model_num: int = 2,
                              node_id: int = None,
                              use_accumulated: bool = True) -> Dict[str, Any]:
        """
        变异predict_config配置

        Args:
            base_config: 基础配置，如果为None则使用默认配置
            mutation_rate: 变异率
            model_type: 模型类型，所有类型都使用STDiT3
            model_num: num
            node_id: 节点ID，用于增量变异管理
            use_accumulated: 是否使用增量变异（基于上一轮结果）

        Returns:
            Dict[str, Any]: 变异后的配置
        """
        # 所有模型类型统一使用STDiT3
        internal_model_type = 'stdit3'

        # 确定要使用的基础配置
        if use_accumulated and node_id is not None and node_id in self.incremental_configs:
            # 使用增量的配置（上一轮的变异结果）
            working_config = self.incremental_configs[node_id].copy()
            print(f"💡 节点 {node_id}: 基于第 {self.current_round} 轮增量结果进行变异")
        elif base_config is not None:
            # 使用提供的基础配置
            if isinstance(base_config, str):
                working_config = base_config
            else:
                working_config = base_config.copy()
            if node_id is not None:
                self.incremental_configs[node_id] = working_config.copy()
            print(f"节点 {node_id}: 使用提供的基础配置进行变异")
        else:
            # 使用默认配置
            # working_config = self.base_stdit3_config.copy()
            # if node_id is not None:
            #     self.incremental_configs[node_id] = working_config.copy()
            # print(f"节点 {node_id}: 使用默认配置进行变异")
            return None

        # 记录变异前的状态
        pre_mutation_config = working_config.copy()

        # 使用从YAML加载的变异参数池
        mutable_params_pool = self.mutable_params_pool

        print(f"对{model_type}模型节点 {node_id} 进行第 {self.current_round + 1} 轮变异，变异率: {mutation_rate}")

        # 记录本轮变异的参数
        mutations_applied = {}

        def _is_mutatable_value(v: Any) -> bool:
            # 只跳过不可变/无效值；允许布尔/标量
            return v is not None and not isinstance(v, (list, tuple, np.ndarray))

        def _key_with_prefix(prefix: Optional[str], name: str) -> str:
            return f"{prefix}.{name}" if prefix else name

        def _select_and_mutate(target_dict: Dict[str, Any], prefix: Optional[str] = None):
            """从候选池里反复随机选取，直到实际变异出 mutnm 个可变参数。"""
            if not isinstance(target_dict, dict):
                return

            eligible_keys = [
                k for k in mutable_params_pool.keys()
                if k in target_dict and _is_mutatable_value(target_dict.get(k))
            ]
            target_k = min(model_num, len(eligible_keys))
            remaining = eligible_keys[:]  # copy

            mutated_cnt = 0
            while mutated_cnt < target_k and remaining:
                param_name = random.choice(remaining)
                remaining.remove(param_name)

                if random.random() >= mutation_rate:
                    continue

                constraints = mutable_params_pool[param_name]
                original_value = target_dict[param_name]
                print(f"  变异参数: {_key_with_prefix(prefix, param_name)}")

                # 优先处理枚举类型参数（例如布尔开关）
                if isinstance(constraints, dict) and 'enum' in constraints:
                    enum_values = constraints['enum']
                    if isinstance(original_value, bool) and all(isinstance(v, bool) for v in enum_values):
                        new_value = not original_value
                    else:
                        candidates = [v for v in enum_values if v != original_value]
                        new_value = random.choice(candidates) if candidates else original_value

                    print(f"    {param_name}: {original_value} -> {new_value}")
                    target_dict[param_name] = new_value
                    mutations_applied[_key_with_prefix(prefix, param_name)] = {
                        'from': original_value,
                        'to': new_value,
                        'constraints': constraints
                    }
                    mutated_cnt += 1
                    continue

                # 处理数值型参数：使用 original_value 类型区分 float/int，避免把 float 约束误走 random.randint
                is_float_value = isinstance(original_value, (float, np.floating))
                float_param_names = ['mlp_ratio', 'class_dropout_prob', 'drop_path', "space_scale", "time_scale",
                                     "num_embeds_ada_norm", "attention_dropout", "hidden_dropout", "dropout_prob"]
                if (param_name in float_param_names) or (param_name == "norm_eps") or is_float_value:
                    rounding = 8 if param_name == "norm_eps" else 3
                    min_val = max(constraints['min_val'], float(original_value) * constraints['min_factor'])
                    max_val = min(constraints['max_val'], float(original_value) * constraints['max_factor'])
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    new_value = random.uniform(min_val, max_val)
                    new_value = round(new_value, rounding)
                else:  # 整数参数
                    min_val = max(constraints['min_val'], int(original_value * constraints['min_factor']))
                    max_val = min(constraints['max_val'], int(original_value * constraints['max_factor']))
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val

                    # 防御性转换：确保 randint 的入参是整数
                    min_val = int(min_val)
                    max_val = int(max_val)

                    # 特殊处理：2 的幂
                    if param_name in ['num_heads', 'num_attention_heads']:
                        possible_values = [2 ** i for i in range(3, 7) if 2 ** i >= min_val and 2 ** i <= max_val]
                        if possible_values:
                            new_value = random.choice(possible_values)
                        else:
                            new_value = max(8, min(32, original_value))
                    else:
                        if param_name in ['num_layers', 'num_heads', 'num_attention_heads']:
                            min_val = max(min_val, 1)
                        new_value = random.randint(min_val, max_val)

                    # 若变异的是 hidden_size：保证能被 heads 的 LCM 整除
                    if param_name == 'hidden_size':
                        nh = target_dict.get('num_heads', None)
                        nah = target_dict.get('num_attention_heads', None)
                        divisors = [h for h in [nh, nah] if isinstance(h, int) and h > 0]
                        if divisors:
                            div = divisors[0]
                            for h in divisors[1:]:
                                div = div * h // math.gcd(div, h)
                            if new_value % div != 0:
                                remainder = new_value % div
                                lower = new_value - remainder
                                upper = lower + div

                                valid_candidates = []
                                min_i = max(min_val, 1)
                                max_i = max_val
                                if lower >= min_i:
                                    valid_candidates.append(lower)
                                if upper <= max_i:
                                    valid_candidates.append(upper)
                                if not valid_candidates:
                                    ceil_multiple = ((min_i + div - 1) // div) * div
                                    floor_multiple = (max_i // div) * div
                                    if ceil_multiple <= max_i:
                                        valid_candidates.append(ceil_multiple)
                                    if floor_multiple >= min_i and floor_multiple != ceil_multiple:
                                        valid_candidates.append(floor_multiple)
                                    if not valid_candidates:
                                        valid_candidates.append(max(min_i, div))
                                new_value = min(valid_candidates, key=lambda v: abs(v - new_value))
                                print(
                                    f"    调整 hidden_size 以整除: {original_value} -> {new_value} (div={div}, heads={divisors})")

                print(f"    {param_name}: {original_value} -> {new_value}")
                target_dict[param_name] = new_value
                mutations_applied[_key_with_prefix(prefix, param_name)] = {
                    'from': original_value,
                    'to': new_value,
                    'constraints': constraints
                }
                mutated_cnt += 1

                # 若变异的是 num_heads / num_attention_heads，则修正 hidden_size 满足两者的 LCM
                if param_name in ['num_heads', 'num_attention_heads'] and 'hidden_size' in target_dict:
                    hs_orig = target_dict['hidden_size']
                    nh_cur = target_dict.get('num_heads', None)
                    nah_cur = target_dict.get('num_attention_heads', None)
                    divisors = [h for h in [nh_cur, nah_cur] if isinstance(h, int) and h > 0]
                    if divisors and isinstance(hs_orig, int):
                        div = divisors[0]
                        for h in divisors[1:]:
                            div = div * h // math.gcd(div, h)
                        if hs_orig % div != 0:
                            remainder = hs_orig % div
                            lower = hs_orig - remainder
                            upper = lower + div
                            hs_constraints = mutable_params_pool.get('hidden_size', {})
                            min_hs = max(hs_constraints.get('min_val', 1), 1)
                            max_hs = hs_constraints.get('max_val', hs_orig)

                            candidates = []
                            if lower >= min_hs:
                                candidates.append(lower)
                            if upper <= max_hs:
                                candidates.append(upper)
                            if not candidates:
                                ceil_multiple = ((min_hs + div - 1) // div) * div
                                floor_multiple = (max_hs // div) * div
                                if ceil_multiple <= max_hs:
                                    candidates.append(ceil_multiple)
                                if floor_multiple >= min_hs and floor_multiple != ceil_multiple:
                                    candidates.append(floor_multiple)
                                if not candidates:
                                    candidates.append(max(min_hs, div))

                            adjusted_hs = min(candidates, key=lambda v: abs(v - hs_orig))
                            if adjusted_hs != hs_orig:
                                hidden_key = _key_with_prefix(prefix, 'hidden_size')
                                print(
                                    f"    修正 hidden_size 以整除: {hs_orig} -> {adjusted_hs} (div={div}, heads={divisors})")
                                target_dict['hidden_size'] = adjusted_hs
                                if hidden_key in mutations_applied:
                                    mutations_applied[hidden_key]['to'] = adjusted_hs
                                else:
                                    mutations_applied[hidden_key] = {
                                        'from': hs_orig,
                                        'to': adjusted_hs,
                                        'constraints': mutable_params_pool.get('hidden_size', {})
                                    }

        # 1) 先对顶层 working_config 进行变异（原行为保留）
        _select_and_mutate(working_config, prefix=None)

        # 2) 如果配置中包含 vision_encoder / vision_projector 子字典，则对其内部参数单独变异
        for vision_key in ['vision_encoder', 'vision_projector']:
            if vision_key in working_config and isinstance(working_config[vision_key], dict):
                _select_and_mutate(working_config[vision_key], prefix=vision_key)

        # 变异input_size和patch_size（特殊处理）
        # if 'input_size' in working_config and random.random() < mutation_rate:
        #     print(f"  变异参数: input_size")
        #     original_size = working_config['input_size']
        #     # 保持相对比例，但允许适度变化
        #     new_size = []
        #     for i, dim in enumerate(original_size):
        #         if i == 0:  # 时间维度
        #             new_dim = max(1, min(32, int(dim * random.uniform(0.5, 2.0))))
        #         else:  # 空间维度
        #             new_dim = max(8, min(128, int(dim * random.uniform(0.7, 1.3))))
        #         new_size.append(new_dim)
        #     print(f"    input_size: {original_size} -> {new_size}")
        #     working_config['input_size'] = new_size
        #     mutations_applied['input_size'] = {
        #         'from': original_size,
        #         'to': new_size
        #     }

        # 最终验证和修复配置
        validated_config = self._validate_and_fix_config(working_config, internal_model_type)

        # 更新增量配置和历史记录
        if node_id is not None:
            self.incremental_configs[node_id] = validated_config.copy()

            # 记录变异历史
            mutation_record = {
                'round': self.current_round + 1,
                'timestamp': datetime.now().isoformat(),
                'pre_mutation': pre_mutation_config,
                'post_mutation': validated_config,
                'mutations_applied': mutations_applied,
                'mutation_rate': mutation_rate,
                'model_type': model_type,
                'use_accumulated': use_accumulated
            }

            if node_id not in self.mutation_history:
                self.mutation_history[node_id] = []
            self.mutation_history[node_id].append(mutation_record)

            print(f"✓ 节点 {node_id} 第 {self.current_round + 1} 轮变异完成，应用了 {len(mutations_applied)} 个参数变异")

        return validated_config

    def advance_round(self):
        """推进到下一轮变异"""
        self.current_round += 1
        print(f"🔄 变异轮次推进到第 {self.current_round + 1} 轮")

    def get_incremental_changes_summary(self, node_id: int) -> Dict[str, Any]:
        """
        获取指定节点的增量变异变化汇总

        Args:
            node_id: 节点ID

        Returns:
            Dict[str, Any]: 增量变化汇总，包含参数变化统计
        """
        if node_id not in self.mutation_history:
            return {
                'node_id': node_id,
                'total_rounds': 0,
                'current_round': self.current_round,
                'incremental_changes': {},
                'total_parameters_mutated': 0
            }

        history = self.mutation_history[node_id]

        # 统计所有参数的变化
        parameter_changes = {}
        total_mutated_params = 0

        for round_record in history:
            mutations = round_record.get('mutations_applied', {})
            for param_name, mutation_info in mutations.items():
                if param_name not in parameter_changes:
                    parameter_changes[param_name] = {
                        'initial': mutation_info['from'],
                        'current': mutation_info['to'],
                        'total_rounds_mutated': 1
                    }
                    total_mutated_params += 1
                else:
                    parameter_changes[param_name]['current'] = mutation_info['to']
                    parameter_changes[param_name]['total_rounds_mutated'] += 1

        return {
            'node_id': node_id,
            'total_rounds': len(history),
            'current_round': self.current_round,
            'incremental_changes': parameter_changes,
            'total_parameters_mutated': total_mutated_params
        }

    def _validate_and_fix_config(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """
        验证和修复配置参数，确保模型能够正常初始化
        """
        fixed_config = config.copy()

        def safe_check_non_positive(value):
            """安全检查值是否为非正数或None"""
            return value is None or value <= 0

        # 所有模型类型统一使用STDiT3验证规则
        print(f"  对{model_type}应用模型验证规则")

        # 1. 确保 hidden_size 能被 num_heads 和 num_attention_heads 的最小公倍数整除
        if 'hidden_size' in fixed_config:
            hidden_size = fixed_config['hidden_size']
            if safe_check_non_positive(hidden_size):
                fixed_config['hidden_size'] = 512
                print(f"  修复 hidden_size: {hidden_size} -> 512 (必须大于0)")
                hidden_size = 512

            heads = []
            if 'num_heads' in fixed_config:
                nh = fixed_config['num_heads']
                if safe_check_non_positive(nh):
                    fixed_config['num_heads'] = 8
                    print(f"  修复 num_heads: {nh} -> 8 (必须大于0)")
                    nh = 8
                heads.append(nh)
            if 'num_attention_heads' in fixed_config:
                nah = fixed_config['num_attention_heads']
                if safe_check_non_positive(nah):
                    fixed_config['num_attention_heads'] = 8
                    print(f"  修复 num_attention_heads: {nah} -> 8 (必须大于0)")
                    nah = 8
                heads.append(nah)

            heads = [h for h in heads if h and h > 0]
            if heads:
                div = heads[0]
                for h in heads[1:]:
                    div = div * h // math.gcd(div, h)
                if hidden_size % div != 0:
                    new_hidden_size = (hidden_size // div + 1) * div
                    fixed_config['hidden_size'] = new_hidden_size
                    print(f"  修复 hidden_size: {hidden_size} -> {new_hidden_size} (必须能被LCM{heads}整除)")

        # 2. 确保其他关键参数的有效性
        if 'num_layers' in fixed_config and safe_check_non_positive(fixed_config['num_layers']):
            fixed_config['num_layers'] = 2
            print(f"  修复 num_layers: {config.get('num_layers')} -> 2")

        # if 'model_max_length' in fixed_config and safe_check_non_positive(fixed_config['model_max_length']):
        #     fixed_config['model_max_length'] = 64
        #     print(f"  修复 model_max_length: {config.get('model_max_length')} -> 64")

        # 3. 确保caption_channels存在且正确（STDiT3必须为4096以匹配T5编码器）
        # if 'caption_channels' not in fixed_config or safe_check_non_positive(fixed_config['caption_channels']) or \
        #         fixed_config['caption_channels'] != 4096:
        #     original_value = fixed_config.get('caption_channels', 'MISSING')
        #     fixed_config['caption_channels'] = 4096
        #     print(f"  修复 caption_channels: {original_value} -> 4096 (必须与T5编码器匹配)")

        # 通用验证规则
        # 确保input_size和patch_size有效
        if 'input_size' in fixed_config:
            input_size = fixed_config['input_size']
            if isinstance(input_size, list) and len(input_size) >= 3:
                # 确保所有维度都大于0
                new_input_size = []
                for i, dim in enumerate(input_size):
                    if dim is None or dim <= 0:
                        new_dim = 8 if i > 0 else 2  # 时间维度默认2，空间维度默认8
                        new_input_size.append(new_dim)
                        print(f"  修复 input_size[{i}]: {dim} -> {new_dim}")
                    else:
                        new_input_size.append(dim)
                fixed_config['input_size'] = new_input_size

        if 'patch_size' in fixed_config:
            patch_size = fixed_config['patch_size']
            if isinstance(patch_size, list) and len(patch_size) >= 3:
                # 确保所有patch维度都大于0
                new_patch_size = []
                for i, dim in enumerate(patch_size):
                    if dim is None or dim <= 0:
                        new_dim = 2 if i > 0 else 1  # 时间维度默认1，空间维度默认2
                        new_patch_size.append(new_dim)
                        print(f"  修复 patch_size[{i}]: {dim} -> {new_dim}")
                    else:
                        new_patch_size.append(dim)
                fixed_config['patch_size'] = new_patch_size

        # 确保通道数有效
        for channel_param in ['in_channels', 'out_channels']:
            if channel_param in fixed_config and safe_check_non_positive(fixed_config[channel_param]):
                fixed_config[channel_param] = 4
                print(f"  修复 {channel_param}: {config.get(channel_param)} -> 4")

        return fixed_config

    def create_mutated_model(self, mutated_config: Dict[str, Any]) -> torch.nn.Module:
        """
        根据变异后的配置创建模型实例

        Args:
            mutated_config: 变异后的配置

        Returns:
            torch.nn.Module: 模型实例
        """
        model_id = mutated_config.get('model_id', 'stdit3')

        print(f"  正在创建模型 (model_id: {model_id})")

        # 统一使用STDiT3
        from mindspeed_mm.models.predictor.dits.stdit3 import STDiT3
        # 移除可能导致问题的参数
        clean_config = {k: v for k, v in mutated_config.items()
                        if k not in ['model_id']}

        # 确保STDiT3的关键参数不为None
        if 'caption_channels' not in clean_config or clean_config['caption_channels'] is None:
            clean_config['caption_channels'] = 4096
            print(f"    修复 caption_channels: None -> 4096")

        # if 'model_max_length' not in clean_config or clean_config['model_max_length'] is None:
        #     clean_config['model_max_length'] = 300
        #     print(f"    修复 model_max_length: None -> 300")

        if 'hidden_size' not in clean_config or clean_config['hidden_size'] is None:
            clean_config['hidden_size'] = 1152
            print(f"    修复 hidden_size: None -> 1152")

        print(
            f"    最终配置检查: caption_channels={clean_config.get('caption_channels')}, model_max_length={clean_config.get('model_max_length')}, hidden_size={clean_config.get('hidden_size')}")

        print("正在创建模型（这一步的时间开销可能较长）")
        model = STDiT3(**clean_config)

        print(f"  模型创建完成，正在移动到设备: {GLOBAL_DEVICE}")

        # 确保模型及其所有参数和缓冲区都移动到全局设备
        model = model.to(GLOBAL_DEVICE)

        # NPU设备特殊处理
        if 'npu' in str(GLOBAL_DEVICE):
            print(f"  检测到NPU设备，执行额外的设备同步")
            # 强制将所有参数和缓冲区移动到NPU设备
            for name, param in model.named_parameters():
                if not param.device == GLOBAL_DEVICE:
                    param.data = param.data.to(GLOBAL_DEVICE)
                    print(f"    参数 {name} 移动到 {GLOBAL_DEVICE}")

            for name, buffer in model.named_buffers():
                if buffer is not None and not buffer.device == GLOBAL_DEVICE:
                    buffer.data = buffer.data.to(GLOBAL_DEVICE)
                    print(f"    缓冲区 {name} 移动到 {GLOBAL_DEVICE}")

            # 设置正确的数据类型
            if hasattr(model, 'half'):
                model = model.half()

        return model

    def save_mutated_config(self, mutated_config: Dict[str, Any],
                            iteration: int, model_type: str = 'stdit3') -> str:
        """
        保存变异后的配置到文件

        Args:
            mutated_config: 变异后的配置
            iteration: 迭代次数
            model_type: 模型类型

        Returns:
            str: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"mutated_{model_type}_{iteration}_{timestamp}.yaml"
        filepath = os.path.join(self.output_dir, filename)

        save_data = {
            'predict_config': mutated_config,
            'metadata': {
                'iteration': iteration,
                'model_type': model_type,
                'created_time': datetime.now().isoformat(),
                'mutated_from': 'base_config'
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(save_data, f, default_flow_style=False, allow_unicode=True)

        print(f"✓ 保存变异配置到: {filepath}")
        return filepath


class MutatedMMNode:
    """变异后的多模态节点"""

    def __init__(self, node_id: int, mutated_config: Dict[str, Any],
                 model_type: str = 'stdit3'):
        """
        初始化变异后的多模态节点

        Args:
            node_id: 节点ID
            mutated_config: 变异后的配置
            model_type: 模型类型（统一使用STDiT3）
        """
        self.node_id = node_id
        self.mutated_config = mutated_config
        self.model_type = model_type
        self.model = None

        print(f"✓ 创建变异多模态节点 {node_id} (类型: {model_type})")

        # 在创建模型前再次验证和修复配置
        print(f"  验证和修复配置...")
        validated_config = self._validate_and_fix_config(mutated_config, model_type)
        self.mutated_config = validated_config  # 使用验证后的配置

        # 直接创建模型实例
        try:
            self.model = self._create_model_directly(validated_config, model_type)
            print(f"  成功创建模型实例")
        except Exception as e:
            print(f"  ✗ 创建模型实例失败: {e}")
            import traceback
            traceback.print_exc()
            self.model = None

    def _validate_and_fix_config(self, config: Dict[str, Any], model_type: str) -> Dict[str, Any]:
        """验证和修复配置参数"""
        fixed_config = config.copy()

        def safe_check_non_positive(value):
            """安全检查值是否为非正数或None"""
            return value is None or value <= 0

        # 所有模型类型统一使用STDiT3验证规则
        print(f"  对{model_type}应用模型验证规则")

        # 1. 确保 hidden_size 能被 num_heads 和 num_attention_heads 的最小公倍数整除
        if 'hidden_size' in fixed_config:
            hidden_size = fixed_config['hidden_size']
            if safe_check_non_positive(hidden_size):
                fixed_config['hidden_size'] = 512
                print(f"  修复 hidden_size: {hidden_size} -> 512")
                hidden_size = 512

            heads = []
            if 'num_heads' in fixed_config:
                nh = fixed_config['num_heads']
                if safe_check_non_positive(nh):
                    fixed_config['num_heads'] = 8
                    print(f"  修复 num_heads: {nh} -> 8")
                    nh = 8
                heads.append(nh)
            if 'num_attention_heads' in fixed_config:
                nah = fixed_config['num_attention_heads']
                if safe_check_non_positive(nah):
                    fixed_config['num_attention_heads'] = 8
                    print(f"  修复 num_attention_heads: {nah} -> 8")
                    nah = 8
                heads.append(nah)

            heads = [h for h in heads if h and h > 0]
            if heads:
                div = heads[0]
                for h in heads[1:]:
                    div = div * h // math.gcd(div, h)
                if hidden_size % div != 0:
                    new_hidden_size = (hidden_size // div + 1) * div
                    fixed_config['hidden_size'] = new_hidden_size
                    print(f"  修复 hidden_size: {hidden_size} -> {new_hidden_size} (LCM={heads})")

        # 2. 确保关键参数有效
        if 'num_layers' in fixed_config and safe_check_non_positive(fixed_config['num_layers']):
            fixed_config['num_layers'] = 2
            print(f"  修复 num_layers: {config.get('num_layers')} -> 2")

        # if 'model_max_length' in fixed_config and safe_check_non_positive(fixed_config['model_max_length']):
        #     fixed_config['model_max_length'] = 64
        #     print(f"  修复 model_max_length: {config.get('model_max_length')} -> 64")

        # 3. 确保caption_channels正确
        if 'caption_channels' not in fixed_config or safe_check_non_positive(fixed_config['caption_channels']) or \
                fixed_config['caption_channels'] != 4096:
            original_value = fixed_config.get('caption_channels', 'MISSING')
            fixed_config['caption_channels'] = 4096
            print(f"  修复 caption_channels: {original_value} -> 4096")

        return fixed_config

    def _create_model_directly(self, config: Dict[str, Any], model_type: str) -> torch.nn.Module:
        """直接创建模型实例"""
        try:
            # 统一使用STDiT3
            from mindspeed_mm.models.predictor.dits.stdit3 import STDiT3

            # 移除可能导致问题的参数
            clean_config = {k: v for k, v in config.items()
                            if k not in ['model_type']}

            # 确保STDiT3的关键参数不为None
            if 'caption_channels' not in clean_config or clean_config['caption_channels'] is None:
                clean_config['caption_channels'] = 4096
                print(f"    修复 caption_channels: None -> 4096")

            # if 'model_max_length' not in clean_config or clean_config['model_max_length'] is None:
            #     clean_config['model_max_length'] = 300
            #     print(f"    修复 model_max_length: None -> 300")

            if 'hidden_size' not in clean_config or clean_config['hidden_size'] is None:
                clean_config['hidden_size'] = 1152
                print(f"    修复 hidden_size: None -> 1152")

            print(
                f"    最终配置检查: caption_channels={clean_config.get('caption_channels')}, model_max_length={clean_config.get('model_max_length')}, hidden_size={clean_config.get('hidden_size')}")

            print("正在创建模型（这一步的时间开销可能较长）")
            model = STDiT3(**clean_config)

            # NPU设备特殊处理
            if 'npu' in str(GLOBAL_DEVICE):
                print(f"NPU设备检测到，移动模型到设备: {GLOBAL_DEVICE}")
                model = model.to(GLOBAL_DEVICE)
                # 强制将所有参数移动到NPU设备
                for param in model.parameters():
                    if not param.is_cuda and not hasattr(param, 'npu'):
                        param.data = param.data.to(GLOBAL_DEVICE)
                for buffer in model.buffers():
                    if not buffer.is_cuda and not hasattr(buffer, 'npu'):
                        buffer.data = buffer.data.to(GLOBAL_DEVICE)

                # 设置正确的数据类型
                if hasattr(model, 'half'):
                    model = model.half()

                print("NPU设备同步完成")
            else:
                model = model.to(GLOBAL_DEVICE)

            return model

        except Exception as e:
            print(f"创建模型失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def forward(self, video: torch.Tensor, timestep: torch.Tensor,
                prompt: torch.Tensor, **kwargs) -> torch.Tensor:
        """执行前向传播"""
        if self.model is None:
            raise RuntimeError(f"节点 {self.node_id} 的模型未初始化")

        try:
            # 设置模型为评估模式
            self.model.eval()

            # 确保所有输入张量都在正确的设备上
            video = video.to(GLOBAL_DEVICE)
            timestep = timestep.to(GLOBAL_DEVICE)
            prompt = prompt.to(GLOBAL_DEVICE)

            # 将kwargs中的张量也移动到正确设备
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    kwargs[key] = value.to(GLOBAL_DEVICE)

            # 为STDiT3模型准备必要的参数
            _, _, _, H, W = video.shape
            if 'height' not in kwargs:
                kwargs['height'] = torch.tensor([H], dtype=torch.long, device=GLOBAL_DEVICE)
            if 'width' not in kwargs:
                kwargs['width'] = torch.tensor([W], dtype=torch.long, device=GLOBAL_DEVICE)
            if 'fps' not in kwargs:
                # 根据模型的数据类型设置fps
                fps_dtype = torch.float16 if self.mutated_config.get('dtype') == 'fp16' else torch.float32
                kwargs['fps'] = torch.tensor([24.0], dtype=fps_dtype, device=GLOBAL_DEVICE)
            if 'prompt_mask' not in kwargs:
                # 创建prompt掩码
                B, _, N_token, _ = prompt.shape
                kwargs['prompt_mask'] = torch.ones(B, N_token, dtype=torch.bool, device=GLOBAL_DEVICE)

            with torch.no_grad():
                output = self.model(video=video, timestep=timestep, prompt=prompt, **kwargs)

            return output

        except Exception as e:
            print(f"✗ 节点 {self.node_id} 前向传播失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def get_config(self) -> Dict[str, Any]:
        """获取配置"""
        return self.mutated_config.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        info = {
            'node_id': self.node_id,
            'model_type': self.model_type,
            'config': self.mutated_config,
            'model_initialized': self.model is not None
        }

        if self.model is not None:
            # 统计模型参数
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            info.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)  # 假设fp32
            })

        return info


def demo_mm_mutation():
    """演示多模态变异系统"""
    print("=== 多模态变异系统演示 ===")

    # 创建变异器
    mutator = MMConfigMutator()

    # 测试STDiT3变异（sora模型类型）
    print("\n--- 测试 Sora (STDiT3) 变异 ---")
    sora_mutated = mutator.mutate_predict_config(
        base_config=None,
        mutation_rate=0.3,
        model_type='sora'
    )

    print("Sora 变异后配置:")
    for key, value in sora_mutated.items():
        print(f"  {key}: {value}")

    # 保存变异配置
    sora_config_path = mutator.save_mutated_config(sora_mutated, 1, 'sora')

    # 测试CogVideoX变异（统一使用STDiT3）
    print("\n--- 测试 CogVideoX (STDiT3) 变异 ---")
    cogvideox_mutated = mutator.mutate_predict_config(
        base_config=None,
        mutation_rate=0.3,
        model_type='cogvideox'
    )

    print("CogVideoX 变异后配置:")
    for key, value in cogvideox_mutated.items():
        print(f"  {key}: {value}")

    cogvideox_config_path = mutator.save_mutated_config(cogvideox_mutated, 1, 'cogvideox')

    # 创建变异节点
    print("\n--- 测试变异节点创建 ---")
    node1 = MutatedMMNode(1, sora_mutated, 'sora')
    node2 = MutatedMMNode(2, cogvideox_mutated, 'cogvideox')

    # 显示节点信息
    print("\n--- 节点信息 ---")
    print("节点1信息:", node1.get_model_info())
    print("节点2信息:", node2.get_model_info())

    print("\n=== 演示完成 ===")


if __name__ == "__main__":
    demo_mm_mutation()
