import copy
import json
import os
import random
from typing import Dict, Any, Any as _Any

from common import SupportedModules
from utils import log_step, log_bullet


class SingleModuleMutator:
    """单模块配置变异器。

    变异规则从 JSON 中读取，格式类似：
    {
        "text_decoder": {
            "num_layers": {"min_factor": ..., "max_factor": ..., "min_val": ..., "max_val": ...},
            "attention_dropout": {"enums": [0.0, 0.1, 0.2]},
            "some_param": {..., "enabled": false}
        },
        "image_encoder": {
            ...
        }
    }
    某参数若含 "enabled": false，则加载时跳过，不参与变异；其余键上的 enabled 会被去掉。
    """

    def __init__(self, schema_path: str):
        self.schema_path = schema_path
        self.mutation_schema = self._load_schema(schema_path)

    @staticmethod
    def _filter_enabled_params(schema: Dict[str, Any]) -> Dict[str, Any]:
        """丢弃 enabled 为 false 的变异项；其余项去掉 enabled 键，避免参与采样逻辑。"""
        out: Dict[str, Any] = {}
        for module_key, params in schema.items():
            if not isinstance(params, dict):
                out[module_key] = params
                continue
            filtered: Dict[str, Any] = {}
            for param_name, spec in params.items():
                if not isinstance(spec, dict):
                    filtered[param_name] = spec
                    continue
                if spec.get("enabled") is False:
                    continue
                cleaned = dict(spec)
                cleaned.pop("enabled", None)
                filtered[param_name] = cleaned
            out[module_key] = filtered
        return out

    @staticmethod
    def _load_schema(schema_path: str) -> Dict[str, Any]:
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Mutation schema json not found: {schema_path}")
        with open(schema_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f"Mutation schema must be a dict, got {type(data)}")
        return SingleModuleMutator._filter_enabled_params(data)

    def mutate(
        self,
        module_type: SupportedModules,
        base_config: Dict[str, Any],
        mutation_num: int = 3,
    ) -> Dict[str, Any]:
        """对某个模块配置进行变异，返回新的配置字典。

        Args:
            module_type: SupportedModules
            base_config: 原始配置（会被复制，不在原地修改）
            mutation_num: 本次希望变异的参数数量上限
        """
        # 使用深拷贝，避免在嵌套结构中意外修改原始配置
        config = copy.deepcopy(base_config) if base_config is not None else {}
        constraints = self.mutation_schema.get(module_type.value, {})
        if not constraints:
            # 没有约束时直接返回原配置
            return config

        # image_encoder 的实际可变参数位于第二层（如 modules.json 中的 "vision_encoder" / "vision_projector"）
        if module_type == SupportedModules.IMAGE_ENCODER:
            # 分别尝试对 vision_encoder 与 vision_projector 进行变异
            for sub_key in ("vision_encoder", "vision_projector"):
                sub_cfg = config.get(sub_key)
                if not isinstance(sub_cfg, dict):
                    continue

                candidate_params = [
                    name for name in constraints.keys()
                    if name in sub_cfg
                ]
                if not candidate_params:
                    continue

                cur_mutation_num = max(1, min(mutation_num, len(candidate_params)))
                random.shuffle(candidate_params)
                selected_params = candidate_params[:cur_mutation_num]

                for param_name in selected_params:
                    param_constraints = constraints[param_name]
                    original_value = sub_cfg.get(param_name)
                    new_value = self._sample_new_value(
                        param_name,
                        original_value,
                        param_constraints,
                    )
                    sub_cfg[param_name] = new_value

                config[sub_key] = sub_cfg
        else:
            # 文本解码器等模块保持原来的单层结构处理方式
            # 找出可变的参数：既在约束中又在当前 config 里
            candidate_params = [
                name for name in constraints.keys()
                if name in config
            ]
            if not candidate_params:
                return config

            mutation_num = max(1, min(mutation_num, len(candidate_params)))
            random.shuffle(candidate_params)
            selected_params = candidate_params[:mutation_num]

            for param_name in selected_params:
                param_constraints = constraints[param_name]
                original_value = config.get(param_name)
                new_value = self._sample_new_value(
                    param_name,
                    original_value,
                    param_constraints,
                )
                config[param_name] = new_value

        # 记录并打印本次变异前后的差异
        self._log_config_diff(module_type, base_config, config)

        return config

    def _sample_new_value(
        self,
        param_name: str,
        original_value: _Any,
        constraints: Dict[str, Any],
    ):
        # 有枚举就优先走枚举
        enums = constraints.get("enums")
        if isinstance(enums, list) and enums:
            # 尽量采样一个不同的值
            candidates = [v for v in enums if v != original_value]
            if not candidates:
                # 所有候选都与原值相同，兜底：布尔值取反，否则原样返回
                if isinstance(original_value, bool):
                    return not original_value
                return original_value
            return random.choice(candidates)

        # 范围 + 缩放因子
        if all(k in constraints for k in ("min_val", "max_val", "min_factor", "max_factor")):
            try:
                base = int(original_value)
            except Exception:
                # 原值不是数字，直接返回原值
                return original_value

            min_val = int(constraints["min_val"])
            max_val = int(constraints["max_val"])
            min_factor = float(constraints["min_factor"])
            max_factor = float(constraints["max_factor"])

            low = max(min_val, int(base * min_factor))
            high = min(max_val, int(base * max_factor))
            if low > high:
                low, high = high, low

            # 对典型结构参数用 2 的幂做采样，类似 withnum_mutation_system 的逻辑
            pow2_params = {
                "num_layers",
                "hidden_size",
                "ffn_hidden_size",
                "num_attention_heads",
                "num_query_groups",
            }
            if param_name in pow2_params:
                candidates = [
                    2 ** i
                    for i in range(1, 13)
                    if low <= 2 ** i <= high and 2 ** i != base
                ]
            else:
                candidates = [v for v in range(low, high + 1) if v != base]

            if not candidates:
                # 兜底：尝试 ±1 微调
                for delta in (-1, 1, 2, -2):
                    new_v = base + delta
                    if low <= new_v <= high and new_v != base:
                        return new_v
                return base

            return random.choice(candidates)

        # 无法识别的约束，直接返回原值
        return original_value

    def _log_config_diff(
        self,
        module_type: SupportedModules,
        before: Dict[str, Any],
        after: Dict[str, Any],
    ) -> None:
        """在控制台打印单次变异中各参数的前后对比。

        为了让嵌套结构（例如 image_encoder 的 vision_encoder/vision_projector）
        更易读，这里会先将配置展平为链式 key，例如：
        vision_encoder.attention_dropout
        vision_projector.add_bias_linear
        """
        if before is None:
            before = {}
        if after is None:
            after = {}

        # 统一模块类型展示
        module_type_str = module_type.value if isinstance(module_type, SupportedModules) else str(module_type)

        log_step("MutationDiff", f"模块类型: {module_type_str}")

        def _flatten(prefix: str, value: Any, acc: Dict[str, Any]) -> None:
            """将嵌套 dict 展平为链式 key -> value."""
            if isinstance(value, dict):
                for k, v in value.items():
                    new_prefix = f"{prefix}.{k}" if prefix else str(k)
                    _flatten(new_prefix, v, acc)
            else:
                acc[prefix] = value

        flat_before: Dict[str, Any] = {}
        flat_after: Dict[str, Any] = {}
        _flatten("", before, flat_before)
        _flatten("", after, flat_after)

        all_keys = sorted(set(flat_before.keys()) | set(flat_after.keys()))
        changed = False
        for key in all_keys:
            old_val = flat_before.get(key, "<MISSING>")
            new_val = flat_after.get(key, "<MISSING>")
            if old_val != new_val:
                changed = True
                log_bullet(f"{key}: {old_val!r} -> {new_val!r}", indent=2)

        if not changed:
            log_bullet("no parameter changed", indent=2)

