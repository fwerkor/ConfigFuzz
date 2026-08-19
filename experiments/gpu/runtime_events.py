from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


EVENT_PREFIX = "CONFIGFUZZ_RUNTIME_EVENT:"


class RuntimeEventRecorder:
    """Emit stable, low-cardinality runtime events once per rank-zero process."""

    def __init__(self, rank: int = 0) -> None:
        self.rank = rank
        self._seen: set[tuple[str, str]] = set()
        self._hooks: list[Any] = []

    def emit(self, kind: str, value: str) -> None:
        if self.rank != 0:
            return
        key = (str(kind), str(value))
        if key in self._seen:
            return
        self._seen.add(key)
        payload = json.dumps(
            {"kind": key[0], "value": key[1]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        print(f"{EVENT_PREFIX}{payload}", flush=True)

    def emit_profile_state(self, profile: Mapping[str, Any]) -> None:
        family = str(profile.get("family", "unknown"))
        self.emit("branch", f"family={family}")

        precision = profile.get("precision", {})
        if isinstance(precision, Mapping) and bool(precision.get("bf16", False)):
            self.emit("branch", "precision=bf16")
        elif isinstance(precision, Mapping) and bool(precision.get("fp16", False)):
            self.emit("branch", "precision=fp16")
        else:
            self.emit("branch", "precision=fp32")

        model = profile.get("model", {})
        if isinstance(model, Mapping):
            heads = _positive_int(model.get("num_attention_heads"))
            groups = _positive_int(model.get("num_query_groups"))
            if heads is not None and groups is not None:
                if groups == heads:
                    mode = "mha"
                elif groups == 1:
                    mode = "mqa"
                else:
                    mode = "gqa"
                self.emit("branch", f"attention_mode={mode}")

        moe = profile.get("moe")
        self.emit("branch", f"moe={'enabled' if isinstance(moe, Mapping) and moe else 'disabled'}")
        if isinstance(moe, Mapping) and moe:
            self.emit("feature", "moe_configuration")
            if _positive_int(moe.get("n_shared_experts")):
                self.emit("feature", "shared_experts")

    def emit_distributed_state(
        self,
        *,
        framework: str,
        world_size: int,
        tp: int = 1,
        pp: int = 1,
        cp: int = 1,
        ep: int = 1,
        distributed_backend: str = "nccl",
    ) -> None:
        self.emit("backend", f"framework={framework}")
        self.emit("backend", f"distributed={distributed_backend}")
        self.emit(
            "topology",
            f"world={int(world_size)},tp={int(tp)},pp={int(pp)},cp={int(cp)},ep={int(ep)}",
        )

    def instrument_model(self, model: Any, profile: Mapping[str, Any]) -> None:
        self._emit_resolved_attention_backend(model)
        for module in model.modules():
            event_values = _module_events(module)
            if not event_values:
                continue

            def hook(_module, _inputs, _output, values=event_values):
                for kind, value in values:
                    self.emit(kind, value)

            self._hooks.append(module.register_forward_hook(hook))

    def close(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def _emit_resolved_attention_backend(self, model: Any) -> None:
        configs: list[Any] = []
        config = getattr(model, "config", None)
        if config is not None:
            configs.append(config)
            for name in ("text_config", "vision_config"):
                nested = getattr(config, name, None)
                if nested is not None:
                    configs.append(nested)
        for cfg in configs:
            implementation = getattr(cfg, "_attn_implementation", None)
            if implementation:
                self.emit("backend", f"attention={implementation}")

        processors = {
            type(getattr(module, "processor")).__name__
            for module in model.modules()
            if getattr(module, "processor", None) is not None
            and "attention" in type(module).__name__.lower()
        }
        for processor in sorted(processors):
            self.emit("backend", f"attention_processor={processor}")


def _positive_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def _module_events(module: Any) -> tuple[tuple[str, str], ...]:
    name = type(module).__name__.lower()
    module_name = type(module).__module__.lower()
    values: list[tuple[str, str]] = []

    if "attention" in name or name.endswith("attn"):
        values.append(("feature", "attention"))
        if "vision" in name or "vision" in module_name:
            values.append(("feature", "vision_attention"))
        else:
            values.append(("feature", "text_attention"))

    if any(token in name for token in ("moe", "sparsemoe", "mixtureofexperts")):
        values.append(("feature", "moe"))
    if "router" in name:
        values.append(("feature", "moe_router"))
    if "expert" in name:
        values.append(("feature", "experts"))

    if "visionmodel" in name or "visionencoder" in name:
        values.append(("feature", "vision_encoder"))
    if "multimodalprojector" in name or "projector" in name and "internvl" in module_name:
        values.append(("feature", "multimodal_projector"))

    if "cogvideoxpatchembed" in name:
        values.append(("feature", "video_patch_embed"))
    if "cogvideoxtransformer3dmodel" in name:
        values.append(("feature", "video_transformer"))

    return tuple(dict.fromkeys(values))
