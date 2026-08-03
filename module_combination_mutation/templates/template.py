import secrets
from collections import OrderedDict

from mindspeed_mm.configs.config import ConfigReader
from mindspeed_mm.models.common.module import MultiModalModule
from modules.text_decoder import TextDecoder
from modules.image_encoder import ImageEncoder
from modules.ae import AE
from it_combine.strategy import ImageTextCombineStrategy

EMBEDDING_METHOD = ["embedding"]


class Template():
    def __init__(self, name: str):
        self.name = name

    def select_modules(self):
        pass

    def test_forward(self):
        pass

    def instantiate(self):
        pass
        
    def dump_graph(self, output_path: str = None) -> str:
        pass


class MMInstance(MultiModalModule):
    def __init__(self, name: str, config: dict, template: Template) -> None:
        self.mm_config = ConfigReader(config)
        super().__init__(config=self.mm_config)
        self.template = template

        # moudles instances
        self.text_embedding_decoder = None
        self.image_encoder = None
        self.image_text_combine_strategy = None
        self.text_decoders = []
        self.ae = None

        # config
        self.config_dict = {
            "template": self.template.name,
            "text_embedding_decoder": None,
            "image_encoder": None,
            "image_text_combine_strategy": None,
            "text_decoders": [],
            "ae": None
        }

    def set_text_embedding_decoder(self, text_embedding_decoder: TextDecoder, config_dict: dict):
        self.text_embedding_decoder = text_embedding_decoder
        self.config_dict["text_embedding_decoder"] = {
            "name": text_embedding_decoder.name,
            "config": config_dict
        }

    def set_image_encoder(self, image_encoder: ImageEncoder, config_dict: dict):
        self.image_encoder = image_encoder
        self.config_dict["image_encoder"] = {
            "name": image_encoder.name,
            "config": config_dict
        }

    def set_image_text_combine_strategy(self, image_text_combine_strategy: ImageTextCombineStrategy):
        self.image_text_combine_strategy = image_text_combine_strategy
        self.config_dict["image_text_combine_strategy"] = image_text_combine_strategy.name

    def add_text_decoder(self, text_decoder: TextDecoder, config_dict: dict):
        self.text_decoders.append(text_decoder)
        self.config_dict["text_decoders"].append({
            "name": text_decoder.name,
            "config": config_dict
        })

    def set_ae(self, ae: AE, config_dict: dict):
        self.ae = ae
        self.config_dict["ae"] = {
            "name": ae.name,
            "config": config_dict
        }

    def get_config_dict(self) -> dict:
        return self.config_dict

    def _module_state_targets(self):
        targets = []
        if self.text_embedding_decoder is not None and getattr(self.text_embedding_decoder, "decoder", None) is not None:
            targets.append(("text_embedding_decoder.", self.text_embedding_decoder.decoder))
        if self.image_encoder is not None and getattr(self.image_encoder, "encoder", None) is not None:
            targets.append(("image_encoder.", self.image_encoder.encoder))
        for idx, text_decoder in enumerate(self.text_decoders):
            decoder_module = getattr(text_decoder, "decoder", None)
            if decoder_module is not None:
                targets.append((f"text_decoders.{idx}.", decoder_module))
        if self.ae is not None and getattr(self.ae, "ae", None) is not None:
            targets.append(("ae.", self.ae.ae))
        return targets

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Include wrapped module weights that are not registered on MMInstance."""
        if destination is None:
            destination = OrderedDict()

        super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)

        for module_prefix, module in self._module_state_targets():
            module_state_dict = module.state_dict(prefix=f"{prefix}{module_prefix}", keep_vars=keep_vars)
            destination.update(module_state_dict)

        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Load wrapped module weights from the custom MMInstance checkpoint layout."""
        result = super().load_state_dict(state_dict, strict=False)
        missing_keys = list(getattr(result, "missing_keys", []))
        unexpected_keys = list(getattr(result, "unexpected_keys", []))
        consumed_keys = set()
        module_targets = self._module_state_targets()
        custom_prefixes = tuple(module_prefix for module_prefix, _ in module_targets)

        if custom_prefixes:
            unexpected_keys = [
                key for key in unexpected_keys
                if not key.startswith(custom_prefixes)
            ]

        for module_prefix, module in module_targets:
            submodule_state_dict = {
                key[len(module_prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(module_prefix)
            }
            if not submodule_state_dict:
                if strict:
                    expected_keys = list(module.state_dict().keys())
                    missing_keys.extend([f"{module_prefix}{key}" for key in expected_keys])
                continue

            load_result = module.load_state_dict(submodule_state_dict, strict=strict)
            consumed_keys.update(f"{module_prefix}{key}" for key in submodule_state_dict.keys())
            missing_keys.extend([f"{module_prefix}{key}" for key in getattr(load_result, "missing_keys", [])])
            unexpected_keys.extend([f"{module_prefix}{key}" for key in getattr(load_result, "unexpected_keys", [])])

        for key in state_dict.keys():
            if key in consumed_keys:
                continue
            if key in unexpected_keys:
                continue
            if key.startswith(custom_prefixes):
                unexpected_keys.append(key)

        missing_keys = list(dict.fromkeys(missing_keys))
        unexpected_keys = list(dict.fromkeys(unexpected_keys))

        if strict and (missing_keys or unexpected_keys):
            problems = []
            if missing_keys:
                problems.append(f"Missing key(s): {missing_keys}")
            if unexpected_keys:
                problems.append(f"Unexpected key(s): {unexpected_keys}")
            raise RuntimeError("Error(s) in loading state_dict for MMInstance:\n\t" + "\n\t".join(problems))

        return type(result)(missing_keys, unexpected_keys)

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

    def forward(self, *args, **kwargs):
        output = self.template.forward(self, *args, **kwargs)
        return output


class TemplateRegistry:
    def __init__(self):
        self.templates = []

    def register(self, template: Template):
        self.templates.append(template)

    def get(self, name: str) -> Template:
        for t in self.templates:
            if t.name == name:
                return t
        raise ValueError(f"No template named '{name}' in TEMPLATE_REGISTRY")

    def random_choice(self) -> Template:
        if not self.templates:
            raise ValueError("No templates registered in TEMPLATE_REGISTRY")
        return secrets.choice(self.templates)

TEMPLATE_REGISTRY = TemplateRegistry()