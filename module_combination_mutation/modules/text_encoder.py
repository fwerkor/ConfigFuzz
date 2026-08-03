import torch
import secrets

class TextEncoder:
    def __init__(self, name: str, config: dict):
        self.name = name
        self.config = config
        self.encoder = None

        self.encoder = self._build()

    def _build(self):
        pass

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

class T5TextEncoder(TextEncoder):
    def __init__(self, name: str, config: dict):
        super().__init__(name, config)

    def _build(self):
        pass

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass