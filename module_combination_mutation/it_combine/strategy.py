import torch
import secrets

class ImageTextCombineStrategy:
    def __init__(self, name: str):
        self.name = name

    def combine(self, image_features: torch.Tensor, text_embeddings: torch.Tensor, input_ids: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pass

class ImageTextCombineStrategyRegistry:
    def __init__(self):
        self.strategies = {}

    def register(self, name: str, strategy: ImageTextCombineStrategy):
        self.strategies[name] = strategy

    def get(self, name: str) -> ImageTextCombineStrategy:
        return self.strategies[name]

    def random_choice(self) -> ImageTextCombineStrategy:
        if not self.strategies:
            raise ValueError("No image text combine strategies registered in IMAGE_TEXT_COMBINE_STRATEGY_REGISTRY")
        return secrets.choice(list(self.strategies.values()))

IMAGE_TEXT_COMBINE_STRATEGY_REGISTRY = ImageTextCombineStrategyRegistry()