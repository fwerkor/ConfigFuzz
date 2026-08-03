from torch import manual_seed as torch_manual_seed

def manual_seed_all(seed: int):
    torch_manual_seed(seed)

def manual_seed(seed: int):
    torch_manual_seed(seed)