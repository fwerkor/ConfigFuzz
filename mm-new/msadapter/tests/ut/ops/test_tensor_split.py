import torch

def test_indices_or_sections_int():
    input_x = torch.arange(9)
    output = torch.tensor_split(input_x, 3)
    print(output)

def test_indices_or_sections_tensor():
    input_x = torch.arange(9)
    output = torch.tensor_split(input_x, torch.tensor(3))
    print(output)
