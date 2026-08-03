import torch
import torch.nn as nn


def test_api_completeness():
    choices = nn.Sequential(nn.ReLU())
    choices.append(nn.Conv1d(10, 10, 3))
    choices.insert(1, nn.Conv3d(10, 10, 3))
    choices.pop(0)
    assert len(choices) == 2

    try:
        nn.Sequential((1.0, 2.0))
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass

    try:
        nn.Sequential(torch.Tensor(2))
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass

    try:
        nn.Sequential(torch.acos)
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass


if __name__ == '__main__':
    test_api_completeness()
