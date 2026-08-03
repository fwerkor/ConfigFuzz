import torch.nn as nn


def test_api_completeness():
    choices = nn.ModuleList([nn.Conv2d(10, 10, 3)])
    choices.append(nn.Conv1d(10, 10, 3))
    choices.extend([nn.Conv1d(10, 10, 3)])
    choices.insert(1, nn.Conv3d(10, 10, 3))
    assert len(choices) == 4

    try:
        nn.ModuleList({'test': nn.ReLU()})
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass

    try:
        nn.ModuleList(0.12)
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass

    try:
        nn.ModuleList([nn.ReLU])
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass


if __name__ == '__main__':
    test_api_completeness()
