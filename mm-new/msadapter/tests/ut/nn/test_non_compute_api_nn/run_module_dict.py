import torch.nn as nn


def test_api_completeness():
    choices = nn.ModuleDict({
        'test1': nn.Conv2d(10, 10, 3)
    })
    choices.add_module('test2', nn.Conv1d(10, 10, 3))
    choices.update({'test2': nn.Conv3d(10, 10, 3)})
    choices.pop('test2')
    assert tuple(choices.keys()) == ('test1',)

    try:
        nn.ModuleDict([1, 2])
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass

    try:
        nn.ModuleDict({(1, 2): nn.ReLU()})
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass

    try:
        nn.ModuleDict({True: nn.ReLU()})
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass

    try:
        nn.ModuleDict({'test': 123})
        assert False, 'The completeness verification failed!'
    except TypeError:
        pass

    try:
        nn.ModuleDict({'re.lu': nn.ReLU()})
        assert False, 'The completeness verification failed!'
    except KeyError:
        pass

    try:
        nn.ModuleDict({'': nn.ReLU()})
        assert False, 'The completeness verification failed!'
    except KeyError:
        pass


if __name__ == '__main__':
    test_api_completeness()
