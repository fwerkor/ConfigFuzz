from typing import Iterable, List, Union

import msadapter
from msadapter import Tensor



def get_rng_state(device: Union[int, str, msadapter.device] = "cuda") -> Tensor:
    r"""Return the random number generator state of the specified GPU as a ByteTensor.

    Args:
        device (msadapter.device or int, optional): The device to return the RNG state of.
            Default: ``'cuda'`` (i.e., ``msadapter.device('cuda')``, the current CUDA device).

    .. warning::
        This function eagerly initializes CUDA.
    """
    
    from . import _lazy_init, current_device
    _lazy_init()
    if isinstance(device, str):
        device = msadapter.device(device)
    elif isinstance(device, int):
        device = msadapter.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = current_device()
    default_generator = msadapter.cuda.default_generators[idx]
    return default_generator.get_state()
