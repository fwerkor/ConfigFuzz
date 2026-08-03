import unittest
from typing import List
from functools import wraps

import torch_npu

def is_iterable(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


class SupportedDevices:
    def __init__(self, supported_devices: List[str]) -> None:
        self.supported_devices = supported_devices

    def __call__(self, fn):
        @wraps(fn)
        def dep_fn(slf, *args, **kwargs):
            device_name = torch_npu.npu.get_device_name(0)[:10]
            if device_name not in self.supported_devices:
                reason = f"Only run on {repr(self.supported_devices)}, current device is {device_name}."
                raise unittest.SkipTest(reason)
            return fn(slf, *args, **kwargs)

        return dep_fn
