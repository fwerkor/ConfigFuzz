import msadapter
import mindspore
from mindspore.common.api import _pynative_executor


has_lapack = True
DisableTorchFunctionSubclass = None


def _jit_set_profiling_executor(mode):
    pass

def _jit_set_profiling_executor(mode):
    pass

def _jit_set_profiling_mode(mode):
    pass

def _jit_override_can_fuse_on_cpu(mode):
    pass

def _jit_override_can_fuse_on_gpu(mode):
    pass

def _jit_set_texpr_fuser_enabled(mode):
    pass

def _debug_set_autodiff_subgraph_inlining(mode):
    pass

def _log_api_usage_once(*args, **kwargs):
    pass

def _jit_set_nvfuser_enabled(mode):
    pass

def _get_privateuse1_backend_name():
    return 'Ascend'

def _is_multithreading_enabled():
    return False

def _set_multithreading_enabled(mode):
    pass

def _accelerator_getAccelerator():
    return msadapter.device('cuda')

def _accelerator_setDeviceIndex(device_index):
    mindspore.set_device("Ascend", device_index)

def _accelerator_getStream():
    return mindspore.runtime.current_stream()

def _accelerator_setStream(stream):
    mindspore.runtime.set_cur_stream(stream)

def _accelerator_synchronizeDevice(device_index):
    mindspore.runtime.synchronize()

def _current_autograd_node():
    return None

def is_grad_enabled():
    return _pynative_executor.requires_grad()


