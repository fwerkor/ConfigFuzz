import os
from packaging import version
import mindspore
from mindspore._c_expression import MSContext # pylint: disable=no-name-in-module, import-error

DEVICE_TARGET = mindspore.get_context('device_target')
USE_PYBOOST = DEVICE_TARGET == 'Ascend'
DEFAULT_DTYPE = mindspore.float32

_MS_VERSION = version.parse(mindspore.__version__)
MS26 = _MS_VERSION >= version.parse('2.6')
MS27 = _MS_VERSION >= version.parse('2.7')
MS271 = _MS_VERSION >= version.parse('2.7.1')
MS28 = _MS_VERSION >= version.parse('2.8')

try:
    import ms_op_plugin
    if not os.getenv("MS_OP_PLUGIN_PATH"):
        raise ImportError("Environment variable \"MS_OP_PLUGIN_PATH\" should be set when using ms_op_plugin.")
    HAS_OP_PLUGIN = True
except ImportError as e:
    HAS_OP_PLUGIN = False

def soc_version():
    if not hasattr(soc_version, 'soc'):
        soc_version.soc = MSContext.get_instance().get_ascend_soc_version()
    return soc_version.soc

def support_bf16():
    if not hasattr(support_bf16, 'flag'):
        support_bf16.flag = soc_version() in ["ascend910b", "ascend910_93"]
    return support_bf16.flag

def on_orange_pi():
    if not hasattr(on_orange_pi, 'flag'):
        on_orange_pi.flag = '310b' in soc_version()
    return on_orange_pi.flag

def set_pyboost(mode: bool):
    """set global pyboost"""
    global USE_PYBOOST
    USE_PYBOOST = mode

def use_pyboost():
    """set global pyboost"""
    return USE_PYBOOST