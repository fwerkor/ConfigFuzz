import sys
from msadapter.proxy import enable_torch_proxy

sys.modules.pop('torch')
enable_torch_proxy()
