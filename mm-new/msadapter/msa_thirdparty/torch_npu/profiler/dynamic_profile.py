"""dynamic_profile"""
from mindspore.profiler import DynamicProfilerMonitor
__all__ = [
    'init',
    'step',
]


class _DynamicProfile:
    """
    DynamicProfile
    """
    _instance = None

    def __new__(cls, path: str = None):
        if cls._instance is None:
            cls._instance = super(_DynamicProfile, cls).__new__(cls)
            cls._instance.dynamic_profiler = DynamicProfilerMonitor(path)
        return cls._instance

    def step(self):
        self.dynamic_profiler.step()


def init(path: str):
    _DynamicProfile(path)


def step():
    _DynamicProfile().step()
