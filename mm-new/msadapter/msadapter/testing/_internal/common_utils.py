import os
import random
import sys
import signal
import unittest
import subprocess
import expecttest
from functools import wraps
import numpy as np
import msadapter


def seed_all(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    msadapter.manual_seed(seed)
    msadapter.use_deterministic_algorithms(True)

def _check_module_exists(name: str) -> bool:
    r"""Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    try:
        import importlib.util
        spec = importlib.util.find_spec(name)
        return spec is not None
    except ImportError:
        return False

TEST_SCIPY = _check_module_exists('scipy')

def skipIfNoLapack(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not msadapter._C.has_lapack:
            raise unittest.SkipTest('PyTorch compiled without Lapack')
        else:
            fn(*args, **kwargs)
    return wrapper

def wait_for_process(p, timeout=None):
    try:
        return p.wait(timeout=timeout)
    except KeyboardInterrupt:
        # Give `p` a chance to handle KeyboardInterrupt. Without this,
        # `pytest` can't print errors it collected so far upon KeyboardInterrupt.
        exit_status = p.wait(timeout=5)
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
            raise
    except subprocess.TimeoutExpired:
        # send SIGINT to give pytest a chance to make xml
        p.send_signal(signal.SIGINT)
        exit_status = None
        try:
            exit_status = p.wait(timeout=5)
        # try to handle the case where p.wait(timeout=5) times out as well as
        # otherwise the wait() call in the finally block can potentially hang
        except subprocess.TimeoutExpired:
            pass
        if exit_status is not None:
            return exit_status
        else:
            p.kill()
        raise
    except:  # noqa: B001,E722, copied from python core library
        p.kill()
        raise
    finally:
        # Always call p.wait() to ensure exit
        p.wait()

def shell(command, cwd=None, env=None, stdout=None, stderr=None, timeout=None):
    sys.stdout.flush()
    sys.stderr.flush()
    # The following cool snippet is copied from Py3 core library subprocess.call
    # only the with
    #   1. `except KeyboardInterrupt` block added for SIGINT handling.
    #   2. In Py2, subprocess.Popen doesn't return a context manager, so we do
    #      `p.wait()` in a `final` block for the code to be portable.
    #
    # https://github.com/python/cpython/blob/71b6c1af727fbe13525fb734568057d78cea33f3/Lib/subprocess.py#L309-L323
    assert not isinstance(command, str), "Command to shell should be a list or tuple of tokens"
    p = subprocess.Popen(command, universal_newlines=True, cwd=cwd, env=env, stdout=stdout, stderr=stderr)
    return wait_for_process(p, timeout=timeout)
