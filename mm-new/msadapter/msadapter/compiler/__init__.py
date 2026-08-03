from functools import wraps

def disable(recursive=False):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator

def is_compiling():
    return False