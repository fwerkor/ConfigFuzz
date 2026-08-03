def require_divisible(value, divisor):
    if value % divisor != 0:
        raise ValueError("not divisible")
