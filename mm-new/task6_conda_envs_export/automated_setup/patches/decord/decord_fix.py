import sys
import warnings

# 尝试导入 decord 并修复
try:
    import decord
    # 如果 decord 没有 cpu 属性，添加一个空实现
    if not hasattr(decord, 'cpu'):
        decord.cpu = lambda: None
        warnings.warn("Patched decord.cpu() as empty function")
    # 确保 decord.cpu 可以被调用
    original_init = decord.__init__ if hasattr(decord, '__init__') else None
except Exception as e:
    warnings.warn(f"Failed to patch decord: {e}")
