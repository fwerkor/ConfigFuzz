# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""
Patch for MSA environment to fix torch.serialization.safe_load_file compatibility.

This module patches torch.serialization to add missing safe_load_file function
that is required by MindSpeed's safetensors patch.
"""
import sys

def patch_serialization():
    """
    Patch torch.serialization to add safe_load_file function.
    """
    try:
        # Import torch (which is msadapter in MSA environment)
        import torch

        # Check if safe_load_file already exists
        if not hasattr(torch.serialization, 'safe_load_file'):
            # Import from msadapter.serialization and inject into torch.serialization
            from msadapter.serialization import safe_load_file, safe_save_file
            torch.serialization.safe_load_file = safe_load_file
            torch.serialization.safe_save_file = safe_save_file

        return True
    except Exception as e:
        print(f"[MSA_PATCH] Warning: Failed to patch serialization: {e}")
        return False


# Auto-patch when this module is imported
patch_success = patch_serialization()
