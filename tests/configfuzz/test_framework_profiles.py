from __future__ import annotations

from configfuzz.framework_profiles import get_framework_profile, list_framework_profiles


def test_requested_framework_profiles_are_available() -> None:
    profiles = {profile.key: profile for profile in list_framework_profiles()}

    assert set(profiles) == {
        "pytorch-cuda",
        "deepspeed",
        "megatron-core",
        "transformers-accelerate",
    }
    assert profiles["pytorch-cuda"].accelerator == "NVIDIA GPU"
    assert "group_size" in profiles["pytorch-cuda"].parameters
    assert "stage" in profiles["deepspeed"].parameters
    assert "tensor_model_parallel_size" in profiles["megatron-core"].parameters
    assert "mixed_precision" in profiles["transformers-accelerate"].parameters


def test_framework_profile_aliases_resolve() -> None:
    assert get_framework_profile("pytorch").key == "pytorch-cuda"
    assert get_framework_profile("megatron").key == "megatron-core"
    assert get_framework_profile("accelerate").key == "transformers-accelerate"
