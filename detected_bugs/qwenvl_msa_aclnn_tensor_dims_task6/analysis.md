# QwenVL2.5 MSA `aclnnInplaceCopyGetWorkspaceSize` Tensor Dimension Limit Bug (Task6 Repro)

> **日期**: 2026-04-29
> **状态**: Task6 运行中稳定复现，与历史记录一致

## 问题描述

在 MSA (MindSpore Adapter) 环境下运行 QwenVL2.5 推理时，`Qwen2VLImageProcessor.__call__` 中的 `image_grid_thw[index].prod()` 因 tensor 维度超过 Ascend 算子限制而崩溃。PTA 环境下正常执行。

## 错误信息

```
RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed, please check!
Ascend Error: AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.
```

## 完整堆栈

```
[rank0]:   File "/shared/mindspeed-mm/MindSpeed-MM/mindspeed_mm/tasks/inference/pipeline/qwen2vl_pipeline.py", line 52, in __call__
[rank0]:     inputs = self.prepare_inputs(prompt=prompt, images=image, videos=video)
[rank0]:   File "/shared/mindspeed-mm/MindSpeed-MM/mindspeed_mm/tasks/inference/pipeline/qwen2vl_pipeline.py", line 102, in prepare_inputs
[rank0]:     inputs = self.image_processor(
[rank0]:   File "/root/anaconda3/envs/msadapter/lib/python3.10/site-packages/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py", line 177, in __call__
[rank0]:     num_image_tokens = image_grid_thw[index].prod() // merge_length
[rank0]: RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed, please check!
[rank0]: Ascend Error: AclNN_Parameter_Error(EZ1001): The self tensor cannot be larger than 8 dimensions.
```

## 触发代码

```python
# transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py:177
num_image_tokens = image_grid_thw[index].prod() // merge_length
```

## 根因分析

MSA 环境下 MindSpore 后端调用 AclNN 算子，限制 tensor 维度 ≤ 8。Qwen2VLImageProcessor 在处理图像时生成的高维 grid tensor 超过此限制。

## 与历史 bug 记录的对比

| 维度 | 历史记录 (2026-04-10) | 本次 Task6 结果 (2026-04-29) |
|------|----------------------|------------------------------|
| 错误信息 | `RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed` | **一致** |
| Ascend 错误码 | `AclNN_Parameter_Error(EZ1001)` | **一致** |
| 触发位置 | `processing_qwen2_5_vl.py:177` | **一致** |
| 状态 | 已知 bug | **再次确认，稳定复现** |

## 环境信息

- **模型**: QwenVL2.5 (推理模式)
- **PTA环境**: mindspeed (正常)
- **MSA环境**: msadapter (崩溃)
- **NPU**: 8 卡 Ascend 910B1
- **Task6 迭代**: iter_1, iter_2 均复现

## Task6 验证结果

| 环境 | 状态 | 指标 |
|------|------|------|
| PTA | 成功 | memory=19360.0MB, time=84991.0ms |
| MSA | 失败 | `RuntimeError: aclnnInplaceCopyGetWorkspaceSize call failed, please check!` |

## 结论

该 bug 在 Task6 多机多轮测试中**稳定复现**，与历史记录完全一致。属于 MSA 环境已知兼容性问题，不影响 Task6 验证流程（已标记为预期失败）。
