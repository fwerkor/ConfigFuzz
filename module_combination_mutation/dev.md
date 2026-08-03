models containing a text decoder: deepseekvl2 (deepseek), glm4.1v (glm4v_lm), internvl2 (internllm), internvl2.5 (internllm), internvl3 (internllm), llava1.5 (none), qwen2.5omni (qwen2_5_omni_thinker), qwen2.5vl (qwen2_5_lm), qwen2vl (qwen2lm), qwen3vl (qwen3_lm), videoalign (videoalign_lm)

models containing a text encoder: cogvideox (T5), sora, qwenvl


mindspeed-mm不支持encoder自建网络结构，只支持从已有权重构建

## 结果目录 (Results Directory)

每次运行会在 `results` 目录下创建一个独立文件夹 `run_YYYYMMDD_HHMMSS`，存放该次运行的所有产物和日志，结构如下：
- `results/run_20260214_091530/train.log` - 训练日志
- `results/run_20260214_091530/profile/` - 性能分析输出

**指定自定义位置：**
- 命令行：`--results-dir /path/to/run_dir`
- Shell 脚本：`RESULTS_DIR=/path/to/base ./module_combination_mutate.sh`（每次仍会在该目录下创建 run_* 子文件夹）

```
bash mm_mutate.sh
bash mm_test.sh ./results/mutate_20260311_084708/configs/
bash mm_test.sh ./results/mutate_20260311_084708/configs/ --no-mm-test-optimizer
```

```
bash ../module_combination_mutation/mm_test.sh --config /data/yd/lm-sv/lmsv_rec/output/2026-04-02-09-50-27/iters/iter_2/core_backup/1-pta-mutate/configs/round_0.json --iterations 5 --save-ckpt

/data/yd/lm-sv/module_combination_mutation/results/_module_combine/test_20260402_101651/ckpts/round_0.pt
```