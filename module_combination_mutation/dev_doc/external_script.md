# 外部脚本控制整个变异+pta/msa跑测流程

参考实现：lmsv_rec/utils/task/task1.py

在module_combination_mutation/task目录下实现

实现流程：

1. （读取配置阶段）从module_combine_config.yaml读取配置到一个自定义的config类中
    配置应包括五部分：全局配置、mutate配置、pta权重生成配置、pta加载权重跑测配置、msa加载权重跑测配置
2. （变异阶段）执行mm_mutate.sh，设置round为1，产生一个模块组合变体，存放在本脚本设置的路径下
3. （pta权重生成阶段）执行mm_test.sh，指定刚才生成的模块组合配置，参考命令：
    conda activate ptaa
    bash mm_test.sh --config ./results/mutate_20260311_084708/configs/round_0.json --save-ckpt
4. （pta加载权重跑测阶段）执行mm_text.sh，指定模型配置文件与权重文件，开启跑测，参考命令：
    conda activate ptaa
    bash mm_test.sh --config ./results/mutate_20260311_084708/configs/round_0.json --load-ckpt --ckpt ./results/test_20260317_103249/ckpts/mutate_20260311_084708_001_round_0.pt
5. （msa加载权重跑测阶段）暂时省略实现，因为msa环境仍未搭建完成
6. （结束）

注意：

1. conda虚拟环境ptaa的名字使用全局变量，方便后续统一修改
2. 执行子任务的代码使用函数封装独立实现，比如变异、pta权重生成等等
