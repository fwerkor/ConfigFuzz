import sys
import os
import shutil
import glob
import argparse  # 仅保留必要导入，删掉subprocess


# 定义清空文件夹的工具函数（无修改）
def clear_folder(folder_path):
    if os.path.exists(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        os.makedirs(folder_path)


# -------------------------- 模型配置映射 --------------------------
MODEL_CONFIGS = {
    'internvl3': {
        # 待变异的json文件
        'original_config_path': '/shared/mm-new/MindSpeed-MM/examples/internvl3/model_8B.json',
        # 变异输出目录
        'mutation_output_dir': '/shared/mm-new/mm_mutation_results/internvl3',
        # PTA脚本
        'pta_script': '/shared/mm-new/MindSpeed-MM/examples/internvl3/finetune_internvl3_8B.sh',
        # MSA脚本
        'msa_script': '/shared/mm-new/MindSpeed-MM/scripts-ms/finetune_internvl3_8B.sh',
        # PTA日志路径
        'pta_log_path': './logs/internvl3/pta',
        # MSA日志路径
        'msa_log_path': './logs/internvl3/msa',
    },
    'opensora': {
        # 待变异的json文件
        'original_config_path': '/shared/mm-new/MindSpeed-MM/examples/opensora1.2/inference_model_102x720x1280.json',
        # 变异输出目录
        'mutation_output_dir': '/shared/mm-new/mm_mutation_results/opensora',
        # PTA脚本
        'pta_script': '/shared/mm-new/MindSpeed-MM/examples/opensora1.2/pretrain_opensora1_2.sh',
        # MSA脚本
        'msa_script': '/shared/mm-new/MindSpeed-MM/scripts-ms/inference_opensora1_2.sh',
        # PTA日志路径
        'pta_log_path': './logs/opensora/pta',
        # MSA日志路径
        'msa_log_path': './logs/opensora/msa',
    },
    'cogvideoX': {
        # 待变异的json文件
        'original_config_path': '/shared/mm-new/MindSpeed-MM/examples/cogvideox/i2v_1.5/model_cogvideox_i2v_1.5.json',
        # 变异输出目录
        'mutation_output_dir': '/shared/mm-new/mm_mutation_results/cogvideoX',
        # PTA脚本
        'pta_script': '/shared/mm-new/MindSpeed-MM/examples/cogvideox/i2v_1.5/pretrain_cogvideox_i2v_1.5.sh',
        # MSA脚本
        'msa_script': '/shared/mm-new/MindSpeed-MM/scripts-ms/pretrain_cogvideox_i2v_1.5.sh',
        # PTA日志路径
        'pta_log_path': './logs/cogvideoX/pta',
        # MSA日志路径
        'msa_log_path': './logs/cogvideoX/msa',
    },
    'qwenvl2.5': {
        # 待变异的json文件
        'original_config_path': '/shared/mm-new/MindSpeed-MM/examples/qwen2.5vl/inference_qwen2_5_vl_7b.json',
        # 变异输出目录
        'mutation_output_dir': '/shared/mm-new/mm_mutation_results/qwenvl2.5',
        # PTA脚本
        'pta_script': '/shared/mm-new/MindSpeed-MM/examples/qwen2.5vl/inference_qwen2_5_vl_7b.sh',
        # MSA脚本
        'msa_script': '/shared/mm-new/MindSpeed-MM/scripts-ms/inference_qwen2_5_vl_7b.sh',
        # PTA日志路径
        'pta_log_path': './logs/qwenvl2.5/pta',
        # MSA日志路径
        'msa_log_path': './logs/qwenvl2.5/msa',
    },
}


# -------------------------- 1. 解析参数 --------------------------
parser = argparse.ArgumentParser(description='模型配置突变与执行脚本')
parser.add_argument('-model', type=str, required=True, choices=list(MODEL_CONFIGS.keys()), help='模型名称: internvl3, opensora, cogvideoX, qwenvl2.5')
parser.add_argument('-mutnm', type=int, default=2, help='每次突变修改的参数个数（默认：2）')
parser.add_argument('-max_valid_mutate_num', type=int, default=100, help='最大有效突变次数（默认：100）')
args = parser.parse_args()

MODEL_NAME = args.model
MODEL_CONFIG = MODEL_CONFIGS[MODEL_NAME]
MUTNM = args.mutnm
MAX_VALID_MUTATE_NUM = args.max_valid_mutate_num

# 从模型配置中获取路径
ORIGINAL_CONFIG_PATH = MODEL_CONFIG['original_config_path']
MUTATION_OUTPUT_DIR = MODEL_CONFIG['mutation_output_dir']
PTA_SSH_PATH = MODEL_CONFIG['pta_script']
MSA_SSH_PATH = MODEL_CONFIG['msa_script']
PTA_LOG_PATH = MODEL_CONFIG['pta_log_path']
MSA_LOG_PATH = MODEL_CONFIG['msa_log_path']

# -------------------------- 2. 文件夹初始化 --------------------------
MM_RESULTS_PATH = MUTATION_OUTPUT_DIR
MSRUN_LOG_PATH = "./msrun_log"
MSA_LOGS_PATH = "./msa_logs"

if os.path.exists(MM_RESULTS_PATH):
    clear_folder(MM_RESULTS_PATH)
else:
    os.makedirs(MM_RESULTS_PATH, exist_ok=True)
if os.path.exists(PTA_LOG_PATH):
    clear_folder(PTA_LOG_PATH)
else:
    os.makedirs(PTA_LOG_PATH, exist_ok=True)
if os.path.exists(MSA_LOG_PATH):
    clear_folder(MSA_LOG_PATH)
else:
    os.makedirs(MSA_LOG_PATH, exist_ok=True)
if os.path.exists(MSA_LOGS_PATH):
    clear_folder(MSA_LOGS_PATH)
else:
    os.makedirs(MSA_LOGS_PATH, exist_ok=True)

# -------------------------- 3. 初始化统计变量 --------------------------
total_valid_mutation_num = 0
total_mutation_num = 0

# -------------------------- 4. 主循环：突变->PTA验证->MSA验证 --------------------------
mutation_exception = None  # 用于传递异常给main.py

while total_valid_mutation_num < MAX_VALID_MUTATE_NUM:
    current_config_path = None
    try:
        # 调用突变函数，传入原始配置路径和突变输出目录
        # 使用conda run在ptaa环境中执行
        import subprocess
        # 创建临时脚本文件来执行突变
        script_content = f"""import sys
sys.path.insert(0, '/shared/mm-new')
from net_mutation.mutate_graph_demo import mutate_json_all
mutate_json_all({MUTNM}, '{ORIGINAL_CONFIG_PATH}', '{MUTATION_OUTPUT_DIR}')
"""
        script_path = '/shared/mm-new/run_mutation_tmp.py'
        with open(script_path, 'w') as f:
            f.write(script_content)
        cmd = f"conda run -n mm-pta bash -c 'cd /shared/mm-new && source ./env_files/ptamm_set.sh && python {script_path}'"
        result = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash',
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            close_fds=True
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        exit_code = result.returncode
        if exit_code != 0:
            raise Exception(f"突变执行失败，退出码：{exit_code}\nstderr: {result.stderr}")
        # 清理临时脚本
        if os.path.exists(script_path):
            os.remove(script_path)
        total_mutation_num += 1

        files = glob.glob(os.path.join(MM_RESULTS_PATH, "*"))
        if not files:
            print("突变未生成配置文件，跳过本轮")
            continue
        files.sort(key=lambda x: os.path.getmtime(x))
        current_config_path = files[-1]

        # PTA验证
        print(f"[DEBUG] 开始执行PTA验证...")
        # PTA脚本需要在MindSpeed-MM目录下执行，使用绝对路径引用env脚本
        pta_workdir = "/shared/mm-new/MindSpeed-MM"
        pta_script_relative = PTA_SSH_PATH.replace("/shared/mm-new/MindSpeed-MM/", "")
        # 使用subprocess获取实际退出码，在容器内执行后获取真实退出码
        cmd_pta = f"docker exec -w {pta_workdir} mindspore_data2 bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate mm-pta && source /shared/mm-new/env_files/ptamm_set.sh && MM_MODEL={current_config_path} bash {pta_script_relative}; exit $(expr $? + 0)'"
        result_pta = subprocess.run(cmd_pta, shell=True, executable='/bin/bash', capture_output=True, text=True, close_fds=True)
        exit_code = result_pta.returncode
        print(f"[DEBUG] PTA执行完成，退出码: {exit_code}")
        # 检查输出中的错误模式
        output_all = (result_pta.stdout or "") + (result_pta.stderr or "")
        error_patterns = ["ERR99999", "KeyError", "FAILED", "failed (exitcode", "Traceback"]
        has_error = any(pattern in output_all for pattern in error_patterns)
        if result_pta.stdout:
            print(result_pta.stdout[-2000:] if len(result_pta.stdout) > 2000 else result_pta.stdout)
        if result_pta.stderr:
            print(result_pta.stderr[-2000:] if len(result_pta.stderr) > 2000 else result_pta.stderr, file=sys.stderr)
        if exit_code != 0 or has_error:
            error_msg = f"PTA命令执行失败，退出码：{exit_code}" if exit_code != 0 else f"PTA输出包含错误模式，退出码：{exit_code}"
            raise Exception(error_msg)

    except Exception as e:
        print(f"PTA执行失败: {e}")
        mutation_exception = e  # 保存异常供main.py判断
        if current_config_path and os.path.exists(current_config_path):
            os.remove(current_config_path) if os.path.isfile(current_config_path) else shutil.rmtree(current_config_path)
        if os.path.exists(PTA_LOG_PATH):
            pta_log_files = glob.glob(os.path.join(PTA_LOG_PATH, "*"))
            if pta_log_files:
                pta_log_files.sort(key=lambda x: os.path.getmtime(x))
                last_pta_log = pta_log_files[-1]
                if os.path.isfile(last_pta_log) or os.path.islink(last_pta_log):
                    os.unlink(last_pta_log)
                elif os.path.isdir(last_pta_log):
                    shutil.rmtree(last_pta_log)
                print(f"已删除PTA_LOG最后文件: {last_pta_log}")
        # 不重新抛出异常，而是继续循环进行下一次突变
        continue

    total_valid_mutation_num += 1
    print("已完成有效突变轮次：", total_valid_mutation_num)
    mutation_exception = None  # 成功则清除异常

    # MSA验证
    try:
        print(f"[DEBUG] 开始执行MSA验证...")
        # MSA脚本需要在lmsv-yd容器中的mm-msa环境执行
        # 工作目录是/shared/mm-new/MindSpeed-MM，脚本在scripts-ms目录下
        msa_workdir = "/shared/mm-new/MindSpeed-MM"
        msa_script_relative = MSA_SSH_PATH.replace("/shared/mm-new/MindSpeed-MM/", "")
        # 使用subprocess获取实际退出码
        cmd_msa = f"docker exec -w {msa_workdir} lmsv-yd bash -c 'source /root/miniconda3/etc/profile.d/conda.sh && conda activate mm-msa && source /data/yd/lm-sv/module_combination_mutation/envs/pta_env.sh && MM_MODEL={current_config_path} bash {msa_script_relative}; exit $(expr $? + 0)'"
        result_msa = subprocess.run(cmd_msa, shell=True, executable='/bin/bash', capture_output=True, text=True, close_fds=True)
        exit_code = result_msa.returncode
        print(f"[DEBUG] MSA执行完成，退出码: {exit_code}")
        # 检查输出中的错误模式
        output_all = (result_msa.stdout or "") + (result_msa.stderr or "")
        error_patterns = ["ERR99999", "KeyError", "FAILED", "failed (exitcode", "Traceback", "msrun: error"]
        has_error = any(pattern in output_all for pattern in error_patterns)
        if result_msa.stdout:
            print(result_msa.stdout[-2000:] if len(result_msa.stdout) > 2000 else result_msa.stdout)
        if result_msa.stderr:
            print(result_msa.stderr[-2000:] if len(result_msa.stderr) > 2000 else result_msa.stderr, file=sys.stderr)
        if exit_code != 0 or has_error:
            error_msg = f"MSA命令执行失败，退出码：{exit_code}" if exit_code != 0 else f"MSA输出包含错误模式，退出码：{exit_code}"
            raise Exception(error_msg)
    except Exception as e:
        print(f"MSA执行失败: {e}")
        mutation_exception = e  # 保存异常
        # MSA失败也继续循环进行下一次突变
        continue

print(f"\n执行完成！")
print(f"总突变次数: {total_mutation_num} | 有效突变次数: {total_valid_mutation_num}")
print(f"最终异常状态: {mutation_exception}")