import os
import numpy as np
import glob
import sys
import re
from pathlib import Path
num_npu = 8


def parse_memory_file(fname):
    p_memory = r'\| \d.*\| \d+.*\|.*\| (\d+).*\|'
    with open(fname, 'r') as f:
        context = f.read().split('\n')
    try:
        mems = []
        for l in context:
            m = re.match(p_memory, l)
            if m:
                mems.append(int(m.group(1)))
        max_mem = max(mems)
    except:
        max_mem = None
    return max_mem / 1024


def parse_script(file):
    with open(file, 'r') as f:
        context = f.read().split('\n')
    p_gbs = r'.*global-batch-size (\d*).*'
    p_len = r'.*seq-length (\d*).*'
    gbs, len = None, None
    for l in context:
        match = re.match(p_gbs, l)
        if match:
            gbs = match.group(1)
        match = re.match(p_len, l)
        if match:
            len = match.group(1)
    return gbs, len


def parse_log_file(file):
    it_pattern = (r'.*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] '
                  r'iteration\s*(\d*)\/.*lm loss: ([\d\.]*).*grad norm: ([\d\.]*).*')
    with open(file, 'r') as f:
        context = f.read().split('\n')
    data = {}
    for l in context:
        match = re.match(it_pattern, l)
        if match:
            data[int(match.group(2))] = match.groups()
    return data


def parse_log_file_mm(file):
    it_pattern = (r'.*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\] '
                  r'iteration\s*(\d*)\/.*loss: ([\d\.]*)')
    with open(file, 'r') as f:
        context = f.read().split('\n')
    data = {}
    for l in context:
        match = re.match(it_pattern, l)
        if match:
            data[int(match.group(2))] = match.groups()
    return data


def clean_args():
    """ clean args """
    option_to_remove = ('--run_mode',
                        '--data_dir',
                        '--ckpt_dir',
                        '--output_dir',
                        '--layout')
    # Process and remove the option from sys.argv
    # Start at 1 to skip the script name
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in option_to_remove:
            # Remove the option and its value
            if i + 1 < len(sys.argv):
                del sys.argv[i:i+2]
            else:
                del sys.argv[i]
        else:
            i += 1


def init_env(mode='megatron'):
    return
    msadapter_path = Path.cwd().parents[3]
    msa_thirdparty_path = msadapter_path / 'msa_thirdparty'
    python_path_ori = os.getenv('PYTHONPATH').replace(msadapter_path.__str__(), '').replace(
        msa_thirdparty_path.__str__(), '')
    python_path_msadapter = f"{msadapter_path}:{msa_thirdparty_path}:{python_path_ori}"

    if mode == 'megatron':
        os.environ['PYTHONPATH'] = python_path_ori
    elif mode == 'mindspore':
        os.environ['PYTHONPATH'] = python_path_msadapter
    print('## PYTHONPATH: ', os.getenv('PYTHONPATH'))


def count_unequal_element(data_expected, data_our, rtol, atol):
    count = 0
    data_len = len(data_expected)
    for i in range(data_len):
        a = data_expected[i]
        b = data_our[i]
        if abs(a - b) > (atol + rtol * abs(b)):
            print("flatten diff index:", i, " expect: ", a, " our:", b, "a/b:", a / b)
        count = count + 1

    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_our)
    count = np.count_nonzero(np.less_equal(error, atol + np.abs(data_our) * rtol))
    print(f"Diff count: {total_count - count}")


def allclose_assert(data_expected, data_our, rtol, atol, print_error_point):
    if np.allclose(data_expected, data_our, rtol, atol) == False:
        if print_error_point:
            count_unequal_element(data_expected, data_our, rtol, atol)
        assert False, "=============== Accuracy test fail !!! ==============="
    else:
        assert True


def calculate_error(y_expect, y_pred):
    if y_expect.dtype == np.bool_:
        y_expect = y_expect.astype(np.int32)
        y_pred = y_pred.astype(np.int32)

    aerror = np.abs(y_expect - y_pred)
    aerror = aerror[~np.isnan(aerror)]
    aerror = aerror[~np.isinf(aerror)]
    rerror = aerror / np.maximum(np.abs(y_expect) + 1e-7, np.abs(y_pred) + 1e-7)
    return rerror, aerror


def compare_data(data_expected,
                 data_our,
                 atol=0.001,
                 rtol=0.001,
                 print_error_point=False,
                 save_path='./output/error/'):
    print(f"'atol' is set to {atol}")
    print(f"'rtol' is set to {rtol}")
    print(f"Golden data shape: {data_expected.shape}")
    print(f"Our data shape: {data_our.shape}")
    assert data_expected.shape == data_our.shape
    ori_shape = data_our.shape
    data_expected = data_expected.flatten()
    data_our = data_our.flatten()

    # calculate error
    relative_error, absolute_error = calculate_error(data_expected, data_our)
    print("============================================")
    print(f"Max absolute error: {np.max(absolute_error)}")
    print(f"Mean absolute error: {np.mean(absolute_error)}")
    print(f"Max relative error: {np.max(relative_error)}")
    print(f"Mean relative error: {np.mean(relative_error)}")
    print("============================================")

    # # save error array
    if np.max(absolute_error) > 0.0 or np.max(relative_error) > 0.0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.save(save_path + '/absolute_error.npy', absolute_error.reshape(ori_shape))
        np.save(save_path + '/relative_error.npy', relative_error.reshape(ori_shape))

    # assert data
    allclose_assert(data_expected, data_our, rtol, atol, print_error_point)


def save_output_data(data, save_dir="data/alone/output/test_megatron/", name='temp', rank='0'):
    """ save output data """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"{save_dir} is not exists !")

    # name: output dout dw
    save_path = os.path.join(save_dir, name + '_' + rank + '.npy')
    np.save(save_path, data)


def compare_all_data(data_dir, compare_types=["_forward", "_backward"], atol=0.0, rtol=0.0, print_error_point=False,
                     weight_dict=None):
    megatron_output_dir = os.path.join(data_dir, "megatron")
    mindspore_output_dir = os.path.join(data_dir, "mindspore")

    for test_type in compare_types:
        # loading megatron output npy file
        megatron_output_dict = {}
        megatron_output_files = glob.glob(os.path.join(megatron_output_dir + test_type, "*.npy"))
        for cur_npy_file in megatron_output_files:
            cur_name = os.path.basename(cur_npy_file).replace(".npy", "")
            if weight_dict != None and "_dw" in cur_name:
                for pattern in cur_name.split(".")[1:-1]:
                    if pattern in weight_dict.keys():
                        cur_name = cur_name.replace(pattern, weight_dict[pattern])

            cur_output = np.load(cur_npy_file)
            megatron_output_dict[cur_name] = cur_output

        # loading mindspore output npy file
        mindspore_output_dict = {}
        mindspore_output_files = glob.glob(os.path.join(mindspore_output_dir + test_type, "*.npy"))
        for cur_npy_file in mindspore_output_files:
            cur_name = os.path.basename(cur_npy_file).replace(".npy", "")
            cur_output = np.load(cur_npy_file)
            mindspore_output_dict[cur_name] = cur_output

        assert len(megatron_output_dict) == len(mindspore_output_dict) and len(megatron_output_dict) > 0

        for name, value in megatron_output_dict.items():
            save_name = os.path.join(data_dir, f"error/{name}_")
            compare_data(value, mindspore_output_dict[name], atol, rtol, print_error_point, save_name)

    print("=============== Accuracy test pass !!! ===============")


def compare_all_data_2(data_dir_pt, data_dir_ms, compare_types=["_forward", "_backward"], atol=0.0, rtol=0.0,
                       print_error_point=False, weight_dict=None):
    megatron_output_dir = os.path.join(data_dir_pt, "megatron")
    mindspore_output_dir = os.path.join(data_dir_ms, "mindspore")

    for test_type in compare_types:
        # loading megatron output npy file
        megatron_output_dict = {}
        megatron_output_files = glob.glob(os.path.join(megatron_output_dir + test_type, "*.npy"))
        for cur_npy_file in megatron_output_files:
            cur_name = os.path.basename(cur_npy_file).replace(".npy", "")
            if weight_dict != None and "_dw" in cur_name:
                for pattern in cur_name.split(".")[1:-1]:
                    if pattern in weight_dict.keys():
                        cur_name = cur_name.replace(pattern, weight_dict[pattern])

            cur_output = np.load(cur_npy_file)
            megatron_output_dict[cur_name] = cur_output

        # loading mindspore output npy file
        mindspore_output_dict = {}
        mindspore_output_files = glob.glob(os.path.join(mindspore_output_dir + test_type, "*.npy"))
        for cur_npy_file in mindspore_output_files:
            cur_name = os.path.basename(cur_npy_file).replace(".npy", "")
            cur_output = np.load(cur_npy_file)
            mindspore_output_dict[cur_name] = cur_output

        assert len(megatron_output_dict) == len(mindspore_output_dict) and len(megatron_output_dict) > 0

        for name, value in megatron_output_dict.items():
            save_name = os.path.join(data_dir_ms, f"error/{name}_")
            compare_data(value, mindspore_output_dict[name], atol, rtol, print_error_point, save_name)

    print("=============== Accuracy test pass !!! ===============")
