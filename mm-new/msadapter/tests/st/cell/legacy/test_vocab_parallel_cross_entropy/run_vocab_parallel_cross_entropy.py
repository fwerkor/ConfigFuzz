import glob
import os
import argparse
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))

import torch
import torch_npu

from mindspore.ops import composite as C
from mindspore.common.api import _pynative_executor

from mindspeed_llm import megatron_adaptor
import megatron.core.parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.module import MegatronModule
from megatron.core.tensor_parallel import vocab_parallel_cross_entropy
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from utils_st import save_output_data


seed = 2024
np.random.seed(seed)
BASE_DIR = os.path.split(os.path.realpath(__file__))[0]

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

class VocabParallelCrossEntropyNet(MegatronModule):
    """ megatron net """
    def __init__(self, config):
        super().__init__(config=config)
        self.vocab_parallel_cross_entropy = vocab_parallel_cross_entropy

    def forward(self, vocab_parallel_logits, target, label_smoothing=0.0):
        """ megatron net forward """
        output = self.vocab_parallel_cross_entropy(vocab_parallel_logits, target, label_smoothing)
        loss = output.float().abs().mean()
        return loss, output


def run_vocab_parallel_cross_entropy_test(data_dir, ckpt_dir, save_output_dir, layout='SBH', run_mode='test_megatron'):
    """ test mindspore """
    # init env
    torch.use_deterministic_algorithms(True)
    initialize_megatron()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model_parallel_cuda_manual_seed(seed)

    # get config
    args = get_args()
    transformer_config = core_transformer_config_from_args(args)
    transformer_config.sequence_parallel = args.sequence_parallel
    transformer_config.vocab_size = args.vocab_size
    transformer_config.seq_length = args.seq_length

    # init model
    network = VocabParallelCrossEntropyNet(transformer_config).cuda()
    network.train(True)

    if args.fp16_lm_cross_entropy:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    # get data and ckpt
    data_dict, ckpt_dict = load_ckpt_and_data(args.micro_batch_size,
                                              args.seq_length,
                                              args.hidden_size,
                                              args.vocab_size,
                                              compute_dtype,
                                              compute_dtype,
                                              layout,
                                              data_dir,
                                              ckpt_dir,
                                              network)

     # load ckpt
    network.load_state_dict(ckpt_dict)

    print(f"all of inputs for model: {data_dict.keys()}")
    data_dict['vocab_parallel_logits'] = data_dict['vocab_parallel_logits'].cuda()
    data_dict['target'] = data_dict['target'].cuda()
    ## no grad with dtype int32, need to check
    data_dict['vocab_parallel_logits'] = data_dict['vocab_parallel_logits'].requires_grad_()
    data_dict['vocab_parallel_logits'].retain_grad()

    # run model
    loss, output = network(**data_dict)
    loss.backward()

    # get grad position
    local_loss = loss.clone()
    x = data_dict['vocab_parallel_logits']
    dout = x.grad
    if ps.get_data_parallel_world_size() > 1:
        # reduce loss
        torch.distributed.all_reduce(loss, group=ps.get_data_parallel_group())
        loss /= ps.get_data_parallel_world_size()
        # reduce dout
        if dout is not None:
            torch.distributed.all_reduce(dout, group=ps.get_data_parallel_group())
            dout /= ps.get_data_parallel_world_size()
        # reduce dw
        for name, param in network.named_parameters():
            cur_dw = param.grad
            torch.distributed.all_reduce(cur_dw, group=ps.get_data_parallel_group())
            cur_dw /= ps.get_data_parallel_world_size()
            param.grad = cur_dw

    tp_rank = ps.get_tensor_model_parallel_rank() if \
                ps.get_tensor_model_parallel_world_size() > 1 else 0
    dp_rank = ps.get_data_parallel_rank() if \
                ps.get_data_parallel_world_size() > 1 else 0

    # save forward
    if run_mode == 'test_megatron':
        forward_file_name = 'megatron_forward'
        backward_file_name = 'megatron_backward'
    else:
        forward_file_name = 'mindspore_forward'
        backward_file_name = 'mindspore_backward'
    save_output_data(output.float().cpu().detach().numpy(),
                     os.path.join(save_output_dir, forward_file_name),
                     'output',
                     f'dp{dp_rank}')
    # save backward
    if x.grad is not None:
        save_output_data(dout.float().cpu().numpy(),
                         os.path.join(save_output_dir, backward_file_name),
                        'dout',
                        f'tp{tp_rank}')
    # save dw
    for name, param in network.named_parameters():
        save_output_data(param.grad.float().cpu().numpy(),
                         os.path.join(save_output_dir, backward_file_name),
                         name + '_dw',
                         f'tp{tp_rank}')


def save_random_ckpt(model, save_path="data/alone/random_ckpt/"):
    """ save random ckpt """
    state_dict = model.state_dict()
    tp_rank = ps.get_tensor_model_parallel_rank() if \
                            ps.get_tensor_model_parallel_world_size() > 1 else 0
    dp_rank = ps.get_data_parallel_rank() if ps.get_data_parallel_world_size() > 1 else 0

    if dp_rank == 0:
        for name, value in state_dict.items():
            if value is None:
                continue
            print(f"tp rank{tp_rank}, saving weight:{name}, shape:{value.shape}")
            np_value = np.random.randn(*value.shape).astype(np.float32)
            save_name = save_path + name + f"_tp{tp_rank}.npy"
            np.save(save_name, np_value)
    torch.distributed.barrier()


def save_random_data(batch_size,
                     seq_length,
                     hidden_size,
                     vocab_size,
                     layout='SBH',
                     save_path='data/alone/random_data/'):
    """ save random data """

    dp_rank = ps.get_data_parallel_rank() if ps.get_data_parallel_world_size() > 1 else 0
    tp_rank = ps.get_tensor_model_parallel_rank() if \
                            ps.get_tensor_model_parallel_world_size() > 1 else 0

    if layout == 'BSH':
        shape = (batch_size, seq_length, vocab_size)
        target_shape = (batch_size, seq_length)
    elif layout == 'SBH':
        shape = (seq_length, batch_size, vocab_size)
        target_shape = (seq_length, batch_size)
    else:
        raise NotImplementedError

    if tp_rank == 0:
        vocab_parallel_logits = np.random.randn(*shape).astype(np.float32) * (dp_rank + 1)
        target = np.random.randint(0, 100, size=target_shape).astype(np.int32) + (dp_rank + 1)
        vocab_parallel_logits_save_name = save_path + f'vocab_parallel_logits_dp{dp_rank}.npy'
        target_save_name = save_path + f'target_dp{dp_rank}.npy'
        np.save(vocab_parallel_logits_save_name, vocab_parallel_logits)
        np.save(target_save_name, target)
    torch.distributed.barrier()


def load_ckpt_and_data(batch_size,
                       seq_length,
                       hidden_size,
                       vocab_size,
                       params_dtype,
                       compute_dtype,
                       layout='SBH',
                       data_dir="data/alone/random_data/",
                       ckpt_dir="data/alone/random_ckpt/",
                       model=None):
    """ load ckpt and data. """
    tp_size = ps.get_tensor_model_parallel_world_size()
    dp_size = ps.get_data_parallel_world_size()
    tp_rank = ps.get_tensor_model_parallel_rank() if tp_size > 1 else 0
    dp_rank = ps.get_data_parallel_rank() if dp_size > 1 else 0

    # save random ckpt
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"{ckpt_dir} is not exists !")
    num_files = len(glob.glob(os.path.join(ckpt_dir, "*.npy")))
    if num_files == 0:
        print("save random ckpt !")
        torch.distributed.barrier()
        save_random_ckpt(model, ckpt_dir)

    # save random data
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} is not exists !")
    num_files = len(glob.glob(os.path.join(data_dir, "*.npy")))
    if num_files == 0:
        torch.distributed.barrier()
        save_random_data(batch_size, seq_length, hidden_size, vocab_size, layout, data_dir)

    # loading data
    data_dict = {}
    data_npy_files = glob.glob(os.path.join(data_dir, f"*_dp{dp_rank}.npy"))
    ori_compute_dtype = compute_dtype
    for cur_npy_file in data_npy_files:
        cur_name = cur_npy_file.split('/')[-1].replace(f"_dp{dp_rank}.npy", "")
        if 'target' in cur_name:
            compute_dtype = torch.int64
        else:
            compute_dtype = ori_compute_dtype
        cur_data = torch.tensor(np.load(cur_npy_file), dtype=compute_dtype)
        data_dict[cur_name] = cur_data

    # loading ckpt
    ckpt_dict = {}
    ckpt_npy_files = glob.glob(os.path.join(ckpt_dir, f"*_tp{tp_rank}.npy"))
    for cur_ckpt_npy_file in ckpt_npy_files:
        cur_name = cur_ckpt_npy_file.split('/')[-1].replace(f"_tp{tp_rank}.npy", "")
        cur_ckpt = torch.tensor(np.load(cur_ckpt_npy_file), dtype=params_dtype)
        ckpt_dict[cur_name] = cur_ckpt
    return data_dict, ckpt_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', type=str, default='test_megatron',
                        help="test_megatron / test_mindspore")
    parser.add_argument('--layout', type=str, default='SBH',
                        help="layout")
    parser.add_argument('--data_dir', type=str, default="data/alone/random_data/",
                        help="load data path")
    parser.add_argument('--ckpt_dir', type=str, default="data/alone/random_ckpt/",
                        help="load ckpt path")
    parser.add_argument('--output_dir', type=str, default="data/alone/output/",
                        help="load ckpt path")
    args, _ = parser.parse_known_args()
    clean_args()

    run_vocab_parallel_cross_entropy_test(args.data_dir, args.ckpt_dir, args.output_dir, args.layout, args.run_mode)