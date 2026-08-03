import glob
import os
import argparse
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))

import torch

from mindspore.ops import composite as C
from mindspore.common.api import _pynative_executor
from mindspeed_llm import megatron_adaptor

import megatron.core.parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.module import MegatronModule
from megatron.legacy.model.language_model import Embedding
from megatron.training.initialize import initialize_megatron
from megatron.training import get_args

from utils_st import save_output_data, clean_args

seed = 2024
np.random.seed(seed)
BASE_DIR = os.path.split(os.path.realpath(__file__))[0]


class EmbeddingNet(MegatronModule):
    """ test net """

    def __init__(self, config, compute_dtype):
        super().__init__(config=config)
        self.embedding = Embedding(
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            max_sequence_length=config.seq_length,
            embedding_dropout_prob=0.0,
            config=config)
        self.compute_dtype = compute_dtype

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        """ megatron net forward """
        output = self.embedding(input_ids, position_ids, tokentype_ids)
        output = output.to(self.compute_dtype)
        loss = output.float().abs().mean()
        return loss, output


def run_embedding_test(data_dir, ckpt_dir, save_output_dir, layout, run_mode):
    """ test embedding """
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

    if args.bf16:
        compute_dtype = torch.bfloat16
    elif args.fp16:
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    # init model
    network = EmbeddingNet(transformer_config, compute_dtype).cuda()
    network.train(True)

    # get data and ckpt
    data_dict, ckpt_dict = load_ckpt_and_data(args.micro_batch_size,
                                              args.seq_length,
                                              args.hidden_size,
                                              compute_dtype,
                                              torch.int32,
                                              layout,
                                              data_dir,
                                              ckpt_dir,
                                              network)

    # load ckpt
    network.load_state_dict(ckpt_dict)

    # no grad with dtype int32, need to check
    print(f"all of inputs for model: {data_dict.keys()}")
    data_dict['input_ids'] = data_dict['input_ids'].cuda()
    data_dict['position_ids'] = data_dict['position_ids'].cuda()

    # run model
    loss, output = network(**data_dict)
    loss.backward()

    # get grad position
    local_loss = loss.clone()
    x = data_dict['input_ids']
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
        network.float()
        for name, param in network.named_parameters():
            # cur_dw = param.grad.astype(torch.float32)
            cur_dw = param.grad
            # print(f"cur_dw dtype: {cur_dw.dtype}", flush=True)
            torch.distributed.all_reduce(cur_dw, group=ps.get_data_parallel_group())
            cur_dw /= ps.get_data_parallel_world_size()
            param.grad = cur_dw

    tp_rank = ps.get_tensor_model_parallel_rank() if \
        ps.get_tensor_model_parallel_world_size() > 1 else 0
    dp_rank = ps.get_data_parallel_rank() if \
        ps.get_data_parallel_world_size() > 1 else 0

    if run_mode == 'test_megatron':
        forward_file_name = 'megatron_forward'
        backward_file_name = 'megatron_backward'
    else:
        forward_file_name = 'mindspore_forward'
        backward_file_name = 'mindspore_backward'

    # save forward
    save_output_data(output.float().cpu().detach().numpy(),
                     os.path.join(save_output_dir, forward_file_name),
                     'output',
                     f'dp{dp_rank}_tp{tp_rank}')
    # save backward
    if dout is not None:
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
                     layout='SBH',
                     save_path='data/alone/random_data/'):
    """ save random data """

    dp_rank = ps.get_data_parallel_rank() if ps.get_data_parallel_world_size() > 1 else 0
    tp_rank = ps.get_tensor_model_parallel_rank() if \
        ps.get_tensor_model_parallel_world_size() > 1 else 0

    if layout == 'BSH':
        shape = (batch_size, seq_length)
    elif layout == 'SBH':
        shape = (seq_length, batch_size)
    else:
        raise NotImplementedError

    if tp_rank == 0:
        print(f"saveing data dp rank {dp_rank}")
        input_ids = np.random.randint(0, 100, size=shape).astype(np.int32) + (dp_rank + 1)
        position_ids = np.random.randint(0, 100, size=shape).astype(np.int32) + (dp_rank + 1)
        input_ids_save_name = save_path + f'input_ids_dp{dp_rank}.npy'
        position_ids_save_name = save_path + f'position_ids_dp{dp_rank}.npy'
        np.save(input_ids_save_name, input_ids)
        np.save(position_ids_save_name, position_ids)
    torch.distributed.barrier()


def load_ckpt_and_data(batch_size,
                       seq_length,
                       hidden_size,
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
        print("save random data !")
        torch.distributed.barrier()
        save_random_data(batch_size, seq_length, hidden_size, layout, data_dir)

    # loading data
    data_dict = {}
    data_npy_files = glob.glob(os.path.join(data_dir, f"*_dp{dp_rank}.npy"))
    for cur_npy_file in data_npy_files:
        cur_data = torch.tensor(np.load(cur_npy_file), dtype=compute_dtype)
        cur_name = cur_npy_file.split('/')[-1].replace(f"_dp{dp_rank}.npy", "")
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

    # run test
    run_embedding_test(args.data_dir, args.ckpt_dir, args.output_dir, args.layout, args.run_mode)
