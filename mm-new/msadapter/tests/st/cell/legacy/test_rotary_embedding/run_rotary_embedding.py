import glob
import os
import argparse
import sys
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir, os.path.pardir))
import torch
import torch_npu

from mindspeed_llm import megatron_adaptor
import megatron.core.parallel_state as ps
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.module import MegatronModule
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
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


class MegatronRotaryEmbeddingNet(MegatronModule):
    """ megatron net """
    def __init__(self, config):
        super().__init__(config=config)
        rotary_dim = config.hidden_size // config.num_attention_heads
        seq_len_interpolation_factor = None
        rotary_base = 10000
        self.rotary_embedding = RotaryEmbedding(
            kv_channels=rotary_dim,
            rotary_percent=config.rotary_percent,
            rotary_interleaved=config.rotary_interleaved,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            rotary_base=rotary_base)

    def forward(self, max_seq_len, offset=0):
        """ megatron net forward """
        output = self.rotary_embedding(max_seq_len, offset)
        return output


def run_rotary_embedding_test(data_dir, ckpt_dir, save_output_dir, layout, run_mode):
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

    # init model
    network = MegatronRotaryEmbeddingNet(args).cuda()
    network.train(True)

    # run model
    output = network(args.seq_length)
    loss = output.float().abs().mean()
    local_loss = loss.clone()
    if ps.get_data_parallel_world_size() > 1:
        # reduce loss
        torch.distributed.all_reduce(loss, group=ps.get_data_parallel_group())
        loss /= ps.get_data_parallel_world_size()

    dp_rank = ps.get_data_parallel_rank() if \
                ps.get_data_parallel_world_size() > 1 else 0

    if run_mode == 'test_megatron':
        forward_file_name = 'megatron_forward'
        backward_file_name = 'megatron_backward'
    else:
        forward_file_name = 'mindspore_forward'
        backward_file_name = 'mindspore_backward'

    # save forward
    save_output_data(output.float().detach().numpy(),
                     os.path.join(save_output_dir, forward_file_name),
                     'output',
                     f'dp{dp_rank}')


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

    run_rotary_embedding_test(args.data_dir, args.ckpt_dir, args.output_dir, args.layout, args.run_mode)