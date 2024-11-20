# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/pytorch/examples/blob/main/mnist/main.py

"""
Example usage at step 1000 (see README.md for a complete example):

    python nino_step.py --ckpt_path model_checkpoints/step_1000.pt --save_path model_checkpoints/step_1000.pt \
     --period 1000 --max_train_steps 10000 --nino_ckpt checkpoints/nino.pt

To avoid GPU OOM for larger models memory, use --nino_mp_device cpu, but this will be much slower and require more RAM.

"""

import argparse
import os
import torch
from optim import NiNo
from utils import Net, get_env_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch implementation of a NiNo step given a history of checkpoints')
    parser.add_argument('--nino_ckpt', type=str, default='checkpoints/nino.pt')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='checkpoint path for the last step before the NiNo step should be applied '
                             '(e.g. model_checkpoints/ckpt_step1000.pt)')
    parser.add_argument('--save_path', type=str, default=None,
                        help='path to save the model after the NiNo step (e.g. model_checkpoints/ckpt.pt)')
    parser.add_argument('--max_train_steps', type=int, default=10000,
                        help='maximum number of iterations to train (used to compute the prediction future horizon k)')
    parser.add_argument('--period', type=int, default=1000,
                        help='number of base opt steps after which to apply NiNo')
    parser.add_argument('--k_decay', type=float, default=2.,
                        help='power of the decay for the future horizon k (the higher, the faster decay).')
    parser.add_argument('--upd_scale', type=float, default=1.,
                        help='scale of the predicted delta.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--nino_device', type=str, default=None,
                        help='NiNo device for parameter update prediction.')
    parser.add_argument('--nino_mp_device', type=str, default=None,
                        help="NiNo's message passing device for parameter update prediction.")
    parser.add_argument('--nino_chunk_size', type=float, default=1e+6,
                        help='number of parameters to process in parallel using NiNo, to trade off time and memory.')
    parser.add_argument('--hf_login', type=str, default=None,
                        help='Hugging Face token for downloading the model/config.'
    )
    parser.add_argument('--verbose', type=int, default=1)
    args = get_env_args(parser.parse_args())
    device = args.device

    # load checkpoints from the list of paths and add them to the opt states
    assert args.ckpt_path is not None, 'No checkpoint paths provided'
    assert args.save_path is not None, 'No save path provided'
    if not os.path.isdir(os.path.dirname(args.save_path)):
        os.mkdir(os.path.dirname(args.save_path))

    nino_device = device if args.nino_device is None else args.nino_device
    opt = NiNo(base_opt=None,
               ckpt=args.nino_ckpt,
               model=None,
               period=args.period,
               max_train_steps=args.max_train_steps,
               nino_device=nino_device,
               message_passing_device=nino_device if args.nino_mp_device is None else args.nino_mp_device,
               verbose=args.verbose,
               p=args.k_decay,
               upd_scale=args.upd_scale,
               chunk_size=int(args.nino_chunk_size))
    ckpt_freq = opt.period // opt.ctx
    # assume args.ckpt_path is some_path/step_{step}.pt or, for train_lm, some_path/step_{step}
    # extract the step number from the path
    last_step = int(args.ckpt_path.split('step')[-1].split('_')[-1].split('.pt')[0])
    done = False
    is_accelerate = os.path.isdir(args.ckpt_path)
    ckpt_name_last = f'step_{last_step}' + ('' if is_accelerate else '.pt')
    for step in range(last_step - ckpt_freq * (opt.ctx - 1), last_step + 1, ckpt_freq):
        ckpt_name = ckpt_name_last.replace(f'step_{last_step}', f'step_{step}')
        ckpt_path = args.ckpt_path.replace(ckpt_name_last, ckpt_name)
        print(f'loading checkpoint {ckpt_path}')

        assert ckpt_path.endswith(ckpt_name) and os.path.exists(ckpt_path), \
            f'ckpt_path={ckpt_path} is invalid/missing for step {step}'

        if opt._model is None:
            if is_accelerate:
                # import huggingface stuff
                from transformers import AutoModelForCausalLM, AutoConfig
                from huggingface_hub import login
                from accelerate import Accelerator, load_checkpoint_in_model
                if args.hf_login:
                    login(token=args.hf_login, add_to_git_credential=True)
                config_path = ckpt_path.replace(ckpt_name, 'config.json')
                model = AutoModelForCausalLM.from_config(
                    AutoConfig.from_pretrained(config_path)).to(nino_device)
                accelerator = Accelerator()
                model = accelerator.prepare(model)
            else:
                state_dict = torch.load(ckpt_path, map_location=nino_device)
                model_args = state_dict['model_args']
                model = Net(**model_args).to(nino_device)

            print(model,
                  'params', sum({p.data_ptr(): p.numel() for p in model.parameters()}.values()),
                  'total param norm',
                  torch.norm(torch.stack([torch.norm(p.data, 2)
                                          for p in model.parameters()]), 2).item())
            graph_feat_path = ckpt_path.replace(ckpt_name, 'graph.pt')
            if os.path.exists(graph_feat_path):
                print('loading cached graph lpe from', graph_feat_path, flush=True)
                lpe = torch.load(graph_feat_path)
                print('loaded graph lpe', lpe.shape)
            else:
                lpe = None
            opt.set_model(model, lpe=lpe)  # construct the neural graph
            if not os.path.exists(graph_feat_path) and hasattr(opt.graph.pyg_graph, 'pos') and opt.graph.pyg_graph.pos is not None:
                print(f'saving graph lpe {opt.graph.pyg_graph.pos.shape} to', graph_feat_path, flush=True)
                torch.save(opt.graph.pyg_graph.pos, graph_feat_path)

        # load the model state and add it to the list of states
        if is_accelerate:
            load_checkpoint_in_model(opt._model, ckpt_path)
        else:
            opt._model.load_state_dict(state_dict['model'])

        if len(opt.states) == opt.ctx - 1:
            opt.step_idx = step - 1  # -1 to properly compute k
            opt.step(nino_fw_device=device if opt.meta_model.is_mlp else None)
            print('saving model to', args.save_path)
            if is_accelerate:
                accelerator.save_state(args.save_path)
            else:
                state_dict['model'] = opt._model.state_dict()
                state_dict['completed_steps'] = opt.step_idx + 1  # restore the total steps
                torch.save(state_dict, args.save_path)

            print('Model and opt saved to', args.save_path)
            print('Done!')
            done = True
            break
        else:
            opt.states.append(torch.cat([p.data.view(-1).to(opt.nino_device) for p in opt._model.parameters()]))

    if not done:
        raise ValueError(f'The NiNo step was not performed, '
                         f'expected {opt.ctx} but got {len(opt.states)}')
