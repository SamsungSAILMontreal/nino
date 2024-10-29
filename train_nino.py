# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/pytorch/examples/blob/main/mnist/main.py

"""
Example usage (train the NiNo model and save it to ./checkpoints/nino_seed0.pt):

    export HF_HOME=/path/to/hf_cache
    python train_nino.py

See more examples in the README.md file.

"""

import argparse
import os
import os.path
import shutil
import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from torch.utils.data import DataLoader
from dataset import SGDDataset, collate_graphs_fn, worker_init_fn
from optim import NiNoModel
from utils import set_seed, mem, get_env_args


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training of the NiNo model')
    parser.add_argument('--data_dir', type=str, default=os.environ['HF_HOME'],
                        help='directory for the SamsungSAILMontreal/nino_metatrain dataset with training trajectories '
                             '(default: $HF_HOME)')
    parser.add_argument('--ctx', type=int, default=5,
                        help='number of parameter states in the model input')
    parser.add_argument('--lpe', type=int, default=8,
                        help='number of laplacian eigenvectors for positional encoding')
    parser.add_argument('--wte_pos_enc', action='store_true', default=False,
                        help='use positional encoding for the word token embeddings')
    parser.add_argument('--seq_len', type=int, default=40,
                        help='max sequence length for DMS')
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='learning rate')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        help='lr scheduler')
    parser.add_argument('--wd', type=float, default=1e-2,
                        help='weight decay')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of parameter trajectories sampled in each batch')
    parser.add_argument('--max_train_steps', type=int, default=20000,
                        help='maximum number of iterations to train')
    parser.add_argument('--grad_clip', type=float, default=1, help='grad clip')
    parser.add_argument('--no_amp', action='store_true', default=False,
                        help='turn off automatic mixed precision, by default it is on')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of workers for data loader')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed')
    args = parser.parse_known_args()[0]
    parser.add_argument('--save_path', type=str, default='./checkpoints/nino_seed{}.pt'.format(args.seed),
                        help='directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='device to use for training')
    parser.add_argument('--verbose', type=int, default=1,
                        help='verbosity level')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args = get_env_args(args)
    return args


def main():

    args = parse_args()

    dset = SGDDataset(root=args.data_dir,  # use cache_dir for the dataset where huggingface will download the data
                      ctx=args.ctx,
                      step=200,  # can be larger, but for our lm1b checkpoints cannot be smaller
                      lpe=args.lpe,
                      seq_len=args.seq_len,
                      verbose=args.verbose)
    train_loader = DataLoader(dset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_graphs_fn,
                              worker_init_fn=worker_init_fn)

    set_seed(args.seed)
    model_args = {'ctx': args.ctx,
                  'lpe': args.lpe,
                  'seq_len': args.seq_len,
                  'max_feat_size': dset.max_feat_size,
                  'wte_pos_enc': args.wte_pos_enc
                  }
    model = NiNoModel(**model_args)
    if args.verbose:
        print('\nNiNo:', model)

    output_dir = os.path.dirname(args.save_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    completed_steps = 0
    if os.path.exists(args.save_path):
        try:
            print('loading NiNo checkpoint from %s' % args.save_path)
            state_dict = torch.load(args.save_path, map_location=args.device)
            if 'state_dict' in state_dict:
                completed_steps = state_dict['step']
                state_dict = state_dict['state_dict']
            result = model.load_state_dict(state_dict)
            print('NiNo with {} params loaded from step {}, ckpt file {}: {}'.format(
                sum([p.numel() for p in model.parameters()]),
                completed_steps,
                args.save_path,
                result))
            set_seed(int(datetime.now().timestamp()))  # seed to make batches different and avoid recurring nan loss
        except Exception as e:
            print('error loading checkpoint %s' % args.save_path, e)
            raise

    model.train().to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    assert len(train_loader) >= args.max_train_steps, (f'only {len(train_loader)} batches for training, '
                                                       f'see __len__() in SGDDataset to increase this number')
    if completed_steps >= args.max_train_steps:
        print(f'the model is already trained for {completed_steps} iterations, exiting...', flush=True)
        exit(0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                     T_max=int(args.max_train_steps),
                                                     eta_min=1e-6)
    for t in range(completed_steps):
        scheduler.step()

    if not args.no_amp:
        scaler = torch.cuda.amp.GradScaler()

    losses = []
    checkpoint = {}
    start_t = time.time()
    print('\nTraining NiNo with {} params for {} steps'.format(
        sum([p.numel() for p in model.parameters()]), args.max_train_steps), flush=True)
    for t, graphs_input in enumerate(train_loader):
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=not args.no_amp):
            graphs_input = graphs_input.to(args.device)
            graphs_output = model(graphs_input)
            loss = F.l1_loss(graphs_output.edge_attr[graphs_input.y_mask],
                             graphs_input.y[graphs_input.y_mask])
            if torch.isnan(loss):
                if 'step' in checkpoint and completed_steps - checkpoint['step'] < 50:
                    # e.g. completed_steps=445 and checkpoint['step']=400
                    # rollback to the last checkpoint
                    if os.path.exists(args.save_path) and os.path.exists(args.save_path + '.bak'):
                        print('restoring the checkpoint before step {}'.format(checkpoint['step']), flush=True)
                        shutil.copyfile(args.save_path + '.bak', args.save_path)  # e.g. step 400 to step 200
                raise ValueError('NaN loss ({}) at step {}'.format(loss, completed_steps + 1))

            if not args.no_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if not args.no_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

        losses.append(loss.item())

        if scheduler is not None:
            scheduler.step()

        if (completed_steps + 1) % args.log_interval == 0 or completed_steps >= args.max_train_steps - 1:

            print('step={}/{}, loss={:.4f}, avg loss={:.4f}, '
                  'lr={:.4e} \t {:.4f} sec/step, g/r={:.2f}/{:.2f}GB'.format(
                completed_steps + 1,
                args.max_train_steps,
                losses[-1],
                np.mean(losses[-1000:]),
                optimizer.param_groups[0]['lr'],
                (time.time() - start_t) / len(losses),
                mem(args.device),
                mem('cpu')),
                flush=True)

        if (completed_steps + 1) % 200 == 0 or completed_steps >= args.max_train_steps - 1:
            try:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'step': completed_steps + 1,
                    'model_args': model_args,
                    'config': args,
                    'losses': losses,
                }
                if os.path.exists(args.save_path):
                    shutil.copyfile(args.save_path, args.save_path + '.bak')  # backup the previous checkpoint (e.g. step 200)
                torch.save(checkpoint, args.save_path)  # save the new checkpoint (e.g. step 400)
                print('saving the checkpoint at step {} done to {}'.format(completed_steps + 1, args.save_path),
                      flush=True)
            except Exception as e:
                print('error in saving the checkpoint', e, flush=True)

        completed_steps += 1
        if completed_steps >= args.max_train_steps:
            break

    print('done at %s' % str(time.strftime('%Y%m%d-%H%M%S')), flush=True)


if __name__ == '__main__':
    main()
    print('Done!')
