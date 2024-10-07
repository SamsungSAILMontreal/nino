# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/pytorch/examples/blob/main/mnist/main.py

"""
Example usage:

    python train_vision.py --task C10-32 --nino_ckpt checkpoints/nino.pt

See more examples in the README.md file.

"""

import argparse
import os.path

import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from optim import NiNo
from utils import set_seed, Net, TASKS, mem, get_env_args


def test(model, data, target, verbose=0):
    model.eval()
    with torch.no_grad():
        output = model(data)
        test_loss = F.cross_entropy(output, target).item()
        correct = torch.sum(torch.argmax(output, -1).eq(target)).item()
        acc = correct / len(data) * 100

    if verbose > 2:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data), acc))
    return {'loss': test_loss, 'acc': acc}


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training with predicting future parameters using NiNo')
    parser.add_argument('--nino_ckpt', type=str, default=None)
    parser.add_argument('--task', type=str, default='FM-16', help='see utils/vision.py for all the tasks')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_train_steps', type=int, default=10000,
                        help='maximum number of iterations to train, early stopping when target is reached')
    parser.add_argument('--period', type=int, default=1000,
                        help='number of base optimizer steps after which to apply NiNo')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--checkpointing_steps', type=int, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    args = get_env_args(args)
    return args


def main():

    args = parse_args()
    device = args.device
    try:
        task = TASKS[args.task]
    except KeyError:
        raise ValueError(f"Task {args.task} not found in {list(TASKS.keys())}")
    print('task', task)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*task['norm'])
        ])

    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': args.num_workers}
    test_kwargs = dict(train_kwargs, shuffle=False, batch_size=10000)
    train_data = eval(f"datasets.{task['dataset']}('../data', train=True, download=True, transform=transform)")

    # in our experiments, we reserved a small subset of the training set for validation, keep the same for consistency
    n_all = len(train_data.targets)
    idx_train = torch.arange(n_all - n_all // 12)
    train_data.data = train_data.data[idx_train]
    train_data.targets = [train_data.targets[i] for i in idx_train]

    set_seed(args.seed)
    generator = torch.Generator()
    train_kwargs['generator'] = generator
    train_loader = torch.utils.data.DataLoader(train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(
        eval(f"datasets.{task['dataset']}('../data', train=False, download=True, transform=transform)"), **test_kwargs)

    # preload the test data to avoid overheads of loading them every time for evaluation
    data_eval, target_eval = next(iter(test_loader))
    data_eval, target_eval = data_eval.to(device, non_blocking=True), target_eval.to(device, non_blocking=True)
    
    set_seed(args.seed)  # set the seed again to make initial weights easily reproducible
    model = Net(**task['net_args']).to(device)
    print(model,
          'params', sum({p.data_ptr(): p.numel() for p in model.parameters()}.values()),
          'total param norm',
          torch.norm(torch.stack([torch.norm(p.data, 2) for p in model.parameters()]), 2).item())

    lr = args.lr if args.lr is not None else task['lr']
    # create a NiNo-based optimizer with AdamW as a base optimizer
    optimizer = NiNo(base_opt=optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd),
                     ckpt=args.nino_ckpt,
                     model=model,
                     period=args.period,
                     max_train_steps=args.max_train_steps,
                     verbose=args.verbose)

    def save(step_idx=None):
        if args.output_dir not in [None, '', 'None', 'none']:
            if not os.path.isdir(args.output_dir):
                os.mkdir(args.output_dir)
            checkpoint_path = os.path.join(args.output_dir,
                                           f'step_{step_idx}.pt' if step_idx else 'ckpt.pt')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.base_opt.state_dict(),  # can save the optimizer state with the history of the past states
                'epoch': epoch,
                'step': step,
                'completed_steps': optimizer.step_idx,
                'model_args': task['net_args'],
                'args': args},
                checkpoint_path)
            print(f'Model and optimizer saved to {checkpoint_path} at '
                  f'epoch={epoch}, '
                  f'step={step}, '
                  f'completed_steps={optimizer.step_idx}', flush=True)

    starting_epoch = 0
    resume_step = 0
    if args.resume_from_checkpoint:
        if not os.path.isfile(args.resume_from_checkpoint):
            print(f"\nWARNING: Resume path {args.resume_from_checkpoint} not found")
        else:
            state_dict = torch.load(args.resume_from_checkpoint, map_location=device)
            model.load_state_dict(state_dict['model'])
            optimizer.base_opt.load_state_dict(state_dict['optimizer'])
            optimizer.step_idx = state_dict['completed_steps']
            starting_epoch = state_dict['epoch']
            resume_step = state_dict['step'] + 1
            if resume_step == len(train_loader):
                starting_epoch += 1
                resume_step = 0
            print(f'Model and optimizer loaded from {args.resume_from_checkpoint}, '
                  f'starting_epoch={starting_epoch}, '
                  f'resume_step={resume_step}, '
                  f'completed_steps={optimizer.step_idx}')
            scores = test(model, data_eval, target_eval, verbose=args.verbose)
            if scores['acc'] >= task['target']:
                print("\nModel already reached target of {:.2f}%>={:.2f}%. Exiting...".format(
                    scores['acc'], task["target"]))
                return

    epochs = int(np.ceil(args.max_train_steps / len(train_loader)))
    losses = []
    start_time = time.time()
    done = False
    print(f'\nTraining {args.task} with {len(train_loader)} batches per epoch for {epochs} epochs')
    for epoch in range(starting_epoch, epochs):
        set_seed(args.seed + epoch)  # set the seed again to make batches the same for nino and adam
        generator.manual_seed(args.seed + epoch)

        for step, (data, target) in enumerate(train_loader, start=resume_step):
            if step >= len(train_loader) or optimizer.step_idx >= args.max_train_steps:
                break

            model.train()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if optimizer.need_grads:
                loss = F.cross_entropy(model(data), target)
                loss.backward()  # only compute gradients for the base optimizer
                closure = None
                losses.append(loss.item())
            else:
                def closure():
                    # eval the loss after the NiNo step to see how it affects the training
                    with torch.no_grad():
                        return F.cross_entropy(model(data), target)

            loss_ = optimizer.step(closure)  # base_opt step or nowcast params every args.period steps using NiNo
            optimizer.zero_grad()
            if loss_ is not None:
                losses.append(loss_.item())

            scores = test(model, data_eval, target_eval, verbose=args.verbose)

            if optimizer.step_idx % args.log_interval == 0:
                print('Train {:04d}/{}: \tTrain loss: {:.4f} \tVal loss: {:.4f} \tVal acc: {:.2f}% '
                      '\t(sec/b={:.3f}, {}={:.3f}G)'.format(
                    optimizer.step_idx,
                    args.max_train_steps, losses[-1], scores['loss'], scores['acc'],
                    (time.time() - start_time) / optimizer.step_idx, device, mem(device)))

            if args.checkpointing_steps is not None and optimizer.step_idx % args.checkpointing_steps == 0:
                save(optimizer.step_idx)  # save the model every args.checkpointing_steps steps

            if scores['acc'] >= task['target']:
                print('\nReached target accuracy of {:.2f}%>={:.2f}% in {} steps '
                      '({:.4f} seconds)'.format(scores['acc'],
                                                task["target"],
                                                optimizer.step_idx,
                                                time.time() - start_time))
                done = True
            if optimizer.step_idx >= args.max_train_steps:
                done = True
            if done:
                break
        resume_step = 0  # reset the start step for the next epoch
        if done:
            break
    save(optimizer.step_idx)  # save the final model

if __name__ == '__main__':
    main()
    print('Done!')
    