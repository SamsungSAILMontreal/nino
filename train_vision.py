# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/pytorch/examples/blob/main/mnist/main.py

"""
Example usage (NiNo's checkpoint checkpoints/nino.pt is used by default):

    python train_vision.py --task C10-32

"""

import argparse
import os.path

import numpy as np
import platform
import subprocess
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from optim import NiNo
from utils import set_seed, Net, TASKS, mem


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
    parser.add_argument('--nino_ckpt', type=str, default='checkpoints/nino.pt')
    parser.add_argument('--task', type=str, default='FM-16', help='see utils/vision.py for all the tasks')
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_steps', type=int, default=10000,
                        help='maximum number of iterations to train, early stopping when target is reached')
    parser.add_argument('--period', type=int, default=1000,
                        help='number of base opt steps after which to apply NiNo')
    parser.add_argument('--seed', type=int, default=1000, help='random seed')
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log_interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()

    print('\nEnvironment:')
    env = {}
    try:
        # print git commit to ease code reproducibility
        env['git commit'] = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
    except Exception as e:
        env['git commit'] = str(e)

    env['hostname'] = platform.node()
    env['torch'] = torch.__version__
    env['torchvision'] = torchvision.__version__
    env['cuda available'] = torch.cuda.is_available()
    env['cudnn enabled'] = cudnn.enabled
    env['cuda version'] = torch.version.cuda
    env['start time'] = time.strftime('%Y%m%d-%H%M%S')
    for x, y in env.items():
        print('{:20s}: {}'.format(x[:20], y))

    args.env = env
    print('\nScript Arguments:', flush=True)
    args_var = vars(args)
    for x in sorted(args_var.keys()):
        y = args_var[x]
        print('{:20s}: {}'.format(x[:20], y))
    print('\n', flush=True)
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
    train_loader = torch.utils.data.DataLoader(
        train_data, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(
        eval(f"datasets.{task['dataset']}('../data', train=False, download=True, transform=transform)"), **test_kwargs)

    # preload the test data to avoid overheads of loading them every time for evaluation
    for data_eval, target_eval in test_loader:
        data_eval, target_eval = data_eval.to(device, non_blocking=True), target_eval.to(device, non_blocking=True)
        break  # only one big batch is expected

    set_seed(args.seed)  # set the seed again to make initial weights easily reproducible
    model = Net(**task['net_args']).to(device)
    print(model, 'params', sum(p.numel() for p in model.parameters()),
          'total param norm',
          torch.norm(torch.stack([torch.norm(p.data, 2) for p in model.parameters()]), 2).item())

    lr = args.lr if args.lr is not None else task['lr']
    # create a NiNo-based opt with AdamW as a base optimizer
    opt = NiNo(base_opt=optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd),
               ckpt=args.nino_ckpt,
               model=model,
               period=args.period,
               max_steps=args.max_steps,
               verbose=args.verbose)

    epochs = int(np.ceil(args.max_steps / len(train_loader)))
    losses = []

    start_time = time.time()
    done = False
    for epoch in range(epochs):
        set_seed(args.seed + epoch)  # set the seed again to make batches the same for nino and adam
        generator.manual_seed(args.seed + epoch)

        for t, (data, target) in enumerate(train_loader):
            model.train()
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            if opt.need_grads:
                loss = F.cross_entropy(model(data), target)
                loss.backward()  # only compute gradients for the base optimizer
                closure = None
                losses.append(loss.item())
            else:
                def closure():
                    # eval the loss after the NiNo step to see how it affects the training
                    with torch.no_grad():
                        return F.cross_entropy(model(data), target)

            loss_ = opt.step(closure)  # base_opt step or nowcast params every args.period steps using NiNo
            opt.zero_grad()
            if loss_ is not None:
                losses.append(loss_.item())

            scores = test(model, data_eval, target_eval, verbose=args.verbose)

            if opt.step_idx % args.log_interval == 0:
                print('Train {:04d}/{}: \tTrain loss: {:.4f} \tVal loss: {:.4f} \tVal acc: {:.2f}% '
                      '\t(sec/b={:.3f}, {}={:.3f}G)'.format(
                       opt.step_idx, args.max_steps, losses[-1], scores['loss'], scores['acc'],
                       (time.time() - start_time) / opt.step_idx, device, mem(device)))

            if scores['acc'] >= task['target']:
                print('\nReached target accuracy of {:.2f}%>={:.2f}% in {} steps '
                      '({:.4f} seconds)'.format(scores['acc'],
                                                task["target"],
                                                opt.step_idx,
                                                time.time() - start_time))
                done = True
            if opt.step_idx >= args.max_steps:
                done = True
            if done:
                break
        if done:
            break

    if args.save_path not in [None, '', 'None', 'none']:

        if not os.path.isdir(args.save_path):
            os.mkdir(args.save_path)
        save_path = os.path.join(args.save_path, 'ckpt.pt')
        torch.save({
            'model': model.state_dict(),
            'opt': opt.base_opt.state_dict(),  # can save the optimizer state with the history of the past states
            'args': args},
            save_path)
        print('Model and opt saved to', save_path)


if __name__ == '__main__':
    main()
    print('Done!')
    