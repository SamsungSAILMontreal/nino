# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
To see the network architecture in each task, run:

    python utils/vision.py

"""

from itertools import chain
import numpy as np
import torch
import torch.nn as nn


class Net(nn.Module):
    """
    Generic neural network with fully connected or convolutional layers.
    The default arguments will create a FM16 network corresponding to the l2o task from
    https://github.com/mkofinas/neural-graphs.
    """
    def __init__(self,
                 hid=(16, 32, 32),
                 activ=nn.ReLU,
                 conv=True,
                 im_size=28,
                 in_channels=1,
                 pool=False,
                 gap=True,
                 stride=1,
                 num_classes=10,
                 kernel_size=3,
                 bias=True):
        super(Net, self).__init__()
        self.hid = (hid,) if not isinstance(hid, (tuple, list)) else hid
        if conv:
            layer, first_dim, last_dim = (
                nn.Conv2d,
                in_channels,
                self.hid[-1] * int(np.ceil(im_size / (2 ** len(self.hid)))) ** 2,
            )
            layer_args = {"kernel_size": kernel_size, "bias": bias}
        else:
            layer, first_dim, last_dim = nn.Linear, in_channels * im_size**2, self.hid[-1]
            layer_args = {"bias": bias}

        if pool:
            last_dim = self.hid[-1] * 9
            pool_hid = 64
        else:
            if gap:
                last_dim = self.hid[-1]
            pool_hid = last_dim

        def layer_fn(i, h):
            if conv:
                return layer(first_dim if i == 0 else self.hid[i - 1], h,
                             stride=1 if pool else (2 if i == 0 else stride),
                             padding=(0 if i == 0 else 'same') if stride == 1 else 1,
                             **layer_args)
            else:
                return layer(first_dim if i == 0 else self.hid[i - 1], h,
                             **layer_args)

        layers = [

                    ([layer_fn(i, h), activ()] + ([
                        nn.MaxPool2d(2, stride=1 if i == 0 else 2)] if pool else []))
                    for i, h in enumerate(self.hid)
                ] + [(([nn.AdaptiveAvgPool2d(1), nn.Flatten()] if gap else [nn.Flatten()]) if conv else []) +
                     ([nn.Linear(last_dim, pool_hid), activ()] if pool else [])]
        self.fc = nn.Sequential(
            *chain.from_iterable(
                layers
            ),
            nn.Linear(pool_hid, num_classes),
        )

    def forward(self, x):
        if isinstance(self.fc[0], nn.Linear):
            x = x.view(len(x), -1)
        x = self.fc(x)
        return x


mnist_norm = ((0.1307,), (0.3081,))
cifar_norm = ((0.49139968, 0.48215827, 0.44653124), (0.24703233, 0.24348505, 0.26158768))

VISION_TASKS = {

    'debug': {
        'net_args': {'hid': (4, 6, 6), 'in_channels': 1, 'num_classes': 10},
        'dataset': 'FashionMNIST',
        'norm': mnist_norm,
        'lr': 0.006,
        'target': 70.0
    },

    # In-distribution tasks (used for training and evaluation)
    'FM-16': {
        'net_args': {'hid': (16, 32, 32), 'in_channels': 1, 'num_classes': 10},
        'dataset': 'FashionMNIST',
        'norm': mnist_norm,
        'lr': 0.006,
        'target': 89.5
    },

    'C10-16': {
            'net_args': {'hid': (16, 32, 32), 'in_channels': 3, 'num_classes': 10},
            'dataset': 'CIFAR10',
            'norm': cifar_norm,
            'lr': 0.003,
            'target': 66.0
        },

    # Out-of-distribution (OOD) tasks (used only for evaluation)
    'FM-32': {
            'net_args': {'hid': (32, 64, 64), 'in_channels': 1, 'num_classes': 10},
            'dataset': 'FashionMNIST',
            'norm': mnist_norm,
            'lr': 0.006,
            'target': 90.5
        },

    'C10-32': {
            'net_args': {'hid': (32, 64, 64), 'in_channels': 3, 'num_classes': 10},
            'dataset': 'CIFAR10',
            'norm': cifar_norm,
            'lr': 0.003,
            'target': 72.5
        },

    'C100-32': {
            'net_args': {'hid': (32, 64, 64), 'in_channels': 3, 'num_classes': 100},
            'dataset': 'CIFAR100',
            'norm': cifar_norm,
            'lr': 0.003,
            'target': 39.0
        },

}

# test the code
if __name__ == '__main__':
    for task, args in VISION_TASKS.items():
        net = Net(**args['net_args'])
        print(f'\nTASK={task}, dataset={args["dataset"]}')
        print(net)
        print('params', sum({p.data_ptr(): p.numel() for p in model.parameters()}.values()))
        print('output', net(torch.randn(1, net.fc[0].weight.shape[1], 28, 28)).shape)

    print('Done!')
