# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Utils.
"""

import random
import numpy as np
import torch
import torchvision
import transformers
import psutil
import os
import platform
import subprocess
import time
from torch.backends import cudnn


process = psutil.Process(os.getpid())


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mem(device=0):
    return (torch.cuda.max_memory_reserved(device) if device != 'cpu' else process.memory_info().rss) / 10 ** 9


def get_env_args(args):
    """
    Updates and prints argparse args with environment information.
    :param args:
    :return:
    """
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
    env['transformers'] = transformers.__version__
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
