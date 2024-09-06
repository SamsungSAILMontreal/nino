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
import psutil
import os


process = psutil.Process(os.getpid())


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mem(device=0):
    return (torch.cuda.max_memory_reserved(device) if device != 'cpu' else process.memory_info().rss) / 10 ** 9
