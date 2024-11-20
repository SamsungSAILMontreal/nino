# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions to scale and unscale model parameters.
"""

import torch


METHODS = ['std', 'std-param', 'min-max', 'min-max-param']

def scale_params(x, model_dict, scales=None, method='std', is_train=False, eps=1e-6):
    # x: psz, seq_len/batch (>=1 for training), state_dim
    assert x.dim() in [2, 3], x.shape
    sz_org = x.shape
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (psz, 1, state_dim)

    if method not in METHODS:
        raise NotImplementedError(method, 'supported methods:', METHODS)

    offset = 0
    compute_scales = scales is None
    if compute_scales:
        scales = []
    per_param = method.endswith('-param')
    is_std = method.startswith('std')
    for layer, (name, p) in enumerate(model_dict.items() if isinstance(model_dict, dict) else model_dict):
        shape = p.shape if isinstance(p, torch.Tensor) else p

        # In training our NiNo models, for transformers we scale them globally.
        # However, when using NiNo on new tasks, scaling is always done per layer, including for transformers.
        # It was a bug, but we found that it actually helps to improve the performance for some reason.
        is_wte_train = name.endswith('wte.weight') and is_train

        n = len(x) if per_param or is_wte_train else shape.numel()
        w = x[offset: offset + n]  # (n, seq_len, state_dim)
        assert len(w) > 0, (name, 'p', shape, 'w', w.shape, 'x', x.shape, 'offset', offset)
        if compute_scales:
            dims = 2 if per_param else (0, 2)
            if is_std:
                mn = torch.mean(w, dim=dims, keepdim=True)
                sd = torch.std(w, dim=dims, keepdim=True)  # (n, seq_len, 1) if per_param else (1, seq_len, 1)
            else:
                sd = torch.amax(w, dim=dims, keepdim=True)[0] - torch.amin(w, dim=dims, keepdim=True)[0]
                mn = torch.zeros_like(sd)

            if not is_std or per_param:
                sd[sd < 1e-2] = 1e-2
        else:
            mn, sd = scales[layer]

        x[offset: offset + n] = (w - mn) / (sd + eps)
        offset += n
        if compute_scales:
            scales.append((mn, sd))
        if per_param or is_wte_train:
            break

    if len(sz_org) == 2:
        x = x.squeeze(1)

    assert offset == len(x), (offset, len(x))

    return x, scales


def unscale_params(x, model_dict, scales, method='std'):
    # x: psz, seq_len/batch (>=1 for training), state_dim (>=1)
    assert x.dim() in [1, 2, 3], x.shape
    if x.dim() == 1:
        x = x.unsqueeze(1)  # (psz, 1)
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (psz, 1, 1)

    if method not in METHODS:
        raise NotImplementedError(method, 'supported methods:', METHODS)

    per_param = method.endswith('-param')

    offset = 0
    for layer, (name, p) in enumerate(model_dict.items() if isinstance(model_dict, dict) else model_dict):
        shape = p.shape if isinstance(p, torch.Tensor) else p
        n = len(x) if per_param else shape.numel()
        w = x[offset: offset + n]  # (n, seq_len, state_dim)
        assert len(w) > 0, (name, 'p', shape, 'w', w.shape, 'x', x.shape, 'offset', offset)
        mn, sd = scales[layer]
        x[offset: offset + n] = w * sd.to(w) + mn.to(w)
        offset += n
        if per_param:
            break

    assert offset == len(x), (offset, len(x))

    return x
