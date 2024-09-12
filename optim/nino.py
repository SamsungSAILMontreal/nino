# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example: opt = NiNo(base_opt=torch.optim.AdamW(...), ...)
See a full example in README, train_vision.py ar train_lm.py.

"""

import torch
import numpy as np
import transformers
import time
from torch.optim import Optimizer
from graph import *
from utils import scale_params, unscale_params, mem
from .model import NiNoModel


class NiNo:
    """
    NiNo optimizer wrapper for PyTorch optimizers.
    """

    def __init__(self,
                 base_opt: Optimizer,  # e.g. Adam
                 model,
                 ckpt='./checkpoints/nino.pt',
                 period=1000,
                 verbose=1,
                 nino_device=None,
                 max_steps=10000,
                 amp=False,
                 **kwargs):
        self.base_opt = base_opt
        if verbose:
            print('base optimizer', base_opt)
        self.ckpt = ckpt
        self.period = period
        self.verbose = verbose
        self.step_idx = 0
        self.max_steps = max_steps
        self.amp = amp
        self.meta_model = None

        # if ckpt is None, use the base optimizer only
        # otherwise, load the NiNo model and initialize the graph
        if ckpt not in [None, 'none', 'None', '']:
            self.states = []
            self._model = model
            self.model_dict = [
                (name, p.shape if isinstance(p, torch.Tensor) else p)
                for name, p in model.named_parameters()
            ]

            # create a neural graph from the model
            if isinstance(model, transformers.GPT2PreTrainedModel):
                self.graph = NeuralGraphGPT(self.model_dict, num_heads=model.config.n_head)
            else:
                self.graph = NeuralGraph(self.model_dict)
            if verbose:
                print('\nNeuralGraph:')
                print('num_nodes', self.graph.pyg_graph.num_nodes)
                print('num_edges', self.graph.pyg_graph.num_edges)
                print('contains_self_loops', self.graph.pyg_graph.contains_self_loops())
                print('pos', self.graph.pyg_graph.pos.shape)
                print('edge_index', self.graph.pyg_graph.edge_index.shape)

            if verbose > 1:
                print('Neural graph visualization is running...')
                self.graph.visualize()

            self.nino_device = next(model.parameters()).device if nino_device is None else nino_device
            self.meta_model = NiNoModel(gnn=True, **kwargs).eval().to(self.nino_device)
            if verbose:
                print(self.meta_model)

            # use base optimizer to store/load states
            assert 'states' not in self.base_opt.state, ('base optimizer already has states',
                                                         list(self.base_opt.state.keys()))
            self.base_opt.state['states'] = []

            self.meta_model.load_state_dict(torch.load(ckpt, map_location=self.nino_device))
            self.ctx = self.meta_model.ctx

            # decay k with steps to predict more in the future at the beginning and less at the end of training
            # e.g. for default setting with 10k max steps: [40 32 26 20 15 11  7  4  2  1]
            p = 2  # power of the decay (the higher, the faster decay)
            self._k_schedule = (np.linspace(self.meta_model.max_seq_len ** (1 / p),
                                            1,
                                            num=max(1, self.max_steps // self.period)) ** p).round().astype(np.int32)
            if self.verbose:
                print('k_schedule', self._k_schedule)

    def _get_k(self):
        idx = min(len(self._k_schedule) - 1, self.step_idx // self.period)
        if self.verbose:
            print('k_schedule idx', idx, 'k', self._k_schedule[idx], flush=True)
        return self._k_schedule[idx]

    def state_dict(self):
        return self.base_opt.state_dict()

    def zero_grad(self):
        self.base_opt.zero_grad()

    @property
    def next_step_nino(self):
        return self.meta_model and (len(self.base_opt.state['states']) == (self.ctx - 1) and
                                    (self.step_idx + 1) % (self.period // self.ctx) == 0)

    @property
    def need_grads(self):
        return not self.next_step_nino

    def step(self, closure=None):

        if self.meta_model and len(self.base_opt.state['states']) == self.ctx:
            self.base_opt.state['states'] = []

        if self.meta_model and (self.step_idx + 1) % (self.period // self.ctx) == 0:
            # get parameters from the optimizer as a concatenated tensor
            # add params to the list of states
            self.base_opt.state['states'].append(torch.cat([p.data.view(-1).to(self.nino_device)
                                                            for p in self._model.parameters()]))
            # can access params via self.base_opt.param_groups, but not trivial for several groups in param_groups
            if self.verbose:
                print('step %d, add state #{}, mem on {}={:.3f}G'.format(self.step_idx + 1,
                                                                         len(self.base_opt.state['states']),
                                                                         mem(self.nino_device)), flush=True)

        if self.meta_model and len(self.base_opt.state['states']) == self.ctx:
            # grads are not need for this step (can omit loss.backward())

            if closure is not None:
                loss = closure()
                if self.verbose:
                    print('loss before NiNo step: %.4f' % loss.item(), flush=True)
            else:
                loss = None

            if self.verbose:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats(self.nino_device)
                    torch.cuda.synchronize()
                print('NiNo step starting: mem on {}={:.3f}G'.format(self.nino_device,
                                                                     mem(self.nino_device)), flush=True)
                start_time = time.time()

            with torch.no_grad():
                # prediction step
                with torch.autocast(device_type='cpu' if self.nino_device == 'cpu' else 'cuda',
                                    enabled=self.amp,
                                    dtype=torch.bfloat16):
                    # using AMP can save memory but may lead to NaNs in the predicted parameters
                    states = torch.stack(self.base_opt.state['states'], dim=1)
                    states, scales = scale_params(states, self.model_dict)
                    self.graph.set_edge_attr(states)
                    self.graph.pyg_graph = self.meta_model(self.graph.pyg_graph.to(self.nino_device), k=self._get_k())
                    x = self.graph.to_vector()
                    if torch.isnan(x).any():
                        raise ValueError('NaNs in the predicted parameters')
                    x = unscale_params(x, self.model_dict, scales)
                    if torch.isnan(x).any():
                        raise ValueError('NaNs in the predicted parameters after scaling')
                # move states to cpu to free gpu mem
                self.base_opt.state['states'] = [s.cpu() for s in self.base_opt.state['states']]

            if self.verbose:
                print('NiNo step finished: {:.3f} sec, mem on {}={:.3f}G'.format(time.time() - start_time,
                                                                                 self.nino_device,
                                                                                 mem(self.nino_device)), flush=True)

            # set the predicted values as the new parameters
            i = 0
            for p in self._model.parameters():
                n = p.numel()
                p.data = x[i: i + n].data.view_as(p).to(p)
                i += n

            if closure is not None:
                loss = closure()
                if self.verbose:
                    print('loss after NiNo step: %.4f' % loss.item(), flush=True)

        else:
            # make sure to compute grads for this step
            loss = self.base_opt.step(closure)

        self.step_idx += 1

        return loss
