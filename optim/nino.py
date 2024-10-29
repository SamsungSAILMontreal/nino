# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example: opt = NiNo(base_opt=torch.optim.AdamW(...), ...)
See a full example in README.md, train_vision.py ar train_lm.py.

"""

import torch
import numpy as np
import transformers
import torchvision
import time
from typing import Optional, Union
from torch.optim import Optimizer
from graph import *
from utils import scale_params, unscale_params, mem
from .model import NiNoModel


class NiNo:
    """
    NiNo optimizer wrapper for PyTorch optimizers.
    """

    def __init__(self,
                 base_opt: Union[Optimizer, None],  # e.g. Adam
                 model: Union[torch.nn.Module, None],
                 ckpt: Optional[str] = './checkpoints/nino.pt',
                 period: Optional[int] = 1000,
                 verbose: Optional[Union[bool, int]] = 1,
                 nino_device: Optional[Union[torch.device, int, str]] = None,
                 message_passing_device: Optional[Union[torch.device, int, str]] = None,
                 max_train_steps: Optional[int] = 10000,
                 amp: Optional[bool] = False,
                 **kwargs):
        """

        :param base_opt: use None if you want to apply NiNo only (see nino_step.py).
        :param model: PyTorch model. Use None to set the model later (see nino_step.py).
        :param ckpt: NiNo checkpoint path.
        :param period: number of steps between NiNo steps.
        :param verbose: print debug info.
        :param nino_device: device for the NiNo model.
        :param message_passing_device: device for the GNN layer.
        :param max_train_steps: maximum number of steps (to compute future horizon k).
        :param amp: Automatic Mixed Precision (AMP) for the NiNo step.
        :param kwargs: NiNo model arguments.
        """
        self.base_opt = base_opt
        if verbose:
            print('base optimizer', base_opt)
        self.ckpt = ckpt
        self.period = period
        self.verbose = verbose
        self.step_idx = 0
        self.max_train_steps = max_train_steps
        self.amp = amp
        self.meta_model = None

        # if ckpt is None, use the base optimizer only
        # otherwise, load the NiNo model and initialize the graph
        if ckpt not in [None, 'none', 'None', '']:

            self.nino_device = next(model.parameters()).device if nino_device is None else nino_device
            state_dict = torch.load(ckpt, map_location=self.nino_device)
            step = state_dict['step'] if 'step' in state_dict else -1
            if 'model_args' in state_dict:
                kwargs.update(state_dict['model_args'])
                print('\n\nkwargs', kwargs)
                state_dict = state_dict['state_dict']
            self.meta_model = NiNoModel(message_passing_device=message_passing_device, **kwargs).eval().to(
                self.nino_device)
            if verbose:
                print(self.meta_model)
            result = self.meta_model.load_state_dict(state_dict)
            if verbose:
                print('NiNo with {} params loaded from step {}, ckpt file {}: {}'.format(
                    sum([p.numel() for p in self.meta_model.parameters()]),
                    step,
                    ckpt,
                    result))
            self.ctx = self.meta_model.ctx

            self.states = []
            # decay k with steps to predict more in the future at the beginning and less at the end of training
            # e.g. for default setting with 10k max steps: [40 33 26 21 16 11  8  5  3  1]
            p = 2  # power of the decay (the higher, the faster decay)
            self._k_schedule = (np.linspace(self.meta_model.seq_len ** (1 / p),
                                            1,
                                            num=max(1, self.max_train_steps // self.period)) ** p).round().astype(np.int32)
            if self.verbose:
                print(f'\nk_schedule values (direct multi-step forecasting) = {self._k_schedule}')
            if model is not None:
                self.set_model(model)
            else:
                self._model = None

    def set_model(self, model, lpe=None, **kwargs):
        """
        Sets the model and creates a neural graph from it.
        Optionally, visualize the neural graph.
        :param model: PyTorch model.
        :param lpe: LPE features if already computed.
        :param kwargs: NeuralGraph arguments.
        :return:
        """
        self._model = model
        self._model_dict = [
            (name, p.shape if isinstance(p, torch.Tensor) else p)
            for name, p in model.named_parameters()
        ]

        # create a neural graph from the model
        kwargs['verbose'] = self.verbose
        if lpe is not None:
            kwargs['lpe'] = lpe
        if self.meta_model.is_mlp:
            kwargs['lpe'] = False
        if isinstance(model, transformers.GPT2PreTrainedModel):
            neural_graph = NeuralGraphGPT
            kwargs['num_heads'] = model.config.n_head
        elif isinstance(model, transformers.BertPreTrainedModel):
            neural_graph = NeuralGraphBERT
            kwargs['num_heads'] = model.config.num_attention_heads
        elif isinstance(model, transformers.LlamaPreTrainedModel):
            neural_graph = NeuralGraphLlama
            kwargs['num_heads'] = model.config.num_attention_heads
            kwargs['num_key_value_heads'] = model.config.num_key_value_heads
        elif isinstance(model, torchvision.models.VisionTransformer):
            neural_graph = NeuralGraphViT
            kwargs['num_heads'] = model.encoder.layers.encoder_layer_0.num_heads
        else:
            neural_graph = NeuralGraph

        start_time = time.time()
        self.graph = neural_graph(self._model_dict, **kwargs)

        if self.verbose:
            print('\nNeural graph for "{}" constructed in {:.3f} sec:\n{}'.format(model.__class__.__name__,
                                                                                  time.time() - start_time,
                                                                                  self.graph))

        if self.verbose > 1:
            print('Neural graph visualization is running...')
            self.graph.visualize()

    def get_k(self, step=None):
        idx = min(len(self._k_schedule) - 1, (self.step_idx if step is None else step) // self.period)
        return self._k_schedule[idx]

    def state_dict(self):
        return self.base_opt.state_dict()

    def zero_grad(self):
        self.base_opt.zero_grad()

    @property
    def next_step_nino(self):
        return self.meta_model and (len(self.states) == (self.ctx - 1) and
                                    (self.step_idx + 1) % (self.period // self.ctx) == 0)

    @property
    def need_grads(self):
        return not self.next_step_nino

    def step(self, closure=None, k=None, nino_fw_device=None):

        if self.meta_model:
            device = nino_fw_device if nino_fw_device is not None else self.nino_device

        if self.meta_model and (self.step_idx + 1) % (self.period // self.ctx) == 0:
            # get parameters from the model as a concatenated tensor
            # add params to the list of states
            self.states.append(torch.cat([p.data.view(-1).to(self.nino_device)
                                          for p in self._model.parameters()]))
            # can access params via self.base_opt.param_groups, but not trivial for several groups in param_groups
            if self.verbose:
                print('step {}, add state #{}, mem on {}={:.3f}G, cpu={:.3f}G'.format(self.step_idx + 1,
                                                                                      len(self.states),
                                                                                      device,
                                                                                      mem(device),
                                                                                      mem('cpu')), flush=True)

        if self.meta_model and len(self.states) == self.ctx:
            # gradients are not needed for this step (can omit loss.backward())

            if closure is not None:
                loss = closure()
                if self.verbose:
                    print('loss before NiNo step: %.4f' % loss.item(), flush=True)
            else:
                loss = None

            if k is None:
                k = self.get_k()
            if self.verbose:
                if torch.cuda.is_available() and self.nino_device != 'cpu':
                    torch.cuda.reset_peak_memory_stats(self.nino_device)
                    torch.cuda.synchronize()
                print('\nNiNo step starting at step {} (k={}): peak mem on {}={:.3f}G, cpu={:.3f}G'.format(
                    self.step_idx + 1,
                    k,
                    device,
                    mem(device),
                    mem('cpu')), flush=True)
                start_time = time.time()

            with torch.no_grad():
                # prediction step
                with torch.autocast(device_type='cpu' if self.nino_device == 'cpu' else 'cuda',
                                    enabled=self.amp,
                                    dtype=torch.bfloat16):
                    # using AMP can save memory but may lead to NaNs in the predicted parameters

                    states = torch.stack(self.states, dim=1)
                    states, scales = scale_params(states, self._model_dict)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.graph.set_edge_attr(states)
                    if self.verbose:
                        print('running the meta model', flush=True)
                    if nino_fw_device is not None:
                        self.meta_model = self.meta_model.to(nino_fw_device)
                    self.graph.pyg_graph = self.meta_model(self.graph.pyg_graph.to(self.nino_device), k=k)
                    if self.graph.pyg_graph.edge_attr.shape[-1] != 1:
                        print('\nWARNING: edge_attr.shape[-1] != 1', self.graph.pyg_graph.edge_attr.shape)
                    x = self.graph.to_vector()
                    if torch.isnan(x).any():
                        raise ValueError('NaNs in the predicted parameters')
                    x = unscale_params(x, self._model_dict, scales)

                self.states = []

            if self.verbose:
                print('NiNo step finished: {:.3f} sec, peak mem on {}={:.3f}G, cpu={:.3f}G\n'.format(
                    time.time() - start_time,
                    device,
                    mem(device),
                    mem('cpu')),
                    flush=True)

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

        if hasattr(self.base_opt, 'sync_gradients'):
            if self.base_opt.sync_gradients:
                self.step_idx += 1
        else:
            self.step_idx += 1

        return loss
