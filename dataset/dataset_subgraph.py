# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/pytorch/examples/blob/main/mnist/main.py

"""
To test/download the dataset, run:

    export HF_HOME=/path/to/hf_cache
    python -m dataset.dataset

    where $HF_HOME is the path to the cache directory for the dataset used by huggingface.

    To download the dataset using huggingface-cli download (should be faster), see download.sh.

"""

import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric as pyg
from torch import arange, zeros, cat, Size, Tensor, tensor
from huggingface_hub import hf_hub_download
from typing import Optional, Union
from utils import VISION_TASKS, LM_TASKS, Net, scale_params
from graph import NeuralGraph, NeuralGraphGPT
from transformers import AutoConfig, AutoModelForCausalLM


class SGDDataset(Dataset):
    def __init__(
            self,
            root: str = os.environ['HF_HOME'],
            tasks: Union[list, tuple] = ('fm-16', 'c10-16', 'lm1b-3-24', 'lm1b-2-32'),
            ctx: int = 5,
            step: int = 200,
            seq_len: int = 40,
            wte_size: int = 1000,
            lpe: int = 8,
            max_samples: Optional[Union[int, list, tuple]] = None,
            verbose: Optional[Union[bool, int]] = 1,
    ):
        """
        Initialize the dataset of SGD (Adam) trajectories.

        :param root: path to the data files (default: $HF_HOME)
        :param tasks: list of tasks to sample from
        :param ctx: number of parameter states in the model input
        :param step: number of steps between the parameter states (the minimum value is 200 for lm1b, 4 for vision)
        :param seq_len: max sequence length for DMS (direct multistep forecasting) in the target
        :param wte_size: number of word token embeddings to sample during training
        :param lpe: number of laplacian eigenvectors for positional encoding
        :param max_samples: maximum number of models to sample from each task
        :param verbose:
        """
        self.verbose = verbose

        assert isinstance(tasks, (list, tuple)), type(tasks)

        self.root = root
        self.data = {}
        self.ctx = ctx
        self.step = step
        self.seq_len = seq_len
        self.wte_size = wte_size

        self.n_trajectories = {}
        self.n_states = {}
        self.graphs = {}
        self.max_feat_size = 0
        self.model_dicts = {}

        for task in tasks:
            if task == 'fm-16':
                n_traj = 300
                n_states = 2688
                n_params = 14378
                metric_names = ('test_loss', 'test_acc', 'train_loss', 'train_acc')
            elif task == 'c10-16':
                n_traj = 300
                n_states = 2513
                n_params = 14666
                metric_names = ('test_loss', 'test_acc', 'train_loss', 'train_acc')
            elif task == 'lm1b-3-24':
                n_traj = 200
                n_states = 124
                n_params = 1252464
                metric_names = ('train_loss',)
            elif task == 'lm1b-2-32':
                n_traj = 200
                n_states = 124
                n_params = 1666464
                metric_names = ('train_loss',)
            else:
                raise NotImplementedError(task)

            def download_get_local_path(name):
                print(f'Loading/downloading {name}...', flush=True)
                return hf_hub_download(
                    repo_id='SamsungSAILMontreal/nino_metatrain',
                    filename=name,
                    repo_type='dataset',
                    revision='mmap',
                    cache_dir=root)

            print(f'\n===== {task.upper()} =====')

            self.data[task] = {}
            n_parts = 2 if task.startswith('lm1b') else 1
            for i in range(n_parts):
                sfx = f'_p{i + 1}' if n_parts > 1 else ''
                mmap_file = download_get_local_path(f'{task}{sfx}.dat')
                self.data[task][i] = (mmap_file,
                                      (n_traj // n_parts, n_states, n_params))
                assert os.path.exists(mmap_file), mmap_file

            mmap_metrics = np.memmap(download_get_local_path(f'{task}_metrics.dat'),
                                     dtype='float16',
                                     mode='r',
                                     shape=(n_traj, n_states, len(metric_names)))

            if max_samples is not None:
                if isinstance(max_samples, (tuple, list)):
                    self.n_trajectories[task]  = max_samples[len(self.n_trajectories)]
                else:
                    self.n_trajectories[task] = max_samples
            else:
                self.n_trajectories[task] = n_traj
            self.n_states[task] = n_states * self.n_trajectories[task]

            # construct the neural graph
            kwargs = {'verbose': verbose, 'lpe': lpe}
            wte_sampled_size = None
            if task.startswith('lm1b'):
                if step < 200:
                    print('\nWARNING: lm1b checkpoints were saved every 200 steps, '
                          'so the minimum step of 200 will be used.')
                task_args = LM_TASKS[task.upper()]
                config = AutoConfig.from_pretrained(task_args['tokenizer'],
                                                    **task_args['net_args'])
                model = AutoModelForCausalLM.from_config(config)

                n, p = list(model.named_parameters())[0]
                assert n.endswith('wte.weight'), (n, p.shape)
                if wte_size < len(p):
                    wte_sampled_size = Size([wte_size, p.shape[1]])
                else:
                    self.wte_size = None
                    print(f'\nWARNING: wte_size={wte_size} is larger than the full size of the wte={p.shape}, '
                          f'so wte_size will be ignored')

                kwargs['num_heads'] = model.config.n_head

                # model = model.transformer.h[0]  # just one layer

                neural_graph = NeuralGraphGPT
            else:
                task_args = VISION_TASKS[task.upper()]
                model = Net(**task_args['net_args'])
                neural_graph = NeuralGraph
            ng = neural_graph(model.named_parameters(), **kwargs)
            if verbose:
                print(f'\n{ng}')  # print graph stats


            layer_ind = torch.zeros_like(ng.pyg_graph.pos_w)
            print('layer_ind', layer_ind.shape, ng.pyg_graph.pos_w.shape)
            for layer, (name, p) in enumerate(ng._model_dict.items()):
                # sz = p.shape if isinstance(p, torch.Tensor) else p
                start, end = ng._edge_dict[name]
                e_ind = ng.pyg_graph.edge_index[0, start:end]
                print(name, e_ind.shape, e_ind.min(), e_ind.max())
                layer_num = name.split('transformer.h.')
                if len(layer_num) > 1:
                    layer_num = int(layer_num[1].split('.')[0]) + 1  # 1-based index
                else:
                    layer_num = 0
                layer_ind[e_ind.min():e_ind.max() + 1] = layer_num
            ng.pyg_graph.layer_ind = layer_ind


            if wte_sampled_size is not None:

                wte_full_size = list(ng._model_dict.values())[0]  # save the full size for later

                # get a subgraph
                config = AutoConfig.from_pretrained(task_args['tokenizer'],
                                                    vocab_size=wte_size,
                                                    **task_args['net_args'])
                model = AutoModelForCausalLM.from_config(config)
                ng = neural_graph(model.named_parameters(), **kwargs)
                ng.wte_full_size = wte_full_size

                neuron_ind = cat((arange(wte_size), arange(wte_full_size[0], len(pos))))
                # Subsample wte
                if pos is not None:
                    # LPE (if used) is computed based on the full (not sampled) model to align with the original model
                    # LPEs are the same for all neurons in the wte, so can subsample in advance
                    ng.pyg_graph.pos = pos[neuron_ind]
                if pos_w is not None:
                    ng.pyg_graph.pos_w = pos_w[neuron_ind]

            self.model_dicts[task] = [(name, p.shape if isinstance(p, Tensor) else p)
                                      for name, p in model.named_parameters()]


            self.graphs[task] = ng
            self.max_feat_size = max(self.max_feat_size, self.graphs[task].max_feat_size)

            print('Num params\t\t: {}'.format(n_params))
            print('Total trajectories\t: {}'.format(self.n_trajectories[task]))
            print('Total states\t\t: {}'.format(self.n_states[task]))

            if verbose:
                for i, metric in enumerate(metric_names):
                    values = mmap_metrics[:, -1, i]
                    print(u'{:15s}\t\t: {:.3f}\u00B1{:.3f} (min={:.3f}, max={:.3f})'.format(
                        metric,
                        np.mean(values),
                        np.std(values),
                        np.min(values),
                        np.max(values)))

        for task in tasks:
            # set the max_feat_size for all graphs, so features will be zero-padded to the max_feat_size
            self.graphs[task].max_feat_size = self.max_feat_size

        print('\nAll tasks:')
        print('Total trajectories\t: {}'.format(sum(self.n_trajectories.values())))
        print('Total states\t\t: {}'.format(sum(self.n_states.values())))

    def __len__(self):
        return int(1e+7)  # some large number to avoid the end of the dataset

    def __getitem__(self, index):

        task = np.random.choice(list(self.data.keys()))  # randomly select a task (e.g. lm1b-3-24)
        sfx = np.random.choice(list(self.data[task].keys()))  # randomly select a part of the task (e.g. p1)

        if isinstance(self.data[task][sfx], tuple):
            # create a mmap object for each worker
            mmap_file, shape = self.data[task][sfx]
            self.data[task][sfx] = np.memmap(mmap_file,
                                             dtype='float16',
                                             mode='r',
                                             shape=shape)

        n_traj, n_states, n_params = self.data[task][sfx].shape  # e.g. (100, 124, 1252464)

        step = max(1, self.step // (200 if task.startswith('lm1b') else 4))  # e.g. 1

        start = np.random.randint(low=0, high=n_states - self.ctx * step + 1)  # from 0 to 119
        ind = np.arange(start, n_states, step)[:self.seq_len + self.ctx]  # e.g. 3-47

        sample_index = np.random.randint(low=0, high=n_traj)  # from 0 to 99 (inclusive)
        ng = self.graphs[task]
        pyg_graph = ng.pyg_graph.clone()

        if task.startswith('lm1b') and self.wte_size is not None:
            wte_name, wte_sampled_size = self.model_dicts[task][0]
            wte_full_size = ng.wte_full_size
            p_ind = np.arange(wte_full_size.numel()).reshape(wte_full_size)  # (50257, D)
            neuron_ind = torch.randperm(wte_full_size[0])[:wte_sampled_size[0]].sort()[0]  # 1000 num from 0 to 50257

            p_ind = p_ind[neuron_ind].flatten()  # (1000, D)
            p_ind = np.concatenate((p_ind, np.arange(wte_full_size.numel(), n_params))).flatten()  # get all indices of sampled params

            if hasattr(pyg_graph, 'pos_w') and pyg_graph.pos_w is not None:
                pyg_graph.pos_w[:len(neuron_ind)] = neuron_ind + 1  # 1-based index

            x = tensor(self.data[task][sfx][sample_index, ind][:, p_ind]).t()  # (p_ind, seq_len + self.ctx)
        else:
            x = tensor(self.data[task][sfx][sample_index, ind, :]).t()

        x, y = x[:, :self.ctx], x[:, self.ctx:]  # input of length 5: 3-7, target of length 40: 8-47

        # create a dms mask for the target (1 for valid values, 0 for padding)
        y_mask = zeros((len(y), self.seq_len), dtype=torch.bool)
        y_mask[:, :y.shape[1]] = 1

        # scale the input and target (based on the input)
        x, scales = scale_params(x, self.model_dicts[task])
        y, _ = scale_params(y, self.model_dicts[task], scales=scales)

        # pad target's last dim to seq_len
        y = F.pad(y, (0, self.seq_len - y.shape[1]))

        pyg_graph.edge_attr = self.graphs[task].to_edges(x)     # padded edge_attr -> (n, 9*5)
        pyg_graph.y = self.graphs[task].to_edges(y)             # padded edge_attr -> (n, 9*40)
        pyg_graph.y_mask = self.graphs[task].to_edges(y_mask)   # padded edge_attr -> (n, 9*40)

        return pyg_graph


def collate_graphs_fn(graphs_list):
    """
    Collate a list of graphs into a batch (a single big graph with disconnected graphs).
    :param graphs_list:
    :return:
    """
    return pyg.data.Batch.from_data_list(graphs_list)

def worker_init_fn(worker_id):
    """
    Initialize the random seed for the worker.
    :param worker_id:
    :return:
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# check the dataset (will also download the dataset if not present locally)
if __name__ == '__main__':

    train_loader = DataLoader(SGDDataset(tasks=('lm1b-3-24', )),
                              batch_size=2,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_graphs_fn,
                              worker_init_fn=worker_init_fn)

    s = time.time()
    for t, graphs in enumerate(train_loader):
        print('t', t,
              'pos', graphs.pos.shape, graphs.pos.dtype,
              'pos_w', graphs.pos_w.shape, graphs.pos_w.dtype, graphs.pos_w.min(), graphs.pos_w.max(), graphs.pos_w[:5],
              'edge_attr', graphs.edge_attr.shape, graphs.edge_attr.dtype,
              'edge_index', graphs.edge_index.shape, graphs.edge_index.dtype,
              'edge_type', graphs.edge_type.shape, graphs.edge_type.dtype,
              'y', graphs.y.shape, graphs.y.dtype,
              'y_mask', graphs.y_mask.shape, graphs.y_mask.dtype,
              'speed=%.4f sec/step' % ((time.time() - s) / (t + 1)))
        # t 8 pos torch.Size([2760, 8]) torch.float32 pos_w torch.Size([2760]) torch.int64 tensor(0) tensor(50212) tensor([ 10,  17,  71,  86, 280]) edge_attr torch.Size([75306, 45]) torch.float16 edge_index torch.Size([2, 75306]) torch.int64 edge_type torch.Size([75306]) torch.int64 y torch.Size([75306, 360]) torch.float16 y_mask torch.Size([75306, 360]) torch.bool speed=2.0043 sec/step
        # t 9 pos torch.Size([192, 8]) torch.float32 pos_w torch.Size([192]) torch.int64 tensor(0) tensor(0) tensor([0, 0, 0, 0, 0]) edge_attr torch.Size([4148, 45]) torch.float16 edge_index torch.Size([2, 4148]) torch.int64 edge_type torch.Size([4148]) torch.int64 y torch.Size([4148, 360]) torch.float16 y_mask torch.Size([4148, 360]) torch.bool speed=1.8133 sec/step
        # t 10 pos torch.Size([192, 8]) torch.float32 pos_w torch.Size([192]) torch.int64 tensor(0) tensor(0) tensor([0, 0, 0, 0, 0]) edge_attr torch.Size([4148, 45]) torch.float16 edge_index torch.Size([2, 4148]) torch.int64 edge_type torch.Size([4148]) torch.int64 y torch.Size([4148, 360]) torch.float16 y_mask torch.Size([4148, 360]) torch.bool speed=1.6528 sec/step
        if t >= 10:
            break