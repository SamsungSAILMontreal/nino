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
from torch import arange, zeros, cat, Size, Tensor, tensor, randint
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
            samples_per_traj: Optional[int] = 4,
            scale_method: str = 'std',
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
        :param samples_per_traj: number of samples to take from each trajectory
        :param scale_method: method to scale the parameters ('std' or other methods from utils.scale.METHODS)
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
        self.samples_per_traj = samples_per_traj
        self.scale_method = scale_method

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
            if task.startswith('lm1b'):
                if step < 200:
                    print('\nWARNING: lm1b checkpoints were saved every 200 steps, '
                          'so the minimum step of 200 will be used.')
                task_args = LM_TASKS[task.upper()]
                config_full = AutoConfig.from_pretrained(task_args['tokenizer'],
                                                         **task_args['net_args'])
                if wte_size >= config_full.vocab_size:
                    self.wte_size = None
                    config = config_full
                    print(f'\nWARNING: wte_size={wte_size} is >= than the full size of the wte={config.vocab_size}, '
                          f'so wte_size will be ignored')
                else:
                    config = AutoConfig.from_pretrained(task_args['tokenizer'],
                                                        vocab_size=wte_size,
                                                        **task_args['net_args'])
                model = AutoModelForCausalLM.from_config(config)
                n, p = list(model.named_parameters())[0]
                assert n.endswith('wte.weight'), (n, p.shape)
                wte_full_size = Size((config_full.vocab_size, p.shape[1]))  # save the full size for later

                kwargs['num_heads'] = model.config.n_head
                neural_graph = NeuralGraphGPT
            else:
                task_args = VISION_TASKS[task.upper()]
                model = Net(**task_args['net_args'])
                neural_graph = NeuralGraph
            ng = neural_graph(model.named_parameters(), **kwargs)
            if verbose:
                print(f'\n{ng}')  # print graph stats

            if task.startswith('lm1b'):
                ng.wte_full_size = wte_full_size

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

        # we avoid using numpy random and use torch random instead,
        # because torch natively supports dataloader workers so random numbers will be different for each worker.
        task = list(self.data.keys())[torch.randperm(len(self.data.keys()))[0]]  # randomly select a task (e.g. lm1b-3-24)
        sfx = list(self.data[task].keys())[torch.randperm(len(self.data[task].keys()))[0]]  # randomly select a part of the task (e.g. p1)

        if isinstance(self.data[task][sfx], tuple):
            # create a mmap object for each worker
            mmap_file, shape = self.data[task][sfx]
            self.data[task][sfx] = np.memmap(mmap_file,
                                             dtype='float16',
                                             mode='r',
                                             shape=shape)

        n_traj, n_states, n_params = self.data[task][sfx].shape  # e.g. (100, 124, 1252464)
        sample_index = randint(low=0, high=n_traj, size=()).item()  # from 0 to 99 (inclusive)

        step = max(1, self.step // (200 if task.startswith('lm1b') else 4))  # e.g. 1
        start = randint(low=0,
                        high=n_states - (self.ctx + min(self.seq_len, 2 * self.ctx) + self.samples_per_traj) * step,
                        size=()).item()  # from 0 to 105
        ng = self.graphs[task]

        ind = np.arange(start, n_states, step)[:self.seq_len + self.ctx]
        sample_wte = task.startswith('lm1b') and self.wte_size is not None
        if sample_wte:
            _, wte_sampled_size = self.model_dicts[task][0]
            wte_full_size = self.graphs[task].wte_full_size
            p_ind = np.arange(wte_full_size.numel()).reshape(wte_full_size)  # (50257, D)
            neuron_ind = torch.randperm(wte_full_size[0])[:wte_sampled_size[0]].sort()[0]  # 1000 num from 0 to 50257
            p_ind = p_ind[neuron_ind].flatten()  # (1000, D)
            p_ind = np.concatenate(
                (p_ind, np.arange(wte_full_size.numel(), n_params))).flatten()  # get all indices of sampled params

            x_all = tensor(self.data[task][sfx][sample_index, ind][:, p_ind]).t()  # (p_ind, seq_len + self.ctx + self.samples_per_traj)

        else:
            x_all = tensor(self.data[task][sfx][sample_index, ind, :]).t()

        if hasattr(ng.pyg_graph, 'pos') and ng.pyg_graph.pos is not None:
            # LPE's sign is non-deterministic, so when training we randomly flip the sign as a data augmentation way
            # https://pytorch-geometric.readthedocs.io/en/2.4.0/_modules/torch_geometric/transforms/add_positional_encoding.html
            # Use the same aug for all the samples in the same trajectory
            aug = randint(0, 2, size=(1, ng.pyg_graph.pos.shape[1])) * 2 - 1  # pos is (n, 8)

        graphs = []
        # sample several sequences from the same trajectory, which performs better than one sequence per trajectory
        for shift in range(self.samples_per_traj):
            pyg_graph = ng.pyg_graph.clone()  # pytorch geometric graph object

            if sample_wte and hasattr(pyg_graph, 'pos_w') and pyg_graph.pos_w is not None:
                pyg_graph.pos_w[:len(neuron_ind)] = neuron_ind + 1  # 1-based index (0 for non-wte neurons)

            if hasattr(pyg_graph, 'pos') and pyg_graph.pos is not None:
                pyg_graph.pos *= aug  # augment the laplacian positional encoding (LPE)

            x, y = x_all[:, shift:shift+self.ctx].clone(), x_all[:, shift+self.ctx:].clone()  # input of length 5: 3-7, target of length 40: 8-47
            assert y.shape[1] >= 2 * self.ctx, (y.shape, task, sfx, 'sample_index', sample_index,
                                                'n_states', n_states, 'ind', len(ind), ind)

            # create a dms mask for the target (1 for valid values, 0 for padding)
            y_mask = zeros((len(y), self.seq_len), dtype=torch.bool)
            y_mask[:, :y.shape[1]] = 1

            # scale the input and target (based on the input)
            x, scales = scale_params(x, self.model_dicts[task], method=self.scale_method, is_train=True)
            y, _ = scale_params(y, self.model_dicts[task], scales=scales, is_train=True)

            # pad target's last dim to seq_len
            y = F.pad(y, (0, self.seq_len - y.shape[1]))  # (p_ind, seq_len)

            pyg_graph.edge_attr = self.graphs[task].to_edges(x)     # padded edge_attr -> (n, 9*ctx)
            pyg_graph.y = self.graphs[task].to_edges(y)             # padded edge_attr -> (n, 9*seq_len)
            pyg_graph.y_mask = self.graphs[task].to_edges(y_mask)   # padded edge_attr -> (n, 9*seq_len)

            pyg_graph.y_mask[pyg_graph.edge_type >= 10] = 0  # mask out the auxiliary parameters
            graphs.append(pyg_graph)

        return graphs


def collate_graphs_fn(graphs_list):
    """
    Collate a list of graphs or a list of graph lists into a batch (a single big graph with disconnected graphs).
    :param graphs_list: list of graphs or a list of graph lists
    :return: single pyg graph object
    """
    if isinstance(graphs_list[0], list):
        return pyg.data.Batch.from_data_list([g for g_lst in graphs_list for g in g_lst])
    return pyg.data.Batch.from_data_list(graphs_list)

# check the dataset (will also download the dataset if not present locally)
if __name__ == '__main__':

    train_loader = DataLoader(SGDDataset(tasks=('fm-16', 'c10-16', 'lm1b-3-24', 'lm1b-2-32')),
                              batch_size=2,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_graphs_fn)

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