# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/pytorch/examples/blob/main/mnist/main.py

"""
To test the dataset, run:

    export HF_HOME=/path/to/hf_cache
    python -m train.dataset

    where $HF_HOME is the path to the cache directory for the dataset used by huggingface.

"""

import os
import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_geometric as pyg
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
            max_samples: int = None,
            sub_graph: bool = True,
            verbose: Optional[Union[bool, int]] = 1,
    ):
        """

        :param root: path to the data files
        :param tasks:
        :param ctx:
        :param step:
        :param seq_len:
        :param wte_size:
        :param lpe:
        :param max_samples:
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
        self.sub_graph = sub_graph

        self.n_trajectories = {}
        self.n_states = {}
        self.graphs = {}
        self.max_feat_size = 0
        metrics = {}

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
                return hf_hub_download(
                    repo_id='SamsungSAILMontreal/nino_metatrain',
                    filename=name,
                    repo_type='dataset',
                    revision='mmap',
                    cache_dir=root)

            self.data[task] = {}
            n_parts = 2 if task.startswith('lm1b') else 1
            for i in range(n_parts):
                sfx = f'_p{i + 1}' if n_parts > 1 else ''
                print(f'Loading/downloading dataset {task}{sfx}...', flush=True)
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
                    print('WARNING: lm1b checkpoints were saved every 200 steps, '
                          'so the minimum step of 200 will be used.')
                task_args = LM_TASKS[task.upper()]
                config = AutoConfig.from_pretrained(task_args['tokenizer'],
                                                    **task_args['net_args'])
                model = AutoModelForCausalLM.from_config(config)
                if sub_graph:
                    # task_args['net_args']['n_layer'] = 1
                    del model.transformer.wte, model.transformer.wpe, model.lm_head

                kwargs['num_heads'] = model.config.n_head
                neural_graph = NeuralGraphGPT
            else:
                task_args = VISION_TASKS[task.upper()]
                model = Net(**task_args['net_args'])
                neural_graph = NeuralGraph
            self.graphs[task] = neural_graph(model.named_parameters(), **kwargs)
            print(f'\n{task.upper()} graph:', self.graphs[task])
            self.max_feat_size = max(self.max_feat_size, self.graphs[task].max_feat_size)
            # LPE is calculated based on the full (not subsampled) model

            if task == 'lm1b-3-24':
                # x = np.array(self.data[task]['0'][120]['data'])
                # x3 = np.array(self.data[task]['1'][120]['data'])
                sfx = 0
                mmap_file, shape = self.data[task][sfx]
                d = np.memmap(mmap_file,
                                             dtype='float16',
                                             mode='r',
                                             shape=shape)
                x = d[0, 120]
                x3 = d[1, 120]
                x2 = np.load('lm1b_tx3hid24_0_120.np.npy')
                print(task, 'x', x.shape, 'x2', x2.shape, 'x3', x3.shape, np.allclose(x, x2), np.allclose(x3, x2))
            elif task == 'lm1b-2-32':
                # x = np.array(self.data[task]['1'][11]['data'])
                sfx = 0
                mmap_file, shape = self.data[task][sfx]
                d = np.memmap(mmap_file,
                              dtype='float16',
                              mode='r',
                              shape=shape)
                x = d[1, 11]
                x3 = d[1, 12]

                x2 = np.load('lm1b_tx2hid32_1_11.np.npy')
                print(task, 'x', x.shape, 'x2', x2.shape, 'x3', x3.shape, np.allclose(x, x2), np.allclose(x3, x2))

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
        x = torch.tensor(self.data[task][sfx][sample_index, ind]).t()  # (n_params, seq_len + self.ctx)
        x, target = x[:, :self.ctx], x[:, self.ctx:]  # input of length 5: 3-7, target of length 40: 8-47

        # create a mask for the target (1 for valid values, 0 for padding)
        mask = torch.zeros((len(target), self.seq_len), dtype=torch.bool)
        mask[:, :target.shape[1]] = 1

        # pad target's last dim to seq_len
        target = F.pad(target, (0, self.seq_len - target.shape[1]))

        # scale the input and target (based on the input)
        x, scales = scale_params(x, self.graphs[task]._model_dict)
        target, _ = scale_params(target, self.graphs[task]._model_dict, scales=scales)

        # ts = np.random.randint(0, len(x))
        graph = self.graphs[task].pyg_graph.clone()
        graph.edge_attr = self.graphs[task].to_edges(x)
        graph.y = self.graphs[task].to_edges(target)
        graph.mask = self.graphs[task].to_edges(mask)

        #     offset = int(task.split('-')[2]) * (1024 + 50257)  # TODO: calculate based on the actual model
        #     # for name, p in self.graphs[task]._model_dict.items():
        #     #     if name.endswith(('wte.weight', 'wpe.weight')):
        #     #         offset += p.numel()
        #             # p_ind_all = np.arange(p.numel()).reshape(p)  # (50257, 32)
        #             # print(name, p_ind_all.shape)
        #             # p_ind = np.array(sorted(
        #             #     np.random.permutation(len(p_ind_all))[:self.wte_size]))  # 1000 numbers from 0 to 50257
        #     trajectory_wte = trajectory[:offset]
        #     trajectory = trajectory[offset:]
        #     print('trajectory', trajectory.shape, 'wte', trajectory_wte.shape, 'offset', offset)
        #     # trajectory torch.Size([21720, 5]) wte torch.Size([1230744, 5]) offset 1230744
        #     graph.wte = trajectory_wte
        # graph = self.graphs[task].set_edge_attr(trajectory, graph=graph)

        # if task.startswith('lm1b'):
        #     graph.pos_w
        # print('task', task, 'sfx', sfx,
        #       'd_idx', sample_index,
        #       # 'inds', ind,
        #       'n_states', n_states,
        #       'trajectory', trajectory.shape, trajectory.dtype,
        #       'graph', graph.edge_attr.shape, graph.edge_attr.dtype, 'num_nodes', graph.num_nodes,
        #       time.time() - s, 'sec')
        # task lm1b-2-32 sfx 1 d_idx 84 inds [37 38 39 40 41] n_states 124 trajectory torch.Size([5, 1666464]) torch.float16 graph torch.Size([26298, 45]) torch.float16 0.35678982734680176 sec
        return graph


def collate_graphs_fn(graphs):
    """
    Collate a list of graphs into a batch.
    :param graphs:
    :return:
    """
    graphs = pyg.data.Batch.from_data_list(graphs)
    return graphs

def worker_init_fn(worker_id):
    """
    Initialize the random seed for the worker.
    :param worker_id:
    :return:
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


# check the dataset
if __name__ == '__main__':

    train_loader = DataLoader(SGDDataset(),
                              batch_size=2,
                              shuffle=True,
                              num_workers=0,
                              collate_fn=collate_graphs_fn,
                              worker_init_fn=worker_init_fn)

    s = time.time()
    for t, graphs_input in enumerate(train_loader):
        print('t', t,
              'edge_attr', graphs_input.edge_attr.shape, graphs_input.edge_attr.dtype,
              'edge_index', graphs_input.edge_index.shape, graphs_input.edge_index.dtype,
              'edge_type', graphs_input.edge_type.shape, graphs_input.edge_type.dtype,
              'y', graphs_input.y.shape, graphs_input.y.dtype,
              'speed=%.4f sec/step' % ((time.time() - s) / (t + 1)))
        # t 9
        # edge_attr torch.Size([28355, 45]) torch.float16
        # edge_index torch.Size([2, 28355]) torch.int64
        # edge_type torch.Size([28355]) torch.int64
        # y torch.Size([28355, 360]) torch.float16
        # speed=0.3962 sec/step
        if t >= 9:
            break