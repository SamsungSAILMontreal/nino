# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
To test the NeuralGraph class with a simple ConvNet, run:

    python graph/graph.py

"""

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import arange, zeros
from torch_geometric.utils import to_networkx, add_self_loops
from torch_geometric.transforms import ToUndirected, Compose, AddSelfLoops, AddLaplacianEigenvectorPE


class NeuralGraph:

    def __init__(self,
                 model_dict,
                 num_heads=None,
                 improved=True,
                 use_param_types=True,
                 wte_sampling_size=None,
                 max_feat_size=None,
                 model_type=None,
                 lpe=8,
                 pos_w=True,
                 self_loops=True):
        """
        :param model_dict: list obtained using model.named_parameters() or list/dict of (name, shape) tuples.
        :param num_heads: number of heads in msa layers
        :param improved: whether to use the original Neural Graph or the improved one from the NiNo paper
        :param use_param_types: whether to use the parameter types in the neural graph
        :param wte_sampling_size: used when wte layer params in x are subsampled
        :param max_feat_size: maximum parameter feature size such as 3x3=9 for conv,
                so that total node/edge feature size is max_feat_size * state_dim.
        :param model_type: model type, must be None or one of 'generic', 'gpt2', 'bert', 'vit'
        :param lpe: number of laplacian eigenvectors for positional encoding
        :param pos_w: whether to include positional embeddings for wte layers of transformers in the neural graph
        :param self_loops: whether to add self-loops to the neural graph (useful to better propagate node/edge features)
        """

        self.model_dict = model_dict if isinstance(model_dict, list) else list(model_dict)
        self.num_heads = num_heads
        self.improved = improved
        self.use_param_types = use_param_types
        self.wte_sampling_size = 0 if wte_sampling_size in ['none', 'None', None] else wte_sampling_size
        self.max_feat_size = 1 if max_feat_size in ['none', 'None', None] else max_feat_size
        self.lpe = lpe
        self.pos_w = pos_w
        self.self_loops = self_loops

        if model_type is None:
            self.model_type = 'generic'
            for name, _ in model_dict:
                if name.startswith(('wte.', 'wpe.')):
                    self.model_type = 'gpt2'
                    break
                elif name.startswith(('class_token', 'conv_proj')):
                    self.model_type = 'vit'
                    break
                elif name.startswith('attention.'):
                    self.model_type = 'bert'
                    break
                elif name.endswith(('word_embeddings.weight', 'position_embeddings.weight')):
                    raise NotImplementedError('Bert models are only supported when a single BertLayer is provided. '
                                              'Passing model_type=\'generic\' will avoid this error, but will lead '
                                              'to an inaccurate neural graph.')
        else:
            assert model_type in ['generic', 'gpt2', 'bert',
                                  'vit'], f'instead of {model_type} must be one of these types'
            self.model_type = model_type

        self._transformer = self.model_type in ['gpt2', 'bert', 'vit']
        self._is_single_tx_block = self.model_type in ['bert']

        self._construct_generic()

        if self.lpe:
            self._add_lpe()

        assert self.self_loops == self.pyg_graph.contains_self_loops(), \
            'self-loops check fail indicates that neural graphs is likely constructed incorrectly'

    def _param_type(self, name, sz):
        """ Returns the type of the parameter based on its name and shape.

        :param name: parameter name
        :param sz: parameter shape (torch.Size or tuple)
        :return: one of the following parameter types:
                0 - dummy params (e.g. for zero-padding),
                1 - fc,
                2 - biases and any other params that don't fit into the other categories,
                3 - conv,
                4 - bn/ln,
                5 - word embeddings (wte),
                6 - pos enc (wpe),
                7 - c_attn (e.g. c_attn.q),
                8 - c_attn.k,
                9 - c_attn.v,
                10 - residual,
                11 - attn heads,
                12 - reserved,
                13 - self-loops.
        """

        if self._transformer:
            raise NotImplementedError('_param_type is not yet supported for transformers!')

        if len(sz) == 1 and name.find('.weight') >= 0:
            t = 4  # bn/ln
        elif len(sz) in [2, 3] and self._transformer and name.endswith(self.cls_names):
            t = 5  # word embeddings
        elif len(sz) in [2, 3] and self._transformer and name.endswith(self.pos_names):
            t = 6  # pos enc
        elif len(sz) == 2 and self._transformer and name.endswith(self.attn + self.attn_q):
            t = 7  # attn query weights
        elif len(sz) == 2 and self._transformer and name.endswith(self.attn_k):
            t = 8  # attn key weights
        elif len(sz) == 2 and self._transformer and name.endswith(self.attn_v):
            t = 9  # attn value weights
        elif len(sz) == 4:
            t = 3   # conv
        elif len(sz) == 2:
            t = 1   # fc
        elif name.find('.head') >= 0 and not name.endswith('.weight'):
            t = 11  # attn heads
        elif name.find('.self_loop') >= 0 and not name.endswith('.weight'):
            t = 13  # self-loops
        elif name.find('.res') >= 0 and not name.endswith('.weight'):
            t = 10  # residual
        else:
            t = 2   # biases and any other params that don't fit into the other categories
        return t

    def _construct_generic(self):
        """
        Constructs a pyg.data.Data object for a generic model (with fc/conv layers).
        :return:
        """

        if self._transformer:
            raise ValueError('_construct_generic is not intended for transformers!')

        param_types = [] if self.use_param_types else None
        edge_index = {}
        c_off, r_off = 0, 0
        for layer, (name, p) in enumerate(self.model_dict):
            sz = p.shape if isinstance(p, torch.Tensor) else p

            layer_name = name[:name.rfind('.')]
            if layer_name not in edge_index:
                edge_index[layer_name] = {}

            # assume the weights are in the form (out, in, ...)
            # but in some cases like GPT2 layers it's (in, out, ...), so this won't be correct
            n_out, n_in = sz[0], sz[1] if len(sz) > 1 else 1
            if len(sz) == 1 and 'weight' in edge_index[layer_name]:
                c_off = max(0, c_off - n_out)  # bias
            elif layer == 0:
                c_off = n_in

            r = arange(r_off, r_off + n_in).view(-1, 1).expand(n_in, n_out).flatten()
            c = arange(c_off, c_off + n_out).view(1, -1).expand(n_in, n_out).flatten()

            if len(sz) > 1:
                edge_index[layer_name]['weight'] = torch.stack((r, c))
            else:
                edge_index[layer_name]['bias'] = torch.stack((r, c))

            x = zeros(n_in, n_out).flatten()
            if self.use_param_types:
                param_types.append(x + self._param_type(name, sz))
            if len(sz) > 2:
                self.max_feat_size = max(self.max_feat_size, sz[2:].numel())

            r_off += n_in
            c_off += n_out

        # move indices to take into account the bias/norms
        offset = 0
        for layer_name, ei in edge_index.items():
            if 'bias' in ei:
                offset += 1
                ei['bias'][1] += offset
                ei['weight'][1] += offset

            edge_index[layer_name] = torch.cat([ei['weight'], ei['bias']], dim=1)

        edge_index = torch.cat(list(edge_index.values()), dim=1)
        if self.use_param_types:
            param_types = torch.cat(param_types, dim=0).long()
        n_nodes = edge_index.max().item() + 1

        if self.self_loops:
            edge_index, param_types = add_self_loops(
                edge_index,
                edge_attr=param_types,
                fill_value=self._param_type('.self_loop', [1]),
                num_nodes=n_nodes,
            )

        self.pyg_graph = pyg.data.Data(edge_index=edge_index,
                                       edge_type=param_types,
                                       pos_w=zeros(n_nodes, dtype=torch.long)  # dummy pos enc (only for transformers)
                                       )

        return self.pyg_graph

    def _add_lpe(self):
        """
        Computes Laplacian eigenvector positional encodings that are used as the neural graph node features.
        :return:
        """
        transform = [] if self.pyg_graph.contains_self_loops() else [AddSelfLoops()]
        transform = Compose(transform + [ToUndirected(),
                                         AddLaplacianEigenvectorPE(k=self.lpe, is_undirected=True)])
        device = self.pyg_graph.edge_index.device
        self.pyg_graph.pos = transform(self.pyg_graph.to('cpu')).laplacian_eigenvector_pe.to(device)

    def set_edge_attr(self, states):
        """
        Sets the edge attributes of the neural graph using the states.
        :param states: list of model states or a tensor of shape (num_params, state_dim)
        :return:
        """
        offset = 0
        edge_feat = []
        states = torch.stack(states, dim=1) if isinstance(states, list) else states
        if states.dim() == 3:
            states = states.squeeze(1)
        assert states.dim() == 2, states.shape

        for layer, (name, p) in enumerate(self.model_dict):
            sz = p.shape if isinstance(p, torch.Tensor) else p
            n = sz.numel()
            w = states[offset:offset + n, :].view(*sz, -1)
            if len(sz) > 2:
                w = w.flatten(2, -2)
            elif len(sz) == 1:
                w = w.unsqueeze(1).unsqueeze(2)
            elif len(sz) == 2:
                w = w.unsqueeze(2)
            assert w.dim() == 4, w.shape
            w = w.permute(1, 0, 2, 3)  # make in_dim before out_dim for neural graphs
            w = F.pad(w, pad=(0, 0, 0, self.max_feat_size - w.shape[-2]))  # e.g. torch.Size([1, 4, 3*3, 5])
            offset += n
            edge_feat.append(w.flatten(0, 1).flatten(1, 2))
        self.pyg_graph.edge_attr = torch.cat(edge_feat, dim=0)

        if self.self_loops:
            # append self-loop features to the edge_attr
            # should correspond to the appended edge_index values in self.pyg_graph.edge_index
            self_loops = zeros(self.pyg_graph.num_nodes, self.max_feat_size, states.shape[1]).to(
                self.pyg_graph.edge_attr)
            self_loops[:, :1, :] = 2
            self.pyg_graph.edge_attr = torch.cat((self.pyg_graph.edge_attr,
                                                  self_loops.flatten(1, 2)), dim=0)

    def to_vector(self, clean_up=True):
        offset = 0
        x = []
        for layer, (name, p) in enumerate(self.model_dict):
            sz = p.shape if isinstance(p, torch.Tensor) else p
            n_out, n_in = sz[0], sz[1] if len(sz) > 1 else 1
            n = n_out * n_in
            w = self.pyg_graph.edge_attr[offset:offset + n].view(n_in, n_out, -1)
            w = w.permute(1, 0, 2)  # make out_dim before in_dim for pytorch
            w = w[:, :, :sz[2:].numel() if len(sz) > 2 else 1]
            offset += n
            x.append(w.flatten())
        x = torch.cat(x)
        if clean_up:
            del self.pyg_graph.edge_attr  # edge_attr not need after prediction
        return x

    def visualize(self, fig_size=(5, 5), edge_attr_key='edge_type', edge_attr_dim=0, remove_self_loops=True, path='./'):
        """
        Visualizes the neural graph as an adjacency matrix and a networkx graph.
        By default, edge_types are used as edge attributes.
        :param fig_size:
        :param edge_attr_key:
        :param edge_attr_dim:
        :param remove_self_loops:
        :param path:
        :return:
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import networkx as nx

        if edge_attr_key == 'edge_type':
            # adj image colors and edge colors for networkx
            # list of tuples (string color value or colors[], weight)

            colors_weights = [('black', 0.25),  # 0
                              ('#8e7cc3', 0.5),  # 1
                              ('#333333', 0.25),  # 2
                              ('#777733', 1),  # 3
                              ('#17becf', 2),  # 4
                              ('#d62728', 1),  # 5
                              ('#1f77b4', 2),  # 6
                              ('#f6b26b', 3),  # 7
                              ('#e377c2', 1),  # 8
                              ('#76a5af', 1.5),  # 9
                              ('#7f7f7f', 2),  # 10
                              ('#a64d79', 1.5),  # 11
                              ('#ff7f0e', 0.25),  # 12
                              ('#2ca02c', 0.25)]  # 13
            n_types = len(colors_weights)
            cmap = mpl.colors.LinearSegmentedColormap.from_list('cmap',
                                                                [(k / (n_types - 1), v[0])
                                                                 for k, v in enumerate(colors_weights)])
            pyg_graph = self.pyg_graph
            bounds = np.arange(n_types + 1)
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
            cbar_kwargs = {'ticks': bounds + 0.5, 'format': '%1i'}
        else:
            pyg_graph = self.pyg_graph.clone()
            pyg_graph.edge_attr = pyg_graph.edge_attr[:, edge_attr_dim]
            cmap = plt.get_cmap()
            n_types = cmap.N
            norm = None
            cbar_kwargs = {}

        g = to_networkx(pyg_graph,
                        edge_attrs=[edge_attr_key],
                        remove_self_loops=remove_self_loops)
        adj = nx.adjacency_matrix(g, weight=edge_attr_key, dtype=np.float32).todense()
        adj[adj == 0] = np.nan
        plt.figure(figsize=fig_size)
        plt.imshow(adj, cmap=cmap, norm=norm)
        plt.colorbar(label=edge_attr_key, fraction=0.046, pad=0.04, **cbar_kwargs)
        plt.grid(lw=0.25, which='minor')
        plt.grid(lw=0.5, which='major')
        x_ = np.arange(len(adj))
        plt.xticks(x_[::10] - 0.5, x_[::10], fontsize=10)
        plt.yticks(x_[::10] - 0.55, x_[::10], fontsize=10)
        try:
            plt.xticks(x_ - 0.5, labels=None, minor=True)
            plt.yticks(x_ - 0.5, labels=None, minor=True)
        except Exception as e:
            print(e, '\nTry upgrading matplotlib.')

        plt.tight_layout()
        plt.savefig(path + 'adj.png', transparent=False)
        plt.gca().set_rasterized(True)
        plt.savefig(path + 'adj.pdf', transparent=True)
        plt.show()

        edges = g.edges()
        colors = []
        weights = []

        for u, v in edges:
            if edge_attr_key == 'edge_type':
                colors.append(g[u][v][edge_attr_key])
                weights.append(colors_weights[int(g[u][v][edge_attr_key])][1])
            else:
                colors.append('black')
                weights.append(abs(g[u][v][edge_attr_key]))

        plt.figure(figsize=fig_size)
        nx.draw(g, pos=nx.drawing.nx_pydot.graphviz_layout(g),
                edge_color=colors,
                width=weights,
                node_size=50,
                node_color='white',
                edgecolors='gray',
                edge_cmap=cmap,
                edge_vmin=0 if edge_attr_key == 'edge_type' else None,
                edge_vmax=n_types - 1 if edge_attr_key == 'edge_type' else None)
        plt.tight_layout()
        plt.savefig(path + 'graph.png')
        plt.savefig(path + 'graph.pdf', transparent=True)
        plt.show()


def test_graphs():
    """
    Test the NeuralGraph class with a simple ConvNet.
    :return:
    """

    class ConvNet(torch.nn.Module):
        def __init__(self, in_dim=3, hid_dim=(4, 6), num_classes=10):
            super().__init__()
            self.fc = torch.nn.Sequential(torch.nn.Conv2d(in_dim, hid_dim[0], 3),
                                          torch.nn.ReLU(),
                                          torch.nn.Conv2d(hid_dim[0], hid_dim[1], 3),
                                          torch.nn.ReLU(),
                                          torch.nn.AdaptiveAvgPool2d(1),
                                          torch.nn.Flatten(),
                                          torch.nn.Linear(hid_dim[1], num_classes))

        def forward(self, x):
            return self.fc(x)

    net = ConvNet()
    graph = NeuralGraph(net.named_parameters())
    print('NeuralGraph for a simple ConvNet:')
    print('num_nodes', graph.pyg_graph.num_nodes)
    print('num_edges', graph.pyg_graph.num_edges)
    print('contains_self_loops', graph.pyg_graph.contains_self_loops())
    print('pos', graph.pyg_graph.pos.shape)
    print('edge_index', graph.pyg_graph.edge_index.shape)
    # print('edge_attr', graph.pyg_graph.edge_attr.shape)  # not yet set
    graph.visualize(path='./conv_')
    return


if __name__ == '__main__':
    test_graphs()
    print('Done!')
