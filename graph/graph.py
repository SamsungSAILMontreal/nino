# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural graph for ConvNets according to the NiNo paper. Can also work for MLPs.

To test the NeuralGraph class with a simple ConvNet, run:

    python graph/graph.py

In case of import errors, you can run it as a module:

    python -m graph.graph

"""
import os.path
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch import arange, zeros, ones
from torch_geometric.utils import to_networkx, add_self_loops
from torch_geometric.transforms import ToUndirected, Compose, AddSelfLoops, AddLaplacianEigenvectorPE


class NeuralGraph:

    def __init__(self,
                 model_dict,
                 lpe=8,
                 max_feat_size=None,
                 use_param_types=True,
                 self_loops=True,
                 model_first_dim_out=True):
        """
        Generic neural graph constructor for a model with fc/conv layers.
        :param model_dict: list obtained using model.named_parameters() or list/dict of (name, shape) tuples
        :param lpe: number of laplacian eigenvectors for positional encoding
        :param max_feat_size: maximum parameter feature size such as 3x3=9 for conv,
                so that total node/edge feature size is max_feat_size * state_dim.
        :param use_param_types: whether to use the parameter types in the neural graph
        :param self_loops: whether to add self-loops to the neural graph (useful to better propagate node/edge features)
        :param model_first_dim_out: whether the model's first dimension is the output dimension
                (True in nn.Linear, nn.Conv2d, but False in GPT2 layers)
        """

        self._model_dict = {
            name: p.shape if isinstance(p, torch.Tensor) else p
            for name, p in model_dict
        }  # dict of {name: shape}

        self.use_param_types = use_param_types
        self.max_feat_size = 1 if max_feat_size in ['none', 'None', None] else max_feat_size
        self.lpe = lpe
        self.self_loops = self_loops
        self.model_first_dim_out = model_first_dim_out

        self._model_dict = self._update_model_dict()
        self._construct()
        if self.lpe:
            self._add_lpe()
        if self.pyg_graph.contains_isolated_nodes():
            print('\nWARNING: isolated nodes found, which indicates that the neural graph '
                  'is likely constructed incorrectly\n')
        if self.self_loops != self.pyg_graph.contains_self_loops():
            print('\nWARNING: self-loops check fail indicates that the neural graph '
                  'is likely constructed incorrectly\n')

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
                ... (other types, e.g. for transformers)
                10 - residual,
                13 - self-loops.
        """

        is_w = name.endswith('.weight')
        if len(sz) == 1 and is_w:
            t = 4  # bn/ln
        elif len(sz) == 4:
            t = 3   # conv
        elif len(sz) == 2:
            t = 1   # fc
        elif not is_w and name.find('.res') >= 0:
            t = 10  # residual
        elif not is_w and name.find('.self_loop') >= 0:
            t = 13  # self-loops
        else:
            t = 2   # biases and any other params that don't fit into the other categories
        return t

    def _permute(self, w, name, sz):
        if self.model_first_dim_out:
            return w.permute(1, 0, *range(2, w.dim()))  # swap in_dim and out_dim for neural graphs
        else:
            return w

    def _update_model_dict(self):
        """
        Updates model_dict by adding auxiliary structural modules, e.g. residuals, heads.
        :return:
        """
        return self._model_dict

    def _construct(self):
        """
        Constructs a pyg.data.Data object for a generic model (with fc/conv layers).
        :return:
        """
        param_types = [] if self.use_param_types else None
        edge_index = {}
        offset_same_neurons = {}
        c_off, r_off = 0, 0
        for layer, (name, sz) in enumerate(self._model_dict.items()):
            param_type = self._param_type(name, sz)
            layer_name, key = name[:name.rfind('.')], name[name.rfind('.') + 1:]
            if layer_name not in edge_index:
                edge_index[layer_name] = {}

            # assume the weights are in the form (out, in, ...)
            # but in some cases like GPT2 layers it's (in, out, ...), so this won't be correct,
            # so it is necessary to use an appropriate neural graph class.
            n_out, n_in = sz[0], sz[1] if len(sz) > 1 else 1
            if not self.model_first_dim_out and len(sz) >= 2:
                n_in, n_out = n_out, n_in

            if len(sz) == 1:
                c_off = max(0, c_off - n_out)  # bias
            elif layer == 0:
                c_off = n_in

            r = arange(r_off, r_off + n_in)
            c = arange(c_off, c_off + n_out)
            edge_index[layer_name][key] = torch.stack((r.view(-1, 1).expand(n_in, n_out).flatten(),
                                                       c.view(1, -1).expand(n_in, n_out).flatten()))
            if c_off not in offset_same_neurons:
                offset_same_neurons[c_off] = []
            offset_same_neurons[c_off].append((1 if len(sz) == 1 and param_type != 10 else 0, layer_name, key))

            if self.use_param_types:
                param_types.append(zeros(n_in * n_out) + param_type)
            if len(sz) > 2:
                self.max_feat_size = max(self.max_feat_size, sz[2:].numel())

            r_off += n_in
            c_off += n_out

        # move column indices to take into account the bias/norms
        col_offset = 0

        for c_off in offset_same_neurons:
            col_offset += sum([c[0] for c in offset_same_neurons[c_off]])
            for (_, layer_name, key) in offset_same_neurons[c_off]:
                edge_index[layer_name][key][1] += col_offset

        self._edge_dict = {}  # map names to edge indices to use set_edge_attr easier
        edge_idx = 0
        for layer_name, ei in edge_index.items():
            for key in ei:
                n = edge_index[layer_name][key].shape[1]
                self._edge_dict[f'{layer_name}.{key}'] = (edge_idx, edge_idx + n)
                edge_idx += n
            edge_index[layer_name] = torch.cat(list(ei.values()), dim=1)
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

    def _get_weight(self, states, offset, name, sz):
        n = sz.numel()
        try:
            w = states[offset:offset + n, :].view(*sz, -1)
        except Exception as e:
            print(f'Error: {e}, states: {states.shape}, name: {name}, offset: {offset}, n: {n}, sz: {sz}')
            raise e
        self._param_vector_index[name] = arange(n).view(sz) + offset
        offset += n
        return w, offset

    def set_edge_attr(self, states):
        """
        Sets the edge attributes of the neural graph using the states.
        :param states: list of model states or a tensor of shape (num_params, state_dim)
        :return:
        """
        states = torch.stack(states, dim=1) if isinstance(states, list) else states
        if states.dim() == 3:
            states = states.squeeze(1)
        elif states.dim() == 1:
            states = states.unsqueeze(1)
        assert states.dim() == 2, states.shape
        self.pyg_graph.edge_attr = zeros(self.pyg_graph.edge_index.shape[1],
                                         self.max_feat_size * states.shape[1]).to(states)
        self._n_params = len(states)
        self._param_vector_index = {}  # to keep indices and convert back to_vector easier
        offset = 0
        for layer, (name, p) in enumerate(self._model_dict.items()):
            sz = p.shape if isinstance(p, torch.Tensor) else p
            param_type = self._param_type(name, sz)
            start, end = self._edge_dict[name]
            if param_type >= 10:
                w = ones(*sz, states.shape[-1]).to(states)  # fixed weights for residual/heads
            else:
                w, offset = self._get_weight(states, offset, name, sz)

            if len(sz) > 2:
                w = w.flatten(2, -2)
            elif len(sz) == 1:
                w = w.unsqueeze(1).unsqueeze(2)
            elif len(sz) == 2:
                w = w.unsqueeze(2)
            assert w.dim() == 4, w.shape

            w = self._permute(w, name, sz)  # make in_dim before out_dim for neural graphs
            if w.shape[-2] < self.max_feat_size:
                w = F.pad(w, pad=(0, 0, 0, self.max_feat_size - w.shape[-2]))  # e.g. torch.Size([1, 4, 3*3, 5])

            self.pyg_graph.edge_attr[start: end, :] = w.flatten(0, 1).flatten(1, 2)

        assert end == self.pyg_graph.edge_index.shape[1] - self.pyg_graph.num_nodes, (end,
                                                                                      self.pyg_graph.edge_index.shape,
                                                                                      self.pyg_graph.num_nodes)
        if self.self_loops:
            # append self-loop features to the edge_attr
            # should correspond to the appended edge_index values in self.pyg_graph.edge_index
            self_loops = zeros(self.pyg_graph.num_nodes, self.max_feat_size, states.shape[1]).to(
                self.pyg_graph.edge_attr)
            self_loops[:, :1, :] = 2

            self.pyg_graph.edge_attr[end:] =  self_loops.flatten(1, 2)

    def to_vector(self, edge_attr_dim=0, clean_up=True):
        """
        Converts neural graph's edge attributes to a parameter vector.
        :param edge_attr_dim: edge attribute dimension to use for conversion
        :param clean_up: delete edge_attr after conversion
        :return:
        """
        x = zeros(self._n_params).to(self.pyg_graph.edge_attr)
        for layer, (name, sz) in enumerate(self._model_dict.items()):
            if name not in self._param_vector_index:
                continue
            start, end = self._edge_dict[name]
            n_out, n_in = sz[0], sz[1] if len(sz) > 1 else 1
            w = self.pyg_graph.edge_attr[start: end].view(n_in, n_out, self.max_feat_size, -1)
            w = w[:, :, :sz[2:].numel() if len(sz) > 2 else 1, edge_attr_dim]
            w = self._permute(w, name, sz)  # make out_dim before in_dim for pytorch
            x[self._param_vector_index[name].flatten()] = w.flatten()
        if clean_up:
            del self.pyg_graph.edge_attr  # edge_attr not need after prediction
        return x

    def visualize(self,
                  fig_size=(10, 10),
                  edge_attr_key='edge_type',
                  edge_attr_dim=0,
                  remove_self_loops=True,
                  path='./results/',
                  show=False):
        """
        Visualizes the neural graph as an adjacency matrix and a networkx graph.
        By default, edge_types are used as edge attributes.
        :param fig_size:
        :param edge_attr_key:
        :param edge_attr_dim:
        :param remove_self_loops:
        :param path: path to save the plots
        :param show: whether to show the plots
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
        plt.gca().xaxis.tick_top()
        try:
            plt.xticks(x_ - 0.5, labels=None, minor=True)
            plt.yticks(x_ - 0.5, labels=None, minor=True)
        except Exception as e:
            print(e, '\nTry upgrading matplotlib.')

        plt.tight_layout()
        try:
            d = os.path.dirname(path)
            if d and not os.path.exists(d):
                os.makedirs(d)
            plt.savefig(path + 'adj.png', transparent=False)
            plt.gca().set_rasterized(True)
            plt.savefig(path + 'adj.pdf', transparent=True)
            if show:
                plt.show()
        except Exception as e:
            print(e)

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
        try:
            d = os.path.dirname(path)
            if d and not os.path.exists(d):
                os.makedirs(d)
            plt.savefig(path + 'graph.png')
            plt.savefig(path + 'graph.pdf', transparent=True)
            if show:
                plt.show()
        except Exception as e:
            print(e)

def run_test(model, graph, name=''):
    print(model)
    print('params:', sum({p.data_ptr(): p.numel() for p in model.parameters()}.values()))
    print(f'NeuralGraph for {name.upper()}:')
    print('num_nodes', graph.pyg_graph.num_nodes)
    print('num_edges', graph.pyg_graph.num_edges)
    print('contains_self_loops', graph.pyg_graph.contains_self_loops())
    if graph.lpe:
        print('pos', graph.pyg_graph.pos.shape)
    print('edge_index', graph.pyg_graph.edge_index.shape)
    graph.visualize(fig_size=(15, 15), path=f'./results/{name}_')
    params = torch.cat([p.data.flatten() for n, p in model.named_parameters()])
    graph.set_edge_attr([params, 2 * params])  # add the second state for debugging
    print('edge_attr', graph.pyg_graph.edge_attr.shape)  # only set after calling set_edge_attr
    graph.visualize(fig_size=(15, 15), edge_attr_key='edge_attr', path=f'./results/{name}_param_')
    x = graph.to_vector()
    print('graph converted back to params correctly: {}\n'.format(torch.allclose(params, x)))

def test_graph_cnn():
    """
    Test the NeuralGraph class for a simple ConvNet.
    Add batch norm layers for debugging a neural graph.
    :return:
    """

    import torch.nn as nn

    class ConvNet(nn.Module):
        def __init__(self, in_dim=3, hid_dim=(4, 6), num_classes=10):
            super().__init__()
            self.fc = nn.Sequential(nn.Conv2d(in_dim, hid_dim[0], 3, bias=False),
                                    nn.BatchNorm2d(hid_dim[0]),
                                    nn.ReLU(),
                                    nn.Conv2d(hid_dim[0], hid_dim[1], 3),
                                    nn.BatchNorm2d(hid_dim[1]),
                                    nn.ReLU(),
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Flatten(),
                                    nn.Linear(hid_dim[1], num_classes))

        def forward(self, x):
            return self.fc(x)

    model = ConvNet()
    graph = NeuralGraph(model.named_parameters())
    run_test(model, graph, name='conv')


if __name__ == '__main__':
    test_graph_cnn()
    print('Done!')
