# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Abstract class for neural graph of transformers.
Used as a base class for NeuralGraphGPT and other transformer models.
"""

import torch
import torch_geometric as pyg
from torch import arange, zeros
from torch_geometric.utils import add_self_loops
from graph import NeuralGraph, run_test


class NeuralGraphTransformer(NeuralGraph):

    _names = {n: ' ' for n in ['cls', 'pos', 'type', 'attn_q', 'attn_k', 'attn_v', 'mlp_res']}

    def __init__(self,
                 model_dict,
                 lpe=8,
                 max_feat_size=None,
                 use_param_types=True,
                 self_loops=True,
                 model_first_dim_out=True,
                 num_heads=None,
                 num_key_value_heads=None,
                 pos_w=True,
                 wte_sampling_size=None,
                 verbose=True,
                 ):
        """
        Constructs a neural graph for GPT style transformers.

        :param model_dict: list obtained using model.named_parameters() or list/dict of (name, shape) tuples
        :param lpe: number of laplacian eigenvectors for positional encoding
        :param max_feat_size: maximum parameter feature size such as 3x3=9 for conv,
                so that total node/edge feature size is max_feat_size * state_dim.
        :param use_param_types: whether to use the parameter types in the neural graph
        :param self_loops: whether to add self-loops to the neural graph (useful to better propagate node/edge features)
        :param model_first_dim_out: whether the model's first dimension is the output dimension
                (True in nn.Linear, nn.Conv2d, but False in GPT2 layers)
        :param num_heads: number of attention heads in the transformer
        :param num_key_value_heads: number of key/value heads in the transformer (for Grouped-Query Attention)
        :param pos_w: whether to include positional embeddings for wte layers of transformers in the neural graph
        :param wte_sampling_size: used when wte layer params in x are subsampled (for meta-training)
        :param verbose: whether to print the graph statistics
        """
        assert num_heads is not None, 'num_heads must be provided for transformers!'
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.pos_w = pos_w
        self.wte_sampling_size = 0 if wte_sampling_size in ['none', 'None', None] else wte_sampling_size
        super().__init__(model_dict,
                         lpe=lpe,
                         max_feat_size=max_feat_size,
                         use_param_types=use_param_types,
                         self_loops=self_loops,
                         model_first_dim_out=model_first_dim_out,
                         verbose=verbose)

    def _param_type(self, name, sz):
        """ Returns the type of the parameter based on its name and shape.

        :param name: parameter name
        :param sz: parameter shape (torch.Size or tuple)
        :return: one of the following parameter types:
                0 - dummy params (e.g. for zero-padding),
                1 - fc,
                2 - biases, embeddings and any other params that don't fit into the other categories,
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

        is_w = name.endswith('.weight')
        is_b = name.endswith('.bias')
        if len(sz) == 1 and is_w:
            t = 4  # bn/ln
        elif len(sz) in [2, 3] and 'cls' in self._names and name.endswith(self._names['cls']):
            t = 5  # word embeddings, class_token
        elif len(sz) in [2, 3] and 'pos' in self._names and name.endswith(self._names['pos']):
            t = 6  # pos enc
        elif len(sz) == 2 and 'attn_q' in self._names and name.endswith(self._names['attn_q']):
            t = 7  # attn/query weights
        elif len(sz) == 2 and 'attn_k' in self._names and name.endswith(self._names['attn_k']):
            t = 8  # attn key weights
        elif len(sz) == 2 and 'attn_v' in self._names and name.endswith(self._names['attn_v']):
            t = 9  # attn value weights
        elif len(sz) == 4:
            t = 3  # conv
        elif len(sz) == 2 and name.find('embed') < 0:
            t = 1  # fc
        elif not is_w and not is_b and name.find('.head') >= 0:
            t = 11  # attn heads
        elif not is_w and not is_b and name.find('.self_loop') >= 0:
            t = 13  # self-loops
        elif not is_w and not is_b and name.find('.res') >= 0:
            t = 10  # residual
        else:
            t = 2  # biases and any other params that don't fit into the other categories
        return t

    def _move_offset(self, name, c_off, r_off, n_out, n_in):
        return c_off, r_off

    def _correct_offsets(self, edge_index, offset_same_neurons):
        return edge_index, offset_same_neurons

    def _construct(self):
        """
        Constructs a pyg.data.Data object for a transformer model with msa/embed/linear layers.
        :return:
        """

        param_types = [] if self.use_param_types else None
        edge_index = {}
        offset_same_neurons = {}
        c_off, r_off = 0, 0
        c_q, r_qv, c_v, c_v_end = 0, 0, 0, 0  # query and value col and row offsets to use as reference points
        is_q_bias, is_k_bias, is_v_bias = False, False, False
        for layer, (name, sz) in enumerate(self._model_dict.items()):
            param_type = self._param_type(name, sz)
            layer_name, key = name[:name.rfind('.')], name[name.rfind('.') + 1:]
            if layer_name not in edge_index:
                edge_index[layer_name] = {}

            # assume the weights are in the form (out, in, ...)
            n_out, n_in = sz[0], sz[1] if len(sz) > 1 else 1
            if ((not self.model_first_dim_out and len(sz) >= 2) or
                    name.endswith((self._names['cls'], self._names['pos'], self._names['type']))):
                # in the GPT2 layers it's (in, out, ...), assuming no nn.linear layers in GPT2
                n_out, n_in = n_in, n_out

            if len(sz) == 1 and key != 'res':
                c_off = max(0, c_off - n_out)  # bias
            elif layer == 0:
                c_off = n_in

            is_embed = param_type == 6 or name.endswith(self._names['type'])  # positional/type embeddings

            if is_embed:
                c_off -= n_out  # to make it connect to the same neurons as the previous layer
            elif name.endswith(self._names['attn_q']):
                is_q_bias = name.replace(self._names['attn_q'],
                                         self._names['attn_k_bias']) in self._model_dict
                is_k_bias = name.replace(self._names['attn_q'],
                                         self._names['attn_k_bias']) in self._model_dict
                is_v_bias = name.replace(self._names['attn_q'],
                                         self._names['attn_v_bias']) in self._model_dict
                r_off += int(is_k_bias)  # for the key bias
                c_q, r_qv, c_v = c_off, r_off, c_off + n_out
            elif name.endswith(self._names['attn_k']):
                c_off, r_off = c_q, r_qv
            elif name.endswith(self._names['attn_v']):
                c_off, r_off = c_v, r_qv
                c_v_end = c_off + n_out
            elif name.endswith(self._names['attn_q_bias']):
                c_off, r_off = c_q, r_qv + n_out + self.num_heads
            elif name.endswith(self._names['attn_k_bias']):
                c_off, r_off = c_q - n_out, r_qv - 1
            elif name.endswith(self._names['attn_v_bias']):
                c_off = c_v
                r_off = r_qv + n_out + self.num_heads + int(
                    name.replace(self._names['attn_v_bias'],
                                 self._names['attn_q_bias']) in self._model_dict)
            elif param_type == 11:  # auxiliary head connection
                # to put the heads on the c_q or c_v columns
                c_off = c_q if layer_name.endswith(self._names['layer_q']) else c_v
            else:
                c_off, r_off = self._move_offset(name, c_off, r_off, n_out, n_in)

            if key == 'res':
                if name.endswith(self._names['value_res']):
                    r_off = r_qv
                    c_off = c_v_end
                else:
                    r_off -= n_out + int(is_v_bias)
                # create a diagonal matrix
                edge_index[layer_name][key] = torch.stack((arange(r_off, r_off + n_out),
                                                           arange(c_off, c_off + n_out)))
            else:
                # create a dense edge matrix
                r = arange(r_off, r_off + n_in)
                c = arange(c_off, c_off + n_out)
                edge_index[layer_name][key] = torch.stack((r.view(-1, 1).expand(n_in, n_out).flatten(),
                                                           c.view(1, -1).expand(n_in, n_out).flatten()))
            if c_off not in offset_same_neurons:
                offset_same_neurons[c_off] = []
            offset_same_neurons[c_off].append((1 if (len(sz) == 1 or (param_type == 7 and is_q_bias)) and
                                                    param_type != 10 else
                                               (n_in if is_embed else 0),
                                               layer_name, key))

            if self.use_param_types:
                param_types.append(zeros(n_in * n_out) + param_type)
            if len(sz) > 2:
                self.max_feat_size = max(self.max_feat_size, sz[2:].numel())

            if key == 'res':
                c_off -= n_out  # so that the next layer has the same col offset and appropriate row offset
                if name.endswith(self._names['value_res']):
                    r_off += n_out + self.num_heads + c_v_end - c_v + int(is_k_bias) + int(is_v_bias) - 1
                else:
                    r_off += n_out - int(not is_v_bias)

            r_off += n_in
            c_off += n_out

        # move indices in some custom way if needed
        edge_index, offset_same_neurons = self._correct_offsets(edge_index, offset_same_neurons)

        # move column indices to take into account the bias/norms
        col_offset = 0
        for c_off in sorted(offset_same_neurons.keys()):
            col_offset += sum([c[0] for c in offset_same_neurons[c_off]])
            for (_, layer_name, key) in offset_same_neurons[c_off]:
                is_w = key.endswith('weight')
                is_b = key.endswith('bias')
                if layer_name.endswith(self._names['layer_v']) and is_w:
                    col_offset -= self.num_key_value_heads + is_v_bias
                edge_index[layer_name][key][1] += col_offset
                if key.find('head') >= 0 and not is_w and not is_b:
                    head_ind = int(key[key.find('head') + 4:])
                    edge_index[layer_name][key][1] += head_ind * edge_index[layer_name][key].shape[1]
                elif layer_name.endswith(self._names['layer_k']) and is_w:
                    edge_index[layer_name][key] = edge_index[layer_name][key][[1, 0], :]  # transpose


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
        pos_w = zeros(n_nodes, dtype=torch.long)
        if self.pos_w:
            for layer, (name, sz) in enumerate(self._model_dict.items()):
                if name.endswith(self._names['cls']):
                    n_out, n_in = sz[0], sz[1] if len(sz) > 1 else 1
                    if n_out <= n_in:
                        print(f'\nWARNING: n_out ({n_out}) <= n_in ({n_in}) for', name, sz)
                    pos_w[:n_out] = arange(n_out, dtype=torch.long) + 1
                    break

        self.pyg_graph = pyg.data.Data(edge_index=edge_index,
                                       edge_type=param_types,
                                       pos_w=pos_w  # positional embeddings for wte layers
                                       )
        print('num_nodes', self.pyg_graph.num_nodes)
        print('num_edges', self.pyg_graph.num_edges)
        print('contains_self_loops', self.pyg_graph.contains_self_loops())
        print('edge_index', self.pyg_graph.edge_index.shape)
        return self.pyg_graph
