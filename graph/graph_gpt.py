# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
To test the NeuralGraph class with a GPT2 model, run:

    python graph/graph_gpt.py

"""
import torch
import torch_geometric as pyg
from torch import arange, zeros
from torch_geometric.utils import add_self_loops
from graph import NeuralGraph


class NeuralGraphGPT(NeuralGraph):

    _names = {
        'cls': 'wte.weight',
        'pos': 'wpe.weight',
        'attn': 'attn.c_attn.weight',
        'attn_q': 'attn.c_attn.q.weight',
        'attn_k': 'attn.c_attn.k.weight',
        'attn_v': 'attn.c_attn.v.weight',
    }
    for n in ['', '_q', '_k', '_v']:
        _names[f'attn{n}_bias'] = _names[f'attn{n}'].replace('weight', 'bias')


    def __init__(self,
                 model_dict,
                 lpe=8,
                 max_feat_size=None,
                 use_param_types=True,
                 self_loops=True,
                 model_first_dim_out=False,
                 num_heads=None,
                 pos_w=True,
                 wte_sampling_size=None,
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
        :param wte_sampling_size: used when wte layer params in x are subsampled
        :param pos_w: whether to include positional embeddings for wte layers of transformers in the neural graph
        """
        assert num_heads is not None, 'num_heads must be provided for transformers!'
        self.num_heads = num_heads
        self.pos_w = pos_w
        self.wte_sampling_size = 0 if wte_sampling_size in ['none', 'None', None] else wte_sampling_size

        super(NeuralGraphGPT, self).__init__(model_dict,
                                             lpe=lpe,
                                             max_feat_size=max_feat_size,
                                             use_param_types=use_param_types,
                                             self_loops=self_loops,
                                             model_first_dim_out=model_first_dim_out)

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

        is_w = name.endswith('.weight')
        if len(sz) == 1 and is_w:
            t = 4  # bn/ln
        elif len(sz) in [2, 3] and name.endswith(self._names['cls']):
            t = 5  # word embeddings, class_token
        elif len(sz) in [2, 3] and name.endswith(self._names['pos']):
            t = 6  # pos enc
        elif len(sz) == 2 and name.endswith((self._names['attn'], self._names['attn_q'])):
            t = 7  # attn/query weights
        elif len(sz) == 2 and name.endswith(self._names['attn_k']):
            t = 8  # attn key weights
        elif len(sz) == 2 and name.endswith(self._names['attn_v']):
            t = 9  # attn value weights
        elif len(sz) == 4:
            t = 3  # conv
        elif len(sz) == 2:
            t = 1  # fc
        elif not is_w and name.find('.head') >= 0:
            t = 11  # attn heads
        elif not is_w and name.find('.self_loop') >= 0:
            t = 13  # self-loops
        elif not is_w and name.find('.res') >= 0:
            t = 10  # residual
        else:
            t = 2  # biases and any other params that don't fit into the other categories
        return t

    def _construct(self):
        """
        Constructs a pyg.data.Data object for a generic model (with fc/conv layers).
        :return:
        """

        # update model_dict by adding auxiliary structural items (residuals, heads)
        model_dict = {}
        for name, sz in self._model_dict.items():
            if name.endswith(self._names['attn']):
                for i, sfx in enumerate(['q', 'k', 'v']):
                    name_ = name.replace(self._names['attn'], self._names[f'attn_{sfx}'])
                    sz_ = (sz[0], sz[1] // 3)
                    model_dict[name_] = torch.Size(sz_)
                    name_b = name.replace('weight', 'bias')
                    if name_b in self._model_dict:
                        sz_ = (self._model_dict[name_b][0] // 3,)
                        name_b = name_.replace('weight', 'bias')
                        if sfx in ['q', 'v']:
                            for head in range(self.num_heads):
                                model_dict[name_b.replace('bias', f'head{head}')] = torch.Size(
                                    (sz_[0] // self.num_heads,))
                        model_dict[name_b] = torch.Size(sz_)
                        if sfx == 'v':
                            model_dict[name_b.replace('bias', 'res')] = torch.Size(sz_)

            elif not name.endswith(self._names['attn_bias']):
                model_dict[name] = sz
                if name.endswith('c_fc.bias'):
                    sz_ = self._model_dict[name.replace('bias', 'weight')]
                    model_dict[name.replace('bias', 'res')] = torch.Size((sz_[0],))
        self._model_dict = model_dict

        param_types = [] if self.use_param_types else None
        edge_index = {}
        offset_same_neurons = {}
        c_off, r_off = 0, 0
        pos_w = [] if self.pos_w else None
        for layer, (name, sz) in enumerate(self._model_dict.items()):
            param_type = self._param_type(name, sz)
            layer_name, param_name = name[:name.rfind('.')], name[name.rfind('.') + 1:]
            key = param_name
            if layer_name not in edge_index:
                edge_index[layer_name] = {}

            # print(name, sz, 'c_off', c_off, 'r_off', r_off, 'param_type', param_type)
            # assume the weights are in the form (out, in, ...)
            n_out, n_in = sz[0], sz[1] if len(sz) > 1 else 1
            if not self.model_first_dim_out and len(sz) >= 2:
                # in the GPT2 layers it's (in, out, ...)
                # assuming no nn.linear layers in GPT2 (lm_head is tied with the embeddings)
                n_out, n_in = n_in, n_out

            if len(sz) == 1:
                c_off = max(0, c_off - n_out)  # bias
            elif layer == 0:
                c_off = n_in

            if param_type == 6:
                c_off -= n_out
            elif name.endswith(self._names['attn_q']):
                r_off += 1  # for the key bias
            elif name.endswith(self._names['attn_k']):
                c_off -= n_out
                r_off -= n_in + self.num_heads + 1
            elif name.endswith(self._names['attn_v']):
                r_off -= n_in + 1
                c_off -= n_out // self.num_heads
            elif name.endswith(self._names['attn_k_bias']):
                r_off -= n_out + 1
                c_off -= n_out
            elif name.endswith(self._names['attn_v_bias']):
                r_off += 1
            elif name.find('head') >= 0:
                c_off -= n_out * (self.num_heads - 1)

            # print(layer_name, key, sz, 'c_off', c_off, 'r_off', r_off, 'param_type', param_type)

            if key == 'res':
                c_off += n_out
                if name.endswith('v.res'):
                    r_off -= n_out + self.num_heads + 2
                else:
                    r_off -= n_out + 1
                r = arange(r_off, r_off + n_out)
                c = arange(c_off, c_off + n_out)
                edge_index[layer_name][key] = torch.stack((r, c))
            else:
                r = arange(r_off, r_off + n_in)
                c = arange(c_off, c_off + n_out)
                edge_index[layer_name][key] = torch.stack((r.view(-1, 1).expand(n_in, n_out).flatten(),
                                                           c.view(1, -1).expand(n_in, n_out).flatten()))
            if c_off not in offset_same_neurons:
                offset_same_neurons[c_off] = []
            offset_same_neurons[c_off].append((1 if (len(sz) == 1 or param_type == 7) and param_type != 10 else
                                               (n_in if param_type == 6 else 0), layer_name, key))

            if self.use_param_types:
                param_types.append(zeros(n_in * n_out) + param_type)
            if len(sz) > 2:
                self.max_feat_size = max(self.max_feat_size, sz[2:].numel())

            if name.endswith(self._names['attn_k_bias']):
                r_off += n_out + 1
                c_off += n_out + (n_out // self.num_heads) - self.num_heads - 1
            elif key == 'res':
                c_off -= n_out
                if name.endswith('v.res'):
                    r_off += 2 * n_out + self.num_heads + 1
                else:
                    r_off += n_out
            elif name.find('head') >= 0:
                c_off += n_out * (self.num_heads - 1)

            r_off += n_in
            c_off += n_out

        # move column indices to take into account the bias/norms
        col_offset = 0

        for c_off in offset_same_neurons:
            col_offset += sum([c[0] for c in offset_same_neurons[c_off]])
            for (_, layer_name, key) in offset_same_neurons[c_off]:
                edge_index[layer_name][key][1] += col_offset
                if key.find('head') >= 0:
                    head_ind = int(key[key.find('head') + 4:])
                    edge_index[layer_name][key][1] += head_ind * edge_index[layer_name][key].shape[1]
                elif layer_name.endswith('.q') and key.endswith('weight'):
                    q_idx = edge_index[layer_name][key]
                elif layer_name.endswith('.k') and key.endswith('weight'):
                    edge_index[layer_name][key] = q_idx[[1, 0], :]

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

        return self.pyg_graph

    def _get_weight(self, states, offset, name, sz):
        for ind, sfx in enumerate(['q', 'k', 'v']):
            if name.endswith(self._names[f'attn_{sfx}']):
                sz_ = torch.Size((sz[0], 3, sz[1]))
                qkv = states[offset:offset + sz_.numel(), :].view(*sz_, -1)
                w = qkv[:, ind, :, :].view(*sz, -1)
                self._param_vector_index[name] = arange(sz_.numel()).view(sz_)[:, ind, :] + offset
                offset += sz_.numel()
                return w, offset
            elif name.endswith(self._names[f'attn_{sfx}_bias']):
                sz_ = torch.Size((3, sz[0]))
                qkv = states[offset:offset + sz_.numel(), :].view(*sz_, -1)
                w = qkv[ind, :, :].view(*sz, -1)
                self._param_vector_index[name] = arange(sz_.numel()).view(sz_)[ind] + offset
                offset += (sz_.numel() if sfx == 'v' else -sz_.numel() * sz[0])
                return w, offset

        return super(NeuralGraphGPT, self)._get_weight(states, offset, name, sz)


def test_graph_gpt():
    """
    Test the NeuralGraph class for a small GPT2 model.
    :return:
    """
    import transformers

    gpt2_config = {'n_embd': 6,
                   'n_layer': 2,
                   'n_head': 2,
                   'vocab_size': 5,
                   'n_positions': 3
                   }
    print('gpt2_config', gpt2_config)
    model = transformers.AutoModelForCausalLM.from_config(transformers.GPT2Config(**gpt2_config))
    print(model)
    print('params:', sum(p.numel() for p in model.parameters()))
    graph = NeuralGraphGPT(model.named_parameters(), num_heads=gpt2_config['n_head'])
    print('NeuralGraph for GPT2:')
    print('num_nodes', graph.pyg_graph.num_nodes)
    print('num_edges', graph.pyg_graph.num_edges)
    print('contains_self_loops', graph.pyg_graph.contains_self_loops())
    print('pos', graph.pyg_graph.pos.shape)
    print('edge_index', graph.pyg_graph.edge_index.shape)
    graph.visualize(fig_size=(15,15), path='./results/gpt2_')
    params = torch.cat([p.data.flatten() for n, p in model.named_parameters()])
    graph.set_edge_attr([params, 2 * params])  # add the second state for debugging
    print('edge_attr', graph.pyg_graph.edge_attr.shape)  # only set after calling set_edge_attr
    graph.visualize(fig_size=(15, 15), edge_attr_key='edge_attr', path='./results/gpt2_param_')
    x = graph.to_vector()
    print('graph converted back to params correctly: {}\n'.format(torch.allclose(params, x)))
    return


if __name__ == '__main__':
    test_graph_gpt()
    print('Done!')
