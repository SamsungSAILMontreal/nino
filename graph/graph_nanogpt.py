# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural graph for NanoGPT transformers (https://github.com/KellerJordan/modded-nanogpt).

This code requires pytorch >= 2.5.1 because of NanoGPT dependencies (see nanogpt.py).

To test the NeuralGraph class with a NanoGPT model, run:

    python graph/graph_nanogpt.py

In case of import errors, you can run it as a module:

    python -m graph.graph_nanogpt

"""

import torch
import torch.nn as nn
from torch import arange
from .graph_transformer import NeuralGraphTransformer, run_test


class NeuralGraphNanoGPT(NeuralGraphTransformer):
    """
    A neural graph for NanoGPT transformers.
    """

    _names = {
        'cls': 'embed.weight',
        'cls_w': 'value_embeds.weight',
        'attn': 'attn.qkv_w',
        'attn_q': 'attn.qkv_w.q.weight',
        'attn_k': 'attn.qkv_w.k.weight',
        'attn_v': 'attn.qkv_w.v.weight',
        'mlp_res': 'c_fc.weight',
    }
    for n in ['', '_q', '_k', '_v']:
        _names[f'attn{n}_bias'] = _names[f'attn{n}'].replace('weight', 'bias')
        if n != '':
            _names[f'layer{n}'] = _names[f'attn{n}'][:_names[f'attn{n}'].rfind('.')]
        if n == '_v':
            _names['value_res'] = _names[f'attn{n}'].replace('weight', 'res')

    def _update_model_dict(self):
        model_dict = {}
        assert self.model_first_dim_out, 'Nano GPT model_first_dim_out assumed to be True'
        self.n_value_embeds = 0
        for name, sz in self._model_dict.items():
            # Since in NanoGPT the qkv weights are stored as a single tensor, we add separate modules for q, k, v
            if name.endswith(self._names['attn']):
                for i, sfx in enumerate(['q', 'k', 'v']):
                    name_ = name.replace(self._names['attn'], self._names[f'attn_{sfx}'])
                    # sz = (3, out_dim, in_dim)
                    sz_ = (sz[1], sz[2])  # out_dim, in_dim
                    model_dict[name_] = torch.Size(sz_)
                    if sfx in ['q', 'v']:
                        for head in range(self.num_heads):
                            model_dict[name_.replace('weight', f'head{head}')] = torch.Size(
                                ((sz_[1] if self.model_first_dim_out else sz_[2]) // self.num_heads,))  # out_dim

                    if sfx == 'v':
                        model_dict[name_.replace('weight', 'res')] = torch.Size(
                            (sz_[1] if self.model_first_dim_out else sz_[0],))  # in_dim

            elif not name.endswith(self._names['attn_bias']):
                if 'value_embeds' in name:
                    # e.g. for 'value_embeds.0.weight' put '0.' in the beginning of the name,
                    # so that we can use name.endswith('value_embeds.weight') to find such embeddings easily
                    lst = name.split('.')
                    name = '.'.join([lst[-2]] + lst[:-2] + lst[-1:])
                    self.n_value_embeds += 1
                elif name.endswith(('skip_weights', 'lambdas')):
                    name = name + '.weight'
                model_dict[name] = sz
                if name.endswith(self._names['mlp_res']):
                    key = name[name.rfind('.') + 1:]
                    sz_ = self._model_dict[name.replace(key, 'weight')]
                    model_dict[name.replace(key, 'res')] = torch.Size(
                        (sz_[1] if self.model_first_dim_out else sz_[0],))  # in_dim
            else:
                raise ValueError(f'assuming no biases in this code: {name}')

        return model_dict

    def _move_offset(self, name, c_off, r_off, n_out, n_in):
        # change column offset for special layers to correctly connect them
        if name.endswith(('skip_weights', 'skip_weights.weight')):
            dim = self._model_dict[self._names.get('cls', ' ')][0]
            c_off += dim * (self.n_value_embeds + 1) + 1
        elif name.endswith(('lambdas', 'lambdas.weight')):
            cls_name = self._names.get('cls', ' ')
            if 'attn.' in name:
                c_off -= 1
            else:
                if cls_name in self._model_dict:
                    c_off -= 1
                else:
                    for n, p in self._model_dict.items():
                        if len(p) >= 2:
                            dim = p[-1]
                            c_off += dim - 2
                            break
        elif name.endswith('attn.c_proj.weight'):
            c_off += 2
        elif name.endswith(self._names.get('cls', ' ')):
            c_off -= self._model_dict['skip_weights.weight'][0]
        elif name.endswith(self._names.get('cls_w', ' ')):
            # move value_embeds to align with embed.weight
            c_off -= n_out
        return c_off, r_off

    def _correct_offsets(self, edge_index, offset_same_neurons):
        # move res col indices because of lambdas
        c_offs = []
        for c_off in offset_same_neurons:
            for n_in, layer_name, key in offset_same_neurons[c_off]:
                if '.'.join([layer_name, key]).endswith(self._names.get('value_res', ' ')):
                    edge_index[layer_name][key][1] += 1
                    c_offs.append((c_off, c_off + 1))
        for c_off, c_off_new in c_offs:
            for n_in, layer_name, key in offset_same_neurons[c_off]:
                if '.'.join([layer_name, key]).endswith(self._names.get('value_res', ' ')):
                    if c_off_new not in offset_same_neurons:
                        offset_same_neurons[c_off_new] = []
                    offset_same_neurons[c_off_new].append((n_in, layer_name, key))
                    offset_same_neurons[c_off].remove((n_in, layer_name, key))
        return edge_index, offset_same_neurons

    def _get_weight(self, states, offset, name, sz):
        # Since in NanoGPT the qkv weights are stored as a single tensor, we need to split them into q, k, v
        for ind, sfx in enumerate(['q', 'k', 'v']):
            if name.endswith(self._names[f'attn_{sfx}']):
                sz_ = torch.Size((3, sz[0], sz[1]))
                qkv = states[offset:offset + sz_.numel(), :].view(*sz_, -1)
                w = qkv[ind, :, :, :].view(*sz, -1)
                self._param_vector_index[name] = arange(sz_.numel()).view(sz_)[ind, :, :] + offset
                if sfx == 'v':
                    offset += sz_.numel()
                return w, offset

        return super()._get_weight(states, offset, name, sz)


def test_graph_nanogpt():
    """
    Test the NeuralGraph class for a NanoGPT model.
    :return:
    """
    from .nanogpt import GPT

    class Hyperparameters:
        train_seq_len = 48 * 1024  # FlexAttention sequence length
        val_seq_len = 4 * 64 * 1024  # FlexAttention sequence length for validation
        # architecture (toy-ish to test neural graph construction)
        vocab_size = 5
        model_dim = 6
        num_layers = 2
        num_heads = 2

    args = Hyperparameters()
    model: nn.Module = GPT(vocab_size=args.vocab_size,
                           num_layers=args.num_layers,
                           num_heads=args.num_heads,
                           model_dim=args.model_dim,
                           max_seq_len=max(args.train_seq_len, args.val_seq_len))
    # model = model.blocks[0]
    num_heads = None
    for m in model.modules():
        if hasattr(m, 'num_heads'):
            num_heads = m.num_heads
    assert num_heads == args.num_heads, (num_heads, args.num_heads)

    graph = NeuralGraphNanoGPT(model.named_parameters(), num_heads=num_heads)
    run_test(model, graph, name='nanogpt')


if __name__ == '__main__':
    test_graph_nanogpt()
    print('Done!')
