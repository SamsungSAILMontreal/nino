# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural graph for GPT2 style transformers according to the NiNo paper.

To test the NeuralGraph class with a GPT2 model, run:

    python graph/graph_gpt.py

In case of import errors, you can run it as a module:

    python -m graph.graph_gpt

"""
import torch
from torch import arange
from .graph_transformer import NeuralGraphTransformer, run_test


class NeuralGraphGPT(NeuralGraphTransformer):
    """
    A neural graph for GPT style transformers.
    """

    _names = {
        'cls': 'wte.weight',
        'pos': 'wpe.weight',
        'attn': 'attn.c_attn.weight',
        'attn_q': 'attn.c_attn.q.weight',
        'attn_k': 'attn.c_attn.k.weight',
        'attn_v': 'attn.c_attn.v.weight',
        'mlp_res': 'c_fc.bias',
    }
    for n in ['', '_q', '_k', '_v']:
        _names[f'attn{n}_bias'] = _names[f'attn{n}'].replace('weight', 'bias')
        if n != '':
            _names[f'layer{n}'] = _names[f'attn{n}'][:_names[f'attn{n}'].rfind('.')]
        if n == '_v':
            _names['value_res'] = _names[f'attn{n}'].replace('weight', 'res')

    def __init__(self, *args, **kwargs):
        if 'model_first_dim_out' not in kwargs:
            kwargs['model_first_dim_out'] = False  # in GPT2 layers the first dimension is the input dimension
        super().__init__(*args, **kwargs)

    def _update_model_dict(self):
        model_dict = {}
        for name, sz in self._model_dict.items():
            # Since in GPT2 the qkv weights are stored as a single tensor, we add separate modules for q, k, v
            if name.endswith(self._names['attn']):
                for i, sfx in enumerate(['q', 'k', 'v']):
                    name_ = name.replace(self._names['attn'], self._names[f'attn_{sfx}'])
                    if self.model_first_dim_out:
                        sz_ = (sz[0] // 3, sz[1])  # out_dim, in_dim
                    else:
                        sz_ = (sz[0], sz[1] // 3)  # in_dim, out_dim
                    model_dict[name_] = torch.Size(sz_)
                    if sfx in ['q', 'v']:
                        for head in range(self.num_heads):
                            model_dict[name_.replace('weight', f'head{head}')] = torch.Size(
                                ((sz_[0] if self.model_first_dim_out else sz_[1]) // self.num_heads,))  # out_dim

                    name_b = name.replace('weight', 'bias')
                    if name_b in self._model_dict:
                        model_dict[name_.replace('weight', 'bias')] = torch.Size((self._model_dict[name_b][0] // 3,))

                    if sfx == 'v':
                        model_dict[name_.replace('weight', 'res')] = torch.Size(
                            (sz_[1] if self.model_first_dim_out else sz_[0],))  # in_dim

            elif not name.endswith(self._names['attn_bias']):
                model_dict[name] = sz
                if name.endswith(self._names['mlp_res']):
                    key = name[name.rfind('.') + 1:]
                    sz_ = self._model_dict[name.replace(key, 'weight')]
                    model_dict[name.replace(key, 'res')] = torch.Size(
                        (sz_[1] if self.model_first_dim_out else sz_[0],))  # in_dim
        return model_dict

    def _get_weight(self, states, offset, name, sz):
        # Since in GPT2 the qkv weights are stored as a single tensor, we need to split them
        for ind, sfx in enumerate(['q', 'k', 'v']):
            if name.endswith(self._names[f'attn_{sfx}']):
                if self.model_first_dim_out:
                    sz_ = torch.Size((3, sz[0], sz[1]))
                    qkv = states[offset:offset + sz_.numel(), :].view(*sz_, -1)
                    w = qkv[ind, :, :, :].view(*sz, -1)
                    self._param_vector_index[name] = arange(sz_.numel()).view(sz_)[ind, :, :] + offset
                else:
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

        return super()._get_weight(states, offset, name, sz)


def test_graph_gpt():
    """
    Test the NeuralGraph class for a small GPT2 model.
    :return:
    """
    import transformers

    config = {
        'n_embd': 6,
        'n_layer': 2,
        'n_head': 2,
        'vocab_size': 5,
        'n_positions': 3
    }
    config = transformers.GPT2Config(**config)
    print('config', config)
    model = transformers.AutoModelForCausalLM.from_config(config)
    # model = model.transformer.h[0]  # can also work for a single transformer layer
    graph = NeuralGraphGPT(model.named_parameters(), num_heads=config.n_head)
    run_test(model, graph, name='gpt2')


if __name__ == '__main__':
    test_graph_gpt()
    print('Done!')
