# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural graph for Vision Transformers.
This is experimental code that has not been well tested and was not used in the NiNo paper.

To test the NeuralGraph class with a ViT model, run:

    python graph/graph_vit.py

In case of import errors, you can run it as a module:

    python -m graph.graph_vit

"""
import torch
from torch import arange
from .graph_transformer import NeuralGraphTransformer, run_test


class NeuralGraphViT(NeuralGraphTransformer):
    """
    A neural graph for Vision Transformers.
    """

    _names = {
        'cls': 'class_token',
        'pos': 'pos_embedding',
        'type': ' ',
        'attn': 'in_proj_weight',
        'attn_q': 'in_proj_q.weight',
        'attn_k': 'in_proj_k.weight',
        'attn_v': 'in_proj_v.weight',
        'mlp_res': 'mlp.0.bias',
    }
    for n in ['', '_q', '_k', '_v']:
        _names[f'attn{n}_bias'] = _names[f'attn{n}'].replace('weight', 'bias')
        if n != '':
            _names[f'layer{n}'] = _names[f'attn{n}'][:_names[f'attn{n}'].rfind('.')]
        if n == '_v':
            _names['value_res'] = _names[f'attn{n}'].replace('weight', 'res')

    def _update_model_dict(self):
        model_dict = {}
        for name, sz in self._model_dict.items():
            # Since in ViT the qkv weights are stored as a single tensor, we add separate modules for q, k, v
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
                if name.endswith((self._names['cls'], self._names['pos'])):
                    model_dict[name + '.weight'] = sz[1:]
                    if name.endswith(self._names['cls']):
                        self._names['cls'] += '.weight'
                    elif name.endswith(self._names['pos']):
                        self._names['pos'] += '.weight'
                else:
                    model_dict[name] = sz
                if name.endswith(self._names['mlp_res']):
                    key = name[name.rfind('.') + 1:]
                    sz_ = self._model_dict[name.replace(key, 'weight')]
                    model_dict[name.replace(key, 'res')] = torch.Size(
                        (sz_[1] if self.model_first_dim_out else sz_[0],))  # in_dim
        return model_dict

    def _move_offset(self, name, c_off, r_off, n_out, n_in):
        if name.endswith('conv_proj.weight'):
            c_off -= n_out
        return c_off, r_off

    def _correct_offsets(self, edge_index, offset_same_neurons):
        for c_off in offset_same_neurons:
            items = []
            for n_in, layer_name, key in offset_same_neurons[c_off]:
                if layer_name  == 'conv_proj' and key == 'weight':
                    r = edge_index[layer_name][key][0]
                    n_in = (r[-1] - r[0] + 1).item()
                items.append((n_in, layer_name, key))
            offset_same_neurons[c_off] = items
        return edge_index, offset_same_neurons

    def _get_weight(self, states, offset, name, sz):
        # Since in ViT the qkv weights are stored as a single tensor, we need to split them
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


def test_graph_vit():
    """
    Test the NeuralGraph class for a small ViT model.
    :return:
    """
    import torchvision

    config = {
        'hidden_dim': 6,
        'mlp_dim': 24,
        'num_layers': 2,
        'num_heads': 2,
        'patch_size': 11,
        'image_size': 33,
        'num_classes': 5
    }
    print('config', config)
    model = torchvision.models.vision_transformer._vision_transformer(
        weights=None,
        progress=False,
        **config
    )
    # model = model.encoder.layers.encoder_layer_0  # can also work for a single transformer layer

    # can verify the correctness of the output and input dimensions in the graph
    # model.encoder.layers.encoder_layer_0.self_attention.in_proj_weight.data[2, :] = torch.arange(6).to(
    #     model.encoder.layers.encoder_layer_0.self_attention.in_proj_weight)  # query
    # model.encoder.layers.encoder_layer_1.self_attention.in_proj_weight.data[7, :] = torch.arange(6).to(
    #     model.encoder.layers.encoder_layer_1.self_attention.in_proj_weight)  # key

    graph = NeuralGraphViT(model.named_parameters(), num_heads=model.encoder.layers.encoder_layer_0.num_heads)
    run_test(model, graph, name='vit')


if __name__ == '__main__':
    test_graph_vit()
    print('Done!')
