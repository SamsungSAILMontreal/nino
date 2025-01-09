# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural graph for Vision Transformers of open-clip.
This is experimental code that has not been well tested and was not used in the NiNo paper.

To test the NeuralGraph class with a ViT model, run:

    python graph/graph_vit_clip.py

In case of import errors, you can run it as a module:

    python -m graph.graph_vit_clip

"""
import torch
from .graph_vit import NeuralGraphViT, run_test


class NeuralGraphViTCLIP(NeuralGraphViT):
    """
    A neural graph for Vision Transformers of open-clip.
    """

    _names = {
        'conv1': 'conv1',
        'cls': 'class_embedding',
        'cls_w': 'class_embedding.weight',
        'pos': 'positional_embedding',
        'pos_w': 'positional_embedding.weight',
        'attn': 'in_proj_weight',
        'attn_q': 'in_proj_q.weight',
        'attn_k': 'in_proj_k.weight',
        'attn_v': 'in_proj_v.weight',
        'mlp_res': 'mlp.c_fc.bias',
    }
    for n in ['', '_q', '_k', '_v']:
        _names[f'attn{n}_bias'] = _names[f'attn{n}'].replace('weight', 'bias')
        if n != '':
            _names[f'layer{n}'] = _names[f'attn{n}'][:_names[f'attn{n}'].rfind('.')]
        if n == '_v':
            _names['value_res'] = _names[f'attn{n}'].replace('weight', 'res')

    def _move_offset(self, name, c_off, r_off, n_out, n_in):
        if name.endswith(self._names['conv1'] + '.weight'):
            c_off -= n_out
        elif name == 'proj.weight':
            r_off -= n_in
            c_off -= n_out
        return c_off, r_off

    def _correct_offsets(self, edge_index, offset_same_neurons):
        # max offset in the edge_index
        col_offset = max(offset_same_neurons.keys())
        row_offset = 0
        for n_in, layer_name, key in offset_same_neurons[col_offset]:
            row_offset = max(row_offset, edge_index[layer_name][key][0].max())

        # move and transpose the proj layer to the end
        layer_name = 'proj'
        key = 'weight'
        if layer_name in edge_index and key in edge_index[layer_name]:
            n_in = edge_index[layer_name][key][1, -1] - edge_index[layer_name][key][1, 0] + 1
            edge_index[layer_name][key] = edge_index[layer_name][key][[1, 0], :]  # transpose because out_dim is second in the proj layer but should be first in neural graphs
            # move the proj layer to the end of the edge_index
            edge_index[layer_name][key][0, :] += row_offset
            edge_index[layer_name][key][1, :] += col_offset + n_in - edge_index[layer_name][key][1, :].min()

            # adjust the offset_same_neurons accordingly
            for c_off in offset_same_neurons:
                for i, lst_item in enumerate(offset_same_neurons[c_off]):
                    # n_in, layer_name_, key_
                    if layer_name == lst_item[1] and key == lst_item[2]:
                        lst_item = offset_same_neurons[c_off].pop(i)
                        offset_same_neurons[col_offset].append(lst_item)
                        break
                if layer_name == lst_item[1] and key == lst_item[2]:
                    break

        for c_off in offset_same_neurons:
            items = []
            for n_in, layer_name, key in offset_same_neurons[c_off]:
                if layer_name == self._names['conv1'] and key == 'weight':
                    r = edge_index[layer_name][key][0]
                    n_in = (r[-1] - r[0] + 1).item()
                items.append((n_in, layer_name, key))
            offset_same_neurons[c_off] = items

        return edge_index, offset_same_neurons


def test_graph_vit():
    """
    Test the NeuralGraph class for a small ViT model.
    :return:
    """
    from open_clip.model import VisualTransformer

    config = {
        'width': 6,
        'mlp_ratio': 4,
        'layers': 2,
        'heads': 2,
        'patch_size': 11,
        'image_size': 33,
        'output_dim': 3
    }
    print('config', config)
    model = VisualTransformer(**config)

    # model = model.transformer.resblocks[0]  # can also work for a single transformer layer

    # can verify the correctness of the output and input dimensions in the graph
    # model.transformer.resblocks[0].attn.in_proj_weight.data[2, :] = torch.arange(6).to(
    #     model.transformer.resblocks[0].attn.in_proj_weight)  # query
    # model.transformer.resblocks[1].attn.in_proj_weight.data[7, :] = torch.arange(6).to(
    #     model.transformer.resblocks[1].attn.in_proj_weight)  # key

    num_heads = None
    for m in model.modules():
        if hasattr(m, 'num_heads'):
            num_heads = m.num_heads
    graph = NeuralGraphViTCLIP(model.named_parameters(), num_heads=num_heads)
    run_test(model, graph, name='vit_clip')


if __name__ == '__main__':
    test_graph_vit()
    print('Done!')
