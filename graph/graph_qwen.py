# Copyright (c) 2024-2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural graph for Qwen style transformers.
This is experimental code that has not been well tested and was not used in the NiNo paper.

To test the NeuralGraph class with a Qwen model, run:

    python graph/graph_qwen.py

In case of import errors, you can run it as a module:

    python -m graph.graph_qwen

"""
import torch
from .graph_llama import NeuralGraphLlama, run_test


class NeuralGraphQwen(NeuralGraphLlama):
    """
    A neural graph for Qwen style transformers.
    """

    def _move_offset(self, name, c_off, r_off, n_out, n_in):
        # q_norm and k_norm will be moved in _correct_offsets, so need to offset here
        c_off, r_off = super()._move_offset(name, c_off, r_off, n_out, n_in)
        if name.endswith('_norm.weight'):
            r_off -= 1

        return c_off, r_off

    def _correct_offsets(self, edge_index, offset_same_neurons):
        # move res col indices because of q_norm/k_norm layers
        # q_norm and k_norm weights can be modeled as connections of neurons within q_proj and k_proj
        c_offs = []
        qk_pos = {}
        for c_off in offset_same_neurons:
            for n_in, layer_name, key in offset_same_neurons[c_off]:
                param_name = '.'.join([layer_name, key])
                if param_name.endswith(('q_proj.weight', 'k_proj.weight')):
                    if layer_name not in qk_pos:
                        qk_pos[layer_name] = edge_index[layer_name][key]
                elif param_name.endswith(('q_norm.weight', 'k_norm.weight')):
                    q_pos = qk_pos[layer_name.replace('_norm', '_proj')]
                    dim = (q_pos[0, :].max() - q_pos[0, :].min() + 1).item()
                    q_pos = q_pos[:, 0]
                    n_heads = self.num_heads
                    head_dim = edge_index[layer_name][key].shape[1]
                    if layer_name.endswith('q_norm'):
                        edge_index[layer_name][key][0, :] = torch.arange(q_pos[0] + dim + n_heads,
                                                                         q_pos[0] + dim + n_heads + head_dim)
                        edge_index[layer_name][key][1, :] = torch.arange(q_pos[1] + head_dim,
                                                                         q_pos[1] + head_dim + head_dim)
                        c_offs.append((c_off, q_pos[1] + head_dim))
                    else:
                        edge_index[layer_name][key][0, :] = torch.arange(q_pos[0], q_pos[0] + head_dim)
                        edge_index[layer_name][key][1, :] = torch.arange(q_pos[1] + head_dim - dim,
                                                                         q_pos[1] + head_dim + head_dim - dim)
                        c_offs.append((c_off, q_pos[1] + head_dim - dim))

        for c_off, c_off_new in c_offs:
            for n_in, layer_name, key in offset_same_neurons[c_off]:
                param_name = '.'.join([layer_name, key])
                if param_name.endswith('_norm.weight') or layer_name.endswith(('q_proj', 'k_proj', 'v_proj')):
                    if c_off_new not in offset_same_neurons:
                        offset_same_neurons[c_off_new] = []
                    offset_same_neurons[c_off_new].append(
                        (0 if param_name.endswith('_norm.weight') else n_in, layer_name, key))
                    offset_same_neurons[c_off].remove((n_in, layer_name, key))
        return edge_index, offset_same_neurons

def test_graph_qwen():
    """
    Test the NeuralGraph class for a small Qwen model.
    :return:
    """
    import transformers

    config = {
        'hidden_size': 12,
        'intermediate_size': 24,
        'num_hidden_layers': 2,
        'num_attention_heads': 6,
        'num_key_value_heads': 2,
        'head_dim': 2,
        'vocab_size': 15,
    }

    config = transformers.Qwen3Config(**config)
    print('config', config)
    model = transformers.AutoModelForCausalLM.from_config(config)
    # model = model.model.layers[0]  # can also work for a single transformer layer

    # can verify the correctness of the output and input dimensions in the graph
    # d = model.config.hidden_size
    # model.model.layers[0].input_layernorm.weight.data = torch.arange(d).to(model.model.layers[0].input_layernorm.weight)
    # model.model.layers[1].input_layernorm.weight.data = torch.arange(d).to(model.model.layers[1].input_layernorm.weight)
    # model.model.layers[0].self_attn.q_proj.weight.data[1, :] = torch.arange(d).to(
    #     model.model.layers[0].self_attn.q_proj.weight)
    # model.model.layers[1].self_attn.k_proj.weight.data[1, :] = torch.arange(d).to(
    #     model.model.layers[1].self_attn.k_proj.weight)

    # input_layernorm should go before self-attention, but we keep it as is for simplicity
    # potentially, it is possible to re-order the layers in the model

    graph = NeuralGraphQwen(model.named_parameters(),
                            num_heads=config.num_attention_heads,
                            num_key_value_heads=config.num_key_value_heads)
    run_test(model, graph, name='qwen')


if __name__ == '__main__':
    import sys
    test_graph_qwen()
    print('Done!')
