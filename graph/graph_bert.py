# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural graph for BERT style transformers.
This is experimental code that has not been well tested and was not used in the NiNo paper.

To test the NeuralGraph class with a BERT model, run:

    python graph/graph_bert.py

In case of import errors, you can run it as a module:

    python -m graph.graph_bert

"""
import torch
from .graph_transformer import NeuralGraphTransformer, run_test


class NeuralGraphBERT(NeuralGraphTransformer):

    _names = {
        'cls': 'word_embeddings.weight',
        'pos': 'position_embeddings.weight',
        'type': 'token_type_embeddings.weight',
        'attn_q': 'attention.self.query.weight',
        'attn_k': 'attention.self.key.weight',
        'attn_v': 'attention.self.value.weight',
        'mlp_res': 'intermediate.dense.bias',
    }
    for n in ['_q', '_k', '_v']:
        _names[f'attn{n}_bias'] = _names[f'attn{n}'].replace('weight', 'bias')
        _names[f'layer{n}'] = _names[f'attn{n}'][:_names[f'attn{n}'].rfind('.')]
        if n == '_v':
            _names['value_res'] = _names[f'attn{n}'].replace('weight', 'res')


    def _update_model_dict(self):
        model_dict = {}
        for name, sz in self._model_dict.items():
            for i, sfx in enumerate(['q', 'k', 'v']):
                if name.endswith(self._names[f'attn_{sfx}']):
                    model_dict[name] = sz
                    if sfx in ['q', 'v']:
                        for head in range(self.num_heads):
                            model_dict[name.replace('weight', f'head{head}')] = torch.Size(
                                ((sz[0] if self.model_first_dim_out else sz[1]) // self.num_heads,))  # out_dim
                    name_b = name.replace('weight', 'bias')
                    if name_b in self._model_dict:
                        model_dict[name_b] = self._model_dict[name_b]
                    if sfx == 'v':
                        model_dict[name.replace('weight', 'res')] = torch.Size(
                            (sz[1] if self.model_first_dim_out else sz[0],))  # in_dim

                elif not name.endswith(self._names[f'attn_{sfx}_bias']):
                    model_dict[name] = sz
                    if name.endswith(self._names['mlp_res']):
                        key = name[name.rfind('.') + 1:]
                        sz_ = self._model_dict[name.replace(key, 'weight')]
                        model_dict[name.replace(key, 'res')] = torch.Size(
                            (sz_[1] if self.model_first_dim_out else sz_[0],))  # in_dim
        return model_dict

    def _move_offset(self, name, c_off, r_off, n_out, n_in):
        if name.endswith('cls.predictions.bias'):
            r_off -= 1
        return c_off, r_off

    def _correct_offsets(self, edge_index, offset_same_neurons):
        # treat the special case of the cls.predictions.bias layer
        # connect it to bert.embeddings.word_embeddings.weight that has the same neurons
        layer_name = 'cls.predictions'
        key = 'bias'
        col_offset = 0
        for c_off in sorted(offset_same_neurons.keys()):
            col_offset = c_off + sum([c[0] for c in offset_same_neurons[c_off]])
            break

        if layer_name in edge_index and key in edge_index[layer_name]:
            edge_index[layer_name][key][0, :] = max(0, col_offset - 1)
            edge_index[layer_name][key][1, :] -= edge_index[layer_name][key][1, 0].item()
            edge_index[layer_name][key] = edge_index[layer_name][key][[1, 0], :]

            for layer_name in edge_index:
                for key in edge_index[layer_name]:
                    edge_index[layer_name][key][1, :] += 1
                    if (layer_name.find('bert.embeddings') < 0 and
                            not layer_name.endswith('cls.predictions') and
                            not (layer_name.endswith('bert.encoder.layer.0.attention.self.key') and key == 'bias')):
                        edge_index[layer_name][key][0, :] += 1

            for c_off in offset_same_neurons:
                # list offset_same_neurons[c_off] contains tuples (n_in, layer_name, key)
                # remove the item from the offset_same_neurons[c_off] list where layer_name == 'cls.predictions' and key == 'bias'
                offset_same_neurons[c_off] = [(n_in, layer_name, key) for n_in, layer_name, key in offset_same_neurons[c_off]
                                              if not (layer_name == 'cls.predictions' and key == 'bias')]


        return edge_index, offset_same_neurons


def test_graph_bert():
    """
    Test the NeuralGraph class for a small BERT model.
    :return:
    """
    import transformers

    config = {
        'hidden_size': 6,
        'intermediate_size': 24,
        'num_hidden_layers': 2,
        'num_attention_heads': 3,
        'vocab_size': 10,
        'max_position_embeddings': 7,
    }
    config = transformers.BertConfig(**config)
    print('config', config)
    model = transformers.AutoModelForCausalLM.from_config(config)
    # model = model.bert.encoder.layer[0]  # can also work for a single transformer layer

    # can verify the correctness of the output and input dimensions in the graph
    # model.cls.predictions.bias.data = torch.arange(10).to(model.cls.predictions.bias)
    # model.bert.embeddings.word_embeddings.weight.data = torch.rand(60).view_as(
    #     model.bert.embeddings.word_embeddings.weight).to(model.bert.embeddings.word_embeddings.weight)
    # model.bert.encoder.layer[0].attention.self.query.weight.data[2, :] = torch.arange(6).to(model.bert.encoder.layer[0].attention.self.query.weight.data)
    # model.bert.encoder.layer[1].attention.self.key.weight.data[2, :] = torch.arange(6).to(model.bert.encoder.layer[1].attention.self.key.weight.data)

    graph = NeuralGraphBERT(model.named_parameters(), num_heads=config.num_attention_heads)
    run_test(model, graph, name='bert')


if __name__ == '__main__':
    test_graph_bert()
    print('Done!')
