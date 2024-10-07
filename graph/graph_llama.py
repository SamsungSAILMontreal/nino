# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Neural graph for Llama style transformers.

To test the NeuralGraph class with a Llama model, run:

    python graph/graph_llama.py [hugging_face_token -- optional]

In case of import errors, you can run it as a module:

    python -m graph.graph_llama [hugging_face_token -- optional]

"""
import torch
from .graph_transformer import NeuralGraphTransformer, run_test


class NeuralGraphLlama(NeuralGraphTransformer):
    """
    A neural graph for Llama style transformers.
    """

    _names = {
        'cls': 'embed_tokens.weight',
        'pos': ' ',
        'type': ' ',
        'attn_q': 'self_attn.q_proj.weight',
        'attn_k': 'self_attn.k_proj.weight',
        'attn_v': 'self_attn.v_proj.weight',
        'mlp_res1': 'mlp.gate_proj.weight',
        'mlp_res2': 'mlp.up_proj.weight',
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
                        heads = self.num_heads if sfx == 'q' else self.num_key_value_heads
                        for head in range(heads):
                            model_dict[name.replace('weight', f'head{head}')] = torch.Size(
                                ((sz[0] if self.model_first_dim_out else sz[1]) // heads,))  # out_dim
                    name_b = name.replace('weight', 'bias')
                    if name_b in self._model_dict:
                        model_dict[name_b] = self._model_dict[name_b]
                    if sfx == 'v':
                        model_dict[name.replace('weight', 'res')] = torch.Size(
                            (sz[1] if self.model_first_dim_out else sz[0],))  # in_dim

                elif not name.endswith(self._names[f'attn_{sfx}_bias']):
                    model_dict[name] = sz
                    # gate_proj and up_proj are connected to the same neurons,
                    # we model their connectivity using a residual connection
                    if name.endswith((self._names['mlp_res1'], self._names['mlp_res2'])):
                        key = name[name.rfind('.') + 1:]
                        sz_ = self._model_dict[name.replace(key, 'weight')]
                        model_dict[name.replace(key, 'res')] = torch.Size(
                            (sz_[1] if self.model_first_dim_out else sz_[0],))  # in_dim
        return model_dict

    def _move_offset(self, name, c_off, r_off, n_out, n_in):
        # gate_proj and up_proj perform element-wise operations on the same neurons,
        # so we need to shift the offsets to align them
        if name.endswith('gate_proj.weight'):
            c_off += n_in
        elif name.endswith('up_proj.weight'):
            c_off -= n_out
        return c_off, r_off


def test_graph_llama(token=None):
    """
    Test the NeuralGraph class for a small Llama model.
    :param token: Hugging Face token for downloading the model/config.
    :return:
    """
    import transformers

    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=True)
        config = transformers.AutoConfig.from_pretrained(
            'meta-llama/Meta-Llama-3.1-8B',
        )
        # make the model tiny for testing and visualization
        config.hidden_size = 12
        config.intermediate_size = 24
        config.num_hidden_layers = 2
        config.num_attention_heads = 6
        config.num_key_value_heads = 2
        config.vocab_size = 15
    else:
        config = {
            'hidden_size': 12,
            'intermediate_size': 24,
            'num_hidden_layers': 2,
            'num_attention_heads': 6,
            'num_key_value_heads': 2,
            'vocab_size': 15,
        }
        config = transformers.LlamaConfig(**config)
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

    graph = NeuralGraphLlama(model.named_parameters(),
                             num_heads=config.num_attention_heads,
                             num_key_value_heads=config.num_key_value_heads)
    run_test(model, graph, name='llama')


if __name__ == '__main__':
    import sys
    test_graph_llama(sys.argv[1] if len(sys.argv) > 1 else None)
    print('Done!')
