# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
To see the network architecture in each task, run:

    python utils/lm.py

"""

from transformers import AutoConfig, AutoModelForCausalLM


LM_TASKS = {

    'LM1B-3-24': {
        'net_args': {'n_embd': 24, 'n_layer': 3, 'n_head': 3},
        'dataset': 'lm1b',
        'cfg': 'plain_text',
        'tokenizer': 'gpt2',
        'lr': 0.0002,
        'epochs': 1,
        'target': 352
    },

    'LM1B-2-32': {
        'net_args': {'n_embd': 32, 'n_layer': 2, 'n_head': 2},
        'dataset': 'lm1b',
        'cfg': 'plain_text',
        'tokenizer': 'gpt2',
        'lr': 0.0002,
        'epochs': 1,
        'target': 319
    },

    'LM1B-3-64': {
        'net_args': {'n_embd': 64, 'n_layer': 3, 'n_head': 4},
        'dataset': 'lm1b',
        'cfg': 'plain_text',
        'tokenizer': 'gpt2',
        'lr': 0.0002,
        'epochs': 1,
        'target': 181
    },

    'WIKI-3-64': {
        'net_args': {'n_embd': 64, 'n_layer': 3, 'n_head': 4},
        'dataset': 'wikitext',
        'cfg': 'wikitext-103-raw-v1',
        'tokenizer': 'gpt2',
        'lr': 0.0002,
        'epochs': 4,
        'target': 147
    },

    'WIKI-4-128': {
        'net_args': {'n_embd': 128, 'n_layer': 4, 'n_head': 4},
        'dataset': 'wikitext',
        'cfg': 'wikitext-103-raw-v1',
        'tokenizer': 'gpt2',
        'lr': 0.0002,
        'epochs': 4,
        'target': 77
    },

    'WIKI-6-384': {
        'net_args': {'n_embd': 384, 'n_layer': 6, 'n_head': 6},
        'dataset': 'wikitext',
        'cfg': 'wikitext-103-raw-v1',
        'tokenizer': 'gpt2',
        'lr': 0.0002,
        'epochs': 4,
        'target': 38
    },

    'WIKI-6-384-llama': {
        'net_args': {'hidden_size': 384, 'num_hidden_layers': 6,
                     'num_attention_heads': 6, 'num_key_value_heads': 2, 'intermediate_size': 384 * 4},
        'dataset': 'wikitext',
        'cfg': 'wikitext-103-raw-v1',
        'tokenizer': 'meta-llama/Meta-Llama-3.1-8B',
        'lr': 0.0002,
        'epochs': 4,
        'target': 24
    }

}

# test the code
if __name__ == '__main__':
    for task, args in LM_TASKS.items():
        # might need to log in to huggingface using the user token (see train_lm.py for example)
        config = AutoConfig.from_pretrained(args['tokenizer'], **args['net_args'])
        model = AutoModelForCausalLM.from_config(config)
        print(f'\nTASK={task}, dataset={args["dataset"]}')
        print('config', config)
        print(type(model), 'params={:.2f}M'.format(sum({p.data_ptr():
                                                           p.numel() for p in model.parameters()}.values()) / 10 ** 6))

    print('Done!')

