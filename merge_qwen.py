# Copyright (c) 2024-2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

This is experimental code that has not been well tested and was not used in the NiNo paper.

Merge Qwen models using NiNo or simple weight averaging.

NiNo:
    python merge_qwen.py meta_merge /path/to/save/merged/model

Weight averaging:
    python merge_qwen.py weight_avg /path/to/save/merged/model

The saved model can then be evaluated with lm-eval or other tools.

"""

import os
import sys
import torch


if __name__ == "__main__":

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from merge_vit import merge_nino

    if len(sys.argv) != 4:
        print("Usage: python merge_qwen.py <method> <model_base> <save_path>")
        print("Example: python merge_qwen.py meta_merge Qwen3-0.6B /path/to/save/merged/model")
        sys.exit(1)

    method = sys.argv[1]
    model_base = sys.argv[2]
    save_path = sys.argv[3]

    models, tokenizer = [], None
    if method == 'meta_merge':
        model_name = f'Qwen/{model_base}'
        print(f"Loading model: {model_name}")
        models.append(AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='cpu'))

    for model_name in [f'SamsungSAILMontreal/{model_base}-Math',
                       f'SamsungSAILMontreal/{model_base}-Fr']:
        print(f"Loading model: {model_name}")
        models.append(AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='cpu'))
        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)

    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
    except Exception as e:
        print(e, flush=True)

    if method == 'meta_merge':

        token_emb = 0
        for ind, model_ in enumerate(models):
            if ind > 0:
                # accumulate token embeddings of fine-tuned models for averaging later
                token_emb += model_.model.embed_tokens.weight.data.to('cpu').clone()
            # use only first 1k embeddings in the vocab to reduce memory usage in meta-merge
            model_.model.embed_tokens.weight.data = model_.model.embed_tokens.weight.data[:1024, :]

        model = merge_nino(models,
                           save_path,
                           k_range=range(1, 6),  # average over k=1..5
                           subgraph=True,
                           upd_scale=0.3,
                           edge_sample_ratio=0.05
                           )

        # average params of token embeddings of fine-tuned models (excluding the base model)
        model.model.embed_tokens.weight.data = token_emb.to(model.model.embed_tokens.weight.data.device) / (len(models) - 1)

    elif method == 'weight_avg':
        merged_state_dict = {}
        for key in models[0].state_dict().keys():
            merged_state_dict[key] = sum(model.state_dict()[key] for model in models) / len(models)
        merged_model = AutoModelForCausalLM.from_pretrained(models[0].name_or_path)
        merged_model.load_state_dict(merged_state_dict)  # Load the averaged state dict into the new model
    else:
        raise ValueError(f"Unknown merging method: {method}")

    try:
        tokenizer.save_pretrained(save_path)
    except Exception as e:
        print(e, flush=True)
        print('failed to save tokenizer to', save_path, flush=True)

    try:
        model.save_pretrained(save_path,
                              safe_serialization=True,
                              max_shard_size='10GB' if '-FP8' in save_path else '4GB')
        print('merged model saved to', save_path, flush=True)
    except Exception as e:
        print(e, flush=True)
        print('failed to save the merged model to', save_path, flush=True)
        raise
