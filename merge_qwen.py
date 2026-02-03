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
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_nino(models, save_path):

    from optim import NiNo

    assert len(models) >= 3, (f'At least three models are required (got {len(models)} instead): '
                              f'pretrained/base model, fine-tuned model 1, fine-tuned model 2, etc.')

    opt = NiNo(base_opt=None,
               model=None,
               ckpt='./checkpoints/nino_no_posw.pt',
               verbose=1,
               subgraph=True,
               upd_scale=0.1,
               edge_sample_ratio=0.05,
               nino_device='auto',
               chunk_size=int(10**6))

    graph_feat_path = os.path.join(save_path, 'graph_feat.pt')
    if os.path.exists(graph_feat_path):
        print('loading cached graph lpe from', graph_feat_path, flush=True)
        lpe = torch.load(graph_feat_path)
        print('loaded graph lpe', f'{len(lpe)} blocks' if isinstance(lpe, list) else lpe.shape)
    else:
        lpe = None

    opt.set_model(models[0], lpe=lpe)  # construct the neural graph
    if not os.path.exists(graph_feat_path):
        # cache the lpe for future reuse
        if isinstance(opt.graph, list):
            pos_lst = []
            for g in opt.graph:
                if hasattr(g.pyg_graph, 'pos') and g.pyg_graph.pos is not None:
                    pos_lst.append(g.pyg_graph.pos)
            print(f'saving graph lpe {len(pos_lst)}-{pos_lst[0].shape} to', graph_feat_path, flush=True)
            torch.save(pos_lst, graph_feat_path)
        else:
            if hasattr(opt.graph.pyg_graph, 'pos') and opt.graph.pyg_graph.pos is not None:
                print(f'saving graph lpe {opt.graph.pyg_graph.pos.shape} to', graph_feat_path, flush=True)
                torch.save(opt.graph.pyg_graph.pos, graph_feat_path)

    params_tasks = []
    for model_ in models:
        params_tasks.append(torch.cat([p.data.view(-1).to('cpu') for n, p in model_.named_parameters()]))

    if len(models) == 3:
        # merging two tasks
        opt.states.extend([params_tasks[i] for i in [0, 1, 2, 2, 1]])
    elif len(models) == 4:
        # merging three tasks
        opt.states.extend([params_tasks[i] for i in [0, 1, 2, 2, 3]])
    elif len(models) == 5:
        # merging four tasks
        opt.states.extend(params_tasks)
    else:
        raise NotImplementedError(f'Merging {len(models)} tasks is not supported because NiNo has 5 features.')
    opt.step(k=5)
    return opt._model

def model_merge(models, method='weight_avg'):
    if method == 'weight_avg':
        merged_state_dict = {}
        for key in models[0].state_dict().keys():
            merged_state_dict[key] = sum(model.state_dict()[key] for model in models) / len(models)
        merged_model = AutoModelForCausalLM.from_pretrained(models[0].name_or_path)
        merged_model.load_state_dict(merged_state_dict)  # Load the averaged state dict into the new model
    else:
        raise ValueError(f"Unknown merging method: {method}")
    return merged_model

if __name__ == "__main__":

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
            os.makedirs(save_path)
    except Exception as e:
        print(e, flush=True)

    if method == 'meta_merge':
        model = merge_nino(models, save_path)
    else:
        model = model_merge(models, method=method)

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
