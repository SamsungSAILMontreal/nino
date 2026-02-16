# Copyright (c) 2024-2026. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""

This is experimental code that has not been well tested and was not used in the NiNo paper.

This script merges two or more ViT models fine-tuned on different tasks using pretrained NiNo.
The script also compares the results to task arithmetic.

Setup:

    Put the ViT checkpoints in the ./task_vectors/checkpoints/ folder as per the instructions on
    https://github.com/mlfoundations/task_vectors?tab=readme-ov-file#checkpoints

    See https://github.com/mlfoundations/task_vectors/issues/1 for dataset preprocessing instructions.
    Specifically, for DTD after downloading and extracting the dataset, I ran the following commands:

    for f in $(cat dtd/labels/train1.txt); do class=$(dirname "$f"); mkdir -p train/$class; cp dtd/images/$f train/$class/; done
    for f in $(cat dtd/labels/val1.txt); do class=$(dirname "$f"); mkdir -p train/$class; cp dtd/images/$f train/$class/; done
    for f in $(cat dtd/labels/test1.txt); do class=$(dirname "$f"); mkdir -p val/$class; cp dtd/images/$f val/$class/; done

Example usage:

    # 2 tasks:
    python merge_vit.py --eval-datasets DTD,RESISC45 --data-location ../data

    # 4 tasks:
    python merge_vit.py --eval-datasets DTD,RESISC45,MNIST,SVHN --data-location ../data

"""

import torch
import os
import sys


@torch.no_grad()
def merge_nino(models, save_path, k_range=range(1,6), **kwargs):

    from optim import NiNo

    assert len(models) >= 3, (f'At least three models are required (got {len(models)} instead): '
                              f'pretrained/base model, fine-tuned model 1, fine-tuned model 2, etc.')

    opt = NiNo(base_opt=None,
               model=None,
               ckpt='./checkpoints/nino_no_posw.pt',
               verbose=1,
               nino_device='auto',
               chunk_size=int(10 ** 6),
                **kwargs)

    graph_feat_path = os.path.join(save_path, 'graph_feat.pt')
    if os.path.exists(graph_feat_path):
        print('loading cached graph lpe from', graph_feat_path, flush=True)
        lpe = torch.load(graph_feat_path)
        print('loaded graph lpe', f'{len(lpe)} blocks' if isinstance(lpe, list) else lpe.shape)
    else:
        lpe = None

    opt.set_model(models[0], lpe=lpe)  # construct the neural graph based on the pretrained model's structure
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

    results = []
    for k in k_range:
        params_tasks = []
        for model_ in models:
            params_tasks.append(torch.cat([p.data.view(-1).to('cpu').clone() for n, p in model_.named_parameters()]))
        if len(models) == 3:
            # merging two tasks
            opt.states.extend([params_tasks[i] for i in [0,1,2,2,1]])  # some heuristic order that works well
        elif len(models) == 4:
            # merging three tasks
            opt.states.extend([params_tasks[i] for i in [0,1,2,2,3]])  # some heuristic order (not tested)
        elif len(models) == 5:
            # merging four tasks
            opt.states.extend(params_tasks)  # the order as is, can be optimized
        else:
            raise NotImplementedError(f'Merging {len(models)} tasks is not supported because NiNo has 5 features.')
        opt.step(k=k)
        results.append(opt._model.state_dict().copy())  # store the merged model for this k
    # now average the resulted models from different k values
    merged_state_dict = {}
    for key in results[0].keys():
        merged_state_dict[key] = sum(result[key] for result in results) / len(results)
    opt._model.load_state_dict(merged_state_dict)  # Load the averaged state dict into the model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return opt._model


if __name__ == '__main__':

    sys.path.append('./task_vectors')
    from task_vectors.src.task_vectors import TaskVector
    from task_vectors.src.eval import eval_single_dataset
    from task_vectors.src.args import parse_arguments

    # Config
    args = parse_arguments()
    model = args.model
    print('model:', model)
    datasets = args.eval_datasets # ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']
    print('datasets to merge:', datasets)
    assert 1 < len(datasets) < 5, f'This script currently supports merging 2-4 tasks only ({len(datasets)} provided).'
    ckpt_path = './task_vectors/checkpoints'
    args.save = f'{ckpt_path}/{model}'
    pretrained_checkpoint = f'{ckpt_path}/{model}/zeroshot.pt'

    # Baseline task vectors/weight averaging
    task_vectors = []
    for dataset in datasets:
        task_vectors.append(TaskVector(pretrained_checkpoint,
                                       f'{ckpt_path}/{model}/{dataset}/finetuned.pt'))

    image_encoder = task_vectors[0].apply_to(pretrained_checkpoint, scaling_coef=0)
    print('\nevaluating the pretrained model (zero-shot acc)...')
    acc = []
    for dataset in datasets:
        acc.append(eval_single_dataset(image_encoder, dataset, args)['top1'])
    print(f'Zero-shot Accuracy: {100 * sum(acc) / len(acc):.2f}%')

    print('\nevaluating the finetuned models individually...')
    for task in range(len(task_vectors)):
        image_encoder = task_vectors[task].apply_to(pretrained_checkpoint, scaling_coef=1)
        acc = []
        for dataset in datasets:
            acc.append(eval_single_dataset(image_encoder, dataset, args)['top1'])
        print(f'Task Vector {datasets[task]} Accuracy: {100 * sum(acc) / len(acc):.2f}%')

    task_vector_sum = sum(task_vectors)
    image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=1/(len(task_vectors)))  # weight averaging
    print('\nevaluating the baseline task vectors merging...')
    acc = []
    for dataset in datasets:
        acc.append(eval_single_dataset(image_encoder, dataset, args)['top1'])
    print(f'Merged Task Vectors Accuracy: {100 * sum(acc) / len(acc):.2f}%')

    # NiNo merging
    with torch.no_grad():
        pretrained = torch.load(pretrained_checkpoint, weights_only=False)
        models = [pretrained.model.visual]

        unsupported_params = {}  # special handling for these layers
        for n, p in pretrained.named_parameters():
            if 'visual.' not in n or 'conv1.weight' in n:
                unsupported_params[n] = 0
        print('unsupported_params', unsupported_params)

        for dataset in datasets:
            model_ = torch.load(f'./task_vectors/checkpoints/{model}/{dataset}/finetuned.pt', weights_only=False)
            models.append(model_.model.visual)
            for key in unsupported_params:
                unsupported_params[key] += model_.state_dict()[key].clone()

        for model_ in models:
            # downsample model.visual.conv1.weight of shape 16x16 to 3x3 by using pytorch bilinear interpolation
            # this is necessary, since the NiNo models accepts maximum 9 features per edge
            model_.conv1.weight.data = torch.nn.functional.interpolate(
                model_.conv1.weight.data, size=(3, 3), mode='bilinear', align_corners=False)

        # apply NiNo merging only on the visual encoder for now
        pretrained.model.visual = merge_nino(models,
                                             args.save,
                                             k_range=range(1, 6),  # average over k=1..5
                                             subgraph=False, # ViT models fit in memory without using subgraphs
                                             upd_scale=2.5,  # this can be tuned for better merging performance
                                             edge_sample_ratio=0.2,  # can reduce to fit in memory
                                             )

        # average params of other layers (conv1, token_emb, ln_final weights)
        for n, p in pretrained.named_parameters():
            if n in unsupported_params:
                print('averaging unsupported param', n, p.shape)
                p.data = unsupported_params[n] / len(datasets)

    print('\nevaluating NiNo-merged task vectors...')
    acc = []
    for dataset in datasets:
        acc.append(eval_single_dataset(pretrained, dataset, args)['top1'])
    print(f'NiNo Merged Task Vectors Accuracy: {100 * sum(acc) / len(acc):.2f}%')
