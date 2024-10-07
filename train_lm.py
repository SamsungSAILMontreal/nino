# Copyright (c) 2024. Samsung Electronics Co., Ltd.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Based on https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py

"""
Example usage:

    python train_lm.py --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
     --num_train_epochs 4 --layers 3 --dim 64 --heads 4 --nino_ckpt checkpoints/nino.pt

To train a Llama3-based model with Grouped-Query Attention using checkpoints/nino_no_posw.pt in this case,
since the tokenizer is different from GPT2 used to train NiNo (checkpoints/nino.pt):

    python train_lm.py --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
     --num_train_epochs 4 --layers 3 --dim 64 --heads 4 --heads_key_value 2 \
     --tokenizer_name meta-llama/Meta-Llama-3.1-8B --hf_login $HUGGING_FACE_TOKEN \
     --nino_ckpt checkpoints/nino_no_posw.pt

where $HUGGING_FACE_TOKEN is your Hugging Face token.

See more examples in the README.md file.

"""

#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import gc
import logging
import math
import time
import numpy as np
import os
import json
from pathlib import Path
from datetime import timedelta
import datasets
import torch
from itertools import chain
from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.logging import get_logger
from datasets import load_dataset
from huggingface_hub import login, HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler
)
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from optim import NiNo
from utils import set_seed, mem, get_env_args


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.33.0.dev0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the dataset directory.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Path to the dataset cache.",
    )
    parser.add_argument(
        "--nino_ckpt",
        type=str,
        default=None,
        help="path to the NiNo checkpoint.",
    )
    parser.add_argument(
        "--nino_device",
        type=str,
        default=None,
        help="NiNo device for parameter update prediction.",
    )
    parser.add_argument(
        "--nino_mp_device",
        type=str,
        default=None,
        help="NiNo's message passing device for parameter update prediction.",
    )
    parser.add_argument(
        "--hf_login",
        type=str,
        default=None,
        help="Hugging Face token for downloading the model/config."
    )
    parser.add_argument(
        "--target",
        type=float,
        default=147,
        help="target validation perplexity "
             "(default is 147 for the WikiText dataset and a 3 layer transformer with a hidden size 64).",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=1000,
        help="number of base opt steps after which to apply NiNo.",
    )
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="skip eval",
    )
    parser.add_argument(
        "--eval_freq",
        type=int,
        default=500,
        help="eval the model every 500 steps.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=12,
        help="transformer num of layers.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=64,
        help="transformer dimensionality.",
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
        help="transformer num of heads.",
    )
    parser.add_argument(
        "--heads_key_value",
        type=int,
        default=None,
        help="transformer num of key value heads (for Grouped-Query Attention).",
    )
    parser.add_argument(
        "--sample_config",
        action="store_true",
        help="sample config for dataset creation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run the script in the debugging mode.",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv, txt or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv, txt or a json file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='gpt2',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="constant",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default='no', help="Choose from 'no','fp16','bf16 or 'fp8'."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=0, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        type=bool,
        default=False,
        help=(
            "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
            "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
            "execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    args = get_env_args(args)
    return args


def main():
    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {'mixed_precision': args.mixed_precision}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    kwargs_handler = InitProcessGroupKwargs(timeout=timedelta(seconds=10000))
    print('kwargs_handlers timeout', kwargs_handler.timeout)
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps,
                              kwargs_handlers=[kwargs_handler],
                              **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
                # Create repo and retrieve repo_id
                api = HfApi()
                repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

                with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    if args.hf_login:
        login(token=args.hf_login, add_to_git_credential=True)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name,
                                    args.dataset_config_name,
                                    data_dir=args.data_dir,
                                    cache_dir=args.cache_dir)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]" if args.debug else
                f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.config_name:
        config = AutoConfig.from_pretrained(
            args.config_name,
            trust_remote_code=args.trust_remote_code,
        )
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        if args.sample_config:
            layers = int(np.random.randint(4, 10))
            dim_max = 64 if layers > 6 else 96
            dim = int(np.random.choice(np.arange(32, dim_max + 1, 32)))
            heads = int(np.random.choice([4, 12 if (dim % 12) == 0 else 8]))
        else:
            layers = args.layers
            dim = args.dim
            heads = args.heads

        config = transformers.AutoConfig.from_pretrained(
            args.tokenizer_name,
        )

        if args.tokenizer_name == 'gpt2':
            print('GPT2Config:\t', 'dim', dim, 'layers', layers, 'heads', heads, flush=True)
            config.n_embd = dim
            config.n_layer = layers
            config.n_head = heads

        elif args.tokenizer_name.lower().find('llama') >= 0:
            print('LlamaConfig:\t', 'dim', dim, 'layers', layers, 'heads', heads, flush=True)
            # make the model tiny for testing and visualization
            config.hidden_size = dim
            config.intermediate_size = dim * 4
            config.num_hidden_layers = layers
            config.num_attention_heads = heads
            config.num_key_value_heads = heads // 2 if args.heads_key_value is None else args.heads_key_value
            print('\nCONFIG:', config)

        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
        logger.info(f"Training new model from scratch")

    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], truncation=True, max_length=1024)

    print('block_size', args.block_size, 'tokenizer.model_max_length', tokenizer.model_max_length, flush=True)

    with accelerator.main_process_first():
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with accelerator.main_process_first():
        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc=f"Grouping texts in chunks of {block_size}",
        )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    # Log a few random samples from the training set:
    # for index in random.sample(range(len(train_dataset)), 3):
    #     logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size,
        num_workers=4 if args.dataset_name == 'wikitext' else 8
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size,
        num_workers=4 if args.dataset_name == 'wikitext' else 8
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    optimizer = NiNo(base_opt=optimizer,
                     ckpt=args.nino_ckpt,
                     model=model,
                     period=args.period,
                     max_train_steps=args.max_train_steps,
                     nino_device=args.nino_device,
                     message_passing_device=args.nino_mp_device,
                     amp=args.mixed_precision != 'no',
                     verbose=args.verbose)  # haven't tested with distributed training

    if args.output_dir is not None:
        checkpoint_last = os.path.join(args.output_dir,
                                       'checkpoint_epoch%d_step%d.pt' % (args.num_train_epochs,
                                                                         args.max_train_steps))
        print('checkpoint_last', checkpoint_last)
        if os.path.isfile(checkpoint_last):
            print('checkpoint_last already exists, exiting', checkpoint_last)
            exit(0)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("clm_no_trainer", experiment_config)

    def get_eval_loss(model, epoch_, step_, total_step_):
        print('running eval... mem=%.4fG ' % mem(accelerator.device), flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        model.eval()
        loss_ = []
        for _, batch_ in enumerate(eval_dataloader):
            with torch.no_grad():
                outputs_ = model(**batch_)
            loss_.append(accelerator.gather_for_metrics(outputs_.loss.repeat(
                args.per_device_eval_batch_size)).data.cpu())
        loss_ = torch.cat(loss_)
        try:
            eval_loss_ = torch.mean(loss_)
            perplexity_ = math.exp(eval_loss_)
        except OverflowError:
            eval_loss_ = float("inf")
            perplexity_ = float("inf")

        n = len(loss_)
        if torch.cuda.is_available():
            loss_ = None
            torch.cuda.empty_cache()
            gc.collect()  # some unclear memory leak, doing manual clean up

        print(f"epoch {epoch_ + 1}, step {step_ + 1}, total step {total_step_ + 1}: "
                    f"perplexity: {perplexity_} eval_loss: {eval_loss_}, "
                    f"eval batches: {n}", flush=True)
        return eval_loss_, perplexity_

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    starting_epoch = 0
    completed_steps = 0
    resume_step = None
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if not os.path.exists(args.resume_from_checkpoint):
            print(f"\nWARNING: Resume path {args.resume_from_checkpoint} not found")
        else:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
                checkpoint_path = path
                path = os.path.basename(checkpoint_path)

            accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
            accelerator.load_state(checkpoint_path)
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                completed_steps = starting_epoch * num_update_steps_per_epoch
            else:
                # need to multiply `gradient_accumulation_steps` to reflect real steps
                resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len(train_dataloader)
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len(train_dataloader)

            optimizer.step_idx = completed_steps
            print(f'Model and opt loaded from {checkpoint_path}, '
                  f'starting_epoch={starting_epoch}, resume_step={resume_step}, total_step={optimizer.step_idx}')
            if not args.skip_eval:
                eval_loss, eval_ppl = get_eval_loss(model, starting_epoch, resume_step, completed_steps)
                if args.target is not None and eval_ppl <= args.target:
                    print("\nModel already reached target of {:.2f}<={:.2f}. Exiting...".format(
                        eval_ppl, args.target))
                    return

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    print(model,
          '\nTotal size={:.2f}M params'.format(n_params / 10 ** 6),
          '\nTotal param norm={:.4f}'.format(
          torch.norm(torch.stack([torch.norm(p.data, 2) for p in model.parameters()]), 2).item()))

    def save(step_idx=None):
        if (accelerator.is_main_process and args.output_dir not in [None, '', 'None', 'none'] and
                accelerator.sync_gradients):
            accelerator.wait_for_everyone()
            if step_idx:
                output_dir = os.path.join(args.output_dir, f"step_{step_idx}")
            else:
                output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")

            if not os.path.isdir(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            accelerator.save_state(output_dir)

            print(f'Model and optimizer saved to {output_dir} at '
                  f'epoch={epoch}, '
                  f'step={step}, '
                  f'completed_steps={completed_steps}', flush=True)
            if step_idx is None:
                tokenizer.save_pretrained(args.output_dir)

    eval_losses = {}
    losses = []
    done = False
    start_time = time.time()
    with accelerator.autocast():
        for epoch in range(starting_epoch, args.num_train_epochs):
            model.train()
            if args.with_tracking:
                total_loss = 0
            if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
                # We skip the first `n` batches in the dataloader when resuming from a checkpoint
                active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
            else:
                active_dataloader = train_dataloader
            for step, batch in enumerate(active_dataloader):

                if not args.skip_eval and (completed_steps + 1) % args.eval_freq == 0 and completed_steps not in eval_losses:
                    # eval before the step because after the NiNo step the performance can drop for a few iterations
                    eval_loss, eval_ppl = get_eval_loss(model, epoch, step, completed_steps)
                    eval_losses[completed_steps] = eval_loss.item()
                    if args.target is not None and eval_ppl <= args.target:
                        print('\nReached target perplexity of {:.2f}<={:.2f} in {} steps '
                              '({:.4f} seconds)'.format(eval_ppl,
                                                        args.target,
                                                        completed_steps,
                                                        time.time() - start_time))
                        done = True
                        break

                model.train()
                with accelerator.accumulate(model):
                    if optimizer.need_grads:
                        loss = model(**batch).loss
                        # We keep track of the loss at each epoch
                        if args.with_tracking:
                            total_loss += loss.detach().float()
                        accelerator.backward(loss)  # only compute gradients for the base optimizer
                        closure = None
                        losses.append(loss.item())
                    else:
                        # prediction step
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # manually free the memory before the NiNo step
                        def closure():
                            # eval the loss after the NiNo step to see how it affects the training
                            with torch.no_grad():
                                return model(**batch).loss

                    loss_ = optimizer.step(closure)  # base_opt step or nowcast params every args.period steps using NiNo
                    if loss_ is not None:
                        losses.append(loss_.item())
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()  # manually free the memory after the NiNo step

                    if np.isnan(losses[-1]):
                        raise ValueError("NaN detected!", "loss", losses[-1])

                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                if isinstance(checkpointing_steps, int) and completed_steps % checkpointing_steps == 0:
                    save(completed_steps)
                if completed_steps >= args.max_train_steps:
                    done = True
                if done:
                    break
            if done:
                break

    # eval at the end of training
    if not args.skip_eval and completed_steps not in eval_losses:
        eval_loss, eval_ppl = get_eval_loss(model, epoch, step, completed_steps)
        eval_losses[completed_steps] = eval_loss.item()
        if args.target is not None and eval_ppl <= args.target:
            print('\nReached target perplexity of {:.2f}<={:.2f} in {} steps '
                  '({:.4f} seconds)'.format(eval_ppl,
                                            args.target,
                                            completed_steps,
                                            time.time() - start_time))

    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        save(completed_steps)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"eval_losses": eval_losses}, f)


if __name__ == "__main__":
    main()
