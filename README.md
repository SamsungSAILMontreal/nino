# Accelerating Training with Neuron Interaction and Nowcasting Networks

<pre>
<a href="https://arxiv.org/abs/2409.04434/">Accelerating Training with Neuron Interaction and Nowcasting Networks</a>
<a href="http://bknyaz.github.io/">Boris Knyazev</a>, <a href="https://amoudgl.github.io/">Abhinav Moudgil</a>, <a href="https://www.guillaumelajoie.com/">Guillaume Lajoie</a>, <a href="https://eugenium.github.io/">Eugene Belilovsky</a>, <a href="https://www.iro.umontreal.ca/~slacoste/">Simon Lacoste-Julien</a>

</pre>

[![arXiv](https://img.shields.io/badge/arXiv-2403.12143-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2409.04434)

[marktechpost](https://www.marktechpost.com/2024/09/17/nino-a-novel-machine-learning-approach-to-accelerate-neural-network-training-through-neuron-interaction-and-nowcasting/)

## Intro

**Neuron interaction and Nowcasting (NiNo) model** 

We introduce the NiNo model predicting future (nowcasting) parameters by learning neuron interaction 
in vision and language tasks. 
We feed c (c=5 by default) past parameter states as input to NiNo and 
nowcast K states leveraging [neural graph structure](https://github.com/mkofinas/neural-graphs) and graph neural networks.
For a new optimization task, NiNo is applied rarely over time: only once per 1k steps of Adam 
(or another base optimizer).
<figure> <img src="figs/fig_main.png" height="250"></figure>

**Using NiNo with Adam**

Adam without and with nowcasting using our NiNo model on a language task that NiNo has not seen during its training.
<figure> <img src="figs/fig_intro.png" height="250"></figure>



## Requirements

The experiments from our paper can be run using a single GPU with <= 80GB of memory.

- python >= 3.8
- pytorch >= 2.0
- torch_geometric
- transformers
- datasets
- other optional dependencies (networkx, pydot)

## Updates

- [x] Initial code release with a pretrained NiNo model (see the [`checkpoints`](checkpoints) folder).
  - [x] `nino.pt` - default NiNo model (assume the GPT2 tokenizer)
  - [x] `nino_no_posw.pt` - NiNo without positional encoding for word embeddings (can be used for arbitrary models and tokenizers including Llama)
  - [x] `nino_h32.pt` - NiNo with hidden size 32 instead of default 128
  - [x] `nino_mlp.pt` - WNN+ model (does not use graphs)
  - [x] `nino_towers4.pt` - NiNo with 4 towers in the message passing step for better efficiency
- [x] Neural graphs and evaluation script for convnet tasks.
- [x] Neural graphs and evaluation script for transformer tasks:
  - [x] GPT2
  - [x] BERT (experimental code)
  - [x] Llama (experimental code, see a graph for a smaller variant of `meta-llama/Meta-Llama-3.1-8B` in the [`results`](results) folder)
  - [x] Vision Transformer (experimental code)
- [ ] Training dataset and training code for NiNo. 

## Pretrained NiNo models

We provide the checkpoint for our best performing NiNo model at `checkpoints/nino.pt`.
 
## Usage

### Example

Training loop with NiNo for some language model:

```python
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from optim import NiNo

model = AutoModelForCausalLM.from_config(...)  # some model

# NiNo is implemented as a wrapper around the base optimizer
# any optimizer other than Adam should also be possible to use with NiNo
opt = NiNo(base_opt=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2),
           ckpt='checkpoints/nino.pt',
           message_passing_device=None,  # can use 'cpu' when NiNo is applied to larger models 
           model=model,
           period=1000,
           max_steps=10000)
for step in range(10000):
    if opt.need_grads:  # True/False based on the step number and period
        opt.zero_grad()  # zero out gradients
        data, targets = ...  # get some batch of data
        # base optimizer step (majority of the time)
        outputs = model(data)  # forward pass
        loss = F.cross_entropy(outputs, targets)  # compute some loss
        loss.backward()  # only compute gradients for the base optimizer            
    opt.step()  # base_opt step or nowcast params every 1000 steps using NiNo    
    ...
```

### Reproducing the results from our paper

#### Optimization in vision tasks

Evaluate on all vision tasks:
```commandline
for task in FM-16 C10-16 FM-32 C10-32 C100-32; 
do for seed in $(seq 1000 1000 10000); 
do python train_vision.py --task $task --seed $seed | tee -a results.log; done; done
```

To evaluate without the NiNo model, run with `--nino_ckpt none`.
You should get the results similar Table 1 and 2 in the paper.

Use `--verbose 2` for graph visualization and more detailed output.

#### Optimization in language tasks

Single seed training on the Wiki/3-64 task:

```commandline
python train_lm.py --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 --num_train_epochs 4 --layers 3 --dim 64 --heads 4
```

For LM1B tasks, use `--dataset_name lm1b --dataset_config_name plain_text`.

## Contributing

Pull requests and github issues are welcome. For major changes, please open an issue first to discuss what you would like to change.

## LICENSE

MIT, see the [LICENSE](LICENSE) file.


## Citation

```
@misc{knyazev2024accelerating,
  title={Accelerating Training with Neuron Interaction and Nowcasting Networks}, 
  author={Boris Knyazev and Abhinav Moudgil and Guillaume Lajoie and Eugene Belilovsky and Simon Lacoste-Julien},
  year={2024},
  eprint={2409.04434},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2409.04434}, 
}
```
