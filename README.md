from ghn3.bert import outputfrom transformers import AutoModelForPreTraining

# Accelerating Training with Neuron Interaction and Nowcasting Networks

<pre>
<a href="https://arxiv.org/abs/2409.04434/">Accelerating Training with Neuron Interaction and Nowcasting Networks</a>
<a href="http://bknyaz.github.io/">Boris Knyazev</a>, <a href="https://amoudgl.github.io/">Abhinav Moudgil</a>, <a href="https://www.guillaumelajoie.com/">Guillaume Lajoie</a>, <a href="https://eugenium.github.io/">Eugene Belilovsky</a>, <a href="https://www.iro.umontreal.ca/~slacoste/">Simon Lacoste-Julien</a>,

</pre>

[![arXiv](https://img.shields.io/badge/arXiv-2403.12143-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2409.04434)

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

- [x] Initial code release with a pretrained NiNo model.
- [x] Neural graphs and evaluation script for vision tasks.
- [x] Neural graphs and evaluation script for language tasks with transformers.
- [ ] Training code for NiNo. 

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
