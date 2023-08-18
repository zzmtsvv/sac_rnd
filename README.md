<!-- https://wandb.ai/zzmtsvv/sac_rnd/runs/d03hrwpr?workspace=user-zzmtsvv -->

# Anti-Exploration by Random Network Distillation on PyTorch

This repository contains possible (not ideal one actually) PyTorch implementation of [SAC RND](https://arxiv.org/abs/2301.13616) with the [wandb](https://wandb.ai/zzmtsvv/sac_rnd?workspace=user-zzmtsvv) integration. It is based on [official realization](https://github.com/tinkoff-ai/CORL/blob/howuhh/sac-rnd/algorithms/sac_rnd_jax.py) written on Jax.

## Setup
In order to be able to run code, just install the requirements:
```
python install -r requirements.txt
```
Anyway, you would also need to install mujoco stuff by your own, you can follow [the steps from the authors](https://github.com/tinkoff-ai/sac-rnd)

if you want to train the model, setup `rnd_config` in `config.py`, initialize `SACRNDTrainer` in `trainer.py` and run its `train` method:
```python3
from trainer import SACRNDTrainer

trainer = SACRNDTrainer()
trainer.train()
```
if you find any bugs and mistakes in the code, please contact me :)

