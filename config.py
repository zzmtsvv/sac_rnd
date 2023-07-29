import torch
from dataclasses import dataclass


@dataclass
class rnd_config:
    project: str = "sac_rnd"
    group: str = "first_attempt"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path: str = "weights/sac_rnd.pt"

    actor_lr: float = 1e-3
    edac_init: bool = False
    critic_lr: float = 1e-3
    beta_lr: float = 1e-3
    hidden_dim: int = 256
    gamma: float = 0.99
    tau: float = 5e-3
    actor_alpha: float = 1.0
    critic_alpha: float = 1.0
    num_critics: int = 2
    critic_layernorm: bool = True

    rnd_learning_rate: float = 3e-4
    rnd_hidden_dim: int = 256
    rnd_embedding_dim: int = 32
    rnd_num_epochs: int = 1

    dataset_name: str = "halfcheetah-medium-v2"  # "walker2d-medium-v2"
    batch_size: int = 1024
    num_epochs: int = 3000
    num_updates_on_epoch: int = 1000
    logging_interval: int = 10
    normalize_reward: bool = False

    eval_episodes: int = 10
    eval_period: int = 50

    train_seed: int = 10
    eval_seed: int = 42
