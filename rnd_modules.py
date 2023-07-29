# check rnd_architecture.PNG in `paper` folder for visualization.
from typing import Tuple, Dict
import torch
from torch import nn

try:
    from rnd_utils import RunningMeanStd
except ModuleNotFoundError:
    from sac_rnd.rnd_utils import RunningMeanStd
# from torch.nn import functional as F


class PredictorNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 4) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        self.bilinear = nn.Bilinear(state_dim, action_dim, hidden_dim)
        
        layers = [nn.ReLU()]
        for _ in range(num_hidden_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        z = self.layers(self.bilinear(states, actions))
        return z


class GatingModule(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim

        self.tanh_module = nn.Sequential(
            nn.Linear(action_dim, embedding_dim),
            nn.Tanh()
        )
        self.sigmoid_module = nn.Sequential(
            nn.Linear(state_dim, embedding_dim),
            nn.Sigmoid()
        )
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        out = self.tanh_module(actions) * self.sigmoid_module(states)
        return out


class FiLM(nn.Module):
    '''
        Feature-wise Linear Modulation
    '''
    def __init__(self,
                 in_features: int,
                 out_features: int) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = nn.Linear(in_features, 2 * out_features)
    
    def forward(self,
                states: torch.Tensor,
                h: torch.Tensor) -> torch.Tensor:
        gamma, beta = torch.split(self.linear(states), self.out_features, dim=-1)

        return gamma * h + beta


class PriorNetwork(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 4) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers

        base_network = [
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(num_hidden_layers - 3):
            base_network.append(nn.Linear(hidden_dim, hidden_dim))
            base_network.append(nn.ReLU())
        
        base_network.append(nn.Linear(hidden_dim, hidden_dim))
        self.base_network = nn.Sequential(*base_network)

        self.film = FiLM(state_dim, hidden_dim)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor) -> torch.Tensor:
        h = self.base_network(actions)
        h = self.film(states, h)
        z = self.head(h)
        return z


class RND(nn.Module):
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 embedding_dim: int,
                 state_mean: torch.Tensor,
                 state_std: torch.Tensor,
                 action_mean: torch.Tensor,
                 action_std: torch.Tensor,
                 max_action: float = 1.0,
                 hidden_dim: int = 256,
                 num_hidden_layers: int = 4) -> None:
        super().__init__()

        self.state_mean, self.state_std = state_mean, state_std
        self.action_mean, self.action_std = action_mean, action_std

        self.loss_fn = nn.MSELoss(reduction="none")

        self.rms = RunningMeanStd()
        self.max_action = max_action
        
        self.predictor = PredictorNetwork(state_dim,
                                          action_dim,
                                          embedding_dim,
                                          hidden_dim,
                                          num_hidden_layers)
        self.predictor.train()

        self.prior = PriorNetwork(state_dim,
                                  action_dim,
                                  embedding_dim,
                                  hidden_dim,
                                  num_hidden_layers)
        self.disable_prior_grads()
        self.prior.eval()
    
    def disable_prior_grads(self):
        for p in self.prior.parameters():
            p.requires_grad = False
    
    def normalize(self,
                  state: torch.Tensor,
                  action: torch.Tensor,
                  eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
        state = (state - self.state_mean) / (self.state_std + eps)
        action = (action - self.action_mean) / (self.action_std + eps)

        return state, action
    
    def forward(self,
                states: torch.Tensor,
                actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.prior.eval()

        states, actions = self.normalize(states, actions)

        predictor_out = self.predictor(states, actions)
        prior_out = self.prior(states, actions)

        return predictor_out, prior_out
    
    def loss(self,
             states: torch.Tensor,
             actions: torch.Tensor) -> torch.Tensor:
        '''
            outputs unreduced vector with shape as [batch_size, embedding_dim]
        '''
        predictor_out, prior_out = self(states, actions)

        loss = self.loss_fn(predictor_out, prior_out)
        return loss
    
    def rnd_bonus(self,
                  state: torch.Tensor,
                  action: torch.Tensor) -> torch.Tensor:
        bonus = self.loss(state, action).sum(dim=1) / self.rms.std
        return bonus
    
    def update_rnd(self,
                   states: torch.Tensor,
                   actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw_loss = self.loss(states, actions).sum(dim=1)
        loss = raw_loss.mean(dim=0)

        self.rms.update(raw_loss)

        # made for logging
        random_actions = torch.rand_like(actions)
        random_actions = 2 * self.max_action * random_actions - self.max_action
        rnd_random = self.rnd_bonus(states, random_actions).mean()

        update_info = {
            "rnd/loss": loss.item(),
            "rnd/running_std": self.rms.std.item(),
            "rnd/data": loss / self.rms.std.item(),
            "rnd/random": rnd_random.item()
            }
        
        return loss, update_info

if __name__ == "__main__":
    rnd = RND(17, 6, 32, None, None, None, None)

    for p in rnd.parameters():
        print(p.requires_grad)
