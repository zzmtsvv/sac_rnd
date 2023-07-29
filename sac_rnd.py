import torch
from torch.nn import functional as F
from typing import Dict, Any, Tuple
from copy import deepcopy

from modules import Actor, EnsembledCritic
from rnd_modules import RND


class SAC_RND:
    def __init__(self,
                 actor: Actor,
                 actor_optim: torch.optim.Optimizer,
                 critic: EnsembledCritic,
                 critic_optim: torch.optim.Optimizer,
                 rnd: RND,
                 actor_alpha: float = 1.0,
                 critic_alpha: float = 1.0,
                 beta_lr: float = 1e-3,
                 gamma: float = 0.99,
                 tau: float = 5e-3,
                 device: str = "cpu") -> None:
        self.device = device

        self.max_action = rnd.max_action

        self.rnd = rnd.to(device)
        self.actor_alpha = actor_alpha
        self.critic_alpha = critic_alpha

        self.actor = actor.to(device)
        self.actor_optim = actor_optim
        
        self.critic = critic.to(device)
        with torch.no_grad():
            self.target_critic = deepcopy(critic)
        self.critic_optim = critic_optim

        self.gamma = gamma
        self.tau = tau

        '''
        In the official implementation the coefficient names for regularizers are switched
        (alpha for log prob and beta for anti-exploration rnd bonus)
        However, I try to stick to the paper's notation and use these terms
        (check sac_rnd.PNG in `paper` folder for details)
        '''
        self.target_entropy = -float(self.actor.action_dim)
        self.log_beta = torch.tensor([0.0], dtype=torch.float32, device=device, requires_grad=True)
        self.beta_optim = torch.optim.Adam([self.log_beta], lr=beta_lr)
        self.beta = self.log_beta.exp().detach()
    
    def train_offline_step(self,
                   state: torch.Tensor,
                   action: torch.Tensor,
                   reward: torch.Tensor,
                   next_state: torch.Tensor,
                   done: torch.Tensor) -> Dict[str, Any]:
        '''
            The update in official implementation is made in order beta-actor-critic

            In this implementation the order is as in the paper: critic-actor-beta
        '''
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)

            rnd_penalty = self.rnd.rnd_bonus(next_state, next_action)
            
            q_next = self.target_critic(next_state, next_action).min(0).values
            q_next = q_next - self.beta * next_action_log_prob - self.critic_alpha * rnd_penalty

            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            q_target = reward + self.gamma * (1 - done) * q_next.unsqueeze(-1)
        
        q_values = self.critic(state, action)
        q_mean = q_values[0].mean().detach()
        
        critic_loss = F.mse_loss(q_values, q_target.view(1, -1))
            
        self.critic_optim.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optim.step()
        
        # actor step
        pi, log_prob = self.actor(state, need_log_prob=True)
        q_values = self.critic(state, pi)
            
        q_min = q_values.min(0).values
        
        with torch.no_grad():
            rnd_penalty = self.rnd.rnd_bonus(state, action)

        actor_loss = (self.beta.detach() * log_prob - q_min + self.actor_alpha * rnd_penalty).mean()

        # for logging
        actor_entropy = -log_prob.mean().detach()
        
        random_actions = torch.rand_like(action)
        random_actions = 2 * self.max_action * random_actions - self.max_action
        rnd_policy = rnd_penalty.mean()
        rnd_random = self.rnd.rnd_bonus(state, random_actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        
        beta_loss = (-self.log_beta * (log_prob.detach() + self.target_entropy)).mean()

        self.beta_optim.zero_grad()
        beta_loss.backward()
        self.beta_optim.step()
        
        self.beta = self.log_beta.exp().detach()

        self.soft_critic_update()

        return {
            "sac_offline/actor_loss": actor_loss.item(),
            "sac_offline/actor_batch_entropy": actor_entropy.item(),
            "sac_offline/rnd_policy": rnd_policy.item(),
            "sac_offline/rnd_random": rnd_random.item(),
            "sac_offline/critic_loss": critic_loss.item(),
            "sac_offline/q_mean": q_mean.item()
        }

    def soft_critic_update(self):
        for param, tgt_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            tgt_param.data.copy_(self.tau * param.data + (1 - self.tau) * tgt_param.data)

    def state_dict(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_beta": self.log_beta.item(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "beta_optim": self.beta_optim.state_dict(),
            "rnd": self.rnd.state_dict(),
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.target_critic.load_state_dict(state_dict["target_critic"])
        
        self.log_beta.data[0] = state_dict["log_beta"]
        self.beta = self.log_beta.exp().detach()

        self.actor_optim.load_state_dict(state_dict["actor_optim"])
        self.critic_optim.load_state_dict(state_dict["critic_optim"])
        self.beta_optim.load_state_dict(state_dict["beta_optim"])

        self.rnd.load_state_dict(state_dict["rnd"])
