from tqdm import trange
from typing import Tuple
import numpy as np
import torch
from torch.optim import Adam
import gym

from dataset import ReplayBuffer
from rnd_modules import RND
from modules import Actor, EnsembledCritic
from sac_rnd import SAC_RND
from config import rnd_config
from utils import seed_everything, make_dir

import wandb

import d4rl


class SACRNDTrainer:
    def __init__(self,
                 cfg=rnd_config) -> None:
        make_dir("weights")

        self.device = cfg.device
        self.batch_size = cfg.batch_size
        self.cfg = cfg

        self.action_dim = 6
        self.state_dim = 17

        self.eval_env = gym.make(cfg.dataset_name)
        self.eval_env.seed(cfg.eval_seed)
        d4rl_dataset = d4rl.qlearning_dataset(self.eval_env)

        self.buffer = ReplayBuffer(self.state_dim, self.action_dim)
        # self.buffer.from_d4rl(d4rl_dataset)
        self.buffer.from_json(cfg.dataset_name)

        seed_everything(cfg.train_seed)
    
    def train_rnd(self) -> RND:
        (self.state_mean, self.state_std), (self.action_mean, self.action_std) = self.buffer.get_moments()

        rnd = RND(self.state_dim,
                  self.action_dim,
                  self.cfg.rnd_embedding_dim,
                  self.state_mean,
                  self.state_std,
                  self.action_mean,
                  self.action_std,
                  hidden_dim=self.cfg.rnd_hidden_dim).to(self.device)
        rnd_optim = Adam(rnd.predictor.parameters(), lr=self.cfg.rnd_learning_rate)

        for epoch in trange(self.cfg.rnd_num_epochs, desc="RND Epochs"):

            for _ in trange(self.cfg.num_updates_on_epoch, desc="RND Iterations"):
                states, actions, _, _, _, = self.buffer.sample(self.batch_size)

                loss, update_info = rnd.update_rnd(states, actions)
                rnd_optim.zero_grad()
                loss.backward()
                rnd_optim.step()

                wandb.log(update_info)
        
        return rnd
    
    def train(self):
        '''
            - setup rnd and wandb
            - train rnd
            - setup sac rnd
            - train sac rnd

        '''
        run_name = f"sac_rnd_" + str(self.cfg.train_seed)
        print(f"Training starts on {self.cfg.device} 🚀")

        with wandb.init(project=self.cfg.project, group=self.cfg.group, name=run_name, job_type="offline_training"):
            wandb.config.update({k: v for k, v in self.cfg.__dict__.items() if not k.startswith("__")})

            rnd = self.train_rnd()
            rnd.eval()

            actor = Actor(self.state_dim, self.action_dim, self.cfg.hidden_dim)
            actor_optim = Adam(actor.parameters(), lr=self.cfg.actor_lr)
            critic = EnsembledCritic(self.state_dim, self.action_dim, self.cfg.hidden_dim, layer_norm=self.cfg.critic_layernorm)
            critic_optim = Adam(critic.parameters(), lr=self.cfg.critic_lr)

            self.sac_rnd = SAC_RND(actor,
                                   actor_optim,
                                   critic,
                                   critic_optim,
                                   rnd,
                                   self.cfg.actor_alpha,
                                   self.cfg.critic_alpha,
                                   self.cfg.beta_lr,
                                   self.cfg.gamma,
                                   self.cfg.tau,
                                   self.device)
            
            for epoch in trange(self.cfg.num_epochs, desc="Offline SAC Epochs"):
                update_info_total = {
                    "sac_offline/actor_loss": 0,
                    "sac_offline/actor_batch_entropy": 0,
                    "sac_offline/rnd_policy": 0,
                    "sac_offline/rnd_random": 0,
                    "sac_offline/critic_loss": 0,
                    "sac_offline/q_mean": 0
                    }

                for _ in range(self.cfg.num_updates_on_epoch):
                    state, action, reward, next_state, done = self.buffer.sample(self.batch_size)

                    update_info = self.sac_rnd.train_offline_step(state,
                                                                  action,
                                                                  reward,
                                                                  next_state,
                                                                  done)
                    
                    for k, v in update_info.items():
                        update_info_total[k] += v
                
                for k, v in update_info_total.items():
                    update_info_total[k] /= self.cfg.num_updates_on_epoch
                
                wandb.log(update_info)

                # if epoch % self.cfg.eval_period == 0 or epoch == self.cfg.num_epochs - 1:
                    
                #     eval_returns = self.eval_actor()
                    
                #     normalized_score = self.eval_env.get_normalized_score(eval_returns) * 100.0

                #     wandb.log({
                #         "eval/return_mean": np.mean(eval_returns),
                #         "eval/return_std": np.std(eval_returns),
                #         "eval/normalized_score_mean": np.mean(normalized_score),
                #         "eval/normalized_score_std": np.std(normalized_score)
                #     })
        
        wandb.finish()

    @torch.no_grad()
    def eval_actor(self) -> np.ndarray:
        self.eval_env.seed(self.cfg.eval_seed)
        self.sac_rnd.actor.eval()
        episode_rewards = []
        
        for _ in range(self.cfg.eval_episodes):

            state, done = self.eval_env.reset(), False
            episode_reward = 0.0

            while not done:
                action = self.sac_rnd.actor.act(state, self.device)
                state, reward, done, _ = self.eval_env.step(action)
                episode_reward += reward
            episode_rewards.append(episode_reward)

        self.sac_rnd.actor.train()
        return np.array(episode_rewards)
    
    def save(self):
        state_dict = self.sac_rnd.state_dict()
        torch.save(state_dict, self.cfg.checkpoint_path)

    def load(self, map_location: str = "cpu"):
        state_dict = torch.load(self.cfg.checkpoint_path, map_location=map_location)

        actor = Actor(self.state_dim, self.action_dim, self.cfg.hidden_dim)
        actor_optim = Adam(actor.parameters(), lr=self.cfg.actor_lr)
        critic = EnsembledCritic(self.state_dim, self.action_dim, self.cfg.hidden_dim, layer_norm=self.cfg.critic_layernorm)
        critic_optim = Adam(critic.parameters(), lr=self.cfg.critic_lr)

        rnd = RND(self.state_dim,
                  self.action_dim,
                  self.cfg.rnd_embedding_dim,
                  self.state_mean,
                  self.state_std,
                  self.action_mean,
                  self.action_std,
                  hidden_dim=self.cfg.rnd_hidden_dim)
        
        self.sac_rnd = SAC_RND(actor,
                               actor_optim,
                               critic,
                               critic_optim,
                               rnd,
                               self.cfg.actor_alpha,
                               self.cfg.critic_alpha,
                               self.cfg.beta_lr,
                               self.cfg.gamma,
                               self.cfg.tau,
                               self.device)
        
        self.sac_rnd.load_state_dict(state_dict)
