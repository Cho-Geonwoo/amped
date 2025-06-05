from collections import OrderedDict
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class SkillActor(nn.Module):
    def __init__(self, obs_dim, skill_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        layers = []
        layers.append(nn.Linear(feature_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, skill_dim))
        self.policy = nn.Sequential(*layers)
        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        logits = self.policy(h)
        return logits


class SkillCritic(nn.Module):
    def __init__(self, obs_dim, skill_dim, feature_dim, hidden_dim):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )
        self.q_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, skill_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, obs):
        h = self.trunk(obs)
        q_values = self.q_net(h)
        return q_values


class SkillSelectorAgent:
    def __init__(
        self,
        obs_shape,
        skill_dim,
        device,
        actor_lr,
        critic_lr,
        feature_dim,
        hidden_dim,
        update_every_steps,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=20000,
    ):
        self.skill_dim = skill_dim
        self.device = device
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.update_every_steps = update_every_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

        self.encoder = nn.Identity()
        self.obs_dim = obs_shape[0]

        self.actor = SkillActor(self.obs_dim, skill_dim, feature_dim, hidden_dim).to(
            device
        )
        self.critic = SkillCritic(self.obs_dim, skill_dim, feature_dim, hidden_dim).to(
            device
        )
        self.target_critic = SkillCritic(
            self.obs_dim, skill_dim, feature_dim, hidden_dim
        ).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.encoder_opt = None

    def get_epsilon(self, step):
        """epsilon decay schedule"""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(
            -1.0 * step / self.epsilon_decay
        )
        return epsilon

    def train(self, training=True):
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_opt is not None:
            self.encoder.train(training)

    def eval(self):
        self.train(False)

    def act(self, obs, step, eval_mode=False):
        self.step_count = step
        obs_tensor = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        encoded_obs = self.encoder(obs_tensor)
        logits = self.actor(encoded_obs)
        dist = torch.distributions.Categorical(logits=logits)
        epsilon = self.get_epsilon(step)
        if eval_mode:
            skill_idx = torch.argmax(logits, dim=-1)
        else:
            if random.random() < epsilon:
                skill_idx = torch.tensor(
                    [random.randrange(self.skill_dim)], device=self.device
                )
            else:
                skill_idx = dist.sample()
        skill_onehot = np.zeros(self.skill_dim).astype(np.float32)
        skill_onehot[skill_idx] = 1.0
        return skill_onehot

    def update(self, replay_iter, step, gamma=0.99, tau=0.005):

        metrics = {}
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, *meta = utils.to_torch(
            batch, self.device
        )
        if len(meta) > 0:
            stored_skill = meta[0]
        else:
            stored_skill = None

        obs_enc = self.encoder(obs)
        next_obs_enc = self.encoder(next_obs)

        q_values = self.critic(obs_enc)
        if stored_skill is not None:
            skill_idx = stored_skill.argmax(dim=1, keepdim=True)
        else:
            logits = self.actor(obs_enc)
            dist = torch.distributions.Categorical(logits=logits)
            sampled_skill = dist.sample()
            skill_idx = sampled_skill.unsqueeze(1)
        current_q = q_values.gather(1, skill_idx)

        with torch.no_grad():
            next_q = self.target_critic(next_obs_enc)  # (batch, skill_dim)
            next_q_max, _ = next_q.max(dim=1, keepdim=True)
            target_q = reward + discount * gamma * next_q_max

        critic_loss = F.mse_loss(current_q, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        metrics["skill_critic_loss"] = critic_loss.item()

        logits = self.actor(obs_enc)
        dist = torch.distributions.Categorical(logits=logits)
        sampled_skill = dist.sample()  # (batch,)
        log_prob = dist.log_prob(sampled_skill)
        q_values_for_actor = self.critic(obs_enc)
        actor_q = q_values_for_actor.gather(1, sampled_skill.unsqueeze(1)).squeeze(1)

        entropy = dist.entropy()
        beta = 0.01

        actor_loss = -(log_prob * actor_q).mean() - beta * entropy.mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()
        metrics["skill_actor_loss"] = actor_loss.item()

        # --- Target Critic Soft Update ---
        utils.soft_update_params(self.critic, self.target_critic, tau)

        return metrics
