# This file is based on
# https://github.com/Baichenjia/CeSD/blob/main/agent/ensemble_ddpg.py
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This file has been modified for the AMPED
# Note: Author information anonymized for double-blind review.

from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import utils


# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features, out_features, ensemble_size):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(
            torch.empty(ensemble_size, in_features, out_features)
        )
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        o = x @ self.weight + self.bias
        # print("in VectorizedLinear:", x.shape, o.shape)
        return o


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
        )

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h


class Actor(nn.Module):
    def __init__(
        self, obs_type, obs_dim, action_dim, skill_dim, feature_dim, hidden_dim
    ):
        super().__init__()

        feature_dim = feature_dim if obs_type == "pixels" else hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + skill_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Tanh(),
        )

        policy_layers = []
        policy_layers += [nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True)]
        # add additional hidden layer for pixels
        if obs_type == "pixels":
            policy_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std, skill=None):
        if skill is not None:
            obs = torch.cat([obs, skill], dim=1)  # TODO

        h = self.trunk(obs)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(
        self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim, ensemble_size
    ):
        super().__init__()

        self.obs_type = obs_type
        self.ensemble_size = ensemble_size
        self.obs_dim = obs_dim

        # for states actions come in the beginning
        trunk_dim = hidden_dim // 2
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, trunk_dim),  # TODO
            nn.LayerNorm(trunk_dim),
            nn.Tanh(),
        )

        def make_q():
            q_layers = []
            q_layers += [
                VectorizedLinear(trunk_dim, hidden_dim, ensemble_size),
                nn.ReLU(inplace=True),
            ]
            q_layers += [VectorizedLinear(hidden_dim, 1, ensemble_size)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = (
            obs if self.obs_type == "pixels" else torch.cat([obs, action], dim=-1)
        )  # (1024, 30)
        h = self.trunk(inpt)  # (1024, 512)

        h = h.unsqueeze(0).repeat_interleave(
            self.ensemble_size, dim=0
        )  # (16, 1024, 512)
        action = action.unsqueeze(0).repeat_interleave(
            self.ensemble_size, dim=0
        )  # (16, 1024, 6)
        # assert h.shape == (16, 1024, 512) and action.shape == (16, 1024, 6)

        h = (
            torch.cat([h, action], dim=-1) if self.obs_type == "pixels" else h
        )  # (16, 1024, 1024)

        q1 = self.Q1(h)  # shape (16, 1024, 1)
        q2 = self.Q2(h)  # shape (16, 1024, 1)

        return q1, q2


class CriticWithPrior(nn.Module):
    def __init__(self, prior, main, prior_scale):
        super().__init__()
        self.prior_network = prior
        self.main_network = main
        self.prior_scale = prior_scale

    def forward(self, obs, action):
        q1_prior, q2_prior = self.prior_network(obs, action)
        q1_main, q2_main = self.main_network(obs, action)
        return (
            q1_prior.detach() * self.prior_scale + q1_main,
            q2_prior.detach() * self.prior_scale + q2_main,
        )

    @property
    def Q1(self):
        return self.main_network.Q1

    @property
    def Q2(self):
        return self.main_network.Q2

    @property
    def trunk(self):
        return self.main_network.trunk


class EnsembleDDPGAgent:
    def __init__(
        self,
        name,
        reward_free,
        obs_type,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        num_expl_steps,
        update_every_steps,
        stddev_schedule,
        nstep,
        batch_size,
        stddev_clip,
        init_critic,
        use_tb,
        use_wandb,
        update_encoder,
        ensemble_size,  # TODO
        meta_dim=0,
    ):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        self.ensemble_size = ensemble_size

        # models
        if obs_type == "pixels":
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0]

        self.actor = Actor(
            obs_type,
            self.obs_dim,
            self.action_dim,
            self.skill_dim,
            feature_dim,
            hidden_dim,
        ).to(device)
        self.critic = Critic(
            obs_type,
            self.obs_dim,
            self.action_dim,
            feature_dim,
            hidden_dim,
            ensemble_size,
        ).to(
            device
        )  # TODO
        self.critic_target = Critic(
            obs_type,
            self.obs_dim,
            self.action_dim,
            feature_dim,
            hidden_dim,
            ensemble_size,
        ).to(
            device
        )  # TODO
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        if obs_type == "pixels":
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)  # (1, 40)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(
        self, obs, skill, action, reward, discount, next_obs, step, mask=None
    ):
        metrics = dict()
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev, skill=skill)
            next_action = dist.sample(clip=self.stddev_clip)  # (1024, 6)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(
                target_Q1, target_Q2
            )  # target_V.shape=(16, 1024, 1), reward.shape=(1024)
            target_Q = reward.squeeze().unsqueeze(0).unsqueeze(-1) + (
                discount * target_V
            )  # (16, 1024, 1)

        Q1, Q2 = self.critic(obs, action)  # Q.shape=(1024,1), Q.shape=(16, 1024, 1)
        if mask is not None:
            # assert mask.shape == (16, 1024, 1)
            Q1, Q2, target_Q = Q1 * mask, Q2 * mask, target_Q * mask

        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, skill, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev, skill=skill)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)  # Q.shape=(16, 1024, 1)   skill.shape=(1024, 16)

        # Optimize the corresponding Q-function in ensemble (using masks)
        Q_skill = skill.T.detach() * Q.squeeze()  # Q_skill=(16, 1024)

        actor_loss = -Q_skill.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["actor_ent"] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)
