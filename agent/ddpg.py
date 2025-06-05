# This file is based on
# https://github.com/rll-research/url_benchmark/tree/main/agent/ddpg.py
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This file has been modified for the AMPED
# Note: Author information anonymized for double-blind review.

from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


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
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == "pixels" else hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
        )

        policy_layers = []
        policy_layers += [nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True)]
        # add additional hidden layer for pixels
        if obs_type == "pixels":
            policy_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == "pixels":
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim, feature_dim), nn.LayerNorm(feature_dim), nn.Tanh()
            )
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            )
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [nn.Linear(trunk_dim, hidden_dim), nn.ReLU(inplace=True)]
            if obs_type == "pixels":
                q_layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == "pixels" else torch.cat([obs, action], dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == "pixels" else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1, q2


class DDPGAgent:
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
        update_encoder=False,
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

        # models
        if obs_type == "pixels":
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim

        self.actor = Actor(
            obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim
        ).to(device)

        self.critic = Critic(
            obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim
        ).to(device)
        self.critic_target = Critic(
            obs_type, self.obs_dim, self.action_dim, feature_dim, hidden_dim
        ).to(device)
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

    def eval(self):
        self.train(False)

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
        inpt = torch.cat(inputs, dim=-1)
        # assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().detach().numpy()[0]

    def update_critic_with_gradient_conflict_solver(
        self,
        obs,
        action,
        intr_reward,
        apt_reward,
        discount,
        next_obs,
        step,
        update=True,
    ):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)

            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)

            target_Q_intr = intr_reward + discount * target_V
            target_Q_apt = apt_reward + discount * target_V

        Q1, Q2 = self.critic(obs, action)

        critic_loss_intr = F.mse_loss(Q1, target_Q_intr) + F.mse_loss(Q2, target_Q_intr)
        critic_loss_apt = F.mse_loss(Q1, target_Q_apt) + F.mse_loss(Q2, target_Q_apt)

        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss_intr.backward(retain_graph=True)

        grad_intr = []
        for p in self.critic.parameters():
            if p.grad is not None:
                grad_intr.append(p.grad.clone())
            else:
                grad_intr.append(None)

        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss_apt.backward(retain_graph=True)

        grad_apt = []
        for p in self.critic.parameters():
            if p.grad is not None:
                grad_apt.append(p.grad.clone())
            else:
                grad_apt.append(None)

        dot = 0.0
        for g_i, g_a in zip(grad_intr, grad_apt):
            if g_i is not None and g_a is not None:
                dot += (g_i * g_a).sum()

        metrics["gradient_dot"] = dot

        # self.becl_cic_ratio is [0.0, 1.0], if 1.0, project intr into apt,
        # if 0.0, project apt into intr
        # if between value, use random coin flip to decide which one to project

        if dot < 0 and update:
            # grad_apt의 norm^2 계산
            norm_apt = 0.0

            project_ratio = np.random.binomial(1, self.becl_cic_ratio)
            if project_ratio == 1:
                for g_a in grad_apt:
                    if g_a is not None:
                        norm_apt += (g_a * g_a).sum()

                norm_apt = norm_apt + 1e-12  # 분모가 0이 되지 않도록 작은 값 더하기
                proj_scale = dot / norm_apt

                # grad_intr <- grad_intr - ( (grad_intr·grad_apt) / ||grad_apt||^2 ) * grad_apt
                for i, (g_i, g_a) in enumerate(zip(grad_intr, grad_apt)):
                    if g_i is not None and g_a is not None:
                        grad_intr[i] = g_i - proj_scale * g_a
            else:
                for g_i in grad_intr:
                    if g_i is not None:
                        norm_apt += (g_i * g_i).sum()

                norm_apt = norm_apt + 1e-12
                proj_scale = dot / norm_apt

                for i, (g_i, g_a) in enumerate(zip(grad_intr, grad_apt)):
                    if g_i is not None and g_a is not None:
                        grad_apt[i] = g_a - proj_scale * g_i

        # 8) 최종 gradient = grad_intr + grad_apt
        final_grad = []
        for g_i, g_a in zip(grad_intr, grad_apt):
            if g_i is None and g_a is None:
                final_grad.append(None)
            elif g_i is None:
                final_grad.append(g_a)
            elif g_a is None:
                final_grad.append(g_i)
            else:
                final_grad.append(g_i + g_a)

        if update:
            self.critic_opt.zero_grad(set_to_none=True)

            # critic 파라미터에 최종 gradient 주입
            for p, g in zip(self.critic.parameters(), final_grad):
                if p.requires_grad and g is not None:
                    p.grad = g

            # 실제 업데이트
            self.critic_opt.step()

            if self.use_tb or self.use_wandb:
                metrics["critic_loss_intr"] = critic_loss_intr.item()
                metrics["critic_loss_apt"] = critic_loss_apt.item()

        return metrics

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
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

    def update_actor(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

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

    def update(self, replay_iter, step):
        metrics = dict()
        # import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step)
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
