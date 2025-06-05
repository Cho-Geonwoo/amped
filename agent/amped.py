import copy
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent


class RND(nn.Module):
    def __init__(
        self,
        obs_dim,
        hidden_dim,
        rnd_rep_dim,
        encoder,
        aug,
        obs_shape,
        obs_type,
        clip_val=5.0,
    ):
        super().__init__()
        self.clip_val = clip_val
        self.aug = aug

        if obs_type == "pixels":
            self.normalize_obs = nn.BatchNorm2d(obs_shape[0], affine=False)
        else:
            self.normalize_obs = nn.BatchNorm1d(obs_shape[0], affine=False)

        self.predictor = nn.Sequential(
            encoder,
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rnd_rep_dim),
        )
        self.target = nn.Sequential(
            copy.deepcopy(encoder),
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, rnd_rep_dim),
        )

        for param in self.target.parameters():
            param.requires_grad = False

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = self.aug(obs)
        obs = self.normalize_obs(obs)
        obs = torch.clamp(obs, -self.clip_val, self.clip_val)
        prediction, target = self.predictor(obs), self.target(obs)
        prediction_error = torch.square(target.detach() - prediction).mean(
            dim=-1, keepdim=True
        )
        return prediction_error


class CIC(nn.Module):
    def __init__(self, obs_dim, skill_dim, hidden_dim, project_skill):
        super().__init__()
        self.obs_dim = obs_dim
        self.skill_dim = skill_dim

        self.state_net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.skill_dim),
        )

        self.pred_net = nn.Sequential(
            nn.Linear(2 * self.skill_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.skill_dim),
        )

        if project_skill:
            self.skill_net = nn.Sequential(
                nn.Linear(self.skill_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.skill_dim),
            )
        else:
            self.skill_net = nn.Identity()

        self.apply(utils.weight_init)

    def forward(self, state, next_state, skill):
        assert len(state.size()) == len(next_state.size())
        state = self.state_net(state)
        next_state = self.state_net(next_state)
        query = self.skill_net(skill)
        key = self.pred_net(torch.cat([state, next_state], 1))
        return query, key


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RMS(object):
    def __init__(self, epsilon=1e-4, shape=(1,)):
        self.M = torch.zeros(shape).to(device)
        self.S = torch.ones(shape).to(device)
        self.n = epsilon

    def __call__(self, x):
        bs = x.size(0)
        delta = torch.mean(x, dim=0) - self.M
        new_M = self.M + delta * bs / (self.n + bs)
        new_S = (
            self.S * self.n
            + torch.var(x, dim=0) * bs
            + (delta**2) * self.n * bs / (self.n + bs)
        ) / (self.n + bs)

        self.M = new_M
        self.S = new_S
        self.n += bs

        return self.M, self.S


class APTArgs:
    def __init__(
        self,
        knn_k=16,
        knn_avg=True,
        rms=True,
        knn_clip=0.0005,
    ):
        self.knn_k = knn_k
        self.knn_avg = knn_avg
        self.rms = rms
        self.knn_clip = knn_clip


rms = RMS()


def compute_apt_reward(source, target, args):

    b1, b2 = source.size(0), target.size(0)
    # (b1, 1, c) - (1, b2, c) -> (b1, 1, c) - (1, b2, c) -> (b1, b2, c) -> (b1, b2)
    sim_matrix = torch.norm(
        source[:, None, :].view(b1, 1, -1) - target[None, :, :].view(1, b2, -1),
        dim=-1,
        p=2,
    )
    reward, _ = sim_matrix.topk(
        args.knn_k, dim=1, largest=False, sorted=True
    )  # (b1, k)

    if not args.knn_avg:  # only keep k-th nearest neighbor
        reward = reward[:, -1]
        reward = reward.reshape(-1, 1)  # (b1, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(
            reward - args.knn_clip, torch.zeros_like(reward).to(device)
        )  # (b1, )
    else:  # average over all k nearest neighbors
        reward = reward.reshape(-1, 1)  # (b1 * k, 1)
        if args.rms:
            moving_mean, moving_std = rms(reward)
            reward = reward / moving_std
        reward = torch.max(reward - args.knn_clip, torch.zeros_like(reward).to(device))
        reward = reward.reshape((b1, args.knn_k))  # (b1, k)
        reward = reward.mean(dim=1)  # (b1,)
    reward = torch.log(reward + 1.0)
    return reward


class BECL(nn.Module):
    def __init__(self, tau_dim, feature_dim, hidden_dim):
        super().__init__()

        self.embed = nn.Sequential(
            nn.Linear(tau_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

        self.project_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.apply(utils.weight_init)

    def forward(self, tau):
        features = self.embed(tau)
        features = self.project_head(features)
        return features


class AmpedAgent(DDPGAgent):
    def __init__(
        self,
        update_skill_every_step,
        skill_dim,
        update_encoder,
        contrastive_update_rate,
        temperature,
        alpha,
        beta,
        skill,
        project_skill,
        update_rep,
        becl_cic_ratio,
        **kwargs
    ):
        self.becl_cic_ratio = becl_cic_ratio
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.update_encoder = update_encoder
        self.contrastive_update_rate = contrastive_update_rate
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        # specify skill in fine-tuning stage if needed
        self.skill = int(skill) if skill >= 0 else np.random.choice(self.skill_dim)
        self.update_rep = update_rep
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim
        self.batch_size = kwargs["batch_size"]
        # create actor and critic
        super().__init__(**kwargs)

        # net
        self.becl = BECL(
            self.obs_dim - self.skill_dim, self.skill_dim, kwargs["hidden_dim"]
        ).to(self.device)

        self.Lambda = nn.Parameter(torch.eye(skill_dim, device=self.device))

        self.rnd = RND(
            self.obs_dim - self.skill_dim,
            self.hidden_dim,
            self.hidden_dim,
            self.encoder,
            self.aug,
            self.obs_shape,
            self.obs_type,
        ).to(self.device)

        # optimizers
        self.rnd_opt = torch.optim.Adam(self.rnd.parameters(), lr=self.lr)
        self.becl_opt = torch.optim.Adam(
            list(self.becl.parameters()) + [self.Lambda], lr=self.lr
        )

        self.cic = CIC(
            self.obs_dim - skill_dim, skill_dim, kwargs["hidden_dim"], project_skill
        ).to(self.device)

        # optimizers
        self.cic_optimizer = torch.optim.Adam(self.cic.parameters(), lr=self.lr)

        self.intrinsic_reward_rms = utils.RMS(device=self.device)

        self.rnd.train()
        self.cic.train()
        self.becl.train()

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, "skill"),)

    def init_meta(self):
        skill = np.zeros(self.skill_dim).astype(np.float32)
        if not self.reward_free:
            skill[self.skill] = 1.0
        else:
            skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta["skill"] = skill
        return meta

    def set_meta(self, seed_skill):
        skill = np.zeros(self.skill_dim).astype(np.float32)
        skill[seed_skill] = 1.0
        meta = OrderedDict()
        meta["skill"] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_contrastive(self, state, skills):
        metrics = dict()
        features = self.becl(state)
        logits = self.compute_ani_nce_loss(features, skills, self.Lambda)
        loss = logits.mean()

        self.becl_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.becl_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics["contrastive_loss"] = loss.item()

        return metrics

    def compute_intr_reward(self, skills, state, metrics):

        # compute contrastive reward
        features = self.becl(state)
        contrastive_reward = torch.exp(
            -self.compute_ani_nce_loss(features, skills, self.Lambda)
        )

        intr_reward = contrastive_reward
        if self.use_tb or self.use_wandb:
            metrics["contrastive_reward"] = contrastive_reward.mean().item()

        return intr_reward

    def compute_ani_nce_loss(self, features, skills, Lambda=None):
        labels = torch.argmax(skills, dim=-1)  # (b,)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long()  # (b, b)
        labels = labels.to(self.device)

        b, c = features.shape
        diff = features.unsqueeze(1) - features.unsqueeze(0)  # (b, b, c)
        if Lambda is None:
            dist_matrix = (diff**2).sum(dim=-1)  # (b, b) -> Euclidean squared distance
        else:
            diff_L = torch.matmul(diff, Lambda)  # (b, b, c)
            dist_matrix = (diff_L * diff).sum(dim=-1)  # (b, b)

        mask = torch.eye(b, dtype=torch.bool).to(self.device)
        dist_matrix = dist_matrix[~mask].view(b, -1)  # (b, b-1)
        labels = labels[~mask].view(b, -1)  # (b, b-1)

        dist_matrix = dist_matrix / self.temperature

        exp_neg_dist = torch.exp(-dist_matrix)  # (b, b-1)

        pick_one_positive_sample_idx = torch.argmax(labels, dim=-1, keepdim=True)
        pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(
            -1, pick_one_positive_sample_idx, 1
        )

        positives = torch.sum(
            exp_neg_dist * pick_one_positive_sample_idx, dim=-1, keepdim=True
        )  # (b, 1)
        negatives = torch.sum(exp_neg_dist, dim=-1, keepdim=True)  # (b, 1)

        eps = torch.as_tensor(1e-6).to(self.device)
        loss = -torch.log(positives / (negatives + eps) + eps)  # (b, 1)
        return loss

    def compute_cpc_loss(self, obs, next_obs, skill):
        temperature = self.temperature
        eps = 1e-6
        query, key = self.cic.forward(obs, next_obs, skill)
        query = F.normalize(query, dim=1)
        key = F.normalize(key, dim=1)
        cov = torch.mm(query, key.T)  # (b,b)
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)  # (b,)
        row_sub = (
            torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        )
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        pos = torch.exp(torch.sum(query * key, dim=-1) / temperature)  # (b,)
        loss = -torch.log(pos / (neg + eps))  # (b,)
        return loss, cov / temperature

    def update_cic(self, obs, skill, next_obs, step):
        metrics = dict()

        loss, logits = self.compute_cpc_loss(obs, next_obs, skill)
        loss = loss.mean()
        self.cic_optimizer.zero_grad()
        loss.backward()
        self.cic_optimizer.step()

        if self.use_tb or self.use_wandb:
            metrics["cic_loss"] = loss.item()
            metrics["cic_logits"] = logits.norm()

        return metrics

    def update_rnd(self, obs, step):
        metrics = dict()

        prediction_error = self.rnd(obs)
        loss = prediction_error.mean()

        self.rnd_opt.zero_grad()
        loss.backward()
        self.rnd_opt.step()

        if self.use_tb or self.use_wandb:
            metrics["rnd_loss"] = loss.item()

        return metrics

    @torch.no_grad()
    def compute_rnd_reward(self, obs, step):
        prediction_error = self.rnd(obs)
        _, intr_reward_var = self.intrinsic_reward_rms(prediction_error)
        reward = prediction_error / (torch.sqrt(intr_reward_var) + 1e-8)
        return reward

    @torch.no_grad()
    def compute_apt_reward(self, obs, next_obs):
        args = APTArgs()
        source = self.cic.state_net(obs)
        target = self.cic.state_net(next_obs)
        reward = compute_apt_reward(source, target, args)  # (b,)
        return reward.unsqueeze(-1)  # (b,1)

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device
        )

        with torch.no_grad():
            obs = self.aug_and_encode(obs)
            next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            metrics.update(self.update_contrastive(next_obs, skill))

            if self.update_rep:
                metrics.update(self.update_cic(obs, skill, next_obs, step))

            metrics.update(self.update_rnd(obs, step))
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, next_obs, metrics)
                apt_reward = self.compute_apt_reward(next_obs, next_obs)
                rnd_reward = self.compute_rnd_reward(obs, step)

            if self.use_tb or self.use_wandb:
                metrics["intr_reward"] = intr_reward.mean().item()
                metrics["apt_reward"] = apt_reward.mean().item()
                metrics["rnd_reward"] = rnd_reward.mean().item()

            apt_reward = self.alpha * apt_reward + self.beta * rnd_reward
            reward = intr_reward + apt_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics["batch_reward"] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        if self.reward_free:
            metrics.update(
                self.update_critic_with_gradient_conflict_solver(
                    obs.detach(),
                    action,
                    intr_reward,
                    apt_reward,
                    discount,
                    next_obs.detach(),
                    step,
                )
            )
        else:
            # update critic
            metrics.update(
                self.update_critic(
                    obs.detach(), action, reward, discount, next_obs.detach(), step
                )
            )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics

    def init_from(self, other):
        super().init_from(other)
        self.becl.load_state_dict(other.becl.state_dict())
        self.rnd.load_state_dict(other.rnd.state_dict())
        self.cic.load_state_dict(other.cic.state_dict())
