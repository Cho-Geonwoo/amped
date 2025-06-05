# This file is based on
# https://github.com/Baichenjia/CeSD/blob/main/agent/cesd.py
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# This file has been modified for the AMPED
# Note: Author information anonymized for double-blind review.

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict
from torch import distributions as pyd
import utils
from agent.ensemble_ddpg import EnsembleDDPGAgent

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

        self.next_state_net = nn.Sequential(
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


class Proto(nn.Module):
    def __init__(self, obs_dim, T, num_protos, batch_size, num_iters, net):
        super().__init__()
        self.num_iters = num_iters
        self.T = T
        self.num_protos = num_protos
        self.batch_size = batch_size
        self.net = net  # feature extractor

        self.protos = nn.Linear(num_protos, num_protos, bias=False)
        self.apply(utils.weight_init)

    def forward(self, s, t):
        # normalize prototypes. s.shape=(1024, 24), t.shape=(1024, 24)
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)  # C.shape=(num_cluster, 24)
        self.num_sample = s.shape[0]

        with torch.no_grad():
            s = self.net(s)  # s.shape=(1024, 24)
        s = F.normalize(s, dim=1, p=2)
        scores_s = self.protos(s)
        log_p_s = F.log_softmax(
            scores_s / self.T, dim=1
        )  # log_p_s.shape=(1024, num_cluster)
        # print("\n log_p_s:", log_p_s.shape)

        with torch.no_grad():
            t = self.net(t)  # s.shape=(1024, 24)
            t = F.normalize(t, dim=1, p=2)
            scores_t = self.protos(t)  # (1024, num_cluster)
            q_t = self.sinkhorn(scores_t)  # q_t.shape=(1024, num_cluster)
            # print("\n q_t:", q_t.shape)

        loss = -(q_t * log_p_s).sum(dim=1).mean()
        return loss

    def calculate_cluster(self, obs):
        # normalize
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

        with torch.no_grad():
            z = self.net(obs)  # obs.shape=(1024, 24)
        z = F.normalize(z, dim=1, p=2)  # (1024, 16)

        # cluster weight
        scores = self.protos(z)  # (1024, num_cluster)
        p = F.softmax(scores / self.T, dim=1)  # (1024, num_cluster)

        idx = pyd.Categorical(p).sample()  # (1024,)
        assert idx.shape == (z.shape[0],)

        # extract samples from each cluster
        cluster_samples, cluster_index = dict(), dict()
        for i in range(self.num_protos):
            cluster_samples[i] = obs[idx == i]
            cluster_index[i] = np.arange(z.shape[0])[idx.cpu().numpy() == i]
        # print("cluster ", i, ", shape:", cluster_samples[i].shape, cluster_index[i].shape, cluster_index[i])  # Count of Sample of Each Cluster
        return cluster_samples, cluster_index

    def create_mask_matrix(self, cluster_index):
        matrix = np.zeros((self.num_protos, int(self.batch_size)))
        for i in range(self.num_protos):
            matrix[i][cluster_index[i]] = 1
        return torch.from_numpy(matrix).unsqueeze(-1)

    def sinkhorn(self, scores):
        def remove_infs(x):
            # print("**", x.shape, torch.isfinite(x).shape, (x[torch.isfinite(x)]).shape)
            m = x[torch.isfinite(x)].max().item()
            x[torch.isinf(x)] = m
            return x

        Q = scores / self.T
        Q -= Q.max()

        Q = torch.exp(Q).T
        Q = remove_infs(Q)
        Q /= Q.sum()

        r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
        c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
        for it in range(self.num_iters):
            u = Q.sum(dim=1)
            u = remove_infs(r / u)
            Q *= u.unsqueeze(dim=1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.T


class CeSDAgent(EnsembleDDPGAgent):
    def __init__(
        self,
        update_skill_every_step,
        scale,
        project_skill,
        rew_type,
        update_rep,
        temp,
        ensemble_size,
        proto_T,
        proto_num_iters,
        constrain_factor,
        domain,
        **kwargs
    ):
        self.temp = temp
        self.skill_dim = ensemble_size
        self.ensemble_size = ensemble_size
        self.update_skill_every_step = update_skill_every_step
        self.scale = scale
        self.project_skill = project_skill
        self.rew_type = rew_type
        self.update_rep = update_rep
        self.batch_size = kwargs["batch_size"]
        self.constrain_factor = constrain_factor[domain]
        self.proto_num_iters = proto_num_iters[domain]
        # print("\n\n ******Domain:", domain, self.constrain_factor, self.proto_num_iters)

        kwargs["meta_dim"] = self.skill_dim
        kwargs["ensemble_size"] = ensemble_size
        # create actor and critic
        super().__init__(**kwargs)
        # create cic first
        self.cic = CIC(
            self.obs_dim, self.skill_dim, kwargs["hidden_dim"], project_skill
        ).to(kwargs["device"])

        # Optimizers
        self.cic_optimizer = torch.optim.Adam(self.cic.parameters(), lr=self.lr)
        self.cic.train()

        # Proto
        self.proto = Proto(
            obs_dim=self.obs_dim,
            T=proto_T,
            num_protos=ensemble_size,
            batch_size=self.batch_size,
            num_iters=self.proto_num_iters,
            net=self.cic.state_net,
        ).cuda()
        self.proto_optimizer = torch.optim.Adam(self.proto.parameters(), lr=self.lr)

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, "skill"),)

    def init_meta(self, meta=None):
        if meta is not None:
            return meta
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta["skill"] = skill
        return meta

    def update_meta(self, meta, step, time_step):
        if step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def compute_cpc_loss(self, obs, next_obs, skill):
        temperature = self.temp
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

    def compute_intr_reward(self, obs, skill, next_obs, step):
        with torch.no_grad():
            loss, logits = self.compute_cpc_loss(obs, next_obs, skill)

        reward = loss
        reward = reward.clone().detach().unsqueeze(-1)

        return reward * self.scale

    @torch.no_grad()
    def compute_apt_reward(self, obs, next_obs):
        args = APTArgs()
        source = self.cic.state_net(obs)
        target = self.cic.state_net(next_obs)
        reward = compute_apt_reward(source, target, args)  # (b,)
        return reward.unsqueeze(-1)  # (b,1)

    def update_proto(self, obs, next_obs, step):
        loss = self.proto(obs, next_obs)
        self.proto_optimizer.zero_grad()
        loss.backward()
        self.proto_optimizer.step()
        return dict()

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

        mask = None
        if self.reward_free:
            if self.update_rep:
                metrics.update(self.update_cic(obs, skill, next_obs, step))

            # Cluster. Calculate Reward for Each Cluster
            intr_reward = torch.zeros(obs.shape[0]).to(self.device)
            cluster_samples, cluster_index = self.proto.calculate_cluster(next_obs)
            count_of_big_cluster = 0
            for i in range(self.ensemble_size):
                next_obs_cluster = cluster_samples[i]  # (number of samples, obs_dim)
                # print("\bnext_obs_cluster:", next_obs_cluster.shape)
                if next_obs_cluster.shape[0] > 16:
                    skill_cluster = skill.argmax(-1)[cluster_index[i]]  # (c_size, 1)
                    countB = torch.sum(
                        skill_cluster != i
                    )  # calculate the intrinsic reward for policy constraints
                    intrinsicB = 1.0 / (1.0 + countB)

                    intr_reward_cluster = self.compute_apt_reward(
                        next_obs_cluster, next_obs_cluster
                    ).squeeze()
                    # print("intr_reward_cluster:", intr_reward_cluster.shape)
                    intr_reward[cluster_index[i]] = (
                        intr_reward_cluster + intrinsicB * self.constrain_factor
                    )
                    count_of_big_cluster += 1

            # if step % 100 == 0 and count_of_big_cluster != 16:
            # print("## count of big cluster:", count_of_big_cluster)
            mask = self.proto.create_mask_matrix(cluster_index).to(
                self.device
            )  # (num_of_ensemble, 1024) 激活的位置是1
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            if self.reward_free:
                metrics["extr_reward"] = extr_reward.mean().item()
                metrics["intr_reward"] = intr_reward.mean().item()
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(
                obs, skill, action, reward, discount, next_obs, step, mask
            )
        )

        # update actor
        metrics.update(self.update_actor(obs, skill, step))

        # TODO: update proto
        if self.reward_free:
            metrics.update(self.update_proto(obs, next_obs, step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics
