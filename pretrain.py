import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import torch

if torch.cuda.is_available():
    os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
    if "DISPLAY" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"
    else:
        os.environ["MUJOCO_GL"] = "glfw"

from pathlib import Path

import hydra
import numpy as np
import wandb
from dm_env import specs

import dmc
import random
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        cfg.seed = random.randint(0, 100000)
        utils.set_seed_everywhere(cfg.seed)
        if torch.cuda.is_available():
            self.device = torch.device(cfg.device)
        else:
            self.device = torch.device("cpu")
            cfg.device = "cpu"

        config = {}

        for k, v in cfg.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    config[k + "." + kk] = vv
            else:
                config[k] = v

        # create logger
        if cfg.use_wandb:
            exp_name = "_".join(
                [
                    cfg.experiment,
                    cfg.agent.name,
                    cfg.domain,
                    cfg.obs_type,
                    str(cfg.seed),
                ]
            )
            wandb.login(key=cfg.wandb_key)
            wandb.init(
                project="amped", group=cfg.agent.name, name=exp_name, config=config
            )

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        # create envs
        assert (
            self.cfg.domain in PRIMAL_TASKS
        ), f"{self.cfg.domain} not in {PRIMAL_TASKS}"

        self.train_env = dmc.make(
            PRIMAL_TASKS[self.cfg.domain],
            cfg.obs_type,
            cfg.frame_stack,
            cfg.action_repeat,
            cfg.seed,
        )
        self.eval_env = dmc.make(
            PRIMAL_TASKS[self.cfg.domain],
            cfg.obs_type,
            cfg.frame_stack,
            cfg.action_repeat,
            cfg.seed,
        )

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # create data storage
        self.replay_storage = ReplayBufferStorage(
            data_specs, meta_specs, self.work_dir / "buffer"
        )

        # create replay buffer
        self.replay_loader = make_replay_loader(
            self.replay_storage,
            cfg.replay_buffer_size,
            cfg.batch_size,
            cfg.replay_buffer_num_workers,
            False,
            cfg.nstep,
            cfg.discount,
        )
        self._replay_iter = None

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, meta, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                total_reward += time_step.reward
                step += 1

            episode += 1

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        # self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                # self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                # try to save snapshot
                episode_step = 0
                episode_reward = 0

            if self.global_frame in self.cfg.snapshots:
                self.save_snapshot()

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(
                    time_step.observation, meta, self.global_step, eval_mode=False
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")
                if self.cfg.use_wandb:
                    wandb.log(metrics)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            episode_step += 1
            self._global_step += 1

        self.save_snapshot()

    def load_snapshot(self):
        snapshot_file = Path(self.cfg.snapshot)
        with snapshot_file.open("rb") as f:
            payload = torch.load(f, map_location=self.cfg.device)
        self.agent.init_from(payload["agent"])
        self.cfg.num_seed_frames = payload["_global_step"] / 10
        self.cfg.num_train_frames -= payload["_global_step"] + self.cfg.num_seed_frames

        self._global_episode = payload["_global_episode"]

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
        print(snapshot)
        keys_to_save = ["agent", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)


@hydra.main(config_path=".", config_name="pretrain")
def main(cfg):
    from pretrain import Workspace as W

    workspace = W(cfg)
    snapshot = Path(cfg.snapshot)
    if snapshot.exists() and not snapshot.is_dir():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    try:
        workspace.train()
    except KeyboardInterrupt:
        print("interrupted")
        workspace.save_snapshot()
        exit()


if __name__ == "__main__":
    main()
