import random
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import wandb
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
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader

if torch.cuda.is_available():
    from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

import logging


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def make_skill_selector(obs_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
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
        self.train_env = dmc.make(
            cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed
        )
        self.eval_env = dmc.make(
            cfg.task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed
        )

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )

        self.skill_selector = make_skill_selector(
            self.train_env.observation_spec(),
            cfg.skill_selector,
        )

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()["agent"]
            self.agent.init_from(pretrained_agent)

        # get meta specs
        self.meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        self.data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # create data storage
        self.replay_storage = ReplayBufferStorage(
            self.data_specs, self.meta_specs, self.work_dir / "buffer"
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

        # create video recorders
        if cfg.save_video:
            self.video_recorder = VideoRecorder(self.work_dir)
        if cfg.save_train_video:
            self.train_video_recorder = TrainVideoRecorder(self.work_dir)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    def reset_replay_buffer(self):
        self.replay_storage = ReplayBufferStorage(
            self.data_specs, self.meta_specs, self.work_dir / "buffer"
        )

        # create replay buffer
        self.replay_loader = make_replay_loader(
            self.replay_storage,
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            False,
            self.cfg.nstep,
            self.cfg.discount,
        )
        self._replay_iter = None

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

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            meta_skill = self.skill_selector.act(
                time_step.observation, self.global_step, eval_mode=True
            )
            meta = {"skill": meta_skill}

            if self.cfg.save_video:
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, meta, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                if self.cfg.save_video:
                    self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

                meta_skill = self.skill_selector.act(
                    time_step.observation, self.global_step, eval_mode=True
                )
                meta = {"skill": meta_skill}

            episode += 1
            if self.cfg.save_video:
                self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            if "skill" in meta:
                log("skill", meta["skill"].argmax())

    def train(self):
        self._global_step = 0
        self._global_episode = 0

        time_step = self.train_env.reset()
        meta = self.skill_selector.act(
            time_step.observation, self._global_step, eval_mode=False
        )
        wandb.log({"skill": meta.argmax()})
        self.replay_storage.add(time_step, {"skill": meta})
        if self.cfg.save_train_video:
            self.train_video_recorder.init(time_step.observation)

        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                if self.cfg.save_train_video:
                    self.train_video_recorder.save(f"{self.global_frame}.mp4")
                time_step = self.train_env.reset()
                meta = self.skill_selector.act(
                    time_step.observation, self._global_step, eval_mode=False
                )
                self.replay_storage.add(time_step, {"skill": meta})
                if self.cfg.save_train_video:
                    self.train_video_recorder.init(time_step.observation)
                wandb.log({"episode_reward": episode_reward})
                episode_step, episode_reward = 0, 0

            if eval_every_step(self.global_step):
                self.agent.eval()
                self.eval()
                self.agent.train()

            meta = self.skill_selector.act(
                time_step.observation, self._global_step, eval_mode=False
            )
            wandb.log({"skill": meta.argmax()})

            action = self.agent.act(
                time_step.observation,
                {"skill": meta},
                self._global_step,
                eval_mode=False,
            )

            if not seed_until_step(self.global_step):
                skill_selector_metrics = self.skill_selector.update(
                    self.replay_iter, self._global_step
                )
                metrics = self.agent.update(self.replay_iter, self._global_step)
                metrics.update(skill_selector_metrics)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")
                if self.cfg.use_wandb:
                    wandb.log(metrics)

            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, {"skill": meta})
            if self.cfg.save_train_video:
                self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split("_", 1)
        snapshot_dir = (
            snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name
        )

        def try_load(seed):
            snapshot = (
                "../../../../../"
                / snapshot_dir
                / str(seed)
                / f"snapshot_{self.cfg.snapshot_ts}.pt"
            )
            logging.info(
                "loading model :{},cwd is {}".format(str(snapshot), str(Path.cwd()))
            )
            if not snapshot.exists():
                logging.error("no such a pretrain model")
                return None
            with snapshot.open("rb") as f:
                payload = torch.load(f, map_location="cpu")
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        assert payload is not None

        return payload


@hydra.main(config_path=".", config_name="finetunev2")
def main(cfg):
    from finetunev2 import Workspace as W

    root_dir = Path.cwd()
    logging.basicConfig(encoding="utf-8", level=logging.DEBUG)
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
