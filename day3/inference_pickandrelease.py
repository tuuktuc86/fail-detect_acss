
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.trainers.torch import SequentialTrainer
import ppo
import argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as T
from torchvision.transforms import functional as F
import gymnasium as gym
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed 


set_seed()  # e.g. `set_seed(42)` for fixed seed


# Isaac Lab 관련 라이브러리 임포트
from isaaclab.app import AppLauncher


# Argparse로 CLI 인자 파싱 및 Omniverse 앱 실행
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import SceneEntityCfg


# 커스텀 환경 시뮬레이션 환경 config 파일 임포트
from task.lift.custom_pickplace_env_cfg_3_3 import YCBPickPlaceEnvCfg

# gymnasium 라이브러리를 활용한 시뮬레이션 환경 선언
from task.lift.config.ik_abs_env_cfg_3_3 import FrankaYCBPickPlaceEnvCfg
gym.register(
    id="Isaac-Lift-Cube-Franka-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaYCBPickPlaceEnvCfg,
    },
    disable_env_checker=True,
)

num_envs = 1

# 환경 및 설정 파싱
env_cfg: YCBPickPlaceEnvCfg = parse_env_cfg(
    "Isaac-Lift-Cube-Franka-Custom-v0",
    device=args_cli.device,
    num_envs=num_envs,
    use_fabric=not args_cli.disable_fabric,
   
)

def make_env(id = "Isaac-Lift-Cube-Franka-Custom-v0"):

    # 환경 생성 및 초기화
    env = gym.make(id, cfg=env_cfg)
    env = wrap_env(env)
    return env

def main():
    env = make_env()
    env.reset()
    device = env.device
    models = {}
    models["policy"] = ppo.Shared(env.observation_space, env.action_space, device)
    models["value"] = models["policy"]  # same instance: shared model
    agent = PPO(models=models,
                memory=None,
                cfg=ppo.set_config(env = env, device=device),
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=device)
    agent.load("/fail-detect_acss/runs/torch/isaac-Lift-Franka-v0/25-09-16_13-16-31-370387_PPO/checkpoints/best_agent.pt")

    for m in agent.models.values():
        m.eval()
    max_steps = 600
    episodes=5
    for ep in range(episodes):
        obs, _ = env.reset()
        ep_ret = 0.0
        steps = 0
        while True:
            # 행동 계산(정책의 평균 행동을 사용하도록 eval 모드 + no_grad)
            actions, _, _ = agent.act(obs, timestep=0, timesteps=0)
            #actions[0, -1] = torch.where(actions[0, -1] < 0, torch.tensor(-1.), torch.tensor(1.))
            print(actions)
        
            obs, rew, terminated, truncated, info = env.step(actions)
            ep_ret += rew.mean().item() if torch.is_tensor(rew) else float(rew)
            steps += 1


            done = bool(getattr(terminated, "any", lambda: terminated)()) or \
                    bool(getattr(truncated, "any", lambda: truncated)())
            if done or (max_steps is not None and steps >= max_steps):
                print(f"[Episode {ep+1}] return={ep_ret:.3f}, steps={steps}")
                break


if __name__ == "__main__":
    main()