import argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as T
from torchvision.transforms import functional as F
import gymnasium as gym
from day3.skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from day3.skrl.envs.loaders.torch import load_isaaclab_env
from day3.skrl.envs.wrappers.torch import wrap_env
from day3.skrl.memories.torch import RandomMemory
from day3.skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from day3.skrl.resources.preprocessors.torch import RunningStandardScaler
from day3.skrl.resources.schedulers.torch import KLAdaptiveLR
from day3.skrl.trainers.torch import SequentialTrainer
from day3.skrl.utils import set_seed 


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
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import SceneEntityCfg


# 커스텀 환경 시뮬레이션 환경 config 파일 임포트
from lift.lift_env_cfg import LiftEnvCfg

# gymnasium 라이브러리를 활용한 시뮬레이션 환경 선언
from lift.config.franka.ik_rel_env_cfg import FrankaCubeLiftEnvCfg # i change it to rel
# from lift.config.franka.joint_pos_env_cfg import FrankaCubeLiftEnvCfg

gym.register(
    id="Isaac-Lift-Cube-Franka-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaCubeLiftEnvCfg,
    },
    disable_env_checker=True,
)

num_envs =4096 #please check headless. or it may cause a probnlem

# 환경 및 설정 파싱
env_cfg: LiftEnvCfg = parse_env_cfg(
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