import gymnasium as gym
import torch
import torch.nn as nn
import sys
import os
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 이제 IsaacLab/IsaacSim 관련 모듈 import
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from task.lift.config.ik_abs_env_cfg_3_3 import FrankaYCBPickPlaceEnvCfg


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "skrl")))
from skrl.envs.wrappers.torch import wrap_env

# gymnasium 환경 등록
gym.register(
    id="Isaac-Lift-Cube-Franka-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": FrankaYCBPickPlaceEnvCfg,
    },
    disable_env_checker=True,
)

def make_env(num_envs=1, device="cuda:0"):
    env_cfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-Custom-v0",
        device=device,
        num_envs=num_envs,
        use_fabric=True,
    )
    env_cfg.sim.headless = False
    env = gym.make("Isaac-Lift-Cube-Franka-Custom-v0", cfg=env_cfg)
    #env = wrap_env(env)
    env.reset()
    return env, env_cfg


# def main():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     env = make_env(num_envs=1, device=device)

#     print("[INFO] 환경 생성 성공 ✅")
#     print("관측 공간:", env.observation_space)
#     print("행동 공간:", env.action_space)

#     env.close()
#     print("[INFO] 환경 테스트 종료")

# if __name__ == "__main__":
#     main()
