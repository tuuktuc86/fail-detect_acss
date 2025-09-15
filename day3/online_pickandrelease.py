import argparse
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial.transform import Rotation as R
from torchvision import transforms as T
from torchvision.transforms import functional as F
import gymnasium as gym
import set_env
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


def main():
    """메인 함수"""
    # 환경 갯수(1개로 고정)
    

    env = set_env.make_env()
    obs, _ = env.reset()
    device = env.device
    print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")

    # # 환경 연산 디바이스(gpu)
    # device = env.unwrapped.scene.device
    # max_trajectories = 5
    # total_traj = 0 
    # max_steps_per_traj = 300

    # actions = torch.tensor(
    #     [[0.4000, 0.0000, 0.200, -0.0573, 0.9845, 0.1090, 0.1252, 1.0000]],
    #     device="cuda:0"
    # )
    # 시뮬레이션 루프
    while 1:
        
        print(f"[INFO] Starting trajectory {total_traj+1}/{max_trajectories}")
        
        done = False
        step_count = 0
        save_count=0
        is_success = False
       
        while not done and step_count < max_steps_per_traj:

                

            # 로봇의 End-Effector 위치와 자세를 기반으로 actions 계산
            robot_data = env.unwrapped.scene["robot"].data
            env.unwrapped.scene["robot"].data.applied_torque[0, -2:] = 0.0
            ee_frame_sensor = env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)

            
            if step_count > 20 :
                actions = torch.tensor(
                    [[0.4000, 0.0000, 0.030, -0.0573, 0.9845, 0.1090, 0.1252, 1.0000]],
                    device="cuda:0"
                )
                actions[0][:3] = env.unwrapped.scene._rigid_objects['object_0']._data.root_pos_w
                actions[0][2]-=0.3


            obs, rewards, terminated, truncated, info = env.step(actions)
            


            prev = actions 
            step_count += 1

            # 시뮬레이션 종료 여부 체크
            dones = terminated | truncated
            if all(dones):
                done = True
                if terminated:
                    print("Episode terminated")
                else:
                    print("Episode truncated")

        
        traj_length = save_count
        total_traj += 1 
    # 환경 종료 및 시뮬레이션 종료
    env.close()
    simulation_app.close()       
# 메인 함수 실행
if __name__ == "__main__":
    main()