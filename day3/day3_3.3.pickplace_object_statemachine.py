import argparse
import os
import datetime
import glob
import random
import torch
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import clip
from torchvision import transforms as T
from torchvision.transforms import functional as F
import gymnasium as gym
from collections.abc import Sequence
import open3d as o3d
# o3d.visualization.Visualizer().destroy_window() #rendering X 
import matplotlib.pyplot as plt
from PIL import Image

# Isaac Lab 관련 라이브러리 임포트
from isaaclab.app import AppLauncher

# Argparse로 CLI 인자 파싱 및 Omniverse 앱 실행
parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms
#import omni.kit.viewport.utility as viewport_utils #use this wha

# AILAB-summer-school-2025/cgnet 폴더에 접근하기 위한 시스템 파일 경로 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isaacsim.util.debug_draw import _debug_draw
from pxr import Gf

draw = _debug_draw.acquire_debug_draw_interface()

# 카메라 렌더링 옵션 --enable_cameras flag 를 대신하기 위함
import carb
carb_settings_iface = carb.settings.get_settings()
carb_settings_iface.set_bool("/isaaclab/cameras_enabled", True)


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
# ---------- helpers ----------
def parse_deltas(s: str):
    s = s.strip()
    if not s:
        return None                     # 0개: 그대로
    vals = [float(x) for x in s.split()] # 공백으로 분리
    return vals[:8]                      # 최대 8개만

def next_actions(prev: torch.Tensor, delta=None, total_dim=8, device="cuda:0", clip=True):
    if prev.dim()==1: prev = prev.unsqueeze(0)
    out = prev.clone().to(device)
    if delta is not None:
        inc = torch.as_tensor(delta, dtype=out.dtype, device=device).flatten()
        k = min(inc.numel(), total_dim)
        out[:, :k] += inc[:k]
    if clip: out = out.clamp(-1.0, 1.0)
    return out


#refer_data = np.load("dataset_all_afterpregrasp_t3.npz")

def main():
    """메인 함수"""
    # 환경 갯수(1개로 고정)
    num_envs = 1

    # 환경 및 설정 파싱
    env_cfg: YCBPickPlaceEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-Custom-v0",
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )

    # 환경 생성 및 초기화
    env = gym.make("Isaac-Lift-Cube-Franka-Custom-v0", cfg=env_cfg)

    
    obs, _ = env.reset()

    print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")
    
    # 환경 관측 카메라 시점 셋팅

    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api import World
    import isaacsim.core.utils.numpy.rotations as rot_utils
    import numpy as np
    import matplotlib.pyplot as pl

    # 환경 연산 디바이스(gpu)
    device = env.unwrapped.scene.device
    max_trajectories = 5
    total_traj = 0 
    max_steps_per_traj = 300

    # actions = torch.tensor(
    #     [[0.3960, -0.0809, 0.3323, -0.0573, 0.9845, 0.1090, 0.1252, -1.0000]],
    #     device="cuda:0"
    # )
    actions = torch.tensor(
        [[0.4000, 0.0000, 0.200, -0.0573, 0.9845, 0.1090, 0.1252, 1.0000]],
        device="cuda:0"
    )
    # 시뮬레이션 루프
    while simulation_app.is_running() and total_traj < max_trajectories:
        env.reset()
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
            #env.unwrapped.unwrapped.scene.state['rigid_object']['object_0'] 
            #{'root_pose': tensor([[0.4000, 0.0000, 0.6000, 1.0000, 0.0000, 0.0000, 0.0000]],
       #device='cuda:0'), 'root_velocity': tensor([[0., 0., 0., 0., 0., 0.]], device='cuda:0')}
            # s = input("delta 입력(공백 구분, 0~8개, 빈 입력=유지, q=종료): ").strip()
            # if s.lower() == "q":
            #     break
            # delta = parse_deltas(s)  # None | list[float]
            # actions = next_actions(prev, delta, total_dim=8, device=device, clip=True)  
            
            if step_count > 20 :
                actions = torch.tensor(
                    [[0.4000, 0.0000, 0.030, -0.0573, 0.9845, 0.1090, 0.1252, 1.0000]],
                    device="cuda:0"
                )
                actions[0][:3] = env.unwrapped.scene._rigid_objects['object_0']._data.root_pos_w
                actions[0][2]-=0.3

            # if step_count > 50 :
            #     actions = torch.tensor(
            #         [[0.4000, 0.0000, 0.030, -0.0573, 0.9845, 0.1090, 0.1252, 1.0000]],
            #         device="cuda:0"
            #     )
            #     actions[0][:3] = env.unwrapped.scene._rigid_objects['object_0']._data.root_pos_w
            #     actions[0][2]-=0.5
                
            # if step_count > 80 :
            #     actions = torch.tensor(
            #         [[0.4000, 0.0000, 0.030, -0.0573, 0.9845, 0.1090, 0.1252, -1.0000]],
            #         device="cuda:0"
            #     )
            #     actions[0][:3] = env.unwrapped.scene._rigid_objects['object_0']._data.root_pos_w
            #     actions[0][2]-=0.5
                
            # if step_count > 150 :
            #     actions = torch.tensor(
            #     [[0.4000, 0.0000, 0.330, -0.0573, 0.9845, 0.1090, 0.1252, -1.0000]],
            #     device="cuda:0"
            # )

            obs, rewards, terminated, truncated, info = env.step(actions)
            
            print(f"actions={actions}")
            print(f"ee_pose={ee_pose}")
            print(f"rewards={rewards}")

            
            # p = (0.4, 0.0, 0.53)          # 월드 좌표
            # c = (1.0, 0.0, 0.0, 1.0)     # RGBA 빨강
            # s = 20                       # 픽셀 크기(가시성 위해 크게)

            # draw.draw_points([p], [c], [s])

            prev = actions 
            step_count += 1

            # 시뮬레이션 종료 여부 체크
            dones = terminated | truncated
            if dones:
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