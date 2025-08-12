import argparse
import os
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
import matplotlib.pyplot as plt

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

# AILAB-summer-school-2025/cgnet 폴더에 접근하기 위한 시스템 파일 경로 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    env.reset()
    print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")

    # 환경 관측 카메라 시점 셋팅

    env.unwrapped.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    #env.unwrapped.sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.0, 0.0, 0.5])
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.api import World
    import isaacsim.core.utils.numpy.rotations as rot_utils
    import numpy as np
    import matplotlib.pyplot as plt
    # 카메라 Prim 추가
    camera2 = Camera(
        prim_path="/World/camera",
        position=np.array([0.16304, 0.4922, 1.7601]),
        frequency=50,
        resolution=(640, 480),
        orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, -90]), degrees=True),)
    camera2.set_world_pose(np.array([0.16304, 0.4922, 1.7601]), rot_utils.euler_angles_to_quats(np.array([0, 0, -90]), degrees=True), camera_axes="usd")
    camera2.initialize()
    camera2.add_motion_vectors_to_frame()

    # 초기 action 설정
    actions = torch.zeros(env.action_space.shape, device='cuda:0')
    actions[:, :7] = torch.tensor([ 0.3, 0,  0.65, 0,  1, 0, 0], device='cuda:0')  # (x, y, z, qw, qx, qy, qz)
    dones = torch.tensor([False], device='cuda:0')

    # 시뮬레이션 루프
    while simulation_app.is_running():
    # Add 
        rgb_image = camera2.get_rgba() # (H, W, 3)
        print(rgb_image.shape)
        #print(camera.orientation)
        if rgb_image.shape[0] != 0:
            rgb = rgb_image[:, :, :3]

            # [0.0, 1.0] → [0, 255]로 정규화 및 uint8 변환
            rgb_uint8 = (rgb * 255).astype(np.uint8)

            # RGB → BGR 변환 (OpenCV 저장용)
            bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite("camera_rgb_image.png", bgr)
            print("Saved RGB image to camera_rgb_image.png")
        dones = env.step(actions)[-2]

# 메인 함수 실행
if __name__ == "__main__":
    main()

    simulation_app.close()