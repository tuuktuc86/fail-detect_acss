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
import omni.kit.viewport.utility as viewport_utils

# AILAB-summer-school-2025/cgnet 폴더에 접근하기 위한 시스템 파일 경로 추가
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Detection 모델 라이브러리 임포트
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision

# Contact-GraspNet 모델 라이브러리 임포트
from cgnet.utils.config import cfg_from_yaml_file
from cgnet.tools import builder
from cgnet.inference_cgnet import inference_cgnet

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


# Detection 모델 설정
DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(DIR_PATH, 'data/checkpoint/maskrcnn_ckpt/maskrcnn_trained_model_refined.pth') # <-- 사전 학습된 Weight
NUM_CLASSES = 79  # 모델 구조는 학습 때와 동일해야 함
CONFIDENCE_THRESHOLD = 0.5

YCB_OBJECT_CLASSES = sorted([
    '002_master_chef_can', '008_pudding_box', '014_lemon', '021_bleach_cleanser', '029_plate', '036_wood_block', '044_flat_screwdriver', '054_softball', '061_foam_brick', '065_c_cups', '065_i_cups', '072_b_toy_airplane', '073_c_lego_duplo',
    '003_cracker_box', '009_gelatin_box', '015_peach', '022_windex_bottle', '030_fork', '037_scissors', '048_hammer', '055_baseball', '062_dice', '065_d_cups', '065_j_cups', '072_c_toy_airplane', '073_d_lego_duplo',
    '004_sugar_box', '010_potted_meat_can', '016_pear', '024_bowl', '031_spoon', '038_padlock', '050_medium_clamp', '056_tennis_ball', '063_a_marbles', '065_e_cups', '070_a_colored_wood_blocks', '072_d_toy_airplane', '073_e_lego_duplo',
    '005_tomato_soup_can', '011_banana', '017_orange', '025_mug', '032_knife', '040_large_marker', '051_large_clamp', '057_racquetball', '063_b_marbles', '065_f_cups', '070_b_colored_wood_blocks', '072_e_toy_airplane', '073_f_lego_duplo',
    '006_mustard_bottle', '012_strawberry', '018_plum', '026_sponge', '033_spatula', '042_adjustable_wrench', '052_extra_large_clamp', '058_golf_ball', '065_a_cups', '065_g_cups', '071_nine_hole_peg_test', '073_a_lego_duplo', '073_g_lego_duplo',
    '007_tuna_fish_can', '013_apple', '019_pitcher_base', '028_skillet_lid', '035_power_drill', '043_phillips_screwdriver', '053_mini_soccer_ball', '059_chain', '065_b_cups', '065_h_cups', '072_a_toy_airplane', '073_b_lego_duplo', '077_rubiks_cube'
])
CLASS_NAME = ['BACKGROUND'] + YCB_OBJECT_CLASSES

def depth2pc(depth, K, rgb=None):
    """ 뎁스 이미지를 포인트 클라우드로 변환하는 함수 """
    
    mask = np.where(depth > 0)
    x, y = mask[1], mask[0]
    
    normalized_x = (x.astype(np.float32)-K[0,2])
    normalized_y = (y.astype(np.float32)-K[1,2])
    
    world_x = normalized_x * depth[y, x] / K[0,0]
    world_y = normalized_y * depth[y, x] / K[1,1]
    world_z = depth[y, x]
    
    if rgb is not None:
        rgb = rgb[y, x]
    
    pc = np.vstack([world_x, world_y, world_z]).T
    return (pc, rgb)

def get_world_bbox(depth, K, bb): 
    """ Bounding Box의 좌표를 포인트 클라우드 기준 좌표로 변환하는 함수 """

    image_width = depth.shape[1]
    image_height = depth.shape[0]

    x_min, x_max = bb[0], bb[2]
    y_min, y_max = bb[1], bb[3]
    
    if y_min < 0:
        y_min = 0
    if y_max >= image_height:
        y_max = image_height-1
    if x_min < 0:
        x_min = 0
    if x_max >=image_width:
        x_max = image_width-1

    z_0, z_1 = depth[int(y_min), int(x_min)], depth[int(y_max), int(x_max)]
    
    def to_world(x, y, z):
        """ 뎁스 포인트를 3D 포인트로 변환하는 함수 """
        world_x = (x - K[0, 2]) * z / K[0, 0]
        world_y = (y - K[1, 2]) * z / K[1, 1]
        return world_x, world_y, z
    
    x_min_w, y_min_w, z_min_w = to_world(x_min, y_min, z_0)
    x_max_w, y_max_w, z_max_w = to_world(x_max, y_max, z_1)
    
    return x_min_w, y_min_w, x_max_w, y_max_w

def get_model_instance_segmentation(num_classes):
    """ 학습 때와 동일한 구조로 Mask R-CNN 모델을 생성합니다. """
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def get_random_color():
    """ 시각화를 위해 랜덤 RGB 색상을 생성합니다. """
    return [random.randint(50, 255) for _ in range(3)] # 너무 어둡지 않은 색상

def CLIP_transform(img_tensor, output_size, fill=0, padding_mode='constant'):
    """ CLIP 모델 입력에 맞게 변환 """

    # 긴 축을 기준으로 리사이즈될 크기 계산
    _, h, w = img_tensor.shape
    if h > w:
        new_h = output_size
        new_w = int(w * (new_h / h))
    else: # w >= h
        new_w = output_size
        new_h = int(h * (new_w / w))

    # 비율을 유지하며 리사이즈
    img_tensor = F.resize(img_tensor, [new_h, new_w])

    # 목표 크기에 맞게 패딩을 추가 (left, top, right, bottom)
    pad_left = (output_size - new_w) // 2
    pad_top = (output_size - new_h) // 2
    pad_right = output_size - new_w - pad_left
    pad_bottom = output_size - new_h - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    # 패딩을 적용하여 이미지를 반환
    padded_img = F.pad(img_tensor, padding, fill, padding_mode)

    # mean/std를 사용해 정규화
    normalized_img = F.normalize(padded_img.to(torch.float32), 
                                 mean=[0.48145466, 0.4578275, 0.40821073], 
                                 std=[0.26862954, 0.26130258, 0.27577711])
    
    return normalized_img

from PIL import Image
# trajectory 끝난 후 log_dir 안의 front_view_* 이미지를 gif로 묶기
def make_gif_from_images(image_dir, pattern="front_view_*.png", gif_name="trajectory.gif", duration=200):
    # 저장된 이미지 경로 정렬
    frames = []
    imgs = sorted(glob.glob(os.path.join(image_dir, pattern)))
    for img_path in imgs:
        frame = Image.open(img_path)
        frames.append(frame)

    if frames:
        frames[0].save(
            os.path.join(image_dir, gif_name),
            save_all=True,
            append_images=frames[1:],
            duration=duration,   # frame 간격(ms)
            loop=0              # 0이면 무한 반복
        )
        print(f"[INFO] GIF saved at {os.path.join(image_dir, gif_name)}")
    else:
        print("[WARNING] No images found for GIF creation.")


class GripperState:
    """ 로봇 제어를 위한 그리퍼 state 정의 """
    OPEN = 1.0
    CLOSE = -1.0

class PickAndPlaceSmState:
    """ 로봇 제어를 위한  상황 state 정의 """
    REST = 0
    PREDICT = 1
    READY = 2
    PREGRASP = 3
    GRASP = 4
    CLOSE = 5
    LIFT = 6
    MOVE_TO_BIN = 7
    LOWER = 8
    RELEASE = 9
    BACK = 10
    BACK_TO_READY = 11

class PickAndPlaceSmWaitTime:
    """ 각 pick-and-place 상황 state 별 대기 시간(초) 정의 """
    REST = 3.0
    PREDICT = 0.0
    READY = 0.5
    PREGRASP = 1.0
    GRASP = 0.5
    CLOSE = 1.0
    LIFT = 0.5
    MOVE_TO_BIN = 0.5
    LOWER = 0.5
    RELEASE = 0.5
    BACK = 0.5
    BACK_TO_READY = 0.5

from td3_bc_agent import TD3_BC_Agent   
from replay_buffer import ReplayBuffer  

def main():
    num_envs = 1
    env_cfg: YCBPickPlaceEnvCfg = parse_env_cfg(
        "Isaac-Lift-Cube-Franka-Custom-v0",
        device=args_cli.device,
        num_envs=num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env = gym.make("Isaac-Lift-Cube-Franka-Custom-v0", cfg=env_cfg)
    device = env.unwrapped.scene.device

    # TD3+BC Agent 
    state_dim = 16   # ee_pose(7) + gripper(2) + grasp_pose(7)
    action_dim = 8   # ee_pose_delta(7) + gripper_action(1)
    max_action = 1.0
    agent = TD3_BC_Agent(state_dim, action_dim, max_action, device=device)

    # Replay Buffer 
    replay_buffer = ReplayBuffer(state_dim, action_dim, max_size=1_000_000)

    # hyper parameter 
    max_steps_per_traj = 700
    max_trajectories = 100
    batch_size = 256
    total_it = 0

    for traj_idx in range(max_trajectories):
        obs = env.reset()[0]
        done = False
        step_count = 0
        episode_reward = 0

        # ---- state 초기화 ----
        ee_pose = obs["policy"][0, :7]  # EE pose
        gripper_state = obs["policy"][0, 7:9]  # gripper
        grasp_pose = torch.zeros(7)  # placeholder: clip/grasp 
        pregrasp_pose = torch.zeros(7)

        state = torch.cat([ee_pose, gripper_state, grasp_pose, pregrasp_pose], dim=-1).to(device)

        while not done and step_count < max_steps_per_traj:
            # 행동 선택
            if total_it < 10_000:
                action = (torch.rand(action_dim) * 2 - 1).to(device)
            else:
                action = agent.select_action(state.unsqueeze(0))  # (1, action_dim)
                action = torch.tensor(action, device=device, dtype=torch.float32).squeeze(0)

            # 환경 step
            next_obs, reward, terminated, truncated, info = env.step(action.unsqueeze(0))
            done = bool(terminated or truncated)
            episode_reward += reward.item()

            # 다음 state 구성
            ee_pose = next_obs["policy"][0, :7]
            gripper_state = next_obs["policy"][0, 7:9]
            # grasp_pose, pregrasp_pose 는 detection/clip/grasp 모델 결과 반영 가능
            next_state = torch.cat([ee_pose, gripper_state, grasp_pose, pregrasp_pose], dim=-1).to(device)

            # Replay buffer에 저장
            replay_buffer.add(
                state.cpu().numpy(),
                action.cpu().numpy(),
                reward.item(),
                next_state.cpu().numpy(),
                float(done)
            )

            state = next_state
            step_count += 1
            total_it += 1

            # 학습 단계
            if total_it >= 1000:  # warmup 이후 학습 시작
                agent.train(replay_buffer, batch_size)

        print(f"[INFO] Trajectory {traj_idx+1}: reward={episode_reward:.2f}, steps={step_count}")

    env.close()
    simulation_app.close()
