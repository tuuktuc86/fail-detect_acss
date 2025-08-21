# train_ssl_3to1_11in_8out.py
import argparse, os, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

IN_DIM  = 11
OUT_DIM = 8
PAST_N  = 3

class NPZNextStepDataset(Dataset):
    def __init__(self, npz_path: str):
        self.path  = npz_path
        # 키/길이만 1회 읽고 닫기
        with np.load(self.path, allow_pickle=False) as f:
            self.keys = sorted(f.files)
            self.lens = {k: f[k].shape[0] for k in self.keys}

        # 인덱스 구축 (target = t+1)
        self.index = []
        for k in self.keys:
            T = self.lens[k]
            if T >= 2:
                for t in range(0, T-1):
                    self.index.append((k, t+1))

        # 정규화 통계 (한 번만 열어 계산)
        with np.load(self.path, allow_pickle=False) as f:
            all_obs = np.concatenate([f[k] for k in self.keys], axis=0).astype(np.float32)
        self.mean_in  = torch.from_numpy(all_obs.mean(0))
        self.std_in   = torch.from_numpy(all_obs.std(0) + 1e-6)
        self.mean_out = self.mean_in[:OUT_DIM].clone()
        self.std_out  = self.std_in[:OUT_DIM].clone()

    def __len__(self): return len(self.index)

    def __getitem__(self, i):
        k, tgt = self.index[i]  # tgt in [1..T-1]
        # 안전하게 매 호출 시 파일 열기
        with np.load(self.path, allow_pickle=False) as f:
            ep = f[k]  # (T,11)

            # 최근 3스텝 복제 패딩
            frames = []
            for dt in range(PAST_N, 0, -1):
                idx = tgt - dt
                if idx < 0: idx = 0
                frames.append(ep[idx])

            x = torch.from_numpy(np.stack(frames, axis=0)).float()   # (3,11)
            y = torch.from_numpy(ep[tgt, :OUT_DIM]).float()          # (8,)
        return x, y
class GRUForecast(nn.Module):
    def __init__(self, hidden=256, layers=2):
        super().__init__()
        self.gru = nn.GRU(input_size=IN_DIM, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, OUT_DIM),
        )
    def forward(self, x):                 # x: (B,3,11)
        h, _ = self.gru(x)                # (B,3,H)
        return self.head(h[:, -1, :])     # (B,8)

from collections import deque
import torch

class ObsHistory:
    def __init__(self, hist_len=3):
        self.hist_len = hist_len
        self.buffer = deque(maxlen=hist_len)

    def add(self, obs):
        """obs: torch.Tensor (11,) 같은 단일 observation"""
        self.buffer.append(obs.detach().clone())

    def get(self):
        """(hist_len, obs_dim) tensor 반환 (복제 패딩 포함)"""
        if len(self.buffer) == 0:
            raise ValueError("No obs in buffer yet!")
        
        if len(self.buffer) == 1:
            # 같은 obs 3개
            out = [self.buffer[0]] * self.hist_len
        elif len(self.buffer) == 2:
            # 첫 obs + 마지막 obs 복제 2개
            out = [self.buffer[0], self.buffer[1], self.buffer[1]]
        else:
            # 이미 3개 이상 → deque 특성상 최근 3개만 유지됨
            out = list(self.buffer)
        
        return torch.stack(out, dim=0)   # (3, obs_dim)


@torch.no_grad()
def predict_next8(obs_hist_3, ckpt_path, cpu=False):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    device = torch.device("cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = GRUForecast(hidden=cfg["hidden"], layers=cfg["layers"]).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()

    # 🔥 stats는 바로 torch.tensor로 변환
    stats = np.load(os.path.join(cfg["out"], "norm_stats.npz"))
    mi = torch.tensor(stats["mean_in"], dtype=torch.float32, device=device)
    si = torch.tensor(stats["std_in"], dtype=torch.float32, device=device)
    mo = torch.tensor(stats["mean_out"], dtype=torch.float32, device=device)
    so = torch.tensor(stats["std_out"], dtype=torch.float32, device=device)

    # 입력 obs_hist_3 (3,11)
    x = torch.as_tensor(obs_hist_3, dtype=torch.float32, device=device).unsqueeze(0)  # (1,3,11)
    x = (x - mi) / si
    y = model(x)[0]
    y = y * so + mo
    return y.cpu().numpy()




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
# args_cli.headless = True
# args_cli.enable_cameras = True
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


class PickAndPlaceSm:
    """
    로봇이 물체를 집어 옮기는(Pick-and-Place) 작업을 상태머신(State Machine)으로 구현.
    각 단계별로 End-Effector 위치와 그리퍼 상태를 지정해줌.

    0. REST: 로봇이 초기자세 상태에 있습니다.
    1. PREDICT: 파지 예측을 수행합니다.
    2. READY: 로봇이 초기자세 상태에 위치하고, 그리퍼를 CLOSE 상태로 둡니다.
    3. PREGRASP: 타겟 물체 앞쪽의 pre-grasp 자세로 이동합니다.
    4. GRASP: 엔드이펙터를 타겟 물체에 grasp 자세로 접근합니다.
    5. CLOSE: 그리퍼를 닫아 물체를 집습니다.
    6. LIFT: 물체를 들어올립니다.
    7. MOVE_TO_BIN: 물체를 목표 xy 위치(바구니)로 이동시키고, 높이도 특정 높이까지 유지합니다.
    8. LOWER: 물체를 낮은 z 위치까지 내립니다.
    9. RELEASE: 그리퍼를 열어 물체를 놓습니다.
    10. BACK: 엔드이펙터를 바구니 위의 특정 높이로 다시 이동시킵니다.
    11. BACK_TO_READY: 엔드이펙터를 원래 초기 위치로 이동시킵니다.
    """
    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu", position_threshold=0.01):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # state machine 파라미터 값(1)
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        self.position_threshold = position_threshold

        # state machine 파라미터 값(2)
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # 목표 로봇 끝단(end-effector) 자세 및 그리퍼 상태
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs, 1), 0.0, device=self.device)

        # 물체 이미지를 취득하기 위한 준비 자세
        self.ready_pose = torch.tensor([[ 3.0280e-01, -5.6916e-02,  6.2400e-01, -1.4891e-10,  1.0000e+00, 8.4725e-11, -8.7813e-10]], device=self.device)  # (x, y, z, qw, qx, qy, qz)
        self.ready_pose = self.ready_pose.repeat(num_envs, 1)

        # 물체를 상자에 두기 위해 상자 위에 위치하는 자세
        self.bin_pose = torch.tensor([[ 0.2, 0.6, 0.55, 0, 1, 0, 0]], device=self.device)   # (x, y, z, qw, qx, qy, qz)
        self.bin_pose = self.bin_pose.repeat(num_envs, 1)

        # 물체를 안정적으로 상자에 두기 위한 낮은 자세
        self.bin_lower_pose = torch.tensor([[ 0.2, 0.6, 0.35, 0, 1, 0, 0]], device=self.device)   # (x, y, z, qw, qx, qy, qz)
        self.bin_lower_pose = self.bin_lower_pose.repeat(num_envs, 1)

        # Contact-GraspNet 추론 값을 담기위한 변수 선언
        self.grasp_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.pregrasp_pose = torch.zeros((self.num_envs, 7), device=self.device)

        # Gripper가 원하는 위치에 도달하지 못하는 경우, statemachine이 멈추는 것을 방지하기 위한 변수 선언
        self.stack_ee_pose = []
        self.noise_start_step = [None for _ in range(self.num_envs)]

    # env idx 를 통한 reset 상태 실행
    def reset_idx(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = PickAndPlaceSmState.REST
        self.sm_wait_time[env_ids] = 0.0

    ##################################### State Machine #####################################
    # 로봇의 end-effector 및 그리퍼의 목표 상태 계산
    def compute(self, ee_pose: torch.Tensor, grasp_pose: torch.Tensor, pregrasp_pose: torch.Tensor, robot_data, current_step):
        ee_pos = ee_pose[:, :3]
        ee_pos[:, 2] -= 0.5

        # 각 environment에 반복적으로 적용
        for i in range(self.num_envs):
            state = self.sm_state[i]
            # 각 상태에 따른 로직 구현
            if state == PickAndPlaceSmState.REST:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 특정 시간 동안 대기
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.REST:
                    # 다음 state 로 전환 및 state 시간 초기화
                    self.sm_state[i] = PickAndPlaceSmState.PREDICT
                    self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.PREDICT:
                # 다음 state 로 전환 및 state 시간 초기화
                self.sm_state[i] = PickAndPlaceSmState.READY
                self.sm_wait_time[i] = 0.0
                
            elif state == PickAndPlaceSmState.READY:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.READY:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.PREGRASP
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.PREGRASP:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = pregrasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.PREGRASP:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.GRASP
                        self.sm_wait_time[i] = 0.0
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []

            elif state == PickAndPlaceSmState.GRASP:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = grasp_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.GRASP:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.CLOSE
                        self.sm_wait_time[i] = 0.0
                        self.stack_ee_pose = []
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []

            elif state == PickAndPlaceSmState.CLOSE:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = ee_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 특정 시간 동안 대기
                if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.CLOSE:
                    # 다음 state 로 전환 및 state 시간 초기화
                    self.sm_state[i] = PickAndPlaceSmState.LIFT
                    self.sm_wait_time[i] = 0.0
                    # 일정 높이로 들어 올릴 위치 설정
                    self.lift_pose = grasp_pose[i]
                    self.lift_pose[2] = self.lift_pose[2] + 0.4

            elif state == PickAndPlaceSmState.LIFT:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.lift_pose 
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.LIFT:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.MOVE_TO_BIN
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.MOVE_TO_BIN:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.bin_pose[i]

                # # ##[failcase4] change mis put to bin 
                # if self.noise_start_step[i] is None:
                #     self.noise_start_step[i]= current_step
                #     print(f"[INFO] Noise first added at step {current_step}")
                # self.test_noise = torch.tensor([
                #     np.random.uniform(-0.6, 0.1),  
                #     np.random.uniform(-0.7, 0.1), 
                #     0.0
                # ], device=self.bin_pose.device)
                # self.des_ee_pose[i, :3] += self.test_noise    
                # # ## ----------------------

                self.des_gripper_state[i] = GripperState.CLOSE
                # 현재 state에서의 end-effector position을 저장
                self.stack_ee_pose.append(ee_pos[i])
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.MOVE_TO_BIN:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.LOWER
                        self.sm_wait_time[i] = 0.0
                        self.stack_ee_pose = []
                # end-effector의 위치가 일정 step 이상 바뀌지 않을때, 다음 state 로 전환 및 state 시간 초기화
                else:
                    if len(self.stack_ee_pose) > 50:
                        if torch.linalg.norm(ee_pos[i] - self.stack_ee_pose[-30]) < self.position_threshold:                       
                            self.sm_state[i] = PickAndPlaceSmState.CLOSE
                            self.sm_wait_time[i] = 0.0
                            self.stack_ee_pose = []
                self.position_threshold = 0.01

            elif state == PickAndPlaceSmState.LOWER:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.bin_lower_pose[i]
                self.des_gripper_state[i] = GripperState.CLOSE
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.LOWER:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.RELEASE
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.RELEASE:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.bin_lower_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.RELEASE:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.BACK
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.BACK:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.bin_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.BACK:
                        # 다음 state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.BACK_TO_READY
                        self.sm_wait_time[i] = 0.0

            elif state == PickAndPlaceSmState.BACK_TO_READY:
                # 목표 end-effector 자세 및 그리퍼 상태 정의
                self.des_ee_pose[i] = self.ready_pose[i]
                self.des_gripper_state[i] = GripperState.OPEN
                # 목표자세 도딜시 특정 시간 동안 대기
                if torch.linalg.norm(ee_pos[i] - self.des_ee_pose[i, :3]) < self.position_threshold:
                    if self.sm_wait_time[i] >= PickAndPlaceSmWaitTime.BACK_TO_READY:
                        # 남은 물체를 잡기 위해, PREDICT state 로 전환 및 state 시간 초기화
                        self.sm_state[i] = PickAndPlaceSmState.PREDICT
                        self.sm_wait_time[i] = 0.0
                        
            # state machine 단위시간 경과
            self.sm_wait_time[i] += self.dt

            actions = torch.cat([self.des_ee_pose, self.des_gripper_state], dim=-1)

        return actions
    ###############################################################################################

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
    hist = ObsHistory(hist_len=3)

    user_text = None

    # 환경 생성 및 초기화
    env = gym.make("Isaac-Lift-Cube-Franka-Custom-v0", cfg=env_cfg)
    
    # Get selected object names from the environment config
    # from task.lift.custom_pickplace_env_cfg_3_3 import SELECTED_OBJECT_NAMES
    # selected_names = SELECTED_OBJECT_NAMES
    
    # # Generate a random permutation for picking order (e.g., if selected_names = ['1', '2', '3'], might become ['1', '3', '2'])
    # pick_order = selected_names.copy()
    # random.shuffle(pick_order)
    # chosen_selected_names = pick_order.copy()  # Copy for saving (original shuffled order)
    
    env.reset()
    print(f"Environment reset. Number of environments: {env.unwrapped.num_envs}")
    
    # 환경 관측 카메라 시점 셋팅
    env.unwrapped.sim.set_camera_view(eye=[2.0, 0.0, 1.5], target=[0.0, 0.0, 0.5]) # person view 
    #env.unwrapped.sim.set_camera_view(eye=[0.0, 0.0, 5.0], target=[0.0, 0.0, 0.0])  # top view 

    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.sensors.camera import Camera
    from isaacsim.core.api import World
    import isaacsim.core.utils.numpy.rotations as rot_utils
    import numpy as np
    import matplotlib.pyplot as plt
    
    camera = Camera(
        prim_path="/World/camera",
        position=np.array([3.67, 0.218, 1.0]),
        frequency=50,
        resolution=(640, 480),
        orientation = rot_utils.euler_angles_to_quats(np.array([0, 88, 90]), degrees=True),)
    camera.set_world_pose(np.array([3.7245, 0.218, 1.053]), rot_utils.euler_angles_to_quats(np.array([88, 0, 90]), degrees=True), camera_axes="usd")
    camera.initialize()
    camera.add_motion_vectors_to_frame()

    camera2 = Camera(
        prim_path="/World/camera2",
        position=np.array([0.16304, 0.4922, 5]),
        frequency=50,
        resolution=(640, 480),
        orientation = rot_utils.euler_angles_to_quats(np.array([0, 0, -90]), degrees=True),)
    camera2.set_world_pose(np.array([0.16304, 0.4922, 5]), rot_utils.euler_angles_to_quats(np.array([0, 0, -90]), degrees=True), camera_axes="usd")
    camera2.initialize()
    camera2.add_motion_vectors_to_frame()

    # 환경 연산 디바이스(gpu)
    device = env.unwrapped.scene.device

    # Detection 모델 로드
    detection_model = get_model_instance_segmentation(NUM_CLASSES)
    detection_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    detection_model.eval()
    detection_model.to(device)

    # Contact-GraspNet 모델 config를 불러오기 위한 경로 설정
    grasp_model_config_path = os.path.join(DIR_PATH, 'cgnet/configs/config.yaml')
    grasp_model_config = cfg_from_yaml_file(grasp_model_config_path)

    # Contact-GraspNet 모델 선언 및 checkpoint 입력을 통한 모델 weight 로드
    grasp_model = builder.model_builder(grasp_model_config.model)
    grasp_model_path = os.path.join(DIR_PATH, 'data/checkpoint/contact_grasp_ckpt/ckpt-iter-60000_gc6d.pth')
    builder.load_model(grasp_model, grasp_model_path)
    grasp_model.to(device)
    grasp_model.eval()

    # CLIP 모델 로드
    clip_model, preprocess = clip.load("ViT-B/32", device='cpu')

    print("[INFO]: Setup complete...")

    # 로봇 pick-and-place 제어를 위한 State machine 선언
    pick_and_place_sm = PickAndPlaceSm(
        dt=env_cfg.sim.dt * env_cfg.decimation,
        num_envs=num_envs,
        device=device,
        position_threshold=0.01
    )

    # 환경에서 robot handeye camera 변수 불러오기
    robot_camera = env.unwrapped.scene.sensors['camera']
    robot_camera2 = env.unwrapped.scene.sensors['camera2']

    # 카메라 인트린식(intrinsics)
    K = robot_camera.data.intrinsic_matrices.squeeze().cpu().numpy()
    
    # Create a directory for saving logs
    # log_dir = f"simulation_logs_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    # os.makedirs(log_dir, exist_ok=True)    
    max_steps_per_traj=500
    max_trajectories=50
    total_traj = 0
    dataset = {
        "EE_pose": [],
        "obs": [],
        "applied_torque": [],
    }
    # 시뮬레이션 루프
    while simulation_app.is_running() and total_traj < max_trajectories:
        env.reset()
        pick_and_place_sm.reset_idx()
        print(f"[INFO] Starting trajectory {total_traj+1}/{max_trajectories}")
        
        done = False
        step_count = 0
        save_count=0
        is_success = False
        dataset = {
            "EE_pose": [],
            "obs": [],
            "applied_torque": [],
        }
        # 우선 success/failure 라벨 없는 폴더 생성
        log_dir = f"simulation_traj_{total_traj}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "front_view"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "wrist_view"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "top_view"), exist_ok=True)
        while not done and step_count < max_steps_per_traj:
        # 모델 추론 상태 - 학습 연산 비활성화
            with torch.no_grad(): #with torch.inference_mode():
                # env 별 시뮬레이션 루프 실행
                for env_num in range(num_envs):
                    # 현재 state가 Precdict일때, Detection-GraspPrediction-CLIP 순으로 추론 진행
                    if pick_and_place_sm.sm_state[env_num] == PickAndPlaceSmState.PREDICT:
                        # 시각화를 위한 RGB 이미지 및 Depth 이미지 얻기

                        image_ = robot_camera.data.output["rgb"][env_num]
                        image = image_.permute(2, 0, 1)  # (channels, height, width)
                        img_np = image_.detach().cpu().numpy()

                        # Continue with depth processing if needed
                        normalized_image = (image - image.min()) / (image.max() - image.min())
                        depth = robot_camera.data.output["distance_to_image_plane"][env_num]
                        depth_np = depth.squeeze().detach().cpu().numpy()                    # 취득한 Depth 이미지를 통한 Point Cloud 생성
                        if num_envs > 1:

                            pc, _ = depth2pc(depth_np, K[env_num])
                        else:
                            pc, _ = depth2pc(depth_np, K)

                ############################ Detection Model Inference ############################
                        print("Running detection inference...")

                        # Detection 모델 추론
                        with torch.no_grad():
                            prediction = detection_model([normalized_image])
                        
                        # 결과 후처리 및 시각화
                        img_np = cv2.cvtColor(np.array(img_np), cv2.COLOR_RGB2BGR)
                        
                        # Bbox와 텍스트를 그릴 이미지 레이어
                        img_with_boxes = img_np.copy()

                        # 마스크를 그릴 투명한 이미지 레이어
                        mask_overlay = img_np.copy()
                        
                        # prediction에서 pred_scores, pred_boxes, pred_masks, pred_labels 값을 추출
                        pred_scores = prediction[0]['scores'].cpu().numpy()
                        pred_boxes = prediction[0]['boxes'].cpu().numpy()
                        pred_masks = prediction[0]['masks'].cpu().numpy()
                        pred_labels = prediction[0]['labels'].cpu().numpy()

                        print(f"Found {len(pred_scores)} objects. Visualizing valid results...")

                        # Detection 결과를 rgb 이미지 위에 표시
                        # 각 인스턴스에 대한 크롭된 이미지 저장용
                        crop_images = []
                        # 바운딩 박스 정보 저장용
                        bboxes = []
                        for i in range(len(pred_scores)):
                            score = pred_scores[i]
                            label_id = pred_labels[i]

                            # 신뢰도와 레이블 ID를 함께 확인하여 Background 제외
                            if score > CONFIDENCE_THRESHOLD and label_id != 0:
                                color = get_random_color()
                                
                                # --- Bbox와 텍스트 그리기 ---
                                box = pred_boxes[i]
                                x1, y1, x2, y2 = map(int, box)
                                #cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
                                
                                class_names = CLASS_NAME
                                label_text = f"{class_names[label_id]}: {score:.2f}"
                                cv2.putText(img_with_boxes, label_text, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # --- 마스크 그리기 ---
                                mask = pred_masks[i, 0]
                                binary_mask = (mask > 0.5) # Boolean mask
                                # 마스크 영역에만 색상 적용
                                mask_overlay[binary_mask] = color

                                # --- 바운딩 박스와 크롭된 이미지 저장 ---
                                bboxes.append(box)
                                crop_image = image[:, int(y1):int(y2), int(x1):int(x2)]
                                crop_images.append(crop_image)

                        # Bbox와 Mask를 분리해서 그린 후 마지막에 한 번만 합성
                        alpha = 0.5 # 마스크 투명도
                        #final_result = cv2.addWeighted(mask_overlay, alpha, img_with_boxes, 1 - alpha, 0)

                        # plt로 한번에 보여주기
                        # if crop_images:
                        #     # 총 이미지 개수: final_result + crop된 이미지들
                        #     num_total_images = 1 + len(crop_images)
                        #     cols = 4  # 한 줄에 보여줄 이미지 수
                        #     rows = (num_total_images + cols - 1) // cols
                        #     fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
                        #     fig.suptitle('Detection Results', fontsize=16)

                        #     # axes가 1차원 배열이 되도록 조정
                        #     if rows * cols == 1:
                        #         axes = [axes]
                        #     else:
                        #         axes = axes.flatten()

                        #     # 첫번째 subplot에 final_result 이미지 표시
                        #     #final_result_rgb = cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB)
                        #     # axes[0].imshow(final_result_rgb)
                        #     # axes[0].set_title('Robot View with Detections')
                        #     # axes[0].axis('off')

                        #     # 두번째 subplot부터 crop된 이미지들 표시
                        #     valid_preds_count = 0
                        #     for i in range(len(pred_scores)):
                        #         score = pred_scores[i]
                        #         label_id = pred_labels[i]

                        #         if score > CONFIDENCE_THRESHOLD and label_id != 0:
                        #             ax = axes[valid_preds_count + 1]
                                    
                        #             # # 이미지 Display
                        #             # img_to_show = crop_images[valid_preds_count].permute(1, 2, 0).cpu().numpy()
                        #             # # ax.imshow(img_to_show)
                                    
                        #             # # 각 이미지의 Class 이름 표시
                        #             # title = f"{CLASS_NAME[label_id]}\nScore: {score:.2f}"
                        #             # ax.set_title(title)
                        #             # ax.axis('off')
                                    
                        #             valid_preds_count += 1

                        #     # 사용하지 않는 subplots 끄기
                        #     for i in range(num_total_images, len(axes)):
                        #         axes[i].axis('off')

                        #     # Layout 조정 및 결과 이미지 저장
                        #     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                        #     plt.savefig("data/detection_result.png")

                ####################################################################################

                ############################ CLIP Model Inference ##################################
                        # 각 crop image 별 input text와의 유사도 저장
                        probs =  []
                        #user_text = pick_order.pop(0)
                        user_text = "bowl" #sys.stdin.readline().strip()
                        
                        for crop_image in crop_images:
                            # text를 deep learning 모델에 넣기 위해 token으로 변환
                            text = clip.tokenize([user_text]).to(device)
                            
                            # image를 clip model의 input size로 변환
                            crop_image = CLIP_transform(crop_image, 224)
                            crop_image = crop_image.unsqueeze(0)
                            
                            # CLIP 모델 추론  및 유사도 저장 (현장 강의 컴퓨터 메모리 문제로 cpu 연산)
                            with torch.no_grad():
                                logits_per_image, logits_per_text = clip_model(crop_image.cpu(), text.cpu())
                                probs.append(logits_per_image.cpu().numpy())
                                
                        # crop image 중에서 input text와 가장 유사도가 큰 이미지 선택
                        target_obj_idx = np.argmax(np.array(probs))
                        target_obj_bbox = bboxes[target_obj_idx]
                        target_image = crop_images[target_obj_idx].permute(1, 2, 0).cpu().numpy()
                        
                        # BGR 이미지를 RGB로 변환 
                        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
                        
                        # CLIP model 결과 cv2로 저장
                        #cv2.imwrite(f'data/CLIP_result_{user_text}.png', target_image)

                ####################################################################################

                ############################ Grasp Model Inference ##################################
                        # targe object의 bbox 위치를 image 좌표에서 world 좌표로 변환
                        if num_envs > 1:
                            x_min_w, y_min_w, x_max_w, y_max_w = get_world_bbox(depth_np, K[env_num], target_obj_bbox)
                        else:
                            x_min_w, y_min_w, x_max_w, y_max_w = get_world_bbox(depth_np, K, target_obj_bbox)

                        # Robot의 end-effector 위치 얻기
                        robot_entity_cfg = SceneEntityCfg("robot", body_names=["panda_hand"])
                        robot_entity_cfg.resolve(env.unwrapped.scene)
                        hand_body_id = robot_entity_cfg.body_ids[0]
                        hand_pose_w = env.unwrapped.scene["robot"].data.body_state_w[:, hand_body_id, :]  # (num_envs, 13)

                        if pc is not None:
                            offset = 0.08 # 바닥이 너무 조금 나올 경우, 바닥에 파지점이 생김
                            # target object가 있는 부분의 point cloud를 world bbox 기준으로 offset을 주고 crop
                            pc = pc[pc[:, 0] > x_min_w-offset]
                            pc = pc[pc[:, 0] < x_max_w+offset]
                            pc = pc[pc[:, 1] > y_min_w-offset]
                            pc = pc[pc[:, 1] < y_max_w+offset]
                            pc = pc[pc[:, 2] > 0.4]  # z
                            
                            # target object의 3d point cloud 시각화
                            # pc_o3d = o3d.geometry.PointCloud()
                            # pc_o3d.points = o3d.utility.Vector3dVector(pc)
                            # o3d.visualization.draw_geometries([pc_o3d])
                            
                            # Contact-GraspNet 모델 추론
                            rot_ee, trans_ee, width = inference_cgnet(pc, grasp_model, device, hand_pose_w, env)
                            print(f"[INFO] Received ee coordinates from inference_cgnet")
                            print(f"[INFO] Gripper width: {width}")
                            
                            # 예측한 파지점을 Isaaclab 형식으로 변환 (rotation matrix -> quat)
                            grasp_rot = rot_ee
                            pregrasp_pos = trans_ee
                            grasp_quat = R.from_matrix(grasp_rot).as_quat()  # (x, y, z, w)
                            grasp_quat = np.array([grasp_quat[3], grasp_quat[0], grasp_quat[1], grasp_quat[2]]) # (w, x, y, z)
                            
                            # rotation matrix를 사용하여 예측한 파지점의 offset 맞추기
                            z_axis = grasp_rot[:, 2]
                            grasp_pos = pregrasp_pos + z_axis * 0.085 # change : miss put point  

                            # 예측한 파지점 pose를 torch tensor로 변환
                            pregrasp_pose = np.concatenate([pregrasp_pos, grasp_quat])
                            grasp_pose = np.concatenate([grasp_pos, grasp_quat])
                            pregrasp_pose = torch.tensor(pregrasp_pose, device=device).unsqueeze(0)
                            grasp_pose = torch.tensor(grasp_pose, device=device).unsqueeze(0)

                            # State machine 에 grasp 및 pregrasp 자세 업데이트
                            pick_and_place_sm.grasp_pose[env_num] = grasp_pose[0]
                            pick_and_place_sm.pregrasp_pose[env_num] = pregrasp_pose[0]
                ####################################################################################

                

                # 로봇의 End-Effector 위치와 자세를 기반으로 actions 계산
                robot_data = env.unwrapped.scene["robot"].data
                env.unwrapped.scene["robot"].data.applied_torque[0, -2:] = 0.0
                ee_frame_sensor = env.unwrapped.scene["ee_frame"]
                tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - env.unwrapped.scene.env_origins
                tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
                ee_pose = torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1)
                imitation_obs = torch.cat([ee_pose[0], pick_and_place_sm.des_gripper_state[0], env.unwrapped.unwrapped.scene.state['rigid_object']['object_0']['root_pose'][0][:3]], dim=0)
                hist.add(imitation_obs)
                # print("step = ", save_count)
                # print(env.unwrapped.unwrapped.scene.state['rigid_object']['object_0']['root_pose'])
                # print(ee_pose)
                # print(pick_and_place_sm.des_gripper_state)
                # imitation_obs = torch.cat([ee_pose[0], pick_and_place_sm.des_gripper_state[0], env.unwrapped.unwrapped.scene.state['rigid_object']['object_0']['root_pose'][0][:3]], dim=0)   # (1,14)
                # print(imitation_obs.shape)
#env.unwrapped.unwrapped.scene.state['rigid_object']['object_0']['root_pose']
                # state machine 을 통한 action 값 출력
                actions = pick_and_place_sm.compute(
                    ee_pose=ee_pose,
                    grasp_pose=pick_and_place_sm.grasp_pose,
                    pregrasp_pose=pick_and_place_sm.pregrasp_pose,
                    robot_data=robot_data,
                    current_step= step_count, 
                )  
                #print("ee_pose = ", ee_pose)
                # 환경에 대한 액션을 실행
                
                
                
                #print("step_count = ", step_count)
                
                #print(f"obs = ", obs)
                if pick_and_place_sm.sm_state >=1: #after predict
                    
                    
                    #print("save_count = ", save_count)
                    if save_count % 5 == 0:
                        dataset["EE_pose"].append(ee_pose[0])
                        dataset["obs"].append(obs['policy'][0])
                        dataset["applied_torque"].append(robot_data.applied_torque[0])  
                        rgb_image = camera.get_rgba()
                        if rgb_image.shape[0] != 0:
                            rgb = rgb_image[:, :, :3]
                            if rgb.max() <= 1.0:
                                rgb = (rgb * 255).astype(np.uint8)
                            else:
                                rgb = rgb.astype(np.uint8)
                            img_pil_ = Image.fromarray(rgb)
                            # img_pil_.save(os.path.join(log_dir, f'front_view_{freq}.png'))
                            new_w, new_h = img_pil_.size[0] // 2, img_pil_.size[1] // 2
                            img_pil_resized = img_pil_.resize((new_w, new_h), Image.Resampling.LANCZOS)

                            img_pil_resized.save(os.path.join(log_dir, f'front_view/front_view_{save_count}.png'))

                        rgb_image = camera2.get_rgba()
                        if rgb_image.shape[0] != 0:
                            rgb = rgb_image[:, :, :3]
                            if rgb.max() <= 1.0:
                                rgb = (rgb * 255).astype(np.uint8)
                            else:
                                rgb = rgb.astype(np.uint8)
                            img_pil_ = Image.fromarray(rgb)
                            #img_pil_.save(os.path.join(log_dir, f'top_view_{freq}.png'))
                            new_w, new_h = img_pil_.size[0] // 2, img_pil_.size[1] // 2
                            img_pil_resized = img_pil_.resize((new_w, new_h), Image.Resampling.LANCZOS)

                            img_pil_resized.save(os.path.join(log_dir, f'top_view/top_view_{save_count}.png'))

                        # Save robot_camera2 image (im2)
                        im2 = robot_camera2.data.output['rgb'][env_num]
                        if len(im2.shape) == 3:  # Expected shape: (channels, height, width) or (height, width, channels)
                            if im2.shape[0] in [3, 4]:  # If channels-first (e.g., (3, height, width))
                                im2 = im2.permute(1, 2, 0)  # Convert to (height, width, channels)
                            im2_np = im2.detach().cpu().numpy()
                            if im2_np.max() <= 1.0:
                                im2_np = (im2_np * 255).astype(np.uint8)
                            else:
                                im2_np = im2_np.astype(np.uint8)
                            img_pil = Image.fromarray(im2_np)
                            new_w, new_h = img_pil.size[0] // 2, img_pil.size[1] // 2
                            img_pil_resized = img_pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
                            img_pil_resized.save(os.path.join(log_dir, f'wrist_view/wrist_view_{save_count}.png'))                       
                            #img_pil.save(os.path.join(log_dir, f'wrist_view_{freq}.png'))
                    save_count+=1
                    # # Save viewport image
                    # vp = viewport_utils.get_active_viewport()
                    # viewport_utils.capture_viewport_to_file(vp, os.path.join(log_dir, f'viewport_{freq}.png'))
                    # Concatenate states into a dictionary and save as .npz (multi-array file)
                    # robotstate = torch.cat([ee_pose, torch.tensor(obs["policy"][0, 7:9].tolist(), device=ee_pose.device).unsqueeze(0)], dim=-1).cpu().numpy()
                    # data_dict = {
                    #     'robot_state': robotstate,
                    # }
                    # np.savez(os.path.join(log_dir, f'states_{step_count}.npz'), **data_dict)

                obs_hist_3= hist.get()
                pred_next = predict_next8(obs_hist_3, "/AILAB-summer-school-2025/next8/best.pt")
                print(pred_next)
                obs, rewards, terminated, truncated, info = env.step(pred_next)
                print(actions)
                # print("===================")
                # print(ee_pose[0])
                # print(obs['policy'][0])
                # print(robot_data.applied_torque)
                step_count += 1

                # 시뮬레이션 종료 여부 체크
                dones = terminated | truncated
                if dones:
                    done = True
                    if terminated:
                        is_success = True
                        print("Episode terminated")
                    else:
                        print("Episode truncated")

        #print(f"eepose len = {len(dataset['EE_pose'])}, obs len = {len(dataset['obs'])}, applied_torque len = {len(dataset['applied_torque'])}")
        for k in dataset:
            dataset[k] = np.array([
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
                for v in dataset[k]
            ])
        np.savez(os.path.join(log_dir, "robot_state.npz"), **dataset)
        traj_length = save_count
        if is_success:
            final_log_dir = f"{log_dir}_len{traj_length}_success"
        else:
            noise_starting_point = pick_and_place_sm.noise_start_step[env_num]
            if noise_starting_point is not None:
                final_log_dir = f"{log_dir}_len{traj_length}_failure_{noise_starting_point}step"
            else:
                final_log_dir = f"{log_dir}_len{traj_length}_failure"
        os.rename(log_dir, final_log_dir)
        
        print(f"[INFO] Trajectory {total_traj+1} saved at {final_log_dir}")
        total_traj += 1 
    # 환경 종료 및 시뮬레이션 종료
    env.close()
    simulation_app.close()       
# 메인 함수 실행
if __name__ == "__main__":
    main()