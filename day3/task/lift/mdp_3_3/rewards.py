# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# def object_is_lifted(
#     env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object_0")
#     # env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg
# ) -> torch.Tensor:
#     """Reward the agent for lifting the object above the minimal height."""
#     object: RigidObject = env.scene[object_cfg.name]
#     #print("object_height = ", object.data.root_pos_w[:, 2])
#     print(f"lift = {torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)}")
#     return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
    # object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the rewardsused quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)
    
    # print(f"reaching_distance = {object_ee_distance}")
    # print(f"reaching = {1 - torch.tanh(object_ee_distance / std)}")
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """Reward the agent for moving object toward bin pose + z-offset using tanh-kernel."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    bin_obj: RigidObject = env.scene[bin_cfg.name]

    # bin 중심 좌표 [N, 3]
    bin_pos = bin_obj.data.root_pos_w[:, :3]

    # 목표 위치 = bin 위치 + z축 0.5m offset
    des_pos_w = bin_pos + torch.tensor([0.0, 0.0, 0.2], device=bin_pos.device)

    # object와 목표 위치 사이 거리
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)

    # print(f"goal_distacne = {distance}")
    # print(f"object_heigtht = {object.data.root_pos_w[:, 2]}")
    # print(f"goal_reward = {(object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))}")

    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def fixed_bin(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """Reward the agent for staying within tolerance of the goal pose."""
    bin: RigidObject = env.scene[bin_cfg.name]
    des_pos_w = torch.tensor([0.2000, 0.6000, 0.5699], device="cuda:0")

    # 현재 bin과 목표 위치 사이 거리
    distance = torch.norm(des_pos_w - bin.data.root_pos_w[:, :3], dim=1)

    # 허용 오차
    tolerance = 0.01

    # tolerance 안 → 0
    # tolerance 밖 → (distance - tolerance) / std → tanh 정규화
    reward = torch.where(
        distance <= tolerance,
        torch.zeros_like(distance),
        torch.tanh((distance - tolerance) / std)
    )
    print(f"bin_distance = {distance}")
    return reward


def release(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
    # object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin")
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    #  bin: RigidObject = env.scene[bin_cfg.name]
    command = env.command_manager.get_command(command_name)
    ee_w = ee_frame.data.target_pos_w[..., 0:3, :]
    # compute the desired position in the world frame
    des_pos_w = torch.tensor([0.2000, 0.6000, 0.5699], device="cuda:0")

    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    # print(f"release_dist = {(ee_w[0][0][0] > -0.1 and ee_w[0][0][0] < 0.5)*(ee_w[0][0][1] > 0.25 and ee_w[0][0][1] < 0.8) *(ee_w[0][0][2] > minimal_height) * (1 - torch.tanh(distance / std))}")
    return (ee_w[0][0][0] > 0.0 and ee_w[0][0][0] < 0.5)*(ee_w[0][0][1] > 0.4 and ee_w[0][0][1] < 0.8) *(ee_w[0][0][2] > minimal_height) * (1 - torch.tanh(distance / std))


def object_in_bin_without_ee_near_bin_sparse(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    # 객체-빈 근접 판정 임계값
    threshold_xy: float = 0.25,
    threshold_z: float = 0.05,
    # EE-빈 근접 판정 임계값 (EE는 이 범위 "밖"이어야 보상)
    ee_threshold: float = 0.4,
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    bin_obj: RigidObject = env.scene[bin_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    bin_pos = bin_obj.data.root_pos_w           # [N, 3]
    obj_pos = obj.data.root_pos_w               # [N, 3]
    ee_pos  = ee_frame.data.target_pos_w[..., 0, :]  # [N, 3]

    # 객체-빈 거리
    d_xy = torch.norm(bin_pos[:, :2] - obj_pos[:, :2], dim=1)   # [N]
    d_z  = torch.abs(bin_pos[:, 2] - obj_pos[:, 2])             # [N]
    in_bin = (d_xy < threshold_xy) & (d_z < threshold_z)        # [N] bool

    # EE-빈 거리
    d_ee_bin = torch.norm(ee_pos - bin_pos, dim=1)              # [N]
    ee_outside = d_ee_bin > ee_threshold                        # [N] bool

    # 조건: 객체는 bin 안, EE는 bin 밖
    reward_bool = in_bin & ee_outside                           # [N] bool
    return reward_bool.float()                                   # sparse: 0 or 1
