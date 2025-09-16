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
    print(f"reaching = {1 - torch.tanh(object_ee_distance / std)}")
    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
    # object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]

    des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    print(f"goal_dist = {(object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))}")
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))

def fixed_bin(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    bin_cfg: SceneEntityCfg = SceneEntityCfg("bin"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    bin: RigidObject = env.scene[bin_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_w = command[:, :3]

    #des_pos_w, _ = combine_frame_transforms(robot.data.root_state_w[:, :3], robot.data.root_state_w[:, 3:7], des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - bin.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    print(f"bin_fixed = {torch.tanh(distance / std)}")
    return torch.tanh(distance / std)


def release(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object_0"),
    # object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    # bin_cfg: SceneEntityCfg = SceneEntityCfg("bin")
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    ee_frame: RigidObject = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    # bin: RigidObject = env.scene[bin_cfg.name]

    command = env.command_manager.get_command(command_name)
    ee_w = ee_frame.data.target_pos_w[..., 0:3, :]
    # compute the desired position in the world frame
    des_pos_w = command[:, :3]

    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w[:, :3], dim=1)
    # rewarded if the object is lifted above the threshold
    print(f"release_dist = {(ee_w[0][0][0] > -0.1 and ee_w[0][0][0] < 0.5)*(ee_w[0][0][1] > 0.25 and ee_w[0][0][1] < 0.8) *(ee_w[0][0][2] > minimal_height) * (1 - torch.tanh(distance / std))}")
    return (ee_w[0][0][0] > -0.1 and ee_w[0][0][0] < 0.5)*(ee_w[0][0][1] > 0.25 and ee_w[0][0][1] < 0.8) *(ee_w[0][0][2] > minimal_height) * (1 - torch.tanh(distance / std))