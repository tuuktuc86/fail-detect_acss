import os
import numpy as np
import pickle
import re

def collect_robot_states_as_nested_dict(root_dir, output_file="merged_robot_state.pkl"):
    merged_data = {}

    # (1) episode 번호 정수 기준 정렬을 위한 리스트 생성
    episode_dirs = []
    for name in os.listdir(root_dir):
        dir_path = os.path.join(root_dir, name)
        if os.path.isdir(dir_path) and name.startswith("success_episode") and "robot_state.npz" in os.listdir(dir_path):
            # 에피소드 번호 추출
            match = re.search(r"success_episode(\d+)_steps\d+", name)
            if match:
                episode_number = int(match.group(1))
                episode_dirs.append((episode_number, dir_path))

    # (2) episode_number 기준으로 정렬
    episode_dirs.sort(key=lambda x: x[0])

    # (3) 정렬된 순서로 데이터 로딩
    for episode_number, dir_path in episode_dirs:
        episode_key = f"episode{episode_number}"

        state_path = os.path.join(dir_path, "robot_state.npz")
        data = np.load(state_path, allow_pickle=True)

        merged_data[episode_key] = {
            "EE_pose": data["EE_pose"],
            "obs": data["obs"],
            "applied_torque": data["applied_torque"]
        }

    # (4) 저장
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)

    print(f"Saved merged data with {len(merged_data)} episodes to {output_file}")
    return merged_data

collect_robot_states_as_nested_dict("/AILAB-summer-school-2025/success_data_raw")
