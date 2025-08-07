import os
import numpy as np
import pickle

def collect_robot_states_as_nested_dict(root_dir, output_file="merged_robot_state.pkl"):
    merged_data = {}

    for name in sorted(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, name)

        if os.path.isdir(dir_path) and name.startswith("success_episode") and "robot_state.npz" in os.listdir(dir_path):
            # 에피소드 이름 설정 (e.g., episode1, episode2)
            episode_number = name.split("_")[1].replace("episode", "")
            episode_key = f"episode{episode_number}"

            # robot_state.npz 로드
            state_path = os.path.join(dir_path, "robot_state.npz")
            data = np.load(state_path, allow_pickle=True)

            merged_data[episode_key] = {
                "EE_pose": data["EE_pose"],
                "obs": data["obs"],
                "applied_torque": data["applied_torque"]
            }

    # .pkl로 저장 (딕셔너리 안에 딕셔너리 형태 유지 가능)
    with open(output_file, "wb") as f:
        pickle.dump(merged_data, f)

    print(f"Saved merged data with {len(merged_data)} episodes to {output_file}")
    return merged_data
collect_robot_states_as_nested_dict("/AILAB-summer-school-2025/success_data_raw")
