import os
import numpy as np

def collect_robot_states(root_dir, output_file="total_robot_state.npz"):
    # 병합된 데이터 저장할 딕셔너리
    merged_data = {}
    
    # 디렉토리 순회
    for name in sorted(os.listdir(root_dir)):
        dir_path = os.path.join(root_dir, name)
        if os.path.isdir(dir_path) and name.startswith("success_episode") and "robot_state.npz" in os.listdir(dir_path):
            # 에피소드 이름을 key로 사용 (e.g., episode1, episode2, ...)
            episode_key = name.replace("success_", "")  # 또는 단순히 name 자체 써도 됨
            
            # robot_state.npz 로드
            state_path = os.path.join(dir_path, "robot_state.npz")
            data = np.load(state_path, allow_pickle=True)

            # numpy object는 dict로 변환
            merged_data[episode_key] = {
                "EE_pose": data["EE_pose"],
                "joint_": data["obs"],
                "applied_torque": data["applied_torque"]
            }
    
    # np.savez로 저장 (dictionary of dictionaries는 np.savez로 바로 저장 불가 → workaround 필요)
    # → 각 episode의 key를 flatten 해서 저장
    save_dict = {}
    for ep, contents in merged_data.items():
        for key, value in contents.items():
            save_dict[f"{ep}/{key}"] = value

    # 저장
    np.savez(output_file, **save_dict)
    print(f"Saved merged robot_state.npz with {len(merged_data)} episodes to {output_file}")

    return merged_data  # optional: 반환해도 좋음

collect_robot_states("/AILAB-summer-school-2025/success_data_raw")