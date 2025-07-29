import os
import numpy as np
import shutil

base_dir = os.path.expanduser("/AILAB-summer-school-2025/success_traj")
output_npz = os.path.join(base_dir, "success_traj_state.npz")
output_img_dir = os.path.join(base_dir, "success_traj_img")

# 결과 저장용 리스트
all_states = []

# 이미지 폴더 구조 생성
for view in ["front_view", "top_view", "wrist_view"]:
    os.makedirs(os.path.join(output_img_dir, view), exist_ok=True)

# 각 simulation_traj_* 폴더 순회
for traj_folder in sorted(os.listdir(base_dir)):
    traj_path = os.path.join(base_dir, traj_folder)
    if not os.path.isdir(traj_path) or not traj_folder.startswith("simulation_traj_"):
        continue

    # npz 파일 읽기
    for file in sorted(os.listdir(traj_path)):
        if file.endswith(".npz") and file.startswith("states_"):
            file_path = os.path.join(traj_path, file)
            data = np.load(file_path)
            for key in data.files:
                all_states.append(data[key])

    # 이미지 복사 (파일명에 traj_folder 이름 붙이기)
    for file in sorted(os.listdir(traj_path)):
        if file.endswith(".png"):
            if "front_view" in file:
                dest = os.path.join(output_img_dir, "front_view", f"{traj_folder}_{file}")
            elif "top_view" in file:
                dest = os.path.join(output_img_dir, "top_view", f"{traj_folder}_{file}")
            elif "wrist_view" in file:
                dest = os.path.join(output_img_dir, "wrist_view", f"{traj_folder}_{file}")
            else:
                continue
            shutil.copy(os.path.join(traj_path, file), dest)

# npz로 저장
np.savez_compressed(output_npz, states=np.array(all_states, dtype=object))

print(f"모든 states 저장 완료: {output_npz}")
print(f"이미지 저장 완료: {output_img_dir}")
