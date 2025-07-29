import os
import numpy as np
from PIL import Image

# 경로 설정 (저장된 npz와 이미지 폴더가 실제로 있는 경로로 수정)
base_dir = "/AILAB-summer-school-2025/success_traj"
output_npz = os.path.join(base_dir, "success_traj_state.npz")
output_img_dir = os.path.join(base_dir, "success_traj_img")

traj_folders = [
    f for f in os.listdir(base_dir) 
    if os.path.isdir(os.path.join(base_dir, f)) and f.startswith("simulation_traj_")
]
num_trajs = len(traj_folders)

# npz 파일 구조 확인
npz_info = {}
if os.path.exists(output_npz):
    data = np.load(output_npz, allow_pickle=True)
    npz_info['keys'] = list(data.keys())
    npz_info['shapes'] = {k: data[k].shape for k in data.keys()}
    npz_info['dtypes'] = {k: data[k].dtype for k in data.keys()}
    data.close()
else:
    npz_info['error'] = f"NPZ file not found at {output_npz}"

# 이미지 파일 크기 확인
img_info = {}
for view in ["front_view", "top_view", "wrist_view"]:
    view_dir = os.path.join(output_img_dir, view)
    if os.path.exists(view_dir):
        sizes = []
        for file in os.listdir(view_dir):
            if file.endswith(".png"):
                path = os.path.join(view_dir, file)
                with Image.open(path) as img:
                    sizes.append(img.size)
        if sizes:
            img_info[view] = {
                "num_images": len(sizes),
                "first_image_size": sizes[0],
                "all_sizes_equal": len(set(sizes)) == 1
            }
        else:
            img_info[view] = {"num_images": 0}
    else:
        img_info[view] = {"error": f"Directory not found at {view_dir}"}

print(f"전체 Trajectory 개수: {num_trajs}")
print("NPZ Info:", npz_info)
print("Image Info:", img_info)