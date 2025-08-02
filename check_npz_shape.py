import numpy as np

# 파일 경로
file_path = "/AILAB-summer-school-2025/simulation_traj_1_20250802_144038_len462_success/robot_state.npz"

data = np.load(file_path, allow_pickle=True)

for key in data.files:
    array = data[key]
    print(f"🔍 Checking key: {key}")

    # 요소가 1차원 object 배열일 경우 (비정형 데이터)
    if array.dtype == object:
        shapes = [np.shape(v) for v in array]
    else:
        shapes = [np.shape(row) for row in array]

    unique_shapes = set(shapes)
    if len(unique_shapes) == 1:
        print(f"✅ All {len(shapes)} entries in '{key}' have the same shape: {unique_shapes.pop()}")
    else:
        print(f"❌ Inconsistent shapes in '{key}':")
        for i, s in enumerate(shapes):
            print(f"  - Index {i}: shape = {s}")