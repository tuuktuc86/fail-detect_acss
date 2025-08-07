
import os
import numpy as np
import pickle
with open("/AILAB-summer-school-2025/scripts/merged_robot_state.pkl", "rb") as f:
    data = pickle.load(f)

print(data.keys())           # episode1, episode2, ...
print(data["episode1"].keys())  # EE_pose, obs, applied_torque
print(data["episode1"]["applied_torque"].shape)  # 예: (T, 29)

