import numpy as np
# 로드
data = np.load("/AILAB-summer-school-2025/scripts/total_robot_state.npz")

# 사용 예시
for key in data.files:
    print(key, data[key].shape)

# 예: 'episode1/obs', 'episode2/EE_pose' 등의 키가 생성됨
print(data[episode1])