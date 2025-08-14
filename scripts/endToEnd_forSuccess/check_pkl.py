import pickle
import torch
import numpy as np
# pkl 파일 경로
net_path = "/AILAB-summer-school-2025/scripts/endToEnd_forSuccess/dataset/success_data_resnet18_pca_robotO.pkl"
gnet_path = "/AILAB-summer-school-2025/scripts/endToEnd_forSuccess/dataset/success_data_resnet18_pca_robotX.pkl"

with open(net_path, "rb") as f: net_data = pickle.load(f)
with open(gnet_path, "rb") as f: gnet_data = pickle.load(f)

def flat192(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x.ravel()[:192]

k1 = list(net_data.keys())[0]
k2 = list(gnet_data.keys())[0]
print(k1)

a = flat192(net_data[k1])
b = flat192(gnet_data[k2])

# 정확 비교
same_exact = np.array_equal(a, b)
print("정확 동일 여부(192개):", same_exact)

# 부동소수 오차 허용 비교
same_close = np.allclose(a, b, rtol=1e-6, atol=1e-8)
print("근사 동일 여부(192개):", same_close)

# 차이 위치 일부 출력
diff_idx = np.where(~np.isclose(a, b, rtol=1e-6, atol=1e-8))[0]
print("상이한 인덱스(최초 10개):", diff_idx[:10])

print("net[:10] =", a[:10])
print("gnet[:10] =", b[:10])