import os
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

# 동일한 구조의 모델 정의
resnet18 = models.resnet18()
resnet18.fc = torch.nn.Identity()

# 저장된 weight 로드
weight_path = "/AILAB-summer-school-2025/scripts/model/resnet18_512.pth"  # 경로 수정
resnet18.load_state_dict(torch.load(weight_path, map_location=device))
resnet18 = resnet18.to(device)
resnet18.eval()

# weight 확인용 hash
print("Weight hash:", hash(str(resnet18.state_dict())))
# ✅ 2. 이미지 전처리 transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ✅ 3. 입력 경로 및 파라미터 설정
npz_path = "/AILAB-summer-school-2025/dataset_preprocessing/image_features(resnet_18).npz"
success_data_raw = "/AILAB-summer-school-2025/success_data_raw"
episode_idx = 1  # 검증할 에피소드 인덱스
stride = 5       # 저장된 프레임 간격

# ✅ 4. .npz 파일 로딩
data = np.load(npz_path, allow_pickle=True)
front_view = data["front_view"]
ep_feat_seq = front_view[episode_idx]  # shape: [T, 512]
T = ep_feat_seq.shape[0]

# ✅ 5. episode 디렉토리 이름 찾기 (저장 시 기준 정렬)
episode_dirs = sorted(
    [d for d in os.listdir(success_data_raw) if d.startswith("success_episode")],
    key=lambda name: int(re.search(r"success_episode(\d+)", name).group(1))
)
episode_name = episode_dirs[episode_idx]
episode_path = os.path.join(success_data_raw, episode_name, "front_view")

print(f"🔍 Checking episode: {episode_name}, T = {T}, stride = {stride}")

# ✅ 6. 점검 루프
num_fail = 0
for i in range(T):
    frame_number = i * stride
    filename = f"front_view_{frame_number}.png"  # no zero padding
    filepath = os.path.join(episode_path, filename)

    if not os.path.exists(filepath):
        print(f"❌ Missing file: {filepath}")
        num_fail += 1
        continue

    image = Image.open(filepath).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    #print("shape = ", img_tensor.shape)
    with torch.no_grad():
        feat_from_resnet = resnet18(img_tensor).cpu().numpy().squeeze()

    feat_from_npz = ep_feat_seq[i]
    is_same = np.allclose(feat_from_npz, feat_from_resnet, atol=1e-6)

    if not is_same:
        diff = np.abs(feat_from_npz - feat_from_resnet)
        print(f"⚠️  Frame {i} (image {frame_number}.png): mismatch")
        print(f"   → Max diff: {np.max(diff):.6f}, Mean diff: {np.mean(diff):.6f}")
        num_fail += 1

if num_fail == 0:
    print("✅ All features match.")
else:
    print(f"❌ {num_fail}/{T} frames failed consistency check.")
