import os
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

# ✅ ResNet18 feature extractor
resnet18 = models.resnet18(pretrained=True)
resnet18.fc = torch.nn.Identity()  # remove final fc
resnet18 = resnet18.to(device)
resnet18.eval()

# ✅ Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ✅ 특정 view에 맞는 PNG 파일 정렬 및 feature 추출
def extract_features(image_dir, view_name):
    def extract_index(filename):
        match = re.search(rf'{view_name}_(\d+)\.png', filename)
        return int(match.group(1)) if match else -1

    image_files = sorted(
        [f for f in os.listdir(image_dir) if f.endswith(".png")],
        key=extract_index
    )

    features = []
    for fname in image_files:
        img = Image.open(os.path.join(image_dir, fname)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet18(img_tensor)
        features.append(feat.cpu().numpy()[0])

    return np.stack(features, axis=0)  # shape: [T, 512]

# ✅ 에피소드 번호 정렬을 위한 유틸
def get_episode_number(ep_dir_name):
    match = re.search(r"success_episode(\d+)_steps(\d+)", ep_dir_name)
    return int(match.group(1)) if match else -1

# ✅ 전체 에피소드 처리 및 .npz 저장
def process_all_episodes(root_dir, output_npz):
    front_list, top_list, wrist_list = [], [], []

    episode_dirs = sorted(
        [d for d in os.listdir(root_dir) if d.startswith("success_episode")],
        key=get_episode_number
    )

    for i, ep_dir in enumerate(tqdm(episode_dirs, desc="Processing episodes")):
        ep_path = os.path.join(root_dir, ep_dir)
        print(f"[{i+1}/{len(episode_dirs)}] Processing: {ep_dir}")

        try:
            front_feat = extract_features(os.path.join(ep_path, "front_view"), "front_view")
            top_feat   = extract_features(os.path.join(ep_path, "top_view"), "top_view")
            wrist_feat = extract_features(os.path.join(ep_path, "wrist_view"), "wrist_view")
        except Exception as e:
            print(f"❌ Error in {ep_dir}: {e}")
            continue

        # ✅ 프레임 수가 다르면 스킵
        T_f, T_t, T_w = len(front_feat), len(top_feat), len(wrist_feat)
        if T_f != T_t or T_t != T_w:
            print(f"⚠️ Skipping {ep_dir} due to mismatched frame counts: front={T_f}, top={T_t}, wrist={T_w}")
            continue


        front_list.append(front_feat)
        top_list.append(top_feat)
        wrist_list.append(wrist_feat)

       
    # ✅ 저장 (object array 형태)

    front_data = np.array(front_list, dtype=object)
    top_data = np.array(top_list, dtype=object)
    wrist_data = np.array(wrist_list, dtype=object)
    # print(front_data.shape)
    # print(len(front_data[0]))
    # print(len(front_data[0][0]))
    np.savez_compressed(
        output_npz,
        front_view=front_data,
        top_view=top_data,
        wrist_view=wrist_data
    )
    print(f"✅ Saved to {output_npz}")


# ✅ 실행
process_all_episodes(
    root_dir="/AILAB-summer-school-2025/success_data_raw",
    output_npz="image_features(resnet18).npz"
)
