import os
import re
import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms
import re

def load_resnet_feature_extractor(device):
    # pretrained ResNet-18, 마지막 fc 층 Identity로 대체
    model = models.resnet18(pretrained=True)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model

def load_pca_model(view):
    pca_path = f'model/model_pca_{view}_view.pkl'
    with open(pca_path, 'rb') as f:
        return pickle.load(f)

def get_sorted_image_paths(dir_path, view):
    # view_view_#.png 에서 # 추출 후 오름차순 정렬
    pattern = re.compile(fr'{view}_view_(\d+)\.png')
    files = []
    for fname in os.listdir(dir_path):
        m = pattern.match(fname)
        if m:
            idx = int(m.group(1))
            files.append((idx, os.path.join(dir_path, fname)))
    files.sort(key=lambda x: x[0])
    return [path for _, path in files]

def process_episode(ep_dir, model, pca_models, device):
    # 1) robot_state.npz에서 EE pose 불러오기
    state = np.load(os.path.join(ep_dir, 'robot_state.npz'))
    if 'eepose' in state:
        ee_pose = state['eepose']           # shape (T,7)
    elif 'EE_pose' in state:
        ee_pose = state['EE_pose']
    else:
        raise KeyError(f'EE pose key not found in {ep_dir}/robot_state.npz')

    # 2) 이미지 전처리 트랜스폼
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std= [0.229,0.224,0.225]),
    ])

    # 3) 각 view별로 ResNet → PCA
    latents = []
    with torch.no_grad():
        for view in ['front', 'top', 'wrist']:
            view_dir = os.path.join(ep_dir, f'{view}_view')
            img_paths = get_sorted_image_paths(view_dir, view)
            feats512 = []
            for img_path in img_paths:
                img = Image.open(img_path).convert('RGB')
                x = transform(img).unsqueeze(0).to(device)
                f512 = model(x).cpu().numpy().reshape(-1)  # (512,)
                feats512.append(f512)
            feats512 = np.stack(feats512, axis=0)           # (T,512)
            feats64  = pca_models[view].transform(feats512) # (T,64)
            latents.append(feats64)

    # 4) front+top+wrist+ee_pose 합치기 → (T,199)
    data_episode = np.concatenate([latents[0], latents[1], latents[2], ee_pose], axis=1)
    return data_episode

def main():
    root_dir = '/AILAB-summer-school-2025/fail_data_raw'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ResNet-18 & PCA 모델 로드
    model = load_resnet_feature_extractor(device)
    pca_models = {
        view: load_pca_model(view)
        for view in ['front', 'top', 'wrist']
    }

    processed = {}
    # 케이스 디렉토리 순회
    for case_name in sorted(os.listdir(root_dir)):
        case_dir = os.path.join(root_dir, case_name)
        if not os.path.isdir(case_dir):
            continue

        # 각 에피소드 디렉토리 순회
        for ep_name in sorted(os.listdir(case_dir)):
            ep_dir = os.path.join(case_dir, ep_name)
            if not os.path.isdir(ep_dir):
                continue

            # 정규표현식으로 정확히 fail{n}_episode{m}_step{s}_noise{v} 패턴만 매칭
            m = re.match(r'^(fail\d+)_episode(\d+)_step\d+_noise(\d+)$', ep_name)
            if not m:
                print(f'[Warning] unexpected folder name "{ep_name}", skipping.')
                continue

            fail_str       = m.group(1)             # ex) 'fail1'
            episode_num    = m.group(2)             # ex) '2'
            noise_val      = int(m.group(3))        # ex) 180
            noise_idx      = noise_val // 5 + 1     # 5로 나눈 몫 + 1
            key            = f'{fail_str}_episode{episode_num}_noise{noise_idx}'

            print(f'Processing {key} ...')
            processed[key] = process_episode(ep_dir, model, pca_models, device)

    # 결과 저장
    out_path = 'fail_data_processed.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(processed, f)
    print(f'Saved processed data to {out_path}')

if __name__ == '__main__':
    main()
