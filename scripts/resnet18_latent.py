import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# 1. ResNet18 모델 불러오기 + FC 레이어 제거
resnet18 = models.resnet18(pretrained=True)
resnet18_feature_extractor = torch.nn.Sequential(*list(resnet18.children())[:-1])  # 마지막 fc 제거
resnet18_feature_extractor.eval()

# 2. 이미지 전처리 transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 3. 이미지 2장 불러오기 (예: image1.jpg, image2.jpg)
img1 = Image.open("/AILAB-summer-school-2025/success_data_raw/success_episode1_steps460/front_view/front_view_45.png").convert("RGB")
img2 = Image.open("/AILAB-summer-school-2025/success_data_raw/success_episode1_steps460/front_view/front_view_350.png").convert("RGB")

# 4. 전처리 적용 및 배치 구성
img1_tensor = transform(img1).unsqueeze(0)  # shape: (1, 3, 224, 224)
img2_tensor = transform(img2).unsqueeze(0)

#batch = torch.cat([img1_tensor, img2_tensor], dim=0)  # shape: (2, 3, 224, 224)

# 5. Feature 추출
with torch.no_grad():
    features1 = resnet18_feature_extractor(img1_tensor)    
    features2 = resnet18_feature_extractor(img2_tensor)    

# 6. 출력
#print("Feature shape:", features.shape)  # (2, 512)
# print("Feature for image 1:", features[0].numpy()[:10])  # 앞 10개 값만 예시
# print("Feature for image 2:", features[1].numpy()[:10])

# from sklearn.decomposition import PCA
# import numpy as np

# # 차원 축소 (squeeze or reshape)
# features1_np = features1.squeeze().numpy().reshape(1, -1)  # shape: (1, 512)
# features2_np = features2.squeeze().numpy().reshape(1, -1)

# # PCA 적용
# pca = PCA(n_components=2)
# compressed_features = pca.fit_transform(np.vstack([features1_np, features2_np]))

print("Compressed Feature 1:", features1)
print("Compressed Feature 2:", features2)