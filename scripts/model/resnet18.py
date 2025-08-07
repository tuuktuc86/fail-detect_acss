import torch
from torchvision import models
import os

# 1. ResNet18 불러오기 (pretrained)
resnet18 = models.resnet18(pretrained=True)

# 2. 마지막 fc 레이어 제거
resnet18.fc = torch.nn.Identity()  # 출력: [B, 512]

# 3. 저장
save_path = "resnet18_512.pth"
torch.save(resnet18.state_dict(), save_path)
print(f"✅ Saved ResNet18 (fc removed) to {save_path}")
