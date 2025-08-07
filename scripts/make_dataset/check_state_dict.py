import torch
from torchvision import models

# 1. 모델 정의 및 weight 로드
resnet1 = models.resnet18(pretrained = True)
resnet1.fc = torch.nn.Identity()

resnet1.eval()

# 2. 동일한 모델을 새로 만들고 다시 weight 로드
resnet2 = models.resnet18()
resnet2.fc = torch.nn.Identity()
resnet2.load_state_dict(torch.load("/AILAB-summer-school-2025/scripts/model/resnet18_512.pth"))
resnet2.eval()

# 3. state_dict 직접 비교
def compare_state_dicts(dict1, dict2):
    for k in dict1:
        if not torch.equal(dict1[k], dict2[k]):
            print(f"❌ Mismatch at key: {k}")
            return False
    return True

same = compare_state_dicts(resnet1.state_dict(), resnet2.state_dict())
print("✅ Weights are exactly the same:" if same else "❌ Weights are different.")
