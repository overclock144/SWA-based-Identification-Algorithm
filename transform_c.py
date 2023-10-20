import torchvision.transforms
from torchvision.transforms import transforms

train_transform = torchvision.transforms.Compose([
    # 600*400 → 256*256
    transforms.Resize((256, 256)),
    # 随机裁剪 → 224*224
    transforms.RandomCrop(224),
    # p=0.5水平翻转
    transforms.RandomHorizontalFlip(),
    # (H, W, C) → (C, H, W) 且归一化
    transforms.ToTensor(),
    # 将数据转换为标准正太分布,使模型更容易收敛
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transform = torchvision.transforms.Compose([
    # 600*400 → 224*224
    transforms.Resize((224, 224)),
    # (H, W, C) → (C, H, W) 且归一化
    transforms.ToTensor(),
    # 将数据转换为标准正太分布,使模型更容易收敛
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
