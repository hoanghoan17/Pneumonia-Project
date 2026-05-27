import torch
import torch.nn as nn
from torchvision import models

def get_resnet18_model(device, freeze_mode='all'):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    if freeze_mode == 'all':
        for param in model.parameters():
            param.requires_grad = False
        print("[*] Freeze TOÀN BỘ backbone.")
    elif freeze_mode == 'selective':
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer3.parameters():
            param.requires_grad = True
        for param in model.layer4.parameters():
            param.requires_grad = True
        print("[*] Fine-tune: Layer 3 + Layer 4.")

    num_ftrs = model.fc.in_features
    # Thêm Dropout layer để giảm overfitting
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, 1)
    )
    return model.to(device)
