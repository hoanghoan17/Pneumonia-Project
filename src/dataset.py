"""
Dataset loader cho Pneumonia Classification.
- Tiền xử lý: CLAHE (cân bằng histogram thích ứng)
- Augmentation mạnh để chống overfitting và domain shift
- WeightedRandomSampler để cân bằng class trong batch
"""
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Su dung CLAHE
def apply_clahe(pil_img):
    img = np.array(pil_img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    return Image.fromarray(img).convert('RGB')


class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Image Index']
        img_path = os.path.join(self.img_dir, img_name)
        label = self.dataframe.iloc[idx]['Label']
        image = Image.open(img_path)
        image = apply_clahe(image)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)


def prepare_dataloaders(data_dir, batch_size=32):
    """Chuẩn bị train/val DataLoaders với class balancing."""
    csv_path = os.path.join(data_dir, 'balanced_labels.csv')
    img_dir = os.path.join(data_dir, 'images')
    df = pd.read_csv(csv_path)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
            transforms.ColorJitter(brightness=0.2, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    }

    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Label']
    )
    train_dataset = PneumoniaDataset(train_df, img_dir, transform=data_transforms['train'])
    val_dataset = PneumoniaDataset(val_df, img_dir, transform=data_transforms['val'])

    # WeightedRandomSampler để cân bằng 1:1 trong mỗi batch
    labels = train_df['Label'].values
    class_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_count
    samples_weight = torch.from_numpy(np.array([weight[int(t)] for t in labels]))
    sampler = WeightedRandomSampler(samples_weight.double(), len(samples_weight))

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )
    return train_loader, val_loader
