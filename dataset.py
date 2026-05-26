import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import numpy as np

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
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.float32)

def prepare_dataloaders(data_dir, batch_size=32):
    csv_path = os.path.join(data_dir, 'balanced_labels.csv')
    img_dir = os.path.join(data_dir, 'images')

    df = pd.read_csv(csv_path)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
    train_dataset = PneumoniaDataset(train_df, img_dir, transform=data_transforms['train'])
    val_dataset = PneumoniaDataset(val_df, img_dir, transform=data_transforms['val'])

    # --- TÍNH TOÁN WEIGHTS CHO RESAMPLING ---
    labels = train_df['Label'].values
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[int(t)] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)

    sampler = WeightedRandomSampler(samples_weight.double(), len(samples_weight))

    # Áp dụng sampler cho train_loader để đạt tỷ lệ 1:1 trong mini-batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2, pin_memory=True)
    # Val loader không cần sampler để phản ánh đúng phân phối thực tế
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
