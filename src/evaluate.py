import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 1. Định nghĩa bộ đọc dữ liệu và tiền xử lý CLAHE giống lúc huấn luyện
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

def evaluate():
    device = torch.device("cpu")
    model_path = '/content/best_model_final.pth'
    data_dir = '/content/pneumonia_data'
    
    csv_path = os.path.join(data_dir, 'balanced_labels.csv')
    img_dir = os.path.join(data_dir, 'images')
    
    # Chia dữ liệu đúng tỷ lệ Validation lúc huấn luyện
    df = pd.read_csv(csv_path)
    _, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    val_dataset = PneumoniaDataset(val_df, img_dir, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
    
    # Khởi tạo mô hình ResNet18
    model = models.resnet18(weights=None)
    
    # SỬA LỖI TẠI ĐÂY: Khai báo lớp FC khớp hoàn toàn với file .pth của bạn
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1)
    )
    
    # Nạp trọng số mô hình
    print("⏳ Đang nạp trọng số mô hình...")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    threshold = checkpoint.get('threshold_f1', 0.455)
    
    all_probs = []
    all_labels = []
    
    print("⏳ AI đang quét tập dữ liệu đánh giá...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
            
            if probs.ndim == 0:
                probs = np.expand_dims(probs, axis=0)
                
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
            
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = (all_probs >= threshold).astype(int)
    
    # Tính toán các chỉ số
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # In kết quả báo cáo
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ ĐÁNH GIÁ TỪ FILE EVALUATE.PY (Threshold: {threshold:.3f})")
    print("="*50)
    print(f"  - Accuracy:    {acc*100:.2f}%")
    print(f"  - Precision:   {precision*100:.2f}%")
    print(f"  - Recall:      {recall*100:.2f}%")
    print(f"  - Specificity: {specificity*100:.2f}%")
    print(f"  - F1-Score:    {f1*100:.2f}%")
    print("-"*50)
    print("\n📝 CHI TIẾT CLASSIFICATION REPORT:")
    print(classification_report(all_labels, all_preds, target_names=['Normal', 'Pneumonia']))

if __name__ == '__main__':
    evaluate()
