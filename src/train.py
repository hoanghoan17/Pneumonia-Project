import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset import prepare_dataloaders
from model import get_resnet18_model

class EarlyStopping:
    def __init__(self, patience=5, path='best_model_v1.pth'):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_loss:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def run_epoch(model, loader, criterion, optimizer, device, phase='train'):
    if phase == 'train':
        model.train()
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                if not next(m.parameters()).requires_grad:
                    m.eval()
    else:
        model.eval()

    running_loss, total = 0.0, 0
    probs_list, labels_list = [], []

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        if phase == 'train':
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)
        probs_list.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        labels_list.extend(labels.detach().cpu().numpy())

    probs_arr = np.array(probs_list)
    labels_arr = np.array(labels_list)
    preds_arr = (probs_arr >= 0.5).astype(int)

    metrics = {
        'loss': running_loss / total,
        'acc': (preds_arr == labels_arr).mean(),
        'auc': roc_auc_score(labels_arr, probs_arr),
        'precision': precision_score(labels_arr, preds_arr, zero_division=0),
        'recall': recall_score(labels_arr, preds_arr, zero_division=0),
        'f1': f1_score(labels_arr, preds_arr, zero_division=0),
        'cm': confusion_matrix(labels_arr, preds_arr) if phase == 'val' else None
    }
    return metrics

def train_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_dataloaders('/content/pneumonia_data', 32)

    pos_weight = torch.tensor([3.0]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history = {k: [] for k in ['t_loss', 'v_loss', 't_acc', 'v_acc', 't_auc', 'v_auc', 'v_recall', 'v_f1']}
    best_cm = None

    # --- STAGE 1: FREEZE ---
    print("\n[STAGE 1] Freeze Backbone with Regularization")
    model = get_resnet18_model(DEVICE, freeze_mode='all')
    # Thêm weight_decay=1e-4 để giảm overfitting
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopper = EarlyStopping(patience=5, path='best_model_v1.pth')

    for epoch in range(8):
        t = run_epoch(model, train_loader, criterion, optimizer, DEVICE, 'train')
        v = run_epoch(model, val_loader, criterion, None, DEVICE, 'val')

        scheduler.step(v['loss'])
        for k, val in zip(history.keys(), [t['loss'], v['loss'], t['acc'], v['acc'], t['auc'], v['auc'], v['recall'], v['f1']]):
            history[k].append(val)

        print(f"E{epoch+1}/8 | Val Acc: {v['acc']:.4f} | Recall: {v['recall']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        early_stopper(v['loss'], model)
        if early_stopper.early_stop: break

    # --- STAGE 2: FINE-TUNE ---
    print("\n[STAGE 2] Fine-tuning with Regularization")
    model.load_state_dict(torch.load('best_model_v1.pth'))
    for param in model.parameters(): param.requires_grad = True
    # Thêm weight_decay=1e-4 cho Stage 2
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopper = EarlyStopping(patience=5, path='best_model_v1.pth')

    for epoch in range(12):
        t = run_epoch(model, train_loader, criterion, optimizer, DEVICE, 'train')
        v = run_epoch(model, val_loader, criterion, None, DEVICE, 'val')

        scheduler.step(v['loss'])
        for k, val in zip(history.keys(), [t['loss'], v['loss'], t['acc'], v['acc'], t['auc'], v['auc'], v['recall'], v['f1']]):
            history[k].append(val)

        print(f"E{len(history['t_loss'])}/20 | Val Acc: {v['acc']:.4f} | Recall: {v['recall']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        if v['loss'] <= min(history['v_loss']): best_cm = v['cm']
        early_stopper(v['loss'], model)
        if early_stopper.early_stop: break

    plt.figure(figsize=(18, 4))
    for i, (m, t_k, v_k) in enumerate([('Loss','t_loss','v_loss'),('Recall & F1','v_recall','v_f1'),('AUC','t_auc','v_auc')]):
        plt.subplot(1, 3, i+1)
        if t_k in history: plt.plot(history[t_k], label='Train')
        plt.plot(history[v_k], label='Val')
        plt.title(m); plt.legend()
    plt.tight_layout(); plt.savefig('learning_curves_full.png'); plt.show()

    if best_cm is not None:
        plt.figure(figsize=(5, 4))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Best Model)'); plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png'); plt.show()

if __name__ == '__main__': train_model()
