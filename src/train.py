"""
Training pipeline 3 giai đoạn:
- Stage 1: Freeze backbone, train classifier head (lr=1e-3)
- Stage 2: Fine-tune Layer 3+4 (lr=1e-5)
- Stage 3: Threshold tuning trên validation set
"""
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve
)
from dataset import prepare_dataloaders
from model import get_resnet18_model


class EarlyStopping:
    """Early stopping dựa trên validation loss."""
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
    """Chạy 1 epoch (train hoặc validate) và trả về dict các metrics."""
    if phase == 'train':
        model.train()
        # Giữ BatchNorm ở eval khi backbone bị freeze
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

    return {
        'loss': running_loss / total,
        'acc': (preds_arr == labels_arr).mean(),
        'auc': roc_auc_score(labels_arr, probs_arr),
        'precision': precision_score(labels_arr, preds_arr, zero_division=0),
        'recall': recall_score(labels_arr, preds_arr, zero_division=0),
        'f1': f1_score(labels_arr, preds_arr, zero_division=0),
        'cm': confusion_matrix(labels_arr, preds_arr) if phase == 'val' else None,
    }


def get_val_predictions(model, val_loader, device):
    """Chạy inference trên val set, trả về (probs, labels) full set."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            p = torch.sigmoid(model(x)).cpu().numpy().flatten()
            all_probs.extend(p)
            all_labels.extend(y.numpy().flatten())
    return np.array(all_probs), np.array(all_labels)


def tune_threshold(all_probs, all_labels, target='f1'):
    """Tìm threshold tối ưu trên val set.

    Args:
        target: 'f1' (cân bằng) hoặc 'medical' (Recall >= 0.90)
    """
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

    if target == 'f1':
        f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        best_idx = f1s[:-1].argmax()
        return (float(thresholds[best_idx]), float(f1s[best_idx]),
                float(precisions[best_idx]), float(recalls[best_idx]))

    elif target == 'medical':
        mask = recalls[:-1] >= 0.90
        if mask.any():
            valid_p = precisions[:-1][mask]
            valid_t = thresholds[mask]
            valid_r = recalls[:-1][mask]
            best_idx = valid_p.argmax()
            return (float(valid_t[best_idx]), None,
                    float(valid_p[best_idx]), float(valid_r[best_idx]))
        return None


def train_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = prepare_dataloaders('/content/pneumonia_data', 32)

    criterion = nn.BCEWithLogitsLoss()

    history = {k: [] for k in ['t_loss', 'v_loss', 't_acc', 'v_acc',
                               't_auc', 'v_auc', 'v_recall', 'v_f1']}
    best_cm = None

    # ===== STAGE 1: FREEZE BACKBONE =====
    print("\n[STAGE 1] Freeze Backbone with Regularization")
    model = get_resnet18_model(DEVICE, freeze_mode='all')
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopper = EarlyStopping(patience=5, path='best_model_v1.pth')

    for epoch in range(8):
        t = run_epoch(model, train_loader, criterion, optimizer, DEVICE, 'train')
        v = run_epoch(model, val_loader, criterion, None, DEVICE, 'val')
        scheduler.step(v['loss'])
        for k, val in zip(history.keys(),
                          [t['loss'], v['loss'], t['acc'], v['acc'],
                           t['auc'], v['auc'], v['recall'], v['f1']]):
            history[k].append(val)
        print(f"E{epoch+1}/8 | Val Acc: {v['acc']:.4f} | Recall: {v['recall']:.4f} | "
              f"F1: {v['f1']:.4f} | AUC: {v['auc']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        early_stopper(v['loss'], model)
        if early_stopper.early_stop:
            print("[!] Early stopping at Stage 1")
            break

    # ===== STAGE 2: FINE-TUNE =====
    print("\n[STAGE 2] Fine-tuning with Regularization")
    model.load_state_dict(torch.load('best_model_v1.pth'))
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)
    early_stopper = EarlyStopping(patience=5, path='best_model_v1.pth')

    for epoch in range(12):
        t = run_epoch(model, train_loader, criterion, optimizer, DEVICE, 'train')
        v = run_epoch(model, val_loader, criterion, None, DEVICE, 'val')
        scheduler.step(v['loss'])
        for k, val in zip(history.keys(),
                          [t['loss'], v['loss'], t['acc'], v['acc'],
                           t['auc'], v['auc'], v['recall'], v['f1']]):
            history[k].append(val)
        print(f"E{len(history['t_loss'])}/20 | Val Acc: {v['acc']:.4f} | "
              f"Recall: {v['recall']:.4f} | F1: {v['f1']:.4f} | AUC: {v['auc']:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        if v['loss'] <= min(history['v_loss']):
            best_cm = v['cm']
        early_stopper(v['loss'], model)
        if early_stopper.early_stop:
            print("[!] Early stopping at Stage 2")
            break

    # ===== STAGE 3: TUNE THRESHOLD =====
    print("\n[STAGE 3] Tuning decision threshold")
    model.load_state_dict(torch.load('best_model_v1.pth'))
    all_probs, all_labels = get_val_predictions(model, val_loader, DEVICE)

    thr_f1, f1_score_val, prec_f1, rec_f1 = tune_threshold(all_probs, all_labels, target='f1')
    print(f"[F1-Optimal] Threshold: {thr_f1:.4f} | F1: {f1_score_val:.4f} | "
          f"P: {prec_f1:.4f} | R: {rec_f1:.4f}")

    medical_result = tune_threshold(all_probs, all_labels, target='medical')
    if medical_result:
        thr_med, _, prec_med, rec_med = medical_result
        print(f"[Medical] Threshold: {thr_med:.4f} | P: {prec_med:.4f} | "
              f"R: {rec_med:.4f} (≥90%)")
    else:
        thr_med = None
        print("[Medical] Không tìm được threshold cho Recall ≥ 90%")

    # Lưu checkpoint với cả 2 threshold
    torch.save({
        'state_dict': model.state_dict(),
        'threshold_f1': thr_f1,
        'threshold_medical': thr_med,
        'val_f1': f1_score_val,
        'val_precision_f1': prec_f1,
        'val_recall_f1': rec_f1,
    }, 'best_model_final.pth')
    print(f"[*] Saved best_model_final.pth")

    # ===== VẼ LEARNING CURVES =====
    plt.figure(figsize=(18, 4))
    for i, (m, t_k, v_k) in enumerate([
        ('Loss', 't_loss', 'v_loss'),
        ('Recall & F1', 'v_recall', 'v_f1'),
        ('AUC', 't_auc', 'v_auc'),
    ]):
        plt.subplot(1, 3, i + 1)
        if t_k in history:
            plt.plot(history[t_k], label='Train')
        plt.plot(history[v_k], label='Val')
        plt.title(m); plt.legend()
    plt.tight_layout()
    plt.savefig('learning_curves_full.png')
    plt.show()

    # ===== CONFUSION MATRICES =====
    fig, axes = plt.subplots(1, 2 if not thr_med else 3, figsize=(6 * (3 if thr_med else 2), 5))
    if best_cm is not None:
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title(f'Threshold = 0.5')
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')

    preds_f1 = (all_probs >= thr_f1).astype(int)
    cm_f1 = confusion_matrix(all_labels, preds_f1)
    sns.heatmap(cm_f1, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'F1-Optimal (Thr = {thr_f1:.3f})')
    axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

    if thr_med:
        preds_med = (all_probs >= thr_med).astype(int)
        cm_med = confusion_matrix(all_labels, preds_med)
        sns.heatmap(cm_med, annot=True, fmt='d', cmap='Reds', ax=axes[2])
        axes[2].set_title(f'Medical (Thr = {thr_med:.3f})')
        axes[2].set_xlabel('Predicted'); axes[2].set_ylabel('Actual')

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.show()


if __name__ == '__main__':
    train_model()
