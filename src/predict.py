"""
Dự đoán ảnh X-quang đơn lẻ (dùng cho demo).
- Tiền xử lý CLAHE
- Test-Time Augmentation (TTA): trung bình prediction của ảnh gốc + ảnh lật ngang
- Sử dụng threshold tối ưu đã lưu trong checkpoint
"""
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from model import get_resnet18_model
from dataset import apply_clahe

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict(image_path, ckpt_path='best_model_final.pth', mode='f1'):
    """Dự đoán 1 ảnh.

    Args:
        image_path: đường dẫn ảnh X-quang
        ckpt_path: checkpoint đã train
        mode: 'f1' (cân bằng) hoặc 'medical' (Recall cao)
    """
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    if mode == 'medical' and ckpt.get('threshold_medical'):
        threshold = ckpt['threshold_medical']
    else:
        threshold = ckpt.get('threshold_f1', 0.455)

    # Load model
    model = get_resnet18_model(DEVICE, freeze_mode='selective')
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    # Tiền xử lý
    img = Image.open(image_path).convert('RGB')
    img_clahe = apply_clahe(img)

    # Test-Time Augmentation
    probs = []
    with torch.no_grad():
        for im in [img_clahe, img_clahe.transpose(Image.FLIP_LEFT_RIGHT)]:
            x = transform(im).unsqueeze(0).to(DEVICE)
            probs.append(torch.sigmoid(model(x)).item())

    prob = float(np.mean(probs))
    label = "PNEUMONIA" if prob >= threshold else "NORMAL"
    confidence = prob if prob >= threshold else (1 - prob)

    print(f"\n{'='*50}")
    print(f"📷 File: {image_path}")
    print(f"{'='*50}")
    print(f"  Probability (Pneumonia): {prob:.4f} ({prob*100:.2f}%)")
    print(f"  Threshold ({mode}):      {threshold}")
    print(f"  Confidence:              {confidence*100:.2f}%")
    print(f"  → PREDICTION: {label}")
    print(f"{'='*50}\n")

    return {'probability': prob, 'label': label,
            'confidence': confidence, 'threshold': threshold}


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [mode]")
        print("  mode: 'f1' (default) or 'medical'")
        sys.exit(1)
    image_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else 'f1'
    predict(image_path, mode=mode)
