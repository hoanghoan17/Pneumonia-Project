# Chest X-Ray Pneumonia Diagnosis using ResNet18 (PyTorch)

An advanced Deep Learning pipeline developed by **Group 5** for detecting Pneumonia from chest X-ray images, built on top of the PyTorch framework and pretrained ResNet18 architecture.

## 🚀 Project Highlights
* **Architecture:** ResNet18 (Transfer Learning + Selective Fine-tuning)
* **Modular Codebase:** Clean separation into `dataset.py`, `model.py`, and `train.py` for professional workflow.
* **Class Imbalance Resolution:** Successfully optimized the clinical safety of the model by tackling dataset imbalance, driving **Recall up to ~80.4%** and reducing dangerous False Negatives by over 66%.

## 📂 Repository Structure
* `dataset.py`: Handles data augmentation, normalization, and balanced DataLoader pipelines.
* `model.py`: Defines the ResNet18 backbone, freezing strategies, and custom binary classification head.
* `train.py`: Controls the 2-stage training loop (Freeze & Unfreeze) with Early Stopping.
* `evaluate.py`: Computes medical imaging evaluation metrics (Accuracy, Recall, F1-score, Confusion Matrix).
* `final.ipynb`: Main Google Colab execution notebook.

## 📊 Experimental Results

### Baseline Model (Imbalanced)
* **Issue:** High overall accuracy but failed dramatically on clinical safety by missing more than half of the pneumonia cases (Recall ~41%).

### Optimized Model (Class-Balanced)
* **Accuracy:** 62.50%
* **F1-Score:** 66.67%
* **Recall (Sensitivity):** **~80.41%** (Crucial for reducing False Negatives in medical diagnosis)

---
*Developed as part of the academic curriculum at Ho Chi Minh City University of Industry and Trade (HUIT).*
