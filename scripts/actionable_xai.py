# "Actionable use of explanations" 

import os
import random
from pathlib import Path
from collections import Counter

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models
from torchvision.transforms import functional as TF

from sklearn.metrics import classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------
# 0) Config
# --------------------------
SEED = 42
DATA_DIR = Path("data/processed")

BATCH_SIZE = 32
IMG_SIZE = 224
NUM_WORKERS = 0  
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4

SAVE_DIR = Path("models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PATH_BEFORE = str(SAVE_DIR / "resnet18_before.pt")
PATH_AFTER  = str(SAVE_DIR / "resnet18_after_v2_topcrop.pt")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)


# --------------------------
# 1) Transforms
# --------------------------


train_tfms_before = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.RandomAffine(degrees=6, translate=(0.02, 0.02), scale=(0.98, 1.02)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.07), ratio=(0.3, 3.3), value=0),
])


eval_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class TopBiasedCrop:
    """
    Random crop biased towards the top of the image, to preserve upper-body cues.
    This aims to reduce over-reliance on the lower garment region (e.g., hemline),
    which can cause Dresses→Skirts confusions.
    """
    def __init__(self, out_size: int = 224, min_scale: float = 0.8, max_scale: float = 1.0, top_bias: float = 0.75):
        self.out_size = out_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.top_bias = top_bias  # 0..1, higher = more top-biased

    def __call__(self, img):
        # img is PIL.Image
        w, h = img.size
        scale = random.uniform(self.min_scale, self.max_scale)
        ch = int(h * scale)
        cw = int(w * scale)

        y_max = max(0, h - ch)
        # biased sampling towards small y (top)
        y = int((random.random() ** (1 + 3 * self.top_bias)) * (y_max + 1))

        x_max = max(0, w - cw)
        x = random.randint(0, x_max) if x_max > 0 else 0

        return TF.resized_crop(img, top=y, left=x, height=ch, width=cw, size=[self.out_size, self.out_size])


train_tfms_after = transforms.Compose([
    transforms.Resize(256),
    TopBiasedCrop(out_size=IMG_SIZE, min_scale=0.8, max_scale=1.0, top_bias=0.75),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(0.15, 0.15, 0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
])


# --------------------------
# 2) Datasets + loaders
# --------------------------
train_ds_b = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms_before)
val_ds     = datasets.ImageFolder(DATA_DIR / "val", transform=eval_tfms)
test_ds    = datasets.ImageFolder(DATA_DIR / "test", transform=eval_tfms)

class_names = train_ds_b.classes
num_classes = len(class_names)
print("classes:", class_names)
print("class_to_idx:", train_ds_b.class_to_idx)

train_loader_b = DataLoader(train_ds_b, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader     = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader    = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

# AFTER uses same val/test transforms, only train changes
train_ds_a = datasets.ImageFolder(DATA_DIR / "train", transform=train_tfms_after)
train_loader_a = DataLoader(train_ds_a, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# class weights (computed on BEFORE train targets; labels are the same)
counts = Counter(train_ds_b.targets)
total = sum(counts.values())
weights = torch.tensor([total / counts[i] for i in range(num_classes)], dtype=torch.float32).to(device)


# --------------------------
# 3) Model + training utils
# --------------------------
def build_model() -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m.to(device)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += x.size(0)

    return total_loss / n, correct / n


def train_and_save(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, save_path: str) -> float:
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_val = 0.0
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer=optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {ep:02d} | train acc {tr_acc:.3f} | val acc {va_acc:.3f}")

    print("Best val acc:", best_val)
    return best_val


def evaluate(model: nn.Module, loader: DataLoader):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x).argmax(1).cpu().numpy()
            preds.extend(p)
            ys.extend(y.numpy())
    return np.array(ys), np.array(preds)


def dresses_to_skirts_count(ys: np.ndarray, preds: np.ndarray, class_to_idx: dict) -> int:
    if "Dresses" not in class_to_idx or "Skirts" not in class_to_idx:
        return -1
    dress = class_to_idx["Dresses"]
    skirt = class_to_idx["Skirts"]
    return int(((ys == dress) & (preds == skirt)).sum())


# --------------------------
# 4) BEFORE experiment
# --------------------------
print("\n=== TRAIN BEFORE ===")
model_before = build_model()
train_and_save(model_before, train_loader_b, val_loader, PATH_BEFORE)

model_before.load_state_dict(torch.load(PATH_BEFORE, map_location=device, weights_only=True))
ys_b, preds_b = evaluate(model_before, test_loader)

print("\n=== BEFORE report ===")
print(classification_report(ys_b, preds_b, target_names=class_names, digits=3))
print("Confusion matrix:\n", confusion_matrix(ys_b, preds_b))
print("Dresses→Skirts BEFORE:", dresses_to_skirts_count(ys_b, preds_b, train_ds_b.class_to_idx))


# --------------------------
# 5) AFTER experiment 
# --------------------------
print("\n=== TRAIN AFTER (TopBiasedCrop) ===")
model_after = build_model()
train_and_save(model_after, train_loader_a, val_loader, PATH_AFTER)

model_after.load_state_dict(torch.load(PATH_AFTER, map_location=device, weights_only=True))
ys_a, preds_a = evaluate(model_after, test_loader)

print("\n=== AFTER report ===")
print(classification_report(ys_a, preds_a, target_names=class_names, digits=3))
print("Confusion matrix:\n", confusion_matrix(ys_a, preds_a))
print("Dresses→Skirts AFTER:", dresses_to_skirts_count(ys_a, preds_a, train_ds_b.class_to_idx))


# --------------------------
# 6) Summary
# --------------------------
before_d2s = dresses_to_skirts_count(ys_b, preds_b, train_ds_b.class_to_idx)
after_d2s  = dresses_to_skirts_count(ys_a, preds_a, train_ds_b.class_to_idx)

print("\n=== SUMMARY ===")
print(f"Dresses→Skirts: {before_d2s} → {after_d2s}")

def save_confusion_matrices_side_by_side(cm_before, cm_after, class_names, out_path,
                                        title_left="BEFORE", title_right="AFTER"):
    """
    Saves a single figure with two confusion matrices side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(
        cm_before, ax=axes[0], annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, cbar=False
    )
    axes[0].set_title(title_left)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    sns.heatmap(
        cm_after, ax=axes[1], annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, cbar=False
    )
    axes[1].set_title(title_right)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


cm_before = confusion_matrix(ys_b, preds_b)
cm_after  = confusion_matrix(ys_a, preds_a)
out_fig = str(SAVE_DIR / "confusion_before_after.png")
save_confusion_matrices_side_by_side(
    cm_before, cm_after, class_names, out_fig,
    title_left="Confusion Matrix (BEFORE)",
    title_right="Confusion Matrix (AFTER)"
)
print("Saved:", out_fig)
