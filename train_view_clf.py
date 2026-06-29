#!/usr/bin/env python3
"""
牙齒視角 CNN 分類器訓練腳本
模型：MobileNetV3-Small（輕量，CPU 推理 < 50ms）
訓練資料：view_clf_data/train/{front,left_side,right_side,upper_occlusal,lower_occlusal}/

用法：
  python train_view_clf.py               # 訓練
  python train_view_clf.py --test img.jpg # 測試單張照片
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets, models
from torch.utils.data import DataLoader

# ── 設定 ────────────────────────────────────────────────────────────────────
DATA_DIR   = Path(__file__).parent / "view_clf_data"
SAVE_PATH  = Path(__file__).parent / "weight" / "view_clf.pt"
IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 30
LR         = 3e-4
CLASSES    = ["front", "left_side", "right_side", "upper_occlusal", "lower_occlusal"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 資料增強 ─────────────────────────────────────────────────────────────────
# 注意：左右側不做水平翻轉（前後置相機 flip 由 app 層處理，server 收到的已是一致方向）
train_tf = T.Compose([
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomCrop(IMG_SIZE),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int):
    """MobileNetV3-Small，只解凍最後分類器層（快速 fine-tune）。"""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # 凍結 backbone
    for p in model.features.parameters():
        p.requires_grad = False
    # 換掉分類頭
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train():
    print(f"[train] device={DEVICE}")
    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=val_tf)
    print(f"[train] train={len(train_ds)} val={len(val_ds)} classes={train_ds.classes}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = build_model(len(train_ds.classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = train_correct = 0
        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_correct = 0
        with torch.no_grad():
            for imgs, labels in val_dl:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                val_correct += (model(imgs).argmax(1) == labels).sum().item()

        train_acc = train_correct / len(train_ds)
        val_acc   = val_correct   / len(val_ds)
        print(f"Epoch {epoch:02d}/{EPOCHS}  "
              f"loss={train_loss/len(train_ds):.4f}  "
              f"train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "state_dict": model.state_dict(),
                "classes":    train_ds.classes,
                "img_size":   IMG_SIZE,
            }, SAVE_PATH)
            print(f"  → saved (best val_acc={best_val_acc:.3f})")

    print(f"\n[done] best val_acc={best_val_acc:.3f}  model={SAVE_PATH}")


def test_image(img_path: str):
    """測試單張照片的預測視角。"""
    from PIL import Image
    ckpt = torch.load(SAVE_PATH, map_location="cpu")
    model = build_model(len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    img = Image.open(img_path).convert("RGB")
    x   = val_tf(img).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    for cls, p in sorted(zip(ckpt["classes"], probs.tolist()), key=lambda t: -t[1]):
        bar = "█" * int(p * 30)
        print(f"  {cls:<20} {p:.3f}  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", metavar="IMAGE", help="測試單張照片")
    args = parser.parse_args()
    if args.test:
        test_image(args.test)
    else:
        train()
