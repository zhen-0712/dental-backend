#!/usr/bin/env python3
"""
牙齒視角 CNN 分類器訓練腳本
模型：MobileNetV3-Small（輕量，CPU 推理 < 50ms）
訓練資料：
  前置鏡頭：view_clf_data/train/  + view_clf_data/val/
  後置鏡頭：view_clf_data/rear_camera/train/  + view_clf_data/rear_camera/val/
  （兩者合併訓練，後置照片套強制水平翻轉以對齊 server 收到的方向）

用法：
  python train_view_clf.py               # 訓練
  python train_view_clf.py --test img.jpg # 測試單張照片
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import datasets, models
from torch.utils.data import DataLoader, ConcatDataset

# ── 設定 ────────────────────────────────────────────────────────────────────
DATA_DIR     = Path(__file__).parent / "view_clf_data"
REAR_CAM_DIR = Path(__file__).parent / "view_clf_data" / "rear_camera"
SAVE_PATH    = Path(__file__).parent / "weight" / "view_clf.pt"
IMG_SIZE     = 224
BATCH_SIZE   = 16
EPOCHS       = 30
LR           = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── 資料增強 ─────────────────────────────────────────────────────────────────
# 前置鏡頭：不做水平翻轉（app 已標準化方向）
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

# 後置鏡頭：原始照片未經 app 翻轉，強制水平翻轉對齊 server 收到的方向
rear_train_tf = T.Compose([
    T.RandomHorizontalFlip(p=1.0),
    T.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    T.RandomCrop(IMG_SIZE),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

rear_val_tf = T.Compose([
    T.RandomHorizontalFlip(p=1.0),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model(num_classes: int):
    """MobileNetV3-Small，只解凍最後分類器層（快速 fine-tune）。"""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    for p in model.features.parameters():
        p.requires_grad = False
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def train():
    print(f"[train] device={DEVICE}")

    # ── 前置鏡頭資料集 ──────────────────────────────────────────────────────
    front_train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    front_val_ds   = datasets.ImageFolder(DATA_DIR / "val",   transform=val_tf)
    classes = front_train_ds.classes

    # ── 後置鏡頭資料集（如果資料夾存在且有內容）──────────────────────────────
    rear_train_ds = rear_val_ds = None
    rear_train_dir = REAR_CAM_DIR / "train"
    rear_val_dir   = REAR_CAM_DIR / "val"

    if rear_train_dir.exists():
        rear_train_ds = datasets.ImageFolder(rear_train_dir, transform=rear_train_tf)
        rear_val_ds   = datasets.ImageFolder(rear_val_dir,   transform=rear_val_tf)
        assert rear_train_ds.classes == classes, \
            f"後置鏡頭類別與前置不一致: {rear_train_ds.classes} vs {classes}"

    # ── 合併資料集 ────────────────────────────────────────────────────────────
    if rear_train_ds:
        train_ds = ConcatDataset([front_train_ds, rear_train_ds])
        val_ds   = ConcatDataset([front_val_ds,   rear_val_ds])
        print(f"[train] 前置 train={len(front_train_ds)} val={len(front_val_ds)}")
        print(f"[train] 後置 train={len(rear_train_ds)} val={len(rear_val_ds)}")
    else:
        train_ds = front_train_ds
        val_ds   = front_val_ds
        print(f"[train] 僅前置鏡頭資料")

    print(f"[train] 合計 train={len(train_ds)} val={len(val_ds)} classes={classes}")

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model     = build_model(len(classes)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
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
                "classes":    classes,
                "img_size":   IMG_SIZE,
            }, SAVE_PATH)
            print(f"  → saved (best val_acc={best_val_acc:.3f})")

    print(f"\n[done] best val_acc={best_val_acc:.3f}  model={SAVE_PATH}")


def test_image(img_path: str):
    """測試單張照片的預測視角。"""
    from PIL import Image
    ckpt  = torch.load(SAVE_PATH, map_location="cpu")
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
