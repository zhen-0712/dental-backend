#!/usr/bin/env python3
"""
extract_annotations.py

把「紅筆圈選標註圖」自動轉成可量化的 ground-truth 座標。

輸入：annotated/ 下的標註圖（檔名需含原圖編號，如 IMG_5030_marked.png）
      —— 可以直接是螢幕截圖，含標題列與灰邊都沒關係，會自動裁掉
輸出：annotated/reference_points.json  （正規化座標 + 筆畫遮罩統計）
      annotated/verify_<name>.png      （疊在原圖上的驗證圖，務必目視確認）

為什麼要自動抽：目測讀 12 張圖的圈選位置誤差太大，而且之後每來一批
新照片都要重做。這支程式讓標註 → 量化評估變成一個指令的事。

用法：
  python extract_annotations.py
  python extract_annotations.py --min-area 30      # 調整筆畫最小面積
"""

import argparse
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ANNOT_DIR = HERE / "annotated"
ORIG_DIR = HERE.parent / "real_color_translet"

# 紅筆筆畫的顏色範圍（數位繪製的純紅，跟照片本身的紫/黃/粉都分得開）
RED_S_MIN = 130          # 飽和度下限：照片裡的粉紅菌斑只有 47~73，拉得開
RED_V_MIN = 100
RED_H_LO = 8             # H <= 8  或  H >= 172
RED_H_HI = 172


def find_photo_region(img):
    """
    截圖通常上方有標題列、周圍有純色邊。照片區域的像素變異遠大於這些純色區塊，
    用每列/每行的標準差找出最大的連續高變異區塊即為照片本體。
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    row_sd = g.std(axis=1)
    col_sd = g.std(axis=0)

    def longest_run(sd):
        thr = max(6.0, sd.max() * 0.15)
        active = sd > thr
        best = (0, len(sd))
        start = None
        best_len = 0
        for i, a in enumerate(np.append(active, False)):
            if a and start is None:
                start = i
            elif not a and start is not None:
                if i - start > best_len:
                    best_len, best = i - start, (start, i)
                start = None
        return best

    y0, y1 = longest_run(row_sd)
    x0, x1 = longest_run(col_sd)
    return x0, y0, x1 - x0, y1 - y0


def red_stroke_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    return (((h <= RED_H_LO) | (h >= RED_H_HI)) &
            (s >= RED_S_MIN) & (v >= RED_V_MIN)).astype(np.uint8) * 255


def match_original(stem):
    m = re.search(r'(IMG_\d+)', stem, re.IGNORECASE)
    if not m:
        return None
    for ext in (".jpg", ".jpeg", ".png", ".JPG"):
        p = ORIG_DIR / f"{m.group(1)}{ext}"
        if p.exists():
            return p
    return None


def process(path: Path, min_area: int):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  ❌ 讀不到 {path.name}")
        return None

    orig_path = match_original(path.stem)
    if orig_path is None:
        print(f"  ❌ {path.name} 找不到對應原圖（檔名需含 IMG_xxxx）")
        return None
    orig = cv2.imread(str(orig_path))
    OH, OW = orig.shape[:2]

    # 1) 裁掉截圖的標題列與邊框
    x, y, w, h = find_photo_region(img)
    crop = img[y:y + h, x:x + w]

    ar_crop, ar_orig = w / h, OW / OH
    warn = ""
    if abs(ar_crop - ar_orig) / ar_orig > 0.04:
        warn = f"  ⚠️ 長寬比不符（截圖 {ar_crop:.3f} vs 原圖 {ar_orig:.3f}），裁切可能不準"

    # 2) 抽紅筆筆畫
    mask = red_stroke_mask(crop)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    marks = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = cents[i]
        bx, by = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # 圈選常是沒閉合的弧線，用 bbox 中心比筆畫重心更接近使用者想指的位置
        marks.append({
            "centroid_norm": [round(cx / w, 4), round(cy / h, 4)],
            "bbox_center_norm": [round((bx + bw / 2) / w, 4),
                                 round((by + bh / 2) / h, 4)],
            "bbox_norm": [round(bx / w, 4), round(by / h, 4),
                          round(bw / w, 4), round(bh / h, 4)],
            "stroke_area_px": int(stats[i, cv2.CC_STAT_AREA]),
            "radius_norm": round(max(bw, bh) / 2 / max(w, h), 4),
        })
    marks.sort(key=lambda m: (m["bbox_center_norm"][1], m["bbox_center_norm"][0]))

    # 3) 驗證圖：把抽出來的點畫回原圖
    vis = orig.copy()
    for k, m in enumerate(marks):
        cx = int(m["bbox_center_norm"][0] * OW)
        cy = int(m["bbox_center_norm"][1] * OH)
        r = max(24, int(m["radius_norm"] * max(OW, OH)))
        cv2.circle(vis, (cx, cy), r, (0, 255, 255), 6)
        cv2.putText(vis, str(k), (cx + r + 6, cy), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 255, 255), 4)
    vis = cv2.resize(vis, (750, 1000))
    cv2.imwrite(str(ANNOT_DIR / f"verify_{orig_path.stem}.png"), vis)

    print(f"  ✅ {path.name} → {orig_path.name}：裁切 {w}×{h}，抽到 {len(marks)} 個標註{warn}")
    return orig_path.stem, {
        "source": path.name,
        "orig_size": [OW, OH],
        "crop_in_screenshot": [int(x), int(y), int(w), int(h)],
        "marks": marks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area", type=int, default=25, help="筆畫最小像素面積")
    args = ap.parse_args()

    if not ANNOT_DIR.exists():
        print(f"❌ 找不到 {ANNOT_DIR}")
        return 1
    files = sorted(p for p in ANNOT_DIR.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                   and not p.name.startswith("verify_"))
    if not files:
        print(f"❌ {ANNOT_DIR} 是空的，請把紅圈標註圖放進去（檔名含 IMG_xxxx）")
        return 1

    print(f"📌 從 {len(files)} 張標註圖抽取 ground truth\n")
    out = {}
    for p in files:
        r = process(p, args.min_area)
        if r:
            out[r[0]] = r[1]

    with open(ANNOT_DIR / "reference_points.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    total = sum(len(v["marks"]) for v in out.values())
    print(f"\n✅ {len(out)} 張、共 {total} 個標註 → annotated/reference_points.json")
    print(f"   ⚠️ 請先看 annotated/verify_*.png 確認位置抽對了再往下做")
    return 0


if __name__ == "__main__":
    sys.exit(main())
