#!/usr/bin/env python3
"""
extract_annotations.py

把「紅筆圈選標註圖」自動轉成可量化的 ground-truth 座標。

輸入：annotated/ 下的標註圖
      —— 可以是螢幕截圖（含標題列、灰邊會自動裁掉）
      —— 也可以是原圖的「局部裁切」，會用多尺度樣板比對定位回原圖
輸出：annotated/reference_points.json  正規化座標（相對原圖）
      annotated/verify_<name>.png      疊在原圖上的驗證圖，務必目視確認

檔名對應規則（擇一）：
  1. 檔名含 IMG_xxxx           → real_color_translet/IMG_xxxx.jpg
  2. 檔名形如 <n>_<view>_test  → real_teeth/<view全名><n>_test.jpg
     view 縮寫：front / left / right / up / low

用法：
  python extract_annotations.py
  python extract_annotations.py --min-area 30
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
ORIG_DIRS = [HERE.parent / "real_color_translet", HERE.parent / "real_teeth"]

# 紅筆筆畫的顏色範圍（數位繪製的純紅，與照片裡的粉紅菌斑 S=47~73 分得開）
RED_S_MIN, RED_V_MIN, RED_H_LO, RED_H_HI = 130, 100, 8, 172

VIEW_ABBR = {"front": "front", "left": "left_side", "right": "right_side",
             "up": "upper_occlusal", "low": "lower_occlusal"}


# ==================================================================
# 對應原圖
# ==================================================================
def match_original(stem: str):
    m = re.search(r'(IMG_\d+)', stem, re.IGNORECASE)
    if m:
        cands = [f"{m.group(1)}{e}" for e in (".jpg", ".jpeg", ".png", ".JPG")]
    else:
        m = re.match(r'(\d+)_+([a-z]+)_test$', stem, re.IGNORECASE)
        if not m:
            return None
        n, abbr = m.group(1), m.group(2).lower()
        if abbr not in VIEW_ABBR:
            return None
        cands = [f"{VIEW_ABBR[abbr]}{n}_test.jpg"]
    for d in ORIG_DIRS:
        for c in cands:
            if (d / c).exists():
                return d / c
    return None


# ==================================================================
# 裁掉截圖的標題列 / 純色邊
# ==================================================================
def find_photo_region(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def longest_run(sd):
        thr = max(6.0, sd.max() * 0.15)
        active = np.append(sd > thr, False)
        best, best_len, start = (0, len(sd)), 0, None
        for i, a in enumerate(active):
            if a and start is None:
                start = i
            elif not a and start is not None:
                if i - start > best_len:
                    best_len, best = i - start, (start, i)
                start = None
        return best

    y0, y1 = longest_run(g.std(axis=1))
    x0, x1 = longest_run(g.std(axis=0))
    return x0, y0, x1 - x0, y1 - y0


def red_stroke_mask(img):
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    return (((h <= RED_H_LO) | (h >= RED_H_HI)) &
            (s >= RED_S_MIN) & (v >= RED_V_MIN)).astype(np.uint8) * 255


# ==================================================================
# 多尺度樣板比對：把裁切圖定位回原圖
# ==================================================================
def locate_in_original(crop_bgr, stroke_mask, orig_bgr, work=1200):
    """
    標註圖可能只是原圖的一塊裁切。先把紅筆畫 inpaint 掉還原底圖，
    再用正規化互相關在多個尺度下搜尋最佳位置。

    回傳 (x, y, w, h)：裁切區塊在原圖座標系中的矩形，以及比對分數。
    """
    clean = cv2.inpaint(crop_bgr, cv2.dilate(stroke_mask, np.ones((5, 5), np.uint8)),
                        3, cv2.INPAINT_TELEA)

    OH, OW = orig_bgr.shape[:2]
    k = work / max(OH, OW)
    ow, oh = int(OW * k), int(OH * k)
    o_small = cv2.cvtColor(cv2.resize(orig_bgr, (ow, oh)), cv2.COLOR_BGR2GRAY)
    c_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    ar = clean.shape[0] / clean.shape[1]          # 高/寬

    best = None
    # 由粗到細：先掃整體尺度，再在最佳值附近細掃
    for lo, hi, step in ((0.30, 1.00, 0.04), (None, None, 0.008)):
        if lo is None:
            lo, hi = max(0.25, best[1] - 0.05), min(1.02, best[1] + 0.05)
        f = lo
        while f <= hi + 1e-9:
            tw = int(ow * f)
            th = int(tw * ar)
            if tw < 24 or th < 24 or tw > ow or th > oh:
                f += step; continue
            tmpl = cv2.resize(c_gray, (tw, th))
            r = cv2.matchTemplate(o_small, tmpl, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(r)
            if best is None or mx > best[0]:
                best = (mx, f, loc, tw, th)
            f += step

    score, f, (bx, by), tw, th = best
    inv = 1.0 / k
    return (bx * inv, by * inv, tw * inv, th * inv), float(score)


# ==================================================================
# 主流程
# ==================================================================
def process(path: Path, min_area: int):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  ❌ 讀不到 {path.name}"); return None

    orig_path = match_original(path.stem)
    if orig_path is None:
        print(f"  ❌ {path.name} 找不到對應原圖"); return None
    orig = cv2.imread(str(orig_path))
    OH, OW = orig.shape[:2]

    # 1) 裁掉截圖邊框
    x, y, w, h = find_photo_region(img)
    crop = img[y:y + h, x:x + w]

    # 2) 抽紅筆筆畫
    mask = cv2.morphologyEx(red_stroke_mask(crop), cv2.MORPH_CLOSE,
                            np.ones((5, 5), np.uint8))

    # 3) 定位回原圖
    (rx, ry, rw, rh), score = locate_in_original(crop, mask, orig)
    sx, sy = rw / crop.shape[1], rh / crop.shape[0]
    flag = "" if score >= 0.55 else f"  ⚠️ 比對分數偏低 {score:.2f}，請務必檢查驗證圖"

    # 4) 筆畫 → 原圖正規化座標
    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    marks = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        bx, by = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # 圈選常是未閉合的弧線，bbox 中心比筆畫重心更接近想指的位置
        cx = rx + (bx + bw / 2) * sx
        cy = ry + (by + bh / 2) * sy
        marks.append({
            "bbox_center_norm": [round(cx / OW, 4), round(cy / OH, 4)],
            "bbox_norm": [round((rx + bx * sx) / OW, 4), round((ry + by * sy) / OH, 4),
                          round(bw * sx / OW, 4), round(bh * sy / OH, 4)],
            "radius_norm": round(max(bw * sx, bh * sy) / 2 / max(OW, OH), 4),
            "stroke_area_px": int(stats[i, cv2.CC_STAT_AREA]),
        })
    marks.sort(key=lambda m: (m["bbox_center_norm"][1], m["bbox_center_norm"][0]))

    # 5) 驗證圖
    vis = orig.copy()
    cv2.rectangle(vis, (int(rx), int(ry)), (int(rx + rw), int(ry + rh)), (255, 200, 0), 5)
    for k, m in enumerate(marks):
        cx, cy = int(m["bbox_center_norm"][0] * OW), int(m["bbox_center_norm"][1] * OH)
        r = max(30, int(m["radius_norm"] * max(OW, OH)))
        cv2.circle(vis, (cx, cy), r, (0, 255, 255), 6)
        cv2.putText(vis, str(k), (cx + r + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 2.0,
                    (0, 255, 255), 4)
    cv2.imwrite(str(ANNOT_DIR / f"verify_{orig_path.stem}.png"),
                cv2.resize(vis, (750, 1000)))

    print(f"  ✅ {path.name} → {orig_path.name}：{len(marks)} 個標註，"
          f"定位分數 {score:.2f}{flag}")
    return orig_path.stem, {
        "source": path.name,
        "orig_size": [OW, OH],
        "crop_rect_in_orig": [round(v, 1) for v in (rx, ry, rw, rh)],
        "match_score": round(score, 3),
        "marks": marks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-area", type=int, default=25)
    args = ap.parse_args()

    files = sorted(p for p in ANNOT_DIR.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                   and not p.name.startswith("verify_"))
    if not files:
        print(f"❌ {ANNOT_DIR} 沒有標註圖"); return 1

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
    print("   ⚠️ 請先看 annotated/verify_*.png 確認定位框與圈選位置都正確")
    return 0


if __name__ == "__main__":
    sys.exit(main())
