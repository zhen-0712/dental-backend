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
DEFAULT_ANNOT_DIR = HERE / "annotated"
DEFAULT_ORIG_DIRS = [HERE.parent / "real_color_translet", HERE.parent / "real_teeth"]

# 紅筆筆畫的偵測條件。
# 紫光照片裡皮膚是紫色，寬鬆條件就夠；但白光照片的嘴唇／皮膚本身就是
# 高飽和紅色（實測 S≈99、H≈6），寬鬆條件會誤判整片皮膚。
# 實測紅筆在 S≥220 時 BGR≈(26,27,215)，近乎純紅，因此改用
# 「高飽和 且 R 明顯大於 G/B」雙條件，不做連通域重建
#（皮膚是一整片巨大連通域，只要有一個種子碰到就會整片被保留）。
STROKE_S_MIN = 180        # 再低到 160 皮膚就開始溢入（連通域面積 1.4k → 9.6k）
STROKE_RG_DIFF = 90
RED_H_LO, RED_H_HI = 10, 170

VIEW_ABBR = {"front": "front", "left": "left_side", "right": "right_side",
             "up": "upper_occlusal", "low": "lower_occlusal"}


# ==================================================================
# 對應原圖
# ==================================================================
def match_original(stem: str, orig_dirs):
    m = re.search(r'(IMG_\d+)', stem, re.IGNORECASE)
    if m:
        cands = [f"{m.group(1)}{e}" for e in (".jpg", ".jpeg", ".png", ".JPG")]
    else:
        m = re.match(r'(\d+)_+([a-z]+)_([a-z]+\d*)$', stem, re.IGNORECASE)
        if not m:
            return None
        n, abbr, suffix = m.group(1), m.group(2).lower(), m.group(3).lower()
        if abbr not in VIEW_ABBR:
            return None
        cands = [f"{VIEW_ABBR[abbr]}{n}_{suffix}.jpg"]
    for d in orig_dirs:
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


def red_stroke_mask(img, s_min=None, rg_diff=None):
    """抽出紅筆筆畫，避免白光照片的嘴唇／皮膚被誤判。"""
    s_min = STROKE_S_MIN if s_min is None else s_min
    rg_diff = STROKE_RG_DIFF if rg_diff is None else rg_diff
    a = img.astype(np.int16)
    b, g, r = a[..., 0], a[..., 1], a[..., 2]
    h, sat, _ = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    m = (((h <= RED_H_LO) | (h >= RED_H_HI)) & (sat >= s_min) &
         ((r - np.maximum(g, b)) > rg_diff)).astype(np.uint8) * 255
    # 筆畫細且有抗鋸齒，先膨脹再閉合把斷點接起來
    m = cv2.dilate(m, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


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
def process(path: Path, min_area: int, annot_dir, orig_dirs, s_min, rg_diff):
    img = cv2.imread(str(path))
    if img is None:
        print(f"  ❌ 讀不到 {path.name}"); return None

    orig_path = match_original(path.stem, orig_dirs)
    if orig_path is None:
        print(f"  ❌ {path.name} 找不到對應原圖"); return None
    orig = cv2.imread(str(orig_path))
    OH, OW = orig.shape[:2]

    # 1) 裁掉截圖邊框
    x, y, w, h = find_photo_region(img)
    crop = img[y:y + h, x:x + w]

    # 2) 抽紅筆筆畫
    mask = red_stroke_mask(crop, s_min, rg_diff)

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
    cv2.imwrite(str(annot_dir / f"verify_{orig_path.stem}.png"),
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
    ap.add_argument("--annot-dir", help="標註圖目錄（預設 fluorescence_test/annotated）")
    ap.add_argument("--orig-dir", action="append",
                    help="原圖搜尋目錄，可重複指定")
    ap.add_argument("--stroke-s", type=int, default=STROKE_S_MIN,
                    help="筆畫飽和度下限。白光照片需 180 以上（皮膚是紅的）；"
                         "紫光照片皮膚是紫色，可放寬到 130")
    ap.add_argument("--stroke-rg", type=int, default=STROKE_RG_DIFF,
                    help="筆畫的 R-max(G,B) 下限，白光 90 / 紫光可用 50")
    args = ap.parse_args()

    ANNOT_DIR = Path(args.annot_dir) if args.annot_dir else DEFAULT_ANNOT_DIR
    ORIG_DIRS = ([Path(d) for d in args.orig_dir] if args.orig_dir
                 else DEFAULT_ORIG_DIRS)

    files = sorted(p for p in ANNOT_DIR.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                   and not p.name.startswith("verify_"))
    if not files:
        print(f"❌ {ANNOT_DIR} 沒有標註圖"); return 1

    print(f"📌 從 {len(files)} 張標註圖抽取 ground truth\n")
    out = {}
    for p in files:
        r = process(p, args.min_area, ANNOT_DIR, ORIG_DIRS, args.stroke_s, args.stroke_rg)
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
