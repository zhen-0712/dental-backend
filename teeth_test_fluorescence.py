#!/usr/bin/env python3
"""
teeth_test_fluorescence.py

紫光燈（405nm 藍紫光螢光）模式的牙菌斑偵測 —— teeth_test.py 的平行替代品。

★ 與 teeth_test.py 完全獨立，兩者互不 import、互不影響。
  由 api_server 依 light_mode 決定跑哪一支：
      light_mode="dye"          → teeth_test.py           （染色劑，原有流程）
      light_mode="fluorescence" → teeth_test_fluorescence.py（本檔，紫光燈）

輸出契約與 teeth_test.py 完全相同，所以下游
（extract_plaque_regions.py → project_plaque_by_fdi.py）一行都不用改：

    teeth_color_test/mask_<view>.jpg   二值菌斑遮罩（原圖解析度）
    teeth_color_test/det_<view>.jpg    輪廓標註圖（人工檢查用）

偵測演算法在 fluorescence_test/fluorescence_core.py，與離線調參工具
plaque_detect_v2.py 共用同一份程式碼，避免兩邊參數漂移。

可用環境變數微調（不設就用 core 的預設）：
    DENTAL_FLUOR_Z          z-score 門檻，越低越靈敏（預設 2.0）
    DENTAL_FLUOR_FP_REL     fp 相對下限倍數（預設 1.5）
    DENTAL_FLUOR_DEBUG      設為 1 時額外輸出 fp/z 熱圖
"""

import os
import sys
import json

import cv2
import numpy as np

sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")
sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth/fluorescence_test")

from user_env import get_paths, setup_user_dirs
import fluorescence_core as fc

_PATHS = get_paths()
setup_user_dirs(_PATHS["user_dir"])

INPUT_DIR = _PATHS["real_teeth"]
OUTPUT_DIR = _PATHS["teeth_color_test"]

# 與 teeth_test.py 相同的五視角
PHOTOS = [
    'front.jpg', 'left_side.jpg', 'right_side.jpg',
    'upper_occlusal.jpg', 'lower_occlusal.jpg',
]


def _cfg_from_env():
    cfg = {}
    if os.environ.get("DENTAL_FLUOR_Z"):
        cfg["z_thresh"] = float(os.environ["DENTAL_FLUOR_Z"])
    if os.environ.get("DENTAL_FLUOR_FP_REL"):
        cfg["fp_rel_min"] = float(os.environ["DENTAL_FLUOR_FP_REL"])
    return cfg


def detect_plaque(image_path, cfg, debug=False):
    name = os.path.basename(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ❌ 讀不到 {name}")
        return None

    OH, OW = img.shape[:2]
    res = fc.detect(img, cfg)
    info = res["info"]

    if "error" in info:
        # 偵測不到牙齒時輸出全黑 mask，讓下游照常跑完（等同「這個視角沒有菌斑」）
        print(f"  ⚠️  {name}: {info['error']}，輸出空 mask")
        mask_full = np.zeros((OH, OW), np.uint8)
        det_full = img.copy()
    else:
        # 放大回原圖解析度，維持與 teeth_test.py 相同的輸出尺寸
        mask_full = cv2.resize(res["plaque_mask"], (OW, OH),
                               interpolation=cv2.INTER_NEAREST)
        _, mask_full = cv2.threshold(mask_full, 127, 255, cv2.THRESH_BINARY)

        det_full = img.copy()
        cnts, _ = cv2.findContours(mask_full, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(det_full, cnts, -1, (0, 0, 255),
                         max(2, round(max(OH, OW) / 500)))

        print(f"  ✅ {name}: 牙齒 {info['tooth_px']:,} px，"
              f"菌斑 {info['plaque_px']:,} px "
              f"({info['plaque_ratio_of_tooth'] * 100:.1f}% of tooth)，"
              f"{len(info['regions'])} 區塊")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "mask_" + name), mask_full)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "det_" + name), det_full)

    if debug and "error" not in info:
        stem = os.path.splitext(name)[0]
        tooth = res["tooth_mask"]
        for tag, arr, lo, hi in (("fp", res["fp"], 0, 45), ("z", res["z"], 0, 5)):
            n = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
            hm = cv2.applyColorMap(n, cv2.COLORMAP_JET)
            out = np.zeros_like(res["plaque_mask"][..., None].repeat(3, 2))
            out[tooth > 0] = hm[tooth > 0]
            cv2.imwrite(os.path.join(OUTPUT_DIR, f"fluor_{tag}_{stem}.png"), out)

    return {"view": os.path.splitext(name)[0], "orig_size": [OW, OH], **info}


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cfg = _cfg_from_env()
    debug = os.environ.get("DENTAL_FLUOR_DEBUG") == "1"

    print("=" * 60)
    print("🔦 紫光燈模式牙菌斑偵測（405nm 螢光 / 光譜解混）")
    print("=" * 60)
    print(f"輸入: {INPUT_DIR}")
    print(f"輸出: {OUTPUT_DIR}")
    if cfg:
        print(f"參數覆寫: {cfg}")

    summary = {}
    found = 0
    for photo in PHOTOS:
        path = os.path.join(str(INPUT_DIR), photo)
        if not os.path.exists(path):
            print(f"  ⚠️  找不到 {photo}，跳過")
            continue
        found += 1
        r = detect_plaque(path, cfg, debug)
        if r:
            summary[r["view"]] = r

    if found == 0:
        print("❌ 五個視角的照片一張都找不到")
        return 1

    with open(os.path.join(OUTPUT_DIR, "fluorescence_summary.json"),
              "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 完成 {found} 個視角，摘要寫入 "
          f"{OUTPUT_DIR}/fluorescence_summary.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
