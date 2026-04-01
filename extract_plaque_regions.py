#!/usr/bin/env python3
"""
extract_plaque_regions.py

對每張照片：
  1. 讀 teeth_color_test/ 已有的 mask_*.jpg（菌斑遮罩）
  2. 用 SegmentAnyTooth 取得牙齒 ROI，保留原始 FDI 數值 mask
  3. 菌斑 AND 牙齒ROI → 純牙齒上的菌斑
  4. 記錄每個菌斑輪廓的：位置、大小、形狀、FDI 編號、在照片的相對座標
  5. 彙整成 plaque_regions.json

輸出到：SegmentAnyTooth/plaque_output/plaque_regions.json
         SegmentAnyTooth/plaque_output/roi_mask_*.png （過濾後的乾淨 mask）
"""

import cv2
import numpy as np
import json
from pathlib import Path

# ==================== 路徑設定 ====================
import sys; sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")
from user_env import get_paths, setup_user_dirs, BASE as _SAT_BASE
_PATHS = get_paths()
setup_user_dirs(_PATHS["user_dir"])
BASE       = _PATHS["user_dir"]
MASK_DIR   = _PATHS["teeth_color_test"]
PHOTO_DIR  = _PATHS["real_teeth"]
OUTPUT_DIR = _PATHS["plaque_output"]
WEIGHT_DIR = _SAT_BASE / "weight"

# ==================== 各視角設定 ====================
VIEW_CONFIG = {
    'front': {
        'mask_file':  'mask_front.jpg',
        'photo_file': 'front.jpg',
        'sat_view':   'front',
        'jaw_split_ratio': 0.50,
        'jaw_label': 'both',
    },
    'left_side': {
        'mask_file':  'mask_left_side.jpg',
        'photo_file': 'left_side.jpg',
        'sat_view':   'left',
        'jaw_split_ratio': None,
        'jaw_label': 'both',
    },
    'right_side': {
        'mask_file':  'mask_right_side.jpg',
        'photo_file': 'right_side.jpg',
        'sat_view':   'right',
        'jaw_split_ratio': None,
        'jaw_label': 'both',
    },
    'upper_occlusal': {
        'mask_file':  'mask_upper_occlusal.jpg',
        'photo_file': 'upper_occlusal.jpg',
        'sat_view':   'upper',
        'jaw_split_ratio': None,
        'jaw_label': 'upper',
    },
    'lower_occlusal': {
        'mask_file':  'mask_lower_occlusal.jpg',
        'photo_file': 'lower_occlusal.jpg',
        'sat_view':   'lower',
        'jaw_split_ratio': None,
        'jaw_label': 'lower',
    },
}

MIN_CONTOUR_AREA = 200   # 最小輪廓面積（像素），過濾雜訊

# ==================== Step 1: 取得牙齒 FDI mask ====================

def get_tooth_roi(photo_path, sat_view):
    """
    呼叫 SegmentAnyTooth，回傳：
      fdi_mask  : 原始 FDI 數值 mask（uint8, 0=背景, 11~48=FDI牙號）
      binary_roi: 二值化 ROI（uint8, 0 or 255），用於 bitwise_and
    """
    try:
        from segmentanytooth import predict
        fdi_mask = predict(
            image_path=str(photo_path),
            view=sat_view,
            weight_dir=str(WEIGHT_DIR),
            sam_batch_size=10
        )
        binary_roi = (fdi_mask > 0).astype(np.uint8) * 255
        tooth_px   = (binary_roi > 0).sum()
        fdis       = [int(v) for v in np.unique(fdi_mask) if v > 0]
        print(f"    ✅ SAT 牙齒區域: {tooth_px:,} px，偵測到 FDI: {fdis}")
        return fdi_mask, binary_roi
    except Exception as e:
        print(f"    ⚠️  SAT 失敗 ({e})，跳過 ROI 過濾")
        return None, None

# ==================== Step 2: 判斷輪廓對應的 FDI 編號 ====================

def get_fdi_for_contour(cnt, fdi_mask, img_h, img_w):
    """
    建立輪廓的填滿 mask，與 fdi_mask 疊合，
    取像素數最多的非零 FDI 值作為該菌斑的所屬牙齒編號。
    手部 / 非牙齒區塊在 fdi_mask 裡全是 0，回傳 None。
    """
    contour_fill = np.zeros((img_h, img_w), dtype=np.uint8)
    cv2.drawContours(contour_fill, [cnt], -1, 255, -1)

    # 只取輪廓範圍內的 FDI 值
    fdi_in_contour = fdi_mask[contour_fill > 0]
    fdi_nonzero    = fdi_in_contour[fdi_in_contour > 0]

    if len(fdi_nonzero) == 0:
        return None   # 手部 / 背景區塊

    counts  = np.bincount(fdi_nonzero.astype(np.int32))
    fdi_val = int(np.argmax(counts))
    return fdi_val

# ==================== Step 3: 計算輪廓特徵 ====================

def contour_features(cnt, img_h, img_w):
    area = float(cv2.contourArea(cnt))
    x, y, w, h = cv2.boundingRect(cnt)
    M = cv2.moments(cnt)
    if M['m00'] > 0:
        cx = M['m10'] / M['m00']
        cy = M['m01'] / M['m00']
    else:
        cx, cy = x + w / 2, y + h / 2

    epsilon = 0.02 * cv2.arcLength(cnt, True)
    approx  = cv2.approxPolyDP(cnt, epsilon, True)
    points  = [[round(float(p[0][0]) / img_w, 4),
                round(float(p[0][1]) / img_h, 4)]
               for p in approx]

    perimeter   = cv2.arcLength(cnt, True)
    circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)

    return {
        'area_px':    round(area),
        'area_ratio': round(area / (img_h * img_w), 5),
        'bbox': {
            'x': round(x / img_w, 4),
            'y': round(y / img_h, 4),
            'w': round(w / img_w, 4),
            'h': round(h / img_h, 4),
        },
        'centroid': {
            'x': round(cx / img_w, 4),
            'y': round(cy / img_h, 4),
        },
        'circularity': round(float(circularity), 3),
        'points_norm': points,
    }

# ==================== 主流程 ====================

print("=" * 60)
print("🦷 提取牙齒上的菌斑區域 + 輸出 JSON")
print("=" * 60)

all_regions = {}

for view_name, cfg in VIEW_CONFIG.items():
    print(f"\n📷 {view_name}")

    mask_path  = MASK_DIR  / cfg['mask_file']
    photo_path = PHOTO_DIR / cfg['photo_file']

    if not mask_path.exists():
        print(f"  ⚠️  找不到 mask: {cfg['mask_file']}")
        continue

    # 讀原始菌斑 mask
    raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw_mask is None:
        print(f"  ⚠️  讀取失敗")
        continue
    _, raw_mask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)
    img_h, img_w = raw_mask.shape

    print(f"  原始菌斑像素: {(raw_mask > 0).sum():,} ({(raw_mask>0).sum()/img_h/img_w*100:.1f}%)")

    # 取得牙齒 ROI（保留 FDI 數值 mask）
    fdi_mask   = None
    binary_roi = None
    if photo_path.exists():
        fdi_mask, binary_roi = get_tooth_roi(photo_path, cfg['sat_view'])
        if fdi_mask is not None:
            # 確保尺寸一致
            if fdi_mask.shape != raw_mask.shape:
                fdi_mask   = cv2.resize(fdi_mask,   (img_w, img_h),
                                        interpolation=cv2.INTER_NEAREST)
                binary_roi = cv2.resize(binary_roi, (img_w, img_h),
                                        interpolation=cv2.INTER_NEAREST)
    else:
        print(f"  ⚠️  找不到原始照片: {cfg['photo_file']}，跳過 ROI 過濾")

    # 套用 ROI 過濾
    if binary_roi is not None:
        filtered_mask = cv2.bitwise_and(raw_mask, binary_roi)
        print(f"  過濾後菌斑像素: {(filtered_mask > 0).sum():,} ({(filtered_mask>0).sum()/img_h/img_w*100:.1f}%)")
    else:
        filtered_mask = raw_mask.copy()
        print(f"  （未做 ROI 過濾）")

    # 存過濾後的 mask
    cv2.imwrite(str(OUTPUT_DIR / f"roi_mask_{view_name}.png"), filtered_mask)

    # debug 疊圖
    if binary_roi is not None:
        debug = cv2.cvtColor(raw_mask // 4, cv2.COLOR_GRAY2BGR)
        debug[binary_roi > 0]    = [0, 60, 0]
        debug[filtered_mask > 0] = [0, 0, 220]
        cv2.imwrite(str(OUTPUT_DIR / f"debug_roi_{view_name}.png"), debug)

    # det 標註圖（綠框=牙齒ROI, 紅框=菌斑, 標註FDI編號）
    if photo_path.exists():
        orig = cv2.imread(str(photo_path))
        if orig is not None:
            orig_h, orig_w = orig.shape[:2]
            det = orig.copy()

            if binary_roi is not None:
                roi_resized = cv2.resize(binary_roi, (orig_w, orig_h),
                                         interpolation=cv2.INTER_NEAREST)
                roi_cnts, _ = cv2.findContours(roi_resized, cv2.RETR_EXTERNAL,
                                                cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(det, roi_cnts, -1, (0, 200, 0), 2)

            fm_resized = cv2.resize(filtered_mask, (orig_w, orig_h),
                                     interpolation=cv2.INTER_NEAREST)
            fm_cnts, _ = cv2.findContours(fm_resized, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            fdi_resized = cv2.resize(fdi_mask, (orig_w, orig_h),
                                     interpolation=cv2.INTER_NEAREST) \
                          if fdi_mask is not None else None

            scale = (orig_w * orig_h) / (img_w * img_h)
            for c in fm_cnts:
                if cv2.contourArea(c) < MIN_CONTOUR_AREA * scale:
                    continue
                cv2.drawContours(det, [c], -1, (0, 0, 255), 2)
                # 在輪廓旁標註 FDI 編號
                if fdi_resized is not None:
                    fdi_val = get_fdi_for_contour(c, fdi_resized, orig_h, orig_w)
                    if fdi_val is not None:
                        rx, ry, rw, rh = cv2.boundingRect(c)
                        cv2.putText(det, str(fdi_val),
                                    (rx, max(ry - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 220, 255), 2, cv2.LINE_AA)

            cv2.imwrite(str(OUTPUT_DIR / f"det_{view_name}.jpg"), det)

    # 提取輪廓
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    regions      = []
    null_fdi_cnt = 0

    for i, cnt in enumerate(contours):
        if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
            continue

        feat = contour_features(cnt, img_h, img_w)

        # FDI 編號
        fdi_val = None
        if fdi_mask is not None:
            fdi_val = get_fdi_for_contour(cnt, fdi_mask, img_h, img_w)
        feat['fdi'] = fdi_val   # None 表示非牙齒區塊（手部等）

        if fdi_val is None:
            null_fdi_cnt += 1

        # 上下顎
        split = cfg['jaw_split_ratio']
        if split is not None:
            jaw = 'upper' if feat['centroid']['y'] < split else 'lower'
        else:
            jaw = cfg['jaw_label']

        feat['jaw']       = jaw
        feat['region_id'] = f"{view_name}_{i:02d}"
        regions.append(feat)

    regions.sort(key=lambda r: r['centroid']['x'])

    if null_fdi_cnt > 0:
        print(f"  ⚠️  {null_fdi_cnt} 個輪廓 FDI=null（非牙齒區塊，手部等）")

    all_regions[view_name] = {
        'image_size': {'w': img_w, 'h': img_h},
        'jaw_label':  cfg['jaw_label'],
        'total_plaque_px_raw':      int((raw_mask > 0).sum()),
        'total_plaque_px_filtered': int((filtered_mask > 0).sum()),
        'roi_available': fdi_mask is not None,
        'region_count':  len(regions),
        'regions':       regions,
    }

    print(f"  輪廓數量（過濾後）: {len(regions)}")

# 輸出 JSON
json_path = OUTPUT_DIR / "plaque_regions.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(all_regions, f, indent=2, ensure_ascii=False)

print(f"\n{'='*60}")
print(f"✅ 完成！")
print(f"   JSON: {json_path}")
print(f"   過濾後 mask: {OUTPUT_DIR}/roi_mask_*.png")
print(f"   Debug 疊圖: {OUTPUT_DIR}/debug_roi_*.png")

print(f"\n📊 各視角摘要:")
for v, data in all_regions.items():
    raw  = data['total_plaque_px_raw']
    filt = data['total_plaque_px_filtered']
    keep = filt / raw * 100 if raw > 0 else 0
    roi  = "✅" if data['roi_available'] else "❌"
    print(f"  {v:20s} ROI:{roi}  輪廓:{data['region_count']:3d}  "
          f"保留 {keep:.0f}% ({raw:,}→{filt:,} px)")