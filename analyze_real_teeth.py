#!/usr/bin/env python3
"""
真實牙齒照片分析 - 簡化版（深度優化版）
只分析 real_teeth 資料夾，不區分 normal/missing

優化項目（在前一版基礎上繼續）：
  1. _measure_teeth_in_mask:
     - 將 (mask == tooth_id) 的計算改為一次性建立 label→索引 lookup table，
       避免每顆牙重複掃描整張 H×W 陣列
     - tooth_mask_2d 只建立一次，共用給 contour_cps 和 tilt_info
     - extract_contour_control_points import 移到模組頂層（避免每顆牙重複 import）
  2. calculate_3d_dimensions: VIEW_WEIGHT_MAP 改為模組級常數（不在函式內每次建立）
  3. 主流程: all_detected_by_view 在 analyze_single_photo 裡一次回傳，不拆分
"""

from segmentanytooth import predict
import cv2
import numpy as np
from pathlib import Path
import json

import sys; sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")
from user_env import get_paths, setup_user_dirs
_PATHS = get_paths()
setup_user_dirs(_PATHS["user_dir"])
BASE_DIR = _PATHS["user_dir"]
REAL_TEETH_DIR = BASE_DIR / "real_teeth_processed"

WEIGHT_DIR = Path("/home/Zhen/projects/SegmentAnyTooth/weight")
OUTPUT_DIR = _PATHS["analysis"]

# ★ 頂層 import（避免每顆牙重複 import）
from tps_deformation import extract_contour_control_points

VIEW_MAPPING = {
    'front.jpg': 'front',
    'left_side.jpg': 'left',
    'right_side.jpg': 'right',
    'upper_occlusal.jpg': 'upper',
    'lower_occlusal.jpg': 'lower'
}

PIXEL_TO_MM = 0.15

MEASUREMENT_DEFINITION = {
    'front': {'width_axis': 'width',  'height_axis': 'height'},
    'upper': {'width_axis': 'width',  'height_axis': 'depth'},
    'lower': {'width_axis': 'width',  'height_axis': 'depth'},
    'left':  {'width_axis': 'depth',  'height_axis': 'height'},
    'right': {'width_axis': 'depth',  'height_axis': 'height'}
}

MIN_VIEWS = {'anterior': 2, 'premolar': 1, 'molar': 1, 'wisdom': 2}
CONFIDENCE_THRESHOLD = 0.4

VIEW_AXIS_WEIGHTS = {
    1: {'front_width': 1.0, 'front_height': 1.0, 'occlusal_width': 0.8, 'side_depth': 0.3, 'side_height': 0.7},
    2: {'front_width': 0.7, 'front_height': 0.9, 'occlusal_width': 0.9, 'side_depth': 0.6, 'side_height': 0.8},
    3: {'front_width': 0.5, 'front_height': 0.8, 'occlusal_width': 1.0, 'side_depth': 0.7, 'side_height': 0.9},
    4: {'front_width': 0.3, 'front_height': 0.7, 'occlusal_width': 1.0, 'side_depth': 0.9, 'side_height': 1.0},
    5: {'front_width': 0.2, 'front_height': 0.6, 'occlusal_width': 1.0, 'side_depth': 0.9, 'side_height': 1.0},
    6: {'front_width': 0.1, 'front_height': 0.4, 'occlusal_width': 1.0, 'side_depth': 1.0, 'side_height': 1.0},
    7: {'front_width': 0.1, 'front_height': 0.3, 'occlusal_width': 1.0, 'side_depth': 1.0, 'side_height': 1.0},
    8: {'front_width': 0.0, 'front_height': 0.2, 'occlusal_width': 1.0, 'side_depth': 1.0, 'side_height': 1.0},
}

ANTERIOR_TEETH = {11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43}

SYMMETRIC_PAIRS = {
    11: 21, 12: 22, 13: 23, 14: 24, 15: 25, 16: 26, 17: 27, 18: 28,
    21: 11, 22: 12, 23: 13, 24: 14, 25: 15, 26: 16, 27: 17, 28: 18,
    31: 41, 32: 42, 33: 43, 34: 44, 35: 45, 36: 46, 37: 47, 38: 48,
    41: 31, 42: 32, 43: 33, 44: 34, 45: 35, 46: 36, 47: 37, 48: 38,
}

# ★ 模組級常數：VIEW_WEIGHT_MAP（不在函式內每次建立 lambda dict）
def _make_view_weight_fn(wd):
    return {
        'front':  (wd['front_width'],    wd['front_height'], 0.0),
        'upper':  (wd['occlusal_width'], 0.0,                wd['occlusal_width']),
        'lower':  (wd['occlusal_width'], 0.0,                wd['occlusal_width']),
        'left':   (0.0,                  wd['side_height'],  wd['side_depth']),
        'right':  (0.0,                  wd['side_height'],  wd['side_depth']),
    }

# 預先為每個 tooth_digit 建立 view weight map
_VIEW_WEIGHT_CACHE: dict[int, dict] = {
    digit: _make_view_weight_fn(wd)
    for digit, wd in VIEW_AXIS_WEIGHTS.items()
}

print("="*70)
print("🦷 真實牙齒照片分析（深度優化版）")
print("="*70)


# ==================== 牙齒位置分類 ====================

def classify_tooth_position(fdi):
    fdi = int(fdi)
    if fdi in [18, 28, 38, 48]: return 'wisdom'
    if fdi in [17, 27, 37, 47, 16, 26, 36, 46]: return 'molar'
    if fdi in [14, 15, 24, 25, 34, 35, 44, 45]: return 'premolar'
    return 'anterior'


# ==================== 傾斜角提取 ====================

def extract_tilt_angle_from_mask(mask_2d, view):
    if mask_2d is None or mask_2d.sum() == 0:
        return None
    mask_uint8 = (mask_2d > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None
    pts = contour.reshape(-1, 2).astype(float)
    x_coords, y_coords = pts[:, 0], pts[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    h_px = y_max - y_min
    if h_px < 5:
        return None
    crown_mask = y_coords < y_min + h_px * 0.20
    neck_mask  = y_coords > y_max - h_px * 0.20
    if crown_mask.sum() < 2 or neck_mask.sum() < 2:
        return None
    crown_x   = x_coords[crown_mask].mean()
    neck_x    = x_coords[neck_mask].mean()
    height_mm = h_px * PIXEL_TO_MM
    offset_mm = (crown_x - neck_x) * PIXEL_TO_MM
    if height_mm < 0.5:
        return None
    tilt_deg  = float(np.degrees(np.arctan2(offset_mm, height_mm)))
    hull      = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    area      = cv2.contourArea(contour)
    solidity  = area / hull_area if hull_area > 0 else 0.0
    confidence = min(1.0, solidity * min(1.0, area / 500.0))
    return {
        'tilt_deg': round(tilt_deg, 2),
        'crown_offset_mm': round(offset_mm, 2),
        'height_mm': round(height_mm, 2),
        'confidence': round(confidence, 3),
        'view': view,
    }


# ==================== 單視角分析（深度優化）====================

def _measure_teeth_in_mask(mask, unique_teeth, view):
    """
    ★ 深度優化版：
    - 預先建立 label→pixel indices 的查找表（一次掃描整張 mask）
    - 每顆牙的 tooth_mask_2d 從查找表直接重建，不重複 (mask == id) 掃描
    - tooth_mask_2d 共用給 contour_cps 和 tilt_info，不建立兩次
    """
    measurement_def = MEASUREMENT_DEFINITION.get(view, {'width_axis': 'width', 'height_axis': 'height'})
    h, w = mask.shape

    # ★ 一次性建立 label→indices 查找表（避免每顆牙重複 np.where）
    flat_mask = mask.ravel()
    # np.argsort on label: 先 sort，再用 searchsorted 找各 label 的範圍
    sorted_order = np.argsort(flat_mask, kind='stable')
    sorted_labels = flat_mask[sorted_order]

    tooth_measurements = {}
    need_tilt = view in ['left', 'right']

    for tooth_id in unique_teeth:
        fdi = int(tooth_id)

        # ★ 用 searchsorted 在已排序 label 陣列裡找該 tooth 的 indices（比 np.where 快）
        lo = np.searchsorted(sorted_labels, tooth_id, side='left')
        hi = np.searchsorted(sorted_labels, tooth_id, side='right')
        if lo >= hi:
            continue
        flat_indices = sorted_order[lo:hi]

        # 從 flat indices 重建 2D mask（只標記這顆牙的像素）
        tooth_mask_2d = np.zeros((h, w), dtype=np.uint8)
        tooth_mask_2d.ravel()[flat_indices] = 255

        contours, _ = cv2.findContours(tooth_mask_2d, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(contour)

        if len(contour) >= 5:
            ellipse     = cv2.fitEllipse(contour)
            _, ellipse_axes, ellipse_angle = ellipse
            minor_axis  = min(ellipse_axes)
            major_axis  = max(ellipse_axes)
        else:
            xb, yb, wb, hb = cv2.boundingRect(contour)
            minor_axis  = float(wb)
            major_axis  = float(hb)
            ellipse_angle = 0.0

        hull      = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity  = float(area_px) / float(hull_area) if hull_area > 0 else 0.0

        if view in ['upper', 'lower']:
            width_mm  = round(minor_axis * PIXEL_TO_MM, 2)
            depth_mm  = round(major_axis * PIXEL_TO_MM, 2)
            height_mm = None
        elif view == 'front':
            width_mm  = round(minor_axis * PIXEL_TO_MM, 2)
            height_mm = round(major_axis * PIXEL_TO_MM, 2)
            depth_mm  = None
        else:
            depth_mm  = round(minor_axis * PIXEL_TO_MM, 2)
            height_mm = round(major_axis * PIXEL_TO_MM, 2)
            width_mm  = None

        # ★ tooth_mask_2d 共用（不再建立第二次）
        contour_cps     = extract_contour_control_points(tooth_mask_2d, view, fdi, PIXEL_TO_MM)
        contour_cp_data = None
        if contour_cps is not None:
            contour_cp_data = {
                'view': view,
                'bbox': contour_cps.get('bbox', {}),
                'points': contour_cps.get('points', {})
            }

        tilt_info = None
        if need_tilt and fdi in ANTERIOR_TEETH:
            tilt_info = extract_tilt_angle_from_mask(tooth_mask_2d, view)

        tooth_measurements[fdi] = {
            'fdi': fdi,
            'pixels': {
                'area':          float(area_px),
                'major_axis':    round(major_axis, 1),
                'minor_axis':    round(minor_axis, 1),
                'ellipse_angle': round(float(ellipse_angle), 1)
            },
            'millimeters': {'width': width_mm, 'height': height_mm, 'depth': depth_mm},
            'solidity': round(solidity, 3),
            'axis_mapping': {
                'bbox_width_represents':  measurement_def['width_axis'],
                'bbox_height_represents': measurement_def['height_axis']
            },
            'view': view,
            'contour_cp': contour_cp_data,
            'tilt_info':  tilt_info,
            'position':   classify_tooth_position(fdi)
        }

    return tooth_measurements


def analyze_single_photo(image_path, view):
    print(f"  📸 {image_path.name} ({view})", end="")
    mask = predict(
        image_path=str(image_path), view=view,
        weight_dir=str(WEIGHT_DIR), sam_batch_size=10
    )
    unique_teeth  = np.unique(mask)
    unique_teeth  = unique_teeth[unique_teeth > 0]
    detected_list = sorted([int(t) for t in unique_teeth])
    print(f" → {len(detected_list)} 顆")
    tooth_measurements = _measure_teeth_in_mask(mask, unique_teeth, view)
    return mask, tooth_measurements, detected_list


# ==================== 可信度計算 ====================

def calculate_confidence_score(tooth_data, position):
    num_views    = tooth_data['num_views']
    min_required = MIN_VIEWS[position]
    view_score   = 1.0 if num_views >= min_required + 1 else \
                   0.7 if num_views == min_required else 0.4

    measurements = tooth_data.get('measurements', [])
    if len(measurements) >= 2:
        major_axes = [m['pixels']['major_axis'] for m in measurements
                      if 'major_axis' in m.get('pixels', {})]
        if len(major_axes) >= 2:
            cv_val = np.std(major_axes) / np.mean(major_axes) if np.mean(major_axes) > 0 else 0
            consistency_score = 1.0 - min(cv_val, 0.5) * 2
        else:
            consistency_score = 0.7
    else:
        consistency_score = 0.7

    return view_score * 0.7 + consistency_score * 0.3


# ==================== 3D 尺寸計算（使用模組級快取）====================

def calculate_3d_dimensions(measurements_list, tooth_id=None):
    tooth_digit = (tooth_id % 10) if tooth_id else 5
    # ★ 使用模組級快取，不在函式內建立 lambda dict
    weight_map  = _VIEW_WEIGHT_CACHE.get(tooth_digit, _VIEW_WEIGHT_CACHE[5])

    dims = {'width': [], 'height': [], 'depth': []}

    for m in measurements_list:
        mm       = m.get('millimeters', {})
        view     = m.get('view', 'unknown')
        solidity = m.get('solidity', 1.0)
        pixels   = m.get('pixels', {})
        area_px  = pixels.get('area', 999)
        angle    = pixels.get('ellipse_angle', 0.0)

        if solidity < 0.45 or area_px < 100:
            continue

        angle_correction = 1.0 - 0.3 * min(abs(angle % 90), 90 - abs(angle % 90)) / 45.0

        weights_tuple = weight_map.get(view)
        if weights_tuple is None:
            w_width = w_height = w_depth = 0.5
        else:
            w_width, w_height, w_depth = weights_tuple

        for axis, base_w in (('width', w_width), ('height', w_height), ('depth', w_depth)):
            val = mm.get(axis)
            if val is not None and val > 0 and base_w > 0:
                dims[axis].append((val, base_w * angle_correction * solidity))

    def filter_and_summarize(pairs):
        if not pairs:
            return {'values': [], 'views': [], 'count': 0, 'mean': None,
                    'std': None, 'min': None, 'max': None}
        vals = np.array([p[0] for p in pairs])
        wgts = np.array([p[1] for p in pairs])

        if len(vals) >= 3:
            keep = (vals > 0) & (vals <= np.median(vals) * 3.0)
            vals, wgts = vals[keep], wgts[keep]
        if len(vals) >= 3:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr    = q3 - q1
            keep   = (vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)
            vals, wgts = vals[keep], wgts[keep]

        if len(vals) == 0:
            return {'values': [], 'views': [], 'count': 0, 'mean': None,
                    'std': None, 'min': None, 'max': None}

        w_sum = wgts.sum()
        wmean = float(np.dot(vals, wgts) / w_sum) if w_sum > 0 else float(vals.mean())
        return {
            'values': vals.tolist(), 'views': [], 'count': len(vals),
            'mean': round(wmean, 2), 'std': round(float(vals.std()), 2),
            'min': round(float(vals.min()), 2), 'max': round(float(vals.max()), 2),
        }

    return {axis: filter_and_summarize(pairs) for axis, pairs in dims.items()}


# ==================== 融合多視角 ====================

def merge_multiview_detections(all_measurements):
    print("\n🔗 融合多視角檢測並計算 3D 尺寸...")

    tooth_view_map: dict[int, list] = {}
    for photo_data in all_measurements.values():
        for tooth_id, mdata in photo_data.items():
            tooth_view_map.setdefault(tooth_id, []).append(mdata)

    merged = {}
    SYMMETRY_DIFF_THRESHOLD = 0.4
    SYMMETRY_WEIGHT         = 0.3

    for tooth_id, measurements_list in tooth_view_map.items():
        views_detected = [m['view'] for m in measurements_list]
        position       = classify_tooth_position(tooth_id)

        tooth_data = {
            'fdi': tooth_id, 'position': position,
            'detected_in_views': views_detected,
            'num_views': len(views_detected),
            'measurements': measurements_list
        }

        if position == 'wisdom':
            avg_solidity = np.mean([m.get('solidity', 0) for m in measurements_list])
            if avg_solidity < 0.80:
                continue

        confidence = calculate_confidence_score(tooth_data, position)
        tooth_data['confidence'] = confidence
        tooth_data['warning']    = (
            'low_confidence'     if confidence < CONFIDENCE_THRESHOLD else
            'insufficient_views' if len(views_detected) < MIN_VIEWS[position] else None
        )
        tooth_data['dimensions_3d'] = calculate_3d_dimensions(measurements_list, tooth_id=tooth_id)

        if tooth_id in ANTERIOR_TEETH:
            tilt_entries = [
                (m['tilt_info']['tilt_deg'], m['tilt_info']['confidence'])
                for m in measurements_list
                if m.get('tilt_info') and m['tilt_info'].get('confidence', 0) > 0.1
            ]
            if tilt_entries:
                angles  = np.array([t[0] for t in tilt_entries])
                weights = np.array([t[1] for t in tilt_entries])
                tooth_data['tilt_3d'] = {
                    'tilt_deg':     round(float(np.dot(angles, weights) / weights.sum()), 2),
                    'confidence':   round(float(weights.mean()), 3),
                    'source_count': len(tilt_entries),
                }
            else:
                tooth_data['tilt_3d'] = None
        else:
            tooth_data['tilt_3d'] = None

        merged[tooth_id] = tooth_data

    # 對稱補全
    for tooth_id, tooth_data in merged.items():
        mirror_id = SYMMETRIC_PAIRS.get(tooth_id)
        if mirror_id not in merged:
            continue
        mirror_data = merged[mirror_id]

        for axis in ('width', 'height', 'depth'):
            dim = tooth_data['dimensions_3d'][axis]
            mir = mirror_data['dimensions_3d'][axis]
            unreliable = (dim['count'] == 0 or dim['count'] == 1 or
                          (dim['std'] and dim['mean'] and dim['std'] / dim['mean'] > 0.3))
            if not unreliable or mir['mean'] is None or dim['mean'] is None:
                continue
            if abs(dim['mean'] - mir['mean']) / max(dim['mean'], mir['mean']) > SYMMETRY_DIFF_THRESHOLD:
                continue
            tooth_data['dimensions_3d'][axis]['mean'] = round(
                dim['mean'] * (1 - SYMMETRY_WEIGHT) + mir['mean'] * SYMMETRY_WEIGHT, 2)
            tooth_data['dimensions_3d'][axis]['symmetry_assisted'] = True

        if tooth_id in ANTERIOR_TEETH:
            my_tilt     = tooth_data.get('tilt_3d')
            mirror_tilt = mirror_data.get('tilt_3d')
            if my_tilt is None and mirror_tilt and mirror_tilt.get('confidence', 0) > 0.3:
                tooth_data['tilt_3d'] = {
                    'tilt_deg':          mirror_tilt['tilt_deg'],
                    'confidence':        mirror_tilt['confidence'] * 0.7,
                    'source_count':      0,
                    'symmetry_assisted': True,
                }

    return merged


def classify_by_position(merged_data):
    classified = {'anterior': [], 'premolar': [], 'molar': [], 'wisdom': []}
    for tooth_id, data in merged_data.items():
        classified[data['position']].append(tooth_id)
    return classified


def identify_suspicious_detections(merged_data):
    suspicious = {'low_confidence': [], 'insufficient_views': [], 'might_be_false': []}
    for tooth_id, data in merged_data.items():
        warning    = data.get('warning')
        confidence = data['confidence']
        if warning == 'low_confidence':    suspicious['low_confidence'].append(tooth_id)
        if warning == 'insufficient_views': suspicious['insufficient_views'].append(tooth_id)
        if confidence < 0.3 and data['num_views'] == 1:
            suspicious['might_be_false'].append(tooth_id)
    return suspicious


# ==================== 主流程 ====================

def main():
    print(f"\n{'='*70}")
    print(f"📂 處理 real_teeth 資料夾")
    print(f"{'='*70}\n")

    all_measurements     = {}
    all_detected_by_view = {}

    for photo_name, view in VIEW_MAPPING.items():
        photo_path = REAL_TEETH_DIR / photo_name
        if not photo_path.exists():
            print(f"  ⚠️  未找到: {photo_name}"); continue
        mask, measurements, detected_list = analyze_single_photo(photo_path, view)
        all_measurements[photo_name]     = measurements
        all_detected_by_view[photo_name] = detected_list

    merged     = merge_multiview_detections(all_measurements)
    classified = classify_by_position(merged)
    suspicious = identify_suspicious_detections(merged)

    total_detected = len(merged)
    reliable     = [t for t, d in merged.items() if d.get('warning') is None]
    questionable = [t for t, d in merged.items() if d.get('warning') is not None]

    print(f"\n📊 檢測統計：")
    print(f"  總計: {total_detected} 顆  可靠: {len(reliable)} 顆  可疑: {len(questionable)} 顆")
    print(f"\n  按位置分類:")
    for pos in ('anterior', 'premolar', 'molar', 'wisdom'):
        label = {'anterior': '前牙', 'premolar': '前臼齒', 'molar': '臼齒', 'wisdom': '智齒'}[pos]
        print(f"    {label}: {len(classified[pos])} 顆 → {sorted(classified[pos])}")

    if suspicious['might_be_false']:
        print(f"\n  ⚠️  可能誤檢: {sorted(suspicious['might_be_false'])}")

    all_model_teeth = (set(range(11, 19)) | set(range(21, 29)) |
                       set(range(31, 39)) | set(range(41, 49)))
    detected_teeth  = set(merged.keys())
    never_detected  = sorted(all_model_teeth - detected_teeth)

    print(f"\n📭 從未出現的牙齒: {len(never_detected)} 顆")
    if never_detected:
        print(f"  {never_detected}")

    result = {
        'total_detected':     total_detected,
        'reliable_count':     len(reliable),
        'questionable_count': len(questionable),
        'classified':   {k: sorted(v) for k, v in classified.items()},
        'suspicious':   {k: sorted(v) for k, v in suspicious.items()},
        'teeth':        merged,
        'by_view':      all_detected_by_view,
        'measurement_definition': MEASUREMENT_DEFINITION,
        'never_detected':  never_detected,
        'detected_teeth':  sorted(detected_teeth)
    }

    result_file = OUTPUT_DIR / "real_teeth_analysis.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n💾 已保存: {result_file}")
    print(f"\n{'='*70}")
    print("✅ 分析完成！")
    print(f"{'='*70}")
    print(f"\n💡 下一步: 執行 create_personalized_3d_real.py")


if __name__ == "__main__":
    main()