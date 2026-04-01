#!/usr/bin/env python3
"""
真實牙齒照片分析 - 簡化版
只分析 real_teeth 資料夾，不區分 normal/missing
"""

from segmentanytooth import predict
import cv2
import numpy as np
from pathlib import Path
import json

# ==================== 設定 ====================

import sys; sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")
from user_env import get_paths, setup_user_dirs
_PATHS = get_paths()
setup_user_dirs(_PATHS["user_dir"])
BASE_DIR = _PATHS["user_dir"]
REAL_TEETH_DIR = BASE_DIR / "real_teeth_processed"

WEIGHT_DIR = Path("/home/Zhen/projects/SegmentAnyTooth/weight")
OUTPUT_DIR = _PATHS["analysis"]

VIEW_MAPPING = {
    'front.jpg': 'front',
    'left_side.jpg': 'left',
    'right_side.jpg': 'right',
    'upper_occlusal.jpg': 'upper',
    'lower_occlusal.jpg': 'lower'
}

PIXEL_TO_MM = 0.15

# ⭐⭐⭐ 測量維度定義（根據視角）
MEASUREMENT_DEFINITION = {
    'front': {
        'width_axis': 'width',
        'height_axis': 'height'
    },
    'upper': {
        'width_axis': 'width',
        'height_axis': 'depth'
    },
    'lower': {
        'width_axis': 'width',
        'height_axis': 'depth'
    },
    'left': {
        'width_axis': 'depth',
        'height_axis': 'height'
    },
    'right': {
        'width_axis': 'depth',
        'height_axis': 'height'
    }
}

MIN_VIEWS = {
    'anterior': 2,
    'premolar': 1,
    'molar': 1,
    'wisdom': 2
}

CONFIDENCE_THRESHOLD = 0.4

# ⭐ 視角加權表：依FDI號碼和視角決定可信度
# 格式：tooth_number末位(1-8) → 各視角對各維度的加權
VIEW_AXIS_WEIGHTS = {
    # 末位1,2 = 門牙：front width最準
    1: {'front_width': 1.0, 'front_height': 1.0, 'occlusal_width': 0.8, 'side_depth': 0.3, 'side_height': 0.7},
    2: {'front_width': 0.7, 'front_height': 0.9, 'occlusal_width': 0.9, 'side_depth': 0.6, 'side_height': 0.8},
    # 末位3 = 犬齒
    3: {'front_width': 0.5, 'front_height': 0.8, 'occlusal_width': 1.0, 'side_depth': 0.7, 'side_height': 0.9},
    # 末位4,5 = 前臼齒：front width很不準
    4: {'front_width': 0.3, 'front_height': 0.7, 'occlusal_width': 1.0, 'side_depth': 0.9, 'side_height': 1.0},
    5: {'front_width': 0.2, 'front_height': 0.6, 'occlusal_width': 1.0, 'side_depth': 0.9, 'side_height': 1.0},
    # 末位6,7,8 = 臼齒：front幾乎無效
    6: {'front_width': 0.1, 'front_height': 0.4, 'occlusal_width': 1.0, 'side_depth': 1.0, 'side_height': 1.0},
    7: {'front_width': 0.1, 'front_height': 0.3, 'occlusal_width': 1.0, 'side_depth': 1.0, 'side_height': 1.0},
    8: {'front_width': 0.0, 'front_height': 0.2, 'occlusal_width': 1.0, 'side_depth': 1.0, 'side_height': 1.0},
}

# 前牙 FDI 編號集合（用於傾斜分析）
ANTERIOR_TEETH = {11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43}

# 對稱牙齒對應表（11↔21, 12↔22, ...）
SYMMETRIC_PAIRS = {
    11: 21, 12: 22, 13: 23, 14: 24, 15: 25, 16: 26, 17: 27, 18: 28,
    21: 11, 22: 12, 23: 13, 24: 14, 25: 15, 26: 16, 27: 17, 28: 18,
    31: 41, 32: 42, 33: 43, 34: 44, 35: 45, 36: 46, 37: 47, 38: 48,
    41: 31, 42: 32, 43: 33, 44: 34, 45: 35, 46: 36, 47: 37, 48: 38,
}

print("="*70)
print("🦷 真實牙齒照片分析")
print("="*70)

# ==================== 牙齒位置分類 ====================

def classify_tooth_position(fdi):
    """根據 FDI 編號分類牙齒位置"""
    fdi = int(fdi)
    
    if fdi in [18, 28, 38, 48]:
        return 'wisdom'
    
    if fdi in [17, 27, 37, 47, 16, 26, 36, 46]:
        return 'molar'
    
    if fdi in [14, 15, 24, 25, 34, 35, 44, 45]:
        return 'premolar'
    
    return 'anterior'

# ==================== ⭐ 傾斜角提取 ====================

def extract_tilt_angle_from_mask(mask_2d, view):
    """
    從側面視角mask提取牙齒頰舌傾斜角。
    正值 = 牙冠偏舌側（內傾），負值 = 牙冠偏頰側（外傾）
    回傳 dict 或 None
    """
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
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    y_min, y_max = y_coords.min(), y_coords.max()
    h_px = y_max - y_min
    if h_px < 5:
        return None

    # 牙冠頂部20% → x中心；頸緣底部20% → x中心
    crown_mask = y_coords < y_min + h_px * 0.20
    neck_mask  = y_coords > y_max - h_px * 0.20

    if crown_mask.sum() < 2 or neck_mask.sum() < 2:
        return None

    crown_x = x_coords[crown_mask].mean()
    neck_x   = x_coords[neck_mask].mean()

    # 偏移換mm，計算傾斜角
    height_mm = h_px * PIXEL_TO_MM
    offset_mm = (crown_x - neck_x) * PIXEL_TO_MM

    if height_mm < 0.5:
        return None

    tilt_deg = float(np.degrees(np.arctan2(offset_mm, height_mm)))

    # 可信度
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    area = cv2.contourArea(contour)
    solidity = area / hull_area if hull_area > 0 else 0.0
    confidence = min(1.0, solidity * min(1.0, area / 500.0))

    return {
        'tilt_deg': round(tilt_deg, 2),
        'crown_offset_mm': round(offset_mm, 2),
        'height_mm': round(height_mm, 2),
        'confidence': round(confidence, 3),
        'view': view,
    }

# ==================== 分析函數 ====================

def analyze_single_photo(image_path, view):
    """分析單張照片"""
    
    print(f"  📸 {image_path.name} ({view})", end="")
    
    mask = predict(
        image_path=str(image_path),
        view=view,
        weight_dir=str(WEIGHT_DIR),
        sam_batch_size=10
    )
    
    unique_teeth = np.unique(mask)
    unique_teeth = unique_teeth[unique_teeth > 0]
    detected_list = sorted([int(t) for t in unique_teeth])
    
    print(f" → {len(detected_list)} 顆")
    
    # 測量每顆牙齒
    tooth_measurements = {}
    measurement_def = MEASUREMENT_DEFINITION.get(view, {
        'width_axis': 'width',
        'height_axis': 'height'
    })
    
    for tooth_id in unique_teeth:
        tooth_mask = (mask == tooth_id).astype(np.uint8)
        contours, _ = cv2.findContours(tooth_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        contour = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(contour)
        
        # ⭐ 橢圓擬合（比bounding box更準確）
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            ellipse_center, ellipse_axes, ellipse_angle = ellipse
            # axes是(短軸,長軸)，取較大的為長軸
            minor_axis = min(ellipse_axes)
            major_axis = max(ellipse_axes)
        else:
            # 點太少無法擬合橢圓，退回bounding box
            x, y, w, h = cv2.boundingRect(contour)
            minor_axis, major_axis = float(w), float(h)
            ellipse_angle = 0.0
        
        # ⭐ 凸包solidity（判斷形狀飽滿度，過低代表可能誤檢）
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area_px) / float(hull_area) if hull_area > 0 else 0.0
        
        # ⭐ 根據視角決定橢圓軸對應的真實維度
        # 咬合面視角：長軸=深度(前後)，短軸=寬度(左右)
        # 正面/側面視角：長軸=高度(上下)，短軸=寬度或深度
        if view in ['upper', 'lower']:
            width_mm = round(minor_axis * PIXEL_TO_MM, 2)
            depth_mm = round(major_axis * PIXEL_TO_MM, 2)
            height_mm = None
        elif view == 'front':
            width_mm = round(minor_axis * PIXEL_TO_MM, 2)
            height_mm = round(major_axis * PIXEL_TO_MM, 2)
            depth_mm = None
        else:  # left, right
            depth_mm = round(minor_axis * PIXEL_TO_MM, 2)
            height_mm = round(major_axis * PIXEL_TO_MM, 2)
            width_mm = None
        
        # ⭐ 提取輪廓控制點（給TPS用）
        from tps_deformation import extract_contour_control_points
        tooth_mask_2d = (mask == tooth_id).astype(np.uint8)
        contour_cps = extract_contour_control_points(
            tooth_mask_2d, view, int(tooth_id), PIXEL_TO_MM
        )
        contour_cp_data = None
        if contour_cps is not None:
            contour_cp_data = {
                'view': view,
                'bbox': contour_cps.get('bbox', {}),
                'points': contour_cps.get('points', {})  # 新版直接就是乾淨的dict
            }

        # ⭐ 傾斜角提取（只在側面視角 + 前牙）
        tilt_info = None
        if view in ['left', 'right'] and int(tooth_id) in ANTERIOR_TEETH:
            tilt_info = extract_tilt_angle_from_mask(tooth_mask_2d, view)

        tooth_measurements[int(tooth_id)] = {
            'fdi': int(tooth_id),
            'pixels': {
                'area': float(area_px),
                'major_axis': round(major_axis, 1),
                'minor_axis': round(minor_axis, 1),
                'ellipse_angle': round(float(ellipse_angle), 1)
            },
            'millimeters': {
                'width': width_mm,
                'height': height_mm,
                'depth': depth_mm
            },
            'solidity': round(solidity, 3),
            'axis_mapping': {
                'bbox_width_represents': measurement_def['width_axis'],
                'bbox_height_represents': measurement_def['height_axis']
            },
            'view': view,
            'contour_cp': contour_cp_data,
            'tilt_info': tilt_info,
            'position': classify_tooth_position(int(tooth_id))
        }
    
    return mask, tooth_measurements, detected_list


def calculate_confidence_score(tooth_data, position):
    """計算可信度分數"""
    num_views = tooth_data['num_views']
    min_required = MIN_VIEWS[position]
    
    if num_views >= min_required + 1:
        view_score = 1.0
    elif num_views == min_required:
        view_score = 0.7
    else:
        view_score = 0.4
    
    measurements = tooth_data.get('measurements', [])
    if len(measurements) >= 2:
        # ⭐ 改用新格式：用橢圓長軸做一致性判斷
        major_axes = [m['pixels']['major_axis'] for m in measurements 
                      if 'major_axis' in m.get('pixels', {})]
        
        if len(major_axes) >= 2:
            cv = np.std(major_axes) / np.mean(major_axes) if np.mean(major_axes) > 0 else 0
            consistency_score = 1.0 - min(cv, 0.5) * 2
        else:
            consistency_score = 0.7
    else:
        consistency_score = 0.7
    
    final_score = view_score * 0.7 + consistency_score * 0.3
    return final_score


def merge_multiview_detections(all_measurements):
    """融合多視角檢測結果"""
    
    print("\n🔗 融合多視角檢測並計算 3D 尺寸...")
    
    all_teeth = set()
    for photo_data in all_measurements.values():
        all_teeth.update(photo_data.keys())
    
    merged = {}
    
    for tooth_id in all_teeth:
        views_detected = []
        measurements_list = []
        
        for photo, teeth_data in all_measurements.items():
            if tooth_id in teeth_data:
                views_detected.append(teeth_data[tooth_id]['view'])
                measurements_list.append(teeth_data[tooth_id])
        
        position = classify_tooth_position(tooth_id)
        
        tooth_data = {
            'fdi': tooth_id,
            'position': position,
            'detected_in_views': views_detected,
            'num_views': len(views_detected),
            'measurements': measurements_list
        }

        # 智齒額外過濾：solidity 太低代表形狀不像真牙
        if position == 'wisdom':
            avg_solidity = np.mean([m.get('solidity', 0) for m in measurements_list])
            if avg_solidity < 0.80:
                continue  # 跳過這顆，不加入 merged
        
        confidence = calculate_confidence_score(tooth_data, position)
        tooth_data['confidence'] = confidence
        
        if confidence < CONFIDENCE_THRESHOLD:
            tooth_data['warning'] = 'low_confidence'
        elif len(views_detected) < MIN_VIEWS[position]:
            tooth_data['warning'] = 'insufficient_views'
        else:
            tooth_data['warning'] = None
        
        tooth_data['dimensions_3d'] = calculate_3d_dimensions(measurements_list, tooth_id=tooth_id)

        # ⭐ 融合傾斜角
        if tooth_id in ANTERIOR_TEETH:
            tilt_entries = []
            for m in measurements_list:
                ti = m.get('tilt_info')
                if ti and ti.get('confidence', 0) > 0.1:
                    tilt_entries.append((ti['tilt_deg'], ti['confidence']))
            if tilt_entries:
                angles  = np.array([t[0] for t in tilt_entries])
                weights = np.array([t[1] for t in tilt_entries])
                tooth_data['tilt_3d'] = {
                    'tilt_deg': round(float(np.dot(angles, weights) / weights.sum()), 2),
                    'confidence': round(float(weights.mean()), 3),
                    'source_count': len(tilt_entries),
                }
            else:
                tooth_data['tilt_3d'] = None
        else:
            tooth_data['tilt_3d'] = None

        merged[tooth_id] = tooth_data
    
    # ⭐ 對稱補全：測量不可靠時參考對側牙
    SYMMETRY_DIFF_THRESHOLD = 0.4  # 差異超過40%就放棄補全
    SYMMETRY_WEIGHT = 0.3           # 對側牙的補全權重
    
    for tooth_id, tooth_data in merged.items():
        mirror_id = SYMMETRIC_PAIRS.get(tooth_id)
        if mirror_id not in merged:
            continue
        mirror_data = merged[mirror_id]
        
        for axis in ['width', 'height', 'depth']:
            dim = tooth_data['dimensions_3d'][axis]
            mir = mirror_data['dimensions_3d'][axis]
            
            # 只在測量不可靠時才觸發
            unreliable = (dim['count'] == 0 or 
                         (dim['count'] == 1) or
                         (dim['std'] is not None and dim['mean'] and dim['std']/dim['mean'] > 0.3))
            
            if not unreliable:
                continue
            if mir['mean'] is None or dim['mean'] is None:
                continue
            
            # 差異太大就放棄
            diff = abs(dim['mean'] - mir['mean']) / max(dim['mean'], mir['mean'])
            if diff > SYMMETRY_DIFF_THRESHOLD:
                continue
            
            # 加權融合（原本值權重更高）
            original_weight = 1.0 - SYMMETRY_WEIGHT
            new_mean = dim['mean'] * original_weight + mir['mean'] * SYMMETRY_WEIGHT
            tooth_data['dimensions_3d'][axis]['mean'] = round(new_mean, 2)
            tooth_data['dimensions_3d'][axis]['symmetry_assisted'] = True

        # ⭐ 傾斜角對稱補全
        if tooth_id in ANTERIOR_TEETH:
            my_tilt     = tooth_data.get('tilt_3d')
            mirror_tilt = mirror_data.get('tilt_3d')
            if my_tilt is None and mirror_tilt and mirror_tilt.get('confidence', 0) > 0.3:
                tooth_data['tilt_3d'] = {
                    'tilt_deg':       mirror_tilt['tilt_deg'],
                    'confidence':     mirror_tilt['confidence'] * 0.7,
                    'source_count':   0,
                    'symmetry_assisted': True,
                }
    
    return merged


def calculate_3d_dimensions(measurements_list, tooth_id=None):
    """從多視角測量計算 3D 尺寸，視角加權 + 橢圓angle修正 + 異常值過濾"""
    
    # 取牙齒末位數字查加權表
    tooth_digit = (tooth_id % 10) if tooth_id else 5
    weights_def = VIEW_AXIS_WEIGHTS.get(tooth_digit, VIEW_AXIS_WEIGHTS[5])
    
    dimensions = {
        'width':  {'values': [], 'views': [], 'weights': []},
        'height': {'values': [], 'views': [], 'weights': []},
        'depth':  {'values': [], 'views': [], 'weights': []},
    }
    
    for m in measurements_list:
        mm      = m.get('millimeters', {})
        view    = m.get('view', 'unknown')
        solidity = m.get('solidity', 1.0)
        pixels  = m.get('pixels', {})
        area_px = pixels.get('area', 999)
        angle   = pixels.get('ellipse_angle', 0.0)  # 橢圓傾斜角
        
        if solidity < 0.45:
            continue
        if area_px < 100:
            continue
        
        # ⭐ ellipse_angle修正：傾斜角越大，長短軸的測量越不準
        # 角度偏離90°/0°越多，代表橢圓擬合的軸向誤差越大
        # 修正係數：完全正立=1.0，45度傾斜=0.7
        angle_deviation = min(abs(angle % 90), 90 - abs(angle % 90)) / 45.0
        angle_correction = 1.0 - 0.3 * angle_deviation  # 最多降30%可信度
        
        # 決定視角對應的加權key
        if view == 'front':
            w_width  = weights_def['front_width']
            w_height = weights_def['front_height']
            w_depth  = 0.0
        elif view in ['upper', 'lower']:
            w_width  = weights_def['occlusal_width']
            w_height = 0.0
            w_depth  = weights_def['occlusal_width']
        elif view in ['left', 'right']:
            w_width  = 0.0
            w_height = weights_def['side_height']
            w_depth  = weights_def['side_depth']
        else:
            w_width = w_height = w_depth = 0.5
        
        # 加入各維度
        for axis, base_weight in [('width', w_width), ('height', w_height), ('depth', w_depth)]:
            val = mm.get(axis)
            if val is not None and val > 0 and base_weight > 0:
                final_weight = base_weight * angle_correction * solidity
                dimensions[axis]['values'].append(val)
                dimensions[axis]['views'].append(view)
                dimensions[axis]['weights'].append(final_weight)
    
    def filter_outliers(values, weights):
        if not values:
            return values, weights
        median = np.median(values)
        keep = [(v, w) for v, w in zip(values, weights) if 0 < v <= median * 3.0]
        if not keep:
            return values, weights
        values, weights = zip(*keep)
        values, weights = list(values), list(weights)
        if len(values) < 3:
            return values, weights
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        keep2 = [(v, w) for v, w in zip(values, weights) if q1 - 1.5*iqr <= v <= q3 + 1.5*iqr]
        if not keep2:
            return values, weights
        values, weights = zip(*keep2)
        return list(values), list(weights)
    
    result = {}
    for axis, data in dimensions.items():
        if data['values']:
            fv, fw = filter_outliers(data['values'], data['weights'])
            fw_arr = np.array(fw)
            fv_arr = np.array(fv)
            w_sum = fw_arr.sum()
            if w_sum > 0:
                weighted_mean = float(np.dot(fv_arr, fw_arr) / w_sum)
            else:
                weighted_mean = float(np.mean(fv_arr))
            result[axis] = {
                'values': data['values'],
                'views':  data['views'],
                'count':  len(fv),
                'mean':   round(weighted_mean, 2),
                'std':    round(float(np.std(fv_arr)), 2),
                'min':    round(float(fv_arr.min()), 2),
                'max':    round(float(fv_arr.max()), 2),
            }
        else:
            result[axis] = {
                'values': [], 'views': [],
                'count': 0, 'mean': None,
                'std': None, 'min': None, 'max': None
            }
    
    return result


def classify_by_position(merged_data):
    """按位置分類牙齒"""
    
    classified = {
        'anterior': [],
        'premolar': [],
        'molar': [],
        'wisdom': []
    }
    
    for tooth_id, data in merged_data.items():
        position = data['position']
        classified[position].append(tooth_id)
    
    return classified


def identify_suspicious_detections(merged_data):
    """識別可疑的檢測結果"""
    
    suspicious = {
        'low_confidence': [],
        'insufficient_views': [],
        'might_be_false': []
    }
    
    for tooth_id, data in merged_data.items():
        warning = data.get('warning')
        confidence = data['confidence']
        
        if warning == 'low_confidence':
            suspicious['low_confidence'].append(tooth_id)
        
        if warning == 'insufficient_views':
            suspicious['insufficient_views'].append(tooth_id)
        
        if confidence < 0.3 and data['num_views'] == 1:
            suspicious['might_be_false'].append(tooth_id)
    
    return suspicious


# ==================== 主流程 ====================

def main():
    """主流程"""
    
    print(f"\n{'='*70}")
    print(f"📂 處理 real_teeth 資料夾")
    print(f"{'='*70}\n")
    
    all_measurements = {}
    all_detected_by_view = {}
    
    for photo_name, view in VIEW_MAPPING.items():
        photo_path = REAL_TEETH_DIR / photo_name
        
        if not photo_path.exists():
            print(f"  ⚠️  未找到: {photo_name}")
            continue
        
        mask, measurements, detected_list = analyze_single_photo(photo_path, view)
        all_measurements[photo_name] = measurements
        all_detected_by_view[photo_name] = detected_list
    
    merged = merge_multiview_detections(all_measurements)
    # ⭐ 咬合面牙弓曲率分析
    def analyze_arch_curvature(all_measurements, view_name):
        """從咬合面視角提取牙弓曲率"""
        photo_key = [k for k in all_measurements.keys() if view_name in k]
        if not photo_key:
            return None
        
        teeth_data = all_measurements[photo_key[0]]
        if len(teeth_data) < 4:
            return None
        
        # 收集每顆牙中心點（pixel座標）
        centers = []
        tooth_ids = []
        for tooth_id, data in teeth_data.items():
            # 從pixels資訊重建中心（這裡用major/minor axis的比例估算）
            tooth_ids.append(tooth_id)
            centers.append(tooth_id)  # placeholder，實際需要mask中心
        
        return {'tooth_count': len(teeth_data), 'view': view_name}
    
    upper_arch = analyze_arch_curvature(all_measurements, 'upper')
    lower_arch = analyze_arch_curvature(all_measurements, 'lower')
    classified = classify_by_position(merged)
    suspicious = identify_suspicious_detections(merged)
    
    # 統計
    total_detected = len(merged)
    reliable = [t for t, d in merged.items() if d.get('warning') is None]
    questionable = [t for t, d in merged.items() if d.get('warning') is not None]
    
    print(f"\n📊 檢測統計：")
    print(f"  總計: {total_detected} 顆")
    print(f"    可靠: {len(reliable)} 顆")
    print(f"    可疑: {len(questionable)} 顆")
    print(f"\n  按位置分類:")
    print(f"    前牙: {len(classified['anterior'])} 顆 → {sorted(classified['anterior'])}")
    print(f"    前臼齒: {len(classified['premolar'])} 顆 → {sorted(classified['premolar'])}")
    print(f"    臼齒: {len(classified['molar'])} 顆 → {sorted(classified['molar'])}")
    print(f"    智齒: {len(classified['wisdom'])} 顆 → {sorted(classified['wisdom'])}")
    
    # 顯示 3D 測量統計
    print(f"\n📐 3D 測量示例（前 3 顆）:")
    for i, tooth_id in enumerate(sorted(reliable)[:3]):
        dims = merged[tooth_id]['dimensions_3d']
        print(f"  牙齒 #{tooth_id}:")
        for axis in ['width', 'height', 'depth']:
            if dims[axis]['count'] > 0:
                mean = dims[axis]['mean']
                std = dims[axis]['std']
                count = dims[axis]['count']
                views = dims[axis]['views']
                print(f"    {axis:6s}: {mean:5.1f}±{std:4.1f} mm ({count} 測量, 來自 {views})")
    
    if suspicious['might_be_false']:
        print(f"\n  ⚠️  可能誤檢: {sorted(suspicious['might_be_false'])}")
    
    # 計算缺牙
    all_model_teeth = set(range(11, 19)) | set(range(21, 29)) | \
                      set(range(31, 39)) | set(range(41, 49))
    detected_teeth = set(merged.keys())
    never_detected = sorted(all_model_teeth - detected_teeth)
    
    print(f"\n📭 照片中從未出現的牙齒: {len(never_detected)} 顆")
    if never_detected:
        print(f"  {never_detected}")
    
    # 保存結果
    result = {
        'total_detected': total_detected,
        'reliable_count': len(reliable),
        'questionable_count': len(questionable),
        'classified': {k: sorted(v) for k, v in classified.items()},
        'suspicious': {k: sorted(v) for k, v in suspicious.items()},
        'teeth': merged,
        'by_view': all_detected_by_view,
        'measurement_definition': MEASUREMENT_DEFINITION,
        'never_detected': never_detected,
        'detected_teeth': sorted(detected_teeth)
    }
    
    result_file = OUTPUT_DIR / "real_teeth_analysis.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 已保存: {result_file}")
    
    print(f"\n{'='*70}")
    print("✅ 分析完成！")
    print(f"{'='*70}")
    
    print(f"\n💡 下一步:")
    print(f"  1. 執行 create_personalized_3d_real.py 生成客製化模型")


if __name__ == "__main__":
    main()