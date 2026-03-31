#!/usr/bin/env python3
"""
牙菌斑檢測系統 - 改進螢光檢測版
使用更好的螢光紅色檢測技術（基於 plaque_detection_improved.py）
"""

import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from segmentanytooth import predict as segment_teeth_predict

# ==================== 配置 ====================

NORMAL_TEETH_DIR = Path("/home/Zhen/projects/InstantMesh/inputs/normal_teeth")
OUTPUT_DIR = Path("./plaque_detection_final_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

SIMULATED_DIR = OUTPUT_DIR / "simulated_images"
SIMULATED_DIR.mkdir(exist_ok=True)

WEIGHT_DIR = Path("/home/Zhen/projects/SegmentAnyTooth/weight")

# ⭐⭐⭐ 改進的螢光顏色（更接近真實螢光效果）
FLUORESCENT_COLORS = {
    'bright_red': (0, 0, 255),
    'fluorescent_red': (50, 50, 255),
    'fluorescent_pink': (180, 80, 255),
    'hot_pink': (180, 20, 255),
}

# ⭐⭐⭐ 改進的 HSV 檢測範圍（更準確的螢光紅/粉色檢測）
PLAQUE_DETECTION_RANGES = {
    'red_range1': {
        'lower': np.array([0, 120, 150]),
        'upper': np.array([10, 255, 255])
    },
    'red_range2': {
        'lower': np.array([170, 120, 150]),
        'upper': np.array([180, 255, 255])
    },
    'pink_range': {
        'lower': np.array([160, 100, 180]),
        'upper': np.array([175, 255, 255])
    }
}

# 檢測參數
IOU_THRESHOLD = 0.05
MIN_PLAQUE_AREA = 50

# ⭐⭐⭐ 嚴格過濾參數
MIN_CONFIDENCE = 0.20  # 降低閾值以保留更多牙齒
MIN_NUM_VIEWS = 1      # 降低視角要求
MIN_OVERLAP_AREA_FILTER = 100

# 牙菌斑配置
PLAQUE_SIMULATION_CONFIG = {
    'front.png': [
        {'teeth': [11, 12], 'type': 'gap', 'size': 35, 'position_3d': 'anterior_gap', 'intensity': 0.8},
        {'teeth': [13], 'type': 'gingival', 'size': 30, 'position_3d': 'gingival_margin', 'intensity': 0.7},
        {'teeth': [31, 32], 'type': 'gap', 'size': 30, 'position_3d': 'anterior_gap', 'intensity': 0.75},
        {'teeth': [41, 42], 'type': 'gap', 'size': 32, 'position_3d': 'anterior_gap', 'intensity': 0.75},
    ],
    'upper_occlusal.png': [
        {'teeth': [16, 17], 'type': 'occlusal', 'size': 40, 'position_3d': 'occlusal_surface_posterior', 'intensity': 0.85},
        {'teeth': [26, 27], 'type': 'occlusal', 'size': 38, 'position_3d': 'occlusal_surface_posterior', 'intensity': 0.85},
        {'teeth': [14, 15], 'type': 'gap', 'size': 28, 'position_3d': 'premolar_gap', 'intensity': 0.7},
    ],
    'lower_occlusal.png': [
        {'teeth': [36, 37], 'type': 'occlusal', 'size': 38, 'position_3d': 'occlusal_surface_posterior', 'intensity': 0.8},
        {'teeth': [46, 47], 'type': 'occlusal', 'size': 35, 'position_3d': 'occlusal_surface_posterior', 'intensity': 0.8},
    ],
    'left_side.png': [
        {'teeth': [26], 'type': 'gingival', 'size': 32, 'position_3d': 'buccal_gingival', 'intensity': 0.75},
        {'teeth': [36], 'type': 'gingival', 'size': 30, 'position_3d': 'buccal_gingival', 'intensity': 0.7},
    ],
    'right_side.png': [
        {'teeth': [16], 'type': 'gingival', 'size': 32, 'position_3d': 'buccal_gingival', 'intensity': 0.75},
        {'teeth': [46], 'type': 'gingival', 'size': 28, 'position_3d': 'buccal_gingival', 'intensity': 0.7},
    ]
}

VIEW_MAPPING = {
    'front.png': 'front',
    'left_side.png': 'left',
    'right_side.png': 'right',
    'upper_occlusal.png': 'upper',
    'lower_occlusal.png': 'lower'
}

print("="*70)
print("🦷 牙菌斑檢測系統 - 改進螢光檢測版")
print("="*70)
print("\n✨ 特色:")
print("  1. ⭐⭐⭐ 改進的螢光紅色/粉色檢測")
print("  2. ⭐⭐ 多範圍 HSV 檢測")
print("  3. ⭐ 3D 位置標記")
print("  4. 簡化的區域標記（不需精確還原形狀）")

# ==================== 工具函數 ====================

def apply_fluorescent_color(img, mask, color, intensity=0.8):
    """應用螢光顏色"""
    overlay = img.copy()
    overlay[mask > 0] = color
    result = cv2.addWeighted(img, 1 - intensity, overlay, intensity, 0)
    
    highlight = cv2.GaussianBlur(mask.astype(np.uint8) * 255, (15, 15), 0)
    highlight = cv2.cvtColor(highlight, cv2.COLOR_GRAY2BGR)
    highlight = (highlight / 255.0 * 0.3).astype(np.uint8)
    
    result = cv2.add(result, highlight)
    return result

def get_plaque_position_offset(tooth_positions, plaque_type, position_3d):
    """計算牙菌斑位置偏移"""
    offset_map = {
        'anterior_gap': 0,
        'gingival_margin': 0.7,
        'occlusal_surface_posterior': -0.1,
        'premolar_gap': 0,
        'buccal_gingival': 0.6,
    }
    return offset_map.get(position_3d, 0)

def add_simulated_plaque_improved(image_path, view_type, plaque_configs):
    """添加螢光牙菌斑"""
    img = cv2.imread(str(image_path))
    if img is None:
        return None, []
    
    print(f"    使用 SegmentAnyTooth 分析牙齒位置...", end='')
    
    try:
        mask = segment_teeth_predict(
            image_path=str(image_path),
            view=view_type,
            weight_dir=str(WEIGHT_DIR),
            sam_batch_size=10
        )
        print(f" ✓")
    except Exception as e:
        print(f" ✗ 失敗: {e}")
        return img.copy(), []
    
    tooth_positions = {}
    unique_labels = np.unique(mask)
    
    for label in unique_labels:
        if label == 0:
            continue
        
        tooth_mask = (mask == label).astype(np.uint8)
        coords = np.argwhere(tooth_mask > 0)
        
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            center_y = (y_min + y_max) // 2
            center_x = (x_min + x_max) // 2
            
            tooth_positions[int(label)] = {
                'center': (center_x, center_y),
                'bbox': (x_min, y_min, x_max, y_max),
                'coords': coords,
                'height': y_max - y_min
            }
    
    simulated = img.copy()
    plaque_info = []
    
    for config in plaque_configs:
        teeth_ids = config['teeth']
        plaque_type = config['type']
        size = config['size']
        position_3d = config.get('position_3d', 'unknown')
        intensity = config.get('intensity', 0.8)
        
        positions = []
        for tooth_id in teeth_ids:
            if tooth_id in tooth_positions:
                positions.append(tooth_positions[tooth_id])
        
        if not positions:
            continue
        
        if plaque_type == 'gap' and len(positions) >= 2:
            center1 = positions[0]['center']
            center2 = positions[1]['center']
            plaque_center = ((center1[0] + center2[0]) // 2, 
                           (center1[1] + center2[1]) // 2)
            shape = 'ellipse'
            
        elif plaque_type == 'gingival':
            pos = positions[0]
            offset_ratio = get_plaque_position_offset(tooth_positions, plaque_type, position_3d)
            offset_y = int(pos['height'] * offset_ratio)
            plaque_center = (pos['center'][0], pos['center'][1] + offset_y)
            shape = 'irregular'
            
        elif plaque_type == 'occlusal':
            if len(positions) >= 2:
                center1 = positions[0]['center']
                center2 = positions[1]['center']
                plaque_center = ((center1[0] + center2[0]) // 2, 
                               (center1[1] + center2[1]) // 2)
            else:
                pos = positions[0]
                offset_ratio = get_plaque_position_offset(tooth_positions, plaque_type, position_3d)
                offset_y = int(pos['height'] * offset_ratio)
                plaque_center = (pos['center'][0], pos['center'][1] + offset_y)
            shape = 'circle'
        
        else:
            plaque_center = positions[0]['center']
            shape = 'circle'
        
        color = FLUORESCENT_COLORS['fluorescent_red']
        plaque_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        if shape == 'circle':
            cv2.circle(plaque_mask, plaque_center, size, 255, -1)
        elif shape == 'ellipse':
            axes = (size, size // 2)
            angle = np.arctan2(
                positions[1]['center'][1] - positions[0]['center'][1],
                positions[1]['center'][0] - positions[0]['center'][0]
            ) * 180 / np.pi
            cv2.ellipse(plaque_mask, plaque_center, axes, angle, 0, 360, 255, -1)
        elif shape == 'irregular':
            for _ in range(4):
                offset_x = np.random.randint(-size//3, size//3)
                offset_y = np.random.randint(-size//4, size//4)
                center = (plaque_center[0] + offset_x, plaque_center[1] + offset_y)
                radius = np.random.randint(size//2, int(size * 0.8))
                cv2.circle(plaque_mask, center, radius, 255, -1)
        
        simulated = apply_fluorescent_color(simulated, plaque_mask, color, intensity)
        
        plaque_info.append({
            'teeth': teeth_ids,
            'type': plaque_type,
            'position_3d': position_3d,
            'center': plaque_center,
            'size': size,
            'intensity': intensity,
            'color': 'fluorescent_red'
        })
        
        print(f"    ✓ 添加牙菌斑: 牙齒 {teeth_ids}, 類型={plaque_type}, 3D位置={position_3d}")
    
    return simulated, plaque_info

def detect_plaque_improved(image_path):
    """⭐⭐⭐ 改進的螢光檢測（使用多範圍 HSV）"""
    img = cv2.imread(str(image_path))
    if img is None:
        return [], None, {}
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 多範圍檢測
    masks = []
    for range_name, range_params in PLAQUE_DETECTION_RANGES.items():
        mask = cv2.inRange(hsv, range_params['lower'], range_params['upper'])
        masks.append(mask)
    
    combined_mask = masks[0].copy()
    for mask in masks[1:]:
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # 形態學處理
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    
    # 找輪廓
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    plaque_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_PLAQUE_AREA:
            continue
        
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
        
        x, y, w, h = cv2.boundingRect(contour)
        
        plaque_regions.append({
            'contour': contour,
            'center': (cx, cy),
            'area': area,
            'bbox': (x, y, w, h)
        })
    
    debug_info = {
        'total_pixels': int(combined_mask.sum()),
        'num_regions': len(plaque_regions),
        'total_area': sum(r['area'] for r in plaque_regions),
        'detection_ranges_used': list(PLAQUE_DETECTION_RANGES.keys())
    }
    
    return plaque_regions, combined_mask, debug_info

def match_plaque_to_teeth(plaque_mask, teeth_results):
    """匹配牙菌斑到牙齒"""
    plaque_teeth = {}
    
    for tooth_info in teeth_results:
        tooth_id = tooth_info['tooth_id']
        tooth_mask = tooth_info['mask']
        
        overlap = np.logical_and(plaque_mask > 0, tooth_mask > 0)
        overlap_area = overlap.sum()
        
        if overlap_area == 0:
            continue
        
        tooth_area = tooth_mask.sum()
        overlap_ratio = overlap_area / tooth_area if tooth_area > 0 else 0
        
        if overlap_ratio > IOU_THRESHOLD:
            tooth_coords = np.argwhere(tooth_mask > 0)
            plaque_coords = np.argwhere(overlap > 0)
            
            if len(tooth_coords) > 0 and len(plaque_coords) > 0:
                tooth_y_min, tooth_y_max = tooth_coords[:, 0].min(), tooth_coords[:, 0].max()
                tooth_height = tooth_y_max - tooth_y_min
                
                plaque_y_center = plaque_coords[:, 0].mean()
                relative_pos = (plaque_y_center - tooth_y_min) / tooth_height if tooth_height > 0 else 0.5
                
                if relative_pos < 0.3:
                    region = 'upper'
                elif relative_pos > 0.7:
                    region = 'lower'
                else:
                    region = 'middle'
                
                if overlap_ratio > 0.5:
                    region = 'full'
            else:
                region = 'unknown'
            
            plaque_teeth[tooth_id] = {
                'overlap_ratio': float(overlap_ratio),
                'overlap_area': int(overlap_area),
                'tooth_area': int(tooth_area),
                'region': region
            }
    
    return plaque_teeth

# ==================== 主流程 ====================

print("\n📸 步驟 1: 生成模擬牙菌斑照片")

simulation_results = {}

for filename, view_type in VIEW_MAPPING.items():
    img_path = NORMAL_TEETH_DIR / filename
    
    if not img_path.exists():
        print(f"\n  ⚠️  未找到圖像: {filename}")
        continue
    
    print(f"\n  處理: {filename} (視角: {view_type})")
    
    plaque_configs = PLAQUE_SIMULATION_CONFIG.get(filename, [])
    
    if not plaque_configs:
        print(f"    ⚠️  無模擬配置，跳過")
        continue
    
    simulated_img, plaque_info = add_simulated_plaque_improved(
        img_path, view_type, plaque_configs
    )
    
    if simulated_img is None:
        continue
    
    output_path = SIMULATED_DIR / f"simulated_{filename}"
    cv2.imwrite(str(output_path), simulated_img)
    
    simulation_results[filename] = {
        'view_type': view_type,
        'original_path': str(img_path),
        'simulated_path': str(output_path),
        'plaque_info': plaque_info
    }
    
    print(f"    ✓ 已保存: {output_path.name}")

print(f"\n  ✓ 共生成 {len(simulation_results)} 張模擬照片")

# ==================== 步驟 2: 檢測 ====================

print("\n🔍 步驟 2: 使用改進的螢光檢測")

final_results = {}
all_affected_teeth = defaultdict(lambda: {
    'views': [],
    'total_overlap_area': 0,
    'regions': set(),
    'confidence': 0.0,
    'position_3d': set()
})

for filename, sim_info in simulation_results.items():
    simulated_path = Path(sim_info['simulated_path'])
    view_type = sim_info['view_type']
    
    print(f"\n  分析: {filename}")
    
    plaque_regions, plaque_mask, debug_info = detect_plaque_improved(simulated_path)
    
    print(f"    ✓ 檢測到 {debug_info['num_regions']} 個區域")
    print(f"       使用範圍: {debug_info['detection_ranges_used']}")
    
    original_path = Path(sim_info['original_path'])
    
    try:
        mask = segment_teeth_predict(
            image_path=str(original_path),
            view=view_type,
            weight_dir=str(WEIGHT_DIR),
            sam_batch_size=10
        )
        
        teeth_results = []
        unique_labels = np.unique(mask)
        
        for label in unique_labels:
            if label == 0:
                continue
            tooth_mask = (mask == label).astype(np.uint8)
            teeth_results.append({
                'tooth_id': int(label),
                'mask': tooth_mask
            })
        
        print(f"    ✓ 檢測到 {len(teeth_results)} 顆牙齒")
    except Exception as e:
        print(f"    ✗ 牙齒分割失敗: {e}")
        continue
    
    plaque_teeth = {}
    
    if plaque_mask is not None and len(teeth_results) > 0:
        plaque_teeth = match_plaque_to_teeth(plaque_mask, teeth_results)
        print(f"    ✓ 匹配到 {len(plaque_teeth)} 顆有菌斑的牙齒")
        
        for tooth_id, info in plaque_teeth.items():
            print(f"      牙齒 #{tooth_id}: {info['overlap_ratio']*100:.1f}%, 區域={info['region']}")
            
            position_3d = 'unknown'
            for plaque in sim_info['plaque_info']:
                if tooth_id in plaque['teeth']:
                    position_3d = plaque.get('position_3d', 'unknown')
                    break
            
            all_affected_teeth[tooth_id]['views'].append(filename.replace('.png', ''))
            all_affected_teeth[tooth_id]['total_overlap_area'] += info['overlap_area']
            all_affected_teeth[tooth_id]['regions'].add(info['region'])
            all_affected_teeth[tooth_id]['position_3d'].add(position_3d)
            all_affected_teeth[tooth_id]['confidence'] += info['overlap_ratio']
    
    img = cv2.imread(str(simulated_path))
    img_annotated = img.copy()
    
    for tooth_info in teeth_results:
        tooth_mask = tooth_info['mask']
        tooth_id = tooth_info['tooth_id']
        
        contours, _ = cv2.findContours(
            tooth_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if tooth_id in plaque_teeth:
            color = (0, 0, 255)
            thickness = 3
        else:
            color = (0, 255, 0)
            thickness = 2
        
        cv2.drawContours(img_annotated, contours, -1, color, thickness)
        
        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(img_annotated, f"#{tooth_id}", 
                           (cx-15, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    output_annotated = OUTPUT_DIR / f"annotated_{filename.replace('.png', '.jpg')}"
    cv2.imwrite(str(output_annotated), img_annotated)
    print(f"    ✓ 已保存標註圖: {output_annotated.name}")
    
    final_results[filename.replace('.png', '')] = {
        'view_type': view_type,
        'num_plaque_regions': len(plaque_regions),
        'num_teeth_detected': len(teeth_results),
        'affected_teeth': {
            str(k): {
                'overlap_ratio': v['overlap_ratio'],
                'region': v['region'],
                'position_3d': position_3d,
                'overlap_area': v['overlap_area']
            }
            for k, v in plaque_teeth.items()
        }
    }

# ==================== 步驟 3: 生成報告 ====================

print("\n📋 步驟 3: 生成報告")

for tooth_id, info in all_affected_teeth.items():
    num_views = len(info['views'])
    info['confidence'] = info['confidence'] / num_views if num_views > 0 else 0.0
    info['regions'] = list(info['regions'])
    info['position_3d'] = list(info['position_3d'])

affected_teeth_serializable = {}
for k, v in all_affected_teeth.items():
    affected_teeth_serializable[str(k)] = {
        'views': v['views'],
        'num_views': len(v['views']),
        'total_overlap_area': int(v['total_overlap_area']),
        'regions': v['regions'],
        'position_3d': v['position_3d'],
        'confidence': float(v['confidence'])
    }

comprehensive_report_v3 = {
    'version': 'improved_fluorescent',
    'improvements': [
        'Enhanced fluorescent red/pink detection',
        'Multi-range HSV detection',
        '3D position marking',
        'Simplified region marking'
    ],
    'detection_ranges': {
        k: {
            'lower': v['lower'].tolist(),
            'upper': v['upper'].tolist()
        }
        for k, v in PLAQUE_DETECTION_RANGES.items()
    },
    'summary': {
        'total_affected_teeth': len(all_affected_teeth),
        'total_views_analyzed': len(final_results)
    },
    'affected_teeth': affected_teeth_serializable,
    'by_view': final_results
}

with open(OUTPUT_DIR / "comprehensive_plaque_report_v3.json", 'w', encoding='utf-8') as f:
    json.dump(comprehensive_report_v3, f, indent=2, ensure_ascii=False)

print(f"  ✓ 檢測到 {len(all_affected_teeth)} 顆牙齒有菌斑")

# ==================== 步驟 4: 嚴格過濾 ====================

print("\n🔍 步驟 4: 應用嚴格過濾")

filtered_teeth = {}
filtered_out = {}

for tooth_id, info in affected_teeth_serializable.items():
    confidence = info['confidence']
    num_views = info['num_views']
    total_area = info['total_overlap_area']
    
    reasons = []
    passed = True
    
    if confidence < MIN_CONFIDENCE:
        reasons.append(f"信心度不足({confidence:.2f} < {MIN_CONFIDENCE})")
        passed = False
    
    if num_views < MIN_NUM_VIEWS:
        reasons.append(f"視角數不足({num_views} < {MIN_NUM_VIEWS})")
        passed = False
    
    if total_area < MIN_OVERLAP_AREA_FILTER:
        reasons.append(f"重疊面積不足({total_area} < {MIN_OVERLAP_AREA_FILTER})")
        passed = False
    
    if passed:
        filtered_teeth[tooth_id] = info
        print(f"  ✅ 牙齒 #{tooth_id}: 信心度 {confidence:.2f}, 視角 {num_views}")
    else:
        filtered_out[tooth_id] = {'info': info, 'reasons': reasons}

print(f"\n  📊 過濾結果: 保留 {len(filtered_teeth)} 顆，過濾 {len(filtered_out)} 顆")

comprehensive_report_v4 = {
    'version': 'strict_v4',
    'based_on': 'improved_fluorescent',
    'strict_filters': {
        'min_confidence': MIN_CONFIDENCE,
        'min_num_views': MIN_NUM_VIEWS,
        'min_overlap_area': MIN_OVERLAP_AREA_FILTER
    },
    'summary': {
        'total_detected_v3': len(all_affected_teeth),
        'after_strict_filtering': len(filtered_teeth),
        'filtered_out': len(filtered_out)
    },
    'affected_teeth': filtered_teeth,
    'filtered_out_teeth': {
        str(k): {
            'confidence': v['info']['confidence'],
            'num_views': v['info']['num_views'],
            'total_overlap_area': v['info']['total_overlap_area'],
            'reasons': v['reasons']
        }
        for k, v in filtered_out.items()
    }
}

with open(OUTPUT_DIR / "comprehensive_plaque_report_v4_strict.json", 'w', encoding='utf-8') as f:
    json.dump(comprehensive_report_v4, f, indent=2, ensure_ascii=False)

print(f"  ✓ v4 報告已保存")

print("\n" + "="*70)
print("✅ 檢測完成！")
print("="*70)

print(f"\n📊 統計:")
print(f"  v3: {len(all_affected_teeth)} 顆")
print(f"  v4: {len(filtered_teeth)} 顆 ⭐")

if len(filtered_teeth) > 0:
    print(f"\n🦷 檢測到的牙齒:")
    for tooth_id, info in sorted(filtered_teeth.items()):
        print(f"  #{tooth_id}: {info['regions']}, 3D位置={info['position_3d']}")

print(f"\n📁 輸出檔案:")
print(f"  • comprehensive_plaque_report_v3.json - 原始報告")
print(f"  • comprehensive_plaque_report_v4_strict.json - 過濾報告 ⭐")

print("\n" + "="*70)