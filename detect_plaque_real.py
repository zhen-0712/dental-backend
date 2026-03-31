#!/usr/bin/env python3
"""
真实牙菌斑检测系统 - 終極優化版
"""

import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from segmentanytooth import predict as segment_teeth_predict

# ==================== 配置 ====================

ORIGINAL_PHOTOS_DIR = Path("./real_teeth_processed")  # ⭐ 使用增強照片
PLAQUE_PHOTOS_DIR = Path("./plaque_photos")

OUTPUT_DIR = Path("./plaque_detection_real")
OUTPUT_DIR.mkdir(exist_ok=True)

WEIGHT_DIR = Path("/home/Zhen/projects/SegmentAnyTooth/weight")

PHOTO_MAPPING = {
    'front': {
        'original': 'test_front.jpg',
        'plaque': 'test_front_result_r.png',
        'view': 'front'
    },
    'left': {
        'original': 'test_left.jpg',
        'plaque': 'test_left_result_r.png',
        'view': 'left'
    },
    'right': {
        'original': 'test_right.jpg',
        'plaque': 'test_right_result_r.png',
        'view': 'right'
    }
}

# ⭐⭐⭐ 檢測參數
MIN_PLAQUE_AREA = 5  # 更低
IOU_THRESHOLD = 0.00001  # 極低
DILATE_PLAQUE = True
DILATE_KERNEL_SIZE = 25  # ⭐ 增加膨脹（從 15 → 25）

print("="*70)
print("🦷 真实牙菌斑检测系统 - 終極優化版")
print("="*70)
print(f"  • 照片來源: {ORIGINAL_PHOTOS_DIR}")
print(f"  • IOU 閾值: {IOU_THRESHOLD}")
print(f"  • 膨脹核: {DILATE_KERNEL_SIZE}px")

# ==================== 工具函數 ====================

def detect_red_plaque(plaque_image_path):
    """檢測紅色牙菌斑"""
    img = cv2.imread(str(plaque_image_path))
    if img is None:
        return [], None, None
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # ⭐ 更寬鬆的紅色範圍
    mask1 = cv2.inRange(hsv, np.array([0, 20, 20]), np.array([15, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([155, 20, 20]), np.array([180, 255, 255]))
    
    combined_mask = cv2.bitwise_or(mask1, mask2)
    
    # 形態學
    kernel = np.ones((5, 5), np.uint8)  # ⭐ 更大的核
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # ⭐⭐⭐ 大幅膨脹
    if DILATE_PLAQUE:
        dilate_kernel = np.ones((DILATE_KERNEL_SIZE, DILATE_KERNEL_SIZE), np.uint8)
        combined_mask = cv2.dilate(combined_mask, dilate_kernel, iterations=1)
    
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
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        plaque_regions.append({
            'contour': contour,
            'center': (cx, cy),
            'area': area,
            'bbox': (x, y, w, h)
        })
    
    return plaque_regions, combined_mask, img.shape[:2]

def match_plaque_to_teeth(plaque_mask, teeth_results, target_shape):
    """匹配牙菌斑到牙齒"""
    
    if plaque_mask.shape != target_shape:
        print(f"      🔧 調整: {plaque_mask.shape} -> {target_shape}")
        plaque_mask = cv2.resize(plaque_mask, (target_shape[1], target_shape[0]), 
                                  interpolation=cv2.INTER_NEAREST)
    
    plaque_teeth = {}
    
    for tooth_info in teeth_results:
        tooth_id = tooth_info['tooth_id']
        tooth_mask = tooth_info['mask']
        
        overlap = np.logical_and(plaque_mask > 0, tooth_mask > 0)
        overlap_area = overlap.sum()
        
        tooth_area = tooth_mask.sum()
        overlap_ratio = overlap_area / tooth_area if tooth_area > 0 else 0
        
        # ⭐⭐⭐ 只要有任何像素重疊就接受
        if overlap_area > 0:
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

print("\n📸 步驟 1: 處理照片")

all_results = {}
all_affected_teeth = defaultdict(lambda: {
    'views': [],
    'total_overlap_area': 0,
    'regions': set(),
    'confidence': 0.0,
    'position_3d': set()
})

for view_name, paths in PHOTO_MAPPING.items():
    print(f"\n  處理: {view_name} 視角")
    
    original_path = ORIGINAL_PHOTOS_DIR / paths['original']
    plaque_path = PLAQUE_PHOTOS_DIR / paths['plaque']
    
    if not original_path.exists():
        print(f"    ⚠️  未找到: {original_path}")
        continue
    
    if not plaque_path.exists():
        print(f"    ⚠️  未找到: {plaque_path}")
        continue
    
    # 識別牙齒
    print(f"    🦷 識別牙齒...", end='')
    
    try:
        mask = segment_teeth_predict(
            image_path=str(original_path),
            view=paths['view'],
            weight_dir=str(WEIGHT_DIR),
            sam_batch_size=10
        )
        
        if mask is None or mask.size == 0 or len(mask.shape) == 0:
            print(f" ✗ 無效")
            teeth_results = []
        else:
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
            
            print(f" ✓ {len(teeth_results)} 顆")
        
        original_img = cv2.imread(str(original_path))
        target_shape = mask.shape if mask is not None else original_img.shape[:2]
        
    except Exception as e:
        print(f" ✗ {e}")
        teeth_results = []
        original_img = cv2.imread(str(original_path))
        target_shape = original_img.shape[:2]
    
    # 檢測牙菌斑
    print(f"    🔍 檢測牙菌斑...", end='')
    plaque_regions, plaque_mask, plaque_shape = detect_red_plaque(plaque_path)
    print(f" ✓ {len(plaque_regions)} 區")
    
    # 匹配
    plaque_teeth = {}
    if plaque_mask is not None and len(teeth_results) > 0:
        plaque_teeth = match_plaque_to_teeth(plaque_mask, teeth_results, target_shape)
        print(f"    ✅ 匹配: {len(plaque_teeth)} 顆")
        
        for tooth_id, info in plaque_teeth.items():
            print(f"      #{tooth_id}: {info['overlap_ratio']*100:.1f}%, {info['region']}")
            
            all_affected_teeth[tooth_id]['views'].append(view_name)
            all_affected_teeth[tooth_id]['total_overlap_area'] += info['overlap_area']
            all_affected_teeth[tooth_id]['regions'].add(info['region'])
            all_affected_teeth[tooth_id]['confidence'] += info['overlap_ratio']
            
            if info['region'] == 'lower':
                all_affected_teeth[tooth_id]['position_3d'].add('gingival_margin')
            elif info['region'] == 'upper':
                all_affected_teeth[tooth_id]['position_3d'].add('incisal_edge')
            elif info['region'] == 'full':
                all_affected_teeth[tooth_id]['position_3d'].add('full_tooth')
            else:
                all_affected_teeth[tooth_id]['position_3d'].add('middle_third')
    elif len(teeth_results) == 0:
        print(f"    ⚠️  無法匹配（無牙齒）")
    
    # 標註圖
    print(f"    🎨 標註...", end='')
    annotated = original_img.copy()
    
    if plaque_mask is not None:
        plaque_mask_resized = cv2.resize(plaque_mask, 
                                         (original_img.shape[1], original_img.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
        
        red_overlay = np.zeros_like(original_img)
        red_overlay[plaque_mask_resized > 0] = [0, 0, 255]
        annotated = cv2.addWeighted(annotated, 0.7, red_overlay, 0.3, 0)
    
    for tooth_info in teeth_results:
        tooth_mask = tooth_info['mask']
        tooth_id = tooth_info['tooth_id']
        
        tooth_mask_resized = cv2.resize(tooth_mask.astype(np.uint8), 
                                        (original_img.shape[1], original_img.shape[0]), 
                                        interpolation=cv2.INTER_NEAREST)
        
        contours, _ = cv2.findContours(tooth_mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        color = (0, 0, 255) if tooth_id in plaque_teeth else (0, 255, 0)
        thickness = 3 if tooth_id in plaque_teeth else 2
        
        cv2.drawContours(annotated, contours, -1, color, thickness)
        
        if len(contours) > 0:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(annotated, f"#{tooth_id}", (cx-20, cy), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    output_annotated = OUTPUT_DIR / f"annotated_{view_name}.jpg"
    cv2.imwrite(str(output_annotated), annotated)
    print(f" ✓")
    
    all_results[view_name] = {
        'view_type': paths['view'],
        'num_plaque_regions': len(plaque_regions),
        'num_teeth_detected': len(teeth_results),
        'affected_teeth': {str(k): {
            'overlap_ratio': v['overlap_ratio'],
            'region': v['region'],
            'overlap_area': v['overlap_area']
        } for k, v in plaque_teeth.items()}
    }

# ==================== 報告 ====================

print("\n📋 步驟 2: 生成報告")

for tooth_id, info in all_affected_teeth.items():
    num_views = len(info['views'])
    info['confidence'] = info['confidence'] / num_views if num_views > 0 else 0.0
    info['regions'] = list(info['regions'])
    info['position_3d'] = list(info['position_3d'])

affected_teeth_serializable = {
    str(k): {
        'views': v['views'],
        'num_views': len(v['views']),
        'total_overlap_area': int(v['total_overlap_area']),
        'regions': v['regions'],
        'position_3d': v['position_3d'],
        'confidence': float(v['confidence'])
    } for k, v in all_affected_teeth.items()
}

comprehensive_report = {
    'version': 'ultimate_v4',
    'source': 'Enhanced photos + dilated plaque',
    'parameters': {
        'iou_threshold': IOU_THRESHOLD,
        'min_plaque_area': MIN_PLAQUE_AREA,
        'dilate_kernel_size': DILATE_KERNEL_SIZE
    },
    'summary': {
        'total_affected_teeth': len(all_affected_teeth),
        'total_views_analyzed': len(all_results)
    },
    'affected_teeth': affected_teeth_serializable,
    'by_view': all_results
}

with open(OUTPUT_DIR / "plaque_detection_report.json", 'w', encoding='utf-8') as f:
    json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)

print(f"  ✓ {len(all_affected_teeth)} 顆牙齒有菌斑")

for tooth_id, info in sorted(affected_teeth_serializable.items()):
    print(f"    #{tooth_id}: {info['views']}, 信心度={info['confidence']:.2f}")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)