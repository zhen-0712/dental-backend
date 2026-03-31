#!/usr/bin/env python3
"""
牙菌斑 3D 整合系統 - 修正版
確保只標記檢測到的牙齒
"""

import cv2
import numpy as np
import json
import trimesh
from pathlib import Path

# ==================== 配置 ====================

PLAQUE_REPORT = Path("./plaque_detection_real/plaque_detection_report.json")
MODEL_FILE = Path("./personalized_3d_models_real/custom_real_teeth.obj")

SEGMENTATION_DIR = Path("./models")
UPPER_SEG_FILE = SEGMENTATION_DIR / "1MWJLE4X_upper.json"
LOWER_SEG_FILE = SEGMENTATION_DIR / "01J9K9S6_lower.json"

OUTPUT_DIR = Path("./plaque_3d_real")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("🦷 牙菌斑 3D 整合系統 - 修正版")
print("="*70)

# ==================== 步驟 1：載入報告 ====================

print("\n📊 步驟 1: 載入檢測報告")

try:
    with open(PLAQUE_REPORT, 'r', encoding='utf-8') as f:
        plaque_report = json.load(f)
    
    affected_teeth = plaque_report.get('affected_teeth', {})
    
    print(f"  ✓ 載入: {len(affected_teeth)} 顆牙齒")
    
    if len(affected_teeth) == 0:
        print(f"\n  ⚠️  沒有檢測到牙菌斑！將生成白色模型。")
    else:
        for tooth_id, info in affected_teeth.items():
            print(f"    #{tooth_id}: {info['position_3d']}, 信心度={info['confidence']:.2f}")
        
except Exception as e:
    print(f"  ❌ 失敗: {e}")
    exit(1)

# ==================== 步驟 2：載入分割標籤 ====================

print("\n📦 步驟 2: 載入分割標籤")

try:
    with open(UPPER_SEG_FILE, 'r') as f:
        upper_seg_data = json.load(f)
        upper_seg_labels = np.array(upper_seg_data["labels"], dtype=np.int32)
    
    with open(LOWER_SEG_FILE, 'r') as f:
        lower_seg_data = json.load(f)
        lower_seg_labels = np.array(lower_seg_data["labels"], dtype=np.int32)
    
    combined_seg_labels = np.concatenate([upper_seg_labels, lower_seg_labels])
    
    print(f"  ✓ 載入: {len(combined_seg_labels)} 個標籤")
    
except Exception as e:
    print(f"  ❌ 失敗: {e}")
    exit(1)

# ==================== 步驟 3：載入 3D 模型 ====================

print("\n🦷 步驟 3: 載入 3D 模型")

try:
    mesh = trimesh.load(str(MODEL_FILE))
    print(f"  ✓ 載入: {len(mesh.vertices)} 個頂點, {len(mesh.faces)} 個面")
    
except Exception as e:
    print(f"  ❌ 失敗: {e}")
    exit(1)

# ==================== 步驟 4：修復頂點數 ====================

print("\n🔧 步驟 4: 檢查頂點數")

num_vertices = len(mesh.vertices)
num_labels = len(combined_seg_labels)

if num_vertices != num_labels:
    print(f"  ⚠️  不匹配: 頂點={num_vertices}, 標籤={num_labels}")
    
    if num_labels > num_vertices:
        combined_seg_labels = combined_seg_labels[:num_vertices]
        print(f"  ✅ 已截斷標籤")
    else:
        padding = np.zeros(num_vertices - num_labels, dtype=np.int32)
        combined_seg_labels = np.concatenate([combined_seg_labels, padding])
        print(f"  ✅ 已填充標籤")
else:
    print(f"  ✅ 完美匹配")

# ==================== 步驟 5：標記牙菌斑（修正版）====================

print("\n🎨 步驟 5: 標記牙菌斑（修正版）")

# ⭐⭐⭐ 初始化為白色（RGBA）
vertex_colors = np.ones((num_vertices, 4), dtype=np.uint8) * 255

# 純紅色
RED_COLOR = np.array([255, 0, 0, 255], dtype=np.uint8)

marked_count = 0
total_marked_vertices = 0

if len(affected_teeth) == 0:
    print("  ⚠️  沒有牙齒需要標記")
else:
    for tooth_id_str, info in affected_teeth.items():
        tooth_id = int(tooth_id_str)
        
        # 找到牙齒的所有頂點
        tooth_mask = (combined_seg_labels == tooth_id)
        tooth_indices = np.where(tooth_mask)[0]
        
        valid_indices = tooth_indices[tooth_indices < num_vertices]
        
        if len(valid_indices) == 0:
            print(f"    ⚠️  未找到牙齒 #{tooth_id}")
            continue
        
        tooth_vertices = mesh.vertices[valid_indices]
        position_3d = info.get('position_3d', [])
        regions = info.get('regions', [])
        confidence = info.get('confidence', 0.0)
        
        # 根據區域決定標記範圍
        position_str = ', '.join(position_3d) if position_3d else 'unknown'
        region_str = ', '.join(regions) if regions else 'unknown'
        
        if 'lower' in regions or 'gingival' in position_str:
            # 下半部（牙齦）
            z_coords = tooth_vertices[:, 2]
            z_threshold = np.percentile(z_coords, 50)
            lower_mask = z_coords < z_threshold
            colored_indices = valid_indices[lower_mask]
            mark_str = '下半部（牙齦）'
            
        elif 'upper' in regions or 'incisal' in position_str:
            # 上半部（切緣）
            z_coords = tooth_vertices[:, 2]
            z_threshold = np.percentile(z_coords, 50)
            upper_mask = z_coords > z_threshold
            colored_indices = valid_indices[upper_mask]
            mark_str = '上半部（切緣）'
            
        elif 'full' in regions:
            # 整顆
            colored_indices = valid_indices
            mark_str = '整顆'
            
        elif 'middle' in regions:
            # 中間 1/3
            z_coords = tooth_vertices[:, 2]
            z_min = np.percentile(z_coords, 33)
            z_max = np.percentile(z_coords, 67)
            middle_mask = (z_coords >= z_min) & (z_coords <= z_max)
            colored_indices = valid_indices[middle_mask]
            mark_str = '中間 1/3'
            
        else:
            # 默認：整顆
            colored_indices = valid_indices
            mark_str = '整顆（默認）'
        
        # ⭐ 應用純紅色（確保只標記這些頂點）
        vertex_colors[colored_indices] = RED_COLOR
        marked_count += 1
        total_marked_vertices += len(colored_indices)
        
        print(f"    ✅ #{tooth_id} ({mark_str}): {len(colored_indices)} 個頂點, 信心度={confidence:.2f}")

print(f"\n  ✅ 共標記 {marked_count} 顆牙齒，{total_marked_vertices} 個頂點")

# ⭐⭐⭐ 驗證：檢查有多少頂點是紅色的
red_vertices = np.all(vertex_colors[:, :3] == [255, 0, 0], axis=1).sum()
white_vertices = np.all(vertex_colors[:, :3] == [255, 255, 255], axis=1).sum()

print(f"\n  📊 顏色分布:")
print(f"    紅色頂點: {red_vertices} ({red_vertices/num_vertices*100:.2f}%)")
print(f"    白色頂點: {white_vertices} ({white_vertices/num_vertices*100:.2f}%)")

if red_vertices > num_vertices * 0.5:
    print(f"\n  ⚠️  警告：超過 50% 的頂點是紅色，這可能不正常！")

# 設置顏色
mesh.visual.vertex_colors = vertex_colors

# ==================== 步驟 6：導出 ====================

print("\n📦 步驟 6: 導出 3D 模型")

ply_path = OUTPUT_DIR / "plaque_real.ply"
mesh.export(str(ply_path), file_type='ply')
print(f"  ✅ PLY: {ply_path.name}")

try:
    glb_path = OUTPUT_DIR / "plaque_real.glb"
    mesh.export(str(glb_path), file_type='glb')
    print(f"  ✅ GLB: {glb_path.name}")
except:
    print(f"  ⚠️  GLB 導出失敗")

obj_path = OUTPUT_DIR / "plaque_real.obj"
mesh.export(str(obj_path), file_type='obj')
print(f"  ✅ OBJ: {obj_path.name}")

# ==================== 生成報告 ====================

print("\n📋 步驟 7: 生成整合報告")

integration_report = {
    'version': 'real_photos_fixed',
    'source_model': str(MODEL_FILE),
    'plaque_report': str(PLAQUE_REPORT),
    'summary': {
        'affected_teeth': len(affected_teeth),
        'marked_teeth': marked_count,
        'marked_vertices': int(total_marked_vertices),
        'total_vertices': num_vertices,
        'red_percentage': float(red_vertices/num_vertices*100)
    },
    'color_distribution': {
        'red_vertices': int(red_vertices),
        'white_vertices': int(white_vertices)
    },
    'output_files': {
        'ply': str(ply_path),
        'glb': str(glb_path) if glb_path.exists() else None,
        'obj': str(obj_path)
    }
}

with open(OUTPUT_DIR / "integration_report.json", 'w', encoding='utf-8') as f:
    json.dump(integration_report, f, indent=2, ensure_ascii=False)

print(f"  ✓ 已保存: integration_report.json")

print("\n" + "="*70)
print("✅ 整合完成！")
print("="*70)

print(f"\n📊 摘要:")
print(f"  標記了 {marked_count} 顆牙齒")
print(f"  紅色頂點: {red_vertices} ({red_vertices/num_vertices*100:.2f}%)")
print(f"  白色頂點: {white_vertices} ({white_vertices/num_vertices*100:.2f}%)")

print(f"\n📁 輸出檔案:")
print(f"  • plaque_real.ply ⭐（推薦）")
print(f"  • plaque_real.glb")
print(f"  • plaque_real.obj")

print("\n💡 如何查看:")
print(f"  線上: https://3dviewer.net/")
print(f"  MeshLab: Render → Show Vertex Colors")

print("\n" + "="*70)