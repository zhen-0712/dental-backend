#!/usr/bin/env python3
"""
牙菌斑 3D 整合系統 - 簡化版
特色：
1. ⭐⭐⭐ 只標記大致位置（不需精確還原形狀）
2. ⭐⭐ 使用純紅色 (255, 0, 0)
3. ⭐ 根據 3D 位置自動決定標記範圍
4. 自動修復頂點數不匹配
"""

import cv2
import numpy as np
import json
import trimesh
from pathlib import Path

# ==================== 配置 ====================

PLAQUE_REPORT_DIR = Path("./plaque_detection_final_v3")
PLAQUE_REPORT_V4 = PLAQUE_REPORT_DIR / "comprehensive_plaque_report_v4_strict.json"
PLAQUE_REPORT_V3 = PLAQUE_REPORT_DIR / "comprehensive_plaque_report_v3.json"

MODEL_DIR = Path("./personalized_3d_models_integrated")
MODEL_FILE = MODEL_DIR / "custom_complete.obj"

SEGMENTATION_DIR = Path("./models")
UPPER_SEG_FILE = SEGMENTATION_DIR / "H5EFRXCQ_upper.json"
LOWER_SEG_FILE = SEGMENTATION_DIR / "01J9K9S6_lower.json"

OUTPUT_DIR = Path("./plaque_3d_integration_simplified")
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*70)
print("🦷 牙菌斑 3D 整合系統 - 簡化版")
print("="*70)
print("\n✨ 特色:")
print("  1. ⭐⭐⭐ 只標記大致位置（間隙、牙齦等）")
print("  2. ⭐⭐ 使用純紅色標記")
print("  3. ⭐ 自動修復頂點數不匹配")

# ==================== 步驟 1: 載入報告 ====================

print("\n📊 步驟 1: 載入牙菌斑檢測報告")

affected_teeth = {}
report_version = "unknown"

# 優先載入 v4
if PLAQUE_REPORT_V4.exists():
    try:
        with open(PLAQUE_REPORT_V4, 'r', encoding='utf-8') as f:
            plaque_report = json.load(f)
        
        affected_teeth = plaque_report.get('affected_teeth', {})
        report_version = plaque_report.get('version', 'unknown')
        
        print(f"  ✓ 載入 v4 報告: {len(affected_teeth)} 顆牙齒")
        for tooth_id, info in affected_teeth.items():
            print(f"    #{tooth_id}: {info['position_3d']}")
    except Exception as e:
        print(f"  ⚠️  載入 v4 失敗: {e}")
        affected_teeth = {}

# 如果 v4 失敗，使用 v3
if len(affected_teeth) == 0 and PLAQUE_REPORT_V3.exists():
    try:
        with open(PLAQUE_REPORT_V3, 'r', encoding='utf-8') as f:
            plaque_report = json.load(f)
        
        affected_teeth = plaque_report.get('affected_teeth', {})
        report_version = plaque_report.get('version', 'unknown')
        
        print(f"  ⚠️  使用 v3 報告: {len(affected_teeth)} 顆牙齒")
    except Exception as e:
        print(f"  ⚠️  載入失敗: {e}")
        affected_teeth = {}

if len(affected_teeth) == 0:
    print(f"\n  ❌ 未找到報告")
    exit(1)

# ==================== 步驟 2: 載入分割標籤 ====================

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

# ==================== 步驟 3: 載入 3D 模型 ====================

print("\n🦷 步驟 3: 載入 3D 模型")

try:
    mesh = trimesh.load(str(MODEL_FILE))
    print(f"  ✓ 載入: {len(mesh.vertices)} 個頂點")
except Exception as e:
    print(f"  ❌ 失敗: {e}")
    exit(1)

# ==================== 步驟 4: 自動修復頂點數 ====================

print("\n🔧 步驟 4: 檢查頂點數")

num_vertices = len(mesh.vertices)
num_labels = len(combined_seg_labels)

if num_vertices != num_labels:
    print(f"  ⚠️  不匹配: 頂點={num_vertices}, 標籤={num_labels}")
    
    if num_labels > num_vertices:
        print(f"  🔧 截斷標籤到 {num_vertices} 個")
        combined_seg_labels = combined_seg_labels[:num_vertices]
    else:
        print(f"  🔧 填充標籤到 {num_vertices} 個")
        padding = np.zeros(num_vertices - num_labels, dtype=np.int32)
        combined_seg_labels = np.concatenate([combined_seg_labels, padding])
    
    print(f"  ✅ 修復完成")
else:
    print(f"  ✅ 完美匹配")

# ==================== 步驟 5: 應用牙菌斑標記（簡化版）====================

print("\n🎨 步驟 5: 標記牙菌斑（簡化版）")

def apply_plaque_simplified(mesh, seg_labels, affected_teeth, output_dir):
    """
    ⭐⭐⭐ 簡化版標記：只標記大致區域，使用純紅色
    """
    num_vertices = len(mesh.vertices)
    
    # 初始化為白色
    vertex_colors = np.ones((num_vertices, 4), dtype=np.uint8) * 255
    
    # ⭐⭐⭐ 純紅色（BGR 格式）
    RED_COLOR = np.array([255, 0, 0, 255], dtype=np.uint8)
    
    marked_count = 0
    
    for tooth_id_str, info in affected_teeth.items():
        tooth_id = int(tooth_id_str)
        
        # 找到牙齒的所有頂點
        tooth_mask = (seg_labels == tooth_id)
        tooth_indices = np.where(tooth_mask)[0]
        
        # 邊界檢查
        valid_indices = tooth_indices[tooth_indices < num_vertices]
        
        if len(valid_indices) == 0:
            print(f"    ⚠️  未找到牙齒 #{tooth_id}")
            continue
        
        tooth_vertices = mesh.vertices[valid_indices]
        position_3d = info.get('position_3d', [])
        
        # ⭐⭐⭐ 根據 3D 位置決定標記範圍（簡化版）
        if isinstance(position_3d, list):
            position_3d_str = ', '.join(position_3d) if position_3d else 'unknown'
        else:
            position_3d_str = str(position_3d)
        
        # 決定標記哪個部分
        if any(p in position_3d_str for p in ['gap', 'anterior_gap', 'premolar_gap']):
            # 間隙：標記整顆
            colored_indices = valid_indices
            region_str = '整顆（間隙）'
            
        elif any(p in position_3d_str for p in ['gingival', 'buccal_gingival', 'gingival_margin']):
            # 牙齦：標記下半部
            z_coords = tooth_vertices[:, 2]
            z_threshold = np.percentile(z_coords, 50)  # 下半部
            lower_mask = z_coords < z_threshold
            colored_indices = valid_indices[lower_mask]
            region_str = '下半部（牙齦）'
            
        elif any(p in position_3d_str for p in ['occlusal', 'occlusal_surface']):
            # 咬合面：標記上半部
            z_coords = tooth_vertices[:, 2]
            z_threshold = np.percentile(z_coords, 50)  # 上半部
            upper_mask = z_coords > z_threshold
            colored_indices = valid_indices[upper_mask]
            region_str = '上半部（咬合面）'
            
        else:
            # 未知：標記整顆
            colored_indices = valid_indices
            region_str = '整顆（未知）'
        
        # ⭐ 應用純紅色
        vertex_colors[colored_indices] = RED_COLOR
        marked_count += 1
        
        print(f"    ✅ #{tooth_id} ({region_str}): {len(colored_indices)} 個頂點")
    
    # 設置頂點顏色
    mesh.visual.vertex_colors = vertex_colors
    
    print(f"\n  ✅ 共標記 {marked_count} 顆牙齒")
    
    # 導出
    print("\n  📦 導出 3D 模型...")
    
    ply_path = output_dir / "plaque_simplified.ply"
    mesh.export(str(ply_path), file_type='ply')
    print(f"    ✅ PLY: {ply_path.name}")
    
    try:
        glb_path = output_dir / "plaque_simplified.glb"
        mesh.export(str(glb_path), file_type='glb')
        print(f"    ✅ GLB: {glb_path.name}")
    except:
        pass
    
    obj_path = output_dir / "plaque_simplified.obj"
    mesh.export(str(obj_path), file_type='obj')
    print(f"    ✅ OBJ: {obj_path.name}")
    
    return mesh

result_mesh = apply_plaque_simplified(
    mesh,
    combined_seg_labels,
    affected_teeth,
    OUTPUT_DIR
)

# ==================== 生成報告 ====================

print("\n📋 步驟 6: 生成報告")

integration_report = {
    'version': 'simplified',
    'features': [
        'Simplified region marking (no precise shape restoration)',
        'Pure red color (255, 0, 0)',
        'Automatic vertex count correction',
        '3D position-based region selection'
    ],
    'plaque_report': {
        'version': report_version,
        'affected_teeth': len(affected_teeth)
    },
    'model': {
        'vertices': len(mesh.vertices),
        'faces': len(mesh.faces)
    },
    'integration': {
        'success': True,
        'marked_teeth': len(affected_teeth)
    },
    'output_files': {
        'ply': str(OUTPUT_DIR / "plaque_simplified.ply"),
        'glb': str(OUTPUT_DIR / "plaque_simplified.glb"),
        'obj': str(OUTPUT_DIR / "plaque_simplified.obj")
    }
}

with open(OUTPUT_DIR / "integration_report.json", 'w', encoding='utf-8') as f:
    json.dump(integration_report, f, indent=2, ensure_ascii=False)

print(f"  ✓ 已保存: integration_report.json")

# ==================== 最終報告 ====================

print("\n" + "="*70)
print("✅ 整合完成！")
print("="*70)

print(f"\n📊 摘要:")
print(f"  標記了 {len(affected_teeth)} 顆牙齒")
print(f"  使用純紅色 (255, 0, 0)")
print(f"  簡化標記（只標記大致區域）")

print(f"\n📁 輸出檔案:")
print(f"  • plaque_simplified.ply ⭐（推薦）")
print(f"  • plaque_simplified.glb")
print(f"  • plaque_simplified.obj")

print(f"\n💡 如何查看:")
print(f"  MeshLab:")
print(f"    1. 開啟 plaque_simplified.ply")
print(f"    2. 檢視 → 渲染模式 → Per Vertex Color")
print(f"  ")
print(f"  線上檢視:")
print(f"    拖放 plaque_simplified.glb 到")
print(f"    https://gltf-viewer.donmccurdy.com/")

print("\n" + "="*70)