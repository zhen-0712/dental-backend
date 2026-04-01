#!/usr/bin/env python3
"""
客製化 3D 模型生成 - 真實牙齒版本
只處理 real_teeth 的分析結果
"""

import json
import numpy as np
import trimesh
from scipy.spatial import KDTree
from pathlib import Path
from extract_control_points import get_tooth_control_points_3d
# tps_deformation 保留檔案，但此版本不使用

# ==================== 配置 ====================

ANALYSIS_DIR = Path("./real_teeth_analysis")
MODELS_DIR = Path("./models")
OUTPUT_DIR = Path("./personalized_3d_models_real")
OUTPUT_DIR.mkdir(exist_ok=True)

# ⭐⭐⭐ 客製化策略
ENABLE_RATIO_ADJUSTMENT = True
MAX_RATIO_SCALE = 1.05
MIN_RATIO_SCALE = 0.85
RATIO_ADJUSTMENT_STRENGTH = 0.5

ENABLE_OVERALL_ADJUSTMENT = True
OVERALL_SCALE_MODE = 'smart'
OVERALL_SCALE_STRENGTH = 1.2
MAX_OVERALL_SCALE = 1.15
MIN_OVERALL_SCALE = 0.85

VIEW_WEIGHTS = {
    'front': 1.5,
    'upper': 1.2,
    'lower': 1.2,
    'left': 0.6,
    'right': 0.6
}

DIMENSION_MAPPING = {
    'front': {'width': 'width', 'height': 'height', 'depth': None},
    'upper': {'width': 'width', 'height': 'depth', 'depth': None},
    'lower': {'width': 'width', 'height': 'depth', 'depth': None},
    'left': {'width': 'depth', 'height': 'height', 'depth': None},
    'right': {'width': 'depth', 'height': 'height', 'depth': None}
}

ENABLE_OUTLIER_FILTERING = True
OUTLIER_IQR_MULTIPLIER = 1.5

# ⭐ 傾斜修正配置（前牙專用）
TILT_CORRECTION_TEETH  = {11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43}
TILT_MIN_CONFIDENCE    = 0.15   # 可信度低於此不使用
TILT_MAX_DEG           = 20.0   # 最大允許修正角度
TILT_CORRECTION_BLEND  = 0.6    # 修正強度（0=不修正，1=完全修正）

# 上下顎對齊參數
UPPER_FLIP = True
UPPER_ROTATE_180 = True
UPPER_ROTATE_ANGLE_Z = -115.0
VERTICAL_SPACING = 10.0
UPPER_JAW_OPENING_ANGLE = -50.0
ARCH_COMPRESSION_RATIO_UPPER = 0.75  # 上顎壓縮（間隙大所以壓更多）
ARCH_COMPRESSION_RATIO_LOWER = 0.82  # 下顎壓縮（已經不錯，維持）
UPPER_TILT_CORRECTION_Y = -8.0    # 已有：前後傾斜
UPPER_TILT_CORRECTION_Z = 3.0      # ⭐ 新增：左右滾動（正值=右側抬高，負值=左側抬高）
ROTATION_PIVOT_RATIO = 0.2
ROTATION_PIVOT_X_RATIO = 0.3
UPPER_POSITION_OFFSET_X = 0.0
UPPER_POSITION_OFFSET_Y = 35.0
UPPER_POSITION_OFFSET_Z = 10.0

print("="*70)
print("🦷 客製化 3D 模型生成 - 真實牙齒版本")
print("="*70)

# ==================== 載入分析結果 ====================

print("\n📊 載入分析結果...")

try:
    with open(ANALYSIS_DIR / "real_teeth_analysis.json") as f:
        analysis = json.load(f)
    print(f"  ✓ 載入分析結果")
except Exception as e:
    print(f"  ❌ 載入失敗: {e}")
    exit(1)

detected_teeth = set(analysis['detected_teeth'])
never_detected = analysis.get('never_detected', [])

print(f"\n📋 牙齒狀態:")
print(f"  檢測到: {len(detected_teeth)} 顆 → {sorted(detected_teeth)}")
print(f"  從未出現: {len(never_detected)} 顆 → {never_detected}")

# ==================== 載入公版模型 ====================

print("\n📦 載入公版模型...")

upper_mesh = trimesh.load(str(MODELS_DIR / "1MWJLE4X_upper.obj"))
lower_mesh = trimesh.load(str(MODELS_DIR / "01J9K9S6_lower.obj"))

with open(MODELS_DIR / "1MWJLE4X_upper.json") as f:
    upper_seg_data = json.load(f)
    upper_seg_labels = np.array(upper_seg_data["labels"], dtype=np.int32)

with open(MODELS_DIR / "01J9K9S6_lower.json") as f:
    lower_seg_data = json.load(f)
    lower_seg_labels = np.array(lower_seg_data["labels"], dtype=np.int32)

print(f"  ✓ 已載入")

# ==================== 工具函數 ====================

def extract_tooth_vertices_indices(seg_labels, tooth_id):
    mask = (seg_labels == tooth_id)
    indices = np.nonzero(mask)[0]
    return indices if len(indices) > 0 else None

def measure_tooth_dimensions(vertices):
    if vertices is None or len(vertices) == 0:
        return None
    
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)
    dimensions = max_coords - min_coords
    
    return {
        'width': float(dimensions[0]),
        'height': float(dimensions[1]),
        'depth': float(dimensions[2]),
        'center': ((min_coords + max_coords) / 2).tolist()
    }

def get_measured_3d_size_from_analysis(tooth_id, analysis_data):
    """從分析數據中提取 3D 測量值"""
    tooth_id_str = str(tooth_id)
    if tooth_id_str not in analysis_data['teeth']:
        return None, None, None, None
    
    tooth_data = analysis_data['teeth'][tooth_id_str]
    
    if 'dimensions_3d' in tooth_data:
        dims_3d = tooth_data['dimensions_3d']
        
        width = dims_3d.get('width', {}).get('mean')
        height = dims_3d.get('height', {}).get('mean')
        depth = dims_3d.get('depth', {}).get('mean')
        
        filter_info = {
            'width': {'filtered': 0, 'total': dims_3d.get('width', {}).get('count', 0)},
            'height': {'filtered': 0, 'total': dims_3d.get('height', {}).get('count', 0)},
            'depth': {'filtered': 0, 'total': dims_3d.get('depth', {}).get('count', 0)}
        }
        
        return width, height, depth, filter_info
    
    return None, None, None, None

def align_upper_lower(upper_mesh, lower_mesh, flip_upper=True, rotate_180=False, 
                      spacing=10.0, opening_angle=0.0, pivot_ratio=0.7,
                      offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """對齊上下顎"""
    aligned_upper = upper_mesh.copy()
    aligned_lower = lower_mesh.copy()
    
    lower_bounds = aligned_lower.bounds
    lower_top = lower_bounds[1, 2]
    
    upper_bounds = aligned_upper.bounds
    upper_center = upper_bounds.mean(axis=0)
    upper_bottom = upper_bounds[0, 2]
    
    if flip_upper:
        rotation_matrix = trimesh.transformations.rotation_matrix(
            np.pi, [1, 0, 0], upper_center
        )
        aligned_upper.apply_transform(rotation_matrix)
        upper_bounds = aligned_upper.bounds
        upper_center = upper_bounds.mean(axis=0)
        upper_bottom = upper_bounds[0, 2]
    
    if rotate_180:
        rotation_matrix_z = trimesh.transformations.rotation_matrix(
            np.radians(UPPER_ROTATE_ANGLE_Z), [0, 0, 1], upper_center
        )
        aligned_upper.apply_transform(rotation_matrix_z)
        upper_bounds = aligned_upper.bounds
        upper_center = upper_bounds.mean(axis=0)
    
    if UPPER_TILT_CORRECTION_Y != 0:
        upper_center = aligned_upper.bounds.mean(axis=0)
        rotation_matrix_tilt = trimesh.transformations.rotation_matrix(
            np.radians(UPPER_TILT_CORRECTION_Y), [0, 1, 0], upper_center
        )
        aligned_upper.apply_transform(rotation_matrix_tilt)
        upper_bounds = aligned_upper.bounds
        upper_center = upper_bounds.mean(axis=0)

    if UPPER_TILT_CORRECTION_Z != 0:
        upper_center = aligned_upper.bounds.mean(axis=0)
        rotation_matrix_roll = trimesh.transformations.rotation_matrix(
            np.radians(UPPER_TILT_CORRECTION_Z), [0, 0, 1], upper_center
        )
        aligned_upper.apply_transform(rotation_matrix_roll)
        upper_bounds = aligned_upper.bounds
        upper_center = upper_bounds.mean(axis=0)
    
    upper_center_xy = aligned_upper.bounds.mean(axis=0)[:2]
    lower_center_xy = aligned_lower.bounds.mean(axis=0)[:2]
    
    translation_xy = lower_center_xy - upper_center_xy
    aligned_upper.vertices[:, :2] += translation_xy
    
    z_offset = lower_top + spacing - upper_bottom
    aligned_upper.vertices[:, 2] += z_offset
    
    if opening_angle != 0:
        upper_bounds = aligned_upper.bounds
        
        y_min = upper_bounds[0, 1]
        y_max = upper_bounds[1, 1]
        pivot_y = y_min + (y_max - y_min) * pivot_ratio
        
        x_min = upper_bounds[0, 0]
        x_max = upper_bounds[1, 0]
        pivot_x = x_min + (x_max - x_min) * ROTATION_PIVOT_X_RATIO
        
        pivot_z = (upper_bounds[0, 2] + upper_bounds[1, 2]) / 2
        
        pivot_point = np.array([pivot_x, pivot_y, pivot_z])
        
        angle_rad = np.radians(opening_angle)
        rotation_matrix_opening = trimesh.transformations.rotation_matrix(
            angle_rad, [1, 0, 0], pivot_point
        )
        
        aligned_upper.apply_transform(rotation_matrix_opening)
    
    if offset_x != 0 or offset_y != 0 or offset_z != 0:
        aligned_upper.vertices[:, 0] += offset_x
        aligned_upper.vertices[:, 1] += offset_y
        aligned_upper.vertices[:, 2] += offset_z
    
    return aligned_upper, aligned_lower

def build_tooth_face_groups(faces, vertex_labels):
    faces = np.asarray(faces, dtype=np.int64)
    vertex_labels = np.asarray(vertex_labels, dtype=np.int64)
    
    face_labels = vertex_labels[faces]
    same_label = (face_labels[:, 0] == face_labels[:, 1]) & \
                 (face_labels[:, 1] == face_labels[:, 2])
    
    valid_faces_idx = np.nonzero(same_label)[0]
    valid_face_labels = face_labels[same_label, 0]
    
    tooth_face_groups = {}
    for idx, t_id in zip(valid_faces_idx, valid_face_labels):
        if t_id == 0:
            continue
        t_id = int(t_id)
        tooth_face_groups.setdefault(t_id, []).append(int(idx))
    
    for k in list(tooth_face_groups.keys()):
        tooth_face_groups[k] = np.array(tooth_face_groups[k], dtype=np.int64)
    
    return tooth_face_groups

def remove_teeth_from_mesh(vertices, faces, vertex_labels, teeth_to_remove):
    if not teeth_to_remove:
        return vertices, faces
    
    faces = np.asarray(faces, dtype=np.int32)
    tooth_face_groups = build_tooth_face_groups(faces, vertex_labels)
    
    mask_faces = np.ones(len(faces), dtype=bool)
    
    for t_id in teeth_to_remove:
        if t_id in tooth_face_groups:
            face_indices = tooth_face_groups[t_id]
            mask_faces[face_indices] = False
    
    remaining_faces = faces[mask_faces]
    return vertices, remaining_faces

def compute_customized_3d_scales(mesh, seg_labels, analysis_data, teeth_list):
    """計算客製化 3D 縮放係數"""
    print(f"\n🔍 計算客製化 3D 縮放係數...")
    
    tooth_scales = {}
    adjusted_count = 0
    all_scale_factors = {'width': [], 'height': [], 'depth': []}
    
    for tooth_id in teeth_list:
        tooth_indices = extract_tooth_vertices_indices(seg_labels, tooth_id)
        if tooth_indices is None:
            continue
        
        tooth_verts = mesh.vertices[tooth_indices]
        dims = measure_tooth_dimensions(tooth_verts)
        if dims is None:
            continue
        
        measured_width, measured_height, measured_depth, filter_info = \
            get_measured_3d_size_from_analysis(tooth_id, analysis_data)
        
        if measured_width is None and measured_height is None and measured_depth is None:
            continue
        
        scale_x_ratio = 1.0
        scale_y_ratio = 1.0
        scale_z_ratio = 1.0
        
        if ENABLE_RATIO_ADJUSTMENT:
            if measured_width is not None and dims['width'] > 0:
                tooth_id_str = str(tooth_id)
                width_count = analysis_data['teeth'][tooth_id_str]['dimensions_3d']['width'].get('count', 1)
                width_std = analysis_data['teeth'][tooth_id_str]['dimensions_3d']['width'].get('std', 0) or 0
                trust = 1.0 if width_count >= 2 else 0.3
                ratio_x = measured_width / dims['width']
                all_scale_factors['width'].append(ratio_x)
                scale_x_ratio = ratio_x ** (RATIO_ADJUSTMENT_STRENGTH * 0.5 * trust)
                scale_x_ratio = np.clip(scale_x_ratio, MIN_RATIO_SCALE, MAX_RATIO_SCALE)
            
            if measured_height is not None and dims['height'] > 0:
                tooth_id_str = str(tooth_id)
                height_count = analysis_data['teeth'][tooth_id_str]['dimensions_3d']['height'].get('count', 1)
                trust = 1.0 if height_count >= 2 else 0.3
                ratio_y = measured_height / dims['height']
                all_scale_factors['height'].append(ratio_y)
                scale_y_ratio = ratio_y ** (RATIO_ADJUSTMENT_STRENGTH * 0.5 * trust)
                scale_y_ratio = np.clip(scale_y_ratio, MIN_RATIO_SCALE, MAX_RATIO_SCALE)
            
            if measured_depth is not None and dims['depth'] > 0:
                tooth_id_str = str(tooth_id)
                depth_count = analysis_data['teeth'][tooth_id_str]['dimensions_3d']['depth'].get('count', 1)
                trust = 1.0 if depth_count >= 2 else 0.3
                ratio_z = measured_depth / dims['depth']
                all_scale_factors['depth'].append(ratio_z)
                scale_z_ratio = ratio_z ** (RATIO_ADJUSTMENT_STRENGTH * 0.5 * trust)
                scale_z_ratio = np.clip(scale_z_ratio, MIN_RATIO_SCALE, MAX_RATIO_SCALE)
        
        has_adjustment = (
            abs(scale_x_ratio - 1.0) > 0.02 or 
            abs(scale_y_ratio - 1.0) > 0.02 or 
            abs(scale_z_ratio - 1.0) > 0.02
        )
        
        if has_adjustment:
            adjusted_count += 1
        
        tooth_scales[tooth_id] = {
            'scale_x': scale_x_ratio,
            'scale_y': scale_y_ratio,
            'scale_z': scale_z_ratio,
            'model_dims': dims,
            'measured_width': measured_width,
            'measured_height': measured_height,
            'measured_depth': measured_depth,
            'has_adjustment': has_adjustment
        }
    
    print(f"  ✓ 需要調整的牙齒: {adjusted_count}/{len(tooth_scales)} 顆")
    
    overall_scale_x = 1.0
    overall_scale_y = 1.0
    overall_scale_z = 1.0

    if ENABLE_OVERALL_ADJUSTMENT:
        for axis_name, scale_list_key, overall_var in [
            ('width',  'width',  None),
            ('height', 'height', None),
            ('depth',  'depth',  None),
        ]:
            factors = all_scale_factors[scale_list_key]
            if not factors:
                continue
            
            median_ratio = float(np.median(factors))
            adjusted = 1.0 + (median_ratio - 1.0) * OVERALL_SCALE_STRENGTH
            adjusted = float(np.clip(adjusted, MIN_OVERALL_SCALE, MAX_OVERALL_SCALE))
            
            if axis_name == 'width':
                overall_scale_x = adjusted
            elif axis_name == 'height':
                overall_scale_y = adjusted
            elif axis_name == 'depth':
                overall_scale_z = adjusted
        
        print(f"\n  ⭐ 三軸整體縮放: X(width)={overall_scale_x:.3f}x  Y(height)={overall_scale_y:.3f}x  Z(depth)={overall_scale_z:.3f}x")

    overall_scale = (overall_scale_x + overall_scale_y + overall_scale_z) / 3.0
    
    for tooth_id in tooth_scales:
        tooth_scales[tooth_id]['overall_scale'] = overall_scale
        tooth_scales[tooth_id]['overall_scale_x'] = overall_scale_x
        tooth_scales[tooth_id]['overall_scale_y'] = overall_scale_y
        tooth_scales[tooth_id]['overall_scale_z'] = overall_scale_z
        tooth_scales[tooth_id]['scale_x'] *= overall_scale_x
        tooth_scales[tooth_id]['scale_y'] *= overall_scale_y
        tooth_scales[tooth_id]['scale_z'] *= overall_scale_z
    
    return tooth_scales

def apply_customized_scaling(mesh, seg_labels, tooth_scales):
    """應用客製化縮放"""
    print("\n⚙️  應用客製化 3D 縮放...")
    
    scaled_mesh = mesh.copy()
    vertices = scaled_mesh.vertices.copy()
    
    adjusted_count = 0
    
    for tooth_id, scales in tooth_scales.items():
        if not scales['has_adjustment'] and scales['overall_scale'] == 1.0:
            continue
        
        tooth_indices = extract_tooth_vertices_indices(seg_labels, tooth_id)
        if tooth_indices is None:
            continue
        
        tooth_verts = vertices[tooth_indices]
        center = tooth_verts.mean(axis=0)
        
        relative_verts = tooth_verts - center
        
        scale_matrix = np.array([
            scales['scale_x'],
            scales['scale_y'],
            scales['scale_z']
        ])
        
        scaled_relative = relative_verts * scale_matrix
        vertices[tooth_indices] = scaled_relative + center
        
        adjusted_count += 1
    
    print(f"  ✓ 共調整 {adjusted_count} 顆牙齒")
    
    scaled_mesh.vertices = vertices
    return scaled_mesh

def normalize_anterior_incisal_height(mesh, seg_labels, analysis_data, teeth_list):
    print("\n📐 前牙切緣高度正規化...")

    NORMALIZE_TEETH = {11, 12, 21, 22, 31, 32, 41, 42}
    BLEND = 0.8

    vertices = mesh.vertices.copy()

    tooth_info = {}
    for tooth_id in teeth_list:
        if tooth_id not in NORMALIZE_TEETH:
            continue
        indices = np.nonzero(seg_labels == tooth_id)[0]
        if len(indices) == 0:
            continue
        tooth_verts = vertices[indices]
        z_top = float(tooth_verts[:, 2].max())
        z_bot = float(tooth_verts[:, 2].min())
        current_h = z_top - z_bot

        tooth_id_str = str(tooth_id)
        if tooth_id_str not in analysis_data['teeth']:
            continue
        dims = analysis_data['teeth'][tooth_id_str].get('dimensions_3d', {})
        measured_h = dims.get('height', {}).get('mean')
        if measured_h is None or measured_h <= 0:
            continue

        tooth_info[tooth_id] = {
            'indices': indices,
            'z_top': z_top,
            'z_bot': z_bot,
            'current_h': current_h,
            'measured_h': measured_h,
        }

    if not tooth_info:
        print("  ⚠ 無有效前牙資料，跳過")
        result = mesh.copy()
        result.vertices = vertices
        return result

    avg_z_bot = np.mean([d['z_bot'] for d in tooth_info.values()])

    corrected = 0
    for tooth_id, d in tooth_info.items():
        indices = d['indices']
        z_top = d['z_top']
        z_bot = d['z_bot']
        current_h = d['current_h']
        measured_h = d['measured_h']

        target_z_top = z_bot + measured_h
        z_shift = (target_z_top - z_top) * BLEND

        if abs(z_shift) < 0.1:
            continue

        tooth_verts = vertices[indices]
        z_range = z_top - z_bot + 1e-6
        blend_weight = np.clip((tooth_verts[:, 2] - z_bot) / z_range, 0, 1)
        blend_weight = blend_weight ** 0.5

        vertices[indices, 2] += z_shift * blend_weight
        corrected += 1
        print(f"    #{tooth_id}: 測量高{measured_h:.1f}mm  切緣偏移{z_shift:+.2f}mm")

    sample_id = list(tooth_info.keys())[0]
    d = tooth_info[sample_id]
    print(f"  [DEBUG] #{sample_id}: z_bot={d['z_bot']:.1f}  z_top={d['z_top']:.1f}  current_h={d['current_h']:.1f}  measured_h={d['measured_h']:.1f}")

    print(f"  ✓ 切緣高度正規化: {corrected} 顆")
    result = mesh.copy()

    SHORTEN_TEETH = {11, 21}
    SHORTEN_RATIO = 0.85

    for tooth_id in SHORTEN_TEETH:
        if tooth_id not in tooth_info:
            continue
        indices = tooth_info[tooth_id]['indices']
        tooth_verts = vertices[indices]
        z_top = tooth_info[tooth_id]['z_top']
        
        z_vals = tooth_verts[:, 2]
        z_min = z_vals.min()
        z_max = z_vals.max()
        
        n_bins = 20
        bin_edges = np.linspace(z_min, z_max, n_bins + 1)
        widths = []
        for i in range(n_bins):
            layer_mask = (z_vals >= bin_edges[i]) & (z_vals < bin_edges[i+1])
            if layer_mask.sum() < 3:
                widths.append(np.inf)
                continue
            layer_verts = tooth_verts[layer_mask]
            w = layer_verts[:, 0].max() - layer_verts[:, 0].min()
            widths.append(w)
        
        widths = np.array(widths)
        lower_half = widths[:n_bins//2]
        cej_bin = np.argmin(lower_half)
        z_cej = bin_edges[cej_bin + 1]
        
        print(f"    #{tooth_id}: CEJ估算z={z_cej:.1f} (z_top={z_top:.1f})")
        
        crown_mask = tooth_verts[:, 2] > z_cej
        if crown_mask.sum() == 0:
            continue
        
        rel_z = tooth_verts[crown_mask, 2] - z_cej
        vertices[indices[crown_mask], 2] = z_cej + rel_z * SHORTEN_RATIO
        print(f"    #{tooth_id}: 牙冠縮短至 {SHORTEN_RATIO*100:.0f}%")

    LIFT_TEETH = {11, 21}
    LIFT_MM = 2.0
    PUSH_IN_MM = 1.5

    for tooth_id in LIFT_TEETH:
        indices = np.nonzero(seg_labels == tooth_id)[0]
        if len(indices) == 0:
            continue
        vertices[indices, 2] += LIFT_MM
        vertices[indices, 1] += PUSH_IN_MM
        print(f"    #{tooth_id}: 往上平移 {LIFT_MM:.1f}mm, 往舌側 {PUSH_IN_MM:.1f}mm")

    result.vertices = vertices
    return result

def compress_arch_spacing(mesh, seg_labels, compression=0.82):
    """壓縮牙弓讓牙齒間隙縮小，牙齦平滑過渡避免摺痕"""
    compressed = mesh.copy()
    vertices = compressed.vertices.copy()
    arch_center_x = vertices[:, 0].mean()
    arch_center_y = vertices[:, 1].mean()
    
    n_verts = len(vertices)
    delta = np.zeros((n_verts, 2))
    
    unique_teeth = np.unique(seg_labels)
    unique_teeth = unique_teeth[unique_teeth > 0]
    
    for tooth_id in unique_teeth:
        indices = np.nonzero(seg_labels == tooth_id)[0]
        if len(indices) == 0:
            continue
        tooth_center = vertices[indices].mean(axis=0)
        dx = (tooth_center[0] - arch_center_x) * (compression - 1.0)
        dy = (tooth_center[1] - arch_center_y) * (compression - 1.0)
        delta[indices, 0] = dx
        delta[indices, 1] = dy
    
    gum_indices = np.nonzero(seg_labels == 0)[0]
    tooth_indices_all = np.nonzero(seg_labels > 0)[0]
    
    if len(gum_indices) > 0 and len(tooth_indices_all) > 0:
        gum_verts = vertices[gum_indices]
        tooth_verts = vertices[tooth_indices_all]
        tooth_deltas = delta[tooth_indices_all]
        
        z_min = vertices[:, 2].min()
        z_max = vertices[:, 2].max()
        z_range = z_max - z_min + 1e-6
        
        gum_z_weight = np.clip((gum_verts[:, 2] - z_min) / z_range, 0, 1)
        gum_z_weight = gum_z_weight ** 0.1
        
        from scipy.spatial import KDTree
        tree = KDTree(tooth_verts[:, :2])
        
        distances, neighbor_idx = tree.query(gum_verts[:, :2], k=min(50, len(tooth_verts)))
        
        distances = np.maximum(distances, 1e-6)
        weights = 1.0 / (distances ** 1.0)
        weight_sum = weights.sum(axis=1, keepdims=True)
        weights = weights / weight_sum
        
        gum_dx = (weights * tooth_deltas[neighbor_idx, 0]).sum(axis=1)
        gum_dy = (weights * tooth_deltas[neighbor_idx, 1]).sum(axis=1)
        
        delta[gum_indices, 0] = gum_dx * gum_z_weight
        delta[gum_indices, 1] = gum_dy * gum_z_weight
    
    vertices[:, 0] += delta[:, 0]
    vertices[:, 1] += delta[:, 1]
    
    compressed.vertices = vertices
    return compressed

def apply_tilt_correction(mesh, seg_labels, analysis_data, teeth_list):
    print("\n📐 前牙傾斜修正...")

    vertices = mesh.vertices.copy()
    corrected_count = 0
    skipped_count   = 0

    for tooth_id in teeth_list:
        if tooth_id not in TILT_CORRECTION_TEETH:
            continue

        tooth_id_str = str(tooth_id)
        if tooth_id_str not in analysis_data['teeth']:
            skipped_count += 1
            continue

        tilt_3d = analysis_data['teeth'][tooth_id_str].get('tilt_3d')
        if tilt_3d is None:
            skipped_count += 1
            continue

        conf     = tilt_3d.get('confidence', 0.0)
        tilt_deg = tilt_3d.get('tilt_deg', 0.0)

        if conf < TILT_MIN_CONFIDENCE:
            skipped_count += 1
            continue

        correction_deg = -np.clip(tilt_deg, -TILT_MAX_DEG, TILT_MAX_DEG) * TILT_CORRECTION_BLEND

        if tooth_id in {11, 21}:
            skipped_count += 1
            continue
            
        if abs(correction_deg) < 0.5:
            skipped_count += 1
            continue

        indices = np.nonzero(seg_labels == tooth_id)[0]
        if len(indices) == 0:
            skipped_count += 1
            continue

        tooth_verts = vertices[indices]
        z_bot = tooth_verts[:, 2].min()
        pivot = tooth_verts.mean(axis=0)
        pivot[2] = tooth_verts[:, 2].min() + (tooth_verts[:, 2].max() - tooth_verts[:, 2].min()) * 0.3
        pivot[2] = z_bot

        angle_rad = np.radians(correction_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

        rel = tooth_verts - pivot
        new_y =  rel[:, 1] * cos_a - rel[:, 2] * sin_a
        new_z =  rel[:, 1] * sin_a + rel[:, 2] * cos_a
        rel[:, 1] = new_y
        rel[:, 2] = new_z

        vertices[indices] = rel + pivot
        corrected_count += 1
        print(f"    #{tooth_id}: {tilt_deg:+.1f}° → 修正 {correction_deg:+.1f}° (可信度 {conf:.2f})")

    print(f"  ✓ 傾斜修正: {corrected_count} 顆")
    if skipped_count > 0:
        print(f"  ⚠ 跳過: {skipped_count} 顆")

    result_mesh = mesh.copy()
    result_mesh.vertices = vertices
    return result_mesh

def apply_incisal_curvature(mesh, seg_labels, analysis_data, teeth_list):
    print("\n📐 前牙切緣輪廓修正...")

    INCISAL_TEETH = {11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43}
    BLEND_STRENGTH = 0.8
    CROWN_TOP_RATIO = 0.30
    MAX_Z_SHIFT = 1.5

    vertices = mesh.vertices.copy()
    corrected_count = 0

    for tooth_id in teeth_list:
        if tooth_id not in INCISAL_TEETH:
            continue

        tooth_data = analysis_data['teeth'].get(str(tooth_id))
        if tooth_data is None:
            continue

        front_cp = None
        for m in tooth_data.get('measurements', []):
            if m.get('view') == 'front' and m.get('contour_cp'):
                front_cp = m['contour_cp']['points']
                break
        if front_cp is None:
            continue

        p_msl = front_cp.get('mesial_third', {})
        p_mid = front_cp.get('incisal_mid', {})
        p_dst = front_cp.get('distal_third', {})

        x_msl = p_msl.get('x')
        z_msl = p_msl.get('z_rel')
        x_mid = p_mid.get('x')
        z_mid_cp = p_mid.get('z_rel')
        x_dst = p_dst.get('x')
        z_dst = p_dst.get('z_rel')

        if any(v is None for v in [x_msl, z_msl, x_mid, z_mid_cp, x_dst, z_dst]):
            continue

        z_baseline = (z_msl + z_mid_cp + z_dst) / 3.0
        dz_msl = (z_msl - z_baseline) * BLEND_STRENGTH
        dz_mid = (z_mid_cp - z_baseline) * BLEND_STRENGTH
        dz_dst = (z_dst - z_baseline) * BLEND_STRENGTH

        dz_msl = np.clip(dz_msl, -MAX_Z_SHIFT, MAX_Z_SHIFT)
        dz_mid = np.clip(dz_mid, -MAX_Z_SHIFT, MAX_Z_SHIFT)
        dz_dst = np.clip(dz_dst, -MAX_Z_SHIFT, MAX_Z_SHIFT)

        if max(abs(dz_msl), abs(dz_mid), abs(dz_dst)) < 0.2:
            continue

        indices = np.nonzero(seg_labels == tooth_id)[0]
        if len(indices) == 0:
            continue

        tooth_verts = vertices[indices]
        z_min = tooth_verts[:, 2].min()
        z_max = tooth_verts[:, 2].max()
        z_range = z_max - z_min
        if z_range < 0.1:
            continue

        x_center = tooth_verts[:, 0].mean()
        x_half_span = (tooth_verts[:, 0].max() - tooth_verts[:, 0].min()) / 2.0
        if x_half_span < 0.1:
            continue

        z_threshold = z_max - z_range * CROWN_TOP_RATIO
        crown_mask = tooth_verts[:, 2] >= z_threshold
        if crown_mask.sum() == 0:
            continue

        height_blend = np.zeros(len(tooth_verts))
        height_blend[crown_mask] = (
            (tooth_verts[crown_mask, 2] - z_threshold) / (z_max - z_threshold + 1e-6)
        )

        verts_x_rel = (tooth_verts[:, 0] - x_center)

        cp_xs = np.array([x_msl, x_mid, x_dst])
        cp_dzs = np.array([dz_msl, dz_mid, dz_dst])

        coeffs = np.polyfit(cp_xs, cp_dzs, 2)
        z_offsets = np.polyval(coeffs, verts_x_rel)

        x_cp_min = cp_xs.min()
        x_cp_max = cp_xs.max()
        out_of_range = (verts_x_rel < x_cp_min) | (verts_x_rel > x_cp_max)
        z_offsets[out_of_range] = np.clip(
            z_offsets[out_of_range], -MAX_Z_SHIFT * 0.5, MAX_Z_SHIFT * 0.5
        )

        vertices[indices, 2] += z_offsets * height_blend

        corrected_count += 1
        print(f"    #{tooth_id}: 近中{dz_msl:+.2f} 中間{dz_mid:+.2f} 遠中{dz_dst:+.2f} mm")

    print(f"  ✓ 切緣輪廓修正: {corrected_count} 顆")
    result_mesh = mesh.copy()
    result_mesh.vertices = vertices
    return result_mesh

def apply_asymmetric_width(mesh, seg_labels, analysis_data, teeth_list):
    print("\n📐 前牙近遠中非對稱修正...")

    ASYM_TEETH = {11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43}
    BLEND_STRENGTH = 0.5
    MAX_SCALE_DIFF = 0.15

    vertices = mesh.vertices.copy()
    corrected_count = 0

    for tooth_id in teeth_list:
        if tooth_id not in ASYM_TEETH:
            continue

        tooth_data = analysis_data['teeth'].get(str(tooth_id))
        if tooth_data is None:
            continue

        front_cp = None
        for m in tooth_data.get('measurements', []):
            if m.get('view') == 'front' and m.get('contour_cp'):
                front_cp = m['contour_cp']['points']
                break
        if front_cp is None:
            continue

        x_msl = front_cp.get('mesial_top', {}).get('x')
        x_dst = front_cp.get('distal_top', {}).get('x')
        x_mid_cp = front_cp.get('incisal_mid', {}).get('x')

        if x_msl is None or x_dst is None or x_mid_cp is None:
            continue

        mesial_half = abs(x_mid_cp - x_msl)
        distal_half = abs(x_dst - x_mid_cp)
        total = mesial_half + distal_half

        if total < 0.5:
            continue

        mesial_ratio = mesial_half / total
        distal_ratio = distal_half / total
        asymmetry = mesial_ratio - 0.5

        if mesial_ratio > 0.85 or distal_ratio > 0.85:
            print(f"    #{tooth_id}: 跳過（cp異常 近中{mesial_ratio:.2f}/遠中{distal_ratio:.2f}）")
            continue

        scale_mesial = 1.0 - asymmetry * BLEND_STRENGTH
        scale_distal = 1.0 + asymmetry * BLEND_STRENGTH

        scale_mesial = np.clip(scale_mesial, 1.0 - MAX_SCALE_DIFF, 1.0 + MAX_SCALE_DIFF)
        scale_distal = np.clip(scale_distal, 1.0 - MAX_SCALE_DIFF, 1.0 + MAX_SCALE_DIFF)

        indices = np.nonzero(seg_labels == tooth_id)[0]
        if len(indices) == 0:
            continue

        tooth_verts = vertices[indices]
        x_center = tooth_verts[:, 0].mean()

        rel_x = tooth_verts[:, 0] - x_center
        mesial_mask = rel_x < 0
        distal_mask = rel_x >= 0

        vertices[indices[mesial_mask], 0] = x_center + rel_x[mesial_mask] * scale_mesial
        vertices[indices[distal_mask], 0] = x_center + rel_x[distal_mask] * scale_distal

        corrected_count += 1
        print(f"    #{tooth_id}: 近中{mesial_ratio:.2f} 遠中{distal_ratio:.2f} → 縮放 {scale_mesial:.3f}/{scale_distal:.3f}")

    print(f"  ✓ 近遠中非對稱修正: {corrected_count} 顆")
    result_mesh = mesh.copy()
    result_mesh.vertices = vertices
    return result_mesh

def apply_zlayer_width_profile(mesh, seg_labels, analysis_data, teeth_list):
    print("\n📐 Z-layer 寬度剖面修正...")

    PROFILE_TEETH = {11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43}
    BLEND_STRENGTH = 0.5
    MAX_SCALE = 1.20
    MIN_SCALE = 0.80

    vertices = mesh.vertices.copy()
    corrected_count = 0

    for tooth_id in teeth_list:
        if tooth_id not in PROFILE_TEETH:
            continue

        tooth_data = analysis_data['teeth'].get(str(tooth_id))
        if tooth_data is None:
            continue

        front_cp = None
        for m in tooth_data.get('measurements', []):
            if m.get('view') == 'front' and m.get('contour_cp'):
                front_cp = m['contour_cp']['points']
                break
        if front_cp is None:
            continue

        def get_pt(name):
            p = front_cp.get(name, {})
            x = p.get('x')
            z = p.get('z_rel')
            return (x, z) if x is not None and z is not None else None

        p_mn = get_pt('mesial_neck')
        p_mt = get_pt('mesial_third')
        p_mp = get_pt('mesial_top')
        p_dn = get_pt('distal_neck')
        p_dt = get_pt('distal_third')
        p_dp = get_pt('distal_top')

        if any(p is None for p in [p_mn, p_mt, p_mp, p_dn, p_dt, p_dp]):
            continue

        z_neck  = (p_mn[1] + p_dn[1]) / 2.0
        z_third = (p_mt[1] + p_dt[1]) / 2.0
        z_top   = (p_mp[1] + p_dp[1]) / 2.0

        x_mesial_at_z = np.array([p_mn[0], p_mt[0], p_mp[0]])
        x_distal_at_z = np.array([p_dn[0], p_dt[0], p_dp[0]])
        z_layers = np.array([z_neck, z_third, z_top])

        sort_idx = np.argsort(z_layers)
        z_layers = z_layers[sort_idx]
        x_mesial_at_z = x_mesial_at_z[sort_idx]
        x_distal_at_z = x_distal_at_z[sort_idx]

        half_width_mesial = np.abs(x_mesial_at_z)
        half_width_distal = np.abs(x_distal_at_z)

        indices = np.nonzero(seg_labels == tooth_id)[0]
        if len(indices) == 0:
            continue

        tooth_verts = vertices[indices]
        z_min = tooth_verts[:, 2].min()
        z_max = tooth_verts[:, 2].max()
        z_range = z_max - z_min
        if z_range < 0.1:
            continue

        x_center = tooth_verts[:, 0].mean()

        z_cp_min = z_layers.min()
        z_cp_max = z_layers.max()
        z_cp_range = z_cp_max - z_cp_min + 1e-6

        def cp_z_to_model_z(z_cp):
            ratio = (z_cp - z_cp_min) / z_cp_range
            return z_min + ratio * z_range

        model_z_layers = np.array([cp_z_to_model_z(z) for z in z_layers])

        x_min_tooth = tooth_verts[:, 0].min()
        x_max_tooth = tooth_verts[:, 0].max()
        current_mesial_half = x_center - x_min_tooth
        current_distal_half = x_max_tooth - x_center

        if current_mesial_half < 0.1 or current_distal_half < 0.1:
            continue

        verts_z = tooth_verts[:, 2]
        verts_x_rel = tooth_verts[:, 0] - x_center

        target_mesial = np.interp(verts_z, model_z_layers, half_width_mesial)
        target_distal = np.interp(verts_z, model_z_layers, half_width_distal)

        scale_mesial = target_mesial / current_mesial_half
        scale_distal = target_distal / current_distal_half

        scale_mesial = 1.0 + (scale_mesial - 1.0) * BLEND_STRENGTH
        scale_distal = 1.0 + (scale_distal - 1.0) * BLEND_STRENGTH

        scale_mesial = np.clip(scale_mesial, MIN_SCALE, MAX_SCALE)
        scale_distal = np.clip(scale_distal, MIN_SCALE, MAX_SCALE)

        mesial_mask = verts_x_rel < 0
        distal_mask = verts_x_rel >= 0

        new_x = tooth_verts[:, 0].copy()
        new_x[mesial_mask] = x_center + verts_x_rel[mesial_mask] * scale_mesial[mesial_mask]
        new_x[distal_mask] = x_center + verts_x_rel[distal_mask] * scale_distal[distal_mask]

        vertices[indices, 0] = new_x
        corrected_count += 1

        sm_neck  = 1.0 + (half_width_mesial[0] / current_mesial_half - 1.0) * BLEND_STRENGTH
        sm_top   = 1.0 + (half_width_mesial[-1] / current_mesial_half - 1.0) * BLEND_STRENGTH
        sd_neck  = 1.0 + (half_width_distal[0] / current_distal_half - 1.0) * BLEND_STRENGTH
        sd_top   = 1.0 + (half_width_distal[-1] / current_distal_half - 1.0) * BLEND_STRENGTH
        sm_neck  = np.clip(sm_neck, MIN_SCALE, MAX_SCALE)
        sm_top   = np.clip(sm_top,  MIN_SCALE, MAX_SCALE)
        sd_neck  = np.clip(sd_neck, MIN_SCALE, MAX_SCALE)
        sd_top   = np.clip(sd_top,  MIN_SCALE, MAX_SCALE)
        print(f"    #{tooth_id}: neck×{sm_neck:.2f}/{sd_neck:.2f}  top×{sm_top:.2f}/{sd_top:.2f}")

    print(f"  ✓ Z-layer 寬度修正: {corrected_count} 顆")
    result_mesh = mesh.copy()
    result_mesh.vertices = vertices
    return result_mesh

# ==================== 主流程 ====================

print("\n📐 步驟 1: 基礎對齊...")

scaled_upper_base = upper_mesh.copy()
scaled_lower_base = lower_mesh.copy()

scaled_upper_base, scaled_lower_base = align_upper_lower(
    scaled_upper_base, scaled_lower_base,
    flip_upper=UPPER_FLIP,
    rotate_180=UPPER_ROTATE_180,
    spacing=VERTICAL_SPACING,
    opening_angle=UPPER_JAW_OPENING_ANGLE,
    pivot_ratio=ROTATION_PIVOT_RATIO,
    offset_x=UPPER_POSITION_OFFSET_X,
    offset_y=UPPER_POSITION_OFFSET_Y,
    offset_z=UPPER_POSITION_OFFSET_Z
)

print(f"  ✓ 對齊完成")

print("\n📐 步驟 2: 客製化 3D 縮放...")

upper_teeth = [t for t in detected_teeth if t < 30 and t not in never_detected]
lower_teeth = [t for t in detected_teeth if t >= 30 and t not in never_detected]

upper_scales = compute_customized_3d_scales(
    scaled_upper_base, upper_seg_labels, analysis, upper_teeth
)

lower_scales = compute_customized_3d_scales(
    scaled_lower_base, lower_seg_labels, analysis, lower_teeth
)

scaled_upper_custom = apply_customized_scaling(
    scaled_upper_base, upper_seg_labels, upper_scales
)
scaled_upper_custom = normalize_anterior_incisal_height(
    scaled_upper_custom, upper_seg_labels, analysis, upper_teeth
)
scaled_upper_custom = compress_arch_spacing(scaled_upper_custom, upper_seg_labels, ARCH_COMPRESSION_RATIO_UPPER)
scaled_upper_custom = apply_tilt_correction(
    scaled_upper_custom, upper_seg_labels, analysis, upper_teeth
)
scaled_upper_custom = apply_incisal_curvature(
    scaled_upper_custom, upper_seg_labels, analysis, upper_teeth
)
scaled_upper_custom = apply_asymmetric_width(
    scaled_upper_custom, upper_seg_labels, analysis, upper_teeth
)
# scaled_upper_custom = apply_zlayer_width_profile(
#     scaled_upper_custom, upper_seg_labels, analysis, upper_teeth
# )


scaled_lower_custom = apply_customized_scaling(
    scaled_lower_base, lower_seg_labels, lower_scales
)
scaled_lower_custom = compress_arch_spacing(scaled_lower_custom, lower_seg_labels, ARCH_COMPRESSION_RATIO_LOWER)
scaled_lower_custom = apply_tilt_correction(
    scaled_lower_custom, lower_seg_labels, analysis, lower_teeth
)
scaled_lower_custom = apply_incisal_curvature(
    scaled_lower_custom, lower_seg_labels, analysis, lower_teeth
)
scaled_lower_custom = normalize_anterior_incisal_height(
    scaled_lower_custom, lower_seg_labels, analysis, lower_teeth
)
scaled_lower_custom = apply_asymmetric_width(
    scaled_lower_custom, lower_seg_labels, analysis, lower_teeth
)
# scaled_lower_custom = apply_zlayer_width_profile(
#     scaled_lower_custom, lower_seg_labels, analysis, lower_teeth
# )

print(f"  ✓ 牙弓壓縮: 上顎 {ARCH_COMPRESSION_RATIO_UPPER}x, 下顎 {ARCH_COMPRESSION_RATIO_LOWER}x")

all_tooth_scales = {**upper_scales, **lower_scales}

# ==================== 生成模型 ====================

print("\n🦷 生成客製化模型...")

upper_to_remove = [t for t in never_detected if t < 30]
lower_to_remove = [t for t in never_detected if t >= 30]

upper_v, upper_f = remove_teeth_from_mesh(
    scaled_upper_custom.vertices, scaled_upper_custom.faces,
    upper_seg_labels, upper_to_remove
)
lower_v, lower_f = remove_teeth_from_mesh(
    scaled_lower_custom.vertices, scaled_lower_custom.faces,
    lower_seg_labels, lower_to_remove
)

upper_mesh_final = trimesh.Trimesh(vertices=upper_v, faces=upper_f, process=True)
upper_mesh_final.remove_degenerate_faces()
upper_mesh_final.remove_duplicate_faces()
lower_mesh_final = trimesh.Trimesh(vertices=lower_v, faces=lower_f, process=True)
lower_mesh_final.remove_degenerate_faces()
lower_mesh_final.remove_duplicate_faces()

full_mesh = trimesh.util.concatenate([upper_mesh_final, lower_mesh_final])

output_file = OUTPUT_DIR / "custom_real_teeth.obj"
full_mesh.export(str(output_file))

print(f"  ✓ 已保存: {output_file.name}")

# 保存報告
overall_scale_upper = upper_scales[list(upper_scales.keys())[0]]['overall_scale'] if upper_scales else 1.0
overall_scale_lower = lower_scales[list(lower_scales.keys())[0]]['overall_scale'] if lower_scales else 1.0

report = {
    'version': 'real_teeth',
    'teeth_info': {
        'detected': sorted(detected_teeth),
        'never_detected': never_detected,
        'adjusted_count': sum(1 for s in all_tooth_scales.values() if s.get('has_adjustment', False))
    },
    'overall_scale': {
        'upper': float(overall_scale_upper),
        'lower': float(overall_scale_lower)
    },
    'tooth_scales': {
        str(k): {
            'scale_x': float(v['scale_x']),
            'scale_y': float(v['scale_y']),
            'scale_z': float(v['scale_z']),
            'overall_scale': float(v.get('overall_scale', 1.0))
        } for k, v in all_tooth_scales.items() if v.get('has_adjustment', False)
    }
}

with open(OUTPUT_DIR / "report_real_teeth.json", 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print("✅ 客製化模型完成！")
print("="*70)

print(f"\n📊 生成摘要:")
print(f"  檢測到: {len(detected_teeth)} 顆")
print(f"  從未出現: {len(never_detected)} 顆")
print(f"  客製化調整: {sum(1 for s in all_tooth_scales.values() if s.get('has_adjustment', False))} 顆")
print(f"  整體縮放: 上顎 {overall_scale_upper:.3f}x, 下顎 {overall_scale_lower:.3f}x")

print(f"\n📁 輸出檔案:")
size = output_file.stat().st_size / (1024 * 1024)
print(f"  • {output_file.name} ({size:.1f} MB)")

# ==================== [NEW] 額外輸出分離客製化模型 + seg_labels ====================
print("\n💾 [NEW] 輸出分離客製化模型 + seg_labels...")

# 上顎客製化模型（remove_teeth 之前的完整版，頂點順序與 upper_seg_labels 對應）
upper_only_path = OUTPUT_DIR / "custom_upper_only.obj"
scaled_upper_custom.export(str(upper_only_path))
print(f"  ✓ 上顎客製化模型: {upper_only_path.name}  ({len(scaled_upper_custom.vertices):,} 頂點)")

# 下顎客製化模型
lower_only_path = OUTPUT_DIR / "custom_lower_only.obj"
scaled_lower_custom.export(str(lower_only_path))
print(f"  ✓ 下顎客製化模型: {lower_only_path.name}  ({len(scaled_lower_custom.vertices):,} 頂點)")

# seg_labels（FDI 標籤陣列，與頂點順序一對一對應）
upper_labels_path = OUTPUT_DIR / "upper_seg_labels.npy"
lower_labels_path = OUTPUT_DIR / "lower_seg_labels.npy"
np.save(str(upper_labels_path), upper_seg_labels)
np.save(str(lower_labels_path), lower_seg_labels)
print(f"  ✓ 上顎 seg_labels: {upper_labels_path.name}  FDI: {sorted(set(upper_seg_labels[upper_seg_labels>0]))}")
print(f"  ✓ 下顎 seg_labels: {lower_labels_path.name}  FDI: {sorted(set(lower_seg_labels[lower_seg_labels>0]))}")

# 驗證頂點數量一致
assert len(scaled_upper_custom.vertices) == len(upper_seg_labels), \
    f"上顎頂點數 {len(scaled_upper_custom.vertices)} != seg_labels {len(upper_seg_labels)}"
assert len(scaled_lower_custom.vertices) == len(lower_seg_labels), \
    f"下顎頂點數 {len(scaled_lower_custom.vertices)} != seg_labels {len(lower_seg_labels)}"
print(f"  ✅ 頂點數量與 seg_labels 一致驗證通過")

# 同時輸出 GLB 供網頁展示
glb_path = OUTPUT_DIR / "custom_real_teeth.glb"
full_mesh.export(str(glb_path))
print(f"  ✓ GLB 網頁展示版: {glb_path.name}")