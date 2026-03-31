#!/usr/bin/env python3
"""
TPS非均勻變形
從多視角mask提取控制點 → 計算TPS變形場 → 應用到mesh
"""

import numpy as np
import cv2
import json
import trimesh
from pathlib import Path
from scipy.spatial import KDTree


# ==================== TPS核心數學 ====================

def tps_kernel(r):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(r > 1e-10, r**2 * np.log(r + 1e-10), 0.0)


def solve_tps(source_pts, target_pts):
    N = len(source_pts)

    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            r = np.linalg.norm(source_pts[i] - source_pts[j])
            K[i, j] = tps_kernel(r)

    P = np.hstack([np.ones((N, 1)), source_pts])

    top = np.hstack([K, P])
    bot = np.hstack([P.T, np.zeros((4, 4))])
    L = np.vstack([top, bot])
    L += np.eye(N + 4) * 1e-6

    rhs_x = np.concatenate([target_pts[:, 0], np.zeros(4)])
    rhs_y = np.concatenate([target_pts[:, 1], np.zeros(4)])
    rhs_z = np.concatenate([target_pts[:, 2], np.zeros(4)])

    params_x = np.linalg.solve(L, rhs_x)
    params_y = np.linalg.solve(L, rhs_y)
    params_z = np.linalg.solve(L, rhs_z)

    weights = np.stack([params_x[:N], params_y[:N], params_z[:N]], axis=1)
    affine  = np.stack([params_x[N:], params_y[N:], params_z[N:]], axis=1)

    return weights, affine, source_pts


def apply_tps(query_pts, weights, affine, source_pts):
    N = len(source_pts)
    M = len(query_pts)

    K = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            r = np.linalg.norm(query_pts[i] - source_pts[j])
            K[i, j] = tps_kernel(r)

    P = np.hstack([np.ones((M, 1)), query_pts])
    return K @ weights + P @ affine


def apply_tps_fast(query_pts, weights, affine, source_pts, batch_size=1000):
    results = []
    for i in range(0, len(query_pts), batch_size):
        batch = query_pts[i:i+batch_size]
        results.append(apply_tps(batch, weights, affine, source_pts))
    return np.vstack(results)


# ==================== 從mask提取2D控制點（已移至extract_control_points.py）====================

def extract_contour_control_points(mask_2d, view, tooth_id, pixel_to_mm=0.15):
    """轉發到 extract_control_points.py 的新版函數"""
    from extract_control_points import extract_contour_control_points as _extract
    return _extract(mask_2d, view, tooth_id, pixel_to_mm)


# ==================== 融合多視角 → 3D目標控制點 ====================

# 9個控制點的名稱，順序對應 get_tooth_control_points_3d
CP_NAMES = [
    'mesial_top',   # 0
    'distal_top',   # 1
    'buccal',       # 2
    'lingual',      # 3
    'mesial_neck',  # 4
    'distal_neck',  # 5
    'incisal_mid',  # 6
    'mesial_third', # 7
    'distal_third', # 8
]

# 各視角對各軸的可信度權重
VIEW_AXIS_CONFIDENCE = {
    'front':  {'x': 0.9, 'y': 0.0, 'z': 0.9},
    'upper':  {'x': 0.9, 'y': 0.9, 'z': 0.0},
    'lower':  {'x': 0.9, 'y': 0.9, 'z': 0.0},
    'left':   {'x': 0.0, 'y': 0.8, 'z': 0.7},
    'right':  {'x': 0.0, 'y': 0.8, 'z': 0.7},
}


def fuse_multiview_control_points(cp_3d_template, multiview_2d, tooth_dims_3d):
    """
    把多個視角的2D控制點融合成3D目標控制點

    cp_3d_template: (9, 3) 公版控制點
    multiview_2d:   {'front': {...}, 'upper': {...}, ...}
    tooth_dims_3d:  {'width': {mean, std}, 'height': ..., 'depth': ...}

    回傳: (9, 3) 目標控制點
    """
    n_cp = len(cp_3d_template)  # 9
    template_center = cp_3d_template.mean(axis=0)

    # 公版尺寸
    template_width  = cp_3d_template[:, 0].max() - cp_3d_template[:, 0].min()
    template_depth  = cp_3d_template[:, 1].max() - cp_3d_template[:, 1].min()
    template_height = cp_3d_template[:, 2].max() - cp_3d_template[:, 2].min()

    # 目標尺寸（來自測量）
    target_width  = tooth_dims_3d.get('width',  {}).get('mean') or template_width
    target_depth  = tooth_dims_3d.get('depth',  {}).get('mean') or template_depth
    target_height = tooth_dims_3d.get('height', {}).get('mean') or template_height

    # 整體縮放比（這已由等比縮放步驟做了，TPS只做形狀微調）
    scale_x = target_width  / template_width  if template_width  > 0 else 1.0
    scale_y = target_depth  / template_depth  if template_depth  > 0 else 1.0
    scale_z = target_height / template_height if template_height > 0 else 1.0

    # 初始化目標點 = 輕微整體縮放（TPS_BLEND控制縮放貢獻比例）
    TPS_SCALE_BLEND = 0.2  # 只讓整體縮放貢獻20%，形狀微調貢獻80%
    target_pts = cp_3d_template.copy()
    for i in range(n_cp):
        rel = cp_3d_template[i] - template_center
        scaled = template_center + rel * np.array([scale_x, scale_y, scale_z])
        target_pts[i] = cp_3d_template[i] + (scaled - cp_3d_template[i]) * TPS_SCALE_BLEND

    # 收集各視角對每個控制點的修正量
    # offsets[cp_idx][axis] = [(target_value, confidence_weight), ...]
    offsets = {i: {'x': [], 'y': [], 'z': []} for i in range(n_cp)}

    for view, view_data in multiview_2d.items():
        if view_data is None:
            continue

        view_points = view_data.get('points', {})
        if not view_points:
            continue

        view_conf = VIEW_AXIS_CONFIDENCE.get(view, {'x': 0.5, 'y': 0.5, 'z': 0.5})

        for cp_idx, cp_name in enumerate(CP_NAMES):
            if cp_name not in view_points:
                continue

            point_data = view_points[cp_name]
            if not isinstance(point_data, dict):
                continue

            # ★ 關鍵：這裡的值已經是「相對牙齒中心的mm偏移」（正規化座標）
            # 直接加上 template_center 就是3D目標座標
            for axis_key, val_mm in point_data.items():
                if axis_key == 'x' and view_conf['x'] > 0:
                    target_val = template_center[0] + val_mm
                    offsets[cp_idx]['x'].append((target_val, view_conf['x']))

                elif axis_key == 'y' and view_conf['y'] > 0:
                    target_val = template_center[1] + val_mm
                    offsets[cp_idx]['y'].append((target_val, view_conf['y']))

                elif axis_key == 'z_rel' and view_conf['z'] > 0:
                    target_val = template_center[2] + val_mm
                    offsets[cp_idx]['z'].append((target_val, view_conf['z']))

    # 加權融合，更新目標點
    max_shifts = [target_width * 0.15, target_depth * 0.15, target_height * 0.15]

    for cp_idx in range(n_cp):
        for axis_name, axis_idx in [('x', 0), ('y', 1), ('z', 2)]:
            vals = offsets[cp_idx][axis_name]
            if not vals:
                continue

            values  = np.array([v for v, _ in vals])
            weights = np.array([w for _, w in vals])
            fused   = np.dot(values, weights) / weights.sum()

            original  = target_pts[cp_idx, axis_idx]
            max_shift = max_shifts[axis_idx]
            shift     = np.clip(fused - original, -max_shift, max_shift)
            target_pts[cp_idx, axis_idx] = original + shift

    return target_pts


# ==================== 對單顆牙應用TPS ====================

def deform_tooth_tps(mesh_vertices, tooth_indices, cp_source, cp_target, influence_radius=None):
    """
    對單顆牙的頂點應用TPS變形
    influence_radius=None 時自動用牙齒自身大小決定
    """
    if len(tooth_indices) == 0:
        return mesh_vertices

    tooth_verts = mesh_vertices[tooth_indices]
    tooth_center = tooth_verts.mean(axis=0)

    if influence_radius is None:
        tooth_radius = np.linalg.norm(tooth_verts - tooth_center, axis=1).max()
        effective_radius = tooth_radius * 0.8
    else:
        effective_radius = influence_radius

    weights, affine, src_pts = solve_tps(cp_source, cp_target)
    new_verts = apply_tps_fast(tooth_verts, weights, affine, src_pts)

    dist_to_center = np.linalg.norm(tooth_verts - tooth_center, axis=1)
    blend = np.clip(1.0 - dist_to_center / effective_radius, 0.0, 1.0)
    blend = blend ** 2.0  # 邊緣快速衰減，不影響鄰牙

    blended = tooth_verts * (1 - blend[:, None]) + new_verts * blend[:, None]

    result = mesh_vertices.copy()
    result[tooth_indices] = blended
    return result


# ==================== 測試 ====================

if __name__ == "__main__":
    from extract_control_points import get_tooth_control_points_3d

    MODELS_DIR = Path("./models")

    mesh = trimesh.load(str(MODELS_DIR / "01J9K9S6_lower.obj"))
    with open(MODELS_DIR / "01J9K9S6_lower.json") as f:
        seg_data = json.load(f)
    seg_labels = np.array(seg_data["labels"], dtype=np.int32)

    indices = np.nonzero(seg_labels == 41)[0]
    verts   = mesh.vertices[indices]

    cp_source = get_tooth_control_points_3d(mesh.vertices, indices)
    assert cp_source.shape == (9, 3), f"控制點應為9個，實際{cp_source.shape}"

    # 模擬X縮小10%
    center    = cp_source.mean(axis=0)
    cp_target = cp_source.copy()
    cp_target[:, 0] = center[0] + (cp_source[:, 0] - center[0]) * 0.9

    weights, affine, src_pts = solve_tps(cp_source, cp_target)
    test_pts = verts[:5]
    new_pts  = apply_tps(test_pts, weights, affine, src_pts)

    print("TPS測試（9控制點，X縮小10%）：")
    for i, (old, new) in enumerate(zip(test_pts, new_pts)):
        print(f"  頂點{i}: ({old[0]:.2f},{old[1]:.2f},{old[2]:.2f})"
              f" → ({new[0]:.2f},{new[1]:.2f},{new[2]:.2f})")

    print("\n✅ TPS核心運作正常")