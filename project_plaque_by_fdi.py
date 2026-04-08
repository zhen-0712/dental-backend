#!/usr/bin/env python3
"""
project_plaque_by_fdi.py  ── SAT mask UV 對應版（優化版）

優化項目：
  - save_debug_projection: 改用 numpy 向量化繪點（避免逐點 Python 迴圈）
  - get_plaque_hit_verts: 批次 boolean indexing
  - run_sat_for_view: 不變（GPU 操作）
  - 主流程: 預先建立 fdi_map 時用 np.where 向量化，染色步驟向量化
"""

import cv2
import numpy as np
import trimesh
import json
from pathlib import Path

import sys; sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")
from user_env import get_paths, setup_user_dirs, BASE as _SAT_BASE
_PATHS = get_paths()
setup_user_dirs(_PATHS["user_dir"])
BASE         = _PATHS["user_dir"]
ROI_MASK_DIR = _PATHS["plaque_output"]
MODEL_DIR    = _PATHS["model_dir"]
OUTPUT_DIR   = _PATHS["plaque_output"]
WEIGHT_DIR   = _SAT_BASE / "weight"

UPPER_OBJ    = MODEL_DIR / "custom_upper_only.obj"
LOWER_OBJ    = MODEL_DIR / "custom_lower_only.obj"
UPPER_LABELS = MODEL_DIR / "upper_seg_labels.npy"
LOWER_LABELS = MODEL_DIR / "lower_seg_labels.npy"
MODEL_PATH   = MODEL_DIR / "custom_real_teeth.obj"

COLOR_NORMAL = np.array([0.92, 0.86, 0.80])

VIEW_CONFIG = {
    'front': {
        'sat_view': 'front', 'roi_mask': 'roi_mask_front.png',
        'photo_file': 'real_teeth_processed/front.jpg',
        'proj_u_axis': 0, 'proj_v_axis': 2,
        'flip_u': True,  'flip_v': True,
        'scale_u': 1.0,  'scale_v': 0.55,
        'offset_u': 0.0, 'offset_v': -1.05,
        'vert_clip_pct': 5,
    },
    'left_side': {
        'sat_view': 'left', 'roi_mask': 'roi_mask_left_side.png',
        'photo_file': 'real_teeth_processed/left_side.jpg',
        'proj_u_axis': 1, 'proj_v_axis': 2,
        'flip_u': False, 'flip_v': True,
        'scale_u': 1.0,  'scale_v': 0.55,
        'offset_u': -0.45, 'offset_v': -0.60,
        'vert_clip_pct': 5,
    },
    'right_side': {
        'sat_view': 'right', 'roi_mask': 'roi_mask_right_side.png',
        'photo_file': 'real_teeth_processed/right_side.jpg',
        'proj_u_axis': 1, 'proj_v_axis': 2,
        'flip_u': True,  'flip_v': True,
        'scale_u': 1.0,  'scale_v': 0.55,
        'offset_u': -0.45, 'offset_v': -0.20,
        'vert_clip_pct': 10,
    },
    'upper_occlusal': {
        'sat_view': 'upper', 'roi_mask': 'roi_mask_upper_occlusal.png',
        'photo_file': 'real_teeth_processed/upper_occlusal.jpg',
        'proj_u_axis': 0, 'proj_v_axis': 1,
        'flip_u': True,  'flip_v': True,
        'scale_u': 1.6,  'scale_v': 1.15,
        'offset_u': 0.0, 'offset_v': 0.0,
        'vert_clip_pct': 5,
    },
    'lower_occlusal': {
        'sat_view': 'lower', 'roi_mask': 'roi_mask_lower_occlusal.png',
        'photo_file': 'real_teeth_processed/lower_occlusal.jpg',
        'proj_u_axis': 0, 'proj_v_axis': 1,
        'flip_u': True,  'flip_v': False,
        'scale_u': 1.0,  'scale_v': 1.0,
        'offset_u': 0.0, 'offset_v': 0.0,
        'vert_clip_pct': 5,
    },
}


# ==================== 工具函式 ====================

def build_fdi_map(upper_verts, upper_labels, lower_verts, lower_labels):
    fdi_map = {}
    for fdi in np.unique(upper_labels):
        if fdi == 0: continue
        idx = np.where(upper_labels == fdi)[0]
        fdi_map[int(fdi)] = ('upper', upper_verts[idx], idx)
    for fdi in np.unique(lower_labels):
        if fdi == 0: continue
        idx = np.where(lower_labels == fdi)[0]
        fdi_map[int(fdi)] = ('lower', lower_verts[idx], idx)
    return fdi_map


def get_sat_bbox(fdi_mask_sat, fdi):
    sat_h, sat_w = fdi_mask_sat.shape
    ys, xs = np.where(fdi_mask_sat == fdi)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()),
            max(int(xs.max() - xs.min()), 1),
            max(int(ys.max() - ys.min()), 1),
            sat_w, sat_h)


def clip_tooth_verts(tooth_verts, clip_pct):
    if clip_pct <= 0:
        return tooth_verts
    lo, hi = clip_pct, 100 - clip_pct
    clipped = tooth_verts.copy()
    for ax in range(3):
        v = tooth_verts[:, ax]
        clipped[:, ax] = np.clip(v, np.percentile(v, lo), np.percentile(v, hi))
    return clipped


def project_tooth_verts(tooth_verts, cfg, sat_bbox):
    u_ax = cfg['proj_u_axis']
    v_ax = cfg['proj_v_axis']
    px_min, py_min, bbox_w, bbox_h, sat_w, sat_h = sat_bbox

    clip_pct  = cfg.get('vert_clip_pct', 0)
    ref_verts = clip_tooth_verts(tooth_verts, clip_pct)

    u_vals = tooth_verts[:, u_ax];  v_vals = tooth_verts[:, v_ax]
    u_ref  = ref_verts[:, u_ax];    v_ref  = ref_verts[:, v_ax]

    u_range = max(u_ref.max() - u_ref.min(), 1e-6)
    v_range = max(v_ref.max() - v_ref.min(), 1e-6)

    u_norm = (u_vals - u_ref.min()) / u_range
    v_norm = (v_vals - v_ref.min()) / v_range

    if cfg['flip_u']: u_norm = 1.0 - u_norm
    if cfg['flip_v']: v_norm = 1.0 - v_norm

    su = cfg.get('scale_u', 1.0);  sv = cfg.get('scale_v', 1.0)
    ou = cfg.get('offset_u', 0.0); ov = cfg.get('offset_v', 0.0)
    u_adj = (u_norm - 0.5) * su + 0.5 + ou
    v_adj = (v_norm - 0.5) * sv + 0.5 + ov

    px = np.clip((u_adj * bbox_w + px_min).astype(np.int32), 0, sat_w - 1)
    py = np.clip((v_adj * bbox_h + py_min).astype(np.int32), 0, sat_h - 1)
    return px, py


def get_plaque_hit_verts(fdi, tooth_verts, tooth_indices, fdi_mask_sat, roi_resized, cfg):
    sat_bbox = get_sat_bbox(fdi_mask_sat, fdi)
    if sat_bbox is None:
        return np.array([], dtype=np.int64)
    px, py = project_tooth_verts(tooth_verts, cfg, sat_bbox)
    hit_mask = roi_resized[py, px] > 0
    return tooth_indices[hit_mask]

def _get_plaque_hit_verts_with_bbox(tooth_verts, tooth_indices, roi_resized, cfg, sat_bbox):
    """已有 sat_bbox 時直接使用，不重複查找"""
    px, py = project_tooth_verts(tooth_verts, cfg, sat_bbox)
    hit_mask = roi_resized[py, px] > 0
    return tooth_indices[hit_mask]


def save_debug_projection(view_name, cfg, fdi_mask_sat, roi_mask, fdi_map, debug_dir):
    """
    優化版：改用 numpy 向量化繪點取代逐點 cv2.circle Python 迴圈
    大幅減少當牙齒頂點多（>1000）時的繪圖時間
    """
    sat_h, sat_w = fdi_mask_sat.shape
    roi_r = cv2.resize(roi_mask, (sat_w, sat_h), interpolation=cv2.INTER_NEAREST) \
            if roi_mask.shape[:2] != (sat_h, sat_w) else roi_mask.copy()

    img = np.zeros((sat_h, sat_w, 3), dtype=np.uint8)
    img[fdi_mask_sat > 0] = [60, 60, 60]
    img[roi_r > 0] = [255, 255, 255]

    for fdi in sorted(int(v) for v in np.unique(fdi_mask_sat) if v > 0):
        if fdi not in fdi_map:
            continue
        _, tooth_verts, _ = fdi_map[fdi]
        sat_bbox = get_sat_bbox(fdi_mask_sat, fdi)
        if sat_bbox is None:
            continue
        px_min, py_min, bw, bh, sw, sh = sat_bbox
        px, py = project_tooth_verts(tooth_verts, cfg, sat_bbox)

        # ★ 向量化繪點：隨機抽樣後直接 numpy 賦值，不用 cv2.circle 迴圈
        n = len(px)
        if n > 0:
            sidx = np.random.choice(n, min(200, n), replace=False)
            # 把每個採樣點周圍 1px 設為綠色（模擬 radius=1 circle）
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    gy = np.clip(py[sidx] + dy, 0, sat_h - 1)
                    gx = np.clip(px[sidx] + dx, 0, sat_w - 1)
                    img[gy, gx] = [0, 200, 0]

        # 紅：命中菌斑（向量化）
        hit = roi_r[py, px] > 0
        hpx, hpy = px[hit], py[hit]
        if len(hpx) > 0:
            hidx = np.random.choice(len(hpx), min(100, len(hpx)), replace=False)
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ry = np.clip(hpy[hidx] + dy, 0, sat_h - 1)
                    rx = np.clip(hpx[hidx] + dx, 0, sat_w - 1)
                    img[ry, rx] = [0, 0, 255]

        # FDI 標籤 + bbox
        cv2.putText(img, str(fdi),
                    (px_min + bw // 2 - 8, py_min + bh // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(img, (px_min, py_min),
                      (px_min + bw, py_min + bh), (200, 100, 0), 1)

    out = debug_dir / f"debug_proj_{view_name}.png"
    cv2.imwrite(str(out), img)
    print(f"    🖼️  {out.name}  [綠=投影, 紅=命中, 白=菌斑, 藍框=SAT bbox]")


def run_sat_for_view(view_name, cfg):
    photo_path = BASE / cfg['photo_file']
    if not photo_path.exists():
        print(f"    ⚠️  找不到照片: {cfg['photo_file']}")
        return None
    try:
        from segmentanytooth import predict
        fdi_mask = predict(
            image_path=str(photo_path),
            view=cfg['sat_view'],
            weight_dir=str(WEIGHT_DIR),
            sam_batch_size=10
        )
        print(f"    SAT 偵測 FDI: {sorted(int(v) for v in np.unique(fdi_mask) if v > 0)}")
        return fdi_mask
    except Exception as e:
        print(f"    ⚠️  SAT 失敗: {e}")
        return None


# ==================== 主流程 ====================
print("=" * 60)
print("🦷 菌斑投射 → 客製化 3D 模型（優化版）")
print("=" * 60)

print("\n📦 載入客製化模型...")
upper_mesh   = trimesh.load(str(UPPER_OBJ))
lower_mesh   = trimesh.load(str(LOWER_OBJ))
upper_labels = np.load(str(UPPER_LABELS))
lower_labels = np.load(str(LOWER_LABELS))

upper_verts = np.array(upper_mesh.vertices)
lower_verts = np.array(lower_mesh.vertices)
n_upper = len(upper_verts)
n_lower = len(lower_verts)

assert len(upper_labels) == n_upper
assert len(lower_labels) == n_lower

print(f"  上顎: {n_upper:,} 頂點  FDI: {sorted(set(upper_labels[upper_labels>0]))}")
print(f"  下顎: {n_lower:,} 頂點  FDI: {sorted(set(lower_labels[lower_labels>0]))}")

fdi_map = build_fdi_map(upper_verts, upper_labels, lower_verts, lower_labels)
print(f"  FDI 映射: {len(fdi_map)} 顆牙  {sum(len(v[2]) for v in fdi_map.values()):,} 頂點")

upper_votes = np.zeros(n_upper, dtype=np.float32)
lower_votes = np.zeros(n_lower, dtype=np.float32)
fdi_plaque_summary = {}
sat_plaque_fdi_set = set()

print("\n🔍 逐視角 SAT mask UV 投射...")
debug_dir = OUTPUT_DIR / "debug_proj"
debug_dir.mkdir(exist_ok=True)

for view_name, cfg in VIEW_CONFIG.items():
    print(f"\n  📷 {view_name}  "
          f"[scale=({cfg.get('scale_u',1):.2f},{cfg.get('scale_v',1):.2f})  "
          f"offset=({cfg.get('offset_u',0):.2f},{cfg.get('offset_v',0):.2f})  "
          f"clip={cfg.get('vert_clip_pct',0)}%]")

    roi_mask_path = ROI_MASK_DIR / cfg['roi_mask']
    if not roi_mask_path.exists():
        print(f"    ⚠️  找不到 roi_mask，跳過"); continue

    roi_mask = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)
    if roi_mask is None or roi_mask.max() == 0:
        print(f"    ⚠️  roi_mask 為空，跳過"); continue
    print(f"    roi_mask 菌斑像素: {int((roi_mask > 0).sum()):,}")

    print(f"    跑 SAT...")
    fdi_mask_sat = run_sat_for_view(view_name, cfg)
    if fdi_mask_sat is None:
        continue

    sat_h, sat_w = fdi_mask_sat.shape
    roi_resized = cv2.resize(roi_mask, (sat_w, sat_h), interpolation=cv2.INTER_NEAREST) \
                  if roi_mask.shape[:2] != (sat_h, sat_w) else roi_mask

    save_debug_projection(view_name, cfg, fdi_mask_sat, roi_mask, fdi_map, debug_dir)

    view_hits = 0
    for fdi in sorted(int(v) for v in np.unique(fdi_mask_sat) if v > 0):
        if fdi not in fdi_map:
            continue

        jaw, tooth_verts, tooth_indices = fdi_map[fdi]

        # ★ sat_bbox 只算一次，共用給 hit_verts 和 plaque_on_tooth
        sat_bbox = get_sat_bbox(fdi_mask_sat, fdi)
        if sat_bbox is None:
            continue
        px_min, py_min, bbox_w, bbox_h, sat_w, sat_h = sat_bbox

        hit_idx = _get_plaque_hit_verts_with_bbox(
            tooth_verts, tooth_indices, roi_resized, cfg, sat_bbox)
        if len(hit_idx) == 0:
            continue

        # ★ 只在 bbox 範圍內計算 plaque_on_tooth（不建立全圖 mask）
        roi_crop   = roi_resized[py_min:py_min+bbox_h, px_min:px_min+bbox_w]
        fdi_crop   = (fdi_mask_sat[py_min:py_min+bbox_h, px_min:px_min+bbox_w] == fdi)
        plaque_on_tooth = int(roi_crop[fdi_crop].sum() // 255)

        # 記錄所有 SAT 偵測到的 FDI（不管有無菌斑）
        sat_plaque_fdi_set.add(fdi)

        if plaque_on_tooth == 0:
            continue

        weight = np.log1p(plaque_on_tooth)

        if jaw == 'upper':
            upper_votes[hit_idx] += weight
        else:
            lower_votes[hit_idx] += weight

        if fdi not in fdi_plaque_summary:
            fdi_plaque_summary[fdi] = {'jaw': jaw, 'views': [],
                                        'total_plaque_px': 0, 'hit_verts': 0}
        fdi_plaque_summary[fdi]['total_plaque_px'] += plaque_on_tooth
        fdi_plaque_summary[fdi]['hit_verts']       += len(hit_idx)
        if view_name not in fdi_plaque_summary[fdi]['views']:
            fdi_plaque_summary[fdi]['views'].append(view_name)

        view_hits += len(hit_idx)
        print(f"    FDI {fdi:02d} ({jaw}): 菌斑像素={plaque_on_tooth}  命中={len(hit_idx)} 頂點")

    print(f"    本視角命中頂點: {view_hits:,}")


# ==================== 染色（向量化）====================
print(f"\n🎨 染色...")

upper_colors = np.tile(np.append(COLOR_NORMAL, 1.0), (n_upper, 1)).astype(np.float32)
lower_colors = np.tile(np.append(COLOR_NORMAL, 1.0), (n_lower, 1)).astype(np.float32)

def apply_plaque_color(votes, colors):
    """向量化染色（避免兩次 boolean indexing）"""
    is_p = votes > 0
    if is_p.sum() > 0:
        inten = 0.5 + 0.5 * votes[is_p] / (votes[is_p].max() + 1e-8)
        colors[is_p, 0] = inten
        colors[is_p, 1] = 0.05
        colors[is_p, 2] = 0.05
    return colors, is_p

upper_colors, is_p_u = apply_plaque_color(upper_votes, upper_colors)
lower_colors, is_p_l = apply_plaque_color(lower_votes, lower_colors)

n_plaque = int(is_p_u.sum()) + int(is_p_l.sum())
n_total  = n_upper + n_lower
print(f"  上顎菌斑頂點: {is_p_u.sum():,} / {n_upper:,}")
print(f"  下顎菌斑頂點: {is_p_l.sum():,} / {n_lower:,}")
print(f"  合計: {n_plaque:,} / {n_total:,} ({n_plaque/n_total*100:.2f}%)")

# ★ 一次 concatenate（避免中間暫存）
all_colors_u8 = (np.concatenate([upper_colors, lower_colors], axis=0) * 255).astype(np.uint8)


# ==================== 輸出 ====================
print(f"\n💾 輸出...")

_analysis_path = _PATHS["analysis"] / "real_teeth_analysis.json"
_never_detected = []
if _analysis_path.exists():
    import json as _json
    _never_detected = _json.loads(_analysis_path.read_text()).get("never_detected", [])
    print(f"  缺牙移除: {_never_detected}")

combined_raw    = trimesh.util.concatenate([upper_mesh, lower_mesh])
combined_labels = np.concatenate([upper_labels, lower_labels])

# ★ 向量化建立 keep_mask（原本用迴圈）
keep_mask = np.ones(len(combined_raw.vertices), dtype=bool)
if _never_detected:
    remove_mask = np.isin(combined_labels, _never_detected)
    keep_mask[remove_mask] = False

keep_indices = np.where(keep_mask)[0]
index_map = np.full(len(combined_raw.vertices), -1, dtype=np.int64)
index_map[keep_indices] = np.arange(len(keep_indices))

out_verts      = np.array(combined_raw.vertices)[keep_indices]
all_colors_u8  = all_colors_u8[keep_indices]
_old_faces     = np.array(combined_raw.faces)
_face_keep     = keep_mask[_old_faces].all(axis=1)
out_faces      = index_map[_old_faces[_face_keep]]
print(f"  移除缺牙後: {len(out_verts):,} 頂點  {len(out_faces):,} 面")


def export_mesh(path, verts, faces, colors):
    trimesh.Trimesh(vertices=verts, faces=faces,
                    vertex_colors=colors, process=False).export(str(path))


ply = OUTPUT_DIR / "plaque_by_fdi.ply"
export_mesh(ply, out_verts, out_faces, all_colors_u8)
print(f"  ✅ PLY: {ply.name}  ({ply.stat().st_size/1024/1024:.1f} MB)")

glb = OUTPUT_DIR / "plaque_by_fdi.glb"
export_mesh(glb, out_verts, out_faces, all_colors_u8)
print(f"  ✅ GLB: {glb.name}  ({glb.stat().st_size/1024/1024:.1f} MB)")

obj = OUTPUT_DIR / "plaque_by_fdi.obj"
with open(obj, 'w') as f:
    f.write("# plaque_by_fdi - vertex color OBJ\n\n")
    # ★ 一次性寫入（batch string 避免逐行 format）
    lines = []
    for v, c in zip(out_verts, all_colors_u8):
        lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} "
                     f"{c[0]/255.:.4f} {c[1]/255.:.4f} {c[2]/255.:.4f}")
    f.write('\n'.join(lines) + '\n\n')
    face_lines = [f"f {r[0]+1} {r[1]+1} {r[2]+1}" for r in out_faces]
    f.write('\n'.join(face_lines) + '\n')
print(f"  ✅ OBJ: {obj.name}")

stats = {
    'total_vertices': n_total,
    'plaque_vertices': n_plaque,
    'plaque_ratio': round(n_plaque / n_total, 4),
    'sat_plaque_fdi_count': len(sat_plaque_fdi_set),
    'fdi_plaque_summary': {str(k): v for k, v in sorted(fdi_plaque_summary.items())},
    'view_config_scale_offset': {
        vn: {
            'scale_u':       vc.get('scale_u', 1.0),
            'scale_v':       vc.get('scale_v', 1.0),
            'offset_u':      vc.get('offset_u', 0.0),
            'offset_v':      vc.get('offset_v', 0.0),
            'vert_clip_pct': vc.get('vert_clip_pct', 0),
        }
        for vn, vc in VIEW_CONFIG.items()
    }
}
json_out = OUTPUT_DIR / "plaque_by_fdi_stats.json"
with open(json_out, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"  ✅ JSON: {json_out.name}")

print(f"\n{'='*60}")
print(f"✅ 完成！菌斑覆蓋率: {n_plaque/n_total*100:.2f}%")
print(f"\n📊 各 FDI 菌斑統計:")
for fdi, info in sorted(fdi_plaque_summary.items()):
    print(f"  FDI {fdi:02d} ({info['jaw']}): "
          f"菌斑像素={info['total_plaque_px']:,}  "
          f"命中={info['hit_verts']} 頂點  視角:{info['views']}")
print(f"\n💡 GLB → https://gltf-viewer.donmccurdy.com")
print(f"   PLY → MeshLab: Render → Color → Per Vertex")