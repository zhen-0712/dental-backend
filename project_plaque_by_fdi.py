#!/usr/bin/env python3
"""
project_plaque_by_fdi.py  ── SAT mask UV 對應版（per-tooth bbox + scale/offset 微調）

流程：
  對每個視角：
  1. 載入 roi_mask_*.png
  2. 用 SAT 跑一次 predict，取得 fdi_mask
  3. 對每顆 FDI 牙：
       a. 從 seg_labels 取出客製化模型的 3D 頂點
       b. 用 SAT 上這顆牙的 bbox 做 normalize，把 3D 頂點投影過去
       c. 套用 scale/offset 微調
       d. 查詢 roi_mask 是否命中菌斑
  4. 輸出染色後的 PLY / GLB / OBJ
"""

import cv2
import numpy as np
import trimesh
import json
from pathlib import Path

# ==================== 路徑設定 ====================
BASE         = Path("/home/Zhen/projects/SegmentAnyTooth")
ROI_MASK_DIR = BASE / "plaque_output"
MODEL_DIR    = BASE / "personalized_3d_models_real"
OUTPUT_DIR   = BASE / "plaque_output"
WEIGHT_DIR   = BASE / "weight"
OUTPUT_DIR.mkdir(exist_ok=True)

UPPER_OBJ    = MODEL_DIR / "custom_upper_only.obj"
LOWER_OBJ    = MODEL_DIR / "custom_lower_only.obj"
UPPER_LABELS = MODEL_DIR / "upper_seg_labels.npy"
LOWER_LABELS = MODEL_DIR / "lower_seg_labels.npy"
MODEL_PATH   = MODEL_DIR / "custom_real_teeth.obj"

COLOR_NORMAL = np.array([0.92, 0.86, 0.80])

# ==================== 各視角設定 ====================
# proj_u_axis / proj_v_axis : 投影平面的兩個軸（0=X, 1=Y, 2=Z）
# flip_u / flip_v           : 是否翻轉軸方向
# scale_u / scale_v         : 投影縮放（以 SAT bbox 中心為基準）
# offset_u / offset_v       : 中心偏移（單位：SAT bbox 的比例）
#   負值 = 往左/往上，正值 = 往右/往下
# vert_clip_pct             : 頂點離群值裁切（百分位數），0=不裁切，5=裁切最外5%
#   用來解決某顆牙頂點含牙根/異常範圍導致 bbox 被拉大的問題

VIEW_CONFIG = {
    'front': {
        'sat_view':   'front',
        'roi_mask':   'roi_mask_front.png',
        'photo_file': 'real_teeth_processed/front.jpg',
        'proj_u_axis': 0, 'proj_v_axis': 2,
        'flip_u': True,  'flip_v': True,
        'scale_u': 1.0,  'scale_v': 0.55,
        'offset_u': 0.0, 'offset_v': -1.05,
        'vert_clip_pct': 5,
    },
    'left_side': {
        'sat_view':   'left',
        'roi_mask':   'roi_mask_left_side.png',
        'photo_file': 'real_teeth_processed/left_side.jpg',
        'proj_u_axis': 1, 'proj_v_axis': 2,
        'flip_u': False, 'flip_v': True,
        'scale_u': 1.0,  'scale_v': 0.55,
        'offset_u': -0.45, 'offset_v': -0.60,
        'vert_clip_pct': 5,
    },
    'right_side': {
        'sat_view':   'right',
        'roi_mask':   'roi_mask_right_side.png',
        'photo_file': 'real_teeth_processed/right_side.jpg',
        'proj_u_axis': 1, 'proj_v_axis': 2,
        'flip_u': True,  'flip_v': True,
        'scale_u': 1.0,  'scale_v': 0.55,
        'offset_u': -0.45, 'offset_v': -0.20,
        # FDI 15 的頂點含大量牙根/異常點，裁切 10% 離群值
        'vert_clip_pct': 10,
    },
    'upper_occlusal': {
        'sat_view':   'upper',
        'roi_mask':   'roi_mask_upper_occlusal.png',
        'photo_file': 'real_teeth_processed/upper_occlusal.jpg',
        # 咬合面照：照片水平 = 牙弓左右 = 3D X 軸
        #           照片垂直 = 牙弓前後深度 = 3D Y 軸
        # flip_u=True 讓左右方向對齊照片（病人右側在照片右）
        # flip_v=True 讓前牙（Y 最小）出現在照片下方（符合上顎咬合照）
        'proj_u_axis': 0, 'proj_v_axis': 1,
        'flip_u': True,  'flip_v': True,
        'scale_u': 1.6,  'scale_v': 1.15,
        'offset_u': 0.0, 'offset_v': 0.0,
        'vert_clip_pct': 5,
    },
    'lower_occlusal': {
        'sat_view':   'lower',
        'roi_mask':   'roi_mask_lower_occlusal.png',
        'photo_file': 'real_teeth_processed/lower_occlusal.jpg',
        # 下顎咬合面照：從下往上拍，前後方向和上顎相反
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
    """從 fdi_mask_sat 取出這顆牙的 bbox，回傳 (px_min, py_min, w, h, sat_w, sat_h) 或 None"""
    sat_h, sat_w = fdi_mask_sat.shape
    ys, xs = np.where(fdi_mask_sat == fdi)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()),
            max(int(xs.max() - xs.min()), 1),
            max(int(ys.max() - ys.min()), 1),
            sat_w, sat_h)


def clip_tooth_verts(tooth_verts, clip_pct):
    """
    裁切頂點的離群值（percentile clipping）。
    對 u/v 兩個投影軸各自裁切最外 clip_pct% 的頂點，
    回傳裁切後的頂點陣列（形狀不變，但 range 縮小）。
    這只影響 normalize 用的 min/max，不刪除任何頂點。
    """
    if clip_pct <= 0:
        return tooth_verts
    lo, hi = clip_pct, 100 - clip_pct
    clipped = tooth_verts.copy()
    for ax in range(3):
        v = tooth_verts[:, ax]
        vmin = np.percentile(v, lo)
        vmax = np.percentile(v, hi)
        clipped[:, ax] = np.clip(v, vmin, vmax)
    return clipped


def project_tooth_verts(tooth_verts, cfg, sat_bbox):
    """
    把 3D 頂點投影到 SAT 上這顆牙的 bbox 像素座標。

    流程：
      0. 用 percentile clipping 排除離群頂點對 normalize range 的影響
      1. 取兩個投影軸，normalize 到 [0,1]（以裁切後的 range 為基準）
      2. flip
      3. 套用 scale/offset（以 0.5 為中心縮放，再平移）
      4. 映射到 SAT bbox 像素座標

    sat_bbox = (px_min, py_min, bbox_w, bbox_h, sat_w, sat_h)
    """
    u_ax = cfg['proj_u_axis']
    v_ax = cfg['proj_v_axis']
    px_min, py_min, bbox_w, bbox_h, sat_w, sat_h = sat_bbox

    # Step 0: percentile clipping（只用來算 normalize range，不刪頂點）
    clip_pct = cfg.get('vert_clip_pct', 0)
    ref_verts = clip_tooth_verts(tooth_verts, clip_pct)

    u_vals = tooth_verts[:, u_ax]
    v_vals = tooth_verts[:, v_ax]
    u_ref  = ref_verts[:, u_ax]
    v_ref  = ref_verts[:, v_ax]

    # 用 ref_verts 的 range 做 normalize（排除異常頂點影響）
    u_min, u_max = u_ref.min(), u_ref.max()
    v_min, v_max = v_ref.min(), v_ref.max()
    u_range = max(u_max - u_min, 1e-6)
    v_range = max(v_max - v_min, 1e-6)

    u_norm = (u_vals - u_min) / u_range
    v_norm = (v_vals - v_min) / v_range

    if cfg['flip_u']: u_norm = 1.0 - u_norm
    if cfg['flip_v']: v_norm = 1.0 - v_norm

    # scale / offset 微調（以 0.5 為中心縮放再平移）
    su = cfg.get('scale_u', 1.0);  sv = cfg.get('scale_v', 1.0)
    ou = cfg.get('offset_u', 0.0); ov = cfg.get('offset_v', 0.0)
    u_adj = (u_norm - 0.5) * su + 0.5 + ou
    v_adj = (v_norm - 0.5) * sv + 0.5 + ov

    px = np.clip((u_adj * bbox_w + px_min).astype(np.int32), 0, sat_w - 1)
    py = np.clip((v_adj * bbox_h + py_min).astype(np.int32), 0, sat_h - 1)
    return px, py


def get_plaque_hit_verts(fdi, tooth_verts, tooth_indices,
                         fdi_mask_sat, roi_resized, cfg):
    """回傳命中菌斑的 tooth_indices 子集"""
    sat_bbox = get_sat_bbox(fdi_mask_sat, fdi)
    if sat_bbox is None:
        return np.array([], dtype=np.int64)
    px, py = project_tooth_verts(tooth_verts, cfg, sat_bbox)
    return tooth_indices[roi_resized[py, px] > 0]


def save_debug_projection(view_name, cfg, fdi_mask_sat, roi_mask, fdi_map, debug_dir):
    """
    輸出 debug_proj_{view_name}.png
      白色 = roi_mask 菌斑
      灰色 = SAT 牙齒區域
      藍框 = SAT bbox
      黃字 = FDI 編號
      綠點 = 3D 頂點投影（抽樣）
      紅點 = 命中菌斑的頂點
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

        # 綠：抽樣 200
        sidx = np.random.choice(len(px), min(200, len(px)), replace=False)
        for i in sidx:
            cv2.circle(img, (px[i], py[i]), 1, (0, 200, 0), -1)

        # 紅：命中菌斑
        hit = roi_r[py, px] > 0
        hpx, hpy = px[hit], py[hit]
        if len(hpx) > 0:
            hidx = np.random.choice(len(hpx), min(100, len(hpx)), replace=False)
            for i in hidx:
                cv2.circle(img, (hpx[i], hpy[i]), 2, (0, 0, 255), -1)

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
print("🦷 菌斑投射 → 客製化 3D 模型（per-tooth bbox + scale/offset）")
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
        print(f"    ⚠️  找不到 roi_mask，跳過")
        continue

    roi_mask = cv2.imread(str(roi_mask_path), cv2.IMREAD_GRAYSCALE)
    if roi_mask is None or roi_mask.max() == 0:
        print(f"    ⚠️  roi_mask 為空，跳過")
        continue
    print(f"    roi_mask 菌斑像素: {int((roi_mask > 0).sum()):,}")

    print(f"    跑 SAT...")
    fdi_mask_sat = run_sat_for_view(view_name, cfg)
    if fdi_mask_sat is None:
        continue

    sat_h, sat_w = fdi_mask_sat.shape
    roi_resized = cv2.resize(roi_mask, (sat_w, sat_h), interpolation=cv2.INTER_NEAREST) \
                  if roi_mask.shape[:2] != (sat_h, sat_w) else roi_mask

    # debug 圖
    save_debug_projection(view_name, cfg, fdi_mask_sat, roi_mask, fdi_map, debug_dir)

    view_hits = 0
    for fdi in sorted(int(v) for v in np.unique(fdi_mask_sat) if v > 0):
        if fdi not in fdi_map:
            continue

        jaw, tooth_verts, tooth_indices = fdi_map[fdi]

        hit_idx = get_plaque_hit_verts(
            fdi, tooth_verts, tooth_indices,
            fdi_mask_sat, roi_resized, cfg)

        if len(hit_idx) == 0:
            continue

        tooth_mask_2d = (fdi_mask_sat == fdi).astype(np.uint8)
        plaque_on_tooth = int(
            cv2.bitwise_and(roi_resized, roi_resized, mask=tooth_mask_2d).sum() / 255)

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

# ==================== 染色 ====================
print(f"\n🎨 染色...")

upper_colors = np.tile(np.append(COLOR_NORMAL, 1.0), (n_upper, 1)).astype(np.float32)
lower_colors = np.tile(np.append(COLOR_NORMAL, 1.0), (n_lower, 1)).astype(np.float32)

is_p_u = upper_votes > 0
if is_p_u.sum() > 0:
    inten = 0.5 + 0.5 * upper_votes[is_p_u] / (upper_votes[is_p_u].max() + 1e-8)
    upper_colors[is_p_u, 0] = inten
    upper_colors[is_p_u, 1] = 0.05
    upper_colors[is_p_u, 2] = 0.05

is_p_l = lower_votes > 0
if is_p_l.sum() > 0:
    inten = 0.5 + 0.5 * lower_votes[is_p_l] / (lower_votes[is_p_l].max() + 1e-8)
    lower_colors[is_p_l, 0] = inten
    lower_colors[is_p_l, 1] = 0.05
    lower_colors[is_p_l, 2] = 0.05

n_plaque = int(is_p_u.sum()) + int(is_p_l.sum())
n_total  = n_upper + n_lower
print(f"  上顎菌斑頂點: {is_p_u.sum():,} / {n_upper:,}")
print(f"  下顎菌斑頂點: {is_p_l.sum():,} / {n_lower:,}")
print(f"  合計: {n_plaque:,} / {n_total:,} ({n_plaque/n_total*100:.2f}%)")

all_colors_u8 = (np.concatenate([upper_colors, lower_colors], axis=0) * 255).astype(np.uint8)

# ==================== 輸出 ====================
print(f"\n💾 輸出...")

# 從 analysis JSON 取得缺牙清單
_analysis_path = BASE / "real_teeth_analysis" / "real_teeth_analysis.json"
_never_detected = []
if _analysis_path.exists():
    import json as _json
    _never_detected = _json.loads(_analysis_path.read_text()).get("never_detected", [])
    print(f"  缺牙移除: {_never_detected}")

# combined + labels（含缺牙完整版）
combined_raw = trimesh.util.concatenate([upper_mesh, lower_mesh])
combined_labels = np.concatenate([upper_labels, lower_labels])

# 找出要保留的頂點（移除缺牙）
keep_mask = np.ones(len(combined_raw.vertices), dtype=bool)
for _t in _never_detected:
    keep_mask[combined_labels == _t] = False

keep_indices = np.where(keep_mask)[0]
index_map = np.full(len(combined_raw.vertices), -1, dtype=np.int64)
index_map[keep_indices] = np.arange(len(keep_indices))

# 過濾頂點、顏色、面
out_verts = np.array(combined_raw.vertices)[keep_indices]
all_colors_u8 = all_colors_u8[keep_indices]
_old_faces = np.array(combined_raw.faces)
_face_keep = keep_mask[_old_faces].all(axis=1)
out_faces = index_map[_old_faces[_face_keep]]
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
    for v, c in zip(out_verts, all_colors_u8):
        f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} "
                f"{c[0]/255.:.4f} {c[1]/255.:.4f} {c[2]/255.:.4f}\n")
    f.write("\n")
    for face in out_faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
print(f"  ✅ OBJ: {obj.name}")

stats = {
    'total_vertices': n_total,
    'plaque_vertices': n_plaque,
    'plaque_ratio': round(n_plaque / n_total, 4),
    'fdi_plaque_summary': {str(k): v for k, v in sorted(fdi_plaque_summary.items())},
    'view_config_scale_offset': {
        vn: {
            'scale_u':      vc.get('scale_u', 1.0),
            'scale_v':      vc.get('scale_v', 1.0),
            'offset_u':     vc.get('offset_u', 0.0),
            'offset_v':     vc.get('offset_v', 0.0),
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