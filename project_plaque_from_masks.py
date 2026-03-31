#!/usr/bin/env python3
"""
project_plaque_from_masks.py
從 teeth_color_test/ 資料夾讀現有的 mask_*.jpg，
直接投影到 3D 模型，輸出 json / obj / ply / glb
放在 SegmentAnyTooth/plaque_output/ 資料夾
"""

import cv2
import numpy as np
import trimesh
import json
from pathlib import Path

# ==================== 路徑設定 ====================
BASE       = Path("/home/Zhen/projects/SegmentAnyTooth")
MASK_DIR   = BASE / "teeth_color_test"
MODEL_PATH = BASE / "personalized_3d_models_real" / "custom_real_teeth.obj"
OUTPUT_DIR = BASE / "plaque_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ==================== 每個視角的 mask 檔名 + 投影設定 ====================
# mask_file: teeth_color_test/ 裡的 mask 檔案
# cam_dir:   相機方向向量（從哪個方向看過來）
# weight:    可信度（front 最高，occlusal 次之，side 最低）
# note:      簡短說明這個視角的問題
VIEW_CONFIG = {
    'front': {
        'mask_file': 'mask_front2_test.jpg',
        # 模型 Z 全為負，正面從 +Z 方向看（Z=-32 最淺處是牙齒正面）
        'cam_dir': (0, 0, 1),
        'weight': 1.0,
        'note': '正面，可信度最高'
    },
    'left_side': {
        'mask_file': 'mask_left_side2_test.jpg',
        # 從右側（+X）看過去拍到左側牙齒
        'cam_dir': (1, 0, 0),
        'weight': 0.5,
        'note': '左側，有手指誤判，降低可信度'
    },
    'right_side': {
        'mask_file': 'mask_right_side2_test.jpg',
        # 從左側（-X）看過去拍到右側牙齒
        'cam_dir': (-1, 0, 0),
        'weight': 0.5,
        'note': '右側，有手指誤判，降低可信度'
    },
    'upper_occlusal': {
        'mask_file': 'mask_upper_occlusal2_test.jpg',
        # 從上方（+Y）往下看咬合面
        'cam_dir': (0, 1, 0),
        'weight': 0.3,
        'note': '上咬合，口腔內部全部白，誤判嚴重'
    },
    'lower_occlusal': {
        'mask_file': 'mask_lower_occlusal2_test.jpg',
        # 從下方（-Y）往上看咬合面
        'cam_dir': (0, -1, 0),
        'weight': 0.8,
        'note': '下咬合，偵測範圍較小但精準'
    },
}

# ==================== 投影門檻 ====================
PLAQUE_THRESHOLD = 0.30   # 加權菌斑比例門檻（降低以保留更多真實菌斑）
PROJ_MARGIN      = 5      # 投影座標邊界允許像素

COLOR_PLAQUE = np.array([0.95, 0.05, 0.05])   # 紅色
COLOR_NORMAL = np.array([0.92, 0.86, 0.80])   # 牙色

# ==================== 投影函數 ====================

def build_projection(cam_dir, center, extent, img_size):
    """
    座標系：X=左右, Y=上下, Z=前後（Z全負，Z=-32最前面）
    front(+Z)  → 投影到 X-Y 平面，u=X(左→右), v=Y(下→上)
    upper(+Y)  → 投影到 X-Z 平面，u=X, v=Z(深→淺)
    lower(-Y)  → 投影到 X-Z 平面，u=X(翻轉), v=Z
    left(+X)   → 投影到 Y-Z 平面，u=Z, v=Y
    right(-X)  → 投影到 Y-Z 平面，u=Z(翻轉), v=Y
    """
    cx, cy, cz = cam_dir
    H, W = img_size

    if abs(cz) > 0.5:
        # 正面：投影到 X-Y 平面
        # 上顎 Y∈[6.78,47.4]，下顎 Y∈[-33.8,6.78]，中心偏上
        y_min  = -33.82
        y_max  =  47.39
        y_mid  = (y_max + y_min) * 0.5   # 6.785
        y_half = (y_max - y_min) * 0.5   # 40.6
        def proj(pts):
            u = (pts[:,0] - center[0]) / extent[0]
            v = (pts[:,1] - y_mid) / y_half
            if cz > 0:
                u = -u
            # scale=0.35 把模型壓到圖像內
            # y_offset=-0.10 讓牙齒往上移（菌斑在圖像25~35%處）
            scale  = 0.35
            y_offset = -0.10
            px = (u  * scale + 0.5) * W
            py = (-v * scale + 0.5 + y_offset) * H
            return np.stack([px, py], axis=1)
        normal = np.array([0., 0., float(cz)])

    elif abs(cy) > 0.5:
        # 咬合面：投影到 X-Z 平面
        def proj(pts):
            u = (pts[:,0] - center[0]) / extent[0]   # X: 左右
            v = (pts[:,2] - center[2]) / extent[2]   # Z: 前後
            if cy > 0:  # 從上往下看
                u = -u
            px = (u * 0.85 + 0.5) * W
            py = (v * 0.85 + 0.5) * H
            return np.stack([px, py], axis=1)
        normal = np.array([0., float(cy), 0.])

    else:
        # 側面：投影到 Z-Y 平面
        def proj(pts):
            u = (pts[:,2] - center[2]) / extent[2]   # Z: 前後
            v = (pts[:,1] - center[1]) / extent[1]   # Y: 上下
            if cx > 0:  # 從+X（右側）看
                u = -u
            px = (u * 0.85 + 0.5) * W
            py = (-v * 0.85 + 0.5) * H
            return np.stack([px, py], axis=1)
        normal = np.array([float(cx), 0., 0.])

    n = normal / (np.linalg.norm(normal) + 1e-8)
    return proj, n

# ==================== 主流程 ====================

print("="*60)
print("🦷 從現有 mask 投影牙菌斑到 3D 模型")
print("="*60)

# 載入模型
print(f"\n📦 載入模型: {MODEL_PATH.name}")
mesh     = trimesh.load(str(MODEL_PATH))
verts    = np.array(mesh.vertices)
n_verts  = len(verts)
bounds   = mesh.bounds
center   = bounds.mean(axis=0)
extent   = (bounds[1] - bounds[0]) * 0.5 + 1e-6
vnormals = np.array(mesh.vertex_normals)
print(f"  頂點: {n_verts:,}  面: {len(mesh.faces):,}")

vote_sum   = np.zeros(n_verts, dtype=np.float32)
weight_sum = np.zeros(n_verts, dtype=np.float32)
view_stats = {}

# 各視角投影
print(f"\n🔍 各視角投影...")

for view_name, cfg in VIEW_CONFIG.items():
    mask_path = MASK_DIR / cfg['mask_file']
    if not mask_path.exists():
        print(f"\n  ⚠️  {view_name}: 找不到 {cfg['mask_file']}")
        continue

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"\n  ⚠️  {view_name}: 讀取失敗")
        continue

    # 二值化（mask 可能是灰階 jpg，重新二值化）
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    H, W = mask.shape
    plaque_px = (mask > 0).sum()
    total_px  = H * W
    print(f"\n  📷 {view_name}  (w={cfg['weight']}, {cfg['note']})")
    print(f"    mask 菌斑像素: {plaque_px} ({plaque_px/total_px*100:.1f}%)")

    proj_fn, _ = build_projection(cfg['cam_dir'], center, extent, (H, W))

    # 背面剔除
    cam_vec = np.array(cfg['cam_dir'], dtype=float)
    cam_vec /= np.linalg.norm(cam_vec) + 1e-8
    dot     = vnormals @ (-cam_vec)
    visible = np.where(dot > -0.1)[0]

    # ⭐ front 視角：分上下顎分別對應 mask 的上半/下半
    if view_name == 'front':
        UPPER_Y = 6.78
        upper_idx = visible[verts[visible, 1] >= UPPER_Y]
        lower_idx = visible[verts[visible, 1] <  UPPER_Y]
        hit_total = 0

        for jaw_idx, jaw_half in [(upper_idx, 'upper'), (lower_idx, 'lower')]:
            u_j = -(verts[jaw_idx, 0] - center[0]) / extent[0]
            if jaw_half == 'upper':
                y_lo, y_hi = UPPER_Y, 47.39
            else:
                y_lo, y_hi = -33.82, UPPER_Y
            y_mid_j  = (y_lo + y_hi) * 0.5
            y_half_j = (y_hi - y_lo) * 0.5
            v_j = (verts[jaw_idx, 1] - y_mid_j) / y_half_j
            px_j = (u_j  * 0.35 + 0.5) * W
            py_j = (-v_j * 0.45 + 0.5) * (H // 2)

            if jaw_half == 'upper':
                mask_crop = mask[:H//2, :]
            else:
                mask_crop = mask[H//2:, :]

            in_b = (px_j >= -PROJ_MARGIN) & (px_j < W + PROJ_MARGIN) &                    (py_j >= -PROJ_MARGIN) & (py_j < H // 2 + PROJ_MARGIN)
            valid_jaw = jaw_idx[in_b]
            ix_c = np.clip(px_j[in_b], 0, W - 1).astype(np.int32)
            iy_c = np.clip(py_j[in_b], 0, H // 2 - 1).astype(np.int32)

            vals = mask_crop[iy_c, ix_c].astype(np.float32) / 255.0
            w = cfg['weight']
            vote_sum[valid_jaw]   += vals * w
            weight_sum[valid_jaw] += w
            hit_total += int((vals > 0.5).sum())

        hit = hit_total
        valid_idx = visible
        print(f"    投影頂點: {len(visible):,}  命中菌斑: {hit:,}  [上下顎分開]")

    else:
        pts2d = proj_fn(verts[visible])
        px, py = pts2d[:,0], pts2d[:,1]
        in_b   = (px >= -PROJ_MARGIN) & (px < W+PROJ_MARGIN) &                  (py >= -PROJ_MARGIN) & (py < H+PROJ_MARGIN)

        valid_idx = visible[in_b]
        ix = np.clip(px[in_b], 0, W-1).astype(np.int32)
        iy = np.clip(py[in_b], 0, H-1).astype(np.int32)

        vals = mask[iy, ix].astype(np.float32) / 255.0
        w    = cfg['weight']
        vote_sum[valid_idx]   += vals * w
        weight_sum[valid_idx] += w

        hit = (vals > 0.5).sum()
        print(f"    投影頂點: {len(valid_idx):,}  命中菌斑: {hit:,}")


    view_stats[view_name] = {
        'weight': w,
        'mask_plaque_ratio': round(plaque_px/total_px, 4),
        'projected_vertices': int(len(valid_idx)),
        'hit_vertices': int(hit),
        'note': cfg['note']
    }

# 染色
print(f"\n🎨 計算頂點染色 (門檻={PLAQUE_THRESHOLD})...")
safe_w    = np.where(weight_sum > 0, weight_sum, 1.0)
ratio     = vote_sum / safe_w
ratio[weight_sum == 0] = 0.0
is_plaque = ratio >= PLAQUE_THRESHOLD

n_plaque = int(is_plaque.sum())
print(f"  菌斑頂點: {n_plaque:,} / {n_verts:,} ({n_plaque/n_verts*100:.1f}%)")

colors = np.zeros((n_verts, 4), dtype=np.float32)
colors[~is_plaque, :3] = COLOR_NORMAL
colors[~is_plaque, 3]  = 1.0
intensity = np.clip(ratio[is_plaque], 0.4, 1.0)
colors[is_plaque, 0] = intensity
colors[is_plaque, 1] = 0.05
colors[is_plaque, 2] = 0.05
colors[is_plaque, 3] = 1.0
colors_u8 = (colors * 255).astype(np.uint8)

# ==================== 輸出 ====================
print(f"\n💾 輸出檔案到 {OUTPUT_DIR.name}/")

# 1. PLY
ply_path = OUTPUT_DIR / "plaque_teeth.ply"
trimesh.Trimesh(
    vertices=mesh.vertices,
    faces=mesh.faces,
    vertex_colors=colors_u8,
    process=False
).export(str(ply_path))
mb = ply_path.stat().st_size / 1024 / 1024
print(f"  ✅ PLY: {ply_path.name}  ({mb:.1f} MB)")

# 2. OBJ (頂點色寫在 v 行)
obj_path = OUTPUT_DIR / "plaque_teeth.obj"
with open(obj_path, 'w') as f:
    f.write("# plaque_teeth - vertex color OBJ\n")
    f.write(f"# plaque_vertices: {n_plaque} / {n_verts}\n\n")
    for v, c in zip(mesh.vertices, colors_u8):
        f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f} "
                f"{c[0]/255.:.4f} {c[1]/255.:.4f} {c[2]/255.:.4f}\n")
    f.write("\n")
    for face in mesh.faces:
        f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
print(f"  ✅ OBJ: {obj_path.name}")

# 3. GLB (trimesh 直接 export，帶材質)
glb_path = OUTPUT_DIR / "plaque_teeth.glb"
# 為 GLB 建立帶顏色的 mesh（trimesh GLB 用 vertex_colors）
glb_mesh = trimesh.Trimesh(
    vertices=mesh.vertices,
    faces=mesh.faces,
    vertex_colors=colors_u8,
    process=False
)
glb_mesh.export(str(glb_path))
mb_glb = glb_path.stat().st_size / 1024 / 1024
print(f"  ✅ GLB: {glb_path.name}  ({mb_glb:.1f} MB)")

# 4. JSON 統計
stats = {
    'model': str(MODEL_PATH.name),
    'total_vertices': n_verts,
    'plaque_vertices': n_plaque,
    'plaque_ratio': round(n_plaque / n_verts, 4),
    'threshold': PLAQUE_THRESHOLD,
    'per_view': view_stats,
    'color': {
        'plaque': 'red (intensity proportional to confidence)',
        'normal': 'ivory'
    }
}
json_path = OUTPUT_DIR / "plaque_stats.json"
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)
print(f"  ✅ JSON: {json_path.name}")

print(f"\n{'='*60}")
print(f"✅ 完成！菌斑覆蓋率: {n_plaque/n_verts*100:.1f}%")
print(f"📁 輸出目錄: {OUTPUT_DIR}")
print(f"""
💡 查看方式：
  PLY/OBJ → MeshLab: Render → Color → Per Vertex
  GLB     → 直接拖入 https://gltf-viewer.donmccurdy.com
            或 Windows 3D 檢視器
{'='*60}
⚠️  注意目前各視角品質：
  - front:          可信 (w=1.0)
  - lower_occlusal: 可信 (w=0.8)  
  - left/right:     有手指誤判 (w=0.5)
  - upper_occlusal: 嚴重誤判 (w=0.3)，建議重拍
""")