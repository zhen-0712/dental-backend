#!/usr/bin/env python3
"""
控制點提取工具
從公版模板自動找每顆牙的控制點，從mask提取對應的2D控制點

核心改動：所有2D控制點座標改為「相對於牙齒自身bbox的正規化比例」
不管牙齒在照片的哪個位置，提取的形狀特徵都一致
"""

import numpy as np
import trimesh
import json
from pathlib import Path


def get_tooth_control_points_3d(mesh_vertices, tooth_indices):
    """
    從公版模板的牙齒頂點中，自動找9個控制點

    控制點定義（在牙齒局部座標系）：
    0: 近中最高點  (x最小側, z最高)
    1: 遠中最高點  (x最大側, z最高)
    2: 頰側最突點  (y最大)
    3: 舌側最突點  (y最小)
    4: 近中頸緣點  (x最小, z最低)
    5: 遠中頸緣點  (x最大, z最低)
    6: 切緣中點    (x中央, z最高) ← 新增，捕捉切緣弧線
    7: 近中1/3高點 (x偏左, z最高) ← 新增
    8: 遠中1/3高點 (x偏右, z最高) ← 新增
    """
    if tooth_indices is None or len(tooth_indices) == 0:
        return None

    verts = mesh_vertices[tooth_indices]

    z_min, z_max = verts[:, 2].min(), verts[:, 2].max()
    x_min, x_max = verts[:, 0].min(), verts[:, 0].max()

    z_mid = (z_min + z_max) * 0.5
    crown_mask = verts[:, 2] > z_mid
    neck_mask  = verts[:, 2] < z_mid + (z_max - z_min) * 0.1

    crown_verts = verts[crown_mask] if crown_mask.any() else verts
    neck_verts  = verts[neck_mask]  if neck_mask.any()  else verts

    cp = np.zeros((9, 3))

    # 0: 近中最高點
    left_mask = crown_verts[:, 0] < np.percentile(crown_verts[:, 0], 30)
    cp[0] = crown_verts[left_mask][crown_verts[left_mask][:, 2].argmax()] \
            if left_mask.any() else crown_verts[crown_verts[:, 2].argmax()]

    # 1: 遠中最高點
    right_mask = crown_verts[:, 0] > np.percentile(crown_verts[:, 0], 70)
    cp[1] = crown_verts[right_mask][crown_verts[right_mask][:, 2].argmax()] \
            if right_mask.any() else crown_verts[crown_verts[:, 2].argmax()]

    # 2: 頰側最突點
    cp[2] = verts[verts[:, 1].argmax()]

    # 3: 舌側最突點
    cp[3] = verts[verts[:, 1].argmin()]

    # 4: 近中頸緣點
    if neck_verts.shape[0] > 0:
        left_neck = neck_verts[:, 0] < np.percentile(neck_verts[:, 0], 30)
        cp[4] = neck_verts[left_neck].mean(axis=0) if left_neck.any() else neck_verts.mean(axis=0)
    else:
        cp[4] = verts[verts[:, 2].argmin()]

    # 5: 遠中頸緣點
    if neck_verts.shape[0] > 0:
        right_neck = neck_verts[:, 0] > np.percentile(neck_verts[:, 0], 70)
        cp[5] = neck_verts[right_neck].mean(axis=0) if right_neck.any() else neck_verts.mean(axis=0)
    else:
        cp[5] = verts[verts[:, 2].argmin()]

    # 6: 切緣中點（crown頂部x中央）
    mid_mask = (crown_verts[:, 0] > np.percentile(crown_verts[:, 0], 35)) & \
               (crown_verts[:, 0] < np.percentile(crown_verts[:, 0], 65))
    cp[6] = crown_verts[mid_mask][crown_verts[mid_mask][:, 2].argmax()] \
            if mid_mask.any() else (cp[0] + cp[1]) / 2

    # 7: 近中1/3高點
    lm_mask = crown_verts[:, 0] < np.percentile(crown_verts[:, 0], 45)
    cp[7] = crown_verts[lm_mask][crown_verts[lm_mask][:, 2].argmax()] \
            if lm_mask.any() else cp[0]

    # 8: 遠中1/3高點
    rm_mask = crown_verts[:, 0] > np.percentile(crown_verts[:, 0], 55)
    cp[8] = crown_verts[rm_mask][crown_verts[rm_mask][:, 2].argmax()] \
            if rm_mask.any() else cp[1]

    return cp  # shape (9, 3)


def extract_contour_control_points(mask_2d, view, tooth_id, pixel_to_mm=0.15):
    """
    從單張照片的mask提取輪廓控制點。

    ★ 核心改動：座標全部改為「相對於牙齒自身bbox的正規化偏移（mm）」
      公式：norm_offset = (pixel_pos - bbox_center) / bbox_size * actual_mm_size
      這樣牙齒在照片任何位置/縮放都能正確對應到3D控制點。
    """
    import cv2

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
    y_coords = pts[:, 1]  # 圖像y向下

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()
    w_px = x_max - x_min  # bbox寬度（pixel）
    h_px = y_max - y_min  # bbox高度（pixel）
    cx   = (x_min + x_max) / 2
    cy   = (y_min + y_max) / 2

    if w_px < 1 or h_px < 1:
        return None

    # ★ 正規化函數：把pixel偏移換成「佔bbox比例 × 實際mm尺寸」
    # 這樣不管牙齒在照片哪個位置，輸出的mm值都代表形狀本身的相對位置
    def norm_x(px):
        """相對bbox中心的X偏移，正規化到[-0.5, 0.5]再乘上bbox_mm寬度"""
        return ((px - cx) / w_px) * (w_px * pixel_to_mm)

    def norm_y(py):
        """相對bbox中心的Y偏移（y軸向上為正），正規化後乘高度mm"""
        return ((cy - py) / h_px) * (h_px * pixel_to_mm)

    # 找特徵點（輪廓極值點）
    left_pt  = pts[x_coords.argmin()]
    right_pt = pts[x_coords.argmax()]
    top_pt   = pts[y_coords.argmin()]   # y最小=圖像最上方
    bot_pt   = pts[y_coords.argmax()]   # y最大=圖像最下方

    # 四個角點（距bbox四角最近的輪廓點）
    for_corners = [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]
    tl, tr, bl, br = [pts[np.linalg.norm(pts - np.array(c), axis=1).argmin()]
                      for c in for_corners]

    # 切緣區域（上半部輪廓）的中央最低點 → 捕捉切緣內凹
    upper_half_mask = y_coords < cy
    if upper_half_mask.sum() > 3:
        upper_pts = pts[upper_half_mask]
        # 在上半部找x最靠近中央、y最大的點（切緣最低凹處）
        mid_region = np.abs(upper_pts[:, 0] - cx) < w_px * 0.25
        if mid_region.any():
            incisal_mid = upper_pts[mid_region][upper_pts[mid_region][:, 1].argmax()]
        else:
            incisal_mid = upper_pts[upper_pts[:, 1].argmax()]
    else:
        incisal_mid = np.array([cx, y_min])

    # 近中1/3和遠中1/3的上緣點
    left_third_mask  = x_coords < cx - w_px * 0.1
    right_third_mask = x_coords > cx + w_px * 0.1
    top_left_mask    = left_third_mask  & (y_coords < cy)
    top_right_mask   = right_third_mask & (y_coords < cy)

    mesial_third_pt = pts[top_left_mask][pts[top_left_mask][:, 1].argmin()] \
                      if top_left_mask.any() else tl
    distal_third_pt = pts[top_right_mask][pts[top_right_mask][:, 1].argmin()] \
                      if top_right_mask.any() else tr

    # 中線左右的頰舌側中點
    left_mid_mask  = x_coords < cx
    right_mid_mask = x_coords > cx
    left_mid  = pts[left_mid_mask][np.abs(y_coords[left_mid_mask] - cy).argmin()] \
                if left_mid_mask.any() else left_pt
    right_mid = pts[right_mid_mask][np.abs(y_coords[right_mid_mask] - cy).argmin()] \
                if right_mid_mask.any() else right_pt

    bbox_info = {'w': w_px, 'h': h_px, 'cx': cx, 'cy': cy}

    # ==================== 各視角回傳 ====================

    if view == 'front':
        # 正面視角：
        #   圖像X（左右） → 牙齒X（width，近遠中方向）
        #   圖像Y（上下） → 牙齒Z（height，冠根方向，圖像y向下所以反轉）
        return {
            'view': view,
            'bbox': bbox_info,
            'points': {
                # 9個控制點對應 get_tooth_control_points_3d 的順序
                'mesial_top':   {'x': norm_x(tl[0]),            'z_rel': norm_y(tl[1])},
                'distal_top':   {'x': norm_x(tr[0]),            'z_rel': norm_y(tr[1])},
                'buccal':       {'x': norm_x(left_mid[0])},                              # front看不到y方向
                'lingual':      {'x': norm_x(right_mid[0])},
                'mesial_neck':  {'x': norm_x(bl[0]),            'z_rel': norm_y(bl[1])},
                'distal_neck':  {'x': norm_x(br[0]),            'z_rel': norm_y(br[1])},
                'incisal_mid':  {'x': norm_x(incisal_mid[0]),   'z_rel': norm_y(incisal_mid[1])},
                'mesial_third': {'x': norm_x(mesial_third_pt[0]),'z_rel': norm_y(mesial_third_pt[1])},
                'distal_third': {'x': norm_x(distal_third_pt[0]),'z_rel': norm_y(distal_third_pt[1])},
            },
        }

    elif view in ['upper', 'lower']:
        # 咬合面視角：
        #   圖像X → 牙齒X（width）
        #   圖像Y → 牙齒Y（depth，頰舌方向）
        return {
            'view': view,
            'bbox': bbox_info,
            'points': {
                'mesial_top':   {'x': norm_x(tl[0]),       'y': norm_y(tl[1])},
                'distal_top':   {'x': norm_x(tr[0]),       'y': norm_y(tr[1])},
                'buccal':       {'x': norm_x(top_pt[0]),   'y': norm_y(top_pt[1])},
                'lingual':      {'x': norm_x(bot_pt[0]),   'y': norm_y(bot_pt[1])},
                'mesial_neck':  {'x': norm_x(bl[0]),       'y': norm_y(bl[1])},
                'distal_neck':  {'x': norm_x(br[0]),       'y': norm_y(br[1])},
                'incisal_mid':  {'x': norm_x(cx),          'y': norm_y(cy)},             # 咬合面中心
                'mesial_third': {'x': norm_x(left_mid[0]), 'y': norm_y(left_mid[1])},
                'distal_third': {'x': norm_x(right_mid[0]),'y': norm_y(right_mid[1])},
            },
        }

    elif view in ['left', 'right']:
        # 側面視角：
        #   圖像X → 牙齒Y（depth）
        #   圖像Y → 牙齒Z（height，反轉）
        return {
            'view': view,
            'bbox': bbox_info,
            'points': {
                'mesial_top':   {'z_rel': norm_y(tl[1])},
                'distal_top':   {'z_rel': norm_y(tr[1])},
                'buccal':       {'y': norm_x(left_pt[0]),  'z_rel': norm_y(left_pt[1])},
                'lingual':      {'y': norm_x(right_pt[0]), 'z_rel': norm_y(right_pt[1])},
                'mesial_neck':  {'z_rel': norm_y(bl[1])},
                'distal_neck':  {'z_rel': norm_y(br[1])},
                'incisal_mid':  {'z_rel': norm_y(top_pt[1])},
                'mesial_third': {'z_rel': norm_y(mesial_third_pt[1])},
                'distal_third': {'z_rel': norm_y(distal_third_pt[1])},
            },
        }

    return None


if __name__ == "__main__":
    MODELS_DIR = Path("./models")

    mesh = trimesh.load(str(MODELS_DIR / "01J9K9S6_lower.obj"))
    with open(MODELS_DIR / "01J9K9S6_lower.json") as f:
        seg_data = json.load(f)
    seg_labels = np.array(seg_data["labels"], dtype=np.int32)

    indices = np.nonzero(seg_labels == 41)[0]
    cp = get_tooth_control_points_3d(mesh.vertices, indices)

    print("41號牙控制點（9個）：")
    names = ['近中最高', '遠中最高', '頰側最突', '舌側最突',
             '近中頸緣', '遠中頸緣', '切緣中點', '近中1/3', '遠中1/3']
    for name, point in zip(names, cp):
        print(f"  {name}: x={point[0]:.1f}, y={point[1]:.1f}, z={point[2]:.1f}")