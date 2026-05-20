#!/usr/bin/env python3
"""
teeth_test.py（優化版）

優化項目：
  - ml_classify_mask: KNN 推論改為分批次（避免一次建立超大矩陣）
  - process_vivid: 使用 numpy in-place 操作減少中間複製
  - detect_plaque: morphology 核心預先建立，不重複建立
  - 輪廓過濾: 用 numpy 向量化面積過濾（取代逐個 if）
"""

import cv2
import numpy as np
import os

import sys; sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")
from user_env import get_paths, setup_user_dirs
_PATHS = get_paths()
setup_user_dirs(_PATHS["user_dir"])
'''
# =========================
# 1️⃣ 機器學習資料
# =========================
positive_samples = np.array([
[1,154,149],[0,149,151],[0,143,159],[1,141,163],[179,130,187],[179,129,190],
[1,117,187],[2,116,182],[2,127,167],[0,123,183],[2,149,149],[2,131,166],
[179,137,177],[178,136,184],[0,150,160],[0,127,179],[179,132,183],
[2,125,170],[6,152,121],[4,147,134],[2,157,130],[177,139,206],
[179,123,208],[179,149,181],[179,131,197],[1,124,199],[1,137,180],
[1,143,160],[179,137,179],[177,144,203],[177,144,206],[179,142,189],
[179,158,171],[178,131,196],[9,68,215],[6,106,186],[2,115,188],
[1,117,192],[1,145,157],[1,148,152],[1,131,170],[0,126,186],
[179,120,191],[1,134,168],[1,145,158],[0,106,192],[179,113,185],
[2,106,187],[5,93,191],[179,126,162],[179,121,168],[0,110,185],
[179,100,202],[3,140,178],[4,119,189],[2,147,174],[9,105,196],
[5,128,182],[4,152,169],[0,136,195],[0,142,179],[1,131,169],
[5,104,197],[5,87,215],[5,80,217],[2,108,212],[3,108,210],
[1,122,202],[179,135,195],[0,126,203],[0,115,208],[1,122,201],
[1,138,178],[0,146,180],[179,139,200],[0,145,188],[3,99,208],
[1,105,209],[179,127,191],[179,128,191],[0,139,182],[177,133,178],
[179,113,203],[2,130,192],[7,139,185],[11,119,208],[11,123,208],
[8,116,213],[4,111,212],[4,106,216],[6,108,222],[11,125,202],
[2,148,176],[4,158,170],[5,150,179],[5,150,179],[5,142,183],
[11,129,198],[5,158,177],[5,146,185],[3,159,156],[9,132,205],
[12,119,204],[8,135,194],[7,136,189],[7,142,182],[4,150,165],
[5,158,176],[8,139,196],[2,122,182],[3,119,195],[2,97,205],
[3,137,198],[1,102,221],[1,101,218],[4,130,182],[3,132,190],
[2,133,194],[179,129,217],[0,118,208],[178,128,223],[176,115,232],
[176,106,238],[178,124,206],[178,110,231],[1,147,185],[179,135,191],
[177,135,209],[176,143,216],[178,137,206],[179,135,202],[2,184,101],
[3,172,104],[6,167,119],[8,148,129],[6,159,128],[8,154,137],
[3,158,157],[6,127,166],[4,140,162],[4,171,136],[6,115,175],
[5,125,167],[3,109,182],[178,115,193],[2,137,162],[3,123,179],
[1,106,193],[178,106,183],[1,102,202],[175,111,211],[1,130,190],
[175,133,225],[174,117,238],[174,112,244],[175,126,235],[173,104,248],
[178,111,227],[1,109,208],[177,113,215],[178,111,219],[179,126,211],
[0,95,220]
], dtype=np.int32)

negative_samples = np.array([
[18,50,208],[18,52,201],[20,47,195],[18,53,187],[16,67,149],
[18,57,173],[14,64,184],[16,64,183],[15,68,202],[9,83,191],
[13,74,178],[17,45,202],[17,45,204],[17,44,208],[16,56,202],
[12,81,160],[13,42,201],[15,40,221],[23,12,247],[21,14,237],
[14,13,252],[10,19,240],[14,35,229],[14,85,180],[13,94,169],
[13,82,193],[12,93,173],[15,74,175],[16,120,155],[14,128,149],
[14,125,143],[13,70,175],[15,84,149],[20,57,222],[20,63,220],
[5,80,185],[7,73,193],[19,62,214],[26,35,220],[17,80,176],
[17,80,159],[15,58,198],[18,68,154],[16,72,183],[0,0,0],
[0,0,0],[0,0,0],[0,0,0],[0,0,0],[13,100,64],[11,20,104],
[94,26,147],[90,20,200],[15,12,86],[10,85,165],[11,83,166],
[10,63,205],[9,60,221],[13,57,225],[11,66,214],[11,68,185],
[11,68,183],[12,76,179],[12,70,185],[11,62,186],[9,64,191],
[11,69,200],[11,71,195],[11,77,178],[12,86,155],[11,109,136],
[13,117,135],[13,110,146],[12,104,140],[13,121,129],[13,122,128],
[12,81,161],[11,65,217],[15,49,234],[16,40,237],[13,46,229],
[11,53,225],[13,70,179],[0,255,14],[17,245,26],[16,233,23],
[14,66,50],[101,15,136],[15,88,199],[13,136,71],[17,61,210],
[16,74,159],[20,60,152],[16,83,136],[16,83,154],[2,116,90],
[10,144,99],[13,108,118],[10,116,125],[11,108,139],[12,94,176],
[13,84,191],[14,88,189],[14,87,191],[15,74,183],[15,72,183],
[16,70,182],[15,71,155],[13,76,167],[11,135,115],[14,73,204],
[15,55,207],[21,14,236],[19,19,232],[30,11,238],[16,81,192],
[14,106,135],[14,95,150],[12,102,142],[14,68,180],[14,77,191],
[0,255,30],[178,195,47],[16,58,163],[17,58,176],[15,57,194],
[18,51,213],[15,58,208],[14,58,206],[13,61,202],[13,60,205],
[15,97,155],[11,71,168],[14,62,192],[14,64,188],[15,64,171],
[15,67,164],[14,91,174],[13,87,167],[14,82,184],[13,82,195],
[8,85,184],[10,77,193],[9,72,209],[10,64,218],[8,63,218],
[10,64,203],[10,56,213],[7,59,224],[7,64,226],[12,80,166],
[16,60,195],[17,55,205],[18,47,213],[17,45,211],[16,46,204],
[13,57,191],[11,61,188],[12,83,179],[14,93,170],[13,98,161],
[13,92,169],[14,76,179],[11,67,179],[14,118,158],[16,97,168],
[17,77,178],[12,117,142],[13,107,150],[13,101,154],[13,100,155],
[13,116,143],[11,88,157],[11,76,179],[12,78,163],[12,86,154],
[12,80,162],[13,63,194],[13,61,204],[14,38,216],[13,36,224],
[18,11,242],[16,43,213],[21,22,236],[19,28,224],[14,42,221]
], dtype=np.int32)

X = np.vstack((positive_samples, negative_samples)).astype(np.float32)
y = np.hstack((np.ones(len(positive_samples)),
               np.zeros(len(negative_samples)))).astype(np.int32).reshape(-1, 1)

knn = cv2.ml.KNearest_create()
knn.train(X, cv2.ml.ROW_SAMPLE, y)
'''
# ★ 預先建立 morphology 核心（避免每張圖重建）
_MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# =========================
# 2️⃣ 增艷處理（in-place 優化）
# =========================
def process_vivid(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    # ★ in-place 乘飽和度，減少一次陣列分配
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255)
    img_sat = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gamma = 1.3
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(256)], dtype=np.uint8)
    img_gamma = cv2.LUT(img_sat, table)
    return cv2.convertScaleAbs(img_gamma, alpha=1.1, beta=-15)

'''
# =========================
# 3️⃣ ML 像素分類（批次優化）
# =========================
_KNN_BATCH = 65536   # 每批 64K 像素（約 256×256），平衡記憶體與速度

def ml_classify_mask(hsv_img):
    """
    分批次 KNN 推論：避免一次建立整張圖的超大距離矩陣
    原本整張 1080p 圖（~2M 像素）一次推，改為每批 64K
    """
    h, w = hsv_img.shape[:2]
    flat = hsv_img.reshape(-1, 3).astype(np.float32)
    n = len(flat)

    result = np.zeros(n, dtype=np.float32)
    for start in range(0, n, _KNN_BATCH):
        batch = flat[start:start + _KNN_BATCH]
        _, res, _, _ = knn.findNearest(batch, k=5)
        result[start:start + _KNN_BATCH] = res.ravel()

    mask = (result == 1).reshape(h, w).astype(np.uint8) * 255
    return mask

'''
# =========================
# 4️⃣ 主偵測流程
# =========================
'''
def detect_plaque(image_path):
    output_dir = str(_PATHS['teeth_color_test'])
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print("❌ 找不到圖片")
        return

    vivid   = process_vivid(img)
    blurred = cv2.GaussianBlur(vivid, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # ML mask（批次推論）
    # ml_mask = ml_classify_mask(hsv)

    # 傳統 HSV mask
    # Plaque dye: pink/rose on tooth surface (moderate sat, moderate-high value)
    # Gum: dark saturated red (high sat S≥150, lower value V≤185) → excluded separately
    lower_A = np.array([0,   40, 160])
    upper_A = np.array([15, 255, 255])
    lower_B = np.array([160, 40, 160])
    upper_B = np.array([180, 255, 255])
    hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_A, upper_A),
        cv2.inRange(hsv, lower_B, upper_B)
    )

    # Exclude dark saturated red/orange (dental model gum + plastic base)
    gum_excl_A = cv2.inRange(hsv, np.array([0,   140, 50]), np.array([22,  255, 190]))
    gum_excl_B = cv2.inRange(hsv, np.array([155, 140, 50]), np.array([180, 255, 190]))
    gum_mask   = cv2.bitwise_or(gum_excl_A, gum_excl_B)

    # 融合 + 型態學（使用預先建立的核心）
    final_mask = cv2.bitwise_and(ml_mask, hsv_mask)
    final_mask = cv2.bitwise_and(final_mask, cv2.bitwise_not(gum_mask))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    # ★ 向量化輪廓面積過濾（取代 if 迴圈）
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output     = img.copy()
    clean_mask = np.zeros_like(final_mask)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 80]
    if valid_contours:
        cv2.drawContours(output,     valid_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(clean_mask, valid_contours, -1, 255, -1)

    name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, "det_"  + name), output)
    cv2.imwrite(os.path.join(output_dir, "mask_" + name), clean_mask)

    print("✅ ML + HSV 偵測完成")
    print(f"📂 {output_dir}")
'''
def detect_plaque(image_path):
    output_dir = str(_PATHS['teeth_color_test'])
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        print("❌ 找不到圖片")
        return

    vivid   = process_vivid(img)
    blurred = cv2.GaussianBlur(vivid, (5, 5), 0)
    hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # --- 🟢 1. 牙菌斑偵測區間 (重新放寬門檻) ---
    # 將 S 降回 50，確保能抓到大部分菌斑，但 H 上限鎖在 172 避開正紫色牙齦
    lower_plaque = np.array([145, 50, 80]) 
    upper_plaque = np.array([172, 255, 255])
    
    lower_red = np.array([0, 50, 150])
    upper_red = np.array([15, 255, 255])

    hsv_mask = cv2.bitwise_or(
        cv2.inRange(hsv, lower_plaque, upper_plaque),
        cv2.inRange(hsv, lower_red, upper_red)
    )

    # --- 🔴 2. 排除遮罩 (精準剔除模型與牙齦) ---
    
    # A. 排除「淡色反光與陰影」(S < 45 的區域全殺)
    shadow_excl = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 45, 255]))
    
    # B. 🔥 核心修正：精準排除牙齦模型色 🔥
    # 根據圖片，牙齦色主要落在 H[173-180] 且 S 在 [100-180] 之間
    # 我們精準定義這個區間作為排除禁區
    gum_model_excl = cv2.inRange(hsv, np.array([170, 40, 40]), np.array([180, 180, 240]))
    
    # C. 廣域底座與深色排除
    gum_base_excl = cv2.inRange(hsv, np.array([0, 130, 0]), np.array([30, 255, 180]))

    # 合併所有排除遮罩
    gum_mask = cv2.bitwise_or(shadow_excl, gum_model_excl)
    gum_mask = cv2.bitwise_or(gum_mask, gum_base_excl)

    # --- 3. 最終融合 ---
    # 這裡多加一個步驟：把「太亮」的牙齦邊緣也順便去掉
    final_mask = cv2.bitwise_and(hsv_mask, cv2.bitwise_not(gum_mask))
    
    # 型態學：先開再閉
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN,  _MORPH_KERNEL)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, _MORPH_KERNEL)

    # 輪廓偵測與過濾
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output     = img.copy()
    clean_mask = np.zeros_like(final_mask)

    # 面積門檻維持 150 即可，不用太高
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 150]
    
    if valid_contours:
        cv2.drawContours(output,     valid_contours, -1, (0, 0, 255), 2)
        cv2.drawContours(clean_mask, valid_contours, -1, 255, -1)

    name = os.path.basename(image_path)
    cv2.imwrite(os.path.join(output_dir, "det_"  + name), output)
    cv2.imwrite(os.path.join(output_dir, "mask_" + name), clean_mask)

    print(f"✅ 修正版偵測完成：已調降菌斑飽和度門檻，並精準鎖定模型牙齦色排除。")


# =========================
# 執行（五張照片）
# =========================
BASE_PATH = str(_PATHS["real_teeth"])
PHOTOS = [
    'front.jpg', 'left_side.jpg', 'right_side.jpg',
    'upper_occlusal.jpg', 'lower_occlusal.jpg',
]

for photo in PHOTOS:
    input_path = os.path.join(BASE_PATH, photo)
    if os.path.exists(input_path):
        print(f"\n處理: {photo}")
        detect_plaque(input_path)
    else:
        print(f"⚠️  找不到: {input_path}")