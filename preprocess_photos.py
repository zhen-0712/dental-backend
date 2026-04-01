#!/usr/bin/env python3
"""
照片預處理 v4 - 高品質照片輕度處理版
適用：手機/相機拍攝的清晰照片（不是 ESP32-CAM）
策略：保守處理，不破壞原始品質
"""

import cv2
import numpy as np

import sys; sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")
from user_env import get_paths, setup_user_dirs, get_user_dir
_PATHS = get_paths()
setup_user_dirs(_PATHS["user_dir"])
INPUT_DIR  = _PATHS["real_teeth"]
OUTPUT_DIR = _PATHS["real_teeth_proc"]

# ==================== 視角設定 ====================
# light_type 根據你的照片來源調整：
#   'phone_normal'  → 手機/相機自然光（大部分照片）
#   'phone_405nm'   → 手機拍攝但有405nm光源（如果有的話）
#   'endoscope'     → 內視鏡暗場（如果下咬合還是用裝置拍）
PHOTO_CONFIG = {
    'front.jpg':          'phone_normal',
    'right_side.jpg':     'phone_normal',
    'left_side.jpg':      'phone_normal',
    'upper_occlusal.jpg': 'phone_normal',
    'lower_occlusal.jpg': 'phone_normal',  # 如果是手機拍的就用 phone_normal
}

# ==================== 模糊評分 ====================
def blur_score(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ==================== 處理函數 ====================

def suppress_highlights(img, threshold=240, strength=0.6):
    """
    壓制高光（口水反光）
    threshold: 高光判斷門檻（越高越保守，建議 235~250）
    strength:  填補強度（0=不填補，1=完全 inpaint）
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    # 高光 = 極亮(v>threshold) + 低飽和(s<50)
    mask = ((v_ch > threshold) & (s_ch < 50)).astype(np.uint8) * 255

    highlight_pixels = mask.sum() // 255
    if highlight_pixels == 0:
        return img, 0

    # 只在高光面積不要太大時才處理（避免把整顆牙都當高光）
    total_pixels = img.shape[0] * img.shape[1]
    highlight_ratio = highlight_pixels / total_pixels

    if highlight_ratio > 0.15:
        # 高光太多，可能判斷錯誤，只輕微壓制
        print(f"     ⚠️  高光佔比 {highlight_ratio*100:.1f}%，改用輕壓模式")
        v_ch_new = np.where(v_ch > threshold,
                            (v_ch * 0.85).astype(np.uint8),
                            v_ch)
        result = cv2.cvtColor(cv2.merge([h_ch, s_ch, v_ch_new]), cv2.COLOR_HSV2BGR)
        return result, highlight_pixels

    # 正常：膨脹 + Inpaint
    kernel = np.ones((3, 3), np.uint8)
    mask_d = cv2.dilate(mask, kernel, iterations=1)
    result = cv2.inpaint(img, mask_d, 3, cv2.INPAINT_TELEA)
    return result, highlight_pixels

def gentle_clahe(img, clip=1.2, tile=8):
    """
    輕度 CLAHE
    clip 很小（1.2），只是微調對比，不做激進增強
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def mild_white_balance(img, strength=0.4):
    """
    輕度白平衡（只修正輕微偏色）
    strength: 0=不調整, 1=完全灰世界校正
    比完整灰世界更保守，避免過度改變顏色
    """
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)

    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    overall = (b_mean + g_mean + r_mean) / 3.0

    # 完整校正係數
    b_scale_full = np.clip(overall / (b_mean + 1e-6), 0.7, 1.5)
    g_scale_full = np.clip(overall / (g_mean + 1e-6), 0.7, 1.3)
    r_scale_full = np.clip(overall / (r_mean + 1e-6), 0.7, 1.2)

    # 用 strength 混合（部分校正）
    b_scale = 1.0 + (b_scale_full - 1.0) * strength
    g_scale = 1.0 + (g_scale_full - 1.0) * strength
    r_scale = 1.0 + (r_scale_full - 1.0) * strength

    result = cv2.merge([
        np.clip(b * b_scale, 0, 255),
        np.clip(g * g_scale, 0, 255),
        np.clip(r * r_scale, 0, 255),
    ]).astype(np.uint8)
    return result

def gentle_sharpen(img, strength=0.5):
    """
    輕度銳化（Unsharp Mask）
    strength 很小，只是讓邊緣稍微清晰
    """
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def moderate_clahe_for_405nm(img, clip=2.0, tile=8):
    """405nm 光源照片用的稍強 CLAHE"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

def strong_white_balance_405nm(img):
    """405nm 強偏色校正"""
    img_float = img.astype(np.float32)
    b, g, r = cv2.split(img_float)
    b_mean, g_mean, r_mean = np.mean(b), np.mean(g), np.mean(r)
    overall = (b_mean + g_mean + r_mean) / 3.0
    b_scale = np.clip(overall / (b_mean + 1e-6), 0.5, 2.5)
    g_scale = np.clip(overall / (g_mean + 1e-6), 0.5, 2.0)
    r_scale = np.clip(overall / (r_mean + 1e-6), 0.5, 1.3)
    return cv2.merge([
        np.clip(b * b_scale, 0, 255),
        np.clip(g * g_scale, 0, 255),
        np.clip(r * r_scale, 0, 255),
    ]).astype(np.uint8)

def gamma_correct(img, gamma=1.8):
    table = np.array([(i / 255.0) ** (1.0 / gamma) * 255
                      for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, table)

def pad_to_square(img, target=512):
    h, w = img.shape[:2]
    scale = target / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    pad_h, pad_w = target - nh, target - nw
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right  = pad_w // 2, pad_w - pad_w // 2
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                               cv2.BORDER_CONSTANT, value=[128, 128, 128])

# ==================== 各光源流程 ====================

def process_phone_normal(img):
    """
    手機/相機高品質自然光照片
    只做最低限度處理
    """
    score = blur_score(img)
    print(f"     模糊分數: {score:.0f} ", end="")
    print("(清晰✅)" if score > 200 else "(稍模糊)" if score > 80 else "(模糊⚠️)")

    # 1. 壓制高光反光
    result, n_highlights = suppress_highlights(img, threshold=240, strength=0.6)
    print(f"     高光像素: {n_highlights}")

    # 2. 輕度白平衡（只校正輕微偏色）
    result = mild_white_balance(result, strength=0.3)

    # 3. 輕度 CLAHE（clip=1.2，非常保守）
    result = gentle_clahe(result, clip=1.2, tile=8)

    # 4. 輕度銳化（只在稍模糊時）
    if score < 300:
        result = gentle_sharpen(result, strength=0.4)

    return result

def process_phone_405nm(img):
    """
    手機拍攝但有 405nm 光源
    需要白平衡校正，但銳化保守
    """
    score = blur_score(img)
    print(f"     模糊分數: {score:.0f}")

    result = strong_white_balance_405nm(img)
    result = moderate_clahe_for_405nm(result, clip=2.0)
    result = gentle_sharpen(result, strength=0.6)
    return result

def process_endoscope(img):
    """
    內視鏡暗場（如果下咬合用裝置拍）
    """
    score = blur_score(img)
    print(f"     模糊分數: {score:.0f}")

    result = gamma_correct(img, gamma=1.8)
    lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    if score < 100:
        result = gentle_sharpen(result, strength=1.2)
    else:
        result = gentle_sharpen(result, strength=0.6)
    return result

# ==================== 主流程 ====================

PROCESSORS = {
    'phone_normal': process_phone_normal,
    'phone_405nm':  process_phone_405nm,
    'endoscope':    process_endoscope,
}

print("="*60)
print("📸 照片預處理 v4 - 高品質照片輕度版")
print("="*60)
print(f"輸入: {INPUT_DIR}")
print(f"輸出: {OUTPUT_DIR}")

count = 0
for photo_name, light_type in PHOTO_CONFIG.items():
    input_path = INPUT_DIR / photo_name
    if not input_path.exists():
        print(f"\n  ⚠️  找不到: {photo_name}")
        continue

    print(f"\n{'─'*50}")
    print(f"  📷 {photo_name}  [{light_type}]")

    img = cv2.imread(str(input_path))
    if img is None:
        print(f"  ❌ 無法讀取")
        continue

    h, w = img.shape[:2]
    print(f"     原始尺寸: {w}x{h}")

    processor = PROCESSORS[light_type]
    result = processor(img)
    final = pad_to_square(result, target=512)

    out_path = (OUTPUT_DIR / photo_name).with_suffix('.jpg')
    cv2.imwrite(str(out_path), final)
    print(f"  ✅ 輸出: {out_path.name} (512x512)")
    count += 1

print(f"\n{'='*60}")
print(f"✅ 完成！共 {count} 張")
print(f"{'='*60}")
print(f"\n💡 下一步: python analyze_real_teeth.py")