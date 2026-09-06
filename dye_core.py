#!/usr/bin/env python3
"""
dye_core.py

染色劑（disclosing agent）模式的牙菌斑偵測 —— 真實牙齒版。

★ 這個模組只服務 DENTAL_MODEL_TYPE="regular"（真實牙齒）。
  假牙教學模型（teaching）仍走 teeth_test.py 內原有的 HSV 門檻，一字未改。

────────────────────────────────────────────────────────────────
為什麼真實牙齒不能用教學模型那組門檻
────────────────────────────────────────────────────────────────
原本的門檻是針對「塑膠假牙模型」調的，排除項 gum_model_excl / gum_base_excl
鎖定的是模型牙齦與底座的特定顏色。真實口腔沒有那種顏色，而真實牙齦、嘴唇、
皮膚全都是粉紅到洋紅——跟染劑同一個色系。實測直接套用會把
**28~55% 的整張影像**判成菌斑。

────────────────────────────────────────────────────────────────
方法
────────────────────────────────────────────────────────────────
1. 牙齒 ROI 用 SAT（YOLO + SAM）取得，並帶 FDI 齒號。
   白光照片正是 SegmentAnyTooth 的原始設計場景，實測每視角可切出
   10~14 顆牙、ROI 僅佔全圖 3.6~8.3%，遠優於任何顏色規則。

2. 排除「上下排咬合接觸帶」。
   上排牙齒會在下排牙齒緊鄰接觸點處投下陰影，那條帶會被誤判成染劑。
   實測原本有 26% 的偵測落在該處。判定方式是找「上排 FDI 與下排 FDI
   都靠近」的區域，不是靠亮度——實測亮度分不開（標註內 167 / 標註外 168）。

3. 在 ROI 內用 **Lab a\* 的局部對比**判定，而非絕對顏色門檻。
   染劑在牙面上是淡粉紅薄膜，絕對 R/G 可能只比乾淨琺瑯質高 0.08，
   但肉眼看得到——因為人眼是「跟周圍比較」。局部背景相減正是模擬這件事。
   絕對門檻實測會先砍掉真正的染色區、只留下更紅的牙齦，方向相反。

實測（5 張真實牙齒染色劑照片、23 個標註區域）：
   偵測佔 ROI  28~46% → 7.6%    圈外誤標 14.6% → 5.6%    區域命中 42% → 52%
"""

import os
import numpy as np
import cv2

DEFAULT_CONFIG = {
    "work_long_side": 900,      # 統一工作解析度，避免參數隨原圖尺寸失準
    "pre_blur_sigma": 2.0,
    "local_win": 141,           # 局部背景視窗（工作解析度下的 px）
    "z_thresh": 1.0,            # 局部殘差 / MAD
    # 最小區塊面積。原設 200 會把顏色訊號正確、z 值通過門檻的小區塊也濾掉
    # ——實測 b5 的兩個標註區塊 z=1.44/1.65（均通過 z_thresh=1.0），開/閉運算
    # 後仍有 193px / 149px，卻因低於 200 被判定為雜訊濾除。拿掉此門檻，
    # 純靠 z-score 與形態學開閉運算本身去除雜訊。
    "min_area": 0,
    "morph_ksize": 7,
    "occlusal_band_px": 35,     # 上下排咬合接觸帶的排除寬度，設 0 停用

    # SAT 的 YOLO 信心門檻。預設 0.20 會漏掉畫面中偏小的牙齒——實測某批
    # 受測者牙齒較小、照片解析度較低時，front 視角在 0.20 只偵測到 13 顆，
    # 降到 0.05 可得 23 顆，標註被 ROI 涵蓋的比例從 12% 提升到 28%。
    # 僅影響染色劑模式；extract_plaque_regions.py 的下游 ROI 過濾不受影響。
    "sat_conf": 0.20,
}

UPPER_FDI = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27, 18, 28]
LOWER_FDI = [31, 32, 33, 34, 35, 36, 37, 41, 42, 43, 44, 45, 46, 47, 38, 48]

_SAM = None
_YOLO = None


def pad_to_square(img, target=512):
    """置中縮放成正方形，灰底填充。與 preprocess_photos.py 相同的幾何，
    但不做高光壓制／白平衡／CLAHE——那些會讓 SAT 在小尺寸、低解析度的
    照片上表現變差（實測 b2 front 涵蓋率 12%→28%、right_side 10%→36%）。
    菌斑偵測需要保留原始色彩訊號，SAT 只需要幾何形狀，兩者用同一張圖
    反而互相拖累。"""
    h, w = img.shape[:2]
    s = target / max(h, w)
    nh, nw = int(h * s), int(w * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    ph, pw = target - nh, target - nw
    return cv2.copyMakeBorder(r, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2,
                              cv2.BORDER_CONSTANT, value=[128, 128, 128])


def load_models(weight_dir):
    """延遲載入 SAT 模型，同一個 process 內只載一次。"""
    global _SAM, _YOLO
    if _SAM is not None:
        return _SAM, _YOLO
    from segmentanytooth import get_model_path
    from sam import sam_load
    from ultralytics import YOLO
    from utils import suppress_stdout
    with suppress_stdout():
        _SAM = sam_load(get_model_path("sam", str(weight_dir)))
        _YOLO = {v: YOLO(model=get_model_path(v, str(weight_dir)))
                 for v in ("front", "right", "upper", "lower")}
    return _SAM, _YOLO


def sat_fdi_mask(proc_img, sat_view, weight_dir, conf=None):
    """
    對前處理後的 512×512 影像跑 SAT，回傳同尺寸的 FDI 遮罩（0=非牙齒）。
    左側視角沿用 extract_plaque_regions 的做法：水平翻轉後借用 right 模型。
    """
    from segmentanytooth import LEFT_CLASSES
    from sam import sam_predict
    from utils import suppress_stdout
    sam, yolo = load_models(weight_dir)
    if conf is None:
        conf = DEFAULT_CONFIG["sat_conf"]

    flip = sat_view == "left"
    img = cv2.flip(proc_img, 1) if flip else proc_img
    with suppress_stdout():
        r = yolo["right" if flip else sat_view].predict(
            img, conf=conf, save=False, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return np.zeros(proc_img.shape[:2], np.uint8)

    names = LEFT_CLASSES if flip else r.names
    boxes = r.boxes.xyxy.cpu().numpy()
    clss = r.boxes.cls.cpu().numpy().astype(np.int32)
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]
    if clss.ndim == 0:
        clss = clss[np.newaxis]
    if flip:
        w = img.shape[1]
        img = cv2.flip(img, 1)
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

    with suppress_stdout():
        masks = sam_predict(sam=sam, boxes_xyxy=boxes,
                            image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                            batch_size=10)
    fdi = np.zeros(proc_img.shape[:2], np.uint8)
    for c, m in zip(clss, masks):
        fdi[m == 1] = int(names[c][-2:])
    return fdi


def unpad_to_original(fdi_512, orig_w, orig_h, target=512):
    """preprocess_photos.pad_to_square 的反運算：去掉置中留白再放大回原圖。"""
    s = target / max(orig_h, orig_w)
    nh, nw = int(orig_h * s), int(orig_w * s)
    top, left = (target - nh) // 2, (target - nw) // 2
    return cv2.resize(fdi_512[top:top + nh, left:left + nw],
                      (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def build_roi(fdi, cfg):
    """牙齒 ROI，並排除上下排咬合接觸帶。"""
    roi = fdi > 0
    bp = cfg["occlusal_band_px"]
    if bp:
        k = np.ones((bp, bp), np.uint8)
        du = cv2.dilate(np.isin(fdi, UPPER_FDI).astype(np.uint8), k)
        dl = cv2.dilate(np.isin(fdi, LOWER_FDI).astype(np.uint8), k)
        roi = roi & ~((du > 0) & (dl > 0))
    return roi


def detect(orig_bgr, fdi_orig, cfg=None):
    """
    輸入原圖與同尺寸的 FDI 遮罩，回傳原圖解析度的二值菌斑遮罩與統計。
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    OH, OW = orig_bgr.shape[:2]
    s = cfg["work_long_side"] / max(OH, OW)
    im = cv2.resize(orig_bgr, (int(OW * s), int(OH * s)), interpolation=cv2.INTER_AREA)
    fdi = cv2.resize(fdi_orig, im.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    roi = build_roi(fdi, cfg)
    empty = np.zeros((OH, OW), np.uint8)
    if roi.sum() < 500:
        return empty, {"error": "牙齒區域太小", "tooth_px": int(roi.sum())}

    a = cv2.cvtColor(cv2.GaussianBlur(im, (0, 0), cfg["pre_blur_sigma"]),
                     cv2.COLOR_BGR2LAB)[..., 1].astype(np.float32)
    m8 = roi.astype(np.float32)
    win = cfg["local_win"] | 1
    num = cv2.boxFilter(a * m8, -1, (win, win), normalize=False)
    den = cv2.boxFilter(m8, -1, (win, win), normalize=False)
    res = a - num / (den + 1e-3)

    rv = res[roi]
    med = float(np.median(rv))
    mad = max(float(np.median(np.abs(rv - med))) * 1.4826, 0.3)
    m = (((res - med) / mad >= cfg["z_thresh"]) & roi).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg["morph_ksize"],) * 2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.array([False] + [stats[i, cv2.CC_STAT_AREA] >= cfg["min_area"]
                               for i in range(1, n)])
    m = np.where(keep[labels], 255, 0).astype(np.uint8)

    full = cv2.resize(m, (OW, OH), interpolation=cv2.INTER_NEAREST)
    _, full = cv2.threshold(full, 127, 255, cv2.THRESH_BINARY)
    return full, {
        "tooth_px": int(roi.sum()),
        "plaque_px": int((m > 0).sum()),
        "plaque_ratio_of_tooth": round(float((m > 0).sum()) / max(int(roi.sum()), 1), 4),
        "region_count": int(keep.sum()),
        "residual_mad": round(mad, 3),
        "fdi_detected": sorted(int(v) for v in np.unique(fdi) if v > 0),
    }
