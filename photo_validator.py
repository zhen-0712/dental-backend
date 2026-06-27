#!/usr/bin/env python3
"""
照片品質與視角驗證器

Layer 1 (純演算法, ~50ms):
  - 模糊偵測  — Laplacian variance
  - 亮度檢查  — 過暗 / 過曝
  - 無效圖片  — 無法解碼

Layer 2 (幾何視角分類, ~100-150ms, lazy load):
  - 只用 front 模型（偵測全部 32 顆牙）
  - 取出 FDI 編號 + bbox 中心位置
  - 套幾何規則（門牙有無、左右對稱、弓形寬度）判斷視角
  - 不需額外訓練資料
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

# ── 常數 ──────────────────────────────────────────────────────────────────

BLUR_THRESHOLD  = 80
BRIGHTNESS_MIN  = 35
BRIGHTNESS_MAX  = 215

_WEIGHT_DIR = Path(__file__).parent / "weight"

VIEW_NAMES_ZH = {
    "front":          "正面",
    "left_side":      "左側",
    "right_side":     "右側",
    "upper_occlusal": "上顎咬合面",
    "lower_occlusal": "下顎咬合面",
}

# ── 資料結構 ─────────────────────────────────────────────────────────────

@dataclass
class PhotoError:
    view:    str
    code:    str      # "invalid" | "blur" | "dark" | "overexposed" | "wrong_angle"
    message: str

@dataclass
class ValidationResult:
    ok:     bool
    errors: list[PhotoError] = field(default_factory=list)


# ── Layer 1: 基本品質檢查 ─────────────────────────────────────────────────

def _to_cv2(raw: bytes) -> np.ndarray | None:
    arr = np.frombuffer(raw, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def validate_quality(view: str, raw: bytes) -> list[PhotoError]:
    """Layer 1：模糊 + 亮度檢查，回傳錯誤清單（空 = 通過）。"""
    errors: list[PhotoError] = []
    name = VIEW_NAMES_ZH.get(view, view)

    img = _to_cv2(raw)
    if img is None:
        errors.append(PhotoError(view=view, code="invalid",
                                  message=f"{name}：無法讀取圖片，請重新上傳"))
        return errors

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < BLUR_THRESHOLD:
        errors.append(PhotoError(view=view, code="blur",
                                  message=f"{name}：照片太模糊，請重新拍攝"))

    mean = float(img.mean())
    if mean < BRIGHTNESS_MIN:
        errors.append(PhotoError(view=view, code="dark",
                                  message=f"{name}：照片太暗，請在光線充足處拍攝"))
    elif mean > BRIGHTNESS_MAX:
        errors.append(PhotoError(view=view, code="overexposed",
                                  message=f"{name}：照片過曝，請避免強光直射"))

    return errors


# ── Layer 2: 幾何視角分類（front 模型 + FDI 位置規則）───────────────────

_yolo_clf: dict = {}

# FDI 象限
_FDI_UPPER_R = frozenset({11, 12, 13, 14, 15, 16, 17, 18})
_FDI_UPPER_L = frozenset({21, 22, 23, 24, 25, 26, 27, 28})
_FDI_LOWER_L = frozenset({31, 32, 33, 34, 35, 36, 37, 38})
_FDI_LOWER_R = frozenset({41, 42, 43, 44, 45, 46, 47, 48})
_FDI_UPPER        = _FDI_UPPER_R | _FDI_UPPER_L
_FDI_LOWER        = _FDI_LOWER_R | _FDI_LOWER_L
_FDI_PATIENT_RIGHT = _FDI_UPPER_R | _FDI_LOWER_R   # 患者右側（1x + 4x）
_FDI_PATIENT_LEFT  = _FDI_UPPER_L | _FDI_LOWER_L   # 患者左側（2x + 3x）
_FDI_POSTERIORS   = frozenset({
    14,15,16,17,18, 24,25,26,27,28,
    34,35,36,37,38, 44,45,46,47,48,
})
_FDI_FRONT_TEETH  = frozenset({11,12,13, 21,22,23, 31,32, 41,42})

# 判斷門檻
_GEO_RATIO_THRESH   = 0.78   # expected / best 低於此才報錯
_GEO_MIN_DETECTIONS = 2      # 偵測太少顆牙 → 不判斷


def _get_yolo_model(model_key: str):
    if model_key not in _yolo_clf:
        from ultralytics import YOLO
        from segmentanytooth import get_model_path
        print(f"[ViewCLF] Loading {model_key} model...", flush=True)
        _yolo_clf[model_key] = YOLO(model=get_model_path(model_key, str(_WEIGHT_DIR)))
    return _yolo_clf[model_key]


def _extract_detections(result, model) -> list[dict]:
    """YOLO result → [{fdi, cx, cy, conf}]，cx/cy 為 0-1 正規化。"""
    if result.boxes is None or len(result.boxes) == 0:
        return []
    h, w = result.orig_shape
    out = []
    for i in range(len(result.boxes)):
        cls_name = model.names[int(result.boxes.cls[i].item())]  # e.g. 'f11'
        fdi_str = cls_name[1:]   # 去掉前綴 f/r/u/l
        if not fdi_str.isdigit():
            continue
        fdi  = int(fdi_str)
        conf = float(result.boxes.conf[i].item())
        xyxy = result.boxes.xyxy[i].cpu().numpy()
        cx = float((xyxy[0] + xyxy[2]) / 2) / w
        cy = float((xyxy[1] + xyxy[3]) / 2) / h
        out.append({'fdi': fdi, 'cx': cx, 'cy': cy, 'conf': conf})
    return out


def _geo_scores(dets: list[dict]) -> dict[str, float]:
    """
    根據 front 模型偵測到的牙齒 FDI 編號與位置，計算各視角幾何合理分數 (0–1)。

    老師建議核心：以門牙為基準，數出現了哪幾顆、偏向哪一側，推斷視角。
      - 正面  : 11 與 21 同時出現，重心置中，前牙多於後牙
      - 右側面: 患者右側（FDI 1x/4x）牙齒佔多數，後牙可見
      - 左側面: 患者左側（FDI 2x/3x）牙齒佔多數，後牙可見
      - 咬合面: 同顎牙齒 ≥ 4 顆，水平弓形展開寬度大
    """
    if not dets:
        return {v: 0.0 for v in ['front','right_side','left_side','upper_occlusal','lower_occlusal']}

    fdis = [d['fdi'] for d in dets]
    fset = set(fdis)
    cxs  = [d['cx'] for d in dets]
    n    = len(dets)

    cx_mean = sum(cxs) / n
    arch_w  = (max(cxs) - min(cxs)) if n > 1 else 0.0

    n_up = sum(1 for f in fdis if f in _FDI_UPPER)
    n_lo = sum(1 for f in fdis if f in _FDI_LOWER)
    n_pr = sum(1 for f in fdis if f in _FDI_PATIENT_RIGHT)
    n_pl = sum(1 for f in fdis if f in _FDI_PATIENT_LEFT)
    has_11 = 11 in fset
    has_21 = 21 in fset

    def clamp(v: float) -> float:
        return min(max(v, 0.0), 1.0)

    # ── 正面 ──────────────────────────────────────────────────────────────
    sf = 0.0
    # 雙側門牙同時出現：但要確保左右牙齒數量均衡，否則是側面視角
    if has_11 and has_21:
        side_imbalance = abs(n_pr - n_pl) / max(n, 1)
        if side_imbalance <= 0.25:
            sf += 0.40   # 左右均衡 → 正面
        else:
            sf += 0.12   # 兩顆門牙都有，但明顯偏向一側 → 可能是側面
    elif has_11 or has_21:
        sf += 0.10
    if 0.28 < cx_mean < 0.72:  sf += 0.20  # 牙齒重心置中
    # 正面必須同時看到上下顎牙齒；只見一顎 → 懲罰（咬合面特徵）
    if n_up > 0 and n_lo > 0:  sf += 0.20
    else:                       sf -= 0.15  # 只見一顎牙，不像正面
    n_ft = sum(1 for f in fdis if f in _FDI_FRONT_TEETH)
    n_po = sum(1 for f in fdis if f in _FDI_POSTERIORS)
    if n_ft >= n_po:           sf += 0.20  # 前牙數量 ≥ 後牙
    # 弓形太寬 → 可能是咬合面
    if arch_w > 0.65:          sf -= 0.15

    # ── 右側面（患者右側 FDI 1x/4x 為主）────────────────────────────────
    sr = 0.0
    if n_pr >= 2 and n_pr > n_pl:
        dominance = min(n_pr / max(n_pl, 1), 4) / 4
        sr += 0.40 * dominance
    if not (has_11 and has_21):  sr += 0.20
    r_po = sum(1 for f in fdis if f in {13,14,15,16,17,18, 43,44,45,46,47,48})
    if r_po >= 2:                sr += 0.30
    if n >= 3:                   sr += 0.10

    # ── 左側面（患者左側 FDI 2x/3x 為主）────────────────────────────────
    sl = 0.0
    if n_pl >= 2 and n_pl > n_pr:
        dominance = min(n_pl / max(n_pr, 1), 4) / 4
        sl += 0.40 * dominance
    if not (has_11 and has_21):  sl += 0.20
    l_po = sum(1 for f in fdis if f in {23,24,25,26,27,28, 33,34,35,36,37,38})
    if l_po >= 2:                sl += 0.30
    if n >= 3:                   sl += 0.10

    # ── 上顎咬合面 ────────────────────────────────────────────────────────
    su = 0.0
    if n_up / max(n, 1) > 0.65:  su += 0.40
    if arch_w > 0.50:             su += 0.30
    if n_up >= 4:                 su += 0.30
    # 只看到上顎牙，沒有下顎 → 強烈咬合面指標
    if n_lo == 0 and n_up >= 3:   su += 0.20

    # ── 下顎咬合面 ────────────────────────────────────────────────────────
    sd = 0.0
    if n_lo / max(n, 1) > 0.65:  sd += 0.40
    if arch_w > 0.50:             sd += 0.30
    if n_lo >= 4:                 sd += 0.30
    # 只看到下顎牙，沒有上顎 → 強烈咬合面指標
    if n_up == 0 and n_lo >= 3:   sd += 0.20

    return {
        'front':          clamp(sf),
        'right_side':     clamp(sr),
        'left_side':      clamp(sl),
        'upper_occlusal': clamp(su),
        'lower_occlusal': clamp(sd),
    }


def validate_view_angle(view: str, raw: bytes) -> list[PhotoError]:
    """
    Layer 2：幾何視角分類。
    只用 front 模型（偵測全部 32 顆牙），取出 FDI 編號 + bbox 位置，
    套幾何規則判斷上傳視角是否正確。
    left_side 額外對水平翻轉圖再跑一次，取兩方向的較高分。
    """
    try:
        img = _to_cv2(raw)
        if img is None:
            return []

        model    = _get_yolo_model("front")
        r_normal = model.predict(img, verbose=False)[0]
        dets     = _extract_detections(r_normal, model)

        if view == 'left_side':
            # 翻轉後跑一次：翻轉圖的 right_side 分數 = 原圖 left_side 分數
            r_flip    = model.predict(cv2.flip(img, 1), verbose=False)[0]
            dets_flip = _extract_detections(r_flip, model)
            s_n = _geo_scores(dets)
            s_f = _geo_scores(dets_flip)
            geo = {
                'front':          max(s_n['front'],          s_f['front']),
                'right_side':     s_n['right_side'],
                'left_side':      max(s_n['left_side'],      s_f['right_side']),
                'upper_occlusal': max(s_n['upper_occlusal'], s_f['upper_occlusal']),
                'lower_occlusal': max(s_n['lower_occlusal'], s_f['lower_occlusal']),
            }
        else:
            geo = _geo_scores(dets)

        n_dets     = len(dets)
        best_view  = max(geo, key=lambda k: geo[k])
        best_score = geo[best_view]
        exp_score  = geo.get(view, 0.0)

        scores_str = " ".join(f"{k}={v:.2f}" for k, v in geo.items())
        print(
            f"[ViewCLF] geo n={n_dets} view={view} {scores_str} "
            f"→ best={best_view}({best_score:.2f}) exp={exp_score:.2f}",
            flush=True,
        )

        # 偵測牙齒太少 or 整體分數太低 → 沒有足夠信心，不擋
        if n_dets < _GEO_MIN_DETECTIONS or best_score < 0.20:
            return []

        if best_view != view:
            ratio = exp_score / best_score if best_score > 0 else 0.0
            if ratio < _GEO_RATIO_THRESH:
                expected_zh  = VIEW_NAMES_ZH.get(view, view)
                predicted_zh = VIEW_NAMES_ZH.get(best_view, best_view)
                return [PhotoError(
                    view=view,
                    code="wrong_angle",
                    message=f"{expected_zh}：這張照片看起來像{predicted_zh}，請確認上傳到正確位置",
                )]

    except Exception as e:
        print(f"[ViewCLF ERROR] view={view} {e}", flush=True)
    return []


# ── 對外主入口 ────────────────────────────────────────────────────────────

def validate_uploads(
    uploads: dict,
    *,
    skip_angle_check: bool = False,
) -> ValidationResult:
    """
    驗證所有視角照片。

    uploads: {"front": UploadFile, "left_side": UploadFile, ...}
    回傳 ValidationResult；呼叫端在 save_uploads 之前執行，
    result.ok == False 時應回傳 HTTP 422。

    注意：此函式會 seek(0) 後讀取但不會消耗 UploadFile，
    save_uploads 仍可正常讀取。
    """
    all_errors: list[PhotoError] = []

    for view, upload in uploads.items():
        upload.file.seek(0)
        raw = upload.file.read()
        upload.file.seek(0)  # reset 給後續 save_uploads 使用

        # Layer 1
        q_errors = validate_quality(view, raw)
        all_errors.extend(q_errors)

        # Layer 2：只在 Layer 1 通過時才跑，節省時間
        if not q_errors and not skip_angle_check:
            all_errors.extend(validate_view_angle(view, raw))

    return ValidationResult(ok=len(all_errors) == 0, errors=all_errors)
