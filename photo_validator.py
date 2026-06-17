#!/usr/bin/env python3
"""
照片品質與視角驗證器

Layer 1 (純演算法, ~50ms):
  - 模糊偵測  — Laplacian variance
  - 亮度檢查  — 過暗 / 過曝
  - 無效圖片  — 無法解碼

Layer 2 (YOLO 單模型, ~100-150ms, lazy load):
  - 只跑「預期視角」的 YOLO 模型
  - 偵測分數（置信度總和）過低 → 這張照片在此視角幾乎看不到牙齒 → 報錯
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

# API 視角 → (YOLO model key, 是否水平翻轉)
_API_TO_YOLO: dict[str, tuple[str, bool]] = {
    "front":          ("front", False),
    "left_side":      ("right", True),   # left 用 right 模型 + 水平翻轉
    "right_side":     ("right", False),
    "upper_occlusal": ("upper", False),
    "lower_occlusal": ("lower", False),
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


# ── Layer 2: YOLO 視角驗證 ────────────────────────────────────────────────

_yolo_clf: dict = {}

# 預期視角分數 / 最佳視角分數 低於此值才報錯
_SCORE_RATIO_THRESHOLD = 0.78


def _get_yolo_model(model_key: str):
    """Lazy load：只在第一次用到該模型時才載入。"""
    if model_key not in _yolo_clf:
        from ultralytics import YOLO
        from segmentanytooth import get_model_path
        print(f"[ViewCLF] Loading {model_key} model...", flush=True)
        _yolo_clf[model_key] = YOLO(model=get_model_path(model_key, str(_WEIGHT_DIR)))
    return _yolo_clf[model_key]


def _detect_score(model_key: str, img: np.ndarray) -> float:
    r = _get_yolo_model(model_key).predict(img, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return 0.0
    return float(r.boxes.conf.cpu().numpy().sum())


def validate_view_angle(view: str, raw: bytes) -> list[PhotoError]:
    """Layer 2：同時跑 4 個 YOLO 模型，比較哪個視角偵測分數最高。
    預期視角分數 / 最佳分數 低於門檻 → 照片可能放錯視角。
    """
    try:
        img = _to_cv2(raw)
        if img is None:
            return []

        img_flip = cv2.flip(img, 1)

        scores = {
            "front":          _detect_score("front", img),
            "right_side":     _detect_score("right", img),
            "left_side":      _detect_score("right", img_flip),
            "upper_occlusal": _detect_score("upper", img),
            "lower_occlusal": _detect_score("lower", img),
        }

        best_view    = max(scores, key=lambda k: scores[k])
        best_score   = scores[best_view]
        expected_score = scores.get(view, 0.0)

        print(f"[ViewCLF] view={view} best={best_view}({best_score:.2f}) expected={expected_score:.2f} ratio={expected_score/best_score:.2f}", flush=True)

        # 所有模型幾乎都偵測不到牙齒 → 不在這裡擋
        if best_score < 0.5:
            return []

        if best_view != view:
            ratio = expected_score / best_score if best_score > 0 else 0.0
            if ratio < _SCORE_RATIO_THRESHOLD:
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
