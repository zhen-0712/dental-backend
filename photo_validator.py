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
_cnn_clf:  dict = {}   # CNN 視角分類器（lazy load）

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

# CNN 開關（False = 純幾何模式，方便測試）
_CNN_ENABLED = True

# 判斷門檻
_GEO_RATIO_THRESH   = 0.70   # combined_exp / best_geo 低於此才報錯
_GEO_MIN_DETECTIONS = 2      # 偵測太少顆牙 → 不判斷
_GEO_WEIGHT         = 0.40   # 幾何分數權重
_SPEC_WEIGHT        = 0.60   # 專屬模型平均 confidence 權重

# 各視角最低可見牙齒數（低於此 → 開口不足或角度偏差）
_MIN_TEETH_BY_VIEW: dict[str, int] = {
    'front':          5,
    'left_side':      3,
    'right_side':     3,
    'upper_occlusal': 4,
    'lower_occlusal': 4,
}

# 每個視角對應的專屬 YOLO 模型（left_side 用翻轉圖跑 right 模型）
_SPEC_MODEL_KEY: dict[str, str] = {
    'right_side':     'right',
    'left_side':      'right',
    'upper_occlusal': 'upper',
    'lower_occlusal': 'lower',
}


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

    # ── cx_mean 牙齒重心側向判別（最可靠的左右信號）──────────────────────
    # 前置相機（app 已標準化方向）：右側照→牙齒偏左(cx<0.5)；左側照→牙齒偏右(cx>0.5)
    # 資料驗證：left_side cx_mean ≥0.589；right_side cx_mean ≤0.544
    if cx_mean < 0.45:      # 牙齒明顯偏左 → 強烈右側信號
        sr += 0.35
        sl -= 0.35
    elif cx_mean > 0.58:    # 牙齒明顯偏右 → 強烈左側信號
        sl += 0.35
        sr -= 0.35

    # ── 虎牙-門牙座標法（補充判別，當 cx_mean 落在灰色地帶）─────────────
    # 原理：側面照中，門牙在齒弓前端；cx_mean 無法分辨時，用門牙-虎牙相對位置補強。
    # 資料驗證：left_side lateral≈+0.20~+0.27；right_side lateral≈-0.08~-0.13
    incisor_dets = [d for d in dets if d['fdi'] in (11, 21)]
    canine_dets  = [d for d in dets if d['fdi'] in (13, 23)]
    if incisor_dets and canine_dets and 0.45 <= cx_mean <= 0.58:
        i_cx    = sum(d['cx'] for d in incisor_dets) / len(incisor_dets)
        c_cx    = sum(d['cx'] for d in canine_dets)  / len(canine_dets)
        lateral = i_cx - c_cx   # >0 → 左側, <0 → 右側
        if lateral > 0.05:
            sl += 0.30
            sr -= 0.15
        elif lateral < -0.05:
            sr += 0.30
            sl -= 0.15

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


def _check_insufficient_opening(view: str, dets: list[dict]) -> list[PhotoError]:
    """
    根據偵測到的牙齒數判斷開口是否足夠。
    低於各視角門檻 → 提示使用者張口更大或調整角度。
    """
    n = len(dets)
    threshold = _MIN_TEETH_BY_VIEW.get(view, 3)
    if 0 < n < threshold:
        view_zh = VIEW_NAMES_ZH.get(view, view)
        return [PhotoError(view=view, code="insufficient_opening",
            message=f"{view_zh}：照片中可見牙齒太少（{n} 顆），請張嘴更大或調整拍攝角度")]
    return []


def _cnn_classify(img) -> dict[str, float]:
    """CNN 單次推理，回傳所有視角信心分數 dict。模型不存在回傳空 dict。"""
    clf_path = _WEIGHT_DIR / "view_clf.pt"
    if not clf_path.exists():
        return {}

    if "model" not in _cnn_clf:
        import torch
        import torchvision.transforms as T
        import torchvision.models as tvm
        import torch.nn as nn
        ckpt = torch.load(str(clf_path), map_location="cpu")
        m = tvm.mobilenet_v3_small()
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(ckpt["classes"]))
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        _cnn_clf["model"]   = m
        _cnn_clf["classes"] = ckpt["classes"]
        _cnn_clf["tf"] = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        print("[ViewCLF] CNN classifier loaded", flush=True)

    try:
        import torch
        from PIL import Image
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x   = _cnn_clf["tf"](Image.fromarray(rgb)).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(_cnn_clf["model"](x), dim=1)[0].tolist()
        return dict(zip(_cnn_clf["classes"], probs))
    except Exception:
        return {}


def _spec_avg_conf(view: str, img) -> float:
    """
    跑對應視角的專屬 YOLO 模型，回傳所有偵測的平均 confidence（0-1）。
    left_side 用水平翻轉圖跑 right 模型（兩側對稱）。
    front 不需要專屬模型，直接回傳中性值 1.0（由幾何分數主導）。
    """
    model_key = _SPEC_MODEL_KEY.get(view)
    if not model_key:
        return 1.0  # front 視角：幾何分數已足夠

    model    = _get_yolo_model(model_key)
    run_img  = cv2.flip(img, 1) if view == 'left_side' else img
    result   = model.predict(run_img, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        return 0.0
    confs = result.boxes.conf.cpu().numpy()
    return float(confs.mean())


def validate_view_angle(view: str, raw: bytes, mirror: bool = False) -> list[PhotoError]:
    """
    Layer 2：視角分類。

    mirror=True 表示後置鏡頭原始照片（app 尚未翻轉），
    驗證前先水平翻轉使方向與前置鏡頭一致。

    決策順序：
      1. YOLO 偵測牙齒數 → 開口不足檢查（n 太少直接提示）
      2. CNN ≥ 0.55 → 通過；CNN ≤ 0.25 → 報錯
      3. CNN 不確定（0.25-0.55）→ 幾何備援（含虎牙門牙座標法）
    """
    try:
        img = _to_cv2(raw)
        if img is None:
            return []
        if mirror:
            img = cv2.flip(img, 1)

        # ── Step 1: YOLO 偵測（開口檢查 + 幾何備援共用）────────────────
        front_model = _get_yolo_model("front")
        r_normal    = front_model.predict(img, verbose=False)[0]
        dets        = _extract_detections(r_normal, front_model)

        # ── Step 2: 開口不足檢查（優先於角度判斷，直接提示）───────────────
        # 咬合面視角：front 模型無法可靠計牙數，改用各自專屬模型
        if view in ('upper_occlusal', 'lower_occlusal'):
            occ_key   = 'upper' if view == 'upper_occlusal' else 'lower'
            occ_model = _get_yolo_model(occ_key)
            r_occ     = occ_model.predict(img, verbose=False)[0]
            dets_occ  = _extract_detections(r_occ, occ_model)
            opening_err = _check_insufficient_opening(view, dets_occ)
        else:
            opening_err = _check_insufficient_opening(view, dets)
        if opening_err:
            n_check = len(dets_occ) if view in ('upper_occlusal', 'lower_occlusal') else len(dets)
            print(f"[ViewCLF] view={view} n={n_check} → INSUFFICIENT_OPENING", flush=True)
            return opening_err

        # ── Step 3: CNN 主要決策 ─────────────────────────────────────────
        def make_error(predicted: str) -> list[PhotoError]:
            exp_zh  = VIEW_NAMES_ZH.get(view, view)
            pred_zh = VIEW_NAMES_ZH.get(predicted, predicted)
            return [PhotoError(view=view, code="wrong_angle",
                message=f"{exp_zh}：這張照片看起來像{pred_zh}，請確認上傳到正確位置")]

        if _CNN_ENABLED:
            cnn_probs = _cnn_classify(img)
            cnn_score = cnn_probs.get(view, -1.0)
            cnn_top   = max(cnn_probs, key=cnn_probs.get) if cnn_probs else None

            # 左右側視角：CNN 對方向較不敏感，拉大幾何備援區間讓 cx_mean 多參與
            # 其他視角：維持原本嚴格門檻
            is_side = view in ('left_side', 'right_side')
            pass_thresh = 0.65 if is_side else 0.55
            fail_thresh = 0.15 if is_side else 0.25

            if cnn_score >= pass_thresh:
                print(f"[ViewCLF] view={view} cnn={cnn_score:.2f} top={cnn_top} → PASS", flush=True)
                return []
            if 0.0 <= cnn_score <= fail_thresh:
                if is_side:
                    # 左右側視角：先排除對向側高信心（此時一定是傳錯照片）
                    opposite = 'right_side' if view == 'left_side' else 'left_side'
                    cnn_opp = cnn_probs.get(opposite, 0.0)
                    if cnn_opp >= pass_thresh:
                        print(f"[ViewCLF] view={view} cnn={cnn_score:.2f} {opposite}={cnn_opp:.2f} → FAIL", flush=True)
                        return make_error(opposite)
                    # CNN 不確定方向（top 為正面/其他）→ 交幾何備援
                    print(f"[ViewCLF] view={view} cnn={cnn_score:.2f} {opposite}={cnn_opp:.2f} → side geo fallback", flush=True)
                else:
                    print(f"[ViewCLF] view={view} cnn={cnn_score:.2f} top={cnn_top} → FAIL", flush=True)
                    return make_error(cnn_top or view)
            cnn_top_fallback = cnn_top
        else:
            cnn_score        = -1.0
            cnn_top_fallback = None
            print(f"[ViewCLF] view={view} CNN_DISABLED → geo only", flush=True)

        # ── Step 4: CNN 不確定 → 幾何備援（虎牙門牙座標法已整合於 _geo_scores）
        if view == 'left_side':
            r_flip    = front_model.predict(cv2.flip(img, 1), verbose=False)[0]
            dets_flip = _extract_detections(r_flip, front_model)
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

        n_dets         = len(dets)
        best_geo_view  = max(geo, key=lambda k: geo[k])
        best_geo_score = geo[best_geo_view]
        geo_exp        = geo.get(view, 0.0)
        predicted      = cnn_top_fallback if cnn_top_fallback and cnn_top_fallback != view else best_geo_view

        scores_str = " ".join(f"{k}={v:.2f}" for k, v in geo.items())
        print(
            f"[ViewCLF] view={view} cnn={cnn_score:.2f} geo_n={n_dets} {scores_str} "
            f"best_geo={best_geo_view}({best_geo_score:.2f})",
            flush=True,
        )

        if n_dets < _GEO_MIN_DETECTIONS or best_geo_score < 0.20:
            return []

        if view == 'front':
            if best_geo_view != view:
                ratio = geo_exp / best_geo_score if best_geo_score > 0 else 1.0
                if best_geo_score >= 0.90 or ratio < _GEO_RATIO_THRESH:
                    return make_error(predicted)
            return []

        # 其他視角：幾何 ratio 決定
        ratio = geo_exp / best_geo_score if best_geo_score > 0 else 1.0
        if ratio < _GEO_RATIO_THRESH:
            return make_error(predicted)

    except Exception as e:
        print(f"[ViewCLF ERROR] view={view} {e}", flush=True)
    return []


# ── 對外主入口 ────────────────────────────────────────────────────────────

def validate_uploads(
    uploads: dict,
    *,
    mirror: bool = False,
    skip_angle_check: bool = False,
) -> ValidationResult:
    """
    驗證所有視角照片。

    uploads: {"front": UploadFile, "left_side": UploadFile, ...}
    mirror:  True = 後置鏡頭（app 尚未翻轉），驗證前先水平翻轉
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
            all_errors.extend(validate_view_angle(view, raw, mirror=mirror))

    return ValidationResult(ok=len(all_errors) == 0, errors=all_errors)
