#!/usr/bin/env python3
"""
fluorescence_core.py

405nm 藍紫光螢光牙菌斑偵測的核心演算法（純函式，import 時無副作用）。

被兩個地方共用：
  - fluorescence_test/plaque_detect_v2.py   離線測試與調參（含標註命中率評估）
  - teeth_test_fluorescence.py              正式 pipeline 的偵測步驟

★ 這個模組不碰既有的染色劑流程（teeth_test.py）。兩條路徑完全獨立。

────────────────────────────────────────────────────────────────
方法：三端元光譜解混（spectral unmixing）
────────────────────────────────────────────────────────────────
每個像素的 BGR 視為三種光的線性疊加：

    observed = a·V + fe·E + fp·P

    V = 405nm 激發光的反射（幾乎不含綠色分量）
    E = 琺瑯質的綠色自體螢光
    P = 菌斑的紅色螢光（porphyrin，~635nm）

三通道三未知數 → 每像素直接解 3×3 線性系統，取 fp 當菌斑強度。
牙齒與牙齦交界看起來偏紅，但那份紅可由 V 完全解釋，fp 會趨近 0，
所以邊緣偽影不會被誤判成菌斑。

牙齒 ROI 用 G 通道自體螢光 + 通道比值閘門。實測只有牙齒的綠色會壓過藍色
（B/G 0.76~0.84），手術鋪單 1.36~1.72、舌頭 1.87~3.08、手指 1.85，
用比值就能把入鏡的雜物一刀切開。
"""

import cv2
import numpy as np

DEFAULT_CONFIG = {
    # ---- 工作解析度 ----
    "work_long_side": 1000,       # 處理時縮到長邊 N px（原圖 2880 慢且雜訊多）
    "pre_blur_sigma": 2.0,        # 解混前的輕微高斯，壓感測器雜訊

    # ---- 端元估計 ----
    "excite_sample_pct": 20,      # 取 G 最低的前 N% 像素估計激發光 V
    # 405nm 紫光的物理約束：藍必為最大分量、綠必須很小。自動估計違反時
    # 代表取樣到的不是紫色背景（有環境光時會取到紅棕色牙齦），退回預設值。
    "excite_fallback_bgr": [1.0, 0.033, 0.367],   # 暗場照片估出的乾淨值
    "excite_min_b": 0.95,
    "excite_max_g": 0.15,
    "enamel_sample_pct": 10,      # 取牙齒內 a* 最低（最綠）的前 N% 估計 E
    "plaque_endmember": [0.0, 0.0, 1.0],   # P：紅色螢光 (B,G,R)

    # ---- 牙齒 ROI ----
    "tooth_g_min_abs": 45,        # G 絕對下限，擋純暗背景
    "tooth_otsu_scale": 0.85,     # Otsu 門檻 × 此係數
    "tooth_bg_max": 1.00,         # B/G 上限：只有牙齒的綠色自體螢光會壓過藍色
    "tooth_rg_max": 1.60,         # R/G 上限：排除舌頭、嘴唇、手指等偏紅組織
    "tooth_min_area_ratio": 3e-4, # 小於全圖此比例的連通域丟掉
    "tooth_close_ratio": 0.012,   # 形態學閉運算核 = long_side × ratio
    "tooth_erode_ratio": 0.004,   # 往內縮一點（解混已處理邊緣，不需縮太多）

    # ---- 菌斑判定 ----
    "local_win_ratio": 0.061,     # 局部背景視窗 = long_side × ratio（≈一顆牙寬）
    "z_thresh": 2.5,              # 局部殘差 / MAD 的門檻（3.0 保守 / 2.5 平衡 / 2.0 靈敏）
    # fp 至少要是該影像牙齒 fp 中位數的幾倍。
    # 原設 1.5，但那會在環境光下失效：環境光把整張圖的 fp 抬高，下限跟著水漲
    # 船高，反而砍掉通過 z 門檻的真訊號（實測 test2 有 98.8% 被砍）。
    # 這道閘門原本用途是「防止在均勻乾淨的牙齒上亂觸發」，該職責現已由
    # image_gate_min 承擔（用 fp/fe 比值、曝光無關），因此放寬到 1.0。
    "fp_rel_min": 1.0,
    "violet_guard_max": 1.25,     # 原圖 B/G 超過此值視為紫光污染 → 排除
    "specular_v_min": 245,        # 高光排除：V ≥ 此值且 S ≤ specular_s_max
    "specular_s_max": 60,
    # ---- 影像層級閘門 ----
    # MAD 自適應門檻是純相對的，永遠會標出每張圖最紅的前幾 %，
    # 即使那張圖根本沒有菌斑。加一道絕對判準：整張影像牙齒區域的
    # fp/fe 第 90 百分位若低於門檻，直接判定「這張圖沒有菌斑」。
    # fp/fe 是曝光無關的比值（實測 29 張的 fe 變異 CV 僅 12%），
    # 因此這個絕對值可跨影像比較。設為 0 可停用。
    "image_gate_pct": 90,
    "image_gate_min": 0.12,

    "plaque_min_area_ratio": 6e-5,  # 菌斑最小面積（佔全圖比例）
    "plaque_open_ratio": 0.004,   # 菌斑 mask 開運算核
}


# ==================================================================
# 小工具
# ==================================================================
def ellipse(size_f):
    s = max(3, int(round(size_f)) | 1)            # 保證奇數且 ≥ 3
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def odd(size_f):
    return max(3, int(round(size_f)) | 1)


def scale_to_working(img, long_side):
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA), scale


# ==================================================================
# 端元估計與解混
# ==================================================================
def estimate_excitation(img_bgr, cfg):
    """
    405nm 激發光是反射光，本身幾乎不含綠色分量，
    所以取 G 最低的一群像素（背景黏膜、口腔深處）的平均色即為其方向。

    但這個假設在有診療室環境光時會失效——最暗的一群變成紅棕色牙齦而非
    紫色背景，估出來的 V 會變成紅色最大，物理上不可能。因此加上約束：
    藍必為最大分量、綠必須很小，違反就退回暗場校準值。

    回傳 (V, fallback_used)。
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    thr = np.percentile(g, cfg["excite_sample_pct"])
    sel = (g <= thr) & (b + g + r > 30)           # 排除全黑
    if sel.sum() < 100:
        sel = g <= np.percentile(g, 50)
    v = np.array([b[sel].mean(), g[sel].mean(), r[sel].mean()], dtype=np.float32)
    v /= (v.max() + 1e-6)

    if v[0] < cfg["excite_min_b"] or v[1] > cfg["excite_max_g"]:
        return np.array(cfg["excite_fallback_bgr"], dtype=np.float32), True
    return v, False


def estimate_enamel(img_bgr, tooth_mask, sample_pct):
    """牙齒 ROI 內 a* 最低（最綠、最乾淨）的像素平均色 = 琺瑯質自體螢光端元。"""
    a = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)[:, :, 1].astype(np.float32)
    inside = tooth_mask > 0
    if inside.sum() < 100:
        return np.array([0.85, 1.0, 1.0], dtype=np.float32)
    sel = inside & (a <= np.percentile(a[inside], sample_pct))
    e = img_bgr.astype(np.float32)[sel].mean(0)
    return (e / (e.max() + 1e-6)).astype(np.float32)


def unmix(img_f32, V, E, P):
    """
    解 M @ [a, fe, fp] = observed，M 的三個 column 分別是 V / E / P。
    回傳 (a, fe, fp)，其中 fp 就是紅色螢光豐度 = 菌斑強度。
    """
    M = np.stack([V, E, P], axis=1)
    if abs(np.linalg.det(M)) < 1e-6:
        raise ValueError("端元向量接近共線，無法解混")
    f = np.einsum('ij,hwj->hwi', np.linalg.inv(M), img_f32)
    return f[..., 0], f[..., 1], np.clip(f[..., 2], 0, None)


# ==================================================================
# 牙齒 ROI
# ==================================================================
def tooth_mask_from_green(img_bgr, cfg, long_side):
    """
    405nm 下琺瑯質有很強的綠色自體螢光，軟組織幾乎沒有。

    但只靠 G 亮度不夠：有診療室環境光時，手術鋪單、舌頭、嘴唇、手指的 G 也很高。
    再加上 B/G 與 R/G 兩道閘門才切得乾淨。
    """
    blur = cv2.GaussianBlur(img_bgr, (0, 0), max(1.0, long_side * 0.004))
    b, g_blur, r = cv2.split(blur.astype(np.float32))

    otsu_t, _ = cv2.threshold(g_blur.astype(np.uint8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(cfg["tooth_g_min_abs"], otsu_t * cfg["tooth_otsu_scale"])

    mask = ((g_blur >= thr) &
            (b / (g_blur + 1.0) < cfg["tooth_bg_max"]) &
            (r / (g_blur + 1.0) < cfg["tooth_rg_max"])).astype(np.uint8) * 255

    k_close = ellipse(long_side * cfg["tooth_close_ratio"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_close)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = mask.size * cfg["tooth_min_area_ratio"]
    keep = np.array([False] + [stats[i, cv2.CC_STAT_AREA] >= min_area
                               for i in range(1, n)])
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)

    mask = cv2.erode(mask, ellipse(long_side * cfg["tooth_erode_ratio"]))
    return mask, float(thr)


# ==================================================================
# 局部背景相減 + MAD 自適應門檻
# ==================================================================
def local_residual(fp, tooth_mask, win):
    """
    只用牙齒內的像素做局部平均（normalized box filter）再相減。
    消除左右照明梯度，也避免牙齒外的黑色背景把平均拉低。
    """
    m = (tooth_mask > 0).astype(np.float32)
    num = cv2.boxFilter(fp * m, -1, (win, win), normalize=False)
    den = cv2.boxFilter(m, -1, (win, win), normalize=False)
    return fp - num / (den + 1e-3)


def adaptive_plaque_mask(fp, residual, tooth_mask, orig_bgr, cfg, long_side):
    inside = tooth_mask > 0
    rv = residual[inside]
    med_r = float(np.median(rv))
    mad = max(float(np.median(np.abs(rv - med_r))) * 1.4826, 0.5)
    z = (residual - med_r) / mad

    fp_med = float(np.median(fp[inside]))
    fp_floor = fp_med * cfg["fp_rel_min"]

    raw = (inside & (z >= cfg["z_thresh"]) & (fp >= fp_floor)).astype(np.uint8) * 255

    # 排除項：紫光污染 + 鏡面高光
    b, g, _ = cv2.split(orig_bgr.astype(np.float32))
    violet = (b / (g + 1.0)) > cfg["violet_guard_max"]
    hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    specular = ((hsv[:, :, 2] >= cfg["specular_v_min"]) &
                (hsv[:, :, 1] <= cfg["specular_s_max"]))
    raw[violet | specular] = 0

    # 形態學 + 最小面積
    k = ellipse(long_side * cfg["plaque_open_ratio"])
    clean = cv2.morphologyEx(raw, cv2.MORPH_OPEN, k)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(clean, 8)
    min_area = clean.size * cfg["plaque_min_area_ratio"]
    keep = np.array([False] + [stats[i, cv2.CC_STAT_AREA] >= min_area
                               for i in range(1, n)])
    final = np.where(keep[labels], 255, 0).astype(np.uint8)

    regions = [{
        "id": int(i),
        "area_px": int(stats[i, cv2.CC_STAT_AREA]),
        "centroid": [round(float(cents[i][0]), 1), round(float(cents[i][1]), 1)],
        "bbox": [int(stats[i, cv2.CC_STAT_LEFT]), int(stats[i, cv2.CC_STAT_TOP]),
                 int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT])],
        "fp_mean": round(float(fp[labels == i].mean()), 2),
        "z_max": round(float(z[labels == i].max()), 2),
    } for i in range(1, n) if keep[i]]
    regions.sort(key=lambda r: -r["area_px"])

    stats_out = {"fp_median_in_tooth": round(fp_med, 2),
                 "fp_floor": round(fp_floor, 2),
                 "residual_mad": round(mad, 3)}
    return final, z, regions, stats_out


# ==================================================================
# 一站式入口
# ==================================================================
def detect(img_bgr, cfg=None):
    """
    輸入原始 BGR 影像（任意尺寸），回傳偵測結果。

    回傳 dict：
        plaque_mask  二值 mask（工作解析度）
        tooth_mask   牙齒 ROI（工作解析度）
        fp / z       中間圖層，debug 用
        work_size    (w, h)
        info         端元、統計、區塊清單
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    work, scale = scale_to_working(img_bgr, cfg["work_long_side"])
    h, w = work.shape[:2]
    long_side = max(h, w)

    tooth, g_thr = tooth_mask_from_green(work, cfg, long_side)
    if int((tooth > 0).sum()) < 500:
        empty = np.zeros((h, w), np.uint8)
        return {"plaque_mask": empty, "tooth_mask": tooth,
                "fp": np.zeros((h, w), np.float32), "z": np.zeros((h, w), np.float32),
                "work_size": (w, h), "scale": scale,
                "info": {"error": "牙齒區域太小，無法偵測", "tooth_px": int((tooth > 0).sum())}}

    V, v_fallback = estimate_excitation(work, cfg)
    E = estimate_enamel(work, tooth, cfg["enamel_sample_pct"])
    P = np.array(cfg["plaque_endmember"], dtype=np.float32)

    smooth = cv2.GaussianBlur(work, (0, 0), cfg["pre_blur_sigma"]).astype(np.float32)
    _, fe, fp = unmix(smooth, V, E, P)

    # ---- 影像層級閘門 ----
    inside = tooth > 0
    gate_val = float(np.percentile((fp / np.maximum(fe, 1.0))[inside],
                                   cfg["image_gate_pct"]))
    if cfg["image_gate_min"] > 0 and gate_val < cfg["image_gate_min"]:
        empty = np.zeros((h, w), np.uint8)
        return {
            "plaque_mask": empty, "tooth_mask": tooth,
            "fp": fp, "z": np.zeros((h, w), np.float32),
            "work_size": (w, h), "scale": scale,
            "info": {
                "excitation_endmember_bgr": [round(float(x), 4) for x in V],
                "excitation_fallback_used": bool(v_fallback),
                "enamel_endmember_bgr": [round(float(x), 4) for x in E],
                "tooth_threshold_g": round(g_thr, 1),
                "tooth_px": int(inside.sum()),
                "plaque_px": 0, "plaque_ratio_of_tooth": 0.0,
                "image_gate_value": round(gate_val, 4),
                "image_gate_min": cfg["image_gate_min"],
                "image_gate_triggered": True,
                "regions": [],
            },
        }

    win = odd(long_side * cfg["local_win_ratio"])
    resid = local_residual(fp, tooth, win)
    plaque, z, regions, st = adaptive_plaque_mask(fp, resid, tooth, work, cfg, long_side)

    tooth_px = int((tooth > 0).sum())
    return {
        "plaque_mask": plaque, "tooth_mask": tooth, "fp": fp, "z": z,
        "work_size": (w, h), "scale": scale,
        "info": {
            "excitation_endmember_bgr": [round(float(x), 4) for x in V],
            "excitation_fallback_used": bool(v_fallback),
            "enamel_endmember_bgr": [round(float(x), 4) for x in E],
            "tooth_threshold_g": round(g_thr, 1),
            "tooth_px": tooth_px,
            "plaque_px": int((plaque > 0).sum()),
            "plaque_ratio_of_tooth": round(int((plaque > 0).sum()) / tooth_px, 4),
            "local_window_px": win,
            "image_gate_value": round(gate_val, 4),
            "image_gate_min": cfg["image_gate_min"],
            "image_gate_triggered": False,
            **st,
            "regions": regions,
        },
    }
