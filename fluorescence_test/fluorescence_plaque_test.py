#!/usr/bin/env python3
"""
fluorescence_plaque_test.py

405nm 藍紫光螢光照片的牙菌斑偵測 —— 獨立測試腳本。

★ 這支程式不 import、不修改、不寫入現有 pipeline 的任何路徑。
  輸入：real_color_translet/ 的原圖   輸出：fluorescence_test/output/

────────────────────────────────────────────────────────────────
為什麼不能沿用 teeth_test.py 的 HSV 門檻
────────────────────────────────────────────────────────────────
藍紫光下整張影像 R≈B 高、G 低，菌斑的 HSV 色相在 H=6~54 之間亂跳、
飽和度只有 47~73，落在現行 H∈[145,172]∪[0,15] 門檻之外。實測把現行
規則跑在這批照片上，標註的粉紅螢光一個都沒抓到，反而在嘴唇、燈珠
上誤觸發。

────────────────────────────────────────────────────────────────
核心方法：三端元光譜解混（spectral unmixing）
────────────────────────────────────────────────────────────────
每個像素的 BGR 視為三種光源的線性疊加：

    observed = a·V + fe·E + fp·P

    V = 405nm 激發光的反射（幾乎不含綠色分量）
    E = 琺瑯質的綠色自體螢光
    P = 菌斑的紅色螢光（porphyrin，~635nm）

三通道、三未知數 → 每個像素直接解 3×3 線性系統，取 fp 當菌斑強度。

這樣做的關鍵好處：牙齒與紫色牙齦的交界處雖然「看起來偏紅」，但那份
紅完全可以由 V 解釋，fp 會趨近 0。先前用 Lab a* 局部對比時，最強的
訊號其實是這圈邊緣偽影而不是菌斑；換成解混後偽影消失。

最後再對 fp 做局部背景相減 + MAD 自適應門檻，消除左右照明梯度，
並讓門檻不綁死絕對數值（換相機／距離／曝光都不用重調）。

用法：
  python fluorescence_plaque_test.py                  # 跑預設兩張
  python fluorescence_plaque_test.py a.jpg b.jpg      # 指定圖片
  python fluorescence_plaque_test.py --z 2.0          # 放寬靈敏度
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# ==================================================================
# 路徑（唯讀既有資料夾，只寫入自己的 output/）
# ==================================================================
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE.parent / "real_color_translet"
OUTPUT_DIR = HERE / "output"

# ==================================================================
# 可調參數 —— 所有魔術數字集中在這裡
# ==================================================================
CONFIG = {
    # ---- Stage 0: 工作解析度 ----
    "work_long_side": 1000,       # 處理時縮到長邊 N px（原圖 2880 慢且雜訊多）
    "pre_blur_sigma": 2.0,        # 解混前的輕微高斯，壓感測器雜訊

    # ---- Stage 1: 端元估計 ----
    "excite_sample_pct": 20,      # 取 G 最低的前 N% 像素估計激發光 V
    "enamel_sample_pct": 10,      # 取牙齒內 a* 最低（最綠）的前 N% 估計 E
    "plaque_endmember": [0.0, 0.0, 1.0],   # P：紅色螢光 (B,G,R)

    # ---- Stage 2: 牙齒 ROI（綠色自體螢光）----
    "tooth_g_min_abs": 45,        # G 絕對下限，擋純暗背景
    "tooth_otsu_scale": 0.85,     # Otsu 門檻 × 此係數
    "tooth_min_area_ratio": 3e-4, # 小於全圖此比例的連通域丟掉
    "tooth_close_ratio": 0.012,   # 形態學閉運算核 = long_side × ratio
    "tooth_erode_ratio": 0.004,   # 往內縮一點（解混已處理邊緣，不需縮太多）

    # ---- Stage 3: 菌斑判定 ----
    "local_win_ratio": 0.061,     # 局部背景視窗 = long_side × ratio（≈一顆牙寬）
    "z_thresh": 2.0,              # 局部殘差 / MAD 的門檻（2.5 保守 / 2.0 平衡 / 1.6 靈敏）
    "fp_rel_min": 1.5,            # fp 至少要是該影像牙齒 fp 中位數的幾倍
    "violet_guard_max": 1.25,     # 原圖 B/G 超過此值視為紫光污染 → 排除
    "specular_v_min": 245,        # 高光排除：V ≥ 此值且 S ≤ specular_s_max
    "specular_s_max": 60,
    "plaque_min_area_ratio": 6e-5,  # 菌斑最小面積（佔全圖比例）
    "plaque_open_ratio": 0.004,   # 菌斑 mask 開運算核

    # ---- Stage 4: 輸出 ----
    "save_debug_panel": True,
}

# 我從你紅筆圈選位置反推出的粉紅螢光座標（工作解析度 750×1000 下），
# 只用來驗證有沒有抓到，完全不參與演算。
REFERENCE_POINTS = {
    "IMG_4955": [(460, 833), (610, 830)],
    "IMG_4956": [(438, 790), (590, 775), (645, 710)],
}


# ==================================================================
# Stage 0
# ==================================================================
def load_and_scale(path: Path, long_side: int):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"讀不到影像: {path}")
    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    return cv2.resize(img, (round(w * scale), round(h * scale)),
                      interpolation=cv2.INTER_AREA), scale


def _ellipse(size_f):
    s = max(3, int(round(size_f)) | 1)            # 保證奇數且 ≥ 3
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))


def _odd(size_f):
    return max(3, int(round(size_f)) | 1)


# ==================================================================
# Stage 1: 端元估計
# ==================================================================
def estimate_excitation(img_bgr, sample_pct):
    """
    405nm 激發光是反射光，本身幾乎不含綠色分量，
    所以取 G 最低的一群像素（背景黏膜、口腔深處）的平均色即為其方向。
    """
    b, g, r = cv2.split(img_bgr.astype(np.float32))
    thr = np.percentile(g, sample_pct)
    sel = (g <= thr) & (b + g + r > 30)           # 排除全黑
    if sel.sum() < 100:
        sel = g <= np.percentile(g, 50)
    v = np.array([b[sel].mean(), g[sel].mean(), r[sel].mean()], dtype=np.float32)
    return v / (v.max() + 1e-6)


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
# Stage 2: 牙齒 ROI（綠色自體螢光）
# ==================================================================
def tooth_mask_from_green(img_bgr, cfg, long_side):
    """
    405nm 下琺瑯質有很強的綠色自體螢光，軟組織幾乎沒有。
    G 通道單獨就能把整排牙齒切出來，在這種暗場照片上比 YOLO/SAM 更穩。
    """
    g = img_bgr[:, :, 1]
    g_blur = cv2.GaussianBlur(g, (0, 0), max(1.0, long_side * 0.004))

    otsu_t, _ = cv2.threshold(g_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(cfg["tooth_g_min_abs"], otsu_t * cfg["tooth_otsu_scale"])
    mask = (g_blur >= thr).astype(np.uint8) * 255

    k_close = _ellipse(long_side * cfg["tooth_close_ratio"])
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_close)

    # 丟掉太小的連通域（反光燈珠、雜點）
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = mask.size * cfg["tooth_min_area_ratio"]
    keep = np.array([False] + [stats[i, cv2.CC_STAT_AREA] >= min_area
                               for i in range(1, n)])
    mask = np.where(keep[labels], 255, 0).astype(np.uint8)

    mask = cv2.erode(mask, _ellipse(long_side * cfg["tooth_erode_ratio"]))
    return mask, float(thr)


# ==================================================================
# Stage 3: 局部背景相減 + MAD 自適應門檻
# ==================================================================
def local_residual(fp, tooth_mask, win):
    """
    只用牙齒內的像素做局部平均（normalized box filter），
    再相減。消除左右照明梯度，也避免牙齒外的黑色背景把平均拉低。
    """
    m = (tooth_mask > 0).astype(np.float32)
    num = cv2.boxFilter(fp * m, -1, (win, win), normalize=False)
    den = cv2.boxFilter(m, -1, (win, win), normalize=False)
    return fp - num / (den + 1e-3)


def adaptive_plaque_mask(fp, residual, tooth_mask, orig_bgr, cfg, long_side):
    inside = tooth_mask > 0
    rv = residual[inside]
    med_r = float(np.median(rv))
    mad = float(np.median(np.abs(rv - med_r))) * 1.4826
    mad = max(mad, 0.5)
    z = (residual - med_r) / mad

    fp_med = float(np.median(fp[inside]))
    fp_floor = fp_med * cfg["fp_rel_min"]

    raw = (inside & (z >= cfg["z_thresh"]) & (fp >= fp_floor)).astype(np.uint8) * 255

    # --- 排除項：紫光污染 + 鏡面高光 ---
    b, g, _ = cv2.split(orig_bgr.astype(np.float32))
    violet = (b / (g + 1.0)) > cfg["violet_guard_max"]
    hsv = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2HSV)
    specular = ((hsv[:, :, 2] >= cfg["specular_v_min"]) &
                (hsv[:, :, 1] <= cfg["specular_s_max"]))
    raw[violet | specular] = 0

    # --- 形態學 + 最小面積 ---
    k = _ellipse(long_side * cfg["plaque_open_ratio"])
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
# Stage 4: 輸出
# ==================================================================
def render_overlay(orig, tooth_mask, plaque_mask, ref_pts=None):
    vis = orig.copy()
    vis[tooth_mask > 0] = (0.8 * vis[tooth_mask > 0]
                           + 0.2 * np.array([0, 255, 0])).astype(np.uint8)
    vis[plaque_mask > 0] = (0.25 * vis[plaque_mask > 0]
                            + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(plaque_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 255, 255), 1)
    for (x, y) in (ref_pts or []):
        cv2.circle(vis, (x, y), 15, (0, 255, 255), 2)
    return vis


def render_panel(orig, fp, z, tooth_mask, overlay, ref_pts):
    def masked_heat(arr, lo, hi, cmap=cv2.COLORMAP_JET):
        n = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        h = cv2.applyColorMap(n, cmap)
        out = np.zeros_like(orig)
        out[tooth_mask > 0] = h[tooth_mask > 0]
        for (x, y) in ref_pts:
            cv2.circle(out, (x, y), 15, (255, 255, 255), 2)
        return out

    tiles = [
        ("1. original", orig),
        ("2. tooth ROI (green autofluor.)", cv2.cvtColor(tooth_mask, cv2.COLOR_GRAY2BGR)),
        ("3. fp = red fluorescence", masked_heat(fp, 0, 45)),
        ("4. z-score (local)", masked_heat(z, 0, 5)),
        ("5. result", overlay),
    ]
    h, w = orig.shape[:2]
    th, tw = h // 2, w // 2
    cells = []
    for name, im in tiles:
        c = cv2.resize(im, (tw, th))
        cv2.putText(c, name, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cells.append(c)
    while len(cells) % 3:
        cells.append(np.zeros((th, tw, 3), np.uint8))
    return np.vstack([np.hstack(cells[i:i + 3]) for i in range(0, len(cells), 3)])


# ==================================================================
# 主流程
# ==================================================================
def process(path: Path, cfg: dict):
    stem = path.stem
    print(f"\n{'=' * 64}\n📷 {path.name}\n{'=' * 64}")

    orig, scale = load_and_scale(path, cfg["work_long_side"])
    h, w = orig.shape[:2]
    long_side = max(h, w)
    print(f"  工作解析度: {w}×{h}（原圖縮放 {scale:.3f}×）")

    smooth = cv2.GaussianBlur(orig, (0, 0), cfg["pre_blur_sigma"]).astype(np.float32)

    # Stage 2 先跑，估計 E 需要牙齒 ROI
    tooth, g_thr = tooth_mask_from_green(orig, cfg, long_side)
    tooth_px = int((tooth > 0).sum())
    print(f"  牙齒 ROI: G 門檻={g_thr:.0f}，{tooth_px:,} px "
          f"({tooth_px / tooth.size * 100:.1f}%)")
    if tooth_px < 500:
        print("  ⚠️  牙齒區域太小，跳過")
        return None

    # Stage 1
    V = estimate_excitation(orig, cfg["excite_sample_pct"])
    E = estimate_enamel(orig, tooth, cfg["enamel_sample_pct"])
    P = np.array(cfg["plaque_endmember"], dtype=np.float32)
    print(f"  端元 (B,G,R)  激發光 V={V.round(3)}  琺瑯質 E={E.round(3)}")

    _, _, fp = unmix(smooth, V, E, P)

    # Stage 3
    win = _odd(long_side * cfg["local_win_ratio"])
    resid = local_residual(fp, tooth, win)
    plaque, z, regions, st = adaptive_plaque_mask(fp, resid, tooth, orig, cfg, long_side)
    n_px = int((plaque > 0).sum())
    print(f"  局部視窗={win}px  fp 中位數={st['fp_median_in_tooth']} "
          f"殘差 MAD={st['residual_mad']}")
    print(f"  菌斑: {n_px:,} px，佔牙齒面積 {n_px / tooth_px * 100:.1f}%，"
          f"{len(regions)} 個區塊")

    # 驗證探針
    ref = REFERENCE_POINTS.get(stem, [])
    ref_s = [(round(x * w / 750), round(y * h / 1000)) for x, y in ref]
    hits, zs = [], []
    for (x, y) in ref_s:
        win_m = plaque[max(0, y - 12):y + 13, max(0, x - 12):x + 13]
        hits.append(bool((win_m > 0).any()))
        zs.append(round(float(z[max(0, y - 3):y + 4, max(0, x - 3):x + 4].mean()), 2))
    if ref:
        print(f"  🎯 標註點命中 {sum(hits)}/{len(hits)}  "
              f"{['✅' if s else '❌' for s in hits]}  z={zs}")

    # Stage 4
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay = render_overlay(orig, tooth, plaque, ref_s)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_overlay.png"), overlay)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_mask.png"), plaque)
    if cfg["save_debug_panel"]:
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}_panel.png"),
                    render_panel(orig, fp, z, tooth, overlay, ref_s))

    return {
        "image": path.name,
        "work_size": [w, h],
        "excitation_endmember_bgr": [round(float(x), 4) for x in V],
        "enamel_endmember_bgr": [round(float(x), 4) for x in E],
        "tooth_threshold_g": round(g_thr, 1),
        "tooth_px": tooth_px,
        "plaque_px": n_px,
        "plaque_ratio_of_tooth": round(n_px / tooth_px, 4),
        "reference_point_hits": hits,
        "reference_point_z": zs,
        **st,
        "regions": regions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="影像路徑（預設跑 real_color_translet/ 全部）")
    ap.add_argument("--z", type=float, help="z-score 門檻（越低越靈敏）")
    ap.add_argument("--fp-rel", type=float, help="fp 相對下限倍數")
    ap.add_argument("--win", type=float, help="局部視窗比例，如 0.061")
    ap.add_argument("--no-panel", action="store_true", help="不輸出 debug 面板")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.z is not None:
        cfg["z_thresh"] = args.z
    if args.fp_rel is not None:
        cfg["fp_rel_min"] = args.fp_rel
    if args.win is not None:
        cfg["local_win_ratio"] = args.win
    if args.no_panel:
        cfg["save_debug_panel"] = False

    if args.images:
        paths = [Path(p) for p in args.images]
    else:
        paths = sorted(DEFAULT_INPUT_DIR.glob("*.jpg")) + \
                sorted(DEFAULT_INPUT_DIR.glob("*.png"))
    if not paths:
        print(f"❌ 找不到影像，請確認 {DEFAULT_INPUT_DIR}")
        return 1

    print(f"⚙️  z_thresh={cfg['z_thresh']}  fp_rel_min={cfg['fp_rel_min']}  "
          f"local_win_ratio={cfg['local_win_ratio']}")

    results = [r for r in (process(p, cfg) for p in paths) if r]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "results.json", "w", encoding="utf-8") as f:
        json.dump({"config": cfg, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完成，輸出於 {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
