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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fluorescence_core as fc

# 演算法全部委派給 core，本檔只負責 CLI、標註評估與繪圖
load_and_scale        = lambda p, n: fc.scale_to_working(cv2.imread(str(p)), n)
estimate_excitation   = fc.estimate_excitation
estimate_enamel       = fc.estimate_enamel
unmix                 = fc.unmix
tooth_mask_from_green = fc.tooth_mask_from_green
local_residual        = fc.local_residual
adaptive_plaque_mask  = fc.adaptive_plaque_mask
_ellipse              = fc.ellipse
_odd                  = fc.odd

# ==================================================================
# 路徑（唯讀既有資料夾，只寫入自己的 output/）
# ==================================================================
HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE.parent / "real_color_translet"
OUTPUT_DIR = HERE / "output"

# ==================================================================
# 可調參數 —— 所有魔術數字集中在這裡
# ==================================================================
# 演算法參數一律以 fluorescence_core.DEFAULT_CONFIG 為準，
# 這裡只覆寫離線測試專屬的項目，避免兩邊參數漂移。
CONFIG = {
    **fc.DEFAULT_CONFIG,
    "ref_tolerance_ratio": 0.05,  # 標註點命中容差半徑 = 長邊 × 此比例
    "save_debug_panel": True,
}

# 標註 ground truth：正規化座標 [x, y]，相對原圖尺寸。
# 來源 annotated/reference_points.json（extract_annotations.py 產出，像素級）
# 或 annotated/manual_points.json（目測抄錄，誤差較大）。前者優先。
def load_reference_points():
    """
    合併兩個來源，reference_points.json（程式抽取，像素級）優先覆蓋
    manual_points.json（目測抄錄，誤差較大）。這樣舊批次的標註不會因為
    新批次改用程式抽取而消失。
    """
    out, src = {}, []
    for fn in ("manual_points.json", "reference_points.json"):
        p = HERE / "annotated" / fn
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        cnt = 0
        for k, v in raw.items():
            if k.startswith("_"):
                continue
            marks = v.get("marks", [])
            if marks and isinstance(marks[0], dict):
                out[k] = [m["bbox_center_norm"] for m in marks]
            else:
                out[k] = [list(m) for m in marks]
            cnt += 1
        src.append(f"{fn}({cnt} 張)")
    if src:
        print(f"📌 標註來源: {' + '.join(src)} → 共 {len(out)} 張、"
              f"{sum(len(v) for v in out.values())} 個點")
    else:
        print("⚠️  annotated/ 沒有標註檔，跳過命中率評估")
    return out


REFERENCE_POINTS = load_reference_points()


# ==================================================================
# Stage 0
# ==================================================================
# ==================================================================
# Stage 1: 端元估計
# ==================================================================
# ==================================================================
# Stage 2: 牙齒 ROI（綠色自體螢光）
# ==================================================================
# ==================================================================
# Stage 3: 局部背景相減 + MAD 自適應門檻
# ==================================================================
# ==================================================================
# Stage 4: 輸出
# ==================================================================
def render_overlay(orig, tooth_mask, plaque_mask):
    vis = orig.copy()
    vis[tooth_mask > 0] = (0.8 * vis[tooth_mask > 0]
                           + 0.2 * np.array([0, 255, 0])).astype(np.uint8)
    vis[plaque_mask > 0] = (0.25 * vis[plaque_mask > 0]
                            + 0.75 * np.array([0, 0, 255])).astype(np.uint8)
    cnts, _ = cv2.findContours(plaque_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, (255, 255, 255), 1)
    return vis


def render_panel(orig, fp, z, tooth_mask, overlay):
    def masked_heat(arr, lo, hi, cmap=cv2.COLORMAP_JET):
        n = np.clip((arr - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
        h = cv2.applyColorMap(n, cmap)
        out = np.zeros_like(orig)
        out[tooth_mask > 0] = h[tooth_mask > 0]
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
    V, _ = estimate_excitation(orig, cfg)
    E = estimate_enamel(orig, tooth, cfg["enamel_sample_pct"])
    P = np.array(cfg["plaque_endmember"], dtype=np.float32)
    print(f"  端元 (B,G,R)  激發光 V={V.round(3)}  琺瑯質 E={E.round(3)}")

    _, fe, fp = unmix(smooth, V, E, P)

    # 影像層級閘門：整張圖的 fp/fe 太低就判定「這張沒有菌斑」
    gate_val = float(np.percentile((fp / np.maximum(fe, 1.0))[tooth > 0],
                                   cfg["image_gate_pct"]))
    gated = cfg["image_gate_min"] > 0 and gate_val < cfg["image_gate_min"]

    # Stage 3
    win = _odd(long_side * cfg["local_win_ratio"])
    resid = local_residual(fp, tooth, win)
    plaque, z, regions, st = adaptive_plaque_mask(fp, resid, tooth, orig, cfg, long_side)
    if gated:
        plaque = np.zeros_like(plaque)
        regions = []
    n_px = int((plaque > 0).sum())
    print(f"  局部視窗={win}px  fp 中位數={st['fp_median_in_tooth']} "
          f"殘差 MAD={st['residual_mad']}")
    print(f"  影像閘門 p{cfg['image_gate_pct']}(fp/fe)={gate_val:.3f} "
          f"(門檻 {cfg['image_gate_min']})"
          + ("  🚫 判定為無菌斑，輸出空結果" if gated else "  ✅ 通過"))
    print(f"  菌斑: {n_px:,} px，佔牙齒面積 {n_px / tooth_px * 100:.1f}%，"
          f"{len(regions)} 個區塊")

    # 驗證探針
    ref = REFERENCE_POINTS.get(stem, [])
    ref_s = [(round(nx * w), round(ny * h)) for nx, ny in ref]
    tol = int(round(long_side * cfg["ref_tolerance_ratio"]))
    hits, zs = [], []
    for (x, y) in ref_s:
        win_m = plaque[max(0, y - tol):y + tol + 1, max(0, x - tol):x + tol + 1]
        hits.append(bool((win_m > 0).any()))
        zs.append(round(float(z[max(0, y - 4):y + 5, max(0, x - 4):x + 5].mean()), 2))

    # 區塊層級 precision：每個偵測區塊的質心是否落在任一標註的容差內。
    # 像素級 IoU 需要全圖標註成本太高，區塊層級也更貼近實際用途
    #（醫師在意的是「有沒有指對牙」而不是輪廓精確度）。
    matched = 0
    for r in regions:
        cx, cy = r["centroid"]
        if any((cx - x) ** 2 + (cy - y) ** 2 <= tol ** 2 for x, y in ref_s):
            matched += 1
    n_reg = len(regions)
    precision = round(matched / n_reg, 3) if n_reg else None
    if ref:
        print(f"  🎯 recall {sum(hits)}/{len(hits)}（容差 {tol}px）"
              f" {''.join('✅' if s else '❌' for s in hits)}"
              f"   precision {matched}/{n_reg}"
              + (f" = {precision:.0%}" if precision is not None else ""))

    # Stage 4
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overlay = render_overlay(orig, tooth, plaque)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_overlay.png"), overlay)
    cv2.imwrite(str(OUTPUT_DIR / f"{stem}_mask.png"), plaque)
    if cfg["save_debug_panel"]:
        cv2.imwrite(str(OUTPUT_DIR / f"{stem}_panel.png"),
                    render_panel(orig, fp, z, tooth, overlay))

    return {
        "image": path.name,
        "work_size": [w, h],
        "excitation_endmember_bgr": [round(float(x), 4) for x in V],
        "enamel_endmember_bgr": [round(float(x), 4) for x in E],
        "tooth_threshold_g": round(g_thr, 1),
        "tooth_px": tooth_px,
        "plaque_px": n_px,
        "plaque_ratio_of_tooth": round(n_px / tooth_px, 4),
        "image_gate_value": round(gate_val, 4),
        "image_gate_triggered": bool(gated),
        "reference_point_hits": hits,
        "reference_point_z": zs,
        "regions_matched": matched,
        "region_precision": precision,
        **st,
        "regions": regions,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*", help="影像路徑（預設跑 real_color_translet/ 全部）")
    ap.add_argument("--z", type=float, help="z-score 門檻（越低越靈敏）")
    ap.add_argument("--gate", type=float,
                    help="影像層級閘門 p90(fp/fe) 下限，設 0 停用")
    ap.add_argument("--fp-rel", type=float, help="fp 相對下限倍數")
    ap.add_argument("--win", type=float, help="局部視窗比例，如 0.061")
    ap.add_argument("--no-panel", action="store_true", help="不輸出 debug 面板")
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.z is not None:
        cfg["z_thresh"] = args.z
    if args.gate is not None:
        cfg["image_gate_min"] = args.gate
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
