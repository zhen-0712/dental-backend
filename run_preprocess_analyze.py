#!/usr/bin/env python3
"""
run_preprocess_analyze.py - 合併照片預處理與 SAT 分析，並行載入模型加速

流程：
  Phase 1 (並行):
    Thread A → preprocess_photos.main()  (CV2 影像處理，寫入 real_teeth_processed/)
    Thread B → _preload_models()          (從磁碟載入 SAM + YOLO，通常 10-20 秒)
  Phase 2 (序列):
    analyze_real_teeth.main(sam_model, yolo_models)  (模型已就緒，立即開始 inference)
"""

import sys
sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WEIGHT_DIR = Path("/home/Zhen/projects/SegmentAnyTooth/weight")


def _load_models():
    from segmentanytooth import get_model_path
    from sam import sam_load
    from ultralytics import YOLO
    from utils import suppress_stdout
    print("🔧 [並行] SAM + YOLO 模型載入中...")
    with suppress_stdout():
        sam = sam_load(get_model_path("sam", str(WEIGHT_DIR)))
        yolo_models = {
            v: YOLO(model=get_model_path(v, str(WEIGHT_DIR)))
            for v in ['front', 'right', 'upper', 'lower']
        }
    print("✅ [並行] 模型載入完成")
    return sam, yolo_models


def _run_preprocess():
    import preprocess_photos
    preprocess_photos.main()


# ── Phase 1：預處理照片 & 載入模型，同時並行 ──
print("="*70)
print("🚀 Phase 1：照片預處理 + 模型載入（並行）")
print("="*70)

with ThreadPoolExecutor(max_workers=2) as pool:
    f_preprocess = pool.submit(_run_preprocess)
    f_models     = pool.submit(_load_models)
    f_preprocess.result()          # 等待預處理完成（analyze 需要檔案在磁碟上）
    sam_model, yolo_models = f_models.result()

# ── Phase 2：使用預載模型執行 SAT 分析 ──
print("\n" + "="*70)
print("🚀 Phase 2：SAT 牙齒分析（使用預載模型）")
print("="*70)

import analyze_real_teeth
analyze_real_teeth.main(sam_model=sam_model, yolo_models=yolo_models)
