#!/usr/bin/env python3
"""
restore_real_plaque.py  ── 還原真實菌斑分析結果

從 plaque_output/backup_real/ 還原備份，覆蓋假模型輸出。
"""
import shutil, json
from pathlib import Path

import sys; sys.path.insert(0, "/home/Zhen/projects/SegmentAnyTooth")
from user_env import get_paths, setup_user_dirs
_PATHS = get_paths()
setup_user_dirs(_PATHS["user_dir"])
OUTPUT_DIR = _PATHS["plaque_output"]

backup_dir = OUTPUT_DIR / "backup_real"
FILES = ["plaque_by_fdi.ply", "plaque_by_fdi.glb",
         "plaque_by_fdi.obj", "plaque_by_fdi_stats.json"]

print("=" * 50)
print("♻️  還原真實菌斑分析結果")
print("=" * 50)

# 檢查目前是否為假資料
stats_path = OUTPUT_DIR / "plaque_by_fdi_stats.json"
if stats_path.exists():
    data = json.loads(stats_path.read_text())
    if data.get("_mock"):
        print("  目前為假資料模式 → 還原中...")
    else:
        print("  目前已是真實資料，仍繼續還原...")

if not backup_dir.exists():
    print(f"\n❌ 找不到備份目錄: {backup_dir}")
    print("   請先執行 generate_mock_plaque.py 才會產生備份")
    raise SystemExit(1)

restored = []
missing  = []
for fname in FILES:
    src = backup_dir / fname
    dst = OUTPUT_DIR / fname
    if src.exists():
        shutil.copy2(src, dst)
        restored.append(fname)
        print(f"  ✅ 還原: {fname}")
    else:
        missing.append(fname)
        print(f"  ⚠️  備份中找不到: {fname}")

print(f"\n{'='*50}")
if missing:
    print(f"⚠️  部分檔案未還原: {missing}")
else:
    print(f"✅ 全部 {len(restored)} 個檔案還原完畢")
    print(f"   真實分析結果已恢復")
