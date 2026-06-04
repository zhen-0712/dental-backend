#!/usr/bin/env python3
"""
用 teaching 模型重新跑菌斑分析，結果存為新的 DB 紀錄。
"""
import os, sys, shutil, json, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

BASE     = Path("/home/Zhen/projects/SegmentAnyTooth")
DATA2    = BASE / "data" / "2"
PYTHON   = "/home/Zhen/anaconda3/envs/triposr/bin/python"
TZ_TPE   = timezone(timedelta(hours=8))
USER_ID  = 2

def now_taipei():
    return datetime.now(TZ_TPE).replace(tzinfo=None)

# ── 1. 建立臨時 user 目錄 ──────────────────────────────────────────────────────
tmp_dir = BASE / "data" / "_tmp_teaching_run"
if tmp_dir.exists():
    shutil.rmtree(tmp_dir)
tmp_dir.mkdir()

# 建立需要的子目錄
(tmp_dir / "plaque_output").mkdir()
(tmp_dir / "personalized_3d_models_real").mkdir()
(tmp_dir / "personalized_3d_models_real_teaching").mkdir()

# symlink 唯讀目錄（photos, processed, analysis, color）
for name in ["real_teeth", "real_teeth_processed", "real_teeth_analysis", "teeth_color_test"]:
    src = DATA2 / name
    dst = tmp_dir / name
    if src.exists():
        dst.symlink_to(src)

# 複製 teaching 模型檔案進 personalized_3d_models_real
teaching_src = DATA2 / "personalized_3d_models_real_teaching"
model_dst    = tmp_dir / "personalized_3d_models_real"

for fname in ["custom_upper_only.obj", "custom_lower_only.obj",
              "upper_seg_labels.npy", "lower_seg_labels.npy"]:
    shutil.copy2(teaching_src / fname, model_dst / fname)

# teaching 只有 custom_real_teeth_teaching.obj，改名放進去
shutil.copy2(teaching_src / "custom_real_teeth_teaching.obj",
             model_dst / "custom_real_teeth.obj")

# teaching symlink (給需要讀 _teaching 的路徑備用)
(tmp_dir / "personalized_3d_models_real_teaching").rmdir()
(tmp_dir / "personalized_3d_models_real_teaching").symlink_to(teaching_src)

print(f"✅ 臨時目錄建立完成：{tmp_dir}")

# ── 2. 在 DB 建立新的 plaque analysis 紀錄 ────────────────────────────────────
sys.path.insert(0, str(BASE))
from database import SessionLocal, Analysis, AnalysisType, AnalysisStatus

db = SessionLocal()
analysis = Analysis(
    user_id=USER_ID,
    type=AnalysisType.plaque,
    status=AnalysisStatus.running,
    created_at=now_taipei(),
)
db.add(analysis)
db.commit()
db.refresh(analysis)
analysis_id = analysis.id
print(f"✅ 新 analysis 記錄 ID：{analysis_id}")

# ── 3. 跑 pipeline ─────────────────────────────────────────────────────────────
env = os.environ.copy()
env["DENTAL_USER_DIR"]   = str(tmp_dir)
env["DENTAL_MODEL_TYPE"] = "teaching"
env["PYTHONWARNINGS"]    = "ignore"

def run(script):
    print(f"▶ 執行 {script} ...")
    result = subprocess.run(
        [PYTHON, str(BASE / script)],
        capture_output=True, text=True,
        cwd=str(BASE), timeout=600, env=env
    )
    if result.returncode != 0:
        print(f"❌ {script} 失敗：\n{result.stderr[-2000:]}")
        return False, result.stderr
    print(f"✅ {script} 完成")
    return True, ""

ok, err = run("teeth_test.py")
if not ok:
    analysis.status = AnalysisStatus.failed
    analysis.error_msg = f"teeth_test failed: {err}"
    db.commit(); db.close()
    shutil.rmtree(tmp_dir)
    sys.exit(1)

ok, err = run("extract_plaque_regions.py")
if not ok:
    analysis.status = AnalysisStatus.failed
    analysis.error_msg = f"extract_plaque failed: {err}"
    db.commit(); db.close()
    shutil.rmtree(tmp_dir)
    sys.exit(1)

ok, err = run("project_plaque_by_fdi.py")
if not ok:
    analysis.status = AnalysisStatus.failed
    analysis.error_msg = f"project_plaque failed: {err}"
    db.commit(); db.close()
    shutil.rmtree(tmp_dir)
    sys.exit(1)

# ── 4. 收集結果並存回 DB ──────────────────────────────────────────────────────
plaque_out  = tmp_dir / "plaque_output"
stats_path  = plaque_out / "plaque_by_fdi_stats.json"
tooth_path  = tmp_dir / "real_teeth_analysis" / "real_teeth_analysis.json"
glb_src     = plaque_out / "plaque_by_fdi.glb"
obj_src     = plaque_out / "plaque_by_fdi.obj"

stats = {}
if stats_path.exists():
    with open(stats_path) as f:
        stats = json.load(f)

tooth_json = {}
if tooth_path.exists():
    with open(tooth_path) as f:
        tooth_json = json.load(f)

# 複製 GLB/OBJ 到 data/2/plaque_output 並用 _t 後綴標記
dest_dir  = DATA2 / "plaque_output"
date_str  = now_taipei().strftime("%Y%m%d")
glb_fname = f"plaque_{date_str}_{analysis_id}_t.glb"
obj_fname = f"plaque_{date_str}_{analysis_id}_t.obj"

if glb_src.exists():
    shutil.copy2(glb_src, dest_dir / glb_fname)
    print(f"✅ GLB 複製到 {dest_dir / glb_fname}")

if obj_src.exists():
    shutil.copy2(obj_src, dest_dir / obj_fname)
    print(f"✅ OBJ 複製到 {dest_dir / obj_fname}")

result = {
    "glb_url":        f"/files/{glb_fname}",
    "obj_url":        f"/files/{obj_fname}",
    "stats":          stats,
    "tooth_analysis": tooth_json,
    "model_source":   "teaching",
}

analysis.status       = AnalysisStatus.done
analysis.completed_at = now_taipei()
analysis.result_json  = json.dumps(result, ensure_ascii=False)
db.commit()
db.close()

print(f"\n🎉 完成！analysis_id={analysis_id}")
print(f"   GLB: /files/{glb_fname}")
print(f"   菌斑覆蓋率: {stats.get('plaque_ratio', 'N/A')}")
print(f"   有菌斑牙齒: {stats.get('sat_plaque_fdi_count', 'N/A')}")

# ── 5. 清理臨時目錄 ───────────────────────────────────────────────────────────
shutil.rmtree(tmp_dir)
print(f"🧹 臨時目錄已清除")
