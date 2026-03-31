#!/usr/bin/env python3
"""
牙齒分析 API Server
POST /analyze     - 上傳5張照片，開始完整 pipeline
GET  /status/{id} - 查詢任務進度
GET  /result/{id} - 取得結果檔案連結
GET  /files/{filename} - 下載輸出檔案
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import shutil
import subprocess
import json
from pathlib import Path
from typing import Optional

app = FastAPI(title="牙齒分析 API")

from fastapi.staticfiles import StaticFiles

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE       = Path("/home/Zhen/projects/SegmentAnyTooth")
TASKS_DIR  = BASE / "api_tasks"
OUTPUT_DIR = BASE / "plaque_output"
TASKS_DIR.mkdir(exist_ok=True)

# 任務狀態存在記憶體（重啟會清空，夠用了）
tasks = {}

EXPECTED_VIEWS = ["front", "left_side", "right_side", "upper_occlusal", "lower_occlusal"]

def run_pipeline(task_id: str, photo_dir: Path):
    try:
        tasks[task_id]["status"] = "running"
        tasks[task_id]["step"]   = "preprocessing"

        # Step 1: 照片預處理
        result = subprocess.run(
            ["conda", "run", "-n", "triposr", "python",
             str(BASE / "preprocess_photos.py"), "--input_dir", str(photo_dir)],
            capture_output=True, text=True, cwd=str(BASE)
        )
        if result.returncode != 0:
            raise Exception(f"preprocess failed: {result.stderr}")

        tasks[task_id]["step"] = "analyzing"

        # Step 2: 分析牙齒
        result = subprocess.run(
            ["conda", "run", "-n", "triposr", "python",
             str(BASE / "analyze_real_teeth.py")],
            capture_output=True, text=True, cwd=str(BASE)
        )
        if result.returncode != 0:
            raise Exception(f"analyze failed: {result.stderr}")

        tasks[task_id]["step"] = "creating_3d"

        # Step 3: 建立 3D 模型
        result = subprocess.run(
            ["conda", "run", "-n", "triposr", "python",
             str(BASE / "create_personalized_3d_real.py")],
            capture_output=True, text=True, cwd=str(BASE)
        )
        if result.returncode != 0:
            raise Exception(f"create_3d failed: {result.stderr}")

        tasks[task_id]["step"] = "projecting_plaque"

        # Step 4: 菌斑投射
        result = subprocess.run(
            ["conda", "run", "-n", "triposr", "python",
             str(BASE / "project_plaque_by_fdi.py")],
            capture_output=True, text=True, cwd=str(BASE)
        )
        if result.returncode != 0:
            raise Exception(f"plaque projection failed: {result.stderr}")

        # 讀取結果 JSON
        stats_path = OUTPUT_DIR / "plaque_by_fdi_stats.json"
        stats = {}
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

        tasks[task_id]["status"] = "done"
        tasks[task_id]["step"]   = "done"
        tasks[task_id]["result"] = {
            "glb_url":  f"/files/plaque_by_fdi.glb",
            "ply_url":  f"/files/plaque_by_fdi.ply",
            "obj_url":  f"/files/plaque_by_fdi.obj",
            "stats_url": f"/files/plaque_by_fdi_stats.json",
            "stats": stats,
        }

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"]  = str(e)


@app.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    front:          UploadFile = File(...),
    left_side:      UploadFile = File(...),
    right_side:     UploadFile = File(...),
    upper_occlusal: UploadFile = File(...),
    lower_occlusal: UploadFile = File(...),
):
    task_id  = str(uuid.uuid4())[:8]
    photo_dir = TASKS_DIR / task_id
    photo_dir.mkdir()

    # 儲存上傳的照片
    files = {
        "front": front,
        "left_side": left_side,
        "right_side": right_side,
        "upper_occlusal": upper_occlusal,
        "lower_occlusal": lower_occlusal,
    }
    for name, upload in files.items():
        dest = photo_dir / f"{name}.jpg"
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)

    tasks[task_id] = {"status": "queued", "step": "waiting"}
    background_tasks.add_task(run_pipeline, task_id, photo_dir)

    return {"task_id": task_id, "status": "queued"}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"error": "task not found"})
    return tasks[task_id]


@app.get("/result/{task_id}")
def get_result(task_id: str):
    if task_id not in tasks:
        return JSONResponse(status_code=404, content={"error": "task not found"})
    t = tasks[task_id]
    if t["status"] != "done":
        return JSONResponse(status_code=400, content={"status": t["status"]})
    return t["result"]


@app.get("/files/{filename}")
def get_file(filename: str):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "file not found"})
    return FileResponse(str(path))


@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="/home/Zhen/projects/dental-web/static"), name="static")
app.mount("/", StaticFiles(directory="/home/Zhen/projects/dental-web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)