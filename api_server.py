#!/usr/bin/env python3
"""
牙齒分析 API Server

端點：
  POST /init          - 上傳5張照片，跑初始化（建立客製化3D模型）
  POST /plaque        - 上傳5張菌斑照片，跑菌斑分析（需先初始化）
  GET  /status/{id}   - 查詢任務進度
  GET  /result/{id}   - 取得結果
  GET  /files/{name}  - 下載輸出檔案
  GET  /model_status  - 檢查是否已初始化
  GET  /health        - 健康檢查
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

app = FastAPI(title="牙齒分析 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE       = Path("/home/Zhen/projects/SegmentAnyTooth")
REAL_TEETH = BASE / "real_teeth"
OUTPUT_DIR = BASE / "plaque_output"
MODEL_DIR  = BASE / "personalized_3d_models_real"
PYTHON     = "/home/Zhen/anaconda3/envs/triposr/bin/python"

OUTPUT_DIR.mkdir(exist_ok=True)
REAL_TEETH.mkdir(exist_ok=True)

tasks = {}

VIEW_FILENAMES = {
    "front":          "front2_test.jpg",
    "left_side":      "left_side2_test.jpg",
    "right_side":     "right_side2_test.jpg",
    "upper_occlusal": "upper_occlusal2_test.jpg",
    "lower_occlusal": "lower_occlusal2_test.jpg",
}


def run_script(script_name: str) -> tuple[bool, str]:
    result = subprocess.run(
        [PYTHON, str(BASE / script_name)],
        capture_output=True,
        text=True,
        cwd=str(BASE),
        timeout=600,
    )
    if result.returncode != 0:
        return False, result.stderr[-2000:]
    return True, ""


def save_uploads(uploads: dict):
    for view, upload in uploads.items():
        dest = REAL_TEETH / VIEW_FILENAMES[view]
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)


def make_upload_params():
    return {
        "front":          File(...),
        "left_side":      File(...),
        "right_side":     File(...),
        "upper_occlusal": File(...),
        "lower_occlusal": File(...),
    }


# ==================== 初始化流程 ====================

def run_init_pipeline(task_id: str):
    try:
        tasks[task_id]["status"] = "running"

        tasks[task_id]["step"] = "preprocessing"
        ok, err = run_script("preprocess_photos.py")
        if not ok:
            raise Exception(f"preprocess failed:\n{err}")

        tasks[task_id]["step"] = "analyzing"
        ok, err = run_script("analyze_real_teeth.py")
        if not ok:
            raise Exception(f"analyze failed:\n{err}")

        tasks[task_id]["step"] = "creating_3d"
        ok, err = run_script("create_personalized_3d_real.py")
        if not ok:
            raise Exception(f"create_3d failed:\n{err}")

        model_exists = (MODEL_DIR / "custom_upper_only.obj").exists() and \
                       (MODEL_DIR / "custom_lower_only.obj").exists()

        tasks[task_id]["status"] = "done"
        tasks[task_id]["step"]   = "done"
        tasks[task_id]["result"] = {
            "message":     "初始化完成，3D 模型已建立",
            "model_ready": model_exists,
        }

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"]  = str(e)


# ==================== 菌斑分析流程 ====================

def run_plaque_pipeline(task_id: str):
    try:
        tasks[task_id]["status"] = "running"

        if not (MODEL_DIR / "custom_upper_only.obj").exists():
            raise Exception("尚未初始化，請先執行初始化流程")

        tasks[task_id]["step"] = "detecting_plaque"
        ok, err = run_script("color_test/teeth_test.py")
        if not ok:
            raise Exception(f"teeth_test failed:\n{err}")

        tasks[task_id]["step"] = "extracting_regions"
        ok, err = run_script("extract_plaque_regions.py")
        if not ok:
            raise Exception(f"extract_plaque failed:\n{err}")

        tasks[task_id]["step"] = "projecting_plaque"
        ok, err = run_script("project_plaque_by_fdi.py")
        if not ok:
            raise Exception(f"project_plaque failed:\n{err}")

        stats = {}
        stats_path = OUTPUT_DIR / "plaque_by_fdi_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

        tasks[task_id]["status"] = "done"
        tasks[task_id]["step"]   = "done"
        tasks[task_id]["result"] = {
            "glb_url":   "/files/plaque_by_fdi.glb",
            "ply_url":   "/files/plaque_by_fdi.ply",
            "obj_url":   "/files/plaque_by_fdi.obj",
            "stats_url": "/files/plaque_by_fdi_stats.json",
            "stats":     stats,
        }

    except Exception as e:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"]  = str(e)


# ==================== Endpoints ====================

@app.post("/init")
async def init_model(
    background_tasks: BackgroundTasks,
    front:            UploadFile = File(...),
    left_side:        UploadFile = File(...),
    right_side:       UploadFile = File(...),
    upper_occlusal:   UploadFile = File(...),
    lower_occlusal:   UploadFile = File(...),
):
    task_id = str(uuid.uuid4())[:8]
    save_uploads({
        "front": front, "left_side": left_side,
        "right_side": right_side, "upper_occlusal": upper_occlusal,
        "lower_occlusal": lower_occlusal,
    })
    tasks[task_id] = {"status": "queued", "step": "waiting", "type": "init"}
    background_tasks.add_task(run_init_pipeline, task_id)
    return {"task_id": task_id, "status": "queued", "type": "init"}


@app.post("/plaque")
async def analyze_plaque(
    background_tasks: BackgroundTasks,
    front:            UploadFile = File(...),
    left_side:        UploadFile = File(...),
    right_side:       UploadFile = File(...),
    upper_occlusal:   UploadFile = File(...),
    lower_occlusal:   UploadFile = File(...),
):
    task_id = str(uuid.uuid4())[:8]
    save_uploads({
        "front": front, "left_side": left_side,
        "right_side": right_side, "upper_occlusal": upper_occlusal,
        "lower_occlusal": lower_occlusal,
    })
    tasks[task_id] = {"status": "queued", "step": "waiting", "type": "plaque"}
    background_tasks.add_task(run_plaque_pipeline, task_id)
    return {"task_id": task_id, "status": "queued", "type": "plaque"}


@app.get("/model_status")
def model_status():
    ready = (MODEL_DIR / "custom_upper_only.obj").exists() and \
            (MODEL_DIR / "custom_lower_only.obj").exists()
    return {"model_ready": ready}


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
    # 先查 plaque_output，再查 real_teeth_analysis 和 personalized_3d_models_real
    for search_dir in [OUTPUT_DIR, BASE / "real_teeth_analysis", BASE / "personalized_3d_models_real"]:
        path = search_dir / filename
        if path.exists():
            # GLB 需要正確的 media type
            media_type = "model/gltf-binary" if filename.endswith(".glb") else None
            return FileResponse(
                str(path),
                media_type=media_type,
                headers={"Access-Control-Allow-Origin": "*"},
            )
    return JSONResponse(status_code=404, content={"error": "file not found"})


@app.get("/health")
def health():
    return {"status": "ok"}


# StaticFiles 必須在所有 endpoint 之後
app.mount("/static", StaticFiles(
    directory="/home/Zhen/projects/dental-web/static"), name="static")
app.mount("/", StaticFiles(
    directory="/home/Zhen/projects/dental-web", html=True), name="web")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)