#!/usr/bin/env python3
"""
牙齒分析 API Server（含帳號管理）

Auth:
  POST /auth/register  - 註冊
  POST /auth/login     - 登入，回傳 JWT
  GET  /auth/me        - 取得目前使用者

Analysis:
  POST /init           - 初始化（需登入）
  POST /plaque         - 菌斑分析（需登入）
  GET  /status/{id}    - 查詢任務進度
  GET  /result/{id}    - 取得結果
  GET  /analyses       - 取得歷史分析清單（需登入）
  GET  /files/{name}   - 下載檔案

System:
  GET  /model_status   - 檢查模型狀態
  GET  /health         - 健康檢查
"""

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import uvicorn, uuid, shutil, subprocess, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
TZ_TAIPEI = timezone(timedelta(hours=8))
def now_taipei(): return datetime.now(TZ_TAIPEI).replace(tzinfo=None), timezone, timedelta
TZ_TAIPEI = timezone(timedelta(hours=8))
def now_taipei(): return datetime.now(TZ_TAIPEI).replace(tzinfo=None)

from database import get_db, init_db, Analysis, AnalysisType, AnalysisStatus, User
from auth import (create_token, decode_token, authenticate_user,
                  create_user, get_user_by_email, get_user_by_id)

app = FastAPI(title="DentalVis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE       = Path("/home/Zhen/projects/SegmentAnyTooth")
DATA_DIR   = BASE / "data"          # 每個 user 的資料根目錄
MODEL_DIR  = BASE / "personalized_3d_models_real"
PYTHON     = "/home/Zhen/anaconda3/envs/triposr/bin/python"
security   = HTTPBearer(auto_error=False)

DATA_DIR.mkdir(exist_ok=True)

# 初始化資料庫
init_db()

# task 暫存（task_id → analysis_id）
tasks: dict[str, dict] = {}

# ==================== Auth 工具 ====================

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    if not credentials:
        raise HTTPException(status_code=401, detail="未登入")
    payload = decode_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Token 無效或已過期")
    user = get_user_by_id(db, int(payload["sub"]))
    if not user:
        raise HTTPException(status_code=401, detail="使用者不存在")
    return user

def get_current_user_optional(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User | None:
    if not credentials:
        return None
    try:
        return get_current_user(credentials, db)
    except:
        return None

# ==================== Pydantic Models ====================

class RegisterRequest(BaseModel):
    email: str
    name:  str
    password: str

class LoginRequest(BaseModel):
    email:    str
    password: str

# ==================== Auth Endpoints ====================

@app.post("/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_email(db, req.email):
        raise HTTPException(status_code=400, detail="Email 已被使用")
    user = create_user(db, req.email, req.name, req.password)
    token = create_token(user.id, user.email)
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.name}}

@app.post("/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = authenticate_user(db, req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Email 或密碼錯誤")
    token = create_token(user.id, user.email)
    return {"token": token, "user": {"id": user.id, "email": user.email, "name": user.name}}

@app.get("/auth/me")
def me(user: User = Depends(get_current_user)):
    return {"id": user.id, "email": user.email, "name": user.name,
            "created_at": user.created_at.isoformat()}

# ==================== 工具函式 ====================

def user_data_dir(user_id: int) -> Path:
    """取得並初始化使用者資料夾"""
    d = DATA_DIR / str(user_id)
    # 建立所有子目錄
    for sub in ["real_teeth", "real_teeth_processed", "real_teeth_analysis",
                "personalized_3d_models_real", "teeth_color_test", "plaque_output"]:
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d

def run_script(script_name: str, user_dir: Path = None) -> tuple[bool, str]:
    import os
    env = os.environ.copy()
    if user_dir:
        env["DENTAL_USER_DIR"] = str(user_dir)
    result = subprocess.run(
        [PYTHON, str(BASE / script_name)],
        capture_output=True, text=True,
        cwd=str(BASE), timeout=600, env=env
    )
    if result.returncode != 0:
        return False, result.stderr[-2000:]
    return True, ""

VIEW_FILENAMES = {
    "front":          "front.jpg",
    "left_side":      "left_side.jpg",
    "right_side":     "right_side.jpg",
    "upper_occlusal": "upper_occlusal.jpg",
    "lower_occlusal": "lower_occlusal.jpg",
}

def save_uploads(uploads: dict, real_teeth_dir: Path):
    real_teeth_dir.mkdir(parents=True, exist_ok=True)
    for view, upload in uploads.items():
        dest = real_teeth_dir / VIEW_FILENAMES[view]
        upload.file.seek(0)
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)

# ==================== 初始化流程 ====================

def run_init_pipeline(task_id: str, analysis_id: int, user_id: int):
    db = next(get_db())
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first() if analysis_id else None
    udir = user_data_dir(user_id) if user_id else BASE
    try:
        if analysis:
            analysis.status = AnalysisStatus.running
            db.commit()

        tasks[task_id]["step"] = "preprocessing"
        ok, err = run_script("preprocess_photos.py", udir)
        if not ok: raise Exception(f"preprocess failed:\n{err}")

        tasks[task_id]["step"] = "analyzing"
        ok, err = run_script("analyze_real_teeth.py", udir)
        if not ok: raise Exception(f"analyze failed:\n{err}")

        tasks[task_id]["step"] = "creating_3d"
        ok, err = run_script("create_personalized_3d_real.py", udir)
        if not ok: raise Exception(f"create_3d failed:\n{err}")

        umodel_dir = udir / "personalized_3d_models_real"
        model_ready = umodel_dir.exists() and (umodel_dir / "custom_upper_only.obj").exists()
        # 讀取牙齒分析 JSON 存入 result
        tooth_json = {}
        tooth_path = udir / "real_teeth_analysis" / "real_teeth_analysis.json"
        if tooth_path.exists():
            with open(tooth_path) as f:
                tooth_json = json.load(f)
        result = {
            "message": "初始化完成",
            "model_ready": model_ready,
            "tooth_analysis": tooth_json,
        }

        if analysis:
            analysis.status       = AnalysisStatus.done
            analysis.completed_at = now_taipei()
            analysis.result_json  = json.dumps(result)
            db.commit()

        tasks[task_id].update({"status": "done", "step": "done", "result": result})

    except Exception as e:
        if analysis:
            analysis.status    = AnalysisStatus.failed
            analysis.error_msg = str(e)
            db.commit()
        tasks[task_id].update({"status": "failed", "error": str(e)})
    finally:
        db.close()

# ==================== 菌斑分析流程 ====================

def run_plaque_pipeline(task_id: str, analysis_id: int, user_id: int):
    db = next(get_db())
    analysis = db.query(Analysis).filter(Analysis.id == analysis_id).first() if analysis_id else None
    udir = user_data_dir(user_id) if user_id else BASE
    try:
        if analysis:
            analysis.status = AnalysisStatus.running
            db.commit()

        if not (MODEL_DIR / "custom_upper_only.obj").exists():
            raise Exception("尚未初始化，請先執行初始化流程")

        tasks[task_id]["step"] = "detecting_plaque"
        ok, err = run_script("color_test/teeth_test.py", udir)
        if not ok: raise Exception(f"teeth_test failed:\n{err}")

        tasks[task_id]["step"] = "extracting_regions"
        ok, err = run_script("extract_plaque_regions.py", udir)
        if not ok: raise Exception(f"extract_plaque failed:\n{err}")

        tasks[task_id]["step"] = "projecting_plaque"
        ok, err = run_script("project_plaque_by_fdi.py", udir)
        if not ok: raise Exception(f"project_plaque failed:\n{err}")

        stats = {}
        stats_path = udir / "plaque_output" / "plaque_by_fdi_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

        result = {
            "glb_url":   "/files/plaque_by_fdi.glb",
            "obj_url":   "/files/plaque_by_fdi.obj",
            "stats":     stats,
        }

        if analysis:
            analysis.status       = AnalysisStatus.done
            analysis.completed_at = now_taipei()
            analysis.result_json  = json.dumps(result, ensure_ascii=False)
            db.commit()

        tasks[task_id].update({"status": "done", "step": "done", "result": result})

    except Exception as e:
        if analysis:
            analysis.status    = AnalysisStatus.failed
            analysis.error_msg = str(e)
            db.commit()
        tasks[task_id].update({"status": "failed", "error": str(e)})
    finally:
        db.close()

# ==================== Analysis Endpoints ====================

@app.post("/init")
async def init_model(
    background_tasks: BackgroundTasks,
    front:            UploadFile = File(...),
    left_side:        UploadFile = File(...),
    right_side:       UploadFile = File(...),
    upper_occlusal:   UploadFile = File(...),
    lower_occlusal:   UploadFile = File(...),
    user: User | None = Depends(get_current_user_optional),
    db:   Session     = Depends(get_db),
):
    _udir = user_data_dir(user.id) if user else BASE
    save_uploads({"front": front, "left_side": left_side, "right_side": right_side,
                  "upper_occlusal": upper_occlusal, "lower_occlusal": lower_occlusal},
                 _udir / "real_teeth")

    task_id = str(uuid.uuid4())[:8]
    analysis_id = None

    if user:
        analysis = Analysis(user_id=user.id, type=AnalysisType.init)
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        analysis_id = analysis.id

    _uid = user.id if user else None
    tasks[task_id] = {"status": "queued", "step": "waiting", "type": "init",
                      "analysis_id": analysis_id}
    background_tasks.add_task(run_init_pipeline, task_id, analysis_id, _uid)
    return {"task_id": task_id, "status": "queued", "type": "init"}

@app.post("/plaque")
async def analyze_plaque(
    background_tasks: BackgroundTasks,
    front:            UploadFile = File(...),
    left_side:        UploadFile = File(...),
    right_side:       UploadFile = File(...),
    upper_occlusal:   UploadFile = File(...),
    lower_occlusal:   UploadFile = File(...),
    user: User | None = Depends(get_current_user_optional),
    db:   Session     = Depends(get_db),
):
    _udir = user_data_dir(user.id) if user else BASE
    save_uploads({"front": front, "left_side": left_side, "right_side": right_side,
                  "upper_occlusal": upper_occlusal, "lower_occlusal": lower_occlusal},
                 _udir / "real_teeth")

    task_id = str(uuid.uuid4())[:8]
    analysis_id = None

    if user:
        analysis = Analysis(user_id=user.id, type=AnalysisType.plaque)
        db.add(analysis)
        db.commit()
        db.refresh(analysis)
        analysis_id = analysis.id

    _uid = user.id if user else None
    tasks[task_id] = {"status": "queued", "step": "waiting", "type": "plaque",
                      "analysis_id": analysis_id}
    background_tasks.add_task(run_plaque_pipeline, task_id, analysis_id, _uid)
    return {"task_id": task_id, "status": "queued", "type": "plaque"}

@app.get("/analyses")
def get_analyses(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    analyses = db.query(Analysis).filter(Analysis.user_id == user.id)\
                 .order_by(Analysis.created_at.desc()).all()
    return [
        {
            "id":           a.id,
            "type":         a.type,
            "status":       a.status,
            "created_at":   a.created_at.isoformat(),
            "completed_at": a.completed_at.isoformat() if a.completed_at else None,
            "result":       json.loads(a.result_json) if a.result_json else None,
        }
        for a in analyses
    ]

@app.get("/model_status")
def model_status(user: User | None = Depends(get_current_user_optional)):
    if user:
        udir = user_data_dir(user.id)
        umodel_dir = udir / "personalized_3d_models_real"
    else:
        umodel_dir = MODEL_DIR
    ready = (umodel_dir / "custom_upper_only.obj").exists() and             (umodel_dir / "custom_lower_only.obj").exists()
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
def get_file(filename: str, user: User | None = Depends(get_current_user_optional)):
    search_dirs = []
    if user:
        udir = user_data_dir(user.id)
        search_dirs += [
            udir / "plaque_output",
            udir / "real_teeth_analysis",
            udir / "personalized_3d_models_real",
        ]
    search_dirs += [
        BASE / "plaque_output",
        BASE / "real_teeth_analysis",
        BASE / "personalized_3d_models_real",
    ]
    for search_dir in search_dirs:
        path = search_dir / filename
        if path.exists():
            media_type = "model/gltf-binary" if filename.endswith(".glb") else None
            return FileResponse(str(path), media_type=media_type,
                                headers={"Access-Control-Allow-Origin": "*"})
    return JSONResponse(status_code=404, content={"error": "file not found"})

@app.get("/health")
def health():
    return {"status": "ok"}

app.mount("/static", StaticFiles(directory="/home/Zhen/projects/dental-web/static"), name="static")
app.mount("/", StaticFiles(directory="/home/Zhen/projects/dental-web", html=True), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)