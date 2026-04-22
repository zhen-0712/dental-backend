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

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException, status, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
import uvicorn, uuid, shutil, subprocess, json
from pathlib import Path
from datetime import datetime, timezone, timedelta
import requests
from flask import Flask, Response, request
TZ_TAIPEI = timezone(timedelta(hours=8))
def now_taipei(): return datetime.now(TZ_TAIPEI).replace(tzinfo=None)

from database import get_db, init_db, Analysis, AnalysisType, AnalysisStatus, User
from email_notify import send_analysis_done, send_analysis_failed
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
    env["PYTHONWARNINGS"] = "ignore"   # suppress model-registry UserWarnings flooding stderr
    result = subprocess.run(
        [PYTHON, str(BASE / script_name)],
        capture_output=True, text=True,
        cwd=str(BASE), timeout=600, env=env
    )
    if result.returncode != 0:
        # Filter ONLY pure UserWarning lines (keep Traceback / Exception lines)
        def _clean(text: str) -> str:
            keep = []
            for l in text.splitlines():
                if ('UserWarning' in l or
                        ('Overwriting' in l and 'registry' in l) or
                        ('register_model' in l and 'fn_wrapper' in l)):
                    continue
                keep.append(l)
            return '\n'.join(keep).strip()
        cleaned = _clean(result.stderr) or _clean(result.stdout)
        raw     = result.stderr[-2000:] or result.stdout[-2000:]
        err     = cleaned or raw or f"process exited with code {result.returncode} (no output)"
        return False, err[-2000:]
    return True, ""

VIEW_FILENAMES = {
    "front":          "front.jpg",
    "left_side":      "left_side.jpg",
    "right_side":     "right_side.jpg",
    "upper_occlusal": "upper_occlusal.jpg",
    "lower_occlusal": "lower_occlusal.jpg",
}

def save_uploads(uploads: dict, real_teeth_dir: Path, mirror: bool = False):
    import cv2, numpy as np
    real_teeth_dir.mkdir(parents=True, exist_ok=True)
    # Clear any leftover multi-photo files from previous multi-mode run
    for view in VIEW_FILENAMES:
        for old in real_teeth_dir.glob(f"{view}_[0-9]*.jpg"):
            old.unlink(missing_ok=True)
    for view, upload in uploads.items():
        dest = real_teeth_dir / VIEW_FILENAMES[view]
        upload.file.seek(0)
        raw = upload.file.read()
        if mirror:
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.flip(img, 1)   # horizontal flip
                cv2.imwrite(str(dest), img)
                continue
        with open(dest, "wb") as f:
            f.write(raw)

def save_multi_uploads(uploads: dict, real_teeth_dir: Path, mirror: bool = False):
    """Save multiple photos per view as {view}_0.jpg, {view}_1.jpg …"""
    import cv2, numpy as np
    real_teeth_dir.mkdir(parents=True, exist_ok=True)
    # Clear any leftover single-photo files from previous single-mode run
    for view, fname in VIEW_FILENAMES.items():
        p = real_teeth_dir / fname
        if p.exists():
            p.unlink()
    for view, file_list in uploads.items():
        for i, upload in enumerate(file_list):
            dest = real_teeth_dir / f"{view}_{i}.jpg"
            upload.file.seek(0)
            raw = upload.file.read()
            if mirror:
                arr = np.frombuffer(raw, np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    img = cv2.flip(img, 1)
                    cv2.imwrite(str(dest), img)
                    continue
            with open(dest, "wb") as f:
                f.write(raw)

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

        # 寄送完成通知
        if user_id:
            _db2 = next(get_db())
            try:
                _u = get_user_by_id(_db2, user_id)
                if _u and _u.email:
                    send_analysis_done(_u.email, _u.name, "init")
            finally:
                _db2.close()

    except Exception as e:
        if analysis:
            analysis.status    = AnalysisStatus.failed
            analysis.error_msg = str(e)
            db.commit()
        tasks[task_id].update({"status": "failed", "error": str(e)})
        # 寄送失敗通知
        if user_id:
            _db2 = next(get_db())
            try:
                _u = get_user_by_id(_db2, user_id)
                if _u and _u.email:
                    send_analysis_failed(_u.email, _u.name, "init")
            finally:
                _db2.close()
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

        umodel_dir = udir / "personalized_3d_models_real"
        if not (umodel_dir / "custom_upper_only.obj").exists():
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

        # 補存 tooth_analysis，讓 history 卡片能做多視角驗證計算
        tooth_json = {}
        tooth_path = udir / "real_teeth_analysis" / "real_teeth_analysis.json"
        if tooth_path.exists():
            with open(tooth_path) as f2:
                tooth_json = json.load(f2)
        result = {
            "glb_url":       "/files/plaque_by_fdi.glb",
            "obj_url":       "/files/plaque_by_fdi.obj",
            "stats":         stats,
            "tooth_analysis": tooth_json,
        }

        if analysis:
            analysis.status       = AnalysisStatus.done
            analysis.completed_at = now_taipei()
            analysis.result_json  = json.dumps(result, ensure_ascii=False)
            db.commit()

        tasks[task_id].update({"status": "done", "step": "done", "result": result})

        # 寄送完成通知
        if user_id:
            _db2 = next(get_db())
            try:
                _u = get_user_by_id(_db2, user_id)
                if _u and _u.email:
                    send_analysis_done(_u.email, _u.name, "plaque")
            finally:
                _db2.close()

    except Exception as e:
        if analysis:
            analysis.status    = AnalysisStatus.failed
            analysis.error_msg = str(e)
            db.commit()
        tasks[task_id].update({"status": "failed", "error": str(e)})
        # 寄送失敗通知
        if user_id:
            _db2 = next(get_db())
            try:
                _u = get_user_by_id(_db2, user_id)
                if _u and _u.email:
                    send_analysis_failed(_u.email, _u.name, "plaque")
            finally:
                _db2.close()
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
    mirror:           str = Form("0"),
    user: User | None = Depends(get_current_user_optional),
    db:   Session     = Depends(get_db),
):
    _udir = user_data_dir(user.id) if user else BASE
    save_uploads({"front": front, "left_side": left_side, "right_side": right_side,
                  "upper_occlusal": upper_occlusal, "lower_occlusal": lower_occlusal},
                 _udir / "real_teeth", mirror=mirror == "1")

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
    mirror:           str = Form("0"),
    user: User | None = Depends(get_current_user_optional),
    db:   Session     = Depends(get_db),
):
    _udir = user_data_dir(user.id) if user else BASE
    save_uploads({"front": front, "left_side": left_side, "right_side": right_side,
                  "upper_occlusal": upper_occlusal, "lower_occlusal": lower_occlusal},
                 _udir / "real_teeth", mirror=mirror == "1")

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

@app.post("/init_multi")
async def init_model_multi(
    background_tasks: BackgroundTasks,
    request: Request,
    user: User | None = Depends(get_current_user_optional),
    db:   Session     = Depends(get_db),
):
    """多張照片模式：每個視角可上傳多張，後端聯集辨識結果。
    FormData 格式：front_0, front_1, left_side_0 … (檔名 = {view}_{index})
    """
    form = await request.form()
    mirror = str(form.get("mirror", "0")) == "1"
    uploads: dict[str, list] = {}
    for key, value in form.multi_items():
        # 用 duck typing 判斷是否為上傳檔（相容 FastAPI/Starlette 兩種 UploadFile）
        if not hasattr(value, 'filename') or not hasattr(value, 'file'):
            continue
        # 解析 view 名稱：找最長符合的 view 前綴
        matched = next(
            (v for v in VIEW_FILENAMES if key.startswith(v + "_") and key[len(v)+1:].isdigit()),
            None
        )
        if matched:
            uploads.setdefault(matched, []).append(value)

    if not uploads:
        raise HTTPException(status_code=400, detail="未收到任何照片")

    _udir = user_data_dir(user.id) if user else BASE
    save_multi_uploads(uploads, _udir / "real_teeth", mirror=mirror)

    task_id = str(uuid.uuid4())[:8]
    analysis_id = None
    if user:
        analysis = Analysis(user_id=user.id, type=AnalysisType.init)
        db.add(analysis); db.commit(); db.refresh(analysis)
        analysis_id = analysis.id

    _uid = user.id if user else None
    tasks[task_id] = {"status": "queued", "step": "waiting", "type": "init",
                      "analysis_id": analysis_id}
    background_tasks.add_task(run_init_pipeline, task_id, analysis_id, _uid)
    return {"task_id": task_id, "status": "queued", "type": "init"}


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
def get_file(filename: str, token: str = None, user: User | None = Depends(get_current_user_optional), db: Session = Depends(get_db)):
    # Support token via query param (for model-viewer which can't set headers)
    if token and not user:
        payload = decode_token(token)
        if payload:
            user = get_user_by_id(db, int(payload["sub"]))
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

@app.get("/favicon.ico")
def favicon():
    p = Path("/home/Zhen/projects/dental-web/favicon.ico")
    if p.exists():
        return FileResponse(str(p))
    return JSONResponse(status_code=204, content={})

@app.get("/favicon.svg")
def favicon_svg():
    p = Path("/home/Zhen/projects/dental-web/favicon.svg")
    if p.exists():
        return FileResponse(str(p), media_type="image/svg+xml")
    return JSONResponse(status_code=204, content={})

@app.get("/health")
def health():
    return {"status": "ok"}

# ===== Photo Quality Check =====
_VIEW_REQ = {
    "front":          {"minPink": 0.04, "minBrightness": 60},
    "left_side":      {"minPink": 0.03, "minBrightness": 55},
    "right_side":     {"minPink": 0.03, "minBrightness": 55},
    "upper_occlusal": {"minPink": 0.02, "minBrightness": 50},
    "lower_occlusal": {"minPink": 0.02, "minBrightness": 50},
}

@app.post("/check_photo")
async def check_photo(file: UploadFile = File(...), view: str = Form("front")):
    import cv2, numpy as np
    contents = await file.read()
    arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"ok": False, "issues": ["照片讀取失敗"], "tips": [], "stats": {}}

    small = cv2.resize(img, (160, 120))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    avg_brightness = float(gray.mean())
    blur_score     = float(cv2.Laplacian(gray, cv2.CV_64F).var())   # higher = sharper

    b, g, r = small[:,:,0].astype(int), small[:,:,1].astype(int), small[:,:,2].astype(int)
    pink_mask  = (r > 120) & (g < r * 0.85) & (b < r * 0.80) & (r > 80)
    pink_ratio = float(pink_mask.sum()) / (160 * 120)

    req    = _VIEW_REQ.get(view, _VIEW_REQ["front"])
    issues, tips = [], []

    if blur_score < 80:
        issues.append("照片可能模糊")
        tips.append("請確保手機對焦後再拍攝")
    if avg_brightness < req["minBrightness"]:
        issues.append("照片太暗")
        tips.append("請在光線充足的地方拍攝，或開閃光燈")
    elif avg_brightness > 220:
        issues.append("照片過曝")
        tips.append("請避免直接對著強光拍攝")
    if pink_ratio < req["minPink"]:
        if view in ("upper_occlusal", "lower_occlusal"):
            issues.append("看不到足夠的牙齒區域")
            tips.append("請盡量張嘴，讓咬合面完整露出")
        else:
            issues.append("嘴巴開口不足或角度偏差")
            tips.append("請張大嘴，確保牙齒清楚可見")

    return {
        "ok":     len(issues) == 0,
        "issues": issues,
        "tips":   tips[:1],
        "stats":  {
            "brightness": round(avg_brightness),
            "sharpness":  round(blur_score, 1),
            "toothArea":  f"{pink_ratio*100:.1f}%",
        },
    }

@app.get("/static/js/{filename:path}")
async def serve_js(filename: str):
    path = Path("/home/Zhen/projects/dental-web/static/js") / filename
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "not found"})
    return FileResponse(
        str(path), media_type="application/javascript",
        headers={"Cache-Control": "no-cache, must-revalidate"}
    )

app.mount("/static", StaticFiles(directory="/home/Zhen/projects/dental-web/static"), name="static")
app.mount("/", StaticFiles(directory="/home/Zhen/projects/dental-web", html=True), name="web")
# 樹梅派
import requests
from fastapi import Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse

# --- [修改區] 請填入樹莓派當前的 IP ---
PI_IP = "192.168.50.254" 

@app.get("/pi_interface/{path:path}")
@app.get("/pi_interface/")
async def pi_proxy(request: Request, path: str = ""):
    # 建立連向樹莓派的網址
    pi_url = f"http://{PI_IP}:8080/{path}"
    
    # 取得原始請求的參數 (例如 ?query=... 之類的)
    params = dict(request.query_params)
    
    try:
        # 轉發請求給樹莓派
        resp = requests.get(pi_url, params=params, timeout=5)
        
        # 將樹莓派的回應傳回給工作站網頁
        return Response(
            content=resp.content, 
            status_code=resp.status_code, 
            media_type=resp.headers.get("Content-Type")
        )
    except Exception as e:
        return HTMLResponse(content=f"無法連線至樹莓派: {str(e)}", status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)