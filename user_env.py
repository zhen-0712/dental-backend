#!/usr/bin/env python3
"""
user_env.py - 使用者資料夾路徑管理

所有 pipeline 腳本透過環境變數 DENTAL_USER_DIR 取得使用者專屬資料夾。
若未設定則 fallback 到原本的路徑（向下相容）。
"""
import os
from pathlib import Path

BASE = Path("/home/Zhen/projects/SegmentAnyTooth")

def get_user_dir() -> Path:
    """取得目前使用者的資料夾，由環境變數 DENTAL_USER_DIR 指定"""
    user_dir = os.environ.get("DENTAL_USER_DIR")
    if user_dir:
        return Path(user_dir)
    # fallback：舊路徑（向下相容）
    return BASE

def get_paths(user_dir: Path = None) -> dict:
    """取得所有路徑的字典"""
    if user_dir is None:
        user_dir = get_user_dir()

    return {
        "user_dir":         user_dir,
        "real_teeth":       user_dir / "real_teeth",
        "real_teeth_proc":  user_dir / "real_teeth_processed",
        "analysis":         user_dir / "real_teeth_analysis",
        "model_dir":        user_dir / "personalized_3d_models_real",
        "teeth_color_test": user_dir / "teeth_color_test",
        "plaque_output":    user_dir / "plaque_output",
        # 共用資源（不依 user）
        "models":           BASE / "models",
        "weight":           BASE / "weight",
    }

def setup_user_dirs(user_dir: Path):
    """建立使用者所需的所有資料夾"""
    paths = get_paths(user_dir)
    for key, path in paths.items():
        if key not in ("models", "weight"):
            path.mkdir(parents=True, exist_ok=True)
    return paths

if __name__ == "__main__":
    # 測試
    d = get_user_dir()
    print(f"User dir: {d}")
    for k, v in get_paths(d).items():
        print(f"  {k}: {v}")