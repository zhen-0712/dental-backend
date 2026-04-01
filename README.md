# DentalVis Backend

個人化牙齒 3D 模型生成與牙菌斑視覺化系統的後端 pipeline。

本專案為 NCU 牙科 AI 研究專題，基於 [SegmentAnyTooth](https://github.com/thangngoc89/SegmentAnyTooth) 開發。

---

## 專案簡介

上傳五個角度的口腔照片，系統自動完成：

1. 照片預處理（白平衡、對比增強）
2. 牙齒辨識與 FDI 編號標註（基於 SegmentAnyTooth）
3. 個人化 3D 牙齒模型生成（含缺牙移除）
4. 牙菌斑 KNN+HSV 偵測 → 2D → 3D 投射與染色
5. 輸出 PLY / GLB / OBJ 格式供前端顯示

---

## 環境需求

```bash
conda create -n triposr python=3.10
conda activate triposr
pip install -r requirements.txt
conda install -c conda-forge libgl -y  # headless 環境需要
```

模型權重（SegmentAnyTooth）請依照原專案說明申請下載，放至 `weight/` 資料夾。  
公版 3D 牙齒模型放至 `models/` 資料夾。

---

## 執行方式

```bash
conda activate triposr
cd /path/to/SegmentAnyTooth
nohup python api_server.py > api_server.log 2>&1 &
```

API 預設跑在 `0.0.0.0:8080`，對應端點：

| 端點 | 方法 | 說明 |
|------|------|------|
| `/init` | POST | 上傳五張照片，建立個人化 3D 模型（一次性） |
| `/plaque` | POST | 上傳五張照片，執行菌斑分析（可重複） |
| `/status/{task_id}` | GET | 查詢任務進度 |
| `/result/{task_id}` | GET | 取得結果與 3D 檔案連結 |
| `/files/{filename}` | GET | 下載輸出檔案（GLB / PLY / OBJ / JSON） |
| `/model_status` | GET | 檢查 3D 模型是否已初始化 |
| `/health` | GET | 伺服器健康檢查 |

### 上傳欄位（`/init` 與 `/plaque` 相同）

| 欄位名 | 說明 |
|--------|------|
| `front` | 正面照片 |
| `left_side` | 左側面照片 |
| `right_side` | 右側面照片 |
| `upper_occlusal` | 上顎咬合面照片 |
| `lower_occlusal` | 下顎咬合面照片 |

---

## 兩段式流程

```
【初始化 /init】（一次性，建立個人化模型）
preprocess_photos.py          照片預處理
    ↓
analyze_real_teeth.py         SAT 牙齒辨識 + FDI 標註
    ↓
create_personalized_3d_real.py    個人化 3D 模型生成（含缺牙移除、GLB 輸出）

【菌斑分析 /plaque】（可重複執行）
color_test/teeth_test.py      KNN + HSV 菌斑偵測 → mask
    ↓
extract_plaque_regions.py     菌斑區域提取 + SAT ROI 過濾
    ↓
project_plaque_by_fdi.py      菌斑投射到 3D 模型（含缺牙移除）→ GLB 輸出
```

---

## 輸出檔案

| 檔案 | 路徑 | 說明 |
|------|------|------|
| `custom_real_teeth.obj` | `personalized_3d_models_real/` | 個人化牙齒模型（OBJ） |
| `custom_real_teeth.glb` | `personalized_3d_models_real/` | 個人化牙齒模型（GLB，網頁展示用） |
| `custom_upper_only.obj` | `personalized_3d_models_real/` | 上顎模型（含 seg_labels 對應） |
| `custom_lower_only.obj` | `personalized_3d_models_real/` | 下顎模型（含 seg_labels 對應） |
| `upper_seg_labels.npy` | `personalized_3d_models_real/` | 上顎 FDI 標籤陣列 |
| `lower_seg_labels.npy` | `personalized_3d_models_real/` | 下顎 FDI 標籤陣列 |
| `real_teeth_analysis.json` | `real_teeth_analysis/` | 牙齒分析結果（FDI、缺牙、尺寸） |
| `plaque_by_fdi.glb` | `plaque_output/` | 菌斑染色 3D 模型（GLB） |
| `plaque_by_fdi_stats.json` | `plaque_output/` | 各 FDI 菌斑統計 |

---

## 引用

本專案使用 SegmentAnyTooth 進行牙齒辨識與 FDI 編號：

> Nguyen, K. D., Hoang, H. T., Doan, T.-P. H., Dao, K. Q., Wang, D.-H., & Hsu, M.-L. (2025).
> *SegmentAnyTooth: An open-source deep learning framework for tooth enumeration and segmentation in intraoral photos.*
> Journal of Dental Sciences. https://doi.org/10.1016/j.jds.2025.01.003

```bibtex
@article{Nguyen2025SegmentAnyTooth,
  title={SegmentAnyTooth: An open-source deep learning framework for tooth enumeration and segmentation in intraoral photos},
  author={Nguyen, Khoa D. and Hoang, Huy T. and Doan, Thi-Phuong-Hoa and Dao, Kim-Quyen and Wang, Ding-Han and Hsu, Min-Ling},
  journal={Journal of Dental Sciences},
  year={2025},
  doi={10.1016/j.jds.2025.01.003}
}
```

---

## License

程式碼採 MIT License。  
SegmentAnyTooth 模型權重採原作者 Non-Commercial License，商業使用需另行取得授權。