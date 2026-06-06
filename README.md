# Smile Guardian — Backend

個人化牙齒 3D 模型生成與牙菌斑視覺化系統的後端 pipeline。

本專案為 NCU 牙科 AI 研究專題，基於 [SegmentAnyTooth](https://github.com/thangngoc89/SegmentAnyTooth) 開發。

> Demo：[http://140.115.51.163:40111](http://140.115.51.163:40111)
> | GitHub：[Web Frontend](https://github.com/Yung-Chen-Chang/dental-web) | [Mobile App](https://github.com/Yung-Chen-Chang/dentalvis-app)

---

## 系統簡介

上傳五個角度的口腔照片，系統自動完成：

1. 照片預處理（白平衡、對比增強、統一縮放至 512×512）
2. 牙齒辨識與 FDI 編號標註（基於 SegmentAnyTooth：YOLO + SAM 雙模型）
3. 個人化 3D 牙齒模型生成（含缺牙移除、並行 GLB/OBJ 輸出）
4. 牙菌斑 KNN+HSV 偵測 → 2D → 3D 投射與染色（per-tooth FDI 統計）
5. 輸出 PLY / GLB / OBJ 供前端 `<model-viewer>` 與 Expo app 顯示

支援兩種模型模式：
- **regular**：一般使用者的真實牙齒
- **teaching**：教學用假牙模型（預設缺牙清單 18, 28, 31, 38, 46, 48；降低 YOLO 信心閾值至 0.10）

---

## 硬體整合

本系統支援搭配 **Raspberry Pi** 拍攝裝置使用：

- Raspberry Pi 架設鏡頭串流（MJPEG）與 GPIO 控制
- GPIO 同步觸發 LED 補光燈（閃光同步拍攝）
- Pi 控制台：即時預覽、拍攝、瀏覽歷史照片
- 選取五角度照片後，直接透過 REST API POST 至後端進行分析

---

## 環境需求

```bash
conda create -n triposr python=3.10
conda activate triposr
pip install -r requirements.txt
conda install -c conda-forge libgl -y  # headless 環境需要
```

模型權重（SegmentAnyTooth）請依照原專案說明申請下載，放至 `weight/` 資料夾。
公版 3D 牙齒底模放至 `models/` 資料夾。

---

## 執行方式

```bash
conda activate triposr
cd /path/to/SegmentAnyTooth
nohup python api_server.py > api_server.log 2>&1 &
```

API 預設跑在 `0.0.0.0:8080`。

---

## API 端點

| 端點 | 方法 | 說明 |
|------|------|------|
| `/init` | POST | 上傳五張照片，建立個人化 3D 模型（一次性） |
| `/init_multi` | POST | 同上，每角度可上傳多張照片 |
| `/plaque` | POST | 上傳五張照片，執行菌斑分析（可重複） |
| `/status/{task_id}` | GET | 查詢任務進度 |
| `/files/{filename}` | GET | 下載輸出檔案（GLB / PLY / OBJ / JSON） |
| `/model_status` | GET | 檢查 3D 模型是否已初始化 |
| `/analyses` | GET | 取得歷史分析列表 |
| `/health` | GET | 伺服器健康檢查 |

### 上傳欄位（`/init`、`/init_multi`、`/plaque` 共用）

| 欄位名 | 說明 |
|--------|------|
| `front` | 正面照片 |
| `left_side` | 左側面照片（使用者自身左側） |
| `right_side` | 右側面照片（使用者自身右側） |
| `upper_occlusal` | 上顎咬合面照片 |
| `lower_occlusal` | 下顎咬合面照片 |
| `model_type` | `regular`（預設）或 `teaching`（假牙模型） |
| `mirror` | `1` 表示照片需水平翻轉（前置相機） |

---

## 兩段式 Pipeline

```
【初始化 /init】（一次性，建立個人化模型）
preprocess_photos.py              照片預處理（白平衡、縮放至 512×512）
    ↓  （並行）
analyze_real_teeth.py             SAT 牙齒辨識 + FDI 標註
                                  （ThreadPoolExecutor，4 張同時 inference）
    ↓
create_personalized_3d_real.py    個人化 3D 模型生成
                                  （缺牙移除、並行輸出 base.glb / upper.obj / lower.obj）

【菌斑分析 /plaque】（可重複執行）
preprocess_photos.py              照片預處理（同上）
    ↓
teeth_test.py                     KNN + HSV 菌斑偵測 → 二值化 mask
    ↓
extract_plaque_regions.py         菌斑區域提取 + SAT ROI 過濾（per-tooth bbox）
    ↓
project_plaque_by_fdi.py          菌斑 2D→3D 投射（per-FDI 頂點投票）→ plaque.glb
```

### 加速優化

- **模型預載**：`run_preprocess_analyze.py` 讓預處理與 SAM+YOLO 模型載入並行，Phase 2 啟動時模型已就緒
- **YOLO 預熱**：在進入 ThreadPoolExecutor 前對所有 YOLO 模型執行 dummy predict，觸發 `model.fuse()`，避免多執行緒競態（`AttributeError: bn`）
- **並行 inference**：`ThreadPoolExecutor(max_workers=4)` 並行處理五張照片
- **並行匯出**：base.glb / upper.obj / lower.obj / teaching.glb 四個 mesh 同時匯出

---

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `custom_real_teeth.glb` | 個人化牙齒模型（GLB，網頁展示） |
| `custom_real_teeth.obj` | 個人化牙齒模型（OBJ） |
| `custom_upper_only.obj` | 上顎模型 |
| `custom_lower_only.obj` | 下顎模型 |
| `upper_seg_labels.npy` | 上顎 FDI 標籤陣列 |
| `lower_seg_labels.npy` | 下顎 FDI 標籤陣列 |
| `real_teeth_analysis.json` | 牙齒分析結果（FDI、缺牙、可信度） |
| `plaque_by_fdi.glb` | 菌斑染色 3D 模型（GLB） |
| `plaque_by_fdi_stats.json` | 各 FDI 菌斑覆蓋率統計 |

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
