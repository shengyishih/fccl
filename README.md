# FCCL 製程優化與基材厚度分析

本專案提供 FCCL 厚度與製程參數資料、資料分析流程，以及以 Random Forest 評估異常批次的 Streamlit 儀表板。儀表板以五個分頁呈現批次資料庫、跨幅厚度剖面、製程偏差與根因、調機模擬，以及產線工作站統計分析；製程設備參數改由頁面右上角齒輪圖示與文字按鈕開啟側邊對話視窗。

## 專案內容

- `fccl_thickness_dataset.csv`：原始資料集，包含 492 筆批次紀錄與厚度量測、品質檢驗及製程參數。
- `generate_fccl_thickness_dataset.ipynb`：原始資料集的產生流程。
- `02_data_preparation.ipynb`：資料清理、探索性資料分析（EDA）、特徵轉換與 CSV 匯出。
- `03_random_forest_training.ipynb`：訓練並評估異常分類模型。
- `04_model_predict.ipynb`：載入模型並示範預測。
- `05_app.py`：Streamlit 儀表板；讀取下方專案 CSV，並在調機模擬中載入專案 joblib 模型。
- 工作站統計分析：依產線、產品、日期統計工作站參數偏差、QCP 超界及同批厚度 NG 關聯；缺少站內實測欄位的站點會標示資料不足。
- `models/random_forest_anomaly.joblib`：使用厚度剖面與批次製程欄位預測注入異常的示範模型。
- `models/equipment_anomaly_model.joblib`：以設備參數偏差預測設備異常，供調機模擬器使用。
- `data/processed/fccl_equipment_specs.json`：頁面右上角設備參數視窗及工作站分析使用的產品設備規格。
- `data/processed/`：清理後的批次資料、逐測點資料及製程偏差特徵。

## 執行資料準備 notebook

在專案根目錄啟動 Jupyter，並開啟 `02_data_preparation.ipynb`。依序執行所有儲存格：

```powershell
uv sync
uv run python -m jupyter lab
```

Notebook 會讀取專案根目錄的 `fccl_thickness_dataset.csv`，並建立 `data/processed/` 資料夾，輸出以下檔案：

- `fccl_cleaned.csv`：批次層級清理資料，保留原始欄位。
- `fccl_measurements_long.csv`：將 10 個厚度測點整理為長格式，每個測點各一列。
- `fccl_process_deviation_features.csv`：實際製程值相對目標值的偏差及偏差百分比。

輸出 CSV 使用 UTF-8 with BOM 編碼，方便以 Windows Excel 開啟。

## 執行 Streamlit 應用

```powershell
uv sync
uv run python -m streamlit run 05_app.py
```

應用程式使用 `data/processed/fccl_cleaned.csv` 和 `data/processed/fccl_process_deviation_features.csv` 顯示批次資料及製程偏差，並使用 `models/equipment_anomaly_model.joblib` 評估調機模擬後的設備異常分類。所有檔案均取自本專案，不會改讀其他部署頁面上的資料集。

## 資料處理原則

- 移除完全重複列並修剪文字欄位前後空白；日期欄位轉為日期格式。
- 批次實測參數依產品型號對應設備規格產生；WS-05 溫度在 1L／2L 使用亞胺化核心峰值設定，在 3L 使用熱滾輪鋼輥表面溫度設定。所有批次實測值皆有值；資料為合成示範，不代表產線歷史量測。
- 合成資料中的異常標記、異常量測值及 QC 結果會保留，供 EDA 和後續分析使用。
- 每種注入厚度異常均在合成資料中配置對應設備參數偏差，讓設備模型有可觀察的輸入訊號；這是示範用的合成因果設定。
- WS-04 Zone 1-2、3-4、5-6 的批次實際溫度也是依各型號 QCP Target 產生的合成示範值，不是產線歷史量測值。
- WS-05 批次溫度依產品型號對應站別規格產生：1L／2L 為亞胺化核心峰值，3L 為熱滾輪鋼輥表面溫度；不是跨站通用的同一種溫度量測。
- `Potential_Root_Cause_Synthetic` 是合成資料中的假設根因，不代表實際調查結論。
- 10 點量測 QC 結果依 LSL/USL 判定；設備偏差和模型機率是另一項訊號，兩者不可互相取代。
- 異常標籤與根因欄位用於合成示範；模型結果不代表已在真實產線驗證。
- 厚度測點的規格判定使用每筆資料的 LSL 與 USL，邊界值視為符合規格。

## 中文圖表字型

Notebook 會依序尋找 Microsoft JhengHei、Microsoft YaHei、Noto Sans CJK TC/SC、PingFang TC 等字型。若圖表中的中文顯示為方框，請安裝可用的中文字型並重新啟動 Jupyter kernel 後再執行。

## 環境需求

專案使用 Python 3.12 以上版本；相依套件與版本需求列於 `pyproject.toml` 和 `uv.lock`。可使用 [uv](https://docs.astral.sh/uv/) 安裝並管理環境。
