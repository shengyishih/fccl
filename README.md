# FCCL 製程優化與基材厚度分析

本專案提供 FCCL 厚度與製程參數資料、資料分析流程，以及以 Random Forest 評估異常批次的 Streamlit 儀表板。互動頁面採用五個分頁，呈現批次資料庫、跨幅厚度剖面、製程偏差與根因、調機模擬，以及檢測 SOP。

## 專案內容

- `fccl_thickness_dataset.csv`：原始資料集，包含 492 筆批次紀錄與厚度量測、品質檢驗及製程參數。
- `generate_fccl_thickness_dataset.ipynb`：原始資料集的產生流程。
- `02_data_preparation.ipynb`：資料清理、探索性資料分析（EDA）、特徵轉換與 CSV 匯出。
- `03_random_forest_training.ipynb`：訓練並評估異常分類模型。
- `04_model_predict.ipynb`：載入模型並示範預測。
- `05_app.py`：Streamlit 儀表板；讀取下方專案 CSV，並在調機模擬中載入專案 joblib 模型。
- `models/random_forest_anomaly.joblib`：已訓練模型。
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

應用程式使用 `data/processed/fccl_cleaned.csv` 和 `data/processed/fccl_process_deviation_features.csv` 顯示批次資料及製程偏差，並使用 `models/random_forest_anomaly.joblib` 評估調機模擬後的異常分類。所有檔案均取自本專案，不會改讀其他部署頁面上的資料集。

## 資料處理原則

- 移除完全重複列並修剪文字欄位前後空白；日期欄位轉為日期格式。
- 溫度欄位的缺值會保留，不會在沒有製程依據時自行填補。
- 合成資料中的異常標記、異常量測值及 QC 結果會保留，供 EDA 和後續分析使用。
- `Potential_Root_Cause_Synthetic` 是合成資料中的假設根因，不代表實際調查結論。
- 異常標籤與根因欄位用於合成示範；模型結果不代表已在真實產線驗證。
- 厚度測點的規格判定使用每筆資料的 LSL 與 USL，邊界值視為符合規格。

## 中文圖表字型

Notebook 會依序尋找 Microsoft JhengHei、Microsoft YaHei、Noto Sans CJK TC/SC、PingFang TC 等字型。若圖表中的中文顯示為方框，請安裝可用的中文字型並重新啟動 Jupyter kernel 後再執行。

## 環境需求

專案使用 Python 3.12 以上版本；相依套件與版本需求列於 `pyproject.toml` 和 `uv.lock`。可使用 [uv](https://docs.astral.sh/uv/) 安裝並管理環境。
