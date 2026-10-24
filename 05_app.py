# -*- coding: utf-8 -*-
"""
FCCL 製程優化與基材厚度分析系統 (FCCL Process Optimization Dashboard)
專為 FCCL 製程工程師與品管 QC/QA 設計
標準規範: IPC-TM-650 Method 2.2.18 / 2.2.4 基材機械厚度檢測法
可直接上傳 GitHub 並發佈至 Streamlit Community Cloud (https://share.streamlit.io)
"""

import json
import os
import hashlib
from pathlib import Path
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(
    page_title="FCCL 製程優化與基材厚度分析系統",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "processed" / "fccl_cleaned.csv"
DEVIATION_PATH = ROOT / "data" / "processed" / "fccl_process_deviation_features.csv"
EQUIPMENT_MODEL_PATH = ROOT / "models" / "equipment_anomaly_model.joblib"
EQUIPMENT_SPECS_PATH = ROOT / "data" / "processed" / "fccl_equipment_specs.json"

# Custom Sleek Industrial CSS
st.markdown("""
<style>
    .stApp {
        background-color: #ffffff;
        color: #0f172a;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        color: #0f172a;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] > div {
        background-color: #ffffff;
    }
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] label {
        color: #0f172a;
    }
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    .metric-title {
        color: #64748b;
        font-size: 0.8rem;
        margin-bottom: 4px;
        font-weight: 500;
    }
    .metric-val {
        color: #0f172a;
        font-size: 1.5rem;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 800;
    }
    .badge-pass {
        background-color: #ecfdf5;
        color: #047857;
        border: 1px solid #a7f3d0;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .badge-ng {
        background-color: #fef2f2;
        color: #b91c1c;
        border: 1px solid #fecaca;
        padding: 2px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# 2. Domain Specs & Anomaly Meta Definitions
FCCL_PRODUCT_SPECS = {
    'FCCL-1L-18/25': {'structure': '1L', 'cu1': 18, 'pi': 37, 'nominal': 55.0, 'tol': 1.5, 'desc': '單面無膠基板 (18µm RA銅箔 + 37µm 澆鑄PI膜)'},
    'FCCL-1L-12/12': {'structure': '1L', 'cu1': 12, 'pi': 12, 'nominal': 24.0, 'tol': 1.0, 'desc': '單面極薄基板 (12µm ED銅箔 + 12µm PI層)'},
    'FCCL-2L-12/20/12': {'structure': '2L', 'cu1': 12, 'pi': 20, 'cu2': 12, 'nominal': 44.0, 'tol': 1.0, 'desc': '雙面無膠板 (12µm Cu + 20µm PI芯層 + 12µm Cu)'},
    'FCCL-2L-12/25/12': {'structure': '2L', 'cu1': 12, 'pi': 25, 'cu2': 12, 'nominal': 49.0, 'tol': 1.0, 'desc': '雙面無膠高頻板 (12µm Cu + 25µm 絕緣PI + 12µm Cu)'},
    'FCCL-2L-09/12/09': {'structure': '2L', 'cu1': 9, 'pi': 12, 'cu2': 9, 'nominal': 30.0, 'tol': 1.0, 'desc': '超微細雙面板 (9µm 超薄銅 + 12µm PI + 9µm 銅)'},
    'FCCL-2L-12/12/12': {'structure': '2L', 'cu1': 12, 'pi': 12, 'cu2': 12, 'nominal': 36.0, 'tol': 1.0, 'desc': '雙面薄型對稱板 (12µm Cu + 12µm PI + 12µm Cu)'},
    'FCCL-2L-18/25/18': {'structure': '2L', 'cu1': 18, 'pi': 25, 'cu2': 18, 'nominal': 61.0, 'tol': 1.5, 'desc': '雙面通用耐折板 (18µm RA銅 + 25µm PI + 18µm 銅)'},
    'FCCL-2L-18/50/18': {'structure': '2L', 'cu1': 18, 'pi': 50, 'cu2': 18, 'nominal': 86.0, 'tol': 1.5, 'desc': '雙面高剛性厚型板 (18µm Cu + 50µm 厚PI芯 + 18µm Cu)'},
    'FCCL-3L-18/12/18': {'structure': '3L', 'cu1': 18, 'pi': 12.5, 'cu2': 18, 'adh': 24, 'nominal': 72.5, 'tol': 1.5, 'desc': '三層有膠耐折板 (18µm Cu + 雙面膠 24µm + 12.5µm PI)'},
    'FCCL-3L-35/20/35': {'structure': '3L', 'cu1': 35, 'pi': 20, 'cu2': 35, 'adh': 70, 'nominal': 160.0, 'tol': 2.5, 'desc': '三層高載流厚銅板 (35µm 1oz銅 + 雙面膠 + 20µm PI)'}
}

ANOMALY_META = {
    'Normal': {'name_zh': '製程正常受控', 'cause': '製程參數在穩定管制區間內，隨機誤差分佈良好', 'action': '維持現行模頭間隙與注膠泵浦轉速設定，標準預防保養'},
    'Center_Bulge': {'name_zh': '幅寬中間突起 (中高/兩側偏薄)', 'cause': '貼合壓輪 (Nip Roll) 中間形變撓曲或乾燥風刀中央風壓熱風不均', 'action': '調整壓輪中央冠狀補償 (Crown compensation)、校正烘箱橫向風刀出風均勻度'},
    'Wedge': {'name_zh': '楔形偏差 (單側偏厚/傾斜)', 'cause': '塗佈模頭左右螺栓間隙不對稱 (Slot Die Gap Imbalance)', 'action': '調校模頭傳動側 (DS) 與操作側 (OS) 差速微調螺栓，補償模頭水平平行度'},
    'Local_Spike': {'name_zh': '局部孤點突起/刮痕突峰', 'cause': '塗佈模唇口結膠/局部微粒污染 (Lip Scratch / Particle contamination)', 'action': '停機擦拭模頭出膠唇口，更換供膠管路精密過濾濾芯 (5µm filter)'},
    'Drift': {'name_zh': '隨長度漸進式漂移 (升厚/降厚)', 'cause': '烘箱乾燥溫度逐漸溫升膨脹，或收放卷張力感測器熱漂移', 'action': '校正烘箱各溫區閉迴路 PID 溫控器，檢查放卷磁粉/伺服張力計校零'},
    'Mean_Shift': {'name_zh': '全幅整體厚度偏移', 'cause': '供料齒輪泵轉速 (Pump Flow) 設定偏差或樹脂膠水固成份/黏度配比漂移', 'action': '校正精密計量齒輪泵流量，檢測膠槽黏度計並即時調整溶劑稀釋配比'}
}

# 3. Data Loader with Intelligent Path Resolution
@st.cache_data
def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到專案批次資料：{CSV_PATH}")
    df_clean = pd.read_csv(CSV_PATH)
    if DEVIATION_PATH.exists():
        df_dev = pd.read_csv(DEVIATION_PATH)
        # Merge deviations on Batch_ID
        dev_cols = [c for c in df_dev.columns if c not in df_clean.columns or c == 'Batch_ID']
        df = pd.merge(df_clean, df_dev[dev_cols], on='Batch_ID', how='left')
    else:
        df = df_clean

    df["Sample_Date"] = pd.to_datetime(df["Sample_Date"], errors="coerce")

    # Calculate Process Capability (Cp, Cpk, Ca)
    def calc_spc(row):
        pts = [row[f'Point_{i}_um'] for i in range(1, 11)]
        mean = np.mean(pts)
        std = max(0.001, np.std(pts, ddof=1))
        usl = row['USL_um']
        lsl = row['LSL_um']
        nom = row['Nominal_Thickness_um']

        cp = (usl - lsl) / (6 * std)
        cpu = (usl - mean) / (3 * std)
        cpl = (mean - lsl) / (3 * std)
        cpk = min(cpu, cpl)
        ca = (mean - nom) / ((usl - lsl) / 2) if (usl - lsl) > 0 else 0
        return pd.Series([round(cp, 3), round(cpk, 3), round(ca, 3)], index=['cp', 'cpk', 'ca'])

    spc_df = df.apply(calc_spc, axis=1)
    df[['cp', 'cpk', 'ca']] = spc_df

    return df


@st.cache_resource
def load_equipment_anomaly_model():
    if not EQUIPMENT_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到設備異常分類模型：{EQUIPMENT_MODEL_PATH}")
    return joblib.load(EQUIPMENT_MODEL_PATH)


@st.cache_data
def load_equipment_specs():
    if not EQUIPMENT_SPECS_PATH.exists():
        raise FileNotFoundError(f"設備參數規格檔不存在：{EQUIPMENT_SPECS_PATH}")
    with EQUIPMENT_SPECS_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


EQUIPMENT_PARAMETERS = [
    ("speed", "產線速度", "m/min"),
    ("unwind", "放捲張力", "N/m"),
    ("viscosity", "膠液黏度", "mPa·s"),
    ("gap", "模頭開縫", "µm"),
    ("pump", "計量泵流量", "mL/min"),
    ("wet", "塗布濕膜厚度", "µm"),
    ("temp", "亞胺化溫度", "°C"),
    ("pressure", "壓合線壓力", "N/mm"),
    ("cure", "熟化溫度", "°C"),
    ("rewind", "收捲張力", "N/m"),
]


def recommend_equipment_adjustments(row, history, model, feature_columns):
    """Use the equipment model to score parameter candidates for rows deviating by 1% or more."""
    baseline = build_equipment_model_input(row, feature_columns)
    classes = list(model.named_steps["classifier"].classes_)
    anomaly_index = classes.index(1)
    baseline_probability = float(model.predict_proba(baseline)[0][anomaly_index])

    line_product = history[
        history["Line"].astype(str).eq(str(row["Line"]))
        & history["Product_Model"].astype(str).eq(str(row["Product_Model"]))
    ]
    product_history = history[history["Product_Model"].astype(str).eq(str(row["Product_Model"]))]
    reference = line_product if len(line_product) >= 5 else product_history
    if len(reference) < 5:
        reference = history

    recommendations = []
    for key, label, _unit in EQUIPMENT_PARAMETERS:
        actual_col, target_col = f"{key}_actual", f"{key}_target"
        if f"{key}_deviation" not in feature_columns or target_col not in row.index:
            continue
        actual = pd.to_numeric(pd.Series([row.get(actual_col)]), errors="coerce").iloc[0]
        target = pd.to_numeric(pd.Series([row.get(target_col)]), errors="coerce").iloc[0]
        if pd.isna(actual) or pd.isna(target):
            continue
        deviation_pct = ((actual - target) / target * 100) if target else 0.0
        if abs(float(deviation_pct)) < 1.0:
            continue

        observed = pd.to_numeric(reference[actual_col], errors="coerce").dropna()
        if observed.empty:
            continue
        lower, upper = observed.quantile(0.05), observed.quantile(0.95)
        candidates = set(np.linspace(lower, upper, 21).tolist())
        candidates.update((float(actual), float(target)))

        best_value, best_probability = float(actual), baseline_probability
        for candidate_value in candidates:
            trial = build_equipment_model_input(
                row,
                feature_columns,
                actual_overrides={actual_col: float(candidate_value)},
            )
            probability = float(model.predict_proba(trial)[0][anomaly_index])
            if probability < best_probability:
                best_value, best_probability = float(candidate_value), probability

        recommendations.append({
            "設備參數": label,
            "單位": _unit,
            "目前實際值": float(actual),
            "設備目標值": float(target),
            "偏差率（%）": float(deviation_pct),
            "建議調整值": best_value,
            "建議方向": "維持現值" if np.isclose(best_value, actual) else ("降低" if best_value < actual else "提高"),
            "目前異常機率（%）": baseline_probability * 100,
            "建議後異常機率（%）": best_probability * 100,
            "機率下降（百分點）": (baseline_probability - best_probability) * 100,
        })
    return pd.DataFrame(recommendations).sort_values("偏差率（%）", key=lambda values: values.abs(), ascending=False) if recommendations else pd.DataFrame()


def build_equipment_model_input(row, feature_columns, actual_overrides=None):
    """Build the deviation features expected by equipment_anomaly_model.joblib."""
    actual_overrides = actual_overrides or {}
    values = {}
    for column in feature_columns:
        if column in ("Line", "Product_Model"):
            values[column] = row[column]
            continue

        if column.endswith("_deviation_pct"):
            parameter_key = column.removesuffix("_deviation_pct")
            target = pd.to_numeric(pd.Series([row.get(f"{parameter_key}_target")]), errors="coerce").iloc[0]
            actual = pd.to_numeric(
                pd.Series([actual_overrides.get(f"{parameter_key}_actual", row.get(f"{parameter_key}_actual"))]),
                errors="coerce",
            ).iloc[0]
            deviation = actual - target
            values[column] = deviation / target * 100 if pd.notna(target) and target != 0 else np.nan
        elif column.endswith("_deviation"):
            parameter_key = column.removesuffix("_deviation")
            target = pd.to_numeric(pd.Series([row.get(f"{parameter_key}_target")]), errors="coerce").iloc[0]
            actual = pd.to_numeric(
                pd.Series([actual_overrides.get(f"{parameter_key}_actual", row.get(f"{parameter_key}_actual"))]),
                errors="coerce",
            ).iloc[0]
            values[column] = actual - target if pd.notna(actual) and pd.notna(target) else np.nan
        else:
            values[column] = row.get(column, np.nan)

    return pd.DataFrame([values], columns=feature_columns)

try:
    df_raw = load_data()
except Exception as e:
    st.error(f"資料庫載入失敗: {e}")
    st.stop()

# 4. Sidebar Controls & Filtering
st.sidebar.markdown("### 🎛️ FCCL 製程品管篩選器")

line_types = {"1L": "單層", "2L": "雙層", "3L": "三層"}
line_options = {"全部產線": None}
for line_code, structure in line_types.items():
    product_count = int(df_raw.loc[df_raw["Line"].astype(str) == line_code, "Product_Model"].nunique())
    line_options[f"{line_code} {structure}產線（{product_count} 種產品）"] = line_code
line_filter = st.sidebar.selectbox("生產線別 (Production Line)", list(line_options))
line_key = line_options[line_filter]

line_data = df_raw if line_key is None else df_raw[df_raw["Line"].astype(str) == line_key]
model_options = ["全部型號"] + sorted(line_data["Product_Model"].dropna().astype(str).unique().tolist())
model_filter = st.sidebar.selectbox("產品型號 (Product Model)", model_options)

qc_filter = st.sidebar.radio("QC 檢驗判定", ["ALL (全部)", "PASS (合格)", "NG (超差)"], horizontal=True)

anomaly_labels = {
    "正常": "Normal",
    "中央隆起": "Center_Bulge",
    "楔形偏差": "Wedge",
    "局部突點": "Local_Spike",
    "厚度漂移": "Drift",
    "整體厚度偏移": "Mean_Shift",
}
anomaly_options = {"全部模式": None, **{label: code for label, code in anomaly_labels.items()}}
anomaly_filter = st.sidebar.selectbox("截面異常模式", list(anomaly_options))
anomaly_key = anomaly_options[anomaly_filter]

date_mode = st.sidebar.selectbox("日期篩選方式", ["不限制", "單日", "月份", "季度", "年度"])
date_source = line_data.copy()
if model_filter != "全部型號":
    date_source = date_source[date_source["Product_Model"] == model_filter]
if qc_filter.startswith("PASS"):
    date_source = date_source[date_source["QC_Result"] == "PASS"]
elif qc_filter.startswith("NG"):
    date_source = date_source[date_source["QC_Result"] == "NG"]
if anomaly_key is not None:
    date_source = date_source[date_source["Anomaly_Pattern"] == anomaly_key]
available_dates = pd.to_datetime(date_source["Sample_Date"], errors="coerce").dropna()
if date_mode == "單日":
    date_options = sorted(available_dates.dt.strftime("%Y-%m-%d").unique().tolist(), reverse=True)
elif date_mode == "月份":
    date_options = sorted(available_dates.dt.strftime("%Y-%m").unique().tolist(), reverse=True)
elif date_mode == "季度":
    date_options = sorted((available_dates.dt.year.astype(str) + "年第" + available_dates.dt.quarter.astype(str) + "季").unique().tolist(), reverse=True)
elif date_mode == "年度":
    date_options = sorted(available_dates.dt.year.astype(str).unique().tolist(), reverse=True)
else:
    date_options = []
date_selection = st.sidebar.selectbox("選擇日期", date_options) if date_options else None

# Filter Dataframe
df_filtered = line_data.copy()
if model_filter != "全部型號":
    df_filtered = df_filtered[df_filtered['Product_Model'] == model_filter]
if qc_filter.startswith("PASS"):
    df_filtered = df_filtered[df_filtered['QC_Result'] == "PASS"]
elif qc_filter.startswith("NG"):
    df_filtered = df_filtered[df_filtered['QC_Result'] == "NG"]
if anomaly_key is not None:
    df_filtered = df_filtered[df_filtered['Anomaly_Pattern'] == anomaly_key]
if date_selection is not None:
    sample_dates = pd.to_datetime(df_filtered["Sample_Date"], errors="coerce")
    if date_mode == "單日":
        date_mask = sample_dates.dt.strftime("%Y-%m-%d") == date_selection
    elif date_mode == "月份":
        date_mask = sample_dates.dt.strftime("%Y-%m") == date_selection
    elif date_mode == "季度":
        date_mask = sample_dates.dt.year.astype("Int64").astype(str) + "年第" + sample_dates.dt.quarter.astype("Int64").astype(str) + "季" == date_selection
    else:  # 年度
        date_mask = sample_dates.dt.year.astype("Int64").astype(str) == date_selection
    df_filtered = df_filtered[date_mask]

if len(df_filtered) == 0:
    st.warning("查無符合篩選條件之批次數據，請放寬篩選條件。")
    st.stop()

# Let a selected row in the batch table drive the current analysis batch and KPI cards.
filter_signature = (line_key, model_filter, qc_filter, anomaly_key, date_mode, date_selection)
table_widget_key = "batch_database_table_" + hashlib.sha1(repr(filter_signature).encode("utf-8")).hexdigest()[:12]
table_state = st.session_state.get(table_widget_key)
selected_rows = getattr(getattr(table_state, "selection", None), "rows", []) if table_state is not None else []
if selected_rows and 0 <= selected_rows[0] < len(df_filtered):
    current_batch = df_filtered.iloc[selected_rows[0]]
else:
    current_batch = df_filtered.sort_values("Sample_Date").iloc[-1]

# 5. Current batch summary cards
st.markdown("## 📐 FCCL 製程優化與基材厚度分析儀表板")
st.markdown("*QC/QA 機械厚度檢測法 (IPC-TM-650 2.2.18) · 跨幅 10 點連續剖面與機台製程參數優化*")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">當前批號型號</div>
        <div class="metric-val" style="font-size:1.1rem; color:#fde047;">{current_batch['Product_Model']}</div>
        <small style="color:#94a3b8;">標稱: {current_batch['Nominal_Thickness_um']}µm ±{current_batch['Tolerance_um']}µm</small>
    </div>
    """, unsafe_allow_html=True)

with col5:
    is_pass = current_batch['QC_Result'] == 'PASS'
    badge_cls = "badge-pass" if is_pass else "badge-ng"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">當前批號 QC 判定</div>
        <div class="metric-val" style="color:{'#34d399' if is_pass else '#f87171'};">{current_batch['QC_Result']}</div>
        <small style="color:#94a3b8;">均值: {current_batch['Thickness_Mean_um']:.2f}µm (Cpk {current_batch['cpk']:.2f})</small>
    </div>
    """, unsafe_allow_html=True)

with col6:
    pattern_meta = ANOMALY_META.get(current_batch['Anomaly_Pattern'], ANOMALY_META['Normal'])
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">截面特徵模式</div>
        <div class="metric-val" style="font-size:1.05rem; color:#f472b6;">{current_batch['Anomaly_Pattern']}</div>
        <small style="color:#cbd5e1;">{pattern_meta['name_zh'].split(' ')[0]}</small>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# 6. Tabbed Dashboard Views
tabs = st.tabs([
    "📋 批次全數據庫 (Batch Explorer)",
    "📐 10點跨幅剖面 (TD Profiler)",
    "⚙️ 機台參數偏差與根因 (Deviation)",
    "🛠️ 製程調機模擬器 (Simulator)",
    "⚙️ 製程設備參數 (Equipment Settings)"
])

# ==========================================
# TAB 1: Batch Explorer Table & CSV Export
# ==========================================
with tabs[0]:
    st.markdown("### 📋 批次全數據庫")
    database_view = df_filtered[[
        'Batch_ID', 'Sample_Date', 'Line', 'Product_Model', 'cpk', 'QC_Result', 'Anomaly_Pattern'
    ]].copy()
    database_view["Line"] = database_view["Line"].map({"1L": "1L（單層）", "2L": "2L（雙層）", "3L": "3L（三層）"}).fillna(database_view["Line"])
    database_view["QC_Result"] = database_view["QC_Result"].map({"PASS": "合格", "NG": "不合格"}).fillna(database_view["QC_Result"])
    anomaly_names = {code: label for label, code in anomaly_labels.items()}
    database_view["Anomaly_Pattern"] = database_view["Anomaly_Pattern"].map(anomaly_names).fillna(database_view["Anomaly_Pattern"])
    database_view = database_view.rename(columns={
        "Batch_ID": "批次編號",
        "Sample_Date": "取樣日期",
        "Line": "生產線",
        "Product_Model": "產品型號",
        "cpk": "Cpk",
        "QC_Result": "品質判定",
        "Anomaly_Pattern": "截面異常模式",
    })
    st.dataframe(
        database_view,
        hide_index=True,
        use_container_width=True,
        key=table_widget_key,
        on_select="rerun",
        selection_mode="single-row",
    )

# ==========================================
# TAB 2: 10-Point Transverse Thickness Profiler
# ==========================================
with tabs[1]:
    st.markdown("### 📐 基材機械厚度跨幅 10 點剖面 (Cross-Web TD Profile)")
    st.caption(f"批號: {current_batch['Batch_ID']} · 型號: {current_batch['Product_Model']} · 取樣日期: {current_batch['Sample_Date']:%Y-%m-%d} · 檢測方法: IPC-TM-650 2.2.18")

    points = [current_batch[f'Point_{i}_um'] for i in range(1, 11)]
    nominal = current_batch['Nominal_Thickness_um']
    usl = current_batch['USL_um']
    lsl = current_batch['LSL_um']
    tol = current_batch['Tolerance_um']

    x_labels = [f"P{i}<br>(OS)" if i == 1 else (f"P{i}<br>(DS)" if i == 10 else f"P{i}") for i in range(1, 11)]

    fig_prof = go.Figure()

    # Tolerance envelope shaded zone
    fig_prof.add_hrect(
        y0=lsl, y1=usl,
        fillcolor="rgba(16, 185, 129, 0.1)",
        line_width=0,
        annotation_text="合格公差帶 (Spec Zone)",
        annotation_position="top left",
        annotation_font_color="#10b981"
    )

    # USL / LSL Lines
    fig_prof.add_hline(y=usl, line_dash="dash", line_color="#ef4444", annotation_text=f"USL: {usl:.2f}µm", annotation_position="top right")
    fig_prof.add_hline(y=lsl, line_dash="dash", line_color="#ef4444", annotation_text=f"LSL: {lsl:.2f}µm", annotation_position="bottom right")
    fig_prof.add_hline(y=nominal, line_dash="dot", line_color="#38bdf8", annotation_text=f"Target: {nominal:.2f}µm", annotation_position="top left")

    # Actual Profile Curve
    fig_prof.add_trace(go.Scatter(
        x=x_labels,
        y=points,
        mode='lines+markers+text',
        name='實測厚度 (Measured)',
        line=dict(color='#38bdf8' if current_batch['QC_Result'] == 'PASS' else '#fb7185', width=3, shape='spline'),
        marker=dict(size=10, color=['#ef4444' if (p > usl or p < lsl) else '#ffffff' for p in points], line=dict(color='#0284c7', width=2)),
        text=[f"{p:.2f}µm" for p in points],
        textposition="top center",
        textfont=dict(family="JetBrains Mono", size=11, color="#f8fafc")
    ))

    fig_prof.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#090d16",
        height=380,
        margin=dict(l=50, r=80, t=30, b=40),
        yaxis=dict(title="基材實體厚度 (µm)", gridcolor="#1e293b", zeroline=False),
        xaxis=dict(title="幅寬檢測點位 (TD 方向: 操作側 OS ───► 傳動側 DS)", gridcolor="#1e293b")
    )

    st.plotly_chart(fig_prof, use_container_width=True)

    # Bottom Profile Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("平均厚度 (Mean X-bar)", f"{current_batch['Thickness_Mean_um']:.3f} µm", f"Δ {(current_batch['Thickness_Mean_um'] - nominal):+.3f} µm")
    c2.metric("厚度標準差 (Std.Dev σ)", f"{current_batch['Thickness_Std_um']:.3f} µm")
    c3.metric("全幅極差 (Range R)", f"{current_batch['Thickness_Range_um']:.3f} µm", f"公差帶: {tol*2:.1f} µm")
    c4.metric("單批能力指數 (Cpk)", f"{current_batch['cpk']:.2f}", f"Cp {current_batch['cp']:.2f}")

# ==========================================
# TAB 3: Process Parameter Deviation & Root Cause
# ==========================================
with tabs[2]:
    st.markdown("### ⚙️ 機台製程參數偏差瀑布圖與缺陷成因關聯")
    st.caption("對比目標設定值 Target 與實際生產讀值 Actual 之偏差百分比")

    parameter_impacts = {
        "speed": "線速變動會影響塗布與乾燥時間。",
        "unwind": "放卷張力會影響進料穩定與膜面伸展。",
        "viscosity": "黏度會影響膠液流變與塗布穩定度。",
        "gap": "模頭開縫會影響塗布量與橫向均勻度。",
        "pump": "計量泵流量會影響塗布量與整體厚度。",
        "wet": "濕膜厚度會影響乾燥後的基材厚度。",
        "temp": "亞胺化溫度會影響固化與尺寸穩定。",
        "pressure": "壓合線壓力會影響貼合與厚度分布。",
        "cure": "熟化溫度會影響黏著與殘餘應力。",
        "rewind": "收捲張力會影響捲芯壓痕與卷材平整度。",
    }
    params_data = [
        {
            "name": label,
            "dev_pct": current_batch.get(f"{key}_deviation_pct"),
            "impact": parameter_impacts[key],
        }
        for key, label, _unit in EQUIPMENT_PARAMETERS
        if pd.notna(current_batch.get(f"{key}_target"))
        and pd.notna(current_batch.get(f"{key}_actual"))
        and pd.notna(current_batch.get(f"{key}_deviation_pct"))
    ]

    p_df = pd.DataFrame(params_data)

    fig_waterfall = go.Figure(go.Bar(
        y=p_df['name'],
        x=p_df['dev_pct'],
        orientation='h',
        marker=dict(color=['#ef4444' if abs(x) > 2.0 else ('#f59e0b' if abs(x) > 1.0 else '#10b981') for x in p_df['dev_pct']]),
        text=[f"{x:+.2f}%" for x in p_df['dev_pct']],
        textposition="outside"
    ))

    fig_waterfall.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#090d16",
        height=400,
        margin=dict(l=150, r=80, t=30, b=40),
        xaxis=dict(title="參數偏差百分比 (%)", zerolinecolor="#64748b", zerolinewidth=1.5, gridcolor="#1e293b"),
        yaxis=dict(autorange="reversed")
    )

    st.plotly_chart(fig_waterfall, use_container_width=True)

    # Diagnostic recommendation callout
    st.info(f"**【製程根因推定結論】**: {pattern_meta['cause']}\n\n**【工程師處置建議】**: {pattern_meta['action']}")

# ==========================================
# TAB 4: Process Optimization Simulator
# ==========================================
with tabs[3]:
    st.markdown("### 🛠️ 製程參數調機即時模擬器 (Parameter Optimizer)")

    model_bundle = load_equipment_anomaly_model()
    model = model_bundle["model"]
    feature_columns = model_bundle["feature_columns"]
    recommendations = recommend_equipment_adjustments(current_batch, df_raw, model, feature_columns)
    batch_id = str(current_batch["Batch_ID"])

    st.subheader("目前批次設備參數偏差")
    parameter_status_rows = []
    for parameter_key, label, unit in EQUIPMENT_PARAMETERS:
        actual = pd.to_numeric(pd.Series([current_batch.get(f"{parameter_key}_actual")]), errors="coerce").iloc[0]
        target = pd.to_numeric(pd.Series([current_batch.get(f"{parameter_key}_target")]), errors="coerce").iloc[0]
        if pd.isna(actual) or pd.isna(target):
            continue
        deviation_pct = (float(actual) - float(target)) / float(target) * 100 if target else np.nan
        parameter_status_rows.append({
            "設備參數": label,
            "Recipe Target": float(target),
            "目前實際值": float(actual),
            "偏差率（%）": deviation_pct,
            "偏差狀態": "達 1% 門檻" if pd.notna(deviation_pct) and abs(deviation_pct) >= 1 else "未達 1% 門檻",
            "單位": unit,
        })
    st.dataframe(
        pd.DataFrame(parameter_status_rows).round(4),
        hide_index=True,
        width="stretch",
    )

    st.subheader("模型建議調整數據")
    if recommendations.empty:
        st.info("目前沒有單項設備偏差達 1%，因此不產生單項調機建議；這不代表批次 QC 判定合格。")
    else:
        recommendation_view = recommendations[[
            "設備參數", "建議調整值", "單位", "建議方向", "目前異常機率（%）",
            "建議後異常機率（%）", "機率下降（百分點）"
        ]].copy()
        recommendation_view["建議調整值"] = recommendation_view["建議調整值"].round(4)
        for column in ["目前異常機率（%）", "建議後異常機率（%）", "機率下降（百分點）"]:
            recommendation_view[column] = recommendation_view[column].round(2)
        st.dataframe(recommendation_view, hide_index=True, use_container_width=True)
        if recommendations["機率下降（百分點）"].max() <= 0:
            st.info("模型在候選範圍內未找到可降低異常機率的單項調整，建議維持目前設定並依製程規範檢查。")

        recommendation_labels = recommendations["設備參數"].tolist()
        selected_recommendation = st.selectbox("選擇要帶入的模型建議", recommendation_labels, key=f"recommendation_{batch_id}")
        if st.button("套用建議值到模擬參數", key=f"apply_recommendation_{batch_id}"):
            chosen = recommendations.loc[recommendations["設備參數"].eq(selected_recommendation)].iloc[0]
            parameter_key = next(key for key, label, _unit in EQUIPMENT_PARAMETERS if label == selected_recommendation)
            st.session_state[f"sim_{batch_id}_{parameter_key}_actual"] = float(chosen["建議調整值"])
            st.rerun()

    st.subheader("產品型號適用設備參數")
    adjusted = {}
    with st.form(key=f"simulation_form_{batch_id}"):
        st.caption("參數名稱、單位及 Recipe Target 取自目前產品型號的專案資料；不適用或缺值的參數不顯示。")
        input_columns = st.columns(2)
        visible_parameters = []
        for index, (parameter_key, label, unit) in enumerate(EQUIPMENT_PARAMETERS):
            actual_col, target_col = f"{parameter_key}_actual", f"{parameter_key}_target"
            if f"{parameter_key}_deviation" not in feature_columns:
                continue
            actual = pd.to_numeric(pd.Series([current_batch.get(actual_col)]), errors="coerce").iloc[0]
            target = pd.to_numeric(pd.Series([current_batch.get(target_col)]), errors="coerce").iloc[0]
            if pd.isna(actual) or pd.isna(target):
                continue
            visible_parameters.append((parameter_key, label, unit, actual_col, float(actual), float(target)))
            widget_key = f"sim_{batch_id}_{actual_col}"
            with input_columns[index % 2]:
                adjusted[actual_col] = st.number_input(
                    f"{label}（{unit}）｜產品目標 {target:g}",
                    value=float(actual),
                    step=max(abs(float(target)) * 0.01, 0.01),
                    format="%.4f",
                    key=widget_key,
                )
        submitted = st.form_submit_button("以設備參數執行模型預測", type="primary")

    if submitted:
        model_input = build_equipment_model_input(current_batch, feature_columns, adjusted)
        class_ids = list(model.named_steps["classifier"].classes_)
        st.session_state[f"equipment_simulation_result_{batch_id}"] = {
            "predicted_class": int(model.predict(model_input)[0]),
            "anomaly_probability": float(model.predict_proba(model_input)[0][class_ids.index(1)]),
            "adjusted": adjusted,
        }

    simulation = st.session_state.get(f"equipment_simulation_result_{batch_id}")
    result_column, comparison_column = st.columns([1, 2])
    if simulation:
        with result_column:
            st.subheader("模型預測結果")
            st.metric("預測判定", "異常" if simulation["predicted_class"] == 1 else "正常")
            st.metric("異常機率", f"{simulation['anomaly_probability']:.1%}")
        changes = []
        for parameter_key, label, unit, actual_col, actual, target in visible_parameters:
            adjusted_value = float(simulation["adjusted"].get(actual_col, actual))
            changes.append({
                "設備參數": label,
                "單位": unit,
                "Recipe Target": target,
                "批次實際值": actual,
                "模擬調整值": adjusted_value,
            })
        comparison = pd.DataFrame(changes)
        with comparison_column:
            st.subheader("設備參數調整比較")
            st.dataframe(
                comparison.round(4),
                hide_index=True,
                width="stretch",
                alt="模擬調整前後設備參數比較",
            )

    else:
        with comparison_column:
            st.info("輸入本產品型號適用的模擬參數後，按「以設備參數執行模型預測」查看結果。")

# ==========================================
# TAB 5: Product-specific equipment settings
# ==========================================
with tabs[4]:
    st.markdown("### 製程設備參數設定要求")
    st.caption("依設備技術參數規格表，按目前產線與產品型號列出 WS-01 至 WS-09 的 Recipe Target、公差範圍及 QCP 管制項目。")

    equipment_specs = load_equipment_specs()
    available_lines = sorted({spec["line"] for spec in equipment_specs.values()})
    default_line = str(line_key) if line_key is not None else str(current_batch["Line"])
    if line_key is None:
        active_line = st.selectbox(
            "產線",
            available_lines,
            index=available_lines.index(default_line) if default_line in available_lines else 0,
            key="equipment_spec_line_all",
        )
    else:
        active_line = str(line_key)
        st.caption(f"產線：{active_line}（依左側產線篩選同步）")
    line_models = sorted(
        model for model, spec in equipment_specs.items()
        if spec["line"] == active_line
    )
    if not line_models:
        st.warning(f"規格表中沒有 {active_line} 產線的設備參數。")
    else:
        preferred_model = model_filter if model_filter in line_models else str(current_batch["Product_Model"])
        selected_model = st.selectbox(
            "產品型號",
            line_models,
            index=line_models.index(preferred_model) if preferred_model in line_models else 0,
            key=f"equipment_spec_model_{active_line}",
        )
        product_spec = equipment_specs[selected_model]

        st.markdown(f"**目前產線：{active_line}｜產品型號：{selected_model}**")
        summary_columns = st.columns(3)
        summary_columns[0].metric("產品層數", product_spec["layers"])
        summary_columns[1].metric("產品總厚度", f"{product_spec['total_thickness_um']:g} µm")
        summary_columns[2].metric("目標良率", f"≥ {product_spec['yield_target_pct']:g}%")

        for station in product_spec["stations"]:
            station_title = f"{station['code']}｜{station['name']}"
            with st.expander(station_title, expanded=station["code"] == "WS-01"):
                station_rows = [
                    {
                        "設備技術參數": parameter["name"].replace(" (*QCP)", ""),
                        "Recipe Target": parameter["target"],
                        "管制下限": parameter["lower"],
                        "管制上限": parameter["upper"],
                        "單位": parameter["unit"],
                        "管制項目": "QCP" if parameter["qcp"] else "一般設定",
                    }
                    for parameter in station["parameters"]
                ]
                st.dataframe(
                    pd.DataFrame(station_rows),
                    hide_index=True,
                    width="stretch",
                    alt=f"{selected_model} {station['code']} 設備參數與管制範圍",
                )

# ==========================================
