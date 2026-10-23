import json
import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="FCCL PE 製程管理", page_icon=":material/precision_manufacturing:", layout="wide")
ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "data" / "processed" / "fccl_cleaned.csv"
MODEL_PATH = ROOT / "models" / "random_forest_anomaly.joblib"
DB_PATH = Path(os.environ.get("FCCL_DB_PATH", ROOT / "data" / "process_database.sqlite3"))
PARAMETERS = [("speed", "產線速度", "m/min"), ("unwind", "放卷張力", "N"), ("viscosity", "膠液黏度", "cP"), ("gap", "塗佈間隙", "µm"), ("pump", "計量泵流量", "mL/min"), ("wet", "濕膜厚度", "µm"), ("temp", "烘箱溫度", "°C"), ("pressure", "壓合壓力", "bar"), ("cure", "熟化條件", "°C"), ("rewind", "收卷張力", "N")]
INSPECTION_COLUMNS = ["Batch_ID", "Sample_Date", "Line", "Product_Model", "QC_Result", "Nominal_Thickness_um", "LSL_um", "USL_um", "Thickness_Mean_um", "Thickness_Std_um", "Thickness_Range_um", "Injected_Anomaly"]


def connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH, timeout=30)


def initialize_db():
    """匯入批次 CSV 至本機 SQLite，並建立改善工單資料表。"""
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"找不到批次資料：{CSV_PATH}")
    with connect_db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS app_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.execute("CREATE TABLE IF NOT EXISTS process_work_orders (work_order_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, batch_id TEXT NOT NULL, production_line TEXT NOT NULL, product_model TEXT NOT NULL, predicted_class INTEGER NOT NULL, anomaly_probability REAL NOT NULL, parameter_changes_json TEXT NOT NULL, status TEXT NOT NULL)")
        stamp = str(CSV_PATH.stat().st_mtime_ns)
        old = conn.execute("SELECT value FROM app_metadata WHERE key='batch_source_mtime_ns'").fetchone()
        exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='batches'").fetchone()
        if not exists or old is None or old[0] != stamp:
            pd.read_csv(CSV_PATH).to_sql("batches", conn, if_exists="replace", index=False)
            conn.execute("INSERT OR REPLACE INTO app_metadata VALUES('batch_source_mtime_ns', ?)", (stamp,))


@st.cache_data(ttl=60, max_entries=2)
def load_batches():
    initialize_db()
    with connect_db() as conn:
        return pd.read_sql_query("SELECT * FROM batches", conn, parse_dates=["Sample_Date"])


@st.cache_resource
def load_model(path):
    # 僅載入專案內可信任的 Joblib 模型檔。
    return joblib.load(path)


def load_orders():
    initialize_db()
    with connect_db() as conn:
        return pd.read_sql_query("SELECT * FROM process_work_orders ORDER BY created_at DESC", conn)


def save_order(order):
    with connect_db() as conn:
        conn.execute("INSERT INTO process_work_orders VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            order["work_order_id"], order["created_at"], order["batch_id"], order["production_line"],
            order["product_model"], order["predicted_class"], order["anomaly_probability"],
            json.dumps(order["parameter_changes"], ensure_ascii=False), order["status"]))


def filter_batches(data, key, anomaly_only=False):
    """提供產線、產品型號、月份篩選。"""
    result = data.copy()
    c1, c2, c3 = st.columns(3)
    lines = sorted(result["Line"].dropna().astype(str).unique())
    line = c1.selectbox("產線", ["全部"] + lines, key=key + "_line")
    if line != "全部":
        result = result[result["Line"].astype(str) == line]
    models = sorted(result["Product_Model"].dropna().astype(str).unique())
    product = c2.selectbox("產品型號", ["全部"] + models, key=key + "_model")
    if product != "全部":
        result = result[result["Product_Model"].astype(str) == product]
    months = sorted(result["Month"].dropna().astype(str).unique())
    month = c3.selectbox("月份", ["全部"] + months, key=key + "_month")
    if month != "全部":
        result = result[result["Month"].astype(str) == month]
    if anomaly_only:
        result = result[result["Injected_Anomaly"].eq(1)]
    return result


def classify_wave(values, lsl, usl):
    """依檢測點外觀分類波動型態，僅作描述用途。"""
    values = np.asarray(values, dtype=float)
    tolerance = max((usl - lsl) / 2, 1e-9)
    center = np.mean(values[3:7])
    edges = np.mean(np.r_[values[:2], values[-2:]])
    if np.any((values < lsl) | (values > usl)):
        return "規格超限"
    if center - edges > tolerance * 0.30:
        return "中央隆起"
    if edges - center > tolerance * 0.30:
        return "邊緣隆起"
    if values[-1] - values[0] > tolerance * 0.50:
        return "遞增趨勢"
    if values[0] - values[-1] > tolerance * 0.50:
        return "遞減趨勢"
    if np.ptp(values) > tolerance * 0.80:
        return "高變異"
    return "穩定"


def wave_data(row):
    return pd.DataFrame({"測點": [f"測點 {i}" for i in range(1, 11)], "實際厚度": [row[f"Point_{i}_um"] for i in range(1, 11)], "規格下限": row["LSL_um"], "規格上限": row["USL_um"]})



def recommend_parameters(row, model, feature_columns, history):
    """逐一試算設備實際值，列出可降低模型異常機率的單參數候選值。"""
    base = pd.DataFrame([{column: row[column] for column in feature_columns}])
    class_ids = list(model.named_steps["classifier"].classes_)
    anomaly_index = class_ids.index(1)
    baseline_probability = float(model.predict_proba(base)[0][anomaly_index])
    recommendations = []

    # 優先以相同產線與產品型號的歷史範圍限制候選值，資料少時逐級放寬範圍。
    product_history = history[history["Product_Model"].astype(str).eq(str(row["Product_Model"]))]
    line_product_history = product_history[product_history["Line"].astype(str).eq(str(row["Line"]))]
    reference = line_product_history if len(line_product_history) >= 5 else product_history
    if len(reference) < 5:
        reference = history

    for key, label, unit in PARAMETERS:
        actual_col, target_col = f"{key}_actual", f"{key}_target"
        if actual_col not in feature_columns or target_col not in feature_columns:
            continue
        current = float(row[actual_col]) if pd.notna(row[actual_col]) else float(row[target_col])
        target = float(row[target_col]) if pd.notna(row[target_col]) else current
        observed = pd.to_numeric(reference[actual_col], errors="coerce").dropna()
        if observed.empty:
            continue
        lower, upper = observed.quantile(0.05), observed.quantile(0.95)
        scale = max(abs(target), abs(current), 1.0)
        deltas = [0.005, 0.01, 0.02, 0.05, 0.10]
        candidates = {current}
        if lower <= target <= upper:
            candidates.add(target)
        for fraction in deltas:
            candidates.add(float(np.clip(current - scale * fraction, lower, upper)))
            candidates.add(float(np.clip(current + scale * fraction, lower, upper)))
        best_value, best_probability = current, baseline_probability
        for candidate_value in sorted(candidates):
            if np.isclose(candidate_value, current):
                continue
            trial = base.copy()
            trial.loc[0, actual_col] = candidate_value
            probability = float(model.predict_proba(trial)[0][anomaly_index])
            if probability < best_probability:
                best_value, best_probability = candidate_value, probability
        reduction = baseline_probability - best_probability
        if reduction > 1e-6:
            recommendations.append({
                "設備參數": label,
                "欄位": actual_col,
                "目前實際值": current,
                "設備目標值": target,
                "建議調整值": best_value,
                "建議方向": "降低" if best_value < current else "提高",
                "異常機率（基準）": baseline_probability,
                "異常機率（單項調整）": best_probability,
                "機率下降（百分點）": reduction * 100,
                "單位": unit,
            })
    return pd.DataFrame(recommendations).sort_values("機率下降（百分點）", ascending=False) if recommendations else pd.DataFrame()


try:
    batches = load_batches()
except Exception as exc:
    st.error(f"批次資料載入失敗：{exc}", title="無法載入資料")
    st.stop()

st.title("FCCL PE 製程管理")
st.caption("依產線檢視厚度品質與設備參數，並以異常分類模型支援改善評估。")
st.info("目前批次資料由專案 CSV 匯入本機 SQLite，改善工單也儲存在此 SQLite 檔案。專案尚未提供工廠正式資料庫連線設定。", title="資料來源說明")
with st.sidebar:
    st.header("功能分頁")
    page = st.radio("選擇功能", ["檢測數據分析", "製程設備參數", "異常模擬與改善工單"], key="main_page")
    st.divider()
    st.caption(f"批次數：{len(batches):,}")
    st.caption(f"本機資料庫：{DB_PATH.name}")

if page == "檢測數據分析":
    st.header("檢測數據分析")
    st.write("本頁只顯示厚度檢測和 QC 資訊，不呈現製程設備參數。")
    filtered = filter_batches(batches, "inspection")
    if filtered.empty:
        st.info("目前篩選條件沒有批次資料。")
        st.stop()
    st.dataframe(filtered[[c for c in INSPECTION_COLUMNS if c in filtered.columns]], hide_index=True, alt="批次厚度檢測與 QC 數據")
    batch_id = st.selectbox("選擇批次檢視量測波動", filtered["Batch_ID"].astype(str).tolist(), key="inspection_batch")
    row = filtered.loc[filtered["Batch_ID"].astype(str).eq(batch_id)].iloc[0]
    values = np.array([row[f"Point_{i}_um"] for i in range(1, 11)], dtype=float)
    c1, c2, c3 = st.columns(3)
    c1.metric("批次平均厚度", f"{row['Thickness_Mean_um']:.3f} µm", border=True)
    c2.metric("量測全距", f"{row['Thickness_Range_um']:.3f} µm", border=True)
    c3.metric("波動圖類型", classify_wave(values, row["LSL_um"], row["USL_um"]), border=True)
    st.line_chart(wave_data(row), x="測點", y=["實際厚度", "規格下限", "規格上限"], alt=f"批次 {batch_id} 十點厚度量測與規格界限")
    st.caption("波動圖類型由十點量測的形狀規則推估，僅供檢視，不代表根因診斷或統計管制圖判異。")

elif page == "製程設備參數":
    st.header("製程設備參數")
    filtered = filter_batches(batches, "equipment")
    if filtered.empty:
        st.info("目前篩選條件沒有批次資料。")
        st.stop()
    batch_id = st.selectbox("選擇批次", filtered["Batch_ID"].astype(str).tolist(), key="equipment_batch")
    row = filtered.loc[filtered["Batch_ID"].astype(str).eq(batch_id)].iloc[0]
    rows = []
    for key, label, unit in PARAMETERS:
        target, actual = f"{key}_target", f"{key}_actual"
        if target in batches and actual in batches:
            rows.append({"設備參數": label, "預設目標值": row[target], "批次實際值": row[actual], "偏差（實際−目標）": row[actual] - row[target] if pd.notna(row[actual]) else np.nan, "單位": unit})
    equipment = pd.DataFrame(rows)
    st.subheader("設備預設參數與批次實績")
    st.dataframe(equipment, hide_index=True, alt=f"批次 {batch_id} 設備目標參數與實際值")
    st.bar_chart(equipment.set_index("設備參數")[["預設目標值", "批次實際值"]], horizontal=True, alt=f"批次 {batch_id} 設備預設值與實際參數比較")
    st.caption("預設參數取自批次資料中的 *_target 欄位；此資料集未附獨立的設備參數主檔。")

else:
    st.header("異常製程模擬與改善工單")
    if not MODEL_PATH.exists():
        st.error(f"找不到模型檔：{MODEL_PATH}，請先完成模型訓練。", title="模型尚未部署")
        st.stop()
    try:
        bundle = load_model(str(MODEL_PATH))
        model = bundle["model"]
        feature_columns = bundle["feature_columns"]
    except Exception as exc:
        st.error(f"模型載入失敗：{exc}", title="無法載入模型")
        st.stop()
    filtered = filter_batches(batches, "simulation", anomaly_only=True)
    if filtered.empty:
        st.info("篩選條件內沒有標記為異常的批次。")
        st.stop()
    batch_id = st.selectbox("選擇異常批次", filtered["Batch_ID"].astype(str).tolist(), key="simulation_batch")
    row = filtered.loc[filtered["Batch_ID"].astype(str).eq(batch_id)].iloc[0]
    st.caption(f"產線 {row['Line']}｜產品 {row['Product_Model']}｜批次 {batch_id}")
    st.warning("模型預測異常分類機率，不會預測調參後的厚度曲線。厚度波動圖維持原始檢測數據；設備參數比較圖會隨模擬設定更新。", title="模擬結果範圍")

    st.subheader("模型建議可調參數")
    recommendations = recommend_parameters(row, model, feature_columns, batches)
    if recommendations.empty:
        st.info("目前模型在歷史資料範圍內找不到可降低異常機率的單一參數調整方案。可查看各設備參數後自行輸入模擬值。")
    else:
        recommendation_view = recommendations.drop(columns=["欄位"]).copy()
        for column in ["目前實際值", "設備目標值", "建議調整值"]:
            recommendation_view[column] = recommendation_view[column].round(4)
        recommendation_view[["異常機率（基準）", "異常機率（單項調整）"]] = recommendation_view[["異常機率（基準）", "異常機率（單項調整）"]].round(4)
        recommendation_view["機率下降（百分點）"] = recommendation_view["機率下降（百分點）"].round(2)
        st.dataframe(recommendation_view, hide_index=True, alt="模型試算後可降低異常機率的單項設備參數建議")
        st.caption("每次只改一個參數並固定其他輸入；候選值限制在同產線/產品型號歷史 5–95 百分位範圍（樣本不足時放寬）。此範圍不代表設備允許操作範圍，請先依核准製程規範及安全限制確認。")
        suggestion_labels = recommendations["設備參數"].tolist()
        selected_suggestion = st.selectbox("選擇要帶入模擬的建議", suggestion_labels, key=f"recommendation_choice_{batch_id}")
        if st.button("將建議值帶入模擬設定", key=f"apply_recommendation_{batch_id}"):
            chosen = recommendations.loc[recommendations["設備參數"].eq(selected_suggestion)].iloc[0]
            widget_key = f"sim_{batch_id}_{chosen['欄位']}"
            st.session_state[widget_key] = float(chosen["建議調整值"])
            st.rerun()

    with st.form(key=f"simulation_form_{batch_id}"):
        st.subheader("模擬調整設備參數")
        st.caption("修改實際值後送出模擬；預測結果只作方案比較。")
        adjusted = {}
        columns = st.columns(2)
        for index, (key, label, unit) in enumerate(PARAMETERS):
            actual, target = f"{key}_actual", f"{key}_target"
            if actual not in feature_columns or target not in feature_columns:
                continue
            with columns[index % 2]:
                default = float(row[actual]) if pd.notna(row[actual]) else float(row[target])
                target_value = float(row[target]) if pd.notna(row[target]) else 0.0
                adjusted[actual] = st.number_input(f"{label}調整後實際值（{unit}），目標 {target_value:g}", value=default, step=max(abs(target_value) * 0.01, 0.01), format="%.4f", key=f"sim_{batch_id}_{actual}")
        submitted = st.form_submit_button("執行模擬預測", type="primary")

    if submitted:
        missing = sorted(set(feature_columns) - set(batches.columns))
        if missing:
            st.error(f"模型所需欄位不存在：{missing}")
        else:
            candidate = pd.DataFrame([{column: row[column] for column in feature_columns}])
            for column, value in adjusted.items():
                candidate.loc[:, column] = value
            probabilities = model.predict_proba(candidate)[0]
            classes = list(model.named_steps["classifier"].classes_)
            st.session_state["simulation_result"] = {
                "batch_id": batch_id,
                "adjusted": adjusted,
                "predicted_class": int(model.predict(candidate)[0]),
                "probability": float(probabilities[classes.index(1)]),
            }

    simulation = st.session_state.get("simulation_result")
    if simulation and simulation["batch_id"] == batch_id:
        result, prob = simulation["predicted_class"], simulation["probability"]
        a, b = st.columns(2)
        a.metric("模擬預測結果", "異常" if result == 1 else "正常", border=True)
        b.metric("預測異常機率", f"{prob:.1%}", border=True)
        st.subheader("檢測波動圖（原始量測）")
        st.line_chart(wave_data(row), x="測點", y=["實際厚度", "規格下限", "規格上限"], alt=f"批次 {batch_id} 原始厚度波動圖")
        rows = []
        for key, label, unit in PARAMETERS:
            actual, target = f"{key}_actual", f"{key}_target"
            if actual in simulation["adjusted"]:
                rows.append({"設備參數": label, "預設值": float(row[target]), "原始實際值": float(row[actual]) if pd.notna(row[actual]) else np.nan, "模擬調整值": float(simulation["adjusted"][actual]), "單位": unit})
        comparison = pd.DataFrame(rows)
        st.subheader("模擬參數與設備預設值")
        st.dataframe(comparison, hide_index=True, alt="設備預設值、原始實際值與模擬調整值")
        st.bar_chart(comparison.set_index("設備參數")[["預設值", "原始實際值", "模擬調整值"]], horizontal=True, alt="模擬設備參數與預設值比較")
        changes = []
        for key, label, unit in PARAMETERS:
            actual, target = f"{key}_actual", f"{key}_target"
            if actual in simulation["adjusted"]:
                changes.append({"parameter": key, "label": label, "unit": unit, "original": float(row[actual]) if pd.notna(row[actual]) else None, "target": float(row[target]) if pd.notna(row[target]) else None, "adjusted": float(simulation["adjusted"][actual])})
        order = {
            "work_order_id": str(uuid.uuid4()),
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "batch_id": batch_id, "production_line": str(row["Line"]),
            "product_model": str(row["Product_Model"]), "predicted_class": result,
            "anomaly_probability": prob, "parameter_changes": changes, "status": "待審核",
        }
        with st.expander("預覽改善工單"):
            st.json(order)
        st.download_button("下載工單 JSON", data=json.dumps(order, ensure_ascii=False, indent=2), file_name=f"process_work_order_{batch_id}.json", mime="application/json")
        if st.button("確認調整方案並儲存至製程資料庫", type="primary", key=f"save_{batch_id}"):
            try:
                save_order(order)
                st.success(f"工單已儲存，編號：{order['work_order_id']}")
            except sqlite3.IntegrityError:
                st.error("工單編號重複，請重新儲存。")
            except Exception as exc:
                st.error(f"工單儲存失敗：{exc}")

    st.divider()
    st.subheader("近期改善工單")
    orders = load_orders()
    if orders.empty:
        st.caption("尚無已儲存的改善工單。")
    else:
        st.dataframe(orders, hide_index=True, alt="本機 SQLite 近期改善工單清單")
