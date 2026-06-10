from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"
DATA_PATH = ROOT / "data" / "processed" / "cleaned_churn_data.csv"

st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def load_model_package():
    if not MODEL_PATH.exists():
        st.error("Модель не знайдено. Спочатку запустіть: python src/train_model.py")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        st.error("Очищений датасет не знайдено. Спочатку запустіть notebooks/EDA.ipynb")
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    return None


package = load_model_package()
model = package["model"]
feature_names = package["feature_names"]
best_model_name = package.get("best_model_name", "Saved model")
df = load_dataset()
metrics = load_metrics()

st.title("📡 Прогнозування відтоку клієнтів телеком-компанії")
st.write(
    "Застосунок прогнозує, чи може клієнт припинити користування послугами компанії, "
    "на основі характеристик підписки, рахунків та якості сервісу."
)

with st.expander("Про датасет", expanded=False):
    st.write("Використовується очищений датасет: `data/processed/cleaned_churn_data.csv`")
    st.write(f"Кількість клієнтів: **{df.shape[0]}**")
    st.write(f"Кількість ознак: **{len(feature_names)}**")
    st.dataframe(df.head(10), use_container_width=True)

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Дані нового клієнта")

    with st.form("prediction_form"):
        is_tv_subscriber = st.selectbox("TV subscriber", [0, 1], index=1)
        is_movie_package_subscriber = st.selectbox("Movie package subscriber", [0, 1], index=0)
        subscription_age = st.number_input("Subscription age", min_value=0.0, max_value=20.0, value=1.0, step=0.1)
        bill_avg = st.number_input("Average bill", min_value=0.0, max_value=500.0, value=20.0, step=1.0)
        reamining_contract = st.number_input("Remaining contract", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        service_failure_count = st.number_input("Service failure count", min_value=0, max_value=50, value=0, step=1)
        download_avg = st.number_input("Download average", min_value=0.0, max_value=10000.0, value=50.0, step=1.0)
        upload_avg = st.number_input("Upload average", min_value=0.0, max_value=1000.0, value=5.0, step=1.0)
        download_over_limit = st.number_input("Download over limit", min_value=0, max_value=10, value=0, step=1)

        submitted = st.form_submit_button("Predict churn")

with right_col:
    st.subheader("Результат прогнозу")

    if submitted:
        client_data = {
            "is_tv_subscriber": is_tv_subscriber,
            "is_movie_package_subscriber": is_movie_package_subscriber,
            "subscription_age": subscription_age,
            "bill_avg": bill_avg,
            "reamining_contract": reamining_contract,
            "service_failure_count": service_failure_count,
            "download_avg": download_avg,
            "upload_avg": upload_avg,
            "download_over_limit": download_over_limit,
        }

        input_df = pd.DataFrame([client_data], columns=feature_names)
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])

        st.metric("Ймовірність відтоку", f"{probability:.2%}")

        if probability >= 0.7:
            st.error("Високий ризик відтоку клієнта")
        elif probability >= 0.4:
            st.warning("Середній ризик відтоку клієнта")
        else:
            st.success("Низький ризик відтоку клієнта")

        st.write("Введені дані:")
        st.dataframe(input_df, use_container_width=True)
    else:
        st.info("Заповніть форму зліва та натисніть Predict churn.")

st.divider()
st.subheader("Якість моделі")
st.write(f"Використана модель: **{best_model_name}**")

if metrics:
    best_results = metrics["results"][metrics["best_model"]]
    metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
    metric_col_1.metric("Accuracy", f"{best_results['accuracy']:.3f}")
    metric_col_2.metric("Precision", f"{best_results['precision']:.3f}")
    metric_col_3.metric("Recall", f"{best_results['recall']:.3f}")
    metric_col_4.metric("F1-score", f"{best_results['f1_score']:.3f}")

    st.write("Порівняння моделей:")
    comparison = []
    for model_name, values in metrics["results"].items():
        comparison.append(
            {
                "model": model_name,
                "accuracy": values["accuracy"],
                "precision": values["precision"],
                "recall": values["recall"],
                "f1_score": values["f1_score"],
            }
        )
    st.dataframe(pd.DataFrame(comparison), use_container_width=True)
else:
    st.info("Файл metrics.json не знайдено. Запустіть навчання моделі для відображення метрик.")
