from pathlib import Path
import json
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "processed" / "cleaned_churn_data.csv"
MODEL_PATH = ROOT / "models" / "churn_model.pkl"
METRICS_PATH = ROOT / "models" / "metrics.json"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Не знайдено data/processed/cleaned_churn_data.csv. "
            "Спочатку запустіть notebooks/EDA.ipynb."
        )

    return pd.read_csv(DATA_PATH)


def evaluate_model(model, X_test, y_test) -> dict:
    predictions = model.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "f1_score": float(f1_score(y_test, predictions)),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def main() -> None:
    df = load_data()

    X = df.drop(columns=["churn"])
    y = df["churn"]

    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), feature_names),
        ]
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
    }

    results = {}
    best_model = None
    best_model_name = None
    best_f1 = -1.0

    for model_name, classifier in models.items():
        print(f"Training: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", classifier),
            ]
        )

        pipeline.fit(X_train, y_train)

        metrics = evaluate_model(pipeline, X_test, y_test)
        results[model_name] = metrics

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = pipeline
            best_model_name = model_name

    model_package = {
        "model": best_model,
        "feature_names": feature_names,
        "best_model_name": best_model_name,
        "data_path": str(DATA_PATH.relative_to(ROOT)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model_package, file)

    output = {
        "target": "churn",
        "data_used": str(DATA_PATH.relative_to(ROOT)),
        "best_model": best_model_name,
        "test_size": 0.2,
        "random_state": 42,
        "results": results,
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4, ensure_ascii=False)

    print(f"Best model: {best_model_name}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()