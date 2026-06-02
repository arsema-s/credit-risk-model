import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import (
    train_test_split
)

from sklearn.linear_model import (
    LogisticRegression
)

from sklearn.ensemble import (
    RandomForestClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent.parent
)

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "model_data.csv"
)

df = pd.read_csv(DATA_PATH)

X = df.drop(
    columns=[
        "CustomerId",
        "is_high_risk"
    ]
)

y = df["is_high_risk"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
)


def evaluate_model(
    model,
    model_name
):

    model.fit(
        X_train,
        y_train
    )

    predictions = (
        model.predict(X_test)
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    with mlflow.start_run(
        run_name=model_name
    ):

        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1",
            f1
        )

        mlflow.log_metric(
            "roc_auc",
            roc_auc
        )

        mlflow.sklearn.log_model(
            model,
            model_name
        )

    print(
        f"\n{model_name}"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall: {recall:.4f}"
    )

    print(
        f"F1: {f1:.4f}"
    )

    print(
        f"ROC AUC: {roc_auc:.4f}"
    )
    return {
    "model": model,
    "model_name": model_name,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc
    }


lr_results = evaluate_model(
    LogisticRegression(
        max_iter=1000
    ),
    "LogisticRegression"
)

rf_results = evaluate_model(
    RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ),
    "RandomForest"
)

results = [
    lr_results,
    rf_results
]

best_model = max(
    results,
    key=lambda x: x["roc_auc"]
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "credit_risk_model.pkl"
)

joblib.dump(
    best_model["model"],
    MODEL_PATH
)

print(
    f"\nBest Model: "
    f"{best_model['model_name']}"
)

print(
    f"ROC AUC: "
    f"{best_model['roc_auc']:.4f}"
)