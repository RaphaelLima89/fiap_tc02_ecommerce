"""Avalia o MLP treinado contra um baseline Scikit-Learn no conjunto de teste."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from recsys.config import get_settings
from recsys.models.factory import ModelFactory, ModelType

FEATURE_COLUMNS = [
    "times_ordered",
    "avg_add_to_cart_order",
    "last_order_number",
    "avg_days_since_prior_order",
]


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Calcula accuracy, precision, recall, F1 e ROC-AUC."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def load_mlp(settings, train_params, input_dim: int) -> torch.nn.Module:
    """Reconstrói a arquitetura do MLP e carrega os pesos salvos."""
    model = ModelFactory.create_model(
        ModelType(train_params["model_type"]),
        input_dim=input_dim,
    )
    state_dict = torch.load(settings.models_dir / "model.pt")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main() -> None:
    settings = get_settings()
    params = load_params()
    train_params = params["train"]

    train_df = pd.read_parquet(settings.data_processed_dir / "train.parquet")
    test_df = pd.read_parquet(settings.data_processed_dir / "test.parquet")

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["reordered"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["reordered"]

    # --- baseline: Logistic Regression ---
    baseline = LogisticRegression(max_iter=1000)
    baseline.fit(X_train, y_train)
    baseline_proba = baseline.predict_proba(X_test)[:, 1]
    baseline_pred = (baseline_proba >= 0.5).astype(int)
    baseline_metrics = compute_metrics(y_test, baseline_pred, baseline_proba)

    # --- MLP ---
    mlp = load_mlp(settings, train_params, input_dim=len(FEATURE_COLUMNS))
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    with torch.no_grad():
        logits = mlp(X_test_tensor)
        mlp_proba = torch.sigmoid(logits).squeeze().numpy()
    mlp_pred = (mlp_proba >= 0.5).astype(int)
    mlp_metrics = compute_metrics(y_test, mlp_pred, mlp_proba)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("recsys-reorder-prediction")

    with mlflow.start_run():
        mlflow.log_metrics({f"baseline_{k}": v for k, v in baseline_metrics.items()})
        mlflow.log_metrics({f"mlp_{k}": v for k, v in mlp_metrics.items()})

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "eval_metrics.json", "w") as f:
        json.dump({"baseline": baseline_metrics, "mlp": mlp_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
