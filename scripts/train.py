"""Treina o modelo MLP de recomendação e registra o experimento no MLflow."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pandas as pd
import torch
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

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


def main() -> None:
    settings = get_settings()
    params = load_params()
    train_params = params["train"]
    seed = params["preprocess"]["random_seed"]

    torch.manual_seed(seed)

    train_df = pd.read_parquet(settings.data_processed_dir / "train.parquet")
    X = torch.tensor(train_df[FEATURE_COLUMNS].values, dtype=torch.float32)
    y = torch.tensor(train_df["reordered"].values, dtype=torch.float32).unsqueeze(1)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=train_params["batch_size"], shuffle=True)

    model = ModelFactory.create_model(
        ModelType(train_params["model_type"]),
        input_dim=len(FEATURE_COLUMNS),
    )
    optimizer = optim.Adam(model.parameters(), lr=train_params["learning_rate"])
    criterion = nn.BCEWithLogitsLoss()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("recsys-reorder-prediction")

    with mlflow.start_run():
        mlflow.log_params(train_params)
        mlflow.log_param("random_seed", seed)

        model.train()
        final_loss = 0.0
        for epoch in range(train_params["epochs"]):
            epoch_loss = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * xb.size(0)
            epoch_loss /= len(dataset)
            final_loss = epoch_loss
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"epoch {epoch + 1}/{train_params['epochs']} - loss: {epoch_loss:.4f}")

        settings.models_dir.mkdir(parents=True, exist_ok=True)
        model_path = settings.models_dir / "model.pt"
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(str(model_path))

    metrics_dir = Path("metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "train_metrics.json", "w") as f:
        json.dump({"final_train_loss": final_loss}, f, indent=2)


if __name__ == "__main__":
    main()
