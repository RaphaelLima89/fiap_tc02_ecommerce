"""Divide o dataset de interações em treino/teste e normaliza as features numéricas."""

from __future__ import annotations

import joblib
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from recsys.config import get_settings

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
    test_size = params["feature_engineering"]["test_size"]
    seed = params["preprocess"]["random_seed"]

    processed_dir = settings.data_processed_dir
    interactions = pd.read_parquet(processed_dir / "interactions.parquet")

    train_df, test_df = train_test_split(
        interactions,
        test_size=test_size,
        random_state=seed,
        stratify=interactions["reordered"],
    )

    train_df[FEATURE_COLUMNS] = train_df[FEATURE_COLUMNS].fillna(0)
    test_df[FEATURE_COLUMNS] = test_df[FEATURE_COLUMNS].fillna(0)

    scaler = StandardScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    test_df[FEATURE_COLUMNS] = scaler.transform(test_df[FEATURE_COLUMNS])

    joblib.dump(scaler, processed_dir / "scaler.joblib")
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)


if __name__ == "__main__":
    main()
