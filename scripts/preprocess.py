"""Preprocessamento de dados brutos do Instacart"""

from __future__ import annotations

import yaml

from recsys.config import get_settings
from recsys.data.preprocessing import SessionAggregationStrategy


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main() -> None:
    settings = get_settings()
    load_params()

    raw_dir = settings.data_raw_dir
    processed_dir = settings.data_processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    orders = pd.read_csv(raw_dir / "orders.csv")
    prior = pd.read_csv(raw_dir / "order_products__prior.csv")
    train = pd.read_csv(raw_dir / "order_products__train.csv")

    prior = prior.merge(
        orders[["order_id", "user_id", "order_number", "days_since_prior_order"]],
        on="order_id",
    )
    train = train.merge(orders[["order_id", "user_id"]], on="order_id")

    strategy = SessionAggregationStrategy()
    interactions = strategy.transform(prior)

    labels = train[["user_id", "product_id"]].assign(reordered=1)
    interactions = interactions.merge(labels, on=["user_id", "product_id"], how="left")
    interactions["reordered"] = interactions["reordered"].fillna(0).astype(int)

    interactions.to_parquet(processed_dir / "interactions.parquet", index=False)


if __name__ == "__main__":
    main()
