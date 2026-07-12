from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class PreprocessingStrategy(ABC):
    """Interface para estratégias de pré-processamento."""

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transforma os dados de entrada.

        Args:
            data (pd.DataFrame): Dados de entrada a serem transformados.

        Returns:
            pd.DataFrame: Dados transformados.
        """
        raise NotImplementedError


class SessionAggregationStrategy(PreprocessingStrategy):
    """Estratégia de agregação de sessões."""

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return (
            data.groupby(["user_id", "product_id"])
            .agg(
                times_ordered=("order_id", "count"),
                avg_add_to_cart_order=("add_to_cart_order", "mean"),
                last_order_number=("order_number", "max"),
                avg_days_since_prior_order=("days_since_prior_order", "mean"),
            )
            .reset_index()
        )
