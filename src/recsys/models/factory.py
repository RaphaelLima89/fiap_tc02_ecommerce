from __future__ import annotations

from enum import StrEnum

import torch.nn as nn


class ModelType(StrEnum):
    """Tipos de modelo suportados pela fábrica."""

    MLP = "mlp"
    EMBEDDING = "embedding"


class ModelFactory:
    """Cria instâncias de modelo a partir de um tipo e parâmetros

    Centraliza a criação de modelos, permitindo a adição de novos tipos
    de modelo sem alterar o código existente.
    """

    @staticmethod
    def create_model(model_type: ModelType, **kwargs: object) -> nn.Module:
        """Instancia um modelo com base no tipo fornecido e nos parâmetros adicionais.

        Args:
            model_type (ModelType): O tipo de modelo a ser criado.
            **kwargs: Hiperparâmetros adicionais.

        Returns:
            Instancia de nn.Module pronta para treino.

        Raises:
            ValueError: Se o tipo de modelo fornecido não for suportado.
        """
        if model_type == ModelType.MLP:
            raise NotImplementedError("MLP model não implementado ainda.")
        if model_type == ModelType.EMBEDDING:
            raise NotImplementedError("Embedding model não implementado ainda.")
        raise ValueError(f"Tipo de modelo '{model_type}' não suportado.")
