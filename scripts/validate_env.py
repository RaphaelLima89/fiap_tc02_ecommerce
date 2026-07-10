"""
Etapa para validar se as variáveis de ambiente estão configuradas corretamente.

Uso:
    poetry run python scripts/validate_env.py
"""

from __future__ import annotations

import sys

from pydantic import ValidationError

from recsys.config import Settings, get_settings


def main() -> int:
    """
    Carrega as configurações do projeto e retorna sucesso ou falha na validação.

    Returns:
        int: código de saída (0 = sucesso, 1 = falha)
    """

    try:
        settings = get_settings()
    except ValidationError as exception:
        print("Erro ao validar as variáveis de ambiente:", file=sys.stderr)
        print(exception, file=sys.stderr)
        return 1

    print("Configurações carregadas com sucesso:")
    for field_name in Settings.model_fields:
        print(f"  {field_name}: {getattr(settings, field_name)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
