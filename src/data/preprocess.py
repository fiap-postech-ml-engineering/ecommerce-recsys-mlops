"""Script de pré-processamento — ponto de entrada do stage "preprocess" do DVC.

Carrega os eventos brutos (`data/raw/`, via `load_or_build_dataset()`), agrega em
interações usuário-item ponderadas (`build_interactions()`) e persiste o resultado em
`data/processed/interactions.parquet`, consumido pelos stages "train" e "evaluate".
"""

import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.data.loader import load_or_build_dataset
from src.data.preprocessor import build_interactions

logger = logging.getLogger(__name__)

INTERACTIONS_PATH = DATA_DIR / "processed" / "interactions.parquet"


def _save_interactions(
    interactions: pd.DataFrame, interactions_path: Path = INTERACTIONS_PATH
) -> Path:
    """Persiste as interações processadas em parquet.

    Args:
        interactions: DataFrame retornado por `build_interactions()`.
        interactions_path: Caminho de destino do parquet.

    Returns:
        Caminho onde o artefato foi salvo.
    """
    interactions_path.parent.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(interactions_path, index=False)
    return interactions_path


def main() -> None:
    """Constrói `data/processed/interactions.parquet` a partir dos eventos brutos."""
    events = load_or_build_dataset()
    interactions = build_interactions(events)

    interactions_path = _save_interactions(interactions)
    logger.info("Interações salvas em: %s", interactions_path)


if __name__ == "__main__":
    main()
