"""Carregamento e consolidação do dataset RetailRocket."""

import logging
from pathlib import Path
import shutil

import kagglehub
import pandas as pd

from src.config import DATA_DIR
from src.data.preprocessor import build_dataset

logger = logging.getLogger(__name__)

RAW_FILENAMES = (
    "events.csv",
    "category_tree.csv",
    "item_properties_part1.csv",
    "item_properties_part2.csv",
)
KAGGLE_DATASET_HANDLE = "retailrocket/ecommerce-dataset"
PROCESSED_FILENAME = "dataset_consolidated.csv"


def _ensure_raw_dataset(raw_dir: Path) -> None:
    """Garante que os arquivos brutos do RetailRocket existem em ``raw_dir``.

    Baixa o dataset via kagglehub apenas se algum arquivo de ``RAW_FILENAMES``
    estiver faltando, e copia somente os arquivos ausentes (idempotente).

    Args:
        raw_dir: Diretório local onde os arquivos brutos devem estar.

    Raises:
        FileNotFoundError: Se algum arquivo ainda faltar após o download.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in RAW_FILENAMES if not (raw_dir / name).exists()]
    if not missing:
        return

    logger.info("Baixando dataset RetailRocket via kagglehub (faltam: %s)", missing)
    cache_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))
    for name in missing:
        source = cache_path / name
        shutil.copy2(source, raw_dir / name)

    still_missing = [name for name in RAW_FILENAMES if not (raw_dir / name).exists()]
    if still_missing:
        raise FileNotFoundError(f"Arquivos não encontrados após download: {still_missing}")


def _load_raw_tables(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê os CSVs brutos de eventos e propriedades de item.

    Args:
        raw_dir: Diretório com os arquivos brutos do RetailRocket.

    Returns:
        Tupla ``(events, item_properties)``, com ``events`` contendo a coluna
        ``datetime`` derivada do timestamp em milissegundos.
    """
    events = pd.read_csv(raw_dir / "events.csv")
    events["datetime"] = pd.to_datetime(events["timestamp"], unit="ms")

    item_properties = pd.concat(
        [
            pd.read_csv(raw_dir / "item_properties_part1.csv"),
            pd.read_csv(raw_dir / "item_properties_part2.csv"),
        ],
        ignore_index=True,
    )
    return events, item_properties


def load_or_build_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """Carrega o dataset RetailRocket consolidado, usando cache em disco quando possível.

    Args:
        force_rebuild: Se True, ignora o cache em ``data/processed/`` e reconstrói o
            dataset a partir dos arquivos brutos (baixando-os se necessário).

    Returns:
        DataFrame consolidado com colunas ``user_id``, ``item_id``, ``event``, ``value``,
        ``timestamp``.
    """
    processed_path = DATA_DIR / "processed" / PROCESSED_FILENAME
    if processed_path.exists() and not force_rebuild:
        logger.info("Carregando dataset consolidado do cache: %s", processed_path)
        return pd.read_csv(processed_path, parse_dates=["timestamp"])

    raw_dir = DATA_DIR / "raw"
    _ensure_raw_dataset(raw_dir)
    events, item_properties = _load_raw_tables(raw_dir)
    dataset = build_dataset(events, item_properties)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(processed_path, index=False)
    logger.info("Dataset consolidado salvo em: %s", processed_path)
    return dataset
