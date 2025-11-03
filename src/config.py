"""Central configuration for the IMDb movie trends project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
LOGS_DIR = PROJECT_ROOT / "logs"


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration values for dataset acquisition."""

    datasets: tuple[str, ...] = (
        "title.basics.tsv.gz",
        "title.ratings.tsv.gz",
        "title.akas.tsv.gz",
        "title.principals.tsv.gz",
        "title.crew.tsv.gz",
        "name.basics.tsv.gz",
    )
    imdb_s3_bucket: str = "datasets.imdbws.com"
    request_timeout: int = 60
    chunk_size: int = 1024 * 1024  # 1 MB


@dataclass(frozen=True)
class PreprocessingConfig:
    """Configuration values for preprocessing pipeline."""

    min_year: int = 1900
    max_year: int = 2025
    min_runtime: int = 40
    max_runtime: int = 240
    min_votes: int = 100
    min_genres: int = 1
    max_genres: int = 5
    cache_intermediate: bool = True


download_config = DownloadConfig()
preprocess_config = PreprocessingConfig()


def ensure_directories() -> None:
    """Create required directories if they do not already exist."""

    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        CACHE_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        LOGS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
