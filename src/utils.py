"""Utility helpers used across the IMDb movie trends project."""

from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable

import pandas as pd
from loguru import logger


def configure_logging(log_file: Path) -> None:
    """Configure Loguru logging sinks.

    Parameters
    ----------
    log_file:
        Target file path for persistent logs.
    """

    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    logger.add(log_file, rotation="5 MB", retention="30 days", level="INFO")


def read_tsv(path: Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read a TSV or gzipped TSV file into a DataFrame."""

    if not path.exists():
        msg = f"File not found: {path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    read_kwargs = {
        "sep": "\t",
        "dtype": "string",
        "na_values": ["\\N", ""],
        "keep_default_na": False,
        "low_memory": False,
    }

    if columns is not None:
        read_kwargs["usecols"] = list(columns)

    if path.suffix == ".gz":
        return pd.read_csv(path, compression="gzip", **read_kwargs)

    return pd.read_csv(path, **read_kwargs)


def save_json(data: dict, path: Path) -> None:
    """Persist a dictionary to JSON with pretty formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


@contextmanager
def open_gzip(path: Path) -> Generator[gzip.GzipFile, None, None]:
    """Context manager for reading gzip files."""

    with gzip.open(path, "rb") as gz_file:
        yield gz_file


def safe_cast(series: pd.Series, dtype: str, errors: str = "coerce") -> pd.Series:
    """Cast a pandas Series to a dtype with logging."""

    original_na = series.isna().sum()
    converted = series.astype(dtype, errors=errors)
    new_na = converted.isna().sum()
    if new_na > original_na:
        logger.debug(
            "Casting to %s introduced %d additional missing values", dtype, new_na - original_na
        )
    return converted
