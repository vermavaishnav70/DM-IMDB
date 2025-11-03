"""Download IMDb non-commercial datasets required for the project."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Iterable

import requests
import typer
from loguru import logger

from config import RAW_DATA_DIR, download_config, ensure_directories
from utils import configure_logging


app = typer.Typer(add_completion=False, help=__doc__)


def _s3_url(filename: str) -> str:
    return f"https://{download_config.imdb_s3_bucket}/{filename}"


def _etag_path(target: Path) -> Path:
    return target.with_suffix(target.suffix + ".md5")


def _checksum(path: Path) -> str:
    md5 = hashlib.md5()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()


def _needs_download(target: Path, remote_etag: str | None) -> bool:
    if not target.exists():
        return True
    etag_path = _etag_path(target)
    if not remote_etag or not etag_path.exists():
        # Unable to compare, re-download to be safe
        return True
    return etag_path.read_text(encoding="utf-8").strip() != remote_etag


def _write_etag(target: Path, etag: str | None) -> None:
    if etag:
        _etag_path(target).write_text(etag, encoding="utf-8")


def download_file(filename: str, overwrite: bool = False) -> None:
    url = _s3_url(filename)
    target = RAW_DATA_DIR / filename

    logger.info("Preparing to download %s", filename)
    response = requests.head(url, timeout=download_config.request_timeout)
    response.raise_for_status()
    remote_etag = response.headers.get("ETag", "").replace('"', "")

    if not overwrite and not _needs_download(target, remote_etag):
        logger.info("File already up to date: %s", filename)
        return

    with requests.get(url, timeout=download_config.request_timeout, stream=True) as r:
        r.raise_for_status()
        with target.open("wb") as file:
            for chunk in r.iter_content(chunk_size=download_config.chunk_size):
                if chunk:
                    file.write(chunk)

    checksum = _checksum(target)
    _write_etag(target, remote_etag or checksum)
    logger.success("Downloaded %s (md5=%s)", filename, checksum)


def download_all(datasets: Iterable[str], overwrite: bool = False) -> None:
    ensure_directories()
    for dataset in datasets:
        download_file(dataset, overwrite=overwrite)


@app.command("all")
def download_all_command(
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Force redownload even if file exists",
    )
) -> None:
    """Download the full IMDb dataset bundle used for this project."""

    download_all(download_config.datasets, overwrite=overwrite)


@app.command("single")
def download_single(filename: str, overwrite: bool = typer.Option(False, "--overwrite")) -> None:
    """Download a single dataset file by name."""

    if filename not in download_config.datasets:
        raise typer.BadParameter(
            f"{filename} is not one of the configured datasets: {download_config.datasets}"
        )
    ensure_directories()
    download_file(filename, overwrite=overwrite)


@app.command("list")
def list_datasets() -> None:
    """List the configured IMDb dataset files."""

    for dataset in download_config.datasets:
        typer.echo(dataset)


def main() -> None:
    ensure_directories()
    configure_logging(Path(os.getenv("IMDB_LOG_FILE", "logs/download.log")))
    app()


if __name__ == "__main__":
    main()
