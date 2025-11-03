"""Preprocess IMDb datasets into an analysis-ready movie table."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer
from loguru import logger

from config import PROCESSED_DATA_DIR, RAW_DATA_DIR, ensure_directories, preprocess_config
from utils import configure_logging, read_tsv, safe_cast, save_json


app = typer.Typer(add_completion=False, help=__doc__)


def _load_title_basics(sample: float | None = None) -> pd.DataFrame:
    path = RAW_DATA_DIR / "title.basics.tsv.gz"
    typer.echo(f"Loading title basics from {path}")
    df = read_tsv(path)
    if sample:
        df = df.sample(frac=sample, random_state=42)
    return df


def _load_title_ratings(sample: float | None = None) -> pd.DataFrame:
    path = RAW_DATA_DIR / "title.ratings.tsv.gz"
    typer.echo(f"Loading title ratings from {path}")
    df = read_tsv(path)
    if sample:
        df = df.sample(frac=sample, random_state=42)
    return df


def _load_title_crew() -> pd.DataFrame:
    path = RAW_DATA_DIR / "title.crew.tsv.gz"
    typer.echo(f"Loading title crew from {path}")
    return read_tsv(path, columns=["tconst", "directors", "writers"])


def _load_title_principals() -> pd.DataFrame:
    path = RAW_DATA_DIR / "title.principals.tsv.gz"
    typer.echo(f"Loading title principals from {path}")
    df = read_tsv(path, columns=["tconst", "ordering", "nconst", "category"])
    df["ordering"] = safe_cast(df["ordering"], "float")
    return df


def _load_name_basics() -> pd.DataFrame:
    path = RAW_DATA_DIR / "name.basics.tsv.gz"
    typer.echo(f"Loading name basics from {path}")
    return read_tsv(path, columns=["nconst", "primaryName", "primaryProfession", "birthYear"])


def _categorise_era(year: int) -> str:
    if year < 1960:
        return "Pre-1960"
    if year < 1980:
        return "1960s-1979"
    if year < 2000:
        return "1980s-1999"
    if year < 2010:
        return "2000s"
    if year < 2020:
        return "2010s"
    return "2020s"


def _runtime_category(runtime: float) -> str:
    if np.isnan(runtime):
        return "Unknown"
    if runtime < 60:
        return "Short (<60m)"
    if runtime < 90:
        return "Compact (60-89m)"
    if runtime < 120:
        return "Feature (90-119m)"
    if runtime < 150:
        return "Extended (120-149m)"
    return "Epic (150m+)"


def _prepare_people_data(principals: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    typer.echo("Merging principal cast & crew with names")
    principals = principals.merge(names, on="nconst", how="left")
    principals.sort_values(["tconst", "ordering"], inplace=True)
    grouped = (
        principals.groupby("tconst")
        .apply(
            lambda df: {
                "top_cast": df[df["category"].isin(["actor", "actress"])]["primaryName"].head(5).tolist(),
                "top_cast_ids": df[df["category"].isin(["actor", "actress"])]["nconst"].head(5).tolist(),
            }
        )
        .rename("principal_data")
        .reset_index()
    )
    grouped[["top_cast", "top_cast_ids"]] = grouped["principal_data"].apply(pd.Series)
    grouped.drop(columns=["principal_data"], inplace=True)
    return grouped


def _process(sample: float | None = None) -> pd.DataFrame:
    basics = _load_title_basics(sample)
    ratings = _load_title_ratings(sample)
    crew = _load_title_crew()
    principals = _load_title_principals()
    names = _load_name_basics()

    typer.echo("Filtering and merging core tables")
    basics = basics[basics["titleType"] == "movie"].copy()
    basics["isAdult"] = safe_cast(basics["isAdult"], "float")
    basics = basics[basics["isAdult"] == 0]

    basics["startYear"] = safe_cast(basics["startYear"], "float")
    basics["runtimeMinutes"] = safe_cast(basics["runtimeMinutes"], "float")

    basics = basics[
        basics["startYear"].between(preprocess_config.min_year, preprocess_config.max_year, inclusive="both")
    ]

    basics = basics[
        basics["runtimeMinutes"].between(
            preprocess_config.min_runtime, preprocess_config.max_runtime, inclusive="both"
        )
    ]

    merged = basics.merge(ratings, on="tconst", how="inner")
    merged["numVotes"] = safe_cast(merged["numVotes"], "float")
    merged = merged[merged["numVotes"] >= preprocess_config.min_votes]

    typer.echo("Processing genres")
    merged["genres"] = merged["genres"].fillna("Unknown")
    merged["genre_list"] = merged["genres"].str.split(",")
    merged["num_genres"] = merged["genre_list"].apply(len)
    merged = merged[
        merged["num_genres"].between(preprocess_config.min_genres, preprocess_config.max_genres)
    ]
    merged["primary_genre"] = merged["genre_list"].apply(lambda g: g[0] if g else "Unknown")
    merged["secondary_genre"] = merged["genre_list"].apply(lambda g: g[1] if len(g) > 1 else np.nan)

    typer.echo("Enriching with crew information")
    crew["directors"] = crew["directors"].fillna("").str.split(",")
    crew["writers"] = crew["writers"].fillna("").str.split(",")
    crew_info = crew.rename(columns={"directors": "director_ids", "writers": "writer_ids"})

    people_data = _prepare_people_data(principals, names)

    merged = merged.merge(crew_info, on="tconst", how="left")
    merged = merged.merge(people_data, on="tconst", how="left")

    typer.echo("Deriving analytical features")
    merged["decade"] = (merged["startYear"] // 10 * 10).astype(int)
    merged["era"] = merged["startYear"].apply(lambda year: _categorise_era(int(year)))
    merged["runtime_category"] = merged["runtimeMinutes"].apply(_runtime_category)
    merged["popularity_score"] = np.log10(merged["numVotes"] + 1)
    merged["success_score"] = merged["averageRating"].astype(float) * merged["popularity_score"]

    columns = [
        "tconst",
        "primaryTitle",
        "originalTitle",
        "startYear",
        "runtimeMinutes",
        "averageRating",
        "numVotes",
        "primary_genre",
        "secondary_genre",
        "genres",
        "genre_list",
        "num_genres",
        "decade",
        "era",
        "runtime_category",
        "popularity_score",
        "success_score",
        "director_ids",
        "writer_ids",
        "top_cast",
        "top_cast_ids",
    ]

    result = merged[columns].copy()
    result["director_ids"] = result["director_ids"].apply(lambda d: d if isinstance(d, list) else [])
    result["writer_ids"] = result["writer_ids"].apply(lambda w: w if isinstance(w, list) else [])
    result["top_cast"] = result["top_cast"].apply(lambda c: c if isinstance(c, list) else [])
    result["top_cast_ids"] = result["top_cast_ids"].apply(lambda c: c if isinstance(c, list) else [])

    return result


def _serialize_lists(df: pd.DataFrame) -> pd.DataFrame:
    list_columns = ["genre_list", "director_ids", "writer_ids", "top_cast", "top_cast_ids"]
    for column in list_columns:
        if column in df.columns:
            df[column] = df[column].apply(json.dumps)
    return df


def _generate_stats(df: pd.DataFrame) -> dict[str, Any]:
    stats = {
        "records": int(len(df)),
        "year_range": {
            "min": int(df["startYear"].min()),
            "max": int(df["startYear"].max()),
        },
        "rating": {
            "mean": float(df["averageRating"].mean()),
            "median": float(df["averageRating"].median()),
        },
        "runtime": {
            "mean": float(df["runtimeMinutes"].mean()),
            "median": float(df["runtimeMinutes"].median()),
        },
        "votes": {
            "total": float(df["numVotes"].sum()),
            "mean": float(df["numVotes"].mean()),
        },
        "genre_counts": df["primary_genre"].value_counts().head(10).to_dict(),
    }
    return stats


def _data_dictionary() -> dict[str, str]:
    return {
        "tconst": "Unique identifier for the title in IMDb",
        "primaryTitle": "The more popular title / the title used by the filmmakers on promotional materials",
        "originalTitle": "Original title, in the original language",
        "startYear": "Release year",
        "runtimeMinutes": "Primary runtime of the movie in minutes",
        "averageRating": "IMDb weighted average user rating (0-10 scale)",
        "numVotes": "Number of votes the title has received",
        "primary_genre": "First genre listed for the movie",
        "secondary_genre": "Second genre listed, if any",
        "genres": "Pipe-separated list of up to three genres as provided by IMDb",
        "genre_list": "JSON array of genres",
        "num_genres": "Count of listed genres",
        "decade": "Decade derived from startYear",
        "era": "Broad era bucket used for analysis",
        "runtime_category": "Categorical runtime bucket",
        "popularity_score": "Log-scaled vote count proxy for popularity",
        "success_score": "Composite score = averageRating ? popularity_score",
        "director_ids": "JSON array of director IMDb IDs",
        "writer_ids": "JSON array of writer IMDb IDs",
        "top_cast": "JSON array of up to five top-billed cast names",
        "top_cast_ids": "JSON array of IMDb IDs for the top-billed cast",
    }


@app.command()
def run(
    sample_frac: float = typer.Option(
        None,
        help="Optional fraction of the dataset to process (useful for testing).",
    ),
    output_file: Path = typer.Option(
        PROCESSED_DATA_DIR / "movies_processed.csv",
        help="Path to save the processed dataset.",
    ),
    stats_file: Path = typer.Option(
        PROCESSED_DATA_DIR / "preprocessing_stats.json",
        help="Path to save summary statistics.",
    ),
    dictionary_file: Path = typer.Option(
        PROCESSED_DATA_DIR / "data_dictionary.json",
        help="Path to save the data dictionary.",
    ),
) -> None:
    """Execute the preprocessing workflow."""

    ensure_directories()
    configure_logging(Path(os.getenv("IMDB_LOG_FILE", "logs/preprocessing.log")))

    logger.info("Starting preprocessing pipeline")
    df = _process(sample=sample_frac)
    stats = _generate_stats(df)

    logger.info("Writing outputs")
    serialize_ready = _serialize_lists(df.copy())
    output_file.parent.mkdir(parents=True, exist_ok=True)
    serialize_ready.to_csv(output_file, index=False)
    save_json(stats, stats_file)
    save_json(_data_dictionary(), dictionary_file)

    logger.success(
        "Preprocessing complete | records=%d | output=%s",
        len(df),
        output_file,
    )


def main() -> None:
    ensure_directories()
    app()


if __name__ == "__main__":
    main()
