from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import pandas as pd


def read_tsv_gz(path: str, nrows: Optional[int] = None, usecols: Optional[list] = None) -> pd.DataFrame:
    """Read a gzipped TSV into a DataFrame treating '\\N' as NA.

    Parameters:
      - path: gzipped TSV file path
      - nrows: if set, only read this many rows (useful for dev)
      - usecols: list of columns to read (saves memory)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    print(f"Reading {path} (nrows={nrows}) ...")
    return pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
        na_values=["\\N"],
        dtype=str,
        usecols=usecols,
        nrows=nrows,
        low_memory=False,
    )


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Convert obvious numeric-ish columns if present
    for col in ("startYear", "runtimeMinutes"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_and_merge(
    basics_path: str,
    ratings_path: str,
    out_path: str,
    sample_path: Optional[str] = None,
    nrows: Optional[int] = None,
    drop_missing_genres: bool = False,
    title_types: Optional[list] = None,
) -> pd.DataFrame:
    """Load basics + ratings, filter, convert types, and merge.

    NOTE: This function now ONLY returns the merged dataframe; saving is
    performed later after optional enrichment steps so that the final
    output includes directors/actors if requested.
    """
    basics = read_tsv_gz(basics_path, nrows=nrows)
    print(f"Basics raw shape: {basics.shape}")

    # Default: keep 'movie' only (but allow different selection)
    if title_types is None:
        title_types = ["movie"]

    if "titleType" in basics.columns:
        before = len(basics)
        basics = basics[basics["titleType"].isin(title_types)].copy()
        print(f"Filtered titleType {title_types}: {before} -> {len(basics)}")
    else:
        print("Warning: 'titleType' not present in basics; skipping filter.")

    basics = convert_numeric_columns(basics)

    ratings = read_tsv_gz(ratings_path, nrows=nrows)
    print(f"Ratings shape: {ratings.shape}")

    # Ensure tconst present
    if "tconst" not in basics.columns or "tconst" not in ratings.columns:
        raise KeyError("'tconst' must be present in both files to merge.")

    merged = basics.merge(ratings, on="tconst", how="inner", suffixes=("", "_ratings"))
    print(f"Merged shape: {merged.shape}")

    # Keep a useful subset for downstream work (safe: keep extras if you want)
    keep_cols = [
        "tconst",
        "primaryTitle",
        "originalTitle",
        "titleType",
        "isAdult",
        "startYear",
        "runtimeMinutes",
        "genres",
        "averageRating",
        "numVotes",
    ]
    # keep whichever of these exist
    keep_cols = [c for c in keep_cols if c in merged.columns]
    merged = merged[keep_cols]

    # Optional drop of rows missing primaryTitle or genres
    if drop_missing_genres and "genres" in merged.columns:
        before = len(merged)
        merged = merged.dropna(subset=["genres"])
        print(f"Dropped rows missing genres: {before} -> {len(merged)}")

    # Add decade column (safe handling of NaN)
    if "startYear" in merged.columns:
        merged["decade"] = (merged["startYear"] // 10) * 10
        merged["decade"] = merged["decade"].fillna(-1).astype(int)
    else:
        merged["decade"] = -1

    return merged


def enrich_with_crew(merged: pd.DataFrame, crew_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Attach directors and writers columns from title.crew."""
    try:
        crew = read_tsv_gz(crew_path, nrows=nrows)
    except FileNotFoundError:
        print(f"Warning: crew file not found: {crew_path}. Skipping crew enrichment.")
        return merged

    # crew has tconst, directors, writers (both comma-separated nconsts)
    for col in ("directors", "writers"):
        if col not in crew.columns:
            crew[col] = None
    crew = crew[["tconst", "directors", "writers"]].copy()
    enriched = merged.merge(crew, on="tconst", how="left")
    print("Enriched with crew (directors, writers).")
    return enriched


def enrich_with_principals(merged: pd.DataFrame, principals_path: str, nrows: Optional[int] = None, top_k: int = 3) -> pd.DataFrame:
    """Attach top_k actors/actresses per movie as a comma-separated list of nconst."""
    try:
        principals = read_tsv_gz(principals_path, nrows=nrows)
    except FileNotFoundError:
        print(f"Warning: principals file not found: {principals_path}. Skipping principals enrichment.")
        return merged

    # keep relevant columns
    cols = [c for c in ("tconst", "ordering", "nconst", "category") if c in principals.columns]
    principals = principals[cols].copy()

    # Keep only actor/actress (and maybe director if present)
    if "category" in principals.columns:
        mask = principals["category"].isin(["actor", "actress"])
        principals = principals[mask]

    # order by ordering (if exists) then take top_k per tconst
    if "ordering" in principals.columns:
        principals["ordering"] = pd.to_numeric(principals["ordering"], errors="coerce")
        principals = principals.sort_values(["tconst", "ordering"])
    topactors = principals.groupby("tconst")["nconst"].apply(lambda s: ",".join(s.head(top_k))).reset_index()
    topactors = topactors.rename(columns={"nconst": "top_actors"})
    enriched = merged.merge(topactors, on="tconst", how="left")
    print(f"Enriched with principals (top {top_k} actors per movie).")
    return enriched


def enrich_with_tmdb(merged: pd.DataFrame, tmdb_path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Attach revenue and budget from TMDB dataset."""
    if not os.path.exists(tmdb_path):
        print(f"Warning: TMDB file not found: {tmdb_path}. Skipping TMDB enrichment.")
        return merged

    print(f"Reading TMDB data from {tmdb_path} ...")
    # TMDB csv has headers: id, title, ..., revenue, budget, imdb_id, ...
    # We only need imdb_id, revenue, budget
    try:
        tmdb = pd.read_csv(
            tmdb_path,
            usecols=["imdb_id", "revenue", "budget","director"],
            nrows=nrows,
            dtype={"revenue": float, "budget": float, "imdb_id": str}
        )
    except ValueError as e:
        print(f"Error reading TMDB file: {e}. Skipping TMDB enrichment.")
        return merged

    # Drop rows without imdb_id
    tmdb = tmdb.dropna(subset=["imdb_id"])
    
    # Merge on tconst (IMDb) == imdb_id (TMDB)
    # Note: merged dataframe has 'tconst'
    enriched = merged.merge(tmdb, left_on="tconst", right_on="imdb_id", how="left")
    
    # Drop the redundant imdb_id column from TMDB
    enriched = enriched.drop(columns=["imdb_id"])
    
    print(f"Enriched with TMDB data (revenue, budget). Shape: {enriched.shape}")
    return enriched


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Load, merge, preview IMDb datasets (basics + ratings) and optionally enrich")
    p.add_argument("--basics", default="data/raw/title.basics.tsv.gz", help="Path to title.basics.tsv.gz")
    p.add_argument("--ratings", default="data/raw/title.ratings.tsv.gz", help="Path to title.ratings.tsv.gz")
    p.add_argument("--out", default="data/processed/movies_clean.csv", help="Output cleaned CSV path")
    p.add_argument("--sample", default="data/processed/movies_sample.csv", help="Sample CSV path (first 5000 rows)")
    p.add_argument("--nrows", type=int, default=None, help="If set, read only this many rows from each input (dev mode)")
    p.add_argument("--drop-missing-genres", action="store_true", help="Drop rows missing genres")
    p.add_argument("--include-crew", action="store_true", help="Include title.crew enrichment (directors/writers)")
    p.add_argument("--crew-path", default="data/raw/title.crew.tsv.gz", help="Path to title.crew.tsv.gz")
    p.add_argument("--include-principals", action="store_true", help="Include title.principals enrichment (top actors)")
    p.add_argument("--principals-path", default="data/raw/title.principals.tsv.gz", help="Path to title.principals.tsv.gz")
    p.add_argument("--top-actors", type=int, default=3, help="Number of top actors to extract per movie")
    p.add_argument("--include-tmdb", action="store_true", help="Include TMDB enrichment (revenue/budget)")
    p.add_argument("--tmdb-path", default="data/raw/TMDB All Movies Dataset.csv", help="Path to TMDB dataset")
    p.add_argument("--title-types", default="movie", help="Comma-separated list of titleType to keep, e.g. 'movie,tvMovie'")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # ensure processed dir exists for defaults
    ensure_dir_for_file(args.out)
    if args.sample:
        ensure_dir_for_file(args.sample)

    title_types = [s.strip() for s in args.title_types.split(",") if s.strip()]

    merged = load_and_merge(
        basics_path=args.basics,
        ratings_path=args.ratings,
        out_path=args.out,
        sample_path=args.sample,
        nrows=args.nrows,
        drop_missing_genres=args.drop_missing_genres,
        title_types=title_types,
    )

    # Optional enrichment steps (these can be slow for large files)
    if args.include_crew:
        merged = enrich_with_crew(merged, crew_path=args.crew_path, nrows=args.nrows)

    if args.include_principals:
        merged = enrich_with_principals(
            merged,
            principals_path=args.principals_path,
            nrows=args.nrows,
            top_k=args.top_actors,
        )

    if args.include_tmdb:
        merged = enrich_with_tmdb(
            merged,
            tmdb_path=args.tmdb_path,
            nrows=args.nrows,
        )

    # Combine directors and actors into one convenience column if available
    if "directors" in merged.columns or "top_actors" in merged.columns:
        dir_series = merged.get("directors", pd.Series([None]*len(merged)))
        act_series = merged.get("top_actors", pd.Series([None]*len(merged)))
        def _combine(d, a):
            parts = []
            if pd.notna(d) and d:
                parts.append(str(d))
            if pd.notna(a) and a:
                parts.append(str(a))
            return ",".join(parts) if parts else None
        merged["directors_and_actors"] = [ _combine(d, a) for d, a in zip(dir_series, act_series) ]
        print("Added combined column 'directors_and_actors'.")

    # Save final enriched dataset and sample
    ensure_dir_for_file(args.out)
    merged.to_csv(args.out, index=False)
    print(f"Saved final dataset (rows={len(merged)}) with columns: {merged.columns.tolist()} -> {args.out}")
    if args.sample:
        ensure_dir_for_file(args.sample)
        merged.head(5000).to_csv(args.sample, index=False)
        print(f"Saved sample (first 5000 rows) to: {args.sample}")

    print("Final dataframe shape:", merged.shape)
    print("Columns:", merged.columns.tolist())
    print("Done.")


if __name__ == "__main__":
    main(argv=sys.argv[1:] + ["--include-tmdb"] )
