#!/usr/bin/env python3
"""
Build speaker-vowel statistics from track token-level feature files.

Input:
    data/processed/track_token_features/*_track_features.csv

Output:
    data/processed/tables/track_stats_speaker_vowel.csv

One output row = one audio x vowel.

For each numeric feature starting at f1s_value20:
    mean
    variance
    median
    p10
    p25
    p75
    p90
    IQR
    MAD
    p90_minus_p10

Labels:
    y is grouped with i.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


STATS = [
    "mean",
    "variance",
    "median",
    "p10",
    "p25",
    "p75",
    "p90",
    "IQR",
    "MAD",
    "p90_minus_p10",
]


def normalize_colname(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_colname(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in normalized:
            return normalized[key]

    return None


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def normalize_vowel(value) -> str | None:
    if pd.isna(value):
        return None

    v = str(value).strip().lower()

    if v == "":
        return None

    # Here y is grouped with i.
    if v == "y":
        return "i"

    return v


def compute_stats(series: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(series, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()

    if len(x) == 0:
        return {stat: np.nan for stat in STATS}

    mean = float(x.mean())

    if len(x) >= 2:
        variance = float(x.var(ddof=1))
    else:
        variance = np.nan

    median = float(x.median())
    p10 = float(x.quantile(0.10))
    p25 = float(x.quantile(0.25))
    p75 = float(x.quantile(0.75))
    p90 = float(x.quantile(0.90))

    mad = float((x - median).abs().median())

    return {
        "mean": mean,
        "variance": variance,
        "median": median,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "IQR": p75 - p25,
        "MAD": mad,
        "p90_minus_p10": p90 - p10,
    }


def get_audio_name(df: pd.DataFrame, path: Path) -> str:
    audio_col = find_col(df, ["audio"])

    if audio_col is not None:
        values = df[audio_col].dropna()
        if len(values) > 0:
            return str(values.iloc[0])

    return path.stem.replace("_track_features", "")


def get_feature_columns(df: pd.DataFrame, start_column: str) -> list[str]:
    start_col_real = find_col(df, [start_column])

    if start_col_real is None:
        raise ValueError(
            f"Could not find start column '{start_column}'. "
            f"Available columns are: {list(df.columns)}"
        )

    start_idx = list(df.columns).index(start_col_real)

    candidate_cols = list(df.columns[start_idx:])

    numeric_feature_cols = []

    for col in candidate_cols:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() > 0:
            numeric_feature_cols.append(col)

    return numeric_feature_cols


def build_stats_for_file(
    path: Path,
    vowels: list[str],
    start_column: str,
) -> pd.DataFrame:
    df = read_csv_safely(path)

    label_col = find_col(df, ["label"])
    if label_col is None:
        raise ValueError(f"No label column found in {path}")

    audio_name = get_audio_name(df, path)

    df = df.copy()
    df["vowel_normalized"] = df[label_col].apply(normalize_vowel)

    feature_cols = get_feature_columns(df, start_column)

    rows = []

    for vowel in vowels:
        subset = df[df["vowel_normalized"] == vowel]

        row = {
            "audio": audio_name,
            "source_file": path.name,
            "vowel": vowel,
            "n_tokens": int(len(subset)),
        }

        for feature in feature_cols:
            stats = compute_stats(subset[feature])

            for stat_name, stat_value in stats.items():
                row[f"{feature}_{stat_name}"] = stat_value

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build audio-vowel statistics from track token feature files."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_track_features.csv files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path, preferably inside data/processed/tables.",
    )

    parser.add_argument(
        "--start-column",
        default="f1s_value20",
        help="First feature column to analyze. Default: f1s_value20.",
    )

    parser.add_argument(
        "--vowels",
        nargs="+",
        default=["a", "e", "i", "o", "u"],
        help="Vowels to include after normalization. Default: a e i o u.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    files = sorted(input_dir.glob("*_track_features.csv"))

    if not files:
        raise FileNotFoundError(f"No *_track_features.csv files found in {input_dir}")

    all_rows = []

    for path in files:
        print(f"Processing {path.name}")
        stats_df = build_stats_for_file(
            path=path,
            vowels=args.vowels,
            start_column=args.start_column,
        )
        all_rows.append(stats_df)

    output = pd.concat(all_rows, ignore_index=True)

    # Keep main columns first.
    first_cols = ["audio", "source_file", "vowel", "n_tokens"]
    ordered = [c for c in first_cols if c in output.columns]
    remaining = [c for c in output.columns if c not in ordered]
    output = output[ordered + remaining]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Done. Output written to: {output_path}")
    print(f"Rows: {len(output)}")
    print(f"Columns: {len(output.columns)}")


if __name__ == "__main__":
    main()