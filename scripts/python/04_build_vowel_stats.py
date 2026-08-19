#!/usr/bin/env python3
from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd


PARAMS = ["dur", "f1", "f2", "f3", "b1", "b2", "b3", "f0"]

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


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Finds a column ignoring case.
    """
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def normalize_vowel(label) -> str:
    """
    Normalizes vowel labels for analysis.
    In this project, new-fave separates y and i, but they are grouped as i.
    """
    if pd.isna(label):
        return np.nan

    vowel = str(label).strip().lower()

    if vowel == "y":
        return "i"

    return vowel


def audio_id_from_value(value) -> str:
    """
    Creates a clean audio id from file_name when available.
    """
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    return Path(value).stem


def compute_stats(series: pd.Series) -> dict[str, float]:
    """
    Computes the requested statistics, ignoring NaN values.

    variance uses sample variance, ddof=1.
    MAD is the raw median absolute deviation:
        median(abs(x - median(x)))
    """
    x = pd.to_numeric(series, errors="coerce").dropna()

    if len(x) == 0:
        return {stat: np.nan for stat in STATS}

    p10 = x.quantile(0.10)
    p25 = x.quantile(0.25)
    p75 = x.quantile(0.75)
    p90 = x.quantile(0.90)
    median = x.median()

    return {
        "mean": x.mean(),
        "variance": x.var(ddof=1) if len(x) > 1 else np.nan,
        "median": median,
        "p10": p10,
        "p25": p25,
        "p75": p75,
        "p90": p90,
        "IQR": p75 - p25,
        "MAD": (x - median).abs().median(),
        "p90_minus_p10": p90 - p10,
    }


def build_clean_dataframe(input_dir: Path) -> pd.DataFrame:
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    all_rows = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        label_col = find_column(df, ["label", "vowel", "phone"])
        file_col = find_column(df, ["file_name", "filename", "audio", "audio_id"])

        if label_col is None:
            raise ValueError(f"Could not find vowel/label column in {csv_file}")

        if file_col is not None:
            df["audio"] = df[file_col].apply(audio_id_from_value)
        else:
            df["audio"] = csv_file.stem

        df["vowel"] = df[label_col].apply(normalize_vowel)

        # Standardize acoustic columns.
        column_map = {}

        source_columns = {
            "dur": ["dur", "duration"],
            "f1": ["F1", "f1"],
            "f2": ["F2", "f2"],
            "f3": ["F3", "f3"],
            "f0": ["f0", "F0", "pitch"],
        }

        for target, candidates in source_columns.items():
            col = find_column(df, candidates)
            if col is None:
                warnings.warn(f"Column for {target} not found in {csv_file}; filling with NaN.")
                df[target] = np.nan
            else:
                df[target] = pd.to_numeric(df[col], errors="coerce")

        # Bandwidths:
        # Prefer B1_Hz/B2_Hz/B3_Hz.
        # If only B1/B2/B3 exist, detect whether they look log-scaled.
        for b in ["b1", "b2", "b3"]:
            hz_col = find_column(df, [f"{b.upper()}_Hz", f"{b}_Hz"])
            raw_col = find_column(df, [b.upper(), b])

            if hz_col is not None:
                df[b] = pd.to_numeric(df[hz_col], errors="coerce")

            elif raw_col is not None:
                raw = pd.to_numeric(df[raw_col], errors="coerce")

                # In our new-fave points, B1/B2/B3 may be natural-log bandwidths.
                # If values are small, assume log-scale and convert with exp().
                median_raw = raw.dropna().median()

                if pd.notna(median_raw) and median_raw < 20:
                    warnings.warn(
                        f"{csv_file.name}: {raw_col} looks log-scaled; converting {b} = exp({raw_col})."
                    )
                    df[b] = np.exp(raw)
                else:
                    warnings.warn(
                        f"{csv_file.name}: using {raw_col} directly as Hz for {b}."
                    )
                    df[b] = raw

            else:
                warnings.warn(f"Bandwidth column for {b} not found in {csv_file}; filling with NaN.")
                df[b] = np.nan

        keep_cols = ["audio", "vowel"] + PARAMS
        all_rows.append(df[keep_cols])

    clean = pd.concat(all_rows, ignore_index=True)

    # Remove empty vowel labels.
    clean = clean.dropna(subset=["audio", "vowel"])

    return clean


def build_stats_table(clean: pd.DataFrame, vowels: list[str] | None) -> pd.DataFrame:
    if vowels is not None and len(vowels) > 0:
        vowels = [normalize_vowel(v) for v in vowels]
        clean = clean[clean["vowel"].isin(vowels)].copy()
    else:
        vowels = sorted(clean["vowel"].dropna().unique().tolist())

    audio_ids = sorted(clean["audio"].dropna().unique().tolist())

    rows = []

    for audio in audio_ids:
        for vowel in vowels:
            group = clean[(clean["audio"] == audio) & (clean["vowel"] == vowel)]

            row = {
                "audio": audio,
                "vowel": vowel,
                "n_tokens": int(len(group)),
            }

            for param in PARAMS:
                stats = compute_stats(group[param]) if len(group) > 0 else {
                    stat: np.nan for stat in STATS
                }

                for stat_name in STATS:
                    row[f"{param}_{stat_name}"] = stats[stat_name]

            rows.append(row)

    result = pd.DataFrame(rows)

    # Explicit column order.
    ordered_cols = ["audio", "vowel", "n_tokens"]
    for param in PARAMS:
        for stat in STATS:
            ordered_cols.append(f"{param}_{stat}")

    return result[ordered_cols]


def main():
    parser = argparse.ArgumentParser(
        description="Build speaker-vowel statistical summaries from new-fave points CSV files."
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing new-fave points CSV files.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file.",
    )

    parser.add_argument(
        "--vowels",
        nargs="*",
        default=None,
        help="Vowels to include. Example: --vowels a e i o u. The label y is automatically grouped with i.",
    )

    args = parser.parse_args()

    clean = build_clean_dataframe(args.input_dir)
    stats = build_stats_table(clean, args.vowels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(args.output, index=False)

    print(f"Saved: {args.output}")
    print(f"Rows: {len(stats)}")
    print(f"Columns: {len(stats.columns)}")
    print()
    print("Vowels in output:")
    print(stats["vowel"].value_counts().sort_index())


if __name__ == "__main__":
    main()