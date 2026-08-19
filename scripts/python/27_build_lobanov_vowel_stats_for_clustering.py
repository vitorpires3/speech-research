#!/usr/bin/env python3
"""
Build speaker-level vowel median table after Lobanov normalization.

Input:
- data/processed/new_fave_points/*.csv

Output:
- data/processed/tables/vowel_stats_speaker_vowel_lobanov.csv

The output deliberately uses the columns f1_median and f2_median because
script 10 expects these names. But the values are Lobanov-normalized z-scores,
not Hz.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


VOWEL_COL_CANDIDATES = ["vowel", "phone", "label", "vowel_label", "phone_label"]
F1_COL_CANDIDATES = ["F1", "f1", "F1_Hz", "f1_hz"]
F2_COL_CANDIDATES = ["F2", "f2", "F2_Hz", "f2_hz"]
F3_COL_CANDIDATES = ["F3", "f3", "F3_Hz", "f3_hz"]


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_col(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    lookup = {norm_col(c): c for c in df.columns}

    for cand in candidates:
        key = norm_col(cand)
        if key in lookup:
            return lookup[key]

    raise ValueError(
        f"Could not find {role} column. "
        f"Candidates: {candidates}. "
        f"Available: {list(df.columns)}"
    )


def audio_id_from_file(path: Path) -> str:
    stem = path.stem

    suffixes = [
        "_new_fave_points",
        "_fasttrack_points",
        "_points",
        ".new_fave_points",
        ".points",
    ]

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True

    stem = re.sub(r"\.v\d+$", "", stem).strip()
    return stem


def parse_label_map(items: list[str]) -> dict[str, str]:
    out = {}

    for item in items:
        if ":" not in item:
            continue

        a, b = item.split(":", 1)
        out[a.strip().lower()] = b.strip().lower()

    return out


def clean_vowel(value, vowels: set[str], label_map: dict[str, str]) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip().lower()

    if s in label_map:
        s = label_map[s]

    s = re.sub(r"[0-9ˈˌː:\.\s_-]+", "", s)

    if s in label_map:
        s = label_map[s]

    if s in vowels:
        return s

    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--label-map", nargs="*", default=["y:i"])
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    vowels = set(args.vowels)
    label_map = parse_label_map(args.label_map)

    frames = []

    for path in sorted(input_dir.glob("*.csv")):
        df = pd.read_csv(path)

        if df.empty:
            continue

        vowel_col = find_col(df, VOWEL_COL_CANDIDATES, "vowel")
        f1_col = find_col(df, F1_COL_CANDIDATES, "F1")
        f2_col = find_col(df, F2_COL_CANDIDATES, "F2")

        # F3 is optional here; clustering script 10 uses F1/F2.
        f3_col = None
        try:
            f3_col = find_col(df, F3_COL_CANDIDATES, "F3")
        except Exception:
            pass

        # Important: create dataframe with the same index as the source file.
        # Otherwise scalar assignment to an empty dataframe can produce NaN audio IDs.
        sub = pd.DataFrame(index=df.index)
        sub["audio"] = audio_id_from_file(path)
        sub["vowel_raw"] = df[vowel_col]
        sub["vowel"] = df[vowel_col].apply(lambda x: clean_vowel(x, vowels, label_map))
        sub["f1"] = pd.to_numeric(df[f1_col], errors="coerce")
        sub["f2"] = pd.to_numeric(df[f2_col], errors="coerce")

        if f3_col is not None:
            sub["f3"] = pd.to_numeric(df[f3_col], errors="coerce")
        else:
            sub["f3"] = np.nan

        sub = sub.dropna(subset=["vowel", "f1", "f2"])
        sub = sub[sub["vowel"].isin(vowels)]
        sub = sub[(sub["f1"] > 0) & (sub["f2"] > 0)]

        frames.append(sub)

    if not frames:
        raise RuntimeError("No valid point data found.")

    tokens = pd.concat(frames, ignore_index=True)

    # Lobanov per speaker/audio.
    for formant in ["f1", "f2", "f3"]:
        if tokens[formant].notna().sum() == 0:
            continue

        mean = tokens.groupby("audio")[formant].transform("mean")
        std = tokens.groupby("audio")[formant].transform("std")

        tokens[f"{formant}_lobanov"] = (tokens[formant] - mean) / std.replace(0, np.nan)

    # Robust named aggregation.
    # This avoids pandas MultiIndex column-name issues.
    agg_items = {
        "n_tokens": ("vowel", "size"),

        # These names are deliberately f1_median/f2_median because
        # script 10 expects these names. Values are Lobanov z-scores.
        "f1_median": ("f1_lobanov", "median"),
        "f2_median": ("f2_lobanov", "median"),
        "f1_mean": ("f1_lobanov", "mean"),
        "f2_mean": ("f2_lobanov", "mean"),
        "f1_std": ("f1_lobanov", "std"),
        "f2_std": ("f2_lobanov", "std"),

        # Keep original Hz values for checking/comparison.
        "f1_hz_median": ("f1", "median"),
        "f2_hz_median": ("f2", "median"),
        "f1_hz_mean": ("f1", "mean"),
        "f2_hz_mean": ("f2", "mean"),
        "f1_hz_std": ("f1", "std"),
        "f2_hz_std": ("f2", "std"),
    }

    if "f3_lobanov" in tokens.columns and tokens["f3_lobanov"].notna().any():
        agg_items.update({
            "f3_lobanov_median": ("f3_lobanov", "median"),
            "f3_lobanov_mean": ("f3_lobanov", "mean"),
            "f3_lobanov_std": ("f3_lobanov", "std"),
            "f3_hz_median": ("f3", "median"),
            "f3_hz_mean": ("f3", "mean"),
            "f3_hz_std": ("f3", "std"),
        })

    out = (
        tokens
        .groupby(["audio", "vowel"], as_index=False)
        .agg(**agg_items)
    )

    out["normalization"] = "lobanov_by_audio"
    out = out.sort_values(["audio", "vowel"]).reset_index(drop=True)

    out.to_csv(output, index=False)

    print("")
    print("Done.")
    print("Output:", output)
    print("Rows:", len(out))
    print("Audios:", out["audio"].nunique())
    print("")
    print("Rows by vowel:")
    print(out["vowel"].value_counts().reindex(args.vowels))


if __name__ == "__main__":
    main()
