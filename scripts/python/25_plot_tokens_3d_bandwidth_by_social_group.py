#!/usr/bin/env python3
"""
3D token-level B1-B2-B3 plots by social group.

For each vowel and each selected social variable, this script creates an
interactive Plotly 3D scatterplot using formant bandwidths.

Default social variables:
- CountryRes
- Sex
- Income_group

Important:
new-FAVE/FastTrack bandwidths may be in Hz or natural-log scale depending
on the exported columns. This script tries to detect that automatically.

Outputs:
- token_points_b1b2b3_with_social_full.csv
- bandwidth_conversion_report.csv
- token_counts_by_audio_vowel_social.csv
- token_counts_by_vowel_social_group.csv
- plot_data_by_<social_var>.csv
- one HTML 3D plot per vowel/social variable
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


VOWEL_COL_CANDIDATES = [
    "vowel", "phone", "label", "vowel_label", "phone_label",
    "ipa", "arpa", "vclass", "vowel_class"
]

B1_COL_CANDIDATES = ["B1_Hz", "b1_hz", "B1Hz", "b1hz", "B1", "b1"]
B2_COL_CANDIDATES = ["B2_Hz", "b2_hz", "B2Hz", "b2hz", "B2", "b2"]
B3_COL_CANDIDATES = ["B3_Hz", "b3_hz", "B3Hz", "b3hz", "B3", "b3"]

TOKEN_ID_CANDIDATES = ["id", "token_id", "vowel_id", "phone_id", "measurement_id"]

EXTRA_COL_CANDIDATES = [
    "beg", "end", "start", "stop", "time", "duration", "dur",
    "word", "speaker", "file", "filename", "file_name"
]


def require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "Plotly is not installed. Install it with:\n\n"
            "pip install plotly\n"
        ) from exc
    return go


def normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, candidates: list[str], explicit: str | None, role: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Column '{explicit}' was requested for {role}, but it was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    normalized_to_original = {normalize_col_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]

    raise ValueError(
        f"Could not automatically detect column for {role}. "
        f"Available columns: {list(df.columns)}."
    )


def find_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_to_original = {normalize_col_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]

    return None


def parse_label_map(items: list[str]) -> dict[str, str]:
    mapping = {}

    for item in items:
        if ":" not in item:
            continue

        src, dst = item.split(":", 1)
        mapping[src.strip().lower()] = dst.strip().lower()

    return mapping


def clean_vowel_label(value, vowels: set[str], label_map: dict[str, str]) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip().lower()

    if s in label_map:
        s = label_map[s]

    s2 = re.sub(r"[0-9ˈˌː:\.\s_-]+", "", s)

    if s2 in label_map:
        s2 = label_map[s2]

    if s2 in vowels:
        return s2

    return None


def audio_id_from_file(path: Path) -> str:
    stem = path.stem

    for suffix in [
        "_new_fave_points",
        "_fasttrack_points",
        "_points",
        ".new_fave_points",
        ".points",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def looks_like_hz_column(colname: str) -> bool:
    name = normalize_col_name(colname)
    return "hz" in name


def convert_bandwidth_to_hz(series: pd.Series, colname: str, mode: str) -> tuple[pd.Series, str]:
    """
    Convert bandwidth to Hz if needed.

    mode:
    - auto: if column name contains Hz, keep; otherwise if median < 20, exp()
    - hz: keep as Hz
    - log: exp()
    """
    x = pd.to_numeric(series, errors="coerce")

    positive = x[x > 0]
    median_value = float(positive.median()) if len(positive) else np.nan

    if mode == "hz":
        return x, "kept_as_hz_forced"

    if mode == "log":
        return np.exp(x), "converted_from_log_forced"

    if looks_like_hz_column(colname):
        return x, "kept_as_hz_column_name"

    # Heuristic:
    # Natural-log bandwidth values usually sit around 3 to 7.
    # Bandwidths in Hz usually sit much higher.
    if np.isfinite(median_value) and median_value < 20:
        return np.exp(x), "converted_from_log_auto"

    return x, "kept_as_hz_auto"


def load_points(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(args.points_dir)
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    vowels = set(args.vowels)
    label_map = parse_label_map(args.label_map)

    frames = []
    report_rows = []

    for path in files:
        df = pd.read_csv(path)

        if df.empty:
            continue

        vowel_col = find_column(df, VOWEL_COL_CANDIDATES, args.vowel_col, "vowel")
        b1_col = find_column(df, B1_COL_CANDIDATES, args.b1_col, "b1")
        b2_col = find_column(df, B2_COL_CANDIDATES, args.b2_col, "b2")
        b3_col = find_column(df, B3_COL_CANDIDATES, args.b3_col, "b3")
        token_col = find_optional_column(df, TOKEN_ID_CANDIDATES)

        keep_cols = [vowel_col, b1_col, b2_col, b3_col]

        if token_col is not None:
            keep_cols.append(token_col)

        for extra in EXTRA_COL_CANDIDATES:
            col = find_optional_column(df, [extra])
            if col is not None and col not in keep_cols:
                keep_cols.append(col)

        sub = df[keep_cols].copy()

        sub["audio"] = audio_id_from_file(path)
        sub["source_file"] = path.name
        sub["vowel_raw"] = sub[vowel_col]
        sub["vowel"] = sub[vowel_col].apply(
            lambda x: clean_vowel_label(x, vowels=vowels, label_map=label_map)
        )

        sub["b1_raw"] = pd.to_numeric(sub[b1_col], errors="coerce")
        sub["b2_raw"] = pd.to_numeric(sub[b2_col], errors="coerce")
        sub["b3_raw"] = pd.to_numeric(sub[b3_col], errors="coerce")

        sub["b1"], b1_action = convert_bandwidth_to_hz(sub[b1_col], b1_col, args.bandwidth_mode)
        sub["b2"], b2_action = convert_bandwidth_to_hz(sub[b2_col], b2_col, args.bandwidth_mode)
        sub["b3"], b3_action = convert_bandwidth_to_hz(sub[b3_col], b3_col, args.bandwidth_mode)

        report_rows.append({
            "source_file": path.name,
            "audio": audio_id_from_file(path),
            "b1_column": b1_col,
            "b2_column": b2_col,
            "b3_column": b3_col,
            "b1_action": b1_action,
            "b2_action": b2_action,
            "b3_action": b3_action,
            "b1_raw_median": float(sub["b1_raw"].dropna().median()) if sub["b1_raw"].notna().any() else np.nan,
            "b2_raw_median": float(sub["b2_raw"].dropna().median()) if sub["b2_raw"].notna().any() else np.nan,
            "b3_raw_median": float(sub["b3_raw"].dropna().median()) if sub["b3_raw"].notna().any() else np.nan,
            "b1_hz_median": float(sub["b1"].dropna().median()) if sub["b1"].notna().any() else np.nan,
            "b2_hz_median": float(sub["b2"].dropna().median()) if sub["b2"].notna().any() else np.nan,
            "b3_hz_median": float(sub["b3"].dropna().median()) if sub["b3"].notna().any() else np.nan,
        })

        if token_col is not None:
            sub["token_id"] = sub[token_col]
        else:
            sub["token_id"] = np.arange(len(sub))

        sub = sub.dropna(subset=["vowel", "b1", "b2", "b3"])
        sub = sub[sub["vowel"].isin(args.vowels)]
        sub = sub[(sub["b1"] > 0) & (sub["b2"] > 0) & (sub["b3"] > 0)]

        # Optional safety cap to avoid insane bandwidth artifacts in plots.
        if args.max_bandwidth_hz and args.max_bandwidth_hz > 0:
            sub = sub[
                (sub["b1"] <= args.max_bandwidth_hz)
                & (sub["b2"] <= args.max_bandwidth_hz)
                & (sub["b3"] <= args.max_bandwidth_hz)
            ]

        frames.append(sub)

    if not frames:
        raise RuntimeError("No valid B1/B2/B3 vowel points found.")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["audio", "vowel"]).reset_index(drop=True)

    report = pd.DataFrame(report_rows)

    return out, report


def load_metadata(args) -> pd.DataFrame:
    meta = pd.read_csv(args.metadata)

    if args.metadata_key not in meta.columns:
        raise ValueError(
            f"Metadata key '{args.metadata_key}' not found. "
            f"Available columns: {list(meta.columns)}"
        )

    missing_social = [col for col in args.social_vars if col not in meta.columns]

    if missing_social:
        raise ValueError(
            f"Missing social columns in metadata: {missing_social}\n"
            f"Available columns: {list(meta.columns)}"
        )

    keep = [args.metadata_key] + args.social_vars
    meta = meta[keep].copy()
    meta = meta.rename(columns={args.metadata_key: "audio"})

    for col in args.social_vars:
        meta[col] = meta[col].astype("string").str.strip()
        meta[col] = meta[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return meta


def merge_points_metadata(points: pd.DataFrame, metadata: pd.DataFrame, social_vars: list[str]) -> pd.DataFrame:
    merged = points.merge(metadata, on="audio", how="left", indicator=True)

    match_report = (
        merged[["audio", "_merge"]]
        .drop_duplicates()
        .groupby("_merge")
        .size()
        .reset_index(name="n_audios")
    )

    print("")
    print("Metadata merge report:")
    print(match_report.to_string(index=False))

    unmatched = (
        merged.loc[merged["_merge"] != "both", "audio"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if unmatched:
        print("")
        print("Unmatched audios:")
        for audio in unmatched:
            print(" -", audio)

    merged = merged.drop(columns=["_merge"])

    for col in social_vars:
        merged[col] = merged[col].fillna("Unknown")

    return merged


def make_color_map(categories: list[str]) -> dict[str, str]:
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
        "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
        "#98df8a", "#ff9896", "#c5b0d5", "#c49c94",
    ]

    categories = [str(c) for c in categories]
    return {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}


def downsample_for_plot(df: pd.DataFrame, args, social_var: str | None = None) -> pd.DataFrame:
    if args.plot_every is not None and args.plot_every > 1:
        df = df.iloc[::args.plot_every].copy()

    if args.max_points_per_group is None or args.max_points_per_group <= 0:
        return df.copy()

    if social_var is None:
        group_cols = ["vowel"]
    else:
        group_cols = ["vowel", social_var]

    chunks = []

    for _, group in df.groupby(group_cols, sort=False):
        n = min(len(group), args.max_points_per_group)

        if n > 0:
            chunks.append(group.sample(n=n, random_state=args.random_state))

    if not chunks:
        return df.iloc[0:0].copy()

    return pd.concat(chunks, ignore_index=True)


def make_hover_text(df: pd.DataFrame, social_var: str) -> list[str]:
    texts = []

    optional = [c for c in ["word", "time", "dur", "token_id"] if c in df.columns]

    for _, row in df.iterrows():
        lines = [
            f"audio: {row['audio']}",
            f"vowel: /{row['vowel']}/",
            f"{social_var}: {row[social_var]}",
            f"B1: {row['b1']:.1f} Hz",
            f"B2: {row['b2']:.1f} Hz",
            f"B3: {row['b3']:.1f} Hz",
        ]

        for col in optional:
            value = row.get(col)
            if pd.notna(value):
                lines.append(f"{col}: {value}")

        texts.append("<br>".join(lines))

    return texts


def update_scene(fig, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1100,
        height=850,
        legend_title_text="Groupe social",
        scene=dict(
            xaxis=dict(title="B1 (Hz)"),
            yaxis=dict(title="B2 (Hz)"),
            zaxis=dict(title="B3 (Hz)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )


def plot_vowel_by_social(df: pd.DataFrame, vowel: str, social_var: str, output_dir: Path, args):
    go = require_plotly()

    d0 = df[df["vowel"] == vowel].copy()

    if d0.empty:
        return

    d0 = downsample_for_plot(d0, args, social_var=social_var)

    categories = sorted(d0[social_var].dropna().astype(str).unique())
    color_map = make_color_map(categories)

    fig = go.Figure()

    for category in categories:
        d = d0[d0[social_var].astype(str) == category].copy()

        if d.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=d["b1"],
                y=d["b2"],
                z=d["b3"],
                mode="markers",
                name=str(category),
                marker=dict(
                    size=args.point_size,
                    color=color_map[str(category)],
                    opacity=args.alpha,
                ),
                text=make_hover_text(d, social_var),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    update_scene(fig, f"Voyelle /{vowel}/ — B1-B2-B3 par {social_var}")

    out = output_dir / f"vowel_{vowel}_by_{social_var}_b1b2b3_3d.html"
    fig.write_html(out, include_plotlyjs="cdn", full_html=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--points-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--metadata-key", default="Time.code.and.speaker")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--social-vars", nargs="+", default=["CountryRes", "Sex", "Income_group"])
    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--label-map", nargs="*", default=["y:i"])

    parser.add_argument("--vowel-col", default=None)
    parser.add_argument("--b1-col", default=None)
    parser.add_argument("--b2-col", default=None)
    parser.add_argument("--b3-col", default=None)

    parser.add_argument(
        "--bandwidth-mode",
        choices=["auto", "hz", "log"],
        default="auto",
        help="How to interpret B1/B2/B3 columns.",
    )

    parser.add_argument(
        "--max-bandwidth-hz",
        type=float,
        default=3000,
        help="Remove bandwidths above this value after conversion. Use 0 to disable.",
    )

    parser.add_argument(
        "--plot-every",
        type=int,
        default=None,
        help="Optional plotting-only systematic subsampling: keep one row every N rows.",
    )

    parser.add_argument(
        "--max-points-per-group",
        type=int,
        default=8000,
        help="Plotting-only cap per vowel/social group. Use 0 for no cap.",
    )

    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    points, conversion_report = load_points(args)
    metadata = load_metadata(args)

    merged = merge_points_metadata(points, metadata, args.social_vars)

    merged.to_csv(output_dir / "token_points_b1b2b3_with_social_full.csv", index=False)
    conversion_report.to_csv(output_dir / "bandwidth_conversion_report.csv", index=False)

    counts = (
        merged
        .groupby(["vowel", "audio", *args.social_vars])
        .size()
        .reset_index(name="n_tokens")
        .sort_values(["vowel", "audio"])
    )
    counts.to_csv(output_dir / "token_counts_by_audio_vowel_social.csv", index=False)

    social_counts = []

    for var in args.social_vars:
        tmp = (
            merged
            .groupby(["vowel", var])
            .size()
            .reset_index(name="n_tokens")
            .rename(columns={var: "social_group"})
        )
        tmp["social_variable"] = var
        social_counts.append(tmp)

    social_counts_df = pd.concat(social_counts, ignore_index=True)
    social_counts_df = social_counts_df[
        ["social_variable", "vowel", "social_group", "n_tokens"]
    ].sort_values(["social_variable", "vowel", "social_group"])

    social_counts_df.to_csv(output_dir / "token_counts_by_vowel_social_group.csv", index=False)

    for var in args.social_vars:
        plot_df = downsample_for_plot(merged, args, social_var=var)
        plot_df.to_csv(output_dir / f"plot_data_by_{var}.csv", index=False)

    for var in args.social_vars:
        for vowel in args.vowels:
            plot_vowel_by_social(merged, vowel, var, output_dir, args)

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print("")
    print("Full token rows:", len(merged))
    print("Audios:", merged["audio"].nunique())
    print("")
    print("Token counts by vowel:")
    print(merged["vowel"].value_counts().reindex(args.vowels))
    print("")
    print("Bandwidth conversion summary:")
    print(conversion_report[["b1_action", "b2_action", "b3_action"]].drop_duplicates().to_string(index=False))
    print("")
    print("Social variables:")
    for var in args.social_vars:
        print("")
        print(var)
        print(merged[var].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
