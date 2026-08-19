#!/usr/bin/env python3
"""
3D plots of track-based vowel movement features by social group.

Uses track token feature files generated from new-FAVE/FastTrack tracks.

Main idea:
- For each token, use formant values at 20% and 80% of the vowel.
- Compute:
    delta_f1 = f1s_value80 - f1s_value20
    delta_f2 = f2s_value80 - f2s_value20
    delta_f3 = f3s_value80 - f3s_value20

Outputs:
- full merged token-level CSV
- summary by vowel/social group
- token-level 3D delta plots
- median 20% -> 80% vector plots by social group

Default social delta_f3 = f3s_value80 - f3s_value20

Outputs:
- full merged token-level CSV
- summary by vowel/social group
- token-level 3D delta plots
- median 20% -> 80% vector plots by social group

Default social variables:
- CountryRes
- Sex
- Income_group
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

AUDIO_COL_CANDIDATES = [
    "audio", "_audio_id", "file", "file_name", "source_file", "_source_file"
]

TOKEN_ID_CANDIDATES = [
    "id", "token_id", "vowel_id", "phone_id", "measurement_id"
]


FEATURE_CANDIDATES = {
    "f1_20": [
        "f1s_value20", "f1_s_value20", "F1_s_value20",
        "f1_value20", "F1_value20", "f1s_p20", "f1_p20",
        "f1s_20", "f1_20"
    ],
    "f2_20": [
        "f2s_value20", "f2_s_value20", "F2_s_value20",
        "f2_value20", "F2_value20", "f2s_p20", "f2_p20",
        "f2s_20", "f2_20"
    ],
    "f3_20": [
        "f3s_value20", "f3_s_value20", "F3_s_value20",
        "f3_value20", "F3_value20", "f3s_p20", "f3_p20",
        "f3s_20", "f3_20"
    ],
    "f1_80": [
        "f1s_value80", "f1_s_value80", "F1_s_value80",
        "f1_value80", "F1_value80", "f1s_p80", "f1_p80",
        "f1s_80", "f1_80"
    ],
    "f2_80": [
        "f2s_value80", "f2_s_value80", "F2_s_value80",
        "f2_value80", "F2_value80", "f2s_p80", "f2_p80",
        "f2s_80", "f2_80"
    ],
    "f3_80": [
        "f3s_value80", "f3_s_value80", "F3_s_value80",
        "f3_value80", "F3_value80", "f3s_p80", "f3_p80",
        "f3s_80", "f3_80"
    ],
    "delta_f1": [
        "f1s_delta20_80", "f1_s_delta20_80", "F1_s_delta20_80",
        "f1_delta20_80", "F1_delta20_80", "delta_f1", "f1_delta"
    ],
    "delta_f2": [
        "f2s_delta20_80", "f2_s_delta20_80", "F2_s_delta20_80",
        "f2_delta20_80", "F2_delta20_80", "delta_f2", "f2_delta"
    ],
    "delta_f3": [
        "f3s_delta20_80", "f3_s_delta20_80", "F3_s_delta20_80",
        "f3_delta20_80", "F3_delta20_80", "delta_f3", "f3_delta"
    ],
}


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


def find_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_to_original = {normalize_col_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]

    return None


def find_required_column(df: pd.DataFrame, candidates: list[str], role: str) -> str:
    col = find_optional_column(df, candidates)

    if col is None:
        raise ValueError(
            f"Could not find column for {role}.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return col


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
        "_track_features",
        "_tracks_track_features",
        "_new_fave_tracks_track_features",
        "_new_fave_tracks",
        "_tracks",
        "_features",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def clean_audio_id(value) -> str | None:
    """
    Normalize audio identifiers so track feature files can be matched
    against social_metadata.csv.

    Examples:
    - 061ea7d0-Audio_SP16.v0 -> 061ea7d0-Audio_SP16
    - 061ea7d0-Audio_SP16.v0_track_features -> 061ea7d0-Audio_SP16
    -  34ad326e-Audio_SP7.v0 -> 34ad326e-Audio_SP7
    """
    if pd.isna(value):
        return None

    s = str(value).strip()

    if not s:
        return None

    s = Path(s).name.strip()

    # Strip common file extensions.
    for ext in [".csv", ".txt", ".tsv"]:
        if s.lower().endswith(ext):
            s = s[: -len(ext)].strip()

    suffixes = [
        "_new_fave_tracks_track_features",
        "_tracks_track_features",
        "_track_features",
        "_new_fave_tracks",
        "_fasttrack_tracks",
        "_tracks",
        "_features",
        "_points",
    ]

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                changed = True

    # Remove version suffix used in track files: .v0, .v1, etc.
    s = re.sub(r"\.v\d+$", "", s).strip()

    return s



def load_track_features(args) -> pd.DataFrame:
    input_dir = Path(args.track_features_dir)
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    vowels = set(args.vowels)
    label_map = parse_label_map(args.label_map)

    frames = []
    diagnostics = []

    for path in files:
        df = pd.read_csv(path)

        if df.empty:
            continue

        vowel_col = find_required_column(df, VOWEL_COL_CANDIDATES, "vowel")
        audio_col = find_optional_column(df, AUDIO_COL_CANDIDATES)
        token_col = find_optional_column(df, TOKEN_ID_CANDIDATES)

        found = {}
        for key, candidates in FEATURE_CANDIDATES.items():
            found[key] = find_optional_column(df, candidates)

        need_values = ["f1_20", "f2_20", "f3_20", "f1_80", "f2_80", "f3_80"]
        has_values = all(found[k] is not None for k in need_values)

        has_deltas = all(found[k] is not None for k in ["delta_f1", "delta_f2", "delta_f3"])

        if not has_values and not has_deltas:
            diagnostics.append({
                "source_file": path.name,
                "status": "skipped_missing_value20_80_and_delta_columns",
                **found,
            })
            continue

        keep_cols = [vowel_col]

        if audio_col is not None:
            keep_cols.append(audio_col)

        if token_col is not None:
            keep_cols.append(token_col)

        for col in found.values():
            if col is not None and col not in keep_cols:
                keep_cols.append(col)

        sub = df[keep_cols].copy()

        if audio_col is not None:
            sub["audio"] = sub[audio_col].apply(clean_audio_id)
        else:
            sub["audio"] = audio_id_from_file(path)

        sub["source_file"] = path.name
        sub["vowel_raw"] = sub[vowel_col]
        sub["vowel"] = sub[vowel_col].apply(
            lambda x: clean_vowel_label(x, vowels=vowels, label_map=label_map)
        )

        if token_col is not None:
            sub["token_id"] = sub[token_col]
        else:
            sub["token_id"] = np.arange(len(sub))

        for key, col in found.items():
            if col is not None:
                sub[key] = pd.to_numeric(sub[col], errors="coerce")
            else:
                sub[key] = np.nan

        if has_values:
            sub["delta_f1"] = sub["f1_80"] - sub["f1_20"]
            sub["delta_f2"] = sub["f2_80"] - sub["f2_20"]
            sub["delta_f3"] = sub["f3_80"] - sub["f3_20"]

        else:
            # If only deltas exist, we can still plot delta space,
            # but vector plots will not be possible.
            sub["f1_20"] = np.nan
            sub["f2_20"] = np.nan
            sub["f3_20"] = np.nan
            sub["f1_80"] = np.nan
            sub["f2_80"] = np.nan
            sub["f3_80"] = np.nan

        sub = sub.dropna(subset=["vowel", "delta_f1", "delta_f2", "delta_f3"])
        sub = sub[sub["vowel"].isin(args.vowels)]

        # Optional artifact filter on extreme movement values.
        if args.max_abs_delta_hz and args.max_abs_delta_hz > 0:
            sub = sub[
                (sub["delta_f1"].abs() <= args.max_abs_delta_hz)
                & (sub["delta_f2"].abs() <= args.max_abs_delta_hz)
                & (sub["delta_f3"].abs() <= args.max_abs_delta_hz)
            ]

        diagnostics.append({
            "source_file": path.name,
            "status": "loaded",
            "n_rows_loaded": len(sub),
            **found,
        })

        frames.append(sub)

    diag_df = pd.DataFrame(diagnostics)

    if not frames:
        diag_path = Path(args.output_dir) / "track_feature_column_diagnostics.csv"
        diag_path.parent.mkdir(parents=True, exist_ok=True)
        diag_df.to_csv(diag_path, index=False)
        raise RuntimeError(
            "No valid track feature rows found. "
            f"Diagnostics saved to: {diag_path}"
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["audio", "vowel"]).reset_index(drop=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(Path(args.output_dir) / "track_feature_column_diagnostics.csv", index=False)

    return out


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
    meta["audio"] = meta["audio"].apply(clean_audio_id)

    for col in args.social_vars:
        meta[col] = meta[col].astype("string").str.strip()
        meta[col] = meta[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    return meta


def merge_metadata(df: pd.DataFrame, metadata: pd.DataFrame, social_vars: list[str]) -> pd.DataFrame:
    merged = df.merge(metadata, on="audio", how="left", indicator=True)

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

    return {str(cat): palette[i % len(palette)] for i, cat in enumerate(categories)}


def downsample_for_plot(df: pd.DataFrame, args, social_var: str) -> pd.DataFrame:
    if args.plot_every is not None and args.plot_every > 1:
        df = df.iloc[::args.plot_every].copy()

    if args.max_points_per_group is None or args.max_points_per_group <= 0:
        return df.copy()

    chunks = []

    for _, group in df.groupby(["vowel", social_var], sort=False):
        n = min(len(group), args.max_points_per_group)

        if n > 0:
            chunks.append(group.sample(n=n, random_state=args.random_state))

    if not chunks:
        return df.iloc[0:0].copy()

    return pd.concat(chunks, ignore_index=True)


def hover_delta_text(df: pd.DataFrame, social_var: str) -> list[str]:
    texts = []

    for _, row in df.iterrows():
        texts.append(
            "<br>".join([
                f"audio: {row['audio']}",
                f"vowel: /{row['vowel']}/",
                f"{social_var}: {row[social_var]}",
                f"ΔF1 80-20: {row['delta_f1']:.1f} Hz",
                f"ΔF2 80-20: {row['delta_f2']:.1f} Hz",
                f"ΔF3 80-20: {row['delta_f3']:.1f} Hz",
                f"token_id: {row['token_id']}",
            ])
        )

    return texts


def update_delta_scene(fig, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1100,
        height=850,
        legend_title_text="Groupe social",
        scene=dict(
            xaxis=dict(title="ΔF2 = F2_80 - F2_20 (Hz)"),
            yaxis=dict(title="ΔF1 = F1_80 - F1_20 (Hz)"),
            zaxis=dict(title="ΔF3 = F3_80 - F3_20 (Hz)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )


def update_formant_scene(fig, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1100,
        height=850,
        legend_title_text="Groupe social",
        scene=dict(
            xaxis=dict(title="F2 (Hz)", autorange="reversed"),
            yaxis=dict(title="F1 (Hz)", autorange="reversed"),
            zaxis=dict(title="F3 (Hz)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )


def plot_delta_tokens(df: pd.DataFrame, vowel: str, social_var: str, output_dir: Path, args):
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
                x=d["delta_f2"],
                y=d["delta_f1"],
                z=d["delta_f3"],
                mode="markers",
                name=str(category),
                marker=dict(
                    size=args.point_size,
                    color=color_map[str(category)],
                    opacity=args.alpha,
                ),
                text=hover_delta_text(d, social_var),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    update_delta_scene(fig, f"Voyelle /{vowel}/ — dynamique ΔF1-ΔF2-ΔF3 par {social_var}")

    fig.write_html(
        output_dir / f"vowel_{vowel}_by_{social_var}_delta20_80_f1f2f3_3d.html",
        include_plotlyjs="cdn",
        full_html=True,
    )


def make_summary(df: pd.DataFrame, social_vars: list[str]) -> pd.DataFrame:
    rows = []

    numeric_cols = [
        "f1_20", "f2_20", "f3_20",
        "f1_80", "f2_80", "f3_80",
        "delta_f1", "delta_f2", "delta_f3",
    ]

    for social_var in social_vars:
        for keys, group in df.groupby(["vowel", social_var], dropna=False):
            vowel, social_group = keys

            row = {
                "social_variable": social_var,
                "vowel": vowel,
                "social_group": social_group,
                "n_tokens": len(group),
                "n_audios": group["audio"].nunique(),
            }

            for col in numeric_cols:
                if col in group.columns:
                    row[f"{col}_mean"] = group[col].mean()
                    row[f"{col}_median"] = group[col].median()
                    row[f"{col}_std"] = group[col].std()
                    row[f"{col}_iqr"] = group[col].quantile(0.75) - group[col].quantile(0.25)

            # Movement magnitude in 3D.
            mag = np.sqrt(
                group["delta_f1"] ** 2
                + group["delta_f2"] ** 2
                + group["delta_f3"] ** 2
            )

            row["delta_magnitude_mean"] = mag.mean()
            row["delta_magnitude_median"] = mag.median()
            row["delta_magnitude_iqr"] = mag.quantile(0.75) - mag.quantile(0.25)

            rows.append(row)

    return pd.DataFrame(rows)


def plot_median_vectors(summary: pd.DataFrame, vowel: str, social_var: str, output_dir: Path):
    go = require_plotly()

    d0 = summary[
        (summary["vowel"] == vowel)
        & (summary["social_variable"] == social_var)
    ].copy()

    if d0.empty:
        return

    # Need p20 and p80 columns. If unavailable, skip.
    needed = [
        "f1_20_median", "f2_20_median", "f3_20_median",
        "f1_80_median", "f2_80_median", "f3_80_median",
    ]

    if any(col not in d0.columns for col in needed):
        return

    d0 = d0.dropna(subset=needed)

    if d0.empty:
        return

    categories = sorted(d0["social_group"].astype(str).unique())
    color_map = make_color_map(categories)

    fig = go.Figure()

    for _, row in d0.iterrows():
        category = str(row["social_group"])

        x = [row["f2_20_median"], row["f2_80_median"]]
        y = [row["f1_20_median"], row["f1_80_median"]]
        z = [row["f3_20_median"], row["f3_80_median"]]

        hover = [
            "<br>".join([
                f"{social_var}: {category}",
                f"vowel: /{vowel}/",
                "point: 20%",
                f"F1_20 median: {row['f1_20_median']:.1f} Hz",
                f"F2_20 median: {row['f2_20_median']:.1f} Hz",
                f"F3_20 median: {row['f3_20_median']:.1f} Hz",
                f"n_tokens: {int(row['n_tokens'])}",
                f"n_audios: {int(row['n_audios'])}",
            ]),
            "<br>".join([
                f"{social_var}: {category}",
                f"vowel: /{vowel}/",
                "point: 80%",
                f"F1_80 median: {row['f1_80_median']:.1f} Hz",
                f"F2_80 median: {row['f2_80_median']:.1f} Hz",
                f"F3_80 median: {row['f3_80_median']:.1f} Hz",
                f"ΔF1 median: {row['delta_f1_median']:.1f} Hz",
                f"ΔF2 median: {row['delta_f2_median']:.1f} Hz",
                f"ΔF3 median: {row['delta_f3_median']:.1f} Hz",
                f"|Δ| median: {row['delta_magnitude_median']:.1f} Hz",
            ]),
        ]

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers+text",
                name=category,
                text=["20%", "80%"],
                textposition="top center",
                line=dict(width=7, color=color_map[category]),
                marker=dict(size=6, color=color_map[category]),
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
            )
        )

    update_formant_scene(fig, f"Voyelle /{vowel}/ — vecteurs médians 20% → 80% par {social_var}")

    fig.write_html(
        output_dir / f"vowel_{vowel}_by_{social_var}_median_vectors20_80_f1f2f3_3d.html",
        include_plotlyjs="cdn",
        full_html=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--track-features-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--metadata-key", default="Time.code.and.speaker")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--social-vars", nargs="+", default=["CountryRes", "Sex", "Income_group"])
    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--label-map", nargs="*", default=["y:i"])

    parser.add_argument(
        "--max-abs-delta-hz",
        type=float,
        default=1500,
        help="Remove tokens with absurd |delta| above this threshold. Use 0 to disable.",
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
        default=6000,
        help="Plotting-only cap per vowel/social group. Use 0 for no cap.",
    )

    parser.add_argument("--point-size", type=float, default=2.2)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tracks = load_track_features(args)
    metadata = load_metadata(args)
    merged = merge_metadata(tracks, metadata, args.social_vars)

    merged.to_csv(output_dir / "track_token_deltas_f1f2f3_with_social_full.csv", index=False)

    counts = (
        merged
        .groupby(["vowel", "audio", *args.social_vars])
        .size()
        .reset_index(name="n_tokens")
        .sort_values(["vowel", "audio"])
    )
    counts.to_csv(output_dir / "track_token_counts_by_audio_vowel_social.csv", index=False)

    summary = make_summary(merged, args.social_vars)
    summary.to_csv(output_dir / "track_delta_summary_by_vowel_social_group.csv", index=False)

    for social_var in args.social_vars:
        plot_df = downsample_for_plot(merged, args, social_var=social_var)
        plot_df.to_csv(output_dir / f"plot_data_delta_by_{social_var}.csv", index=False)

    for social_var in args.social_vars:
        for vowel in args.vowels:
            plot_delta_tokens(merged, vowel, social_var, output_dir, args)
            plot_median_vectors(summary, vowel, social_var, output_dir)

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print("")
    print("Rows:", len(merged))
    print("Audios:", merged["audio"].nunique())
    print("")
    print("Token counts by vowel:")
    print(merged["vowel"].value_counts().reindex(args.vowels))
    print("")
    print("Delta summary, first rows:")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
