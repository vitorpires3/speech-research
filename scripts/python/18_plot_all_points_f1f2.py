#!/usr/bin/env python3
"""
Plot all F1-F2 token-level points from new-FAVE points files.

Outputs:
- combined CSV with all F1/F2 token-level points
- one CSV per vowel
- counts by audio/vowel
- speaker/vowel centers
- speaker style mapping
- one all-vowels plot
- one plot per vowel

Conventional vowel-space plot:
x-axis = F2
y-axis = F1
both axes inverted.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VOWEL_COL_CANDIDATES = [
    "vowel", "phone", "label", "vowel_label", "phone_label",
    "ipa", "arpa", "vclass", "vowel_class"
]

F1_COL_CANDIDATES = [
    "F1", "f1", "F1_Hz", "f1_hz", "f1hz", "F1Hz"
]

F2_COL_CANDIDATES = [
    "F2", "f2", "F2_Hz", "f2_hz", "f2hz", "F2Hz"
]

TOKEN_ID_CANDIDATES = [
    "id", "token_id", "vowel_id", "phone_id", "measurement_id"
]

EXTRA_COL_CANDIDATES = [
    "beg", "end", "start", "stop", "time", "duration", "dur",
    "word", "speaker", "file", "filename"
]

VOWEL_COLORS = {
    "a": "#d62728",
    "e": "#ff7f0e",
    "i": "#2ca02c",
    "o": "#1f77b4",
    "u": "#9467bd",
}

# 7 colors x 7 markers = 49 possible speaker styles.
SPEAKER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2"
]

SPEAKER_MARKERS = ["o", "^", "s", "D", "P", "X", "v"]


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
        f"Available columns: {list(df.columns)}. "
        f"Try passing --{role}-col explicitly."
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
        "_new_fave_points", "_fasttrack_points", "_points",
        ".new_fave_points", ".points"
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def load_points(args) -> pd.DataFrame:
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    vowels = set(args.vowels)
    label_map = parse_label_map(args.label_map)

    frames = []

    for path in files:
        df = pd.read_csv(path)

        if df.empty:
            continue

        vowel_col = find_column(df, VOWEL_COL_CANDIDATES, args.vowel_col, "vowel")
        f1_col = find_column(df, F1_COL_CANDIDATES, args.f1_col, "f1")
        f2_col = find_column(df, F2_COL_CANDIDATES, args.f2_col, "f2")
        token_col = find_optional_column(df, TOKEN_ID_CANDIDATES)

        keep_cols = [vowel_col, f1_col, f2_col]

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

        sub["f1"] = pd.to_numeric(sub[f1_col], errors="coerce")
        sub["f2"] = pd.to_numeric(sub[f2_col], errors="coerce")

        if token_col is not None:
            sub["token_id"] = sub[token_col]
        else:
            sub["token_id"] = np.arange(len(sub))

        sub = sub.dropna(subset=["vowel", "f1", "f2"])
        sub = sub[sub["vowel"].isin(args.vowels)]
        sub = sub[(sub["f1"] > 0) & (sub["f2"] > 0)]

        frames.append(sub)

    if not frames:
        raise RuntimeError(
            "No valid F1/F2 vowel points found. "
            "Check --vowel-col, --f1-col, --f2-col and --label-map."
        )

    out = pd.concat(frames, ignore_index=True)
    return out


def make_speaker_style_table(df: pd.DataFrame) -> pd.DataFrame:
    audios = sorted(df["audio"].dropna().unique())

    rows = []
    for idx, audio in enumerate(audios):
        color = SPEAKER_COLORS[idx % len(SPEAKER_COLORS)]
        marker = SPEAKER_MARKERS[(idx // len(SPEAKER_COLORS)) % len(SPEAKER_MARKERS)]

        rows.append({
            "speaker_code": idx + 1,
            "audio": audio,
            "color": color,
            "marker": marker,
        })

    return pd.DataFrame(rows)


def downsample_for_plot(df: pd.DataFrame, args) -> pd.DataFrame:
    """
    Safe plotting-only subsampling.

    This avoids pandas groupby.apply(), which may return a malformed empty
    dataframe depending on the pandas version/environment.
    """
    if args.max_points_per_audio_vowel is None or args.max_points_per_audio_vowel <= 0:
        return df.copy()

    if df.empty:
        return df.copy()

    chunks = []

    for _, group in df.groupby(["audio", "vowel"], sort=False):
        n = min(len(group), args.max_points_per_audio_vowel)

        if n <= 0:
            continue

        chunks.append(
            group.sample(
                n=n,
                random_state=args.random_state,
            )
        )

    if not chunks:
        return df.iloc[0:0].copy()

    out = pd.concat(chunks, ignore_index=True)

    # Defensive check.
    required = {"audio", "vowel", "f1", "f2"}
    missing = required - set(out.columns)

    if missing:
        raise RuntimeError(
            f"Downsampled dataframe lost required columns: {sorted(missing)}. "
            f"Columns are: {list(out.columns)}"
        )

    return out


def setup_vowel_axes(ax, title: str):
    ax.set_title(title)
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)


def save_plot(fig, output_base: Path, dpi: int):
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def make_all_vowels_plot(df_plot: pd.DataFrame, centers: pd.DataFrame, args, output_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 8))

    for vowel in args.vowels:
        d = df_plot[df_plot["vowel"] == vowel]
        if d.empty:
            continue

        ax.scatter(
            d["f2"],
            d["f1"],
            s=args.point_size,
            alpha=args.alpha,
            color=VOWEL_COLORS.get(vowel, None),
            label=f"/{vowel}/",
            linewidths=0,
            rasterized=True,
        )

    ax.scatter(
        centers["f2_median"],
        centers["f1_median"],
        s=22,
        marker="x",
        color="black",
        linewidths=0.8,
        alpha=0.65,
        label="centres locuteur-voyelle",
    )

    setup_vowel_axes(ax, "Tous les points F1-F2 — toutes les voyelles")
    ax.legend(loc="best", fontsize=9, frameon=True)

    save_plot(fig, output_dir / "all_vowels_all_points_f1f2", args.dpi)


def make_one_vowel_plot(
    df_plot: pd.DataFrame,
    centers: pd.DataFrame,
    vowel: str,
    args,
    output_dir: Path,
    speaker_styles: pd.DataFrame,
):
    d = df_plot[df_plot["vowel"] == vowel].copy()
    c = centers[centers["vowel"] == vowel].copy()

    fig, ax = plt.subplots(figsize=(9, 8))

    if args.speaker_style_grid:
        d = d.merge(speaker_styles, on="audio", how="left")
        c = c.merge(speaker_styles[["audio", "speaker_code"]], on="audio", how="left")

        for _, style in speaker_styles.iterrows():
            ds = d[d["audio"] == style["audio"]]

            if ds.empty:
                continue

            ax.scatter(
                ds["f2"],
                ds["f1"],
                s=args.point_size,
                alpha=args.alpha,
                color=style["color"],
                marker=style["marker"],
                linewidths=0,
                rasterized=True,
            )

        ax.scatter(
            c["f2_median"],
            c["f1_median"],
            s=42,
            marker="x",
            color="black",
            linewidths=1.0,
            alpha=0.85,
            label="centre médian par locuteur",
        )

        for _, row in c.iterrows():
            if pd.notna(row.get("speaker_code")):
                ax.text(
                    row["f2_median"],
                    row["f1_median"],
                    str(int(row["speaker_code"])),
                    fontsize=6,
                    alpha=0.85,
                )

    elif args.color_by_audio:
        audio_codes = pd.Categorical(d["audio"]).codes

        scatter = ax.scatter(
            d["f2"],
            d["f1"],
            s=args.point_size,
            alpha=args.alpha,
            c=audio_codes,
            cmap="tab20",
            linewidths=0,
            rasterized=True,
        )

        cb = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("Code locuteur/audio")

        ax.scatter(
            c["f2_median"],
            c["f1_median"],
            s=36,
            marker="x",
            color="black",
            linewidths=0.9,
            alpha=0.8,
            label="centre par locuteur",
        )

    else:
        ax.scatter(
            d["f2"],
            d["f1"],
            s=args.point_size,
            alpha=args.alpha,
            color=VOWEL_COLORS.get(vowel, None),
            linewidths=0,
            rasterized=True,
        )

        ax.scatter(
            c["f2_median"],
            c["f1_median"],
            s=36,
            marker="x",
            color="black",
            linewidths=0.9,
            alpha=0.8,
            label="centre par locuteur",
        )

    if args.label_speaker_centers and not args.speaker_style_grid:
        for idx, row in c.reset_index(drop=True).iterrows():
            ax.text(
                row["f2_median"],
                row["f1_median"],
                str(idx + 1),
                fontsize=6,
                alpha=0.8,
            )

    setup_vowel_axes(ax, f"Tous les points F1-F2 — voyelle /{vowel}/")
    ax.legend(loc="best", fontsize=9, frameon=True)

    save_plot(fig, output_dir / f"vowel_{vowel}_all_points_f1f2", args.dpi)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--label-map", nargs="*", default=["y:i"])

    parser.add_argument("--vowel-col", default=None)
    parser.add_argument("--f1-col", default=None)
    parser.add_argument("--f2-col", default=None)

    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--alpha", type=float, default=0.16)
    parser.add_argument("--dpi", type=int, default=250)

    parser.add_argument("--color-by-audio", action="store_true")
    parser.add_argument("--label-speaker-centers", action="store_true")

    parser.add_argument(
        "--speaker-style-grid",
        action="store_true",
        help="For per-vowel plots, encode each speaker with a stable color+marker combination.",
    )

    parser.add_argument(
        "--max-points-per-audio-vowel",
        type=int,
        default=None,
        help="Plotting-only subsampling: maximum number of points per audio/vowel.",
    )

    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_points(args)

    # Full token-level data.
    df.to_csv(output_dir / "all_points_f1f2_all_vowels.csv", index=False)

    # Optional plotting-only subsampling.
    df_plot = downsample_for_plot(df, args)

    if len(df_plot) != len(df):
        df_plot.to_csv(output_dir / "all_points_f1f2_plot_subsample.csv", index=False)

    counts = (
        df.groupby(["audio", "vowel"])
        .size()
        .reset_index(name="n_points")
        .sort_values(["audio", "vowel"])
    )
    counts.to_csv(output_dir / "all_points_f1f2_counts_by_audio_vowel.csv", index=False)

    centers = (
        df.groupby(["audio", "vowel"])
        .agg(
            n_points=("f1", "size"),
            f1_median=("f1", "median"),
            f2_median=("f2", "median"),
            f1_mean=("f1", "mean"),
            f2_mean=("f2", "mean"),
        )
        .reset_index()
        .sort_values(["vowel", "audio"])
    )
    centers.to_csv(output_dir / "all_points_f1f2_speaker_vowel_centers.csv", index=False)

    speaker_styles = make_speaker_style_table(df)
    speaker_styles.to_csv(output_dir / "speaker_style_mapping.csv", index=False)

    for vowel in args.vowels:
        d = df[df["vowel"] == vowel].copy()
        d.to_csv(output_dir / f"points_f1f2_vowel_{vowel}.csv", index=False)

    make_all_vowels_plot(df_plot, centers, args, output_dir)

    for vowel in args.vowels:
        if (df["vowel"] == vowel).any():
            make_one_vowel_plot(
                df_plot,
                centers,
                vowel,
                args,
                output_dir,
                speaker_styles=speaker_styles,
            )

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print("")
    print("Total points by vowel:")
    print(df.groupby("vowel").size().reindex(args.vowels))
    print("")
    print("Number of audios detected:", df["audio"].nunique())
    print("Number of rows in full CSV:", len(df))
    print("Number of rows used for plotting:", len(df_plot))


if __name__ == "__main__":
    main()
