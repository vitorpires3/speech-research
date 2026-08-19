#!/usr/bin/env python3
"""
Make F1-F2 plots from already generated CSVs.

Expected structure:
input-root/
  H1/all_points_f1f2_all_vowels.csv
  H2/all_points_f1f2_all_vowels.csv

Outputs:
output-root/
  H1/*.png, *.pdf
  H2/*.png, *.pdf

Plots:
- all vowels together
- one plot per vowel
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


VOWEL_COLORS = {
    "a": "#d62728",
    "e": "#ff7f0e",
    "i": "#2ca02c",
    "o": "#1f77b4",
    "u": "#9467bd",
}

SPEAKER_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2"
]

SPEAKER_MARKERS = ["o", "^", "s", "D", "P", "X", "v"]


def make_speaker_style_table(df: pd.DataFrame) -> pd.DataFrame:
    audios = sorted(df["audio"].dropna().unique())

    rows = []
    for idx, audio in enumerate(audios):
        rows.append({
            "speaker_code": idx + 1,
            "audio": audio,
            "color": SPEAKER_COLORS[idx % len(SPEAKER_COLORS)],
            "marker": SPEAKER_MARKERS[(idx // len(SPEAKER_COLORS)) % len(SPEAKER_MARKERS)],
        })

    return pd.DataFrame(rows)


def downsample(df: pd.DataFrame, max_points: int | None, random_state: int) -> pd.DataFrame:
    if max_points is None or max_points <= 0:
        return df.copy()

    return (
        df.groupby(["audio", "vowel"], group_keys=False)
        .apply(lambda x: x.sample(
            n=min(len(x), max_points),
            random_state=random_state
        ))
        .reset_index(drop=True)
    )


def compute_centers(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["audio", "vowel"])
        .agg(
            n_points=("f1", "size"),
            f1_median=("f1", "median"),
            f2_median=("f2", "median"),
            f1_mean=("f1", "mean"),
            f2_mean=("f2", "mean"),
        )
        .reset_index()
    )


def setup_axes(ax, title: str):
    ax.set_title(title)
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)


def save(fig, path_base: Path, dpi: int):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path_base.with_suffix(".png"), dpi=dpi)
    fig.savefig(path_base.with_suffix(".pdf"))
    plt.close(fig)


def plot_all_vowels(df_plot: pd.DataFrame, centers: pd.DataFrame, half: str, output_dir: Path, args):
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
            color=VOWEL_COLORS.get(vowel),
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
        alpha=0.7,
        label="centres médians locuteur-voyelle",
    )

    setup_axes(ax, f"{half} — Tous les points F1-F2 — toutes les voyelles")
    ax.legend(loc="best", fontsize=9, frameon=True)

    save(fig, output_dir / "all_vowels_all_points_f1f2", args.dpi)


def plot_one_vowel(
    df_plot: pd.DataFrame,
    centers: pd.DataFrame,
    speaker_styles: pd.DataFrame,
    vowel: str,
    half: str,
    output_dir: Path,
    args,
):
    d = df_plot[df_plot["vowel"] == vowel].copy()
    c = centers[centers["vowel"] == vowel].copy()

    if d.empty:
        return

    d = d.merge(speaker_styles, on="audio", how="left")
    c = c.merge(speaker_styles[["audio", "speaker_code"]], on="audio", how="left")

    fig, ax = plt.subplots(figsize=(9, 8))

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
        s=46,
        marker="x",
        color="black",
        linewidths=1.0,
        alpha=0.9,
        label="centre médian par locuteur",
    )

    for _, row in c.iterrows():
        if pd.notna(row["speaker_code"]):
            ax.text(
                row["f2_median"],
                row["f1_median"],
                str(int(row["speaker_code"])),
                fontsize=6,
                alpha=0.9,
            )

    setup_axes(ax, f"{half} — Tous les points F1-F2 — voyelle /{vowel}/")
    ax.legend(loc="best", fontsize=9, frameon=True)

    save(fig, output_dir / f"vowel_{vowel}_all_points_f1f2", args.dpi)


def process_half(half: str, args):
    input_csv = Path(args.input_root) / half / "all_points_f1f2_all_vowels.csv"
    output_dir = Path(args.output_root) / half
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")

    df = pd.read_csv(input_csv)

    needed = {"audio", "vowel", "f1", "f2"}
    missing = needed - set(df.columns)

    if missing:
        raise ValueError(f"{input_csv} is missing columns: {sorted(missing)}")

    df = df[df["vowel"].isin(args.vowels)].copy()
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df["f2"] = pd.to_numeric(df["f2"], errors="coerce")
    df = df.dropna(subset=["audio", "vowel", "f1", "f2"])
    df = df[(df["f1"] > 0) & (df["f2"] > 0)]

    df_plot = downsample(df, args.max_points_per_audio_vowel, args.random_state)

    centers = compute_centers(df)
    speaker_styles = make_speaker_style_table(df)

    centers.to_csv(output_dir / "speaker_vowel_centers_for_plot.csv", index=False)
    speaker_styles.to_csv(output_dir / "speaker_style_mapping.csv", index=False)

    if len(df_plot) != len(df):
        df_plot.to_csv(output_dir / "plot_subsample_used.csv", index=False)

    plot_all_vowels(df_plot, centers, half, output_dir, args)

    for vowel in args.vowels:
        plot_one_vowel(df_plot, centers, speaker_styles, vowel, half, output_dir, args)

    print(f"{half}: wrote plots to {output_dir}")
    print(f"{half}: full rows = {len(df)}, plot rows = {len(df_plot)}, audios = {df['audio'].nunique()}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)

    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--max-points-per-audio-vowel", type=int, default=200)
    parser.add_argument("--point-size", type=float, default=12)
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--dpi", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)

    args = parser.parse_args()

    for half in ["H1", "H2"]:
        process_half(half, args)

    print("")
    print("Done.")
    print(f"Output root: {args.output_root}")


if __name__ == "__main__":
    main()
