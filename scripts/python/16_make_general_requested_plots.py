#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VOWELS = ["i", "e", "a", "o", "u"]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def full_audio_label(audio):
    return str(audio)


def save_bar_plot(df, x_col, y_col, title, ylabel, output_path, zero_line=False):
    df = df.copy()
    df = df.sort_values(y_col, ascending=True)

    labels = df[x_col].astype(str).tolist()
    values = df[y_col].to_numpy(dtype=float)

    fig_width = max(14, len(df) * 0.42)

    fig, ax = plt.subplots(figsize=(fig_width, 6))

    x = np.arange(len(df))

    ax.bar(x, values)

    if zero_line:
        ax.axhline(0, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_vowel_overlap_plots(overlap_df, output_dir):
    overlap_df = overlap_df.copy()
    overlap_df["audio_label"] = overlap_df["audio"].apply(full_audio_label)

    # H1 and H2 plots
    for half in ["H1", "H2"]:
        for vowel in VOWELS:
            sub = overlap_df[
                (overlap_df["half"] == half)
                & (overlap_df["vowel"] == vowel)
            ].copy()

            if sub.empty:
                continue

            output_path = output_dir / f"overlap_{half}_vowel_{vowel}_by_audio_sorted.png"

            save_bar_plot(
                df=sub,
                x_col="audio_label",
                y_col="overlap_percent_with_any_other_vowel",
                title=f"Overlap of /{vowel}/ by audio - {half}",
                ylabel=f"/{vowel}/ overlap with any other vowel (%)",
                output_path=output_path,
                zero_line=False,
            )

    # Difference H2 - H1
    h1 = overlap_df[overlap_df["half"] == "H1"][
        ["audio", "vowel", "overlap_percent_with_any_other_vowel"]
    ].rename(columns={"overlap_percent_with_any_other_vowel": "overlap_H1"})

    h2 = overlap_df[overlap_df["half"] == "H2"][
        ["audio", "vowel", "overlap_percent_with_any_other_vowel"]
    ].rename(columns={"overlap_percent_with_any_other_vowel": "overlap_H2"})

    diff = h1.merge(h2, on=["audio", "vowel"], how="inner")
    diff["audio_label"] = diff["audio"].apply(full_audio_label)
    diff["overlap_diff_H2_minus_H1"] = diff["overlap_H2"] - diff["overlap_H1"]

    diff.to_csv(output_dir / "overlap_difference_H2_minus_H1_by_audio_vowel.csv", index=False)

    for vowel in VOWELS:
        sub = diff[diff["vowel"] == vowel].copy()

        if sub.empty:
            continue

        output_path = output_dir / f"overlap_diff_H2_minus_H1_vowel_{vowel}_by_audio_sorted.png"

        save_bar_plot(
            df=sub,
            x_col="audio_label",
            y_col="overlap_diff_H2_minus_H1",
            title=f"Overlap change for /{vowel}/ by audio: H2 - H1",
            ylabel=f"/{vowel}/ overlap difference H2 - H1 (%)",
            output_path=output_path,
            zero_line=True,
        )


def make_accuracy_plot(summary_df, output_dir):
    df = summary_df.copy()
    df["audio_label"] = df["audio"].apply(full_audio_label)

    output_path = output_dir / "cross_half_classification_accuracy_by_audio_sorted.png"

    save_bar_plot(
        df=df,
        x_col="audio_label",
        y_col="classification_accuracy_mean",
        title="Cross-half classification accuracy by audio",
        ylabel="Mean cross-half classification accuracy",
        output_path=output_path,
        zero_line=False,
    )


def make_silhouette_scatter(summary_df, output_dir):
    df = summary_df.dropna(subset=["silhouette_H1", "silhouette_H2"]).copy()
    df["audio_label"] = df["audio"].apply(full_audio_label)

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(df["silhouette_H1"], df["silhouette_H2"], s=55)

    for row in df.itertuples(index=False):
        ax.text(row.silhouette_H1, row.silhouette_H2, row.audio_label, fontsize=7)

    min_val = min(df["silhouette_H1"].min(), df["silhouette_H2"].min())
    max_val = max(df["silhouette_H1"].max(), df["silhouette_H2"].max())

    pad = (max_val - min_val) * 0.08
    if not np.isfinite(pad) or pad == 0:
        pad = 0.05

    lo = min_val - pad
    hi = max_val + pad

    ax.plot([lo, hi], [lo, hi], linewidth=1)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    ax.set_xlabel("Silhouette H1")
    ax.set_ylabel("Silhouette H2")
    ax.set_title("Silhouette comparison: H1 × H2")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "silhouette_H1_vs_H2_scatter.png", dpi=300)
    plt.close(fig)


def make_summary_tables(overlap_df, summary_df, output_dir):
    h1 = overlap_df[overlap_df["half"] == "H1"][
        ["audio", "vowel", "overlap_percent_with_any_other_vowel"]
    ].rename(columns={"overlap_percent_with_any_other_vowel": "overlap_H1"})

    h2 = overlap_df[overlap_df["half"] == "H2"][
        ["audio", "vowel", "overlap_percent_with_any_other_vowel"]
    ].rename(columns={"overlap_percent_with_any_other_vowel": "overlap_H2"})

    wide = h1.merge(h2, on=["audio", "vowel"], how="inner")
    wide["overlap_diff_H2_minus_H1"] = wide["overlap_H2"] - wide["overlap_H1"]

    wide.to_csv(output_dir / "overlap_H1_H2_diff_long.csv", index=False)

    pivot = wide.pivot(index="audio", columns="vowel", values=["overlap_H1", "overlap_H2", "overlap_diff_H2_minus_H1"])
    pivot.columns = [f"{metric}_{vowel}" for metric, vowel in pivot.columns]
    pivot = pivot.reset_index()

    keep_cols = [
        "audio",
        "classification_accuracy_mean",
        "silhouette_H1",
        "silhouette_H2",
        "silhouette_diff_H2_minus_H1",
    ]

    available_keep_cols = [col for col in keep_cols if col in summary_df.columns]

    general = summary_df[available_keep_cols].merge(pivot, on="audio", how="left")
    general.to_csv(output_dir / "requested_general_plot_data_wide.csv", index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        default="results/all_audio_half_region_profiles_level80",
        help="Folder containing the general CSV outputs.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output folder for the requested plots. Default: input-root/general_requested_plots",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)

    if args.output_dir is None:
        output_dir = input_root / "general_requested_plots"
    else:
        output_dir = Path(args.output_dir)

    ensure_dir(output_dir)

    overlap_path = input_root / "general_total_region_overlap_by_vowel_all.csv"
    summary_path = input_root / "general_audio_summary.csv"

    if not overlap_path.exists():
        raise FileNotFoundError(f"Missing file: {overlap_path}")

    if not summary_path.exists():
        raise FileNotFoundError(f"Missing file: {summary_path}")

    overlap_df = pd.read_csv(overlap_path)
    summary_df = pd.read_csv(summary_path)

    make_vowel_overlap_plots(overlap_df, output_dir)
    make_accuracy_plot(summary_df, output_dir)
    make_silhouette_scatter(summary_df, output_dir)
    make_summary_tables(overlap_df, summary_df, output_dir)

    print()
    print("=== Requested general plots generated ===")
    print(f"Input root: {input_root}")
    print(f"Output dir: {output_dir}")
    print()
    print("Generated:")
    print("- 5 H1 overlap bar plots")
    print("- 5 H2 overlap bar plots")
    print("- 5 H2-H1 overlap difference bar plots")
    print("- 1 cross-half classification accuracy bar plot")
    print("- 1 silhouette H1 × H2 scatter plot")
    print("- summary CSV tables")
    print()
    print("Main files:")
    print(f"- {output_dir / 'cross_half_classification_accuracy_by_audio_sorted.png'}")
    print(f"- {output_dir / 'silhouette_H1_vs_H2_scatter.png'}")
    print(f"- {output_dir / 'requested_general_plot_data_wide.csv'}")
    print(f"- {output_dir / 'overlap_H1_H2_diff_long.csv'}")


if __name__ == "__main__":
    main()
