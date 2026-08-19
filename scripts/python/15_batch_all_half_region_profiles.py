#!/usr/bin/env python3

from pathlib import Path
import argparse
import subprocess
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VOWEL_ORDER = ["i", "e", "a", "o", "u"]


def clean_audio_id(value):
    s = str(value).strip()
    s = Path(s).name

    for ext in [".wav", ".flac", ".mp3", ".csv", ".TextGrid", ".textgrid", ".seg"]:
        if s.endswith(ext):
            s = s[: -len(ext)]

    s = re.sub(r"\.v[0-9]+$", "", s)

    for suffix in [
        "_points",
        "-points",
        "_point",
        "-point",
        "_tracks",
        "-tracks",
        "_track",
        "-track",
        "_track_features",
        "-track_features",
    ]:
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]

    return s


def short_label(audio):
    """
    Label used in the general plots.

    Important:
    Use the full audio/file id, not only Audio_SPxx, because several files
    may share the same speaker number.
    """
    return str(audio)


def run_command(cmd, log_prefix):
    result = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
    )

    return {
        "command": " ".join(cmd),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "log_prefix": log_prefix,
    }


def read_csv_if_exists(path):
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def first_or_nan(df, col):
    if df.empty or col not in df.columns:
        return np.nan

    values = df[col].dropna()

    if values.empty:
        return np.nan

    return values.iloc[0]


def row_for_half(df, half):
    if df.empty or "half" not in df.columns:
        return pd.DataFrame()

    return df[df["half"] == half].copy()


def pair_name(row):
    if pd.isna(row.get("vowel_1", np.nan)) or pd.isna(row.get("vowel_2", np.nan)):
        return ""

    return f"{row['vowel_1']}-{row['vowel_2']}"


def build_audio_summary(audio, centroid_dir, region_dir):
    half_summary = read_csv_if_exists(centroid_dir / "half_summary.csv")
    similarity = read_csv_if_exists(centroid_dir / "half_similarity_summary.csv")
    total_overlap = read_csv_if_exists(region_dir / "total_region_overlap_by_vowel.csv")
    pairwise_overlap = read_csv_if_exists(region_dir / "pairwise_region_overlap.csv")

    h1 = row_for_half(half_summary, "H1")
    h2 = row_for_half(half_summary, "H2")

    overlap_h1 = row_for_half(total_overlap, "H1")
    overlap_h2 = row_for_half(total_overlap, "H2")

    pair_h1 = row_for_half(pairwise_overlap, "H1")
    pair_h2 = row_for_half(pairwise_overlap, "H2")

    if not pair_h1.empty and "jaccard_overlap_percent" in pair_h1.columns:
        top_pair_h1 = pair_h1.sort_values("jaccard_overlap_percent", ascending=False).iloc[0]
        top_pair_h1_name = pair_name(top_pair_h1)
        top_pair_h1_jaccard = top_pair_h1["jaccard_overlap_percent"]
    else:
        top_pair_h1_name = ""
        top_pair_h1_jaccard = np.nan

    if not pair_h2.empty and "jaccard_overlap_percent" in pair_h2.columns:
        top_pair_h2 = pair_h2.sort_values("jaccard_overlap_percent", ascending=False).iloc[0]
        top_pair_h2_name = pair_name(top_pair_h2)
        top_pair_h2_jaccard = top_pair_h2["jaccard_overlap_percent"]
    else:
        top_pair_h2_name = ""
        top_pair_h2_jaccard = np.nan

    def most_overlapped_name(df):
        if df.empty or "overlap_percent_with_any_other_vowel" not in df.columns:
            return ""
        row = df.sort_values("overlap_percent_with_any_other_vowel", ascending=False).iloc[0]
        return row["vowel"]

    def most_overlapped_value(df):
        if df.empty or "overlap_percent_with_any_other_vowel" not in df.columns:
            return np.nan
        row = df.sort_values("overlap_percent_with_any_other_vowel", ascending=False).iloc[0]
        return row["overlap_percent_with_any_other_vowel"]

    mean_overlap_h1 = (
        float(overlap_h1["overlap_percent_with_any_other_vowel"].mean())
        if not overlap_h1.empty and "overlap_percent_with_any_other_vowel" in overlap_h1.columns
        else np.nan
    )

    mean_overlap_h2 = (
        float(overlap_h2["overlap_percent_with_any_other_vowel"].mean())
        if not overlap_h2.empty and "overlap_percent_with_any_other_vowel" in overlap_h2.columns
        else np.nan
    )

    row = {
        "audio": audio,
        "short_label": short_label(audio),

        "n_tokens_H1": first_or_nan(h1, "n_tokens"),
        "n_tokens_H2": first_or_nan(h2, "n_tokens"),
        "min_tokens_per_vowel_H1": first_or_nan(h1, "min_tokens_per_vowel"),
        "min_tokens_per_vowel_H2": first_or_nan(h2, "min_tokens_per_vowel"),

        "vowel_space_area_H1": first_or_nan(h1, "vowel_space_area"),
        "vowel_space_area_H2": first_or_nan(h2, "vowel_space_area"),
        "vowel_space_area_diff_H2_minus_H1": first_or_nan(h2, "vowel_space_area") - first_or_nan(h1, "vowel_space_area"),

        "silhouette_H1": first_or_nan(h1, "silhouette_by_vowel"),
        "silhouette_H2": first_or_nan(h2, "silhouette_by_vowel"),
        "silhouette_diff_H2_minus_H1": first_or_nan(h2, "silhouette_by_vowel") - first_or_nan(h1, "silhouette_by_vowel"),

        "mean_pairwise_separation_H1": first_or_nan(h1, "mean_pairwise_separation_score"),
        "mean_pairwise_separation_H2": first_or_nan(h2, "mean_pairwise_separation_score"),
        "mean_pairwise_separation_diff_H2_minus_H1": first_or_nan(h2, "mean_pairwise_separation_score") - first_or_nan(h1, "mean_pairwise_separation_score"),

        "mean_within_vowel_dispersion_H1": first_or_nan(h1, "mean_within_vowel_dispersion"),
        "mean_within_vowel_dispersion_H2": first_or_nan(h2, "mean_within_vowel_dispersion"),
        "mean_within_vowel_dispersion_diff_H2_minus_H1": first_or_nan(h2, "mean_within_vowel_dispersion") - first_or_nan(h1, "mean_within_vowel_dispersion"),

        "mean_centroid_shift": first_or_nan(similarity, "mean_centroid_shift"),
        "median_centroid_shift": first_or_nan(similarity, "median_centroid_shift"),
        "max_centroid_shift": first_or_nan(similarity, "max_centroid_shift"),
        "distance_matrix_correlation": first_or_nan(similarity, "distance_matrix_correlation"),
        "classification_accuracy_mean": first_or_nan(similarity, "classification_accuracy_mean"),

        "mean_region_overlap_H1": mean_overlap_h1,
        "mean_region_overlap_H2": mean_overlap_h2,
        "mean_region_overlap_diff_H2_minus_H1": mean_overlap_h2 - mean_overlap_h1,

        "most_overlapped_vowel_H1": most_overlapped_name(overlap_h1),
        "most_overlapped_vowel_H1_percent": most_overlapped_value(overlap_h1),
        "most_overlapped_vowel_H2": most_overlapped_name(overlap_h2),
        "most_overlapped_vowel_H2_percent": most_overlapped_value(overlap_h2),

        "top_pair_H1": top_pair_h1_name,
        "top_pair_H1_jaccard": top_pair_h1_jaccard,
        "top_pair_H2": top_pair_h2_name,
        "top_pair_H2_jaccard": top_pair_h2_jaccard,
    }

    return row


def save_general_tables(output_root, audio_dirs):
    audio_summaries = []
    half_summary_all = []
    similarity_all = []
    total_overlap_all = []
    pairwise_overlap_all = []
    ellipse_all = []

    for audio_dir in audio_dirs:
        audio = audio_dir.name

        centroid_dir = audio_dir / "centroid_profile"
        region_dir = audio_dir / "region_overlap"

        audio_summaries.append(
            build_audio_summary(
                audio=audio,
                centroid_dir=centroid_dir,
                region_dir=region_dir,
            )
        )

        half_summary = read_csv_if_exists(centroid_dir / "half_summary.csv")
        if not half_summary.empty:
            half_summary.insert(0, "audio", audio)
            half_summary_all.append(half_summary)

        similarity = read_csv_if_exists(centroid_dir / "half_similarity_summary.csv")
        if not similarity.empty:
            if "audio" not in similarity.columns:
                similarity.insert(0, "audio", audio)
            similarity_all.append(similarity)

        total_overlap = read_csv_if_exists(region_dir / "total_region_overlap_by_vowel.csv")
        if not total_overlap.empty:
            total_overlap_all.append(total_overlap)

        pairwise_overlap = read_csv_if_exists(region_dir / "pairwise_region_overlap.csv")
        if not pairwise_overlap.empty:
            pairwise_overlap_all.append(pairwise_overlap)

        ellipse = read_csv_if_exists(region_dir / "ellipse_parameters.csv")
        if not ellipse.empty:
            ellipse_all.append(ellipse)

    summary_df = pd.DataFrame(audio_summaries)
    summary_df = summary_df.sort_values("audio")

    summary_df.to_csv(output_root / "general_audio_summary.csv", index=False)

    if half_summary_all:
        pd.concat(half_summary_all, ignore_index=True).to_csv(
            output_root / "general_half_summary_all.csv",
            index=False,
        )

    if similarity_all:
        pd.concat(similarity_all, ignore_index=True).to_csv(
            output_root / "general_similarity_summary_all.csv",
            index=False,
        )

    if total_overlap_all:
        pd.concat(total_overlap_all, ignore_index=True).to_csv(
            output_root / "general_total_region_overlap_by_vowel_all.csv",
            index=False,
        )

    if pairwise_overlap_all:
        pd.concat(pairwise_overlap_all, ignore_index=True).to_csv(
            output_root / "general_pairwise_region_overlap_all.csv",
            index=False,
        )

    if ellipse_all:
        pd.concat(ellipse_all, ignore_index=True).to_csv(
            output_root / "general_ellipse_parameters_all.csv",
            index=False,
        )

    return summary_df


def plot_mean_region_overlap_by_audio(summary_df, output_path):
    df = summary_df.dropna(subset=["mean_region_overlap_H1", "mean_region_overlap_H2"]).copy()

    if df.empty:
        return

    df = df.sort_values("mean_region_overlap_diff_H2_minus_H1")

    labels = df["short_label"].tolist()
    x = np.arange(len(df))
    width = 0.4

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x - width / 2, df["mean_region_overlap_H1"], width, label="H1")
    ax.bar(x + width / 2, df["mean_region_overlap_H2"], width, label="H2")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel("Mean region overlap (%)")
    ax.set_title("Mean vowel-region overlap by audio: H1 vs H2")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_vowel_overlap_mean(output_root):
    path = output_root / "general_total_region_overlap_by_vowel_all.csv"

    if not path.exists():
        return

    df = pd.read_csv(path)

    if df.empty:
        return

    agg = (
        df.groupby(["half", "vowel"], as_index=False)
        .agg(
            mean_overlap=("overlap_percent_with_any_other_vowel", "mean"),
            median_overlap=("overlap_percent_with_any_other_vowel", "median"),
        )
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(VOWEL_ORDER))
    width = 0.4

    for i, half in enumerate(["H1", "H2"]):
        sub = agg[agg["half"] == half].set_index("vowel").reindex(VOWEL_ORDER)
        values = sub["mean_overlap"].to_numpy(dtype=float)

        ax.bar(x + (i - 0.5) * width, values, width, label=half)

    ax.set_xticks(x)
    ax.set_xticklabels([f"/{v}/" for v in VOWEL_ORDER])
    ax.set_ylabel("Mean overlap with any other vowel (%)")
    ax.set_title("Mean vowel-region overlap across all audios")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_root / "general_vowel_overlap_mean_H1_H2.png", dpi=300)
    plt.close(fig)


def plot_centroid_shift_vs_accuracy(summary_df, output_path):
    df = summary_df.dropna(subset=["mean_centroid_shift", "classification_accuracy_mean"]).copy()

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(df["mean_centroid_shift"], df["classification_accuracy_mean"], s=55)

    for row in df.itertuples(index=False):
        ax.text(row.mean_centroid_shift, row.classification_accuracy_mean, row.short_label, fontsize=7)

    ax.set_xlabel("Mean centroid shift H1→H2 (Hz)")
    ax.set_ylabel("Cross-half classification accuracy")
    ax.set_title("Centroid stability vs cross-half predictability")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_overlap_change_vs_silhouette_change(summary_df, output_path):
    df = summary_df.dropna(
        subset=["mean_region_overlap_diff_H2_minus_H1", "silhouette_diff_H2_minus_H1"]
    ).copy()

    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(df["mean_region_overlap_diff_H2_minus_H1"], df["silhouette_diff_H2_minus_H1"], s=55)

    for row in df.itertuples(index=False):
        ax.text(
            row.mean_region_overlap_diff_H2_minus_H1,
            row.silhouette_diff_H2_minus_H1,
            row.short_label,
            fontsize=7,
        )

    ax.axvline(0, linewidth=1)
    ax.axhline(0, linewidth=1)

    ax.set_xlabel("Mean region overlap change: H2 - H1 (%)")
    ax.set_ylabel("Silhouette change: H2 - H1")
    ax.set_title("Change in regional overlap vs change in token separation")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_plots(output_root, summary_df):
    plot_mean_region_overlap_by_audio(
        summary_df,
        output_root / "general_mean_region_overlap_H1_H2_by_audio.png",
    )

    plot_vowel_overlap_mean(output_root)

    plot_centroid_shift_vs_accuracy(
        summary_df,
        output_root / "general_centroid_shift_vs_accuracy.png",
    )

    plot_overlap_change_vs_silhouette_change(
        summary_df,
        output_root / "general_overlap_change_vs_silhouette_change.png",
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--points-dir",
        default="data/processed/new_fave_points",
        help="Pasta com arquivos de points.",
    )

    parser.add_argument(
        "--audio-dir",
        default="data/raw/sound",
        help="Pasta com arquivos de áudio.",
    )

    parser.add_argument(
        "--output-root",
        required=True,
        help="Pasta raiz dos resultados.",
    )

    parser.add_argument(
        "--region-level",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--outlier-z",
        type=float,
        default=3.5,
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pula áudios que já têm resultados principais gerados.",
    )

    parser.add_argument(
        "--max-audios",
        type=int,
        default=None,
        help="Opcional: processar só os primeiros N áudios, para teste rápido.",
    )

    args = parser.parse_args()

    project_root = Path.cwd()
    points_dir = Path(args.points_dir)
    audio_dir = Path(args.audio_dir)
    output_root = Path(args.output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    script12 = project_root / "scripts/python/12_split_points_by_audio_half.py"
    script13 = project_root / "scripts/python/13_analyze_half_vowel_profile.py"
    script14 = project_root / "scripts/python/14_vowel_region_overlap.py"

    for script in [script12, script13, script14]:
        if not script.exists():
            raise FileNotFoundError(f"Script não encontrado: {script}")

    points_files = sorted(points_dir.glob("*.csv"))

    if args.max_audios is not None:
        points_files = points_files[: args.max_audios]

    if not points_files:
        raise ValueError(f"Nenhum CSV encontrado em {points_dir}")

    errors = []
    processed_audio_dirs = []

    print()
    print("=== Batch half/profile/region analysis ===")
    print(f"Points dir: {points_dir}")
    print(f"Audio dir: {audio_dir}")
    print(f"Output root: {output_root}")
    print(f"Files to process: {len(points_files)}")
    print()

    for i, points_file in enumerate(points_files, start=1):
        audio = clean_audio_id(points_file.stem)
        audio_dir_out = output_root / audio

        split_dir = audio_dir_out / "points_halves"
        centroid_dir = audio_dir_out / "centroid_profile"
        region_dir = audio_dir_out / "region_overlap"

        processed_audio_dirs.append(audio_dir_out)

        final_marker = region_dir / "total_region_overlap_by_vowel.csv"

        if args.skip_existing and final_marker.exists():
            print(f"[{i}/{len(points_files)}] Skipping existing: {audio}")
            continue

        audio_dir_out.mkdir(parents=True, exist_ok=True)

        print(f"[{i}/{len(points_files)}] Processing: {audio}")

        cmd_split = [
            sys.executable,
            str(script12),
            "--points-dir",
            str(points_dir),
            "--audio-dir",
            str(audio_dir),
            "--audio",
            audio,
            "--output-dir",
            str(split_dir),
        ]

        result_split = run_command(cmd_split, f"{audio} :: split")

        if result_split["returncode"] != 0:
            errors.append(
                {
                    "audio": audio,
                    "step": "split",
                    "returncode": result_split["returncode"],
                    "command": result_split["command"],
                    "stdout": result_split["stdout"],
                    "stderr": result_split["stderr"],
                }
            )
            print(f"  ERROR in split: {audio}")
            continue

        points_with_halves = split_dir / "all_points_with_halves.csv"

        cmd_centroid = [
            sys.executable,
            str(script13),
            "--points-file",
            str(points_with_halves),
            "--audio",
            audio,
            "--output-dir",
            str(centroid_dir),
        ]

        result_centroid = run_command(cmd_centroid, f"{audio} :: centroid_profile")

        if result_centroid["returncode"] != 0:
            errors.append(
                {
                    "audio": audio,
                    "step": "centroid_profile",
                    "returncode": result_centroid["returncode"],
                    "command": result_centroid["command"],
                    "stdout": result_centroid["stdout"],
                    "stderr": result_centroid["stderr"],
                }
            )
            print(f"  ERROR in centroid profile: {audio}")
            continue

        cmd_region = [
            sys.executable,
            str(script14),
            "--points-file",
            str(points_with_halves),
            "--audio",
            audio,
            "--output-dir",
            str(region_dir),
            "--region-level",
            str(args.region_level),
            "--outlier-z",
            str(args.outlier_z),
            "--grid-size",
            str(args.grid_size),
        ]

        result_region = run_command(cmd_region, f"{audio} :: region_overlap")

        if result_region["returncode"] != 0:
            errors.append(
                {
                    "audio": audio,
                    "step": "region_overlap",
                    "returncode": result_region["returncode"],
                    "command": result_region["command"],
                    "stdout": result_region["stdout"],
                    "stderr": result_region["stderr"],
                }
            )
            print(f"  ERROR in region overlap: {audio}")
            continue

        print(f"  OK: {audio}")

    error_df = pd.DataFrame(errors)
    error_path = output_root / "batch_error_log.csv"
    error_df.to_csv(error_path, index=False)

    existing_audio_dirs = [
        path
        for path in sorted(output_root.iterdir())
        if path.is_dir()
        and (path / "centroid_profile").exists()
        and (path / "region_overlap").exists()
    ]

    summary_df = save_general_tables(output_root, existing_audio_dirs)
    make_plots(output_root, summary_df)

    print()
    print("=== Done ===")
    print(f"Audio folders found: {len(existing_audio_dirs)}")
    print(f"Errors: {len(error_df)}")
    print()
    print("General outputs:")
    print(f"- {output_root / 'general_audio_summary.csv'}")
    print(f"- {output_root / 'general_half_summary_all.csv'}")
    print(f"- {output_root / 'general_similarity_summary_all.csv'}")
    print(f"- {output_root / 'general_total_region_overlap_by_vowel_all.csv'}")
    print(f"- {output_root / 'general_pairwise_region_overlap_all.csv'}")
    print(f"- {output_root / 'general_ellipse_parameters_all.csv'}")
    print(f"- {output_root / 'general_mean_region_overlap_H1_H2_by_audio.png'}")
    print(f"- {output_root / 'general_vowel_overlap_mean_H1_H2.png'}")
    print(f"- {output_root / 'general_centroid_shift_vs_accuracy.png'}")
    print(f"- {output_root / 'general_overlap_change_vs_silhouette_change.png'}")
    print(f"- {error_path}")

    if len(error_df) > 0:
        print()
        print("Some errors occurred. See batch_error_log.csv.")


if __name__ == "__main__":
    main()
