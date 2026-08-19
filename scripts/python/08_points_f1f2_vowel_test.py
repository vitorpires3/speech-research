#!/usr/bin/env python3
"""
First exploratory statistical test on points data.

Goal:
    Test whether vowels are separated in the F1-F2 acoustic space.

Input:
    data/processed/tables/vowel_stats_speaker_vowel.csv

Uses:
    f1_median and f2_median

Outputs:
    - F1 x F2 vowel plot
    - vowel centroids table
    - pairwise centroid distances table
    - blocked PERMANOVA result

The PERMANOVA permutation is blocked by speaker/audio:
    labels are shuffled only within each audio.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


DEFAULT_VOWELS = ["a", "e", "i", "o", "u"]


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


def normalize_vowel(value, group_y_with_i: bool = True) -> str | None:
    if pd.isna(value):
        return None

    v = str(value).strip().lower()

    if v == "":
        return None

    if group_y_with_i and v == "y":
        return "i"

    return v


def add_confidence_ellipse(ax, x, y, label=None):
    """
    Adds an approximate 95% confidence/data ellipse for a 2D cloud.

    This is descriptive, not the statistical test itself.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 3:
        return

    cov = np.cov(x, y)

    if not np.all(np.isfinite(cov)):
        return

    vals, vecs = np.linalg.eigh(cov)

    if np.any(vals <= 0):
        return

    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

    # sqrt(chi-square quantile 0.95 with 2 df) ~= sqrt(5.991)
    scale_95 = 2.4477

    width, height = 2 * scale_95 * np.sqrt(vals)

    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=1.8,
        alpha=0.8,
        label=label,
    )

    ax.add_patch(ellipse)


def compute_pseudo_f(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Euclidean PERMANOVA-style pseudo-F using sums of squares.

    For Euclidean distances, this is equivalent to comparing
    between-group and within-group multivariate sums of squares.
    """
    labels = np.asarray(labels)
    groups = [g for g in sorted(pd.unique(labels)) if pd.notna(g)]

    n = X.shape[0]
    g_count = len(groups)

    if n <= g_count or g_count < 2:
        return {
            "pseudo_F": np.nan,
            "R2": np.nan,
            "SS_total": np.nan,
            "SS_between": np.nan,
            "SS_within": np.nan,
            "df_between": np.nan,
            "df_within": np.nan,
        }

    grand_centroid = X.mean(axis=0)

    ss_total = float(np.sum((X - grand_centroid) ** 2))

    ss_within = 0.0

    for group in groups:
        Xg = X[labels == group]

        if len(Xg) == 0:
            continue

        centroid = Xg.mean(axis=0)
        ss_within += float(np.sum((Xg - centroid) ** 2))

    ss_between = ss_total - ss_within

    df_between = g_count - 1
    df_within = n - g_count

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    pseudo_f = ms_between / ms_within if ms_within > 0 else np.nan
    r2 = ss_between / ss_total if ss_total > 0 else np.nan

    return {
        "pseudo_F": float(pseudo_f),
        "R2": float(r2),
        "SS_total": float(ss_total),
        "SS_between": float(ss_between),
        "SS_within": float(ss_within),
        "df_between": int(df_between),
        "df_within": int(df_within),
    }


def permute_labels_within_blocks(labels: np.ndarray, blocks: np.ndarray, rng) -> np.ndarray:
    labels_perm = labels.copy()

    for block in pd.unique(blocks):
        idx = np.where(blocks == block)[0]

        if len(idx) <= 1:
            continue

        labels_perm[idx] = rng.permutation(labels_perm[idx])

    return labels_perm


def blocked_permanova(
    X: np.ndarray,
    labels: np.ndarray,
    blocks: np.ndarray,
    permutations: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)

    observed = compute_pseudo_f(X, labels)
    observed_f = observed["pseudo_F"]

    permuted_f_values = []

    for _ in range(permutations):
        permuted_labels = permute_labels_within_blocks(labels, blocks, rng)
        perm_result = compute_pseudo_f(X, permuted_labels)
        permuted_f_values.append(perm_result["pseudo_F"])

    permuted_f_values = np.asarray(permuted_f_values, dtype=float)
    valid = np.isfinite(permuted_f_values)

    if not np.isfinite(observed_f) or valid.sum() == 0:
        p_value = np.nan
    else:
        p_value = (1 + np.sum(permuted_f_values[valid] >= observed_f)) / (1 + valid.sum())

    result = observed.copy()
    result["p_value"] = float(p_value)
    result["permutations"] = int(permutations)
    result["valid_permutations"] = int(valid.sum())

    return result


def make_centroid_tables(df: pd.DataFrame, vowel_col: str, f1_col: str, f2_col: str):
    centroids = (
        df.groupby(vowel_col)
        .agg(
            n=("audio", "count"),
            f1_median_mean=(f1_col, "mean"),
            f2_median_mean=(f2_col, "mean"),
            f1_median_sd=(f1_col, "std"),
            f2_median_sd=(f2_col, "std"),
        )
        .reset_index()
        .rename(columns={vowel_col: "vowel"})
    )

    distance_rows = []

    centroid_points = {
        row["vowel"]: np.array([row["f1_median_mean"], row["f2_median_mean"]], dtype=float)
        for _, row in centroids.iterrows()
    }

    for v1, v2 in combinations(centroid_points.keys(), 2):
        dist = float(np.linalg.norm(centroid_points[v1] - centroid_points[v2]))

        distance_rows.append(
            {
                "vowel_1": v1,
                "vowel_2": v2,
                "centroid_distance_f1f2": dist,
            }
        )

    distances = pd.DataFrame(distance_rows)
    distances = distances.sort_values("centroid_distance_f1f2", ascending=False)

    return centroids, distances


def make_plot(
    df: pd.DataFrame,
    vowel_col: str,
    f1_col: str,
    f2_col: str,
    output_path: Path,
    title: str,
    show: bool,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    vowels = [v for v in DEFAULT_VOWELS if v in set(df[vowel_col])]
    remaining = [v for v in sorted(pd.unique(df[vowel_col])) if v not in vowels]
    vowels = vowels + remaining

    for vowel in vowels:
        sub = df[df[vowel_col] == vowel]

        x = sub[f2_col].to_numpy(dtype=float)
        y = sub[f1_col].to_numpy(dtype=float)

        ax.scatter(
            x,
            y,
            s=45,
            alpha=0.75,
            label=f"/{vowel}/",
        )

        add_confidence_ellipse(ax, x, y)

        centroid_x = np.nanmean(x)
        centroid_y = np.nanmean(y)

        ax.text(
            centroid_x,
            centroid_y,
            f"/{vowel}/",
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
        )

    ax.set_title(title)
    ax.set_xlabel("F2 median (Hz) — eixo invertido")
    ax.set_ylabel("F1 median (Hz) — eixo invertido")

    # Classical vowel plot orientation.
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.grid(True, alpha=0.25)
    ax.legend(title="Vogal", loc="best")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)

    if show:
        plt.show()

    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="F1-F2 vowel separation test using points statistics."
    )

    parser.add_argument(
        "--input",
        default="data/processed/tables/vowel_stats_speaker_vowel.csv",
        help="Input CSV with points statistics.",
    )

    parser.add_argument(
        "--plot-output",
        default="results/plots/points_f1f2_vowel_clusters.png",
        help="Output path for the F1-F2 plot.",
    )

    parser.add_argument(
        "--tables-dir",
        default="results/tables",
        help="Directory for output tables.",
    )

    parser.add_argument(
        "--f1-col",
        default="f1_median",
        help="Column used as F1 feature. Default: f1_median.",
    )

    parser.add_argument(
        "--f2-col",
        default="f2_median",
        help="Column used as F2 feature. Default: f2_median.",
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=9999,
        help="Number of permutations for blocked PERMANOVA. Default: 9999.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    parser.add_argument(
        "--min-tokens",
        type=int,
        default=0,
        help="Minimum token count per audio-vowel row. Default: 0.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot after saving.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    plot_output = Path(args.plot_output)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    audio_col = find_col(df, ["audio"])
    vowel_col = find_col(df, ["vowel", "label"])
    n_tokens_col = find_col(df, ["n_tokens", "ntokens"])

    f1_col = find_col(df, [args.f1_col])
    f2_col = find_col(df, [args.f2_col])

    if audio_col is None:
        raise ValueError("Could not find audio column.")

    if vowel_col is None:
        raise ValueError("Could not find vowel/label column.")

    if f1_col is None:
        raise ValueError(f"Could not find F1 column: {args.f1_col}")

    if f2_col is None:
        raise ValueError(f"Could not find F2 column: {args.f2_col}")

    df = df.copy()

    df["audio"] = df[audio_col].astype(str)
    df["vowel_test"] = df[vowel_col].apply(lambda x: normalize_vowel(x, group_y_with_i=True))

    df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")
    df[f2_col] = pd.to_numeric(df[f2_col], errors="coerce")

    df = df[df["vowel_test"].isin(DEFAULT_VOWELS)]

    if n_tokens_col is not None and args.min_tokens > 0:
        df[n_tokens_col] = pd.to_numeric(df[n_tokens_col], errors="coerce")
        df = df[df[n_tokens_col] >= args.min_tokens]

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["audio", "vowel_test", f1_col, f2_col])

    if df.empty:
        raise ValueError("No valid data after filtering.")

    clean_output = tables_dir / "points_f1f2_vowel_test_data.csv"
    df[["audio", "vowel_test", f1_col, f2_col]].to_csv(clean_output, index=False)

    X = df[[f1_col, f2_col]].to_numpy(dtype=float)
    labels = df["vowel_test"].to_numpy()
    blocks = df["audio"].to_numpy()

    permanova_result = blocked_permanova(
        X=X,
        labels=labels,
        blocks=blocks,
        permutations=args.permutations,
        seed=args.seed,
    )

    permanova_df = pd.DataFrame([permanova_result])
    permanova_output = tables_dir / "points_f1f2_vowel_permanova.csv"
    permanova_df.to_csv(permanova_output, index=False)

    centroids, distances = make_centroid_tables(
        df=df,
        vowel_col="vowel_test",
        f1_col=f1_col,
        f2_col=f2_col,
    )

    centroids_output = tables_dir / "points_f1f2_vowel_centroids.csv"
    distances_output = tables_dir / "points_f1f2_vowel_centroid_distances.csv"

    centroids.to_csv(centroids_output, index=False)
    distances.to_csv(distances_output, index=False)

    title = "Vowel separation in F1-F2 space from points data"

    make_plot(
        df=df,
        vowel_col="vowel_test",
        f1_col=f1_col,
        f2_col=f2_col,
        output_path=plot_output,
        title=title,
        show=args.show,
    )

    print("")
    print("=== F1-F2 vowel test ===")
    print(f"Input: {input_path}")
    print(f"Rows used: {len(df)}")
    print(f"Features: {f1_col}, {f2_col}")
    print("")
    print("PERMANOVA blocked by audio:")
    print(permanova_df.to_string(index=False))
    print("")
    print("Vowel centroids:")
    print(centroids.to_string(index=False))
    print("")
    print("Largest centroid distances:")
    print(distances.head(10).to_string(index=False))
    print("")
    print("Outputs:")
    print(f"- Plot: {plot_output}")
    print(f"- Test data: {clean_output}")
    print(f"- PERMANOVA: {permanova_output}")
    print(f"- Centroids: {centroids_output}")
    print(f"- Pairwise centroid distances: {distances_output}")


if __name__ == "__main__":
    main()