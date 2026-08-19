#!/usr/bin/env python3

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform


VOWELS = ["a", "e", "i", "o", "u"]


def norm_col(s):
    return str(s).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def find_col(df, names):
    lookup = {norm_col(c): c for c in df.columns}
    for name in names:
        key = norm_col(name)
        if key in lookup:
            return lookup[key]
    return None


def normalize_vowel(v):
    if pd.isna(v):
        return None
    v = str(v).strip().lower()
    if v == "y":
        return "i"
    return v


def zscore(X):
    mean = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (X - mean) / sd


def silhouette_score_manual(X, labels):
    labels = np.asarray(labels)
    D = squareform(pdist(X, metric="euclidean"))

    values = []

    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False

        if same.sum() == 0:
            values.append(0.0)
            continue

        a = D[i, same].mean()

        b_vals = []
        for lab in sorted(set(labels)):
            if lab == labels[i]:
                continue
            other = labels == lab
            b_vals.append(D[i, other].mean())

        b = min(b_vals)

        if max(a, b) == 0:
            values.append(0.0)
        else:
            values.append((b - a) / max(a, b))

    return float(np.mean(values))


def short_audio_label(x, max_len=16):
    x = str(x)
    if len(x) <= max_len:
        return x
    return x[:max_len]


def make_scatter_plot(sub, vowel, f1_col, f2_col, cluster_col, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    for cluster in sorted(sub[cluster_col].unique()):
        s = sub[sub[cluster_col] == cluster]

        ax.scatter(
            s[f2_col],
            s[f1_col],
            s=70,
            alpha=0.8,
            label=f"Cluster {cluster}",
        )

        for _, row in s.iterrows():
            ax.text(
                row[f2_col],
                row[f1_col],
                short_audio_label(row["audio"]),
                fontsize=7,
                alpha=0.7,
            )

    ax.set_title(f"Speaker clusters for vowel /{vowel}/ using F1-F2 medians")
    ax.set_xlabel("F2 median (Hz) — inverted axis")
    ax.set_ylabel("F1 median (Hz) — inverted axis")

    # Conventional vowel-space orientation
    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.grid(alpha=0.25)
    ax.legend(title="Cluster")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_dendrogram_plot(Z, labels, vowel, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8))

    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
    )

    ax.set_title(f"Hierarchical clustering for vowel /{vowel}/ using F1-F2 medians")
    ax.set_ylabel("Ward distance")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def process_vowel(df, vowel, f1_col, f2_col, plots_dir, fixed_k, max_clusters):
    sub = df[df["vowel_norm"] == vowel].copy()
    sub = sub.dropna(subset=["audio", f1_col, f2_col])

    if len(sub) < 4:
        raise ValueError(f"Too few rows for vowel /{vowel}/.")

    X_raw = sub[[f1_col, f2_col]].to_numpy(dtype=float)
    X = zscore(X_raw)

    Z = linkage(X, method="ward")

    silhouette_rows = []

    max_k = min(max_clusters, len(sub) - 1)

    for k in range(2, max_k + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")
        sil = silhouette_score_manual(X, labels)

        silhouette_rows.append(
            {
                "vowel": vowel,
                "k": k,
                "silhouette_score": sil,
            }
        )

    silhouette_df = pd.DataFrame(silhouette_rows)

    if fixed_k is not None:
        chosen_k = fixed_k
    else:
        chosen_k = int(
            silhouette_df.sort_values("silhouette_score", ascending=False)
            .iloc[0]["k"]
        )

    clusters = fcluster(Z, t=chosen_k, criterion="maxclust")
    sub["cluster"] = clusters
    sub["chosen_k"] = chosen_k

    cluster_summary = (
        sub.groupby("cluster")
        .agg(
            vowel=("vowel_norm", "first"),
            chosen_k=("chosen_k", "first"),
            n_speakers=("audio", "count"),
            f1_median_mean=(f1_col, "mean"),
            f2_median_mean=(f2_col, "mean"),
            f1_median_sd=(f1_col, "std"),
            f2_median_sd=(f2_col, "std"),
            speakers=("audio", lambda x: "; ".join(map(str, x))),
        )
        .reset_index()
    )

    scatter_path = plots_dir / f"{vowel}_f1f2_speaker_clusters.png"
    dendrogram_path = plots_dir / f"{vowel}_dendrogram.png"

    make_scatter_plot(
        sub=sub,
        vowel=vowel,
        f1_col=f1_col,
        f2_col=f2_col,
        cluster_col="cluster",
        output_path=scatter_path,
    )

    make_dendrogram_plot(
        Z=Z,
        labels=sub["audio"].astype(str).to_list(),
        vowel=vowel,
        output_path=dendrogram_path,
    )

    return sub, silhouette_df, cluster_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/tables/vowel_stats_speaker_vowel.csv",
    )

    parser.add_argument(
        "--tables-dir",
        default="results/tables/points_by_vowel",
    )

    parser.add_argument(
        "--plots-dir",
        default="results/plots/points_by_vowel",
    )

    parser.add_argument(
        "--max-clusters",
        type=int,
        default=8,
        help="Maximum k tested for silhouette. Default: 8.",
    )

    parser.add_argument(
        "--fixed-k",
        type=int,
        default=None,
        help="Use fixed k for all vowels. If omitted, best silhouette k is chosen per vowel.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    tables_dir = Path(args.tables_dir)
    plots_dir = Path(args.plots_dir)

    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    audio_col = find_col(df, ["audio"])
    vowel_col = find_col(df, ["vowel", "label"])
    f1_col = find_col(df, ["f1_median"])
    f2_col = find_col(df, ["f2_median"])

    if audio_col is None:
        raise ValueError("Coluna audio não encontrada.")
    if vowel_col is None:
        raise ValueError("Coluna vowel/label não encontrada.")
    if f1_col is None:
        raise ValueError("Coluna f1_median não encontrada.")
    if f2_col is None:
        raise ValueError("Coluna f2_median não encontrada.")

    df = df.copy()
    df["audio"] = df[audio_col].astype(str)
    df["vowel_norm"] = df[vowel_col].apply(normalize_vowel)
    df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")
    df[f2_col] = pd.to_numeric(df[f2_col], errors="coerce")

    all_assignments = []
    all_silhouettes = []
    all_cluster_summaries = []

    for vowel in VOWELS:
        print(f"Processing vowel /{vowel}/")

        assignments, silhouettes, cluster_summary = process_vowel(
            df=df,
            vowel=vowel,
            f1_col=f1_col,
            f2_col=f2_col,
            plots_dir=plots_dir,
            fixed_k=args.fixed_k,
            max_clusters=args.max_clusters,
        )

        all_assignments.append(assignments)
        all_silhouettes.append(silhouettes)
        all_cluster_summaries.append(cluster_summary)

    assignments_df = pd.concat(all_assignments, ignore_index=True)
    silhouettes_df = pd.concat(all_silhouettes, ignore_index=True)
    cluster_summary_df = pd.concat(all_cluster_summaries, ignore_index=True)

    best_k_df = (
        silhouettes_df.sort_values(["vowel", "silhouette_score"], ascending=[True, False])
        .groupby("vowel")
        .head(1)
        .reset_index(drop=True)
        .rename(columns={"k": "best_k", "silhouette_score": "best_silhouette_score"})
    )

    assignments_path = tables_dir / "points_by_vowel_cluster_assignments.csv"
    silhouettes_path = tables_dir / "points_by_vowel_silhouette_by_k.csv"
    best_k_path = tables_dir / "points_by_vowel_best_k.csv"
    cluster_summary_path = tables_dir / "points_by_vowel_cluster_summary.csv"

    cols_to_save = ["audio", "vowel_norm", f1_col, f2_col, "chosen_k", "cluster"]
    assignments_df[cols_to_save].to_csv(assignments_path, index=False)
    silhouettes_df.to_csv(silhouettes_path, index=False)
    best_k_df.to_csv(best_k_path, index=False)
    cluster_summary_df.to_csv(cluster_summary_path, index=False)

    print()
    print("=== Per-vowel F1-F2 speaker clustering ===")
    print(f"Input: {input_path}")
    print()
    print("Best k by vowel:")
    print(best_k_df.to_string(index=False))
    print()
    print("Outputs:")
    print(f"- Cluster assignments: {assignments_path}")
    print(f"- Silhouette by k: {silhouettes_path}")
    print(f"- Best k: {best_k_path}")
    print(f"- Cluster summary: {cluster_summary_path}")
    print(f"- Plots directory: {plots_dir}")


if __name__ == "__main__":
    main()
