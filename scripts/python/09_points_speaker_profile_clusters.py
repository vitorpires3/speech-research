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


def zscore_matrix(X):
    mean = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)

    sd[sd == 0] = 1.0

    Xz = (X - mean) / sd

    return Xz, mean, sd


def pca_svd(Xz):
    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)

    scores = U * S
    loadings = Vt.T

    eigenvalues = (S ** 2) / (Xz.shape[0] - 1)
    explained = eigenvalues / eigenvalues.sum()

    return scores, loadings, explained


def silhouette_score_manual(X, labels):
    labels = np.asarray(labels)
    D = squareform(pdist(X, metric="euclidean"))

    values = []

    for i in range(len(X)):
        same_cluster = labels == labels[i]
        same_cluster[i] = False

        if same_cluster.sum() == 0:
            values.append(0.0)
            continue

        a = D[i, same_cluster].mean()

        b_candidates = []

        for lab in sorted(set(labels)):
            if lab == labels[i]:
                continue

            other_cluster = labels == lab
            b_candidates.append(D[i, other_cluster].mean())

        b = min(b_candidates)

        if max(a, b) == 0:
            values.append(0.0)
        else:
            values.append((b - a) / max(a, b))

    return float(np.mean(values))


def build_speaker_profile(df, audio_col, vowel_col, f1_col, f2_col):
    df = df.copy()

    df["vowel_norm"] = df[vowel_col].apply(normalize_vowel)
    df = df[df["vowel_norm"].isin(VOWELS)]

    df[f1_col] = pd.to_numeric(df[f1_col], errors="coerce")
    df[f2_col] = pd.to_numeric(df[f2_col], errors="coerce")

    f1_pivot = df.pivot_table(
        index=audio_col,
        columns="vowel_norm",
        values=f1_col,
        aggfunc="mean",
    )

    f2_pivot = df.pivot_table(
        index=audio_col,
        columns="vowel_norm",
        values=f2_col,
        aggfunc="mean",
    )

    f1_pivot = f1_pivot.reindex(columns=VOWELS)
    f2_pivot = f2_pivot.reindex(columns=VOWELS)

    f1_pivot.columns = [f"{v}_f1_median" for v in f1_pivot.columns]
    f2_pivot.columns = [f"{v}_f2_median" for v in f2_pivot.columns]

    profile = pd.concat([f1_pivot, f2_pivot], axis=1)
    profile = profile.reset_index().rename(columns={audio_col: "audio"})

    return profile


def make_pca_plot(scores_df, explained, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cluster in sorted(scores_df["cluster"].unique()):
        sub = scores_df[scores_df["cluster"] == cluster]

        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=70,
            alpha=0.8,
            label=f"Cluster {cluster}",
        )

        for _, row in sub.iterrows():
            label = str(row["audio"])

            if len(label) > 18:
                label = label[:18]

            ax.text(
                row["PC1"],
                row["PC2"],
                label,
                fontsize=7,
                alpha=0.7,
            )

    ax.axhline(0, linewidth=1, alpha=0.3)
    ax.axvline(0, linewidth=1, alpha=0.3)

    ax.set_title("Speaker clustering from F1/F2 vowel profile")
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")

    ax.grid(alpha=0.25)
    ax.legend(title="Cluster")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def make_dendrogram(Z, labels, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(13, 8))

    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
    )

    ax.set_title("Hierarchical clustering of speakers from F1/F2 vowel profile")
    ax.set_ylabel("Ward distance")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/tables/vowel_stats_speaker_vowel.csv",
    )

    parser.add_argument(
        "--tables-dir",
        default="results/tables",
    )

    parser.add_argument(
        "--plots-dir",
        default="results/plots",
    )

    parser.add_argument(
        "--n-clusters",
        type=int,
        default=4,
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

    profile = build_speaker_profile(
        df=df,
        audio_col=audio_col,
        vowel_col=vowel_col,
        f1_col=f1_col,
        f2_col=f2_col,
    )

    feature_cols = [c for c in profile.columns if c != "audio"]

    profile_clean = profile.dropna(subset=feature_cols).copy()

    if len(profile_clean) < 3:
        raise ValueError("Poucos locutores com dados completos para PCA/clustering.")

    X = profile_clean[feature_cols].to_numpy(dtype=float)

    Xz, means, sds = zscore_matrix(X)

    scores, loadings, explained = pca_svd(Xz)

    Z = linkage(Xz, method="ward")
    clusters = fcluster(Z, t=args.n_clusters, criterion="maxclust")

    silhouette = silhouette_score_manual(Xz, clusters)

    scores_df = pd.DataFrame(
        {
            "audio": profile_clean["audio"].to_numpy(),
            "cluster": clusters,
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "PC3": scores[:, 2] if scores.shape[1] > 2 else np.nan,
        }
    )

    loadings_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "PC1_loading": loadings[:, 0],
            "PC2_loading": loadings[:, 1],
            "PC3_loading": loadings[:, 2] if loadings.shape[1] > 2 else np.nan,
        }
    )

    explained_df = pd.DataFrame(
        {
            "PC": [f"PC{i + 1}" for i in range(len(explained))],
            "explained_variance_ratio": explained,
            "explained_variance_percent": explained * 100,
        }
    )

    cluster_summary = (
        scores_df.groupby("cluster")
        .agg(
            n_speakers=("audio", "count"),
            speakers=("audio", lambda x: "; ".join(map(str, x))),
            PC1_mean=("PC1", "mean"),
            PC2_mean=("PC2", "mean"),
        )
        .reset_index()
    )

    summary_df = pd.DataFrame(
        [
            {
                "n_speakers_used": len(profile_clean),
                "n_features": len(feature_cols),
                "n_clusters": args.n_clusters,
                "silhouette_score": silhouette,
                "PC1_explained_percent": explained[0] * 100,
                "PC2_explained_percent": explained[1] * 100,
                "PC1_PC2_total_percent": (explained[0] + explained[1]) * 100,
            }
        ]
    )

    profile_path = tables_dir / "points_speaker_profile_f1f2.csv"
    scores_path = tables_dir / "points_speaker_pca_scores.csv"
    loadings_path = tables_dir / "points_speaker_pca_loadings.csv"
    explained_path = tables_dir / "points_speaker_pca_explained_variance.csv"
    clusters_path = tables_dir / "points_speaker_clusters.csv"
    cluster_summary_path = tables_dir / "points_speaker_cluster_summary.csv"
    summary_path = tables_dir / "points_speaker_profile_test_summary.csv"

    profile_clean.to_csv(profile_path, index=False)
    scores_df.to_csv(scores_path, index=False)
    loadings_df.to_csv(loadings_path, index=False)
    explained_df.to_csv(explained_path, index=False)
    scores_df[["audio", "cluster"]].to_csv(clusters_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    pca_plot_path = plots_dir / "points_speaker_pca_clusters.png"
    dendrogram_path = plots_dir / "points_speaker_dendrogram.png"

    make_pca_plot(
        scores_df=scores_df,
        explained=explained,
        output_path=pca_plot_path,
    )

    make_dendrogram(
        Z=Z,
        labels=profile_clean["audio"].astype(str).to_list(),
        output_path=dendrogram_path,
    )

    print()
    print("=== Speaker profile clustering test ===")
    print(f"Input: {input_path}")
    print(f"Speakers used: {len(profile_clean)}")
    print(f"Features used: {len(feature_cols)}")
    print(f"Clusters requested: {args.n_clusters}")
    print()
    print("PCA explained variance:")
    print(explained_df.head(5).to_string(index=False))
    print()
    print(f"Silhouette score: {silhouette:.4f}")
    print()
    print("Cluster summary:")
    print(cluster_summary.to_string(index=False))
    print()
    print("Top PC1 loadings:")
    print(
        loadings_df.assign(abs_loading=loadings_df["PC1_loading"].abs())
        .sort_values("abs_loading", ascending=False)
        .head(10)
        .drop(columns=["abs_loading"])
        .to_string(index=False)
    )
    print()
    print("Top PC2 loadings:")
    print(
        loadings_df.assign(abs_loading=loadings_df["PC2_loading"].abs())
        .sort_values("abs_loading", ascending=False)
        .head(10)
        .drop(columns=["abs_loading"])
        .to_string(index=False)
    )
    print()
    print("Outputs:")
    print(f"- PCA plot: {pca_plot_path}")
    print(f"- Dendrogram: {dendrogram_path}")
    print(f"- Speaker profile: {profile_path}")
    print(f"- PCA scores: {scores_path}")
    print(f"- PCA loadings: {loadings_path}")
    print(f"- Explained variance: {explained_path}")
    print(f"- Cluster assignments: {clusters_path}")
    print(f"- Cluster summary: {cluster_summary_path}")
    print(f"- Test summary: {summary_path}")


if __name__ == "__main__":
    main()
