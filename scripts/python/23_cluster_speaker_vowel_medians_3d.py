#!/usr/bin/env python3
"""
Cluster speaker-level vowel medians in F1-F2-F3 space.

For each vowel:
- reads speaker-level medians
- standardizes f1_median, f2_median, f3_median
- applies Ward hierarchical clustering
- tests k = 2..max_clusters
- selects the best k by average silhouette score
- saves dendrograms, silhouette table, cluster assignments and 3D plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster


def optional_plotly():
    try:
        import plotly.graph_objects as go
        return go
    except ImportError:
        return None


def detect_audio_col(df: pd.DataFrame) -> str:
    for col in ["audio", "file", "file_name", "source_file"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find audio column. Columns: {list(df.columns)}")


def load_data(args) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    audio_col = detect_audio_col(df)

    required = {"vowel", "f1_median", "f2_median", "f3_median"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df[[audio_col, "vowel", "f1_median", "f2_median", "f3_median"]].copy()
    out = out.rename(columns={audio_col: "audio"})

    out["vowel"] = out["vowel"].astype(str).str.lower().replace({"y": "i"})
    out = out[out["vowel"].isin(args.vowels)].copy()

    for col in ["f1_median", "f2_median", "f3_median"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["audio", "vowel", "f1_median", "f2_median", "f3_median"])
    out = out[
        (out["f1_median"] > 0)
        & (out["f2_median"] > 0)
        & (out["f3_median"] > 0)
    ].copy()

    return out


def plot_dendrogram(Z, labels, vowel: str, output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6))

    dendrogram(
        Z,
        labels=labels,
        leaf_rotation=90,
        leaf_font_size=7,
        ax=ax,
    )

    ax.set_title(f"Dendrogramme Ward — voyelle /{vowel}/ — F1-F2-F3")
    ax.set_ylabel("Distance de fusion Ward")
    fig.tight_layout()

    fig.savefig(output_dir / f"vowel_{vowel}_ward_dendrogram_f1f2f3.png", dpi=250)
    fig.savefig(output_dir / f"vowel_{vowel}_ward_dendrogram_f1f2f3.pdf")
    plt.close(fig)


def plot_3d_clusters_static(d: pd.DataFrame, vowel: str, output_dir: Path):
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    clusters = sorted(d["cluster"].unique())

    for cluster in clusters:
        dc = d[d["cluster"] == cluster]
        ax.scatter(
            dc["f2_median"],
            dc["f1_median"],
            dc["f3_median"],
            s=50,
            alpha=0.85,
            label=f"cluster {cluster}",
        )

        for _, row in dc.iterrows():
            ax.text(
                row["f2_median"],
                row["f1_median"],
                row["f3_median"],
                str(row["speaker_code"]),
                fontsize=7,
            )

    ax.set_title(f"Clusters Ward — voyelle /{vowel}/ — F1-F2-F3")
    ax.set_xlabel("F2 median (Hz)")
    ax.set_ylabel("F1 median (Hz)")
    ax.set_zlabel("F3 median (Hz)")

    ax.invert_xaxis()
    ax.invert_yaxis()

    ax.legend(fontsize=8)
    fig.tight_layout()

    fig.savefig(output_dir / f"vowel_{vowel}_clusters_f1f2f3_3d.png", dpi=250)
    fig.savefig(output_dir / f"vowel_{vowel}_clusters_f1f2f3_3d.pdf")
    plt.close(fig)


def plot_3d_clusters_interactive(d: pd.DataFrame, vowel: str, output_dir: Path):
    go = optional_plotly()

    if go is None:
        return

    fig = go.Figure()

    for cluster in sorted(d["cluster"].unique()):
        dc = d[d["cluster"] == cluster].copy()

        hover = []

        for _, row in dc.iterrows():
            hover.append(
                "<br>".join([
                    f"speaker code: {row['speaker_code']}",
                    f"audio: {row['audio']}",
                    f"vowel: /{vowel}/",
                    f"cluster: {row['cluster']}",
                    f"F1 median: {row['f1_median']:.1f} Hz",
                    f"F2 median: {row['f2_median']:.1f} Hz",
                    f"F3 median: {row['f3_median']:.1f} Hz",
                ])
            )

        fig.add_trace(
            go.Scatter3d(
                x=dc["f2_median"],
                y=dc["f1_median"],
                z=dc["f3_median"],
                mode="markers+text",
                text=dc["speaker_code"].astype(str),
                textposition="top center",
                hovertext=hover,
                hovertemplate="%{hovertext}<extra></extra>",
                name=f"cluster {cluster}",
                marker=dict(size=6),
            )
        )

    fig.update_layout(
        title=f"Clusters Ward — voyelle /{vowel}/ — F1-F2-F3",
        template="plotly_white",
        width=1000,
        height=820,
        scene=dict(
            xaxis=dict(title="F2 median (Hz)", autorange="reversed"),
            yaxis=dict(title="F1 median (Hz)", autorange="reversed"),
            zaxis=dict(title="F3 median (Hz)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )

    fig.write_html(
        output_dir / f"vowel_{vowel}_clusters_f1f2f3_3d.html",
        include_plotlyjs="cdn",
    )


def process_vowel(df: pd.DataFrame, vowel: str, args, output_dir: Path):
    d = df[df["vowel"] == vowel].copy()
    d = d.sort_values("audio").reset_index(drop=True)

    if len(d) < 4:
        print(f"Skipping /{vowel}/: not enough points.")
        return [], pd.DataFrame(), pd.DataFrame()

    d["speaker_code"] = np.arange(1, len(d) + 1)

    features = ["f1_median", "f2_median", "f3_median"]
    X = d[features].to_numpy()

    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    Z = linkage(Xz, method="ward")

    plot_dendrogram(Z, labels=d["speaker_code"].astype(str).tolist(), vowel=vowel, output_dir=output_dir)

    max_k = min(args.max_clusters, len(d) - 1)

    metric_rows = []

    for k in range(2, max_k + 1):
        labels = fcluster(Z, t=k, criterion="maxclust")

        if len(np.unique(labels)) < 2:
            continue

        sil = silhouette_score(Xz, labels)
        ch = calinski_harabasz_score(Xz, labels)
        db = davies_bouldin_score(Xz, labels)

        metric_rows.append({
            "vowel": vowel,
            "k": k,
            "silhouette_score": sil,
            "calinski_harabasz_score": ch,
            "davies_bouldin_score": db,
            "n_clusters_observed": len(np.unique(labels)),
        })

    metrics = pd.DataFrame(metric_rows)

    if metrics.empty:
        print(f"Skipping /{vowel}/: no valid silhouette values.")
        return [], pd.DataFrame(), pd.DataFrame()

    # Main criterion: highest silhouette.
    # Tie-breaker: smaller k.
    best_row = (
        metrics.sort_values(
            ["silhouette_score", "k"],
            ascending=[False, True],
        )
        .iloc[0]
    )

    best_k = int(best_row["k"])
    best_labels = fcluster(Z, t=best_k, criterion="maxclust")

    d["cluster"] = best_labels

    # Make cluster labels stable by ordering clusters by F2/F1/F3 center.
    centers = (
        d.groupby("cluster")[features]
        .mean()
        .sort_values(["f2_median", "f1_median", "f3_median"])
        .reset_index()
    )

    remap = {old: new for new, old in enumerate(centers["cluster"], start=1)}
    d["cluster"] = d["cluster"].map(remap)

    cluster_summary = (
        d.groupby(["vowel", "cluster"])
        .agg(
            n_speakers=("audio", "size"),
            f1_median_mean=("f1_median", "mean"),
            f2_median_mean=("f2_median", "mean"),
            f3_median_mean=("f3_median", "mean"),
            f1_median_median=("f1_median", "median"),
            f2_median_median=("f2_median", "median"),
            f3_median_median=("f3_median", "median"),
        )
        .reset_index()
    )

    plot_3d_clusters_static(d, vowel, output_dir)
    plot_3d_clusters_interactive(d, vowel, output_dir)

    best = {
        "vowel": vowel,
        "best_k_by_silhouette": best_k,
        "best_silhouette_score": float(best_row["silhouette_score"]),
        "calinski_harabasz_at_best_k": float(best_row["calinski_harabasz_score"]),
        "davies_bouldin_at_best_k": float(best_row["davies_bouldin_score"]),
        "n_speakers": len(d),
    }

    return [best], metrics, d[["audio", "speaker_code", "vowel", "cluster", *features]], cluster_summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--max-clusters", type=int, default=8)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args)

    all_best = []
    all_metrics = []
    all_assignments = []
    all_summaries = []

    for vowel in args.vowels:
        best, metrics, assignments, summary = process_vowel(df, vowel, args, output_dir)

        all_best.extend(best)

        if not metrics.empty:
            all_metrics.append(metrics)

        if not assignments.empty:
            all_assignments.append(assignments)

        if not summary.empty:
            all_summaries.append(summary)

    best_df = pd.DataFrame(all_best)
    metrics_df = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    assignments_df = pd.concat(all_assignments, ignore_index=True) if all_assignments else pd.DataFrame()
    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()

    best_df.to_csv(output_dir / "f1f2f3_best_k_by_vowel.csv", index=False)
    metrics_df.to_csv(output_dir / "f1f2f3_silhouette_by_k.csv", index=False)
    assignments_df.to_csv(output_dir / "f1f2f3_cluster_assignments.csv", index=False)
    summary_df.to_csv(output_dir / "f1f2f3_cluster_summary.csv", index=False)

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print("")
    print("Best k by vowel:")
    if not best_df.empty:
        print(best_df.to_string(index=False))
    else:
        print("No results.")


if __name__ == "__main__":
    main()
