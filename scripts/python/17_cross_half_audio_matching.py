#!/usr/bin/env python3

from pathlib import Path
import argparse
import itertools
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


VOWELS = ["i", "e", "a", "o", "u"]
PAIRS = list(itertools.combinations(VOWELS, 2))


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def is_pos_number(x):
    try:
        x = float(x)
        return np.isfinite(x) and x > 0
    except Exception:
        return False


def safe_log(x):
    if is_pos_number(x):
        return math.log(float(x))
    return np.nan


def read_required_csv(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    return pd.read_csv(path)


def pivot_rows(rows):
    return pd.DataFrame(rows)


def build_centroid_features(ellipse_df):
    rows = []

    for (audio, half), sub in ellipse_df.groupby(["audio", "half"]):
        row = {
            "audio": audio,
            "half": half,
        }

        for vowel in VOWELS:
            v = sub[sub["vowel"] == vowel]

            if v.empty:
                row[f"centroid_{vowel}_f1"] = np.nan
                row[f"centroid_{vowel}_f2"] = np.nan
            else:
                r = v.iloc[0]
                row[f"centroid_{vowel}_f1"] = r.get("center_f1", np.nan)
                row[f"centroid_{vowel}_f2"] = r.get("center_f2", np.nan)

        rows.append(row)

    return pivot_rows(rows)


def build_geometry_features(ellipse_df):
    rows = []

    for (audio, half), sub in ellipse_df.groupby(["audio", "half"]):
        row = {
            "audio": audio,
            "half": half,
        }

        centers = {}

        for vowel in VOWELS:
            v = sub[sub["vowel"] == vowel]

            if not v.empty:
                r = v.iloc[0]
                centers[vowel] = (
                    float(r.get("center_f1", np.nan)),
                    float(r.get("center_f2", np.nan)),
                )

        for v1, v2 in PAIRS:
            col = f"geom_{v1}_{v2}_distance"

            if v1 in centers and v2 in centers:
                f1a, f2a = centers[v1]
                f1b, f2b = centers[v2]

                if np.isfinite(f1a) and np.isfinite(f2a) and np.isfinite(f1b) and np.isfinite(f2b):
                    row[col] = math.sqrt((f1a - f1b) ** 2 + (f2a - f2b) ** 2)
                else:
                    row[col] = np.nan
            else:
                row[col] = np.nan

        rows.append(row)

    return pivot_rows(rows)


def build_ellipse_features(ellipse_df):
    rows = []

    for (audio, half), sub in ellipse_df.groupby(["audio", "half"]):
        row = {
            "audio": audio,
            "half": half,
        }

        for vowel in VOWELS:
            v = sub[sub["vowel"] == vowel]

            if v.empty:
                row[f"ellipse_{vowel}_log_area"] = np.nan
                row[f"ellipse_{vowel}_log_width"] = np.nan
                row[f"ellipse_{vowel}_log_height"] = np.nan
                row[f"ellipse_{vowel}_log_det_cov"] = np.nan
                row[f"ellipse_{vowel}_sin_2angle"] = np.nan
                row[f"ellipse_{vowel}_cos_2angle"] = np.nan
                continue

            r = v.iloc[0]

            angle = float(r.get("angle_degrees", np.nan))

            if np.isfinite(angle):
                angle_rad = math.radians(angle)
                row[f"ellipse_{vowel}_sin_2angle"] = math.sin(2 * angle_rad)
                row[f"ellipse_{vowel}_cos_2angle"] = math.cos(2 * angle_rad)
            else:
                row[f"ellipse_{vowel}_sin_2angle"] = np.nan
                row[f"ellipse_{vowel}_cos_2angle"] = np.nan

            row[f"ellipse_{vowel}_log_area"] = safe_log(r.get("ellipse_area_analytic", np.nan))
            row[f"ellipse_{vowel}_log_width"] = safe_log(r.get("width_f2_axis", np.nan))
            row[f"ellipse_{vowel}_log_height"] = safe_log(r.get("height_f1_axis", np.nan))
            row[f"ellipse_{vowel}_log_det_cov"] = safe_log(r.get("det_cov", np.nan))

        rows.append(row)

    return pivot_rows(rows)


def build_overlap_features(total_overlap_df, pairwise_overlap_df):
    rows = []

    all_keys = sorted(
        set(zip(total_overlap_df["audio"], total_overlap_df["half"]))
        &
        set(zip(pairwise_overlap_df["audio"], pairwise_overlap_df["half"]))
    )

    for audio, half in all_keys:
        total_sub = total_overlap_df[
            (total_overlap_df["audio"] == audio)
            & (total_overlap_df["half"] == half)
        ]

        pair_sub = pairwise_overlap_df[
            (pairwise_overlap_df["audio"] == audio)
            & (pairwise_overlap_df["half"] == half)
        ]

        row = {
            "audio": audio,
            "half": half,
        }

        for vowel in VOWELS:
            v = total_sub[total_sub["vowel"] == vowel]

            if v.empty:
                row[f"overlap_{vowel}_with_any"] = np.nan
                row[f"unique_{vowel}_percent"] = np.nan
            else:
                r = v.iloc[0]
                row[f"overlap_{vowel}_with_any"] = r.get("overlap_percent_with_any_other_vowel", np.nan)
                row[f"unique_{vowel}_percent"] = r.get("unique_percent_not_overlapped", np.nan)

        for v1, v2 in PAIRS:
            p = pair_sub[
                ((pair_sub["vowel_1"] == v1) & (pair_sub["vowel_2"] == v2))
                |
                ((pair_sub["vowel_1"] == v2) & (pair_sub["vowel_2"] == v1))
            ]

            if p.empty:
                row[f"overlap_{v1}_{v2}_jaccard"] = np.nan
            else:
                row[f"overlap_{v1}_{v2}_jaccard"] = p.iloc[0].get("jaccard_overlap_percent", np.nan)

        rows.append(row)

    return pivot_rows(rows)


def get_common_audios(feature_blocks):
    common = None

    for name, df in feature_blocks.items():
        h1 = set(df[df["half"] == "H1"]["audio"])
        h2 = set(df[df["half"] == "H2"]["audio"])
        both = h1 & h2

        if common is None:
            common = both
        else:
            common = common & both

    if common is None:
        return []

    return sorted(common)


def make_feature_matrix(block_df, audio_list, block_name):
    df = block_df.copy()

    if "audio" not in df.columns or "half" not in df.columns:
        raise ValueError(f"{block_name}: missing audio or half column")

    feature_cols = [c for c in df.columns if c not in ["audio", "half"]]

    if not feature_cols:
        raise ValueError(f"{block_name}: no feature columns")

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

    h1 = (
        df[df["half"] == "H1"]
        .drop_duplicates(subset=["audio"])
        .set_index("audio")
        .reindex(audio_list)
    )

    h2 = (
        df[df["half"] == "H2"]
        .drop_duplicates(subset=["audio"])
        .set_index("audio")
        .reindex(audio_list)
    )

    X = pd.concat([h1[feature_cols], h2[feature_cols]], axis=0)

    # Remove columns entirely missing.
    X = X.dropna(axis=1, how="all")

    if X.shape[1] == 0:
        raise ValueError(f"{block_name}: all feature columns are missing")

    # Fill missing values with the column median.
    for col in X.columns:
        median = X[col].median()

        if not np.isfinite(median):
            median = 0.0

        X[col] = X[col].fillna(median)

    # Standardize features.
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)

    keep = std[std > 0].index.tolist()

    if not keep:
        raise ValueError(f"{block_name}: all features have zero variance")

    X = (X[keep] - mean[keep]) / std[keep]

    n = len(audio_list)

    X_h1 = X.iloc[:n].to_numpy(dtype=float)
    X_h2 = X.iloc[n:].to_numpy(dtype=float)

    return X_h1, X_h2, keep


def rms_distance_matrix(X_h1, X_h2):
    diff = X_h1[:, None, :] - X_h2[None, :, :]
    return np.sqrt(np.mean(diff ** 2, axis=2))


def normalize_matrix_by_median(D):
    finite = D[np.isfinite(D)]

    if len(finite) == 0:
        return D

    med = np.median(finite)

    if not np.isfinite(med) or med <= 0:
        med = 1.0

    return D / med


def build_distance_matrices(feature_blocks, audio_list, normalize_components=True):
    matrices = {}
    feature_counts = {}

    for name, block_df in feature_blocks.items():
        X_h1, X_h2, used_cols = make_feature_matrix(block_df, audio_list, name)

        D = rms_distance_matrix(X_h1, X_h2)

        if normalize_components:
            D = normalize_matrix_by_median(D)

        matrices[name] = D
        feature_counts[name] = len(used_cols)

    return matrices, feature_counts


def combine_matrices(matrices, weights):
    total = None
    w_sum = 0.0

    for name, D in matrices.items():
        w = float(weights.get(name, 0.0))

        if w < 0:
            raise ValueError(f"Negative weight: {name}={w}")

        if w == 0:
            continue

        if total is None:
            total = w * D
        else:
            total = total + w * D

        w_sum += w

    if total is None or w_sum <= 0:
        raise ValueError("At least one weight must be positive.")

    return total / w_sum


def rank_candidates(total_cost, audio_list):
    rows = []

    for i, h1_audio in enumerate(audio_list):
        order = np.argsort(total_cost[i])

        for rank, j in enumerate(order, start=1):
            h2_audio = audio_list[j]

            rows.append(
                {
                    "h1_audio": h1_audio,
                    "candidate_h2_audio": h2_audio,
                    "rank": rank,
                    "cost": float(total_cost[i, j]),
                    "is_true_match": h1_audio == h2_audio,
                }
            )

    return pd.DataFrame(rows)


def hungarian_assignments(total_cost, matrices, audio_list):
    row_idx, col_idx = linear_sum_assignment(total_cost)

    rankings = rank_candidates(total_cost, audio_list)

    audio_to_index = {audio: i for i, audio in enumerate(audio_list)}

    rows = []

    for i, j in zip(row_idx, col_idx):
        h1_audio = audio_list[i]
        predicted_h2 = audio_list[j]
        true_h2 = h1_audio
        true_j = audio_to_index[true_h2]

        ranking_sub = rankings[rankings["h1_audio"] == h1_audio]
        true_row = ranking_sub[ranking_sub["candidate_h2_audio"] == true_h2]
        pred_row = ranking_sub[ranking_sub["candidate_h2_audio"] == predicted_h2]

        sorted_costs = np.sort(total_cost[i])
        best_cost = float(sorted_costs[0])
        second_cost = float(sorted_costs[1]) if len(sorted_costs) > 1 else np.nan

        row = {
            "h1_audio": h1_audio,
            "predicted_h2_audio": predicted_h2,
            "true_h2_audio": true_h2,
            "is_correct": predicted_h2 == true_h2,
            "assigned_cost": float(total_cost[i, j]),
            "true_match_cost": float(total_cost[i, true_j]),
            "true_match_rank": int(true_row["rank"].iloc[0]) if not true_row.empty else np.nan,
            "predicted_match_rank_for_this_h1": int(pred_row["rank"].iloc[0]) if not pred_row.empty else np.nan,
            "best_candidate_cost_for_this_h1": best_cost,
            "second_best_candidate_cost_for_this_h1": second_cost,
            "margin_best_to_second": second_cost - best_cost if np.isfinite(second_cost) else np.nan,
        }

        for name, D in matrices.items():
            row[f"{name}_distance_assigned"] = float(D[i, j])
            row[f"{name}_distance_true"] = float(D[i, true_j])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("h1_audio"), rankings


def make_summary(assignments, rankings, weights, feature_counts):
    n = len(assignments)

    true_ranks = assignments["true_match_rank"].astype(float)

    true_costs = rankings[rankings["is_true_match"]]["cost"]
    false_costs = rankings[~rankings["is_true_match"]]["cost"]

    row = {
        "n_audios": n,
        "hungarian_correct": int(assignments["is_correct"].sum()),
        "hungarian_accuracy": float(assignments["is_correct"].mean()),
        "row_top1_accuracy": float((true_ranks <= 1).mean()),
        "row_top3_accuracy": float((true_ranks <= 3).mean()),
        "row_top5_accuracy": float((true_ranks <= 5).mean()),
        "mean_true_match_rank": float(true_ranks.mean()),
        "median_true_match_rank": float(true_ranks.median()),
        "mean_reciprocal_rank": float((1.0 / true_ranks).mean()),
        "mean_true_cost": float(true_costs.mean()),
        "mean_false_cost": float(false_costs.mean()),
        "median_true_cost": float(true_costs.median()),
        "median_false_cost": float(false_costs.median()),
    }

    for name, w in weights.items():
        row[f"weight_{name}"] = float(w)

    for name, n_features in feature_counts.items():
        row[f"n_features_{name}"] = int(n_features)

    return pd.DataFrame([row])


def save_matrix_csv(matrix, audio_list, path):
    df = pd.DataFrame(matrix, index=audio_list, columns=audio_list)
    df.index.name = "H1_audio"
    df.columns.name = "H2_audio"
    df.to_csv(path)


def plot_cost_heatmap(matrix, audio_list, path):
    fig_size = max(10, len(audio_list) * 0.28)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(audio_list)))
    ax.set_yticks(np.arange(len(audio_list)))

    ax.set_xticklabels(audio_list, rotation=90, fontsize=5)
    ax.set_yticklabels(audio_list, fontsize=5)

    ax.set_xlabel("H2 candidates")
    ax.set_ylabel("H1 audios")
    ax.set_title("Cross-half matching cost matrix")

    fig.colorbar(im, ax=ax, label="Cost")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_true_rank(assignments, path):
    df = assignments.sort_values("true_match_rank", ascending=False).copy()

    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x, df["true_match_rank"])
    ax.axhline(1, linewidth=1)

    ax.set_xticks(x)
    ax.set_xticklabels(df["h1_audio"], rotation=90, fontsize=7)

    ax.set_ylabel("Rank of true H2 match")
    ax.set_title("Rank of true match for each H1 audio")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_true_false_costs(rankings, path):
    true_costs = rankings[rankings["is_true_match"]]["cost"].to_numpy(dtype=float)
    false_costs = rankings[~rankings["is_true_match"]]["cost"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(false_costs, bins=40, alpha=0.65, label="False pairs")
    ax.hist(true_costs, bins=20, alpha=0.85, label="True pairs")

    ax.set_xlabel("Cost")
    ax.set_ylabel("Count")
    ax.set_title("True-pair costs vs false-pair costs")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-root",
        default="results/all_audio_half_region_profiles_level80",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
    )

    parser.add_argument("--w-centroids", type=float, default=0.25)
    parser.add_argument("--w-geometry", type=float, default=0.25)
    parser.add_argument("--w-ellipse", type=float, default=0.25)
    parser.add_argument("--w-overlap", type=float, default=0.25)

    parser.add_argument(
        "--no-normalize-components",
        action="store_true",
        help="Do not normalize each component distance matrix by its median.",
    )

    args = parser.parse_args()

    input_root = Path(args.input_root)

    if args.output_dir is None:
        output_dir = input_root / "cross_half_matching_equal_weights"
    else:
        output_dir = Path(args.output_dir)

    ensure_dir(output_dir)
    ensure_dir(output_dir / "component_distance_matrices")

    ellipse_path = input_root / "general_ellipse_parameters_all.csv"
    total_overlap_path = input_root / "general_total_region_overlap_by_vowel_all.csv"
    pairwise_overlap_path = input_root / "general_pairwise_region_overlap_all.csv"

    ellipse_df = read_required_csv(ellipse_path)
    total_overlap_df = read_required_csv(total_overlap_path)
    pairwise_overlap_df = read_required_csv(pairwise_overlap_path)

    feature_blocks = {
        "centroids": build_centroid_features(ellipse_df),
        "geometry": build_geometry_features(ellipse_df),
        "ellipse": build_ellipse_features(ellipse_df),
        "overlap": build_overlap_features(total_overlap_df, pairwise_overlap_df),
    }

    audio_list = get_common_audios(feature_blocks)

    if not audio_list:
        raise ValueError("No common audios found across all feature blocks.")

    weights = {
        "centroids": args.w_centroids,
        "geometry": args.w_geometry,
        "ellipse": args.w_ellipse,
        "overlap": args.w_overlap,
    }

    matrices, feature_counts = build_distance_matrices(
        feature_blocks,
        audio_list,
        normalize_components=not args.no_normalize_components,
    )

    total_cost = combine_matrices(matrices, weights)

    assignments, rankings = hungarian_assignments(total_cost, matrices, audio_list)
    summary = make_summary(assignments, rankings, weights, feature_counts)

    summary.to_csv(output_dir / "cross_half_matching_summary.csv", index=False)
    assignments.to_csv(output_dir / "cross_half_matching_assignments.csv", index=False)
    rankings.to_csv(output_dir / "cross_half_matching_candidate_rankings.csv", index=False)

    with open(output_dir / "cross_half_matching_weights.json", "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    save_matrix_csv(
        total_cost,
        audio_list,
        output_dir / "cross_half_matching_total_cost_matrix.csv",
    )

    for name, matrix in matrices.items():
        save_matrix_csv(
            matrix,
            audio_list,
            output_dir / "component_distance_matrices" / f"{name}_distance_matrix.csv",
        )

    plot_cost_heatmap(
        total_cost,
        audio_list,
        output_dir / "cross_half_matching_total_cost_heatmap.png",
    )

    plot_true_rank(
        assignments,
        output_dir / "cross_half_true_match_rank_by_audio.png",
    )

    plot_true_false_costs(
        rankings,
        output_dir / "cross_half_true_vs_false_cost_distribution.png",
    )

    print()
    print("=== Cross-half audio matching ===")
    print(f"Input root: {input_root}")
    print(f"Output dir: {output_dir}")
    print(f"Audios used: {len(audio_list)}")
    print()
    print("Weights:")
    for name, w in weights.items():
        print(f"- {name}: {w}")
    print()
    print("Summary:")
    print(summary.to_string(index=False))
    print()
    print("Outputs:")
    print(f"- {output_dir / 'cross_half_matching_summary.csv'}")
    print(f"- {output_dir / 'cross_half_matching_assignments.csv'}")
    print(f"- {output_dir / 'cross_half_matching_candidate_rankings.csv'}")
    print(f"- {output_dir / 'cross_half_matching_total_cost_heatmap.png'}")
    print(f"- {output_dir / 'cross_half_true_match_rank_by_audio.png'}")
    print(f"- {output_dir / 'cross_half_true_vs_false_cost_distribution.png'}")


if __name__ == "__main__":
    main()
