#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency


KEY_CANDIDATES = [
    "audio",
    "file_name",
    "filename",
    "file",
    "fichier",
    "nom_fichier",
    "speaker",
    "speaker_id",
    "speaker_num",
    "locutor",
    "id_audio",
]


MISSING_VALUES = {
    "",
    "nan",
    "na",
    "n/a",
    "none",
    "null",
    "missing",
    "unknown",
    "inconnu",
    "não informado",
    "nao informado",
    "sem informação",
    "sem informacao",
    "?",
    "-",
}


def norm_col(s):
    return str(s).strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def find_col(df, names):
    lookup = {norm_col(c): c for c in df.columns}

    for name in names:
        key = norm_col(name)
        if key in lookup:
            return lookup[key]

    return None


def read_table(path, sheet_name="0"):
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        if sheet_name is None:
            sheet_name = 0
        elif isinstance(sheet_name, str) and sheet_name.isdigit():
            sheet_name = int(sheet_name)

        return pd.read_excel(path, sheet_name=sheet_name)

    if suffix in [".csv", ".txt"]:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, sep=None, engine="python")

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_excel(path, sheet_name=sheet_name)


def normalize_key(value):
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s == "":
        return None

    s = Path(s).name

    # Remove common audio/table extensions while preserving things like .v0
    for ext in [".wav", ".mp3", ".flac", ".TextGrid", ".textgrid", ".csv"]:
        if s.endswith(ext):
            s = s[: -len(ext)]

    return s.strip().lower()


def clean_category(value):
    if pd.isna(value):
        return None

    s = str(value).strip()

    if s.lower() in MISSING_VALUES:
        return None

    return s


def fdr_bh(p_values):
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)

    valid = np.isfinite(p)
    p_valid = p[valid]

    if len(p_valid) == 0:
        return q

    order = np.argsort(p_valid)
    ranked = p_valid[order]

    m = len(ranked)
    adjusted = ranked * m / np.arange(1, m + 1)

    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    q_valid = np.empty_like(adjusted)
    q_valid[order] = adjusted

    q[valid] = q_valid

    return q


def cramers_v(chi2, n, r, c):
    denom = n * min(r - 1, c - 1)

    if denom <= 0:
        return np.nan

    return float(np.sqrt(chi2 / denom))


def effect_label(v):
    if not np.isfinite(v):
        return "undefined"
    if v < 0.10:
        return "very weak"
    if v < 0.20:
        return "weak"
    if v < 0.40:
        return "moderate"
    if v < 0.60:
        return "strong"
    return "very strong"


def maybe_collapse_rare(series, min_count):
    if min_count <= 1:
        return series

    counts = series.value_counts(dropna=True)
    rare = set(counts[counts < min_count].index)

    return series.apply(lambda x: "Other_rare" if x in rare else x)


def split_multivalue_series(series, separators_regex):
    rows = []

    for idx, value in series.items():
        if pd.isna(value):
            continue

        parts = re.split(separators_regex, str(value))

        for part in parts:
            part = clean_category(part)
            if part is not None:
                rows.append((idx, part))

    return rows


def prepare_variable_data(merged, var, split_multivalue, separators_regex, min_category_count):
    base = merged[["audio", "vowel_norm", "cluster", var]].copy()
    base[var] = base[var].apply(clean_category)
    base = base.dropna(subset=[var])

    if not split_multivalue:
        base[var] = maybe_collapse_rare(base[var], min_category_count)
        return base

    expanded_rows = []

    split_rows = split_multivalue_series(base[var], separators_regex)

    for idx, category in split_rows:
        original = base.loc[idx]
        expanded_rows.append(
            {
                "audio": original["audio"],
                "vowel_norm": original["vowel_norm"],
                "cluster": original["cluster"],
                var: category,
            }
        )

    out = pd.DataFrame(expanded_rows)

    if out.empty:
        return out

    out[var] = maybe_collapse_rare(out[var], min_category_count)

    return out


def analyze_one_variable_for_one_vowel(data, vowel, var):
    sub = data[data["vowel_norm"] == vowel].copy()

    if sub.empty:
        return None, None

    contingency = pd.crosstab(sub["cluster"], sub[var])

    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return None, None

    chi2, p_value, dof, expected = chi2_contingency(contingency, correction=False)

    expected_df = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns,
    )

    residuals = (contingency - expected_df) / np.sqrt(expected_df)

    n = int(contingency.to_numpy().sum())
    r, c = contingency.shape
    v = cramers_v(chi2, n, r, c)

    min_expected = float(np.min(expected))
    n_expected_below_5 = int(np.sum(expected < 5))

    result = {
        "vowel": vowel,
        "social_variable": var,
        "n_speakers_or_rows": n,
        "n_clusters": r,
        "n_categories": c,
        "chi2": float(chi2),
        "dof": int(dof),
        "p_value": float(p_value),
        "cramers_v": v,
        "effect_label": effect_label(v),
        "min_expected_count": min_expected,
        "n_cells_expected_below_5": n_expected_below_5,
        "caution_sparse_table": bool(n_expected_below_5 > 0),
    }

    long_rows = []

    row_totals = contingency.sum(axis=1)
    col_totals = contingency.sum(axis=0)
    total = contingency.to_numpy().sum()

    for cluster in contingency.index:
        for category in contingency.columns:
            obs = int(contingency.loc[cluster, category])
            exp = float(expected_df.loc[cluster, category])
            resid = float(residuals.loc[cluster, category])

            long_rows.append(
                {
                    "vowel": vowel,
                    "social_variable": var,
                    "cluster": cluster,
                    "category": category,
                    "observed": obs,
                    "expected": exp,
                    "std_residual": resid,
                    "row_percent": obs / row_totals.loc[cluster] * 100 if row_totals.loc[cluster] > 0 else np.nan,
                    "column_percent": obs / col_totals.loc[category] * 100 if col_totals.loc[category] > 0 else np.nan,
                    "total_percent": obs / total * 100 if total > 0 else np.nan,
                }
            )

    long_df = pd.DataFrame(long_rows)

    return result, long_df


def plot_residual_heatmap(long_df, vowel, var, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = long_df[
        (long_df["vowel"] == vowel)
        & (long_df["social_variable"] == var)
    ].copy()

    if sub.empty:
        return

    pivot_resid = sub.pivot(index="cluster", columns="category", values="std_residual")
    pivot_obs = sub.pivot(index="cluster", columns="category", values="observed")

    fig_width = max(8, min(18, 0.8 * len(pivot_resid.columns) + 3))
    fig_height = max(4, 0.7 * len(pivot_resid.index) + 2)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    vmax = np.nanmax(np.abs(pivot_resid.to_numpy()))

    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0

    im = ax.imshow(
        pivot_resid.to_numpy(),
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        cmap="coolwarm",
    )

    ax.set_title(f"Standardized residuals: /{vowel}/ clusters × {var}")
    ax.set_xlabel(var)
    ax.set_ylabel("Cluster")

    ax.set_xticks(np.arange(len(pivot_resid.columns)))
    ax.set_xticklabels(pivot_resid.columns, rotation=45, ha="right")

    ax.set_yticks(np.arange(len(pivot_resid.index)))
    ax.set_yticklabels(pivot_resid.index)

    for i in range(pivot_resid.shape[0]):
        for j in range(pivot_resid.shape[1]):
            resid = pivot_resid.iloc[i, j]
            obs = pivot_obs.iloc[i, j]

            if pd.isna(resid):
                text = ""
            else:
                text = f"{resid:.1f}\n(n={int(obs)})"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Standardized residual")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def safe_filename(s):
    s = str(s)
    s = re.sub(r"[^A-Za-z0-9_\\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--clusters",
        required=True,
        help="CSV with cluster assignments, e.g. points_by_vowel_cluster_assignments.csv",
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Social metadata file, CSV or XLSX.",
    )

    parser.add_argument(
        "--metadata-sheet",
        default="0",
        help="Sheet name/index for XLSX. Default: 0.",
    )

    parser.add_argument(
        "--cluster-key",
        default="audio",
        help="Key column in cluster file. Default: audio.",
    )

    parser.add_argument(
        "--metadata-key",
        default=None,
        help="Key column in metadata file. If omitted, script tries to detect it.",
    )

    parser.add_argument(
        "--output-dir",
        default="results/tables/social_cluster_associations",
    )

    parser.add_argument(
        "--plots-dir",
        default="results/plots/social_cluster_associations",
    )

    parser.add_argument(
        "--exclude-cols",
        nargs="*",
        default=[],
        help="Metadata columns to exclude from tests.",
    )

    parser.add_argument(
        "--max-categories",
        type=int,
        default=30,
        help="Skip variables with more categories than this. Default: 30.",
    )

    parser.add_argument(
        "--min-category-count",
        type=int,
        default=1,
        help="Collapse categories with fewer than this count into Other_rare. Default: 1.",
    )

    parser.add_argument(
        "--top-n-plots",
        type=int,
        default=15,
        help="Number of top associations to plot as residual heatmaps. Default: 15.",
    )

    parser.add_argument(
        "--split-multivalue",
        action="store_true",
        help="Split cells with multiple values using separators ; | / . Exploratory only.",
    )

    parser.add_argument(
        "--multivalue-separators",
        default=r";|\\||/",
        help="Regex separators for --split-multivalue. Default: ; | /",
    )

    args = parser.parse_args()

    clusters_path = Path(args.clusters)
    metadata_path = Path(args.metadata)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    clusters = read_table(clusters_path)
    metadata = read_table(metadata_path, sheet_name=args.metadata_sheet)

    cluster_key = find_col(clusters, [args.cluster_key])
    if cluster_key is None:
        raise ValueError(f"Cluster key not found: {args.cluster_key}")

    if args.metadata_key is not None:
        metadata_key = find_col(metadata, [args.metadata_key])
        if metadata_key is None:
            raise ValueError(f"Metadata key not found: {args.metadata_key}")
    else:
        metadata_key = find_col(metadata, KEY_CANDIDATES)
        if metadata_key is None:
            raise ValueError(
                "Could not auto-detect metadata key. "
                "Use --metadata-key NOME_DA_COLUNA. "
                f"Available columns: {list(metadata.columns)}"
            )

    vowel_col = find_col(clusters, ["vowel_norm", "vowel", "label"])
    cluster_col = find_col(clusters, ["cluster"])

    if vowel_col is None:
        raise ValueError("Could not find vowel column in cluster file.")

    if cluster_col is None:
        raise ValueError("Could not find cluster column in cluster file.")

    clusters = clusters.copy()
    metadata = metadata.copy()

    clusters["join_key"] = clusters[cluster_key].apply(normalize_key)
    metadata["join_key"] = metadata[metadata_key].apply(normalize_key)

    clusters["audio"] = clusters[cluster_key].astype(str)
    clusters["vowel_norm"] = clusters[vowel_col].astype(str).str.strip().str.lower()
    clusters["cluster"] = clusters[cluster_col].astype(str)

    merged = clusters.merge(
        metadata,
        on="join_key",
        how="left",
        indicator=True,
        suffixes=("", "_meta"),
    )

    match_report = pd.DataFrame(
        [
            {
                "cluster_rows": len(clusters),
                "metadata_rows": len(metadata),
                "merged_rows": len(merged),
                "matched_rows": int((merged["_merge"] == "both").sum()),
                "unmatched_cluster_rows": int((merged["_merge"] != "both").sum()),
                "matched_unique_audios": int(merged.loc[merged["_merge"] == "both", "audio"].nunique()),
                "total_unique_audios": int(merged["audio"].nunique()),
                "cluster_key": cluster_key,
                "metadata_key": metadata_key,
            }
        ]
    )

    match_report_path = output_dir / "social_cluster_match_report.csv"
    match_report.to_csv(match_report_path, index=False)

    unmatched = merged[merged["_merge"] != "both"][["audio", "join_key", "vowel_norm", "cluster"]]
    unmatched_path = output_dir / "social_cluster_unmatched_rows.csv"
    unmatched.to_csv(unmatched_path, index=False)

    exclude = set(args.exclude_cols)
    exclude.update({metadata_key, "join_key"})

    social_cols = []

    for col in metadata.columns:
        if col in exclude:
            continue

        if col == "join_key":
            continue

        non_null = metadata[col].apply(clean_category).dropna()

        if len(non_null) == 0:
            continue

        n_cat = non_null.nunique()

        if n_cat < 2:
            continue

        if n_cat > args.max_categories:
            continue

        social_cols.append(col)

    skipped_cols = []

    for col in metadata.columns:
        if col in social_cols or col in exclude or col == "join_key":
            continue

        non_null = metadata[col].apply(clean_category).dropna()
        skipped_cols.append(
            {
                "column": col,
                "non_missing": int(len(non_null)),
                "n_categories": int(non_null.nunique()) if len(non_null) else 0,
                "reason": "excluded_or_invalid_or_too_many_categories",
            }
        )

    skipped_df = pd.DataFrame(skipped_cols)
    skipped_path = output_dir / "social_cluster_skipped_variables.csv"
    skipped_df.to_csv(skipped_path, index=False)

    all_results = []
    all_long = []

    vowels = sorted(merged["vowel_norm"].dropna().unique())

    for var in social_cols:
        print(f"Testing variable: {var}")

        var_data = prepare_variable_data(
            merged=merged,
            var=var,
            split_multivalue=args.split_multivalue,
            separators_regex=args.multivalue_separators,
            min_category_count=args.min_category_count,
        )

        if var_data.empty:
            continue

        for vowel in vowels:
            result, long_df = analyze_one_variable_for_one_vowel(
                data=var_data,
                vowel=vowel,
                var=var,
            )

            if result is None:
                continue

            all_results.append(result)
            all_long.append(long_df)

    if not all_results:
        raise ValueError("No valid association tests were produced.")

    results_df = pd.DataFrame(all_results)
    results_df["p_fdr"] = fdr_bh(results_df["p_value"])
    results_df = results_df.sort_values(
        ["cramers_v", "p_value"],
        ascending=[False, True],
    )

    long_df = pd.concat(all_long, ignore_index=True)

    ranking_path = output_dir / "social_cluster_association_ranking.csv"
    contingency_long_path = output_dir / "social_cluster_contingency_long.csv"
    merged_path = output_dir / "social_cluster_merged_data.csv"

    results_df.to_csv(ranking_path, index=False)
    long_df.to_csv(contingency_long_path, index=False)
    merged.to_csv(merged_path, index=False)

    top_for_plots = results_df.head(args.top_n_plots)

    for rank, row in enumerate(top_for_plots.itertuples(index=False), start=1):
        vowel = row.vowel
        var = row.social_variable

        plot_name = f"{rank:02d}_{safe_filename(vowel)}_{safe_filename(var)}_residual_heatmap.png"
        plot_path = plots_dir / plot_name

        plot_residual_heatmap(
            long_df=long_df,
            vowel=vowel,
            var=var,
            output_path=plot_path,
        )

    print()
    print("=== Social variables × acoustic clusters ===")
    print(f"Clusters: {clusters_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Cluster key: {cluster_key}")
    print(f"Metadata key: {metadata_key}")
    print()
    print("Match report:")
    print(match_report.to_string(index=False))
    print()
    print(f"Variables tested: {len(social_cols)}")
    print()
    print("Top associations by Cramér's V:")
    display_cols = [
        "vowel",
        "social_variable",
        "n_speakers_or_rows",
        "n_clusters",
        "n_categories",
        "cramers_v",
        "effect_label",
        "p_value",
        "p_fdr",
        "min_expected_count",
        "n_cells_expected_below_5",
    ]
    print(results_df[display_cols].head(20).to_string(index=False))
    print()
    print("Outputs:")
    print(f"- Ranking: {ranking_path}")
    print(f"- Contingency/residuals long table: {contingency_long_path}")
    print(f"- Merged data: {merged_path}")
    print(f"- Match report: {match_report_path}")
    print(f"- Unmatched rows: {unmatched_path}")
    print(f"- Skipped variables: {skipped_path}")
    print(f"- Top residual heatmaps: {plots_dir}")


if __name__ == "__main__":
    main()
