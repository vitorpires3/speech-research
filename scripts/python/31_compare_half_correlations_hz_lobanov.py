#!/usr/bin/env python3
"""Compare H1-H2 Pearson and Spearman correlations.

The analysis compares:

1. non-normalized vowel centroids and geometry in Hz;
2. independently normalized Lobanov centroids and geometry.

For each acoustic feature, the observations are the speakers:
H1 values are correlated with the corresponding H2 values.

The report is generated in English.
"""

from __future__ import annotations

import argparse
import itertools
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import (
    ConstantInputWarning,
    pearsonr,
    spearmanr,
)


VOWELS = ("i", "e", "a", "o", "u")
PAIRS = tuple(itertools.combinations(VOWELS, 2))


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Compare split-half Pearson and Spearman correlations "
            "for non-normalized and Lobanov vowel features."
        )
    )

    parser.add_argument(
        "--hz-root",
        type=Path,
        default=(
            project_root
            / "results"
            / "all_audio_half_region_profiles_level80"
        ),
        help="Directory containing the non-normalized H1/H2 profiles.",
    )

    parser.add_argument(
        "--lobanov-root",
        type=Path,
        default=(
            project_root
            / "results"
            / "all_audio_half_region_profiles_lobanov_median_centroids"
        ),
        help="Directory containing the Lobanov H1/H2 profiles.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "results"
            / "half_correlations_hz_vs_lobanov"
        ),
        help="Output directory.",
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help=(
            "Number of paired speaker bootstrap samples used for "
            "confidence intervals. Default: 2000."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260721,
        help="Random seed used for the bootstrap.",
    )

    return parser.parse_args()


def read_centers(
    input_root: Path,
    representation_name: str,
) -> pd.DataFrame:
    path = (
        input_root
        / "general_ellipse_parameters_all.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"{representation_name}: file not found: {path}"
        )

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    required = {
        "audio",
        "half",
        "vowel",
        "center_f1",
        "center_f2",
    }

    missing = sorted(
        required.difference(df.columns)
    )

    if missing:
        raise ValueError(
            f"{representation_name}: missing columns: {missing}"
        )

    df = df.copy()

    df["audio"] = (
        df["audio"]
        .astype(str)
        .str.strip()
    )

    df["half"] = (
        df["half"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["vowel"] = (
        df["vowel"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for column in ("center_f1", "center_f2"):
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        df.loc[
            ~np.isfinite(df[column]),
            column,
        ] = np.nan

    df = df[
        df["half"].isin(("H1", "H2"))
        & df["vowel"].isin(VOWELS)
    ][
        [
            "audio",
            "half",
            "vowel",
            "center_f1",
            "center_f2",
        ]
    ].copy()

    duplicated = df.duplicated(
        subset=[
            "audio",
            "half",
            "vowel",
        ],
        keep=False,
    )

    if duplicated.any():
        examples = (
            df.loc[
                duplicated,
                [
                    "audio",
                    "half",
                    "vowel",
                ],
            ]
            .drop_duplicates()
            .head(10)
        )

        raise ValueError(
            f"{representation_name}: duplicate audio/half/vowel "
            "keys were found:\n"
            f"{examples.to_string(index=False)}"
        )

    return df


def build_centroid_features(
    centers: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (audio, half), sub in centers.groupby(
        ["audio", "half"],
        sort=True,
    ):
        row: dict[str, object] = {
            "audio": audio,
            "half": half,
        }

        indexed = sub.set_index("vowel")

        for vowel in VOWELS:
            if vowel not in indexed.index:
                row[f"centroid_{vowel}_f1"] = np.nan
                row[f"centroid_{vowel}_f2"] = np.nan
                continue

            row[f"centroid_{vowel}_f1"] = (
                indexed.loc[vowel, "center_f1"]
            )

            row[f"centroid_{vowel}_f2"] = (
                indexed.loc[vowel, "center_f2"]
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_geometry_features(
    centers: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for (audio, half), sub in centers.groupby(
        ["audio", "half"],
        sort=True,
    ):
        row: dict[str, object] = {
            "audio": audio,
            "half": half,
        }

        indexed = sub.set_index("vowel")

        for vowel_1, vowel_2 in PAIRS:
            column = (
                f"geometry_{vowel_1}_{vowel_2}"
            )

            if (
                vowel_1 not in indexed.index
                or vowel_2 not in indexed.index
            ):
                row[column] = np.nan
                continue

            f1_a = indexed.loc[
                vowel_1,
                "center_f1",
            ]

            f2_a = indexed.loc[
                vowel_1,
                "center_f2",
            ]

            f1_b = indexed.loc[
                vowel_2,
                "center_f1",
            ]

            f2_b = indexed.loc[
                vowel_2,
                "center_f2",
            ]

            values = np.asarray(
                [
                    f1_a,
                    f2_a,
                    f1_b,
                    f2_b,
                ],
                dtype=float,
            )

            if not np.all(np.isfinite(values)):
                row[column] = np.nan
                continue

            row[column] = math.sqrt(
                (f1_a - f1_b) ** 2
                + (f2_a - f2_b) ** 2
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_blocks(
    input_root: Path,
    representation_name: str,
) -> dict[str, pd.DataFrame]:
    centers = read_centers(
        input_root,
        representation_name,
    )

    return {
        "Centroids": build_centroid_features(
            centers
        ),
        "Geometry": build_geometry_features(
            centers
        ),
    }


def make_half_pair_table(
    feature_df: pd.DataFrame,
    feature: str,
) -> pd.DataFrame:
    h1 = (
        feature_df[
            feature_df["half"] == "H1"
        ][
            [
                "audio",
                feature,
            ]
        ]
        .drop_duplicates(subset=["audio"])
        .rename(columns={feature: "H1"})
    )

    h2 = (
        feature_df[
            feature_df["half"] == "H2"
        ][
            [
                "audio",
                feature,
            ]
        ]
        .drop_duplicates(subset=["audio"])
        .rename(columns={feature: "H2"})
    )

    result = h1.merge(
        h2,
        on="audio",
        how="inner",
        validate="one_to_one",
    )

    result["H1"] = pd.to_numeric(
        result["H1"],
        errors="coerce",
    )

    result["H2"] = pd.to_numeric(
        result["H2"],
        errors="coerce",
    )

    valid = (
        np.isfinite(result["H1"])
        & np.isfinite(result["H2"])
    )

    return (
        result.loc[valid]
        .sort_values("audio")
        .reset_index(drop=True)
    )


def safe_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan, np.nan

    if (
        np.std(x, ddof=0) == 0
        or np.std(y, ddof=0) == 0
    ):
        return np.nan, np.nan

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            ConstantInputWarning,
        )

        if method == "pearson":
            result = pearsonr(x, y)

        elif method == "spearman":
            result = spearmanr(x, y)

        else:
            raise ValueError(
                f"Unknown correlation method: {method}"
            )

    statistic = float(result.statistic)
    pvalue = float(result.pvalue)

    return statistic, pvalue


def correlation_only(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
) -> float:
    statistic, _ = safe_correlation(
        x,
        y,
        method,
    )

    return statistic


def percentile_interval(
    values: list[float],
) -> tuple[float, float]:
    finite = np.asarray(
        [
            value
            for value in values
            if np.isfinite(value)
        ],
        dtype=float,
    )

    if len(finite) < 20:
        return np.nan, np.nan

    return (
        float(np.percentile(finite, 2.5)),
        float(np.percentile(finite, 97.5)),
    )


def bootstrap_comparison(
    hz_h1: np.ndarray,
    hz_h2: np.ndarray,
    lob_h1: np.ndarray,
    lob_h2: np.ndarray,
    method: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    n = len(hz_h1)

    hz_values: list[float] = []
    lob_values: list[float] = []
    difference_values: list[float] = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        hz_correlation = correlation_only(
            hz_h1[indices],
            hz_h2[indices],
            method,
        )

        lob_correlation = correlation_only(
            lob_h1[indices],
            lob_h2[indices],
            method,
        )

        if np.isfinite(hz_correlation):
            hz_values.append(
                hz_correlation
            )

        if np.isfinite(lob_correlation):
            lob_values.append(
                lob_correlation
            )

        if (
            np.isfinite(hz_correlation)
            and np.isfinite(lob_correlation)
        ):
            difference_values.append(
                lob_correlation
                - hz_correlation
            )

    hz_low, hz_high = percentile_interval(
        hz_values
    )

    lob_low, lob_high = percentile_interval(
        lob_values
    )

    difference_low, difference_high = (
        percentile_interval(
            difference_values
        )
    )

    if difference_values:
        probability_lobanov_higher = float(
            np.mean(
                np.asarray(
                    difference_values
                )
                > 0
            )
        )
    else:
        probability_lobanov_higher = np.nan

    return {
        "hz_ci_low": hz_low,
        "hz_ci_high": hz_high,
        "lobanov_ci_low": lob_low,
        "lobanov_ci_high": lob_high,
        "difference_ci_low": difference_low,
        "difference_ci_high": difference_high,
        "probability_lobanov_higher": (
            probability_lobanov_higher
        ),
    }


def bh_adjust(
    pvalues: pd.Series,
) -> pd.Series:
    values = pd.to_numeric(
        pvalues,
        errors="coerce",
    ).to_numpy(dtype=float)

    adjusted = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    finite_indices = np.where(
        np.isfinite(values)
    )[0]

    if len(finite_indices) == 0:
        return pd.Series(
            adjusted,
            index=pvalues.index,
        )

    finite_values = values[
        finite_indices
    ]

    order = np.argsort(
        finite_values
    )

    ordered_values = finite_values[
        order
    ]

    number = len(ordered_values)

    ordered_adjusted = (
        ordered_values
        * number
        / np.arange(
            1,
            number + 1,
        )
    )

    ordered_adjusted = np.minimum.accumulate(
        ordered_adjusted[::-1]
    )[::-1]

    ordered_adjusted = np.clip(
        ordered_adjusted,
        0,
        1,
    )

    reverse_order = np.empty_like(
        order
    )

    reverse_order[order] = np.arange(
        number
    )

    adjusted_values = ordered_adjusted[
        reverse_order
    ]

    adjusted[
        finite_indices
    ] = adjusted_values

    return pd.Series(
        adjusted,
        index=pvalues.index,
    )


def feature_display_name(
    feature: str,
) -> str:
    if feature.startswith("centroid_"):
        _, vowel, formant = (
            feature.split("_")
        )

        return (
            f"/{vowel}/ {formant.upper()} centroid"
        )

    if feature.startswith("geometry_"):
        _, vowel_1, vowel_2 = (
            feature.split("_")
        )

        return (
            f"/{vowel_1}–{vowel_2}/ distance"
        )

    return feature


def feature_short_name(
    feature: str,
) -> str:
    if feature.startswith("centroid_"):
        _, vowel, formant = (
            feature.split("_")
        )

        return (
            f"{vowel}-{formant.upper()}"
        )

    if feature.startswith("geometry_"):
        _, vowel_1, vowel_2 = (
            feature.split("_")
        )

        return (
            f"{vowel_1}–{vowel_2}"
        )

    return feature


def comparison_label(
    low: float,
    high: float,
) -> str:
    if not (
        np.isfinite(low)
        and np.isfinite(high)
    ):
        return "Unavailable"

    if low > 0:
        return "Lobanov higher"

    if high < 0:
        return "Non-normalized higher"

    return "No clear difference"


def plot_heatmap(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    columns = [
        "pearson_hz",
        "pearson_lobanov",
        "spearman_hz",
        "spearman_lobanov",
    ]

    labels = [
        "Pearson\nHz",
        "Pearson\nLobanov",
        "Spearman\nHz",
        "Spearman\nLobanov",
    ]

    values = results[
        columns
    ].to_numpy(dtype=float)

    row_labels = [
        feature_display_name(feature)
        for feature in results["feature"]
    ]

    height = max(
        8,
        0.42 * len(results),
    )

    figure, axis = plt.subplots(
        figsize=(9, height),
        constrained_layout=True,
    )

    image = axis.imshow(
        values,
        aspect="auto",
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    axis.set_xticks(
        np.arange(len(labels))
    )

    axis.set_xticklabels(
        labels
    )

    axis.set_yticks(
        np.arange(len(row_labels))
    )

    axis.set_yticklabels(
        row_labels,
        fontsize=8,
    )

    for row_index in range(
        values.shape[0]
    ):
        for column_index in range(
            values.shape[1]
        ):
            value = values[
                row_index,
                column_index,
            ]

            text = (
                f"{value:.2f}"
                if np.isfinite(value)
                else "NA"
            )

            axis.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    axis.set_title(
        "H1–H2 correlations by acoustic feature"
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Correlation",
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_representation_comparison(
    results: pd.DataFrame,
    method: str,
    output_path: Path,
) -> None:
    hz_column = f"{method}_hz"
    lobanov_column = (
        f"{method}_lobanov"
    )

    figure, axis = plt.subplots(
        figsize=(9, 8),
        constrained_layout=True,
    )

    markers = {
        "Centroids": "o",
        "Geometry": "^",
    }

    all_values = np.concatenate(
        [
            results[
                hz_column
            ].to_numpy(dtype=float),
            results[
                lobanov_column
            ].to_numpy(dtype=float),
        ]
    )

    finite_values = all_values[
        np.isfinite(all_values)
    ]

    if len(finite_values):
        lower = max(
            -1.0,
            float(finite_values.min())
            - 0.08,
        )

        upper = min(
            1.0,
            float(finite_values.max())
            + 0.08,
        )
    else:
        lower = -1
        upper = 1

    if upper - lower < 0.4:
        midpoint = (
            lower + upper
        ) / 2

        lower = max(
            -1,
            midpoint - 0.25,
        )

        upper = min(
            1,
            midpoint + 0.25,
        )

    for block, block_df in results.groupby(
        "block",
        sort=False,
    ):
        axis.scatter(
            block_df[hz_column],
            block_df[lobanov_column],
            marker=markers.get(
                block,
                "o",
            ),
            s=65,
            alpha=0.75,
            label=block,
        )

        for _, row in block_df.iterrows():
            x_value = row[hz_column]
            y_value = row[
                lobanov_column
            ]

            if not (
                np.isfinite(x_value)
                and np.isfinite(y_value)
            ):
                continue

            axis.annotate(
                feature_short_name(
                    row["feature"]
                ),
                (
                    x_value,
                    y_value,
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    axis.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1,
        label="Equal correlation",
    )

    axis.set_xlim(
        lower,
        upper,
    )

    axis.set_ylim(
        lower,
        upper,
    )

    axis.set_aspect(
        "equal",
        adjustable="box",
    )

    axis.set_xlabel(
        f"{method.capitalize()} correlation — non-normalized"
    )

    axis.set_ylabel(
        f"{method.capitalize()} correlation — Lobanov"
    )

    axis.set_title(
        f"H1–H2 {method.capitalize()} correlation comparison"
    )

    axis.grid(
        alpha=0.25
    )

    axis.legend()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def format_correlation(
    value: float,
    low: float,
    high: float,
) -> str:
    if not np.isfinite(value):
        return "NA"

    if (
        np.isfinite(low)
        and np.isfinite(high)
    ):
        return (
            f"{value:.3f} "
            f"[{low:.3f}, {high:.3f}]"
        )

    return f"{value:.3f}"


def format_probability(
    value: float,
) -> str:
    if not np.isfinite(value):
        return "NA"

    if value < 0.001:
        return "&lt;0.001"

    return f"{value:.3f}"


def format_number(
    value: float,
    decimals: int = 3,
) -> str:
    if not np.isfinite(value):
        return "NA"

    return f"{value:.{decimals}f}"


def create_html_report(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    output_path: Path,
    assets_dir_name: str,
    bootstrap_samples: int,
    hz_root: Path,
    lobanov_root: Path,
) -> None:
    detail_rows = []

    for _, row in results.iterrows():
        pearson_status = comparison_label(
            row[
                "pearson_difference_ci_low"
            ],
            row[
                "pearson_difference_ci_high"
            ],
        )

        spearman_status = comparison_label(
            row[
                "spearman_difference_ci_low"
            ],
            row[
                "spearman_difference_ci_high"
            ],
        )

        detail_rows.append(
            {
                "Feature block": row["block"],
                "Feature": feature_display_name(
                    row["feature"]
                ),
                "N speakers": int(row["n_speakers"]),
                "Pearson — Hz": format_correlation(
                    row["pearson_hz"],
                    row["pearson_hz_ci_low"],
                    row["pearson_hz_ci_high"],
                ),
                "Pearson — Lobanov": format_correlation(
                    row["pearson_lobanov"],
                    row["pearson_lobanov_ci_low"],
                    row["pearson_lobanov_ci_high"],
                ),
                "Pearson Δ<br>(Lobanov − Hz)": (
                    format_correlation(
                        row[
                            "pearson_difference"
                        ],
                        row[
                            "pearson_difference_ci_low"
                        ],
                        row[
                            "pearson_difference_ci_high"
                        ],
                    )
                ),
                "Pearson comparison": pearson_status,
                "Pearson Hz FDR q": format_probability(
                    row["pearson_hz_q"]
                ),
                "Pearson Lobanov FDR q": format_probability(
                    row[
                        "pearson_lobanov_q"
                    ]
                ),
                "Spearman — Hz": format_correlation(
                    row["spearman_hz"],
                    row["spearman_hz_ci_low"],
                    row["spearman_hz_ci_high"],
                ),
                "Spearman — Lobanov": format_correlation(
                    row["spearman_lobanov"],
                    row[
                        "spearman_lobanov_ci_low"
                    ],
                    row[
                        "spearman_lobanov_ci_high"
                    ],
                ),
                "Spearman Δ<br>(Lobanov − Hz)": (
                    format_correlation(
                        row[
                            "spearman_difference"
                        ],
                        row[
                            "spearman_difference_ci_low"
                        ],
                        row[
                            "spearman_difference_ci_high"
                        ],
                    )
                ),
                "Spearman comparison": spearman_status,
                "Spearman Hz FDR q": format_probability(
                    row["spearman_hz_q"]
                ),
                "Spearman Lobanov FDR q": format_probability(
                    row[
                        "spearman_lobanov_q"
                    ]
                ),
            }
        )

    detail_display = pd.DataFrame(
        detail_rows
    )

    summary_display = summary.rename(
        columns={
            "block": "Feature block",
            "n_features": "Number of features",
            "median_pearson_hz": (
                "Median Pearson — Hz"
            ),
            "median_pearson_lobanov": (
                "Median Pearson — Lobanov"
            ),
            "median_spearman_hz": (
                "Median Spearman — Hz"
            ),
            "median_spearman_lobanov": (
                "Median Spearman — Lobanov"
            ),
            "pearson_features_lobanov_higher": (
                "Features with higher Pearson after Lobanov"
            ),
            "spearman_features_lobanov_higher": (
                "Features with higher Spearman after Lobanov"
            ),
        }
    )

    html = f"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">

<title>
H1–H2 correlations: non-normalized versus Lobanov
</title>

<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 2rem;
    line-height: 1.45;
    color: #222;
}}

h1 {{
    margin-bottom: 0.25rem;
}}

h2 {{
    margin-top: 2.5rem;
}}

.description {{
    max-width: 1100px;
    color: #444;
}}

.note {{
    max-width: 1100px;
    padding: 0.9rem 1rem;
    background: #f5f5f5;
    border-left: 4px solid #888;
    margin: 1.2rem 0;
}}

.table-container {{
    overflow-x: auto;
    border: 1px solid #ddd;
    margin-bottom: 2rem;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    min-width: 1200px;
    font-size: 0.82rem;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 0.45rem 0.6rem;
    text-align: right;
    white-space: nowrap;
}}

th {{
    position: sticky;
    top: 0;
    background: #eeeeee;
    z-index: 1;
}}

td:first-child,
th:first-child,
td:nth-child(2),
th:nth-child(2) {{
    text-align: left;
}}

tbody tr:nth-child(even) {{
    background: #fafafa;
}}

tbody tr:hover {{
    background: #f1f5ff;
}}

img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 1rem 0 2.5rem;
    border: 1px solid #ddd;
}}

code {{
    background: #f2f2f2;
    padding: 0.1rem 0.25rem;
}}

</style>
</head>

<body>

<h1>
H1–H2 acoustic-feature correlations
</h1>

<p class="description">
This report compares the split-half stability of vowel
centroids and vowel-space geometry before and after Lobanov
normalization.
For each feature, the H1 values across speakers were correlated
with the corresponding H2 values from the same speakers.
</p>

<div class="note">
<strong>Interpretation:</strong>
Pearson correlation measures linear stability of the numerical
values, whereas Spearman correlation measures stability of the
speaker ranking.
A high correlation does not necessarily imply exact agreement:
H2 may be systematically shifted relative to H1 while preserving
a high correlation.
</div>

<h2>Methods</h2>

<ul>
<li>
Non-normalized source:
<code>{hz_root}</code>
</li>

<li>
Lobanov source:
<code>{lobanov_root}</code>
</li>

<li>
Centroid features:
five vowels × F1/F2 = 10 coordinates.
</li>

<li>
Geometry features:
10 Euclidean distances between pairs of vowel centroids.
</li>

<li>
Bootstrap confidence intervals:
{bootstrap_samples:,} paired resamples of speakers.
</li>

<li>
Correlation difference:
Lobanov correlation minus non-normalized correlation.
</li>

<li>
FDR q-values:
Benjamini–Hochberg correction within each
correlation/representation family.
</li>
</ul>

<h2>Summary by feature block</h2>

<div class="table-container">
"""

    html += summary_display.to_html(
        index=False,
        border=0,
        float_format=lambda value: f"{value:.3f}",
    )

    html += f"""
</div>

<h2>Correlation heatmap</h2>

<p class="description">
Each cell shows the H1–H2 correlation for one acoustic feature.
</p>

<img
    src="{assets_dir_name}/correlation_heatmap.png"
    alt="Correlation heatmap"
>

<h2>Pearson comparison</h2>

<p class="description">
Points above the diagonal have higher Pearson correlation after
Lobanov normalization. Points below the diagonal are more stable
in the non-normalized representation.
</p>

<img
    src="{assets_dir_name}/pearson_hz_vs_lobanov.png"
    alt="Pearson correlation comparison"
>

<h2>Spearman comparison</h2>

<p class="description">
Points above the diagonal preserve speaker ranking better after
Lobanov normalization. Points below the diagonal preserve ranking
better before normalization.
</p>

<img
    src="{assets_dir_name}/spearman_hz_vs_lobanov.png"
    alt="Spearman correlation comparison"
>

<h2>Detailed feature results</h2>

<p class="description">
Values in brackets are 95% paired-speaker bootstrap confidence
intervals. A positive difference means that the Lobanov
correlation is higher.
</p>

<div class="table-container">
"""

    html += detail_display.to_html(
        index=False,
        border=0,
        escape=False,
    )

    html += """
</div>

<h2>Interpretation cautions</h2>

<ul>
<li>
Correlations are calculated across speakers, not separately
within each speaker.
</li>

<li>
The same speakers and the same H1/H2 divisions are used in the
two representations whenever all four values are available.
</li>

<li>
Geometry in Hz and geometry in Lobanov units have different
scales, but correlation itself is scale-free.
</li>

<li>
A high Spearman value with a lower Pearson value suggests that
speaker ordering is stable even when numerical spacing changes.
</li>

<li>
A high Pearson or Spearman correlation should not be interpreted
as proof of exact H1–H2 agreement.
</li>
</ul>

</body>
</html>
"""

    output_path.write_text(
        html,
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if args.bootstrap < 100:
        raise ValueError(
            "--bootstrap must be at least 100."
        )

    hz_root = args.hz_root.resolve()
    lobanov_root = (
        args.lobanov_root.resolve()
    )

    output_dir = args.output_dir.resolve()
    assets_dir = output_dir / "assets"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    assets_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Reading feature tables...")

    hz_blocks = build_feature_blocks(
        hz_root,
        "Non-normalized",
    )

    lobanov_blocks = build_feature_blocks(
        lobanov_root,
        "Lobanov",
    )

    rng = np.random.default_rng(
        args.seed
    )

    result_rows: list[
        dict[str, object]
    ] = []

    for block in ("Centroids", "Geometry"):
        hz_df = hz_blocks[block]
        lobanov_df = lobanov_blocks[
            block
        ]

        feature_columns = [
            column
            for column in hz_df.columns
            if column not in ("audio", "half")
            and column in lobanov_df.columns
        ]

        print(
            f"{block}: "
            f"{len(feature_columns)} features"
        )

        for position, feature in enumerate(
            feature_columns,
            start=1,
        ):
            print(
                f"  [{position:02d}/"
                f"{len(feature_columns):02d}] "
                f"{feature}"
            )

            hz_pairs = make_half_pair_table(
                hz_df,
                feature,
            ).rename(
                columns={
                    "H1": "hz_H1",
                    "H2": "hz_H2",
                }
            )

            lobanov_pairs = (
                make_half_pair_table(
                    lobanov_df,
                    feature,
                )
                .rename(
                    columns={
                        "H1": "lobanov_H1",
                        "H2": "lobanov_H2",
                    }
                )
            )

            paired = hz_pairs.merge(
                lobanov_pairs,
                on="audio",
                how="inner",
                validate="one_to_one",
            )

            numeric_columns = [
                "hz_H1",
                "hz_H2",
                "lobanov_H1",
                "lobanov_H2",
            ]

            valid = np.ones(
                len(paired),
                dtype=bool,
            )

            for column in numeric_columns:
                paired[column] = (
                    pd.to_numeric(
                        paired[column],
                        errors="coerce",
                    )
                )

                valid &= np.isfinite(
                    paired[column]
                )

            paired = (
                paired.loc[valid]
                .reset_index(drop=True)
            )

            n_speakers = len(paired)

            if n_speakers < 3:
                print(
                    "    Skipped: fewer than "
                    "three complete speakers."
                )
                continue

            hz_h1 = paired[
                "hz_H1"
            ].to_numpy(dtype=float)

            hz_h2 = paired[
                "hz_H2"
            ].to_numpy(dtype=float)

            lob_h1 = paired[
                "lobanov_H1"
            ].to_numpy(dtype=float)

            lob_h2 = paired[
                "lobanov_H2"
            ].to_numpy(dtype=float)

            pearson_hz, pearson_hz_p = (
                safe_correlation(
                    hz_h1,
                    hz_h2,
                    "pearson",
                )
            )

            pearson_lob, pearson_lob_p = (
                safe_correlation(
                    lob_h1,
                    lob_h2,
                    "pearson",
                )
            )

            spearman_hz, spearman_hz_p = (
                safe_correlation(
                    hz_h1,
                    hz_h2,
                    "spearman",
                )
            )

            spearman_lob, spearman_lob_p = (
                safe_correlation(
                    lob_h1,
                    lob_h2,
                    "spearman",
                )
            )

            pearson_bootstrap = (
                bootstrap_comparison(
                    hz_h1,
                    hz_h2,
                    lob_h1,
                    lob_h2,
                    "pearson",
                    args.bootstrap,
                    rng,
                )
            )

            spearman_bootstrap = (
                bootstrap_comparison(
                    hz_h1,
                    hz_h2,
                    lob_h1,
                    lob_h2,
                    "spearman",
                    args.bootstrap,
                    rng,
                )
            )

            result_rows.append(
                {
                    "block": block,
                    "feature": feature,
                    "n_speakers": n_speakers,

                    "pearson_hz": pearson_hz,
                    "pearson_hz_p": pearson_hz_p,
                    "pearson_hz_ci_low": (
                        pearson_bootstrap[
                            "hz_ci_low"
                        ]
                    ),
                    "pearson_hz_ci_high": (
                        pearson_bootstrap[
                            "hz_ci_high"
                        ]
                    ),

                    "pearson_lobanov": (
                        pearson_lob
                    ),
                    "pearson_lobanov_p": (
                        pearson_lob_p
                    ),
                    "pearson_lobanov_ci_low": (
                        pearson_bootstrap[
                            "lobanov_ci_low"
                        ]
                    ),
                    "pearson_lobanov_ci_high": (
                        pearson_bootstrap[
                            "lobanov_ci_high"
                        ]
                    ),

                    "pearson_difference": (
                        pearson_lob
                        - pearson_hz
                    ),
                    "pearson_difference_ci_low": (
                        pearson_bootstrap[
                            "difference_ci_low"
                        ]
                    ),
                    "pearson_difference_ci_high": (
                        pearson_bootstrap[
                            "difference_ci_high"
                        ]
                    ),
                    "pearson_probability_lobanov_higher": (
                        pearson_bootstrap[
                            "probability_lobanov_higher"
                        ]
                    ),

                    "spearman_hz": spearman_hz,
                    "spearman_hz_p": spearman_hz_p,
                    "spearman_hz_ci_low": (
                        spearman_bootstrap[
                            "hz_ci_low"
                        ]
                    ),
                    "spearman_hz_ci_high": (
                        spearman_bootstrap[
                            "hz_ci_high"
                        ]
                    ),

                    "spearman_lobanov": (
                        spearman_lob
                    ),
                    "spearman_lobanov_p": (
                        spearman_lob_p
                    ),
                    "spearman_lobanov_ci_low": (
                        spearman_bootstrap[
                            "lobanov_ci_low"
                        ]
                    ),
                    "spearman_lobanov_ci_high": (
                        spearman_bootstrap[
                            "lobanov_ci_high"
                        ]
                    ),

                    "spearman_difference": (
                        spearman_lob
                        - spearman_hz
                    ),
                    "spearman_difference_ci_low": (
                        spearman_bootstrap[
                            "difference_ci_low"
                        ]
                    ),
                    "spearman_difference_ci_high": (
                        spearman_bootstrap[
                            "difference_ci_high"
                        ]
                    ),
                    "spearman_probability_lobanov_higher": (
                        spearman_bootstrap[
                            "probability_lobanov_higher"
                        ]
                    ),
                }
            )

    if not result_rows:
        raise RuntimeError(
            "No valid features were analyzed."
        )

    results = pd.DataFrame(
        result_rows
    )

    for method in (
        "pearson",
        "spearman",
    ):
        for representation in (
            "hz",
            "lobanov",
        ):
            p_column = (
                f"{method}_{representation}_p"
            )

            q_column = (
                f"{method}_{representation}_q"
            )

            results[q_column] = bh_adjust(
                results[p_column]
            )

    block_order = pd.Categorical(
        results["block"],
        categories=[
            "Centroids",
            "Geometry",
        ],
        ordered=True,
    )

    results = (
        results.assign(
            _block_order=block_order
        )
        .sort_values(
            [
                "_block_order",
                "feature",
            ]
        )
        .drop(columns="_block_order")
        .reset_index(drop=True)
    )

    summary_rows = []

    for block in (
        "Centroids",
        "Geometry",
    ):
        subset = results[
            results["block"] == block
        ]

        summary_rows.append(
            {
                "block": block,
                "n_features": len(subset),
                "median_pearson_hz": (
                    subset[
                        "pearson_hz"
                    ].median()
                ),
                "median_pearson_lobanov": (
                    subset[
                        "pearson_lobanov"
                    ].median()
                ),
                "median_spearman_hz": (
                    subset[
                        "spearman_hz"
                    ].median()
                ),
                "median_spearman_lobanov": (
                    subset[
                        "spearman_lobanov"
                    ].median()
                ),
                "pearson_features_lobanov_higher": int(
                    (
                        subset[
                            "pearson_lobanov"
                        ]
                        > subset[
                            "pearson_hz"
                        ]
                    ).sum()
                ),
                "spearman_features_lobanov_higher": int(
                    (
                        subset[
                            "spearman_lobanov"
                        ]
                        > subset[
                            "spearman_hz"
                        ]
                    ).sum()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    results.to_csv(
        output_dir
        / "half_correlation_results.csv",
        index=False,
    )

    summary.to_csv(
        output_dir
        / "half_correlation_summary.csv",
        index=False,
    )

    plot_heatmap(
        results,
        assets_dir
        / "correlation_heatmap.png",
    )

    plot_representation_comparison(
        results,
        "pearson",
        assets_dir
        / "pearson_hz_vs_lobanov.png",
    )

    plot_representation_comparison(
        results,
        "spearman",
        assets_dir
        / "spearman_hz_vs_lobanov.png",
    )

    report_path = (
        output_dir
        / "half_correlation_report.html"
    )

    create_html_report(
        results=results,
        summary=summary,
        output_path=report_path,
        assets_dir_name="assets",
        bootstrap_samples=args.bootstrap,
        hz_root=hz_root,
        lobanov_root=lobanov_root,
    )

    print()
    print("Analysis completed.")
    print(f"Features analyzed: {len(results)}")
    print(
        "HTML report: "
        f"{report_path}"
    )
    print(
        "Detailed CSV: "
        f"{output_dir / 'half_correlation_results.csv'}"
    )
    print(
        "Summary CSV: "
        f"{output_dir / 'half_correlation_summary.csv'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
