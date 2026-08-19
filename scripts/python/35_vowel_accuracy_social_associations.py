#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relate speaker-level vowel-classification balanced accuracy "
            "to social characteristics."
        )
    )

    parser.add_argument(
        "--speaker-results",
        type=Path,
        default=Path(
            "results/vowel_classification_raw_formants_only/"
            "per_speaker_results.csv"
        ),
    )

    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/social_metadata.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vowel_accuracy_social_associations"),
    )

    parser.add_argument(
        "--config-id",
        default="Logistic regression__F1_F2_F3",
    )

    parser.add_argument(
        "--speaker-key",
        default="Time.code.and.speaker",
    )

    parser.add_argument(
        "--sex-column",
        default="Sex",
    )

    parser.add_argument(
        "--country-column",
        default="CountryRes",
    )

    parser.add_argument(
        "--education-column",
        default="Education",
    )

    parser.add_argument(
        "--income-column",
        default="Amount_mw_earnings",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260804,
    )

    return parser.parse_args()


def normalize_id(value: object) -> str:
    """
    Normalize speaker/file identifiers before joining both tables.

    Examples:
        061ea7d0-Audio_SP16.csv
        061ea7d0-Audio_SP16.wav
        061ea7d0-audio_sp16

    become the same identifier.
    """
    if pd.isna(value):
        return ""

    text = Path(str(value).strip()).name.lower()

    text = re.sub(
        r"\.(csv|wav|wave|textgrid|txt|tsv)$",
        "",
        text,
    )

    text = re.sub(
        r"(_tracks_track_features|_track_features|_tracks|_points)$",
        "",
        text,
    )

    return re.sub(r"[^a-z0-9]+", "", text)


def clean_category(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip()

    missing_values = {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "missing",
        "unknown",
        "?",
        "-",
    }

    return out.mask(
        out.str.lower().isin(missing_values) | out.isna()
    )


def parse_income(value: object) -> float:
    """
    Parse numeric income values, including values with commas,
    points, spaces or currency symbols.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        return float(value)

    text = str(value).strip()

    text = re.sub(
        r"[^0-9,\.\-+]",
        "",
        text,
    )

    if text in {"", "+", "-"}:
        return np.nan

    if "," in text and "." in text:
        # Last separator is treated as decimal separator.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")

    elif "," in text:
        parts = text.split(",")

        if len(parts) == 2 and 1 <= len(parts[1]) <= 2:
            text = parts[0] + "." + parts[1]
        else:
            text = text.replace(",", "")

    elif "." in text:
        parts = text.split(".")

        # Treat 1.200 as 1200 rather than 1.2.
        if len(parts) > 2 or (
            len(parts) == 2
            and len(parts[1]) == 3
        ):
            text = text.replace(".", "")

    try:
        return float(text)
    except ValueError:
        return np.nan


def bh_fdr(p_values: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg false-discovery-rate correction.
    """
    result = pd.Series(
        np.nan,
        index=p_values.index,
        dtype=float,
    )

    valid = p_values.dropna().astype(float)

    if valid.empty:
        return result

    ordered = valid.sort_values()

    m = len(ordered)

    adjusted = (
        ordered.to_numpy()
        * m
        / np.arange(1, m + 1)
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    result.loc[ordered.index] = np.clip(
        adjusted,
        0,
        1,
    )

    return result


def welch_t_test(
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> dict:
    """
    Welch independent-samples t-test.

    Used here for Sex, which is expected to contain exactly
    two non-missing categories.
    """
    work = data[
        [group_column, value_column]
    ].dropna()

    groups = sorted(
        work[group_column].astype(str).unique()
    )

    if len(groups) != 2:
        return {
            "test": "Welch t-test",
            "variable": group_column,
            "n": len(work),
            "groups": len(groups),
            "statistic": np.nan,
            "df1": np.nan,
            "df2": np.nan,
            "effect_name": "mean difference",
            "effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "status": (
                "Not run: expected exactly 2 groups, "
                f"found {len(groups)}"
            ),
        }

    group_1 = work.loc[
        work[group_column].astype(str) == groups[0],
        value_column,
    ].to_numpy(float)

    group_2 = work.loc[
        work[group_column].astype(str) == groups[1],
        value_column,
    ].to_numpy(float)

    if min(len(group_1), len(group_2)) < 2:
        return {
            "test": "Welch t-test",
            "variable": group_column,
            "n": len(work),
            "groups": 2,
            "statistic": np.nan,
            "df1": np.nan,
            "df2": np.nan,
            "effect_name": "mean difference",
            "effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "status": (
                "Not run: each group needs at least "
                "2 observations"
            ),
        }

    test_result = stats.ttest_ind(
        group_1,
        group_2,
        equal_var=False,
        nan_policy="omit",
    )

    variance_1 = np.var(
        group_1,
        ddof=1,
    )

    variance_2 = np.var(
        group_2,
        ddof=1,
    )

    standard_error_squared = (
        variance_1 / len(group_1)
        + variance_2 / len(group_2)
    )

    standard_error = math.sqrt(
        standard_error_squared
    )

    degrees_of_freedom = (
        standard_error_squared**2
        /
        (
            ((variance_1 / len(group_1)) ** 2)
            / (len(group_1) - 1)
            +
            ((variance_2 / len(group_2)) ** 2)
            / (len(group_2) - 1)
        )
    )

    mean_difference = (
        np.mean(group_1)
        - np.mean(group_2)
    )

    critical_value = stats.t.ppf(
        0.975,
        degrees_of_freedom,
    )

    ci_low = (
        mean_difference
        - critical_value * standard_error
    )

    ci_high = (
        mean_difference
        + critical_value * standard_error
    )

    return {
        "test": "Welch t-test",
        "variable": group_column,
        "n": len(work),
        "groups": 2,
        "statistic": float(test_result.statistic),
        "df1": float(degrees_of_freedom),
        "df2": np.nan,
        "effect_name": (
            f"mean({groups[0]}) - mean({groups[1]})"
        ),
        "effect": float(mean_difference),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(test_result.pvalue),
        "status": "OK",
    }


def welch_anova(
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> dict:
    """
    Welch one-way ANOVA.

    Groups with fewer than two observations cannot have a
    sample variance and are excluded from the inferential test.
    They remain present in descriptive summaries and graphs.
    """
    work = data[
        [group_column, value_column]
    ].dropna()

    grouped_values = [
        (
            str(group_name),
            group_data[value_column].to_numpy(float),
        )
        for group_name, group_data
        in work.groupby(group_column)
        if len(group_data) >= 2
    ]

    number_of_groups = len(grouped_values)

    if number_of_groups < 2:
        return {
            "test": "Welch ANOVA",
            "variable": group_column,
            "n": sum(
                len(values)
                for _, values in grouped_values
            ),
            "groups": number_of_groups,
            "statistic": np.nan,
            "df1": np.nan,
            "df2": np.nan,
            "effect_name": "range of group means",
            "effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "status": (
                "Not run: fewer than two groups "
                "with n >= 2"
            ),
        }

    arrays = [
        values
        for _, values in grouped_values
    ]

    sample_sizes = np.array(
        [len(values) for values in arrays],
        dtype=float,
    )

    means = np.array(
        [np.mean(values) for values in arrays],
        dtype=float,
    )

    variances = np.array(
        [
            np.var(values, ddof=1)
            for values in arrays
        ],
        dtype=float,
    )

    if np.any(variances <= 0):
        return {
            "test": "Welch ANOVA",
            "variable": group_column,
            "n": int(sample_sizes.sum()),
            "groups": number_of_groups,
            "statistic": np.nan,
            "df1": np.nan,
            "df2": np.nan,
            "effect_name": "range of group means",
            "effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "status": (
                "Not run: at least one group "
                "has zero variance"
            ),
        }

    weights = (
        sample_sizes
        / variances
    )

    total_weight = weights.sum()

    weighted_mean = (
        np.sum(weights * means)
        / total_weight
    )

    numerator = (
        np.sum(
            weights
            * (means - weighted_mean) ** 2
        )
        / (number_of_groups - 1)
    )

    auxiliary_term = np.sum(
        (
            (1 - weights / total_weight) ** 2
        )
        / (sample_sizes - 1)
    )

    denominator = (
        1
        +
        (
            2
            * (number_of_groups - 2)
            / (number_of_groups**2 - 1)
        )
        * auxiliary_term
    )

    f_statistic = (
        numerator
        / denominator
    )

    df1 = (
        number_of_groups
        - 1
    )

    df2 = (
        (number_of_groups**2 - 1)
        /
        (3 * auxiliary_term)
    )

    p_value = stats.f.sf(
        f_statistic,
        df1,
        df2,
    )

    return {
        "test": "Welch ANOVA",
        "variable": group_column,
        "n": int(sample_sizes.sum()),
        "groups": number_of_groups,
        "statistic": float(f_statistic),
        "df1": float(df1),
        "df2": float(df2),
        "effect_name": "range of group means",
        "effect": float(
            means.max() - means.min()
        ),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": float(p_value),
        "status": "OK",
    }


def correlation_test(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    method: str,
) -> dict:
    work = data[
        [x_column, y_column]
    ].dropna().astype(float)

    if (
        len(work) < 3
        or work[x_column].nunique() < 2
        or work[y_column].nunique() < 2
    ):
        return {
            "test": f"{method} correlation",
            "variable": x_column,
            "n": len(work),
            "groups": np.nan,
            "statistic": np.nan,
            "df1": np.nan,
            "df2": np.nan,
            "effect_name": (
                "r"
                if method == "Pearson"
                else "rho"
            ),
            "effect": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "p_value": np.nan,
            "status": (
                "Not run: insufficient non-missing "
                "or non-constant data"
            ),
        }

    if method == "Pearson":
        result = stats.pearsonr(
            work[x_column],
            work[y_column],
        )

        effect_name = "r"

    else:
        result = stats.spearmanr(
            work[x_column],
            work[y_column],
        )

        effect_name = "rho"

    return {
        "test": f"{method} correlation",
        "variable": x_column,
        "n": len(work),
        "groups": np.nan,
        "statistic": float(result.statistic),
        "df1": np.nan,
        "df2": np.nan,
        "effect_name": effect_name,
        "effect": float(result.statistic),
        "ci_low": np.nan,
        "ci_high": np.nan,
        "p_value": float(result.pvalue),
        "status": "OK",
    }


def group_summary(
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
) -> pd.DataFrame:
    result = (
        data[
            [group_column, value_column]
        ]
        .dropna()
        .groupby(group_column)[value_column]
        .agg(
            n="size",
            mean="mean",
            sd="std",
            median="median",
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    result.insert(
        0,
        "variable",
        group_column,
    )

    return result


def create_boxplot(
    data: pd.DataFrame,
    group_column: str,
    value_column: str,
    output_path: Path,
    rng: np.random.Generator,
) -> None:
    work = data[
        [group_column, value_column]
    ].dropna()

    if work.empty:
        return

    groups = sorted(
        work[group_column].astype(str).unique()
    )

    values = [
        work.loc[
            work[group_column].astype(str) == group,
            value_column,
        ].to_numpy(float)
        for group in groups
    ]

    width = max(
        7,
        min(
            16,
            len(groups) * 1.2 + 3,
        ),
    )

    figure, axis = plt.subplots(
        figsize=(width, 5.4)
    )

    axis.boxplot(
        values,
        tick_labels=[
            f"{group}\n(n={len(group_values)})"
            for group, group_values
            in zip(groups, values)
        ],
    )

    for position, group_values in enumerate(
        values,
        start=1,
    ):
        jitter = rng.normal(
            0,
            0.045,
            len(group_values),
        )

        axis.scatter(
            position + jitter,
            group_values,
            alpha=0.75,
            s=28,
        )

    axis.set_ylabel(
        "Speaker-level balanced accuracy"
    )

    axis.set_xlabel(
        group_column
    )

    axis.set_ylim(
        0.15,
        1.0,
    )

    axis.set_title(
        f"Balanced accuracy by {group_column}"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    if len(groups) > 5:
        axis.tick_params(
            axis="x",
            labelrotation=35,
        )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def create_income_plot(
    data: pd.DataFrame,
    output_path: Path,
) -> None:
    work = (
        data[
            ["Income_raw", "balanced_accuracy"]
        ]
        .dropna()
        .astype(float)
        .sort_values("Income_raw")
    )

    if work.empty:
        return

    figure, axis = plt.subplots(
        figsize=(8.5, 5.5)
    )

    axis.scatter(
        work["Income_raw"],
        work["balanced_accuracy"],
        alpha=0.8,
        s=38,
    )

    if (
        len(work) >= 2
        and work["Income_raw"].nunique() >= 2
    ):
        slope, intercept = np.polyfit(
            work["Income_raw"],
            work["balanced_accuracy"],
            1,
        )

        x_line = np.linspace(
            work["Income_raw"].min(),
            work["Income_raw"].max(),
            200,
        )

        axis.plot(
            x_line,
            intercept + slope * x_line,
            linewidth=2,
            label="Least-squares line",
        )

        axis.legend()

    axis.set_xlabel(
        "Raw income"
    )

    axis.set_ylabel(
        "Speaker-level balanced accuracy"
    )

    axis.set_ylim(
        0.15,
        1.0,
    )

    axis.set_title(
        "Raw income and balanced accuracy"
    )

    axis.grid(
        alpha=0.25
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def html_table(
    dataframe: pd.DataFrame,
) -> str:
    if dataframe.empty:
        return "<p><em>No rows available.</em></p>"

    return (
        '<div class="table-wrap">'
        +
        dataframe.to_html(
            index=False,
            border=0,
            na_rep="",
            float_format=lambda value: f"{value:.4f}",
            escape=True,
        )
        +
        "</div>"
    )


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    assets_dir = output_dir / "assets"

    assets_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(
        args.seed
    )

    # ---------------------------------------------------------
    # Speaker-level classifier results
    # ---------------------------------------------------------
    performance = pd.read_csv(
        args.speaker_results,
        low_memory=False,
    )

    required_performance_columns = {
        "speaker",
        "balanced_accuracy",
    }

    missing_columns = (
        required_performance_columns
        - set(performance.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing classifier columns: "
            f"{sorted(missing_columns)}"
        )

    # Some output files may contain all tested configurations.
    # In this case, retain the corrected best configuration.
    if (
        "config_id" in performance.columns
        and performance["config_id"].nunique() > 1
    ):
        selected = performance.loc[
            performance["config_id"].astype(str)
            == args.config_id
        ].copy()

        if selected.empty:
            available = sorted(
                performance["config_id"]
                .astype(str)
                .unique()
            )

            raise ValueError(
                f"Configuration {args.config_id!r} "
                f"not found. Available: {available}"
            )

        performance = selected

    performance["balanced_accuracy"] = pd.to_numeric(
        performance["balanced_accuracy"],
        errors="coerce",
    )

    performance["join_id"] = performance[
        "speaker"
    ].map(normalize_id)

    performance = performance.dropna(
        subset=["balanced_accuracy"]
    )

    if performance["join_id"].duplicated().any():
        duplicated = performance.loc[
            performance["join_id"].duplicated(
                keep=False
            ),
            "speaker",
        ].tolist()

        raise ValueError(
            "Duplicated speaker identifiers in "
            f"the performance data: {duplicated[:10]}"
        )

    # ---------------------------------------------------------
    # Social metadata
    # ---------------------------------------------------------
    metadata = pd.read_csv(
        args.metadata,
        low_memory=False,
    )

    required_metadata_columns = {
        args.speaker_key,
        args.sex_column,
        args.country_column,
        args.education_column,
        args.income_column,
    }

    missing_columns = (
        required_metadata_columns
        - set(metadata.columns)
    )

    if missing_columns:
        raise KeyError(
            "Missing metadata columns: "
            f"{sorted(missing_columns)}"
        )

    metadata = metadata[
        [
            args.speaker_key,
            args.sex_column,
            args.country_column,
            args.education_column,
            args.income_column,
        ]
    ].copy()

    metadata["join_id"] = metadata[
        args.speaker_key
    ].map(normalize_id)

    metadata["Sex"] = clean_category(
        metadata[args.sex_column]
    )

    metadata["CountryRes"] = clean_category(
        metadata[args.country_column]
    )

    metadata["Education"] = clean_category(
        metadata[args.education_column]
    )

    metadata["Income_raw"] = metadata[
        args.income_column
    ].map(parse_income)

    duplicated_metadata = metadata.loc[
        metadata["join_id"].duplicated(
            keep=False
        )
    ].copy()

    if not duplicated_metadata.empty:
        print(
            "WARNING: duplicated metadata identifiers "
            "were found. The first row of each identifier "
            "will be retained."
        )

    metadata = metadata.drop_duplicates(
        "join_id",
        keep="first",
    )

    # ---------------------------------------------------------
    # Merge classification performance and metadata
    # ---------------------------------------------------------
    merged = performance.merge(
        metadata[
            [
                "join_id",
                args.speaker_key,
                "Sex",
                "CountryRes",
                "Education",
                "Income_raw",
            ]
        ],
        on="join_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    match_rate = float(
        (merged["_merge"] == "both").mean()
    )

    merged.to_csv(
        output_dir / "match_audit.csv",
        index=False,
    )

    if match_rate < 0.90:
        raise RuntimeError(
            f"Metadata match rate was only "
            f"{match_rate:.1%}. "
            "Inspect match_audit.csv before continuing."
        )

    data = merged.loc[
        merged["_merge"] == "both"
    ].copy()

    # ---------------------------------------------------------
    # Primary statistical tests
    # ---------------------------------------------------------
    tests = pd.DataFrame(
        [
            welch_t_test(
                data,
                "Sex",
                "balanced_accuracy",
            ),
            welch_anova(
                data,
                "CountryRes",
                "balanced_accuracy",
            ),
            welch_anova(
                data,
                "Education",
                "balanced_accuracy",
            ),
            correlation_test(
                data,
                "Income_raw",
                "balanced_accuracy",
                "Pearson",
            ),
            correlation_test(
                data,
                "Income_raw",
                "balanced_accuracy",
                "Spearman",
            ),
        ]
    )

    # Five primary tests are being performed.
    tests["q_value"] = bh_fdr(
        tests["p_value"]
    )

    # ---------------------------------------------------------
    # Descriptive group summaries
    # ---------------------------------------------------------
    summaries = pd.concat(
        [
            group_summary(
                data,
                "Sex",
                "balanced_accuracy",
            ),
            group_summary(
                data,
                "CountryRes",
                "balanced_accuracy",
            ),
            group_summary(
                data,
                "Education",
                "balanced_accuracy",
            ),
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Figures
    # ---------------------------------------------------------
    create_boxplot(
        data,
        "Sex",
        "balanced_accuracy",
        assets_dir / "accuracy_by_sex.png",
        rng,
    )

    create_boxplot(
        data,
        "CountryRes",
        "balanced_accuracy",
        assets_dir / "accuracy_by_country.png",
        rng,
    )

    create_boxplot(
        data,
        "Education",
        "balanced_accuracy",
        assets_dir / "accuracy_by_education.png",
        rng,
    )

    create_income_plot(
        data,
        assets_dir / "accuracy_by_income.png",
    )

    # ---------------------------------------------------------
    # Save CSV outputs
    # ---------------------------------------------------------
    export_columns = [
        column
        for column in [
            "speaker",
            "balanced_accuracy",
            "accuracy",
            "macro_f1",
            "n_tokens",
            "Sex",
            "CountryRes",
            "Education",
            "Income_raw",
            args.speaker_key,
            "join_id",
        ]
        if column in data.columns
    ]

    data[export_columns].to_csv(
        output_dir
        / "speaker_accuracy_social_merged.csv",
        index=False,
    )

    tests.to_csv(
        output_dir / "primary_tests.csv",
        index=False,
    )

    summaries.to_csv(
        output_dir / "group_summaries.csv",
        index=False,
    )

    duplicated_metadata.to_csv(
        output_dir / "duplicated_metadata_rows.csv",
        index=False,
    )

    merged.loc[
        merged["_merge"] != "both"
    ].to_csv(
        output_dir / "unmatched_speakers.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Automatic HTML interpretation
    # ---------------------------------------------------------
    conclusions = []

    for _, row in tests.iterrows():
        if row["status"] != "OK":
            interpretation = row["status"]

        elif row["q_value"] < 0.05:
            if row["test"] == "Welch t-test":
                interpretation = (
                    "FDR-significant evidence of a "
                    "difference between the two group means"
                )

            elif row["test"] == "Welch ANOVA":
                interpretation = (
                    "FDR-significant evidence that at least "
                    "one group mean differs"
                )

            else:
                direction = (
                    "positive"
                    if row["effect"] > 0
                    else "negative"
                )

                interpretation = (
                    f"FDR-significant {direction} association"
                )

        else:
            if row["test"] == "Welch t-test":
                interpretation = (
                    "no FDR-significant evidence of a "
                    "difference between the two group means"
                )

            elif row["test"] == "Welch ANOVA":
                interpretation = (
                    "no FDR-significant evidence that "
                    "the group means differ"
                )

            else:
                direction = (
                    "positive"
                    if row["effect"] > 0
                    else "negative"
                )

                interpretation = (
                    f"no FDR-significant {direction} association"
                )

        if pd.notna(row["p_value"]):
            numeric_part = (
                f"effect={row['effect']:.4f}, "
                f"p={row['p_value']:.4f}, "
                f"q={row['q_value']:.4f}"
            )
        else:
            numeric_part = "test unavailable"

        conclusions.append(
            "<li>"
            f"<b>{html.escape(str(row['test']))} — "
            f"{html.escape(str(row['variable']))}:</b> "
            f"{html.escape(str(interpretation))} "
            f"({numeric_part})."
            "</li>"
        )

    # ---------------------------------------------------------
    # HTML report
    # ---------------------------------------------------------
    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Vowel accuracy and social characteristics</title>
<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 2rem;
    line-height: 1.45;
    color: #222;
    max-width: 1450px;
}}

h1 {{
    margin-bottom: .25rem;
}}

h2 {{
    margin-top: 2.5rem;
    border-bottom: 2px solid #ddd;
    padding-bottom: .3rem;
}}

.note {{
    background: #f5f5f5;
    border-left: 4px solid #777;
    padding: 1rem;
    margin: 1rem 0;
}}

.good {{
    background: #eef8ee;
    border-left: 4px solid #4c8;
    padding: 1rem;
    margin: 1rem 0;
}}

.warning {{
    background: #fff4e5;
    border-left: 4px solid #d98c00;
    padding: 1rem;
    margin: 1rem 0;
}}

.table-wrap {{
    overflow-x: auto;
    border: 1px solid #ddd;
    margin: 1rem 0 2rem;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    font-size: .84rem;
}}

th,
td {{
    border: 1px solid #ddd;
    padding: .42rem;
    text-align: right;
    white-space: nowrap;
}}

th {{
    background: #eee;
}}

td:first-child,
th:first-child {{
    text-align: left;
}}

img {{
    max-width: 100%;
    border: 1px solid #ddd;
    margin: 1rem 0 2rem;
}}

code {{
    background: #f2f2f2;
    padding: .1rem .25rem;
}}
</style>
</head>

<body>

<h1>
Speaker-level vowel-classification performance
and social characteristics
</h1>

<div class="note">
<b>Dependent variable:</b>
out-of-fold speaker-level <b>balanced accuracy</b>
from the speaker-independent vowel classifier.
Each speaker contributes one performance value,
calculated when that speaker was outside the training set.
</div>

<h2>Automatic summary</h2>

<div class="good">
<ul>
{''.join(conclusions)}
</ul>
</div>

<p>
Benjamini–Hochberg false-discovery-rate correction
was applied jointly to the five primary tests.
The results concern classifier generalization,
not pronunciation quality and not causation.
</p>

<h2>Data and matching audit</h2>

<ul>
<li>
Speaker-level classifier file:
<code>{html.escape(str(args.speaker_results.resolve()))}</code>
</li>

<li>
Social metadata:
<code>{html.escape(str(args.metadata.resolve()))}</code>
</li>

<li>
Classifier observations:
<b>{len(performance)}</b>
</li>

<li>
Matched metadata observations:
<b>{len(data)}</b>
({match_rate:.1%})
</li>

<li>
Sex available:
<b>{data["Sex"].notna().sum()}</b>
</li>

<li>
Country available:
<b>{data["CountryRes"].notna().sum()}</b>
</li>

<li>
Education available:
<b>{data["Education"].notna().sum()}</b>
</li>

<li>
Raw income available:
<b>{data["Income_raw"].notna().sum()}</b>
</li>
</ul>

<h2>Primary inferential tests</h2>

<p>
The Welch t-test compares mean balanced accuracy
between the two sex groups.
Welch ANOVA tests whether at least one mean differs
across country-of-residence or education groups.
Pearson measures linear association with raw income,
whereas Spearman measures monotonic rank association.
</p>

{html_table(tests)}

<h2>Descriptive group summaries</h2>

{html_table(summaries)}

<h2>Sex</h2>

<img
src="assets/accuracy_by_sex.png"
alt="Balanced accuracy by sex"
>

<h2>Country of residence</h2>

<img
src="assets/accuracy_by_country.png"
alt="Balanced accuracy by country of residence"
>

<h2>Education</h2>

<img
src="assets/accuracy_by_education.png"
alt="Balanced accuracy by education"
>

<h2>Raw income</h2>

<img
src="assets/accuracy_by_income.png"
alt="Raw income and balanced accuracy"
>

<h2>Interpretation cautions</h2>

<ul>

<li>
Welch tests assess differences in
<b>mean</b> balanced accuracy.
A non-significant result does not prove that the
complete distributions are identical.
</li>

<li>
Country uses the metadata field
<code>{html.escape(args.country_column)}</code>,
interpreted here as country of residence,
not nationality.
</li>

<li>
Pearson measures linear association,
whereas Spearman measures monotonic rank association.
</li>

<li>
Groups with fewer than two observations are retained
in descriptive summaries but excluded from Welch ANOVA.
</li>

<li>
Small or highly unbalanced social groups produce
limited statistical power and wide uncertainty.
</li>

<li>
This analysis inherits the classifier's definition of
a speaker. Confirm that the classifier identifier
corresponds to a real individual rather than merely
to an audio file.
</li>

<li>
An association may reflect accent, representation in
the training data, recording quality, formant-extraction
quality or confounding among social variables.
</li>

</ul>

<h2>Saved outputs</h2>

<ul>
<li><code>vowel_accuracy_social_report.html</code></li>
<li><code>speaker_accuracy_social_merged.csv</code></li>
<li><code>primary_tests.csv</code></li>
<li><code>group_summaries.csv</code></li>
<li><code>match_audit.csv</code></li>
<li><code>unmatched_speakers.csv</code></li>
<li><code>duplicated_metadata_rows.csv</code></li>
</ul>

</body>
</html>
"""

    report_path = (
        output_dir
        / "vowel_accuracy_social_report.html"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(
        f"Matched speakers: "
        f"{len(data)}/{len(performance)} "
        f"({match_rate:.1%})"
    )

    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
