#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# Experiment configuration
# ============================================================

FEATURE_SETS = {
    "F1_F2_F3": [
        "F1",
        "F2",
        "F3",
    ],
    "F1_F2": [
        "F1",
        "F2",
    ],
    "ALL_SAFE_NUMERIC": [
        "F1",
        "F2",
        "F3",
        "B1_Hz",
        "B2_Hz",
        "B3_Hz",
    ],
}

FEATURE_ORDER = [
    "F1_F2_F3",
    "F1_F2",
    "ALL_SAFE_NUMERIC",
]

MODEL_ORDER = [
    "Logistic regression",
    "Random forest",
    "QDA",
    "Nearest centroid",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare four vowel classifiers using three "
            "acoustic feature sets with speaker-disjoint folds."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed/new_fave_points"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/vowel_classifier_12way"),
    )

    parser.add_argument(
        "--label-column",
        default="label",
    )

    parser.add_argument(
        "--vowels",
        nargs="+",
        default=["a", "e", "i", "o", "u"],
    )

    parser.add_argument(
        "--folds",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--split-search",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--rf-trees",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260727,
    )

    return parser.parse_args()


def slug(text: str) -> str:
    return re.sub(
        r"[^a-zA-Z0-9]+",
        "_",
        text,
    ).strip("_").lower()


# ============================================================
# Data loading
# ============================================================

def load_data(
    folder: Path,
    label_column: str,
    vowels: list[str],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    required_features = sorted(
        {
            feature
            for feature_list in FEATURE_SETS.values()
            for feature in feature_list
        }
    )

    for path in sorted(folder.glob("*.csv")):
        dataframe = pd.read_csv(
            path,
            low_memory=False,
        )

        if label_column not in dataframe.columns:
            print(
                f"Skipping {path.name}: "
                f"no {label_column!r} column"
            )
            continue

        dataframe["vowel"] = (
            dataframe[label_column]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.replace("/", "", regex=False)
        )

        dataframe = dataframe.loc[
            dataframe["vowel"].isin(vowels)
        ].copy()

        for column in [
            "F1",
            "F2",
            "F3",
            "B1",
            "B2",
            "B3",
        ]:
            if column in dataframe.columns:
                dataframe[column] = pd.to_numeric(
                    dataframe[column],
                    errors="coerce",
                )

        # B1/B2/B3 are stored in natural-log scale in
        # the new-fave/FastTrack points output.
        for number in (1, 2, 3):
            source = f"B{number}"
            target = f"B{number}_Hz"

            if target in dataframe.columns:
                dataframe[target] = pd.to_numeric(
                    dataframe[target],
                    errors="coerce",
                )

            elif source in dataframe.columns:
                with np.errstate(
                    over="ignore",
                    invalid="ignore",
                ):
                    dataframe[target] = np.exp(
                        dataframe[source].to_numpy(float)
                    )

        missing = [
            feature
            for feature in required_features
            if feature not in dataframe.columns
        ]

        if missing:
            print(
                f"Skipping {path.name}: "
                f"missing columns {missing}"
            )
            continue

        dataframe["speaker"] = path.stem

        frames.append(
            dataframe[
                [
                    "speaker",
                    "vowel",
                    *required_features,
                ]
            ]
        )

    if not frames:
        raise RuntimeError(
            f"No usable CSV files found in {folder}"
        )

    data = pd.concat(
        frames,
        ignore_index=True,
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    # F1 and F2 are required by all three feature sets.
    data = data.dropna(
        subset=[
            "F1",
            "F2",
        ]
    ).reset_index(drop=True)

    return data


# ============================================================
# Speaker-disjoint fold construction
# ============================================================

def split_quality(
    splits: list[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    groups: np.ndarray,
    vowels: list[str],
) -> float:
    target_fraction = 1.0 / len(splits)

    total_by_vowel = (
        pd.Series(labels)
        .value_counts()
        .reindex(vowels, fill_value=0)
    )

    score = 0.0

    for _, test_indices in splits:
        test_counts = (
            pd.Series(labels[test_indices])
            .value_counts()
            .reindex(vowels, fill_value=0)
        )

        test_fractions = (
            test_counts
            / total_by_vowel
        )

        score += float(
            (
                (
                    test_fractions
                    - target_fraction
                )
                ** 2
            ).sum()
        )

        # Strong penalty if a fold has no test token
        # for one of the five vowels.
        score += 1000.0 * float(
            (test_counts == 0).any()
        )

        speaker_fraction = (
            len(np.unique(groups[test_indices]))
            / len(np.unique(groups))
        )

        score += 0.25 * (
            speaker_fraction
            - target_fraction
        ) ** 2

    return score


def make_splits(
    data: pd.DataFrame,
    number_of_folds: int,
    attempts: int,
    seed: int,
    vowels: list[str],
) -> tuple[
    list[tuple[np.ndarray, np.ndarray]],
    int,
    float,
]:
    labels = data["vowel"].to_numpy()
    groups = data["speaker"].to_numpy()

    dummy_features = np.zeros(
        (len(data), 1)
    )

    best_result = None

    for offset in range(attempts):
        candidate_seed = seed + offset

        splitter = StratifiedGroupKFold(
            n_splits=number_of_folds,
            shuffle=True,
            random_state=candidate_seed,
        )

        candidate_splits = list(
            splitter.split(
                dummy_features,
                labels,
                groups,
            )
        )

        quality = split_quality(
            candidate_splits,
            labels,
            groups,
            vowels,
        )

        if (
            best_result is None
            or quality < best_result[0]
        ):
            best_result = (
                quality,
                candidate_seed,
                candidate_splits,
            )

    if best_result is None:
        raise RuntimeError(
            "No valid grouped split was generated."
        )

    quality, selected_seed, splits = best_result

    return (
        splits,
        selected_seed,
        quality,
    )


# ============================================================
# Models
# ============================================================

def create_model(
    model_name: str,
    seed: int,
    random_forest_trees: int,
) -> Pipeline:
    common_steps = [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        )
    ]

    if model_name == "Logistic regression":
        return Pipeline(
            common_steps
            + [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2500,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )

    if model_name == "Random forest":
        return Pipeline(
            common_steps
            + [
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=random_forest_trees,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                )
            ]
        )

    if model_name == "QDA":
        return Pipeline(
            common_steps
            + [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    QuadraticDiscriminantAnalysis(
                        reg_param=0.1
                    ),
                ),
            ]
        )

    if model_name == "Nearest centroid":
        return Pipeline(
            common_steps
            + [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    NearestCentroid(),
                ),
            ]
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )


# ============================================================
# Plotting
# ============================================================

def plot_accuracy_bars(
    labels: list[str],
    values: np.ndarray,
    title: str,
    output_path: Path,
    horizontal: bool = False,
    baseline: float | None = None,
) -> None:
    values = np.asarray(
        values,
        dtype=float,
    )

    if horizontal:
        figure, axis = plt.subplots(
            figsize=(11, 7.5)
        )

        bars = axis.barh(
            labels,
            values,
        )

        axis.set_xlim(
            0,
            100,
        )

        axis.set_xlabel(
            "Out-of-fold test accuracy (%)"
        )

        axis.grid(
            axis="x",
            alpha=0.25,
        )

        axis.bar_label(
            bars,
            labels=[
                f"{value:.2f}%"
                for value in values
            ],
            padding=3,
        )

        if baseline is not None:
            axis.axvline(
                baseline,
                linestyle="--",
                label=(
                    "Majority-class baseline "
                    f"({baseline:.2f}%)"
                ),
            )

    else:
        figure, axis = plt.subplots(
            figsize=(8.5, 5.2)
        )

        bars = axis.bar(
            labels,
            values,
        )

        axis.set_ylim(
            0,
            100,
        )

        axis.set_ylabel(
            "Out-of-fold test accuracy (%)"
        )

        axis.tick_params(
            axis="x",
            labelrotation=18,
        )

        axis.grid(
            axis="y",
            alpha=0.25,
        )

        axis.bar_label(
            bars,
            labels=[
                f"{value:.2f}%"
                for value in values
            ],
            padding=3,
        )

        if baseline is not None:
            axis.axhline(
                baseline,
                linestyle="--",
                label=(
                    "Majority-class baseline "
                    f"({baseline:.2f}%)"
                ),
            )

    if baseline is not None:
        axis.legend()

    axis.set_title(title)

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    vowels: list[str],
    title: str,
    output_path: Path,
) -> np.ndarray:
    row_sums = matrix.sum(
        axis=1,
        keepdims=True,
    )

    row_percentages = np.divide(
        matrix * 100,
        row_sums,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_sums != 0,
    )

    figure, axis = plt.subplots(
        figsize=(6.2, 5.4)
    )

    image = axis.imshow(
        row_percentages,
        vmin=0,
        vmax=100,
    )

    figure.colorbar(
        image,
        ax=axis,
        label="Row percentage (%)",
    )

    axis.set_xticks(
        range(len(vowels)),
        vowels,
    )

    axis.set_yticks(
        range(len(vowels)),
        vowels,
    )

    axis.set_xlabel(
        "Predicted vowel"
    )

    axis.set_ylabel(
        "True vowel"
    )

    axis.set_title(title)

    for true_index in range(len(vowels)):
        for predicted_index in range(len(vowels)):
            value = row_percentages[
                true_index,
                predicted_index,
            ]

            axis.text(
                predicted_index,
                true_index,
                f"{value:.1f}",
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > 50
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)

    return row_percentages


def html_table(
    dataframe: pd.DataFrame,
) -> str:
    return (
        '<div class="table-wrap">'
        + dataframe.to_html(
            index=False,
            border=0,
            escape=True,
            na_rep="",
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
        + "</div>"
    )


# ============================================================
# Main experiment
# ============================================================

def main() -> None:
    arguments = parse_args()

    vowels = [
        vowel.lower().replace("/", "")
        for vowel in arguments.vowels
    ]

    output_dir = arguments.output_dir
    assets_dir = output_dir / "assets"
    confusion_dir = (
        assets_dir
        / "confusion_matrices"
    )

    confusion_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = load_data(
        arguments.input_dir,
        arguments.label_column,
        vowels,
    )

    splits, selected_split_seed, split_score = (
        make_splits(
            data,
            arguments.folds,
            arguments.split_search,
            arguments.seed,
            vowels,
        )
    )

    true_labels = data["vowel"].to_numpy()
    speakers = data["speaker"].to_numpy()

    summary_rows: list[dict] = []
    fold_rows: list[dict] = []
    vowel_rows: list[dict] = []
    warning_messages: list[str] = []

    confusion_matrices: dict[
        str,
        np.ndarray,
    ] = {}

    # Save which fold tests each speaker.
    fold_assignment_rows = []

    for fold_number, (_, test_indices) in enumerate(
        splits,
        start=1,
    ):
        for speaker in sorted(
            np.unique(
                speakers[test_indices]
            )
        ):
            fold_assignment_rows.append(
                {
                    "speaker": speaker,
                    "test_fold": fold_number,
                }
            )

    pd.DataFrame(
        fold_assignment_rows
    ).to_csv(
        output_dir
        / "speaker_fold_assignment.csv",
        index=False,
    )

    # ========================================================
    # 4 models × 3 feature sets = 12 configurations
    # ========================================================

    for feature_set_name in FEATURE_ORDER:
        feature_columns = FEATURE_SETS[
            feature_set_name
        ]

        features = data[
            feature_columns
        ]

        for model_name in MODEL_ORDER:
            configuration_id = (
                f"{model_name}__{feature_set_name}"
            )

            print(
                f"Running {configuration_id}",
                flush=True,
            )

            out_of_fold_predictions = np.empty(
                len(data),
                dtype=object,
            )

            for fold_number, (
                train_indices,
                test_indices,
            ) in enumerate(
                splits,
                start=1,
            ):
                pipeline = create_model(
                    model_name,
                    arguments.seed + fold_number,
                    arguments.rf_trees,
                )

                with warnings.catch_warnings(
                    record=True
                ) as caught_warnings:
                    warnings.simplefilter(
                        "always"
                    )

                    pipeline.fit(
                        features.iloc[train_indices],
                        true_labels[train_indices],
                    )

                    fold_predictions = pipeline.predict(
                        features.iloc[test_indices]
                    )

                out_of_fold_predictions[
                    test_indices
                ] = fold_predictions

                for warning_item in caught_warnings:
                    warning_messages.append(
                        f"{configuration_id}, "
                        f"fold {fold_number}: "
                        f"{warning_item.message}"
                    )

                fold_rows.append(
                    {
                        "config_id": configuration_id,
                        "model": model_name,
                        "feature_set": (
                            feature_set_name
                        ),
                        "fold": fold_number,
                        "train_tokens": len(
                            train_indices
                        ),
                        "test_tokens": len(
                            test_indices
                        ),
                        "train_speakers": len(
                            np.unique(
                                speakers[
                                    train_indices
                                ]
                            )
                        ),
                        "test_speakers": len(
                            np.unique(
                                speakers[
                                    test_indices
                                ]
                            )
                        ),
                        "speaker_overlap": len(
                            set(
                                speakers[
                                    train_indices
                                ]
                            )
                            & set(
                                speakers[
                                    test_indices
                                ]
                            )
                        ),
                        "accuracy": accuracy_score(
                            true_labels[
                                test_indices
                            ],
                            fold_predictions,
                        ),
                        "balanced_accuracy": (
                            balanced_accuracy_score(
                                true_labels[
                                    test_indices
                                ],
                                fold_predictions,
                            )
                        ),
                        "macro_f1": f1_score(
                            true_labels[
                                test_indices
                            ],
                            fold_predictions,
                            labels=vowels,
                            average="macro",
                            zero_division=0,
                        ),
                    }
                )

            out_of_fold_predictions = (
                out_of_fold_predictions.astype(str)
            )

            configuration_fold_results = (
                pd.DataFrame(
                    [
                        row
                        for row in fold_rows
                        if row["config_id"]
                        == configuration_id
                    ]
                )
            )

            classification = classification_report(
                true_labels,
                out_of_fold_predictions,
                labels=vowels,
                output_dict=True,
                zero_division=0,
            )

            for vowel in vowels:
                vowel_rows.append(
                    {
                        "config_id": (
                            configuration_id
                        ),
                        "model": model_name,
                        "feature_set": (
                            feature_set_name
                        ),
                        "vowel": vowel,
                        "precision": (
                            classification[
                                vowel
                            ]["precision"]
                        ),
                        "recall": (
                            classification[
                                vowel
                            ]["recall"]
                        ),
                        "f1": (
                            classification[
                                vowel
                            ]["f1-score"]
                        ),
                        "support": (
                            classification[
                                vowel
                            ]["support"]
                        ),
                    }
                )

            speaker_balanced_accuracies = []

            for speaker in np.unique(speakers):
                speaker_mask = (
                    speakers == speaker
                )

                speaker_balanced_accuracies.append(
                    balanced_accuracy_score(
                        true_labels[
                            speaker_mask
                        ],
                        out_of_fold_predictions[
                            speaker_mask
                        ],
                    )
                )

            summary_rows.append(
                {
                    "config_id": configuration_id,
                    "model": model_name,
                    "feature_set": (
                        feature_set_name
                    ),
                    "n_features": len(
                        feature_columns
                    ),
                    "features": ", ".join(
                        feature_columns
                    ),

                    # Primary result requested:
                    # percentage of correct test tokens.
                    "accuracy_oof": accuracy_score(
                        true_labels,
                        out_of_fold_predictions,
                    ),

                    # Additional metrics kept because
                    # the five vowel classes are unbalanced.
                    "balanced_accuracy_oof": (
                        balanced_accuracy_score(
                            true_labels,
                            out_of_fold_predictions,
                        )
                    ),
                    "speaker_macro_balanced_accuracy": (
                        np.mean(
                            speaker_balanced_accuracies
                        )
                    ),
                    "macro_f1_oof": f1_score(
                        true_labels,
                        out_of_fold_predictions,
                        labels=vowels,
                        average="macro",
                        zero_division=0,
                    ),
                    "fold_accuracy_mean": (
                        configuration_fold_results[
                            "accuracy"
                        ].mean()
                    ),
                    "fold_accuracy_sd": (
                        configuration_fold_results[
                            "accuracy"
                        ].std(ddof=1)
                    ),
                }
            )

            confusion_matrices[
                configuration_id
            ] = confusion_matrix(
                true_labels,
                out_of_fold_predictions,
                labels=vowels,
            )

    # ========================================================
    # Results tables
    # ========================================================

    summary = pd.DataFrame(
        summary_rows
    ).sort_values(
        "accuracy_oof",
        ascending=False,
    ).reset_index(drop=True)

    summary.insert(
        0,
        "rank",
        np.arange(
            1,
            len(summary) + 1,
        ),
    )

    fold_results = pd.DataFrame(
        fold_rows
    )

    vowel_results = pd.DataFrame(
        vowel_rows
    )

    # Majority-class reference.
    vowel_counts = (
        data["vowel"]
        .value_counts()
        .reindex(
            vowels,
            fill_value=0,
        )
    )

    majority_baseline = (
        100
        * vowel_counts.max()
        / vowel_counts.sum()
    )

    # ========================================================
    # Accuracy graphs
    # ========================================================

    overall_sorted = summary.sort_values(
        "accuracy_oof"
    )

    plot_accuracy_bars(
        labels=(
            overall_sorted["model"]
            + " — "
            + overall_sorted["feature_set"]
        ).tolist(),
        values=(
            100
            * overall_sorted[
                "accuracy_oof"
            ]
        ),
        title=(
            "All 12 classifier × "
            "feature-set combinations"
        ),
        output_path=(
            assets_dir
            / "accuracy_all_12.png"
        ),
        horizontal=True,
        baseline=majority_baseline,
    )

    for feature_set_name in FEATURE_ORDER:
        feature_summary = (
            summary.loc[
                summary["feature_set"]
                == feature_set_name
            ]
            .set_index("model")
            .reindex(MODEL_ORDER)
        )

        plot_accuracy_bars(
            labels=MODEL_ORDER,
            values=(
                100
                * feature_summary[
                    "accuracy_oof"
                ]
            ),
            title=(
                "Test accuracy — "
                f"{feature_set_name}"
            ),
            output_path=(
                assets_dir
                / (
                    "accuracy_"
                    f"{slug(feature_set_name)}"
                    ".png"
                )
            ),
            horizontal=False,
            baseline=majority_baseline,
        )

    # ========================================================
    # All 12 confusion matrices
    # ========================================================

    confusion_html_sections = []
    confusion_long_rows = []

    for _, row in summary.iterrows():
        configuration_id = row[
            "config_id"
        ]

        image_filename = (
            f"{slug(configuration_id)}.png"
        )

        row_percentages = (
            plot_confusion_matrix(
                matrix=confusion_matrices[
                    configuration_id
                ],
                vowels=vowels,
                title=configuration_id.replace(
                    "__",
                    " — ",
                ),
                output_path=(
                    confusion_dir
                    / image_filename
                ),
            )
        )

        matrix = confusion_matrices[
            configuration_id
        ]

        for true_index, true_vowel in enumerate(
            vowels
        ):
            for (
                predicted_index,
                predicted_vowel,
            ) in enumerate(vowels):
                confusion_long_rows.append(
                    {
                        "config_id": (
                            configuration_id
                        ),
                        "true_vowel": (
                            true_vowel
                        ),
                        "predicted_vowel": (
                            predicted_vowel
                        ),
                        "count": matrix[
                            true_index,
                            predicted_index,
                        ],
                        "row_percent": (
                            row_percentages[
                                true_index,
                                predicted_index,
                            ]
                        ),
                    }
                )

        confusion_html_sections.append(
            "<h3>"
            + html.escape(
                configuration_id.replace(
                    "__",
                    " — ",
                )
            )
            + "</h3>"
            + (
                '<img src="assets/'
                "confusion_matrices/"
                f'{image_filename}" '
                'alt="'
                f'{html.escape(configuration_id)}">'
            )
        )

    # ========================================================
    # Save CSV files
    # ========================================================

    summary.to_csv(
        output_dir
        / "model_summary_12way.csv",
        index=False,
    )

    fold_results.to_csv(
        output_dir
        / "fold_results_12way.csv",
        index=False,
    )

    vowel_results.to_csv(
        output_dir
        / "per_vowel_results_12way.csv",
        index=False,
    )

    pd.DataFrame(
        confusion_long_rows
    ).to_csv(
        output_dir
        / "confusion_matrices_12way.csv",
        index=False,
    )

    unique_warnings = list(
        dict.fromkeys(
            warning_messages
        )
    )

    (
        output_dir
        / "warnings.txt"
    ).write_text(
        "\n".join(unique_warnings),
        encoding="utf-8",
    )

    # Percentage columns for presentation.
    display_summary = summary.copy()

    metric_columns = [
        "accuracy_oof",
        "balanced_accuracy_oof",
        "speaker_macro_balanced_accuracy",
        "macro_f1_oof",
        "fold_accuracy_mean",
        "fold_accuracy_sd",
    ]

    for column in metric_columns:
        display_summary[
            f"{column}_percent"
        ] = (
            100
            * display_summary[column]
        )

    display_summary = display_summary[
        [
            "rank",
            "model",
            "feature_set",
            "n_features",
            "accuracy_oof_percent",
            "balanced_accuracy_oof_percent",
            (
                "speaker_macro_"
                "balanced_accuracy_percent"
            ),
            "macro_f1_oof_percent",
            "fold_accuracy_mean_percent",
            "fold_accuracy_sd_percent",
        ]
    ]

    # ========================================================
    # HTML sections for each feature set
    # ========================================================

    feature_set_html = []

    for feature_set_name in FEATURE_ORDER:
        feature_set_html.append(
            f"<h2>{feature_set_name}</h2>"
            + (
                '<img src="assets/'
                f"accuracy_{slug(feature_set_name)}"
                '.png" alt="'
                f'{feature_set_name}">'
            )
            + html_table(
                display_summary.loc[
                    display_summary[
                        "feature_set"
                    ]
                    == feature_set_name
                ]
            )
        )

    best = summary.iloc[0]

    # ========================================================
    # HTML report
    # ========================================================

    report = f"""<!doctype html>
<html lang="en">

<head>
<meta charset="utf-8">

<title>
12-way vowel classifier comparison
</title>

<style>
body {{
    font-family: Arial, Helvetica, sans-serif;
    margin: 2rem;
    line-height: 1.45;
    color: #222;
    max-width: 1500px;
}}

h1 {{
    margin-bottom: .3rem;
}}

h2 {{
    margin-top: 2.5rem;
    border-bottom: 2px solid #ddd;
    padding-bottom: .3rem;
}}

.good,
.note {{
    padding: 1rem;
    margin: 1rem 0;
}}

.good {{
    background: #eef8ee;
    border-left: 4px solid #4c8;
}}

.note {{
    background: #f5f5f5;
    border-left: 4px solid #777;
}}

.table-wrap {{
    overflow-x: auto;
    margin: 1rem 0 2rem;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    font-size: .82rem;
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
Vowel classification:
4 classifiers × 3 feature sets
</h1>

<div class="good">

<b>
Best by ordinary out-of-fold test accuracy:
</b>

{html.escape(str(best["model"]))}
with
{html.escape(str(best["feature_set"]))}

—

<b>
{100 * best["accuracy_oof"]:.2f}%
correct test tokens
</b>.

</div>

<div class="note">

<b>Primary plotted result:</b>

ordinary out-of-fold accuracy, calculated as
the number of correct test-token predictions
divided by the total number of test tokens.

Every speaker is held out once.

Balanced accuracy, speaker-macro balanced accuracy
and Macro-F1 remain in the tables because the vowel
classes are strongly unbalanced, particularly /u/.

</div>

<h2>Dataset</h2>

<ul>

<li>
Retained vowel tokens:
<b>{len(data):,}</b>
</li>

<li>
Speaker/audio groups:
<b>{data["speaker"].nunique()}</b>
</li>

<li>
Vowels:
<b>{", ".join(vowels)}</b>
</li>

<li>
Speaker-disjoint folds:
<b>{arguments.folds}</b>
</li>

<li>
Selected split seed:
<code>{selected_split_seed}</code>
</li>

<li>
Candidate split searches:
<b>{arguments.split_search}</b>
</li>

</ul>

<h2>Feature sets</h2>

{html_table(
    pd.DataFrame(
        [
            {
                "feature_set": name,
                "n_features": len(FEATURE_SETS[name]),
                "features": ", ".join(FEATURE_SETS[name]),
            }
            for name in FEATURE_ORDER
        ]
    )
)}

<h2>All 12 results</h2>

<img
src="assets/accuracy_all_12.png"
alt="All 12 classifier results"
>

{html_table(display_summary)}

{''.join(feature_set_html)}

<h2>Per-vowel metrics</h2>

<p>
Precision, recall and F1 are calculated from all
out-of-fold predictions for each configuration.
</p>

{html_table(vowel_results)}

<h2>Fold-level results</h2>

<p>
Exactly the same speaker-disjoint folds are used for
all 12 configurations.

Selected split seed:
<code>{selected_split_seed}</code>.

Split-search score:
<code>{split_score:.8f}</code>.
</p>

{html_table(fold_results)}

<h2>Confusion matrices</h2>

<p>
Rows are the true vowels and columns are the predicted
vowels. Each row is normalized to 100%.
</p>

{''.join(confusion_html_sections)}

<h2>Methodological notes</h2>

<ul>

<li>
The primary ranking uses ordinary out-of-fold
test accuracy.
</li>

<li>
The same folds are used for every model and feature set.
</li>

<li>
No speaker/audio identifier occurs in both training and
test data within the same fold.
</li>

<li>
Logistic regression uses balanced class weights.
</li>

<li>
Random forest uses balanced subsample class weights.
</li>

<li>
QDA uses regularization parameter
<code>reg_param=0.1</code>.
</li>

<li>
Nearest centroid and the parametric models receive
standardized features where appropriate.
</li>

<li>
Missing acoustic measurements are imputed from the
training-fold median through the model pipeline.
</li>

<li>
B1, B2 and B3 are converted to Hz using
<code>exp(B)</code>.
</li>

<li>
The test folds are not used to tune model
hyperparameters.
</li>

</ul>

<h2>Saved files</h2>

<ul>

<li>
<code>vowel_classifier_12way_report.html</code>
</li>

<li>
<code>model_summary_12way.csv</code>
</li>

<li>
<code>fold_results_12way.csv</code>
</li>

<li>
<code>per_vowel_results_12way.csv</code>
</li>

<li>
<code>confusion_matrices_12way.csv</code>
</li>

<li>
<code>speaker_fold_assignment.csv</code>
</li>

<li>
<code>warnings.txt</code>
</li>

</ul>

</body>
</html>
"""

    report_path = (
        output_dir
        / "vowel_classifier_12way_report.html"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(
        f"Best: {best['model']} "
        f"× {best['feature_set']}"
    )

    print(
        "OOF test accuracy: "
        f"{100 * best['accuracy_oof']:.2f}%"
    )

    print(
        f"Report: {report_path}"
    )


if __name__ == "__main__":
    main()
