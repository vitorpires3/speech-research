#!/usr/bin/env python3
"""Aplica normalização de Lobanov aos CSVs points do new-fave.

O script:

1. lê os CSVs de data/processed/new_fave_points;
2. considera cada arquivo um locutor;
3. calcula os parâmetros de Lobanov usando /a e i o u/ juntas;
4. acrescenta F1_lobanov, F2_lobanov e F3_lobanov;
5. preserva os arquivos originais;
6. produz tabelas e gráficos de diagnóstico.

F1 e F2 correspondem ao uso tradicional do método.
F3 é uma extensão para futuras análises tridimensionais.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FORMANTS = ("F1", "F2", "F3")
TARGET_VOWELS = ("a", "e", "i", "o", "u")
VOWEL_ORDER = ("i", "e", "a", "o", "u")

VOWEL_MARKERS = {
    "a": "o",
    "e": "s",
    "i": "^",
    "o": "D",
    "u": "v",
}

ACCENTED_VOWEL_MAP = str.maketrans(
    {
        "á": "a",
        "é": "e",
        "í": "i",
        "ó": "o",
        "ú": "u",
        "ü": "u",
        "Á": "a",
        "É": "e",
        "Í": "i",
        "Ó": "o",
        "Ú": "u",
        "Ü": "u",
    }
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Aplica normalização de Lobanov aos arquivos points "
            "produzidos pelo new-fave."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data/processed/new_fave_points",
        help="Diretório com os CSVs originais do new-fave.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data/processed/new_fave_points_lobanov",
        help="Diretório onde serão gravados os CSVs normalizados.",
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=project_root / "results/lobanov_points",
        help="Diretório para tabelas e gráficos de diagnóstico.",
    )

    parser.add_argument(
        "--min-tokens-per-vowel",
        type=int,
        default=5,
        help=(
            "Quantidade mínima esperada por locutor e vogal. "
            "Valores abaixo disso são apenas sinalizados."
        ),
    )

    parser.add_argument(
        "--no-speaker-plots",
        action="store_true",
        help="Não gera os gráficos individuais dos locutores.",
    )

    return parser.parse_args()


def normalize_label(value: object) -> str | None:
    """Converte o rótulo para uma das cinco vogais do espanhol."""

    if pd.isna(value):
        return None

    label = str(value).strip().translate(ACCENTED_VOWEL_MAP).lower()

    if label in TARGET_VOWELS:
        return label

    return None


def safe_unique_strings(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).str.strip()

    return sorted(
        value
        for value in values.unique()
        if value
    )


def load_csv_files(
    input_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lê todos os CSVs e cria um DataFrame combinado."""

    csv_paths = sorted(input_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(
            f"Nenhum arquivo CSV encontrado em: {input_dir}"
        )

    frames: list[pd.DataFrame] = []
    inventory_rows: list[dict[str, object]] = []

    required_columns = {"label", *FORMANTS}

    for csv_path in csv_paths:
        print(f"Lendo: {csv_path.name}")

        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            raise RuntimeError(
                f"Não foi possível ler {csv_path}: {exc}"
            ) from exc

        missing = sorted(required_columns.difference(df.columns))

        if missing:
            raise ValueError(
                f"O arquivo {csv_path.name} não contém "
                f"as colunas obrigatórias: {missing}"
            )

        audio_id = csv_path.stem

        df = df.copy()

        df["source_csv"] = csv_path.name
        df["audio_id"] = audio_id
        df["source_row"] = np.arange(len(df), dtype=int)
        df["vowel"] = df["label"].map(normalize_label)

        for formant in FORMANTS:
            df[formant] = pd.to_numeric(
                df[formant],
                errors="coerce",
            )

        speaker_nums = (
            safe_unique_strings(df["speaker_num"])
            if "speaker_num" in df.columns
            else []
        )

        groups = (
            safe_unique_strings(df["group"])
            if "group" in df.columns
            else []
        )

        # Na estrutura atual, cada CSV representa uma gravação/locutor.
        # Caso surjam várias combinações speaker_num/group dentro do
        # mesmo arquivo, elas serão mantidas como chaves distintas.
        if len(speaker_nums) <= 1 and len(groups) <= 1:
            df["speaker_id"] = audio_id

        else:
            if "speaker_num" in df.columns:
                speaker_part = (
                    df["speaker_num"]
                    .astype("string")
                    .fillna("NA")
                )
            else:
                speaker_part = pd.Series(
                    "NA",
                    index=df.index,
                    dtype="string",
                )

            if "group" in df.columns:
                group_part = (
                    df["group"]
                    .astype("string")
                    .fillna("NA")
                )
            else:
                group_part = pd.Series(
                    "NA",
                    index=df.index,
                    dtype="string",
                )

            df["speaker_id"] = (
                audio_id
                + "__sp"
                + speaker_part.astype(str)
                + "__"
                + group_part.astype(str)
            )

        inventory_rows.append(
            {
                "source_csv": csv_path.name,
                "audio_id": audio_id,
                "n_rows": len(df),
                "n_unique_token_ids": (
                    int(df["id"].nunique(dropna=True))
                    if "id" in df.columns
                    else np.nan
                ),
                "speaker_num_values": "|".join(speaker_nums),
                "group_values": "|".join(groups),
                "n_speaker_keys": int(
                    df["speaker_id"].nunique()
                ),
            }
        )

        frames.append(df)

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    inventory = pd.DataFrame(inventory_rows)

    return combined, inventory


def calculate_lobanov_parameters(
    data: pd.DataFrame,
    min_tokens_per_vowel: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcula média e desvio RMS por locutor e formante."""

    parameter_rows: list[dict[str, object]] = []
    count_rows: list[dict[str, object]] = []

    for speaker_id, speaker_df in data.groupby(
        "speaker_id",
        sort=True,
    ):
        vowel_df = speaker_df[
            speaker_df["vowel"].isin(TARGET_VOWELS)
        ].copy()

        counts = (
            vowel_df.groupby(
                "vowel",
                observed=True,
            )
            .size()
            .reindex(
                TARGET_VOWELS,
                fill_value=0,
            )
        )

        missing_vowels = [
            vowel
            for vowel in TARGET_VOWELS
            if counts[vowel] == 0
        ]

        low_count_vowels = [
            vowel
            for vowel in TARGET_VOWELS
            if 0 < counts[vowel] < min_tokens_per_vowel
        ]

        count_row: dict[str, object] = {
            "speaker_id": speaker_id,
            "audio_id": speaker_df["audio_id"].iloc[0],
            "n_rows_total": len(speaker_df),
            "n_target_vowel_rows": len(vowel_df),
            "missing_vowels": "|".join(missing_vowels),
            "low_count_vowels": "|".join(low_count_vowels),
            "has_all_five_vowels": len(missing_vowels) == 0,
        }

        for vowel in TARGET_VOWELS:
            count_row[f"n_{vowel}"] = int(counts[vowel])

        count_rows.append(count_row)

        parameter_row: dict[str, object] = {
            "speaker_id": speaker_id,
            "audio_id": speaker_df["audio_id"].iloc[0],
            "n_target_vowel_rows": len(vowel_df),
            "has_all_five_vowels": len(missing_vowels) == 0,
            "missing_vowels": "|".join(missing_vowels),
        }

        for formant in FORMANTS:
            values = pd.to_numeric(
                vowel_df[formant],
                errors="coerce",
            )

            usable = values[
                values.notna()
                & np.isfinite(values)
                & values.gt(0)
            ]

            if len(usable):
                mean_value = float(usable.mean())

                # ddof=0: desvio RMS em torno da média.
                sd_value = float(
                    usable.std(ddof=0)
                )
            else:
                mean_value = np.nan
                sd_value = np.nan

            parameter_row[f"n_{formant}"] = int(
                len(usable)
            )

            parameter_row[f"mean_{formant}_Hz"] = (
                mean_value
            )

            parameter_row[f"sd_{formant}_Hz"] = (
                sd_value
            )

            parameter_row[
                f"valid_{formant}_parameters"
            ] = bool(
                len(usable) >= 2
                and np.isfinite(sd_value)
                and sd_value > 0
            )

        parameter_rows.append(parameter_row)

    parameters = pd.DataFrame(parameter_rows)
    counts = pd.DataFrame(count_rows)

    return parameters, counts


def apply_lobanov(
    data: pd.DataFrame,
    parameters: pd.DataFrame,
) -> pd.DataFrame:
    """Aplica (F - média do locutor) / desvio do locutor."""

    parameter_columns = ["speaker_id"]

    for formant in FORMANTS:
        parameter_columns.extend(
            [
                f"mean_{formant}_Hz",
                f"sd_{formant}_Hz",
            ]
        )

    result = data.merge(
        parameters[parameter_columns],
        on="speaker_id",
        how="left",
        validate="many_to_one",
    )

    target_mask = result["vowel"].isin(
        TARGET_VOWELS
    )

    for formant in FORMANTS:
        original = pd.to_numeric(
            result[formant],
            errors="coerce",
        )

        mean_column = result[
            f"mean_{formant}_Hz"
        ]

        sd_column = result[
            f"sd_{formant}_Hz"
        ]

        valid = (
            target_mask
            & original.notna()
            & np.isfinite(original)
            & original.gt(0)
            & sd_column.notna()
            & np.isfinite(sd_column)
            & sd_column.gt(0)
        )

        normalized_column = (
            f"{formant}_lobanov"
        )

        result[normalized_column] = np.nan

        result.loc[
            valid,
            normalized_column,
        ] = (
            original.loc[valid]
            - mean_column.loc[valid]
        ) / sd_column.loc[valid]

    drop_columns: list[str] = []

    for formant in FORMANTS:
        drop_columns.extend(
            [
                f"mean_{formant}_Hz",
                f"sd_{formant}_Hz",
            ]
        )

    return result.drop(columns=drop_columns)


def build_qc_table(
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    """Verifica se cada locutor ficou com média 0 e DP 1."""

    rows: list[dict[str, object]] = []

    target = normalized[
        normalized["vowel"].isin(TARGET_VOWELS)
    ]

    for speaker_id, speaker_df in target.groupby(
        "speaker_id",
        sort=True,
    ):
        row: dict[str, object] = {
            "speaker_id": speaker_id,
            "audio_id": speaker_df["audio_id"].iloc[0],
            "n_target_vowel_rows": len(speaker_df),
        }

        for formant in FORMANTS:
            column = f"{formant}_lobanov"

            values = pd.to_numeric(
                speaker_df[column],
                errors="coerce",
            )

            values = values[
                values.notna()
                & np.isfinite(values)
            ]

            row[f"mean_{column}"] = (
                float(values.mean())
                if len(values)
                else np.nan
            )

            row[f"sd_{column}"] = (
                float(values.std(ddof=0))
                if len(values)
                else np.nan
            )

            row[f"n_{column}"] = int(
                len(values)
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_speaker_vowel_centers(
    normalized: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula os centros das vogais antes e depois."""

    target = normalized[
        normalized["vowel"].isin(TARGET_VOWELS)
    ].copy()

    aggregation: dict[
        str,
        tuple[str, str],
    ] = {
        "n_tokens": ("vowel", "size"),
    }

    for formant in FORMANTS:
        aggregation[
            f"mean_{formant}_Hz"
        ] = (
            formant,
            "mean",
        )

        aggregation[
            f"median_{formant}_Hz"
        ] = (
            formant,
            "median",
        )

        aggregation[
            f"mean_{formant}_lobanov"
        ] = (
            f"{formant}_lobanov",
            "mean",
        )

        aggregation[
            f"median_{formant}_lobanov"
        ] = (
            f"{formant}_lobanov",
            "median",
        )

    centers = (
        target.groupby(
            [
                "speaker_id",
                "audio_id",
                "vowel",
            ],
            observed=True,
        )
        .agg(**aggregation)
        .reset_index()
    )

    return centers


def mean_pairwise_distance(
    points: np.ndarray,
) -> float:
    """Distância média entre todos os pares de pontos."""

    if len(points) < 2:
        return np.nan

    differences = (
        points[:, None, :]
        - points[None, :, :]
    )

    distances = np.sqrt(
        np.sum(
            differences ** 2,
            axis=2,
        )
    )

    upper_triangle = distances[
        np.triu_indices(
            len(points),
            k=1,
        )
    ]

    if not len(upper_triangle):
        return np.nan

    return float(
        np.mean(upper_triangle)
    )


def build_normalization_effect_table(
    normalized: pd.DataFrame,
    centers: pd.DataFrame,
) -> pd.DataFrame:
    """Compara a dispersão dos centros antes e depois."""

    target = normalized[
        normalized["vowel"].isin(TARGET_VOWELS)
    ]

    global_means = {
        formant: float(
            target[formant].mean()
        )
        for formant in ("F1", "F2")
    }

    global_sds = {
        formant: float(
            target[formant].std(ddof=0)
        )
        for formant in ("F1", "F2")
    }

    effect_rows: list[dict[str, object]] = []

    for vowel in TARGET_VOWELS:
        vowel_centers = centers[
            centers["vowel"] == vowel
        ].copy()

        raw_points = np.column_stack(
            [
                (
                    vowel_centers[
                        "mean_F1_Hz"
                    ].to_numpy()
                    - global_means["F1"]
                )
                / global_sds["F1"],

                (
                    vowel_centers[
                        "mean_F2_Hz"
                    ].to_numpy()
                    - global_means["F2"]
                )
                / global_sds["F2"],
            ]
        )

        normalized_points = vowel_centers[
            [
                "mean_F1_lobanov",
                "mean_F2_lobanov",
            ]
        ].to_numpy(dtype=float)

        raw_points = raw_points[
            np.all(
                np.isfinite(raw_points),
                axis=1,
            )
        ]

        normalized_points = normalized_points[
            np.all(
                np.isfinite(normalized_points),
                axis=1,
            )
        ]

        raw_distance = mean_pairwise_distance(
            raw_points
        )

        normalized_distance = (
            mean_pairwise_distance(
                normalized_points
            )
        )

        if (
            np.isfinite(raw_distance)
            and raw_distance > 0
        ):
            distance_ratio = (
                normalized_distance
                / raw_distance
            )

            compactness_change = (
                100.0
                * (
                    1.0
                    - normalized_distance
                    / raw_distance
                )
            )
        else:
            distance_ratio = np.nan
            compactness_change = np.nan

        effect_rows.append(
            {
                "vowel": vowel,
                "n_speakers_raw": len(raw_points),
                "n_speakers_lobanov": len(
                    normalized_points
                ),
                "mean_pairwise_distance_raw_global_z": (
                    raw_distance
                ),
                "mean_pairwise_distance_lobanov": (
                    normalized_distance
                ),
                "lobanov_to_raw_distance_ratio": (
                    distance_ratio
                ),
                "compactness_change_percent": (
                    compactness_change
                ),
            }
        )

    return pd.DataFrame(effect_rows)


def vowel_style_map() -> dict[
    str,
    tuple[str, str],
]:
    default_colors = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    return {
        vowel: (
            default_colors[
                index % len(default_colors)
            ],
            VOWEL_MARKERS[vowel],
        )
        for index, vowel in enumerate(
            TARGET_VOWELS
        )
    }


def plot_all_centers_before_after(
    centers: pd.DataFrame,
    output_path: Path,
) -> None:
    styles = vowel_style_map()

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        constrained_layout=True,
    )

    panels = (
        (
            axes[0],
            "mean_F2_Hz",
            "mean_F1_Hz",
            "Antes da normalização",
            "F2 (Hz)",
            "F1 (Hz)",
        ),
        (
            axes[1],
            "mean_F2_lobanov",
            "mean_F1_lobanov",
            "Depois da normalização de Lobanov",
            "F2 Lobanov",
            "F1 Lobanov",
        ),
    )

    for (
        axis,
        x_column,
        y_column,
        title,
        x_label,
        y_label,
    ) in panels:
        for vowel in TARGET_VOWELS:
            subset = centers[
                centers["vowel"] == vowel
            ]

            color, marker = styles[vowel]

            axis.scatter(
                subset[x_column],
                subset[y_column],
                label=f"/{vowel}/",
                marker=marker,
                alpha=0.65,
                s=38,
                color=color,
            )

        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

        axis.invert_xaxis()
        axis.invert_yaxis()

        axis.grid(alpha=0.25)
        axis.legend()

    figure.suptitle(
        "Centros vocálicos dos locutores: antes e depois"
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_compactness(
    effect: pd.DataFrame,
    output_path: Path,
) -> None:
    ordered = (
        effect.set_index("vowel")
        .reindex(TARGET_VOWELS)
        .reset_index()
    )

    x = np.arange(len(ordered))
    width = 0.36

    figure, axis = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    axis.bar(
        x - width / 2,
        ordered[
            "mean_pairwise_distance_raw_global_z"
        ],
        width,
        label="Antes: Hz com padronização global",
    )

    axis.bar(
        x + width / 2,
        ordered[
            "mean_pairwise_distance_lobanov"
        ],
        width,
        label="Depois: Lobanov",
    )

    axis.set_xticks(
        x,
        [
            f"/{vowel}/"
            for vowel in ordered["vowel"]
        ],
    )

    axis.set_ylabel(
        "Distância média entre centros dos locutores"
    )

    axis.set_title(
        "Compactação dos centros da mesma vogal"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_single_speaker_before_after(
    speaker_centers: pd.DataFrame,
    output_path: Path,
) -> None:
    styles = vowel_style_map()

    available = (
        speaker_centers
        .set_index("vowel")
        .reindex(VOWEL_ORDER)
    )

    available = available.dropna(
        subset=[
            "mean_F1_Hz",
            "mean_F2_Hz",
        ],
        how="any",
    )

    if available.empty:
        return

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5.5),
        constrained_layout=True,
    )

    speaker_id = str(
        speaker_centers[
            "speaker_id"
        ].iloc[0]
    )

    panels = (
        (
            axes[0],
            "mean_F2_Hz",
            "mean_F1_Hz",
            "Antes",
            "F2 (Hz)",
            "F1 (Hz)",
        ),
        (
            axes[1],
            "mean_F2_lobanov",
            "mean_F1_lobanov",
            "Depois de Lobanov",
            "F2 Lobanov",
            "F1 Lobanov",
        ),
    )

    for (
        axis,
        x_column,
        y_column,
        title,
        x_label,
        y_label,
    ) in panels:
        line_data = available.dropna(
            subset=[
                x_column,
                y_column,
            ],
            how="any",
        )

        if not line_data.empty:
            axis.plot(
                line_data[x_column],
                line_data[y_column],
                alpha=0.55,
            )

        for vowel, row in available.iterrows():
            x_value = row.get(x_column)
            y_value = row.get(y_column)

            if not (
                np.isfinite(x_value)
                and np.isfinite(y_value)
            ):
                continue

            color, marker = styles[vowel]

            axis.scatter(
                [x_value],
                [y_value],
                marker=marker,
                s=70,
                color=color,
            )

            axis.annotate(
                f"/{vowel}/",
                (x_value, y_value),
                xytext=(5, 5),
                textcoords="offset points",
            )

        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)

        axis.invert_xaxis()
        axis.invert_yaxis()

        axis.grid(alpha=0.25)

    figure.suptitle(
        f"Sistema vocálico — {speaker_id}"
    )

    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(figure)


def write_per_source_csvs(
    normalized: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Grava uma cópia normalizada de cada CSV original."""

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    internal_columns = {
        "source_csv",
        "audio_id",
        "source_row",
        "speaker_id",
        "vowel",
    }

    for source_csv, source_df in normalized.groupby(
        "source_csv",
        sort=True,
    ):
        source_df = (
            source_df
            .sort_values("source_row")
            .copy()
        )

        original_columns = [
            column
            for column in source_df.columns
            if column not in internal_columns
            and not column.endswith("_lobanov")
        ]

        output_columns = (
            original_columns
            + [
                "vowel",
                "speaker_id",
                "F1_lobanov",
                "F2_lobanov",
                "F3_lobanov",
            ]
        )

        source_df[
            output_columns
        ].to_csv(
            output_dir / source_csv,
            index=False,
        )


def print_summary(
    normalized: pd.DataFrame,
    parameters: pd.DataFrame,
    counts: pd.DataFrame,
    output_dir: Path,
    results_dir: Path,
) -> None:
    n_files = int(
        normalized["source_csv"].nunique()
    )

    n_speakers = int(
        normalized["speaker_id"].nunique()
    )

    n_rows = len(normalized)

    n_target = int(
        normalized[
            "vowel"
        ].isin(TARGET_VOWELS).sum()
    )

    incomplete = int(
        (
            ~counts["has_all_five_vowels"]
        ).sum()
    )

    print()
    print("Normalização de Lobanov concluída.")
    print(f"Arquivos CSV lidos: {n_files}")
    print(f"Chaves de locutor: {n_speakers}")
    print(f"Total de linhas: {n_rows}")
    print(
        "Linhas identificadas como /a e i o u/: "
        f"{n_target}"
    )
    print(
        "Locutores sem pelo menos uma vogal: "
        f"{incomplete}"
    )

    for formant in FORMANTS:
        invalid_count = int(
            (
                ~parameters[
                    f"valid_{formant}_parameters"
                ]
            ).sum()
        )

        print(
            f"Locutores com parâmetros inválidos "
            f"para {formant}: {invalid_count}"
        )

    print()
    print(
        f"CSVs normalizados: {output_dir}"
    )

    print(
        f"Resultados de diagnóstico: {results_dir}"
    )


def main() -> int:
    args = parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    results_dir = args.results_dir.resolve()

    plots_dir = results_dir / "plots"
    speaker_plots_dir = (
        plots_dir / "speakers"
    )

    if args.min_tokens_per_vowel < 1:
        raise ValueError(
            "--min-tokens-per-vowel deve ser pelo menos 1."
        )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not args.no_speaker_plots:
        speaker_plots_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    combined, inventory = load_csv_files(
        input_dir
    )

    parameters, counts = (
        calculate_lobanov_parameters(
            combined,
            min_tokens_per_vowel=(
                args.min_tokens_per_vowel
            ),
        )
    )

    normalized = apply_lobanov(
        combined,
        parameters,
    )

    qc = build_qc_table(normalized)

    centers = build_speaker_vowel_centers(
        normalized
    )

    effect = build_normalization_effect_table(
        normalized,
        centers,
    )

    write_per_source_csvs(
        normalized,
        output_dir,
    )

    combined_output_columns = [
        column
        for column in normalized.columns
        if column != "source_row"
    ]

    normalized[
        combined_output_columns
    ].to_csv(
        output_dir
        / "all_points_lobanov.csv",
        index=False,
    )

    inventory.to_csv(
        results_dir
        / "input_file_inventory.csv",
        index=False,
    )

    parameters.to_csv(
        results_dir
        / "lobanov_parameters_by_speaker.csv",
        index=False,
    )

    counts.to_csv(
        results_dir
        / "token_counts_by_speaker_vowel.csv",
        index=False,
    )

    qc.to_csv(
        results_dir
        / "lobanov_qc_by_speaker.csv",
        index=False,
    )

    centers.to_csv(
        results_dir
        / "speaker_vowel_centers_before_after.csv",
        index=False,
    )

    effect.to_csv(
        results_dir
        / "normalization_effect_by_vowel.csv",
        index=False,
    )

    plot_all_centers_before_after(
        centers,
        plots_dir
        / "speaker_vowel_centers_before_after.png",
    )

    plot_compactness(
        effect,
        plots_dir
        / "same_vowel_compactness_before_after.png",
    )

    if not args.no_speaker_plots:
        for (
            speaker_id,
            speaker_centers,
        ) in centers.groupby(
            "speaker_id",
            sort=True,
        ):
            safe_name = (
                str(speaker_id)
                .replace("/", "_")
            )

            plot_single_speaker_before_after(
                speaker_centers,
                speaker_plots_dir
                / f"{safe_name}_before_after.png",
            )

    print_summary(
        normalized,
        parameters,
        counts,
        output_dir,
        results_dir,
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(
            f"ERRO: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(1) from exc
