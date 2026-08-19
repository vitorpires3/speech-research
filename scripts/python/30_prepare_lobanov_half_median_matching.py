#!/usr/bin/env python3
"""Prepara H1/H2 normalizadas separadamente para o script 17.

Este script NÃO realiza o matching.

Etapas:
1. divide cada áudio em H1/H2;
2. calcula Lobanov independentemente em cada metade;
3. calcula medianas F1/F2 Lobanov por vogal;
4. produz um diretório de entrada compatível com
   17_cross_half_audio_matching.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


VOWELS = ("i", "e", "a", "o", "u")
FORMANTS = ("F1", "F2")

ACCENT_MAP = str.maketrans(
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
            "Divide os áudios em H1/H2, normaliza cada metade "
            "separadamente e prepara medianas para o script 17."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=project_root / "data/processed/new_fave_points",
    )

    parser.add_argument(
        "--base-profile-root",
        type=Path,
        default=(
            project_root
            / "results/all_audio_half_region_profiles_level80"
        ),
        help=(
            "Diretório antigo contendo as três tabelas gerais "
            "lidas pelo script 17."
        ),
    )

    parser.add_argument(
        "--output-data-dir",
        type=Path,
        default=(
            project_root
            / "data/processed"
            / "new_fave_points_halves_lobanov_independent"
        ),
    )

    parser.add_argument(
        "--output-profile-root",
        type=Path,
        default=(
            project_root
            / "results"
            / "all_audio_half_region_profiles_lobanov_median_centroids"
        ),
    )

    parser.add_argument(
        "--min-tokens-per-vowel",
        type=int,
        default=5,
        help=(
            "Mínimo de tokens em cada combinação "
            "áudio × metade × vogal."
        ),
    )

    return parser.parse_args()


def normalize_vowel(value: object) -> str | None:
    if pd.isna(value):
        return None

    value = (
        str(value)
        .strip()
        .translate(ACCENT_MAP)
        .lower()
    )

    return value if value in VOWELS else None


def finite_numeric(series: pd.Series) -> pd.Series:
    result = pd.to_numeric(
        series,
        errors="coerce",
    )

    return result.where(
        np.isfinite(result),
        np.nan,
    )


def assign_halves(
    df: pd.DataFrame,
) -> tuple[pd.Series, str, float]:
    """Divide a gravação em duas metades temporais.

    A coluna time contém a posição absoluta do token na gravação.

    A coluna prop_time NÃO deve ser usada aqui, porque representa
    a posição proporcional da medição dentro do próprio token.
    """

    if "time" not in df.columns:
        raise ValueError(
            "A coluna time não existe no arquivo."
        )

    time = finite_numeric(df["time"])
    valid_time = time.dropna()

    if valid_time.empty:
        raise ValueError(
            "A coluna time não contém valores válidos."
        )

    start_time = float(valid_time.min())
    end_time = float(valid_time.max())

    if end_time <= start_time:
        raise ValueError(
            "Intervalo temporal inválido: "
            f"início={start_time}, fim={end_time}."
        )

    midpoint = (
        start_time
        + end_time
    ) / 2.0

    half = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    half.loc[
        time.notna()
        & time.lt(midpoint)
    ] = "H1"

    half.loc[
        time.notna()
        & time.ge(midpoint)
    ] = "H2"

    return half, "absolute_time_midpoint", midpoint


def calculate_parameters(
    df: pd.DataFrame,
    audio: str,
    half: str,
) -> dict[str, object]:
    """Calcula os parâmetros Lobanov usando as cinco vogais juntas."""

    target = df[
        df["vowel"].isin(VOWELS)
    ]

    row: dict[str, object] = {
        "audio": audio,
        "half": half,
        "n_tokens": len(target),
    }

    for formant in FORMANTS:
        values = finite_numeric(
            target[formant]
        )

        values = values[
            values.notna()
            & values.gt(0)
        ]

        mean_value = (
            float(values.mean())
            if len(values)
            else np.nan
        )

        # ddof=0: desvio RMS em torno da média.
        sd_value = (
            float(values.std(ddof=0))
            if len(values)
            else np.nan
        )

        row[f"n_{formant}"] = len(values)
        row[f"mean_{formant}"] = mean_value
        row[f"sd_{formant}"] = sd_value

        row[f"valid_{formant}"] = bool(
            len(values) >= 2
            and np.isfinite(sd_value)
            and sd_value > 0
        )

    return row


def apply_lobanov(
    df: pd.DataFrame,
    parameters: dict[str, object],
) -> pd.DataFrame:
    result = df.copy()

    target_mask = result[
        "vowel"
    ].isin(VOWELS)

    for formant in FORMANTS:
        output_column = (
            f"{formant}_lobanov_half"
        )

        result[output_column] = np.nan

        mean_value = parameters[
            f"mean_{formant}"
        ]

        sd_value = parameters[
            f"sd_{formant}"
        ]

        if not (
            np.isfinite(mean_value)
            and np.isfinite(sd_value)
            and sd_value > 0
        ):
            continue

        values = finite_numeric(
            result[formant]
        )

        valid = (
            target_mask
            & values.notna()
            & values.gt(0)
        )

        result.loc[
            valid,
            output_column,
        ] = (
            values.loc[valid]
            - mean_value
        ) / sd_value

    return result


def calculate_medians(
    df: pd.DataFrame,
    audio: str,
    half: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for vowel in VOWELS:
        subset = df[
            df["vowel"] == vowel
        ]

        row: dict[str, object] = {
            "audio": audio,
            "half": half,
            "vowel": vowel,
            "n_tokens": len(subset),
        }

        for formant in FORMANTS:
            raw = finite_numeric(
                subset[formant]
            ).dropna()

            normalized = finite_numeric(
                subset[
                    f"{formant}_lobanov_half"
                ]
            ).dropna()

            row[
                f"median_{formant}_Hz"
            ] = (
                float(raw.median())
                if len(raw)
                else np.nan
            )

            row[
                f"median_{formant}_lobanov"
            ] = (
                float(normalized.median())
                if len(normalized)
                else np.nan
            )

        rows.append(row)

    return rows


def prepare_script17_input(
    medians: pd.DataFrame,
    valid_audios: set[str],
    base_root: Path,
    output_root: Path,
) -> None:
    """Substitui os centroides antigos pelas medianas Lobanov.

    As demais colunas são mantidas apenas porque o script 17
    lê e monta todos os blocos antes de aplicar os pesos.
    """

    ellipse_path = (
        base_root
        / "general_ellipse_parameters_all.csv"
    )

    total_overlap_path = (
        base_root
        / "general_total_region_overlap_by_vowel_all.csv"
    )

    pairwise_overlap_path = (
        base_root
        / "general_pairwise_region_overlap_all.csv"
    )

    for path in (
        ellipse_path,
        total_overlap_path,
        pairwise_overlap_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo necessário ausente: {path}"
            )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    ellipse = pd.read_csv(
        ellipse_path
    )

    total_overlap = pd.read_csv(
        total_overlap_path
    )

    pairwise_overlap = pd.read_csv(
        pairwise_overlap_path
    )

    required_keys = {
        "audio",
        "half",
        "vowel",
    }

    if not required_keys.issubset(
        ellipse.columns
    ):
        raise ValueError(
            "A tabela de elipses não contém "
            "audio, half e vowel."
        )

    centers = medians[
        medians["audio"].isin(
            valid_audios
        )
    ][
        [
            "audio",
            "half",
            "vowel",
            "n_tokens",
            "median_F1_lobanov",
            "median_F2_lobanov",
        ]
    ].copy()

    centers = centers.rename(
        columns={
            "median_F1_lobanov": (
                "new_center_f1"
            ),
            "median_F2_lobanov": (
                "new_center_f2"
            ),
        }
    )

    ellipse = ellipse[
        ellipse["audio"]
        .astype(str)
        .isin(valid_audios)
    ].copy()

    for column in (
        "audio",
        "half",
        "vowel",
    ):
        ellipse[column] = (
            ellipse[column]
            .astype(str)
        )

        centers[column] = (
            centers[column]
            .astype(str)
        )

    merged = ellipse.merge(
        centers,
        on=[
            "audio",
            "half",
            "vowel",
        ],
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "Nenhuma chave coincidiu entre "
            "os centroides novos e a tabela antiga."
        )

    merged[
        "center_f1_original"
    ] = merged["center_f1"]

    merged[
        "center_f2_original"
    ] = merged["center_f2"]

    # Colunas lidas pelo build_centroid_features
    # do script 17.
    merged["center_f1"] = (
        merged["new_center_f1"]
    )

    merged["center_f2"] = (
        merged["new_center_f2"]
    )

    merged["center_source"] = (
        "median_independent_half_lobanov"
    )

    merged = merged.drop(
        columns=[
            "new_center_f1",
            "new_center_f2",
        ]
    )

    merged.to_csv(
        output_root
        / "general_ellipse_parameters_all.csv",
        index=False,
    )

    for table, filename in (
        (
            total_overlap,
            "general_total_region_overlap_by_vowel_all.csv",
        ),
        (
            pairwise_overlap,
            "general_pairwise_region_overlap_all.csv",
        ),
    ):
        filtered = table[
            table["audio"]
            .astype(str)
            .isin(valid_audios)
        ]

        filtered.to_csv(
            output_root / filename,
            index=False,
        )


def write_preparation_report(
    output_root: Path,
    inventory: pd.DataFrame,
    parameters: pd.DataFrame,
    counts: pd.DataFrame,
    medians: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> None:
    valid = eligibility[
        eligibility["eligible"]
    ]

    excluded = eligibility[
        ~eligibility["eligible"]
    ]

    summary = pd.DataFrame(
        [
            {
                "input_audios": (
                    inventory[
                        "audio"
                    ].nunique()
                ),
                "valid_audios": len(valid),
                "excluded_audios": len(excluded),
                "normalization": (
                    "Lobanov independente "
                    "por áudio e metade"
                ),
                "centers": (
                    "medianas F1/F2 Lobanov"
                ),
            }
        ]
    )

    html = """
<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Preparação Lobanov H1/H2</title>
<style>
body {
    font-family: sans-serif;
    margin: 2rem;
}
table {
    border-collapse: collapse;
    margin-bottom: 2rem;
    font-size: 0.82rem;
}
th, td {
    border: 1px solid #ccc;
    padding: 0.35rem 0.55rem;
    text-align: right;
}
th {
    background: #f2f2f2;
}
td:first-child, th:first-child {
    text-align: left;
}
</style>
</head>
<body>
<h1>Preparação H1/H2 com Lobanov independente</h1>
<h2>Resumo</h2>
"""

    html += summary.to_html(
        index=False,
        border=0,
    )

    html += "<h2>Elegibilidade dos áudios</h2>"
    html += eligibility.to_html(
        index=False,
        border=0,
    )

    html += "<h2>Inventário da divisão</h2>"
    html += inventory.to_html(
        index=False,
        border=0,
    )

    html += "<h2>Contagem por metade e vogal</h2>"
    html += counts.to_html(
        index=False,
        border=0,
    )

    html += "<h2>Parâmetros Lobanov</h2>"
    html += parameters.to_html(
        index=False,
        border=0,
    )

    html += "<h2>Medianas normalizadas</h2>"
    html += medians.to_html(
        index=False,
        border=0,
    )

    html += "</body></html>"

    (
        output_root
        / "preparation_report.html"
    ).write_text(
        html,
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if args.min_tokens_per_vowel < 1:
        raise ValueError(
            "--min-tokens-per-vowel deve ser >= 1."
        )

    input_dir = args.input_dir.resolve()
    base_root = (
        args.base_profile_root.resolve()
    )
    output_data_dir = (
        args.output_data_dir.resolve()
    )
    output_root = (
        args.output_profile_root.resolve()
    )

    csv_paths = sorted(
        input_dir.glob("*.csv")
    )

    if not csv_paths:
        raise FileNotFoundError(
            f"Nenhum CSV encontrado em {input_dir}"
        )

    output_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_normalized = []
    parameter_rows = []
    median_rows = []
    inventory_rows = []
    count_rows = []

    for number, csv_path in enumerate(
        csv_paths,
        start=1,
    ):
        audio = csv_path.stem

        print(
            f"[{number:02d}/{len(csv_paths):02d}] "
            f"{audio}"
        )

        df = pd.read_csv(
            csv_path,
            low_memory=False,
        )

        required = {
            "label",
            "F1",
            "F2",
        }

        missing = sorted(
            required.difference(df.columns)
        )

        if missing:
            raise ValueError(
                f"{csv_path.name}: "
                f"colunas ausentes: {missing}"
            )

        df = df.copy()

        df["audio"] = audio

        df["vowel"] = df[
            "label"
        ].map(normalize_vowel)

        for formant in FORMANTS:
            df[formant] = finite_numeric(
                df[formant]
            )

        half, method, split_value = (
            assign_halves(df)
        )

        df["half"] = half

        inventory_rows.append(
            {
                "audio": audio,
                "source_csv": csv_path.name,
                "split_method": method,
                "split_value": split_value,
                "n_total": len(df),
                "n_H1": int(
                    (df["half"] == "H1").sum()
                ),
                "n_H2": int(
                    (df["half"] == "H2").sum()
                ),
                "n_without_half": int(
                    df["half"].isna().sum()
                ),
            }
        )

        for half_name in ("H1", "H2"):
            half_df = df[
                df["half"] == half_name
            ].copy()

            parameters = calculate_parameters(
                half_df,
                audio,
                half_name,
            )

            parameter_rows.append(
                parameters
            )

            normalized = apply_lobanov(
                half_df,
                parameters,
            )

            half_output_dir = (
                output_data_dir
                / half_name
            )

            half_output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            normalized.to_csv(
                half_output_dir
                / csv_path.name,
                index=False,
            )

            all_normalized.append(
                normalized
            )

            medians = calculate_medians(
                normalized,
                audio,
                half_name,
            )

            median_rows.extend(
                medians
            )

            count_row = {
                "audio": audio,
                "half": half_name,
            }

            for row in medians:
                count_row[
                    f"n_{row['vowel']}"
                ] = row["n_tokens"]

            count_rows.append(
                count_row
            )

    normalized_all = pd.concat(
        all_normalized,
        ignore_index=True,
        sort=False,
    )

    parameters_df = pd.DataFrame(
        parameter_rows
    )

    medians_df = pd.DataFrame(
        median_rows
    )

    inventory_df = pd.DataFrame(
        inventory_rows
    )

    counts_df = pd.DataFrame(
        count_rows
    )

    normalized_all.to_csv(
        output_data_dir
        / "all_points_halves_lobanov.csv",
        index=False,
    )

    parameters_df.to_csv(
        output_root
        / "lobanov_half_parameters.csv",
        index=False,
    )

    medians_df.to_csv(
        output_root
        / "lobanov_half_median_centroids.csv",
        index=False,
    )

    inventory_df.to_csv(
        output_root
        / "split_inventory.csv",
        index=False,
    )

    counts_df.to_csv(
        output_root
        / "token_counts_by_audio_half_vowel.csv",
        index=False,
    )

    # Perfil largo: uma linha por áudio × metade.
    wide = medians_df.pivot(
        index=[
            "audio",
            "half",
        ],
        columns="vowel",
        values=[
            "median_F1_lobanov",
            "median_F2_lobanov",
        ],
    )

    wide.columns = [
        f"{measure}_{vowel}"
        for measure, vowel
        in wide.columns
    ]

    wide = wide.reset_index()

    wide.to_csv(
        output_root
        / "lobanov_half_median_profiles_wide.csv",
        index=False,
    )

    eligibility_rows = []

    for audio in sorted(
        medians_df["audio"].unique()
    ):
        subset = medians_df[
            medians_df["audio"] == audio
        ]

        reasons = []

        for half_name in ("H1", "H2"):
            for vowel in VOWELS:
                row = subset[
                    (subset["half"] == half_name)
                    & (subset["vowel"] == vowel)
                ]

                if row.empty:
                    reasons.append(
                        f"{half_name}/{vowel}:missing"
                    )
                    continue

                n_tokens = int(
                    row["n_tokens"].iloc[0]
                )

                if (
                    n_tokens
                    < args.min_tokens_per_vowel
                ):
                    reasons.append(
                        f"{half_name}/{vowel}:"
                        f"n={n_tokens}"
                    )

                f1 = row[
                    "median_F1_lobanov"
                ].iloc[0]

                f2 = row[
                    "median_F2_lobanov"
                ].iloc[0]

                if not (
                    np.isfinite(f1)
                    and np.isfinite(f2)
                ):
                    reasons.append(
                        f"{half_name}/{vowel}:"
                        "invalid_median"
                    )

        eligibility_rows.append(
            {
                "audio": audio,
                "eligible": (
                    len(reasons) == 0
                ),
                "reasons": "|".join(
                    reasons
                ),
            }
        )

    eligibility_df = pd.DataFrame(
        eligibility_rows
    )

    eligibility_df.to_csv(
        output_root
        / "audio_eligibility.csv",
        index=False,
    )

    valid_audios = set(
        eligibility_df.loc[
            eligibility_df["eligible"],
            "audio",
        ].astype(str)
    )

    excluded = eligibility_df[
        ~eligibility_df["eligible"]
    ]

    excluded.to_csv(
        output_root
        / "excluded_audios.csv",
        index=False,
    )

    if not valid_audios:
        raise ValueError(
            "Nenhum áudio passou pelos critérios."
        )

    prepare_script17_input(
        medians=medians_df,
        valid_audios=valid_audios,
        base_root=base_root,
        output_root=output_root,
    )

    write_preparation_report(
        output_root=output_root,
        inventory=inventory_df,
        parameters=parameters_df,
        counts=counts_df,
        medians=medians_df,
        eligibility=eligibility_df,
    )

    print()
    print("Preparação concluída.")
    print(f"Áudios de entrada: {len(csv_paths)}")
    print(
        "Áudios válidos para matching: "
        f"{len(valid_audios)}"
    )
    print(
        f"Áudios excluídos: {len(excluded)}"
    )
    print(
        f"Dados normalizados: {output_data_dir}"
    )
    print(
        f"Entrada para o script 17: {output_root}"
    )
    print(
        "Relatório: "
        f"{output_root / 'preparation_report.html'}"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())

    except (
        FileNotFoundError,
        ValueError,
        RuntimeError,
        pd.errors.ParserError,
    ) as exc:
        print(
            f"ERRO: {exc}",
            file=sys.stderr,
        )

        raise SystemExit(1) from exc
