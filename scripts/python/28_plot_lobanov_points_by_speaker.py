#!/usr/bin/env python3
"""Gera um gráfico F1 × F2 normalizado para cada locutor.

Cada gráfico contém todos os tokens das vogais /a e i o u/,
utilizando F1_lobanov e F2_lobanov.

Por padrão, todos os gráficos usam os mesmos limites de eixo,
facilitando a comparação direta entre locutores.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


VOWELS = ("a", "e", "i", "o", "u")

VOWEL_STYLES = {
    "a": {
        "marker": "o",
        "label": "/a/",
    },
    "e": {
        "marker": "s",
        "label": "/e/",
    },
    "i": {
        "marker": "^",
        "label": "/i/",
    },
    "o": {
        "marker": "D",
        "label": "/o/",
    },
    "u": {
        "marker": "v",
        "label": "/u/",
    },
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Gera um gráfico F1 × F2 Lobanov com todos os "
            "tokens para cada locutor."
        )
    )

    parser.add_argument(
        "--input-file",
        type=Path,
        default=(
            project_root
            / "data/processed/new_fave_points_lobanov"
            / "all_points_lobanov.csv"
        ),
        help="Arquivo combinado produzido pelo script 27.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "results/lobanov_points/plots"
            / "all_points_by_speaker"
        ),
        help="Pasta onde os gráficos serão gravados.",
    )

    parser.add_argument(
        "--axis-mode",
        choices=("global", "speaker"),
        default="global",
        help=(
            "'global' usa os mesmos limites em todos os gráficos; "
            "'speaker' ajusta os limites individualmente."
        ),
    )

    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
        help="Formato dos gráficos.",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=12.0,
        help="Tamanho dos pontos.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.38,
        help="Transparência dos pontos, entre 0 e 1.",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Resolução das imagens PNG.",
    )

    return parser.parse_args()


def safe_filename(value: object) -> str:
    text = str(value).strip()

    text = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        text,
    )

    return text.strip("_") or "speaker"


def numeric_finite(
    series: pd.Series,
) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.where(
        np.isfinite(numeric),
        np.nan,
    )


def calculate_limits(
    values: pd.Series,
    margin_ratio: float = 0.05,
) -> tuple[float, float]:
    values = numeric_finite(values).dropna()

    if values.empty:
        raise ValueError(
            "Não foi possível calcular os limites dos eixos."
        )

    minimum = float(values.min())
    maximum = float(values.max())

    amplitude = maximum - minimum

    if amplitude == 0:
        amplitude = 1.0

    margin = amplitude * margin_ratio

    return minimum - margin, maximum + margin


def get_display_name(
    speaker_df: pd.DataFrame,
    speaker_id: str,
) -> str:
    if "audio_id" in speaker_df.columns:
        audio_values = (
            speaker_df["audio_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(audio_values) == 1:
            return audio_values[0]

    if "file_name" in speaker_df.columns:
        file_values = (
            speaker_df["file_name"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(file_values) == 1:
            return file_values[0]

    return speaker_id


def save_figure(
    figure: plt.Figure,
    output_base: Path,
    output_format: str,
    dpi: int,
) -> list[Path]:
    generated: list[Path] = []

    if output_format in {"png", "both"}:
        png_path = output_base.with_suffix(".png")

        figure.savefig(
            png_path,
            dpi=dpi,
            bbox_inches="tight",
        )

        generated.append(png_path)

    if output_format in {"pdf", "both"}:
        pdf_path = output_base.with_suffix(".pdf")

        figure.savefig(
            pdf_path,
            bbox_inches="tight",
        )

        generated.append(pdf_path)

    return generated


def plot_speaker(
    speaker_id: str,
    speaker_df: pd.DataFrame,
    output_dir: Path,
    axis_mode: str,
    global_xlim: tuple[float, float],
    global_ylim: tuple[float, float],
    point_size: float,
    alpha: float,
    dpi: int,
    output_format: str,
) -> dict[str, object]:
    speaker_df = speaker_df.copy()

    display_name = get_display_name(
        speaker_df,
        speaker_id,
    )

    if axis_mode == "global":
        x_min, x_max = global_xlim
        y_min, y_max = global_ylim
    else:
        x_min, x_max = calculate_limits(
            speaker_df["F2_lobanov"]
        )

        y_min, y_max = calculate_limits(
            speaker_df["F1_lobanov"]
        )

    figure, axis = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    inventory: dict[str, object] = {
        "speaker_id": speaker_id,
        "display_name": display_name,
        "n_tokens_total": len(speaker_df),
    }

    default_colors = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    for index, vowel in enumerate(VOWELS):
        subset = speaker_df[
            speaker_df["vowel"] == vowel
        ]

        n_tokens = len(subset)

        inventory[f"n_{vowel}"] = n_tokens

        if n_tokens == 0:
            continue

        style = VOWEL_STYLES[vowel]

        axis.scatter(
            subset["F2_lobanov"],
            subset["F1_lobanov"],
            marker=style["marker"],
            s=point_size,
            alpha=alpha,
            color=default_colors[
                index % len(default_colors)
            ],
            label=(
                f"{style['label']} tokens "
                f"(n={n_tokens})"
            ),
            edgecolors="none",
        )

    # O centro geral de cada locutor após Lobanov deve ficar
    # aproximadamente em zero.
    axis.axvline(
        0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.35,
    )

    axis.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.35,
    )

    # Eixos invertidos conforme a convenção dos mapas vocálicos.
    axis.set_xlim(
        x_max,
        x_min,
    )

    axis.set_ylim(
        y_max,
        y_min,
    )

    axis.set_xlabel(
        "F2 normalizado — Lobanov"
    )

    axis.set_ylabel(
        "F1 normalizado — Lobanov"
    )

    axis.set_title(
        f"Espaço vocálico normalizado — {display_name}"
    )

    axis.grid(
        alpha=0.22,
    )

    axis.legend(
        title="Vogal",
        loc="best",
        frameon=True,
    )

    subtitle = (
        f"Todos os tokens normalizados | "
        f"n = {len(speaker_df)}"
    )

    axis.text(
        0.5,
        -0.10,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="top",
    )

    output_base = (
        output_dir
        / safe_filename(speaker_id)
    )

    generated = save_figure(
        figure=figure,
        output_base=output_base,
        output_format=output_format,
        dpi=dpi,
    )

    plt.close(figure)

    inventory["output_files"] = "|".join(
        str(path.name)
        for path in generated
    )

    return inventory


def main() -> int:
    args = parse_args()

    input_file = args.input_file.resolve()
    output_dir = args.output_dir.resolve()

    if not input_file.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {input_file}"
        )

    if not 0 < args.alpha <= 1:
        raise ValueError(
            "--alpha deve estar entre 0 e 1."
        )

    if args.point_size <= 0:
        raise ValueError(
            "--point-size deve ser maior que zero."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Lendo: {input_file}")

    df = pd.read_csv(
        input_file,
        low_memory=False,
    )

    required_columns = {
        "speaker_id",
        "vowel",
        "F1_lobanov",
        "F2_lobanov",
    }

    missing = sorted(
        required_columns.difference(df.columns)
    )

    if missing:
        raise ValueError(
            "O arquivo não contém as colunas necessárias: "
            f"{missing}"
        )

    df = df.copy()

    df["vowel"] = (
        df["vowel"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["F1_lobanov"] = numeric_finite(
        df["F1_lobanov"]
    )

    df["F2_lobanov"] = numeric_finite(
        df["F2_lobanov"]
    )

    df = df[
        df["vowel"].isin(VOWELS)
        & df["speaker_id"].notna()
        & df["F1_lobanov"].notna()
        & df["F2_lobanov"].notna()
    ].copy()

    if df.empty:
        raise ValueError(
            "Nenhum token normalizado utilizável foi encontrado."
        )

    global_xlim = calculate_limits(
        df["F2_lobanov"]
    )

    global_ylim = calculate_limits(
        df["F1_lobanov"]
    )

    speaker_ids = sorted(
        df["speaker_id"]
        .astype(str)
        .unique()
    )

    print(
        f"Locutores encontrados: {len(speaker_ids)}"
    )

    inventory_rows: list[dict[str, object]] = []

    for position, speaker_id in enumerate(
        speaker_ids,
        start=1,
    ):
        print(
            f"[{position:02d}/{len(speaker_ids):02d}] "
            f"Gerando gráfico de {speaker_id}"
        )

        speaker_df = df[
            df["speaker_id"].astype(str)
            == speaker_id
        ].copy()

        inventory_row = plot_speaker(
            speaker_id=speaker_id,
            speaker_df=speaker_df,
            output_dir=output_dir,
            axis_mode=args.axis_mode,
            global_xlim=global_xlim,
            global_ylim=global_ylim,
            point_size=args.point_size,
            alpha=args.alpha,
            dpi=args.dpi,
            output_format=args.format,
        )

        inventory_rows.append(
            inventory_row
        )

    inventory = pd.DataFrame(
        inventory_rows
    )

    inventory.to_csv(
        output_dir / "plot_inventory.csv",
        index=False,
    )

    print()
    print("Gráficos concluídos.")
    print(
        f"Quantidade de locutores: {len(speaker_ids)}"
    )
    print(
        f"Pasta de saída: {output_dir}"
    )
    print(
        "Inventário: "
        f"{output_dir / 'plot_inventory.csv'}"
    )

    if len(speaker_ids) != 43:
        print()
        print(
            "ATENÇÃO: foram encontradas "
            f"{len(speaker_ids)} chaves de locutor, "
            "e não exatamente 43."
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
