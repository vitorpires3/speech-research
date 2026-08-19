#!/usr/bin/env python3
"""Gera mapas vocálicos Lobanov com escala robusta por locutor.

Os limites dos eixos são determinados por percentis, evitando que
poucos valores extremos destruam a visualização da distribuição central.

Os dados originais e normalizados não são modificados.
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

VOWEL_MARKERS = {
    "a": "o",
    "e": "s",
    "i": "^",
    "o": "D",
    "u": "v",
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=(
            "Gera gráficos F1 × F2 Lobanov por locutor, "
            "com limites robustos baseados em percentis."
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
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            project_root
            / "results/lobanov_points/plots"
            / "all_points_by_speaker_robust_zoom"
        ),
    )

    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=0.5,
        help="Percentil inferior usado para definir os eixos.",
    )

    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.5,
        help="Percentil superior usado para definir os eixos.",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.06,
        help="Margem adicional proporcional à amplitude dos eixos.",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=13.0,
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.38,
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
    )

    parser.add_argument(
        "--format",
        choices=("png", "pdf", "both"),
        default="png",
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


def finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return numeric.where(
        np.isfinite(numeric),
        np.nan,
    )


def robust_limits(
    series: pd.Series,
    lower_percentile: float,
    upper_percentile: float,
    margin_ratio: float,
) -> tuple[float, float]:
    values = finite_numeric(series).dropna()

    if values.empty:
        raise ValueError(
            "Não foi possível calcular os limites do gráfico."
        )

    lower = float(
        np.percentile(
            values,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            values,
            upper_percentile,
        )
    )

    amplitude = upper - lower

    if not np.isfinite(amplitude) or amplitude <= 0:
        amplitude = 1.0

    margin = amplitude * margin_ratio

    return lower - margin, upper + margin


def get_display_name(
    speaker_df: pd.DataFrame,
    speaker_id: str,
) -> str:
    if "audio_id" in speaker_df.columns:
        values = (
            speaker_df["audio_id"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(values) == 1:
            return values[0]

    if "file_name" in speaker_df.columns:
        values = (
            speaker_df["file_name"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(values) == 1:
            return values[0]

    return speaker_id


def save_figure(
    figure: plt.Figure,
    output_base: Path,
    output_format: str,
    dpi: int,
) -> list[str]:
    generated: list[str] = []

    if output_format in {"png", "both"}:
        path = output_base.with_suffix(".png")

        figure.savefig(
            path,
            dpi=dpi,
            bbox_inches="tight",
        )

        generated.append(path.name)

    if output_format in {"pdf", "both"}:
        path = output_base.with_suffix(".pdf")

        figure.savefig(
            path,
            bbox_inches="tight",
        )

        generated.append(path.name)

    return generated


def plot_speaker(
    speaker_id: str,
    speaker_df: pd.DataFrame,
    output_dir: Path,
    lower_percentile: float,
    upper_percentile: float,
    margin_ratio: float,
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

    x_min, x_max = robust_limits(
        speaker_df["F2_lobanov"],
        lower_percentile,
        upper_percentile,
        margin_ratio,
    )

    y_min, y_max = robust_limits(
        speaker_df["F1_lobanov"],
        lower_percentile,
        upper_percentile,
        margin_ratio,
    )

    inside_x = speaker_df["F2_lobanov"].between(
        x_min,
        x_max,
        inclusive="both",
    )

    inside_y = speaker_df["F1_lobanov"].between(
        y_min,
        y_max,
        inclusive="both",
    )

    inside_plot = inside_x & inside_y

    n_total = len(speaker_df)
    n_visible = int(inside_plot.sum())
    n_outside = n_total - n_visible

    figure, axis = plt.subplots(
        figsize=(10, 8),
        constrained_layout=True,
    )

    default_colors = (
        plt.rcParams["axes.prop_cycle"]
        .by_key()["color"]
    )

    inventory: dict[str, object] = {
        "speaker_id": speaker_id,
        "display_name": display_name,
        "n_tokens_total": n_total,
        "n_tokens_visible": n_visible,
        "n_tokens_outside_zoom": n_outside,
        "visible_percent": (
            100.0 * n_visible / n_total
            if n_total
            else np.nan
        ),
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }

    for index, vowel in enumerate(VOWELS):
        vowel_mask = (
            speaker_df["vowel"] == vowel
        )

        subset_all = speaker_df[
            vowel_mask
        ]

        subset_visible = speaker_df[
            vowel_mask & inside_plot
        ]

        n_vowel_total = len(subset_all)
        n_vowel_visible = len(subset_visible)

        inventory[f"n_{vowel}_total"] = (
            n_vowel_total
        )

        inventory[f"n_{vowel}_visible"] = (
            n_vowel_visible
        )

        if n_vowel_visible == 0:
            continue

        axis.scatter(
            subset_visible["F2_lobanov"],
            subset_visible["F1_lobanov"],
            marker=VOWEL_MARKERS[vowel],
            s=point_size,
            alpha=alpha,
            color=default_colors[
                index % len(default_colors)
            ],
            edgecolors="none",
            label=(
                f"/{vowel}/ "
                f"(n={n_vowel_visible})"
            ),
        )

    axis.axvline(
        0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.30,
    )

    axis.axhline(
        0,
        linewidth=0.8,
        linestyle="--",
        alpha=0.30,
    )

    # Convenção tradicional dos mapas vocálicos:
    # F2 e F1 crescem na direção inversa.
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

    axis.grid(alpha=0.22)

    axis.legend(
        title="Vogal",
        loc="best",
        frameon=True,
    )

    visible_percent = (
        100.0 * n_visible / n_total
        if n_total
        else 0.0
    )

    subtitle = (
        f"Zoom robusto: percentis "
        f"{lower_percentile:g}–{upper_percentile:g} | "
        f"visíveis: {n_visible}/{n_total} "
        f"({visible_percent:.1f}%) | "
        f"fora do enquadramento: {n_outside}"
    )

    axis.text(
        0.5,
        -0.105,
        subtitle,
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=9,
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
        generated
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

    if not (
        0 <= args.lower_percentile
        < args.upper_percentile
        <= 100
    ):
        raise ValueError(
            "Os percentis devem obedecer a "
            "0 <= inferior < superior <= 100."
        )

    if args.margin < 0:
        raise ValueError(
            "--margin não pode ser negativo."
        )

    if not 0 < args.alpha <= 1:
        raise ValueError(
            "--alpha deve estar entre 0 e 1."
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
            "Colunas necessárias ausentes: "
            f"{missing}"
        )

    df = df.copy()

    df["speaker_id"] = (
        df["speaker_id"]
        .astype("string")
    )

    df["vowel"] = (
        df["vowel"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    df["F1_lobanov"] = finite_numeric(
        df["F1_lobanov"]
    )

    df["F2_lobanov"] = finite_numeric(
        df["F2_lobanov"]
    )

    df = df[
        df["speaker_id"].notna()
        & df["vowel"].isin(VOWELS)
        & df["F1_lobanov"].notna()
        & df["F2_lobanov"].notna()
    ].copy()

    if df.empty:
        raise ValueError(
            "Nenhum token normalizado utilizável."
        )

    speaker_ids = sorted(
        df["speaker_id"]
        .astype(str)
        .unique()
    )

    print(
        f"Locutores encontrados: {len(speaker_ids)}"
    )

    inventory_rows: list[
        dict[str, object]
    ] = []

    for position, speaker_id in enumerate(
        speaker_ids,
        start=1,
    ):
        print(
            f"[{position:02d}/{len(speaker_ids):02d}] "
            f"{speaker_id}"
        )

        speaker_df = df[
            df["speaker_id"].astype(str)
            == speaker_id
        ].copy()

        inventory_rows.append(
            plot_speaker(
                speaker_id=speaker_id,
                speaker_df=speaker_df,
                output_dir=output_dir,
                lower_percentile=(
                    args.lower_percentile
                ),
                upper_percentile=(
                    args.upper_percentile
                ),
                margin_ratio=args.margin,
                point_size=args.point_size,
                alpha=args.alpha,
                dpi=args.dpi,
                output_format=args.format,
            )
        )

    inventory = pd.DataFrame(
        inventory_rows
    )

    inventory.to_csv(
        output_dir
        / "robust_zoom_inventory.csv",
        index=False,
    )

    print()
    print("Gráficos concluídos.")
    print(f"Locutores: {len(speaker_ids)}")
    print(f"Saída: {output_dir}")
    print(
        "Inventário: "
        f"{output_dir / 'robust_zoom_inventory.csv'}"
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
