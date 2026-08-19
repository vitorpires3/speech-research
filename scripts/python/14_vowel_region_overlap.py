#!/usr/bin/env python3

from pathlib import Path
import argparse
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import chi2


VOWEL_ORDER = ["i", "e", "a", "o", "u"]


def norm_col(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(".", "")
        .replace(" ", "")
    )


def find_col(df, candidates):
    lookup = {norm_col(c): c for c in df.columns}

    for candidate in candidates:
        key = norm_col(candidate)

        if key in lookup:
            return lookup[key]

    return None


def clean_audio_id(value):
    s = str(value).strip()
    s = Path(s).name

    for ext in [".wav", ".flac", ".mp3", ".csv", ".TextGrid", ".textgrid", ".seg"]:
        if s.endswith(ext):
            s = s[: -len(ext)]

    s = re.sub(r"\.v[0-9]+$", "", s)

    for suffix in [
        "_points",
        "-points",
        "_point",
        "-point",
        "_tracks",
        "-tracks",
        "_track",
        "-track",
    ]:
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]

    return s


def norm_audio_key(value):
    return clean_audio_id(value).lower()


def normalize_vowel(value):
    if pd.isna(value):
        return None

    s = str(value).strip().lower()

    if s == "":
        return None

    s = re.sub(r"[0-9]", "", s)

    if s in ["y", "iy", "ɪ"]:
        return "i"

    for vowel in VOWEL_ORDER:
        if vowel in s:
            return vowel

    return None


def load_tokens(points_file, audio=None, f1_min=150, f1_max=1200, f2_min=400, f2_max=4000):
    df = pd.read_csv(points_file)

    f1_col = find_col(df, ["f1", "F1"])
    f2_col = find_col(df, ["f2", "F2"])
    half_col = find_col(df, ["half"])
    vowel_col = find_col(df, ["vowel", "vowel_norm", "label", "phone", "segment"])

    if f1_col is None or f2_col is None:
        raise ValueError(f"Não encontrei F1/F2. Colunas disponíveis: {list(df.columns)}")

    if half_col is None:
        raise ValueError("Não encontrei a coluna half. Rode primeiro o script de split.")

    if vowel_col is None:
        raise ValueError(f"Não encontrei coluna de vogal. Colunas disponíveis: {list(df.columns)}")

    audio_col = find_col(df, ["_audio_id", "audio", "file", "filename", "source_file"])

    out = pd.DataFrame()

    if audio_col is not None:
        out["audio"] = df[audio_col].astype(str).apply(clean_audio_id)
    else:
        out["audio"] = clean_audio_id(Path(points_file).stem)

    out["half"] = df[half_col].astype(str)
    out["vowel_raw"] = df[vowel_col]
    out["vowel"] = df[vowel_col].apply(normalize_vowel)
    out["f1"] = pd.to_numeric(df[f1_col], errors="coerce")
    out["f2"] = pd.to_numeric(df[f2_col], errors="coerce")

    time_col = find_col(df, ["_token_time", "time", "point", "point_time", "midpoint"])
    if time_col is not None:
        out["token_time"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        out["token_time"] = np.nan

    if audio is not None:
        target = norm_audio_key(audio)
        out = out[out["audio"].apply(norm_audio_key) == target].copy()

    out = out.dropna(subset=["half", "vowel", "f1", "f2"])
    out = out[out["half"].isin(["H1", "H2"])]
    out = out[out["vowel"].isin(VOWEL_ORDER)]

    out = out[
        (out["f1"] >= f1_min)
        & (out["f1"] <= f1_max)
        & (out["f2"] >= f2_min)
        & (out["f2"] <= f2_max)
    ].copy()

    if out.empty:
        raise ValueError("Depois dos filtros, não sobrou nenhum token.")

    return out


def robust_mad_scale(x):
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))

    if not np.isfinite(mad) or mad == 0:
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            return med, 1.0
        return med, sd

    return med, 1.4826 * mad


def robust_filter_vowel(sub, z_threshold=3.5):
    """
    Remove outliers usando z-score robusto em F1 e F2.

    Não é uma remoção por linha metade/ordem.
    É uma remoção geométrica dentro de cada vogal e metade.
    """

    f1_med, f1_scale = robust_mad_scale(sub["f1"])
    f2_med, f2_scale = robust_mad_scale(sub["f2"])

    z1 = np.abs((sub["f1"] - f1_med) / f1_scale)
    z2 = np.abs((sub["f2"] - f2_med) / f2_scale)

    keep = (z1 <= z_threshold) & (z2 <= z_threshold)

    filtered = sub[keep].copy()

    return filtered


def ellipse_from_tokens(sub, region_level=0.80, z_threshold=3.5, min_tokens=10):
    """
    Estima uma elipse robusta para uma vogal.

    Centro:
      mediana de F1 e F2 após filtro robusto.

    Covariância:
      covariância dos tokens após filtro robusto.

    Tamanho:
      contorno de nível region_level usando distribuição qui-quadrado com 2 dimensões.
      Exemplo: region_level=0.80 gera uma elipse que representa aproximadamente
      a região central de 80% dos tokens, assumindo forma aproximadamente elíptica.
    """

    raw_n = len(sub)

    filtered = robust_filter_vowel(sub, z_threshold=z_threshold)
    filtered_n = len(filtered)

    if filtered_n < min_tokens:
        return None, filtered

    center_f1 = float(filtered["f1"].median())
    center_f2 = float(filtered["f2"].median())

    X = filtered[["f2", "f1"]].to_numpy(dtype=float)

    cov = np.cov(X, rowvar=False)

    if cov.shape != (2, 2):
        return None, filtered

    if not np.all(np.isfinite(cov)):
        return None, filtered

    # Pequena regularização para evitar matriz singular.
    cov = cov + np.eye(2) * 1e-6

    det = float(np.linalg.det(cov))

    if det <= 0:
        return None, filtered

    q = float(chi2.ppf(region_level, df=2))

    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    width = 2.0 * math.sqrt(q * eigvals[0])
    height = 2.0 * math.sqrt(q * eigvals[1])

    angle = math.degrees(math.atan2(eigvecs[1, 0], eigvecs[0, 0]))

    area = math.pi * q * math.sqrt(det)

    ellipse = {
        "center_f1": center_f1,
        "center_f2": center_f2,
        "cov_00_f2": cov[0, 0],
        "cov_01": cov[0, 1],
        "cov_11_f1": cov[1, 1],
        "det_cov": det,
        "region_level": region_level,
        "chi2_q": q,
        "width_f2_axis": width,
        "height_f1_axis": height,
        "angle_degrees": angle,
        "ellipse_area_analytic": area,
        "raw_n_tokens": int(raw_n),
        "filtered_n_tokens": int(filtered_n),
        "removed_outliers": int(raw_n - filtered_n),
        "removed_outliers_percent": float((raw_n - filtered_n) / raw_n * 100) if raw_n > 0 else np.nan,
    }

    return ellipse, filtered


def point_inside_ellipse(x_f2, y_f1, ellipse):
    center = np.array([ellipse["center_f2"], ellipse["center_f1"]], dtype=float)

    cov = np.array(
        [
            [ellipse["cov_00_f2"], ellipse["cov_01"]],
            [ellipse["cov_01"], ellipse["cov_11_f1"]],
        ],
        dtype=float,
    )

    inv_cov = np.linalg.inv(cov)

    X = np.column_stack([x_f2.ravel(), y_f1.ravel()])
    D = X - center

    md2 = np.einsum("ij,jk,ik->i", D, inv_cov, D)

    inside = md2 <= ellipse["chi2_q"]

    return inside.reshape(x_f2.shape)


def build_grid(tokens, ellipses, grid_size=700, margin_ratio=0.08):
    f2_values = list(tokens["f2"].to_numpy(dtype=float))
    f1_values = list(tokens["f1"].to_numpy(dtype=float))

    for ellipse in ellipses.values():
        f2_values.extend(
            [
                ellipse["center_f2"] - ellipse["width_f2_axis"],
                ellipse["center_f2"] + ellipse["width_f2_axis"],
            ]
        )
        f1_values.extend(
            [
                ellipse["center_f1"] - ellipse["height_f1_axis"],
                ellipse["center_f1"] + ellipse["height_f1_axis"],
            ]
        )

    x_min = float(np.nanmin(f2_values))
    x_max = float(np.nanmax(f2_values))
    y_min = float(np.nanmin(f1_values))
    y_max = float(np.nanmax(f1_values))

    x_margin = (x_max - x_min) * margin_ratio
    y_margin = (y_max - y_min) * margin_ratio

    x_min -= x_margin
    x_max += x_margin
    y_min -= y_margin
    y_max += y_margin

    xs = np.linspace(x_min, x_max, grid_size)
    ys = np.linspace(y_min, y_max, grid_size)

    xx, yy = np.meshgrid(xs, ys)

    dx = (x_max - x_min) / (grid_size - 1)
    dy = (y_max - y_min) / (grid_size - 1)
    cell_area = dx * dy

    return xx, yy, cell_area


def compute_overlap_for_half(tokens_half, ellipse_rows, grid_size=700):
    ellipses = {
        row["vowel"]: row
        for row in ellipse_rows
    }

    if len(ellipses) < 2:
        return pd.DataFrame(), pd.DataFrame(), {}

    xx, yy, cell_area = build_grid(tokens_half, ellipses, grid_size=grid_size)

    masks = {}

    for vowel, ellipse in ellipses.items():
        masks[vowel] = point_inside_ellipse(xx, yy, ellipse)

    grid_areas = {
        vowel: float(mask.sum() * cell_area)
        for vowel, mask in masks.items()
    }

    pair_rows = []

    vowels = [v for v in VOWEL_ORDER if v in masks]

    for i, v1 in enumerate(vowels):
        for v2 in vowels[i + 1:]:
            m1 = masks[v1]
            m2 = masks[v2]

            intersection = m1 & m2
            union = m1 | m2

            intersection_area = float(intersection.sum() * cell_area)
            union_area = float(union.sum() * cell_area)

            area_1 = grid_areas[v1]
            area_2 = grid_areas[v2]

            pair_rows.append(
                {
                    "vowel_1": v1,
                    "vowel_2": v2,
                    "area_vowel_1": area_1,
                    "area_vowel_2": area_2,
                    "intersection_area": intersection_area,
                    "union_area": union_area,
                    "overlap_percent_of_vowel_1": intersection_area / area_1 * 100 if area_1 > 0 else np.nan,
                    "overlap_percent_of_vowel_2": intersection_area / area_2 * 100 if area_2 > 0 else np.nan,
                    "jaccard_overlap_percent": intersection_area / union_area * 100 if union_area > 0 else np.nan,
                }
            )

    total_rows = []

    for vowel in vowels:
        own = masks[vowel]

        others_union = np.zeros_like(own, dtype=bool)

        for other_vowel in vowels:
            if other_vowel == vowel:
                continue

            others_union = others_union | masks[other_vowel]

        overlap_with_any = own & others_union

        own_area = grid_areas[vowel]
        overlap_area = float(overlap_with_any.sum() * cell_area)

        total_rows.append(
            {
                "vowel": vowel,
                "ellipse_area_grid": own_area,
                "overlap_area_with_any_other_vowel": overlap_area,
                "overlap_percent_with_any_other_vowel": overlap_area / own_area * 100 if own_area > 0 else np.nan,
                "unique_area_not_overlapped": own_area - overlap_area,
                "unique_percent_not_overlapped": (own_area - overlap_area) / own_area * 100 if own_area > 0 else np.nan,
            }
        )

    return pd.DataFrame(pair_rows), pd.DataFrame(total_rows), masks


def plot_regions(tokens_half, ellipse_rows, half, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    for vowel in VOWEL_ORDER:
        sub = tokens_half[tokens_half["vowel"] == vowel]

        if sub.empty:
            continue

        ax.scatter(
            sub["f2"],
            sub["f1"],
            s=18,
            alpha=0.45,
            label=f"/{vowel}/ tokens",
        )

    for ellipse in ellipse_rows:
        vowel = ellipse["vowel"]

        patch = Ellipse(
            xy=(ellipse["center_f2"], ellipse["center_f1"]),
            width=ellipse["width_f2_axis"],
            height=ellipse["height_f1_axis"],
            angle=ellipse["angle_degrees"],
            fill=False,
            linewidth=2.0,
        )

        ax.add_patch(patch)

        ax.scatter(
            ellipse["center_f2"],
            ellipse["center_f1"],
            s=120,
            marker="X",
        )

        ax.text(
            ellipse["center_f2"],
            ellipse["center_f1"],
            f" /{vowel}/",
            fontsize=12,
            weight="bold",
        )

    ax.set_title(f"Vowel regions in F1-F2 space - {half}")
    ax.set_xlabel("F2 (Hz) — inverted axis")
    ax.set_ylabel("F1 (Hz) — inverted axis")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_overlap_heatmap(total_overlap, half, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = total_overlap[total_overlap["half"] == half].copy()
    sub = sub.set_index("vowel").reindex(VOWEL_ORDER).dropna(subset=["overlap_percent_with_any_other_vowel"])

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    values = sub["overlap_percent_with_any_other_vowel"].to_numpy(dtype=float)
    labels = [f"/{v}/" for v in sub.index]

    ax.bar(labels, values)
    ax.set_ylim(0, max(100, np.nanmax(values) * 1.15))
    ax.set_ylabel("Overlap with any other vowel (%)")
    ax.set_title(f"Total region overlap by vowel - {half}")
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--points-file",
        required=True,
        help="Arquivo CSV com coluna half, normalmente all_points_with_halves.csv.",
    )

    parser.add_argument(
        "--audio",
        default=None,
        help="Opcional: filtrar um áudio específico.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pasta de saída.",
    )

    parser.add_argument(
        "--region-level",
        type=float,
        default=0.80,
        help="Nível da elipse. 0.80 = região central aproximada de 80%% dos tokens.",
    )

    parser.add_argument(
        "--outlier-z",
        type=float,
        default=3.5,
        help="Limiar do filtro robusto por MAD. Default: 3.5.",
    )

    parser.add_argument(
        "--grid-size",
        type=int,
        default=700,
        help="Resolução da grade para estimar áreas de sobreposição. Default: 700.",
    )

    parser.add_argument("--f1-min", type=float, default=150)
    parser.add_argument("--f1-max", type=float, default=1200)
    parser.add_argument("--f2-min", type=float, default=400)
    parser.add_argument("--f2-max", type=float, default=4000)

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens = load_tokens(
        points_file=args.points_file,
        audio=args.audio,
        f1_min=args.f1_min,
        f1_max=args.f1_max,
        f2_min=args.f2_min,
        f2_max=args.f2_max,
    )

    audio_name = tokens["audio"].iloc[0]

    tokens.to_csv(output_dir / "tokens_used.csv", index=False)

    ellipse_all_rows = []
    filtered_all = []
    pairwise_all = []
    total_all = []

    for half in ["H1", "H2"]:
        tokens_half = tokens[tokens["half"] == half].copy()

        ellipse_rows = []

        for vowel in VOWEL_ORDER:
            sub = tokens_half[tokens_half["vowel"] == vowel].copy()

            if sub.empty:
                continue

            ellipse, filtered = ellipse_from_tokens(
                sub,
                region_level=args.region_level,
                z_threshold=args.outlier_z,
            )

            filtered = filtered.copy()
            filtered["half"] = half
            filtered["vowel"] = vowel
            filtered_all.append(filtered)

            if ellipse is None:
                continue

            ellipse["audio"] = audio_name
            ellipse["half"] = half
            ellipse["vowel"] = vowel

            ellipse_rows.append(ellipse)
            ellipse_all_rows.append(ellipse)

        if ellipse_rows:
            pairwise, total, masks = compute_overlap_for_half(
                tokens_half=tokens_half,
                ellipse_rows=ellipse_rows,
                grid_size=args.grid_size,
            )

            if not pairwise.empty:
                pairwise.insert(0, "audio", audio_name)
                pairwise.insert(1, "half", half)
                pairwise_all.append(pairwise)

            if not total.empty:
                total.insert(0, "audio", audio_name)
                total.insert(1, "half", half)
                total_all.append(total)

            plot_regions(
                tokens_half=tokens_half,
                ellipse_rows=ellipse_rows,
                half=half,
                output_path=output_dir / f"{half}_vowel_regions.png",
            )

    ellipse_df = pd.DataFrame(ellipse_all_rows)

    if filtered_all:
        filtered_df = pd.concat(filtered_all, ignore_index=True)
    else:
        filtered_df = pd.DataFrame()

    if pairwise_all:
        pairwise_df = pd.concat(pairwise_all, ignore_index=True)
    else:
        pairwise_df = pd.DataFrame()

    if total_all:
        total_df = pd.concat(total_all, ignore_index=True)
    else:
        total_df = pd.DataFrame()

    ellipse_df.to_csv(output_dir / "ellipse_parameters.csv", index=False)
    filtered_df.to_csv(output_dir / "tokens_after_outlier_filter.csv", index=False)
    pairwise_df.to_csv(output_dir / "pairwise_region_overlap.csv", index=False)
    total_df.to_csv(output_dir / "total_region_overlap_by_vowel.csv", index=False)

    for half in ["H1", "H2"]:
        if not total_df.empty:
            plot_overlap_heatmap(
                total_overlap=total_df,
                half=half,
                output_path=output_dir / f"{half}_total_overlap_by_vowel.png",
            )

    print()
    print("=== Vowel region overlap analysis ===")
    print(f"Audio: {audio_name}")
    print(f"Region level: {args.region_level}")
    print(f"Outlier robust z threshold: {args.outlier_z}")
    print(f"Output dir: {output_dir}")
    print()
    print("Token counts used:")
    print(pd.crosstab(tokens["half"], tokens["vowel"]).reindex(columns=VOWEL_ORDER, fill_value=0).to_string())
    print()
    print("Ellipse parameters:")
    display_cols = [
        "half",
        "vowel",
        "raw_n_tokens",
        "filtered_n_tokens",
        "removed_outliers",
        "removed_outliers_percent",
        "center_f1",
        "center_f2",
        "ellipse_area_analytic",
    ]
    print(ellipse_df[display_cols].to_string(index=False))
    print()
    print("Total overlap by vowel:")
    if not total_df.empty:
        print(total_df.to_string(index=False))
    print()
    print("Pairwise overlap:")
    if not pairwise_df.empty:
        print(pairwise_df.to_string(index=False))
    print()
    print("Outputs principais:")
    print(f"- {output_dir / 'H1_vowel_regions.png'}")
    print(f"- {output_dir / 'H2_vowel_regions.png'}")
    print(f"- {output_dir / 'H1_total_overlap_by_vowel.png'}")
    print(f"- {output_dir / 'H2_total_overlap_by_vowel.png'}")
    print(f"- {output_dir / 'ellipse_parameters.csv'}")
    print(f"- {output_dir / 'pairwise_region_overlap.csv'}")
    print(f"- {output_dir / 'total_region_overlap_by_vowel.csv'}")
    print(f"- {output_dir / 'tokens_after_outlier_filter.csv'}")


if __name__ == "__main__":
    main()
