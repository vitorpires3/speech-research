#!/usr/bin/env python3
"""Class-conditional 3D KDE analysis by vowel and speaker/audio."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.offline import get_plotlyjs
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

VOWELS = ("i", "e", "a", "o", "u")
MASSES = (0.50, 0.80, 0.95)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(
        description="Fit one 3D Gaussian KDE per speaker/audio and vowel."
    )
    p.add_argument(
        "--input-dir",
        type=Path,
        default=root / "data" / "processed" / "new_fave_points",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "kde_3d_by_vowel_speaker",
    )
    p.add_argument("--axes", nargs=3, default=("F1", "F2", "F3"))
    p.add_argument("--label-column", default="label")
    p.add_argument(
        "--speaker-column",
        default="",
        help="Optional class column. Default: CSV filename stem.",
    )
    p.add_argument("--vowels", nargs="+", default=list(VOWELS))
    p.add_argument("--min-points", type=int, default=25)
    p.add_argument("--max-fit-points", type=int, default=1000)
    p.add_argument("--max-plot-points", type=int, default=800)
    p.add_argument(
        "--bandwidths",
        nargs="+",
        type=float,
        default=(0.20, 0.30, 0.45, 0.60, 0.80, 1.00),
    )
    p.add_argument("--cv-folds", type=int, default=3)
    p.add_argument("--grid-size", type=int, default=28)
    p.add_argument("--grid-padding", type=float, default=2.5)
    p.add_argument("--robust-z-cutoff", type=float, default=6.0)
    p.add_argument("--chunk-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20260721)
    args = p.parse_args()

    if args.min_points < 5:
        p.error("--min-points must be >= 5")
    if args.cv_folds < 2:
        p.error("--cv-folds must be >= 2")
    if args.grid_size < 16:
        p.error("--grid-size must be >= 16")
    if any(value <= 0 for value in args.bandwidths):
        p.error("All bandwidths must be positive")
    return args


def stable_seed(text: str, base: int) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def read_csv(path: Path) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8", "utf-8-sig", "latin1"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def resolve_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    mapping = {str(col).lower(): str(col) for col in df.columns}
    if requested.lower() in mapping:
        return mapping[requested.lower()]
    raise KeyError(f"Missing column {requested!r}")


def load_points(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = sorted(args.input_dir.resolve().glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {args.input_dir}")

    frames: list[pd.DataFrame] = []
    inventory: list[dict[str, object]] = []

    for index, path in enumerate(paths, start=1):
        print(f"Reading [{index:02d}/{len(paths):02d}] {path.name}")
        df = read_csv(path)
        try:
            x, y, z = [resolve_column(df, name) for name in args.axes]
            label = resolve_column(df, args.label_column)
            speaker_col = (
                resolve_column(df, args.speaker_column)
                if args.speaker_column.strip()
                else None
            )
        except KeyError as exc:
            inventory.append(
                {"file": path.name, "status": "skipped", "reason": str(exc)}
            )
            continue

        keep = [label, x, y, z] + ([speaker_col] if speaker_col else [])
        sub = df[keep].copy().rename(
            columns={label: "vowel", x: "x", y: "y", z: "z"}
        )
        if speaker_col:
            sub = sub.rename(columns={speaker_col: "speaker"})
            sub["speaker"] = sub["speaker"].astype(str).str.strip()
            sub.loc[sub["speaker"].eq(""), "speaker"] = path.stem
        else:
            sub["speaker"] = path.stem

        sub["source_file"] = path.name
        sub["vowel"] = sub["vowel"].astype(str).str.strip().str.lower()
        for col in ("x", "y", "z"):
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        frames.append(sub)
        inventory.append({"file": path.name, "status": "loaded", "reason": ""})

    if not frames:
        raise RuntimeError("No usable input files")

    points = pd.concat(frames, ignore_index=True)
    finite = np.isfinite(points[["x", "y", "z"]].to_numpy(float)).all(axis=1)
    points = points.loc[finite].reset_index(drop=True)
    return points, pd.DataFrame(inventory)


def robust_filter(
    points: pd.DataFrame,
    vowels: tuple[str, ...],
    cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept: list[pd.DataFrame] = []
    summary: list[dict[str, object]] = []

    for vowel in vowels:
        sub = points[points["vowel"] == vowel].copy()
        mask = np.ones(len(sub), dtype=bool)
        if cutoff > 0 and len(sub):
            for col in ("x", "y", "z"):
                values = sub[col].to_numpy(float)
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                scale = 1.4826 * mad
                if not np.isfinite(scale) or scale <= 0:
                    scale = np.std(values)
                if np.isfinite(scale) and scale > 0:
                    mask &= np.abs(values - median) / scale <= cutoff

        retained = sub.loc[mask].copy()
        if len(retained):
            kept.append(retained)
        summary.append(
            {
                "vowel": vowel,
                "raw_tokens": len(sub),
                "retained_tokens": len(retained),
                "removed_tokens": len(sub) - len(retained),
                "retained_percent": (
                    100 * len(retained) / len(sub) if len(sub) else np.nan
                ),
            }
        )

    if not kept:
        raise RuntimeError("No requested vowel points remained")
    return pd.concat(kept, ignore_index=True), pd.DataFrame(summary)


def capped_sample(
    values: np.ndarray,
    maximum: int,
    seed: int,
) -> np.ndarray:
    if maximum <= 0 or len(values) <= maximum:
        return values
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(values), size=maximum, replace=False))
    return values[indices]


def make_kde(bandwidth: float) -> KernelDensity:
    return KernelDensity(
        bandwidth=bandwidth,
        kernel="gaussian",
        algorithm="ball_tree",
        breadth_first=True,
        atol=1e-6,
        rtol=1e-6,
    )


def choose_bandwidth(
    fit_arrays: dict[str, np.ndarray],
    candidates: tuple[float, ...],
    folds: int,
    seed: int,
) -> tuple[float, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for bandwidth in candidates:
        speaker_scores: list[float] = []
        for speaker, values in fit_arrays.items():
            n_splits = min(folds, len(values))
            if n_splits < 2:
                continue
            splitter = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=stable_seed(f"{speaker}-{bandwidth}", seed),
            )
            fold_scores: list[float] = []
            for train, test in splitter.split(values):
                if len(train) < 2:
                    continue
                model = make_kde(bandwidth).fit(values[train])
                score = model.score_samples(values[test])
                score = score[np.isfinite(score)]
                if len(score):
                    fold_scores.append(float(np.mean(score)))
            if fold_scores:
                speaker_scores.append(float(np.mean(fold_scores)))

        rows.append(
            {
                "bandwidth": bandwidth,
                "mean_speaker_log_likelihood": (
                    float(np.mean(speaker_scores)) if speaker_scores else np.nan
                ),
                "median_speaker_log_likelihood": (
                    float(np.median(speaker_scores)) if speaker_scores else np.nan
                ),
                "n_speakers_scored": len(speaker_scores),
            }
        )

    table = pd.DataFrame(rows)
    valid = table[np.isfinite(table["mean_speaker_log_likelihood"])]
    if valid.empty:
        raise RuntimeError("Bandwidth CV produced no valid score")
    selected = float(
        valid.sort_values(
            ["mean_speaker_log_likelihood", "bandwidth"],
            ascending=[False, True],
        ).iloc[0]["bandwidth"]
    )
    table["selected"] = np.isclose(table["bandwidth"], selected)
    return selected, table


def make_grid(
    values: np.ndarray,
    grid_size: int,
    bandwidth: float,
    padding_factor: float,
) -> tuple[np.ndarray, float]:
    lower = np.quantile(values, 0.005, axis=0) - padding_factor * bandwidth
    upper = np.quantile(values, 0.995, axis=0) + padding_factor * bandwidth
    for axis in range(3):
        if upper[axis] - lower[axis] < bandwidth:
            middle = (upper[axis] + lower[axis]) / 2
            lower[axis] = middle - bandwidth
            upper[axis] = middle + bandwidth

    coords = [np.linspace(lower[i], upper[i], grid_size) for i in range(3)]
    mesh = np.meshgrid(*coords, indexing="ij")
    grid = np.column_stack([part.ravel() for part in mesh])
    cell_volume = float(np.prod([coord[1] - coord[0] for coord in coords]))
    return grid, cell_volume


def score_chunks(
    model: KernelDensity,
    positions: np.ndarray,
    chunk_size: int,
) -> np.ndarray:
    return np.concatenate(
        [
            model.score_samples(positions[start : start + chunk_size])
            for start in range(0, len(positions), chunk_size)
        ]
    )


def normalized_density(log_density: np.ndarray, cell_volume: float) -> np.ndarray:
    density = np.exp(log_density - np.max(log_density))
    integral = float(np.sum(density) * cell_volume)
    if not np.isfinite(integral) or integral <= 0:
        raise RuntimeError("Invalid grid density integral")
    return density / integral


def hdr_values(
    density: np.ndarray,
    cell_volume: float,
) -> dict[str, float]:
    order = np.argsort(density)[::-1]
    sorted_density = density[order]
    cumulative = np.cumsum(sorted_density) * cell_volume
    result: dict[str, float] = {}

    for mass in MASSES:
        position = min(int(np.searchsorted(cumulative, mass)), len(order) - 1)
        threshold = float(sorted_density[position])
        selected = density >= threshold
        volume = float(np.count_nonzero(selected) * cell_volume)
        actual_mass = float(np.sum(density[selected]) * cell_volume)
        suffix = int(round(100 * mass))
        result[f"hdr_{suffix}_volume"] = volume
        result[f"hdr_{suffix}_actual_mass"] = actual_mass
        result[f"hdr_{suffix}_mass_per_volume"] = (
            mass / volume if volume > 0 else np.nan
        )
    return result


def save_cv_plot(table: pd.DataFrame, selected: float, path: Path, vowel: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(
        table["bandwidth"],
        table["mean_speaker_log_likelihood"],
        marker="o",
    )
    ax.axvline(selected, linestyle="--")
    ax.set_xlabel("Bandwidth (standardized units)")
    ax.set_ylabel("Mean held-out log likelihood")
    ax.set_title(f"/{vowel}/ — speaker-balanced bandwidth selection")
    ax.grid(alpha=0.25)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_overlap_heatmap(
    speakers: list[str],
    matrix: np.ndarray,
    path: Path,
    vowel: str,
) -> None:
    size = max(10, 0.30 * len(speakers))
    fig, ax = plt.subplots(figsize=(size, size), constrained_layout=True)
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="equal", cmap="viridis")
    ax.set_xticks(np.arange(len(speakers)))
    ax.set_yticks(np.arange(len(speakers)))
    ax.set_xticklabels(speakers, rotation=90, fontsize=5)
    ax.set_yticklabels(speakers, fontsize=5)
    ax.set_title(f"/{vowel}/ — pairwise KDE overlap")
    fig.colorbar(image, ax=ax, label="Overlap coefficient")
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_volume_plot(metrics: pd.DataFrame, path: Path, vowel: str) -> None:
    table = metrics.sort_values("hdr_80_volume").reset_index(drop=True)
    positions = np.arange(len(table))
    fig, ax = plt.subplots(
        figsize=(11, max(8, 0.28 * len(table))),
        constrained_layout=True,
    )
    ax.plot(table["hdr_50_volume"], positions, marker="o", linestyle="", label="50% HDR")
    ax.plot(table["hdr_80_volume"], positions, marker="s", linestyle="", label="80% HDR")
    ax.plot(table["hdr_95_volume"], positions, marker="^", linestyle="", label="95% HDR")
    ax.set_yticks(positions)
    ax.set_yticklabels(table["speaker"], fontsize=6)
    ax.set_xlabel("Approximate volume (standardized units³)")
    ax.set_ylabel("Speaker/audio")
    ax.set_title(f"/{vowel}/ — highest-density-region volumes")
    ax.grid(axis="x", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plotly_div(
    points: pd.DataFrame,
    metrics: pd.DataFrame,
    axes: tuple[str, str, str],
    max_points: int,
    seed: int,
) -> str:
    fig = go.Figure()
    vowel = str(points["vowel"].iloc[0])

    for speaker in sorted(metrics["speaker"].astype(str)):
        sub = points[points["speaker"].astype(str) == speaker]
        if max_points > 0 and len(sub) > max_points:
            sub = sub.sample(
                n=max_points,
                random_state=stable_seed(f"plot-{vowel}-{speaker}", seed),
            )
        fig.add_trace(
            go.Scatter3d(
                x=sub["x"],
                y=sub["y"],
                z=sub["z"],
                mode="markers",
                name=speaker,
                marker={"size": 2.2, "opacity": 0.35},
                text=[speaker] * len(sub),
                hovertemplate=(
                    "Speaker/audio: %{text}<br>"
                    + f"{axes[0]}: %{{x:.3f}}<br>"
                    + f"{axes[1]}: %{{y:.3f}}<br>"
                    + f"{axes[2]}: %{{z:.3f}}<extra></extra>"
                ),
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=metrics["peak_x_original"],
            y=metrics["peak_y_original"],
            z=metrics["peak_z_original"],
            mode="markers",
            name="KDE peaks",
            marker={"size": 5, "symbol": "diamond", "opacity": 1.0},
            text=metrics["speaker"],
            hovertemplate=(
                "KDE peak — %{text}<br>"
                + f"{axes[0]}: %{{x:.3f}}<br>"
                + f"{axes[1]}: %{{y:.3f}}<br>"
                + f"{axes[2]}: %{{z:.3f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=f"/{vowel}/ — all speaker/audio classes and KDE peaks",
        scene={
            "xaxis_title": axes[0],
            "yaxis_title": axes[1],
            "zaxis_title": axes[2],
        },
        height=760,
        margin={"l": 0, "r": 0, "t": 55, "b": 0},
        legend={"font": {"size": 8}, "itemsizing": "constant"},
    )
    return fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )


def format_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    table = df.head(max_rows).copy() if max_rows else df.copy()
    for col in table.columns:
        if pd.api.types.is_float_dtype(table[col]):
            table[col] = table[col].map(
                lambda value: "NA" if not np.isfinite(value) else f"{value:.4f}"
            )
    return table.to_html(index=False, border=0, escape=True)


def write_report(
    path: Path,
    args: argparse.Namespace,
    axes: tuple[str, str, str],
    overall: pd.DataFrame,
    filtering: pd.DataFrame,
    inventory: pd.DataFrame,
    sections: list[dict[str, object]],
) -> None:
    section_html: list[str] = []

    for section in sections:
        vowel = str(section["vowel"])
        section_html.append(
            f"""
<section>
<h2>Vowel /{html.escape(vowel)}/</h2>
<p class="description">
One Gaussian KDE was fitted for each known speaker/audio class.
Diamond markers in the interactive 3D plot show estimated density peaks.
</p>
{section['plot_div']}
<div class="grid-two">
<div><h3>Bandwidth cross-validation</h3>
<img src="assets/{html.escape(vowel)}_bandwidth_cv.png"></div>
<div><h3>Highest-density-region volumes</h3>
<img src="assets/{html.escape(vowel)}_hdr_volumes.png"></div>
</div>
<h3>Pairwise overlap heatmap</h3>
<img src="assets/{html.escape(vowel)}_overlap_heatmap.png">
<h3>Speaker-level density metrics</h3>
<p class="description">
Smaller HDR volumes indicate greater concentration. Own mass in the dominant
region is the fraction of a class density falling in grid cells where that
class has the highest density under equal priors.
</p>
<div class="table-container">{format_table(section['metrics'])}</div>
<h3>Largest pairwise overlaps</h3>
<div class="table-container">{format_table(section['top_overlaps'], 25)}</div>
</section>
"""
        )

    speaker_rule = (
        html.escape(args.speaker_column)
        if args.speaker_column.strip()
        else "CSV filename stem"
    )
    fit_cap = "all" if args.max_fit_points <= 0 else str(args.max_fit_points)
    plot_cap = "all" if args.max_plot_points <= 0 else str(args.max_plot_points)

    report = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Class-conditional 3D KDE analysis</title>
<style>
body {{font-family:Arial,Helvetica,sans-serif;margin:2rem;line-height:1.45;color:#222}}
h1 {{margin-bottom:.25rem}}
h2 {{margin-top:3rem;border-bottom:2px solid #ddd;padding-bottom:.35rem}}
h3 {{margin-top:1.8rem}}
.description {{max-width:1150px;color:#444}}
.note {{max-width:1150px;background:#f5f5f5;border-left:4px solid #888;padding:.9rem 1rem;margin:1.2rem 0}}
.table-container {{overflow-x:auto;border:1px solid #ddd;margin-bottom:1.8rem}}
table {{border-collapse:collapse;width:100%;font-size:.79rem}}
th,td {{border:1px solid #ddd;padding:.42rem .55rem;text-align:right;white-space:nowrap}}
th {{background:#eee;position:sticky;top:0;z-index:1}}
td:first-child,th:first-child {{text-align:left}}
tbody tr:nth-child(even) {{background:#fafafa}}
img {{max-width:100%;height:auto;border:1px solid #ddd}}
.grid-two {{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:1.5rem}}
code {{background:#f2f2f2;padding:.1rem .25rem}}
</style>
<script>{get_plotlyjs()}</script>
</head>
<body>
<h1>Class-conditional 3D KDE analysis by vowel and speaker</h1>
<p class="description">
This report estimates where each known speaker/audio class is concentrated
inside the three-dimensional acoustic space for each vowel. It does not try
to discover disconnected clusters; overlap is allowed and quantified.
</p>
<div class="note"><strong>Interpretation:</strong>
smaller 50%, 80%, or 95% highest-density-region volumes indicate denser
concentration. Pairwise overlap ranges from 0 (no overlap) to 1 (identical
estimated densities). KDE calculations use globally standardized coordinates.
</div>
<h2>Methods</h2>
<ul>
<li>Input directory: <code>{html.escape(str(args.input_dir.resolve()))}</code></li>
<li>Axes: <code>{html.escape(axes[0])}</code>, <code>{html.escape(axes[1])}</code>, <code>{html.escape(axes[2])}</code></li>
<li>Class label: {speaker_rule}</li>
<li>Minimum points per speaker-vowel KDE: {args.min_points}</li>
<li>Maximum fit points per speaker-vowel: {fit_cap}</li>
<li>Maximum displayed points per speaker-vowel: {plot_cap}</li>
<li>Grid: {args.grid_size}<sup>3</sup> = {args.grid_size ** 3:,} cells per vowel</li>
<li>Bandwidth selection: within-speaker K-fold held-out likelihood, averaged equally across speakers</li>
</ul>
<div class="note">
Volumes and overlaps are numerical grid approximations in standardized units.
For final reporting, repeat with a larger grid and verify that rankings and
conclusions are stable.
</div>
<h2>Overall summary</h2><div class="table-container">{format_table(overall)}</div>
<h2>Token filtering summary</h2><div class="table-container">{format_table(filtering)}</div>
<h2>Input inventory</h2><div class="table-container">{format_table(inventory)}</div>
{''.join(section_html)}
<h2>Interpretation cautions</h2>
<ul>
<li>This is supervised class-conditional density estimation, not unsupervised clustering.</li>
<li>Large overlap is a substantive result, not an algorithmic failure.</li>
<li>In-sample log density is descriptive, not held-out classification performance.</li>
<li>Grid resolution, grid bounds, bandwidth, and outlier filtering affect numerical values.</li>
<li>The analysis quantifies acoustic concentration but does not identify its physiological, linguistic, or social cause.</li>
</ul>
</body></html>"""
    path.write_text(report, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.input_dir = args.input_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    axes = tuple(str(value) for value in args.axes)
    vowels = tuple(str(value).strip().lower() for value in args.vowels)
    candidates = tuple(sorted(set(float(value) for value in args.bandwidths)))

    assets = args.output_dir / "assets"
    matrices = args.output_dir / "overlap_matrices"
    assets.mkdir(parents=True, exist_ok=True)
    matrices.mkdir(parents=True, exist_ok=True)

    print("=== Class-conditional 3D KDE analysis ===")
    print(f"Input: {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Axes: {axes}")

    points, inventory = load_points(args)
    points = points[points["vowel"].isin(vowels)].reset_index(drop=True)
    points, filtering = robust_filter(points, vowels, args.robust_z_cutoff)

    scaler = StandardScaler()
    points[["sx", "sy", "sz"]] = scaler.fit_transform(points[["x", "y", "z"]])

    metric_tables: list[pd.DataFrame] = []
    pair_tables: list[pd.DataFrame] = []
    cv_tables: list[pd.DataFrame] = []
    overall_rows: list[dict[str, object]] = []
    sections: list[dict[str, object]] = []

    for vowel_index, vowel in enumerate(vowels, start=1):
        print(f"\n--- /{vowel}/ [{vowel_index}/{len(vowels)}] ---")
        vowel_points = points[points["vowel"] == vowel].copy()
        counts = vowel_points.groupby("speaker").size()
        speakers = sorted(counts[counts >= args.min_points].index.astype(str))
        vowel_points = vowel_points[
            vowel_points["speaker"].astype(str).isin(speakers)
        ].copy()

        print(f"Retained tokens: {len(vowel_points):,}")
        print(f"Eligible speaker/audio classes: {len(speakers)}")
        if len(speakers) < 2:
            print("Skipped: fewer than two eligible classes")
            continue

        full_arrays: dict[str, np.ndarray] = {}
        fit_arrays: dict[str, np.ndarray] = {}
        for speaker in speakers:
            values = vowel_points[
                vowel_points["speaker"].astype(str) == speaker
            ][["sx", "sy", "sz"]].to_numpy(float)
            full_arrays[speaker] = values
            fit_arrays[speaker] = capped_sample(
                values,
                args.max_fit_points,
                stable_seed(f"fit-{vowel}-{speaker}", args.seed),
            )

        print("Selecting shared bandwidth...")
        bandwidth, cv = choose_bandwidth(
            fit_arrays, candidates, args.cv_folds, stable_seed(vowel, args.seed)
        )
        cv.insert(0, "vowel", vowel)
        cv_tables.append(cv)
        save_cv_plot(cv, bandwidth, assets / f"{vowel}_bandwidth_cv.png", vowel)
        print(f"Selected bandwidth: {bandwidth:g}")

        grid, cell_volume = make_grid(
            vowel_points[["sx", "sy", "sz"]].to_numpy(float),
            args.grid_size,
            bandwidth,
            args.grid_padding,
        )

        densities: list[np.ndarray] = []
        metric_rows: list[dict[str, object]] = []

        for position, speaker in enumerate(speakers, start=1):
            print(f"Fitting/evaluating [{position:02d}/{len(speakers):02d}] {speaker}")
            model = make_kde(bandwidth).fit(fit_arrays[speaker])
            log_grid = score_chunks(model, grid, args.chunk_size)
            density = normalized_density(log_grid, cell_volume).astype(np.float32)
            densities.append(density)

            peak_index = int(np.argmax(density))
            peak_std = grid[peak_index]
            peak_original = scaler.inverse_transform(peak_std.reshape(1, -1))[0]
            self_score = model.score_samples(full_arrays[speaker])
            self_score = self_score[np.isfinite(self_score)]

            row = {
                "vowel": vowel,
                "speaker": speaker,
                "n_retained_tokens": len(full_arrays[speaker]),
                "n_fit_tokens": len(fit_arrays[speaker]),
                "bandwidth": bandwidth,
                "mean_self_log_density": float(np.mean(self_score)),
                "median_self_log_density": float(np.median(self_score)),
                "peak_log_density_grid": float(np.log(max(float(density[peak_index]), 1e-300))),
                "peak_x_standardized": float(peak_std[0]),
                "peak_y_standardized": float(peak_std[1]),
                "peak_z_standardized": float(peak_std[2]),
                "peak_x_original": float(peak_original[0]),
                "peak_y_original": float(peak_original[1]),
                "peak_z_original": float(peak_original[2]),
            }
            row.update(hdr_values(density, cell_volume))
            metric_rows.append(row)

        density_matrix = np.vstack(densities).astype(np.float32)
        total_density = density_matrix.sum(axis=0)
        winners = density_matrix.argmax(axis=0)
        metrics = pd.DataFrame(metric_rows)

        for index, speaker in enumerate(speakers):
            density = density_matrix[index]
            peak = int(np.argmax(density))
            dominant = winners == index
            mask = metrics["speaker"].astype(str) == speaker
            metrics.loc[mask, "dominant_grid_volume"] = (
                np.count_nonzero(dominant) * cell_volume
            )
            metrics.loc[mask, "own_mass_in_dominant_region"] = (
                density[dominant].sum() * cell_volume
            )
            metrics.loc[mask, "equal_prior_posterior_at_own_peak"] = (
                density[peak] / total_density[peak] if total_density[peak] > 0 else np.nan
            )

        overlap_matrix = np.eye(len(speakers), dtype=float)
        pair_rows: list[dict[str, object]] = []

        for first in range(len(speakers)):
            for second in range(first + 1, len(speakers)):
                d1 = density_matrix[first]
                d2 = density_matrix[second]
                overlap = float(np.minimum(d1, d2).sum() * cell_volume)
                bc = float(np.sqrt(d1 * d2).sum() * cell_volume)
                bc = min(max(bc, 0.0), 1.0)
                hellinger = math.sqrt(max(0.0, 1.0 - bc))
                overlap_matrix[first, second] = overlap
                overlap_matrix[second, first] = overlap
                pair_rows.append(
                    {
                        "vowel": vowel,
                        "speaker_1": speakers[first],
                        "speaker_2": speakers[second],
                        "overlap_coefficient": overlap,
                        "bhattacharyya_coefficient": bc,
                        "hellinger_distance": hellinger,
                    }
                )

        pairs = pd.DataFrame(pair_rows).sort_values(
            "overlap_coefficient", ascending=False
        )
        pd.DataFrame(
            overlap_matrix, index=speakers, columns=speakers
        ).to_csv(matrices / f"{vowel}_overlap_matrix.csv")

        save_overlap_heatmap(
            speakers, overlap_matrix, assets / f"{vowel}_overlap_heatmap.png", vowel
        )
        save_volume_plot(metrics, assets / f"{vowel}_hdr_volumes.png", vowel)

        metrics = metrics.sort_values("hdr_80_volume").reset_index(drop=True)
        plot_div = make_plotly_div(
            vowel_points, metrics, axes, args.max_plot_points, args.seed
        )

        metric_tables.append(metrics)
        pair_tables.append(pairs)
        pair_values = pairs["overlap_coefficient"].to_numpy(float)
        overall_rows.append(
            {
                "vowel": vowel,
                "retained_tokens": len(vowel_points),
                "eligible_speakers": len(speakers),
                "selected_bandwidth": bandwidth,
                "median_hdr_50_volume": metrics["hdr_50_volume"].median(),
                "median_hdr_80_volume": metrics["hdr_80_volume"].median(),
                "median_hdr_95_volume": metrics["hdr_95_volume"].median(),
                "median_pairwise_overlap": float(np.median(pair_values)),
                "maximum_pairwise_overlap": float(np.max(pair_values)),
                "most_overlapping_pair": (
                    f"{pairs.iloc[0]['speaker_1']} ↔ {pairs.iloc[0]['speaker_2']}"
                ),
            }
        )
        sections.append(
            {
                "vowel": vowel,
                "metrics": metrics,
                "top_overlaps": pairs,
                "plot_div": plot_div,
            }
        )

    if not metric_tables:
        raise RuntimeError("No vowel produced a valid KDE analysis")

    metrics_all = pd.concat(metric_tables, ignore_index=True)
    pairs_all = pd.concat(pair_tables, ignore_index=True)
    cv_all = pd.concat(cv_tables, ignore_index=True)
    overall = pd.DataFrame(overall_rows)

    metrics_all.to_csv(args.output_dir / "kde_vowel_speaker_metrics.csv", index=False)
    pairs_all.to_csv(args.output_dir / "kde_pairwise_overlap.csv", index=False)
    cv_all.to_csv(args.output_dir / "kde_bandwidth_cv.csv", index=False)
    overall.to_csv(args.output_dir / "kde_overall_summary.csv", index=False)
    filtering.to_csv(args.output_dir / "kde_filter_summary.csv", index=False)
    inventory.to_csv(args.output_dir / "kde_input_inventory.csv", index=False)

    config = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "axes": list(axes),
        "label_column": args.label_column,
        "speaker_column": args.speaker_column or None,
        "vowels": list(vowels),
        "min_points": args.min_points,
        "max_fit_points": args.max_fit_points,
        "max_plot_points": args.max_plot_points,
        "bandwidths": list(candidates),
        "cv_folds": args.cv_folds,
        "grid_size": args.grid_size,
        "grid_padding": args.grid_padding,
        "robust_z_cutoff": args.robust_z_cutoff,
        "seed": args.seed,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    }
    (args.output_dir / "kde_analysis_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    report_path = args.output_dir / "kde_3d_report.html"
    write_report(
        report_path,
        args,
        axes,
        overall,
        filtering,
        inventory,
        sections,
    )

    print("\n=== Analysis completed ===")
    print(f"HTML report: {report_path}")
    print(f"Speaker metrics: {args.output_dir / 'kde_vowel_speaker_metrics.csv'}")
    print(f"Pairwise overlap: {args.output_dir / 'kde_pairwise_overlap.csv'}")
    print(f"Bandwidth CV: {args.output_dir / 'kde_bandwidth_cv.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())