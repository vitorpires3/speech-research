#!/usr/bin/env python3
"""
Build token-level trajectory features from new-fave tracks files.

Input:
    data/processed/new_fave_tracks/*.csv

Output:
    data/processed/track_token_features/<audio>_track_features.csv
    data/processed/tables/all_track_token_features.csv

One output row = one vowel token, identified by id inside one audio.

Main trajectory variables:
    F1_s, F2_s, F3_s, B1, B2, B3, f0

For each variable:
    value20, value50, value80
    delta20_80 = value80 - value20
    range
    slope
    curvature

Additional trajectory measures:
    displacement_f1f2
    trajectory_length_f1f2
    trajectory_ratio_f1f2

    displacement_f1f2f3
    trajectory_length_f1f2f3
    trajectory_ratio_f1f2f3
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROPORTIONAL_POINTS = [0.20, 0.50, 0.80]

CONTEXT_COLUMNS = [
    "file_name",
    "group",
    "speaker_num",
    "label",
    "word",
    "stress",
    "pre_word",
    "fol_word",
    "pre_seg",
    "fol_seg",
    "abs_pre_seg",
    "abs_fol_seg",
    "context",
    "point_heuristic",
    "max_formant",
    "smooth_error",
    "optimized",
]


def normalize_colname(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {normalize_colname(c): c for c in df.columns}

    for candidate in candidates:
        key = normalize_colname(candidate)
        if key in normalized:
            return normalized[key]

    return None


def read_csv_safely(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def first_non_null(series: pd.Series):
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return valid.iloc[0]


def numeric_series(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def build_prop_time(df: pd.DataFrame) -> pd.Series:
    """
    Prefer prop_time if available.
    Otherwise, compute proportional time from time or rel_time.
    """
    prop_col = find_col(df, ["prop_time", "proptime"])
    if prop_col is not None:
        prop = pd.to_numeric(df[prop_col], errors="coerce")
        return prop.clip(lower=0, upper=1)

    time_col = find_col(df, ["time", "rel_time", "reltime"])
    if time_col is None:
        # fallback: equally spaced positions
        if len(df) == 1:
            return pd.Series([0.5], index=df.index, dtype="float64")
        return pd.Series(
            np.linspace(0, 1, len(df)),
            index=df.index,
            dtype="float64",
        )

    t = pd.to_numeric(df[time_col], errors="coerce")

    if t.notna().sum() <= 1:
        return pd.Series(0.5, index=df.index, dtype="float64")

    t_min = t.min()
    t_max = t.max()

    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max == t_min:
        return pd.Series(0.5, index=df.index, dtype="float64")

    return ((t - t_min) / (t_max - t_min)).clip(lower=0, upper=1)


def clean_xy(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    data = pd.DataFrame({"x": x, "y": y})
    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    if data.empty:
        return np.array([]), np.array([])

    # If duplicated prop_time values exist, average them.
    data = data.groupby("x", as_index=False)["y"].mean()
    data = data.sort_values("x")

    return data["x"].to_numpy(dtype=float), data["y"].to_numpy(dtype=float)


def interpolate_at(x: pd.Series, y: pd.Series, target: float) -> float:
    x_arr, y_arr = clean_xy(x, y)

    if len(x_arr) == 0:
        return np.nan

    if len(x_arr) == 1:
        return float(y_arr[0])

    return float(np.interp(target, x_arr, y_arr))


def compute_range(y: pd.Series) -> float:
    y_arr = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

    if len(y_arr) == 0:
        return np.nan

    return float(y_arr.max() - y_arr.min())


def compute_linear_slope(x: pd.Series, y: pd.Series) -> float:
    """
    Linear OLS slope for:
        y = beta0 + beta1 * t

    Uses centered proportional time:
        t = prop_time - 0.5

    For a purely linear model, centering changes only the intercept,
    not the slope.
    """
    x_arr, y_arr = clean_xy(x, y)

    if len(x_arr) < 2:
        return np.nan

    if np.nanstd(x_arr) == 0:
        return np.nan

    t = x_arr - 0.5
    X = np.column_stack([np.ones(len(t)), t])

    try:
        beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
        return float(beta[1])
    except Exception:
        return np.nan


def compute_quadratic_curvature(x: pd.Series, y: pd.Series) -> float:
    """
    Quadratic OLS curvature for:
        y = beta0 + beta1 * t + beta2 * t^2

    Uses centered proportional time:
        t = prop_time - 0.5

    The curvature is beta2.
    """
    x_arr, y_arr = clean_xy(x, y)

    if len(x_arr) < 3:
        return np.nan

    if len(np.unique(x_arr)) < 3:
        return np.nan

    t = x_arr - 0.5
    X = np.column_stack([np.ones(len(t)), t, t**2])

    try:
        beta = np.linalg.lstsq(X, y_arr, rcond=None)[0]
        return float(beta[2])
    except Exception:
        return np.nan


def trajectory_values_on_interval(
    x: pd.Series,
    values: list[pd.Series],
    start: float = 0.20,
    end: float = 0.80,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Build a shared proportional-time grid between start and end,
    including interpolated endpoints.
    """
    base_x = pd.to_numeric(x, errors="coerce")
    valid_x = base_x.replace([np.inf, -np.inf], np.nan).dropna()

    if len(valid_x) == 0:
        return np.array([]), []

    inner = valid_x[(valid_x >= start) & (valid_x <= end)].to_numpy(dtype=float)
    grid = np.unique(np.concatenate([[start], inner, [end]]))
    grid = np.sort(grid)

    interpolated_values = []

    for y in values:
        x_arr, y_arr = clean_xy(base_x, y)

        if len(x_arr) == 0:
            return np.array([]), []

        if len(x_arr) == 1:
            interp_y = np.repeat(y_arr[0], len(grid))
        else:
            interp_y = np.interp(grid, x_arr, y_arr)

        interpolated_values.append(interp_y)

    return grid, interpolated_values


def compute_displacement_and_length(
    x: pd.Series,
    values: list[pd.Series],
    ratio_min_displacement: float,
    start: float = 0.20,
    end: float = 0.80,
) -> tuple[float, float, float]:
    """
    Computes:
        displacement = straight-line distance between 20% and 80%
        trajectory_length = accumulated path length between 20% and 80%
        ratio = trajectory_length / displacement
    """
    grid, ys = trajectory_values_on_interval(x, values, start=start, end=end)

    if len(grid) < 2 or len(ys) == 0:
        return np.nan, np.nan, np.nan

    Y = np.column_stack(ys)

    if np.isnan(Y).any():
        return np.nan, np.nan, np.nan

    displacement = float(np.linalg.norm(Y[-1, :] - Y[0, :]))

    diffs = np.diff(Y, axis=0)
    step_distances = np.linalg.norm(diffs, axis=1)
    trajectory_length = float(np.sum(step_distances))

    if not np.isfinite(displacement) or displacement < ratio_min_displacement:
        ratio = np.nan
    else:
        ratio = float(trajectory_length / displacement)

    return displacement, trajectory_length, ratio


def prepare_bandwidth_series(df: pd.DataFrame, hz_candidates: list[str], raw_candidates: list[str]) -> pd.Series:
    """
    Prefer B*_Hz if available.

    If only B1/B2/B3 exists, new-fave/FastTrack may store bandwidths
    in natural-log scale. If the median is very small, assume log-scale
    and convert with exp().
    """
    hz_col = find_col(df, hz_candidates)
    if hz_col is not None:
        return numeric_series(df, hz_col)

    raw_col = find_col(df, raw_candidates)
    raw = numeric_series(df, raw_col)

    finite = raw.replace([np.inf, -np.inf], np.nan).dropna()

    if len(finite) == 0:
        return raw

    # Bandwidth in Hz is normally much larger than 20.
    # Log bandwidths are typically around 4-8.
    if finite.median() < 20:
        return np.exp(raw)

    return raw


def extract_features_for_group(
    group: pd.DataFrame,
    audio_name: str,
    source_file: str,
    variable_series: dict[str, pd.Series],
    ratio_min_displacement: float,
) -> dict:
    prop_time = build_prop_time(group)

    row = {
        "audio": audio_name,
        "source_file": source_file,
        "id": first_non_null(group["id"]) if "id" in group.columns else np.nan,
        "n_track_points": int(len(group)),
    }

    for col in CONTEXT_COLUMNS:
        real_col = find_col(group, [col])
        if real_col is not None and real_col not in row:
            row[col] = first_non_null(group[real_col])

    dur_col = find_col(group, ["dur", "duration"])
    if dur_col is not None:
        row["dur"] = first_non_null(pd.to_numeric(group[dur_col], errors="coerce"))
    else:
        time_col = find_col(group, ["time"])
        if time_col is not None:
            t = pd.to_numeric(group[time_col], errors="coerce")
            row["dur"] = float(t.max() - t.min()) if t.notna().sum() >= 2 else np.nan
        else:
            row["dur"] = np.nan

    for var_name, series in variable_series.items():
        y = series.loc[group.index]

        v20 = interpolate_at(prop_time, y, 0.20)
        v50 = interpolate_at(prop_time, y, 0.50)
        v80 = interpolate_at(prop_time, y, 0.80)

        row[f"{var_name}_value20"] = v20
        row[f"{var_name}_value50"] = v50
        row[f"{var_name}_value80"] = v80
        row[f"{var_name}_delta20_80"] = v80 - v20 if np.isfinite(v20) and np.isfinite(v80) else np.nan
        row[f"{var_name}_range"] = compute_range(y)
        row[f"{var_name}_slope"] = compute_linear_slope(prop_time, y)
        row[f"{var_name}_curvature"] = compute_quadratic_curvature(prop_time, y)

    f1 = variable_series["f1s"].loc[group.index]
    f2 = variable_series["f2s"].loc[group.index]
    f3 = variable_series["f3s"].loc[group.index]

    disp2, length2, ratio2 = compute_displacement_and_length(
        prop_time,
        [f1, f2],
        ratio_min_displacement=ratio_min_displacement,
        start=0.20,
        end=0.80,
    )

    disp3, length3, ratio3 = compute_displacement_and_length(
        prop_time,
        [f1, f2, f3],
        ratio_min_displacement=ratio_min_displacement,
        start=0.20,
        end=0.80,
    )

    row["displacement_f1f2"] = disp2
    row["trajectory_length_f1f2"] = length2
    row["trajectory_ratio_f1f2"] = ratio2

    row["displacement_f1f2f3"] = disp3
    row["trajectory_length_f1f2f3"] = length3
    row["trajectory_ratio_f1f2f3"] = ratio3

    return row


def build_features_for_file(
    input_path: Path,
    output_path: Path,
    ratio_min_displacement: float,
) -> pd.DataFrame:
    df = read_csv_safely(input_path)

    id_col = find_col(df, ["id"])
    if id_col is None:
        raise ValueError(f"No id column found in {input_path}")

    if id_col != "id":
        df = df.rename(columns={id_col: "id"})

    audio_col = find_col(df, ["file_name", "filename", "audio"])
    if audio_col is not None:
        audio_name = str(first_non_null(df[audio_col]))
        if audio_name == "nan":
            audio_name = input_path.stem
    else:
        audio_name = input_path.stem

    f1s_col = find_col(df, ["F1_s", "F1s"])
    f2s_col = find_col(df, ["F2_s", "F2s"])
    f3s_col = find_col(df, ["F3_s", "F3s"])

    # Fallback to raw formants if smoothed tracks are unavailable.
    if f1s_col is None:
        f1s_col = find_col(df, ["F1"])
    if f2s_col is None:
        f2s_col = find_col(df, ["F2"])
    if f3s_col is None:
        f3s_col = find_col(df, ["F3"])

    variable_series = {
        "f1s": numeric_series(df, f1s_col),
        "f2s": numeric_series(df, f2s_col),
        "f3s": numeric_series(df, f3s_col),
        "b1": prepare_bandwidth_series(df, ["B1_Hz", "B1Hz"], ["B1"]),
        "b2": prepare_bandwidth_series(df, ["B2_Hz", "B2Hz"], ["B2"]),
        "b3": prepare_bandwidth_series(df, ["B3_Hz", "B3Hz"], ["B3"]),
        "f0": numeric_series(df, find_col(df, ["f0", "F0"])),
    }

    rows = []

    for _, group in df.groupby("id", sort=False):
        row = extract_features_for_group(
            group=group,
            audio_name=audio_name,
            source_file=input_path.name,
            variable_series=variable_series,
            ratio_min_displacement=ratio_min_displacement,
        )
        rows.append(row)

    out = pd.DataFrame(rows)

    # Put identification/context columns first when available.
    first_cols = [
        "audio",
        "source_file",
        "id",
        "label",
        "word",
        "stress",
        "pre_word",
        "fol_word",
        "pre_seg",
        "fol_seg",
        "abs_pre_seg",
        "abs_fol_seg",
        "context",
        "dur",
        "n_track_points",
    ]

    ordered = [c for c in first_cols if c in out.columns]
    remaining = [c for c in out.columns if c not in ordered]
    out = out[ordered + remaining]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build token-level trajectory features from new-fave tracks CSV files."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing new-fave tracks CSV files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where one feature CSV per audio will be written.",
    )

    parser.add_argument(
        "--combined-output",
        default=None,
        help="Optional path for one concatenated CSV with all audios.",
    )

    parser.add_argument(
        "--ratio-min-displacement",
        type=float,
        default=20.0,
        help="Minimum displacement in Hz required to compute trajectory_ratio. Default: 20.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    all_outputs = []

    for input_path in csv_files:
        output_name = f"{input_path.stem}_track_features.csv"
        output_path = output_dir / output_name

        print(f"Processing {input_path.name} -> {output_path}")

        out = build_features_for_file(
            input_path=input_path,
            output_path=output_path,
            ratio_min_displacement=args.ratio_min_displacement,
        )

        all_outputs.append(out)

    if args.combined_output:
        combined = pd.concat(all_outputs, ignore_index=True)
        combined_path = Path(args.combined_output)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, index=False)
        print(f"Combined output written to {combined_path}")

    print("Done.")


if __name__ == "__main__":
    main()