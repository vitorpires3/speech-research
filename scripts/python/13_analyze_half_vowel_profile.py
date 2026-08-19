#!/usr/bin/env python3

from pathlib import Path
import argparse
import itertools
import re
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


VOWEL_ORDER = ["i", "e", "a", "o", "u"]
CONSECUTIVE_PAIRS = [("i", "e"), ("e", "a"), ("a", "o"), ("o", "u"), ("u", "i")]


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
    lookup = {norm_col(col): col for col in df.columns}

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


def euclidean(x1, y1, x2, y2):
    return float(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def convex_hull_area(points):
    points = [
        (float(x), float(y))
        for x, y in points
        if np.isfinite(x) and np.isfinite(y)
    ]

    points = sorted(set(points))

    if len(points) < 3:
        return np.nan

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]

    if len(hull) < 3:
        return np.nan

    area = 0.0

    for p1, p2 in zip(hull, hull[1:] + hull[:1]):
        area += p1[0] * p2[1] - p2[0] * p1[1]

    return abs(area) / 2.0


def simple_silhouette(df):
    if len(df) < 3:
        return np.nan

    labels = df["vowel_norm"].to_numpy()
    X = df[["f1", "f2"]].to_numpy(dtype=float)

    if len(set(labels)) < 2:
        return np.nan

    diff = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    scores = []

    for i in range(len(df)):
        own_mask = labels == labels[i]

        if own_mask.sum() <= 1:
            continue

        a = distances[i, own_mask].sum() / (own_mask.sum() - 1)

        b_values = []

        for other_label in sorted(set(labels) - {labels[i]}):
            other_mask = labels == other_label
            if other_mask.sum() > 0:
                b_values.append(distances[i, other_mask].mean())

        if not b_values:
            continue

        b = min(b_values)

        denominator = max(a, b)

        if denominator > 0:
            scores.append((b - a) / denominator)

    if not scores:
        return np.nan

    return float(np.mean(scores))


def per_vowel_silhouette(df):
    if len(df) < 3:
        return pd.DataFrame()

    labels = df["vowel_norm"].to_numpy()
    X = df[["f1", "f2"]].to_numpy(dtype=float)

    if len(set(labels)) < 2:
        return pd.DataFrame()

    diff = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    rows = []

    for i in range(len(df)):
        own_mask = labels == labels[i]

        if own_mask.sum() <= 1:
            continue

        a = distances[i, own_mask].sum() / (own_mask.sum() - 1)

        b_values = []

        for other_label in sorted(set(labels) - {labels[i]}):
            other_mask = labels == other_label
            if other_mask.sum() > 0:
                b_values.append(distances[i, other_mask].mean())

        if not b_values:
            continue

        b = min(b_values)
        denominator = max(a, b)

        if denominator > 0:
            rows.append(
                {
                    "half": df.iloc[i]["half"],
                    "vowel": labels[i],
                    "silhouette_token": (b - a) / denominator,
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)

    return (
        out.groupby(["half", "vowel"], as_index=False)
        .agg(
            silhouette_mean=("silhouette_token", "mean"),
            silhouette_median=("silhouette_token", "median"),
            n_silhouette_tokens=("silhouette_token", "size"),
        )
    )


def load_tokens(points_file, audio=None, f1_min=150, f1_max=1200, f2_min=400, f2_max=4000):
    df = pd.read_csv(points_file)

    f1_col = find_col(df, ["f1", "F1"])
    f2_col = find_col(df, ["f2", "F2"])
    half_col = find_col(df, ["half"])
    vowel_col = find_col(df, ["vowel", "vowel_norm", "label", "phone", "segment"])

    if f1_col is None or f2_col is None:
        raise ValueError(f"Não encontrei colunas F1/F2. Colunas disponíveis: {list(df.columns)}")

    if half_col is None:
        raise ValueError("Não encontrei a coluna 'half'. Rode primeiro o script 12_split_points_by_audio_half.py.")

    if vowel_col is None:
        raise ValueError(f"Não encontrei coluna de vogal/label. Colunas disponíveis: {list(df.columns)}")

    audio_col = find_col(df, ["_audio_id", "audio", "file", "filename", "source_file"])

    out = pd.DataFrame()

    if audio_col is not None:
        out["audio"] = df[audio_col].astype(str).apply(clean_audio_id)
    else:
        out["audio"] = clean_audio_id(Path(points_file).stem)

    out["half"] = df[half_col].astype(str)
    out["vowel_raw"] = df[vowel_col]
    out["vowel_norm"] = df[vowel_col].apply(normalize_vowel)
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

    out = out.dropna(subset=["vowel_norm", "f1", "f2"])
    out = out[out["vowel_norm"].isin(VOWEL_ORDER)]
    out = out[out["half"].isin(["H1", "H2"])]

    out = out[
        (out["f1"] >= f1_min)
        & (out["f1"] <= f1_max)
        & (out["f2"] >= f2_min)
        & (out["f2"] <= f2_max)
    ].copy()

    if out.empty:
        raise ValueError("Depois dos filtros, não sobrou nenhum token.")

    return out


def compute_centroids(tokens):
    rows = []

    for half in ["H1", "H2"]:
        half_df = tokens[tokens["half"] == half]

        for vowel in VOWEL_ORDER:
            sub = half_df[half_df["vowel_norm"] == vowel]

            if sub.empty:
                continue

            f1_centroid = float(sub["f1"].median())
            f2_centroid = float(sub["f2"].median())

            distances = np.sqrt(
                (sub["f1"] - f1_centroid) ** 2
                + (sub["f2"] - f2_centroid) ** 2
            )

            rows.append(
                {
                    "half": half,
                    "vowel": vowel,
                    "n_tokens": int(len(sub)),
                    "f1_centroid": f1_centroid,
                    "f2_centroid": f2_centroid,
                    "f1_mean": float(sub["f1"].mean()),
                    "f2_mean": float(sub["f2"].mean()),
                    "f1_sd": float(sub["f1"].std(ddof=1)) if len(sub) > 1 else np.nan,
                    "f2_sd": float(sub["f2"].std(ddof=1)) if len(sub) > 1 else np.nan,
                    "f1_iqr": float(sub["f1"].quantile(0.75) - sub["f1"].quantile(0.25)),
                    "f2_iqr": float(sub["f2"].quantile(0.75) - sub["f2"].quantile(0.25)),
                    "within_dispersion_mean": float(distances.mean()),
                    "within_dispersion_median": float(distances.median()),
                }
            )

    return pd.DataFrame(rows)


def compute_pairwise_distances(centroids):
    rows = []

    for half in ["H1", "H2"]:
        c = centroids[centroids["half"] == half].copy()

        lookup = {
            row.vowel: row
            for row in c.itertuples(index=False)
        }

        for v1, v2 in itertools.combinations(VOWEL_ORDER, 2):
            if v1 not in lookup or v2 not in lookup:
                continue

            r1 = lookup[v1]
            r2 = lookup[v2]

            distance = euclidean(
                r1.f1_centroid,
                r1.f2_centroid,
                r2.f1_centroid,
                r2.f2_centroid,
            )

            pooled_dispersion = math.sqrt(
                (r1.within_dispersion_mean ** 2 + r2.within_dispersion_mean ** 2) / 2
            )

            separation_score = (
                distance / pooled_dispersion
                if pooled_dispersion > 0
                else np.nan
            )

            rows.append(
                {
                    "half": half,
                    "vowel_1": v1,
                    "vowel_2": v2,
                    "pair": f"{v1}-{v2}",
                    "centroid_distance": distance,
                    "pooled_within_dispersion": pooled_dispersion,
                    "pairwise_separation_score": separation_score,
                    "is_consecutive": (v1, v2) in CONSECUTIVE_PAIRS or (v2, v1) in CONSECUTIVE_PAIRS,
                }
            )

    return pd.DataFrame(rows)


def compute_vowel_differentiation(centroids, silhouette_by_vowel):
    rows = []

    for half in ["H1", "H2"]:
        c = centroids[centroids["half"] == half].copy()

        for row in c.itertuples(index=False):
            others = c[c["vowel"] != row.vowel]

            if others.empty:
                continue

            nearest_vowel = None
            nearest_distance = np.inf
            nearest_dispersion = np.nan

            for other in others.itertuples(index=False):
                distance = euclidean(
                    row.f1_centroid,
                    row.f2_centroid,
                    other.f1_centroid,
                    other.f2_centroid,
                )

                if distance < nearest_distance:
                    nearest_distance = distance
                    nearest_vowel = other.vowel
                    nearest_dispersion = other.within_dispersion_mean

            pooled_dispersion = math.sqrt(
                (row.within_dispersion_mean ** 2 + nearest_dispersion ** 2) / 2
            )

            differentiation_score = (
                nearest_distance / pooled_dispersion
                if pooled_dispersion > 0
                else np.nan
            )

            overlap_risk = (
                pooled_dispersion / nearest_distance
                if nearest_distance > 0
                else np.nan
            )

            rows.append(
                {
                    "half": half,
                    "vowel": row.vowel,
                    "nearest_vowel": nearest_vowel,
                    "nearest_centroid_distance": nearest_distance,
                    "within_dispersion_mean": row.within_dispersion_mean,
                    "pooled_dispersion_with_nearest": pooled_dispersion,
                    "differentiation_score": differentiation_score,
                    "overlap_risk": overlap_risk,
                }
            )

    diff = pd.DataFrame(rows)

    if not silhouette_by_vowel.empty:
        diff = diff.merge(
            silhouette_by_vowel,
            on=["half", "vowel"],
            how="left",
        )

    return diff


def classify_with_centroids(train_centroids, test_tokens):
    if train_centroids.empty or test_tokens.empty:
        return pd.DataFrame(), np.nan, pd.DataFrame()

    C = train_centroids[["f1_centroid", "f2_centroid"]].to_numpy(dtype=float)
    labels = train_centroids["vowel"].to_numpy()

    test = test_tokens.copy()
    X = test[["f1", "f2"]].to_numpy(dtype=float)

    diff = X[:, None, :] - C[None, :, :]
    distances = np.sqrt(np.sum(diff ** 2, axis=2))

    nearest_idx = np.argmin(distances, axis=1)
    test["predicted_vowel"] = labels[nearest_idx]
    test["nearest_centroid_distance"] = distances[np.arange(len(test)), nearest_idx]

    accuracy = float((test["vowel_norm"] == test["predicted_vowel"]).mean())

    confusion = pd.crosstab(
        test["vowel_norm"],
        test["predicted_vowel"],
        rownames=["true_vowel"],
        colnames=["predicted_vowel"],
    )

    for vowel in VOWEL_ORDER:
        if vowel not in confusion.index:
            confusion.loc[vowel] = 0
        if vowel not in confusion.columns:
            confusion[vowel] = 0

    confusion = confusion.loc[VOWEL_ORDER, VOWEL_ORDER]

    return confusion, accuracy, test


def compute_half_summary(tokens, centroids, pairwise):
    rows = []

    for half in ["H1", "H2"]:
        sub = tokens[tokens["half"] == half]
        c = centroids[centroids["half"] == half]
        p = pairwise[pairwise["half"] == half]

        counts = sub["vowel_norm"].value_counts()

        centroid_points = list(zip(c["f1_centroid"], c["f2_centroid"]))

        rows.append(
            {
                "half": half,
                "n_tokens": int(len(sub)),
                "n_vowels_present": int(c["vowel"].nunique()),
                "min_tokens_per_vowel": int(min([counts.get(v, 0) for v in VOWEL_ORDER])),
                "vowel_space_area": convex_hull_area(centroid_points),
                "mean_pairwise_centroid_distance": float(p["centroid_distance"].mean()) if not p.empty else np.nan,
                "min_pairwise_centroid_distance": float(p["centroid_distance"].min()) if not p.empty else np.nan,
                "mean_pairwise_separation_score": float(p["pairwise_separation_score"].mean()) if not p.empty else np.nan,
                "min_pairwise_separation_score": float(p["pairwise_separation_score"].min()) if not p.empty else np.nan,
                "mean_within_vowel_dispersion": float(c["within_dispersion_mean"].mean()) if not c.empty else np.nan,
                "silhouette_by_vowel": simple_silhouette(sub),
            }
        )

    return pd.DataFrame(rows)


def compute_comparison(tokens, centroids, pairwise, half_summary):
    h1_centroids = centroids[centroids["half"] == "H1"]
    h2_centroids = centroids[centroids["half"] == "H2"]

    centroid_join = h1_centroids.merge(
        h2_centroids,
        on="vowel",
        suffixes=("_H1", "_H2"),
    )

    shift_rows = []

    for row in centroid_join.itertuples(index=False):
        shift = euclidean(
            row.f1_centroid_H1,
            row.f2_centroid_H1,
            row.f1_centroid_H2,
            row.f2_centroid_H2,
        )

        shift_rows.append(
            {
                "vowel": row.vowel,
                "f1_centroid_H1": row.f1_centroid_H1,
                "f2_centroid_H1": row.f2_centroid_H1,
                "f1_centroid_H2": row.f1_centroid_H2,
                "f2_centroid_H2": row.f2_centroid_H2,
                "f1_shift_H2_minus_H1": row.f1_centroid_H2 - row.f1_centroid_H1,
                "f2_shift_H2_minus_H1": row.f2_centroid_H2 - row.f2_centroid_H1,
                "centroid_shift_distance": shift,
            }
        )

    centroid_shifts = pd.DataFrame(shift_rows)

    p1 = pairwise[pairwise["half"] == "H1"][["pair", "centroid_distance"]]
    p2 = pairwise[pairwise["half"] == "H2"][["pair", "centroid_distance"]]

    distance_join = p1.merge(p2, on="pair", suffixes=("_H1", "_H2"))

    if len(distance_join) >= 2:
        x = distance_join["centroid_distance_H1"].to_numpy(dtype=float)
        y = distance_join["centroid_distance_H2"].to_numpy(dtype=float)

        if np.std(x) > 0 and np.std(y) > 0:
            distance_matrix_correlation = float(np.corrcoef(x, y)[0, 1])
        else:
            distance_matrix_correlation = np.nan
    else:
        distance_matrix_correlation = np.nan

    h1_tokens = tokens[tokens["half"] == "H1"]
    h2_tokens = tokens[tokens["half"] == "H2"]

    conf_h1_to_h2, acc_h1_to_h2, pred_h1_to_h2 = classify_with_centroids(
        h1_centroids,
        h2_tokens,
    )

    conf_h2_to_h1, acc_h2_to_h1, pred_h2_to_h1 = classify_with_centroids(
        h2_centroids,
        h1_tokens,
    )

    s1 = half_summary[half_summary["half"] == "H1"].iloc[0]
    s2 = half_summary[half_summary["half"] == "H2"].iloc[0]

    comparison = pd.DataFrame(
        [
            {
                "mean_centroid_shift": float(centroid_shifts["centroid_shift_distance"].mean()) if not centroid_shifts.empty else np.nan,
                "median_centroid_shift": float(centroid_shifts["centroid_shift_distance"].median()) if not centroid_shifts.empty else np.nan,
                "max_centroid_shift": float(centroid_shifts["centroid_shift_distance"].max()) if not centroid_shifts.empty else np.nan,
                "distance_matrix_correlation": distance_matrix_correlation,
                "vowel_space_area_H1": s1["vowel_space_area"],
                "vowel_space_area_H2": s2["vowel_space_area"],
                "vowel_space_area_abs_diff": abs(s2["vowel_space_area"] - s1["vowel_space_area"]),
                "mean_pairwise_distance_H1": s1["mean_pairwise_centroid_distance"],
                "mean_pairwise_distance_H2": s2["mean_pairwise_centroid_distance"],
                "mean_pairwise_distance_abs_diff": abs(s2["mean_pairwise_centroid_distance"] - s1["mean_pairwise_centroid_distance"]),
                "silhouette_H1": s1["silhouette_by_vowel"],
                "silhouette_H2": s2["silhouette_by_vowel"],
                "silhouette_abs_diff": abs(s2["silhouette_by_vowel"] - s1["silhouette_by_vowel"]),
                "classification_accuracy_H1_centroids_to_H2_tokens": acc_h1_to_h2,
                "classification_accuracy_H2_centroids_to_H1_tokens": acc_h2_to_h1,
                "classification_accuracy_mean": float(np.nanmean([acc_h1_to_h2, acc_h2_to_h1])),
            }
        ]
    )

    predictions = pd.concat(
        [
            pred_h1_to_h2.assign(direction="H1_centroids_to_H2_tokens"),
            pred_h2_to_h1.assign(direction="H2_centroids_to_H1_tokens"),
        ],
        ignore_index=True,
    )

    return comparison, centroid_shifts, distance_join, conf_h1_to_h2, conf_h2_to_h1, predictions


def plot_vowel_space(tokens, centroids, half, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sub = tokens[tokens["half"] == half]
    c = centroids[centroids["half"] == half]

    fig, ax = plt.subplots(figsize=(8, 6))

    for vowel in VOWEL_ORDER:
        sv = sub[sub["vowel_norm"] == vowel]

        if sv.empty:
            continue

        ax.scatter(
            sv["f2"],
            sv["f1"],
            s=18,
            alpha=0.45,
            label=f"/{vowel}/ tokens",
        )

        cv = c[c["vowel"] == vowel]

        if not cv.empty:
            row = cv.iloc[0]

            ax.scatter(
                row["f2_centroid"],
                row["f1_centroid"],
                s=140,
                marker="X",
            )

            ax.text(
                row["f2_centroid"],
                row["f1_centroid"],
                f" /{vowel}/",
                fontsize=12,
                weight="bold",
            )

    polygon = c.set_index("vowel").reindex(VOWEL_ORDER).dropna(subset=["f1_centroid", "f2_centroid"])

    if len(polygon) >= 3:
        x = polygon["f2_centroid"].tolist()
        y = polygon["f1_centroid"].tolist()

        x.append(x[0])
        y.append(y[0])

        ax.plot(x, y, linewidth=1.5)

    ax.set_title(f"F1 × F2 vowel space - {half}")
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_centroid_shift(centroids, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    h1 = centroids[centroids["half"] == "H1"]
    h2 = centroids[centroids["half"] == "H2"]

    merged = h1.merge(h2, on="vowel", suffixes=("_H1", "_H2"))

    fig, ax = plt.subplots(figsize=(8, 6))

    for row in merged.itertuples(index=False):
        ax.scatter(row.f2_centroid_H1, row.f1_centroid_H1, s=100, marker="o")
        ax.scatter(row.f2_centroid_H2, row.f1_centroid_H2, s=100, marker="X")

        ax.annotate(
            "",
            xy=(row.f2_centroid_H2, row.f1_centroid_H2),
            xytext=(row.f2_centroid_H1, row.f1_centroid_H1),
            arrowprops=dict(arrowstyle="->", lw=1.5),
        )

        ax.text(row.f2_centroid_H1, row.f1_centroid_H1, f"/{row.vowel}/ H1", fontsize=9)
        ax.text(row.f2_centroid_H2, row.f1_centroid_H2, f"/{row.vowel}/ H2", fontsize=9)

    ax.set_title("Centroid shift from H1 to H2")
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_differentiation(vowel_diff, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    positions = np.arange(len(VOWEL_ORDER))
    width = 0.35

    for i, half in enumerate(["H1", "H2"]):
        sub = vowel_diff[vowel_diff["half"] == half].set_index("vowel").reindex(VOWEL_ORDER)

        values = sub["differentiation_score"].to_numpy(dtype=float)

        ax.bar(
            positions + (i - 0.5) * width,
            values,
            width=width,
            label=half,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"/{v}/" for v in VOWEL_ORDER])
    ax.set_ylabel("Differentiation score")
    ax.set_title("Vowel differentiation by half")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--points-file",
        required=True,
        help="CSV gerado pelo split: all_points_with_halves.csv",
    )

    parser.add_argument(
        "--audio",
        default=None,
        help="Opcional: selecionar um áudio específico.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pasta de saída da análise.",
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

    centroids = compute_centroids(tokens)
    pairwise = compute_pairwise_distances(centroids)
    silhouette_by_vowel = per_vowel_silhouette(tokens)
    vowel_diff = compute_vowel_differentiation(centroids, silhouette_by_vowel)
    half_summary = compute_half_summary(tokens, centroids, pairwise)

    (
        comparison,
        centroid_shifts,
        distance_join,
        conf_h1_to_h2,
        conf_h2_to_h1,
        predictions,
    ) = compute_comparison(tokens, centroids, pairwise, half_summary)

    comparison.insert(0, "audio", audio_name)

    tokens.to_csv(output_dir / "tokens_used.csv", index=False)
    centroids.to_csv(output_dir / "centroids_by_half.csv", index=False)
    pairwise.to_csv(output_dir / "pairwise_centroid_distances.csv", index=False)
    pairwise[pairwise["is_consecutive"]].to_csv(output_dir / "consecutive_centroid_distances.csv", index=False)
    vowel_diff.to_csv(output_dir / "vowel_differentiation_by_half.csv", index=False)
    half_summary.to_csv(output_dir / "half_summary.csv", index=False)
    comparison.to_csv(output_dir / "half_similarity_summary.csv", index=False)
    centroid_shifts.to_csv(output_dir / "centroid_shifts_H1_H2.csv", index=False)
    distance_join.to_csv(output_dir / "pairwise_distance_comparison_H1_H2.csv", index=False)
    predictions.to_csv(output_dir / "cross_half_nearest_centroid_predictions.csv", index=False)

    conf_h1_to_h2.to_csv(output_dir / "confusion_H1_centroids_to_H2_tokens.csv")
    conf_h2_to_h1.to_csv(output_dir / "confusion_H2_centroids_to_H1_tokens.csv")

    plot_vowel_space(tokens, centroids, "H1", output_dir / "H1_f1_f2_vowel_space.png")
    plot_vowel_space(tokens, centroids, "H2", output_dir / "H2_f1_f2_vowel_space.png")
    plot_centroid_shift(centroids, output_dir / "centroid_shift_H1_to_H2.png")
    plot_differentiation(vowel_diff, output_dir / "vowel_differentiation_by_half.png")

    print()
    print("=== Half-vowel profile analysis ===")
    print(f"Audio: {audio_name}")
    print(f"Output dir: {output_dir}")
    print()
    print("Token counts:")
    print(pd.crosstab(tokens["half"], tokens["vowel_norm"]).reindex(columns=VOWEL_ORDER, fill_value=0).to_string())
    print()
    print("Half summary:")
    print(half_summary.to_string(index=False))
    print()
    print("Similarity summary:")
    print(comparison.to_string(index=False))
    print()
    print("Centroid shifts:")
    print(centroid_shifts.to_string(index=False))
    print()
    print("Vowel differentiation:")
    print(vowel_diff.to_string(index=False))
    print()
    print("Outputs principais:")
    print(f"- {output_dir / 'H1_f1_f2_vowel_space.png'}")
    print(f"- {output_dir / 'H2_f1_f2_vowel_space.png'}")
    print(f"- {output_dir / 'centroid_shift_H1_to_H2.png'}")
    print(f"- {output_dir / 'vowel_differentiation_by_half.png'}")
    print(f"- {output_dir / 'half_similarity_summary.csv'}")
    print(f"- {output_dir / 'centroids_by_half.csv'}")
    print(f"- {output_dir / 'vowel_differentiation_by_half.csv'}")
    print(f"- {output_dir / 'consecutive_centroid_distances.csv'}")


if __name__ == "__main__":
    main()
