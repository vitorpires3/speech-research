#!/usr/bin/env python3
"""
Interactive 3D F1-F2-F3 plots from new-FAVE points files.

Outputs:
- all_points_f1f2f3_full.csv
- all_points_f1f2f3_plot_half.csv
- counts_by_audio_vowel.csv
- speaker_style_mapping.csv
- all_vowels_f1f2f3_3d.html
- vowel_a_f1f2f3_3d.html
- vowel_e_f1f2f3_3d.html
- vowel_i_f1f2f3_3d.html
- vowel_o_f1f2f3_3d.html
- vowel_u_f1f2f3_3d.html

Plot convention:
x = F2
y = F1
z = F3

F1 and F2 axes are reversed, following the usual vowel-space convention.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go


VOWEL_COL_CANDIDATES = [
    "vowel", "phone", "label", "vowel_label", "phone_label",
    "ipa", "arpa", "vclass", "vowel_class"
]

F1_COL_CANDIDATES = ["F1", "f1", "F1_Hz", "f1_hz", "f1hz", "F1Hz"]
F2_COL_CANDIDATES = ["F2", "f2", "F2_Hz", "f2_hz", "f2hz", "F2Hz"]
F3_COL_CANDIDATES = ["F3", "f3", "F3_Hz", "f3_hz", "f3hz", "F3Hz"]

TOKEN_ID_CANDIDATES = ["id", "token_id", "vowel_id", "phone_id", "measurement_id"]

EXTRA_COL_CANDIDATES = [
    "beg", "end", "start", "stop", "time", "duration", "dur",
    "word", "speaker", "file", "filename", "file_name"
]

VOWEL_COLORS = {
    "a": "rgba(214, 39, 40, 0.55)",
    "e": "rgba(255, 127, 14, 0.55)",
    "i": "rgba(44, 160, 44, 0.55)",
    "o": "rgba(31, 119, 180, 0.55)",
    "u": "rgba(148, 103, 189, 0.55)",
}

SPEAKER_COLORS = [
    "rgba(31, 119, 180, 0.55)",
    "rgba(255, 127, 14, 0.55)",
    "rgba(44, 160, 44, 0.55)",
    "rgba(214, 39, 40, 0.55)",
    "rgba(148, 103, 189, 0.55)",
    "rgba(140, 86, 75, 0.55)",
    "rgba(227, 119, 194, 0.55)",
]

# Plotly 3D marker symbols are more limited than matplotlib.
SPEAKER_MARKERS = [
    "circle", "diamond", "square", "cross", "x",
    "circle-open", "diamond-open"
]


def normalize_col_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def find_column(df: pd.DataFrame, candidates: list[str], explicit: str | None, role: str) -> str:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(
                f"Column '{explicit}' was requested for {role}, but it was not found. "
                f"Available columns: {list(df.columns)}"
            )
        return explicit

    normalized_to_original = {normalize_col_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]

    raise ValueError(
        f"Could not automatically detect column for {role}. "
        f"Available columns: {list(df.columns)}. "
        f"Try passing --{role}-col explicitly."
    )


def find_optional_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_to_original = {normalize_col_name(c): c for c in df.columns}

    for cand in candidates:
        key = normalize_col_name(cand)
        if key in normalized_to_original:
            return normalized_to_original[key]

    return None


def parse_label_map(items: list[str]) -> dict[str, str]:
    mapping = {}

    for item in items:
        if ":" not in item:
            continue

        src, dst = item.split(":", 1)
        mapping[src.strip().lower()] = dst.strip().lower()

    return mapping


def clean_vowel_label(value, vowels: set[str], label_map: dict[str, str]) -> str | None:
    if pd.isna(value):
        return None

    s = str(value).strip().lower()

    if s in label_map:
        s = label_map[s]

    # Remove stress, digit, length and separator markers.
    s2 = re.sub(r"[0-9ˈˌː:\.\s_-]+", "", s)

    if s2 in label_map:
        s2 = label_map[s2]

    if s2 in vowels:
        return s2

    return None


def audio_id_from_file(path: Path) -> str:
    stem = path.stem

    for suffix in [
        "_new_fave_points",
        "_fasttrack_points",
        "_points",
        ".new_fave_points",
        ".points",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]

    return stem


def load_points(args) -> pd.DataFrame:
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    vowels = set(args.vowels)
    label_map = parse_label_map(args.label_map)

    frames = []

    for path in files:
        df = pd.read_csv(path)

        if df.empty:
            continue

        vowel_col = find_column(df, VOWEL_COL_CANDIDATES, args.vowel_col, "vowel")
        f1_col = find_column(df, F1_COL_CANDIDATES, args.f1_col, "f1")
        f2_col = find_column(df, F2_COL_CANDIDATES, args.f2_col, "f2")
        f3_col = find_column(df, F3_COL_CANDIDATES, args.f3_col, "f3")
        token_col = find_optional_column(df, TOKEN_ID_CANDIDATES)

        keep_cols = [vowel_col, f1_col, f2_col, f3_col]

        if token_col is not None:
            keep_cols.append(token_col)

        for extra in EXTRA_COL_CANDIDATES:
            col = find_optional_column(df, [extra])
            if col is not None and col not in keep_cols:
                keep_cols.append(col)

        sub = df[keep_cols].copy()

        sub["audio"] = audio_id_from_file(path)
        sub["source_file"] = path.name
        sub["vowel_raw"] = sub[vowel_col]
        sub["vowel"] = sub[vowel_col].apply(
            lambda x: clean_vowel_label(x, vowels=vowels, label_map=label_map)
        )

        sub["f1"] = pd.to_numeric(sub[f1_col], errors="coerce")
        sub["f2"] = pd.to_numeric(sub[f2_col], errors="coerce")
        sub["f3"] = pd.to_numeric(sub[f3_col], errors="coerce")

        if token_col is not None:
            sub["token_id"] = sub[token_col]
        else:
            sub["token_id"] = np.arange(len(sub))

        sub = sub.dropna(subset=["vowel", "f1", "f2", "f3"])
        sub = sub[sub["vowel"].isin(args.vowels)]
        sub = sub[(sub["f1"] > 0) & (sub["f2"] > 0) & (sub["f3"] > 0)]

        frames.append(sub)

    if not frames:
        raise RuntimeError(
            "No valid F1/F2/F3 vowel points found. "
            "Check --vowel-col, --f1-col, --f2-col, --f3-col and --label-map."
        )

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["audio", "vowel"]).reset_index(drop=True)

    return out


def take_alternating_half(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one token, remove the next one, independently for each audio/vowel.

    This keeps the sample balanced across speakers and vowels.
    """
    df = df.copy()
    df["_row_in_audio_vowel"] = df.groupby(["audio", "vowel"]).cumcount()
    return df[df["_row_in_audio_vowel"] % 2 == 0].copy()


def make_speaker_style_table(df: pd.DataFrame) -> pd.DataFrame:
    audios = sorted(df["audio"].dropna().unique())

    rows = []

    for idx, audio in enumerate(audios):
        rows.append({
            "speaker_code": idx + 1,
            "audio": audio,
            "color": SPEAKER_COLORS[idx % len(SPEAKER_COLORS)],
            "marker": SPEAKER_MARKERS[(idx // len(SPEAKER_COLORS)) % len(SPEAKER_MARKERS)],
        })

    return pd.DataFrame(rows)


def make_hover_text(df: pd.DataFrame) -> list[str]:
    bits = []

    optional_cols = [c for c in ["word", "time", "dur", "token_id"] if c in df.columns]

    for _, row in df.iterrows():
        text = [
            f"audio: {row['audio']}",
            f"vowel: /{row['vowel']}/",
            f"F1: {row['f1']:.1f} Hz",
            f"F2: {row['f2']:.1f} Hz",
            f"F3: {row['f3']:.1f} Hz",
        ]

        for col in optional_cols:
            val = row.get(col)
            if pd.notna(val):
                text.append(f"{col}: {val}")

        bits.append("<br>".join(text))

    return bits


def update_scene(fig: go.Figure, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1100,
        height=850,
        legend_title_text="Groupe",
        scene=dict(
            xaxis=dict(title="F2 (Hz)", autorange="reversed"),
            yaxis=dict(title="F1 (Hz)", autorange="reversed"),
            zaxis=dict(title="F3 (Hz)"),
            camera=dict(
                eye=dict(x=1.7, y=1.7, z=1.1)
            ),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )


def plot_all_vowels(df: pd.DataFrame, args, output_dir: Path):
    fig = go.Figure()

    for vowel in args.vowels:
        d = df[df["vowel"] == vowel].copy()

        if d.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=d["f2"],
                y=d["f1"],
                z=d["f3"],
                mode="markers",
                name=f"/{vowel}/",
                marker=dict(
                    size=args.point_size,
                    color=VOWEL_COLORS.get(vowel, "rgba(80,80,80,0.5)"),
                    symbol="circle",
                ),
                text=make_hover_text(d),
                hovertemplate="%{text}<extra></extra>",
            )
        )

    update_scene(fig, "F1 × F2 × F3 — tous les tokens, toutes les voyelles")

    fig.write_html(
        output_dir / "all_vowels_f1f2f3_3d.html",
        include_plotlyjs="cdn",
        full_html=True,
    )


def plot_one_vowel(df: pd.DataFrame, vowel: str, speaker_styles: pd.DataFrame, args, output_dir: Path):
    d0 = df[df["vowel"] == vowel].copy()

    if d0.empty:
        return

    d0 = d0.merge(speaker_styles, on="audio", how="left")

    fig = go.Figure()

    for _, style in speaker_styles.iterrows():
        d = d0[d0["audio"] == style["audio"]].copy()

        if d.empty:
            continue

        speaker_code = int(style["speaker_code"])
        audio = style["audio"]

        fig.add_trace(
            go.Scatter3d(
                x=d["f2"],
                y=d["f1"],
                z=d["f3"],
                mode="markers",
                name=f"{speaker_code:02d} | {audio}",
                marker=dict(
                    size=args.point_size,
                    color=style["color"],
                    symbol=style["marker"],
                ),
                text=make_hover_text(d),
                hovertemplate="%{text}<extra></extra>",
                visible=True,
            )
        )

    update_scene(fig, f"F1 × F2 × F3 — tous les tokens de /{vowel}/ par locuteur")

    fig.write_html(
        output_dir / f"vowel_{vowel}_f1f2f3_3d.html",
        include_plotlyjs="cdn",
        full_html=True,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--label-map", nargs="*", default=["y:i"])

    parser.add_argument("--vowel-col", default=None)
    parser.add_argument("--f1-col", default=None)
    parser.add_argument("--f2-col", default=None)
    parser.add_argument("--f3-col", default=None)

    parser.add_argument(
        "--no-alternating-half",
        action="store_true",
        help="Use all tokens instead of keeping one token out of two.",
    )

    parser.add_argument(
        "--point-size",
        type=float,
        default=2.2,
        help="Marker size in Plotly 3D.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_df = load_points(args)
    full_df.to_csv(output_dir / "all_points_f1f2f3_full.csv", index=False)

    if args.no_alternating_half:
        plot_df = full_df.copy()
    else:
        plot_df = take_alternating_half(full_df)

    plot_df.to_csv(output_dir / "all_points_f1f2f3_plot_half.csv", index=False)

    counts = (
        full_df.groupby(["audio", "vowel"])
        .size()
        .reset_index(name="n_full_tokens")
        .merge(
            plot_df.groupby(["audio", "vowel"])
            .size()
            .reset_index(name="n_plot_tokens"),
            on=["audio", "vowel"],
            how="left",
        )
        .fillna({"n_plot_tokens": 0})
    )

    counts.to_csv(output_dir / "counts_by_audio_vowel.csv", index=False)

    speaker_styles = make_speaker_style_table(full_df)
    speaker_styles.to_csv(output_dir / "speaker_style_mapping.csv", index=False)

    for vowel in args.vowels:
        plot_df[plot_df["vowel"] == vowel].to_csv(
            output_dir / f"plot_half_vowel_{vowel}.csv",
            index=False,
        )

    plot_all_vowels(plot_df, args, output_dir)

    for vowel in args.vowels:
        plot_one_vowel(plot_df, vowel, speaker_styles, args, output_dir)

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print("")
    print("Full rows:", len(full_df))
    print("Rows used for plotting:", len(plot_df))
    print("Audios:", full_df["audio"].nunique())
    print("")
    print("Vowel counts in plot data:")
    print(plot_df["vowel"].value_counts().reindex(args.vowels))
    print("")
    print("Open the HTML files with your browser.")


if __name__ == "__main__":
    main()
