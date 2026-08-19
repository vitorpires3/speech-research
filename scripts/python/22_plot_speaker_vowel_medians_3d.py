#!/usr/bin/env python3
"""
Plot speaker-level vowel medians in F1-F2-F3 space.

Input:
- data/processed/tables/vowel_stats_speaker_vowel.csv

Expected columns:
- audio
- vowel
- f1_median
- f2_median
- f3_median

Outputs:
- interactive Plotly HTML plots
- CSVs used for plotting

Plot convention:
x = F2
y = F1
z = F3

F1 and F2 are reversed, as in conventional vowel-space plots.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VOWEL_COLORS = {
    "a": "rgba(214, 39, 40, 0.75)",
    "e": "rgba(255, 127, 14, 0.75)",
    "i": "rgba(44, 160, 44, 0.75)",
    "o": "rgba(31, 119, 180, 0.75)",
    "u": "rgba(148, 103, 189, 0.75)",
}

SPEAKER_COLORS = [
    "rgba(31, 119, 180, 0.75)",
    "rgba(255, 127, 14, 0.75)",
    "rgba(44, 160, 44, 0.75)",
    "rgba(214, 39, 40, 0.75)",
    "rgba(148, 103, 189, 0.75)",
    "rgba(140, 86, 75, 0.75)",
    "rgba(227, 119, 194, 0.75)",
]

SPEAKER_MARKERS = [
    "circle", "diamond", "square", "cross", "x",
    "circle-open", "diamond-open"
]


def require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "Plotly is not installed. Install it with:\n\n"
            "pip install plotly\n"
        ) from exc
    return go


def detect_audio_col(df: pd.DataFrame) -> str:
    for col in ["audio", "file", "file_name", "source_file"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find audio column. Columns: {list(df.columns)}")


def make_speaker_style_table(df: pd.DataFrame, audio_col: str) -> pd.DataFrame:
    audios = sorted(df[audio_col].dropna().unique())
    rows = []

    for idx, audio in enumerate(audios):
        rows.append({
            "speaker_code": idx + 1,
            "audio": audio,
            "color": SPEAKER_COLORS[idx % len(SPEAKER_COLORS)],
            "marker": SPEAKER_MARKERS[(idx // len(SPEAKER_COLORS)) % len(SPEAKER_MARKERS)],
        })

    return pd.DataFrame(rows)


def load_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(args.input)

    audio_col = detect_audio_col(df)

    required = {"vowel", "f1_median", "f2_median", "f3_median"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    out = df[[audio_col, "vowel", "f1_median", "f2_median", "f3_median"]].copy()
    out = out.rename(columns={audio_col: "audio"})

    out["vowel"] = out["vowel"].astype(str).str.lower()
    out["vowel"] = out["vowel"].replace({"y": "i"})
    out = out[out["vowel"].isin(args.vowels)].copy()

    for col in ["f1_median", "f2_median", "f3_median"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["audio", "vowel", "f1_median", "f2_median", "f3_median"])
    out = out[
        (out["f1_median"] > 0)
        & (out["f2_median"] > 0)
        & (out["f3_median"] > 0)
    ].copy()

    if out.empty:
        raise RuntimeError("No valid rows left after filtering.")

    speaker_styles = make_speaker_style_table(out, "audio")
    return out, speaker_styles


def hover_text(df: pd.DataFrame) -> list[str]:
    texts = []

    for _, row in df.iterrows():
        texts.append(
            "<br>".join([
                f"audio: {row['audio']}",
                f"vowel: /{row['vowel']}/",
                f"F1 median: {row['f1_median']:.1f} Hz",
                f"F2 median: {row['f2_median']:.1f} Hz",
                f"F3 median: {row['f3_median']:.1f} Hz",
            ])
        )

    return texts


def update_scene(fig, title: str):
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1000,
        height=820,
        legend_title_text="Groupe",
        scene=dict(
            xaxis=dict(title="F2 median (Hz)", autorange="reversed"),
            yaxis=dict(title="F1 median (Hz)", autorange="reversed"),
            zaxis=dict(title="F3 median (Hz)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.1)),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )


def plot_all_vowels(df: pd.DataFrame, args, output_dir: Path):
    go = require_plotly()
    fig = go.Figure()

    for vowel in args.vowels:
        d = df[df["vowel"] == vowel].copy()

        if d.empty:
            continue

        fig.add_trace(
            go.Scatter3d(
                x=d["f2_median"],
                y=d["f1_median"],
                z=d["f3_median"],
                mode="markers+text" if args.show_labels else "markers",
                text=d["audio"] if args.show_labels else hover_text(d),
                hovertext=hover_text(d),
                hovertemplate="%{hovertext}<extra></extra>",
                name=f"/{vowel}/",
                marker=dict(
                    size=args.point_size,
                    color=VOWEL_COLORS.get(vowel, "rgba(80,80,80,0.75)"),
                    symbol="circle",
                ),
            )
        )

    update_scene(fig, "Médianes F1-F2-F3 par locuteur — toutes les voyelles")
    fig.write_html(output_dir / "all_vowels_medians_f1f2f3_3d.html", include_plotlyjs="cdn")


def plot_one_vowel(df: pd.DataFrame, speaker_styles: pd.DataFrame, vowel: str, args, output_dir: Path):
    go = require_plotly()

    d0 = df[df["vowel"] == vowel].copy()

    if d0.empty:
        return

    d0 = d0.merge(speaker_styles, on="audio", how="left")

    fig = go.Figure()

    for _, style in speaker_styles.iterrows():
        d = d0[d0["audio"] == style["audio"]].copy()

        if d.empty:
            continue

        code = int(style["speaker_code"])

        fig.add_trace(
            go.Scatter3d(
                x=d["f2_median"],
                y=d["f1_median"],
                z=d["f3_median"],
                mode="markers+text",
                text=[str(code)] * len(d),
                textposition="top center",
                hovertext=hover_text(d),
                hovertemplate="%{hovertext}<extra></extra>",
                name=f"{code:02d} | {style['audio']}",
                marker=dict(
                    size=args.point_size,
                    color=style["color"],
                    symbol=style["marker"],
                ),
            )
        )

    update_scene(fig, f"Médianes F1-F2-F3 par locuteur — voyelle /{vowel}/")
    fig.write_html(output_dir / f"vowel_{vowel}_medians_f1f2f3_3d.html", include_plotlyjs="cdn")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--point-size", type=float, default=5.5)
    parser.add_argument("--show-labels", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, speaker_styles = load_data(args)

    df.to_csv(output_dir / "speaker_vowel_medians_f1f2f3.csv", index=False)
    speaker_styles.to_csv(output_dir / "speaker_style_mapping.csv", index=False)

    for vowel in args.vowels:
        df[df["vowel"] == vowel].to_csv(
            output_dir / f"speaker_vowel_medians_f1f2f3_vowel_{vowel}.csv",
            index=False,
        )

    plot_all_vowels(df, args, output_dir)

    for vowel in args.vowels:
        plot_one_vowel(df, speaker_styles, vowel, args, output_dir)

    print("")
    print("Done.")
    print(f"Output directory: {output_dir}")
    print("")
    print("Rows:", len(df))
    print("Audios:", df["audio"].nunique())
    print("")
    print("Counts by vowel:")
    print(df["vowel"].value_counts().reindex(args.vowels))


if __name__ == "__main__":
    main()
