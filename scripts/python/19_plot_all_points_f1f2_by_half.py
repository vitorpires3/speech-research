#!/usr/bin/env python3
"""
Generate token-level F1-F2 plots separately for H1 and H2.

It reuses:
- scripts/python/12_split_points_by_audio_half.py
- scripts/python/18_plot_all_points_f1f2.py
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str]):
    print("")
    print("Running:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--points-dir", default="data/processed/new_fave_points")
    parser.add_argument("--audio-dir", default="data/raw/sound")

    parser.add_argument(
        "--split-output-dir",
        default="data/processed/all_points_halves_for_plots",
    )

    parser.add_argument(
        "--plot-output-root",
        default="results/plots/all_points_f1f2_by_half",
    )

    parser.add_argument("--vowels", nargs="+", default=["a", "e", "i", "o", "u"])
    parser.add_argument("--label-map", nargs="*", default=["y:i"])

    parser.add_argument("--skip-split", action="store_true")

    parser.add_argument("--color-by-audio", action="store_true")
    parser.add_argument("--label-speaker-centers", action="store_true")
    parser.add_argument("--speaker-style-grid", action="store_true")

    parser.add_argument("--max-points-per-audio-vowel", default=None)
    parser.add_argument("--point-size", default="5")
    parser.add_argument("--alpha", default="0.16")

    args = parser.parse_args()

    split_output_dir = Path(args.split_output_dir)
    plot_output_root = Path(args.plot_output_root)

    split_script = Path("scripts/python/12_split_points_by_audio_half.py")
    plot_script = Path("scripts/python/18_plot_all_points_f1f2.py")

    if not split_script.exists():
        raise FileNotFoundError(f"Missing split script: {split_script}")

    if not plot_script.exists():
        raise FileNotFoundError(f"Missing plot script: {plot_script}")

    if not args.skip_split:
        run([
            "python3",
            str(split_script),
            "--points-dir",
            args.points_dir,
            "--audio-dir",
            args.audio_dir,
            "--output-dir",
            str(split_output_dir),
        ])

    for half in ["H1", "H2"]:
        half_input_dir = split_output_dir / half
        half_output_dir = plot_output_root / half

        if not half_input_dir.exists():
            raise FileNotFoundError(
                f"Expected half directory not found: {half_input_dir}. "
                "Check whether script 12 generated H1/ and H2/ folders."
            )

        cmd = [
            "python3",
            str(plot_script),
            "--input-dir",
            str(half_input_dir),
            "--output-dir",
            str(half_output_dir),
            "--vowels",
            *args.vowels,
        ]

        if args.label_map:
            cmd.extend(["--label-map", *args.label_map])

        cmd.extend([
            "--point-size",
            str(args.point_size),
            "--alpha",
            str(args.alpha),
        ])

        if args.color_by_audio:
            cmd.append("--color-by-audio")

        if args.label_speaker_centers:
            cmd.append("--label-speaker-centers")

        if args.speaker_style_grid:
            cmd.append("--speaker-style-grid")

        if args.max_points_per_audio_vowel is not None:
            cmd.extend([
                "--max-points-per-audio-vowel",
                str(args.max_points_per_audio_vowel),
            ])

        run(cmd)

    print("")
    print("Done.")
    print(f"H1 outputs: {plot_output_root / 'H1'}")
    print(f"H2 outputs: {plot_output_root / 'H2'}")
    print("")
    print("Open with:")
    print(f"xdg-open {plot_output_root}")


if __name__ == "__main__":
    main()
