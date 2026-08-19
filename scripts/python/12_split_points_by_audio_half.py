#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import pandas as pd
import numpy as np


TIME_CANDIDATES = [
    "time",
    "point",
    "point_time",
    "measurement_time",
    "midpoint",
    "mid_point",
    "vowel_time",
    "t",
]

START_CANDIDATES = [
    "beg",
    "begin",
    "start",
    "xmin",
    "start_time",
    "t1",
]

END_CANDIDATES = [
    "end",
    "stop",
    "xmax",
    "end_time",
    "t2",
]


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


def clean_audio_id(name):
    """
    Normaliza nomes como:
    0e6115f2-Audio_SP19.v0.wav
    0e6115f2-Audio_SP19_points.csv
    0e6115f2-Audio_SP19
    """

    s = str(name).strip()
    s = Path(s).name

    for ext in [".wav", ".flac", ".mp3", ".csv", ".TextGrid", ".textgrid", ".seg"]:
        if s.endswith(ext):
            s = s[: -len(ext)]

    s = re.sub(r"\.v[0-9]+$", "", s)

    suffixes = [
        "_points",
        "-points",
        "_point",
        "-point",
        "_tracks",
        "-tracks",
        "_track",
        "-track",
    ]

    for suffix in suffixes:
        if s.lower().endswith(suffix):
            s = s[: -len(suffix)]

    return s


def norm_audio_key(name):
    return clean_audio_id(name).lower()


def get_audio_duration(audio_path):
    """
    Tenta obter a duração real do áudio usando soundfile.
    Se soundfile não estiver disponível ou falhar, retorna None.
    """

    try:
        import soundfile as sf

        info = sf.info(str(audio_path))
        return float(info.frames) / float(info.samplerate)

    except Exception:
        return None


def find_matching_audio(points_file, audio_dir):
    """
    Procura um wav/flac/mp3 com o mesmo id base do arquivo de points.
    """

    if audio_dir is None:
        return None

    audio_dir = Path(audio_dir)

    if not audio_dir.exists():
        return None

    target_key = norm_audio_key(points_file.stem)

    audio_extensions = ["*.wav", "*.flac", "*.mp3"]

    for pattern in audio_extensions:
        for audio_path in audio_dir.glob(pattern):
            if norm_audio_key(audio_path.stem) == target_key:
                return audio_path

    return None


def get_token_time(df):
    """
    Pega o tempo do token.

    Prioridade:
    1. coluna explícita de tempo: time, point, midpoint etc.
    2. média entre começo e fim: (beg + end) / 2
    """

    time_col = find_col(df, TIME_CANDIDATES)

    if time_col is not None:
        token_time = pd.to_numeric(df[time_col], errors="coerce")
        return token_time, f"time column: {time_col}"

    start_col = find_col(df, START_CANDIDATES)
    end_col = find_col(df, END_CANDIDATES)

    if start_col is not None and end_col is not None:
        start = pd.to_numeric(df[start_col], errors="coerce")
        end = pd.to_numeric(df[end_col], errors="coerce")
        token_time = (start + end) / 2
        return token_time, f"midpoint from columns: {start_col}, {end_col}"

    raise ValueError(
        "Não encontrei coluna de tempo nem colunas de começo/fim. "
        f"Colunas disponíveis: {list(df.columns)}"
    )


def process_points_file(points_file, audio_dir, output_dir):
    points_file = Path(points_file)
    output_dir = Path(output_dir)

    df = pd.read_csv(points_file)

    token_time, time_basis = get_token_time(df)

    audio_path = find_matching_audio(points_file, audio_dir)
    audio_duration = None

    if audio_path is not None:
        audio_duration = get_audio_duration(audio_path)

    if audio_duration is not None:
        split_time = audio_duration / 2
        split_basis = "audio_duration"
    else:
        valid_times = token_time.dropna()

        if valid_times.empty:
            raise ValueError(f"Arquivo sem tempos válidos: {points_file}")

        split_time = float(valid_times.max()) / 2
        split_basis = "max_token_time_fallback"

    out = df.copy()

    out["_source_file"] = points_file.name
    out["_audio_id"] = clean_audio_id(points_file.stem)
    out["_token_time"] = token_time
    out["_audio_duration"] = audio_duration
    out["_split_time"] = split_time
    out["_time_basis"] = time_basis
    out["_split_basis"] = split_basis

    out["half"] = np.where(out["_token_time"] <= split_time, "H1", "H2")

    h1 = out[out["half"] == "H1"].copy()
    h2 = out[out["half"] == "H2"].copy()

    output_dir.mkdir(parents=True, exist_ok=True)

    h1_dir = output_dir / "H1"
    h2_dir = output_dir / "H2"

    h1_dir.mkdir(parents=True, exist_ok=True)
    h2_dir.mkdir(parents=True, exist_ok=True)

    h1_path = h1_dir / points_file.name
    h2_path = h2_dir / points_file.name

    h1.to_csv(h1_path, index=False)
    h2.to_csv(h2_path, index=False)

    summary = {
        "source_file": points_file.name,
        "audio_id": clean_audio_id(points_file.stem),
        "audio_path": str(audio_path) if audio_path is not None else "",
        "audio_duration": audio_duration,
        "split_time": split_time,
        "time_basis": time_basis,
        "split_basis": split_basis,
        "total_rows": len(out),
        "H1_rows": len(h1),
        "H2_rows": len(h2),
        "min_token_time": float(token_time.min()) if token_time.notna().any() else np.nan,
        "max_token_time": float(token_time.max()) if token_time.notna().any() else np.nan,
    }

    return out, summary


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--points-dir",
        required=True,
        help="Pasta com os CSVs de points do new-FAVE.",
    )

    parser.add_argument(
        "--audio-dir",
        default="data/raw/sound",
        help="Pasta com os arquivos de áudio. Default: data/raw/sound",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Pasta de saída para os points divididos.",
    )

    parser.add_argument(
        "--audio",
        default=None,
        help="Opcional: processar só um áudio específico, exemplo 0e6115f2-Audio_SP19.",
    )

    args = parser.parse_args()

    points_dir = Path(args.points_dir)
    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    points_files = sorted(points_dir.glob("*.csv"))

    if args.audio is not None:
        target_key = norm_audio_key(args.audio)

        points_files = [
            path
            for path in points_files
            if norm_audio_key(path.stem) == target_key
        ]

    if not points_files:
        raise ValueError("Nenhum arquivo de points encontrado para processar.")

    all_rows = []
    summaries = []

    for points_file in points_files:
        print(f"Processing: {points_file}")

        split_df, summary = process_points_file(
            points_file=points_file,
            audio_dir=audio_dir,
            output_dir=output_dir,
        )

        all_rows.append(split_df)
        summaries.append(summary)

    combined = pd.concat(all_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    combined_path = output_dir / "all_points_with_halves.csv"
    summary_path = output_dir / "split_points_summary.csv"

    combined.to_csv(combined_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print()
    print("=== Split points by audio half ===")
    print(f"Files processed: {len(points_files)}")
    print()
    print("Summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Outputs:")
    print(f"- H1 files: {output_dir / 'H1'}")
    print(f"- H2 files: {output_dir / 'H2'}")
    print(f"- Combined file: {combined_path}")
    print(f"- Summary file: {summary_path}")


if __name__ == "__main__":
    main()
