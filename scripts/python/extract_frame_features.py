from pathlib import Path
import argparse
import math

import numpy as np
import pandas as pd
import parselmouth
from praatio import textgrid


def detect_tier_name(tg, candidates):
    tier_names = list(tg.tierNames)

    lowered = {
        name.lower(): name
        for name in tier_names
    }

    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]

    for name in tier_names:
        lname = name.lower()

        for candidate in candidates:
            if candidate.lower() in lname:
                return name

    return None


def interval_to_tuple(interval):
    if hasattr(interval, "start"):
        return interval.start, interval.end, interval.label

    return interval[0], interval[1], interval[2]


def get_entries(tg, tier_name):
    if tier_name is None:
        return []

    tier = tg.getTier(tier_name)

    entries = [
        interval_to_tuple(entry)
        for entry in tier.entries
    ]

    entries.sort(key=lambda x: (x[0], x[1]))

    return entries


def find_interval_index(entries, time_value, current_index):
    if not entries:
        return None

    index = current_index

    if index is None:
        index = 0

    while (
        index < len(entries)
        and time_value >= entries[index][1]
    ):
        index += 1

    if index >= len(entries):
        return None

    start, end, label = entries[index]

    if start <= time_value < end:
        return index

    return None


def safe_formant_value(formant, number, time_value):
    try:
        value = formant.get_value_at_time(
            number,
            time_value,
        )

        if value is None or math.isnan(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def safe_bandwidth_value(formant, number, time_value):
    try:
        value = formant.get_bandwidth_at_time(
            number,
            time_value,
        )

        if value is None or math.isnan(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def safe_intensity_value(intensity, time_value):
    try:
        value = intensity.get_value(
            time_value,
        )

        if value is None or math.isnan(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def safe_pitch_value(pitch, time_value):
    try:
        value = pitch.get_value_at_time(
            time_value,
        )

        if value is None or math.isnan(value):
            return np.nan

        return value

    except Exception:
        return np.nan


def get_context(entries, index):
    if index is None:
        return "", "", "", ""

    current_start, current_end, current_label = entries[index]

    pre_label = ""
    fol_label = ""

    if index > 0:
        pre_label = entries[index - 1][2]

    if index < len(entries) - 1:
        fol_label = entries[index + 1][2]

    return (
        current_label,
        pre_label,
        fol_label,
        current_end - current_start,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract frame-level formants, bandwidths, "
            "intensity, f0 and TextGrid labels."
        )
    )

    parser.add_argument(
        "--audio",
        required=True,
        help="Path to WAV file.",
    )

    parser.add_argument(
        "--textgrid",
        required=True,
        help="Path to TextGrid file.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--step",
        type=float,
        default=0.02,
        help="Time step in seconds. Default: 0.02.",
    )

    parser.add_argument(
        "--max-formant",
        type=float,
        default=5500,
        help="Maximum formant in Hz. Default: 5500.",
    )

    parser.add_argument(
        "--n-formants",
        type=int,
        default=5,
        help="Number of formants estimated by Praat. Default: 5.",
    )

    parser.add_argument(
        "--window-length",
        type=float,
        default=0.025,
        help="Formant analysis window length. Default: 0.025.",
    )

    parser.add_argument(
        "--pitch-floor",
        type=float,
        default=75,
        help="Pitch floor in Hz. Default: 75.",
    )

    parser.add_argument(
        "--pitch-ceiling",
        type=float,
        default=600,
        help="Pitch ceiling in Hz. Default: 600.",
    )

    args = parser.parse_args()

    audio_path = Path(args.audio)
    textgrid_path = Path(args.textgrid)
    output_path = Path(args.output)

    print(f"Audio: {audio_path}")
    print(f"TextGrid: {textgrid_path}")
    print(f"Output: {output_path}")
    print(f"Step: {args.step}")

    sound = parselmouth.Sound(str(audio_path))

    tg = textgrid.openTextgrid(
        str(textgrid_path),
        includeEmptyIntervals=True,
    )

    phone_tier_name = detect_tier_name(
        tg,
        [
            "phone",
            "phones",
            "phoneme",
            "phonemes",
            "segment",
            "segments",
        ],
    )

    word_tier_name = detect_tier_name(
        tg,
        [
            "word",
            "words",
            "transcription",
        ],
    )

    print(f"Detected phone tier: {phone_tier_name}")
    print(f"Detected word tier: {word_tier_name}")

    phone_entries = get_entries(
        tg,
        phone_tier_name,
    )

    word_entries = get_entries(
        tg,
        word_tier_name,
    )

    duration = sound.get_total_duration()

    print(f"Duration: {duration:.3f} seconds")
    print(f"Phone intervals: {len(phone_entries):,}")
    print(f"Word intervals: {len(word_entries):,}")


    #Pegando os valores dos formantes de fato
    formant = sound.to_formant_burg(
        time_step=args.step,
        max_number_of_formants=args.n_formants,
        maximum_formant=args.max_formant,
        window_length=args.window_length,
    )

    intensity = sound.to_intensity(
        time_step=args.step,
    )

    pitch = sound.to_pitch(
        time_step=args.step,
        pitch_floor=args.pitch_floor,
        pitch_ceiling=args.pitch_ceiling,
    )

    times = np.arange(
        0,
        duration,
        args.step,
    )

    rows = []

    phone_index = 0
    word_index = 0

    for counter, time_value in enumerate(times, start=1):
        phone_index = find_interval_index(
            phone_entries,
            time_value,
            phone_index,
        )

        word_index = find_interval_index(
            word_entries,
            time_value,
            word_index,
        )

        phone, pre_seg, fol_seg, phone_dur = get_context(
            phone_entries,
            phone_index,
        )

        word, pre_word, fol_word, word_dur = get_context(
            word_entries,
            word_index,
        )

        row = {
            "time": time_value,
            "phone": phone,
            "word": word,
            "F1": safe_formant_value(
                formant,
                1,
                time_value,
            ),
            "F2": safe_formant_value(
                formant,
                2,
                time_value,
            ),
            "F3": safe_formant_value(
                formant,
                3,
                time_value,
            ),
            "B1": safe_bandwidth_value(
                formant,
                1,
                time_value,
            ),
            "B2": safe_bandwidth_value(
                formant,
                2,
                time_value,
            ),
            "B3": safe_bandwidth_value(
                formant,
                3,
                time_value,
            ),
            "intensity": safe_intensity_value(
                intensity,
                time_value,
            ),
            "f0": safe_pitch_value(
                pitch,
                time_value,
            ),
            "pre_seg": pre_seg,
            "fol_seg": fol_seg,
            "pre_word": pre_word,
            "fol_word": fol_word,
            "phone_dur": phone_dur,
            "word_dur": word_dur,
            "max_formant": args.max_formant,
            "step": args.step,
            "file_name": audio_path.stem,
        }

        rows.append(row)

        if counter % 10000 == 0:
            print(
                f"Processed {counter:,} frames "
                f"up to {time_value:.2f}s"
            )

    data = pd.DataFrame(rows)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"CSV created: {output_path}")
    print(f"Rows: {len(data):,}")
    print(
        f"Time range: "
        f"{data['time'].min():.3f}-"
        f"{data['time'].max():.3f}s"
    )


if __name__ == "__main__":
    main()