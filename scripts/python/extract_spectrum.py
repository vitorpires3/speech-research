from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import parselmouth

from config import freq_min, freq_max


PathLike = Union[str, Path]


def extract_spectrum(
    audio_path: PathLike,
    center_time: float,
    window_duration: float,
    min_frequency: float = freq_min,
    max_frequency: float = freq_max,
    output_path: Optional[PathLike] = None,
    window_shape: parselmouth.WindowShape = (
        parselmouth.WindowShape.GAUSSIAN1
    ),
    relative_width: float = 1.0,
    convert_to_mono: bool = True,
    normalize_magnitude: bool = True,
    calculate_relative_db: bool = True,
) -> pd.DataFrame:
    """
    Extracts the complex spectrum of a short audio segment.

    The analyzed segment is centered at center_time and has a total
    duration defined by window_duration.

    Parameters
    ----------
    audio_path:
        Path to the WAV or another audio format supported by Praat.

    center_time:
        Center of the analyzed segment, in seconds.

    window_duration:
        Total duration of the analyzed segment, in seconds.
        Example: 0.020 means 20 milliseconds.

    min_frequency:
        Minimum frequency exported to the result, in Hz.

    max_frequency:
        Maximum frequency exported to the result, in Hz.

    output_path:
        Optional path for saving the resulting CSV.
        When None, no file is saved.

    window_shape:
        Window applied to the extracted audio segment.

    relative_width:
        Relative width of the Praat window.

    convert_to_mono:
        Converts stereo audio to mono before analysis.

    normalize_magnitude:
        Adds magnitude_normalized, with the largest magnitude equal to 1.

    calculate_relative_db:
        Adds magnitude_relative_db, with the largest magnitude equal to 0 dB.

    Returns
    -------
    pandas.DataFrame
        Table containing frequency, real part, imaginary part,
        magnitude and phase.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    if window_duration <= 0:
        raise ValueError(
            "window_duration must be greater than zero."
        )

    if center_time < 0:
        raise ValueError(
            "center_time cannot be negative."
        )

    if min_frequency < 0:
        raise ValueError(
            "min_frequency cannot be negative."
        )

    if max_frequency <= min_frequency:
        raise ValueError(
            "max_frequency must be greater than min_frequency."
        )

    sound = parselmouth.Sound(
        str(audio_path)
    )

    if convert_to_mono and sound.n_channels > 1:
        sound = sound.convert_to_mono()

    start_time = (
        center_time
        - window_duration / 2
    )

    end_time = (
        center_time
        + window_duration / 2
    )

    if start_time < sound.xmin:
        raise ValueError(
            "The analysis window starts before the audio begins. "
            f"Requested start: {start_time:.6f} s. "
            f"Audio starts at: {sound.xmin:.6f} s."
        )

    if end_time > sound.xmax:
        raise ValueError(
            "The analysis window ends after the audio finishes. "
            f"Requested end: {end_time:.6f} s. "
            f"Audio ends at: {sound.xmax:.6f} s."
        )

    segment = sound.extract_part(
        from_time=start_time,
        to_time=end_time,
        window_shape=window_shape,
        relative_width=relative_width,
        preserve_times=False,
    )

    spectrum = segment.to_spectrum(
        fast=True
    )

    frequencies = spectrum.xs()

    real_values = spectrum.values[0]

    imaginary_values = spectrum.values[1]

    dataframe = pd.DataFrame(
        {
            "frequency_hz": frequencies,
            "real": real_values,
            "imaginary": imaginary_values,
        }
    )

    frequency_filter = (
        (dataframe["frequency_hz"] >= min_frequency)
        & (dataframe["frequency_hz"] <= max_frequency)
    )

    dataframe = dataframe.loc[
        frequency_filter
    ].copy()

    dataframe["magnitude"] = np.hypot(
        dataframe["real"],
        dataframe["imaginary"],
    )

    dataframe["phase_rad"] = np.arctan2(
        dataframe["imaginary"],
        dataframe["real"],
    )

    dataframe["phase_deg"] = np.degrees(
        dataframe["phase_rad"]
    )

    if normalize_magnitude:
        maximum_magnitude = dataframe[
            "magnitude"
        ].max()

        if maximum_magnitude > 0:
            dataframe["magnitude_normalized"] = (
                dataframe["magnitude"]
                / maximum_magnitude
            )
        else:
            dataframe["magnitude_normalized"] = 0.0

    if calculate_relative_db:
        if "magnitude_normalized" not in dataframe.columns:
            maximum_magnitude = dataframe[
                "magnitude"
            ].max()

            if maximum_magnitude > 0:
                normalized_magnitude = (
                    dataframe["magnitude"]
                    / maximum_magnitude
                )
            else:
                normalized_magnitude = pd.Series(
                    0.0,
                    index=dataframe.index,
                )
        else:
            normalized_magnitude = dataframe[
                "magnitude_normalized"
            ]

        safe_magnitude = np.maximum(
            normalized_magnitude,
            np.finfo(float).tiny,
        )

        dataframe["magnitude_relative_db"] = (
            20
            * np.log10(safe_magnitude)
        )

    dataframe.insert(
        0,
        "center_time_s",
        center_time,
    )

    dataframe.insert(
        1,
        "window_start_s",
        start_time,
    )

    dataframe.insert(
        2,
        "window_end_s",
        end_time,
    )

    dataframe.insert(
        3,
        "window_duration_s",
        window_duration,
    )

    dataframe.reset_index(
        drop=True,
        inplace=True,
    )

    if output_path is not None:
        output_path = Path(output_path)

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            output_path,
            index=False,
        )

    return dataframe


def extract_spectrum_interval(
    audio_path,
    start_time,
    end_time,
    min_frequency=freq_min,
    max_frequency=freq_max,
    output_path=None
):
    """
    Extracts one spectrum from an entire time interval.
    """

    if end_time <= start_time:
        raise ValueError(
            "end_time must be greater than start_time."
        )

    center_time = (
        start_time + end_time
    ) / 2

    interval_duration = (
        end_time - start_time
    )

    return extract_spectrum(
        audio_path=audio_path,
        center_time=center_time,
        window_duration=interval_duration,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        output_path=output_path
    )