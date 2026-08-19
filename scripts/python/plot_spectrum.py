from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import pandas as pd

from correct_spectrum_csv import correct_spectrum_csv


PathLike = Union[str, Path]


def plot_spectrum(
    csv_path: PathLike,
    title: str = "Spectrum Magnitude and Phase"
) -> tuple:
    """
    Standardizes a spectrum CSV and plots magnitude and phase
    in two vertically stacked graphs within the same figure.

    Parameters
    ----------
    csv_path:
        Path to the spectrum CSV file.

    title:
        Main title of the figure.

    Returns
    -------
    tuple
        Figure, axes, and standardized spectrum DataFrame.
    """

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Spectrum file not found: {csv_path}"
        )

    # Standardize the CSV and overwrite the source file
    spectrum = correct_spectrum_csv(
        input_path=csv_path,
        output_path=csv_path
    )

    # Create one figure with two vertically stacked graphs
    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 9),
        sharex=True
    )

    magnitude_axis = axes[0]
    phase_axis = axes[1]

    # -----------------------------------------------------
    # Magnitude graph
    # -----------------------------------------------------

    magnitude_axis.plot(
        spectrum["frequency_hz"],
        spectrum["magnitude"]
    )

    magnitude_axis.set_title(
        "Magnitude"
    )

    magnitude_axis.set_ylabel(
        "Magnitude"
    )

    magnitude_axis.grid(
        True,
        alpha=0.3
    )

    # -----------------------------------------------------
    # Phase graph
    # -----------------------------------------------------

    phase_axis.plot(
        spectrum["frequency_hz"],
        spectrum["phase_deg"]
    )

    phase_axis.set_title(
        "Phase"
    )

    phase_axis.set_xlabel(
        "Frequency (Hz)"
    )

    phase_axis.set_ylabel(
        "Phase (degrees)"
    )

    phase_axis.set_ylim(
        -180,
        180
    )

    phase_axis.grid(
        True,
        alpha=0.3
    )

    # Main title for the complete figure
    figure.suptitle(
        title,
        fontsize=14
    )

    # Adjust spacing between graphs
    figure.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    return (
        figure,
        axes,
        spectrum
    )