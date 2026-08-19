from pathlib import Path

import matplotlib.pyplot as plt

from plot_spectrum import plot_spectrum
from extract_spectrum import (
    extract_spectrum,
    extract_spectrum_interval,
)

from config import freq_min, freq_max

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

project_root = Path(__file__).resolve().parents[2]

audio_file = (
    project_root
    / "data"
    / "raw"
    / "sound"
    / "0e6115f2-Audio_SP19.v0.wav"
)

output_directory = (
    project_root
    / "data"
    / "processed"
    / "spectrum_test"
)

output_directory.mkdir(
    parents=True,
    exist_ok=True
)


# ---------------------------------------------------------
# Test parameters
# ---------------------------------------------------------

min_frequency = freq_min
max_frequency = freq_max

point_time = 54.36

# Local window centered at 54.36 seconds
point_window_duration = 0.010

interval_start = 54.24
interval_end = 54.58


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------

point_csv = (
    output_directory
    / "SP19_spectrum_at_54_360s.csv"
)

interval_csv = (
    output_directory
    / "SP19_spectrum_54_240s_to_54_580s.csv"
)


# ---------------------------------------------------------
# Check input audio
# ---------------------------------------------------------

if not audio_file.exists():
    raise FileNotFoundError(
        f"Audio file not found: {audio_file}"
    )


# ---------------------------------------------------------
# Extract local spectrum at 54.36 seconds
# ---------------------------------------------------------

point_spectrum = extract_spectrum(
    audio_path=audio_file,
    center_time=point_time,
    window_duration=point_window_duration,
    min_frequency=min_frequency,
    max_frequency=max_frequency,
    output_path=point_csv
)


# ---------------------------------------------------------
# Extract spectrum from the entire interval
# ---------------------------------------------------------

interval_spectrum = extract_spectrum_interval(
    audio_path=audio_file,
    start_time=interval_start,
    end_time=interval_end,
    min_frequency=min_frequency,
    max_frequency=max_frequency,
    output_path=interval_csv
)


# ---------------------------------------------------------
# Create both graphs
# ---------------------------------------------------------

plot_spectrum(
    csv_path=point_csv,
    title=(
        "Local Spectrum at 54.36 s "
        "(10 ms Window)"
    )
)

plot_spectrum(
    csv_path=interval_csv,
    title=(
        "Spectrum of the Complete Interval "
        "54.24–54.58 s"
    )
)


# ---------------------------------------------------------
# Execution information
# ---------------------------------------------------------

print("Spectrum extraction completed.")
print()
print(f"Audio file: {audio_file}")
print()
print("Local spectrum:")
print(f"  Center time: {point_time:.3f} s")
print(
    f"  Window duration: "
    f"{point_window_duration * 1000:.1f} ms"
)
print(f"  CSV: {point_csv}")
print(f"  Frequency bins: {len(point_spectrum)}")
print()
print("Complete interval spectrum:")
print(f"  Start time: {interval_start:.3f} s")
print(f"  End time: {interval_end:.3f} s")
print(
    f"  Duration: "
    f"{interval_end - interval_start:.3f} s"
)
print(f"  CSV: {interval_csv}")
print(f"  Frequency bins: {len(interval_spectrum)}")


# Display both graph windows
plt.show()