from pathlib import Path
import re

import pandas as pd


CHUNK_DIR = Path(
    "data/processed/new_fave_chunks/"
    "42d56bbf-Audio_SP23"
)

OUTPUT_FILE = Path(
    "data/processed/new_fave_points/"
    "42d56bbf-Audio_SP23.csv"
)

FILE_PATTERN = re.compile(
    r"42d56bbf-Audio_SP23_part(\d+)_points\.csv$"
)

CHUNK_DURATION = 600.0


def get_part_number(path: Path) -> int:
    match = FILE_PATTERN.fullmatch(path.name)

    if match is None:
        raise ValueError(
            f"Unexpected chunk filename: {path.name}"
        )

    return int(match.group(1))


def main() -> None:
    chunk_files = []

    for path in CHUNK_DIR.glob(
        "42d56bbf-Audio_SP23_part*_points.csv"
    ):
        if FILE_PATTERN.fullmatch(path.name):
            chunk_files.append(path)

    chunk_files.sort(key=get_part_number)

    if not chunk_files:
        raise FileNotFoundError(
            f"No chunk CSV files found in {CHUNK_DIR}"
        )

    print(f"Chunk CSV files found: {len(chunk_files)}")

    expected_parts = list(
        range(
            1,
            get_part_number(chunk_files[-1]) + 1,
        )
    )

    found_parts = [
        get_part_number(path)
        for path in chunk_files
    ]

    if found_parts != expected_parts:
        raise ValueError(
            "Missing or duplicated chunk parts.\n"
            f"Expected: {expected_parts}\n"
            f"Found: {found_parts}"
        )

    dataframes = []
    reference_columns = None

    for path in chunk_files:
        part_number = get_part_number(path)
        offset = (part_number - 1) * CHUNK_DURATION

        data = pd.read_csv(path)

        if data.empty:
            raise ValueError(
                f"CSV is empty: {path.name}"
            )

        if reference_columns is None:
            reference_columns = list(data.columns)

        elif list(data.columns) != reference_columns:
            raise ValueError(
                f"Column mismatch in {path.name}"
            )

        if "time" not in data.columns:
            raise ValueError(
                f"Column 'time' not found in {path.name}"
            )

        data["chunk_part"] = part_number
        data["chunk_offset"] = offset

        # Preserve the time relative to the chunk.
        data["chunk_time"] = data["time"]

        # Restore the time relative to the full recording.
        data["time"] = data["chunk_time"] + offset

        # Preserve the original ID from the chunk.
        if "id" in data.columns:
            data["original_chunk_id"] = (
                data["id"].astype("string")
            )

            data["id"] = (
                f"part{part_number:02d}_"
                + data["original_chunk_id"].fillna("")
            )

        # Preserve the chunk file name and restore the
        # original recording name.
        if "file_name" in data.columns:
            data["chunk_file_name"] = data["file_name"]
            data["file_name"] = (
                "42d56bbf-Audio_SP23.v0"
            )

        dataframes.append(data)

        print(
            f"Loaded part {part_number:02d}: "
            f"{len(data):,} rows | "
            f"offset={offset:.0f}s | "
            f"time={data['time'].min():.3f}-"
            f"{data['time'].max():.3f}s"
        )

    merged = pd.concat(
        dataframes,
        ignore_index=True,
    )

    sort_columns = ["time"]

    if "id" in merged.columns:
        sort_columns.append("id")

    merged = merged.sort_values(
        by=sort_columns,
        kind="stable",
    ).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    print()
    print(f"Merged CSV created: {OUTPUT_FILE}")
    print(f"Total rows: {len(merged):,}")

    if "id" in merged.columns:
        print(
            f"Unique IDs: "
            f"{merged['id'].nunique():,}"
        )

    print(
        f"Parts: "
        f"{sorted(merged['chunk_part'].unique())}"
    )

    print(
        f"Time range: "
        f"{merged['time'].min():.3f}-"
        f"{merged['time'].max():.3f} s"
    )


if __name__ == "__main__":
    main()