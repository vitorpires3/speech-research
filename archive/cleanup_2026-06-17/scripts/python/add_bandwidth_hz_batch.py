from pathlib import Path

import numpy as np
import pandas as pd


INPUT_DIR = Path(
    "data/processed/new_fave_points"
)

REQUIRED_COLUMNS = {
    "B1",
    "B2",
    "B3",
}

OUTPUT_COLUMNS = [
    "B1_Hz",
    "B2_Hz",
    "B3_Hz",
]


def process_csv(csv_path: Path) -> tuple[int, int]:
    data = pd.read_csv(csv_path)

    missing_columns = (
        REQUIRED_COLUMNS - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing columns in {csv_path.name}: "
            f"{sorted(missing_columns)}"
        )

    data["B1_Hz"] = np.exp(
        pd.to_numeric(
            data["B1"],
            errors="coerce",
        )
    )

    data["B2_Hz"] = np.exp(
        pd.to_numeric(
            data["B2"],
            errors="coerce",
        )
    )

    data["B3_Hz"] = np.exp(
        pd.to_numeric(
            data["B3"],
            errors="coerce",
        )
    )

    invalid_rows = data[
        OUTPUT_COLUMNS
    ].isna().any(axis=1).sum()

    data.to_csv(
        csv_path,
        index=False,
        encoding="utf-8",
    )

    return len(data), int(invalid_rows)


def main() -> None:
    csv_files = sorted(
        INPUT_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_DIR}"
        )

    print(
        f"CSV files found: {len(csv_files)}"
    )

    total_rows = 0
    total_invalid = 0
    processed_files = 0
    failed_files = 0

    for csv_path in csv_files:
        try:
            rows, invalid_rows = process_csv(
                csv_path
            )

            processed_files += 1
            total_rows += rows
            total_invalid += invalid_rows

            print(
                f"OK: {csv_path.name} | "
                f"rows={rows:,} | "
                f"invalid bandwidth rows="
                f"{invalid_rows:,}"
            )

        except Exception as error:
            failed_files += 1

            print(
                f"ERROR: {csv_path.name} | "
                f"{type(error).__name__}: "
                f"{error}"
            )

    print()
    print("=" * 70)
    print("FINISHED")
    print("=" * 70)
    print(
        f"Processed files: {processed_files}"
    )
    print(
        f"Failed files: {failed_files}"
    )
    print(
        f"Total rows: {total_rows:,}"
    )
    print(
        f"Rows with invalid bandwidth values: "
        f"{total_invalid:,}"
    )


if __name__ == "__main__":
    main()