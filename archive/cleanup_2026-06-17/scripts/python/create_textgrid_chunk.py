from pathlib import Path
import argparse

from praatio import textgrid


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a time-aligned TextGrid chunk."
    )

    parser.add_argument("input_textgrid", type=Path)
    parser.add_argument("output_textgrid", type=Path)
    parser.add_argument("start", type=float)
    parser.add_argument("end", type=float)

    args = parser.parse_args()

    if not args.input_textgrid.exists():
        raise FileNotFoundError(args.input_textgrid)

    if args.end <= args.start:
        raise ValueError("End time must be greater than start time.")

    original = textgrid.openTextgrid(
        str(args.input_textgrid),
        includeEmptyIntervals=True,
    )

    chunk = original.crop(
        args.start,
        args.end,
        mode="truncated",
        rebaseToZero=True,
    )

    args.output_textgrid.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    chunk.save(
        str(args.output_textgrid),
        format="long_textgrid",
        includeBlankSpaces=True,
    )

    print(f"Created: {args.output_textgrid}")
    print(f"Original range: {args.start:.2f}–{args.end:.2f} s")
    print(f"Chunk range: 0.00–{args.end - args.start:.2f} s")


if __name__ == "__main__":
    main()