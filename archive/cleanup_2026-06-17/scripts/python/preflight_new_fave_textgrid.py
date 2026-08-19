from __future__ import annotations

import argparse
from pathlib import Path

from aligned_textgrid import AlignedTextGrid, Phone, Word


VOWEL_LABELS = {
    "a", "e", "i", "o", "u", "y",
    "á", "é", "í", "ó", "ú", "ý",
    "A", "E", "I", "O", "U", "Y",
    "Á", "É", "Í", "Ó", "Ú", "Ý",
}


def describe_interval(interval: object) -> str:
    return (
        f"type={type(interval).__name__}, "
        f"label={getattr(interval, 'label', None)!r}, "
        f"start={getattr(interval, 'start', None)}, "
        f"end={getattr(interval, 'end', None)}"
    )


def validate_within_chain(interval: object) -> tuple[bool, list[str]]:
    """
    Reproduces the important part of FastTrack's hierarchy lookup:

        phone -> word -> tier/group

    It fails if the chain reaches None before reaching an object
    representing the tier group.
    """
    chain: list[str] = []
    current = interval
    visited: set[int] = set()

    while current is not None:
        current_id = id(current)

        if current_id in visited:
            chain.append("CYCLE DETECTED")
            return False, chain

        visited.add(current_id)
        chain.append(describe_interval(current))

        current_type = type(current).__name__

        if current_type == "TierGroup":
            return True, chain

        parent = getattr(current, "within", None)

        # Depending on the aligned_textgrid version, the hierarchy may
        # reach an object whose type name contains TierGroup.
        if parent is not None and "TierGroup" in type(parent).__name__:
            chain.append(describe_interval(parent))
            return True, chain

        current = parent

    chain.append("None")
    return False, chain


def validate_textgrid(path: Path) -> int:
    print(f"TextGrid: {path}")

    grid = AlignedTextGrid(
        textgrid_path=str(path),
        entry_classes=[Word, Phone],
    )

    if len(grid) == 0:
        print("ERROR: no tier groups found.")
        return 1

    group = grid[0]

    words = list(group.Word)
    phones = list(group.Phone)

    print(f"Words: {len(words):,}")
    print(f"Phones: {len(phones):,}")

    errors: list[str] = []
    warnings: list[str] = []

    # 1. Basic interval validation.
    for tier_name, intervals in [
        ("words", words),
        ("phones", phones),
    ]:
        previous = None

        for index, interval in enumerate(intervals):
            if interval.start >= interval.end:
                errors.append(
                    f"INVALID {tier_name.upper()} INTERVAL\n"
                    f"  index={index}\n"
                    f"  {describe_interval(interval)}"
                )

            if previous is not None and interval.start < previous.end:
                errors.append(
                    f"OVERLAP IN {tier_name.upper()}\n"
                    f"  previous index={index - 1}: "
                    f"{describe_interval(previous)}\n"
                    f"  current index={index}: "
                    f"{describe_interval(interval)}\n"
                    f"  overlap={previous.end - interval.start:.6f}s"
                )

            if previous is None or interval.end > previous.end:
                previous = interval

    # 2. Check only non-empty vowel intervals, because those are the
    # intervals relevant to the new-fave vowel extraction.
    target_phones = [
        phone
        for phone in phones
        if str(phone.label).strip() in VOWEL_LABELS
    ]

    print(f"Target vowel phones: {len(target_phones):,}")

    for index, phone in enumerate(target_phones):
        valid, chain = validate_within_chain(phone)

        if not valid:
            errors.append(
                "BROKEN HIERARCHY FOR VOWEL PHONE\n"
                f"  target index={index}\n"
                f"  {describe_interval(phone)}\n"
                + "\n".join(
                    f"  level {level}: {item}"
                    for level, item in enumerate(chain)
                )
            )

            # Ten examples are enough for diagnosis.
            if sum(
                error.startswith(
                    "BROKEN HIERARCHY FOR VOWEL PHONE"
                )
                for error in errors
            ) >= 10:
                break

    # 3. Check whether words and their first/last phones share boundaries.
    mismatch_count = 0
    largest_mismatch = 0.0

    for word in words:
        contained = list(getattr(word, "contains", []))

        if not contained:
            if str(word.label).strip():
                warnings.append(
                    "NON-EMPTY WORD WITHOUT PHONES\n"
                    f"  {describe_interval(word)}"
                )
            continue

        first_phone = contained[0]
        last_phone = contained[-1]

        start_diff = abs(first_phone.start - word.start)
        end_diff = abs(last_phone.end - word.end)

        for difference in (start_diff, end_diff):
            if difference > 0.0005:
                mismatch_count += 1
                largest_mismatch = max(
                    largest_mismatch,
                    difference,
                )

    print(f"Boundary mismatches: {mismatch_count}")
    print(f"Largest mismatch: {largest_mismatch:.6f}s")

    if mismatch_count:
        warnings.append(
            f"{mismatch_count} word/phone boundary mismatches; "
            f"largest={largest_mismatch:.6f}s"
        )

    print()

    if errors:
        print(f"FAILED: {len(errors)} structural error(s).")

        for number, error in enumerate(errors, start=1):
            print()
            print(f"[ERROR {number}]")
            print(error)

        print()
        print("Do not run new-fave for this TextGrid.")
        return 1

    print("PASSED: no blocking structural errors detected.")

    if warnings:
        print()
        print("Warnings:")

        for warning in warnings[:20]:
            print(f"- {warning}")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a TextGrid before running new-fave. "
            "This does not process the audio."
        )
    )

    parser.add_argument(
        "textgrid",
        type=Path,
        help="Path to the UTF-8 TextGrid.",
    )

    args = parser.parse_args()

    if not args.textgrid.exists():
        raise FileNotFoundError(args.textgrid)

    raise SystemExit(validate_textgrid(args.textgrid))


if __name__ == "__main__":
    main()