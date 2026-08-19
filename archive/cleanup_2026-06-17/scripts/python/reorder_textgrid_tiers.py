from pathlib import Path
import re


INPUT_DIR = Path("data/processed/new_fave_label")
OUTPUT_DIR = Path("data/processed/new_fave_label_fixed")

DESIRED_ORDER = ["words", "phones"]


def get_tier_name(block: str) -> str:
    match = re.search(r'(?m)^\s*name\s*=\s*"([^"]+)"\s*$', block)

    if match is None:
        raise ValueError("Could not identify the tier name.")

    return match.group(1)


def replace_item_number(block: str, item_number: int) -> str:
    return re.sub(
        r'(?m)^(\s*)item\s*\[\d+\]:\s*$',
        rf"\1item [{item_number}]:",
        block,
        count=1,
    )


def reorder_textgrid(input_path: Path, output_path: Path) -> None:
    text = input_path.read_text(encoding="utf-8")

    item_matches = list(
        re.finditer(
            r'(?m)^\s{4}item\s*\[\d+\]:\s*$',
            text,
        )
    )

    if not item_matches:
        raise ValueError("No TextGrid tier blocks were found.")

    header = text[: item_matches[0].start()]
    blocks: dict[str, str] = {}

    for index, match in enumerate(item_matches):
        block_start = match.start()

        if index + 1 < len(item_matches):
            block_end = item_matches[index + 1].start()
        else:
            block_end = len(text)

        block = text[block_start:block_end]
        tier_name = get_tier_name(block)

        if tier_name in blocks:
            raise ValueError(f"Duplicate tier name: {tier_name}")

        blocks[tier_name] = block

    missing_tiers = [
        tier_name
        for tier_name in DESIRED_ORDER
        if tier_name not in blocks
    ]

    if missing_tiers:
        raise ValueError(
            f"Required tiers not found: {', '.join(missing_tiers)}"
        )

    extra_tiers = [
        tier_name
        for tier_name in blocks
        if tier_name not in DESIRED_ORDER
    ]

    if extra_tiers:
        raise ValueError(
            f"Unexpected tiers found: {', '.join(extra_tiers)}"
        )

    ordered_blocks = []

    for item_number, tier_name in enumerate(DESIRED_ORDER, start=1):
        ordered_blocks.append(
            replace_item_number(
                blocks[tier_name],
                item_number,
            )
        )

    output_text = header + "".join(ordered_blocks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")


def main() -> None:
    input_files = sorted(INPUT_DIR.glob("*.TextGrid"))

    if not input_files:
        raise FileNotFoundError(
            f"No TextGrid files found in {INPUT_DIR}"
        )

    success_count = 0

    for input_path in input_files:
        output_path = OUTPUT_DIR / input_path.name

        try:
            reorder_textgrid(input_path, output_path)
            print(f"OK: {input_path.name}")
            success_count += 1

        except Exception as error:
            print(f"ERROR: {input_path.name}: {error}")

    print()
    print(f"Processed successfully: {success_count}")
    print(f"Input files: {len(input_files)}")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()