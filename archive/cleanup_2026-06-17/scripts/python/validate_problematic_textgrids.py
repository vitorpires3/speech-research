from dataclasses import dataclass
from pathlib import Path
import re


TEXTGRID_DIR = Path("data/processed/new_fave_label")
REPORT_PATH = Path("docs/new_fave_validation_report.txt")

PROBLEM_FILES = [
    "1c403477-Audio_SP13.v0.TextGrid",
    "1d46eb6e-Audio_SP18.v0.TextGrid",
    "2988c6fc-Audio_SP13.v0.TextGrid",
    "42d56bbf-Audio_SP23.v0.TextGrid",
    "49ae7020-Audio_SP17.v0.TextGrid",
    "501e6487-Audio_SP19.v0.TextGrid",
    "5278b5b9-Audio_SP15.v0.TextGrid",
    "5586979c-Audio_SP18.v0.TextGrid",
    "5819ce7a-Audio_SP11.v0.TextGrid",
    "7aee4437-Audio_SP12.v0.TextGrid",
    "7c3c98d2-Audio_SP13.v0.TextGrid",
    "85c7895f-Audio_SP16.v0.TextGrid",
    "a452055c-Audio_SP11.v0.TextGrid",
    "d4c363bf-Audio_SP13.v0.TextGrid",
    "e90f8e4d-Audio_SP22.v0.TextGrid",
]

SKIP_FILES = {
    "42d56bbf-Audio_SP23.v0.TextGrid",
}


@dataclass
class Interval:
    tier_name: str
    number: int
    xmin: float
    xmax: float
    text: str


def extract_tier_blocks(content: str) -> list[str]:
    matches = list(
        re.finditer(
            r"(?m)^\s{4}item\s+\[\d+\]:\s*$",
            content,
        )
    )

    blocks: list[str] = []

    for index, match in enumerate(matches):
        start = match.start()

        if index + 1 < len(matches):
            end = matches[index + 1].start()
        else:
            end = len(content)

        blocks.append(content[start:end])

    return blocks


def extract_tier_name(block: str) -> str:
    match = re.search(
        r'(?m)^\s*name\s*=\s*"([^"]*)"\s*$',
        block,
    )

    if match is None:
        return "<unknown>"

    return match.group(1)


def extract_intervals(
    block: str,
    tier_name: str,
) -> list[Interval]:
    interval_matches = list(
        re.finditer(
            r"(?m)^\s{8}intervals\s+\[(\d+)\]:\s*$",
            block,
        )
    )

    intervals: list[Interval] = []

    for index, match in enumerate(interval_matches):
        start = match.start()

        if index + 1 < len(interval_matches):
            end = interval_matches[index + 1].start()
        else:
            end = len(block)

        interval_block = block[start:end]
        interval_number = int(match.group(1))

        xmin_match = re.search(
            r"(?m)^\s*xmin\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$",
            interval_block,
        )

        xmax_match = re.search(
            r"(?m)^\s*xmax\s*=\s*([-+]?\d+(?:\.\d+)?)\s*$",
            interval_block,
        )

        text_match = re.search(
            r'(?m)^\s*text\s*=\s*"(.*)"\s*$',
            interval_block,
        )

        if xmin_match is None or xmax_match is None:
            continue

        xmin = float(xmin_match.group(1))
        xmax = float(xmax_match.group(1))
        text = text_match.group(1) if text_match else ""

        intervals.append(
            Interval(
                tier_name=tier_name,
                number=interval_number,
                xmin=xmin,
                xmax=xmax,
                text=text,
            )
        )

    return intervals


def validate_tier(
    intervals: list[Interval],
) -> list[str]:
    errors: list[str] = []

    previous: Interval | None = None

    for interval in intervals:
        if interval.xmin > interval.xmax:
            errors.append(
                "INVALID INTERVAL\n"
                f"  Tier: {interval.tier_name}\n"
                f"  Interval: {interval.number}\n"
                f"  Start: {interval.xmin}\n"
                f"  End: {interval.xmax}\n"
                f"  Text: {interval.text!r}"
            )

        if (
            previous is not None
            and interval.xmin < previous.xmax
        ):
            errors.append(
                "OVERLAPPING INTERVALS\n"
                f"  Tier: {interval.tier_name}\n"
                f"  Previous interval {previous.number}: "
                f"({previous.xmin}, {previous.xmax}, "
                f"{previous.text!r})\n"
                f"  Current interval {interval.number}: "
                f"({interval.xmin}, {interval.xmax}, "
                f"{interval.text!r})\n"
                f"  Overlap: "
                f"{previous.xmax - interval.xmin:.6f} s"
            )

        if (
            previous is None
            or interval.xmax > previous.xmax
        ):
            previous = interval

    return errors


def validate_textgrid(path: Path) -> list[str]:
    content = path.read_text(encoding="utf-8")
    tier_blocks = extract_tier_blocks(content)

    if not tier_blocks:
        return ["NO TIERS FOUND"]

    errors: list[str] = []

    for block in tier_blocks:
        tier_name = extract_tier_name(block)
        intervals = extract_intervals(block, tier_name)

        if not intervals:
            errors.append(
                f"NO INTERVALS FOUND IN TIER: {tier_name}"
            )
            continue

        errors.extend(validate_tier(intervals))

    return errors


def main() -> None:
    report_lines: list[str] = []

    for file_name in PROBLEM_FILES:
        path = TEXTGRID_DIR / file_name

        print("=" * 80)
        print(file_name)

        report_lines.append("=" * 80)
        report_lines.append(file_name)

        if file_name in SKIP_FILES:
            message = (
                "SKIPPED: previously caused out-of-memory "
                "during new-fave processing."
            )

            print(message)
            report_lines.append(message)
            continue

        if not path.exists():
            message = f"FILE NOT FOUND: {path}"

            print(message)
            report_lines.append(message)
            continue

        try:
            errors = validate_textgrid(path)

        except UnicodeDecodeError as error:
            message = f"UTF-8 READING ERROR: {error}"

            print(message)
            report_lines.append(message)
            continue

        except Exception as error:
            message = (
                f"UNEXPECTED VALIDATION ERROR: "
                f"{type(error).__name__}: {error}"
            )

            print(message)
            report_lines.append(message)
            continue

        if not errors:
            message = (
                "No invalid or overlapping intervals "
                "were detected by this validator."
            )

            print(message)
            report_lines.append(message)
            continue

        for error in errors:
            print(error)
            print()

            report_lines.append(error)
            report_lines.append("")

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print()
    print(f"Report created: {REPORT_PATH}")


if __name__ == "__main__":
    main()