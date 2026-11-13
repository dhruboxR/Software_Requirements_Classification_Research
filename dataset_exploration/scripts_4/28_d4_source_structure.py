
from pathlib import Path
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    ROOT / "A Unified Multi-Source Dataset of Software Require"
)

TARGET = "Classified_AI_Requirements.xlsx"

matches = list(DATASET_ROOT.rglob(TARGET))

if len(matches) != 1:
    raise RuntimeError(
        f"Expected exactly one {TARGET}, found {len(matches)}"
    )

path = matches[0]

print("=" * 80)
print("SCRIPT 28: D4 ORIGINAL SOURCE STRUCTURE")
print("=" * 80)

print("\nOriginal workbook:", path)
print("File size:", path.stat().st_size, "bytes")

wb = load_workbook(
    path,
    read_only=True,
    data_only=True
)

print("\nWORKSHEETS")
print(wb.sheetnames)

for sheet in wb:
    print("\n" + "=" * 80)
    print("SHEET:", sheet.title)
    print("=" * 80)

    print("Reported rows:", sheet.max_row)
    print("Reported columns:", sheet.max_column)

    print("\nFIRST 8 ROWS")

    for row_number, row in enumerate(
        sheet.iter_rows(values_only=True),
        start=1
    ):
        if row_number > 8:
            break

        print(f"\nRow {row_number}:")

        for col_number, value in enumerate(
            row,
            start=1
        ):
            if value is None:
                continue

            text = str(value).replace("\n", " ").strip()

            if len(text) > 200:
                text = text[:200] + "..."

            print(f"  Column {col_number}: {text}")

wb.close()

print("\nInspection complete.")
print("Original workbook unchanged.")
