from pathlib import Path

import pandas as pd


FILE = Path(
    "data/raw/expanded_biofilm/GSE191103/"
    "GSE191103_ProcessedDataMatrix_biofilm_lasR.xlsx"
)


print()
print("=" * 100)
print("CF-STATE — Audit 35: inspect published GSE191103 count matrix")
print("=" * 100)


if not FILE.exists():
    raise FileNotFoundError(
        "Published matrix not found: " + str(FILE)
    )


# ============================================================
# Inspect workbook
# ============================================================

excel = pd.ExcelFile(FILE)

print()
print("Sheets:")
for sheet in excel.sheet_names:
    print("  " + sheet)


# ============================================================
# Inspect every sheet
# ============================================================

for sheet in excel.sheet_names:

    print()
    print("=" * 100)
    print("SHEET: " + sheet)
    print("=" * 100)

    df = pd.read_excel(
        FILE,
        sheet_name=sheet,
    )

    n_rows = df.shape[0]
    n_cols = df.shape[1]

    print("Rows: " + str(n_rows))
    print("Columns: " + str(n_cols))

    print()
    print("Column names:")

    for column in df.columns:
        print("  " + str(column))

    print()
    print("First five rows:")

    print(
        df.head().to_string(
            index=False
        )
    )


# ============================================================
# Use first sheet for sample search
# ============================================================

df = pd.read_excel(
    FILE,
    sheet_name=excel.sheet_names[0],
)


TARGETS = [
    "PA14_BF_1",
    "PA14_BF_2",
    "PA14_BF_3",
    "PA14_PL_1",
    "PA14_PL_2",
    "PA14_PL_3",
]


print()
print("=" * 100)
print("WT PA14 SAMPLE COLUMN SEARCH")
print("=" * 100)


matched_columns = {}


for target in TARGETS:

    matches = []

    target_lower = target.lower()

    for column in df.columns:

        column_text = str(column)

        if target_lower in column_text.lower():
            matches.append(column_text)

    matched_columns[target] = matches

    print(
        target
        + ": "
        + str(matches)
    )


# ============================================================
# Published count totals
# ============================================================

print()
print("=" * 100)
print("PUBLISHED WT PA14 COUNT TOTALS")
print("=" * 100)


for target in TARGETS:

    matches = matched_columns[target]

    if len(matches) != 1:

        print(
            target
            + ": cannot calculate total because "
            + str(len(matches))
            + " matching columns were found"
        )

        continue

    column = matches[0]

    numeric = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    total = numeric.sum()

    nonzero = int(
        (numeric > 0).sum()
    )

    print(
        target
        + " | total = "
        + format(total, ",.0f")
        + " | nonzero rows = "
        + str(nonzero)
    )


print()
print("=" * 100)
print("AUDIT 35 COMPLETE")
print("=" * 100)
print()
