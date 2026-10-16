from pathlib import Path
import pandas as pd

ROOT = Path("external/EPathDash/GEO_Datasets/Pseudomonas")

DESIGN_DIR = ROOT / "Design_Matrices"
COUNT_DIR = ROOT / "Count_Tables"

STUDIES = [
    "GSE136111",
    "GSE203348",
    "GSE223663",
    "GSE87213",
]


print()
print("=" * 100)
print("CF-STATE — Biofilm count/design alignment inspection")
print("=" * 100)


for accession in STUDIES:

    design_file = DESIGN_DIR / f"{accession}_design.csv"
    count_file = COUNT_DIR / f"{accession}_counts.csv"

    print()
    print("=" * 100)
    print(accession)
    print("=" * 100)

    design = pd.read_csv(
        design_file,
        encoding="utf-8-sig",
    )

    counts = pd.read_csv(
        count_file,
        encoding="utf-8-sig",
    )

    design.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in design.columns
    ]

    counts.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in counts.columns
    ]

    print()
    print("DESIGN")
    print("-" * 100)

    print(f"Shape: {design.shape}")
    print(f"Columns: {list(design.columns)}")

    print()
    print(design.to_string(index=True))

    print()
    print("COUNT TABLE")
    print("-" * 100)

    print(f"Shape: {counts.shape}")
    print(f"Number of design rows: {len(design)}")
    print(f"Number of count-table columns: {counts.shape[1]}")

    print()
    print("Count-table columns:")
    for i, col in enumerate(counts.columns):
        print(f"{i:3d}: {col}")

    print()
    print("First 3 rows / first 10 columns:")

    preview_cols = counts.columns[:10]

    print(
        counts[
            preview_cols
        ].head(3).to_string(index=False)
    )

    print()
    print(
        "Difference between count columns and design rows: "
        f"{counts.shape[1] - len(design)}"
    )


print()
print("=" * 100)
print("Inspection complete")
print("=" * 100)
