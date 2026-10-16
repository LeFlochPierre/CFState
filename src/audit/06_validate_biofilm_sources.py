from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path("external/EPathDash/GEO_Datasets/Pseudomonas")
COUNT_DIR = ROOT / "Count_Tables"

STUDIES = [
    "GSE136111",
    "GSE203348",
    "GSE223663",
    "GSE87213",
]


def load_counts(accession):
    path = COUNT_DIR / f"{accession}_counts.csv"

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    return df


def vector_hash(values):
    """
    Hash an expression vector after converting to float64.

    Exact duplicates should produce the same hash.
    """
    arr = pd.to_numeric(
        pd.Series(values),
        errors="coerce"
    ).to_numpy(dtype=np.float64)

    return hashlib.sha256(
        arr.tobytes()
    ).hexdigest()


print()
print("=" * 100)
print("CF-STATE — Biofilm source validation")
print("=" * 100)

tables = {}

sample_records = []

for accession in STUDIES:

    df = load_counts(accession)
    tables[accession] = df

    gene_col = df.columns[0]
    sample_cols = df.columns[1:]

    numeric = df[sample_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    values = numeric.to_numpy(dtype=float)

    finite = values[np.isfinite(values)]

    if finite.size:
        integer_fraction = np.mean(
            np.isclose(finite, np.round(finite))
        )
    else:
        integer_fraction = np.nan

    print()
    print("-" * 100)
    print(accession)
    print("-" * 100)

    print(f"Genes/features:       {len(df)}")
    print(f"Samples:              {len(sample_cols)}")
    print(f"Gene identifier col:  {gene_col}")
    print(f"Missing values:       {numeric.isna().sum().sum()}")
    print(f"Integer-like values:  {integer_fraction:.4f}")

    if finite.size:
        print(f"Minimum:              {finite.min()}")
        print(f"Maximum:              {finite.max()}")

    print()
    print("Sample columns:")

    for col in sample_cols:

        h = vector_hash(numeric[col])

        sample_records.append(
            {
                "study": accession,
                "sample": col,
                "hash": h,
            }
        )

        print(f"  {col}")


# ------------------------------------------------------------------
# Exact duplicate expression vectors
# ------------------------------------------------------------------

samples = pd.DataFrame(sample_records)

duplicate_groups = (
    samples.groupby("hash")
    .filter(lambda x: len(x) > 1)
    .sort_values(["hash", "study", "sample"])
)

print()
print("=" * 100)
print("EXACT DUPLICATE SAMPLE VECTORS")
print("=" * 100)

if duplicate_groups.empty:

    print("No exact duplicate sample vectors found.")

else:

    for h, group in duplicate_groups.groupby("hash"):

        print()
        print(f"Hash: {h[:16]}...")

        for _, row in group.iterrows():

            print(
                f"  {row['study']:12s}  "
                f"{row['sample']}"
            )


# ------------------------------------------------------------------
# Repeated sample names across studies
# ------------------------------------------------------------------

print()
print("=" * 100)
print("REPEATED SAMPLE NAMES ACROSS STUDIES")
print("=" * 100)

name_counts = (
    samples.groupby("sample")["study"]
    .nunique()
)

repeated_names = name_counts[
    name_counts > 1
].index

if len(repeated_names) == 0:

    print("No repeated sample names found.")

else:

    for name in repeated_names:

        subset = samples[
            samples["sample"] == name
        ]

        print()
        print(name)

        for _, row in subset.iterrows():

            print(f"  {row['study']}")


# ------------------------------------------------------------------
# Compare GSE203348 directly against WT columns from GSE223663
# ------------------------------------------------------------------

print()
print("=" * 100)
print("GSE203348 vs GSE223663 DIRECT CHECK")
print("=" * 100)

a = tables["GSE203348"]
b = tables["GSE223663"]

gene_a = a.columns[0]
gene_b = b.columns[0]

shared_samples = sorted(
    set(a.columns[1:])
    &
    set(b.columns[1:])
)

print(f"Shared sample names: {len(shared_samples)}")

for sample in shared_samples:

    if len(a) != len(b):

        same = False

    else:

        same_genes = (
            a[gene_a].astype(str).to_numpy()
            ==
            b[gene_b].astype(str).to_numpy()
        ).all()

        va = pd.to_numeric(
            a[sample],
            errors="coerce"
        ).to_numpy(dtype=float)

        vb = pd.to_numeric(
            b[sample],
            errors="coerce"
        ).to_numpy(dtype=float)

        same_values = np.array_equal(
            va,
            vb,
            equal_nan=True,
        )

        same = same_genes and same_values

    print(
        f"{sample:25s} exact_match={same}"
    )


OUT = Path(
    "results/audit/biofilm_sample_fingerprints.csv"
)

OUT.parent.mkdir(
    parents=True,
    exist_ok=True,
)

samples.to_csv(
    OUT,
    index=False,
)

print()
print(f"Saved fingerprints: {OUT}")
print()
