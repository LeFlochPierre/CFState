from pathlib import Path
import re
import pandas as pd
import numpy as np


ROOT = Path("external/EPathDash/GEO_Datasets/Pseudomonas")
COUNT_DIR = ROOT / "Count_Tables"

OUT_DIR = Path("results/audit/gene_identifiers")
OUT_DIR.mkdir(parents=True, exist_ok=True)


STUDIES = [
    "GSE136111",
    "GSE203348",
    "GSE87213",
]


def classify_identifier(x):
    x = str(x).strip()

    if x == "" or x.lower() == "nan":
        return "blank"

    # PAO1 locus tags: PA0001, PA1234, etc.
    if re.fullmatch(r"PA\d{4}", x, flags=re.IGNORECASE):
        return "PAO1_locus_tag"

    # PA14 locus tags: PA14_12340 etc.
    if re.fullmatch(r"PA14_\d+", x, flags=re.IGNORECASE):
        return "PA14_locus_tag"

    # Simple gene-symbol-like strings
    if re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_.-]*",
        x,
    ):
        return "gene_symbol_or_other"

    return "other"


print()
print("=" * 100)
print("CF-STATE — Gene identifier and feature audit")
print("=" * 100)


summary_records = []


for accession in STUDIES:

    path = COUNT_DIR / f"{accession}_counts.csv"

    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
    )

    df.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in df.columns
    ]

    gene_col = df.columns[0]
    sample_cols = df.columns[1:]

    identifiers = (
        df[gene_col]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    numeric = df[sample_cols].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # ---------------------------------------------------------
    # Identifier statistics
    # ---------------------------------------------------------

    n_rows = len(df)

    blank_mask = (
        identifiers.eq("")
        |
        identifiers.str.lower().eq("nan")
    )

    n_blank = int(blank_mask.sum())

    nonblank = identifiers[~blank_mask]

    n_unique = int(nonblank.nunique())

    duplicate_mask = (
        nonblank.duplicated(keep=False)
    )

    duplicated = nonblank[
        duplicate_mask
    ]

    n_duplicate_rows = len(duplicated)

    n_duplicate_ids = duplicated.nunique()

    id_classes = (
        identifiers
        .map(classify_identifier)
        .value_counts()
    )

    # ---------------------------------------------------------
    # Expression-value statistics
    # ---------------------------------------------------------

    total_values = numeric.size

    missing_values = int(
        numeric.isna().sum().sum()
    )

    zero_values = int(
        (numeric == 0).sum().sum()
    )

    finite_values = numeric.to_numpy(
        dtype=float
    )

    finite_values = finite_values[
        np.isfinite(finite_values)
    ]

    if len(finite_values):

        integer_like_fraction = float(
            np.mean(
                np.isclose(
                    finite_values,
                    np.round(finite_values),
                )
            )
        )

    else:

        integer_like_fraction = np.nan

    # ---------------------------------------------------------
    # Print report
    # ---------------------------------------------------------

    print()
    print("=" * 100)
    print(accession)
    print("=" * 100)

    print(f"Feature column:             {gene_col}")
    print(f"Feature rows:               {n_rows}")
    print(f"Unique nonblank IDs:        {n_unique}")
    print(f"Blank IDs:                  {n_blank}")
    print(f"Rows with duplicated IDs:   {n_duplicate_rows}")
    print(f"Distinct duplicated IDs:    {n_duplicate_ids}")
    print()

    print("Identifier classes:")
    print(
        id_classes.to_string()
    )

    print()
    print("First 25 identifiers:")
    print(
        identifiers.head(25).to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # Duplicates
    # ---------------------------------------------------------

    if n_duplicate_ids > 0:

        duplicate_counts = (
            duplicated
            .value_counts()
            .rename_axis("identifier")
            .reset_index(name="n_occurrences")
        )

        print()
        print("Most frequent duplicated IDs:")
        print(
            duplicate_counts
            .head(30)
            .to_string(index=False)
        )

        duplicate_out = (
            OUT_DIR
            / f"{accession}_duplicated_identifiers.csv"
        )

        duplicate_counts.to_csv(
            duplicate_out,
            index=False,
        )

    # ---------------------------------------------------------
    # Missing values
    # ---------------------------------------------------------

    missing_by_sample = (
        numeric.isna()
        .sum()
        .sort_values(ascending=False)
    )

    missing_by_sample = missing_by_sample[
        missing_by_sample > 0
    ]

    missing_by_feature_mask = (
        numeric.isna().any(axis=1)
    )

    missing_features = df.loc[
        missing_by_feature_mask,
        [gene_col] + list(sample_cols),
    ]

    print()
    print("Expression matrix:")
    print(f"Samples:                    {len(sample_cols)}")
    print(f"Total expression values:    {total_values}")
    print(f"Missing values:             {missing_values}")
    print(f"Zero values:                {zero_values}")
    print(
        f"Integer-like fraction:      "
        f"{integer_like_fraction:.6f}"
    )

    if not missing_by_sample.empty:

        print()
        print("Missing values by sample:")
        print(
            missing_by_sample.to_string()
        )

        print()
        print(
            f"Features containing >=1 missing value: "
            f"{missing_by_feature_mask.sum()}"
        )

        print()
        print("First missing-value features:")
        print(
            missing_features
            .head(30)
            .to_string(index=False)
        )

        missing_out = (
            OUT_DIR
            / f"{accession}_missing_values.csv"
        )

        missing_features.to_csv(
            missing_out,
            index=False,
        )

    # ---------------------------------------------------------
    # Save identifier table
    # ---------------------------------------------------------

    id_table = pd.DataFrame(
        {
            "identifier": identifiers,
            "identifier_class": identifiers.map(
                classify_identifier
            ),
        }
    )

    id_table.to_csv(
        OUT_DIR
        / f"{accession}_identifiers.csv",
        index=False,
    )

    summary_records.append(
        {
            "study": accession,
            "feature_column": gene_col,
            "n_features": n_rows,
            "n_unique_nonblank": n_unique,
            "n_blank": n_blank,
            "n_duplicate_rows": n_duplicate_rows,
            "n_duplicate_ids": n_duplicate_ids,
            "n_samples": len(sample_cols),
            "missing_values": missing_values,
            "zero_values": zero_values,
            "integer_like_fraction": integer_like_fraction,
        }
    )


# ============================================================
# Combined summary
# ============================================================

summary = pd.DataFrame(
    summary_records
)

summary_file = (
    OUT_DIR
    / "gene_identifier_audit_summary.csv"
)

summary.to_csv(
    summary_file,
    index=False,
)


print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

print(
    summary.to_string(index=False)
)

print()
print(f"Saved audit files to: {OUT_DIR}")
print(f"Saved summary:        {summary_file}")
print()
