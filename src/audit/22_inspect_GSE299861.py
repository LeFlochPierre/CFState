from pathlib import Path
import re

import numpy as np
import pandas as pd
import requests


# ============================================================
# Paths
# ============================================================

OUT_DIR = Path(
    "data/raw/expanded_biofilm/GSE299861"
)

AUDIT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# GEO supplementary files
# ============================================================

BASE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE299nnn/GSE299861/suppl/"
)

FILES = {
    "PAO1":
        "GSE299861_PAO1_all.fpkm.txt.gz",

    "MPAO1":
        "GSE299861_MPAO1_all.fpkm.txt.gz",
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 22: "
    "inspect expansion candidate GSE299861"
)
print("=" * 100)


# ============================================================
# Download
# ============================================================

for strain, filename in FILES.items():

    destination = (
        OUT_DIR / filename
    )

    if not destination.exists():

        url = (
            BASE_URL
            + filename
        )

        print()
        print(
            f"Downloading {strain}:"
        )

        print(url)

        response = requests.get(
            url,
            timeout=60,
        )

        response.raise_for_status()

        destination.write_bytes(
            response.content
        )

        print(
            f"Saved: {destination}"
        )

    else:

        print()
        print(
            f"Already downloaded: "
            f"{destination}"
        )


# ============================================================
# Helpers
# ============================================================

def integer_like_fraction(
    values,
):

    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(arr)
    ]

    if len(arr) == 0:
        return np.nan

    return float(
        np.mean(
            np.isclose(
                arr,
                np.round(arr),
            )
        )
    )


# ============================================================
# Inspect each matrix
# ============================================================

summary_records = []


for strain, filename in FILES.items():

    path = (
        OUT_DIR / filename
    )

    print()
    print("=" * 100)
    print(strain)
    print("=" * 100)

    df = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
    )

    print(
        f"Shape: {df.shape}"
    )

    print()
    print("Columns:")

    for column in df.columns:

        print(
            f"  {column}"
        )


    # --------------------------------------------------------
    # Feature identifier column
    # --------------------------------------------------------

    feature_col = df.columns[0]

    feature_ids = (
        df[feature_col]
        .astype(str)
        .str.strip()
    )


    print()
    print(
        f"Feature column: "
        f"{feature_col}"
    )

    print(
        f"Feature rows: "
        f"{len(feature_ids)}"
    )

    print(
        f"Unique feature IDs: "
        f"{feature_ids.nunique()}"
    )

    print(
        f"Duplicate feature rows: "
        f"{feature_ids.duplicated(keep=False).sum()}"
    )


    # --------------------------------------------------------
    # Identifier nomenclature
    # --------------------------------------------------------

    pao1_mask = (
        feature_ids
        .str.fullmatch(
            r"PA\d{4}"
        )
    )

    pa14_mask = (
        feature_ids
        .str.fullmatch(
            r"PA14_\d+"
        )
    )

    print()
    print("Identifier classes:")

    print(
        f"  PAO1 locus tags: "
        f"{pao1_mask.sum()}"
    )

    print(
        f"  PA14 locus tags: "
        f"{pa14_mask.sum()}"
    )

    other_count = (~pao1_mask & ~pa14_mask).sum()
    print(f"  Other: {other_count}")


    print()
    print(
        "First 20 feature IDs:"
    )

    print(
        feature_ids
        .head(20)
        .to_string(index=False)
    )


    # --------------------------------------------------------
    # Expression columns
    # --------------------------------------------------------

    expression_cols = list(
        df.columns[1:]
    )

    expression = (
        df[
            expression_cols
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )


    values = (
        expression
        .to_numpy(
            dtype=float
        )
    )

    finite = values[
        np.isfinite(values)
    ]


    n_missing = int(
        np.isnan(values).sum()
    )

    n_zero = int(
        np.nansum(
            values == 0
        )
    )


    print()
    print("Expression QC:")

    print(
        f"  Sample/value columns: "
        f"{len(expression_cols)}"
    )

    print(
        f"  Missing values: "
        f"{n_missing}"
    )

    print(
        f"  Zero values: "
        f"{n_zero}"
    )

    print(
        f"  Integer-like fraction: "
        f"{integer_like_fraction(finite):.6f}"
    )


    if len(finite):

        print(
            f"  Minimum: "
            f"{np.min(finite):.6g}"
        )

        print(
            f"  Median: "
            f"{np.median(finite):.6g}"
        )

        print(
            f"  95th percentile: "
            f"{np.percentile(finite, 95):.6g}"
        )

        print(
            f"  Maximum: "
            f"{np.max(finite):.6g}"
        )


    # --------------------------------------------------------
    # Per-column summaries
    # --------------------------------------------------------

    sample_records = []

    print()
    print("Per-column summary:")

    for column in expression_cols:

        values_col = (
            pd.to_numeric(
                df[column],
                errors="coerce",
            )
            .to_numpy(
                dtype=float
            )
        )

        rec = {
            "strain":
                strain,

            "column":
                column,

            "missing":
                int(
                    np.isnan(
                        values_col
                    ).sum()
                ),

            "nonzero":
                int(
                    np.nansum(
                        values_col != 0
                    )
                ),

            "sum":
                float(
                    np.nansum(
                        values_col
                    )
                ),

            "median":
                float(
                    np.nanmedian(
                        values_col
                    )
                ),

            "max":
                float(
                    np.nanmax(
                        values_col
                    )
                ),
        }

        sample_records.append(
            rec
        )


    sample_df = pd.DataFrame(
        sample_records
    )

    print(
        sample_df.to_string(
            index=False
        )
    )


    sample_df.to_csv(
        AUDIT_DIR
        / f"{strain}_column_summary.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Save non-standard identifiers
    # --------------------------------------------------------

    nonstandard = pd.DataFrame(
        {
            "feature_id":
                feature_ids[
                    ~pao1_mask
                    &
                    ~pa14_mask
                ]
        }
    )

    nonstandard.to_csv(
        AUDIT_DIR
        / f"{strain}_nonstandard_ids.csv",
        index=False,
    )


    summary_records.append(
        {
            "strain":
                strain,

            "n_rows":
                len(df),

            "n_expression_columns":
                len(expression_cols),

            "n_unique_features":
                feature_ids.nunique(),

            "n_pao1_ids":
                int(
                    pao1_mask.sum()
                ),

            "n_pa14_ids":
                int(
                    pa14_mask.sum()
                ),

            "n_other_ids":
                int(
                    (
                        ~pao1_mask
                        &
                        ~pa14_mask
                    ).sum()
                ),

            "n_missing_values":
                n_missing,

            "integer_like_fraction":
                integer_like_fraction(
                    finite
                ),
        }
    )


# ============================================================
# Overall summary
# ============================================================

summary = pd.DataFrame(
    summary_records
)

summary.to_csv(
    AUDIT_DIR
    / "GSE299861_matrix_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 22 SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)

print()
print(
    f"Raw files saved to: "
    f"{OUT_DIR}"
)

print(
    f"Audit outputs saved to: "
    f"{AUDIT_DIR}"
)

print()
