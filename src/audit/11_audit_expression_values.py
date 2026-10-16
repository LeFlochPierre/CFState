from pathlib import Path
import numpy as np
import pandas as pd


COUNT_DIR = Path(
    "external/EPathDash/GEO_Datasets/Pseudomonas/Count_Tables"
)

HARM_DIR = Path(
    "results/audit/gene_harmonisation"
)

OUT_DIR = Path(
    "results/audit/expression_qc"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Primary samples
# ============================================================

PRIMARY_SAMPLES = {
    "GSE136111": {
        "B_1": "biofilm",
        "B_2": "biofilm",
        "B_3": "biofilm",
        "P_1": "planktonic",
        "P_2": "planktonic",
        "P_3": "planktonic",
    },

    "GSE203348": {
        "BioFWTBR1_S10": "biofilm",
        "BioFWTBR2_S11": "biofilm",
        "BioFWTBR3_S12": "biofilm",
        "PlankWTBR1_S1": "planktonic",
        "PlankWTBR2_S2": "planktonic",
        "PlankWTBR3_S3": "planktonic",
    },

    "GSE87213": {
        "WTB3.raw": "biofilm",
        "WTB4.raw": "biofilm",
        "WTB6.raw": "biofilm",
        "WTP4.raw": "planktonic",
        "WTP5.raw": "planktonic",
        "WTP6.raw": "planktonic",
    },
}


# ============================================================
# Common canonical gene universe
# ============================================================

common_df = pd.read_csv(
    HARM_DIR
    / "common_pao1_gene_universe.csv"
)

COMMON_GENES = set(
    common_df["canonical_pao1"]
    .astype(str)
    .str.strip()
)

print()
print("=" * 100)
print("CF-STATE — Audit 11: pre-normalisation expression QC")
print("=" * 100)

print(
    f"Common canonical PAO1 genes: "
    f"{len(COMMON_GENES)}"
)


# ============================================================
# Helpers
# ============================================================

def integer_like_fraction(values):

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


def build_mapped_table(study):

    raw = pd.read_csv(
        COUNT_DIR / f"{study}_counts.csv",
        encoding="utf-8-sig",
    )

    raw.columns = [
        str(c)
        .replace("\ufeff", "")
        .strip()
        for c in raw.columns
    ]

    source_ids = (
        raw.iloc[:, 0]
        .astype(str)
        .str.strip()
    )

    sample_cols = list(
        PRIMARY_SAMPLES[study].keys()
    )

    missing_sample_cols = [
        x
        for x in sample_cols
        if x not in raw.columns
    ]

    if missing_sample_cols:
        raise ValueError(
            f"{study}: primary samples missing "
            f"from count table: {missing_sample_cols}"
        )

    numeric = raw[
        sample_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    # ----------------------------------------
    # Attach canonical PAO1 IDs
    # ----------------------------------------

    if study == "GSE136111":

        mapping = pd.read_csv(
            HARM_DIR
            / "GSE136111_mapping_audit.csv"
        )

        mapping_ids = (
            mapping["original_id"]
            .astype(str)
            .str.strip()
        )

        # Mapping file was generated row-by-row,
        # so ordering must remain identical.
        if not source_ids.reset_index(
            drop=True
        ).equals(
            mapping_ids.reset_index(
                drop=True
            )
        ):
            raise ValueError(
                "GSE136111 mapping order does "
                "not match count table."
            )

        canonical = (
            mapping["canonical_pao1"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    elif study == "GSE203348":

        canonical = source_ids.copy()

    elif study == "GSE87213":

        mapping = pd.read_csv(
            HARM_DIR
            / "GSE87213_mapping_audit.csv"
        )

        mapping_ids = (
            mapping["original_id"]
            .astype(str)
            .str.strip()
        )

        if not source_ids.reset_index(
            drop=True
        ).equals(
            mapping_ids.reset_index(
                drop=True
            )
        ):
            raise ValueError(
                "GSE87213 mapping order does "
                "not match count table."
            )

        canonical = (
            mapping["canonical_pao1"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    else:
        raise ValueError(study)

    mapped = pd.DataFrame(
        {
            "source_id": source_ids,
            "canonical_pao1": canonical,
        }
    )

    mapped = pd.concat(
        [
            mapped.reset_index(drop=True),
            numeric.reset_index(drop=True),
        ],
        axis=1,
    )

    mapped = mapped[
        mapped["canonical_pao1"].isin(
            COMMON_GENES
        )
    ].copy()

    return mapped


# ============================================================
# Run study-level QC
# ============================================================

study_records = []
sample_records = []
duplicate_records = []

for study in PRIMARY_SAMPLES:

    mapped = build_mapped_table(
        study
    )

    sample_cols = list(
        PRIMARY_SAMPLES[study].keys()
    )

    expression = mapped[
        sample_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    canonical = mapped[
        "canonical_pao1"
    ]

    unique_genes = set(
        canonical
    )

    missing_common = sorted(
        COMMON_GENES
        - unique_genes
    )

    duplicated_mask = canonical.duplicated(
        keep=False
    )

    duplicated = mapped[
        duplicated_mask
    ].copy()

    duplicate_genes = sorted(
        duplicated[
            "canonical_pao1"
        ].unique()
    )

    all_values = expression.to_numpy(
        dtype=float
    )

    finite = all_values[
        np.isfinite(all_values)
    ]

    n_missing = int(
        np.isnan(
            all_values
        ).sum()
    )

    n_zero = int(
        np.nansum(
            all_values == 0
        )
    )

    n_total = int(
        all_values.size
    )

    print()
    print("=" * 100)
    print(study)
    print("=" * 100)

    print(
        f"Rows mapping into common universe: "
        f"{len(mapped)}"
    )

    print(
        f"Unique common genes represented:   "
        f"{len(unique_genes)} / "
        f"{len(COMMON_GENES)}"
    )

    print(
        f"Missing common genes:              "
        f"{len(missing_common)}"
    )

    print(
        f"Canonical genes with >1 row:       "
        f"{len(duplicate_genes)}"
    )

    print(
        f"Missing expression values:         "
        f"{n_missing}"
    )

    print(
        f"Zero expression values:            "
        f"{n_zero}"
    )

    print(
        f"Zero fraction:                     "
        f"{n_zero / n_total:.6f}"
    )

    print(
        f"Integer-like fraction:             "
        f"{integer_like_fraction(finite):.6f}"
    )

    if len(finite):

        print(
            f"Minimum:                           "
            f"{np.min(finite):.6g}"
        )

        print(
            f"Median:                            "
            f"{np.median(finite):.6g}"
        )

        print(
            f"95th percentile:                   "
            f"{np.percentile(finite, 95):.6g}"
        )

        print(
            f"99th percentile:                   "
            f"{np.percentile(finite, 99):.6g}"
        )

        print(
            f"Maximum:                           "
            f"{np.max(finite):.6g}"
        )

    # ----------------------------------------
    # Duplicate mapping audit
    # ----------------------------------------

    if duplicate_genes:

        print()
        print("Duplicate canonical mappings:")

        for gene in duplicate_genes:

            block = mapped[
                mapped["canonical_pao1"]
                == gene
            ]

            arr = block[
                sample_cols
            ].to_numpy(
                dtype=float
            )

            exact = True

            for i in range(
                1,
                len(arr),
            ):

                if not np.allclose(
                    arr[0],
                    arr[i],
                    equal_nan=True,
                ):
                    exact = False
                    break

            max_diff = 0.0

            if len(arr) > 1:

                diffs = []

                for i in range(
                    1,
                    len(arr),
                ):

                    diff = np.abs(
                        arr[0]
                        - arr[i]
                    )

                    finite_diff = diff[
                        np.isfinite(diff)
                    ]

                    if len(finite_diff):
                        diffs.append(
                            finite_diff.max()
                        )

                if diffs:
                    max_diff = max(
                        diffs
                    )

            sources = block[
                "source_id"
            ].tolist()

            print(
                f"  {gene}: "
                f"{sources} | "
                f"identical={exact} | "
                f"max_abs_diff={max_diff}"
            )

            duplicate_records.append(
                {
                    "study": study,
                    "canonical_pao1": gene,
                    "source_ids": "|".join(
                        sources
                    ),
                    "n_rows": len(block),
                    "expression_identical": exact,
                    "max_abs_diff": max_diff,
                }
            )

    # ----------------------------------------
    # Missing genes / missing values
    # ----------------------------------------

    missing_value_rows = mapped[
        expression.isna().any(
            axis=1
        )
    ]

    if len(missing_value_rows):

        print()
        print(
            "Genes containing >=1 missing "
            "value after common-gene filtering:"
        )

        print(
            missing_value_rows[
                [
                    "source_id",
                    "canonical_pao1",
                ]
                + sample_cols
            ]
            .to_string(
                index=False
            )
        )

        missing_value_rows.to_csv(
            OUT_DIR
            / f"{study}_common_genes_with_missing_values.csv",
            index=False,
        )

    pd.DataFrame(
        {
            "canonical_pao1": missing_common
        }
    ).to_csv(
        OUT_DIR
        / f"{study}_missing_common_genes.csv",
        index=False,
    )

    # ----------------------------------------
    # Per-sample QC
    # ----------------------------------------

    print()
    print("Per-sample summary:")

    current_sample_rows = []

    for sample in sample_cols:

        values = pd.to_numeric(
            mapped[sample],
            errors="coerce",
        ).to_numpy(
            dtype=float
        )

        finite_values = values[
            np.isfinite(values)
        ]

        state = PRIMARY_SAMPLES[
            study
        ][sample]

        rec = {
            "study": study,
            "sample": sample,
            "state": state,
            "n_genes": len(values),
            "missing": int(
                np.isnan(values).sum()
            ),
            "nonzero": int(
                np.nansum(
                    values != 0
                )
            ),
            "total_expression": float(
                np.nansum(values)
            ),
            "mean": float(
                np.nanmean(values)
            ),
            "median": float(
                np.nanmedian(values)
            ),
            "p75": float(
                np.nanpercentile(
                    values,
                    75,
                )
            ),
            "p95": float(
                np.nanpercentile(
                    values,
                    95,
                )
            ),
            "p99": float(
                np.nanpercentile(
                    values,
                    99,
                )
            ),
            "max": float(
                np.nanmax(values)
            ),
            "integer_like_fraction":
                integer_like_fraction(
                    finite_values
                ),
        }

        sample_records.append(
            rec
        )

        current_sample_rows.append(
            rec
        )

    current_df = pd.DataFrame(
        current_sample_rows
    )

    print(
        current_df[
            [
                "sample",
                "state",
                "missing",
                "nonzero",
                "total_expression",
                "median",
                "p99",
                "max",
                "integer_like_fraction",
            ]
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------
    # Study summary
    # ----------------------------------------

    study_records.append(
        {
            "study": study,
            "n_rows_common": len(mapped),
            "n_unique_common_genes": len(
                unique_genes
            ),
            "n_missing_common_genes": len(
                missing_common
            ),
            "n_duplicate_canonical_genes": len(
                duplicate_genes
            ),
            "n_missing_values": n_missing,
            "n_zero_values": n_zero,
            "zero_fraction":
                n_zero / n_total,
            "integer_like_fraction":
                integer_like_fraction(
                    finite
                ),
            "minimum":
                np.min(finite)
                if len(finite)
                else np.nan,
            "median":
                np.median(finite)
                if len(finite)
                else np.nan,
            "p99":
                np.percentile(
                    finite,
                    99,
                )
                if len(finite)
                else np.nan,
            "maximum":
                np.max(finite)
                if len(finite)
                else np.nan,
        }
    )


# ============================================================
# Save summaries
# ============================================================

study_summary = pd.DataFrame(
    study_records
)

sample_summary = pd.DataFrame(
    sample_records
)

duplicate_summary = pd.DataFrame(
    duplicate_records
)

study_summary.to_csv(
    OUT_DIR
    / "study_expression_qc_summary.csv",
    index=False,
)

sample_summary.to_csv(
    OUT_DIR
    / "sample_expression_qc_summary.csv",
    index=False,
)

duplicate_summary.to_csv(
    OUT_DIR
    / "duplicate_canonical_mapping_qc.csv",
    index=False,
)


print()
print("=" * 100)
print("STUDY SUMMARY")
print("=" * 100)

print(
    study_summary.to_string(
        index=False
    )
)

print()
print(
    f"Saved QC outputs to: "
    f"{OUT_DIR}"
)
print()
