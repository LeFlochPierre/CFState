from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PUBLISHED_FILE = Path(
    "data/raw/expanded_biofilm/GSE191103/"
    "GSE191103_ProcessedDataMatrix_biofilm_lasR.xlsx"
)

MAPPING_FILE = Path(
    "results/audit/expanded_biofilm/GSE191103/"
    "published_id_mapping/"
    "published_to_PAO1_unique_mappings.csv"
)

FROZEN_FILE = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE191103/finalization"
)

PROCESSED_DIR = Path(
    "data/processed/expanded_biofilm/GSE191103"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 37: "
    "finalize GSE191103 published counts"
)
print("=" * 100)


# ============================================================
# Sample definitions
# ============================================================

SAMPLE_MAP = {
    "PA14_BF_1": "biofilm_PA14_1",
    "PA14_BF_2": "biofilm_PA14_2",
    "PA14_BF_3": "biofilm_PA14_3",
    "PA14_PL_1": "planktonic_PA14_1",
    "PA14_PL_2": "planktonic_PA14_2",
    "PA14_PL_3": "planktonic_PA14_3",
}


STATES = {
    "biofilm_PA14_1": "biofilm",
    "biofilm_PA14_2": "biofilm",
    "biofilm_PA14_3": "biofilm",
    "planktonic_PA14_1": "planktonic",
    "planktonic_PA14_2": "planktonic",
    "planktonic_PA14_3": "planktonic",
}


SAMPLES = list(
    STATES.keys()
)


# ============================================================
# Load published data
# ============================================================

published = pd.read_excel(
    PUBLISHED_FILE,
    sheet_name="ProcessedDataMatrix",
)


mapping = pd.read_csv(
    MAPPING_FILE
)


frozen = pd.read_csv(
    FROZEN_FILE
)


# ============================================================
# Identify frozen gene column
# ============================================================

frozen_gene_col = None


for candidate in [
    "canonical_pao1",
    "GeneID",
    "gene",
    "locus_tag",
]:

    if candidate in frozen.columns:
        frozen_gene_col = candidate
        break


if frozen_gene_col is None:

    if frozen.shape[1] == 1:
        frozen_gene_col = frozen.columns[0]

    else:
        raise RuntimeError(
            "Could not identify frozen gene column."
        )


frozen_genes = (
    frozen[
        frozen_gene_col
    ]
    .astype(str)
    .tolist()
)


if len(frozen_genes) != 4858:

    raise RuntimeError(
        "Expected 4,858 frozen genes, found "
        + str(len(frozen_genes))
    )


# ============================================================
# Basic mapping checks
# ============================================================

required_mapping_columns = [
    "Locus_ID",
    "canonical_pao1",
]


for column in required_mapping_columns:

    if column not in mapping.columns:
        raise RuntimeError(
            "Missing mapping column: " + column
        )


if mapping[
    "canonical_pao1"
].duplicated().any():

    raise RuntimeError(
        "Canonical PAO1 mappings are not unique."
    )


# ============================================================
# Restrict mapping to frozen genes
# ============================================================

mapping_frozen = (
    mapping.loc[
        mapping[
            "canonical_pao1"
        ].isin(
            frozen_genes
        ),
        [
            "Locus_ID",
            "canonical_pao1",
        ],
    ]
    .copy()
)


print()
print("=" * 100)
print("FROZEN GENE MAPPING")
print("=" * 100)

print(
    "Frozen genes expected: "
    + str(len(frozen_genes))
)

print(
    "Frozen mappings found: "
    + str(len(mapping_frozen))
)


if len(mapping_frozen) != 4858:

    raise RuntimeError(
        "Expected exactly 4,858 frozen mappings."
    )


# ============================================================
# Extract six WT PA14 sample columns
# ============================================================

required_columns = [
    "Locus_ID"
] + list(
    SAMPLE_MAP.keys()
)


for column in required_columns:

    if column not in published.columns:
        raise RuntimeError(
            "Missing published column: " + column
        )


published_subset = (
    published[
        required_columns
    ]
    .copy()
)


# ============================================================
# Merge published rows with canonical PAO1 mapping
# ============================================================

merged = (
    mapping_frozen.merge(
        published_subset,
        on="Locus_ID",
        how="left",
        validate="one_to_one",
    )
)


sample_columns_original = list(
    SAMPLE_MAP.keys()
)


if merged[
    sample_columns_original
].isna().any().any():

    raise RuntimeError(
        "Missing published count values after mapping."
    )


# ============================================================
# Rename sample columns
# ============================================================

merged = merged.rename(
    columns=SAMPLE_MAP
)


# ============================================================
# Build genes x samples count matrix
# ============================================================

counts = (
    merged[
        [
            "canonical_pao1"
        ] + SAMPLES
    ]
    .set_index(
        "canonical_pao1"
    )
)


# Force exact original frozen gene ordering
counts = counts.loc[
    frozen_genes
]


counts = counts.apply(
    pd.to_numeric,
    errors="raise",
)


if counts.isna().any().any():

    raise RuntimeError(
        "Count matrix contains missing values."
    )


if (counts < 0).any().any():

    raise RuntimeError(
        "Count matrix contains negative values."
    )


print()
print("=" * 100)
print("FINAL COUNT MATRIX")
print("=" * 100)

print(
    "Genes: "
    + str(counts.shape[0])
)

print(
    "Samples: "
    + str(counts.shape[1])
)

print(
    "Missing values: "
    + str(
        int(
            counts.isna().sum().sum()
        )
    )
)


# ============================================================
# Library summaries
# ============================================================

library_records = []


for sample in SAMPLES:

    total_counts = int(
        counts[
            sample
        ].sum()
    )

    nonzero_genes = int(
        (
            counts[
                sample
            ]
            > 0
        ).sum()
    )

    zero_genes = (
        len(frozen_genes)
        - nonzero_genes
    )

    library_records.append(
        {
            "sample":
                sample,

            "state":
                STATES[sample],

            "total_counts_4858":
                total_counts,

            "nonzero_genes_4858":
                nonzero_genes,

            "zero_genes_4858":
                zero_genes,
        }
    )


library_df = pd.DataFrame(
    library_records
)


print()
print("=" * 100)
print("LIBRARY SUMMARY")
print("=" * 100)

print(
    library_df.to_string(
        index=False
    )
)


library_df.to_csv(
    OUT_DIR
    / "GSE191103_library_summary.csv",
    index=False,
)


# ============================================================
# Within-sample percentile ranks
# ============================================================

rank_matrix = (
    counts.T
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


# ============================================================
# Sample correlations
# ============================================================

correlation = (
    rank_matrix.T
    .corr(
        method="spearman"
    )
)


print()
print("=" * 100)
print("SAMPLE-SAMPLE SPEARMAN CORRELATION")
print("=" * 100)

print(
    correlation
    .round(3)
    .to_string()
)


correlation.to_csv(
    OUT_DIR
    / "GSE191103_rank_sample_correlations.csv"
)


# ============================================================
# Same-state vs different-state correlations
# ============================================================

pair_records = []


for i, sample_a in enumerate(
    SAMPLES
):

    for sample_b in SAMPLES[
        i + 1:
    ]:

        same_state = (
            STATES[sample_a]
            ==
            STATES[sample_b]
        )

        rho = float(
            correlation.loc[
                sample_a,
                sample_b,
            ]
        )

        pair_records.append(
            {
                "sample_a":
                    sample_a,

                "sample_b":
                    sample_b,

                "same_state":
                    same_state,

                "spearman":
                    rho,
            }
        )


pair_df = pd.DataFrame(
    pair_records
)


same_values = (
    pair_df.loc[
        pair_df[
            "same_state"
        ],
        "spearman",
    ]
)


different_values = (
    pair_df.loc[
        ~pair_df[
            "same_state"
        ],
        "spearman",
    ]
)


same_median = float(
    same_values.median()
)

different_median = float(
    different_values.median()
)


print()
print("=" * 100)
print("WITHIN-STATE VS BETWEEN-STATE")
print("=" * 100)

print(
    "Same-state median correlation: "
    + format(
        same_median,
        ".4f",
    )
)

print(
    "Different-state median correlation: "
    + format(
        different_median,
        ".4f",
    )
)


pair_df.to_csv(
    OUT_DIR
    / "GSE191103_pairwise_correlations.csv",
    index=False,
)


# ============================================================
# Save genes x samples matrix
# ============================================================

genes_x_samples = (
    counts
    .reset_index()
)


genes_x_samples.to_csv(
    PROCESSED_DIR
    / "GSE191103_PA14_counts_4858.csv",
    index=False,
)


# ============================================================
# Save samples x genes matrix
# ============================================================

samples_x_genes = (
    counts.T
    .copy()
)


samples_x_genes.insert(
    0,
    "state",
    [
        STATES[sample]
        for sample in samples_x_genes.index
    ],
)


samples_x_genes.insert(
    0,
    "study",
    "GSE191103",
)


samples_x_genes.insert(
    0,
    "sample",
    samples_x_genes.index,
)


samples_x_genes.to_csv(
    PROCESSED_DIR
    / "GSE191103_PA14_samples_x_4858.csv",
    index=False,
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 100)
print("AUDIT 37 COMPLETE")
print("=" * 100)

print(
    "Frozen Study #5 matrix:"
)

print(
    PROCESSED_DIR
    / "GSE191103_PA14_samples_x_4858.csv"
)

print()

print(
    "GSE191103 is now harmonised to the "
    "same frozen 4,858-gene CF-State universe."
)

print()
