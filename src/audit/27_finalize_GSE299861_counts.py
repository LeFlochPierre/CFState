from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

COUNT_DIR = Path(
    "results/audit/expanded_biofilm/"
    "GSE299861/reprocessing/counts"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/"
    "GSE299861/finalization"
)

PROCESSED_DIR = Path(
    "data/processed/expanded_biofilm/GSE299861"
)

PUBLISHED_FILE = Path(
    "data/raw/expanded_biofilm/GSE299861/"
    "GSE299861_PAO1_all.fpkm.txt.gz"
)

CLEAN_GENE_FILE = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
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
    "CF-STATE — Audit 27: "
    "finalize GSE299861 PAO1 counts"
)
print("=" * 100)


# ============================================================
# Sample order
# ============================================================

SAMPLES = [
    "biofilm_PAO1_1",
    "biofilm_PAO1_2",
    "biofilm_PAO1_3",
    "planktonic_PAO1_1",
    "planktonic_PAO1_2",
    "planktonic_PAO1_3",
]

STATES = {
    "biofilm_PAO1_1": "biofilm",
    "biofilm_PAO1_2": "biofilm",
    "biofilm_PAO1_3": "biofilm",
    "planktonic_PAO1_1": "planktonic",
    "planktonic_PAO1_2": "planktonic",
    "planktonic_PAO1_3": "planktonic",
}


# ============================================================
# 1. Inspect featureCounts assignment categories
# ============================================================

summary_file = (
    COUNT_DIR
    / "counts_s2.txt.summary"
)

summary = pd.read_csv(
    summary_file,
    sep="\t",
    index_col=0,
)


# Rename BAM columns to simple sample names
rename_columns = {}

for column in summary.columns:

    sample = (
        Path(column)
        .name
        .replace(
            ".sorted.bam",
            "",
        )
    )

    rename_columns[
        column
    ] = sample


summary = summary.rename(
    columns=rename_columns
)


print()
print("=" * 100)
print(
    "FEATURECOUNTS ASSIGNMENT BREAKDOWN"
)
print("=" * 100)


# Convert each column to percentages
summary_percent = (
    summary.copy()
    .astype(float)
)

for sample in summary_percent.columns:

    total = (
        summary_percent[
            sample
        ].sum()
    )

    summary_percent[
        sample
    ] = (
        100.0
        * summary_percent[
            sample
        ]
        / total
    )


# Print all categories
display_summary = (
    summary_percent
    .round(2)
)

print(
    display_summary.to_string()
)


summary_percent.to_csv(
    OUT_DIR
    / "featureCounts_s2_assignment_percent.csv"
)


# ============================================================
# Main reason reads were not assigned
# ============================================================

print()
print("=" * 100)
print(
    "DOMINANT UNASSIGNED CATEGORY PER SAMPLE"
)
print("=" * 100)


unassigned = (
    summary_percent
    .drop(
        index="Assigned",
        errors="ignore",
    )
)


dominant_records = []


for sample in SAMPLES:

    sample_unassigned = (
        unassigned[
            sample
        ]
        .sort_values(
            ascending=False
        )
    )

    top_category = (
        sample_unassigned.index[0]
    )

    top_percent = float(
        sample_unassigned.iloc[0]
    )

    assigned_percent = float(
        summary_percent.loc[
            "Assigned",
            sample,
        ]
    )

    dominant_records.append(
        {
            "sample":
                sample,

            "state":
                STATES[sample],

            "assigned_percent":
                assigned_percent,

            "dominant_unassigned_category":
                top_category,

            "dominant_unassigned_percent":
                top_percent,
        }
    )


dominant_df = pd.DataFrame(
    dominant_records
)


print(
    dominant_df.to_string(
        index=False
    )
)


dominant_df.to_csv(
    OUT_DIR
    / "dominant_unassigned_categories.csv",
    index=False,
)


# ============================================================
# 2. Load reverse-stranded counts
# ============================================================

count_file = (
    COUNT_DIR
    / "counts_s2.txt"
)


counts_raw = pd.read_csv(
    count_file,
    sep="\t",
    comment="#",
)


print()
print("=" * 100)
print("COUNT MATRIX")
print("=" * 100)

print(
    f"FeatureCounts rows: "
    f"{len(counts_raw)}"
)


gene_col = None

for candidate in [
    "Geneid",
    "GeneID",
    "gene",
]:

    if candidate in counts_raw.columns:

        gene_col = candidate
        break


if gene_col is None:

    raise RuntimeError(
        "Could not identify gene ID column "
        f"in {count_file}"
    )


# Identify BAM count columns
metadata_columns = {
    gene_col,
    "Chr",
    "Start",
    "End",
    "Strand",
    "Length",
}


bam_columns = [
    column
    for column in counts_raw.columns
    if column
    not in metadata_columns
]


counts = pd.DataFrame(
    index=(
        counts_raw[
            gene_col
        ]
        .astype(str)
    )
)


for column in bam_columns:

    sample = (
        Path(column)
        .name
        .replace(
            ".sorted.bam",
            "",
        )
    )

    counts[
        sample
    ] = pd.to_numeric(
        counts_raw[
            column
        ],
        errors="raise",
    ).to_numpy()


# Force desired sample ordering
counts = counts[
    SAMPLES
]


print(
    f"Unique genes: "
    f"{counts.index.nunique()}"
)

print(
    f"Samples: "
    f"{counts.shape[1]}"
)

print(
    f"Missing values: "
    f"{counts.isna().sum().sum()}"
)

print(
    f"Negative values: "
    f"{(counts < 0).sum().sum()}"
)

vals = counts.to_numpy()
non_int_count = (~np.isclose(vals, np.round(vals))).sum()
print(f"Non-integer values: {non_int_count}")


# ============================================================
# 3. Load frozen 4,858-gene universe
# ============================================================

clean = pd.read_csv(
    CLEAN_GENE_FILE
)


gene_column = None


for candidate in [
    "canonical_pao1",
    "GeneID",
    "gene",
    "locus_tag",
]:

    if candidate in clean.columns:

        gene_column = candidate
        break


if gene_column is None:

    # If file consists of a single column,
    # accept it automatically.
    if clean.shape[1] == 1:

        gene_column = (
            clean.columns[0]
        )

    else:

        raise RuntimeError(
            "Could not identify canonical gene "
            "column in clean_gene_universe.csv"
        )


clean_genes = (
    clean[
        gene_column
    ]
    .astype(str)
    .tolist()
)


clean_gene_set = set(
    clean_genes
)


count_gene_set = set(
    counts.index
)


missing = (
    clean_gene_set
    - count_gene_set
)


extra = (
    count_gene_set
    - clean_gene_set
)


print()
print("=" * 100)
print(
    "FROZEN CF-STATE GENE UNIVERSE"
)
print("=" * 100)

print(
    f"Expected genes: "
    f"{len(clean_genes)}"
)

overlap_count = len(clean_gene_set & count_gene_set)
print(f"Present in GSE299861: {overlap_count}")

print(
    f"Missing: "
    f"{len(missing)}"
)

print(
    f"Additional canonical genes: "
    f"{len(extra)}"
)


if missing:

    print()
    print(
        "ERROR: frozen genes are missing:"
    )

    print(
        sorted(
            missing
        )[:50]
    )

    raise RuntimeError(
        "Cannot freeze matrix because "
        "some CF-State genes are missing."
    )


# ============================================================
# Restrict exactly to the existing 4,858 genes
# ============================================================

counts_4858 = (
    counts.loc[
        clean_genes
    ]
    .copy()
)


# ============================================================
# 4. Library summaries
# ============================================================

library_records = []


for sample in SAMPLES:

    all_total = int(
        counts[
            sample
        ].sum()
    )

    frozen_total = int(
        counts_4858[
            sample
        ].sum()
    )

    nonzero = int(
        (
            counts_4858[
                sample
            ]
            > 0
        ).sum()
    )

    library_records.append(
        {
            "sample":
                sample,

            "state":
                STATES[sample],

            "all_5529_assigned":
                all_total,

            "frozen_4858_counts":
                frozen_total,

            "nonzero_4858_genes":
                nonzero,

            "zero_4858_genes":
                (
                    4858
                    - nonzero
                ),
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


# ============================================================
# 5. Compare against authors' published Count totals
#
# This is NOT gene-level validation because the published
# feature IDs are custom and unresolved.
#
# We only compare total count scale.
# ============================================================

published = pd.read_csv(
    PUBLISHED_FILE,
    sep="\t",
    compression="gzip",
)


published_mapping = {
    "biofilm PAO1-1_Count":
        "biofilm_PAO1_1",

    "biofilm PAO1-2_Count":
        "biofilm_PAO1_2",

    "biofilm PAO1-3_Count":
        "biofilm_PAO1_3",

    "planktonic PAO1-1_Count":
        "planktonic_PAO1_1",

    "planktonic PAO1-2_Count":
        "planktonic_PAO1_2",

    "planktonic PAO1-3_Count":
        "planktonic_PAO1_3",
}


published_totals = {}


for column, sample in (
    published_mapping.items()
):

    published_totals[
        sample
    ] = float(
        pd.to_numeric(
            published[
                column
            ],
            errors="coerce",
        ).sum()
    )


library_df[
    "published_count_total"
] = library_df[
    "sample"
].map(
    published_totals
)


library_df[
    "our_vs_published_ratio"
] = (
    library_df[
        "all_5529_assigned"
    ]
    /
    library_df[
        "published_count_total"
    ]
)


print()
print("=" * 100)
print(
    "OUR COUNTS VS PUBLISHED COUNT TOTALS"
)
print("=" * 100)

print(
    library_df[
        [
            "sample",
            "state",
            "all_5529_assigned",
            "published_count_total",
            "our_vs_published_ratio",
        ]
    ].to_string(
        index=False
    )
)


library_df.to_csv(
    OUT_DIR
    / "GSE299861_library_summary.csv",
    index=False,
)


# ============================================================
# 6. Within-sample percentile ranks
# ============================================================

rank_4858 = (
    counts_4858.T
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
    rank_4858.T
    .corr(
        method="spearman"
    )
)


print()
print("=" * 100)
print(
    "SAMPLE-SAMPLE SPEARMAN CORRELATION"
)
print("=" * 100)

print(
    correlation
    .round(3)
    .to_string()
)


correlation.to_csv(
    OUT_DIR
    / "GSE299861_rank_sample_correlations.csv"
)


# ============================================================
# Within-state versus between-state correlations
# ============================================================

pair_records = []


for i, sample_a in enumerate(
    SAMPLES
):

    for sample_b in SAMPLES[
        i + 1:
    ]:

        corr = float(
            correlation.loc[
                sample_a,
                sample_b,
            ]
        )

        same_state = (
            STATES[sample_a]
            ==
            STATES[sample_b]
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
                    corr,
            }
        )


pair_df = pd.DataFrame(
    pair_records
)


same_state_values = (
    pair_df.loc[
        pair_df[
            "same_state"
        ],
        "spearman",
    ]
)


different_state_values = (
    pair_df.loc[
        ~pair_df[
            "same_state"
        ],
        "spearman",
    ]
)


print()
print("=" * 100)
print(
    "WITHIN-STATE VS BETWEEN-STATE"
)
print("=" * 100)

print(
    f"Same-state median correlation: "
    f"{same_state_values.median():.4f}"
)

print(
    f"Different-state median correlation: "
    f"{different_state_values.median():.4f}"
)


pair_df.to_csv(
    OUT_DIR
    / "GSE299861_pairwise_correlations.csv",
    index=False,
)


# ============================================================
# 7. Save canonical 4,858-gene matrix
#
# genes x samples
# ============================================================

counts_out = (
    counts_4858
    .reset_index()
    .rename(
        columns={
            "index":
                "canonical_pao1"
        }
    )
)


counts_out.to_csv(
    PROCESSED_DIR
    / "GSE299861_PAO1_counts_4858.csv",
    index=False,
)


# samples x genes version
sample_matrix = (
    counts_4858.T
    .copy()
)

sample_matrix.insert(
    0,
    "state",
    [
        STATES[
            sample
        ]
        for sample
        in sample_matrix.index
    ],
)

sample_matrix.insert(
    0,
    "study",
    "GSE299861",
)

sample_matrix.insert(
    0,
    "sample",
    sample_matrix.index,
)


sample_matrix.to_csv(
    PROCESSED_DIR
    / "GSE299861_PAO1_samples_x_4858.csv",
    index=False,
)


# ============================================================
# Final
# ============================================================

print()
print("=" * 100)
print("AUDIT 27 COMPLETE")
print("=" * 100)

print(
    f"Frozen Study #4 matrix:"
)

print(
    PROCESSED_DIR
    / "GSE299861_PAO1_samples_x_4858.csv"
)

print()

print(
    "IMPORTANT:"
)

print(
    "The matrix has now been harmonised "
    "to the original frozen 4,858-gene "
    "CF-State universe."
)

print(
    "Do not yet use GSE299861 to redefine "
    "the original three-study signature."
)

print()
