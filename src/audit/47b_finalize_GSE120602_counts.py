from pathlib import Path
import re

import pandas as pd


RAW_DIR = Path(
    "data/raw/expanded_biofilm/GSE120602/published_counts"
)

FROZEN_FILE = Path(
    "data/processed/biofilm/clean_gene_universe.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE120602/finalization"
)

PROCESSED_DIR = Path(
    "data/processed/expanded_biofilm/GSE120602"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SAMPLES = [
    {
        "sample": "persister_cipro_1",
        "gsm": "GSM3405609",
        "condition": "persister_ciprofloxacin",
        "state": "persister_ciprofloxacin",
        "replicate": 1,
        "filename": "GSM3405609_plusCiproLarge1_readcount.txt.gz",
    },
    {
        "sample": "persister_cipro_2",
        "gsm": "GSM3405610",
        "condition": "persister_ciprofloxacin",
        "state": "persister_ciprofloxacin",
        "replicate": 2,
        "filename": "GSM3405610_plusCiproLarge2_readcount.txt.gz",
    },
    {
        "sample": "stationary_1",
        "gsm": "GSM3405611",
        "condition": "stationary",
        "state": "stationary",
        "replicate": 1,
        "filename": "GSM3405611_minusCiproLarge1_readcount.txt.gz",
    },
    {
        "sample": "stationary_2",
        "gsm": "GSM3405612",
        "condition": "stationary",
        "state": "stationary",
        "replicate": 2,
        "filename": "GSM3405612_minusCiproLarge2_readcount.txt.gz",
    },
    {
        "sample": "planktonic_1",
        "gsm": "GSM3405613",
        "condition": "planktonic",
        "state": "planktonic",
        "replicate": 1,
        "filename": "GSM3405613_PAPlanktonic1_readcount.txt.gz",
    },
    {
        "sample": "planktonic_2",
        "gsm": "GSM3405614",
        "condition": "planktonic",
        "state": "planktonic",
        "replicate": 2,
        "filename": "GSM3405614_PAPlanktonic2_readcount.txt.gz",
    },
    {
        "sample": "biofilm_1",
        "gsm": "GSM3405615",
        "condition": "biofilm",
        "state": "biofilm",
        "replicate": 1,
        "filename": "GSM3405615_PABiofilm1_readcount.txt.gz",
    },
    {
        "sample": "biofilm_2",
        "gsm": "GSM3405616",
        "condition": "biofilm",
        "state": "biofilm",
        "replicate": 2,
        "filename": "GSM3405616_PABiofilm2_readcount.txt.gz",
    },
]


print()
print("=" * 100)
print(
    "CF-STATE — Audit 47b: "
    "inspect and freeze GSE120602 published counts"
)
print("=" * 100)


# ============================================================
# Load frozen universe
# ============================================================

frozen = pd.read_csv(
    FROZEN_FILE
)

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


if len(set(frozen_genes)) != 4858:

    raise RuntimeError(
        "Frozen gene universe contains duplicates."
    )


frozen_set = set(
    frozen_genes
)


# ============================================================
# Read all published count files
# ============================================================

sample_series = {}
all_gene_sets = {}
inspection_records = []


for info in SAMPLES:

    path = (
        RAW_DIR
        / info["filename"]
    )

    if not path.exists():

        raise RuntimeError(
            "Missing file: "
            + str(path)
        )


    table = pd.read_csv(
        path,
        sep="\t",
        compression="gzip",
    )


    if table.shape[1] != 2:

        raise RuntimeError(
            info["gsm"]
            + " expected 2 columns, found "
            + str(table.shape[1])
        )


    gene_col = table.columns[0]
    count_col = table.columns[1]


    genes = (
        table[
            gene_col
        ]
        .astype(str)
    )


    counts = pd.to_numeric(
        table[
            count_col
        ],
        errors="raise",
    )


    if genes.duplicated().any():

        raise RuntimeError(
            info["gsm"]
            + " contains duplicate gene IDs."
        )


    if counts.isna().any():

        raise RuntimeError(
            info["gsm"]
            + " contains missing counts."
        )


    if (
        counts
        < 0
    ).any():

        raise RuntimeError(
            info["gsm"]
            + " contains negative counts."
        )


    gene_set = set(
        genes
    )


    all_gene_sets[
        info["sample"]
    ] = gene_set


    canonical_like = sum(
        bool(
            re.fullmatch(
                r"PA\d{4}",
                gene,
            )
        )
        for gene in gene_set
    )


    frozen_present = len(
        frozen_set
        & gene_set
    )


    frozen_missing = (
        4858
        - frozen_present
    )


    sample_series[
        info["sample"]
    ] = pd.Series(
        counts.to_numpy(),
        index=genes.to_numpy(),
        name=info["sample"],
    )


    inspection_records.append(
        {
            "sample": info["sample"],
            "gsm": info["gsm"],
            "condition": info["condition"],
            "rows": len(table),
            "unique_genes": len(gene_set),
            "PAxxxx_like": canonical_like,
            "frozen_present": frozen_present,
            "frozen_missing": frozen_missing,
            "total_counts_all_rows": int(
                counts.sum()
            ),
            "nonzero_all_rows": int(
                (
                    counts
                    > 0
                ).sum()
            ),
        }
    )


inspection = pd.DataFrame(
    inspection_records
)


print()
print("=" * 100)
print("PUBLISHED COUNT-TABLE STRUCTURE")
print("=" * 100)

print(
    inspection.to_string(
        index=False
    )
)


inspection.to_csv(
    OUT_DIR
    / "GSE120602_published_count_structure.csv",
    index=False,
)


# ============================================================
# Gene-set consistency
# ============================================================

sample_names = [
    info["sample"]
    for info in SAMPLES
]


reference_genes = all_gene_sets[
    sample_names[0]
]


gene_set_records = []


for sample in sample_names:

    current = all_gene_sets[
        sample
    ]

    gene_set_records.append(
        {
            "sample": sample,
            "same_as_first_sample":
                current == reference_genes,

            "genes_only_in_sample":
                len(
                    current
                    - reference_genes
                ),

            "genes_missing_vs_first":
                len(
                    reference_genes
                    - current
                ),
        }
    )


gene_set_df = pd.DataFrame(
    gene_set_records
)


print()
print("=" * 100)
print("GENE-SET CONSISTENCY")
print("=" * 100)

print(
    gene_set_df.to_string(
        index=False
    )
)


gene_set_df.to_csv(
    OUT_DIR
    / "GSE120602_gene_set_consistency.csv",
    index=False,
)


# ============================================================
# Frozen coverage
# ============================================================

if (
    inspection[
        "frozen_missing"
    ]
    > 0
).any():

    print()
    print("=" * 100)
    print("FROZEN GENE COVERAGE FAILURE")
    print("=" * 100)

    for info in SAMPLES:

        sample = info["sample"]

        missing = sorted(
            frozen_set
            - all_gene_sets[
                sample
            ]
        )

        if missing:

            print()
            print(
                sample
                + ": "
                + str(len(missing))
                + " missing"
            )

            print(
                ", ".join(
                    missing[:100]
                )
            )

    raise RuntimeError(
        "At least one sample does not contain "
        "all 4,858 frozen genes."
    )


print()
print("=" * 100)
print("FROZEN GENE COVERAGE")
print("=" * 100)

print(
    "Frozen genes expected: 4858"
)

print(
    "Frozen genes present in every sample: 4858"
)

print(
    "Frozen genes missing: 0"
)


# ============================================================
# Build 4858 x 8 count matrix
# ============================================================

counts = pd.DataFrame(
    {
        sample: sample_series[
            sample
        ].reindex(
            frozen_genes
        )

        for sample in sample_names
    },
    index=frozen_genes,
)


if counts.shape != (
    4858,
    8,
):

    raise RuntimeError(
        "Unexpected matrix shape: "
        + str(counts.shape)
    )


if counts.isna().any().any():

    raise RuntimeError(
        "Missing values after frozen-gene reindexing."
    )


counts = counts.astype(
    int
)


# ============================================================
# Metadata
# ============================================================

metadata = pd.DataFrame(
    [
        {
            "sample": info["sample"],
            "study": "GSE120602",
            "gsm": info["gsm"],
            "state": info["state"],
            "condition": info["condition"],
            "replicate": info["replicate"],
        }

        for info in SAMPLES
    ]
)


metadata.to_csv(
    OUT_DIR
    / "GSE120602_canonical_manifest.tsv",
    sep="\t",
    index=False,
)


# ============================================================
# Library summary
# ============================================================

library_records = []


for info in SAMPLES:

    sample = info["sample"]

    values = counts[
        sample
    ]


    total_counts = int(
        values.sum()
    )


    nonzero_genes = int(
        (
            values
            > 0
        ).sum()
    )


    zero_genes = (
        4858
        - nonzero_genes
    )


    library_records.append(
        {
            "sample": sample,
            "condition": info["condition"],
            "total_counts_4858": total_counts,
            "nonzero_genes_4858": nonzero_genes,
            "zero_genes_4858": zero_genes,
            "zero_fraction":
                zero_genes
                / 4858.0,
        }
    )


library_df = pd.DataFrame(
    library_records
)


print()
print("=" * 100)
print("FROZEN-UNIVERSE LIBRARY SUMMARY")
print("=" * 100)

print(
    library_df.to_string(
        index=False
    )
)


library_df.to_csv(
    OUT_DIR
    / "GSE120602_library_summary.csv",
    index=False,
)


# ============================================================
# Sample correlations
# ============================================================

rank_matrix = (
    counts.T
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


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
    .round(
        3
    )
    .to_string()
)


correlation.to_csv(
    OUT_DIR
    / "GSE120602_sample_correlations.csv"
)


# ============================================================
# Replicate QC
# ============================================================

condition_samples = {
    "persister_ciprofloxacin": [
        "persister_cipro_1",
        "persister_cipro_2",
    ],

    "stationary": [
        "stationary_1",
        "stationary_2",
    ],

    "planktonic": [
        "planktonic_1",
        "planktonic_2",
    ],

    "biofilm": [
        "biofilm_1",
        "biofilm_2",
    ],
}


replicate_records = []


for condition, pair in condition_samples.items():

    rho = float(
        correlation.loc[
            pair[0],
            pair[1],
        ]
    )


    replicate_records.append(
        {
            "condition": condition,
            "sample_a": pair[0],
            "sample_b": pair[1],
            "spearman": rho,
        }
    )


replicate_df = pd.DataFrame(
    replicate_records
)


print()
print("=" * 100)
print("WITHIN-CONDITION REPLICATE CORRELATIONS")
print("=" * 100)

print(
    replicate_df.to_string(
        index=False
    )
)


replicate_df.to_csv(
    OUT_DIR
    / "GSE120602_replicate_correlations.csv",
    index=False,
)


# ============================================================
# Condition-centroid correlations
# ============================================================

condition_profiles = {}


for condition, members in condition_samples.items():

    condition_profiles[
        condition
    ] = (
        rank_matrix.loc[
            members
        ]
        .mean(
            axis=0
        )
    )


condition_df = pd.DataFrame(
    condition_profiles
)


condition_correlation = (
    condition_df.corr(
        method="spearman"
    )
)


print()
print("=" * 100)
print("CONDITION-CENTROID CORRELATIONS")
print("=" * 100)

print(
    condition_correlation
    .round(
        3
    )
    .to_string()
)


condition_correlation.to_csv(
    OUT_DIR
    / "GSE120602_condition_centroid_correlations.csv"
)


# ============================================================
# Save matrices
# ============================================================

genes_x_samples = (
    counts
    .reset_index()
    .rename(
        columns={
            "index": "canonical_pao1"
        }
    )
)


genes_x_samples.to_csv(
    PROCESSED_DIR
    / "GSE120602_PAO1_counts_4858.csv",
    index=False,
)


samples_x_genes = (
    counts.T
    .reset_index()
    .rename(
        columns={
            "index": "sample"
        }
    )
)


samples_x_genes = metadata.merge(
    samples_x_genes,
    on="sample",
    how="inner",
    validate="one_to_one",
)


samples_x_genes.to_csv(
    PROCESSED_DIR
    / "GSE120602_PAO1_samples_x_4858.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 47b COMPLETE")
print("=" * 100)

print(
    "Frozen matrix:"
)

print(
    PROCESSED_DIR
    / "GSE120602_PAO1_samples_x_4858.csv"
)

print()

print(
    "If QC is satisfactory, GSE120602 is ready "
    "for frozen 251-gene validation."
)

print()
