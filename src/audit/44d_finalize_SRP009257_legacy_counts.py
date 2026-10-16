from pathlib import Path
import pandas as pd


COUNT_FILE = Path(
    "results/audit/expanded_biofilm/SRP009257/"
    "reprocessing/counts/counts_legacy_s1.txt"
)

MAPPING_FILE = Path(
    "results/audit/expanded_biofilm/GSE191103/"
    "PAO1_PA14_clean_1to1_mapping.csv"
)

FROZEN_FILE = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
)

MANIFEST_FILE = Path(
    "results/audit/expanded_biofilm/SRP009257/"
    "SRP009257_canonical_manifest.tsv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/SRP009257/finalization"
)

PROCESSED_DIR = Path(
    "data/processed/expanded_biofilm/SRP009257"
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
    "CF-STATE — Audit 44d: "
    "finalize SRP009257 legacy counts"
)
print("=" * 100)


SAMPLES = [
    "planktonic_P4_1",
    "planktonic_P4_2",
    "planktonic_P12_1",
    "planktonic_P12_2",
    "biofilm_B24_1",
    "biofilm_B24_2",
    "biofilm_B48_1",
    "biofilm_B48_2",
]


# ============================================================
# Load inputs
# ============================================================

counts_raw = pd.read_csv(
    COUNT_FILE,
    sep="\t",
    comment="#",
)

mapping = pd.read_csv(
    MAPPING_FILE
)

frozen = pd.read_csv(
    FROZEN_FILE
)

manifest = pd.read_csv(
    MANIFEST_FILE,
    sep="\t",
)


# ============================================================
# Frozen gene list
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
# featureCounts sample columns
# ============================================================

if "Geneid" not in counts_raw.columns:

    raise RuntimeError(
        "Geneid missing from legacy featureCounts table."
    )


metadata_columns = {
    "Geneid",
    "Chr",
    "Start",
    "End",
    "Strand",
    "Length",
}


bam_columns = [
    column
    for column in counts_raw.columns
    if column not in metadata_columns
]


if len(bam_columns) != 8:

    raise RuntimeError(
        "Expected 8 featureCounts sample columns, found "
        + str(len(bam_columns))
    )


rename_map = {}


for column in bam_columns:

    sample = Path(
        column
    ).name

    if sample.endswith(
        ".sorted.bam"
    ):

        sample = sample[
            :-len(
                ".sorted.bam"
            )
        ]

    rename_map[
        column
    ] = sample


counts_raw = counts_raw.rename(
    columns=rename_map
)


observed_samples = set(
    rename_map.values()
)

expected_samples = set(
    SAMPLES
)


if observed_samples != expected_samples:

    raise RuntimeError(
        "featureCounts sample names do not match expected samples. "
        "Observed: "
        + str(
            sorted(
                observed_samples
            )
        )
    )


# ============================================================
# Strict orthology checks
# ============================================================

required_mapping_columns = [
    "pa14_locus_tag",
    "canonical_pao1",
]


for column in required_mapping_columns:

    if column not in mapping.columns:

        raise RuntimeError(
            "Missing mapping column: "
            + column
        )


if mapping[
    "pa14_locus_tag"
].duplicated().any():

    raise RuntimeError(
        "Duplicate PA14 locus tags in strict mapping."
    )


if mapping[
    "canonical_pao1"
].duplicated().any():

    raise RuntimeError(
        "Duplicate PAO1 genes in strict mapping."
    )


# ============================================================
# PA14 counts -> canonical PAO1
# ============================================================

pa14_counts = (
    counts_raw[
        [
            "Geneid"
        ]
        + SAMPLES
    ]
    .copy()
    .rename(
        columns={
            "Geneid":
                "pa14_locus_tag"
        }
    )
)


if pa14_counts[
    "pa14_locus_tag"
].duplicated().any():

    raise RuntimeError(
        "Duplicate Geneid values in legacy counts."
    )


mapped = pa14_counts.merge(
    mapping[
        [
            "pa14_locus_tag",
            "canonical_pao1",
        ]
    ],
    on="pa14_locus_tag",
    how="inner",
    validate="one_to_one",
)


mapped_frozen = (
    mapped.loc[
        mapped[
            "canonical_pao1"
        ].isin(
            frozen_genes
        )
    ]
    .copy()
)


observed_frozen = set(
    mapped_frozen[
        "canonical_pao1"
    ]
)


missing_frozen = sorted(
    set(
        frozen_genes
    )
    - observed_frozen
)


print()
print("=" * 100)
print("FROZEN GENE COVERAGE")
print("=" * 100)

print(
    "Frozen genes expected: "
    + str(
        len(
            frozen_genes
        )
    )
)

print(
    "Frozen genes present: "
    + str(
        len(
            observed_frozen
        )
    )
)

print(
    "Frozen genes missing: "
    + str(
        len(
            missing_frozen
        )
    )
)


if missing_frozen:

    print()

    print(
        "Missing genes: "
        + ", ".join(
            missing_frozen
        )
    )

    raise RuntimeError(
        "Frozen gene universe is incomplete."
    )


# ============================================================
# Build 4858 x 8 count matrix
# ============================================================

counts = (
    mapped_frozen[
        [
            "canonical_pao1"
        ]
        + SAMPLES
    ]
    .set_index(
        "canonical_pao1"
    )
    .loc[
        frozen_genes
    ]
)


counts = counts.apply(
    pd.to_numeric,
    errors="raise",
)


if counts.shape != (
    4858,
    8,
):

    raise RuntimeError(
        "Unexpected count matrix shape: "
        + str(
            counts.shape
        )
    )


if counts.isna().any().any():

    raise RuntimeError(
        "Missing values in frozen count matrix."
    )


if (
    counts
    < 0
).any().any():

    raise RuntimeError(
        "Negative counts detected."
    )


# ============================================================
# Manifest
# ============================================================

manifest = manifest.set_index(
    "sample"
)


missing_manifest = sorted(
    set(
        SAMPLES
    )
    - set(
        manifest.index
    )
)


if missing_manifest:

    raise RuntimeError(
        "Samples missing from manifest: "
        + str(
            missing_manifest
        )
    )


# ============================================================
# Library summary
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
        4858
        - nonzero_genes
    )

    state = str(
        manifest.loc[
            sample,
            "state",
        ]
    )

    condition = str(
        manifest.loc[
            sample,
            "condition",
        ]
    )

    library_records.append(
        {
            "sample":
                sample,

            "state":
                state,

            "condition":
                condition,

            "total_counts_4858":
                total_counts,

            "nonzero_genes_4858":
                nonzero_genes,

            "zero_genes_4858":
                zero_genes,

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
print("LIBRARY SUMMARY")
print("=" * 100)

print(
    library_df.to_string(
        index=False
    )
)


library_df.to_csv(
    OUT_DIR
    / "SRP009257_library_summary.csv",
    index=False,
)


# ============================================================
# Rank-normalised profiles
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
    / "SRP009257_sample_correlations.csv"
)


# ============================================================
# Replicate QC
# ============================================================

condition_samples = {
    "P4": [
        "planktonic_P4_1",
        "planktonic_P4_2",
    ],
    "P12": [
        "planktonic_P12_1",
        "planktonic_P12_2",
    ],
    "B24": [
        "biofilm_B24_1",
        "biofilm_B24_2",
    ],
    "B48": [
        "biofilm_B48_1",
        "biofilm_B48_2",
    ],
}


replicate_records = []


for condition, sample_pair in condition_samples.items():

    sample_a = sample_pair[0]
    sample_b = sample_pair[1]

    rho = float(
        correlation.loc[
            sample_a,
            sample_b,
        ]
    )

    replicate_records.append(
        {
            "condition":
                condition,

            "sample_a":
                sample_a,

            "sample_b":
                sample_b,

            "spearman":
                rho,
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
    / "SRP009257_replicate_correlations.csv",
    index=False,
)


# ============================================================
# Condition centroid correlations
# ============================================================

condition_profiles = {}


for condition, condition_sample_names in condition_samples.items():

    condition_profiles[
        condition
    ] = (
        rank_matrix.loc[
            condition_sample_names
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
    / "SRP009257_condition_centroid_correlations.csv"
)


# ============================================================
# Save frozen matrices
# ============================================================

genes_x_samples = (
    counts
    .reset_index()
)


genes_x_samples.to_csv(
    PROCESSED_DIR
    / "SRP009257_PA14_counts_4858.csv",
    index=False,
)


samples_x_genes = (
    counts.T
    .copy()
)


samples_x_genes.insert(
    0,
    "condition",
    [
        str(
            manifest.loc[
                sample,
                "condition",
            ]
        )
        for sample
        in samples_x_genes.index
    ],
)


samples_x_genes.insert(
    0,
    "state",
    [
        str(
            manifest.loc[
                sample,
                "state",
            ]
        )
        for sample
        in samples_x_genes.index
    ],
)


samples_x_genes.insert(
    0,
    "study",
    "SRP009257",
)


samples_x_genes.insert(
    0,
    "sample",
    samples_x_genes.index,
)


samples_x_genes.to_csv(
    PROCESSED_DIR
    / "SRP009257_PA14_samples_x_4858.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 44d COMPLETE")
print("=" * 100)

print(
    "Frozen matrix: "
    + str(
        PROCESSED_DIR
        / "SRP009257_PA14_samples_x_4858.csv"
    )
)

print()

print(
    "SRP009257 is harmonised to the full "
    "4,858-gene CF-State universe."
)

print()
