from pathlib import Path

import numpy as np
import pandas as pd


PLATFORM_FILE = Path(
    "results/audit/expanded_biofilm/GSE12207/"
    "probe_mapping/GPL84_platform_with_PA_mapping.csv"
)

PROBE_MATRIX_FILE = Path(
    "results/audit/expanded_biofilm/GSE12207/"
    "probe_mapping/GSE12207_probe_expression_matrix.csv"
)

FROZEN_FILE = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
)

SIGNATURE_FILE = Path(
    "results/audit/core_biofilm_signature/"
    "core_biofilm_signature_fdr05.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE12207/"
    "finalization"
)

PROCESSED_DIR = Path(
    "data/processed/expanded_biofilm/GSE12207"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SAMPLE_INFO = {
    "GSM307112": ("planktonic_exponential", 1),
    "GSM307113": ("planktonic_exponential", 2),
    "GSM307114": ("planktonic_exponential", 3),

    "GSM307115": ("planktonic_stationary", 1),
    "GSM307116": ("planktonic_stationary", 2),
    "GSM307117": ("planktonic_stationary", 3),

    "GSM307118": ("colony_15h", 1),
    "GSM307119": ("colony_15h", 2),
    "GSM307120": ("colony_15h", 3),

    "GSM307121": ("colony_40h", 1),
    "GSM307122": ("colony_40h", 2),
    "GSM307123": ("colony_40h", 3),

    "GSM307124": ("biofilm_3d", 1),
    "GSM307125": ("biofilm_3d", 2),
    "GSM307126": ("biofilm_3d", 3),
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 51: "
    "finalize GSE12207 canonical gene matrix"
)
print("=" * 100)


# ============================================================
# Load
# ============================================================

platform = pd.read_csv(
    PLATFORM_FILE
)

probe_matrix = pd.read_csv(
    PROBE_MATRIX_FILE
)

frozen = pd.read_csv(
    FROZEN_FILE
)

signature = pd.read_csv(
    SIGNATURE_FILE
)


# ============================================================
# Frozen genes
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
        "Expected 4,858 frozen genes."
    )


frozen_set = set(
    frozen_genes
)


# ============================================================
# Frozen 251 signature
# ============================================================

if "canonical_pao1" not in signature.columns:

    raise RuntimeError(
        "Signature file missing canonical_pao1."
    )


signature_genes = (
    signature[
        "canonical_pao1"
    ]
    .astype(str)
    .tolist()
)


if len(signature_genes) != 251:

    raise RuntimeError(
        "Expected 251 frozen signature genes."
    )


signature_set = set(
    signature_genes
)


# ============================================================
# Canonical GPL84 mapping
# ============================================================

required_platform_columns = [
    "ID",
    "canonical_pao1",
]


for column in required_platform_columns:

    if column not in platform.columns:

        raise RuntimeError(
            "Missing platform column: "
            + column
        )


canonical = (
    platform.loc[
        platform[
            "canonical_pao1"
        ].notna(),
        [
            "ID",
            "canonical_pao1",
        ],
    ]
    .copy()
)


canonical[
    "canonical_pao1"
] = (
    canonical[
        "canonical_pao1"
    ]
    .astype(str)
)


canonical = (
    canonical.loc[
        canonical[
            "canonical_pao1"
        ].isin(
            frozen_set
        )
    ]
    .copy()
)


if canonical[
    "ID"
].duplicated().any():

    raise RuntimeError(
        "Duplicate probe IDs after filtering."
    )


if canonical[
    "canonical_pao1"
].duplicated().any():

    raise RuntimeError(
        "A frozen PAO1 gene has multiple GPL84 probes."
    )


observed_genes = set(
    canonical[
        "canonical_pao1"
    ]
)


missing_frozen = sorted(
    frozen_set
    - observed_genes
)


missing_signature = sorted(
    signature_set
    - observed_genes
)


print()
print("=" * 100)
print("CROSS-PLATFORM GENE COVERAGE")
print("=" * 100)

print(
    "Original frozen universe: "
    + str(
        len(
            frozen_genes
        )
    )
)

print(
    "GSE12207 measurable frozen genes: "
    + str(
        len(
            observed_genes
        )
    )
)

print(
    "Frozen genes unavailable on GPL84: "
    + str(
        len(
            missing_frozen
        )
    )
)

print(
    "Frozen 251 genes unavailable on GPL84: "
    + str(
        len(
            missing_signature
        )
    )
)


print()

print(
    "Missing frozen genes:"
)

print(
    ", ".join(
        missing_frozen
    )
)


print()

print(
    "Missing frozen 251-signature genes:"
)

if missing_signature:

    print(
        ", ".join(
            missing_signature
        )
    )

else:

    print(
        "None."
    )


# ============================================================
# Build probe -> gene matrix
# ============================================================

if "probe_id" not in probe_matrix.columns:

    raise RuntimeError(
        "Probe matrix missing probe_id."
    )


expression_columns = [
    column
    for column in probe_matrix.columns
    if column != "probe_id"
]


expected_samples = set(
    SAMPLE_INFO
)


if set(
    expression_columns
) != expected_samples:

    raise RuntimeError(
        "Unexpected GSM columns in probe matrix."
    )


mapped = canonical.merge(
    probe_matrix,
    left_on="ID",
    right_on="probe_id",
    how="inner",
    validate="one_to_one",
)


if len(mapped) != len(observed_genes):

    raise RuntimeError(
        "Unexpected mapped gene count."
    )


gene_matrix = (
    mapped[
        [
            "canonical_pao1"
        ]
        + sorted(
            SAMPLE_INFO
        )
    ]
    .set_index(
        "canonical_pao1"
    )
)


# Preserve original frozen-gene ordering,
# while omitting only the five GPL84-unavailable genes.

available_order = [
    gene
    for gene in frozen_genes
    if gene in observed_genes
]


gene_matrix = gene_matrix.loc[
    available_order
]


if gene_matrix.shape != (
    4853,
    15,
):

    raise RuntimeError(
        "Expected 4,853 x 15 matrix, found "
        + str(
            gene_matrix.shape
        )
    )


if gene_matrix.isna().any().any():

    raise RuntimeError(
        "Missing values in gene matrix."
    )


# ============================================================
# Metadata
# ============================================================

metadata_records = []


for gsm in sorted(
    SAMPLE_INFO
):

    condition, replicate = SAMPLE_INFO[
        gsm
    ]

    metadata_records.append(
        {
            "sample":
                gsm,

            "study":
                "GSE12207",

            "gsm":
                gsm,

            "condition":
                condition,

            "replicate":
                replicate,
        }
    )


metadata = pd.DataFrame(
    metadata_records
)


# ============================================================
# Gene-level QC
# ============================================================

rank_matrix = (
    gene_matrix.T
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


condition_samples = {}


for gsm, info in SAMPLE_INFO.items():

    condition = info[0]

    condition_samples.setdefault(
        condition,
        []
    ).append(
        gsm
    )


replicate_records = []


for condition, samples in condition_samples.items():

    pairwise = []


    for i in range(
        len(
            samples
        )
    ):

        for j in range(
            i + 1,
            len(
                samples
            )
        ):

            pairwise.append(
                float(
                    correlation.loc[
                        samples[i],
                        samples[j],
                    ]
                )
            )


    replicate_records.append(
        {
            "condition":
                condition,

            "median_spearman":
                float(
                    np.median(
                        pairwise
                    )
                ),

            "min_spearman":
                float(
                    np.min(
                        pairwise
                    )
                ),

            "max_spearman":
                float(
                    np.max(
                        pairwise
                    )
                ),
        }
    )


replicate_df = pd.DataFrame(
    replicate_records
)


condition_profiles = {}


for condition, samples in condition_samples.items():

    condition_profiles[
        condition
    ] = (
        rank_matrix.loc[
            samples
        ]
        .mean(
            axis=0
        )
    )


condition_profile_df = pd.DataFrame(
    condition_profiles
)


condition_correlation = (
    condition_profile_df.corr(
        method="spearman"
    )
)


print()
print("=" * 100)
print("GENE-LEVEL WITHIN-CONDITION REPLICATE QC")
print("=" * 100)

print(
    replicate_df.to_string(
        index=False
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


# ============================================================
# Save
# ============================================================

gene_matrix.reset_index().to_csv(
    PROCESSED_DIR
    / "GSE12207_PAO1_expression_4853_genes_x_samples.csv",
    index=False,
)


samples_x_genes = (
    gene_matrix.T
    .reset_index()
    .rename(
        columns={
            "index":
                "sample"
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
    / "GSE12207_PAO1_samples_x_4853.csv",
    index=False,
)


pd.DataFrame(
    {
        "canonical_pao1":
            available_order
    }
).to_csv(
    OUT_DIR
    / "GSE12207_cross_platform_gene_universe_4853.csv",
    index=False,
)


pd.DataFrame(
    {
        "canonical_pao1":
            missing_frozen
    }
).to_csv(
    OUT_DIR
    / "GSE12207_missing_frozen_genes.csv",
    index=False,
)


pd.DataFrame(
    {
        "canonical_pao1":
            missing_signature
    }
).to_csv(
    OUT_DIR
    / "GSE12207_missing_251_signature_genes.csv",
    index=False,
)


replicate_df.to_csv(
    OUT_DIR
    / "GSE12207_gene_level_replicate_QC.csv",
    index=False,
)


correlation.to_csv(
    OUT_DIR
    / "GSE12207_gene_level_sample_correlations.csv"
)


condition_correlation.to_csv(
    OUT_DIR
    / "GSE12207_condition_centroid_correlations.csv"
)


print()
print("=" * 100)
print("AUDIT 51 COMPLETE")
print("=" * 100)

print(
    "GSE12207 frozen cross-platform matrix:"
)

print(
    PROCESSED_DIR
    / "GSE12207_PAO1_samples_x_4853.csv"
)

print()

print(
    "No missing genes were imputed."
)

print()
