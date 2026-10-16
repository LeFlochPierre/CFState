from pathlib import Path

import numpy as np 
import pandas as pd 
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


PUBLISHED_FILE = Path(
    "data/raw/expanded_biofilm/GSE299861/published_expression/"
    "GSE299861_PAO1_all.fpkm.txt.gz"
)

MAPPING_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "published_expression_mapping/"
    "Audit59b_high_confidence_PAO1DT_to_PA_mapping.csv"
)

MODEL_FILE = Path(
    "data/processed/cross_study_modeling/"
    "CFSTATE_binary_7studies_40samples_4853genes_ranked.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "published_vs_reprocessed"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


META_COLUMNS = [
    "study",
    "sample",
    "state",
    "label",
    "source_condition",
]


PUBLISHED_COLUMNS = {
    "biofilm PAO1-1_Count": "biofilm_PAO1_1",
    "biofilm PAO1-2_Count": "biofilm_PAO1_2",
    "biofilm PAO1-3_Count": "biofilm_PAO1_3",
    "planktonic PAO1-1_Count": "planktonic_PAO1_1",
    "planktonic PAO1-2_Count": "planktonic_PAO1_2",
    "planktonic PAO1-3_Count": "planktonic_PAO1_3",
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 59c: "
    "GSE299861 published counts vs reprocessed counts"
)
print("=" * 100)


# ============================================================
# Load files
# ============================================================

published = pd.read_csv(
    PUBLISHED_FILE,
    sep="\t",
    compression="gzip",
)

mapping = pd.read_csv(
    MAPPING_FILE
)

model = pd.read_csv(
    MODEL_FILE
)


for column in [
    "GeneId",
] + list(
    PUBLISHED_COLUMNS.keys()
):

    if column not in published.columns:

        raise RuntimeError(
            "Missing published column: "
            + column
        )


for column in [
    "PAO1DT_ID",
    "canonical_pao1",
]:

    if column not in mapping.columns:

        raise RuntimeError(
            "Missing mapping column: "
            + column
        )


for column in META_COLUMNS:

    if column not in model.columns:

        raise RuntimeError(
            "Missing model metadata column: "
            + column
        )


model_genes = [
    column
    for column in model.columns
    if column not in META_COLUMNS
]


if len(model_genes) != 4853:

    raise RuntimeError(
        "Expected 4,853 model genes."
    )


# ============================================================
# Restrict high-confidence mapping to modelling genes
# ============================================================

mapping[
    "PAO1DT_ID"
] = (
    mapping[
        "PAO1DT_ID"
    ]
    .astype(str)
)


mapping[
    "canonical_pao1"
] = (
    mapping[
        "canonical_pao1"
    ]
    .astype(str)
)


common_mapping = mapping.loc[
    mapping[
        "canonical_pao1"
    ].isin(
        model_genes
    )
].copy()


if common_mapping[
    "PAO1DT_ID"
].duplicated().any():

    raise RuntimeError(
        "Duplicate PAO1DT IDs."
    )


if common_mapping[
    "canonical_pao1"
].duplicated().any():

    raise RuntimeError(
        "Duplicate canonical PAO1 IDs."
    )


comparison_genes = (
    common_mapping[
        "canonical_pao1"
    ]
    .tolist()
)


if len(comparison_genes) != 4164:

    raise RuntimeError(
        "Expected 4,164 high-confidence common genes, found "
        + str(len(comparison_genes))
    )


print()
print("=" * 100)
print("COMPARISON UNIVERSE")
print("=" * 100)

print(
    "High-confidence genes used: "
    + str(len(comparison_genes))
)

print(
    "Coverage of 4,853-gene model universe: "
    + format(
        100.0
        * len(comparison_genes)
        / 4853.0,
        ".2f",
    )
    + "%"
)


# ============================================================
# Build authors' published count matrix
# ============================================================

published_mapped = published.merge(
    common_mapping[
        [
            "PAO1DT_ID",
            "canonical_pao1",
        ]
    ],
    left_on="GeneId",
    right_on="PAO1DT_ID",
    how="inner",
    validate="one_to_one",
)


if len(published_mapped) != 4164:

    raise RuntimeError(
        "Expected 4,164 mapped published rows."
    )


published_gene_by_sample = (
    published_mapped[
        [
            "canonical_pao1",
        ]
        + list(
            PUBLISHED_COLUMNS.keys()
        )
    ]
    .set_index(
        "canonical_pao1"
    )
    .rename(
        columns=PUBLISHED_COLUMNS
    )
)


published_gene_by_sample = (
    published_gene_by_sample.loc[
        comparison_genes
    ]
)


published_gene_by_sample = (
    published_gene_by_sample.apply(
        pd.to_numeric,
        errors="raise",
    )
)


published_samples = list(
    PUBLISHED_COLUMNS.values()
)


if list(
    published_gene_by_sample.columns
) != published_samples:

    raise RuntimeError(
        "Unexpected published sample order."
    )


# ============================================================
# Extract our reprocessed GSE299861 values
#
# The model file already contains within-sample ranks over
# the 4,853-gene universe.
# ============================================================

ours = model.loc[
    model[
        "study"
    ]
    == "GSE299861"
].copy()


if len(ours) != 6:

    raise RuntimeError(
        "Expected 6 GSE299861 samples."
    )


if set(
    ours[
        "sample"
    ]
) != set(
    published_samples
):

    raise RuntimeError(
        "Published and reprocessed sample names do not match."
    )


ours = (
    ours
    .set_index(
        "sample"
    )
    .loc[
        published_samples
    ]
)


# ============================================================
# Re-rank BOTH datasets within exactly the same 4,164 genes
#
# This makes the comparison independent of the 689 genes
# unavailable in the conservative published-data crosswalk.
# ============================================================

published_samples_x_genes = (
    published_gene_by_sample.T
)


published_rank = (
    published_samples_x_genes
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


ours_subset = (
    ours[
        comparison_genes
    ]
)


ours_rank = (
    ours_subset
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


# ============================================================
# Sample-by-sample published vs reprocessed agreement
# ============================================================

sample_records = []


for sample in published_samples:

    rho, _ = spearmanr(
        published_rank.loc[
            sample,
            comparison_genes,
        ].to_numpy(
            dtype=float
        ),
        ours_rank.loc[
            sample,
            comparison_genes,
        ].to_numpy(
            dtype=float
        ),
    )


    sample_records.append(
        {
            "sample":
                sample,

            "state":
                (
                    "biofilm"
                    if sample.startswith(
                        "biofilm"
                    )
                    else "planktonic"
                ),

            "published_total_mapped_counts":
                float(
                    published_samples_x_genes.loc[
                        sample
                    ].sum()
                ),

            "published_vs_reprocessed_rank_spearman":
                float(
                    rho
                ),
        }
    )


sample_agreement = pd.DataFrame(
    sample_records
)


sample_agreement.to_csv(
    OUT_DIR
    / "Audit59c_sample_agreement.csv",
    index=False,
)


print()
print("=" * 100)
print("SAMPLE-BY-SAMPLE AGREEMENT")
print("=" * 100)

print(
    sample_agreement.to_string(
        index=False
    )
)


# ============================================================
# Biofilm - planktonic effect vectors
# ============================================================

biofilm_samples = [
    "biofilm_PAO1_1",
    "biofilm_PAO1_2",
    "biofilm_PAO1_3",
]


planktonic_samples = [
    "planktonic_PAO1_1",
    "planktonic_PAO1_2",
    "planktonic_PAO1_3",
]


published_effect = (
    published_rank.loc[
        biofilm_samples
    ].mean(
        axis=0
    )
    -
    published_rank.loc[
        planktonic_samples
    ].mean(
        axis=0
    )
)


ours_effect = (
    ours_rank.loc[
        biofilm_samples
    ].mean(
        axis=0
    )
    -
    ours_rank.loc[
        planktonic_samples
    ].mean(
        axis=0
    )
)


effect_rho, _ = spearmanr(
    published_effect.to_numpy(
        dtype=float
    ),
    ours_effect.to_numpy(
        dtype=float
    ),
)


nonzero = (
    (
        published_effect
        != 0
    )
    &
    (
        ours_effect
        != 0
    )
)


effect_sign_agreement = float(
    (
        np.sign(
            published_effect.loc[
                nonzero
            ]
        )
        ==
        np.sign(
            ours_effect.loc[
                nonzero
            ]
        )
    ).mean()
)


effect_table = pd.DataFrame(
    {
        "canonical_pao1":
            comparison_genes,

        "published_effect":
            published_effect.loc[
                comparison_genes
            ].to_numpy(),

        "reprocessed_effect":
            ours_effect.loc[
                comparison_genes
            ].to_numpy(),
    }
)


effect_table[
    "same_direction"
] = (
    np.sign(
        effect_table[
            "published_effect"
        ]
    )
    ==
    np.sign(
        effect_table[
            "reprocessed_effect"
        ]
    )
)


effect_table.to_csv(
    OUT_DIR
    / "Audit59c_published_vs_reprocessed_effects.csv",
    index=False,
)


print()
print("=" * 100)
print("PUBLISHED VS REPROCESSED EFFECT VECTOR")
print("=" * 100)

print(
    "Spearman rho: "
    + format(
        float(effect_rho),
        ".6f",
    )
)

print(
    "Sign agreement among non-zero effects: "
    + format(
        100.0
        * effect_sign_agreement,
        ".2f",
    )
    + "%"
)


# ============================================================
# Build other-six-study consensus direction
#
# Re-rank every study within exactly the same 4,164 genes.
#
# GSE299861 is EXCLUDED from consensus construction.
# ============================================================

all_subset = (
    model[
        META_COLUMNS
        + comparison_genes
    ]
    .copy()
)


reranked_values = (
    all_subset[
        comparison_genes
    ]
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


reranked = pd.concat(
    [
        all_subset[
            META_COLUMNS
        ].reset_index(
            drop=True
        ),
        reranked_values.reset_index(
            drop=True
        ),
    ],
    axis=1,
)


other_studies = sorted(
    set(
        reranked[
            "study"
        ]
    )
    - {
        "GSE299861",
    }
)


other_effects = {}


for study in other_studies:

    subset = reranked.loc[
        reranked[
            "study"
        ]
        == study
    ]


    bio = subset.loc[
        subset[
            "label"
        ]
        == 1,
        comparison_genes,
    ]


    plank = subset.loc[
        subset[
            "label"
        ]
        == 0,
        comparison_genes,
    ]


    other_effects[
        study
    ] = (
        bio.mean(
            axis=0
        )
        -
        plank.mean(
            axis=0
        )
    )


other_effect_df = pd.DataFrame(
    other_effects
)


consensus = (
    other_effect_df.median(
        axis=1
    )
)


direction = (
    consensus.loc[
        comparison_genes
    ]
    .to_numpy(
        dtype=float
    )
)


direction_norm = float(
    np.linalg.norm(
        direction
    )
)


if direction_norm == 0:

    raise RuntimeError(
        "Consensus direction has zero norm."
    )


direction = (
    direction
    / direction_norm
)


# ============================================================
# Compare each GSE299861 effect with the six-study consensus
# ============================================================

published_consensus_rho, _ = spearmanr(
    published_effect.loc[
        comparison_genes
    ].to_numpy(
        dtype=float
    ),
    consensus.loc[
        comparison_genes
    ].to_numpy(
        dtype=float
    ),
)


ours_consensus_rho, _ = spearmanr(
    ours_effect.loc[
        comparison_genes
    ].to_numpy(
        dtype=float
    ),
    consensus.loc[
        comparison_genes
    ].to_numpy(
        dtype=float
    ),
)


print()
print("=" * 100)
print("EFFECT AGREEMENT WITH OTHER-SIX-STUDY CONSENSUS")
print("=" * 100)

print(
    "Published-count effect vs consensus rho: "
    + format(
        float(
            published_consensus_rho
        ),
        ".6f",
    )
)

print(
    "Reprocessed effect vs consensus rho: "
    + format(
        float(
            ours_consensus_rho
        ),
        ".6f",
    )
)


# ============================================================
# Consensus projection of AUTHORS' published counts
# ============================================================

published_scores = (
    published_rank[
        comparison_genes
    ]
    .to_numpy(
        dtype=float
    )
    @ direction
)


published_labels = np.array(
    [
        1
        if sample.startswith(
            "biofilm"
        )
        else 0
        for sample in published_samples
    ],
    dtype=int,
)


published_auc = float(
    roc_auc_score(
        published_labels,
        published_scores,
    )
)


published_bio_mean = float(
    published_scores[
        published_labels
        == 1
    ].mean()
)


published_plank_mean = float(
    published_scores[
        published_labels
        == 0
    ].mean()
)


# ============================================================
# Consensus projection of OUR reprocessed counts
# ============================================================

ours_scores = (
    ours_rank.loc[
        published_samples,
        comparison_genes,
    ]
    .to_numpy(
        dtype=float
    )
    @ direction
)


ours_labels = published_labels.copy()


ours_auc = float(
    roc_auc_score(
        ours_labels,
        ours_scores,
    )
)


ours_bio_mean = float(
    ours_scores[
        ours_labels
        == 1
    ].mean()
)


ours_plank_mean = float(
    ours_scores[
        ours_labels
        == 0
    ].mean()
)


# ============================================================
# Print projections
# ============================================================

projection_records = []


for sample, label, score in zip(
    published_samples,
    published_labels,
    published_scores,
):

    projection_records.append(
        {
            "pipeline":
                "authors_published_counts",

            "sample":
                sample,

            "state":
                (
                    "biofilm"
                    if label == 1
                    else "planktonic"
                ),

            "consensus_score":
                float(score),
        }
    )


for sample, label, score in zip(
    published_samples,
    ours_labels,
    ours_scores,
):

    projection_records.append(
        {
            "pipeline":
                "our_reprocessed_counts",

            "sample":
                sample,

            "state":
                (
                    "biofilm"
                    if label == 1
                    else "planktonic"
                ),

            "consensus_score":
                float(score),
        }
    )


projection_df = pd.DataFrame(
    projection_records
)


projection_df.to_csv(
    OUT_DIR
    / "Audit59c_consensus_projection_scores.csv",
    index=False,
)


print()
print("=" * 100)
print("OTHER-SIX-STUDY CONSENSUS PROJECTION")
print("=" * 100)

print()
print(
    "AUTHORS' PUBLISHED COUNTS"
)

print(
    "AUC: "
    + format(
        published_auc,
        ".6f",
    )
)

print(
    "Mean biofilm score: "
    + format(
        published_bio_mean,
        ".6f",
    )
)

print(
    "Mean planktonic score: "
    + format(
        published_plank_mean,
        ".6f",
    )
)

print(
    "Biofilm - planktonic: "
    + format(
        published_bio_mean
        - published_plank_mean,
        ".6f",
    )
)


print()
print(
    "OUR REPROCESSED COUNTS"
)

print(
    "AUC: "
    + format(
        ours_auc,
        ".6f",
    )
)

print(
    "Mean biofilm score: "
    + format(
        ours_bio_mean,
        ".6f",
    )
)

print(
    "Mean planktonic score: "
    + format(
        ours_plank_mean,
        ".6f",
    )
)

print(
    "Biofilm - planktonic: "
    + format(
        ours_bio_mean
        - ours_plank_mean,
        ".6f",
    )
)


print()
print("=" * 100)
print("INDIVIDUAL CONSENSUS SCORES")
print("=" * 100)

print(
    projection_df.to_string(
        index=False
    )
)


# ============================================================
# Save published mapped matrix
# ============================================================

published_output = (
    published_rank
    .reset_index()
    .rename(
        columns={
            "index":
                "sample"
        }
    )
)


published_output.insert(
    1,
    "study",
    "GSE299861",
)


published_output.insert(
    2,
    "state",
    [
        "biofilm"
        if sample.startswith(
            "biofilm"
        )
        else "planktonic"
        for sample in published_output[
            "sample"
        ]
    ],
)


published_output.to_csv(
    OUT_DIR
    / "GSE299861_authors_published_counts_4164genes_ranked.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 59c COMPLETE")
print("=" * 100)

print(
    "This audit compared only the 4,164 genes with "
    "high-confidence PAO1DT -> PA mappings."
)

print()

print(
    "If both published and reprocessed data show the "
    "same reversal against the six-study consensus, "
    "the reversal is not caused by our preprocessing."
)

print()
