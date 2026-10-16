from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


MODEL_FILE = Path(
    "data/processed/cross_study_modeling/"
    "CFSTATE_binary_7studies_40samples_4853genes_ranked.csv"
)

PUBLISHED_RANK_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "published_vs_reprocessed/"
    "GSE299861_authors_published_counts_4164genes_ranked.csv"
)

EFFECT_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "published_vs_reprocessed/"
    "Audit59c_published_vs_reprocessed_effects.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "robust_consensus_reversal"
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


print()
print("=" * 100)
print(
    "CF-STATE — Audit 60: "
    "is GSE299861 broadly reversed against robust consensus genes?"
)
print("=" * 100)


# ============================================================
# Load
# ============================================================

model = pd.read_csv(
    MODEL_FILE
)

published = pd.read_csv(
    PUBLISHED_RANK_FILE
)

effect_reference = pd.read_csv(
    EFFECT_FILE
)


required_effect_columns = [
    "canonical_pao1",
    "published_effect",
]


for column in required_effect_columns:

    if column not in effect_reference.columns:

        raise RuntimeError(
            "Missing effect column: "
            + column
        )


comparison_genes = (
    effect_reference[
        "canonical_pao1"
    ]
    .astype(str)
    .tolist()
)


if len(comparison_genes) != 4164:

    raise RuntimeError(
        "Expected 4,164 comparison genes, found "
        + str(len(comparison_genes))
    )


for gene in comparison_genes:

    if gene not in model.columns:

        raise RuntimeError(
            "Gene missing from model matrix: "
            + gene
        )

    if gene not in published.columns:

        raise RuntimeError(
            "Gene missing from published matrix: "
            + gene
        )


# ============================================================
# Re-rank all binary samples within exactly these 4,164 genes
# ============================================================

model_subset = model[
    META_COLUMNS
    + comparison_genes
].copy()


reranked_values = (
    model_subset[
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
        model_subset[
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


# ============================================================
# Build effect vector for each OTHER study
# ============================================================

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


if len(other_studies) != 6:

    raise RuntimeError(
        "Expected six non-GSE299861 studies."
    )


effect_vectors = {}


for study in other_studies:

    subset = reranked.loc[
        reranked[
            "study"
        ]
        == study
    ]


    biofilm = subset.loc[
        subset[
            "label"
        ]
        == 1,
        comparison_genes,
    ]


    planktonic = subset.loc[
        subset[
            "label"
        ]
        == 0,
        comparison_genes,
    ]


    effect_vectors[
        study
    ] = (
        biofilm.mean(
            axis=0
        )
        -
        planktonic.mean(
            axis=0
        )
    )


effects = pd.DataFrame(
    effect_vectors
)


effects = effects.loc[
    comparison_genes
]


# ============================================================
# Six-study consensus statistics
# ============================================================

n_positive = (
    effects
    > 0
).sum(
    axis=1
)


n_negative = (
    effects
    < 0
).sum(
    axis=1
)


n_zero = (
    effects
    == 0
).sum(
    axis=1
)


majority_count = pd.concat(
    [
        n_positive,
        n_negative,
    ],
    axis=1,
).max(
    axis=1
)


consensus_sign = pd.Series(
    0,
    index=comparison_genes,
    dtype=int,
)


consensus_sign.loc[
    n_positive
    > n_negative
] = 1


consensus_sign.loc[
    n_negative
    > n_positive
] = -1


median_effect = effects.median(
    axis=1
)


mean_effect = effects.mean(
    axis=1
)


# ============================================================
# Published GSE299861 effect
# ============================================================

published_effect = (
    effect_reference
    .set_index(
        "canonical_pao1"
    )[
        "published_effect"
    ]
    .reindex(
        comparison_genes
    )
)


if published_effect.isna().any():

    raise RuntimeError(
        "Missing published GSE299861 effects."
    )


# ============================================================
# Consensus table
# ============================================================

consensus_table = pd.DataFrame(
    {
        "canonical_pao1":
            comparison_genes,

        "n_positive_other6":
            n_positive.loc[
                comparison_genes
            ].to_numpy(),

        "n_negative_other6":
            n_negative.loc[
                comparison_genes
            ].to_numpy(),

        "n_zero_other6":
            n_zero.loc[
                comparison_genes
            ].to_numpy(),

        "majority_count_other6":
            majority_count.loc[
                comparison_genes
            ].to_numpy(),

        "consensus_sign":
            consensus_sign.loc[
                comparison_genes
            ].to_numpy(),

        "median_effect_other6":
            median_effect.loc[
                comparison_genes
            ].to_numpy(),

        "mean_effect_other6":
            mean_effect.loc[
                comparison_genes
            ].to_numpy(),

        "GSE299861_published_effect":
            published_effect.loc[
                comparison_genes
            ].to_numpy(),
    }
)


consensus_table[
    "GSE299861_sign"
] = np.sign(
    consensus_table[
        "GSE299861_published_effect"
    ]
).astype(int)


consensus_table[
    "agrees_with_consensus"
] = (
    consensus_table[
        "GSE299861_sign"
    ]
    ==
    consensus_table[
        "consensus_sign"
    ]
)


consensus_table[
    "opposes_consensus"
] = (
    (
        consensus_table[
            "GSE299861_sign"
        ]
        != 0
    )
    &
    (
        consensus_table[
            "consensus_sign"
        ]
        != 0
    )
    &
    (
        consensus_table[
            "GSE299861_sign"
        ]
        != consensus_table[
            "consensus_sign"
        ]
    )
)


consensus_table.to_csv(
    OUT_DIR
    / "Audit60_gene_consensus_and_GSE299861_effect.csv",
    index=False,
)


# ============================================================
# Define increasingly strict OTHER-SIX consensus sets
# ============================================================

sets = {
    "majority_4of6":
        (
            majority_count
            >= 4
        )
        &
        (
            consensus_sign
            != 0
        ),

    "robust_5of6":
        (
            majority_count
            >= 5
        )
        &
        (
            consensus_sign
            != 0
        ),

    "unanimous_6of6":
        (
            majority_count
            == 6
        )
        &
        (
            consensus_sign
            != 0
        ),

    "robust_5of6_median_abs_ge_0.02":
        (
            majority_count
            >= 5
        )
        &
        (
            median_effect.abs()
            >= 0.02
        ),

    "robust_5of6_median_abs_ge_0.05":
        (
            majority_count
            >= 5
        )
        &
        (
            median_effect.abs()
            >= 0.05
        ),

    "unanimous_6of6_median_abs_ge_0.02":
        (
            majority_count
            == 6
        )
        &
        (
            median_effect.abs()
            >= 0.02
        ),

    "unanimous_6of6_median_abs_ge_0.05":
        (
            majority_count
            == 6
        )
        &
        (
            median_effect.abs()
            >= 0.05
        ),
}


# ============================================================
# Published GSE299861 sample labels
# ============================================================

published_samples = (
    published[
        "sample"
    ]
    .astype(str)
    .tolist()
)


published_labels = (
    published[
        "state"
    ]
    .eq(
        "biofilm"
    )
    .astype(int)
    .to_numpy()
)


if int(
    published_labels.sum()
) != 3:

    raise RuntimeError(
        "Expected three published biofilm samples."
    )


# ============================================================
# Evaluate each consensus set
# ============================================================

set_records = []


for set_name, mask in sets.items():

    genes = (
        mask.loc[
            comparison_genes
        ]
    )


    selected_genes = [
        gene
        for gene in comparison_genes
        if bool(
            genes.loc[
                gene
            ]
        )
    ]


    n_genes = len(
        selected_genes
    )


    if n_genes == 0:

        continue


    consensus_values = (
        median_effect.loc[
            selected_genes
        ]
        .to_numpy(
            dtype=float
        )
    )


    gse_effect_values = (
        published_effect.loc[
            selected_genes
        ]
        .to_numpy(
            dtype=float
        )
    )


    rho, _ = spearmanr(
        consensus_values,
        gse_effect_values,
    )


    valid_sign = (
        (
            consensus_values
            != 0
        )
        &
        (
            gse_effect_values
            != 0
        )
    )


    if int(
        valid_sign.sum()
    ) == 0:

        sign_agreement = np.nan
        sign_opposition = np.nan

    else:

        sign_agreement = float(
            (
                np.sign(
                    consensus_values[
                        valid_sign
                    ]
                )
                ==
                np.sign(
                    gse_effect_values[
                        valid_sign
                    ]
                )
            ).mean()
        )


        sign_opposition = (
            1.0
            - sign_agreement
        )


    # --------------------------------------------------------
    # Projection using only this consensus gene set
    # --------------------------------------------------------

    direction = consensus_values.copy()


    norm = float(
        np.linalg.norm(
            direction
        )
    )


    if norm == 0:

        raise RuntimeError(
            "Zero direction norm for "
            + set_name
        )


    direction = (
        direction
        / norm
    )


    X = (
        published[
            selected_genes
        ]
        .to_numpy(
            dtype=float
        )
    )


    scores = (
        X
        @ direction
    )


    auc = float(
        roc_auc_score(
            published_labels,
            scores,
        )
    )


    biofilm_mean = float(
        scores[
            published_labels
            == 1
        ].mean()
    )


    planktonic_mean = float(
        scores[
            published_labels
            == 0
        ].mean()
    )


    set_records.append(
        {
            "gene_set":
                set_name,

            "n_genes":
                n_genes,

            "effect_spearman":
                float(
                    rho
                ),

            "sign_agreement":
                sign_agreement,

            "sign_opposition":
                sign_opposition,

            "projection_auc":
                auc,

            "mean_biofilm_score":
                biofilm_mean,

            "mean_planktonic_score":
                planktonic_mean,

            "bio_minus_planktonic":
                (
                    biofilm_mean
                    - planktonic_mean
                ),
        }
    )


set_summary = pd.DataFrame(
    set_records
)


set_summary.to_csv(
    OUT_DIR
    / "Audit60_robust_consensus_set_results.csv",
    index=False,
)


print()
print("=" * 100)
print("ROBUST OTHER-SIX CONSENSUS SETS")
print("=" * 100)

print(
    set_summary.to_string(
        index=False
    )
)


# ============================================================
# Which genes contribute most strongly to reversal?
#
# Negative product:
#   other-six consensus says one direction
#   GSE299861 moves strongly in the opposite direction.
# ============================================================

consensus_table[
    "direction_product"
] = (
    consensus_table[
        "median_effect_other6"
    ]
    *
    consensus_table[
        "GSE299861_published_effect"
    ]
)


opposing = (
    consensus_table.loc[
        consensus_table[
            "direction_product"
        ]
        < 0
    ]
    .copy()
)


opposing[
    "opposition_strength"
] = (
    -opposing[
        "direction_product"
    ]
)


opposing = opposing.sort_values(
    "opposition_strength",
    ascending=False,
)


opposing.to_csv(
    OUT_DIR
    / "Audit60_genes_driving_reversal.csv",
    index=False,
)


print()
print("=" * 100)
print("TOP 40 GENES DRIVING THE REVERSAL")
print("=" * 100)

display_columns = [
    "canonical_pao1",
    "n_positive_other6",
    "n_negative_other6",
    "median_effect_other6",
    "GSE299861_published_effect",
    "opposition_strength",
]


print(
    opposing[
        display_columns
    ]
    .head(
        40
    )
    .to_string(
        index=False
    )
)


# ============================================================
# Is reversal concentrated in just a few genes?
# ============================================================

total_opposition = float(
    opposing[
        "opposition_strength"
    ].sum()
)


concentration_records = []


for top_n in [
    10,
    25,
    50,
    100,
    250,
    500,
]:

    n_use = min(
        top_n,
        len(opposing),
    )


    contribution = float(
        opposing[
            "opposition_strength"
        ]
        .head(
            n_use
        )
        .sum()
    )


    fraction = (
        contribution
        / total_opposition
        if total_opposition > 0
        else np.nan
    )


    concentration_records.append(
        {
            "top_n":
                n_use,

            "fraction_of_total_opposition":
                fraction,
        }
    )


concentration = pd.DataFrame(
    concentration_records
)


concentration.to_csv(
    OUT_DIR
    / "Audit60_reversal_concentration.csv",
    index=False,
)


print()
print("=" * 100)
print("REVERSAL CONCENTRATION")
print("=" * 100)

print(
    concentration.to_string(
        index=False
    )
)


# ============================================================
# Simple overall summary
# ============================================================

robust_5_mask = (
    majority_count
    >= 5
)


robust_5_genes = [
    gene
    for gene in comparison_genes
    if bool(
        robust_5_mask.loc[
            gene
        ]
    )
]


robust_other_sign = np.sign(
    median_effect.loc[
        robust_5_genes
    ].to_numpy(
        dtype=float
    )
)


robust_gse_sign = np.sign(
    published_effect.loc[
        robust_5_genes
    ].to_numpy(
        dtype=float
    )
)


valid = (
    (
        robust_other_sign
        != 0
    )
    &
    (
        robust_gse_sign
        != 0
    )
)


robust_agreement = float(
    (
        robust_other_sign[
            valid
        ]
        ==
        robust_gse_sign[
            valid
        ]
    ).mean()
)


print()
print("=" * 100)
print("AUDIT 60 SUMMARY")
print("=" * 100)

print(
    "Other-six studies: "
    + ", ".join(
        other_studies
    )
)

print(
    "High-confidence comparison genes: "
    + str(
        len(
            comparison_genes
        )
    )
)

print(
    "Genes with >=5/6 other studies agreeing in direction: "
    + str(
        len(
            robust_5_genes
        )
    )
)

print(
    "GSE299861 sign agreement on those genes: "
    + format(
        100.0
        * robust_agreement,
        ".2f",
    )
    + "%"
)

print()

print(
    "Interpret the robust-set projection AUCs above:"
)

print(
    "If they remain near 0 even for 5/6 and 6/6 "
    "consensus genes, the reversal is broad rather "
    "than being caused by a small unstable subset."
)

print()
