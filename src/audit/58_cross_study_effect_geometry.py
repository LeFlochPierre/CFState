from pathlib import Path 
from itertools import combinations
import numpy as np
import pandas as pd

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


DATA_FILE = Path(
    "data/processed/cross_study_modeling/"
    "CFSTATE_binary_7studies_40samples_4853genes_ranked.csv"
)

OUT_DIR = Path(
    "results/audit/cross_study_modeling/"
    "Audit58_effect_geometry"
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
    "CF-STATE — Audit 58: "
    "cross-study biofilm-effect geometry"
)
print("=" * 100)


# ============================================================
# Load
# ============================================================

data = pd.read_csv(
    DATA_FILE
)


for column in META_COLUMNS:

    if column not in data.columns:

        raise RuntimeError(
            "Missing metadata column: "
            + column
        )


genes = [
    column
    for column in data.columns
    if column not in META_COLUMNS
]


if len(genes) != 4853:

    raise RuntimeError(
        "Expected 4,853 genes, found "
        + str(len(genes))
    )


studies = sorted(
    data[
        "study"
    ].unique()
)


if len(studies) != 7:

    raise RuntimeError(
        "Expected 7 study families."
    )


# ============================================================
# Calculate one biofilm - planktonic effect vector per study
# ============================================================

effect_vectors = {}


composition_records = []


for study in studies:

    subset = data.loc[
        data[
            "study"
        ]
        == study
    ]


    biofilm = subset.loc[
        subset[
            "label"
        ]
        == 1,
        genes,
    ]


    planktonic = subset.loc[
        subset[
            "label"
        ]
        == 0,
        genes,
    ]


    if len(biofilm) == 0 or len(planktonic) == 0:

        raise RuntimeError(
            "Missing one class in "
            + study
        )


    effect = (
        biofilm.mean(
            axis=0
        )
        -
        planktonic.mean(
            axis=0
        )
    )


    effect_vectors[
        study
    ] = effect


    composition_records.append(
        {
            "study":
                study,

            "n_biofilm":
                len(
                    biofilm
                ),

            "n_planktonic":
                len(
                    planktonic
                ),

            "mean_absolute_effect":
                float(
                    effect.abs().mean()
                ),

            "median_absolute_effect":
                float(
                    effect.abs().median()
                ),

            "fraction_positive":
                float(
                    (
                        effect
                        > 0
                    ).mean()
                ),

            "fraction_negative":
                float(
                    (
                        effect
                        < 0
                    ).mean()
                ),
        }
    )


effects = pd.DataFrame(
    effect_vectors
)


effects.index.name = (
    "canonical_pao1"
)


composition = pd.DataFrame(
    composition_records
)


effects.reset_index().to_csv(
    OUT_DIR
    / "Audit58_study_effect_vectors.csv",
    index=False,
)


composition.to_csv(
    OUT_DIR
    / "Audit58_study_effect_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("STUDY EFFECT MAGNITUDES")
print("=" * 100)

print(
    composition.to_string(
        index=False
    )
)


# ============================================================
# Pairwise effect-vector similarity
# ============================================================

pair_records = []


for study_a, study_b in combinations(
    studies,
    2,
):

    effect_a = (
        effects[
            study_a
        ]
        .to_numpy(
            dtype=float
        )
    )


    effect_b = (
        effects[
            study_b
        ]
        .to_numpy(
            dtype=float
        )
    )


    spearman_rho, _ = spearmanr(
        effect_a,
        effect_b,
    )


    pearson_r = float(
        np.corrcoef(
            effect_a,
            effect_b,
        )[0, 1]
    )


    denominator = (
        np.linalg.norm(
            effect_a
        )
        *
        np.linalg.norm(
            effect_b
        )
    )


    if denominator == 0:

        cosine = np.nan

    else:

        cosine = float(
            np.dot(
                effect_a,
                effect_b,
            )
            / denominator
        )


    nonzero = (
        (
            effect_a
            != 0
        )
        &
        (
            effect_b
            != 0
        )
    )


    if int(
        nonzero.sum()
    ) == 0:

        sign_agreement = np.nan

    else:

        sign_agreement = float(
            (
                np.sign(
                    effect_a[
                        nonzero
                    ]
                )
                ==
                np.sign(
                    effect_b[
                        nonzero
                    ]
                )
            ).mean()
        )


    pair_records.append(
        {
            "study_a":
                study_a,

            "study_b":
                study_b,

            "spearman_rho":
                float(
                    spearman_rho
                ),

            "pearson_r":
                pearson_r,

            "cosine_similarity":
                cosine,

            "sign_agreement":
                sign_agreement,
        }
    )


pairwise = pd.DataFrame(
    pair_records
)


pairwise.to_csv(
    OUT_DIR
    / "Audit58_pairwise_effect_similarity.csv",
    index=False,
)


print()
print("=" * 100)
print("PAIRWISE STUDY EFFECT SIMILARITY")
print("=" * 100)

print(
    pairwise
    .sort_values(
        "spearman_rho",
        ascending=False,
    )
    .to_string(
        index=False
    )
)


# ============================================================
# Similarity matrices
# ============================================================

spearman_matrix = (
    effects.corr(
        method="spearman"
    )
)


pearson_matrix = (
    effects.corr(
        method="pearson"
    )
)


spearman_matrix.to_csv(
    OUT_DIR
    / "Audit58_effect_spearman_matrix.csv"
)


pearson_matrix.to_csv(
    OUT_DIR
    / "Audit58_effect_pearson_matrix.csv"
)


print()
print("=" * 100)
print("SPEARMAN EFFECT CORRELATION MATRIX")
print("=" * 100)

print(
    spearman_matrix
    .round(
        3
    )
    .to_string()
)


# ============================================================
# Leave-one-study-out median consensus direction
#
# For each held-out study:
#
# 1. Use ONLY the other six studies.
# 2. Calculate the median biofilm effect for each gene.
# 3. Project held-out samples onto that vector.
# 4. Evaluate ordering with AUC.
#
# No model fitting, probability calibration or threshold.
# ============================================================

projection_records = []

consensus_similarity_records = []


for held_out_study in studies:

    training_studies = [
        study
        for study in studies
        if study != held_out_study
    ]


    training_effects = effects[
        training_studies
    ]


    consensus = (
        training_effects.median(
            axis=1
        )
    )


    held_out_effect = (
        effects[
            held_out_study
        ]
    )


    consensus_rho, _ = spearmanr(
        consensus.to_numpy(
            dtype=float
        ),
        held_out_effect.to_numpy(
            dtype=float
        ),
    )


    nonzero = (
        (
            consensus
            != 0
        )
        &
        (
            held_out_effect
            != 0
        )
    )


    if int(
        nonzero.sum()
    ) == 0:

        sign_agreement = np.nan

    else:

        sign_agreement = float(
            (
                np.sign(
                    consensus.loc[
                        nonzero
                    ]
                )
                ==
                np.sign(
                    held_out_effect.loc[
                        nonzero
                    ]
                )
            ).mean()
        )


    consensus_similarity_records.append(
        {
            "held_out_study":
                held_out_study,

            "effect_spearman_vs_other6_median":
                float(
                    consensus_rho
                ),

            "sign_agreement_vs_other6_median":
                sign_agreement,

            "consensus_nonzero_genes":
                int(
                    (
                        consensus
                        != 0
                    ).sum()
                ),
        }
    )


    test = data.loc[
        data[
            "study"
        ]
        == held_out_study
    ].copy()


    X = (
        test[
            genes
        ]
        .to_numpy(
            dtype=float
        )
    )


    direction = (
        consensus
        .reindex(
            genes
        )
        .to_numpy(
            dtype=float
        )
    )


    # Normalise only the direction vector.
    # This changes scale, not sample ordering.

    direction_norm = float(
        np.linalg.norm(
            direction
        )
    )


    if direction_norm == 0:

        raise RuntimeError(
            "Zero consensus direction for "
            + held_out_study
        )


    direction = (
        direction
        / direction_norm
    )


    scores = (
        X
        @ direction
    )


    labels = (
        test[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )


    auc = float(
        roc_auc_score(
            labels,
            scores,
        )
    )


    mean_biofilm = float(
        scores[
            labels
            == 1
        ].mean()
    )


    mean_planktonic = float(
        scores[
            labels
            == 0
        ].mean()
    )


    score_difference = (
        mean_biofilm
        - mean_planktonic
    )


    for sample, state, label, score in zip(
        test[
            "sample"
        ],
        test[
            "state"
        ],
        labels,
        scores,
    ):

        projection_records.append(
            {
                "held_out_study":
                    held_out_study,

                "sample":
                    sample,

                "state":
                    state,

                "label":
                    int(
                        label
                    ),

                "loso_consensus_score":
                    float(
                        score
                    ),

                "held_out_auc":
                    auc,

                "held_out_mean_biofilm":
                    mean_biofilm,

                "held_out_mean_planktonic":
                    mean_planktonic,

                "held_out_bio_minus_planktonic":
                    score_difference,
            }
        )


projection = pd.DataFrame(
    projection_records
)


consensus_similarity = pd.DataFrame(
    consensus_similarity_records
)


projection.to_csv(
    OUT_DIR
    / "Audit58_loso_consensus_projection_scores.csv",
    index=False,
)


consensus_similarity.to_csv(
    OUT_DIR
    / "Audit58_loso_consensus_effect_similarity.csv",
    index=False,
)


# ============================================================
# One row per held-out study
# ============================================================

projection_summary = (
    projection[
        [
            "held_out_study",
            "held_out_auc",
            "held_out_mean_biofilm",
            "held_out_mean_planktonic",
            "held_out_bio_minus_planktonic",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        "held_out_study"
    )
)


projection_summary = (
    projection_summary.merge(
        consensus_similarity,
        on="held_out_study",
        how="left",
        validate="one_to_one",
    )
)


projection_summary.to_csv(
    OUT_DIR
    / "Audit58_loso_consensus_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("LEAVE-ONE-STUDY-OUT CONSENSUS DIRECTION")
print("=" * 100)

print(
    projection_summary.to_string(
        index=False
    )
)


# ============================================================
# Overall consensus characteristics
# ============================================================

all_study_median = (
    effects.median(
        axis=1
    )
)


all_study_mean = (
    effects.mean(
        axis=1
    )
)


majority_positive = (
    (
        effects
        > 0
    ).sum(
        axis=1
    )
    >= 4
)


majority_negative = (
    (
        effects
        < 0
    ).sum(
        axis=1
    )
    >= 4
)


majority_direction = (
    majority_positive
    |
    majority_negative
)


unanimous_positive = (
    effects
    > 0
).all(
    axis=1
)


unanimous_negative = (
    effects
    < 0
).all(
    axis=1
)


unanimous_direction = (
    unanimous_positive
    |
    unanimous_negative
)


consensus_gene_table = pd.DataFrame(
    {
        "canonical_pao1":
            genes,

        "median_effect_7studies":
            all_study_median.loc[
                genes
            ].to_numpy(),

        "mean_effect_7studies":
            all_study_mean.loc[
                genes
            ].to_numpy(),

        "n_positive_studies":
            (
                effects.loc[
                    genes
                ]
                > 0
            ).sum(
                axis=1
            ).to_numpy(),

        "n_negative_studies":
            (
                effects.loc[
                    genes
                ]
                < 0
            ).sum(
                axis=1
            ).to_numpy(),

        "majority_same_direction":
            majority_direction.loc[
                genes
            ].to_numpy(),

        "unanimous_same_direction":
            unanimous_direction.loc[
                genes
            ].to_numpy(),
    }
)


consensus_gene_table.to_csv(
    OUT_DIR
    / "Audit58_consensus_gene_directions.csv",
    index=False,
)


print()
print("=" * 100)
print("GENOME-WIDE DIRECTION CONSENSUS")
print("=" * 100)

print(
    "Genes with >=4/7 studies positive or >=4/7 negative: "
    + str(
        int(
            majority_direction.sum()
        )
    )
    + " / 4853"
)

print(
    "Genes with all 7 studies agreeing in direction: "
    + str(
        int(
            unanimous_direction.sum()
        )
    )
    + " / 4853"
)


print()
print("=" * 100)
print("AUDIT 58 COMPLETE")
print("=" * 100)

print(
    "No classifier was fitted."
)

print(
    "LOSO projection uses only the median effect direction "
    "from the other six studies."
)

print()
