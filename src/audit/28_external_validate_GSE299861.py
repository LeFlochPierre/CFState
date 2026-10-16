from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================
# Paths
# ============================================================

SIGNATURE_FILE = Path(
    "results/audit/core_biofilm_signature/"
    "core_biofilm_signature_fdr05.csv"
)

NEW_STUDY_FILE = Path(
    "data/processed/expanded_biofilm/GSE299861/"
    "GSE299861_PAO1_samples_x_4858.csv"
)

OUT_DIR = Path(
    "results/audit/external_validation/GSE299861"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 28: "
    "external validation on GSE299861"
)
print("=" * 100)


# ============================================================
# 1. Load frozen Audit-16 signature
# ============================================================

signature = pd.read_csv(
    SIGNATURE_FILE
)

required_columns = [
    "canonical_pao1",
    "median_effect",
    "direction",
]

missing_columns = [
    column
    for column in required_columns
    if column not in signature.columns
]

if missing_columns:
    raise RuntimeError(
        "Missing required signature columns: "
        + ", ".join(missing_columns)
    )


if len(signature) != 251:
    raise RuntimeError(
        "Expected 251 frozen genes, "
        f"found {len(signature)}."
    )


# ============================================================
# IMPORTANT
#
# median_effect is already SIGNED:
#
# positive = higher in biofilm
# negative = lower in biofilm
#
# Do NOT use consistency_score as a signed value.
# ============================================================

frozen = pd.DataFrame(
    {
        "canonical_pao1":
            signature[
                "canonical_pao1"
            ].astype(str),

        "frozen_effect":
            pd.to_numeric(
                signature[
                    "median_effect"
                ],
                errors="raise",
            ),
    }
)


if frozen[
    "canonical_pao1"
].duplicated().any():
    raise RuntimeError(
        "Frozen signature contains duplicated genes."
    )


n_higher = int(
    (
        frozen[
            "frozen_effect"
        ]
        > 0
    ).sum()
)

n_lower = int(
    (
        frozen[
            "frozen_effect"
        ]
        < 0
    ).sum()
)

n_zero = int(
    (
        frozen[
            "frozen_effect"
        ]
        == 0
    ).sum()
)


print()
print("=" * 100)
print("FROZEN SIGNATURE")
print("=" * 100)

print(
    f"Genes: {len(frozen)}"
)

print(
    f"Higher in biofilm: {n_higher}"
)

print(
    f"Lower in biofilm: {n_lower}"
)

print(
    f"Zero effect: {n_zero}"
)


# Critical sanity check against Audit 16
if n_higher != 118:
    raise RuntimeError(
        "Expected 118 biofilm-high genes."
    )

if n_lower != 133:
    raise RuntimeError(
        "Expected 133 biofilm-low genes."
    )

if n_zero != 0:
    raise RuntimeError(
        "Expected zero zero-effect genes."
    )


frozen[
    "frozen_direction"
] = np.where(
    frozen[
        "frozen_effect"
    ] > 0,
    "higher_in_biofilm",
    "lower_in_biofilm",
)


# ============================================================
# 2. Load external GSE299861 dataset
# ============================================================

new = pd.read_csv(
    NEW_STUDY_FILE
)


meta_columns = [
    "sample",
    "study",
    "state",
]


for column in meta_columns:

    if column not in new.columns:

        raise RuntimeError(
            f"Missing required column: {column}"
        )


if len(new) != 6:

    raise RuntimeError(
        "Expected 6 GSE299861 samples, "
        f"found {len(new)}."
    )


n_biofilm = int(
    (
        new[
            "state"
        ]
        == "biofilm"
    ).sum()
)

n_planktonic = int(
    (
        new[
            "state"
        ]
        == "planktonic"
    ).sum()
)


if n_biofilm != 3:

    raise RuntimeError(
        "Expected exactly 3 biofilm samples."
    )


if n_planktonic != 3:

    raise RuntimeError(
        "Expected exactly 3 planktonic samples."
    )


gene_columns = [
    column
    for column in new.columns
    if column not in meta_columns
]


if len(gene_columns) != 4858:

    raise RuntimeError(
        "Expected 4,858 genes, "
        f"found {len(gene_columns)}."
    )


X = (
    new[
        gene_columns
    ]
    .astype(float)
)


if X.isna().any().any():

    raise RuntimeError(
        "GSE299861 contains missing expression values."
    )


print()
print("=" * 100)
print("NEW STUDY")
print("=" * 100)

print(
    f"Samples: {len(new)}"
)

print(
    f"Genes: {len(gene_columns)}"
)

print()

print(
    new[
        [
            "sample",
            "state",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 3. Verify frozen genes exist
# ============================================================

new_gene_set = set(
    gene_columns
)

frozen_gene_set = set(
    frozen[
        "canonical_pao1"
    ]
)

missing_genes = sorted(
    frozen_gene_set
    - new_gene_set
)


print()
print(
    "Frozen signature genes missing "
    f"from GSE299861: {len(missing_genes)}"
)


if missing_genes:

    raise RuntimeError(
        "Frozen signature genes are missing."
    )


# ============================================================
# 4. Within-sample percentile ranks
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


# ============================================================
# 5. GSE299861 biofilm - planktonic gene effects
# ============================================================

bio_mask = (
    new[
        "state"
    ]
    == "biofilm"
)

plank_mask = (
    new[
        "state"
    ]
    == "planktonic"
)


bio_mean = (
    X_rank.loc[
        bio_mask
    ]
    .mean(
        axis=0
    )
)

plank_mean = (
    X_rank.loc[
        plank_mask
    ]
    .mean(
        axis=0
    )
)


new_effect = (
    bio_mean
    - plank_mean
)


validation = (
    frozen.copy()
)

validation[
    "GSE299861_effect"
] = (
    validation[
        "canonical_pao1"
    ]
    .map(
        new_effect
    )
)


if validation[
    "GSE299861_effect"
].isna().any():

    raise RuntimeError(
        "A frozen gene has no external effect value."
    )


# ============================================================
# 6. Directional replication
# ============================================================

validation[
    "same_direction"
] = (
    np.sign(
        validation[
            "frozen_effect"
        ]
    )
    ==
    np.sign(
        validation[
            "GSE299861_effect"
        ]
    )
)


validation[
    "strong_same_direction"
] = (
    validation[
        "same_direction"
    ]
    &
    (
        validation[
            "GSE299861_effect"
        ].abs()
        >= 0.05
    )
)


n_same = int(
    validation[
        "same_direction"
    ].sum()
)

n_strong_same = int(
    validation[
        "strong_same_direction"
    ].sum()
)

same_fraction = (
    n_same
    / len(validation)
)

strong_same_fraction = (
    n_strong_same
    / len(validation)
)


# ============================================================
# 7. Correlation between frozen and external effects
# ============================================================

rho, rho_standard_p = spearmanr(
    validation[
        "frozen_effect"
    ],
    validation[
        "GSE299861_effect"
    ],
)


# ============================================================
# 8. Completely frozen single-sample score
#
# No parameters are learned from GSE299861.
#
# score =
# mean rank of original biofilm-high genes
# -
# mean rank of original biofilm-low genes
# ============================================================

up_genes = (
    frozen.loc[
        frozen[
            "frozen_effect"
        ]
        > 0,
        "canonical_pao1",
    ]
    .tolist()
)

down_genes = (
    frozen.loc[
        frozen[
            "frozen_effect"
        ]
        < 0,
        "canonical_pao1",
    ]
    .tolist()
)


if len(up_genes) != 118:

    raise RuntimeError(
        "Incorrect number of biofilm-high genes."
    )


if len(down_genes) != 133:

    raise RuntimeError(
        "Incorrect number of biofilm-low genes."
    )


up_score = (
    X_rank[
        up_genes
    ]
    .mean(
        axis=1
    )
)

down_score = (
    X_rank[
        down_genes
    ]
    .mean(
        axis=1
    )
)


state_score = (
    up_score
    - down_score
)


if state_score.isna().any():

    raise RuntimeError(
        "Frozen state score contains NaN."
    )


sample_results = (
    new[
        [
            "sample",
            "state",
        ]
    ]
    .copy()
)


sample_results[
    "frozen_signature_score"
] = (
    state_score.to_numpy()
)


# ============================================================
# 9. External discrimination
# ============================================================

y_true = (
    sample_results[
        "state"
    ]
    == "biofilm"
).astype(int)


external_auc = (
    roc_auc_score(
        y_true,
        sample_results[
            "frozen_signature_score"
        ],
    )
)


biofilm_score_mean = float(
    sample_results.loc[
        sample_results[
            "state"
        ]
        == "biofilm",
        "frozen_signature_score",
    ].mean()
)


planktonic_score_mean = float(
    sample_results.loc[
        sample_results[
            "state"
        ]
        == "planktonic",
        "frozen_signature_score",
    ].mean()
)


observed_score_difference = (
    biofilm_score_mean
    - planktonic_score_mean
)


# ============================================================
# 10. Exact permutation test for sample scores
#
# Six samples:
# C(6,3) = 20 balanced assignments.
#
# The signature score stays frozen.
# Only labels change.
# ============================================================

indices = list(
    range(6)
)

scores = (
    sample_results[
        "frozen_signature_score"
    ]
    .to_numpy()
)


sample_permutation_records = []


for bio_tuple in combinations(
    indices,
    3,
):

    bio_indices = set(
        bio_tuple
    )


    perm_labels = np.array(
        [
            1
            if index in bio_indices
            else 0
            for index in indices
        ],
        dtype=int,
    )


    perm_bio_mean = float(
        scores[
            perm_labels == 1
        ].mean()
    )


    perm_plank_mean = float(
        scores[
            perm_labels == 0
        ].mean()
    )


    perm_difference = (
        perm_bio_mean
        - perm_plank_mean
    )


    perm_auc = (
        roc_auc_score(
            perm_labels,
            scores,
        )
    )


    bio_index_text = ";".join(
        str(index)
        for index in sorted(
            bio_indices
        )
    )


    sample_permutation_records.append(
        {
            "biofilm_indices":
                bio_index_text,

            "mean_score_difference":
                perm_difference,

            "auc":
                perm_auc,
        }
    )


sample_permutations = pd.DataFrame(
    sample_permutation_records
)


exact_score_p = float(
    (
        sample_permutations[
            "mean_score_difference"
        ]
        >= observed_score_difference - 1e-12
    ).mean()
)


exact_auc_p = float(
    (
        sample_permutations[
            "auc"
        ]
        >= external_auc - 1e-12
    ).mean()
)


# ============================================================
# 11. Exact permutation tests for gene transfer
# ============================================================

gene_permutation_records = []


frozen_effect_array = (
    frozen[
        "frozen_effect"
    ]
    .to_numpy()
)


for bio_tuple in combinations(
    indices,
    3,
):

    bio_indices = list(
        bio_tuple
    )

    plank_indices = [
        index
        for index in indices
        if index not in bio_indices
    ]


    perm_effect = (
        X_rank.iloc[
            bio_indices
        ]
        .mean(
            axis=0
        )
        -
        X_rank.iloc[
            plank_indices
        ]
        .mean(
            axis=0
        )
    )


    perm_new_effect = (
        frozen[
            "canonical_pao1"
        ]
        .map(
            perm_effect
        )
        .to_numpy()
    )


    perm_same = (
        np.sign(
            frozen_effect_array
        )
        ==
        np.sign(
            perm_new_effect
        )
    )


    perm_strong_same = (
        perm_same
        &
        (
            np.abs(
                perm_new_effect
            )
            >= 0.05
        )
    )


    perm_rho, _ = spearmanr(
        frozen_effect_array,
        perm_new_effect,
    )


    gene_permutation_records.append(
        {
            "n_same_direction":
                int(
                    perm_same.sum()
                ),

            "n_strong_same_direction":
                int(
                    perm_strong_same.sum()
                ),

            "spearman":
                float(
                    perm_rho
                ),
        }
    )


gene_permutations = pd.DataFrame(
    gene_permutation_records
)


same_direction_p = float(
    (
        gene_permutations[
            "n_same_direction"
        ]
        >= n_same
    ).mean()
)


strong_direction_p = float(
    (
        gene_permutations[
            "n_strong_same_direction"
        ]
        >= n_strong_same
    ).mean()
)


rho_exact_p = float(
    (
        gene_permutations[
            "spearman"
        ]
        >= rho - 1e-12
    ).mean()
)


# ============================================================
# 12. Save outputs
# ============================================================

validation = (
    validation
    .sort_values(
        "frozen_effect",
        ascending=False,
    )
)


validation.to_csv(
    OUT_DIR
    / "frozen_251_gene_external_validation.csv",
    index=False,
)


sample_results.to_csv(
    OUT_DIR
    / "GSE299861_frozen_signature_scores.csv",
    index=False,
)


sample_permutations.to_csv(
    OUT_DIR
    / "sample_score_exact_permutations.csv",
    index=False,
)


gene_permutations.to_csv(
    OUT_DIR
    / "gene_direction_exact_permutations.csv",
    index=False,
)


# ============================================================
# 13. Print results
# ============================================================

print()
print("=" * 100)
print(
    "FROZEN 251-GENE EXTERNAL VALIDATION"
)
print("=" * 100)


same_percent = (
    100
    * same_fraction
)

strong_percent = (
    100
    * strong_same_fraction
)


print(
    f"Same direction: "
    f"{n_same} / {len(validation)} "
    f"({same_percent:.1f}%)"
)

print(
    f"Strong same direction: "
    f"{n_strong_same} / {len(validation)} "
    f"({strong_percent:.1f}%)"
)

print()

print(
    f"Effect Spearman rho: "
    f"{rho:.4f}"
)

print(
    f"Standard Spearman p: "
    f"{rho_standard_p:.3e}"
)

print()

print(
    "Exact balanced-label tests:"
)

print(
    f"  Same-direction p: "
    f"{same_direction_p:.4f}"
)

print(
    f"  Strong-direction p: "
    f"{strong_direction_p:.4f}"
)

print(
    f"  Effect-correlation p: "
    f"{rho_exact_p:.4f}"
)


print()
print("=" * 100)
print(
    "FROZEN SIGNATURE SAMPLE SCORES"
)
print("=" * 100)


sample_results_sorted = (
    sample_results
    .sort_values(
        "frozen_signature_score",
        ascending=False,
    )
)


print(
    sample_results_sorted.to_string(
        index=False
    )
)


print()

print(
    f"Mean biofilm score: "
    f"{biofilm_score_mean:.6f}"
)

print(
    f"Mean planktonic score: "
    f"{planktonic_score_mean:.6f}"
)

print(
    f"Biofilm - planktonic difference: "
    f"{observed_score_difference:.6f}"
)

print(
    f"External ROC AUC: "
    f"{external_auc:.3f}"
)

print()

print(
    f"Exact score-difference p: "
    f"{exact_score_p:.4f}"
)

print(
    f"Exact AUC p: "
    f"{exact_auc_p:.4f}"
)


# ============================================================
# Top genes
# ============================================================

print()
print("=" * 100)
print(
    "TOP 20 FROZEN BIOFILM-HIGH GENES"
)
print("=" * 100)


high_table = (
    validation.loc[
        validation[
            "frozen_effect"
        ]
        > 0
    ]
    .sort_values(
        "frozen_effect",
        ascending=False,
    )
    .head(20)
)


print(
    high_table[
        [
            "canonical_pao1",
            "frozen_effect",
            "GSE299861_effect",
            "same_direction",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print(
    "TOP 20 FROZEN BIOFILM-LOW GENES"
)
print("=" * 100)


low_table = (
    validation.loc[
        validation[
            "frozen_effect"
        ]
        < 0
    ]
    .sort_values(
        "frozen_effect",
        ascending=True,
    )
    .head(20)
)


print(
    low_table[
        [
            "canonical_pao1",
            "frozen_effect",
            "GSE299861_effect",
            "same_direction",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 28 COMPLETE")
print("=" * 100)

print(
    "Outputs saved to: "
    f"{OUT_DIR}"
)

print()

print(
    "GSE299861 was not used to select, filter, "
    "weight, or redefine the original 251-gene signature."
)

print()