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
    "data/processed/expanded_biofilm/GSE191103/"
    "GSE191103_PA14_samples_x_4858.csv"
)

OUT_DIR = Path(
    "results/audit/external_validation/GSE191103"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 38: "
    "external validation on GSE191103"
)
print("=" * 100)


# ============================================================
# Load frozen Audit-16 signature
# ============================================================

signature = pd.read_csv(
    SIGNATURE_FILE
)


required_columns = [
    "canonical_pao1",
    "median_effect",
]


for column in required_columns:

    if column not in signature.columns:
        raise RuntimeError(
            "Missing signature column: "
            + column
        )


if len(signature) != 251:

    raise RuntimeError(
        "Expected 251 frozen genes."
    )


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


print()
print("=" * 100)
print("FROZEN SIGNATURE")
print("=" * 100)

print(
    "Genes: "
    + str(len(frozen))
)

print(
    "Higher in biofilm: "
    + str(n_higher)
)

print(
    "Lower in biofilm: "
    + str(n_lower)
)


if n_higher != 118:

    raise RuntimeError(
        "Expected 118 biofilm-high genes."
    )


if n_lower != 133:

    raise RuntimeError(
        "Expected 133 biofilm-low genes."
    )


# ============================================================
# Load GSE191103
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
            "Missing study column: "
            + column
        )


if len(new) != 6:

    raise RuntimeError(
        "Expected six samples."
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
        "Expected three biofilm samples."
    )


if n_planktonic != 3:
    raise RuntimeError(
        "Expected three planktonic samples."
    )


gene_columns = [
    column
    for column in new.columns
    if column not in meta_columns
]


if len(gene_columns) != 4858:

    raise RuntimeError(
        "Expected 4,858 genes."
    )


X = (
    new[
        gene_columns
    ]
    .astype(float)
)


if X.isna().any().any():

    raise RuntimeError(
        "Expression matrix contains missing values."
    )


print()
print("=" * 100)
print("EXTERNAL STUDY")
print("=" * 100)

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
# Check frozen genes
# ============================================================

missing = sorted(
    set(
        frozen[
            "canonical_pao1"
        ]
    )
    - set(
        gene_columns
    )
)


print()
print(
    "Frozen genes missing: "
    + str(len(missing))
)


if missing:

    raise RuntimeError(
        "Frozen genes missing from GSE191103."
    )


# ============================================================
# Within-sample percentile ranks
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


# ============================================================
# External gene effects
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


external_effect = (
    bio_mean
    - plank_mean
)


validation = frozen.copy()


validation[
    "GSE191103_effect"
] = (
    validation[
        "canonical_pao1"
    ]
    .map(
        external_effect
    )
)


if validation[
    "GSE191103_effect"
].isna().any():

    raise RuntimeError(
        "External effect contains missing values."
    )


# ============================================================
# Directional replication
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
            "GSE191103_effect"
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
            "GSE191103_effect"
        ].abs()
        >= 0.05
    )
)


n_same = int(
    validation[
        "same_direction"
    ].sum()
)

n_strong = int(
    validation[
        "strong_same_direction"
    ].sum()
)


same_fraction = (
    n_same
    / len(validation)
)

strong_fraction = (
    n_strong
    / len(validation)
)


# ============================================================
# Effect correlation
# ============================================================

rho, rho_standard_p = spearmanr(
    validation[
        "frozen_effect"
    ],
    validation[
        "GSE191103_effect"
    ],
)


# ============================================================
# Frozen sample score
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
# AUC and observed difference
# ============================================================

y_true = (
    sample_results[
        "state"
    ]
    == "biofilm"
).astype(int)


external_auc = roc_auc_score(
    y_true,
    sample_results[
        "frozen_signature_score"
    ],
)


biofilm_mean = float(
    sample_results.loc[
        sample_results[
            "state"
        ]
        == "biofilm",
        "frozen_signature_score",
    ].mean()
)


planktonic_mean = float(
    sample_results.loc[
        sample_results[
            "state"
        ]
        == "planktonic",
        "frozen_signature_score",
    ].mean()
)


observed_difference = (
    biofilm_mean
    - planktonic_mean
)


# ============================================================
# Exact sample-label permutations
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


sample_perm_records = []


for bio_tuple in combinations(
    indices,
    3,
):

    bio_indices = set(
        bio_tuple
    )

    labels = np.array(
        [
            1 if index in bio_indices else 0
            for index in indices
        ]
    )

    perm_difference = float(
        scores[
            labels == 1
        ].mean()
        -
        scores[
            labels == 0
        ].mean()
    )

    perm_auc = float(
        roc_auc_score(
            labels,
            scores,
        )
    )

    sample_perm_records.append(
        {
            "mean_score_difference":
                perm_difference,

            "auc":
                perm_auc,
        }
    )


sample_permutations = pd.DataFrame(
    sample_perm_records
)


score_exact_p = float(
    (
        sample_permutations[
            "mean_score_difference"
        ]
        >= observed_difference - 1e-12
    ).mean()
)


auc_exact_p = float(
    (
        sample_permutations[
            "auc"
        ]
        >= external_auc - 1e-12
    ).mean()
)


# ============================================================
# Exact gene-direction permutations
# ============================================================

gene_perm_records = []


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

    perm_external = (
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
            perm_external
        )
    )

    perm_strong = (
        perm_same
        &
        (
            np.abs(
                perm_external
            )
            >= 0.05
        )
    )

    perm_rho, _ = spearmanr(
        frozen_effect_array,
        perm_external,
    )

    gene_perm_records.append(
        {
            "n_same_direction":
                int(
                    perm_same.sum()
                ),

            "n_strong_same_direction":
                int(
                    perm_strong.sum()
                ),

            "spearman":
                float(
                    perm_rho
                ),
        }
    )


gene_permutations = pd.DataFrame(
    gene_perm_records
)


same_exact_p = float(
    (
        gene_permutations[
            "n_same_direction"
        ]
        >= n_same
    ).mean()
)


strong_exact_p = float(
    (
        gene_permutations[
            "n_strong_same_direction"
        ]
        >= n_strong
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
# Save
# ============================================================

validation.to_csv(
    OUT_DIR
    / "frozen_251_gene_external_validation.csv",
    index=False,
)


sample_results.to_csv(
    OUT_DIR
    / "GSE191103_frozen_signature_scores.csv",
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
# Print results
# ============================================================

same_percent = (
    100.0
    * same_fraction
)

strong_percent = (
    100.0
    * strong_fraction
)


print()
print("=" * 100)
print(
    "FROZEN 251-GENE EXTERNAL VALIDATION"
)
print("=" * 100)


print(
    "Same direction: "
    + str(n_same)
    + " / 251 ("
    + format(
        same_percent,
        ".1f",
    )
    + "%)"
)


print(
    "Strong same direction: "
    + str(n_strong)
    + " / 251 ("
    + format(
        strong_percent,
        ".1f",
    )
    + "%)"
)


print()

print(
    "Effect Spearman rho: "
    + format(
        rho,
        ".4f",
    )
)

print(
    "Standard Spearman p: "
    + format(
        rho_standard_p,
        ".3e",
    )
)


print()
print(
    "Exact balanced-label tests:"
)

print(
    "  Same-direction p: "
    + format(
        same_exact_p,
        ".4f",
    )
)

print(
    "  Strong-direction p: "
    + format(
        strong_exact_p,
        ".4f",
    )
)

print(
    "  Effect-correlation p: "
    + format(
        rho_exact_p,
        ".4f",
    )
)


print()
print("=" * 100)
print(
    "FROZEN SIGNATURE SAMPLE SCORES"
)
print("=" * 100)


print(
    sample_results
    .sort_values(
        "frozen_signature_score",
        ascending=False,
    )
    .to_string(
        index=False
    )
)


print()

print(
    "Mean biofilm score: "
    + format(
        biofilm_mean,
        ".6f",
    )
)

print(
    "Mean planktonic score: "
    + format(
        planktonic_mean,
        ".6f",
    )
)

print(
    "Biofilm - planktonic difference: "
    + format(
        observed_difference,
        ".6f",
    )
)

print(
    "External ROC AUC: "
    + format(
        external_auc,
        ".3f",
    )
)


print()

print(
    "Exact score-difference p: "
    + format(
        score_exact_p,
        ".4f",
    )
)

print(
    "Exact AUC p: "
    + format(
        auc_exact_p,
        ".4f",
    )
)


print()
print("=" * 100)
print("AUDIT 38 COMPLETE")
print("=" * 100)

print(
    "Outputs saved to: "
    + str(OUT_DIR)
)

print()
