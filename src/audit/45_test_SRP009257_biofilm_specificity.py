from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


SIGNATURE_FILE = Path(
    "results/audit/core_biofilm_signature/"
    "core_biofilm_signature_fdr05.csv"
)

STUDY_FILE = Path(
    "data/processed/expanded_biofilm/SRP009257/"
    "SRP009257_PA14_samples_x_4858.csv"
)

OUT_DIR = Path(
    "results/audit/external_validation/SRP009257_specificity"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 45: "
    "SRP009257 biofilm-specificity test"
)
print("=" * 100)


# ============================================================
# Load frozen 251-gene programme
# ============================================================

signature = pd.read_csv(
    SIGNATURE_FILE
)

required_signature_columns = [
    "canonical_pao1",
    "median_effect",
]

for column in required_signature_columns:

    if column not in signature.columns:
        raise RuntimeError(
            "Missing signature column: "
            + column
        )


if len(signature) != 251:
    raise RuntimeError(
        "Expected 251 frozen signature genes."
    )


signature = pd.DataFrame(
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


if signature[
    "canonical_pao1"
].duplicated().any():

    raise RuntimeError(
        "Duplicate genes in frozen signature."
    )


n_high = int(
    (
        signature[
            "frozen_effect"
        ]
        > 0
    ).sum()
)

n_low = int(
    (
        signature[
            "frozen_effect"
        ]
        < 0
    ).sum()
)


if n_high != 118 or n_low != 133:
    raise RuntimeError(
        "Frozen signature direction counts "
        "do not match 118 high / 133 low."
    )


# ============================================================
# Load SRP009257
# ============================================================

study = pd.read_csv(
    STUDY_FILE
)

meta_columns = [
    "sample",
    "study",
    "state",
    "condition",
]

for column in meta_columns:

    if column not in study.columns:
        raise RuntimeError(
            "Missing study metadata column: "
            + column
        )


expected_conditions = {
    "P4": 2,
    "P12": 2,
    "B24": 2,
    "B48": 2,
}


for condition, expected_n in expected_conditions.items():

    observed_n = int(
        (
            study[
                "condition"
            ]
            == condition
        ).sum()
    )

    if observed_n != expected_n:
        raise RuntimeError(
            "Expected "
            + str(expected_n)
            + " samples for "
            + condition
            + ", found "
            + str(observed_n)
        )


gene_columns = [
    column
    for column in study.columns
    if column not in meta_columns
]


if len(gene_columns) != 4858:
    raise RuntimeError(
        "Expected 4,858 genes, found "
        + str(len(gene_columns))
    )


missing_signature = sorted(
    set(
        signature[
            "canonical_pao1"
        ]
    )
    - set(
        gene_columns
    )
)


if missing_signature:
    raise RuntimeError(
        "Frozen signature genes missing: "
        + ", ".join(
            missing_signature
        )
    )


X = (
    study[
        gene_columns
    ]
    .astype(float)
)


if X.isna().any().any():
    raise RuntimeError(
        "Expression matrix contains missing values."
    )


# ============================================================
# Within-sample percentile ranks
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


up_genes = (
    signature.loc[
        signature[
            "frozen_effect"
        ]
        > 0,
        "canonical_pao1",
    ]
    .tolist()
)


down_genes = (
    signature.loc[
        signature[
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


sample_scores = (
    study[
        [
            "sample",
            "state",
            "condition",
        ]
    ]
    .copy()
)


sample_scores[
    "frozen_signature_score"
] = state_score.to_numpy()


condition_order = [
    "P4",
    "P12",
    "B24",
    "B48",
]


condition_scores = (
    sample_scores
    .groupby(
        "condition"
    )[
        "frozen_signature_score"
    ]
    .agg(
        [
            "mean",
            "min",
            "max",
        ]
    )
    .reindex(
        condition_order
    )
    .reset_index()
)


sample_scores.to_csv(
    OUT_DIR
    / "SRP009257_frozen_signature_sample_scores.csv",
    index=False,
)


condition_scores.to_csv(
    OUT_DIR
    / "SRP009257_frozen_signature_condition_scores.csv",
    index=False,
)


# ============================================================
# Contrasts
#
# Positive = group A > group B.
#
# P12 vs P4 tells us whether stationary phase moves in the
# frozen biofilm direction.
#
# B48 vs B24 tells us whether later biofilm moves further in
# that direction.
# ============================================================

contrasts = [
    {
        "name":
            "Biofilm_all_vs_Planktonic_all",
        "group_a":
            ["B24", "B48"],
        "group_b":
            ["P4", "P12"],
    },
    {
        "name":
            "B24_vs_P4",
        "group_a":
            ["B24"],
        "group_b":
            ["P4"],
    },
    {
        "name":
            "B24_vs_P12",
        "group_a":
            ["B24"],
        "group_b":
            ["P12"],
    },
    {
        "name":
            "B48_vs_P4",
        "group_a":
            ["B48"],
        "group_b":
            ["P4"],
    },
    {
        "name":
            "B48_vs_P12",
        "group_a":
            ["B48"],
        "group_b":
            ["P12"],
    },
    {
        "name":
            "P12_vs_P4",
        "group_a":
            ["P12"],
        "group_b":
            ["P4"],
    },
    {
        "name":
            "B48_vs_B24",
        "group_a":
            ["B48"],
        "group_b":
            ["B24"],
    },
]


frozen_effect = (
    signature[
        "frozen_effect"
    ]
    .to_numpy()
)


signature_gene_order = (
    signature[
        "canonical_pao1"
    ]
    .tolist()
)


scores = (
    sample_scores[
        "frozen_signature_score"
    ]
    .to_numpy()
)


contrast_records = []
gene_effect_records = []


for contrast in contrasts:

    contrast_name = contrast[
        "name"
    ]

    group_a_conditions = contrast[
        "group_a"
    ]

    group_b_conditions = contrast[
        "group_b"
    ]


    group_a_mask = (
        study[
            "condition"
        ]
        .isin(
            group_a_conditions
        )
    )

    group_b_mask = (
        study[
            "condition"
        ]
        .isin(
            group_b_conditions
        )
    )


    subset_mask = (
        group_a_mask
        | group_b_mask
    )


    subset_indices = np.flatnonzero(
        subset_mask.to_numpy()
    )

    group_a_indices = np.flatnonzero(
        group_a_mask.to_numpy()
    )

    group_b_indices = np.flatnonzero(
        group_b_mask.to_numpy()
    )


    n_a = len(
        group_a_indices
    )

    n_b = len(
        group_b_indices
    )


    if n_a == 0 or n_b == 0:
        raise RuntimeError(
            "Empty group in contrast "
            + contrast_name
        )


    # --------------------------------------------------------
    # Frozen score contrast
    # --------------------------------------------------------

    mean_score_a = float(
        scores[
            group_a_indices
        ].mean()
    )

    mean_score_b = float(
        scores[
            group_b_indices
        ].mean()
    )

    score_difference = (
        mean_score_a
        - mean_score_b
    )


    y_true = np.zeros(
        len(study),
        dtype=int,
    )

    y_true[
        group_a_indices
    ] = 1


    observed_auc = float(
        roc_auc_score(
            y_true[
                subset_indices
            ],
            scores[
                subset_indices
            ],
        )
    )


    # --------------------------------------------------------
    # Gene-level effect for this contrast
    # --------------------------------------------------------

    effect_series = (
        X_rank.iloc[
            group_a_indices
        ]
        .mean(
            axis=0
        )
        -
        X_rank.iloc[
            group_b_indices
        ]
        .mean(
            axis=0
        )
    )


    external_effect = (
        effect_series[
            signature_gene_order
        ]
        .to_numpy()
    )


    same_direction = (
        np.sign(
            frozen_effect
        )
        ==
        np.sign(
            external_effect
        )
    )


    strong_same_direction = (
        same_direction
        &
        (
            np.abs(
                external_effect
            )
            >= 0.05
        )
    )


    n_same = int(
        same_direction.sum()
    )

    n_strong = int(
        strong_same_direction.sum()
    )


    effect_rho, effect_rho_p = spearmanr(
        frozen_effect,
        external_effect,
    )


    # --------------------------------------------------------
    # Exact balanced-label permutations
    # --------------------------------------------------------

    local_indices = list(
        range(
            len(
                subset_indices
            )
        )
    )


    local_scores = scores[
        subset_indices
    ]


    subset_rank = (
        X_rank.iloc[
            subset_indices
        ]
        .reset_index(
            drop=True
        )
    )


    permutation_records = []


    for local_a_tuple in combinations(
        local_indices,
        n_a,
    ):

        local_a_list = sorted(
            local_a_tuple
        )

        local_a_set = set(
            local_a_tuple
        )

        local_b = [
            index
            for index in local_indices
            if index not in local_a_set
        ]


        perm_score_difference = float(
            local_scores[
                local_a_list
            ].mean()
            -
            local_scores[
                local_b
            ].mean()
        )


        perm_labels = np.zeros(
            len(
                local_indices
            ),
            dtype=int,
        )

        perm_labels[
            local_a_list
        ] = 1


        perm_auc = float(
            roc_auc_score(
                perm_labels,
                local_scores,
            )
        )


        perm_effect_series = (
            subset_rank.iloc[
                local_a_list
            ]
            .mean(
                axis=0
            )
            -
            subset_rank.iloc[
                local_b
            ]
            .mean(
                axis=0
            )
        )


        perm_external_effect = (
            perm_effect_series[
                signature_gene_order
            ]
            .to_numpy()
        )


        perm_same_direction = (
            np.sign(
                frozen_effect
            )
            ==
            np.sign(
                perm_external_effect
            )
        )


        perm_strong_same_direction = (
            perm_same_direction
            &
            (
                np.abs(
                    perm_external_effect
                )
                >= 0.05
            )
        )


        perm_rho, _ = spearmanr(
            frozen_effect,
            perm_external_effect,
        )


        permutation_records.append(
            {
                "score_difference":
                    perm_score_difference,

                "auc":
                    perm_auc,

                "n_same_direction":
                    int(
                        perm_same_direction.sum()
                    ),

                "n_strong_same_direction":
                    int(
                        perm_strong_same_direction.sum()
                    ),

                "effect_rho":
                    float(
                        perm_rho
                    ),
            }
        )


    permutations = pd.DataFrame(
        permutation_records
    )


    n_permutations = len(
        permutations
    )


    score_exact_p = float(
        (
            permutations[
                "score_difference"
            ]
            >= score_difference - 1e-12
        ).mean()
    )


    auc_exact_p = float(
        (
            permutations[
                "auc"
            ]
            >= observed_auc - 1e-12
        ).mean()
    )


    same_exact_p = float(
        (
            permutations[
                "n_same_direction"
            ]
            >= n_same
        ).mean()
    )


    strong_exact_p = float(
        (
            permutations[
                "n_strong_same_direction"
            ]
            >= n_strong
        ).mean()
    )


    rho_exact_p = float(
        (
            permutations[
                "effect_rho"
            ]
            >= effect_rho - 1e-12
        ).mean()
    )


    contrast_records.append(
        {
            "contrast":
                contrast_name,

            "group_a":
                "+".join(
                    group_a_conditions
                ),

            "group_b":
                "+".join(
                    group_b_conditions
                ),

            "n_group_a":
                n_a,

            "n_group_b":
                n_b,

            "mean_score_a":
                mean_score_a,

            "mean_score_b":
                mean_score_b,

            "score_difference":
                score_difference,

            "auc":
                observed_auc,

            "effect_rho_vs_frozen":
                float(
                    effect_rho
                ),

            "effect_rho_standard_p":
                float(
                    effect_rho_p
                ),

            "same_direction_n":
                n_same,

            "same_direction_fraction":
                n_same / 251.0,

            "strong_same_direction_n":
                n_strong,

            "strong_same_direction_fraction":
                n_strong / 251.0,

            "n_exact_permutations":
                n_permutations,

            "exact_score_p":
                score_exact_p,

            "exact_auc_p":
                auc_exact_p,

            "exact_same_direction_p":
                same_exact_p,

            "exact_strong_direction_p":
                strong_exact_p,

            "exact_effect_rho_p":
                rho_exact_p,
        }
    )


    for gene, frozen_value, external_value in zip(
        signature_gene_order,
        frozen_effect,
        external_effect,
    ):

        gene_effect_records.append(
            {
                "contrast":
                    contrast_name,

                "canonical_pao1":
                    gene,

                "frozen_effect":
                    frozen_value,

                "contrast_effect":
                    external_value,

                "aligned_with_frozen_direction":
                    np.sign(
                        frozen_value
                    )
                    ==
                    np.sign(
                        external_value
                    ),
            }
        )


    permutations.to_csv(
        OUT_DIR
        / (
            contrast_name
            + "_exact_permutations.csv"
        ),
        index=False,
    )


# ============================================================
# Save summaries
# ============================================================

contrast_results = pd.DataFrame(
    contrast_records
)


gene_effects = pd.DataFrame(
    gene_effect_records
)


contrast_results.to_csv(
    OUT_DIR
    / "SRP009257_specificity_contrasts.csv",
    index=False,
)


gene_effects.to_csv(
    OUT_DIR
    / "SRP009257_251_gene_contrast_effects.csv",
    index=False,
)


# ============================================================
# Print sample scores
# ============================================================

print()
print("=" * 100)
print("FROZEN 251-GENE SAMPLE SCORES")
print("=" * 100)

print(
    sample_scores[
        [
            "sample",
            "condition",
            "frozen_signature_score",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("CONDITION MEAN SCORES")
print("=" * 100)

print(
    condition_scores.to_string(
        index=False
    )
)


# ============================================================
# Print contrasts
# ============================================================

display = (
    contrast_results[
        [
            "contrast",
            "score_difference",
            "auc",
            "effect_rho_vs_frozen",
            "same_direction_fraction",
            "strong_same_direction_fraction",
            "n_exact_permutations",
            "exact_score_p",
            "exact_effect_rho_p",
        ]
    ]
    .copy()
)


display[
    "same_direction_percent"
] = (
    100.0
    * display[
        "same_direction_fraction"
    ]
)


display[
    "strong_same_direction_percent"
] = (
    100.0
    * display[
        "strong_same_direction_fraction"
    ]
)


display = display.drop(
    columns=[
        "same_direction_fraction",
        "strong_same_direction_fraction",
    ]
)


print()
print("=" * 100)
print("BIOFILM-SPECIFICITY CONTRASTS")
print("=" * 100)

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# Trajectory summary
# ============================================================

score_lookup = dict(
    zip(
        condition_scores[
            "condition"
        ],
        condition_scores[
            "mean"
        ],
    )
)


print()
print("=" * 100)
print("TRAJECTORY CHECK")
print("=" * 100)


for condition in condition_order:

    value = float(
        score_lookup[
            condition
        ]
    )

    print(
        condition
        + " mean frozen score: "
        + format(
            value,
            ".6f",
        )
    )


p12_shift = (
    score_lookup[
        "P12"
    ]
    -
    score_lookup[
        "P4"
    ]
)


b24_over_p12 = (
    score_lookup[
        "B24"
    ]
    -
    score_lookup[
        "P12"
    ]
)


b48_over_p12 = (
    score_lookup[
        "B48"
    ]
    -
    score_lookup[
        "P12"
    ]
)


print()

print(
    "Stationary-phase shift P12 - P4: "
    + format(
        p12_shift,
        ".6f",
    )
)

print(
    "B24 - P12 specificity margin: "
    + format(
        b24_over_p12,
        ".6f",
    )
)

print(
    "B48 - P12 specificity margin: "
    + format(
        b48_over_p12,
        ".6f",
    )
)


print()
print("=" * 100)
print("AUDIT 45 COMPLETE")
print("=" * 100)

print(
    "Outputs saved to: "
    + str(
        OUT_DIR
    )
)

print()

print(
    "Important: pairwise 2-vs-2 contrasts have only "
    "6 balanced label assignments, so their smallest "
    "possible exact one-sided p-value is 1/6 = 0.1667."
)

print(
    "The pooled 4-vs-4 contrast has 70 balanced "
    "assignments, so its minimum exact p-value is 1/70."
)

print()
