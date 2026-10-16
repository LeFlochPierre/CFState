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
    "data/processed/expanded_biofilm/GSE12207/"
    "GSE12207_PAO1_samples_x_4853.csv"
)

OUT_DIR = Path(
    "results/audit/external_validation/"
    "GSE12207"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 52: "
    "GSE12207 cross-platform frozen-signature validation"
)
print("=" * 100)


# ============================================================
# Load frozen signature
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


# ============================================================
# Load GSE12207
# ============================================================

study = pd.read_csv(
    STUDY_FILE
)


meta_columns = [
    "sample",
    "study",
    "gsm",
    "condition",
    "replicate",
]


for column in meta_columns:

    if column not in study.columns:

        raise RuntimeError(
            "Missing metadata column: "
            + column
        )


expected_conditions = {
    "planktonic_exponential": 3,
    "planktonic_stationary": 3,
    "colony_15h": 3,
    "colony_40h": 3,
    "biofilm_3d": 3,
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


if len(gene_columns) != 4853:

    raise RuntimeError(
        "Expected 4,853 genes, found "
        + str(len(gene_columns))
    )


available_gene_set = set(
    gene_columns
)


signature[
    "available"
] = (
    signature[
        "canonical_pao1"
    ]
    .isin(
        available_gene_set
    )
)


available_signature = (
    signature.loc[
        signature[
            "available"
        ]
    ]
    .copy()
)


missing_signature = (
    signature.loc[
        ~signature[
            "available"
        ],
        "canonical_pao1",
    ]
    .tolist()
)


print()
print("=" * 100)
print("SIGNATURE COVERAGE")
print("=" * 100)

print(
    "Frozen signature genes expected: "
    + str(
        len(signature)
    )
)

print(
    "Signature genes measurable on GPL84: "
    + str(
        len(
            available_signature
        )
    )
)

print(
    "Signature genes unavailable: "
    + str(
        len(
            missing_signature
        )
    )
)

print(
    "Missing signature genes: "
    + (
        ", ".join(
            missing_signature
        )
        if missing_signature
        else "None"
    )
)


if len(
    available_signature
) != 250:

    raise RuntimeError(
        "Expected exactly 250 measurable signature genes."
    )


# ============================================================
# Expression matrix
# ============================================================

X = (
    study[
        gene_columns
    ]
    .apply(
        pd.to_numeric,
        errors="raise",
    )
)


if X.isna().any().any():

    raise RuntimeError(
        "Missing expression values detected."
    )


# ============================================================
# Within-sample percentile ranks
#
# This makes Affymetrix values comparable in spirit to our
# RNA-seq rank-based scoring.
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


up_genes = (
    available_signature.loc[
        available_signature[
            "frozen_effect"
        ]
        > 0,
        "canonical_pao1",
    ]
    .tolist()
)


down_genes = (
    available_signature.loc[
        available_signature[
            "frozen_effect"
        ]
        < 0,
        "canonical_pao1",
    ]
    .tolist()
)


print()
print(
    "Measurable biofilm-higher genes: "
    + str(
        len(
            up_genes
        )
    )
)

print(
    "Measurable biofilm-lower genes: "
    + str(
        len(
            down_genes
        )
    )
)


if (
    len(up_genes)
    + len(down_genes)
    != 250
):

    raise RuntimeError(
        "Available signature directions do not sum to 250."
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
            "gsm",
            "condition",
            "replicate",
        ]
    ]
    .copy()
)


sample_scores[
    "up_score"
] = up_score.to_numpy()


sample_scores[
    "down_score"
] = down_score.to_numpy()


sample_scores[
    "frozen_signature_score"
] = state_score.to_numpy()


# ============================================================
# Condition summaries
# ============================================================

condition_order = [
    "planktonic_exponential",
    "planktonic_stationary",
    "colony_15h",
    "colony_40h",
    "biofilm_3d",
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
    / "GSE12207_frozen_signature_sample_scores.csv",
    index=False,
)


condition_scores.to_csv(
    OUT_DIR
    / "GSE12207_frozen_signature_condition_scores.csv",
    index=False,
)


# ============================================================
# Pairwise biological contrasts
#
# Group A is expected to be MORE biofilm-like than group B.
# ============================================================

contrasts = [
    {
        "name":
            "Biofilm_vs_Exponential",

        "group_a":
            "biofilm_3d",

        "group_b":
            "planktonic_exponential",
    },
    {
        "name":
            "Stationary_vs_Exponential",

        "group_a":
            "planktonic_stationary",

        "group_b":
            "planktonic_exponential",
    },
    {
        "name":
            "Biofilm_vs_Stationary",

        "group_a":
            "biofilm_3d",

        "group_b":
            "planktonic_stationary",
    },
    {
        "name":
            "Colony15_vs_Exponential",

        "group_a":
            "colony_15h",

        "group_b":
            "planktonic_exponential",
    },
    {
        "name":
            "Colony40_vs_Exponential",

        "group_a":
            "colony_40h",

        "group_b":
            "planktonic_exponential",
    },
    {
        "name":
            "Biofilm_vs_Colony15",

        "group_a":
            "biofilm_3d",

        "group_b":
            "colony_15h",
    },
    {
        "name":
            "Biofilm_vs_Colony40",

        "group_a":
            "biofilm_3d",

        "group_b":
            "colony_40h",
    },
]


signature_gene_order = (
    available_signature[
        "canonical_pao1"
    ]
    .tolist()
)


frozen_effect = (
    available_signature[
        "frozen_effect"
    ]
    .to_numpy()
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

    group_a = contrast[
        "group_a"
    ]

    group_b = contrast[
        "group_b"
    ]


    group_a_mask = (
        study[
            "condition"
        ]
        == group_a
    )

    group_b_mask = (
        study[
            "condition"
        ]
        == group_b
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


    if (
        len(group_a_indices) != 3
        or len(group_b_indices) != 3
    ):

        raise RuntimeError(
            "Expected 3-vs-3 comparison for "
            + contrast_name
        )


    # --------------------------------------------------------
    # Score difference
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


    subset_scores = scores[
        subset_indices
    ]


    subset_labels = (
        study.loc[
            subset_mask,
            "condition",
        ]
        .eq(
            group_a
        )
        .astype(int)
        .to_numpy()
    )


    observed_auc = float(
        roc_auc_score(
            subset_labels,
            subset_scores,
        )
    )


    # --------------------------------------------------------
    # Gene-level effect
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


    effect_rho, effect_standard_p = spearmanr(
        frozen_effect,
        external_effect,
    )


    # --------------------------------------------------------
    # Exact 3-vs-3 balanced permutations
    #
    # 6 choose 3 = 20 assignments.
    # Minimum one-sided p = 1/20 = 0.05.
    # --------------------------------------------------------

    subset_rank = (
        X_rank.iloc[
            subset_indices
        ]
        .reset_index(
            drop=True
        )
    )


    local_indices = list(
        range(
            6
        )
    )


    permutation_records = []


    for local_a_tuple in combinations(
        local_indices,
        3,
    ):

        local_a = sorted(
            local_a_tuple
        )

        local_a_set = set(
            local_a
        )

        local_b = [
            index
            for index in local_indices
            if index not in local_a_set
        ]


        perm_score_difference = float(
            subset_scores[
                local_a
            ].mean()
            -
            subset_scores[
                local_b
            ].mean()
        )


        perm_labels = np.zeros(
            6,
            dtype=int,
        )

        perm_labels[
            local_a
        ] = 1


        perm_auc = float(
            roc_auc_score(
                perm_labels,
                subset_scores,
            )
        )


        perm_effect_series = (
            subset_rank.iloc[
                local_a
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


        perm_strong_same = (
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

                "same_direction_n":
                    int(
                        perm_same_direction.sum()
                    ),

                "strong_same_direction_n":
                    int(
                        perm_strong_same.sum()
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


    exact_score_p = float(
        (
            permutations[
                "score_difference"
            ]
            >= score_difference - 1e-12
        ).mean()
    )


    exact_auc_p = float(
        (
            permutations[
                "auc"
            ]
            >= observed_auc - 1e-12
        ).mean()
    )


    exact_same_p = float(
        (
            permutations[
                "same_direction_n"
            ]
            >= n_same
        ).mean()
    )


    exact_strong_p = float(
        (
            permutations[
                "strong_same_direction_n"
            ]
            >= n_strong
        ).mean()
    )


    exact_rho_p = float(
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
                group_a,

            "group_b":
                group_b,

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
                    effect_standard_p
                ),

            "same_direction_n":
                n_same,

            "same_direction_fraction":
                n_same / 250.0,

            "strong_same_direction_n":
                n_strong,

            "strong_same_direction_fraction":
                n_strong / 250.0,

            "n_exact_permutations":
                len(
                    permutations
                ),

            "exact_score_p":
                exact_score_p,

            "exact_auc_p":
                exact_auc_p,

            "exact_same_direction_p":
                exact_same_p,

            "exact_strong_direction_p":
                exact_strong_p,

            "exact_effect_rho_p":
                exact_rho_p,
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
                    bool(
                        np.sign(
                            frozen_value
                        )
                        ==
                        np.sign(
                            external_value
                        )
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
# Save
# ============================================================

contrast_results = pd.DataFrame(
    contrast_records
)


gene_effects = pd.DataFrame(
    gene_effect_records
)


contrast_results.to_csv(
    OUT_DIR
    / "GSE12207_specificity_contrasts.csv",
    index=False,
)


gene_effects.to_csv(
    OUT_DIR
    / "GSE12207_250_gene_contrast_effects.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print("FROZEN SIGNATURE SAMPLE SCORES")
print("=" * 100)

print(
    sample_scores.to_string(
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


display = (
    contrast_results[
        [
            "contrast",
            "score_difference",
            "auc",
            "effect_rho_vs_frozen",
            "same_direction_fraction",
            "strong_same_direction_fraction",
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
print("GSE12207 CONTRAST RESULTS")
print("=" * 100)

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# Trajectory
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
print("BIOLOGICAL TRAJECTORY")
print("=" * 100)


for condition in condition_order:

    print(
        condition
        + ": "
        + format(
            float(
                score_lookup[
                    condition
                ]
            ),
            ".6f",
        )
    )


print()

print(
    "Stationary - exponential: "
    + format(
        score_lookup[
            "planktonic_stationary"
        ]
        -
        score_lookup[
            "planktonic_exponential"
        ],
        ".6f",
    )
)

print(
    "Biofilm - exponential: "
    + format(
        score_lookup[
            "biofilm_3d"
        ]
        -
        score_lookup[
            "planktonic_exponential"
        ],
        ".6f",
    )
)

print(
    "Biofilm - stationary: "
    + format(
        score_lookup[
            "biofilm_3d"
        ]
        -
        score_lookup[
            "planktonic_stationary"
        ],
        ".6f",
    )
)


print()
print("=" * 100)
print("AUDIT 52 COMPLETE")
print("=" * 100)

print(
    "Available frozen-signature genes: 250 / 251"
)

print(
    "PA2463 was structurally unavailable on GPL84 "
    "and was not imputed."
)

print()

print(
    "All pairwise comparisons are 3-vs-3, giving "
    "20 balanced assignments and a minimum exact "
    "one-sided p-value of 0.05."
)

print()
