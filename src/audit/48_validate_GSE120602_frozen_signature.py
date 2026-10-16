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
    "data/processed/expanded_biofilm/GSE120602/"
    "GSE120602_PAO1_samples_x_4858.csv"
)

OUT_DIR = Path(
    "results/audit/external_validation/"
    "GSE120602"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 48: "
    "GSE120602 frozen 251-gene validation and specificity"
)
print("=" * 100)


# ============================================================
# Load frozen signature
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
        "Expected 251 frozen genes, found "
        + str(len(signature))
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


n_up = int(
    (
        signature[
            "frozen_effect"
        ]
        > 0
    ).sum()
)

n_down = int(
    (
        signature[
            "frozen_effect"
        ]
        < 0
    ).sum()
)


if n_up != 118 or n_down != 133:

    raise RuntimeError(
        "Expected 118 biofilm-higher and "
        "133 biofilm-lower genes."
    )


# ============================================================
# Load GSE120602
# ============================================================

study = pd.read_csv(
    STUDY_FILE
)


meta_columns = [
    "sample",
    "study",
    "gsm",
    "state",
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
    "planktonic": 2,
    "stationary": 2,
    "biofilm": 2,
    "persister_ciprofloxacin": 2,
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
        "Expected 4,858 gene columns, found "
        + str(len(gene_columns))
    )


signature_genes = (
    signature[
        "canonical_pao1"
    ]
    .tolist()
)


missing_signature = sorted(
    set(
        signature_genes
    )
    - set(
        gene_columns
    )
)


if missing_signature:

    raise RuntimeError(
        "Missing frozen signature genes: "
        + ", ".join(
            missing_signature
        )
    )


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
# Within-sample percentile-rank normalisation
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
            "gsm",
            "condition",
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
    "planktonic",
    "stationary",
    "biofilm",
    "persister_ciprofloxacin",
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
    / "GSE120602_frozen_signature_sample_scores.csv",
    index=False,
)


condition_scores.to_csv(
    OUT_DIR
    / "GSE120602_frozen_signature_condition_scores.csv",
    index=False,
)


# ============================================================
# Contrasts
#
# Group A is expected to be MORE biofilm-like than group B.
# ============================================================

contrasts = [
    {
        "name": "Biofilm_vs_Planktonic",
        "group_a": "biofilm",
        "group_b": "planktonic",
    },
    {
        "name": "Stationary_vs_Planktonic",
        "group_a": "stationary",
        "group_b": "planktonic",
    },
    {
        "name": "Biofilm_vs_Stationary",
        "group_a": "biofilm",
        "group_b": "stationary",
    },
    {
        "name": "Persister_vs_Planktonic",
        "group_a": "persister_ciprofloxacin",
        "group_b": "planktonic",
    },
    {
        "name": "Biofilm_vs_Persister",
        "group_a": "biofilm",
        "group_b": "persister_ciprofloxacin",
    },
]


frozen_effect = (
    signature[
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


    n_a = len(
        group_a_indices
    )

    n_b = len(
        group_b_indices
    )


    if n_a != 2 or n_b != 2:

        raise RuntimeError(
            "Expected 2-vs-2 samples for "
            + contrast_name
        )


    # --------------------------------------------------------
    # Frozen score comparison
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


    subset_scores = scores[
        subset_indices
    ]


    observed_auc = float(
        roc_auc_score(
            subset_labels,
            subset_scores,
        )
    )


    # --------------------------------------------------------
    # Gene-level contrast effect
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
            signature_genes
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
    # Exact 2-vs-2 balanced-label permutations
    #
    # 4 samples choose 2 = 6 possible assignments.
    # --------------------------------------------------------

    subset_rank = (
        X_rank.iloc[
            subset_indices
        ]
        .reset_index(
            drop=True
        )
    )


    local_indices = [
        0,
        1,
        2,
        3,
    ]


    permutation_records = []


    for local_a_tuple in combinations(
        local_indices,
        2,
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
            4,
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
                signature_genes
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
                n_same / 251.0,

            "strong_same_direction_n":
                n_strong,

            "strong_same_direction_fraction":
                n_strong / 251.0,

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
        signature_genes,
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
# Save results
# ============================================================

contrast_results = pd.DataFrame(
    contrast_records
)


gene_effects = pd.DataFrame(
    gene_effect_records
)


contrast_results.to_csv(
    OUT_DIR
    / "GSE120602_specificity_contrasts.csv",
    index=False,
)


gene_effects.to_csv(
    OUT_DIR
    / "GSE120602_251_gene_contrast_effects.csv",
    index=False,
)


# ============================================================
# Print scores
# ============================================================

print()
print("=" * 100)
print("FROZEN 251-GENE SAMPLE SCORES")
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


# ============================================================
# Print contrasts
# ============================================================

display_columns = [
    "contrast",
    "score_difference",
    "auc",
    "effect_rho_vs_frozen",
    "same_direction_fraction",
    "strong_same_direction_fraction",
    "exact_score_p",
    "exact_effect_rho_p",
]


display = (
    contrast_results[
        display_columns
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
print("GSE120602 CONTRAST RESULTS")
print("=" * 100)

print(
    display.to_string(
        index=False
    )
)


# ============================================================
# Key biological comparisons
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


biofilm_vs_planktonic = (
    score_lookup[
        "biofilm"
    ]
    -
    score_lookup[
        "planktonic"
    ]
)


stationary_vs_planktonic = (
    score_lookup[
        "stationary"
    ]
    -
    score_lookup[
        "planktonic"
    ]
)


biofilm_vs_stationary = (
    score_lookup[
        "biofilm"
    ]
    -
    score_lookup[
        "stationary"
    ]
)


persister_vs_planktonic = (
    score_lookup[
        "persister_ciprofloxacin"
    ]
    -
    score_lookup[
        "planktonic"
    ]
)


biofilm_vs_persister = (
    score_lookup[
        "biofilm"
    ]
    -
    score_lookup[
        "persister_ciprofloxacin"
    ]
)


print()
print("=" * 100)
print("BIOLOGICAL SPECIFICITY CHECK")
print("=" * 100)

print(
    "Biofilm - planktonic: "
    + format(
        biofilm_vs_planktonic,
        ".6f",
    )
)

print(
    "Stationary - planktonic: "
    + format(
        stationary_vs_planktonic,
        ".6f",
    )
)

print(
    "Biofilm - stationary: "
    + format(
        biofilm_vs_stationary,
        ".6f",
    )
)

print(
    "Persister - planktonic: "
    + format(
        persister_vs_planktonic,
        ".6f",
    )
)

print(
    "Biofilm - persister: "
    + format(
        biofilm_vs_persister,
        ".6f",
    )
)


print()
print("=" * 100)
print("AUDIT 48 COMPLETE")
print("=" * 100)

print(
    "Outputs saved to: "
    + str(
        OUT_DIR
    )
)

print()

print(
    "Note: every pairwise contrast contains only "
    "2 vs 2 samples."
)

print(
    "There are therefore only 6 balanced label "
    "assignments and the minimum possible exact "
    "one-sided p-value is 1/6 = 0.1667."
)

print()
