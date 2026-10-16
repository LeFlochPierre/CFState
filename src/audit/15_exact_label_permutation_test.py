from pathlib import Path
from itertools import combinations, product

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATA_FILE = Path(
    "data/processed/biofilm/"
    "all_studies_harmonised_unscaled.csv"
)

OUT_DIR = Path(
    "results/audit/biofilm_permutation_test"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(DATA_FILE)

META = [
    "study",
    "sample",
    "state",
]

GENES = [
    c
    for c in df.columns
    if c not in META
]

metadata = df[META].copy()

X = df[GENES].astype(float)


print()
print("=" * 100)
print("CF-STATE — Audit 15: exact balanced-label permutation test")
print("=" * 100)

print(f"Samples: {X.shape[0]}")
print(f"Genes:   {X.shape[1]}")


# ============================================================
# Convert every sample to within-sample percentile ranks
# Same representation as Audit 14
# ============================================================

X_rank = (
    X.rank(
        axis=1,
        method="average",
        pct=True,
    )
    .to_numpy(
        dtype=float
    )
)


studies = list(
    metadata["study"].unique()
)


# ============================================================
# Study sample indices
# ============================================================

study_indices = {}

for study in studies:

    idx = np.where(
        metadata["study"].to_numpy()
        == study
    )[0]

    if len(idx) != 6:
        raise RuntimeError(
            f"{study}: expected 6 samples, "
            f"found {len(idx)}"
        )

    study_indices[study] = idx


# ============================================================
# Generate all possible 3-vs-3 assignments
#
# For each study there are C(6,3) = 20 assignments.
#
# Each assignment stores which LOCAL sample indices
# are called 'biofilm'. The remaining three are called
# planktonic.
# ============================================================

assignments = list(
    combinations(
        range(6),
        3,
    )
)

print()
print(
    f"Balanced assignments per study: "
    f"{len(assignments)}"
)

print(
    f"Total combinations across studies: "
    f"{len(assignments) ** len(studies)}"
)


# ============================================================
# Precompute effect vectors
#
# effects_by_study[study]
# has shape:
#
#   20 possible labelings × 4858 genes
#
# This makes testing all 8000 combinations fast.
# ============================================================

effects_by_study = {}

for study in studies:

    global_idx = study_indices[
        study
    ]

    block = X_rank[
        global_idx,
        :
    ]

    study_effects = []

    for bio_local in assignments:

        bio_local = np.array(
            bio_local,
            dtype=int,
        )

        plank_local = np.array(
            [
                i
                for i in range(6)
                if i not in bio_local
            ],
            dtype=int,
        )

        bio_mean = block[
            bio_local,
            :
        ].mean(
            axis=0
        )

        plank_mean = block[
            plank_local,
            :
        ].mean(
            axis=0
        )

        effect = (
            bio_mean
            - plank_mean
        )

        study_effects.append(
            effect
        )

    effects_by_study[
        study
    ] = np.vstack(
        study_effects
    )


# ============================================================
# Find the actual label assignment
# ============================================================

actual_assignment_index = {}

for study in studies:

    global_idx = study_indices[
        study
    ]

    local_states = (
        metadata.iloc[
            global_idx
        ]["state"]
        .to_numpy()
    )

    actual_bio_local = tuple(
        np.where(
            local_states == "biofilm"
        )[0]
    )

    if actual_bio_local not in assignments:
        raise RuntimeError(
            f"{study}: actual assignment not found."
        )

    actual_assignment_index[
        study
    ] = assignments.index(
        actual_bio_local
    )


print()
print("Actual assignment indices:")

for study in studies:

    print(
        f"  {study}: "
        f"{actual_assignment_index[study]}"
    )


# ============================================================
# Statistic helper
# ============================================================

def calculate_statistics(
    effect_matrix,
):
    """
    effect_matrix:
        shape = 3 studies × genes
    """

    # ----------------------------------------
    # Pairwise Spearman correlation
    # ----------------------------------------

    rhos = []

    for i in range(
        effect_matrix.shape[0]
    ):

        for j in range(
            i + 1,
            effect_matrix.shape[0],
        ):

            rho, _ = spearmanr(
                effect_matrix[i],
                effect_matrix[j],
            )

            rhos.append(
                float(rho)
            )

    mean_rho = float(
        np.mean(rhos)
    )

    # ----------------------------------------
    # Direction consistency
    # ----------------------------------------

    all_positive = np.all(
        effect_matrix > 0,
        axis=0,
    )

    all_negative = np.all(
        effect_matrix < 0,
        axis=0,
    )

    consistent = (
        all_positive
        |
        all_negative
    )

    n_consistent = int(
        consistent.sum()
    )

    # ----------------------------------------
    # Minimum magnitude across all studies
    # ----------------------------------------

    min_abs = np.min(
        np.abs(
            effect_matrix
        ),
        axis=0,
    )

    n_effect_005 = int(
        (
            consistent
            &
            (min_abs >= 0.05)
        ).sum()
    )

    n_effect_010 = int(
        (
            consistent
            &
            (min_abs >= 0.10)
        ).sum()
    )

    # ----------------------------------------
    # Consensus strength
    # ----------------------------------------

    median_effect = np.median(
        effect_matrix,
        axis=0,
    )

    consensus_strength = float(
        np.sum(
            np.abs(
                median_effect[
                    consistent
                ]
            )
        )
    )

    return {
        "mean_pairwise_spearman":
            mean_rho,

        "n_direction_consistent":
            n_consistent,

        "n_consistent_min_005":
            n_effect_005,

        "n_consistent_min_010":
            n_effect_010,

        "consensus_strength":
            consensus_strength,
    }


# ============================================================
# Actual observed statistics
# ============================================================

actual_effects = np.vstack(
    [
        effects_by_study[
            study
        ][
            actual_assignment_index[
                study
            ]
        ]
        for study in studies
    ]
)


observed = calculate_statistics(
    actual_effects
)


print()
print("=" * 100)
print("OBSERVED STATISTICS")
print("=" * 100)

for key, value in observed.items():

    print(
        f"{key:30s} "
        f"{value}"
    )


# ============================================================
# Exhaustive null distribution
#
# 20^3 = 8000 combinations.
# ============================================================

records = []

study_1 = studies[0]
study_2 = studies[1]
study_3 = studies[2]


for a, b, c in product(
    range(20),
    range(20),
    range(20),
):

    effect_matrix = np.vstack(
        [
            effects_by_study[
                study_1
            ][a],

            effects_by_study[
                study_2
            ][b],

            effects_by_study[
                study_3
            ][c],
        ]
    )

    stats = calculate_statistics(
        effect_matrix
    )

    records.append(
        {
            "assignment_1": a,
            "assignment_2": b,
            "assignment_3": c,
            **stats,
        }
    )


null_df = pd.DataFrame(
    records
)


null_df.to_csv(
    OUT_DIR
    / "exact_null_distribution.csv",
    index=False,
)


# ============================================================
# Exact one-sided permutation p-values
#
# Larger statistic = stronger cross-study agreement.
# Actual configuration is itself one of the 8000 possible
# assignments, so no +1 correction is required.
# ============================================================

test_rows = []

for statistic, observed_value in observed.items():

    null_values = null_df[
        statistic
    ].to_numpy()

    n_equal_or_more_extreme = int(
        np.sum(
            null_values
            >= observed_value
        )
    )

    p_value = (
        n_equal_or_more_extreme
        /
        len(null_values)
    )

    percentile = float(
        100
        * np.mean(
            null_values
            <= observed_value
        )
    )

    test_rows.append(
        {
            "statistic":
                statistic,

            "observed":
                observed_value,

            "null_mean":
                float(
                    np.mean(
                        null_values
                    )
                ),

            "null_median":
                float(
                    np.median(
                        null_values
                    )
                ),

            "null_95th_percentile":
                float(
                    np.percentile(
                        null_values,
                        95,
                    )
                ),

            "null_99th_percentile":
                float(
                    np.percentile(
                        null_values,
                        99,
                    )
                ),

            "observed_percentile":
                percentile,

            "exact_p_value":
                p_value,

            "n_null_at_least_observed":
                n_equal_or_more_extreme,

            "n_total_permutations":
                len(null_values),
        }
    )


result_df = pd.DataFrame(
    test_rows
)


result_df.to_csv(
    OUT_DIR
    / "exact_permutation_test_results.csv",
    index=False,
)


print()
print("=" * 100)
print("EXACT PERMUTATION TEST RESULTS")
print("=" * 100)

print(
    result_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 15 COMPLETE")
print("=" * 100)

print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)

print()
print(
    "Interpretation:"
)

print(
    "Small p-values mean the true biofilm/planktonic "
    "labels produce stronger cross-study agreement "
    "than most other balanced 3-vs-3 labelings."
)

print()
