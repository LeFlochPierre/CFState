from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATA_FILE = Path(
    "data/processed/biofilm/"
    "all_studies_harmonised_unscaled.csv"
)

OUT_DIR = Path(
    "results/audit/cross_study_biofilm_effects"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(DATA_FILE)

META = [
    "study",
    "sample",
    "state",
]

GENES = [
    x for x in df.columns
    if x not in META
]

metadata = df[META].copy()

X = df[GENES].astype(float)


print()
print("=" * 100)
print("CF-STATE — Audit 14: cross-study biofilm effects")
print("=" * 100)

print(f"Samples: {len(df)}")
print(f"Genes:   {len(GENES)}")


# ============================================================
# Rank each gene within each sample
#
# Produces values from ~0 to 1.
# This removes absolute expression-scale differences.
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


# ============================================================
# Calculate biofilm - planktonic effect
# independently inside each study
# ============================================================

studies = list(
    metadata["study"].unique()
)

effects = pd.DataFrame(
    index=GENES
)


for study in studies:

    study_mask = (
        metadata["study"] == study
    )

    biofilm_mask = (
        study_mask
        &
        (metadata["state"] == "biofilm")
    )

    plank_mask = (
        study_mask
        &
        (metadata["state"] == "planktonic")
    )

    biofilm_mean = (
        X_rank.loc[
            biofilm_mask
        ].mean(axis=0)
    )

    plank_mean = (
        X_rank.loc[
            plank_mask
        ].mean(axis=0)
    )

    effect = (
        biofilm_mean
        - plank_mean
    )

    effects[
        study
    ] = effect


# ============================================================
# Pairwise correlation between study effect vectors
# ============================================================

print()
print("=" * 100)
print("PAIRWISE STUDY EFFECT CORRELATIONS")
print("=" * 100)

correlation_records = []

for i in range(len(studies)):

    for j in range(
        i + 1,
        len(studies),
    ):

        a = studies[i]
        b = studies[j]

        rho, pvalue = spearmanr(
            effects[a],
            effects[b],
        )

        correlation_records.append(
            {
                "study_1": a,
                "study_2": b,
                "spearman_rho": rho,
                "p_value": pvalue,
            }
        )

        print(
            f"{a} vs {b}: "
            f"rho={rho:.4f}, "
            f"p={pvalue:.3e}"
        )


pd.DataFrame(
    correlation_records
).to_csv(
    OUT_DIR
    / "study_effect_correlations.csv",
    index=False,
)


# ============================================================
# Direction consistency
# ============================================================

effect_array = effects[
    studies
].to_numpy()


all_positive = np.all(
    effect_array > 0,
    axis=1,
)

all_negative = np.all(
    effect_array < 0,
    axis=1,
)

all_same_direction = (
    all_positive
    |
    all_negative
)


effects[
    "direction_consistent"
] = all_same_direction

effects[
    "direction"
] = np.where(
    all_positive,
    "higher_in_biofilm",
    np.where(
        all_negative,
        "lower_in_biofilm",
        "mixed",
    )
)


# ============================================================
# Consensus effect
#
# Median across studies is robust to one unusually large
# experiment.
# ============================================================

effects[
    "median_effect"
] = effects[
    studies
].median(
    axis=1
)

effects[
    "mean_effect"
] = effects[
    studies
].mean(
    axis=1
)

effects[
    "min_abs_effect"
] = effects[
    studies
].abs().min(
    axis=1
)


# ============================================================
# Sort
# ============================================================

effects.index.name = (
    "canonical_pao1"
)

effects = effects.reset_index()


consistent = effects[
    effects[
        "direction_consistent"
    ]
].copy()


consistent[
    "consensus_strength"
] = (
    consistent[
        "median_effect"
    ].abs()
)


consistent = consistent.sort_values(
    "consensus_strength",
    ascending=False,
)


# ============================================================
# Summary
# ============================================================

n_up = int(
    (
        effects["direction"]
        == "higher_in_biofilm"
    ).sum()
)

n_down = int(
    (
        effects["direction"]
        == "lower_in_biofilm"
    ).sum()
)

n_consistent = (
    n_up + n_down
)


print()
print("=" * 100)
print("DIRECTION CONSISTENCY")
print("=" * 100)

print(
    f"Genes higher in biofilm "
    f"in all 3 studies:   {n_up}"
)

print(
    f"Genes lower in biofilm "
    f"in all 3 studies:   {n_down}"
)

print(
    f"Direction-consistent total:   "
    f"{n_consistent}"
)

print(
    f"Fraction of all genes:        "
    f"{n_consistent / len(GENES):.3f}"
)


# ============================================================
# Strong consensus genes
#
# Require consistent direction AND a minimum effect in
# every single study.
#
# Rank scale ranges from 0 to 1.
# Threshold 0.05 means at least a 5-percentile shift
# in every experiment.
# ============================================================

thresholds = [
    0.01,
    0.025,
    0.05,
    0.10,
]


print()
print("=" * 100)
print("CONSISTENCY BY MINIMUM EFFECT SIZE")
print("=" * 100)

threshold_records = []

for threshold in thresholds:

    strong = consistent[
        consistent[
            "min_abs_effect"
        ] >= threshold
    ]

    up = int(
        (
            strong["direction"]
            == "higher_in_biofilm"
        ).sum()
    )

    down = int(
        (
            strong["direction"]
            == "lower_in_biofilm"
        ).sum()
    )

    print(
        f"Minimum rank shift >= "
        f"{threshold:.3f}: "
        f"{len(strong)} genes "
        f"({up} up, {down} down)"
    )

    threshold_records.append(
        {
            "minimum_rank_effect":
                threshold,

            "n_genes":
                len(strong),

            "higher_in_biofilm":
                up,

            "lower_in_biofilm":
                down,
        }
    )


pd.DataFrame(
    threshold_records
).to_csv(
    OUT_DIR
    / "effect_threshold_summary.csv",
    index=False,
)


# ============================================================
# Top genes
# ============================================================

print()
print("=" * 100)
print("TOP CONSISTENTLY HIGHER-IN-BIOFILM GENES")
print("=" * 100)

top_up = consistent[
    consistent["direction"]
    == "higher_in_biofilm"
].head(25)

print(
    top_up[
        [
            "canonical_pao1",
            *studies,
            "median_effect",
            "min_abs_effect",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("TOP CONSISTENTLY LOWER-IN-BIOFILM GENES")
print("=" * 100)

top_down = consistent[
    consistent["direction"]
    == "lower_in_biofilm"
].head(25)

print(
    top_down[
        [
            "canonical_pao1",
            *studies,
            "median_effect",
            "min_abs_effect",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Save
# ============================================================

effects.to_csv(
    OUT_DIR
    / "all_gene_study_effects.csv",
    index=False,
)

consistent.to_csv(
    OUT_DIR
    / "direction_consistent_genes.csv",
    index=False,
)

top_up.to_csv(
    OUT_DIR
    / "top_25_higher_in_biofilm.csv",
    index=False,
)

top_down.to_csv(
    OUT_DIR
    / "top_25_lower_in_biofilm.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 14 COMPLETE")
print("=" * 100)

print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)
print()
