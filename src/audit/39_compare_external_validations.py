from pathlib import Path 
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


# ============================================================
# Paths
# ============================================================

GSE299861_FILE = Path(
    "results/audit/external_validation/GSE299861/"
    "frozen_251_gene_external_validation.csv"
)

GSE191103_FILE = Path(
    "results/audit/external_validation/GSE191103/"
    "frozen_251_gene_external_validation.csv"
)

OUT_DIR = Path(
    "results/audit/external_validation/"
    "cross_external_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 39: "
    "compare independent external validations"
)
print("=" * 100)


# ============================================================
# Load
# ============================================================

a = pd.read_csv(
    GSE299861_FILE
)

b = pd.read_csv(
    GSE191103_FILE
)


required_a = [
    "canonical_pao1",
    "frozen_effect",
    "GSE299861_effect",
]

required_b = [
    "canonical_pao1",
    "frozen_effect",
    "GSE191103_effect",
]


for column in required_a:
    if column not in a.columns:
        raise RuntimeError(
            "Missing GSE299861 column: "
            + column
        )


for column in required_b:
    if column not in b.columns:
        raise RuntimeError(
            "Missing GSE191103 column: "
            + column
        )


if len(a) != 251:
    raise RuntimeError(
        "Expected 251 GSE299861 genes."
    )


if len(b) != 251:
    raise RuntimeError(
        "Expected 251 GSE191103 genes."
    )


# ============================================================
# Merge
# ============================================================

comparison = (
    a[
        [
            "canonical_pao1",
            "frozen_effect",
            "GSE299861_effect",
        ]
    ]
    .merge(
        b[
            [
                "canonical_pao1",
                "frozen_effect",
                "GSE191103_effect",
            ]
        ],
        on="canonical_pao1",
        suffixes=(
            "_299861",
            "_191103",
        ),
        validate="one_to_one",
    )
)


if len(comparison) != 251:
    raise RuntimeError(
        "External datasets did not merge to 251 genes."
    )


# Make sure frozen effects agree
difference = (
    comparison[
        "frozen_effect_299861"
    ]
    -
    comparison[
        "frozen_effect_191103"
    ]
)


if not np.allclose(
    difference,
    0,
):

    raise RuntimeError(
        "Frozen effects differ between external analyses."
    )


comparison[
    "frozen_effect"
] = comparison[
    "frozen_effect_299861"
]


comparison = comparison.drop(
    columns=[
        "frozen_effect_299861",
        "frozen_effect_191103",
    ]
)


# ============================================================
# Direction checks
# ============================================================

frozen_sign = np.sign(
    comparison[
        "frozen_effect"
    ]
)

sign_299861 = np.sign(
    comparison[
        "GSE299861_effect"
    ]
)

sign_191103 = np.sign(
    comparison[
        "GSE191103_effect"
    ]
)


comparison[
    "same_direction_299861"
] = (
    sign_299861
    ==
    frozen_sign
)


comparison[
    "same_direction_191103"
] = (
    sign_191103
    ==
    frozen_sign
)


comparison[
    "same_direction_both"
] = (
    comparison[
        "same_direction_299861"
    ]
    &
    comparison[
        "same_direction_191103"
    ]
)


# ============================================================
# Strong effect checks
#
# Same threshold used previously:
# |external effect| >= 0.05
# ============================================================

comparison[
    "strong_299861"
] = (
    comparison[
        "same_direction_299861"
    ]
    &
    (
        comparison[
            "GSE299861_effect"
        ].abs()
        >= 0.05
    )
)


comparison[
    "strong_191103"
] = (
    comparison[
        "same_direction_191103"
    ]
    &
    (
        comparison[
            "GSE191103_effect"
        ].abs()
        >= 0.05
    )
)


comparison[
    "strong_both"
] = (
    comparison[
        "strong_299861"
    ]
    &
    comparison[
        "strong_191103"
    ]
)


# ============================================================
# All three directions agree
#
# This means:
#
# discovery sign =
# GSE299861 sign =
# GSE191103 sign
# ============================================================

n_same_both = int(
    comparison[
        "same_direction_both"
    ].sum()
)


n_strong_both = int(
    comparison[
        "strong_both"
    ].sum()
)


# ============================================================
# External-study effect correlation
# ============================================================

external_rho, external_p = spearmanr(
    comparison[
        "GSE299861_effect"
    ],
    comparison[
        "GSE191103_effect"
    ],
)


# ============================================================
# Separate original biofilm-high and biofilm-low
# ============================================================

original_high = (
    comparison[
        "frozen_effect"
    ]
    > 0
)

original_low = (
    comparison[
        "frozen_effect"
    ]
    < 0
)


n_high = int(
    original_high.sum()
)

n_low = int(
    original_low.sum()
)


high_same_both = int(
    (
        comparison[
            "same_direction_both"
        ]
        &
        original_high
    ).sum()
)


low_same_both = int(
    (
        comparison[
            "same_direction_both"
        ]
        &
        original_low
    ).sum()
)


high_strong_both = int(
    (
        comparison[
            "strong_both"
        ]
        &
        original_high
    ).sum()
)


low_strong_both = int(
    (
        comparison[
            "strong_both"
        ]
        &
        original_low
    ).sum()
)


# ============================================================
# Conservative cross-external score
#
# Minimum signed magnitude across both external studies.
#
# Positive values indicate the gene reproduces in the
# discovery direction in both studies.
# ============================================================

aligned_299861 = (
    frozen_sign
    *
    comparison[
        "GSE299861_effect"
    ]
)


aligned_191103 = (
    frozen_sign
    *
    comparison[
        "GSE191103_effect"
    ]
)


comparison[
    "cross_external_min_aligned_effect"
] = np.minimum(
    aligned_299861,
    aligned_191103,
)


comparison[
    "cross_external_mean_aligned_effect"
] = (
    aligned_299861
    +
    aligned_191103
) / 2.0


# ============================================================
# Sort conserved genes
# ============================================================

conserved = (
    comparison.loc[
        comparison[
            "same_direction_both"
        ]
    ]
    .sort_values(
        "cross_external_min_aligned_effect",
        ascending=False,
    )
)


strong_conserved = (
    comparison.loc[
        comparison[
            "strong_both"
        ]
    ]
    .sort_values(
        "cross_external_min_aligned_effect",
        ascending=False,
    )
)


# ============================================================
# Save
# ============================================================

comparison.to_csv(
    OUT_DIR
    / "all_251_cross_external_comparison.csv",
    index=False,
)


conserved.to_csv(
    OUT_DIR
    / "same_direction_in_both_external_studies.csv",
    index=False,
)


strong_conserved.to_csv(
    OUT_DIR
    / "strong_same_direction_in_both_external_studies.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

same_percent = (
    100.0
    * n_same_both
    / 251
)


strong_percent = (
    100.0
    * n_strong_both
    / 251
)


print()
print("=" * 100)
print("CROSS-EXTERNAL REPLICATION")
print("=" * 100)

print(
    "Same direction in BOTH external studies: "
    + str(n_same_both)
    + " / 251 ("
    + format(
        same_percent,
        ".1f",
    )
    + "%)"
)


print(
    "Strong same direction in BOTH: "
    + str(n_strong_both)
    + " / 251 ("
    + format(
        strong_percent,
        ".1f",
    )
    + "%)"
)


print()
print(
    "Original biofilm-high genes reproduced "
    "in both: "
    + str(high_same_both)
    + " / "
    + str(n_high)
)


print(
    "Original biofilm-low genes reproduced "
    "in both: "
    + str(low_same_both)
    + " / "
    + str(n_low)
)


print()
print(
    "Strong original biofilm-high genes "
    "in both: "
    + str(high_strong_both)
)


print(
    "Strong original biofilm-low genes "
    "in both: "
    + str(low_strong_both)
)


print()
print("=" * 100)
print("EXTERNAL STUDY AGREEMENT")
print("=" * 100)

print(
    "GSE299861 vs GSE191103 effect rho: "
    + format(
        external_rho,
        ".4f",
    )
)

print(
    "Standard Spearman p: "
    + format(
        external_p,
        ".3e",
    )
)


print()
print("=" * 100)
print(
    "TOP CONSERVED GENES ACROSS BOTH "
    "EXTERNAL STUDIES"
)
print("=" * 100)


columns_to_show = [
    "canonical_pao1",
    "frozen_effect",
    "GSE299861_effect",
    "GSE191103_effect",
    "cross_external_min_aligned_effect",
]


print(
    conserved[
        columns_to_show
    ]
    .head(30)
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 39 COMPLETE")
print("=" * 100)

print(
    "Outputs saved to: "
    + str(OUT_DIR)
)

print()
