from pathlib import Path
from itertools import combinations

import pandas as pd


MODULE_FILE = Path(
    "results/audit/core_biofilm_modules/"
    "multi_gene_modules.csv"
)

SIGNATURE_FILE = Path(
    "results/audit/core_biofilm_signature/"
    "core_biofilm_signature_fdr05.csv"
)

NEW_STUDY_FILE = Path(
    "data/processed/expanded_biofilm/GSE299861/"
    "GSE299861_PAO1_samples_x_4858.csv"
)

OUT_DIR = Path(
    "results/audit/external_validation/"
    "GSE299861_modules"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 29: "
    "external validation of frozen genomic modules"
)
print("=" * 100)


# ============================================================
# Load frozen modules and signature
# ============================================================

modules = pd.read_csv(
    MODULE_FILE
)

signature = pd.read_csv(
    SIGNATURE_FILE
)


for column in ["module_id", "genes"]:
    if column not in modules.columns:
        raise RuntimeError(
            f"Missing module column: {column}"
        )


for column in [
    "canonical_pao1",
    "median_effect",
]:
    if column not in signature.columns:
        raise RuntimeError(
            f"Missing signature column: {column}"
        )


signature_effect = (
    signature
    .set_index("canonical_pao1")[
        "median_effect"
    ]
    .astype(float)
)


# ============================================================
# Load GSE299861
# ============================================================

new = pd.read_csv(
    NEW_STUDY_FILE
)

meta_columns = [
    "sample",
    "study",
    "state",
]

gene_columns = [
    column
    for column in new.columns
    if column not in meta_columns
]


if len(new) != 6:
    raise RuntimeError(
        "Expected six GSE299861 samples."
    )


n_biofilm = int(
    (new["state"] == "biofilm").sum()
)

n_planktonic = int(
    (new["state"] == "planktonic").sum()
)


if n_biofilm != 3:
    raise RuntimeError(
        "Expected three biofilm samples."
    )

if n_planktonic != 3:
    raise RuntimeError(
        "Expected three planktonic samples."
    )


X = (
    new[
        gene_columns
    ]
    .astype(float)
)


X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


# ============================================================
# Build frozen module definitions
# ============================================================

module_records = []


for _, row in modules.iterrows():

    gene_text = str(
        row["genes"]
    )

    genes = [
        gene.strip()
        for gene in gene_text.split(";")
        if gene.strip()
    ]


    genes = [
        gene
        for gene in genes
        if gene in signature_effect.index
    ]


    if len(genes) < 2:
        continue


    original_effects = (
        signature_effect.loc[
            genes
        ]
    )


    n_positive = int(
        (original_effects > 0).sum()
    )

    n_negative = int(
        (original_effects < 0).sum()
    )


    if n_positive == len(genes):

        expected_sign = 1

    elif n_negative == len(genes):

        expected_sign = -1

    else:

        raise RuntimeError(
            "Mixed original directions in module "
            f"{row['module_id']}."
        )


    missing_genes = [
        gene
        for gene in genes
        if gene not in gene_columns
    ]


    if missing_genes:

        raise RuntimeError(
            "Module genes missing from GSE299861: "
            + ", ".join(missing_genes)
        )


    module_records.append(
        {
            "module_id":
                row["module_id"],

            "n_genes":
                len(genes),

            "genes":
                ";".join(genes),

            "expected_sign":
                expected_sign,

            "original_median_effect":
                float(
                    original_effects.median()
                ),
        }
    )


frozen_modules = pd.DataFrame(
    module_records
)


print()
print(
    "Frozen multi-gene modules: "
    f"{len(frozen_modules)}"
)


# ============================================================
# Calculate module effects for a label assignment
# ============================================================

def calculate_module_effects(
    bio_indices,
):

    bio_indices = list(
        bio_indices
    )

    plank_indices = [
        index
        for index in range(
            len(new)
        )
        if index not in bio_indices
    ]


    result_records = []


    for _, module in frozen_modules.iterrows():

        genes = (
            module["genes"]
            .split(";")
        )


        bio_value = float(
            X_rank.iloc[
                bio_indices
            ][
                genes
            ]
            .mean(axis=1)
            .mean()
        )


        plank_value = float(
            X_rank.iloc[
                plank_indices
            ][
                genes
            ]
            .mean(axis=1)
            .mean()
        )


        external_effect = (
            bio_value
            - plank_value
        )


        expected_sign = int(
            module[
                "expected_sign"
            ]
        )


        aligned_effect = (
            expected_sign
            * external_effect
        )


        result_records.append(
            {
                "module_id":
                    module[
                        "module_id"
                    ],

                "external_effect":
                    external_effect,

                "aligned_effect":
                    aligned_effect,
            }
        )


    return pd.DataFrame(
        result_records
    )


# ============================================================
# Observed labels
# ============================================================

observed_bio_indices = (
    new.index[
        new["state"]
        == "biofilm"
    ]
    .tolist()
)


observed_effects = (
    calculate_module_effects(
        observed_bio_indices
    )
)


module_results = (
    frozen_modules.merge(
        observed_effects,
        on="module_id",
        how="left",
    )
)


module_results[
    "same_direction"
] = (
    module_results[
        "aligned_effect"
    ]
    > 0
)


module_results[
    "strong_same_direction"
] = (
    module_results[
        "aligned_effect"
    ]
    >= 0.05
)


n_same = int(
    module_results[
        "same_direction"
    ].sum()
)

n_strong = int(
    module_results[
        "strong_same_direction"
    ].sum()
)

median_aligned = float(
    module_results[
        "aligned_effect"
    ].median()
)


# ============================================================
# Exact permutation test
#
# 6 samples:
# C(6,3) = 20 balanced label assignments
# ============================================================

permutation_records = []


for bio_tuple in combinations(
    range(6),
    3,
):

    perm_effects = (
        calculate_module_effects(
            bio_tuple
        )
    )


    n_same_perm = int(
        (
            perm_effects[
                "aligned_effect"
            ]
            > 0
        ).sum()
    )


    n_strong_perm = int(
        (
            perm_effects[
                "aligned_effect"
            ]
            >= 0.05
        ).sum()
    )


    median_perm = float(
        perm_effects[
            "aligned_effect"
        ].median()
    )


    permutation_records.append(
        {
            "biofilm_indices":
                ";".join(
                    str(index)
                    for index in bio_tuple
                ),

            "n_same_direction":
                n_same_perm,

            "n_strong_same_direction":
                n_strong_perm,

            "median_aligned_effect":
                median_perm,
        }
    )


permutations = pd.DataFrame(
    permutation_records
)


same_p = float(
    (
        permutations[
            "n_same_direction"
        ]
        >= n_same
    ).mean()
)


strong_p = float(
    (
        permutations[
            "n_strong_same_direction"
        ]
        >= n_strong
    ).mean()
)


median_p = float(
    (
        permutations[
            "median_aligned_effect"
        ]
        >= median_aligned
    ).mean()
)


# ============================================================
# Save
# ============================================================

module_results = (
    module_results.sort_values(
        "aligned_effect",
        ascending=False,
    )
)


module_results.to_csv(
    OUT_DIR
    / "module_external_validation.csv",
    index=False,
)


permutations.to_csv(
    OUT_DIR
    / "module_exact_permutations.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

n_modules = len(
    module_results
)

same_percent = (
    100.0
    * n_same
    / n_modules
)

strong_percent = (
    100.0
    * n_strong
    / n_modules
)


print()
print("=" * 100)
print(
    "MODULE-LEVEL EXTERNAL VALIDATION"
)
print("=" * 100)

print(
    f"Same direction: "
    f"{n_same} / {n_modules} "
    f"({same_percent:.1f}%)"
)

print(
    f"Strong same direction: "
    f"{n_strong} / {n_modules} "
    f"({strong_percent:.1f}%)"
)

print(
    f"Median aligned module effect: "
    f"{median_aligned:.6f}"
)

print()

print(
    f"Exact same-direction p: "
    f"{same_p:.4f}"
)

print(
    f"Exact strong-direction p: "
    f"{strong_p:.4f}"
)

print(
    f"Exact median-effect p: "
    f"{median_p:.4f}"
)


print()
print("=" * 100)
print(
    "STRONGEST REPLICATED MODULES"
)
print("=" * 100)


replicated = (
    module_results[
        [
            "module_id",
            "n_genes",
            "genes",
            "original_median_effect",
            "external_effect",
            "aligned_effect",
        ]
    ]
    .head(15)
)


print(
    replicated.to_string(
        index=False
    )
)


print()
print("=" * 100)
print(
    "STRONGEST REVERSED MODULES"
)
print("=" * 100)


reversed_modules = (
    module_results[
        [
            "module_id",
            "n_genes",
            "genes",
            "original_median_effect",
            "external_effect",
            "aligned_effect",
        ]
    ]
    .tail(15)
    .sort_values(
        "aligned_effect"
    )
)


print(
    reversed_modules.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 29 COMPLETE")
print("=" * 100)

print(
    "Outputs saved to: "
    f"{OUT_DIR}"
)

print()
