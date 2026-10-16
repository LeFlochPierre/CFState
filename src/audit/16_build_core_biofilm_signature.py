from pathlib import Path 
from itertools import combinations, product 
import numpy as np
import pandas as pd
from Bio import SeqIO


DATA_FILE = Path(
    "data/processed/biofilm/"
    "all_studies_harmonised_unscaled.csv"
)

GENBANK_FILE = Path(
    "references/mappings/NC_002516.2.gb"
)

OUT_DIR = Path(
    "results/audit/core_biofilm_signature"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load expression data
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
print("CF-STATE — Audit 16: statistically supported core biofilm signature")
print("=" * 100)

print(f"Samples: {X.shape[0]}")
print(f"Genes:   {X.shape[1]}")


# ============================================================
# Within-sample percentile ranks
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


# ============================================================
# Studies and sample indices
# ============================================================

studies = list(
    metadata["study"].unique()
)

study_indices = {}

for study in studies:

    idx = np.where(
        metadata["study"].to_numpy()
        == study
    )[0]

    if len(idx) != 6:
        raise RuntimeError(
            f"{study}: expected 6 samples, found {len(idx)}"
        )

    study_indices[study] = idx


# ============================================================
# All balanced 3-vs-3 label assignments
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
    f"Total cross-study assignments: "
    f"{len(assignments) ** 3}"
)


# ============================================================
# Precompute gene-effect vectors for every assignment
# ============================================================

effects_by_study = {}

for study in studies:

    block = X_rank[
        study_indices[study],
        :
    ]

    study_effects = []

    for bio_idx in assignments:

        bio_idx = np.array(
            bio_idx,
            dtype=int,
        )

        plank_idx = np.array(
            [
                i
                for i in range(6)
                if i not in bio_idx
            ],
            dtype=int,
        )

        effect = (
            block[bio_idx].mean(axis=0)
            -
            block[plank_idx].mean(axis=0)
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
# Locate actual biological assignment
# ============================================================

actual_assignment_index = {}

for study in studies:

    idx = study_indices[
        study
    ]

    states = metadata.iloc[
        idx
    ]["state"].to_numpy()

    bio_local = tuple(
        np.where(
            states == "biofilm"
        )[0]
    )

    actual_assignment_index[
        study
    ] = assignments.index(
        bio_local
    )


# ============================================================
# Actual study-specific effects
# ============================================================

observed_effects = np.vstack(
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


# ============================================================
# Cross-study consistency score
#
# If all effects > 0 OR all effects < 0:
#     score = minimum absolute effect
#
# Otherwise:
#     score = 0
# ============================================================

def consistency_score(
    effect_matrix,
):

    positive = np.all(
        effect_matrix > 0,
        axis=0,
    )

    negative = np.all(
        effect_matrix < 0,
        axis=0,
    )

    consistent = (
        positive
        |
        negative
    )

    minimum_effect = np.min(
        np.abs(
            effect_matrix
        ),
        axis=0,
    )

    return np.where(
        consistent,
        minimum_effect,
        0.0,
    )


observed_score = consistency_score(
    observed_effects
)


# ============================================================
# Exact empirical null
#
# Count how many of the 8000 balanced relabellings produce
# a consistency score >= the actual score for each gene.
# ============================================================

exceed_count = np.zeros(
    len(GENES),
    dtype=np.int64,
)

n_permutations = 0


for a, b, c in product(
    range(20),
    range(20),
    range(20),
):

    effect_matrix = np.vstack(
        [
            effects_by_study[
                studies[0]
            ][a],

            effects_by_study[
                studies[1]
            ][b],

            effects_by_study[
                studies[2]
            ][c],
        ]
    )

    null_score = consistency_score(
        effect_matrix
    )

    exceed_count += (
        null_score
        >= (
            observed_score
            - 1e-12
        )
    )

    n_permutations += 1


p_values = (
    exceed_count
    /
    n_permutations
)


# ============================================================
# Benjamini-Hochberg FDR
# ============================================================

def benjamini_hochberg(
    pvals,
):

    pvals = np.asarray(
        pvals,
        dtype=float,
    )

    n = len(pvals)

    order = np.argsort(
        pvals
    )

    ranked = pvals[
        order
    ]

    adjusted = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
        )
    )

    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    result = np.empty(
        n,
        dtype=float,
    )

    result[
        order
    ] = adjusted

    return result


q_values = benjamini_hochberg(
    p_values
)


# ============================================================
# Build result table
# ============================================================

result = pd.DataFrame(
    {
        "canonical_pao1":
            GENES,

        studies[0]:
            observed_effects[0],

        studies[1]:
            observed_effects[1],

        studies[2]:
            observed_effects[2],

        "consistency_score":
            observed_score,

        "exact_p_value":
            p_values,

        "fdr_q_value":
            q_values,
    }
)


# ============================================================
# Direction
# ============================================================

positive = np.all(
    observed_effects > 0,
    axis=0,
)

negative = np.all(
    observed_effects < 0,
    axis=0,
)

result["direction"] = np.where(
    positive,
    "higher_in_biofilm",
    np.where(
        negative,
        "lower_in_biofilm",
        "mixed",
    )
)


result[
    "median_effect"
] = np.median(
    observed_effects,
    axis=0,
)


# ============================================================
# Add PAO1 gene names / products from GenBank
# ============================================================

record = SeqIO.read(
    GENBANK_FILE,
    "genbank",
)

annotation = {}

for feature in record.features:

    if feature.type != "CDS":
        continue

    locus_tags = feature.qualifiers.get(
        "locus_tag",
        []
    )

    if not locus_tags:
        continue

    locus = locus_tags[0]

    gene = feature.qualifiers.get(
        "gene",
        [""],
    )[0]

    product_name = feature.qualifiers.get(
        "product",
        [""],
    )[0]

    annotation[
        locus
    ] = {
        "gene_name": gene,
        "product": product_name,
    }


result[
    "gene_name"
] = result[
    "canonical_pao1"
].map(
    lambda x:
        annotation.get(
            x,
            {},
        ).get(
            "gene_name",
            "",
        )
)


result[
    "product"
] = result[
    "canonical_pao1"
].map(
    lambda x:
        annotation.get(
            x,
            {},
        ).get(
            "product",
            "",
        )
)


# ============================================================
# Sort
# ============================================================

result = result.sort_values(
    [
        "fdr_q_value",
        "consistency_score",
    ],
    ascending=[
        True,
        False,
    ],
)


# ============================================================
# Significant sets
# ============================================================

sig_005 = result[
    (
        result["fdr_q_value"]
        <= 0.05
    )
    &
    (
        result["consistency_score"]
        > 0
    )
].copy()


sig_010 = result[
    (
        result["fdr_q_value"]
        <= 0.10
    )
    &
    (
        result["consistency_score"]
        > 0
    )
].copy()


# ============================================================
# Save
# ============================================================

result.to_csv(
    OUT_DIR
    / "all_genes_exact_permutation_results.csv",
    index=False,
)


sig_005.to_csv(
    OUT_DIR
    / "core_biofilm_signature_fdr05.csv",
    index=False,
)


sig_010.to_csv(
    OUT_DIR
    / "core_biofilm_signature_fdr10.csv",
    index=False,
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 100)
print("CORE SIGNATURE SUMMARY")
print("=" * 100)

print(
    f"Genes with FDR <= 0.05: "
    f"{len(sig_005)}"
)

print(
    f"Genes with FDR <= 0.10: "
    f"{len(sig_010)}"
)


if len(sig_005):

    n_up = (
        sig_005["direction"]
        == "higher_in_biofilm"
    ).sum()

    n_down = (
        sig_005["direction"]
        == "lower_in_biofilm"
    ).sum()

    print()
    print(
        f"FDR <= 0.05 higher in biofilm: "
        f"{n_up}"
    )

    print(
        f"FDR <= 0.05 lower in biofilm:  "
        f"{n_down}"
    )


# ============================================================
# Top results
# ============================================================

print()
print("=" * 100)
print("TOP STATISTICALLY SUPPORTED GENES")
print("=" * 100)

cols = [
    "canonical_pao1",
    "gene_name",
    "product",
    *studies,
    "direction",
    "consistency_score",
    "exact_p_value",
    "fdr_q_value",
]

print(
    result.head(40)[
        cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 16 COMPLETE")
print("=" * 100)

print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)

print()
