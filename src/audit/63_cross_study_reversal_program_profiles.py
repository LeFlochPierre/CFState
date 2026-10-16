from pathlib import Path

import numpy as np
import pandas as pd


MODEL_FILE = Path(
    "data/processed/cross_study_modeling/"
    "CFSTATE_binary_7studies_40samples_4853genes_ranked.csv"
)

PUBLISHED_GSE_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "published_vs_reprocessed/"
    "GSE299861_authors_published_counts_4164genes_ranked.csv"
)

EFFECT_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "published_vs_reprocessed/"
    "Audit59c_published_vs_reprocessed_effects.csv"
)

MODULE_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "reversal_biology/"
    "Audit61_genomic_modules_multigene.csv"
)

KEGG_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "reversal_biology/KEGG_enrichment/"
    "Audit62_significant_KEGG_FDR05.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "program_profiles"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


META_COLUMNS = [
    "study",
    "sample",
    "state",
    "label",
    "source_condition",
]


print()
print("=" * 100)
print(
    "CF-STATE — Audit 63: "
    "cross-study profiles of GSE299861 reversal programmes"
)
print("=" * 100)


# ============================================================
# Load comparison universe
# ============================================================

effect_reference = pd.read_csv(
    EFFECT_FILE
)


comparison_genes = (
    effect_reference[
        "canonical_pao1"
    ]
    .astype(str)
    .tolist()
)


if len(comparison_genes) != 4164:

    raise RuntimeError(
        "Expected 4,164 comparison genes."
    )


comparison_set = set(
    comparison_genes
)


# ============================================================
# Load six non-GSE299861 studies
# ============================================================

model = pd.read_csv(
    MODEL_FILE
)


other = model.loc[
    model[
        "study"
    ]
    != "GSE299861"
].copy()


if other[
    "study"
].nunique() != 6:

    raise RuntimeError(
        "Expected six non-GSE299861 studies."
    )


for gene in comparison_genes:

    if gene not in other.columns:

        raise RuntimeError(
            "Missing model gene: "
            + gene
        )


# Re-rank all samples within exactly the 4,164-gene universe.

other_rank_values = (
    other[
        comparison_genes
    ]
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


other_ranked = pd.concat(
    [
        other[
            META_COLUMNS
        ].reset_index(
            drop=True
        ),

        other_rank_values.reset_index(
            drop=True
        ),
    ],
    axis=1,
)


# ============================================================
# Load authors' GSE299861 published-count ranks
# ============================================================

published = pd.read_csv(
    PUBLISHED_GSE_FILE
)


required_published_columns = [
    "sample",
    "study",
    "state",
]


for column in required_published_columns:

    if column not in published.columns:

        raise RuntimeError(
            "Missing published GSE299861 column: "
            + column
        )


for gene in comparison_genes:

    if gene not in published.columns:

        raise RuntimeError(
            "Missing published GSE299861 gene: "
            + gene
        )


if len(published) != 6:

    raise RuntimeError(
        "Expected six published GSE299861 samples."
    )


published[
    "label"
] = (
    published[
        "state"
    ]
    .eq(
        "biofilm"
    )
    .astype(int)
)


# ============================================================
# Build programme definitions
# ============================================================

programme_records = []


# ------------------------------------------------------------
# Significant direction-specific KEGG pathways
# ------------------------------------------------------------

kegg = pd.read_csv(
    KEGG_FILE
)


for _, row in kegg.iterrows():

    gene_set_name = str(
        row[
            "gene_set"
        ]
    )


    # Skip ALL_ROBUST_REVERSED because it duplicates
    # the directional starch/sucrose result.

    if gene_set_name == "ALL_ROBUST_REVERSED":
        continue


    if (
        gene_set_name
        == "NORMALLY_BIOFILM_HIGHER_BUT_LOWER_IN_GSE299861"
    ):

        expected_sign = 1

    elif (
        gene_set_name
        == "NORMALLY_BIOFILM_LOWER_BUT_HIGHER_IN_GSE299861"
    ):

        expected_sign = -1

    else:

        continue


    genes = [
        gene
        for gene in str(
            row[
                "overlap_genes"
            ]
        ).split(
            ";"
        )
        if gene in comparison_set
    ]


    genes = sorted(
        set(
            genes
        )
    )


    if len(genes) < 2:
        continue


    programme_records.append(
        {
            "programme_type":
                "KEGG",

            "programme_id":
                str(
                    row[
                        "pathway"
                    ]
                ),

            "programme_name":
                str(
                    row[
                        "pathway_name"
                    ]
                ),

            "expected_sign":
                expected_sign,

            "n_genes":
                len(
                    genes
                ),

            "genes":
                ";".join(
                    genes
                ),

            "selection_strength":
                float(
                    -np.log10(
                        max(
                            float(
                                row[
                                    "fdr"
                                ]
                            ),
                            1e-300,
                        )
                    )
                ),
        }
    )


# ------------------------------------------------------------
# Genomic modules
#
# Keep modules containing >=3 robust reversed genes.
# ------------------------------------------------------------

modules = pd.read_csv(
    MODULE_FILE
)


modules = modules.loc[
    modules[
        "n_genes"
    ]
    >= 3
].copy()


for _, row in modules.iterrows():

    direction = str(
        row[
            "consensus_direction"
        ]
    )


    if direction == "biofilm_higher_elsewhere":

        expected_sign = 1

    elif direction == "biofilm_lower_elsewhere":

        expected_sign = -1

    else:

        raise RuntimeError(
            "Unknown module direction: "
            + direction
        )


    genes = [
        gene
        for gene in str(
            row[
                "genes"
            ]
        ).split(
            ";"
        )
        if gene in comparison_set
    ]


    genes = sorted(
        set(
            genes
        )
    )


    if len(genes) < 3:
        continue


    gene_names = ""

    if "gene_names" in modules.columns:

        gene_names = str(
            row[
                "gene_names"
            ]
        )


    programme_name = (
        "Module "
        + str(
            int(
                row[
                    "module_id"
                ]
            )
        )
    )


    if (
        gene_names
        and gene_names != "nan"
    ):

        programme_name = (
            programme_name
            + " ["
            + gene_names
            + "]"
        )


    programme_records.append(
        {
            "programme_type":
                "GENOMIC_MODULE",

            "programme_id":
                "module_"
                + str(
                    int(
                        row[
                            "module_id"
                        ]
                    )
                ),

            "programme_name":
                programme_name,

            "expected_sign":
                expected_sign,

            "n_genes":
                len(
                    genes
                ),

            "genes":
                ";".join(
                    genes
                ),

            "selection_strength":
                float(
                    row[
                        "sum_opposition_strength"
                    ]
                ),
        }
    )


programmes = pd.DataFrame(
    programme_records
)


if programmes.empty:

    raise RuntimeError(
        "No programmes constructed."
    )


programmes.to_csv(
    OUT_DIR
    / "Audit63_programme_definitions.csv",
    index=False,
)


print()
print("=" * 100)
print("PROGRAMMES")
print("=" * 100)

print(
    "Significant directional KEGG programmes: "
    + str(
        int(
            (
                programmes[
                    "programme_type"
                ]
                == "KEGG"
            ).sum()
        )
    )
)

print(
    "Genomic modules with >=3 genes: "
    + str(
        int(
            (
                programmes[
                    "programme_type"
                ]
                == "GENOMIC_MODULE"
            ).sum()
        )
    )
)

print(
    "Total programmes: "
    + str(
        len(
            programmes
        )
    )
)


# ============================================================
# Programme effect helper
#
# raw effect:
# mean rank in biofilm - mean rank in planktonic
#
# aligned effect:
# raw effect * expected consensus sign
#
# Therefore:
#   aligned > 0 = behaves like other-study consensus
#   aligned < 0 = reversed
# ============================================================

def calculate_programme_effect(
    table,
    genes,
):

    biofilm = table.loc[
        table[
            "label"
        ]
        == 1,
        genes,
    ]


    planktonic = table.loc[
        table[
            "label"
        ]
        == 0,
        genes,
    ]


    if len(biofilm) == 0:

        raise RuntimeError(
            "No biofilm samples."
        )


    if len(planktonic) == 0:

        raise RuntimeError(
            "No planktonic samples."
        )


    biofilm_gene_mean = (
        biofilm.mean(
            axis=0
        )
    )


    planktonic_gene_mean = (
        planktonic.mean(
            axis=0
        )
    )


    gene_effect = (
        biofilm_gene_mean
        - planktonic_gene_mean
    )


    programme_effect = float(
        gene_effect.mean()
    )


    return programme_effect


# ============================================================
# Calculate effects study-by-study
# ============================================================

study_records = []


other_studies = sorted(
    other_ranked[
        "study"
    ].unique()
)


all_studies = (
    other_studies
    + [
        "GSE299861",
    ]
)


for _, programme in programmes.iterrows():

    genes = str(
        programme[
            "genes"
        ]
    ).split(
        ";"
    )


    expected_sign = int(
        programme[
            "expected_sign"
        ]
    )


    for study in other_studies:

        study_table = other_ranked.loc[
            other_ranked[
                "study"
            ]
            == study
        ]


        raw_effect = calculate_programme_effect(
            study_table,
            genes,
        )


        aligned_effect = (
            raw_effect
            * expected_sign
        )


        study_records.append(
            {
                "programme_type":
                    programme[
                        "programme_type"
                    ],

                "programme_id":
                    programme[
                        "programme_id"
                    ],

                "programme_name":
                    programme[
                        "programme_name"
                    ],

                "n_genes":
                    int(
                        programme[
                            "n_genes"
                        ]
                    ),

                "expected_sign":
                    expected_sign,

                "study":
                    study,

                "raw_biofilm_minus_planktonic":
                    raw_effect,

                "aligned_effect":
                    aligned_effect,

                "reversed_relative_to_consensus":
                    bool(
                        aligned_effect
                        < 0
                    ),
            }
        )


    gse_raw_effect = calculate_programme_effect(
        published,
        genes,
    )


    gse_aligned_effect = (
        gse_raw_effect
        * expected_sign
    )


    study_records.append(
        {
            "programme_type":
                programme[
                    "programme_type"
                ],

            "programme_id":
                programme[
                    "programme_id"
                ],

            "programme_name":
                programme[
                    "programme_name"
                ],

            "n_genes":
                int(
                    programme[
                        "n_genes"
                    ]
                ),

            "expected_sign":
                expected_sign,

            "study":
                "GSE299861",

            "raw_biofilm_minus_planktonic":
                gse_raw_effect,

            "aligned_effect":
                gse_aligned_effect,

            "reversed_relative_to_consensus":
                bool(
                    gse_aligned_effect
                    < 0
                ),
        }
    )


profiles = pd.DataFrame(
    study_records
)


profiles.to_csv(
    OUT_DIR
    / "Audit63_programme_effects_by_study.csv",
    index=False,
)


# ============================================================
# Summarise each programme
# ============================================================

summary_records = []


for programme_id, group in profiles.groupby(
    "programme_id"
):

    programme_type = str(
        group[
            "programme_type"
        ].iloc[
            0
        ]
    )


    programme_name = str(
        group[
            "programme_name"
        ].iloc[
            0
        ]
    )


    n_genes = int(
        group[
            "n_genes"
        ].iloc[
            0
        ]
    )


    other_group = group.loc[
        group[
            "study"
        ]
        != "GSE299861"
    ]


    gse_group = group.loc[
        group[
            "study"
        ]
        == "GSE299861"
    ]


    other_aligned = (
        other_group[
            "aligned_effect"
        ]
        .to_numpy(
            dtype=float
        )
    )


    gse_aligned = float(
        gse_group[
            "aligned_effect"
        ].iloc[
            0
        ]
    )


    n_other_positive = int(
        (
            other_aligned
            > 0
        ).sum()
    )


    n_other_negative = int(
        (
            other_aligned
            < 0
        ).sum()
    )


    median_other = float(
        np.median(
            other_aligned
        )
    )


    mean_other = float(
        np.mean(
            other_aligned
        )
    )


    if median_other != 0:

        reversal_ratio = (
            gse_aligned
            / abs(
                median_other
            )
        )

    else:

        reversal_ratio = np.nan


    summary_records.append(
        {
            "programme_type":
                programme_type,

            "programme_id":
                programme_id,

            "programme_name":
                programme_name,

            "n_genes":
                n_genes,

            "other6_positive_aligned":
                n_other_positive,

            "other6_negative_aligned":
                n_other_negative,

            "median_other6_aligned_effect":
                median_other,

            "mean_other6_aligned_effect":
                mean_other,

            "GSE299861_aligned_effect":
                gse_aligned,

            "GSE299861_reversal_ratio_vs_other6_median":
                reversal_ratio,

            "GSE299861_reversed":
                bool(
                    gse_aligned
                    < 0
                ),
        }
    )


summary = pd.DataFrame(
    summary_records
)


summary = summary.sort_values(
    [
        "GSE299861_reversed",
        "GSE299861_aligned_effect",
    ],
    ascending=[
        False,
        True,
    ],
)


summary.to_csv(
    OUT_DIR
    / "Audit63_programme_summary.csv",
    index=False,
)


# ============================================================
# Wide aligned-effect matrix
# ============================================================

wide = profiles.pivot(
    index=[
        "programme_type",
        "programme_id",
        "programme_name",
    ],
    columns="study",
    values="aligned_effect",
)


wide = wide.reindex(
    columns=all_studies
)


wide.to_csv(
    OUT_DIR
    / "Audit63_aligned_effect_matrix.csv"
)


# ============================================================
# Print KEGG programmes
# ============================================================

print()
print("=" * 100)
print("SIGNIFICANT KEGG PROGRAMME PROFILES")
print("=" * 100)


kegg_summary = summary.loc[
    summary[
        "programme_type"
    ]
    == "KEGG"
]


display_columns = [
    "programme_id",
    "programme_name",
    "n_genes",
    "other6_positive_aligned",
    "median_other6_aligned_effect",
    "GSE299861_aligned_effect",
    "GSE299861_reversal_ratio_vs_other6_median",
]


print(
    kegg_summary[
        display_columns
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "Per-study aligned effects:"
)


kegg_ids = set(
    kegg_summary[
        "programme_id"
    ]
)


kegg_profiles = profiles.loc[
    profiles[
        "programme_id"
    ].isin(
        kegg_ids
    )
]


kegg_wide = kegg_profiles.pivot(
    index=[
        "programme_id",
        "programme_name",
    ],
    columns="study",
    values="aligned_effect",
)


kegg_wide = kegg_wide.reindex(
    columns=all_studies
)


print(
    kegg_wide
    .round(
        4
    )
    .to_string()
)


# ============================================================
# Print strongest genomic modules
# ============================================================

print()
print("=" * 100)
print("TOP GENOMIC MODULE PROGRAMMES")
print("=" * 100)


module_summary = summary.loc[
    summary[
        "programme_type"
    ]
    == "GENOMIC_MODULE"
].copy()


module_summary = module_summary.sort_values(
    "GSE299861_aligned_effect"
)


print(
    module_summary[
        [
            "programme_id",
            "programme_name",
            "n_genes",
            "other6_positive_aligned",
            "median_other6_aligned_effect",
            "GSE299861_aligned_effect",
            "GSE299861_reversal_ratio_vs_other6_median",
        ]
    ]
    .head(
        25
    )
    .to_string(
        index=False
    )
)


# ============================================================
# Programmes consistently aligned in all other six studies
# but reversed in GSE299861
# ============================================================

strict = summary.loc[
    (
        summary[
            "other6_positive_aligned"
        ]
        == 6
    )
    &
    (
        summary[
            "GSE299861_aligned_effect"
        ]
        < 0
    )
].copy()


strict = strict.sort_values(
    "GSE299861_aligned_effect"
)


strict.to_csv(
    OUT_DIR
    / "Audit63_programmes_6of6_elsewhere_but_reversed_GSE299861.csv",
    index=False,
)


print()
print("=" * 100)
print("6/6 CONSISTENT ELSEWHERE, REVERSED IN GSE299861")
print("=" * 100)


if strict.empty:

    print(
        "None."
    )

else:

    print(
        strict[
            [
                "programme_type",
                "programme_id",
                "programme_name",
                "n_genes",
                "median_other6_aligned_effect",
                "GSE299861_aligned_effect",
                "GSE299861_reversal_ratio_vs_other6_median",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# Study-level programme concordance
#
# Among selected programmes, what fraction point in the
# expected consensus direction in each study?
# ============================================================

concordance_records = []


for study in all_studies:

    subset = profiles.loc[
        profiles[
            "study"
        ]
        == study
    ]


    positive_fraction = float(
        (
            subset[
                "aligned_effect"
            ]
            > 0
        ).mean()
    )


    median_aligned = float(
        subset[
            "aligned_effect"
        ].median()
    )


    mean_aligned = float(
        subset[
            "aligned_effect"
        ].mean()
    )


    concordance_records.append(
        {
            "study":
                study,

            "n_programmes":
                len(
                    subset
                ),

            "fraction_programmes_consensus_direction":
                positive_fraction,

            "median_aligned_programme_effect":
                median_aligned,

            "mean_aligned_programme_effect":
                mean_aligned,
        }
    )


concordance = pd.DataFrame(
    concordance_records
)


concordance.to_csv(
    OUT_DIR
    / "Audit63_study_programme_concordance.csv",
    index=False,
)


print()
print("=" * 100)
print("STUDY-LEVEL PROGRAMME CONCORDANCE")
print("=" * 100)

print(
    concordance.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 63 COMPLETE")
print("=" * 100)

print(
    "Aligned effect > 0 means a programme behaves "
    "in the consensus biofilm direction."
)

print(
    "Aligned effect < 0 means that programme is reversed."
)

print()

print(
    "GSE299861 uses the authors' published counts."
)

print(
    "All other studies were re-ranked within exactly "
    "the same 4,164-gene comparison universe."
)

print()
