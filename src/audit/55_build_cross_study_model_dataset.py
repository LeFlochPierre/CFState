from pathlib import Path

import numpy as np
import pandas as pd


OUT_DIR = Path(
    "results/audit/cross_study_modeling"
)

PROCESSED_DIR = Path(
    "data/processed/cross_study_modeling"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


COMMON_GENE_FILE = Path(
    "results/audit/expanded_biofilm/GSE12207/"
    "finalization/"
    "GSE12207_cross_platform_gene_universe_4853.csv"
)


FILES = {
    "GSE136111": Path(
        "data/processed/biofilm/"
        "GSE136111_harmonised_unscaled_sample_by_gene.csv"
    ),

    "GSE203348": Path(
        "data/processed/biofilm/"
        "GSE203348_harmonised_unscaled_sample_by_gene.csv"
    ),

    "GSE87213": Path(
        "data/processed/biofilm/"
        "GSE87213_harmonised_unscaled_sample_by_gene.csv"
    ),

    "GSE299861": Path(
        "data/processed/expanded_biofilm/GSE299861/"
        "GSE299861_PAO1_samples_x_4858.csv"
    ),

    "GSE191103": Path(
        "data/processed/expanded_biofilm/GSE191103/"
        "GSE191103_PA14_samples_x_4858.csv"
    ),

    "GSE120602": Path(
        "data/processed/expanded_biofilm/GSE120602/"
        "GSE120602_PAO1_samples_x_4858.csv"
    ),

    "GSE12207": Path(
        "data/processed/expanded_biofilm/GSE12207/"
        "GSE12207_PAO1_samples_x_4853.csv"
    ),
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 55: "
    "build cross-study biofilm/planktonic modelling dataset"
)
print("=" * 100)


# ============================================================
# Common 4,853-gene universe
# ============================================================

common = pd.read_csv(
    COMMON_GENE_FILE
)


if "canonical_pao1" not in common.columns:

    raise RuntimeError(
        "Common gene file missing canonical_pao1."
    )


common_genes = (
    common[
        "canonical_pao1"
    ]
    .astype(str)
    .tolist()
)


if len(common_genes) != 4853:

    raise RuntimeError(
        "Expected 4,853 common genes, found "
        + str(len(common_genes))
    )


if len(set(common_genes)) != 4853:

    raise RuntimeError(
        "Common gene universe contains duplicates."
    )


common_set = set(
    common_genes
)


# ============================================================
# Helpers
# ============================================================

def check_genes(table, study):

    missing = sorted(
        common_set
        - set(table.columns)
    )

    if missing:

        raise RuntimeError(
            study
            + " is missing "
            + str(len(missing))
            + " common genes."
        )


def make_record(
    study,
    sample,
    state,
    source_condition,
    values,
):

    if state == "biofilm":

        label = 1

    elif state == "planktonic":

        label = 0

    else:

        raise RuntimeError(
            "Invalid binary state: "
            + str(state)
        )


    record = {
        "study": study,
        "sample": sample,
        "state": state,
        "label": label,
        "source_condition": source_condition,
    }


    for gene in common_genes:

        record[
            gene
        ] = values[
            gene
        ]


    return record


records = []


# ============================================================
# GSE136111
# ============================================================

study = "GSE136111"

table = pd.read_csv(
    FILES[
        study
    ]
)

sample_column = table.columns[
    0
]

check_genes(
    table,
    study,
)


for _, row in table.iterrows():

    sample = str(
        row[
            sample_column
        ]
    )


    if sample.startswith(
        "B_"
    ):

        state = "biofilm"

    elif sample.startswith(
        "P_"
    ):

        state = "planktonic"

    else:

        raise RuntimeError(
            "Unknown GSE136111 sample label: "
            + sample
        )


    records.append(
        make_record(
            study,
            sample,
            state,
            state,
            row,
        )
    )


# ============================================================
# GSE203348
# ============================================================

study = "GSE203348"

table = pd.read_csv(
    FILES[
        study
    ]
)

sample_column = table.columns[
    0
]

check_genes(
    table,
    study,
)


for _, row in table.iterrows():

    sample = str(
        row[
            sample_column
        ]
    )


    if sample.startswith(
        "BioFWT"
    ):

        state = "biofilm"

    elif sample.startswith(
        "PlankWT"
    ):

        state = "planktonic"

    else:

        raise RuntimeError(
            "Unknown GSE203348 sample label: "
            + sample
        )


    records.append(
        make_record(
            study,
            sample,
            state,
            state,
            row,
        )
    )


# ============================================================
# GSE87213
# ============================================================

study = "GSE87213"

table = pd.read_csv(
    FILES[
        study
    ]
)

sample_column = table.columns[
    0
]

check_genes(
    table,
    study,
)


for _, row in table.iterrows():

    sample = str(
        row[
            sample_column
        ]
    )


    if sample.startswith(
        "WTB"
    ):

        state = "biofilm"

    elif sample.startswith(
        "WTP"
    ):

        state = "planktonic"

    else:

        raise RuntimeError(
            "Unknown GSE87213 sample label: "
            + sample
        )


    records.append(
        make_record(
            study,
            sample,
            state,
            state,
            row,
        )
    )


# ============================================================
# GSE299861
# ============================================================

study = "GSE299861"

table = pd.read_csv(
    FILES[
        study
    ]
)

check_genes(
    table,
    study,
)


for _, row in table.iterrows():

    state = str(
        row[
            "state"
        ]
    )

    sample = str(
        row[
            "sample"
        ]
    )


    if state not in {
        "biofilm",
        "planktonic",
    }:

        raise RuntimeError(
            "Unexpected GSE299861 state: "
            + state
        )


    records.append(
        make_record(
            study,
            sample,
            state,
            state,
            row,
        )
    )


# ============================================================
# GSE191103
# ============================================================

study = "GSE191103"

table = pd.read_csv(
    FILES[
        study
    ]
)

check_genes(
    table,
    study,
)


for _, row in table.iterrows():

    state = str(
        row[
            "state"
        ]
    )

    sample = str(
        row[
            "sample"
        ]
    )


    if state not in {
        "biofilm",
        "planktonic",
    }:

        raise RuntimeError(
            "Unexpected GSE191103 state: "
            + state
        )


    records.append(
        make_record(
            study,
            sample,
            state,
            state,
            row,
        )
    )


# ============================================================
# GSE120602
#
# Keep ONLY true planktonic and biofilm samples in the
# binary modelling dataset.
#
# Stationary and ciprofloxacin-persister samples remain
# specificity controls.
# ============================================================

study = "GSE120602"

table = pd.read_csv(
    FILES[
        study
    ]
)

check_genes(
    table,
    study,
)


binary = table.loc[
    table[
        "condition"
    ].isin(
        [
            "planktonic",
            "biofilm",
        ]
    )
].copy()


if len(binary) != 4:

    raise RuntimeError(
        "Expected 4 binary GSE120602 samples."
    )


for _, row in binary.iterrows():

    condition = str(
        row[
            "condition"
        ]
    )

    sample = str(
        row[
            "sample"
        ]
    )


    records.append(
        make_record(
            study,
            sample,
            condition,
            condition,
            row,
        )
    )


# ============================================================
# GSE12207
#
# Binary comparison:
#   exponential planktonic
#   vs 3-day flow-cell biofilm
#
# Stationary and colony samples remain specificity controls.
# ============================================================

study = "GSE12207"

table = pd.read_csv(
    FILES[
        study
    ]
)

check_genes(
    table,
    study,
)


binary = table.loc[
    table[
        "condition"
    ].isin(
        [
            "planktonic_exponential",
            "biofilm_3d",
        ]
    )
].copy()


if len(binary) != 6:

    raise RuntimeError(
        "Expected 6 binary GSE12207 samples."
    )


for _, row in binary.iterrows():

    condition = str(
        row[
            "condition"
        ]
    )

    sample = str(
        row[
            "sample"
        ]
    )


    if condition == "biofilm_3d":

        state = "biofilm"

    elif condition == "planktonic_exponential":

        state = "planktonic"

    else:

        raise RuntimeError(
            "Unexpected GSE12207 binary condition."
        )


    records.append(
        make_record(
            study,
            sample,
            state,
            condition,
            row,
        )
    )


# ============================================================
# Assemble binary dataset
# ============================================================

model = pd.DataFrame(
    records
)


metadata_columns = [
    "study",
    "sample",
    "state",
    "label",
    "source_condition",
]


model = model[
    metadata_columns
    + common_genes
]


if model.shape != (
    40,
    4858,
):

    raise RuntimeError(
        "Expected 40 x 4,858 table "
        "(5 metadata + 4,853 genes), found "
        + str(model.shape)
    )


if model[
    common_genes
].isna().any().any():

    raise RuntimeError(
        "Missing expression values in model matrix."
    )


if model[
    [
        "study",
        "sample",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate study/sample combinations."
    )


# ============================================================
# Verify study and label balance
# ============================================================

study_counts = (
    model
    .groupby(
        [
            "study",
            "state",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


print()
print("=" * 100)
print("BINARY STUDY COMPOSITION")
print("=" * 100)

print(
    study_counts.to_string()
)


print()
print(
    "Total samples: "
    + str(
        len(model)
    )
)

print(
    "Biofilm samples: "
    + str(
        int(
            (
                model[
                    "label"
                ]
                == 1
            ).sum()
        )
    )
)

print(
    "Planktonic samples: "
    + str(
        int(
            (
                model[
                    "label"
                ]
                == 0
            ).sum()
        )
    )
)


if len(model) != 40:

    raise RuntimeError(
        "Binary dataset does not contain 40 samples."
    )


if (
    model[
        "label"
    ]
    == 1
).sum() != 20:

    raise RuntimeError(
        "Expected 20 biofilm samples."
    )


if (
    model[
        "label"
    ]
    == 0
).sum() != 20:

    raise RuntimeError(
        "Expected 20 planktonic samples."
    )


if model[
    "study"
].nunique() != 7:

    raise RuntimeError(
        "Expected 7 independent study families."
    )


# ============================================================
# Within-sample percentile ranks
# ============================================================

ranked_values = (
    model[
        common_genes
    ]
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


ranked = pd.concat(
    [
        model[
            metadata_columns
        ].reset_index(
            drop=True
        ),

        ranked_values.reset_index(
            drop=True
        ),
    ],
    axis=1,
)


if ranked.isna().any().any():

    raise RuntimeError(
        "Missing values after rank transformation."
    )


# ============================================================
# Rank sanity checks
# ============================================================

rank_summary = pd.DataFrame(
    {
        "sample":
            ranked[
                "sample"
            ],

        "study":
            ranked[
                "study"
            ],

        "rank_min":
            ranked_values.min(
                axis=1
            ).to_numpy(),

        "rank_median":
            ranked_values.median(
                axis=1
            ).to_numpy(),

        "rank_max":
            ranked_values.max(
                axis=1
            ).to_numpy(),
    }
)


print()
print("=" * 100)
print("WITHIN-SAMPLE RANK TRANSFORMATION")
print("=" * 100)

print(
    rank_summary.to_string(
        index=False
    )
)


# ============================================================
# Save binary modelling dataset
# ============================================================

model.to_csv(
    PROCESSED_DIR
    / "CFSTATE_binary_7studies_40samples_4853genes_raw.csv",
    index=False,
)


ranked.to_csv(
    PROCESSED_DIR
    / "CFSTATE_binary_7studies_40samples_4853genes_ranked.csv",
    index=False,
)


model[
    metadata_columns
].to_csv(
    PROCESSED_DIR
    / "CFSTATE_binary_7studies_40samples_metadata.csv",
    index=False,
)


rank_summary.to_csv(
    OUT_DIR
    / "Audit55_rank_transform_summary.csv",
    index=False,
)


study_counts.to_csv(
    OUT_DIR
    / "Audit55_binary_study_composition.csv"
)


# ============================================================
# Build specificity-control matrix
#
# These samples are NOT part of ordinary binary training.
# ============================================================

challenge_records = []


def add_challenge_rows(
    study_name,
    table,
    selected_conditions,
):

    check_genes(
        table,
        study_name,
    )


    subset = table.loc[
        table[
            "condition"
        ].isin(
            selected_conditions
        )
    ].copy()


    for _, row in subset.iterrows():

        record = {
            "study":
                study_name,

            "sample":
                str(
                    row[
                        "sample"
                    ]
                ),

            "condition":
                str(
                    row[
                        "condition"
                    ]
                ),
        }


        for gene in common_genes:

            record[
                gene
            ] = row[
                gene
            ]


        challenge_records.append(
            record
        )


# GSE120602 non-binary states

table_120602 = pd.read_csv(
    FILES[
        "GSE120602"
    ]
)


add_challenge_rows(
    "GSE120602",
    table_120602,
    [
        "stationary",
        "persister_ciprofloxacin",
    ],
)


# GSE12207 non-binary states

table_12207 = pd.read_csv(
    FILES[
        "GSE12207"
    ]
)


add_challenge_rows(
    "GSE12207",
    table_12207,
    [
        "planktonic_stationary",
        "colony_15h",
        "colony_40h",
    ],
)


# ============================================================
# Entire SRP009257 specificity study
# ============================================================

SRP_FILE = Path(
    "data/processed/expanded_biofilm/SRP009257/"
    "SRP009257_PA14_samples_x_4858.csv"
)


if SRP_FILE.exists():

    srp = pd.read_csv(
        SRP_FILE
    )

    check_genes(
        srp,
        "SRP009257",
    )


    for _, row in srp.iterrows():

        record = {
            "study":
                "SRP009257",

            "sample":
                str(
                    row[
                        "sample"
                    ]
                ),

            "condition":
                str(
                    row[
                        "condition"
                    ]
                ),
        }


        for gene in common_genes:

            record[
                gene
            ] = row[
                gene
            ]


        challenge_records.append(
            record
        )


challenge = pd.DataFrame(
    challenge_records
)


challenge_metadata = [
    "study",
    "sample",
    "condition",
]


challenge = challenge[
    challenge_metadata
    + common_genes
]


challenge_rank_values = (
    challenge[
        common_genes
    ]
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


challenge_ranked = pd.concat(
    [
        challenge[
            challenge_metadata
        ].reset_index(
            drop=True
        ),

        challenge_rank_values.reset_index(
            drop=True
        ),
    ],
    axis=1,
)


challenge_ranked.to_csv(
    PROCESSED_DIR
    / "CFSTATE_specificity_controls_4853genes_ranked.csv",
    index=False,
)


print()
print("=" * 100)
print("SPECIFICITY / CHALLENGE SET")
print("=" * 100)

print(
    challenge_ranked[
        challenge_metadata
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "Challenge samples: "
    + str(
        len(
            challenge_ranked
        )
    )
)


print()
print("=" * 100)
print("AUDIT 55 COMPLETE")
print("=" * 100)

print(
    "Binary modelling matrix:"
)

print(
    PROCESSED_DIR
    / "CFSTATE_binary_7studies_40samples_4853genes_ranked.csv"
)

print()

print(
    "Specificity controls:"
)

print(
    PROCESSED_DIR
    / "CFSTATE_specificity_controls_4853genes_ranked.csv"
)

print()
