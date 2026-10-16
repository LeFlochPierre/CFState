from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BIOFILM_EFFECT_FILE = Path(
    "results/audit/cross_study_modeling/"
    "Audit58_effect_geometry/"
    "Audit58_study_effect_vectors.csv"
)

BINARY_FILE = Path(
    "data/processed/cross_study_modeling/"
    "CFSTATE_binary_7studies_40samples_4853genes_ranked.csv"
)

CONTROL_FILE = Path(
    "data/processed/cross_study_modeling/"
    "CFSTATE_specificity_controls_4853genes_ranked.csv"
)

OUT_DIR = Path(
    "results/audit/cross_study_modeling/"
    "Audit65_controlled_state_atlas"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 65: "
    "controlled physiological-state contrast atlas"
)
print("=" * 100)


# ============================================================
# Load
# ============================================================

biofilm_effects = pd.read_csv(
    BIOFILM_EFFECT_FILE
)

binary = pd.read_csv(
    BINARY_FILE
)

controls = pd.read_csv(
    CONTROL_FILE
)

# Harmonise specificity-control metadata naming.
# The control manifest uses "condition" whereas the
# binary modelling manifest uses "state".

if "state" not in controls.columns:

    if "condition" in controls.columns:

        controls = controls.rename(
            columns={
                "condition":
                    "state"
            }
        )

    else:

        raise RuntimeError(
            "Control file contains neither "
            "'state' nor 'condition'."
        )
        

if "canonical_pao1" not in biofilm_effects.columns:

    raise RuntimeError(
        "Missing canonical_pao1 in Audit 58 effect file."
    )


genes = (
    biofilm_effects[
        "canonical_pao1"
    ]
    .astype(str)
    .tolist()
)


if len(genes) != 4853:

    raise RuntimeError(
        "Expected 4,853 genes."
    )


for table_name, table in [
    ("binary", binary),
    ("controls", controls),
]:

    missing = [
        gene
        for gene in genes
        if gene not in table.columns
    ]


    if missing:

        raise RuntimeError(
            table_name
            + " is missing "
            + str(len(missing))
            + " genes."
        )


# ============================================================
# Inspect metadata
# ============================================================

gene_pattern = re.compile(
    r"^PA\d{4}$"
)


binary_metadata = [
    column
    for column in binary.columns
    if not gene_pattern.fullmatch(
        str(column)
    )
]


control_metadata = [
    column
    for column in controls.columns
    if not gene_pattern.fullmatch(
        str(column)
    )
]


print()
print("=" * 100)
print("METADATA")
print("=" * 100)

print(
    "Binary metadata columns:"
)

print(
    ", ".join(
        binary_metadata
    )
)

print()

print(
    "Control metadata columns:"
)

print(
    ", ".join(
        control_metadata
    )
)


required = [
    "study",
    "sample",
    "state",
]


for column in required:

    if column not in binary.columns:

        raise RuntimeError(
            "Binary file missing column: "
            + column
        )


    if column not in controls.columns:

        raise RuntimeError(
            "Control file missing column: "
            + column
        )


print()
print("=" * 100)
print("CONTROL COMPOSITION")
print("=" * 100)

composition = (
    controls.groupby(
        [
            "study",
            "state",
        ]
    )
    .size()
    .reset_index(
        name="n_samples"
    )
)


print(
    composition.to_string(
        index=False
    )
)


# ============================================================
# Effect helper
# ============================================================

def mean_effect(
    target_rows,
    reference_rows,
):

    target = (
        target_rows[
            genes
        ]
        .mean(
            axis=0
        )
    )


    reference = (
        reference_rows[
            genes
        ]
        .mean(
            axis=0
        )
    )


    return (
        target
        - reference
    )


# ============================================================
# Build contrast vectors
# ============================================================

contrast_vectors = {}

contrast_metadata = []


# ------------------------------------------------------------
# GSE120602
#
# Reference = binary planktonic samples.
# Targets = each specificity-control state.
# ------------------------------------------------------------

study = "GSE120602"


reference = binary.loc[
    (
        binary[
            "study"
        ]
        == study
    )
    &
    (
        binary[
            "label"
        ]
        == 0
    )
]


if len(reference) == 0:

    raise RuntimeError(
        "No GSE120602 planktonic reference."
    )


study_controls = controls.loc[
    controls[
        "study"
    ]
    == study
]


for state in sorted(
    study_controls[
        "state"
    ].unique()
):

    target = study_controls.loc[
        study_controls[
            "state"
        ]
        == state
    ]


    contrast_name = (
        study
        + ":"
        + str(state)
        + "_vs_planktonic"
    )


    contrast_vectors[
        contrast_name
    ] = mean_effect(
        target,
        reference,
    )


    contrast_metadata.append(
        {
            "contrast":
                contrast_name,

            "study":
                study,

            "target_state":
                str(state),

            "reference_state":
                "planktonic",

            "n_target":
                len(target),

            "n_reference":
                len(reference),
        }
    )


# ------------------------------------------------------------
# GSE12207
#
# Reference = exponential planktonic binary samples.
# Targets = stationary / colony states.
# ------------------------------------------------------------

study = "GSE12207"


reference = binary.loc[
    (
        binary[
            "study"
        ]
        == study
    )
    &
    (
        binary[
            "label"
        ]
        == 0
    )
]


if len(reference) == 0:

    raise RuntimeError(
        "No GSE12207 planktonic reference."
    )


study_controls = controls.loc[
    controls[
        "study"
    ]
    == study
]


for state in sorted(
    study_controls[
        "state"
    ].unique()
):

    target = study_controls.loc[
        study_controls[
            "state"
        ]
        == state
    ]


    contrast_name = (
        study
        + ":"
        + str(state)
        + "_vs_exp_planktonic"
    )


    contrast_vectors[
        contrast_name
    ] = mean_effect(
        target,
        reference,
    )


    contrast_metadata.append(
        {
            "contrast":
                contrast_name,

            "study":
                study,

            "target_state":
                str(state),

            "reference_state":
                "exp_planktonic",

            "n_target":
                len(target),

            "n_reference":
                len(reference),
        }
    )


# ------------------------------------------------------------
# SRP009257
#
# Reference = P4.
# Targets = every other state.
# ------------------------------------------------------------

study = "SRP009257"


study_controls = controls.loc[
    controls[
        "study"
    ]
    == study
].copy()


if len(study_controls) == 0:

    raise RuntimeError(
        "No SRP009257 controls found."
    )


states = (
    study_controls[
        "state"
    ]
    .astype(str)
    .unique()
    .tolist()
)


p4_matches = [
    state
    for state in states
    if "p4" in state.lower()
]


if len(p4_matches) != 1:

    print(
        "SRP009257 states:"
    )

    print(
        states
    )

    raise RuntimeError(
        "Could not uniquely identify SRP009257 P4 reference."
    )


p4_state = p4_matches[
    0
]


reference = study_controls.loc[
    study_controls[
        "state"
    ].astype(str)
    == p4_state
]


for state in sorted(
    states
):

    if state == p4_state:

        continue


    target = study_controls.loc[
        study_controls[
            "state"
        ].astype(str)
        == state
    ]


    contrast_name = (
        study
        + ":"
        + state
        + "_vs_"
        + p4_state
    )


    contrast_vectors[
        contrast_name
    ] = mean_effect(
        target,
        reference,
    )


    contrast_metadata.append(
        {
            "contrast":
                contrast_name,

            "study":
                study,

            "target_state":
                state,

            "reference_state":
                p4_state,

            "n_target":
                len(target),

            "n_reference":
                len(reference),
        }
    )


contrast_table = pd.DataFrame(
    contrast_vectors
)


contrast_table.index = genes
contrast_table.index.name = "canonical_pao1"


contrast_table.reset_index().to_csv(
    OUT_DIR
    / "Audit65_control_effect_vectors.csv",
    index=False,
)


contrast_meta = pd.DataFrame(
    contrast_metadata
)


contrast_meta.to_csv(
    OUT_DIR
    / "Audit65_control_contrast_metadata.csv",
    index=False,
)


print()
print("=" * 100)
print("CONTROL CONTRASTS")
print("=" * 100)

print(
    contrast_meta.to_string(
        index=False
    )
)


# ============================================================
# Similarity of every biofilm contrast to every control vector
# ============================================================

biofilm_studies = [
    column
    for column in biofilm_effects.columns
    if column != "canonical_pao1"
]


records = []


for biofilm_study in biofilm_studies:

    bio_vector = (
        biofilm_effects[
            biofilm_study
        ]
        .to_numpy(
            dtype=float
        )
    )


    for control_name in contrast_table.columns:

        control_vector = (
            contrast_table[
                control_name
            ]
            .to_numpy(
                dtype=float
            )
        )


        rho, _ = spearmanr(
            bio_vector,
            control_vector,
        )


        pearson = float(
            np.corrcoef(
                bio_vector,
                control_vector,
            )[0, 1]
        )


        denominator = (
            np.linalg.norm(
                bio_vector
            )
            *
            np.linalg.norm(
                control_vector
            )
        )


        if denominator == 0:

            cosine = np.nan

        else:

            cosine = float(
                np.dot(
                    bio_vector,
                    control_vector,
                )
                / denominator
            )


        records.append(
            {
                "biofilm_study":
                    biofilm_study,

                "control_contrast":
                    control_name,

                "spearman_rho":
                    float(
                        rho
                    ),

                "pearson_r":
                    pearson,

                "cosine_similarity":
                    cosine,
            }
        )


similarity = pd.DataFrame(
    records
)


similarity.to_csv(
    OUT_DIR
    / "Audit65_biofilm_vs_control_similarity.csv",
    index=False,
)


# ============================================================
# Full Spearman matrix
# ============================================================

combined = pd.concat(
    [
        biofilm_effects
        .set_index(
            "canonical_pao1"
        )[
            biofilm_studies
        ],

        contrast_table,
    ],
    axis=1,
)


spearman_matrix = combined.corr(
    method="spearman"
)


spearman_matrix.to_csv(
    OUT_DIR
    / "Audit65_full_contrast_spearman_matrix.csv"
)


# ============================================================
# Top matches for each biofilm study
# ============================================================

top_records = []


for study in biofilm_studies:

    subset = similarity.loc[
        similarity[
            "biofilm_study"
        ]
        == study
    ].copy()


    subset[
        "absolute_spearman"
    ] = subset[
        "spearman_rho"
    ].abs()


    subset = subset.sort_values(
        "absolute_spearman",
        ascending=False,
    )


    top = subset.head(
        5
    ).copy()


    top[
        "rank"
    ] = range(
        1,
        len(top) + 1,
    )


    top_records.append(
        top
    )


top_matches = pd.concat(
    top_records,
    ignore_index=True,
)


top_matches.to_csv(
    OUT_DIR
    / "Audit65_top_control_matches_per_biofilm_study.csv",
    index=False,
)


print()
print("=" * 100)
print("TOP CONTROL-STATE MATCHES FOR EACH BIOFILM STUDY")
print("=" * 100)

for study in biofilm_studies:

    subset = top_matches.loc[
        top_matches[
            "biofilm_study"
        ]
        == study
    ]


    print()
    print(
        study
    )

    print(
        subset[
            [
                "rank",
                "control_contrast",
                "spearman_rho",
                "cosine_similarity",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# Control-control similarities
# ============================================================

control_spearman = contrast_table.corr(
    method="spearman"
)


control_spearman.to_csv(
    OUT_DIR
    / "Audit65_control_vs_control_spearman.csv"
)


print()
print("=" * 100)
print("CONTROL-STATE SPEARMAN MATRIX")
print("=" * 100)

print(
    control_spearman
    .round(
        3
    )
    .to_string()
)


# ============================================================
# Simple strongest positive and negative relationships
# ============================================================

sorted_positive = similarity.sort_values(
    "spearman_rho",
    ascending=False,
)


sorted_negative = similarity.sort_values(
    "spearman_rho",
    ascending=True,
)


print()
print("=" * 100)
print("STRONGEST POSITIVE BIOFILM ↔ CONTROL RELATIONSHIPS")
print("=" * 100)

print(
    sorted_positive[
        [
            "biofilm_study",
            "control_contrast",
            "spearman_rho",
            "cosine_similarity",
        ]
    ]
    .head(
        20
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("STRONGEST NEGATIVE BIOFILM ↔ CONTROL RELATIONSHIPS")
print("=" * 100)

print(
    sorted_negative[
        [
            "biofilm_study",
            "control_contrast",
            "spearman_rho",
            "cosine_similarity",
        ]
    ]
    .head(
        20
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 65 COMPLETE")
print("=" * 100)

print(
    "Positive similarity means the biofilm contrast "
    "resembles that controlled physiological transition."
)

print(
    "Negative similarity means it points in the opposite "
    "transcriptional direction."
)

print()

print(
    "This is descriptive geometry, not a trained classifier."
)

print()
