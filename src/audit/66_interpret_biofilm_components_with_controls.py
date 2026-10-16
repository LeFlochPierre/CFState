from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


AUDIT64_DIR = Path(
    "results/audit/cross_study_modeling/"
    "Audit64_effect_dimensionality"
)

CONTROL_FILE = Path(
    "results/audit/cross_study_modeling/"
    "Audit65_controlled_state_atlas/"
    "Audit65_control_effect_vectors.csv"
)

OUT_DIR = Path(
    "results/audit/cross_study_modeling/"
    "Audit66_component_interpretation"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


ANALYSES = [
    "SIX_WITHOUT_GSE299861",
    "ALL7",
]


N_COMPONENTS = 4


print()
print("=" * 100)
print(
    "CF-STATE — Audit 66: "
    "interpret biofilm-response components using controlled states"
)
print("=" * 100)


# ============================================================
# Load control-state effect vectors
# ============================================================

controls = pd.read_csv(
    CONTROL_FILE
)


if "canonical_pao1" not in controls.columns:

    raise RuntimeError(
        "Control effect file is missing canonical_pao1."
    )


genes = (
    controls[
        "canonical_pao1"
    ]
    .astype(str)
    .tolist()
)


if len(genes) != 4853:

    raise RuntimeError(
        "Expected 4,853 genes, found "
        + str(len(genes))
    )


control_names = [
    column
    for column in controls.columns
    if column != "canonical_pao1"
]


if len(control_names) == 0:

    raise RuntimeError(
        "No control contrasts found."
    )


controls = controls.set_index(
    "canonical_pao1"
)


print()
print(
    "Control contrasts: "
    + str(len(control_names))
)


for control_name in control_names:

    print(
        "  "
        + control_name
    )


# ============================================================
# Analyse each Audit 64 decomposition
# ============================================================

all_records = []


for analysis_name in ANALYSES:

    print()
    print("=" * 100)
    print(
        analysis_name
    )
    print("=" * 100)


    analysis_records = []


    for component_number in range(
        1,
        N_COMPONENTS + 1,
    ):

        loading_file = (
            AUDIT64_DIR
            / (
                analysis_name
                + "_component_"
                + str(component_number)
                + "_gene_loadings.csv"
            )
        )


        if not loading_file.exists():

            raise RuntimeError(
                "Missing component loading file: "
                + str(loading_file)
            )


        loading_table = pd.read_csv(
            loading_file
        )


        required_columns = [
            "canonical_pao1",
            "loading",
        ]


        for column in required_columns:

            if column not in loading_table.columns:

                raise RuntimeError(
                    "Missing column in "
                    + str(loading_file)
                    + ": "
                    + column
                )


        loading_table[
            "canonical_pao1"
        ] = (
            loading_table[
                "canonical_pao1"
            ]
            .astype(str)
        )


        loading = (
            loading_table
            .set_index(
                "canonical_pao1"
            )[
                "loading"
            ]
            .reindex(
                genes
            )
        )


        if loading.isna().any():

            raise RuntimeError(
                "Missing genes in component "
                + str(component_number)
                + " of "
                + analysis_name
            )


        loading_vector = loading.to_numpy(
            dtype=float
        )


        loading_norm = float(
            np.linalg.norm(
                loading_vector
            )
        )


        if loading_norm == 0:

            raise RuntimeError(
                "Zero component loading norm."
            )


        # SVD loadings should already have unit norm,
        # but explicitly normalise for safety.

        loading_unit = (
            loading_vector
            / loading_norm
        )


        for control_name in control_names:

            control_vector = (
                controls[
                    control_name
                ]
                .reindex(
                    genes
                )
                .to_numpy(
                    dtype=float
                )
            )


            control_norm = float(
                np.linalg.norm(
                    control_vector
                )
            )


            if control_norm == 0:

                raise RuntimeError(
                    "Zero control vector: "
                    + control_name
                )


            control_unit = (
                control_vector
                / control_norm
            )


            # Component projection:
            # cosine between the control transition and
            # the SVD gene-loading direction.

            projection = float(
                np.dot(
                    control_unit,
                    loading_unit,
                )
            )


            rho, _ = spearmanr(
                control_vector,
                loading_vector,
            )


            record = {
                "analysis":
                    analysis_name,

                "component":
                    component_number,

                "control_contrast":
                    control_name,

                "projection_cosine":
                    projection,

                "spearman_rho":
                    float(
                        rho
                    ),

                "absolute_projection":
                    abs(
                        projection
                    ),

                "absolute_spearman":
                    abs(
                        float(
                            rho
                        )
                    ),
            }


            analysis_records.append(
                record
            )


    analysis_df = pd.DataFrame(
        analysis_records
    )


    analysis_df.to_csv(
        OUT_DIR
        / (
            "Audit66_"
            + analysis_name
            + "_control_component_similarity.csv"
        ),
        index=False,
    )


    all_records.append(
        analysis_df
    )


    # ========================================================
    # Print component interpretations
    # ========================================================

    for component_number in range(
        1,
        N_COMPONENTS + 1,
    ):

        subset = analysis_df.loc[
            analysis_df[
                "component"
            ]
            == component_number
        ].copy()


        subset = subset.sort_values(
            "absolute_projection",
            ascending=False,
        )


        print()
        print("-" * 100)
        print(
            "COMPONENT "
            + str(component_number)
        )
        print("-" * 100)


        print(
            subset[
                [
                    "control_contrast",
                    "projection_cosine",
                    "spearman_rho",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# Combined output
# ============================================================

combined = pd.concat(
    all_records,
    ignore_index=True,
)


combined.to_csv(
    OUT_DIR
    / "Audit66_all_component_control_similarity.csv",
    index=False,
)


# ============================================================
# Best matching control for each component
# ============================================================

best_records = []


for analysis_name in ANALYSES:

    for component_number in range(
        1,
        N_COMPONENTS + 1,
    ):

        subset = combined.loc[
            (
                combined[
                    "analysis"
                ]
                == analysis_name
            )
            &
            (
                combined[
                    "component"
                ]
                == component_number
            )
        ].copy()


        subset = subset.sort_values(
            "absolute_projection",
            ascending=False,
        )


        best = subset.iloc[
            0
        ]


        second = subset.iloc[
            1
        ]


        best_records.append(
            {
                "analysis":
                    analysis_name,

                "component":
                    component_number,

                "best_control":
                    best[
                        "control_contrast"
                    ],

                "best_projection":
                    float(
                        best[
                            "projection_cosine"
                        ]
                    ),

                "best_spearman":
                    float(
                        best[
                            "spearman_rho"
                        ]
                    ),

                "second_control":
                    second[
                        "control_contrast"
                    ],

                "second_projection":
                    float(
                        second[
                            "projection_cosine"
                        ]
                    ),
            }
        )


best_table = pd.DataFrame(
    best_records
)


best_table.to_csv(
    OUT_DIR
    / "Audit66_best_control_per_component.csv",
    index=False,
)


print()
print("=" * 100)
print("BEST CONTROL MATCH PER COMPONENT")
print("=" * 100)

print(
    best_table.to_string(
        index=False
    )
)


# ============================================================
# Component coverage by available controls
#
# This does NOT estimate variance explained.
# It simply reports how strongly each component aligns with
# at least one controlled transition currently in the atlas.
# ============================================================

coverage_records = []


for analysis_name in ANALYSES:

    for component_number in range(
        1,
        N_COMPONENTS + 1,
    ):

        subset = combined.loc[
            (
                combined[
                    "analysis"
                ]
                == analysis_name
            )
            &
            (
                combined[
                    "component"
                ]
                == component_number
            )
        ]


        max_abs_projection = float(
            subset[
                "absolute_projection"
            ].max()
        )


        max_abs_spearman = float(
            subset[
                "absolute_spearman"
            ].max()
        )


        coverage_records.append(
            {
                "analysis":
                    analysis_name,

                "component":
                    component_number,

                "max_absolute_control_projection":
                    max_abs_projection,

                "max_absolute_control_spearman":
                    max_abs_spearman,
            }
        )


coverage = pd.DataFrame(
    coverage_records
)


coverage.to_csv(
    OUT_DIR
    / "Audit66_component_control_coverage.csv",
    index=False,
)


print()
print("=" * 100)
print("CONTROL COVERAGE OF COMPONENTS")
print("=" * 100)

print(
    coverage.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 66 COMPLETE")
print("=" * 100)

print(
    "Positive projection means the component points "
    "in the same direction as that controlled transition."
)

print(
    "Negative projection means the component points "
    "in the opposite direction."
)

print()

print(
    "Component signs were inherited from Audit 64."
)

print(
    "This audit is descriptive interpretation only; "
    "no classifier or predictive model was fitted."
)

print()
