from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "results/audit/cross_study_modeling/"
    "Audit58_effect_geometry/"
    "Audit58_study_effect_vectors.csv"
)

OUT_DIR = Path(
    "results/audit/cross_study_modeling/"
    "Audit64_effect_dimensionality"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


N_PERMUTATIONS = 2000
RANDOM_SEED = 20261005


print()
print("=" * 100)
print(
    "CF-STATE — Audit 64: "
    "dimensionality of cross-study biofilm response"
)
print("=" * 100)


# ============================================================
# Load study effect vectors
# ============================================================

effects = pd.read_csv(
    INPUT_FILE
)


if "canonical_pao1" not in effects.columns:

    raise RuntimeError(
        "Missing canonical_pao1 column."
    )


study_columns = [
    column
    for column in effects.columns
    if column != "canonical_pao1"
]


if len(study_columns) != 7:

    raise RuntimeError(
        "Expected seven study effect vectors, found "
        + str(len(study_columns))
    )


if "GSE299861" not in study_columns:

    raise RuntimeError(
        "GSE299861 is missing."
    )


print()
print("Studies:")

for study in study_columns:

    print(
        "  "
        + study
    )


print()

print(
    "Genes: "
    + str(len(effects))
)


# ============================================================
# Helpers
# ============================================================

def row_normalise(
    matrix,
):

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )


    if np.any(
        norms == 0
    ):

        raise RuntimeError(
            "Zero-length study effect vector."
        )


    return (
        matrix
        / norms
    )


def svd_statistics(
    matrix,
):

    U, singular_values, Vt = np.linalg.svd(
        matrix,
        full_matrices=False,
    )


    energy = (
        singular_values ** 2
    )


    explained = (
        energy
        / energy.sum()
    )


    cumulative = np.cumsum(
        explained
    )


    effective_dimension = float(
        (
            energy.sum() ** 2
        )
        /
        (
            (
                energy ** 2
            ).sum()
        )
    )


    return (
        U,
        singular_values,
        Vt,
        explained,
        cumulative,
        effective_dimension,
    )


def orient_component(
    study_scores,
    gene_loadings,
):

    # Component sign is mathematically arbitrary.
    # Orient it so that the majority of study scores
    # are positive.

    if np.sum(
        study_scores
    ) < 0:

        study_scores = (
            -study_scores
        )

        gene_loadings = (
            -gene_loadings
        )


    return (
        study_scores,
        gene_loadings,
    )


def rank_k_reconstruction_cosines(
    matrix,
    U,
    singular_values,
    Vt,
):

    records = []


    max_rank = min(
        4,
        len(
            singular_values
        ),
    )


    for rank in range(
        1,
        max_rank + 1,
    ):

        reconstruction = (
            U[
                :,
                :rank
            ]
            @ np.diag(
                singular_values[
                    :rank
                ]
            )
            @ Vt[
                :rank,
                :
            ]
        )


        for row_index in range(
            matrix.shape[
                0
            ]
        ):

            original = matrix[
                row_index
            ]

            reconstructed = reconstruction[
                row_index
            ]


            denominator = (
                np.linalg.norm(
                    original
                )
                *
                np.linalg.norm(
                    reconstructed
                )
            )


            if denominator == 0:

                cosine = np.nan

            else:

                cosine = float(
                    np.dot(
                        original,
                        reconstructed,
                    )
                    / denominator
                )


            records.append(
                {
                    "rank":
                        rank,

                    "study_index":
                        row_index,

                    "reconstruction_cosine":
                        cosine,
                }
            )


    return pd.DataFrame(
        records
    )


def permutation_null(
    matrix,
    n_permutations,
    seed,
):

    rng = np.random.default_rng(
        seed
    )


    n_studies = matrix.shape[
        0
    ]


    null_pc1 = np.empty(
        n_permutations,
        dtype=float,
    )


    null_pc12 = np.empty(
        n_permutations,
        dtype=float,
    )


    for permutation_index in range(
        n_permutations
    ):

        permuted = np.empty_like(
            matrix
        )


        for study_index in range(
            n_studies
        ):

            permuted[
                study_index
            ] = rng.permutation(
                matrix[
                    study_index
                ]
            )


        # Singular values squared are eigenvalues
        # of the small study x study Gram matrix.
        gram = (
            permuted
            @ permuted.T
        )


        eigenvalues = np.linalg.eigvalsh(
            gram
        )


        eigenvalues = np.sort(
            eigenvalues
        )[
            ::-1
        ]


        eigenvalues = np.clip(
            eigenvalues,
            0.0,
            None,
        )


        total = float(
            eigenvalues.sum()
        )


        null_pc1[
            permutation_index
        ] = (
            eigenvalues[
                0
            ]
            / total
        )


        if len(
            eigenvalues
        ) >= 2:

            null_pc12[
                permutation_index
            ] = (
                eigenvalues[
                    :2
                ].sum()
                / total
            )

        else:

            null_pc12[
                permutation_index
            ] = null_pc1[
                permutation_index
            ]


    return (
        null_pc1,
        null_pc12,
    )


# ============================================================
# Run one analysis
# ============================================================

def run_analysis(
    analysis_name,
    selected_studies,
    seed_offset,
):

    print()
    print("=" * 100)
    print(
        analysis_name
    )
    print("=" * 100)


    matrix = (
        effects[
            selected_studies
        ]
        .T
        .to_numpy(
            dtype=float
        )
    )


    normalised = row_normalise(
        matrix
    )


    (
        U,
        singular_values,
        Vt,
        explained,
        cumulative,
        effective_dimension,
    ) = svd_statistics(
        normalised
    )


    # --------------------------------------------------------
    # Component summary
    # --------------------------------------------------------

    component_records = []


    n_components = len(
        singular_values
    )


    for component_index in range(
        n_components
    ):

        component_records.append(
            {
                "analysis":
                    analysis_name,

                "component":
                    component_index + 1,

                "singular_value":
                    float(
                        singular_values[
                            component_index
                        ]
                    ),

                "explained_energy":
                    float(
                        explained[
                            component_index
                        ]
                    ),

                "cumulative_energy":
                    float(
                        cumulative[
                            component_index
                        ]
                    ),
            }
        )


    component_table = pd.DataFrame(
        component_records
    )


    component_table.to_csv(
        OUT_DIR
        / (
            analysis_name
            + "_components.csv"
        ),
        index=False,
    )


    print()
    print(
        "COMPONENT ENERGY"
    )

    print(
        component_table.to_string(
            index=False
        )
    )


    print()

    print(
        "Effective dimensionality: "
        + format(
            effective_dimension,
            ".4f",
        )
    )


    # --------------------------------------------------------
    # Study scores on first components
    # --------------------------------------------------------

    score_records = []


    max_components_to_report = min(
        4,
        n_components,
    )


    oriented_scores = {}


    for component_index in range(
        max_components_to_report
    ):

        scores = (
            U[
                :,
                component_index
            ]
            * singular_values[
                component_index
            ]
        )


        loadings = Vt[
            component_index
        ].copy()


        (
            scores,
            loadings,
        ) = orient_component(
            scores,
            loadings,
        )


        oriented_scores[
            component_index
        ] = scores


        loading_table = pd.DataFrame(
            {
                "canonical_pao1":
                    effects[
                        "canonical_pao1"
                    ],

                "loading":
                    loadings,
            }
        )


        loading_table[
            "absolute_loading"
        ] = loading_table[
            "loading"
        ].abs()


        loading_table = loading_table.sort_values(
            "absolute_loading",
            ascending=False,
        )


        loading_table.to_csv(
            OUT_DIR
            / (
                analysis_name
                + "_component_"
                + str(
                    component_index + 1
                )
                + "_gene_loadings.csv"
            ),
            index=False,
        )


    for study_index, study in enumerate(
        selected_studies
    ):

        record = {
            "analysis":
                analysis_name,

            "study":
                study,
        }


        for component_index in range(
            max_components_to_report
        ):

            record[
                "component_"
                + str(
                    component_index + 1
                )
                + "_score"
            ] = float(
                oriented_scores[
                    component_index
                ][
                    study_index
                ]
            )


        score_records.append(
            record
        )


    score_table = pd.DataFrame(
        score_records
    )


    score_table.to_csv(
        OUT_DIR
        / (
            analysis_name
            + "_study_scores.csv"
        ),
        index=False,
    )


    print()
    print(
        "STUDY SCORES"
    )

    print(
        score_table.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Rank-k reconstruction
    # --------------------------------------------------------

    reconstruction = rank_k_reconstruction_cosines(
        normalised,
        U,
        singular_values,
        Vt,
    )


    reconstruction[
        "study"
    ] = reconstruction[
        "study_index"
    ].map(
        dict(
            enumerate(
                selected_studies
            )
        )
    )


    reconstruction[
        "analysis"
    ] = analysis_name


    reconstruction = reconstruction[
        [
            "analysis",
            "rank",
            "study",
            "reconstruction_cosine",
        ]
    ]


    reconstruction.to_csv(
        OUT_DIR
        / (
            analysis_name
            + "_rank_reconstruction.csv"
        ),
        index=False,
    )


    reconstruction_summary = (
        reconstruction.groupby(
            "rank",
            as_index=False,
        )[
            "reconstruction_cosine"
        ]
        .agg(
            [
                "mean",
                "median",
                "min",
                "max",
            ]
        )
        .reset_index()
    )


    print()
    print(
        "RANK-K RECONSTRUCTION"
    )

    print(
        reconstruction_summary.to_string(
            index=False
        )
    )


    # --------------------------------------------------------
    # Permutation null
    #
    # Independently scramble gene identities within each
    # study. This preserves each study's effect distribution
    # but destroys cross-study gene correspondence.
    # --------------------------------------------------------

    (
        null_pc1,
        null_pc12,
    ) = permutation_null(
        normalised,
        N_PERMUTATIONS,
        RANDOM_SEED
        + seed_offset,
    )


    observed_pc1 = float(
        explained[
            0
        ]
    )


    observed_pc12 = float(
        cumulative[
            min(
                1,
                len(
                    cumulative
                )
                - 1,
            )
        ]
    )


    pc1_p = float(
        (
            1
            + np.sum(
                null_pc1
                >= observed_pc1
            )
        )
        /
        (
            N_PERMUTATIONS
            + 1
        )
    )


    pc12_p = float(
        (
            1
            + np.sum(
                null_pc12
                >= observed_pc12
            )
        )
        /
        (
            N_PERMUTATIONS
            + 1
        )
    )


    null_summary = pd.DataFrame(
        [
            {
                "analysis":
                    analysis_name,

                "n_studies":
                    len(
                        selected_studies
                    ),

                "n_genes":
                    matrix.shape[
                        1
                    ],

                "observed_PC1_energy":
                    observed_pc1,

                "null_PC1_mean":
                    float(
                        null_pc1.mean()
                    ),

                "null_PC1_95th_percentile":
                    float(
                        np.quantile(
                            null_pc1,
                            0.95,
                        )
                    ),

                "PC1_permutation_p":
                    pc1_p,

                "observed_PC1_PC2_cumulative":
                    observed_pc12,

                "null_PC1_PC2_mean":
                    float(
                        null_pc12.mean()
                    ),

                "null_PC1_PC2_95th_percentile":
                    float(
                        np.quantile(
                            null_pc12,
                            0.95,
                        )
                    ),

                "PC1_PC2_permutation_p":
                    pc12_p,

                "effective_dimensionality":
                    effective_dimension,
            }
        ]
    )


    null_summary.to_csv(
        OUT_DIR
        / (
            analysis_name
            + "_permutation_summary.csv"
        ),
        index=False,
    )


    print()
    print(
        "PERMUTATION DIMENSIONALITY TEST"
    )

    print(
        null_summary.to_string(
            index=False
        )
    )


    return {
        "components":
            component_table,

        "scores":
            score_table,

        "reconstruction":
            reconstruction,

        "null":
            null_summary,
    }


# ============================================================
# Analysis A: all seven studies
# ============================================================

all7 = run_analysis(
    analysis_name="ALL7",
    selected_studies=study_columns,
    seed_offset=0,
)


# ============================================================
# Analysis B: exclude GSE299861
# ============================================================

without_299861 = [
    study
    for study in study_columns
    if study != "GSE299861"
]


six = run_analysis(
    analysis_name="SIX_WITHOUT_GSE299861",
    selected_studies=without_299861,
    seed_offset=10000,
)


# ============================================================
# Direct comparison
# ============================================================

comparison = pd.concat(
    [
        all7[
            "null"
        ],
        six[
            "null"
        ],
    ],
    ignore_index=True,
)


comparison.to_csv(
    OUT_DIR
    / "Audit64_dimensionality_comparison.csv",
    index=False,
)


print()
print("=" * 100)
print("ALL7 VS SIX-WITHOUT-GSE299861")
print("=" * 100)

print(
    comparison.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 64 COMPLETE")
print("=" * 100)

print(
    "Effect vectors were L2-normalised study-by-study."
)

print(
    "Therefore this audit examines transcriptional "
    "DIRECTION rather than effect magnitude."
)

print()

print(
    "The permutation null independently shuffles gene "
    "identities within each study while preserving each "
    "study's effect-value distribution."
)

print()

print(
    "If one universal biofilm direction exists, PC1 should "
    "capture a large fraction of energy and rank-1 "
    "reconstruction should represent most studies well."
)

print(
    "If several dimensions are required even after removing "
    "GSE299861, a single universal biofilm axis is not "
    "supported by these datasets."
)

print()
