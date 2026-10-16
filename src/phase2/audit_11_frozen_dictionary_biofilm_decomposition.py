from pathlib import Path
from itertools import combinations
from math import factorial

import numpy as np
import pandas as pd


ROOT = Path.cwd()

DICTIONARY = (
    ROOT
    / "results/phase2/audit_10_multistress_imodulon_dictionary"
    / "stress_contrast_vectors.csv"
)

GSE203348 = (
    ROOT
    / "results/phase2/audit_03_external_pao1_activity_inference"
    / "GSE203348_activity_summary.csv"
)

GSE120602 = (
    ROOT
    / "results/phase2/audit_04_controlled_state_anchoring_GSE120602"
    / "controlled_contrast_vectors.csv"
)

CATALOGUE = (
    ROOT
    / "external/aeruprecise/precise411/iM_table.csv"
)

OUT = (
    ROOT
    / "results/phase2/audit_11_frozen_dictionary_biofilm_decomposition"
)

N_PERM = 5000
SEED = 20261006


def cosine(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom == 0:
        return np.nan

    return float(
        np.dot(x, y)
        / denom
    )


def orthonormal_basis(matrix):
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    if matrix.ndim == 1:
        matrix = matrix[:, None]

    if matrix.shape[1] == 0:
        return np.empty(
            (
                matrix.shape[0],
                0,
            )
        )

    U, s, _ = np.linalg.svd(
        matrix,
        full_matrices=False,
    )

    if len(s) == 0:
        return np.empty(
            (
                matrix.shape[0],
                0,
            )
        )

    tolerance = (
        max(
            matrix.shape
        )
        * np.finfo(float).eps
        * s[0]
    )

    rank = int(
        np.sum(
            s > tolerance
        )
    )

    return U[
        :,
        :rank,
    ]


def explained_fraction(
    target,
    matrix,
):
    target = np.asarray(
        target,
        dtype=float,
    )

    Q = orthonormal_basis(
        matrix
    )

    denominator = float(
        np.dot(
            target,
            target,
        )
    )

    if denominator == 0:
        raise RuntimeError(
            "Zero-length target vector"
        )

    if Q.shape[1] == 0:
        return 0.0

    coefficients = (
        Q.T
        @ target
    )

    return float(
        np.dot(
            coefficients,
            coefficients,
        )
        / denominator
    )


def reconstruct(
    target,
    matrix,
):
    target = np.asarray(
        target,
        dtype=float,
    )

    Q = orthonormal_basis(
        matrix
    )

    if Q.shape[1] == 0:
        return np.zeros_like(
            target
        )

    return (
        Q
        @ (
            Q.T
            @ target
        )
    )


def subset_columns(
    matrix,
    mask,
    n_axes,
):
    indices = [
        i
        for i in range(
            n_axes
        )
        if mask
        & (
            1 << i
        )
    ]

    if not indices:
        return np.empty(
            (
                matrix.shape[0],
                0,
            )
        )

    return matrix[
        :,
        indices,
    ]


def exact_shapley(
    target,
    dictionary_matrix,
    axis_names,
):
    n_axes = len(
        axis_names
    )

    if n_axes > 15:
        raise RuntimeError(
            "Exact Shapley disabled for >15 axes."
        )

    n_subsets = (
        1 << n_axes
    )

    values = {}

    for mask in range(
        n_subsets
    ):
        subset = subset_columns(
            dictionary_matrix,
            mask,
            n_axes,
        )

        values[
            mask
        ] = explained_fraction(
            target,
            subset,
        )

    n_factorial = factorial(
        n_axes
    )

    shapley = []

    full_mask = (
        n_subsets
        - 1
    )

    full_value = values[
        full_mask
    ]

    for i, axis in enumerate(
        axis_names
    ):
        phi = 0.0

        bit = (
            1 << i
        )

        for mask in range(
            n_subsets
        ):
            if mask & bit:
                continue

            k = int(
                mask.bit_count()
            )

            weight = (
                factorial(
                    k
                )
                * factorial(
                    n_axes
                    - k
                    - 1
                )
                / n_factorial
            )

            marginal = (
                values[
                    mask
                    | bit
                ]
                - values[
                    mask
                ]
            )

            phi += (
                weight
                * marginal
            )

        single_value = values[
            bit
        ]

        without_axis = values[
            full_mask
            ^ bit
        ]

        unique_increment = (
            full_value
            - without_axis
        )

        signed_cosine = cosine(
            target,
            dictionary_matrix[
                :,
                i,
            ],
        )

        shapley.append(
            {
                "axis":
                    axis,

                "shapley_explained_energy":
                    phi,

                "fraction_of_dictionary_explanation":
                    (
                        phi
                        / full_value
                        if full_value > 0
                        else np.nan
                    ),

                "single_axis_explained_energy":
                    single_value,

                "unique_increment_in_full_dictionary":
                    unique_increment,

                "signed_cosine":
                    signed_cosine,
            }
        )

    return (
        pd.DataFrame(
            shapley
        ),
        full_value,
    )


def permutation_test(
    target,
    dictionary_matrix,
    observed,
    n_perm,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    n_programmes = (
        dictionary_matrix.shape[
            0
        ]
    )

    null = np.empty(
        n_perm,
        dtype=float,
    )

    # Jointly shuffle iModulon identities across every
    # dictionary axis. This preserves the internal geometry
    # and correlations among the 11 controlled perturbations
    # while destroying correspondence with the biofilm target.
    for i in range(
        n_perm
    ):
        order = rng.permutation(
            n_programmes
        )

        shuffled = dictionary_matrix[
            order,
            :
        ]

        null[
            i
        ] = explained_fraction(
            target,
            shuffled,
        )

    p = (
        1
        + int(
            np.sum(
                null
                >= observed
            )
        )
    ) / (
        n_perm
        + 1
    )

    return (
        null,
        float(
            p
        ),
    )


def effective_dimension(
    matrix,
):
    unit_columns = []

    for j in range(
        matrix.shape[1]
    ):
        x = matrix[
            :,
            j,
        ]

        n = np.linalg.norm(
            x
        )

        if n == 0:
            raise RuntimeError(
                "Zero-length dictionary axis"
            )

        unit_columns.append(
            x / n
        )

    X = np.column_stack(
        unit_columns
    )

    _, s, _ = np.linalg.svd(
        X,
        full_matrices=False,
    )

    energy = (
        s
        ** 2
    )

    energy = (
        energy
        / energy.sum()
    )

    return float(
        1.0
        / np.sum(
            energy
            ** 2
        )
    )


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 11")
    print("Frozen controlled-dictionary decomposition of biofilm states")
    print("=" * 78)

    for path in [
        DICTIONARY,
        GSE203348,
        GSE120602,
        CATALOGUE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # 1. Frozen 11-stress dictionary
    # ------------------------------------------------------------

    dictionary = pd.read_csv(
        DICTIONARY,
        index_col=0,
    )

    if dictionary.shape != (
        116,
        11,
    ):
        raise RuntimeError(
            "Expected dictionary shape (116, 11), found "
            + str(
                dictionary.shape
            )
        )

    if not dictionary.index.is_unique:
        raise RuntimeError(
            "Dictionary iModulon IDs are not unique."
        )

    dictionary = dictionary.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if dictionary.isna().any().any():
        raise RuntimeError(
            "Dictionary contains non-numeric values."
        )

    axis_names = list(
        dictionary.columns
    )

    dictionary_matrix = (
        dictionary.to_numpy(
            dtype=float
        )
    )

    dictionary_rank = int(
        np.linalg.matrix_rank(
            dictionary_matrix
        )
    )

    dictionary_effective_dim = (
        effective_dimension(
            dictionary_matrix
        )
    )

    print()
    print("Frozen dictionary")
    print(
        "  iModulons:",
        dictionary.shape[0],
    )
    print(
        "  controlled axes:",
        dictionary.shape[1],
    )
    print(
        "  algebraic rank:",
        dictionary_rank,
    )
    print(
        "  effective dimension:",
        round(
            dictionary_effective_dim,
            3,
        ),
    )

    print()
    print("Axes")
    for axis in axis_names:
        print(
            " ",
            axis,
        )

    # ------------------------------------------------------------
    # 2. Independent biofilm targets
    # ------------------------------------------------------------

    gse203348 = (
        pd.read_csv(
            GSE203348
        )
        .set_index(
            "imodulon"
        )[
            "delta_biofilm_minus_planktonic"
        ]
        .astype(float)
    )

    gse120602_table = pd.read_csv(
        GSE120602,
        index_col=0,
    )

    gse120602 = (
        gse120602_table[
            "biofilm_vs_planktonic"
        ]
        .astype(float)
    )

    targets = {
        "GSE203348_biofilm":
            gse203348,

        "GSE120602_biofilm":
            gse120602,
    }

    catalogue = (
        pd.read_csv(
            CATALOGUE
        )[
            [
                "name",
                "function",
                "category",
            ]
        ]
        .rename(
            columns={
                "name":
                    "imodulon"
            }
        )
    )

    all_summary = []
    all_attribution = []
    all_residuals = []

    # ------------------------------------------------------------
    # 3. Decompose each biofilm vector
    # ------------------------------------------------------------

    for target_i, (
        target_name,
        target_series,
    ) in enumerate(
        targets.items()
    ):
        common = (
            dictionary.index
            .intersection(
                target_series.index
            )
        )

        if len(
            common
        ) != 116:
            raise RuntimeError(
                target_name
                + ": expected 116 common iModulons, found "
                + str(
                    len(
                        common
                    )
                )
            )

        D = (
            dictionary.loc[
                common,
                axis_names,
            ]
            .to_numpy(
                dtype=float
            )
        )

        target = (
            target_series.loc[
                common
            ]
            .to_numpy(
                dtype=float
            )
        )

        # --------------------------------------------------------
        # Exact Shapley attribution over all 2^11 subsets
        # --------------------------------------------------------

        attribution, explained = exact_shapley(
            target,
            D,
            axis_names,
        )

        attribution.insert(
            0,
            "target",
            target_name,
        )

        attribution = attribution.sort_values(
            "shapley_explained_energy",
            ascending=False,
        )

        all_attribution.append(
            attribution
        )

        # --------------------------------------------------------
        # Full dictionary projection
        # --------------------------------------------------------

        reconstruction = reconstruct(
            target,
            D,
        )

        residual = (
            target
            - reconstruction
        )

        reconstruction_cosine = cosine(
            target,
            reconstruction,
        )

        residual_fraction = (
            1.0
            - explained
        )

        # --------------------------------------------------------
        # Permutation significance
        # --------------------------------------------------------

        null, p = permutation_test(
            target=target,
            dictionary_matrix=D,
            observed=explained,
            n_perm=N_PERM,
            seed=SEED + target_i,
        )

        np.savetxt(
            OUT
            / (
                target_name
                + "_dictionary_permutation_null.csv"
            ),
            null,
            delimiter=",",
        )

        null_mean = float(
            np.mean(
                null
            )
        )

        null_95 = float(
            np.quantile(
                null,
                0.95,
            )
        )

        # --------------------------------------------------------
        # Residual programmes
        # --------------------------------------------------------

        residual_df = pd.DataFrame(
            {
                "target":
                    target_name,

                "imodulon":
                    common,

                "observed_delta":
                    target,

                "dictionary_reconstruction":
                    reconstruction,

                "residual_delta":
                    residual,
            }
        )

        residual_df[
            "abs_residual"
        ] = (
            residual_df[
                "residual_delta"
            ]
            .abs()
        )

        residual_energy = float(
            np.square(
                residual
            ).sum()
        )

        if residual_energy > 0:
            residual_df[
                "residual_energy_fraction"
            ] = (
                np.square(
                    residual_df[
                        "residual_delta"
                    ]
                )
                / residual_energy
            )

        else:
            residual_df[
                "residual_energy_fraction"
            ] = 0.0

        residual_df = residual_df.merge(
            catalogue,
            on="imodulon",
            how="left",
        )

        residual_df = residual_df.sort_values(
            "abs_residual",
            ascending=False,
        )

        all_residuals.append(
            residual_df
        )

        # --------------------------------------------------------
        # Summary
        # --------------------------------------------------------

        all_summary.append(
            {
                "target":
                    target_name,

                "dictionary_rank":
                    dictionary_rank,

                "dictionary_effective_dimension":
                    dictionary_effective_dim,

                "dictionary_fraction_energy_explained":
                    explained,

                "dictionary_reconstruction_cosine":
                    reconstruction_cosine,

                "residual_energy_fraction":
                    residual_fraction,

                "permutation_null_mean":
                    null_mean,

                "permutation_null_95pct":
                    null_95,

                "dictionary_permutation_p":
                    p,
            }
        )

        # --------------------------------------------------------
        # Print target
        # --------------------------------------------------------

        print()
        print("=" * 78)
        print(
            target_name
        )
        print("=" * 78)

        print()
        print("Dictionary subspace")
        print(
            "  fraction energy explained:",
            round(
                explained,
                4,
            ),
        )
        print(
            "  reconstruction cosine:",
            round(
                reconstruction_cosine,
                4,
            ),
        )
        print(
            "  residual energy:",
            round(
                residual_fraction,
                4,
            ),
        )
        print(
            "  permutation null mean:",
            round(
                null_mean,
                4,
            ),
        )
        print(
            "  permutation null 95%:",
            round(
                null_95,
                4,
            ),
        )
        print(
            "  permutation p:",
            round(
                p,
                4,
            ),
        )

        print()
        print("Stress-axis attribution")
        print(
            attribution[
                [
                    "axis",
                    "shapley_explained_energy",
                    "fraction_of_dictionary_explanation",
                    "single_axis_explained_energy",
                    "unique_increment_in_full_dictionary",
                    "signed_cosine",
                ]
            ]
            .round(
                4
            )
            .to_string(
                index=False
            )
        )

        print()
        print("Top unexplained programmes")
        print(
            residual_df[
                [
                    "imodulon",
                    "observed_delta",
                    "dictionary_reconstruction",
                    "residual_delta",
                    "residual_energy_fraction",
                    "function",
                ]
            ]
            .head(
                15
            )
            .round(
                {
                    "observed_delta":
                        3,

                    "dictionary_reconstruction":
                        3,

                    "residual_delta":
                        3,

                    "residual_energy_fraction":
                        4,
                }
            )
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # 4. Save combined outputs
    # ------------------------------------------------------------

    summary = pd.DataFrame(
        all_summary
    )

    attribution = pd.concat(
        all_attribution,
        ignore_index=True,
    )

    residuals = pd.concat(
        all_residuals,
        ignore_index=True,
    )

    summary.to_csv(
        OUT
        / "biofilm_dictionary_summary.csv",
        index=False,
    )

    attribution.to_csv(
        OUT
        / "biofilm_stress_axis_shapley_attribution.csv",
        index=False,
    )

    residuals.to_csv(
        OUT
        / "biofilm_dictionary_residuals.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 5. Cross-target attribution comparison
    # ------------------------------------------------------------

    wide_shapley = attribution.pivot(
        index="axis",
        columns="target",
        values="shapley_explained_energy",
    )

    wide_cosine = attribution.pivot(
        index="axis",
        columns="target",
        values="signed_cosine",
    )

    cross = (
        wide_shapley
        .add_prefix(
            "shapley_"
        )
        .join(
            wide_cosine.add_prefix(
                "cosine_"
            )
        )
        .reset_index()
    )

    cross.to_csv(
        OUT
        / "cross_biofilm_axis_comparison.csv",
        index=False,
    )

    print()
    print("=" * 78)
    print("Cross-biofilm axis comparison")
    print("=" * 78)

    print(
        cross.round(
            4
        ).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # 6. Technical audit gate
    # ------------------------------------------------------------

    shapley_sums = (
        attribution.groupby(
            "target"
        )[
            "shapley_explained_energy"
        ]
        .sum()
    )

    summary_index = (
        summary.set_index(
            "target"
        )
    )

    shapley_ok = True

    for target_name in targets:
        difference = abs(
            shapley_sums.loc[
                target_name
            ]
            - summary_index.loc[
                target_name,
                "dictionary_fraction_energy_explained",
            ]
        )

        if difference > 1e-10:
            shapley_ok = False

    checks = {
        "dictionary_116_by_11":
            dictionary.shape
            == (
                116,
                11,
            ),

        "dictionary_rank_positive":
            dictionary_rank
            > 0,

        "two_biofilm_targets":
            len(
                summary
            )
            == 2,

        "exact_shapley_sums_to_full_explanation":
            shapley_ok,

        "all_results_finite":
            bool(
                np.isfinite(
                    summary.select_dtypes(
                        include=[
                            np.number
                        ]
                    ).to_numpy()
                ).all()
            ),

        "permutation_tests_completed":
            bool(
                summary[
                    "dictionary_permutation_p"
                ]
                .notna()
                .all()
            ),
    }

    passed = all(
        checks.values()
    )

    pd.DataFrame(
        [
            {
                "check":
                    key,

                "pass":
                    value,
            }
            for key, value
            in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 7. Human-readable summary
    # ------------------------------------------------------------

    lines = [
        "# CF-State Phase 2 — Audit 11",
        "",
        "## Objective",
        "",
        "Decompose independent PAO1 biofilm-associated iModulon vectors "
        "against the frozen 11-condition GSE152295 controlled physiological "
        "dictionary.",
        "",
        "## Method",
        "",
        "The analysis uses the span of the 11 controlled perturbation vectors, "
        "rather than ordinary regression coefficients.",
        "",
        "Total dictionary coverage is the fraction of target-vector energy "
        "captured by orthogonal projection onto this frozen subspace.",
        "",
        "Because controlled stress axes are correlated, contributions are "
        "allocated using exact Shapley values across all 2^11 dictionary "
        "subsets. The Shapley contributions sum exactly to total dictionary "
        "explained energy.",
        "",
        "Signed cosine similarity is reported separately to preserve direction.",
        "",
        "A joint programme-label permutation preserves the internal geometry "
        "of the controlled dictionary while testing whether its alignment "
        "with each biofilm vector exceeds chance.",
        "",
        "## Results",
        "",
    ]

    for _, row in summary.iterrows():
        lines.extend(
            [
                "### "
                + str(
                    row[
                        "target"
                    ]
                ),
                "",
                "- Dictionary energy explained: "
                + format(
                    row[
                        "dictionary_fraction_energy_explained"
                    ],
                    ".2%",
                ),
                "- Reconstruction cosine: "
                + format(
                    row[
                        "dictionary_reconstruction_cosine"
                    ],
                    ".4f",
                ),
                "- Residual energy: "
                + format(
                    row[
                        "residual_energy_fraction"
                    ],
                    ".2%",
                ),
                "- Permutation p: "
                + format(
                    row[
                        "dictionary_permutation_p"
                    ],
                    ".4g",
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation boundary",
            "",
            "The Shapley quantities describe how the frozen controlled "
            "perturbation subspace overlaps each biofilm transcriptional "
            "state. They do not imply that a biofilm sample literally "
            "experienced every attributed laboratory stress.",
            "",
            "Axes were frozen before this decomposition; no biofilm target "
            "was used to fit or select them.",
            "",
            "## Decision",
            "",
            "PASS: "
            + str(
                passed
            ),
        ]
    )

    (
        OUT
        / "AUDIT_11_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 11:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_11_SUMMARY.md",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
