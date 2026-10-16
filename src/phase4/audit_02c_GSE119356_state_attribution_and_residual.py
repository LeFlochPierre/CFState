from pathlib import Path
from math import factorial

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


ROOT = Path.cwd()

REGIONAL = (
    ROOT
    / "results/phase4/audit_02a_GSE119356_matched_projection_QC"
    / "matched_region_imodulon_contrasts.csv"
)

POOLED = (
    ROOT
    / "results/phase4/audit_02a_GSE119356_matched_projection_QC"
    / "pooled_invivo_minus_invitro_imodulon_contrast.csv"
)

PRIMARY_FILE = (
    ROOT
    / "results/phase3/audit_04_freeze_crossstrain_state_framework"
    / "phase3_robust_transferable_imodulons.csv"
)

REFERENCE_LIBRARY = (
    ROOT
    / "results/phase4/audit_02b_GSE119356_CF_state_interpretation"
    / "controlled_state_reference_library.csv"
)

PREVIOUS_SPAN_FIT = (
    ROOT
    / "results/phase4/audit_02b_GSE119356_CF_state_interpretation"
    / "controlled_state_subspace_fit.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_02c_GSE119356_state_attribution_and_residual"
)


REGIONS = [
    "BR",
    "ULP",
    "ULC",
    "LLP",
    "LLC",
]


def cosine(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return np.nan

    return float(
        np.dot(
            a,
            b,
        )
        / denominator
    )


def projection_energy_fraction(
    y,
    B,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    if B.shape[1] == 0:
        return 0.0

    beta = np.linalg.lstsq(
        B,
        y,
        rcond=None,
    )[0]

    reconstruction = (
        B
        @ beta
    )

    denominator = float(
        np.dot(
            y,
            y,
        )
    )

    if denominator == 0:
        return np.nan

    return float(
        np.dot(
            reconstruction,
            reconstruction,
        )
        / denominator
    )


def project_vector(
    y,
    B,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    beta = np.linalg.lstsq(
        B,
        y,
        rcond=None,
    )[0]

    reconstruction = (
        B
        @ beta
    )

    residual = (
        y
        - reconstruction
    )

    return (
        beta,
        reconstruction,
        residual,
    )


def exact_shapley(
    y,
    B,
    axis_names,
):
    """
    Exact Shapley decomposition of explained projection energy.

    The value function is:

        v(S) = ||Proj_span(S)(y)||^2 / ||y||^2

    Therefore Shapley values partition the fraction of target
    programme energy explained by the full controlled-state span.

    This is a geometric attribution, NOT a causal decomposition.
    """

    n_axes = len(
        axis_names
    )

    if n_axes > 15:
        raise RuntimeError(
            "Exact Shapley intentionally limited to <=15 axes."
        )

    values = {}

    for mask in range(
        1 << n_axes
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

        if len(
            indices
        ) == 0:
            values[
                mask
            ] = 0.0
        else:
            subset = B[
                :,
                indices
            ]

            values[
                mask
            ] = projection_energy_fraction(
                y,
                subset,
            )

    shapley = np.zeros(
        n_axes,
        dtype=float,
    )

    denominator = factorial(
        n_axes
    )

    for i in range(
        n_axes
    ):
        bit = (
            1 << i
        )

        for mask in range(
            1 << n_axes
        ):
            if mask & bit:
                continue

            subset_size = int(
                mask.bit_count()
            )

            weight = (
                factorial(
                    subset_size
                )
                * factorial(
                    n_axes
                    - subset_size
                    - 1
                )
                / denominator
            )

            marginal = (
                values[
                    mask | bit
                ]
                - values[
                    mask
                ]
            )

            shapley[
                i
            ] += (
                weight
                * marginal
            )

    full_mask = (
        1 << n_axes
    ) - 1

    full_value = values[
        full_mask
    ]

    loo_unique = np.zeros(
        n_axes,
        dtype=float,
    )

    for i in range(
        n_axes
    ):
        without_i = (
            full_mask
            & ~(
                1 << i
            )
        )

        loo_unique[
            i
        ] = (
            full_value
            - values[
                without_i
            ]
        )

    return (
        shapley,
        loo_unique,
        full_value,
        values,
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 02c")
    print("GSE119356 exact state attribution and CF-lung residual")
    print("=" * 78)

    for path in [
        REGIONAL,
        POOLED,
        PRIMARY_FILE,
        REFERENCE_LIBRARY,
        PREVIOUS_SPAN_FIT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Load CF-lung programme shifts
    # ----------------------------------------------------------

    regional = pd.read_csv(
        REGIONAL,
        index_col=0,
    )

    regional.index = (
        regional.index
        .astype(str)
        .str.strip()
    )

    regional = regional.apply(
        pd.to_numeric,
        errors="raise",
    )

    for region in REGIONS:
        if region not in regional.columns:
            raise RuntimeError(
                "Missing region "
                + region
            )

    imodulons = list(
        regional.index
    )

    if len(
        imodulons
    ) != 116:
        raise RuntimeError(
            "Expected 116 iModulons."
        )

    pooled_table = pd.read_csv(
        POOLED,
        index_col=0,
    )

    if pooled_table.shape[1] != 1:
        raise RuntimeError(
            "Expected exactly one pooled vector."
        )

    pooled = pd.to_numeric(
        pooled_table.iloc[
            :,
            0
        ],
        errors="raise",
    )

    pooled.index = (
        pooled.index
        .astype(str)
        .str.strip()
    )

    pooled = pooled.reindex(
        imodulons
    )

    if pooled.isna().any():
        raise RuntimeError(
            "Pooled vector missing iModulons."
        )

    # ----------------------------------------------------------
    # 2. Frozen Phase-3 primary 111 programmes
    # ----------------------------------------------------------

    primary_table = pd.read_csv(
        PRIMARY_FILE
    )

    if "imodulon" not in primary_table.columns:
        raise RuntimeError(
            "Primary programme file missing imodulon column."
        )

    primary_set = set(
        primary_table[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    primary = [
        imodulon
        for imodulon in imodulons
        if imodulon in primary_set
    ]

    if len(
        primary
    ) != 111:
        raise RuntimeError(
            "Expected 111 primary programmes, found "
            + str(
                len(
                    primary
                )
            )
        )

    print()
    print("Frozen interpretation space")
    print(
        "  all programmes:",
        len(
            imodulons
        ),
    )
    print(
        "  primary programmes:",
        len(
            primary
        ),
    )

    # ----------------------------------------------------------
    # 3. Controlled-state reference library
    # ----------------------------------------------------------

    references = pd.read_csv(
        REFERENCE_LIBRARY,
        index_col=0,
    )

    references.index = (
        references.index
        .astype(str)
        .str.strip()
    )

    references = references.apply(
        pd.to_numeric,
        errors="raise",
    )

    references = references.reindex(
        imodulons
    )

    if references.isna().any().any():
        raise RuntimeError(
            "Reference library missing iModulons."
        )

    axis_names = list(
        references.columns
    )

    if len(
        axis_names
    ) != 11:
        raise RuntimeError(
            "Expected 11 controlled-state axes, found "
            + str(
                len(
                    axis_names
                )
            )
        )

    print()
    print("Controlled-state axes")

    for axis in axis_names:
        print(
            " ",
            axis,
        )

    # ----------------------------------------------------------
    # 4. Normalize reference axes in PRIMARY_111 space
    #
    # Scaling does not alter subspaces, but normalization improves
    # numerical conditioning and makes direct cosines transparent.
    # ----------------------------------------------------------

    B_raw = references.loc[
        primary,
        axis_names,
    ].to_numpy(
        dtype=float
    )

    axis_norms = np.linalg.norm(
        B_raw,
        axis=0,
    )

    if (
        axis_norms <= 0
    ).any():
        raise RuntimeError(
            "Zero-norm controlled-state axis."
        )

    B = (
        B_raw
        / axis_norms
    )

    library_rank = int(
        np.linalg.matrix_rank(
            B
        )
    )

    print()
    print("Controlled-state library geometry")
    print(
        "  axes:",
        B.shape[1],
    )
    print(
        "  rank:",
        library_rank,
    )

    # ----------------------------------------------------------
    # 5. Targets
    # ----------------------------------------------------------

    targets = {}

    for region in REGIONS:
        targets[
            region
        ] = regional.loc[
            primary,
            region,
        ]

    targets[
        "POOLED"
    ] = pooled.reindex(
        primary
    )

    # ----------------------------------------------------------
    # 6. Exact Shapley attribution
    # ----------------------------------------------------------

    attribution_rows = []

    summary_rows = []

    residual_vectors = {}

    reconstruction_vectors = {}

    all_shapley_sums = []

    full_span_values = {}

    for target_name, target_series in targets.items():
        y = target_series.to_numpy(
            dtype=float
        )

        (
            shapley,
            loo_unique,
            full_value,
            subset_values,
        ) = exact_shapley(
            y,
            B,
            axis_names,
        )

        beta, reconstruction, residual = project_vector(
            y,
            B,
        )

        residual_vectors[
            target_name
        ] = pd.Series(
            residual,
            index=primary,
            name=target_name,
        )

        reconstruction_vectors[
            target_name
        ] = pd.Series(
            reconstruction,
            index=primary,
            name=target_name,
        )

        full_span_values[
            target_name
        ] = full_value

        shapley_sum = float(
            shapley.sum()
        )

        all_shapley_sums.append(
            shapley_sum
        )

        residual_energy_fraction = float(
            np.dot(
                residual,
                residual,
            )
            / np.dot(
                y,
                y,
            )
        )

        reconstruction_cosine = cosine(
            y,
            reconstruction,
        )

        residual_axis_dot = np.abs(
            B.T
            @ residual
        )

        max_residual_axis_dot = float(
            residual_axis_dot.max()
        )

        summary_rows.append(
            {
                "target":
                    target_name,

                "controlled_state_energy_fraction":
                    full_value,

                "residual_energy_fraction":
                    residual_energy_fraction,

                "reconstruction_cosine":
                    reconstruction_cosine,

                "shapley_sum":
                    shapley_sum,

                "shapley_closure_error":
                    abs(
                        shapley_sum
                        - full_value
                    ),

                "max_abs_residual_axis_dot":
                    max_residual_axis_dot,
            }
        )

        for i, axis in enumerate(
            axis_names
        ):
            reference = B[
                :,
                i
            ]

            pairwise_cosine = cosine(
                y,
                reference,
            )

            pairwise_pearson = float(
                pearsonr(
                    y,
                    reference,
                ).statistic
            )

            explained_share = (
                float(
                    shapley[
                        i
                    ]
                    / full_value
                )
                if full_value > 0
                else np.nan
            )

            attribution_rows.append(
                {
                    "target":
                        target_name,

                    "state_axis":
                        axis,

                    "shapley_energy_fraction":
                        float(
                            shapley[
                                i
                            ]
                        ),

                    "share_of_explained_energy":
                        explained_share,

                    "leave_one_axis_out_unique_energy":
                        float(
                            loo_unique[
                                i
                            ]
                        ),

                    "pairwise_cosine":
                        pairwise_cosine,

                    "pairwise_pearson":
                        pairwise_pearson,

                    "full_model_coefficient":
                        float(
                            beta[
                                i
                            ]
                        ),
                }
            )

    attribution = pd.DataFrame(
        attribution_rows
    )

    summary = pd.DataFrame(
        summary_rows
    )

    print()
    print("Exact controlled-state attribution summary")
    print(
        summary
        .round(
            5
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Print top Shapley-attributed states
    #
    # Pairwise cosine is retained alongside Shapley attribution
    # because Shapley is unsigned explained energy.
    # ----------------------------------------------------------

    print()
    print("Largest Shapley controlled-state attributions")

    for target_name in targets:
        subset = (
            attribution.loc[
                attribution[
                    "target"
                ]
                == target_name
            ]
            .sort_values(
                "shapley_energy_fraction",
                ascending=False,
            )
            .head(
                7
            )
        )

        print()
        print(
            target_name
        )

        print(
            subset[
                [
                    "state_axis",
                    "shapley_energy_fraction",
                    "share_of_explained_energy",
                    "leave_one_axis_out_unique_energy",
                    "pairwise_cosine",
                ]
            ]
            .round(
                4
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 8. Residual programme matrix
    # ----------------------------------------------------------

    residual_matrix = pd.DataFrame(
        residual_vectors
    )

    reconstruction_matrix = pd.DataFrame(
        reconstruction_vectors
    )

    residual_correlation = (
        residual_matrix.corr()
    )

    print()
    print("Regional controlled-state residual correlation")
    print(
        residual_correlation.loc[
            REGIONS,
            REGIONS,
        ]
        .round(
            3
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 9. Pooled residual programme ranking
    # ----------------------------------------------------------

    pooled_residual = residual_matrix[
        "POOLED"
    ]

    pooled_residual_table = pd.DataFrame(
        {
            "imodulon":
                primary,

            "residual_activity":
                pooled_residual.reindex(
                    primary
                ).to_numpy(),

            "abs_residual_activity":
                np.abs(
                    pooled_residual.reindex(
                        primary
                    ).to_numpy()
                ),
        }
    )

    pooled_residual_table = (
        pooled_residual_table
        .sort_values(
            "abs_residual_activity",
            ascending=False,
        )
    )

    print()
    print("Largest pooled residual programmes after all controlled states")
    print(
        pooled_residual_table[
            [
                "imodulon",
                "residual_activity",
            ]
        ]
        .head(
            25
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 10. Residual spatial heterogeneity
    # ----------------------------------------------------------

    regional_residual = residual_matrix[
        REGIONS
    ]

    residual_heterogeneity = pd.DataFrame(
        {
            "imodulon":
                primary,

            "regional_residual_mean":
                regional_residual.mean(
                    axis=1
                ).reindex(
                    primary
                ).to_numpy(),

            "regional_residual_sd":
                regional_residual.std(
                    axis=1,
                    ddof=1,
                ).reindex(
                    primary
                ).to_numpy(),

            "regional_residual_range":
                (
                    regional_residual.max(
                        axis=1
                    )
                    - regional_residual.min(
                        axis=1
                    )
                )
                .reindex(
                    primary
                )
                .to_numpy(),
        }
    )

    residual_heterogeneity = (
        residual_heterogeneity
        .sort_values(
            "regional_residual_sd",
            ascending=False,
        )
    )

    print()
    print("Most spatially heterogeneous residual programmes")
    print(
        residual_heterogeneity[
            [
                "imodulon",
                "regional_residual_mean",
                "regional_residual_sd",
                "regional_residual_range",
            ]
        ]
        .head(
            20
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 11. Consensus residual signal
    #
    # These are programmes that remain shifted in the same
    # direction in every region AFTER removing the entire
    # controlled-state subspace.
    # ----------------------------------------------------------

    consensus_rows = []

    for imodulon in primary:
        values = regional_residual.loc[
            imodulon,
            REGIONS,
        ].to_numpy(
            dtype=float
        )

        all_positive = bool(
            np.all(
                values > 0
            )
        )

        all_negative = bool(
            np.all(
                values < 0
            )
        )

        same_direction = (
            all_positive
            or all_negative
        )

        consensus_rows.append(
            {
                "imodulon":
                    imodulon,

                "consensus_residual_direction":
                    same_direction,

                "direction":
                    (
                        "positive"
                        if all_positive
                        else (
                            "negative"
                            if all_negative
                            else "mixed"
                        )
                    ),

                "regional_residual_mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "minimum_abs_residual":
                    float(
                        np.min(
                            np.abs(
                                values
                            )
                        )
                    ),

                "maximum_abs_residual":
                    float(
                        np.max(
                            np.abs(
                                values
                            )
                        )
                    ),
            }
        )

    consensus = pd.DataFrame(
        consensus_rows
    )

    consensus_ranked = (
        consensus.loc[
            consensus[
                "consensus_residual_direction"
            ]
        ]
        .sort_values(
            "minimum_abs_residual",
            ascending=False,
        )
    )

    print()
    print(
        "Strongest same-direction residual programmes in all 5 CF-lung regions"
    )

    print(
        consensus_ranked[
            [
                "imodulon",
                "direction",
                "regional_residual_mean",
                "minimum_abs_residual",
                "maximum_abs_residual",
            ]
        ]
        .head(
            20
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 12. Validate against Audit 02b span-energy calculations
    # ----------------------------------------------------------

    previous = pd.read_csv(
        PREVIOUS_SPAN_FIT
    )

    previous_primary = previous.loc[
        previous[
            "programme_set"
        ]
        == "PRIMARY_111"
    ].copy()

    previous_lookup = dict(
        zip(
            previous_primary[
                "target"
            ],
            previous_primary[
                "controlled_state_span_energy_fraction"
            ],
        )
    )

    span_differences = []

    for target_name in targets:
        if target_name not in previous_lookup:
            raise RuntimeError(
                "Previous span fit missing target "
                + target_name
            )

        difference = abs(
            full_span_values[
                target_name
            ]
            - float(
                previous_lookup[
                    target_name
                ]
            )
        )

        span_differences.append(
            difference
        )

    maximum_span_difference = float(
        max(
            span_differences
        )
    )

    # ----------------------------------------------------------
    # 13. Pooled residual linearity
    #
    # Projection onto a fixed linear subspace is linear, so the
    # pooled residual should equal the mean regional residual.
    # ----------------------------------------------------------

    mean_regional_residual = (
        residual_matrix[
            REGIONS
        ].mean(
            axis=1
        )
    )

    pooled_residual_difference = float(
        np.max(
            np.abs(
                mean_regional_residual.to_numpy(
                    dtype=float
                )
                - residual_matrix[
                    "POOLED"
                ].to_numpy(
                    dtype=float
                )
            )
        )
    )

    # ----------------------------------------------------------
    # 14. Technical consistency checks
    #
    # No biological state is required to dominate.
    # No minimum residual similarity is imposed.
    # ----------------------------------------------------------

    maximum_shapley_closure_error = float(
        summary[
            "shapley_closure_error"
        ].max()
    )

    maximum_residual_axis_dot = float(
        summary[
            "max_abs_residual_axis_dot"
        ].max()
    )

    minimum_shapley = float(
        attribution[
            "shapley_energy_fraction"
        ].min()
    )

    finite_outputs = bool(
        np.isfinite(
            attribution[
                [
                    "shapley_energy_fraction",
                    "share_of_explained_energy",
                    "leave_one_axis_out_unique_energy",
                    "pairwise_cosine",
                    "pairwise_pearson",
                    "full_model_coefficient",
                ]
            ].to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            residual_matrix.to_numpy(
                dtype=float
            )
        ).all()
    )

    checks = {
        "111_primary_programmes":
            len(
                primary
            )
            == 111,

        "11_controlled_state_axes":
            len(
                axis_names
            )
            == 11,

        "library_rank_ge_8":
            library_rank
            >= 8,

        "shapley_closure_error_lt_1e8":
            maximum_shapley_closure_error
            < 1e-8,

        "no_material_negative_shapley":
            minimum_shapley
            > -1e-10,

        "residual_orthogonal_to_controlled_span":
            maximum_residual_axis_dot
            < 1e-8,

        "matches_audit_02b_span_energy":
            maximum_span_difference
            < 1e-8,

        "pooled_residual_equals_mean_regional_residual":
            pooled_residual_difference
            < 1e-8,

        "finite_outputs":
            finite_outputs,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 02c consistency checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    status = (
        "GSE119356_CF_STATE_ATTRIBUTION_AND_RESIDUAL_ACCEPTED"
        if passed
        else "GSE119356_CF_STATE_ATTRIBUTION_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: Shapley values partition geometric explained "
        "programme energy. They are not causal effect sizes and "
        "do not prove the corresponding literal perturbation is "
        "present in the CF lung."
    )

    # ----------------------------------------------------------
    # 15. Save
    # ----------------------------------------------------------

    attribution.to_csv(
        OUT
        / "exact_controlled_state_shapley_attribution.csv",
        index=False,
    )

    summary.to_csv(
        OUT
        / "controlled_state_attribution_summary.csv",
        index=False,
    )

    reconstruction_matrix.to_csv(
        OUT
        / "controlled_state_reconstructed_CF_vectors.csv"
    )

    residual_matrix.to_csv(
        OUT
        / "controlled_state_residual_CF_vectors.csv"
    )

    residual_correlation.to_csv(
        OUT
        / "controlled_state_residual_correlation.csv"
    )

    pooled_residual_table.to_csv(
        OUT
        / "pooled_CF_residual_programmes.csv",
        index=False,
    )

    residual_heterogeneity.to_csv(
        OUT
        / "regional_residual_programme_heterogeneity.csv",
        index=False,
    )

    consensus.to_csv(
        OUT
        / "regional_residual_direction_consensus.csv",
        index=False,
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

    (
        OUT
        / "AUDIT_02C_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed Audit 02c consistency checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
