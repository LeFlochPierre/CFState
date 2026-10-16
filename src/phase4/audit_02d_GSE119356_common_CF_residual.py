from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

RESIDUAL = (
    ROOT
    / "results/phase4/audit_02c_GSE119356_state_attribution_and_residual"
    / "controlled_state_residual_CF_vectors.csv"
)

SUMMARY = (
    ROOT
    / "results/phase4/audit_02c_GSE119356_state_attribution_and_residual"
    / "controlled_state_attribution_summary.csv"
)

PRIMARY_FILE = (
    ROOT
    / "results/phase3/audit_04_freeze_crossstrain_state_framework"
    / "phase3_robust_transferable_imodulons.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_02d_GSE119356_common_CF_residual"
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


def pearson(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        np.std(a) == 0
        or np.std(b) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def projection_onto_direction(
    vector,
    direction,
):
    vector = np.asarray(
        vector,
        dtype=float,
    )

    direction = np.asarray(
        direction,
        dtype=float,
    )

    denominator = float(
        np.dot(
            direction,
            direction,
        )
    )

    if denominator == 0:
        raise RuntimeError(
            "Zero-norm common residual direction."
        )

    coefficient = float(
        np.dot(
            vector,
            direction,
        )
        / denominator
    )

    component = (
        coefficient
        * direction
    )

    residual = (
        vector
        - component
    )

    total_energy = float(
        np.dot(
            vector,
            vector,
        )
    )

    common_energy = float(
        np.dot(
            component,
            component,
        )
        / total_energy
    )

    local_energy = float(
        np.dot(
            residual,
            residual,
        )
        / total_energy
    )

    return {
        "coefficient":
            coefficient,

        "common_component":
            component,

        "local_component":
            residual,

        "common_energy_fraction":
            common_energy,

        "local_energy_fraction":
            local_energy,
    }


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 02d")
    print("GSE119356 common versus spatial CF-lung residual")
    print("=" * 78)

    for path in [
        RESIDUAL,
        SUMMARY,
        PRIMARY_FILE,
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
    # 1. Load residual vectors
    # ----------------------------------------------------------

    residual = pd.read_csv(
        RESIDUAL,
        index_col=0,
    )

    residual.index = (
        residual.index
        .astype(str)
        .str.strip()
    )

    residual = residual.apply(
        pd.to_numeric,
        errors="raise",
    )

    for region in REGIONS:
        if region not in residual.columns:
            raise RuntimeError(
                "Missing residual region: "
                + region
            )

    if "POOLED" not in residual.columns:
        raise RuntimeError(
            "Missing pooled residual vector."
        )

    primary_table = pd.read_csv(
        PRIMARY_FILE
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
        for imodulon in residual.index
        if imodulon in primary_set
    ]

    if len(
        primary
    ) != 111:
        raise RuntimeError(
            "Expected 111 primary programmes."
        )

    residual = residual.reindex(
        primary
    )

    regional = residual[
        REGIONS
    ]

    pooled = residual[
        "POOLED"
    ]

    print()
    print("Residual programme space")
    print(
        "  programmes:",
        len(
            primary
        ),
    )
    print(
        "  lung regions:",
        len(
            REGIONS
        ),
    )

    # ----------------------------------------------------------
    # 2. Verify pooled residual equals regional mean
    # ----------------------------------------------------------

    regional_mean = regional.mean(
        axis=1
    )

    pooled_difference = float(
        np.max(
            np.abs(
                regional_mean.to_numpy(
                    dtype=float
                )
                - pooled.to_numpy(
                    dtype=float
                )
            )
        )
    )

    print()
    print("Pooled residual consistency")
    print(
        "  max pooled-vs-regional-mean difference:",
        format(
            pooled_difference,
            ".3e",
        ),
    )

    # ----------------------------------------------------------
    # 3. Define common CF residual direction
    #
    # Direction = pooled residual / mean regional residual.
    #
    # This is descriptive, not yet claimed to generalize beyond
    # this patient.
    # ----------------------------------------------------------

    common = regional_mean.copy()

    common_norm = float(
        np.linalg.norm(
            common.to_numpy(
                dtype=float
            )
        )
    )

    if common_norm == 0:
        raise RuntimeError(
            "Common residual has zero norm."
        )

    common_unit = (
        common
        / common_norm
    )

    # ----------------------------------------------------------
    # 4. Common-versus-local decomposition per region
    # ----------------------------------------------------------

    decomposition_rows = []

    local_components = {}

    common_components = {}

    for region in REGIONS:
        vector = regional[
            region
        ]

        result = projection_onto_direction(
            vector.to_numpy(
                dtype=float
            ),
            common.to_numpy(
                dtype=float
            ),
        )

        common_components[
            region
        ] = pd.Series(
            result[
                "common_component"
            ],
            index=primary,
        )

        local_components[
            region
        ] = pd.Series(
            result[
                "local_component"
            ],
            index=primary,
        )

        decomposition_rows.append(
            {
                "region":
                    region,

                "common_direction_coefficient":
                    result[
                        "coefficient"
                    ],

                "common_residual_energy_fraction":
                    result[
                        "common_energy_fraction"
                    ],

                "local_residual_energy_fraction":
                    result[
                        "local_energy_fraction"
                    ],

                "cosine_to_common_residual":
                    cosine(
                        vector,
                        common
                    ),

                "pearson_to_common_residual":
                    pearson(
                        vector,
                        common
                    ),
            }
        )

    decomposition = pd.DataFrame(
        decomposition_rows
    )

    print()
    print("Common-versus-local residual decomposition")
    print(
        decomposition
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Leave-one-region-out common-direction stability
    #
    # For each held-out region:
    # build common residual using ONLY the other four regions,
    # then compare held-out residual with that independent
    # common direction.
    # ----------------------------------------------------------

    loo_rows = []

    loo_common_vectors = {}

    for held_out in REGIONS:
        training_regions = [
            region
            for region in REGIONS
            if region != held_out
        ]

        loo_common = regional[
            training_regions
        ].mean(
            axis=1
        )

        held_vector = regional[
            held_out
        ]

        loo_common_vectors[
            held_out
        ] = loo_common

        result = projection_onto_direction(
            held_vector.to_numpy(
                dtype=float
            ),
            loo_common.to_numpy(
                dtype=float
            ),
        )

        loo_rows.append(
            {
                "held_out_region":
                    held_out,

                "training_regions":
                    ",".join(
                        training_regions
                    ),

                "pearson_to_other_four":
                    pearson(
                        held_vector,
                        loo_common,
                    ),

                "cosine_to_other_four":
                    cosine(
                        held_vector,
                        loo_common,
                    ),

                "energy_along_other_four_common_direction":
                    result[
                        "common_energy_fraction"
                    ],

                "remaining_local_energy_fraction":
                    result[
                        "local_energy_fraction"
                    ],
            }
        )

    loo = pd.DataFrame(
        loo_rows
    )

    print()
    print("Leave-one-region-out common residual validation")
    print(
        loo
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Stability of common residual itself
    #
    # Compare each four-region estimate to the five-region
    # pooled estimate.
    # ----------------------------------------------------------

    common_stability_rows = []

    for held_out in REGIONS:
        loo_common = loo_common_vectors[
            held_out
        ]

        common_stability_rows.append(
            {
                "excluded_region":
                    held_out,

                "pearson_LOO_vs_all5":
                    pearson(
                        loo_common,
                        common
                    ),

                "cosine_LOO_vs_all5":
                    cosine(
                        loo_common,
                        common
                    ),
            }
        )

    common_stability = pd.DataFrame(
        common_stability_rows
    )

    print()
    print("Common residual estimate stability")
    print(
        common_stability
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Rank common residual programmes
    # ----------------------------------------------------------

    common_table = pd.DataFrame(
        {
            "imodulon":
                primary,

            "common_residual_activity":
                common.reindex(
                    primary
                ).to_numpy(),

            "abs_common_residual_activity":
                np.abs(
                    common.reindex(
                        primary
                    ).to_numpy()
                ),
        }
    )

    common_table = common_table.sort_values(
        "abs_common_residual_activity",
        ascending=False,
    )

    print()
    print("Largest common CF-lung residual programmes")
    print(
        common_table[
            [
                "imodulon",
                "common_residual_activity",
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
    # 8. Programme-level leave-one-out sign stability
    #
    # For every programme, ask whether:
    #
    # - all 5 individual regional residuals have same sign
    # - all 5 leave-one-out common estimates have same sign
    #
    # This identifies particularly robust common residual
    # programmes.
    # ----------------------------------------------------------

    programme_stability_rows = []

    for imodulon in primary:
        region_values = regional.loc[
            imodulon,
            REGIONS,
        ].to_numpy(
            dtype=float
        )

        loo_values = np.array(
            [
                loo_common_vectors[
                    region
                ].loc[
                    imodulon
                ]
                for region in REGIONS
            ],
            dtype=float,
        )

        region_all_positive = bool(
            np.all(
                region_values > 0
            )
        )

        region_all_negative = bool(
            np.all(
                region_values < 0
            )
        )

        loo_all_positive = bool(
            np.all(
                loo_values > 0
            )
        )

        loo_all_negative = bool(
            np.all(
                loo_values < 0
            )
        )

        regional_consensus = (
            region_all_positive
            or region_all_negative
        )

        loo_consensus = (
            loo_all_positive
            or loo_all_negative
        )

        same_consensus_direction = (
            (
                region_all_positive
                and loo_all_positive
            )
            or (
                region_all_negative
                and loo_all_negative
            )
        )

        programme_stability_rows.append(
            {
                "imodulon":
                    imodulon,

                "regional_sign_consensus":
                    regional_consensus,

                "LOO_common_sign_consensus":
                    loo_consensus,

                "same_consensus_direction":
                    same_consensus_direction,

                "direction":
                    (
                        "positive"
                        if (
                            region_all_positive
                            and loo_all_positive
                        )
                        else (
                            "negative"
                            if (
                                region_all_negative
                                and loo_all_negative
                            )
                            else "mixed"
                        )
                    ),

                "common_residual_activity":
                    float(
                        common.loc[
                            imodulon
                        ]
                    ),

                "minimum_abs_regional_residual":
                    float(
                        np.min(
                            np.abs(
                                region_values
                            )
                        )
                    ),

                "minimum_abs_LOO_common":
                    float(
                        np.min(
                            np.abs(
                                loo_values
                            )
                        )
                    ),
            }
        )

    programme_stability = pd.DataFrame(
        programme_stability_rows
    )

    stable_programmes = (
        programme_stability.loc[
            programme_stability[
                "same_consensus_direction"
            ]
        ]
        .sort_values(
            "minimum_abs_regional_residual",
            ascending=False,
        )
    )

    print()
    print("Strongest leave-one-region-out stable common programmes")
    print(
        stable_programmes[
            [
                "imodulon",
                "direction",
                "common_residual_activity",
                "minimum_abs_regional_residual",
                "minimum_abs_LOO_common",
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
    # 9. Spatial local residuals
    #
    # These are what remain in each region after removal of
    # the common residual direction.
    # ----------------------------------------------------------

    local_matrix = pd.DataFrame(
        local_components
    )

    local_correlation = (
        local_matrix.corr()
    )

    local_heterogeneity = pd.DataFrame(
        {
            "imodulon":
                primary,

            "local_sd":
                local_matrix.std(
                    axis=1,
                    ddof=1,
                )
                .reindex(
                    primary
                )
                .to_numpy(),

            "local_range":
                (
                    local_matrix.max(
                        axis=1
                    )
                    - local_matrix.min(
                        axis=1
                    )
                )
                .reindex(
                    primary
                )
                .to_numpy(),
        }
    )

    local_heterogeneity = (
        local_heterogeneity
        .sort_values(
            "local_sd",
            ascending=False,
        )
    )

    print()
    print("Local residual correlation after common CF direction removal")
    print(
        local_correlation
        .round(
            3
        )
        .to_string()
    )

    print()
    print("Most spatially heterogeneous local programmes")
    print(
        local_heterogeneity
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
    # 10. Energy accounting relative to original CF programme
    #     shifts
    #
    # Audit 02c records the controlled-state residual fraction.
    # Combining it with the common-direction fraction tells us
    # approximately how much ORIGINAL CF-shift energy is:
    #
    # known controlled state
    # common unexplained residual
    # local unexplained residual
    #
    # per region.
    # ----------------------------------------------------------

    attribution_summary = pd.read_csv(
        SUMMARY
    )

    accounting_rows = []

    for region in REGIONS:
        row = attribution_summary.loc[
            attribution_summary[
                "target"
            ]
            == region
        ]

        if len(
            row
        ) != 1:
            raise RuntimeError(
                "Missing attribution summary for "
                + region
            )

        row = row.iloc[
            0
        ]

        known_fraction = float(
            row[
                "controlled_state_energy_fraction"
            ]
        )

        residual_fraction = float(
            row[
                "residual_energy_fraction"
            ]
        )

        decomposition_row = decomposition.loc[
            decomposition[
                "region"
            ]
            == region
        ].iloc[
            0
        ]

        common_within_residual = float(
            decomposition_row[
                "common_residual_energy_fraction"
            ]
        )

        local_within_residual = float(
            decomposition_row[
                "local_residual_energy_fraction"
            ]
        )

        common_original = (
            residual_fraction
            * common_within_residual
        )

        local_original = (
            residual_fraction
            * local_within_residual
        )

        total = (
            known_fraction
            + common_original
            + local_original
        )

        accounting_rows.append(
            {
                "region":
                    region,

                "known_controlled_state_fraction":
                    known_fraction,

                "common_unexplained_CF_fraction":
                    common_original,

                "local_unexplained_fraction":
                    local_original,

                "energy_sum":
                    total,
            }
        )

    accounting = pd.DataFrame(
        accounting_rows
    )

    print()
    print("Approximate original CF-shift energy accounting")
    print(
        accounting
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 11. Consistency checks
    #
    # We deliberately do NOT require a minimum common-residual
    # similarity: whether the residual is shared is a biological
    # result, not something the audit is allowed to force.
    # ----------------------------------------------------------

    maximum_energy_sum_error = float(
        np.max(
            np.abs(
                accounting[
                    "energy_sum"
                ].to_numpy(
                    dtype=float
                )
                - 1.0
            )
        )
    )

    common_components_matrix = pd.DataFrame(
        common_components
    )

    # Every common component must lie on exactly the same
    # one-dimensional direction.
    common_component_cosines = []

    for region in REGIONS:
        value = cosine(
            common_components_matrix[
                region
            ],
            common,
        )

        common_component_cosines.append(
            abs(
                value
            )
        )

    minimum_common_component_alignment = float(
        min(
            common_component_cosines
        )
    )

    finite_outputs = bool(
        np.isfinite(
            decomposition.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            loo.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            accounting.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy(
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

        "five_regions":
            len(
                REGIONS
            )
            == 5,

        "pooled_equals_regional_mean":
            pooled_difference
            < 1e-8,

        "five_LOO_common_vectors":
            len(
                loo_common_vectors
            )
            == 5,

        "common_components_collinear":
            minimum_common_component_alignment
            > 1.0
            - 1e-10,

        "energy_accounting_closes":
            maximum_energy_sum_error
            < 1e-8,

        "finite_outputs":
            finite_outputs,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 02d consistency checks")

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
        "GSE119356_COMMON_AND_SPATIAL_CF_RESIDUAL_GENERATED"
        if passed
        else "GSE119356_COMMON_CF_RESIDUAL_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: the common residual is shared across spatial "
        "regions of ONE explanted CF lung. It must not yet be "
        "called a universal CF programme. Multi-patient replication "
        "is required."
    )

    # ----------------------------------------------------------
    # 12. Save
    # ----------------------------------------------------------

    common.to_csv(
        OUT
        / "common_CF_lung_residual_vector.csv",
        header=[
            "common_residual_activity"
        ],
    )

    decomposition.to_csv(
        OUT
        / "common_vs_local_residual_decomposition.csv",
        index=False,
    )

    loo.to_csv(
        OUT
        / "leave_one_region_out_common_residual_validation.csv",
        index=False,
    )

    common_stability.to_csv(
        OUT
        / "common_residual_estimate_stability.csv",
        index=False,
    )

    common_table.to_csv(
        OUT
        / "common_CF_residual_programmes_ranked.csv",
        index=False,
    )

    programme_stability.to_csv(
        OUT
        / "common_residual_programme_LOO_stability.csv",
        index=False,
    )

    common_components_matrix.to_csv(
        OUT
        / "regional_common_residual_components.csv"
    )

    local_matrix.to_csv(
        OUT
        / "regional_local_residual_components.csv"
    )

    local_correlation.to_csv(
        OUT
        / "regional_local_residual_correlation.csv"
    )

    local_heterogeneity.to_csv(
        OUT
        / "local_residual_programme_heterogeneity.csv",
        index=False,
    )

    accounting.to_csv(
        OUT
        / "original_CF_shift_energy_accounting.csv",
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
        / "AUDIT_02D_STATUS.txt"
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
            "Failed Audit 02d consistency checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
