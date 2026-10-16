from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

STRICT_FILTER = (
    ROOT
    / "results/phase4/audit_03g_PRJEB24688_low_energy_sensitivity"
    / "strict_sample_filter.csv"
)

RESIDUALS = (
    ROOT
    / "results/phase4/audit_03f_PRJEB24688_multipatient_CF_replication"
    / "PRJEB24688_controlled_state_residuals_108.csv"
)

GSE_REFERENCE = (
    ROOT
    / "results/phase4/audit_03f_PRJEB24688_multipatient_CF_replication"
    / "GSE119356_mask_compatible_common_residual_108.csv"
)

AUDIT03G_COHORT = (
    ROOT
    / "results/phase4/audit_03g_PRJEB24688_low_energy_sensitivity"
    / "ALL_vs_STRICT_cohort_replication.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_03h_freeze_replicated_CF_airway_candidate"
)


PATIENTS = [
    "P24M1",
    "P30M0",
    "P73F1",
    "P77F1",
]

BASELINES = [
    "EARLY",
    "LATE",
]


def patient_from_sample(sample):
    match = re.match(
        r"^(P\d+[MF]\d+)_",
        str(sample),
    )

    if match is None:
        raise RuntimeError(
            "Could not parse patient from "
            + str(sample)
        )

    return match.group(1)


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


def unit_vector(series):
    values = series.to_numpy(
        dtype=float
    )

    norm = float(
        np.linalg.norm(
            values
        )
    )

    if norm <= 0:
        raise RuntimeError(
            "Cannot normalize zero vector."
        )

    return pd.Series(
        values / norm,
        index=series.index,
    )


def permutation_p(
    a,
    b,
    n_perm=10000,
    seed=20261007,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    observed = cosine(
        a,
        b,
    )

    rng = np.random.default_rng(
        seed
    )

    null = np.empty(
        n_perm,
        dtype=float,
    )

    for i in range(
        n_perm
    ):
        null[
            i
        ] = cosine(
            a,
            rng.permutation(
                b
            ),
        )

    p = (
        1
        + int(
            np.sum(
                null >= observed
            )
        )
    ) / (
        n_perm
        + 1
    )

    return observed, float(p)


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 03h")
    print("Freeze replicated CF-airway programme candidate")
    print("=" * 78)

    for path in [
        STRICT_FILTER,
        RESIDUALS,
        GSE_REFERENCE,
        AUDIT03G_COHORT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Frozen strict PRJEB24688 sample set
    # ----------------------------------------------------------

    sample_filter = pd.read_csv(
        STRICT_FILTER
    )

    strict_samples = (
        sample_filter.loc[
            sample_filter[
                "strict_both_baselines_pass"
            ].astype(bool),
            "sample",
        ]
        .astype(str)
        .tolist()
    )

    excluded_samples = (
        sample_filter.loc[
            ~sample_filter[
                "strict_both_baselines_pass"
            ].astype(bool),
            "sample",
        ]
        .astype(str)
        .tolist()
    )

    retained_patients = sorted(
        {
            patient_from_sample(
                sample
            )
            for sample in strict_samples
        }
    )

    print()
    print("Frozen strict PRJEB24688 replication subset")
    print(
        "  retained samples:",
        len(
            strict_samples
        ),
        strict_samples,
    )
    print(
        "  excluded samples:",
        len(
            excluded_samples
        ),
        excluded_samples,
    )
    print(
        "  retained patients:",
        len(
            retained_patients
        ),
        retained_patients,
    )

    # ----------------------------------------------------------
    # 2. Load exact 108-programme residual space
    # ----------------------------------------------------------

    residuals = pd.read_csv(
        RESIDUALS,
        index_col=0,
    )

    residuals.index = (
        residuals.index
        .astype(str)
        .str.strip()
    )

    residuals = residuals.apply(
        pd.to_numeric,
        errors="raise",
    )

    programmes = list(
        residuals.index
    )

    if len(
        programmes
    ) != 108:
        raise RuntimeError(
            "Expected 108 soft-core-stable primary programmes."
        )

    # ----------------------------------------------------------
    # 3. Load frozen GSE119356 discovery direction
    # ----------------------------------------------------------

    gse_table = pd.read_csv(
        GSE_REFERENCE,
        index_col=0,
    )

    if gse_table.shape[
        1
    ] != 1:
        raise RuntimeError(
            "Expected one GSE119356 reference column."
        )

    gse = pd.to_numeric(
        gse_table.iloc[
            :,
            0
        ],
        errors="raise",
    )

    gse.index = (
        gse.index
        .astype(str)
        .str.strip()
    )

    gse = gse.reindex(
        programmes
    )

    if gse.isna().any():
        raise RuntimeError(
            "GSE119356 reference missing programmes."
        )

    gse_unit = unit_vector(
        gse
    )

    # ----------------------------------------------------------
    # 4. Reconstruct strict patient residual vectors
    #
    # Equal weight per sputum biological sample within patient.
    # Keep EARLY and LATE separately first.
    # ----------------------------------------------------------

    patient_baseline_vectors = {}

    design_rows = []

    for patient in PATIENTS:
        patient_samples = [
            sample
            for sample in strict_samples
            if patient_from_sample(
                sample
            )
            == patient
        ]

        if not patient_samples:
            raise RuntimeError(
                "Strict subset contains no sample for "
                + patient
            )

        for baseline in BASELINES:
            columns = [
                sample
                + "__"
                + baseline
                for sample in patient_samples
            ]

            missing = [
                column
                for column in columns
                if column
                not in residuals.columns
            ]

            if missing:
                raise RuntimeError(
                    "Missing residual columns: "
                    + ", ".join(
                        missing
                    )
                )

            vector = residuals[
                columns
            ].mean(
                axis=1
            )

            key = (
                patient
                + "__"
                + baseline
            )

            patient_baseline_vectors[
                key
            ] = vector

            design_rows.append(
                {
                    "patient":
                        patient,

                    "baseline":
                        baseline,

                    "n_sputum_samples":
                        len(
                            patient_samples
                        ),

                    "samples":
                        ",".join(
                            patient_samples
                        ),
                }
            )

    patient_baseline_matrix = pd.DataFrame(
        patient_baseline_vectors
    )

    design = pd.DataFrame(
        design_rows
    )

    # ----------------------------------------------------------
    # 5. Baseline-averaged patient vectors
    #
    # EARLY and LATE are two legitimate laboratory references.
    # To freeze one patient-level direction for future testing,
    # give each baseline equal weight.
    # ----------------------------------------------------------

    patient_vectors = {}

    patient_rows = []

    for patient in PATIENTS:
        early = patient_baseline_matrix[
            patient
            + "__EARLY"
        ]

        late = patient_baseline_matrix[
            patient
            + "__LATE"
        ]

        averaged = (
            early
            + late
        ) / 2.0

        patient_vectors[
            patient
        ] = averaged

        observed_cosine, p = permutation_p(
            averaged,
            gse,
        )

        patient_rows.append(
            {
                "patient":
                    patient,

                "pearson_EARLY_vs_LATE":
                    pearson(
                        early,
                        late,
                    ),

                "cosine_EARLY_vs_LATE":
                    cosine(
                        early,
                        late,
                    ),

                "pearson_baseline_averaged_vs_GSE119356":
                    pearson(
                        averaged,
                        gse,
                    ),

                "cosine_baseline_averaged_vs_GSE119356":
                    observed_cosine,

                "permutation_p":
                    p,
            }
        )

    patient_matrix = pd.DataFrame(
        patient_vectors
    )

    patient_summary = pd.DataFrame(
        patient_rows
    )

    print()
    print("Baseline-averaged patient replication")
    print(
        patient_summary
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. PRJEB24688 replication-cohort direction
    #
    # Two versions:
    #
    # raw_equal_patient:
    #   equal patient weighting, retaining activity magnitudes.
    #
    # directional_equal_patient:
    #   normalize each patient first so no patient dominates
    #   merely because its activity vector has larger magnitude.
    # ----------------------------------------------------------

    raw_equal_patient = patient_matrix.mean(
        axis=1
    )

    patient_unit_vectors = {}

    for patient in PATIENTS:
        patient_unit_vectors[
            patient
        ] = unit_vector(
            patient_matrix[
                patient
            ]
        )

    patient_unit_matrix = pd.DataFrame(
        patient_unit_vectors
    )

    directional_equal_patient = unit_vector(
        patient_unit_matrix.mean(
            axis=1
        )
    )

    raw_equal_patient_unit = unit_vector(
        raw_equal_patient
    )

    # ----------------------------------------------------------
    # 7. Verify consistency with Audit 03g
    #
    # Reconstruct baseline-specific strict cohort vectors and
    # compare their GSE119356 metrics to the saved 03g results.
    # ----------------------------------------------------------

    audit03g = pd.read_csv(
        AUDIT03G_COHORT
    )

    audit03g_strict = audit03g.loc[
        audit03g[
            "analysis_set"
        ]
        == "STRICT"
    ].copy()

    consistency_rows = []

    for baseline in BASELINES:
        baseline_columns = [
            patient
            + "__"
            + baseline
            for patient in PATIENTS
        ]

        vector = patient_baseline_matrix[
            baseline_columns
        ].mean(
            axis=1
        )

        computed_pearson = pearson(
            vector,
            gse,
        )

        computed_cosine = cosine(
            vector,
            gse,
        )

        saved = audit03g_strict.loc[
            audit03g_strict[
                "baseline"
            ]
            == baseline
        ]

        if len(
            saved
        ) != 1:
            raise RuntimeError(
                "Could not uniquely resolve saved 03g "
                + baseline
                + " cohort row."
            )

        saved = saved.iloc[
            0
        ]

        consistency_rows.append(
            {
                "baseline":
                    baseline,

                "computed_pearson":
                    computed_pearson,

                "saved_pearson":
                    float(
                        saved[
                            "pearson_to_GSE119356"
                        ]
                    ),

                "pearson_difference":
                    abs(
                        computed_pearson
                        - float(
                            saved[
                                "pearson_to_GSE119356"
                            ]
                        )
                    ),

                "computed_cosine":
                    computed_cosine,

                "saved_cosine":
                    float(
                        saved[
                            "cosine_to_GSE119356"
                        ]
                    ),

                "cosine_difference":
                    abs(
                        computed_cosine
                        - float(
                            saved[
                                "cosine_to_GSE119356"
                            ]
                        )
                    ),
            }
        )

    consistency = pd.DataFrame(
        consistency_rows
    )

    print()
    print("Reproduction of Audit-03g strict cohort metrics")
    print(
        consistency
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 8. Cross-cohort agreement
    # ----------------------------------------------------------

    raw_cosine, raw_p = permutation_p(
        raw_equal_patient,
        gse,
    )

    directional_cosine, directional_p = permutation_p(
        directional_equal_patient,
        gse_unit,
    )

    cross_cohort = pd.DataFrame(
        [
            {
                "PRJEB24688_replication_vector":
                    "raw_equal_patient",

                "pearson_to_GSE119356":
                    pearson(
                        raw_equal_patient,
                        gse,
                    ),

                "cosine_to_GSE119356":
                    raw_cosine,

                "permutation_p":
                    raw_p,
            },

            {
                "PRJEB24688_replication_vector":
                    "directional_equal_patient",

                "pearson_to_GSE119356":
                    pearson(
                        directional_equal_patient,
                        gse_unit,
                    ),

                "cosine_to_GSE119356":
                    directional_cosine,

                "permutation_p":
                    directional_p,
            },
        ]
    )

    print()
    print("Frozen cross-cohort agreement")
    print(
        cross_cohort
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Two-cohort consensus direction
    #
    # Equal weight to:
    #
    #   cohort 1 = GSE119356
    #   cohort 2 = strict PRJEB24688
    #
    # Both are unit-normalized first.
    #
    # This vector is now FROZEN for testing in an independent
    # third cohort. It is not retroactively used to assess either
    # discovery or replication cohort.
    # ----------------------------------------------------------

    consensus = unit_vector(
        (
            gse_unit
            + directional_equal_patient
        )
        / 2.0
    )

    # ----------------------------------------------------------
    # 10. Jackknife stability of PRJEB24688 direction
    # ----------------------------------------------------------

    jackknife_rows = []

    for excluded_patient in PATIENTS:
        included = [
            patient
            for patient in PATIENTS
            if patient
            != excluded_patient
        ]

        included_units = patient_unit_matrix[
            included
        ]

        jackknife = unit_vector(
            included_units.mean(
                axis=1
            )
        )

        jackknife_rows.append(
            {
                "excluded_patient":
                    excluded_patient,

                "pearson_vs_full_PRJEB24688":
                    pearson(
                        jackknife,
                        directional_equal_patient,
                    ),

                "cosine_vs_full_PRJEB24688":
                    cosine(
                        jackknife,
                        directional_equal_patient,
                    ),

                "pearson_vs_GSE119356":
                    pearson(
                        jackknife,
                        gse_unit,
                    ),

                "cosine_vs_GSE119356":
                    cosine(
                        jackknife,
                        gse_unit,
                    ),
            }
        )

    jackknife = pd.DataFrame(
        jackknife_rows
    )

    print()
    print("PRJEB24688 patient-jackknife stability")
    print(
        jackknife
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 11. Programme-level replicated consensus
    #
    # Strongest possible directional criterion:
    #
    #  - GSE119356 has a non-zero direction
    #  - every one of 4 patients has same sign under EARLY
    #  - every one of 4 patients has same sign under LATE
    #  - all of those signs match GSE119356
    #
    # This yields a conservative programme registry.
    # ----------------------------------------------------------

    programme_rows = []

    for imodulon in programmes:
        reference_value = float(
            gse.loc[
                imodulon
            ]
        )

        early_values = np.array(
            [
                patient_baseline_matrix.loc[
                    imodulon,
                    patient
                    + "__EARLY",
                ]
                for patient in PATIENTS
            ],
            dtype=float,
        )

        late_values = np.array(
            [
                patient_baseline_matrix.loc[
                    imodulon,
                    patient
                    + "__LATE",
                ]
                for patient in PATIENTS
            ],
            dtype=float,
        )

        all_values = np.concatenate(
            [
                early_values,
                late_values,
            ]
        )

        reference_positive = (
            reference_value
            > 0
        )

        reference_negative = (
            reference_value
            < 0
        )

        all_positive = bool(
            np.all(
                all_values
                > 0
            )
        )

        all_negative = bool(
            np.all(
                all_values
                < 0
            )
        )

        replicated_same_direction = bool(
            (
                reference_positive
                and all_positive
            )
            or (
                reference_negative
                and all_negative
            )
        )

        direction = (
            "positive"
            if (
                reference_positive
                and all_positive
            )
            else (
                "negative"
                if (
                    reference_negative
                    and all_negative
                )
                else "not_consensus"
            )
        )

        minimum_abs_patient_baseline = float(
            np.min(
                np.abs(
                    all_values
                )
            )
        )

        programme_rows.append(
            {
                "imodulon":
                    imodulon,

                "GSE119356_activity":
                    reference_value,

                "PRJEB24688_mean_EARLY":
                    float(
                        np.mean(
                            early_values
                        )
                    ),

                "PRJEB24688_mean_LATE":
                    float(
                        np.mean(
                            late_values
                        )
                    ),

                "replicated_same_direction":
                    replicated_same_direction,

                "direction":
                    direction,

                "minimum_abs_PRJEB24688_patient_baseline":
                    minimum_abs_patient_baseline,

                "two_cohort_consensus_loading":
                    float(
                        consensus.loc[
                            imodulon
                        ]
                    ),
            }
        )

    programme_registry = pd.DataFrame(
        programme_rows
    )

    replicated_programmes = (
        programme_registry.loc[
            programme_registry[
                "replicated_same_direction"
            ]
        ]
        .sort_values(
            "minimum_abs_PRJEB24688_patient_baseline",
            ascending=False,
        )
    )

    print()
    print("Conservatively replicated programme directions")
    print(
        "  programmes:",
        len(
            replicated_programmes
        ),
        "/ 108",
    )

    print()
    print("Strongest replicated programme directions")
    print(
        replicated_programmes[
            [
                "imodulon",
                "direction",
                "GSE119356_activity",
                "PRJEB24688_mean_EARLY",
                "PRJEB24688_mean_LATE",
                "minimum_abs_PRJEB24688_patient_baseline",
                "two_cohort_consensus_loading",
            ]
        ]
        .head(
            30
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 12. Frozen vector table
    # ----------------------------------------------------------

    frozen_vectors = pd.DataFrame(
        {
            "GSE119356_discovery_raw":
                gse,

            "GSE119356_discovery_unit":
                gse_unit,

            "PRJEB24688_replication_raw_equal_patient":
                raw_equal_patient,

            "PRJEB24688_replication_unit":
                directional_equal_patient,

            "two_cohort_consensus_unit":
                consensus,
        }
    )

    # ----------------------------------------------------------
    # 13. Consistency checks
    #
    # This audit freezes already-established biology.
    # No new biological similarity threshold is introduced.
    # ----------------------------------------------------------

    max_metric_difference = float(
        max(
            consistency[
                "pearson_difference"
            ].max(),
            consistency[
                "cosine_difference"
            ].max(),
        )
    )

    minimum_jackknife_full_cosine = float(
        jackknife[
            "cosine_vs_full_PRJEB24688"
        ].min()
    )

    unit_norms = {
        "GSE119356_discovery_unit":
            float(
                np.linalg.norm(
                    gse_unit
                )
            ),

        "PRJEB24688_replication_unit":
            float(
                np.linalg.norm(
                    directional_equal_patient
                )
            ),

        "two_cohort_consensus_unit":
            float(
                np.linalg.norm(
                    consensus
                )
            ),
    }

    finite_outputs = bool(
        np.isfinite(
            frozen_vectors.to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            patient_summary.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy(
                dtype=float
            )
        ).all()
    )

    checks = {
        "108_programme_space":
            len(
                programmes
            )
            == 108,

        "7_strict_samples":
            len(
                strict_samples
            )
            == 7,

        "4_patients_retained":
            retained_patients
            == sorted(
                PATIENTS
            ),

        "8_patient_baseline_vectors":
            patient_baseline_matrix.shape[
                1
            ]
            == 8,

        "4_baseline_averaged_patient_vectors":
            patient_matrix.shape[
                1
            ]
            == 4,

        "reproduces_Audit03g_metrics":
            max_metric_difference
            < 1e-6,

        "all_frozen_unit_vectors_norm_1":
            all(
                abs(
                    value
                    - 1.0
                )
                < 1e-10
                for value in unit_norms.values()
            ),

        "all_4_jackknives_generated":
            len(
                jackknife
            )
            == 4,

        "PRJEB_direction_stable_to_patient_jackknife":
            minimum_jackknife_full_cosine
            >= 0.80,

        "finite_outputs":
            finite_outputs,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 03h freeze checks")

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
        "REPLICATED_CF_AIRWAY_PROGRAMME_CANDIDATE_FROZEN"
        if passed
        else "CF_AIRWAY_PROGRAMME_FREEZE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "INTERPRETATION: the frozen candidate is supported by "
        "one spatially sampled explanted CF lung and an independent "
        "multi-patient sputum cohort. It is not yet a universal CF "
        "programme. The two-cohort consensus vector is now fixed "
        "before testing any third cohort."
    )

    # ----------------------------------------------------------
    # 14. Save
    # ----------------------------------------------------------

    design.to_csv(
        OUT
        / "frozen_PRJEB24688_strict_patient_design.csv",
        index=False,
    )

    patient_baseline_matrix.to_csv(
        OUT
        / "strict_patient_baseline_residual_vectors.csv"
    )

    patient_matrix.to_csv(
        OUT
        / "strict_baseline_averaged_patient_residual_vectors.csv"
    )

    patient_summary.to_csv(
        OUT
        / "strict_patient_replication_summary.csv",
        index=False,
    )

    consistency.to_csv(
        OUT
        / "Audit03g_metric_reproduction.csv",
        index=False,
    )

    cross_cohort.to_csv(
        OUT
        / "frozen_cross_cohort_agreement.csv",
        index=False,
    )

    jackknife.to_csv(
        OUT
        / "PRJEB24688_patient_jackknife_stability.csv",
        index=False,
    )

    programme_registry.to_csv(
        OUT
        / "replicated_programme_registry_108.csv",
        index=False,
    )

    replicated_programmes.to_csv(
        OUT
        / "conservatively_replicated_programmes.csv",
        index=False,
    )

    frozen_vectors.to_csv(
        OUT
        / "FROZEN_CF_airway_candidate_vectors.csv"
    )

    pd.DataFrame(
        [
            {
                "vector":
                    key,
                "L2_norm":
                    value,
            }
            for key, value in unit_norms.items()
        ]
    ).to_csv(
        OUT
        / "frozen_vector_norms.csv",
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
            for key, value in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    (
        OUT
        / "AUDIT_03H_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    summary_lines = [
        "# CF-State Phase 4 — Replicated CF-airway programme candidate",
        "",
        "Status: `" + status + "`",
        "",
        "## Discovery cohort",
        "",
        "- GSE119356",
        "- one explanted CF lung",
        "- five spatial regions",
        "- common residual after removal of controlled physiological states",
        "",
        "## Independent replication cohort",
        "",
        "- PRJEB24688",
        "- four CF patients",
        "- seven strict high-quality biological sputum samples",
        "- replication survives EARLY and LATE laboratory reference choice",
        "",
        "## Frozen representation",
        "",
        "- 108 empirically soft-core-stable primary iModulons",
        "- GSE119356 discovery vector retained unchanged",
        "- PRJEB24688 equal-patient replication direction frozen",
        "- two-cohort equal-weight consensus direction frozen for third-cohort testing",
        "",
        "## Interpretation",
        "",
        "The recurrent direction is an independently replicated CF-airway programme candidate.",
        "It must not yet be described as universal or clinically validated.",
        "The next independent cohort must be evaluated against these frozen vectors without refitting them.",
        "",
    ]

    (
        OUT
        / "REPLICATED_CF_AIRWAY_CANDIDATE.md"
    ).write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    if not passed:
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed Audit 03h checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
