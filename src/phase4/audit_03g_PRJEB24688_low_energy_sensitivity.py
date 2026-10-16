from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

QC = (
    ROOT
    / "results/phase4/audit_03f_PRJEB24688_multipatient_CF_replication"
    / "PRJEB24688_sputum_projection_QC.csv"
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

BIO_COUNTS = (
    ROOT
    / "results/phase4/audit_03d_PRJEB24688_published_count_intake"
    / "PRJEB24688_PRIMARY_9_biological_sputum_counts.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_03g_PRJEB24688_low_energy_sensitivity"
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

ENERGY_THRESHOLD = 0.15
CORRELATION_THRESHOLD = 0.40


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

    if np.std(a) == 0 or np.std(b) == 0:
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
        np.dot(a, b)
        / denominator
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

    for i in range(n_perm):
        null[i] = cosine(
            a,
            rng.permutation(b),
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
    print("CF-State Phase 4 - Audit 03g")
    print("PRJEB24688 strict low-projection-energy sensitivity")
    print("=" * 78)

    for path in [
        QC,
        RESIDUALS,
        GSE_REFERENCE,
        BIO_COUNTS,
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
    # 1. Load original Audit-03f technical QC
    # ----------------------------------------------------------

    qc = pd.read_csv(
        QC
    )

    required_columns = [
        "sample",
        "patient",
        "baseline",
        "projection_corr",
        "projection_energy_fraction",
    ]

    for column in required_columns:
        if column not in qc.columns:
            raise RuntimeError(
                "QC table missing "
                + column
            )

    print()
    print("Original Audit-03f projection QC")
    print(
        qc[
            required_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 2. Freeze strict sample rule
    #
    # Keep a biological sample only when BOTH EARLY and LATE
    # contrasts satisfy the ORIGINAL 03f gates:
    #
    # projection correlation >= 0.40
    # projection energy      >= 0.15
    #
    # No thresholds are changed.
    # ----------------------------------------------------------

    sample_rows = []

    for sample, group in qc.groupby(
        "sample"
    ):
        baselines = set(
            group[
                "baseline"
            ]
        )

        if baselines != set(
            BASELINES
        ):
            raise RuntimeError(
                "Expected both baselines for "
                + sample
            )

        minimum_corr = float(
            group[
                "projection_corr"
            ].min()
        )

        minimum_energy = float(
            group[
                "projection_energy_fraction"
            ].min()
        )

        strict_pass = bool(
            minimum_corr
            >= CORRELATION_THRESHOLD
            and minimum_energy
            >= ENERGY_THRESHOLD
        )

        failed_reasons = []

        if minimum_corr < CORRELATION_THRESHOLD:
            failed_reasons.append(
                "projection_corr"
            )

        if minimum_energy < ENERGY_THRESHOLD:
            failed_reasons.append(
                "projection_energy"
            )

        sample_rows.append(
            {
                "sample":
                    sample,

                "patient":
                    patient_from_sample(
                        sample
                    ),

                "minimum_projection_corr":
                    minimum_corr,

                "minimum_projection_energy":
                    minimum_energy,

                "strict_both_baselines_pass":
                    strict_pass,

                "failure_reason":
                    ",".join(
                        failed_reasons
                    ),
            }
        )

    sample_filter = pd.DataFrame(
        sample_rows
    ).sort_values(
        [
            "strict_both_baselines_pass",
            "patient",
            "sample",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    )

    strict_samples = (
        sample_filter.loc[
            sample_filter[
                "strict_both_baselines_pass"
            ],
            "sample",
        ]
        .tolist()
    )

    excluded_samples = (
        sample_filter.loc[
            ~sample_filter[
                "strict_both_baselines_pass"
            ],
            "sample",
        ]
        .tolist()
    )

    print()
    print("Strict sample filter")
    print(
        sample_filter
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Strict samples:",
        len(strict_samples),
        strict_samples,
    )

    print(
        "Excluded samples:",
        len(excluded_samples),
        excluded_samples,
    )

    # ----------------------------------------------------------
    # 3. Depth diagnostic
    #
    # Descriptive only.
    # Do not create a new depth exclusion criterion.
    # ----------------------------------------------------------

    counts = pd.read_csv(
        BIO_COUNTS,
        index_col=0,
    )

    counts = counts.apply(
        pd.to_numeric,
        errors="raise",
    )

    depths = counts.sum(
        axis=0
    )

    depth_rows = []

    for sample in sample_filter[
        "sample"
    ]:
        depth_rows.append(
            {
                "sample":
                    sample,

                "patient":
                    patient_from_sample(
                        sample
                    ),

                "mapped_counts":
                    float(
                        depths[
                            sample
                        ]
                    ),

                "strict_both_baselines_pass":
                    bool(
                        sample_filter.loc[
                            sample_filter[
                                "sample"
                            ]
                            == sample,
                            "strict_both_baselines_pass",
                        ].iloc[0]
                    ),
            }
        )

    depth_qc = pd.DataFrame(
        depth_rows
    )

    print()
    print("Biological-sample mapped depth")
    print(
        depth_qc
        .sort_values(
            "mapped_counts"
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. Load residual programme vectors
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

    if residuals.shape[0] != 108:
        raise RuntimeError(
            "Expected 108 programme residual space."
        )

    reference_table = pd.read_csv(
        GSE_REFERENCE,
        index_col=0,
    )

    if reference_table.shape[1] != 1:
        raise RuntimeError(
            "Expected one GSE119356 reference vector."
        )

    gse_reference = pd.to_numeric(
        reference_table.iloc[:, 0],
        errors="raise",
    )

    gse_reference.index = (
        gse_reference.index
        .astype(str)
        .str.strip()
    )

    gse_reference = gse_reference.reindex(
        residuals.index
    )

    if gse_reference.isna().any():
        raise RuntimeError(
            "GSE119356 reference missing programmes."
        )

    # ----------------------------------------------------------
    # 5. Reconstruct ALL-sample and STRICT-sample patient means
    # ----------------------------------------------------------

    all_samples = sorted(
        set(
            qc[
                "sample"
            ]
        )
    )

    patient_vectors = {}

    patient_rows = []

    for analysis_set, samples in [
        (
            "ALL_9",
            all_samples,
        ),
        (
            "STRICT",
            strict_samples,
        ),
    ]:
        for baseline in BASELINES:
            for patient in PATIENTS:
                patient_samples = [
                    sample
                    for sample in samples
                    if patient_from_sample(
                        sample
                    )
                    == patient
                ]

                if not patient_samples:
                    continue

                columns = [
                    sample
                    + "__"
                    + baseline
                    for sample in patient_samples
                ]

                missing_columns = [
                    column
                    for column in columns
                    if column
                    not in residuals.columns
                ]

                if missing_columns:
                    raise RuntimeError(
                        "Missing residual columns: "
                        + ", ".join(
                            missing_columns
                        )
                    )

                vector = residuals[
                    columns
                ].mean(
                    axis=1
                )

                key = (
                    analysis_set
                    + "__"
                    + patient
                    + "__"
                    + baseline
                )

                patient_vectors[
                    key
                ] = vector

                observed_cosine, p = permutation_p(
                    vector,
                    gse_reference,
                )

                patient_rows.append(
                    {
                        "analysis_set":
                            analysis_set,

                        "patient":
                            patient,

                        "baseline":
                            baseline,

                        "n_samples":
                            len(
                                patient_samples
                            ),

                        "pearson_to_GSE119356":
                            pearson(
                                vector,
                                gse_reference,
                            ),

                        "cosine_to_GSE119356":
                            observed_cosine,

                        "permutation_p":
                            p,
                    }
                )

    patient_results = pd.DataFrame(
        patient_rows
    )

    print()
    print("Patient-level ALL versus STRICT sensitivity")
    print(
        patient_results
        .round(4)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Equal-patient cohort consensus
    # ----------------------------------------------------------

    cohort_vectors = {}

    cohort_rows = []

    for analysis_set in [
        "ALL_9",
        "STRICT",
    ]:
        for baseline in BASELINES:
            keys = [
                analysis_set
                + "__"
                + patient
                + "__"
                + baseline
                for patient in PATIENTS
            ]

            available = [
                key
                for key in keys
                if key
                in patient_vectors
            ]

            if len(
                available
            ) != 4:
                raise RuntimeError(
                    analysis_set
                    + " "
                    + baseline
                    + " does not retain all four patients."
                )

            matrix = pd.DataFrame(
                {
                    key:
                        patient_vectors[
                            key
                        ]
                    for key in available
                }
            )

            cohort = matrix.mean(
                axis=1
            )

            cohort_key = (
                analysis_set
                + "__"
                + baseline
            )

            cohort_vectors[
                cohort_key
            ] = cohort

            observed_cosine, p = permutation_p(
                cohort,
                gse_reference,
            )

            cohort_rows.append(
                {
                    "analysis_set":
                        analysis_set,

                    "baseline":
                        baseline,

                    "n_patients":
                        len(
                            available
                        ),

                    "pearson_to_GSE119356":
                        pearson(
                            cohort,
                            gse_reference,
                        ),

                    "cosine_to_GSE119356":
                        observed_cosine,

                    "permutation_p":
                        p,
                }
            )

    cohort_results = pd.DataFrame(
        cohort_rows
    )

    print()
    print("Equal-patient cohort sensitivity")
    print(
        cohort_results
        .round(4)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Does exclusion materially change the cohort direction?
    # ----------------------------------------------------------

    direction_rows = []

    for baseline in BASELINES:
        all_vector = cohort_vectors[
            "ALL_9__"
            + baseline
        ]

        strict_vector = cohort_vectors[
            "STRICT__"
            + baseline
        ]

        direction_rows.append(
            {
                "baseline":
                    baseline,

                "pearson_ALL_vs_STRICT":
                    pearson(
                        all_vector,
                        strict_vector,
                    ),

                "cosine_ALL_vs_STRICT":
                    cosine(
                        all_vector,
                        strict_vector,
                    ),
            }
        )

    direction_stability = pd.DataFrame(
        direction_rows
    )

    print()
    print("ALL versus STRICT cohort-direction stability")
    print(
        direction_stability
        .round(4)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 8. STRICT residual baseline robustness
    # ----------------------------------------------------------

    strict_baseline_rows = []

    for patient in PATIENTS:
        early = patient_vectors[
            "STRICT__"
            + patient
            + "__EARLY"
        ]

        late = patient_vectors[
            "STRICT__"
            + patient
            + "__LATE"
        ]

        strict_baseline_rows.append(
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
            }
        )

    strict_baseline = pd.DataFrame(
        strict_baseline_rows
    )

    print()
    print("STRICT patient residual robustness to baseline")
    print(
        strict_baseline
        .round(4)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. STRICT leave-one-patient-out internal replication
    # ----------------------------------------------------------

    loo_rows = []

    for baseline in BASELINES:
        for held_out in PATIENTS:
            training = [
                patient
                for patient in PATIENTS
                if patient
                != held_out
            ]

            training_vectors = pd.DataFrame(
                {
                    patient:
                        patient_vectors[
                            "STRICT__"
                            + patient
                            + "__"
                            + baseline
                        ]
                    for patient in training
                }
            )

            training_common = training_vectors.mean(
                axis=1
            )

            held = patient_vectors[
                "STRICT__"
                + held_out
                + "__"
                + baseline
            ]

            loo_rows.append(
                {
                    "baseline":
                        baseline,

                    "held_out_patient":
                        held_out,

                    "pearson_to_other_three":
                        pearson(
                            held,
                            training_common,
                        ),

                    "cosine_to_other_three":
                        cosine(
                            held,
                            training_common,
                        ),
                }
            )

    loo = pd.DataFrame(
        loo_rows
    )

    print()
    print("STRICT leave-one-patient-out internal agreement")
    print(
        loo
        .round(4)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 10. STRICT cross-patient programme consensus
    # ----------------------------------------------------------

    consensus_rows = []

    for imodulon in residuals.index:
        early_values = np.array(
            [
                patient_vectors[
                    "STRICT__"
                    + patient
                    + "__EARLY"
                ].loc[
                    imodulon
                ]
                for patient in PATIENTS
            ],
            dtype=float,
        )

        late_values = np.array(
            [
                patient_vectors[
                    "STRICT__"
                    + patient
                    + "__LATE"
                ].loc[
                    imodulon
                ]
                for patient in PATIENTS
            ],
            dtype=float,
        )

        early_positive = bool(
            np.all(
                early_values
                > 0
            )
        )

        early_negative = bool(
            np.all(
                early_values
                < 0
            )
        )

        late_positive = bool(
            np.all(
                late_values
                > 0
            )
        )

        late_negative = bool(
            np.all(
                late_values
                < 0
            )
        )

        same_positive = (
            early_positive
            and late_positive
        )

        same_negative = (
            early_negative
            and late_negative
        )

        consensus_both = (
            same_positive
            or same_negative
        )

        direction = (
            "positive"
            if same_positive
            else (
                "negative"
                if same_negative
                else "mixed"
            )
        )

        reference_value = float(
            gse_reference.loc[
                imodulon
            ]
        )

        matches_reference = bool(
            (
                direction
                == "positive"
                and reference_value
                > 0
            )
            or (
                direction
                == "negative"
                and reference_value
                < 0
            )
        )

        consensus_rows.append(
            {
                "imodulon":
                    imodulon,

                "same_direction_all_4_patients_both_baselines":
                    consensus_both,

                "direction":
                    direction,

                "mean_EARLY":
                    float(
                        np.mean(
                            early_values
                        )
                    ),

                "mean_LATE":
                    float(
                        np.mean(
                            late_values
                        )
                    ),

                "minimum_abs_EARLY":
                    float(
                        np.min(
                            np.abs(
                                early_values
                            )
                        )
                    ),

                "minimum_abs_LATE":
                    float(
                        np.min(
                            np.abs(
                                late_values
                            )
                        )
                    ),

                "GSE119356_reference_activity":
                    reference_value,

                "matches_GSE119356_direction":
                    matches_reference,
            }
        )

    consensus = pd.DataFrame(
        consensus_rows
    )

    consensus[
        "minimum_abs_across_baselines"
    ] = consensus[
        [
            "minimum_abs_EARLY",
            "minimum_abs_LATE",
        ]
    ].min(
        axis=1
    )

    stable_consensus = (
        consensus.loc[
            consensus[
                "same_direction_all_4_patients_both_baselines"
            ]
        ]
        .sort_values(
            "minimum_abs_across_baselines",
            ascending=False,
        )
    )

    print()
    print(
        "STRICT programmes with same direction in all four patients "
        "under both baselines"
    )

    print(
        stable_consensus[
            [
                "imodulon",
                "direction",
                "mean_EARLY",
                "mean_LATE",
                "minimum_abs_across_baselines",
                "GSE119356_reference_activity",
                "matches_GSE119356_direction",
            ]
        ]
        .head(30)
        .round(4)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 11. Descriptive biological-replication summary
    #
    # This is NOT used as the technical audit PASS.
    # ----------------------------------------------------------

    strict_cohort = cohort_results.loc[
        cohort_results[
            "analysis_set"
        ]
        == "STRICT"
    ].copy()

    strict_patient = patient_results.loc[
        patient_results[
            "analysis_set"
        ]
        == "STRICT"
    ].copy()

    cohort_positive_significant = bool(
        (
            (
                strict_cohort[
                    "cosine_to_GSE119356"
                ]
                > 0
            )
            & (
                strict_cohort[
                    "permutation_p"
                ]
                <= 0.01
            )
        ).all()
    )

    all_patients_positive = bool(
        (
            strict_patient[
                "cosine_to_GSE119356"
            ]
            > 0
        ).all()
    )

    all_patients_significant_005 = bool(
        (
            strict_patient[
                "permutation_p"
            ]
            <= 0.05
        ).all()
    )

    replication_supported = bool(
        cohort_positive_significant
        and all_patients_positive
        and all_patients_significant_005
    )

    print()
    print("Strict-sensitivity biological replication summary")
    print(
        "  cohort positive and p<=0.01 under both baselines:",
        cohort_positive_significant,
    )
    print(
        "  all 4 patients positive under both baselines:",
        all_patients_positive,
    )
    print(
        "  all 4 patients p<=0.05 under both baselines:",
        all_patients_significant_005,
    )
    print(
        "  replication supported in strict sensitivity:",
        replication_supported,
    )

    # ----------------------------------------------------------
    # 12. Technical audit checks
    #
    # No biological similarity threshold is required for PASS.
    # ----------------------------------------------------------

    retained_patients = sorted(
        {
            patient_from_sample(
                sample
            )
            for sample in strict_samples
        }
    )

    strict_qc_rows = qc.loc[
        qc[
            "sample"
        ].isin(
            strict_samples
        )
    ]

    strict_original_gates_pass = bool(
        (
            (
                strict_qc_rows[
                    "projection_corr"
                ]
                >= CORRELATION_THRESHOLD
            )
            & (
                strict_qc_rows[
                    "projection_energy_fraction"
                ]
                >= ENERGY_THRESHOLD
            )
        ).all()
    )

    finite_outputs = bool(
        np.isfinite(
            patient_results.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            cohort_results.select_dtypes(
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
    )

    checks = {
        "original_18_contrasts_present":
            len(
                qc
            )
            == 18,

        "strict_rule_uses_original_thresholds":
            (
                ENERGY_THRESHOLD
                == 0.15
                and CORRELATION_THRESHOLD
                == 0.40
            ),

        "strict_subset_retains_all_4_patients":
            retained_patients
            == sorted(
                PATIENTS
            ),

        "strict_subset_retains_at_least_1_sample_per_patient":
            all(
                any(
                    patient_from_sample(
                        sample
                    )
                    == patient
                    for sample in strict_samples
                )
                for patient in PATIENTS
            ),

        "all_strict_contrasts_pass_original_gates":
            strict_original_gates_pass,

        "both_baselines_recomputed":
            set(
                strict_cohort[
                    "baseline"
                ]
            )
            == set(
                BASELINES
            ),

        "finite_outputs":
            finite_outputs,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 03g technical checks")

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
        "PRJEB24688_LOW_ENERGY_SENSITIVITY_GENERATED"
        if passed
        else "PRJEB24688_LOW_ENERGY_SENSITIVITY_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: Audit 03f remains formally REVIEW_REQUIRED. "
        "Audit 03g does not lower or waive its projection-energy gate."
    )

    if replication_supported:
        print()
        print(
            "BIOLOGICAL RESULT: the GSE119356 residual replication "
            "is supported after excluding every biological sample "
            "that fails the original Audit-03f projection criteria "
            "under either laboratory baseline."
        )
    else:
        print()
        print(
            "BIOLOGICAL RESULT: strict sensitivity does not support "
            "calling the GSE119356 residual replicated across all "
            "patients and both baselines."
        )

    # ----------------------------------------------------------
    # 13. Save
    # ----------------------------------------------------------

    sample_filter.to_csv(
        OUT
        / "strict_sample_filter.csv",
        index=False,
    )

    depth_qc.to_csv(
        OUT
        / "strict_sample_depth_diagnostic.csv",
        index=False,
    )

    patient_results.to_csv(
        OUT
        / "ALL_vs_STRICT_patient_replication.csv",
        index=False,
    )

    cohort_results.to_csv(
        OUT
        / "ALL_vs_STRICT_cohort_replication.csv",
        index=False,
    )

    direction_stability.to_csv(
        OUT
        / "ALL_vs_STRICT_cohort_direction_stability.csv",
        index=False,
    )

    strict_baseline.to_csv(
        OUT
        / "STRICT_patient_baseline_robustness.csv",
        index=False,
    )

    loo.to_csv(
        OUT
        / "STRICT_leave_one_patient_out_replication.csv",
        index=False,
    )

    consensus.to_csv(
        OUT
        / "STRICT_cross_patient_programme_consensus.csv",
        index=False,
    )

    stable_consensus.to_csv(
        OUT
        / "STRICT_stable_programmes_both_baselines.csv",
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
        / "AUDIT_03G_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    biological_status = (
        "STRICT_SENSITIVITY_SUPPORTS_MULTIPATIENT_REPLICATION"
        if replication_supported
        else "STRICT_SENSITIVITY_DOES_NOT_SUPPORT_FULL_REPLICATION"
    )

    (
        OUT
        / "BIOLOGICAL_REPLICATION_STATUS.txt"
    ).write_text(
        biological_status
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
            "Failed Audit 03g checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
