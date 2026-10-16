from pathlib import Path
from io import StringIO
import json
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

SPUTUM_COUNTS = (
    ROOT
    / "results/phase4/audit_03d_PRJEB24688_published_count_intake"
    / "PRJEB24688_PRIMARY_9_biological_sputum_counts.csv"
)

LAB_COUNTS = (
    ROOT
    / "results/phase4/audit_03d_PRJEB24688_published_count_intake"
    / "PRJEB24688_patient_matched_invitro_counts.csv"
)

REPLICATION_MAP = (
    ROOT
    / "results/phase4/audit_03d_PRJEB24688_published_count_intake"
    / "PRJEB24688_frozen_replication_gene_map.csv"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411"
    / "paeru_116_iModulon.json"
)

CALIBRATION = (
    ROOT
    / "results/phase4/audit_03e_softcore_projection_fidelity"
    / "softcore_to_full_activity_calibration.csv"
)

STABLE_PRIMARY = (
    ROOT
    / "results/phase4/audit_03e_softcore_projection_fidelity"
    / "PRJEB24688_empirically_stable_primary_programmes.csv"
)

GSE119356_REGIONAL = (
    ROOT
    / "results/phase4/audit_02a_GSE119356_matched_projection_QC"
    / "matched_region_imodulon_contrasts.csv"
)

STATE_LIBRARY = (
    ROOT
    / "results/phase4/audit_02b_GSE119356_CF_state_interpretation"
    / "controlled_state_reference_library.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_03f_PRJEB24688_multipatient_CF_replication"
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


def json_df(value):
    if isinstance(value, str):
        return pd.read_json(
            StringIO(value)
        )

    if isinstance(value, dict):
        return pd.DataFrame(
            value
        )

    raise RuntimeError(
        "Unexpected serialized model DataFrame."
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

    return (
        observed,
        float(
            p
        ),
    )


def patient_from_sputum(sample):
    match = re.match(
        r"^(P\d+[MF]\d+)_",
        str(
            sample
        ),
    )

    if match is None:
        raise RuntimeError(
            "Could not parse sputum patient from "
            + str(
                sample
            )
        )

    return match.group(
        1
    )


def parse_lab_sample(sample):
    text = str(
        sample
    )

    match = re.match(
        r"^PA-(P\d+[MF]\d+[AB]?)_(3H|24H)_R\d+$",
        text,
    )

    if match is None:
        raise RuntimeError(
            "Could not parse lab sample "
            + text
        )

    isolate = match.group(
        1
    )

    condition = match.group(
        2
    )

    patient_match = re.match(
        r"^(P\d+[MF]\d+)",
        isolate,
    )

    if patient_match is None:
        raise RuntimeError(
            "Could not parse patient from isolate "
            + isolate
        )

    patient = patient_match.group(
        1
    )

    baseline = (
        "EARLY"
        if condition
        == "3H"
        else "LATE"
    )

    return {
        "patient":
            patient,

        "isolate":
            isolate,

        "condition":
            condition,

        "baseline":
            baseline,
    }


def residualize(
    vector,
    reference_matrix,
):
    y = np.asarray(
        vector,
        dtype=float,
    )

    B = np.asarray(
        reference_matrix,
        dtype=float,
    )

    norms = np.linalg.norm(
        B,
        axis=0,
    )

    if (
        norms <= 0
    ).any():
        raise RuntimeError(
            "Zero-norm controlled-state reference."
        )

    B = (
        B
        / norms
    )

    beta = (
        np.linalg.lstsq(
            B,
            y,
            rcond=None,
        )[
            0
        ]
    )

    reconstruction = (
        B
        @ beta
    )

    residual = (
        y
        - reconstruction
    )

    total_energy = float(
        np.dot(
            y,
            y,
        )
    )

    explained = float(
        np.dot(
            reconstruction,
            reconstruction,
        )
        / total_energy
    )

    residual_fraction = float(
        np.dot(
            residual,
            residual,
        )
        / total_energy
    )

    return (
        reconstruction,
        residual,
        explained,
        residual_fraction,
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 03f")
    print("PRJEB24688 multi-patient replication of CF-lung programme")
    print("=" * 78)

    required = [
        SPUTUM_COUNTS,
        LAB_COUNTS,
        REPLICATION_MAP,
        MODEL,
        CALIBRATION,
        STABLE_PRIMARY,
        GSE119356_REGIONAL,
        STATE_LIBRARY,
    ]

    for path in required:
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
    # 1. Model
    # ----------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(
            handle
        )

    M = json_df(
        raw[
            "M"
        ]
    )

    M.index = (
        M.index
        .astype(str)
        .str.strip()
    )

    M.columns = (
        M.columns
        .astype(str)
        .str.strip()
    )

    imodulons = list(
        M.columns
    )

    # ----------------------------------------------------------
    # 2. Replication gene universe + gene lengths
    # ----------------------------------------------------------

    replication_map = pd.read_csv(
        REPLICATION_MAP
    )

    required_map_columns = [
        "pao1_gene",
        "length_bp",
    ]

    for column in required_map_columns:
        if column not in replication_map.columns:
            raise RuntimeError(
                "Replication map missing "
                + column
            )

    replication_map[
        "pao1_gene"
    ] = (
        replication_map[
            "pao1_gene"
        ]
        .astype(str)
        .str.strip()
    )

    replication_map[
        "length_bp"
    ] = pd.to_numeric(
        replication_map[
            "length_bp"
        ],
        errors="raise",
    )

    if (
        replication_map[
            "length_bp"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "Invalid gene lengths."
        )

    genes = list(
        replication_map[
            "pao1_gene"
        ]
    )

    if len(
        genes
    ) != 4925:
        raise RuntimeError(
            "Expected 4925 replication genes."
        )

    lengths = (
        replication_map
        .set_index(
            "pao1_gene"
        )[
            "length_bp"
        ]
        .reindex(
            genes
        )
    )

    # ----------------------------------------------------------
    # 3. Published counts
    # ----------------------------------------------------------

    sputum = pd.read_csv(
        SPUTUM_COUNTS,
        index_col=0,
    )

    lab = pd.read_csv(
        LAB_COUNTS,
        index_col=0,
    )

    for table in [
        sputum,
        lab,
    ]:
        table.index = (
            table.index
            .astype(str)
            .str.strip()
        )

    sputum = sputum.reindex(
        genes
    )

    lab = lab.reindex(
        genes
    )

    if sputum.isna().any().any():
        raise RuntimeError(
            "Sputum count matrix missing replication genes."
        )

    if lab.isna().any().any():
        raise RuntimeError(
            "Lab count matrix missing replication genes."
        )

    sputum = sputum.apply(
        pd.to_numeric,
        errors="raise",
    )

    lab = lab.apply(
        pd.to_numeric,
        errors="raise",
    )

    print()
    print("Published count matrices")
    print(
        "  replication genes:",
        len(
            genes
        ),
    )
    print(
        "  sputum biological samples:",
        sputum.shape[
            1
        ],
    )
    print(
        "  matched laboratory runs:",
        lab.shape[
            1
        ],
    )

    # ----------------------------------------------------------
    # 4. TPM normalization
    #
    # Counts were generated against PA14.
    # length_bp comes from our frozen PA14 projection map.
    # ----------------------------------------------------------

    def counts_to_log_tpm(counts):
        rpk = counts.divide(
            lengths
            / 1000.0,
            axis=0,
        )

        denominator = rpk.sum(
            axis=0
        )

        if (
            denominator <= 0
        ).any():
            raise RuntimeError(
                "Invalid TPM denominator."
            )

        tpm = rpk.divide(
            denominator,
            axis=1,
        ) * 1_000_000.0

        log_tpm = np.log2(
            tpm
            + 1.0
        )

        return (
            tpm,
            log_tpm,
        )

    sputum_tpm, sputum_log = counts_to_log_tpm(
        sputum
    )

    lab_tpm, lab_log = counts_to_log_tpm(
        lab
    )

    print()
    print("TPM normalization")
    print(
        "  sputum TPM sums:",
        round(
            float(
                sputum_tpm.sum(
                    axis=0
                ).min()
            ),
            3,
        ),
        "to",
        round(
            float(
                sputum_tpm.sum(
                    axis=0
                ).max()
            ),
            3,
        ),
    )
    print(
        "  lab TPM sums:",
        round(
            float(
                lab_tpm.sum(
                    axis=0
                ).min()
            ),
            3,
        ),
        "to",
        round(
            float(
                lab_tpm.sum(
                    axis=0
                ).max()
            ),
            3,
        ),
    )

    # ----------------------------------------------------------
    # 5. Freeze patient/isolate/baseline laboratory references
    #
    # First average biological replicate runs within isolate.
    # Then average isolates equally within patient.
    #
    # This matters for P24M1, which has two isolates and unequal
    # numbers of 24H runs.
    # ----------------------------------------------------------

    lab_metadata_rows = []

    for sample in lab_log.columns:
        parsed = parse_lab_sample(
            sample
        )

        lab_metadata_rows.append(
            {
                "sample":
                    sample,

                **parsed,
            }
        )

    lab_metadata = pd.DataFrame(
        lab_metadata_rows
    )

    isolate_reference_vectors = {}
    isolate_reference_rows = []

    for (
        patient,
        isolate,
        baseline,
    ), group in lab_metadata.groupby(
        [
            "patient",
            "isolate",
            "baseline",
        ]
    ):
        samples = group[
            "sample"
        ].tolist()

        vector = lab_log[
            samples
        ].mean(
            axis=1
        )

        key = (
            patient,
            isolate,
            baseline,
        )

        isolate_reference_vectors[
            key
        ] = vector

        isolate_reference_rows.append(
            {
                "patient":
                    patient,

                "isolate":
                    isolate,

                "baseline":
                    baseline,

                "n_runs":
                    len(
                        samples
                    ),

                "runs":
                    ",".join(
                        samples
                    ),
            }
        )

    isolate_reference_info = pd.DataFrame(
        isolate_reference_rows
    )

    patient_reference_vectors = {}
    patient_reference_rows = []

    for patient in PATIENTS:
        for baseline in BASELINES:
            keys = [
                key
                for key in isolate_reference_vectors
                if key[
                    0
                ]
                == patient
                and key[
                    2
                ]
                == baseline
            ]

            if not keys:
                raise RuntimeError(
                    "Missing "
                    + baseline
                    + " lab reference for "
                    + patient
                )

            isolate_vectors = pd.DataFrame(
                {
                    key[
                        1
                    ]:
                        isolate_reference_vectors[
                            key
                        ]
                    for key in keys
                }
            )

            patient_reference = isolate_vectors.mean(
                axis=1
            )

            patient_reference_vectors[
                (
                    patient,
                    baseline,
                )
            ] = patient_reference

            patient_reference_rows.append(
                {
                    "patient":
                        patient,

                    "baseline":
                        baseline,

                    "n_isolates":
                        len(
                            keys
                        ),

                    "isolates":
                        ",".join(
                            sorted(
                                [
                                    key[
                                        1
                                    ]
                                    for key in keys
                                ]
                            )
                        ),
                }
            )

    patient_reference_info = pd.DataFrame(
        patient_reference_rows
    )

    print()
    print("Patient laboratory references")
    print(
        patient_reference_info.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Soft-core projection operator
    # ----------------------------------------------------------

    M_mask = M.reindex(
        genes
    )

    if M_mask.isna().any().any():
        raise RuntimeError(
            "Masked M missing replication genes."
        )

    M_array = M_mask.to_numpy(
        dtype=float
    )

    pinv_M = np.linalg.pinv(
        M_array
    )

    # ----------------------------------------------------------
    # 7. Load empirical calibration
    #
    # For a contrast:
    #
    #   A_full(delta X) ~= slope * A_mask(delta X)
    #
    # Calibration intercept cancels from a difference.
    # ----------------------------------------------------------

    calibration = pd.read_csv(
        CALIBRATION
    )

    calibration[
        "imodulon"
    ] = (
        calibration[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    slopes = (
        calibration
        .set_index(
            "imodulon"
        )[
            "calibration_slope"
        ]
        .reindex(
            imodulons
        )
    )

    if slopes.isna().any():
        raise RuntimeError(
            "Missing calibration slopes."
        )

    # ----------------------------------------------------------
    # 8. Frozen mask-compatible primary programme set
    # ----------------------------------------------------------

    stable = pd.read_csv(
        STABLE_PRIMARY
    )

    stable[
        "imodulon"
    ] = (
        stable[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    stable_programmes = [
        imodulon
        for imodulon in imodulons
        if imodulon
        in set(
            stable[
                "imodulon"
            ]
        )
    ]

    if len(
        stable_programmes
    ) != 108:
        raise RuntimeError(
            "Expected 108 empirically stable primary programmes."
        )

    excluded_primary = [
        imodulon
        for imodulon in [
            "XisF4-2",
            "Sigma-70",
            "MexZ",
        ]
        if imodulon
        not in stable_programmes
    ]

    print()
    print("Soft-core-compatible interpretation set")
    print(
        "  stable primary programmes:",
        len(
            stable_programmes
        ),
    )
    print(
        "  excluded unstable primary programmes:",
        ", ".join(
            excluded_primary
        ),
    )

    # ----------------------------------------------------------
    # 9. Controlled-state library restricted to same 108-space
    # ----------------------------------------------------------

    state_library = pd.read_csv(
        STATE_LIBRARY,
        index_col=0,
    )

    state_library.index = (
        state_library.index
        .astype(str)
        .str.strip()
    )

    state_library = state_library.apply(
        pd.to_numeric,
        errors="raise",
    )

    state_library = state_library.reindex(
        stable_programmes
    )

    if state_library.isna().any().any():
        raise RuntimeError(
            "Controlled-state library missing stable programmes."
        )

    # ----------------------------------------------------------
    # 10. Freeze GSE119356 reference residual in the exact same
    #     108-dimensional space.
    #
    # This is computed without looking at PRJEB24688 biology.
    # ----------------------------------------------------------

    gse_regional = pd.read_csv(
        GSE119356_REGIONAL,
        index_col=0,
    )

    gse_regional.index = (
        gse_regional.index
        .astype(str)
        .str.strip()
    )

    gse_regional = gse_regional.apply(
        pd.to_numeric,
        errors="raise",
    )

    gse_pooled = (
        gse_regional.mean(
            axis=1
        )
        .reindex(
            stable_programmes
        )
    )

    (
        gse_reconstruction,
        gse_common_residual,
        gse_known_fraction,
        gse_residual_fraction,
    ) = residualize(
        gse_pooled.to_numpy(
            dtype=float
        ),
        state_library.to_numpy(
            dtype=float
        ),
    )

    gse_residual = pd.Series(
        gse_common_residual,
        index=stable_programmes,
        name="GSE119356_common_residual_108",
    )

    print()
    print("Frozen mask-compatible GSE119356 reference")
    print(
        "  known-state energy fraction:",
        round(
            gse_known_fraction,
            4,
        ),
    )
    print(
        "  residual energy fraction:",
        round(
            gse_residual_fraction,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 11. Project all 9 sputum samples against BOTH patient
    #     reference states.
    # ----------------------------------------------------------

    activity_vectors = {}
    residual_vectors = {}

    projection_rows = []
    similarity_rows = []

    for sputum_sample in sputum_log.columns:
        patient = patient_from_sputum(
            sputum_sample
        )

        if patient not in PATIENTS:
            raise RuntimeError(
                "Unexpected patient "
                + patient
            )

        for baseline in BASELINES:
            reference = patient_reference_vectors[
                (
                    patient,
                    baseline,
                )
            ]

            delta_x = (
                sputum_log[
                    sputum_sample
                ]
                - reference
            )

            delta_array = delta_x.to_numpy(
                dtype=float
            )

            activity_masked = (
                pinv_M
                @ delta_array
            )

            reconstruction = (
                M_array
                @ activity_masked
            )

            projection_corr = pearson(
                delta_array,
                reconstruction,
            )

            projection_energy = float(
                np.dot(
                    reconstruction,
                    reconstruction,
                )
                / np.dot(
                    delta_array,
                    delta_array,
                )
            )

            projection_error = float(
                np.linalg.norm(
                    delta_array
                    - reconstruction
                )
                / np.linalg.norm(
                    delta_array
                )
            )

            activity_calibrated = (
                activity_masked
                * slopes.to_numpy(
                    dtype=float
                )
            )

            activity_series = pd.Series(
                activity_calibrated,
                index=imodulons,
            )

            stable_activity = activity_series.reindex(
                stable_programmes
            )

            (
                state_reconstruction,
                cf_residual,
                known_fraction,
                residual_fraction,
            ) = residualize(
                stable_activity.to_numpy(
                    dtype=float
                ),
                state_library.to_numpy(
                    dtype=float
                ),
            )

            key = (
                sputum_sample
                + "__"
                + baseline
            )

            activity_vectors[
                key
            ] = stable_activity

            residual_series = pd.Series(
                cf_residual,
                index=stable_programmes,
            )

            residual_vectors[
                key
            ] = residual_series

            residual_cosine, residual_p = permutation_p(
                residual_series,
                gse_residual,
            )

            projection_rows.append(
                {
                    "sample":
                        sputum_sample,

                    "patient":
                        patient,

                    "baseline":
                        baseline,

                    "projection_corr":
                        projection_corr,

                    "projection_energy_fraction":
                        projection_energy,

                    "relative_projection_error":
                        projection_error,

                    "known_state_energy_fraction_108":
                        known_fraction,

                    "residual_energy_fraction_108":
                        residual_fraction,
                }
            )

            similarity_rows.append(
                {
                    "sample":
                        sputum_sample,

                    "patient":
                        patient,

                    "baseline":
                        baseline,

                    "pearson_to_GSE119356_common_residual":
                        pearson(
                            residual_series,
                            gse_residual,
                        ),

                    "cosine_to_GSE119356_common_residual":
                        residual_cosine,

                    "permutation_p":
                        residual_p,
                }
            )

    projection_qc = pd.DataFrame(
        projection_rows
    )

    sample_similarity = pd.DataFrame(
        similarity_rows
    )

    activity_matrix = pd.DataFrame(
        activity_vectors
    )

    residual_matrix = pd.DataFrame(
        residual_vectors
    )

    print()
    print("PRJEB24688 sputum projection QC")
    print(
        projection_qc
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Sample-level replication of GSE119356 common residual")
    print(
        sample_similarity
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 12. Baseline robustness
    #
    # Same sputum sample, different matched laboratory baseline.
    # ----------------------------------------------------------

    baseline_rows = []

    for sputum_sample in sputum_log.columns:
        early = residual_matrix[
            sputum_sample
            + "__EARLY"
        ]

        late = residual_matrix[
            sputum_sample
            + "__LATE"
        ]

        baseline_rows.append(
            {
                "sample":
                    sputum_sample,

                "patient":
                    patient_from_sputum(
                        sputum_sample
                    ),

                "pearson_EARLY_vs_LATE_residual":
                    pearson(
                        early,
                        late,
                    ),

                "cosine_EARLY_vs_LATE_residual":
                    cosine(
                        early,
                        late,
                    ),
            }
        )

    baseline_robustness = pd.DataFrame(
        baseline_rows
    )

    print()
    print("Residual robustness to laboratory baseline choice")
    print(
        baseline_robustness
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 13. Patient-level residuals
    #
    # Equal weight per biological sputum sample within patient.
    # Then equal weight per patient for cohort consensus.
    # ----------------------------------------------------------

    patient_residual_vectors = {}
    patient_rows = []

    for baseline in BASELINES:
        for patient in PATIENTS:
            sample_names = [
                sample
                for sample in sputum_log.columns
                if patient_from_sputum(
                    sample
                )
                == patient
            ]

            keys = [
                sample
                + "__"
                + baseline
                for sample in sample_names
            ]

            patient_residual = residual_matrix[
                keys
            ].mean(
                axis=1
            )

            key = (
                patient
                + "__"
                + baseline
            )

            patient_residual_vectors[
                key
            ] = patient_residual

            observed_cosine, p = permutation_p(
                patient_residual,
                gse_residual,
            )

            patient_rows.append(
                {
                    "patient":
                        patient,

                    "baseline":
                        baseline,

                    "n_sputum_samples":
                        len(
                            sample_names
                        ),

                    "pearson_to_GSE119356":
                        pearson(
                            patient_residual,
                            gse_residual,
                        ),

                    "cosine_to_GSE119356":
                        observed_cosine,

                    "permutation_p":
                        p,
                }
            )

    patient_residual_matrix = pd.DataFrame(
        patient_residual_vectors
    )

    patient_similarity = pd.DataFrame(
        patient_rows
    )

    print()
    print("Patient-level replication of GSE119356 common residual")
    print(
        patient_similarity
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 14. Equal-patient cohort consensus
    # ----------------------------------------------------------

    cohort_rows = []

    cohort_vectors = {}

    for baseline in BASELINES:
        keys = [
            patient
            + "__"
            + baseline
            for patient in PATIENTS
        ]

        cohort = patient_residual_matrix[
            keys
        ].mean(
            axis=1
        )

        cohort_vectors[
            baseline
        ] = cohort

        observed_cosine, p = permutation_p(
            cohort,
            gse_residual,
        )

        cohort_rows.append(
            {
                "baseline":
                    baseline,

                "pearson_to_GSE119356":
                    pearson(
                        cohort,
                        gse_residual,
                    ),

                "cosine_to_GSE119356":
                    observed_cosine,

                "permutation_p":
                    p,
            }
        )

    cohort_similarity = pd.DataFrame(
        cohort_rows
    )

    print()
    print("Equal-patient PRJEB24688 cohort consensus vs GSE119356")
    print(
        cohort_similarity
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Cohort residual baseline robustness")
    print(
        "  Pearson:",
        round(
            pearson(
                cohort_vectors[
                    "EARLY"
                ],
                cohort_vectors[
                    "LATE"
                ],
            ),
            4,
        ),
    )
    print(
        "  cosine:",
        round(
            cosine(
                cohort_vectors[
                    "EARLY"
                ],
                cohort_vectors[
                    "LATE"
                ],
            ),
            4,
        ),
    )

    # ----------------------------------------------------------
    # 15. Leave-one-patient-out internal replication
    #
    # Does each patient align with a common direction estimated
    # from the OTHER three patients?
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

            training_keys = [
                patient
                + "__"
                + baseline
                for patient in training
            ]

            training_common = (
                patient_residual_matrix[
                    training_keys
                ].mean(
                    axis=1
                )
            )

            held_vector = patient_residual_matrix[
                held_out
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
                            held_vector,
                            training_common,
                        ),

                    "cosine_to_other_three":
                        cosine(
                            held_vector,
                            training_common,
                        ),
                }
            )

    loo = pd.DataFrame(
        loo_rows
    )

    print()
    print("Leave-one-patient-out internal CF residual agreement")
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
    # 16. Patient-consensus programme direction
    #
    # Use equal-patient cohort vectors, not nine samples, to
    # prevent patients with more longitudinal samples dominating.
    # ----------------------------------------------------------

    consensus_rows = []

    for imodulon in stable_programmes:
        early_values = np.array(
            [
                patient_residual_matrix.loc[
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
                patient_residual_matrix.loc[
                    imodulon,
                    patient
                    + "__LATE",
                ]
                for patient in PATIENTS
            ],
            dtype=float,
        )

        all_early_positive = bool(
            np.all(
                early_values
                > 0
            )
        )

        all_early_negative = bool(
            np.all(
                early_values
                < 0
            )
        )

        all_late_positive = bool(
            np.all(
                late_values
                > 0
            )
        )

        all_late_negative = bool(
            np.all(
                late_values
                < 0
            )
        )

        same_positive = (
            all_early_positive
            and all_late_positive
        )

        same_negative = (
            all_early_negative
            and all_late_negative
        )

        consensus_both = (
            same_positive
            or same_negative
        )

        consensus_rows.append(
            {
                "imodulon":
                    imodulon,

                "same_direction_all_4_patients_both_baselines":
                    consensus_both,

                "direction":
                    (
                        "positive"
                        if same_positive
                        else (
                            "negative"
                            if same_negative
                            else "mixed"
                        )
                    ),

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
                    float(
                        gse_residual.loc[
                            imodulon
                        ]
                    ),
            }
        )

    consensus = pd.DataFrame(
        consensus_rows
    )

    consensus[
        "matches_GSE119356_direction"
    ] = (
        (
            (
                consensus[
                    "direction"
                ]
                == "positive"
            )
            & (
                consensus[
                    "GSE119356_reference_activity"
                ]
                > 0
            )
        )
        | (
            (
                consensus[
                    "direction"
                ]
                == "negative"
            )
            & (
                consensus[
                    "GSE119356_reference_activity"
                ]
                < 0
            )
        )
    )

    stable_consensus = consensus.loc[
        consensus[
            "same_direction_all_4_patients_both_baselines"
        ]
    ].copy()

    stable_consensus[
        "minimum_abs_across_baselines"
    ] = stable_consensus[
        [
            "minimum_abs_EARLY",
            "minimum_abs_LATE",
        ]
    ].min(
        axis=1
    )

    stable_consensus = stable_consensus.sort_values(
        "minimum_abs_across_baselines",
        ascending=False,
    )

    print()
    print(
        "Strongest residual programmes with the same direction "
        "in all 4 patients under BOTH baselines"
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
    # 17. Technical decision
    #
    # Biological similarity to GSE119356 is deliberately NOT a
    # pass criterion.
    #
    # A failure to replicate would itself be a valid result.
    # ----------------------------------------------------------

    min_projection_corr = float(
        projection_qc[
            "projection_corr"
        ].min()
    )

    min_projection_energy = float(
        projection_qc[
            "projection_energy_fraction"
        ].min()
    )

    min_baseline_residual_cosine = float(
        baseline_robustness[
            "cosine_EARLY_vs_LATE_residual"
        ].min()
    )

    finite_outputs = bool(
        np.isfinite(
            projection_qc.select_dtypes(
                include=[
                    np.number
                ]
            ).to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            patient_similarity.select_dtypes(
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
        "4925_replication_genes":
            len(
                genes
            )
            == 4925,

        "9_primary_sputum_samples":
            sputum.shape[
                1
            ]
            == 9,

        "19_matched_lab_runs":
            lab.shape[
                1
            ]
            == 19,

        "4_patients":
            len(
                PATIENTS
            )
            == 4,

        "108_stable_primary_programmes":
            len(
                stable_programmes
            )
            == 108,

        "18_sample_by_baseline_contrasts":
            len(
                projection_qc
            )
            == 18,

        "all_projection_corr_ge_040":
            min_projection_corr
            >= 0.40,

        "all_projection_energy_ge_015":
            min_projection_energy
            >= 0.15,

        "all_sample_residual_baseline_cosine_ge_050":
            min_baseline_residual_cosine
            >= 0.50,

        "finite_outputs":
            finite_outputs,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 03f technical checks")

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
        "PRJEB24688_MULTIPATIENT_CF_REPLICATION_GENERATED"
        if passed
        else "PRJEB24688_MULTIPATIENT_CF_REPLICATION_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: similarity to GSE119356 is a biological result, "
        "not an audit gate. Audit 03f is allowed to conclude either "
        "replication or non-replication."
    )

    print()
    print(
        "EARLY means matched 3 h / exponential laboratory growth; "
        "LATE means matched 24 h / stationary laboratory growth."
    )

    # ----------------------------------------------------------
    # 18. Save
    # ----------------------------------------------------------

    sputum_tpm.to_csv(
        OUT
        / "PRJEB24688_PRIMARY_sputum_TPM.csv"
    )

    lab_tpm.to_csv(
        OUT
        / "PRJEB24688_matched_lab_TPM.csv"
    )

    lab_metadata.to_csv(
        OUT
        / "matched_lab_sample_metadata.csv",
        index=False,
    )

    isolate_reference_info.to_csv(
        OUT
        / "isolate_level_reference_design.csv",
        index=False,
    )

    patient_reference_info.to_csv(
        OUT
        / "patient_level_reference_design.csv",
        index=False,
    )

    gse_residual.to_csv(
        OUT
        / "GSE119356_mask_compatible_common_residual_108.csv",
        header=True,
    )

    projection_qc.to_csv(
        OUT
        / "PRJEB24688_sputum_projection_QC.csv",
        index=False,
    )

    activity_matrix.to_csv(
        OUT
        / "PRJEB24688_calibrated_CF_activity_contrasts_108.csv"
    )

    residual_matrix.to_csv(
        OUT
        / "PRJEB24688_controlled_state_residuals_108.csv"
    )

    sample_similarity.to_csv(
        OUT
        / "sample_level_GSE119356_residual_replication.csv",
        index=False,
    )

    baseline_robustness.to_csv(
        OUT
        / "sample_residual_baseline_robustness.csv",
        index=False,
    )

    patient_residual_matrix.to_csv(
        OUT
        / "patient_mean_CF_residuals.csv"
    )

    patient_similarity.to_csv(
        OUT
        / "patient_level_GSE119356_residual_replication.csv",
        index=False,
    )

    cohort_similarity.to_csv(
        OUT
        / "cohort_consensus_GSE119356_replication.csv",
        index=False,
    )

    loo.to_csv(
        OUT
        / "leave_one_patient_out_internal_replication.csv",
        index=False,
    )

    consensus.to_csv(
        OUT
        / "cross_patient_programme_consensus.csv",
        index=False,
    )

    stable_consensus.to_csv(
        OUT
        / "stable_cross_patient_programmes_both_baselines.csv",
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
        / "AUDIT_03F_STATUS.txt"
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
            "Failed Audit 03f technical checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
