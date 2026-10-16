from pathlib import Path
from io import StringIO
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path.cwd()

LOG_EXPR = (
    ROOT
    / "results/phase3/audit_03g_GSE55197_PA14_biofilm_composition"
    / "GSE55197_PA14_biofilm_projection_log2TPM_PAO1_space.csv"
)

ACTIVITY = (
    ROOT
    / "results/phase3/audit_03g_GSE55197_PA14_biofilm_composition"
    / "GSE55197_PA14_biofilm_imodulon_activities.csv"
)

COUNTS = (
    ROOT
    / "results/phase3/audit_03g_GSE55197_PA14_biofilm_composition"
    / "GSE55197_PA14_biofilm_projection_counts_PAO1_space.csv"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

TRANSFER = (
    ROOT
    / "results/phase3/audit_03c_GSE55197_composite_ID_resolution"
    / "GSE55197_exact_imodulon_transferability.csv"
)

STATIONARY = (
    ROOT
    / "results/phase3/audit_03d_GSE55197_stationary_crossstrain_projection"
    / "GSE55197_PA14_stationary_programme_vector.csv"
)

OXYGEN = (
    ROOT
    / "results/phase3/audit_03e_GSE55197_anoxic_crossstrain_projection"
    / "GSE55197_anoxic_stationary_residual_decomposition.csv"
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

GSE299861 = (
    ROOT
    / "results/phase2/audit_12f_GSE299861_canonical_imodulon_challenge"
    / "GSE299861_biofilm_programme_vector.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_03h_GSE55197_biofilm_programme_stability"
)


EXP = [
    "PA14_EXP_1",
    "PA14_EXP_2",
]

BIOFILM_GROUPS = {
    "24h": [
        "PA14_BF24_1",
        "PA14_BF24_2",
    ],
    "48h": [
        "PA14_BF48_1",
        "PA14_BF48_2",
    ],
}


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
        "Unexpected serialized DataFrame."
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


def vector_corr(a, b):
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


def sign_agreement(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    return float(
        np.mean(
            np.sign(a)
            == np.sign(b)
        )
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
        null[i] = cosine(
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


def read_vector(
    path,
    value_column,
    imodulons,
):
    table = pd.read_csv(
        path
    )

    index_column = None

    for candidate in [
        "imodulon",
        "iModulon",
        "Unnamed: 0",
    ]:
        if candidate in table.columns:
            index_column = candidate
            break

    if index_column is None:
        raise RuntimeError(
            "Could not identify iModulon column in "
            + str(path)
        )

    if value_column not in table.columns:
        raise RuntimeError(
            str(path)
            + " missing column "
            + value_column
        )

    table[index_column] = (
        table[index_column]
        .astype(str)
        .str.strip()
    )

    vector = pd.to_numeric(
        table.set_index(
            index_column
        )[value_column],
        errors="raise",
    ).reindex(
        imodulons
    )

    if vector.isna().any():
        raise RuntimeError(
            "Missing iModulons in "
            + str(path)
        )

    return vector


def compare(
    target,
    reference,
    programme_set,
):
    a = target.reindex(
        programme_set
    ).to_numpy(
        dtype=float
    )

    b = reference.reindex(
        programme_set
    ).to_numpy(
        dtype=float
    )

    observed_cosine, p = permutation_p(
        a,
        b,
    )

    return {
        "pearson":
            float(
                pearsonr(
                    a,
                    b,
                ).statistic
            ),

        "spearman":
            float(
                spearmanr(
                    a,
                    b,
                ).statistic
            ),

        "cosine":
            observed_cosine,

        "sign_agreement":
            sign_agreement(
                a,
                b,
            ),

        "permutation_p":
            p,
    }


def decompose(
    vector,
    stationary,
    oxygen,
):
    y = vector.to_numpy(
        dtype=float
    )

    s = stationary.to_numpy(
        dtype=float
    )

    o = oxygen.to_numpy(
        dtype=float
    )

    X = np.column_stack(
        [
            s,
            o,
        ]
    )

    beta = (
        np.linalg.pinv(
            X
        )
        @ y
    )

    reconstructed = (
        X
        @ beta
    )

    residual = (
        y
        - reconstructed
    )

    total_energy = float(
        np.dot(
            y,
            y,
        )
    )

    stationary_component = (
        beta[0]
        * s
    )

    oxygen_component = (
        beta[1]
        * o
    )

    return {
        "stationary_coefficient":
            float(
                beta[0]
            ),

        "oxygen_coefficient":
            float(
                beta[1]
            ),

        "stationary_energy_fraction":
            float(
                np.dot(
                    stationary_component,
                    stationary_component,
                )
                / total_energy
            ),

        "oxygen_energy_fraction":
            float(
                np.dot(
                    oxygen_component,
                    oxygen_component,
                )
                / total_energy
            ),

        "combined_energy_fraction":
            float(
                np.dot(
                    reconstructed,
                    reconstructed,
                )
                / total_energy
            ),

        "residual_energy_fraction":
            float(
                np.dot(
                    residual,
                    residual,
                )
                / total_energy
            ),

        "residual":
            pd.Series(
                residual,
                index=vector.index,
            ),
    }


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 03h")
    print("GSE55197 PA14 biofilm programme-level stability")
    print("=" * 78)

    for path in [
        LOG_EXPR,
        ACTIVITY,
        COUNTS,
        MODEL,
        TRANSFER,
        STATIONARY,
        OXYGEN,
        GSE203348,
        GSE120602,
        GSE299861,
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
    # 1. Frozen model
    # ----------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(
            handle
        )

    M = json_df(
        raw["M"]
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
    # 2. Audit 03g outputs
    # ----------------------------------------------------------

    log_expr = pd.read_csv(
        LOG_EXPR,
        index_col=0,
    )

    activity = pd.read_csv(
        ACTIVITY,
        index_col=0,
    )

    counts = pd.read_csv(
        COUNTS,
        index_col=0,
    )

    log_expr.index = (
        log_expr.index
        .astype(str)
        .str.strip()
    )

    activity.index = (
        activity.index
        .astype(str)
        .str.strip()
    )

    activity = activity.reindex(
        imodulons
    )

    M_shared = M.reindex(
        log_expr.index
    )

    if M_shared.isna().any().any():
        raise RuntimeError(
            "Missing model genes."
        )

    pinv_M = np.linalg.pinv(
        M_shared.to_numpy(
            dtype=float
        )
    )

    exp_reference = log_expr[
        EXP
    ].mean(
        axis=1
    )

    # ----------------------------------------------------------
    # 3. Original failed expression gates stay preserved
    # ----------------------------------------------------------

    expression_rows = []

    for timepoint, samples in BIOFILM_GROUPS.items():
        r = float(
            log_expr[
                samples
            ]
            .corr()
            .iloc[
                0,
                1
            ]
        )

        activity_r = float(
            activity[
                samples
            ]
            .corr()
            .iloc[
                0,
                1
            ]
        )

        expression_rows.append(
            {
                "timepoint":
                    timepoint,

                "expression_r":
                    r,

                "activity_r":
                    activity_r,
            }
        )

    original_qc = pd.DataFrame(
        expression_rows
    )

    print()
    print("Original Audit 03g replicate metrics")
    print(
        original_qc
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Audit 03g remains REVIEW REQUIRED; "
        "its expression gates are not altered."
    )

    # ----------------------------------------------------------
    # 4. Individual biofilm replicate programme vectors
    # ----------------------------------------------------------

    replicate_vectors = {}

    projection_rows = []

    for timepoint, samples in BIOFILM_GROUPS.items():
        for sample in samples:
            delta_x = (
                log_expr[
                    sample
                ]
                - exp_reference
            )

            delta_a = (
                pinv_M
                @ delta_x.to_numpy(
                    dtype=float
                )
            )

            vector = pd.Series(
                delta_a,
                index=imodulons,
                name=sample,
            )

            replicate_vectors[
                sample
            ] = vector

            reconstruction = (
                M_shared.to_numpy(
                    dtype=float
                )
                @ delta_a
            )

            projection_rows.append(
                {
                    "timepoint":
                        timepoint,

                    "sample":
                        sample,

                    "projection_corr":
                        vector_corr(
                            delta_x,
                            reconstruction,
                        ),

                    "projection_energy_fraction":
                        float(
                            np.linalg.norm(
                                reconstruction
                            )
                            ** 2
                            / (
                                np.linalg.norm(
                                    delta_x
                                )
                                ** 2
                            )
                        ),

                    "model_compatible_counts":
                        float(
                            counts[
                                sample
                            ].sum()
                        ),

                    "direct_vs_stored_activity_max_diff":
                        float(
                            np.max(
                                np.abs(
                                    delta_a
                                    - activity[
                                        sample
                                    ].to_numpy(
                                        dtype=float
                                    )
                                )
                            )
                        ),
                }
            )

    projection_qc = pd.DataFrame(
        projection_rows
    )

    print()
    print("Individual biofilm replicate projection QC")
    print(
        projection_qc
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Within-timepoint programme stability
    # ----------------------------------------------------------

    stability_rows = []

    group_vectors = {}

    for timepoint, samples in BIOFILM_GROUPS.items():
        first = replicate_vectors[
            samples[0]
        ]

        second = replicate_vectors[
            samples[1]
        ]

        group = (
            first
            + second
        ) / 2.0

        group_vectors[
            timepoint
        ] = group

        stability_rows.append(
            {
                "timepoint":
                    timepoint,

                "replicate_pearson":
                    vector_corr(
                        first,
                        second,
                    ),

                "replicate_cosine":
                    cosine(
                        first,
                        second,
                    ),

                "replicate_1_to_group_pearson":
                    vector_corr(
                        first,
                        group,
                    ),

                "replicate_2_to_group_pearson":
                    vector_corr(
                        second,
                        group,
                    ),

                "replicate_1_to_group_cosine":
                    cosine(
                        first,
                        group,
                    ),

                "replicate_2_to_group_cosine":
                    cosine(
                        second,
                        group,
                    ),
            }
        )

    stability = pd.DataFrame(
        stability_rows
    )

    print()
    print("Biofilm programme-vector stability")
    print(
        stability
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Sensitivity to exponential-control identity
    # ----------------------------------------------------------

    control_rows = []

    for timepoint, samples in BIOFILM_GROUPS.items():
        group = group_vectors[
            timepoint
        ]

        for biofilm_sample in samples:
            for exp_sample in EXP:
                delta_x = (
                    log_expr[
                        biofilm_sample
                    ]
                    - log_expr[
                        exp_sample
                    ]
                )

                vector = pd.Series(
                    pinv_M
                    @ delta_x.to_numpy(
                        dtype=float
                    ),
                    index=imodulons,
                )

                control_rows.append(
                    {
                        "timepoint":
                            timepoint,

                        "contrast":
                            biofilm_sample
                            + "_minus_"
                            + exp_sample,

                        "pearson_to_group":
                            vector_corr(
                                vector,
                                group,
                            ),

                        "cosine_to_group":
                            cosine(
                                vector,
                                group,
                            ),
                    }
                )

    control_sensitivity = pd.DataFrame(
        control_rows
    )

    print()
    print("Sensitivity to individual exponential control")
    print(
        control_sensitivity
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Conserved physiological anchors
    # ----------------------------------------------------------

    stationary = read_vector(
        STATIONARY,
        "stationary_vs_exponential",
        imodulons,
    )

    oxygen = read_vector(
        OXYGEN,
        "anoxic_residual_after_stationary",
        imodulons,
    )

    # ----------------------------------------------------------
    # 8. Individual-replicate decomposition
    # ----------------------------------------------------------

    decomposition_rows = []

    residual_vectors = {}

    for timepoint, samples in BIOFILM_GROUPS.items():
        for sample in samples:
            result = decompose(
                replicate_vectors[
                    sample
                ],
                stationary,
                oxygen,
            )

            residual_vectors[
                sample
            ] = result[
                "residual"
            ]

            decomposition_rows.append(
                {
                    "timepoint":
                        timepoint,

                    "sample":
                        sample,

                    "stationary_coefficient":
                        result[
                            "stationary_coefficient"
                        ],

                    "oxygen_coefficient":
                        result[
                            "oxygen_coefficient"
                        ],

                    "stationary_energy_fraction":
                        result[
                            "stationary_energy_fraction"
                        ],

                    "oxygen_energy_fraction":
                        result[
                            "oxygen_energy_fraction"
                        ],

                    "combined_anchor_energy_fraction":
                        result[
                            "combined_energy_fraction"
                        ],

                    "residual_energy_fraction":
                        result[
                            "residual_energy_fraction"
                        ],
                }
            )

    decomposition = pd.DataFrame(
        decomposition_rows
    )

    print()
    print("Individual-replicate conserved-state decomposition")
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
    # 9. Robust transfer set
    # ----------------------------------------------------------

    transfer = pd.read_csv(
        TRANSFER
    )

    transfer[
        "imodulon"
    ] = (
        transfer[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    transfer = transfer.set_index(
        "imodulon"
    ).reindex(
        imodulons
    )

    robust_mask = (
        (
            transfer[
                "weighted_coverage_absM"
            ]
            >= 0.90
        )
        & (
            transfer[
                "energy_coverage_M2"
            ]
            >= 0.80
        )
    )

    all_programmes = pd.Index(
        imodulons,
        name="ALL_116",
    )

    robust = pd.Index(
        transfer.index[
            robust_mask
        ],
        name="ROBUST_TRANSFER",
    )

    # ----------------------------------------------------------
    # 10. Independent PAO1 biofilm comparators
    # ----------------------------------------------------------

    references = {
        "GSE203348":
            read_vector(
                GSE203348,
                "delta_biofilm_minus_planktonic",
                imodulons,
            ),

        "GSE120602":
            read_vector(
                GSE120602,
                "biofilm_vs_planktonic",
                imodulons,
            ),

        "GSE299861":
            read_vector(
                GSE299861,
                "delta_biofilm_minus_planktonic",
                imodulons,
            ),
    }

    biological_rows = []

    for timepoint, samples in BIOFILM_GROUPS.items():
        for sample in samples:
            raw_vector = replicate_vectors[
                sample
            ]

            residual_vector = residual_vectors[
                sample
            ]

            for reference_name, reference in references.items():
                for programme_set in [
                    all_programmes,
                    robust,
                ]:
                    raw_stats = compare(
                        raw_vector,
                        reference,
                        programme_set,
                    )

                    residual_stats = compare(
                        residual_vector,
                        reference,
                        programme_set,
                    )

                    biological_rows.append(
                        {
                            "timepoint":
                                timepoint,

                            "sample":
                                sample,

                            "vector":
                                "raw",

                            "reference":
                                reference_name,

                            "programme_set":
                                programme_set.name,

                            **raw_stats,
                        }
                    )

                    biological_rows.append(
                        {
                            "timepoint":
                                timepoint,

                            "sample":
                                sample,

                            "vector":
                                "anchor_residual",

                            "reference":
                                reference_name,

                            "programme_set":
                                programme_set.name,

                            **residual_stats,
                        }
                    )

    biological = pd.DataFrame(
        biological_rows
    )

    print()
    print("Individual-replicate PAO1 biofilm agreement")
    print(
        biological
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 11. Programme-level stability acceptance
    #
    # Same principle as Audit 03f:
    # Audit 03g's expression failure remains untouched.
    # This asks whether the programme-level composition is stable.
    # ----------------------------------------------------------

    min_programme_pearson = float(
        stability[
            "replicate_pearson"
        ].min()
    )

    min_programme_cosine = float(
        stability[
            "replicate_cosine"
        ].min()
    )

    min_control_pearson = float(
        control_sensitivity[
            "pearson_to_group"
        ].min()
    )

    min_control_cosine = float(
        control_sensitivity[
            "cosine_to_group"
        ].min()
    )

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

    min_depth = float(
        projection_qc[
            "model_compatible_counts"
        ].min()
    )

    max_direct_diff = float(
        projection_qc[
            "direct_vs_stored_activity_max_diff"
        ].max()
    )

    min_stationary_fraction = (
        decomposition
        .groupby(
            "timepoint"
        )[
            "stationary_energy_fraction"
        ]
        .min()
    )

    checks = {
        "original_24h_expression_failure_preserved":
            float(
                original_qc.loc[
                    original_qc[
                        "timepoint"
                    ]
                    == "24h",
                    "expression_r",
                ].iloc[0]
            )
            < 0.90,

        "original_48h_expression_failure_preserved":
            float(
                original_qc.loc[
                    original_qc[
                        "timepoint"
                    ]
                    == "48h",
                    "expression_r",
                ].iloc[0]
            )
            < 0.90,

        "both_activity_replicate_correlations_ge_075":
            bool(
                (
                    original_qc[
                        "activity_r"
                    ]
                    >= 0.75
                ).all()
            ),

        "both_programme_pair_pearsons_ge_085":
            min_programme_pearson
            >= 0.85,

        "both_programme_pair_cosines_ge_085":
            min_programme_cosine
            >= 0.85,

        "all_control_pair_pearsons_ge_085":
            min_control_pearson
            >= 0.85,

        "all_control_pair_cosines_ge_085":
            min_control_cosine
            >= 0.85,

        "each_replicate_projection_corr_ge_040":
            min_projection_corr
            >= 0.40,

        "each_replicate_projection_energy_ge_015":
            min_projection_energy
            >= 0.15,

        "minimum_model_count_depth_ge_100k":
            min_depth
            >= 100000,

        "direct_projection_matches_stored_activity":
            max_direct_diff
            < 1e-8,

        "robust_programmes_ge_90":
            len(
                robust
            )
            >= 90,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Programme-level stability checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    if passed:
        status = (
            "GSE55197_PA14_BIOFILM_PROGRAMME_SIGNAL_ACCEPTED_WITH_EXPRESSION_QC_CAVEAT"
        )
    else:
        status = (
            "GSE55197_PA14_BIOFILM_PROGRAMME_SIGNAL_NOT_ACCEPTED"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print("Stationary-fraction range by timepoint")

    for timepoint in [
        "24h",
        "48h",
    ]:
        subset = decomposition.loc[
            decomposition[
                "timepoint"
            ]
            == timepoint,
            "stationary_energy_fraction",
        ]

        print(
            " ",
            timepoint,
            ":",
            round(
                float(
                    subset.min()
                ),
                4,
            ),
            "to",
            round(
                float(
                    subset.max()
                ),
                4,
            ),
        )

    print()
    print(
        "Interpretation rule: Audit 03g remains REVIEW REQUIRED. "
        "Audit 03h only determines whether programme-level "
        "composition is sufficiently replicate-stable to retain "
        "as supporting biological evidence."
    )

    # ----------------------------------------------------------
    # 12. Save
    # ----------------------------------------------------------

    original_qc.to_csv(
        OUT
        / "original_03g_replicate_QC.csv",
        index=False,
    )

    projection_qc.to_csv(
        OUT
        / "individual_biofilm_projection_QC.csv",
        index=False,
    )

    stability.to_csv(
        OUT
        / "biofilm_programme_vector_stability.csv",
        index=False,
    )

    control_sensitivity.to_csv(
        OUT
        / "biofilm_control_pair_sensitivity.csv",
        index=False,
    )

    decomposition.to_csv(
        OUT
        / "individual_biofilm_conserved_state_decomposition.csv",
        index=False,
    )

    biological.to_csv(
        OUT
        / "individual_biofilm_crossstudy_agreement.csv",
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
        / "AUDIT_03H_STATUS.txt"
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
            "Failed programme-stability checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
