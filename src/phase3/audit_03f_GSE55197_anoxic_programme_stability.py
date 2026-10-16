from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path.cwd()

LOG_EXPR = (
    ROOT
    / "results/phase3/audit_03e_GSE55197_anoxic_crossstrain_projection"
    / "GSE55197_anoxic_projection_log2TPM_plus1_PAO1_space.csv"
)

ACTIVITY = (
    ROOT
    / "results/phase3/audit_03e_GSE55197_anoxic_crossstrain_projection"
    / "GSE55197_anoxic_imodulon_activities.csv"
)

COUNTS = (
    ROOT
    / "results/phase3/audit_03e_GSE55197_anoxic_crossstrain_projection"
    / "GSE55197_anoxic_projection_gene_counts_PAO1_space.csv"
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

MICRO = (
    ROOT
    / "results/phase2/audit_10_multistress_imodulon_dictionary"
    / "stress_contrast_vectors.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_03f_GSE55197_anoxic_programme_stability"
)

EXPONENTIAL = [
    "PA14_EXP_1",
    "PA14_EXP_2",
]

ANOXIC = [
    "PA14_ANOX_1",
    "PA14_ANOX_2",
]


def read_json_df(value):
    from io import StringIO

    if isinstance(value, str):
        return pd.read_json(
            StringIO(value)
        )

    if isinstance(value, dict):
        return pd.DataFrame(
            value
        )

    raise RuntimeError(
        "Unexpected model DataFrame format."
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

    return (
        observed,
        float(p),
    )


def find_index_column(table):
    for column in [
        "imodulon",
        "iModulon",
        "Unnamed: 0",
    ]:
        if column in table.columns:
            return column

    raise RuntimeError(
        "Could not identify iModulon column."
    )


def read_vector(
    path,
    column,
    imodulons,
):
    table = pd.read_csv(
        path
    )

    index_column = find_index_column(
        table
    )

    if column not in table.columns:
        raise RuntimeError(
            str(path)
            + " missing column "
            + column
        )

    table[index_column] = (
        table[index_column]
        .astype(str)
        .str.strip()
    )

    result = pd.to_numeric(
        table.set_index(
            index_column
        )[column],
        errors="raise",
    ).reindex(
        imodulons
    )

    if result.isna().any():
        raise RuntimeError(
            "Missing iModulons in "
            + str(path)
        )

    return result


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


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 03f")
    print("GSE55197 anoxic programme-level replicate stability")
    print("=" * 78)

    for path in [
        LOG_EXPR,
        ACTIVITY,
        COUNTS,
        MODEL,
        TRANSFER,
        STATIONARY,
        MICRO,
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

    import json

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(
            handle
        )

    M = read_json_df(
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
    # 2. Audit 03e expression and programme outputs
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

    counts.index = (
        counts.index
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
            "Missing PAO1 model genes."
        )

    M_array = M_shared.to_numpy(
        dtype=float
    )

    pinv_M = np.linalg.pinv(
        M_array
    )

    exp_reference = log_expr[
        EXPONENTIAL
    ].mean(
        axis=1
    )

    # ----------------------------------------------------------
    # 3. Original expression-level caveat
    # ----------------------------------------------------------

    anoxic_expression_r = float(
        log_expr[
            ANOXIC
        ]
        .corr()
        .iloc[
            0,
            1
        ]
    )

    anoxic_activity_r = float(
        activity[
            ANOXIC
        ]
        .corr()
        .iloc[
            0,
            1
        ]
    )

    print()
    print("Original Audit 03e replicate metrics")
    print(
        "  anoxic expression r:",
        round(
            anoxic_expression_r,
            4,
        ),
    )
    print(
        "  anoxic activity r:",
        round(
            anoxic_activity_r,
            4,
        ),
    )

    print()
    print(
        "  NOTE: Audit 03e remains REVIEW REQUIRED because "
        "expression r < 0.90."
    )

    # ----------------------------------------------------------
    # 4. Per-anoxic-replicate contrast projection
    # ----------------------------------------------------------

    replicate_rows = []

    replicate_vectors = {}

    for sample in ANOXIC:
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

        reconstruction = (
            M_array
            @ delta_a
        )

        corr = vector_corr(
            delta_x.to_numpy(
                dtype=float
            ),
            reconstruction,
        )

        relative_error = float(
            np.linalg.norm(
                delta_x.to_numpy(
                    dtype=float
                )
                - reconstruction
            )
            / np.linalg.norm(
                delta_x.to_numpy(
                    dtype=float
                )
            )
        )

        energy_fraction = float(
            np.linalg.norm(
                reconstruction
            )
            ** 2
            / (
                np.linalg.norm(
                    delta_x.to_numpy(
                        dtype=float
                    )
                )
                ** 2
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

        stored_difference = float(
            np.max(
                np.abs(
                    vector.to_numpy(
                        dtype=float
                    )
                    - activity[
                        sample
                    ].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        replicate_rows.append(
            {
                "sample":
                    sample,

                "projection_corr":
                    corr,

                "relative_error":
                    relative_error,

                "projection_energy_fraction":
                    energy_fraction,

                "direct_vs_stored_activity_max_diff":
                    stored_difference,

                "model_compatible_counts":
                    float(
                        counts[
                            sample
                        ].sum()
                    ),
            }
        )

    replicate_qc = pd.DataFrame(
        replicate_rows
    )

    print()
    print("Individual anoxic replicate projection QC")
    print(
        replicate_qc
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Group-mean programme vector
    # ----------------------------------------------------------

    group_delta = (
        replicate_vectors[
            "PA14_ANOX_1"
        ]
        + replicate_vectors[
            "PA14_ANOX_2"
        ]
    ) / 2.0

    replicate_programme_rows = []

    for sample in ANOXIC:
        vector = replicate_vectors[
            sample
        ]

        replicate_programme_rows.append(
            {
                "sample":
                    sample,

                "pearson_to_group":
                    vector_corr(
                        vector,
                        group_delta,
                    ),

                "cosine_to_group":
                    cosine(
                        vector,
                        group_delta,
                    ),
            }
        )

    replicate_programme = pd.DataFrame(
        replicate_programme_rows
    )

    replicate_pair_pearson = vector_corr(
        replicate_vectors[
            "PA14_ANOX_1"
        ],
        replicate_vectors[
            "PA14_ANOX_2"
        ],
    )

    replicate_pair_cosine = cosine(
        replicate_vectors[
            "PA14_ANOX_1"
        ],
        replicate_vectors[
            "PA14_ANOX_2"
        ],
    )

    print()
    print("Programme-vector replicate stability")
    print(
        "  ANOX_1 vs ANOX_2 Pearson:",
        round(
            replicate_pair_pearson,
            4,
        ),
    )
    print(
        "  ANOX_1 vs ANOX_2 cosine:",
        round(
            replicate_pair_cosine,
            4,
        ),
    )

    print()
    print(
        replicate_programme
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Sensitivity to choice of individual exponential control
    #
    # Four contrasts:
    # ANOX1-EXP1, ANOX1-EXP2, ANOX2-EXP1, ANOX2-EXP2.
    # ----------------------------------------------------------

    pair_vectors = {}

    pair_rows = []

    for anox in ANOXIC:
        for exp in EXPONENTIAL:
            delta_x = (
                log_expr[
                    anox
                ]
                - log_expr[
                    exp
                ]
            )

            delta_a = (
                pinv_M
                @ delta_x.to_numpy(
                    dtype=float
                )
            )

            name = (
                anox
                + "_minus_"
                + exp
            )

            vector = pd.Series(
                delta_a,
                index=imodulons,
                name=name,
            )

            pair_vectors[
                name
            ] = vector

            pair_rows.append(
                {
                    "contrast":
                        name,

                    "pearson_to_group_mean":
                        vector_corr(
                            vector,
                            group_delta,
                        ),

                    "cosine_to_group_mean":
                        cosine(
                            vector,
                            group_delta,
                        ),
                }
            )

    pair_stability = pd.DataFrame(
        pair_rows
    )

    min_pair_pearson = float(
        pair_stability[
            "pearson_to_group_mean"
        ].min()
    )

    min_pair_cosine = float(
        pair_stability[
            "cosine_to_group_mean"
        ].min()
    )

    print()
    print("Sensitivity to replicate/control pairing")
    print(
        pair_stability
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Robust cross-strain programme set
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

    robust = pd.Index(
        transfer.index[
            robust_mask
        ],
        name="ROBUST_TRANSFER",
    )

    # ----------------------------------------------------------
    # 8. Independent reference vectors
    # ----------------------------------------------------------

    stationary = read_vector(
        STATIONARY,
        "stationary_vs_exponential",
        imodulons,
    )

    micro = read_vector(
        MICRO,
        "microaerophilic",
        imodulons,
    )

    # ----------------------------------------------------------
    # 9. Individual replicate biological agreement
    #
    # Scientific output only — NOT used as the technical
    # programme-stability acceptance gate.
    # ----------------------------------------------------------

    biological_rows = []

    for sample in ANOXIC:
        vector = replicate_vectors[
            sample
        ]

        stat_array = stationary.to_numpy(
            dtype=float
        )

        vector_array = vector.to_numpy(
            dtype=float
        )

        beta = float(
            np.dot(
                vector_array,
                stat_array,
            )
            / np.dot(
                stat_array,
                stat_array,
            )
        )

        residual = pd.Series(
            vector_array
            - beta
            * stat_array,
            index=imodulons,
        )

        for programme_set in [
            pd.Index(
                imodulons,
                name="ALL_116",
            ),
            robust,
        ]:
            raw_stats = compare(
                vector,
                micro,
                programme_set,
            )

            residual_stats = compare(
                residual,
                micro,
                programme_set,
            )

            biological_rows.append(
                {
                    "sample":
                        sample,

                    "programme_set":
                        programme_set.name,

                    "comparison":
                        "raw_anoxic_vs_PAO1_microaerophilic",

                    **raw_stats,
                }
            )

            biological_rows.append(
                {
                    "sample":
                        sample,

                    "programme_set":
                        programme_set.name,

                    "comparison":
                        "stationary_residual_vs_PAO1_microaerophilic",

                    **residual_stats,
                }
            )

    biological = pd.DataFrame(
        biological_rows
    )

    print()
    print("Individual-replicate cross-strain oxygen agreement")
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
    # 10. Programme-level acceptance
    #
    # This does NOT overwrite Audit 03e's failed expression gate.
    # It asks a separate question:
    #
    # Is programme-level inference stable despite the modest
    # gene-level replicate disagreement?
    #
    # Cross-strain biological correlations are deliberately not
    # included in these acceptance gates.
    # ----------------------------------------------------------

    minimum_projection_corr = float(
        replicate_qc[
            "projection_corr"
        ].min()
    )

    minimum_projection_energy = float(
        replicate_qc[
            "projection_energy_fraction"
        ].min()
    )

    maximum_activity_difference = float(
        replicate_qc[
            "direct_vs_stored_activity_max_diff"
        ].max()
    )

    minimum_depth = float(
        replicate_qc[
            "model_compatible_counts"
        ].min()
    )

    checks = {
        "original_expression_QC_failure_preserved":
            anoxic_expression_r
            < 0.90,

        "anoxic_activity_replicate_r_ge_090":
            anoxic_activity_r
            >= 0.90,

        "replicate_programme_pearson_ge_090":
            replicate_pair_pearson
            >= 0.90,

        "replicate_programme_cosine_ge_090":
            replicate_pair_cosine
            >= 0.90,

        "all_four_control_pair_pearsons_ge_085":
            min_pair_pearson
            >= 0.85,

        "all_four_control_pair_cosines_ge_085":
            min_pair_cosine
            >= 0.85,

        "each_anoxic_projection_corr_ge_040":
            minimum_projection_corr
            >= 0.40,

        "each_anoxic_projection_energy_ge_015":
            minimum_projection_energy
            >= 0.15,

        "direct_projection_matches_stored_activity":
            maximum_activity_difference
            < 1e-8,

        "minimum_model_count_depth_ge_200k":
            minimum_depth
            >= 200000,

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
            "GSE55197_ANOXIC_PROGRAMME_SIGNAL_ACCEPTED_WITH_EXPRESSION_QC_CAVEAT"
        )
    else:
        status = (
            "GSE55197_ANOXIC_PROGRAMME_SIGNAL_NOT_ACCEPTED"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "Interpretation rule:"
    )
    print(
        "  Audit 03e remains technically REVIEW REQUIRED."
    )
    print(
        "  Audit 03f determines whether the programme-level "
        "oxygen-family result is sufficiently replicate-stable "
        "to retain as supporting evidence."
    )

    # ----------------------------------------------------------
    # 11. Save
    # ----------------------------------------------------------

    replicate_qc.to_csv(
        OUT
        / "individual_anoxic_projection_QC.csv",
        index=False,
    )

    replicate_programme.to_csv(
        OUT
        / "anoxic_replicate_programme_stability.csv",
        index=False,
    )

    pair_stability.to_csv(
        OUT
        / "anoxic_control_pair_sensitivity.csv",
        index=False,
    )

    biological.to_csv(
        OUT
        / "individual_replicate_crossstrain_oxygen_agreement.csv",
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

    summary = [
        "# CF-State Phase 3 — Audit 03f",
        "",
        "## GSE55197 PA14 anoxic programme-level stability",
        "",
        "Audit 03e remains REVIEW REQUIRED because the two "
        "anoxic expression profiles correlated at r="
        + format(
            anoxic_expression_r,
            ".4f",
        )
        + ", below its pre-specified 0.90 gate.",
        "",
        "Audit 03f asks a separate question: whether the inferred "
        "iModulon programme response is stable to anoxic replicate "
        "identity and exponential-control identity.",
        "",
        "- Anoxic activity replicate r: "
        + format(
            anoxic_activity_r,
            ".4f",
        ),
        "- Anoxic programme-vector Pearson: "
        + format(
            replicate_pair_pearson,
            ".4f",
        ),
        "- Anoxic programme-vector cosine: "
        + format(
            replicate_pair_cosine,
            ".4f",
        ),
        "- Minimum of four control-pair Pearson similarities: "
        + format(
            min_pair_pearson,
            ".4f",
        ),
        "- Minimum of four control-pair cosine similarities: "
        + format(
            min_pair_cosine,
            ".4f",
        ),
        "",
        "## Decision",
        "",
        status,
    ]

    (
        OUT
        / "AUDIT_03F_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary
        )
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
