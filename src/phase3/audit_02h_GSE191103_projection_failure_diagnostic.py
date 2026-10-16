from pathlib import Path
from io import StringIO
import json

import numpy as np
import pandas as pd


ROOT = Path.cwd()

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

COUNTS = (
    ROOT
    / "results/phase3/audit_02a_GSE191103_PA14_intake"
    / "GSE191103_WT_PA14_processed_expression.csv"
)

PROJECTION_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

TPM = (
    ROOT
    / "results/phase3/audit_02g_GSE191103_final_PA14_projection"
    / "GSE191103_projection_compatible_TPM.csv"
)

LOG_EXPR = (
    ROOT
    / "results/phase3/audit_02g_GSE191103_final_PA14_projection"
    / "GSE191103_projection_compatible_log2TPM_plus1_PAO1_space.csv"
)

ACTIVITY = (
    ROOT
    / "results/phase3/audit_02g_GSE191103_final_PA14_projection"
    / "GSE191103_PA14_imodulon_activities.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_02h_GSE191103_projection_failure_diagnostic"
)

BIOFILM = [
    "PA14_BF_1",
    "PA14_BF_2",
    "PA14_BF_3",
]

PLANKTONIC = [
    "PA14_PL_1",
    "PA14_PL_2",
    "PA14_PL_3",
]

ALL_SAMPLES = (
    BIOFILM
    + PLANKTONIC
)


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


def pairwise_table(
    matrix,
    samples,
    group_name,
):
    corr = matrix[
        samples
    ].corr()

    rows = []

    for i in range(
        len(samples)
    ):
        for j in range(
            i + 1,
            len(samples),
        ):
            rows.append(
                {
                    "group":
                        group_name,

                    "sample_1":
                        samples[
                            i
                        ],

                    "sample_2":
                        samples[
                            j
                        ],

                    "pearson_r":
                        float(
                            corr.loc[
                                samples[
                                    i
                                ],
                                samples[
                                    j
                                ],
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        rows
    )


def mean_pairwise(
    matrix,
    samples,
):
    table = pairwise_table(
        matrix,
        samples,
        "tmp",
    )

    return float(
        table[
            "pearson_r"
        ].mean()
    )


def cosine(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(
            a
        )
        * np.linalg.norm(
            b
        )
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


def vector_corr(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        np.std(
            a
        )
        == 0
        or np.std(
            b
        )
        == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[
            0,
            1
        ]
    )


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02h")
    print("GSE191103 PA14 projection-failure diagnostic")
    print("=" * 78)

    for path in [
        MODEL,
        COUNTS,
        PROJECTION_MAP,
        TPM,
        LOG_EXPR,
        ACTIVITY,
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
    # Frozen model
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

    # ----------------------------------------------------------
    # Frozen projection map
    # ----------------------------------------------------------

    projection_map = pd.read_csv(
        PROJECTION_MAP
    )

    projection_map[
        "pao1_gene"
    ] = (
        projection_map[
            "pao1_gene"
        ]
        .astype(str)
        .str.strip()
    )

    projection_map[
        "pa14_gene"
    ] = (
        projection_map[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    pa14_genes = list(
        projection_map[
            "pa14_gene"
        ]
    )

    pao1_genes = list(
        projection_map[
            "pao1_gene"
        ]
    )

    # ----------------------------------------------------------
    # Raw count-depth diagnostic
    # ----------------------------------------------------------

    counts = pd.read_csv(
        COUNTS,
        index_col=0,
    )

    counts.index = (
        counts.index
        .astype(str)
        .str.strip()
    )

    counts = counts.apply(
        pd.to_numeric,
        errors="raise",
    )

    projection_counts = counts.reindex(
        pa14_genes
    )[
        ALL_SAMPLES
    ]

    if projection_counts.isna().any().any():
        raise RuntimeError(
            "Missing projection-universe raw counts."
        )

    depth = pd.DataFrame(
        {
            "projection_gene_counts":
                projection_counts.sum(
                    axis=0
                ),

            "detected_genes":
                (
                    projection_counts
                    > 0
                ).sum(
                    axis=0
                ),
        }
    )

    depth[
        "ge_500k_projection_counts"
    ] = (
        depth[
            "projection_gene_counts"
        ]
        >= 500000
    )

    print()
    print("Projection-universe count depth")
    print(
        depth.to_string()
    )

    # ----------------------------------------------------------
    # Expression + activities from Audit 02g
    # ----------------------------------------------------------

    tpm = pd.read_csv(
        TPM,
        index_col=0,
    )

    tpm.index = (
        tpm.index
        .astype(str)
        .str.strip()
    )

    # TPM output is PA14-indexed. Rename to PAO1 model space.
    pa14_to_pao1 = dict(
        zip(
            projection_map[
                "pa14_gene"
            ],
            projection_map[
                "pao1_gene"
            ],
        )
    )

    tpm = tpm.reindex(
        pa14_genes
    )

    tpm.index = [
        pa14_to_pao1[
            gene
        ]
        for gene in tpm.index
    ]

    tpm = tpm.reindex(
        pao1_genes
    )

    log_expr = pd.read_csv(
        LOG_EXPR,
        index_col=0,
    )

    log_expr.index = (
        log_expr.index
        .astype(str)
        .str.strip()
    )

    log_expr = log_expr.reindex(
        pao1_genes
    )[
        ALL_SAMPLES
    ]

    activity = pd.read_csv(
        ACTIVITY,
        index_col=0,
    )

    activity.index = (
        activity.index
        .astype(str)
        .str.strip()
    )

    activity = activity.reindex(
        M.columns
    )[
        ALL_SAMPLES
    ]

    if (
        log_expr.isna().any().any()
        or activity.isna().any().any()
    ):
        raise RuntimeError(
            "Missing values in Audit 02g outputs."
        )

    # ----------------------------------------------------------
    # Raw pairwise expression correlations
    # ----------------------------------------------------------

    expression_pairs = pd.concat(
        [
            pairwise_table(
                log_expr,
                BIOFILM,
                "biofilm",
            ),
            pairwise_table(
                log_expr,
                PLANKTONIC,
                "planktonic",
            ),
        ],
        ignore_index=True,
    )

    print()
    print("Pairwise log2(TPM+1) expression correlations")
    print(
        expression_pairs
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    expression_pairs.to_csv(
        OUT
        / "pairwise_expression_correlations.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Pairwise activity correlations
    # ----------------------------------------------------------

    activity_pairs = pd.concat(
        [
            pairwise_table(
                activity,
                BIOFILM,
                "biofilm",
            ),
            pairwise_table(
                activity,
                PLANKTONIC,
                "planktonic",
            ),
        ],
        ignore_index=True,
    )

    print()
    print("Pairwise iModulon activity correlations")
    print(
        activity_pairs
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    activity_pairs.to_csv(
        OUT
        / "pairwise_activity_correlations.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Is low-abundance noise driving poor replicate correlations?
    # ----------------------------------------------------------

    mean_tpm = tpm[
        ALL_SAMPLES
    ].mean(
        axis=1
    )

    filter_rows = []

    thresholds = [
        0.0,
        0.1,
        1.0,
        5.0,
        10.0,
        25.0,
    ]

    for threshold in thresholds:
        genes = mean_tpm.index[
            mean_tpm
            >= threshold
        ]

        subset = log_expr.reindex(
            genes
        )

        if len(
            genes
        ) < 100:
            continue

        filter_rows.append(
            {
                "mean_TPM_threshold":
                    threshold,

                "n_genes":
                    len(
                        genes
                    ),

                "biofilm_mean_pairwise_r":
                    mean_pairwise(
                        subset,
                        BIOFILM,
                    ),

                "planktonic_mean_pairwise_r":
                    mean_pairwise(
                        subset,
                        PLANKTONIC,
                    ),
            }
        )

    filter_table = pd.DataFrame(
        filter_rows
    )

    print()
    print("Expression reproducibility after abundance filtering")
    print(
        filter_table
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    filter_table.to_csv(
        OUT
        / "abundance_filter_replicate_QC.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Identify whether one replicate dominates the problem
    # ----------------------------------------------------------

    leave_rows = []

    for group_name, samples in [
        (
            "biofilm",
            BIOFILM,
        ),
        (
            "planktonic",
            PLANKTONIC,
        ),
    ]:
        for excluded in samples:
            retained = [
                sample
                for sample in samples
                if sample != excluded
            ]

            pair_r = float(
                log_expr[
                    retained
                ].corr().iloc[
                    0,
                    1
                ]
            )

            leave_rows.append(
                {
                    "group":
                        group_name,

                    "excluded_sample":
                        excluded,

                    "remaining_pair_r":
                        pair_r,
                }
            )

    leave_table = pd.DataFrame(
        leave_rows
    )

    print()
    print("Leave-one-replicate-out expression agreement")
    print(
        leave_table
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    leave_table.to_csv(
        OUT
        / "leave_one_replicate_out_expression_QC.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Projection fit per individual sample
    # ----------------------------------------------------------

    M_shared = M.reindex(
        pao1_genes
    )

    plank_reference = log_expr[
        PLANKTONIC
    ].mean(
        axis=1
    )

    centered = log_expr.subtract(
        plank_reference,
        axis=0,
    )

    M_array = M_shared.to_numpy(
        dtype=float
    )

    projection_rows = []

    for sample in ALL_SAMPLES:
        x = centered[
            sample
        ].to_numpy(
            dtype=float
        )

        a = activity[
            sample
        ].to_numpy(
            dtype=float
        )

        recon = (
            M_array
            @ a
        )

        corr = vector_corr(
            x,
            recon,
        )

        x_norm = float(
            np.linalg.norm(
                x
            )
        )

        if x_norm > 0:
            relative_error = float(
                np.linalg.norm(
                    x
                    - recon
                )
                / x_norm
            )

            energy_fraction = float(
                np.linalg.norm(
                    recon
                )
                ** 2
                / (
                    x_norm
                    ** 2
                )
            )
        else:
            relative_error = np.nan
            energy_fraction = np.nan

        projection_rows.append(
            {
                "sample":
                    sample,

                "condition":
                    (
                        "biofilm"
                        if sample in BIOFILM
                        else "planktonic"
                    ),

                "projection_corr":
                    corr,

                "relative_error":
                    relative_error,

                "projection_energy_fraction":
                    energy_fraction,
            }
        )

    projection_table = pd.DataFrame(
        projection_rows
    )

    print()
    print("Per-sample projection fit")
    print(
        projection_table
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    projection_table.to_csv(
        OUT
        / "per_sample_projection_fit.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Crucial test:
    # Does the BIOFILM - PLANKTONIC GROUP CONTRAST project better
    # than noisy individual samples?
    # ----------------------------------------------------------

    delta_x = (
        log_expr[
            BIOFILM
        ].mean(
            axis=1
        )
        - log_expr[
            PLANKTONIC
        ].mean(
            axis=1
        )
    )

    pinv_M = np.linalg.pinv(
        M_array
    )

    delta_a_direct = (
        pinv_M
        @ delta_x.to_numpy(
            dtype=float
        )
    )

    delta_a_from_samples = (
        activity[
            BIOFILM
        ].mean(
            axis=1
        )
        - activity[
            PLANKTONIC
        ].mean(
            axis=1
        )
    ).to_numpy(
        dtype=float
    )

    delta_activity_agreement = float(
        np.max(
            np.abs(
                delta_a_direct
                - delta_a_from_samples
            )
        )
    )

    delta_reconstruction = (
        M_array
        @ delta_a_direct
    )

    delta_x_array = delta_x.to_numpy(
        dtype=float
    )

    contrast_corr = vector_corr(
        delta_x_array,
        delta_reconstruction,
    )

    contrast_relative_error = float(
        np.linalg.norm(
            delta_x_array
            - delta_reconstruction
        )
        / np.linalg.norm(
            delta_x_array
        )
    )

    contrast_energy_fraction = float(
        np.linalg.norm(
            delta_reconstruction
        )
        ** 2
        / (
            np.linalg.norm(
                delta_x_array
            )
            ** 2
        )
    )

    print()
    print("Group-mean biofilm-minus-planktonic contrast projection")
    print(
        "  corr(delta X, M @ delta A):",
        round(
            contrast_corr,
            4,
        ),
    )
    print(
        "  relative projection error:",
        round(
            contrast_relative_error,
            4,
        ),
    )
    print(
        "  fraction contrast energy in PAO1 iModulon span:",
        round(
            contrast_energy_fraction,
            4,
        ),
    )
    print(
        "  max direct-vs-sample-derived delta activity difference:",
        format(
            delta_activity_agreement,
            ".3e",
        ),
    )

    # ----------------------------------------------------------
    # Biofilm replicate programme stability relative to the
    # full group-mean programme response
    # ----------------------------------------------------------

    full_delta_a = pd.Series(
        delta_a_direct,
        index=M.columns,
    )

    biofilm_stability_rows = []

    for sample in BIOFILM:
        sample_a = activity[
            sample
        ]

        biofilm_stability_rows.append(
            {
                "sample":
                    sample,

                "pearson_to_group_delta":
                    vector_corr(
                        sample_a.values,
                        full_delta_a.values,
                    ),

                "cosine_to_group_delta":
                    cosine(
                        sample_a.values,
                        full_delta_a.values,
                    ),
            }
        )

    stability = pd.DataFrame(
        biofilm_stability_rows
    )

    print()
    print("Biofilm replicate programme stability")
    print(
        stability
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    stability.to_csv(
        OUT
        / "biofilm_replicate_programme_stability.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Diagnostic interpretation
    # ----------------------------------------------------------

    bio_pairs = expression_pairs.loc[
        expression_pairs[
            "group"
        ]
        == "biofilm",
        "pearson_r",
    ]

    plank_pairs = expression_pairs.loc[
        expression_pairs[
            "group"
        ]
        == "planktonic",
        "pearson_r",
    ]

    best_bio_pair = float(
        bio_pairs.max()
    )

    best_plank_pair = float(
        plank_pairs.max()
    )

    max_filtered_bio = float(
        filter_table[
            "biofilm_mean_pairwise_r"
        ].max()
    )

    max_filtered_plank = float(
        filter_table[
            "planktonic_mean_pairwise_r"
        ].max()
    )

    n_below_500k = int(
        (
            depth[
                "projection_gene_counts"
            ]
            < 500000
        ).sum()
    )

    activity_bio_mean = mean_pairwise(
        activity,
        BIOFILM,
    )

    activity_plank_mean = mean_pairwise(
        activity,
        PLANKTONIC,
    )

    print()
    print("Diagnostic summary")
    print(
        "  samples below 500k projection-universe counts:",
        n_below_500k,
    )
    print(
        "  best biofilm expression pair r:",
        round(
            best_bio_pair,
            4,
        ),
    )
    print(
        "  best planktonic expression pair r:",
        round(
            best_plank_pair,
            4,
        ),
    )
    print(
        "  best filtered biofilm mean r:",
        round(
            max_filtered_bio,
            4,
        ),
    )
    print(
        "  best filtered planktonic mean r:",
        round(
            max_filtered_plank,
            4,
        ),
    )
    print(
        "  biofilm activity mean pairwise r:",
        round(
            activity_bio_mean,
            4,
        ),
    )
    print(
        "  planktonic activity mean pairwise r "
        "(centered diagnostic):",
        round(
            activity_plank_mean,
            4,
        ),
    )
    print(
        "  group-contrast projection correlation:",
        round(
            contrast_corr,
            4,
        ),
    )

    # These are diagnostic categories, not acceptance gates.
    flags = []

    if n_below_500k > 0:
        flags.append(
            "LOW_DEPTH_SAMPLE_PRESENT"
        )

    if (
        best_bio_pair >= 0.90
        or best_plank_pair >= 0.90
    ):
        flags.append(
            "POSSIBLE_SINGLE_REPLICATE_OUTLIER"
        )

    if (
        max_filtered_bio >= 0.75
        and max_filtered_plank >= 0.75
    ):
        flags.append(
            "LOW_ABUNDANCE_NOISE_CONTRIBUTES"
        )

    if (
        contrast_corr >= 0.40
        and activity_bio_mean >= 0.75
    ):
        flags.append(
            "GROUP_PROGRAMME_CONTRAST_RECOVERABLE"
        )

    if contrast_corr < 0.40:
        flags.append(
            "PAO1_IMODULON_SPAN_POOR_CONTRAST_FIT"
        )

    if not flags:
        flags.append(
            "NO_SINGLE_FAILURE_MODE_IDENTIFIED"
        )

    print()
    print("Diagnostic flags")

    for flag in flags:
        print(
            " ",
            flag,
        )

    pd.DataFrame(
        {
            "diagnostic_flag":
                flags
        }
    ).to_csv(
        OUT
        / "diagnostic_flags.csv",
        index=False,
    )

    summary_lines = [
        "# CF-State Phase 3 — Audit 02h",
        "",
        "## GSE191103 projection-failure diagnostic",
        "",
        "Audit 02g remains formally failed. "
        "This diagnostic does not alter or relax its gates.",
        "",
        "### Key metrics",
        "",
        "- Samples below 500k projection-universe counts: "
        + str(
            n_below_500k
        ),
        "- Best biofilm expression pair r: "
        + format(
            best_bio_pair,
            ".4f",
        ),
        "- Best planktonic expression pair r: "
        + format(
            best_plank_pair,
            ".4f",
        ),
        "- Best abundance-filtered biofilm mean r: "
        + format(
            max_filtered_bio,
            ".4f",
        ),
        "- Best abundance-filtered planktonic mean r: "
        + format(
            max_filtered_plank,
            ".4f",
        ),
        "- Biofilm activity mean pairwise r: "
        + format(
            activity_bio_mean,
            ".4f",
        ),
        "- Group-contrast projection correlation: "
        + format(
            contrast_corr,
            ".4f",
        ),
        "- Group-contrast projection energy fraction: "
        + format(
            contrast_energy_fraction,
            ".4f",
        ),
        "",
        "### Diagnostic flags",
        "",
    ]

    for flag in flags:
        summary_lines.append(
            "- "
            + flag
        )

    (
        OUT
        / "AUDIT_02H_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 02h complete."
    )
    print(
        "Summary:",
        OUT
        / "AUDIT_02H_SUMMARY.md",
    )


if __name__ == "__main__":
    main()
