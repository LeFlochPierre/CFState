from pathlib import Path
from io import StringIO
import json

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


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

TRANSFER = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "final_projection_imodulon_transferability.csv"
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

GSE299861_DIR = (
    ROOT
    / "results/phase2/audit_12f_GSE299861_canonical_imodulon_challenge"
)

OUT = (
    ROOT
    / "results/phase3/audit_02g_GSE191103_final_PA14_projection"
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


def mean_pairwise_corr(matrix, columns):
    corr = matrix[
        columns
    ].corr()

    values = []

    for i in range(
        len(columns)
    ):
        for j in range(
            i + 1,
            len(columns),
        ):
            values.append(
                float(
                    corr.iloc[
                        i,
                        j,
                    ]
                )
            )

    return float(
        np.mean(
            values
        )
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


def read_vector(path, value_column, imodulons):
    table = pd.read_csv(
        path
    )

    if "imodulon" in table.columns:
        index_column = "imodulon"
    elif "iModulon" in table.columns:
        index_column = "iModulon"
    elif "Unnamed: 0" in table.columns:
        index_column = "Unnamed: 0"
    else:
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

    table[
        index_column
    ] = (
        table[
            index_column
        ]
        .astype(str)
        .str.strip()
    )

    vector = pd.to_numeric(
        table.set_index(
            index_column
        )[
            value_column
        ],
        errors="raise",
    )

    vector = vector.reindex(
        imodulons
    )

    if vector.isna().any():
        raise RuntimeError(
            str(path)
            + " does not contain all 116 iModulons."
        )

    return vector


def find_gse299861_vector(imodulons):
    if not GSE299861_DIR.exists():
        return None, None

    preferred_columns = [
        "delta_biofilm_minus_planktonic",
        "biofilm_vs_planktonic",
    ]

    candidates = sorted(
        GSE299861_DIR.glob(
            "*.csv"
        )
    )

    for path in candidates:
        try:
            table = pd.read_csv(
                path
            )
        except Exception:
            continue

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
            continue

        observed = set(
            table[
                index_column
            ]
            .astype(str)
            .str.strip()
        )

        overlap = len(
            observed.intersection(
                set(
                    imodulons
                )
            )
        )

        if overlap < 100:
            continue

        for value_column in preferred_columns:
            if value_column in table.columns:
                vector = read_vector(
                    path,
                    value_column,
                    imodulons,
                )

                return vector, path

    return None, None


def comparison_row(
    target,
    reference,
    name,
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

    return {
        "comparison":
            name,

        "programme_set":
            programme_set.name,

        "n_programmes":
            len(
                programme_set
            ),

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
            cosine(
                a,
                b,
            ),

        "sign_agreement":
            sign_agreement(
                a,
                b,
            ),
    }


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02g")
    print("Final GSE191103 WT PA14 cross-strain iModulon projection")
    print("=" * 78)

    for path in [
        MODEL,
        COUNTS,
        PROJECTION_MAP,
        TRANSFER,
        GSE203348,
        GSE120602,
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
    # 1. Frozen aeruPRECISE model
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

    A_train = json_df(
        raw[
            "A"
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

    A_train.index = (
        A_train.index
        .astype(str)
        .str.strip()
    )

    A_train = A_train.reindex(
        M.columns
    )

    if M.shape != (
        5572,
        116,
    ):
        raise RuntimeError(
            "Unexpected aeruPRECISE M shape: "
            + str(
                M.shape
            )
        )

    imodulons = list(
        M.columns
    )

    print()
    print("Frozen PAO1 representation")
    print(
        "  genes:",
        M.shape[
            0
        ],
    )
    print(
        "  iModulons:",
        M.shape[
            1
        ],
    )

    # ----------------------------------------------------------
    # 2. Frozen PA14 projection universe
    # ----------------------------------------------------------

    projection_map = pd.read_csv(
        PROJECTION_MAP
    )

    required_map_columns = {
        "pao1_gene",
        "pa14_gene",
        "length_bp",
    }

    missing_columns = (
        required_map_columns
        - set(
            projection_map.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Projection map missing columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
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

    projection_map[
        "length_bp"
    ] = pd.to_numeric(
        projection_map[
            "length_bp"
        ],
        errors="raise",
    )

    if projection_map[
        "pao1_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate PAO1 genes in frozen projection map."
        )

    if projection_map[
        "pa14_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate PA14 genes in frozen projection map."
        )

    if (
        projection_map[
            "length_bp"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "Non-positive gene length in projection map."
        )

    print()
    print("Frozen PA14 projection universe")
    print(
        "  genes:",
        len(
            projection_map
        ),
    )
    print(
        "  full PAO1 model coverage:",
        format(
            len(
                projection_map
            )
            / M.shape[
                0
            ],
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # 3. Raw counts restricted to the frozen projection universe
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

    for sample in (
        BIOFILM
        + PLANKTONIC
    ):
        if sample not in counts.columns:
            raise RuntimeError(
                "Missing expected sample: "
                + sample
            )

    counts = counts[
        BIOFILM
        + PLANKTONIC
    ]

    pa14_order = list(
        projection_map[
            "pa14_gene"
        ]
    )

    missing_expression = [
        gene
        for gene in pa14_order
        if gene not in counts.index
    ]

    if missing_expression:
        raise RuntimeError(
            "Frozen projection genes absent from expression matrix: "
            + str(
                len(
                    missing_expression
                )
            )
        )

    counts_projection = counts.reindex(
        pa14_order
    ).copy()

    # ----------------------------------------------------------
    # 4. Projection-compatible TPM
    #
    # Denominator is deliberately restricted to the same frozen
    # 5286 model-compatible genes in every sample.
    # ----------------------------------------------------------

    lengths = pd.Series(
        projection_map[
            "length_bp"
        ].to_numpy(
            dtype=float
        ),
        index=projection_map[
            "pa14_gene"
        ],
    )

    lengths = lengths.reindex(
        pa14_order
    )

    rpk = counts_projection.divide(
        lengths
        / 1000.0,
        axis=0,
    )

    rpk_sums = rpk.sum(
        axis=0
    )

    if (
        rpk_sums
        <= 0
    ).any():
        raise RuntimeError(
            "Invalid projection-universe RPK denominator."
        )

    tpm = rpk.divide(
        rpk_sums,
        axis=1,
    ) * 1_000_000.0

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    print()
    print("Projection-compatible normalization")
    print(
        "  TPM sum minimum:",
        round(
            float(
                tpm.sum(
                    axis=0
                ).min()
            ),
            3,
        ),
    )
    print(
        "  TPM sum maximum:",
        round(
            float(
                tpm.sum(
                    axis=0
                ).max()
            ),
            3,
        ),
    )

    plank_expression_r = mean_pairwise_corr(
        log_tpm,
        PLANKTONIC,
    )

    biofilm_expression_r = mean_pairwise_corr(
        log_tpm,
        BIOFILM,
    )

    print(
        "  planktonic replicate mean r:",
        round(
            plank_expression_r,
            4,
        ),
    )
    print(
        "  biofilm replicate mean r:",
        round(
            biofilm_expression_r,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 5. Rename PA14 rows into frozen PAO1 model space
    # ----------------------------------------------------------

    log_tpm_pao1 = log_tpm.copy()

    log_tpm_pao1.index = list(
        projection_map[
            "pao1_gene"
        ]
    )

    if log_tpm_pao1.index.duplicated().any():
        raise RuntimeError(
            "Duplicate PAO1 genes after mapping."
        )

    pao1_order = [
        gene
        for gene in M.index
        if gene in log_tpm_pao1.index
    ]

    log_tpm_pao1 = log_tpm_pao1.reindex(
        pao1_order
    )

    M_shared = M.reindex(
        pao1_order
    )

    model_coverage = (
        len(
            pao1_order
        )
        / M.shape[
            0
        ]
    )

    print()
    print("Cross-strain model alignment")
    print(
        "  shared genes:",
        len(
            pao1_order
        ),
    )
    print(
        "  model coverage:",
        format(
            model_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # 6. Within-experiment planktonic centering
    # ----------------------------------------------------------

    plank_reference = log_tpm_pao1[
        PLANKTONIC
    ].mean(
        axis=1
    )

    X_centered = log_tpm_pao1.subtract(
        plank_reference,
        axis=0,
    )

    max_gene_reference_residual = float(
        X_centered[
            PLANKTONIC
        ].mean(
            axis=1
        ).abs().max()
    )

    # ----------------------------------------------------------
    # 7. iModulon projection
    # ----------------------------------------------------------

    M_array = M_shared.to_numpy(
        dtype=float
    )

    X_array = X_centered.to_numpy(
        dtype=float
    )

    A_array = (
        np.linalg.pinv(
            M_array
        )
        @ X_array
    )

    A_external = pd.DataFrame(
        A_array,
        index=M.columns,
        columns=X_centered.columns,
    )

    reconstructed = (
        M_array
        @ A_array
    )

    projection_corr = float(
        np.corrcoef(
            X_array.ravel(),
            reconstructed.ravel(),
        )[
            0,
            1
        ]
    )

    relative_projection_error = float(
        np.linalg.norm(
            X_array
            - reconstructed
        )
        / np.linalg.norm(
            X_array
        )
    )

    training_sd = float(
        np.nanstd(
            A_train.to_numpy(
                dtype=float
            )
        )
    )

    external_sd = float(
        np.nanstd(
            A_array
        )
    )

    scale_ratio = (
        external_sd
        / training_sd
    )

    centered_plank_activity_r = mean_pairwise_corr(
        A_external,
        PLANKTONIC,
    )

    biofilm_activity_r = mean_pairwise_corr(
        A_external,
        BIOFILM,
    )

    max_activity_reference_residual = float(
        A_external[
            PLANKTONIC
        ].mean(
            axis=1
        ).abs().max()
    )

    print()
    print("Projection QC")
    print(
        "  centered planktonic activity mean r "
        "(diagnostic only):",
        round(
            centered_plank_activity_r,
            4,
        ),
    )
    print(
        "  biofilm activity replicate mean r:",
        round(
            biofilm_activity_r,
            4,
        ),
    )
    print(
        "  max |mean planktonic gene residual|:",
        format(
            max_gene_reference_residual,
            ".3e",
        ),
    )
    print(
        "  max |mean planktonic activity|:",
        format(
            max_activity_reference_residual,
            ".3e",
        ),
    )
    print(
        "  corr(centered X, M@A):",
        round(
            projection_corr,
            4,
        ),
    )
    print(
        "  relative projection error:",
        round(
            relative_projection_error,
            4,
        ),
    )
    print(
        "  external/training activity SD ratio:",
        round(
            scale_ratio,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 8. PA14 biofilm programme vector
    # ----------------------------------------------------------

    delta = (
        A_external[
            BIOFILM
        ].mean(
            axis=1
        )
        - A_external[
            PLANKTONIC
        ].mean(
            axis=1
        )
    )

    delta.name = (
        "delta_biofilm_minus_planktonic"
    )

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

    if transfer[
        [
            "weighted_coverage_absM",
            "energy_coverage_M2",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Missing transferability values."
        )

    # Conservative sensitivity set defined BEFORE inspecting
    # PA14 biological results.
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

    robust_programmes = pd.Index(
        transfer.index[
            robust_mask
        ],
        name="ROBUST_TRANSFER",
    )

    all_programmes = pd.Index(
        imodulons,
        name="ALL_116",
    )

    print()
    print("Cross-strain programme reliability")
    print(
        "  all programmes:",
        len(
            all_programmes
        ),
    )
    print(
        "  robust sensitivity programmes:",
        len(
            robust_programmes
        ),
    )
    print(
        "  excluded from robust sensitivity set:",
        len(
            all_programmes
        )
        - len(
            robust_programmes
        ),
    )

    programme = pd.DataFrame(
        {
            "imodulon":
                imodulons,

            "delta_biofilm_minus_planktonic":
                delta.reindex(
                    imodulons
                ).values,

            "weighted_transfer_coverage":
                transfer[
                    "weighted_coverage_absM"
                ].values,

            "energy_transfer_coverage":
                transfer[
                    "energy_coverage_M2"
                ].values,

            "robust_transfer":
                robust_mask.values,
        }
    )

    programme[
        "abs_delta"
    ] = programme[
        "delta_biofilm_minus_planktonic"
    ].abs()

    programme = programme.sort_values(
        "abs_delta",
        ascending=False,
    )

    print()
    print("Largest PA14 biofilm programme changes")
    print(
        programme[
            [
                "imodulon",
                "delta_biofilm_minus_planktonic",
                "weighted_transfer_coverage",
                "energy_transfer_coverage",
                "robust_transfer",
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
    # 9. Frozen Phase-2 comparator vectors
    # ----------------------------------------------------------

    references = {}

    references[
        "GSE203348"
    ] = read_vector(
        GSE203348,
        "delta_biofilm_minus_planktonic",
        imodulons,
    )

    references[
        "GSE120602"
    ] = read_vector(
        GSE120602,
        "biofilm_vs_planktonic",
        imodulons,
    )

    gse299861_vector, gse299861_path = (
        find_gse299861_vector(
            imodulons
        )
    )

    if gse299861_vector is not None:
        references[
            "GSE299861"
        ] = gse299861_vector

        print()
        print(
            "GSE299861 comparison vector:",
            gse299861_path,
        )
    else:
        print()
        print(
            "WARNING: GSE299861 programme vector "
            "not found automatically; comparison skipped."
        )

    # ----------------------------------------------------------
    # 10. Cross-study agreement:
    #     all 116 vs conservative robust transfer set
    # ----------------------------------------------------------

    comparison_rows = []

    for name, reference in references.items():
        comparison_rows.append(
            comparison_row(
                delta,
                reference,
                "GSE191103_PA14_vs_" + name,
                all_programmes,
            )
        )

        comparison_rows.append(
            comparison_row(
                delta,
                reference,
                "GSE191103_PA14_vs_" + name,
                robust_programmes,
            )
        )

    comparisons = pd.DataFrame(
        comparison_rows
    )

    print()
    print("Cross-study biofilm agreement")
    print(
        comparisons
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 11. Save
    # ----------------------------------------------------------

    tpm.to_csv(
        OUT
        / "GSE191103_projection_compatible_TPM.csv"
    )

    log_tpm_pao1.to_csv(
        OUT
        / "GSE191103_projection_compatible_log2TPM_plus1_PAO1_space.csv"
    )

    A_external.to_csv(
        OUT
        / "GSE191103_PA14_imodulon_activities.csv"
    )

    programme.drop(
        columns=[
            "abs_delta"
        ]
    ).to_csv(
        OUT
        / "GSE191103_PA14_biofilm_programme_vector.csv",
        index=False,
    )

    comparisons.to_csv(
        OUT
        / "GSE191103_cross_study_biofilm_agreement.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "imodulon":
                imodulons,

            "robust_transfer":
                [
                    imodulon
                    in set(
                        robust_programmes
                    )
                    for imodulon
                    in imodulons
                ],

            "weighted_transfer_coverage":
                transfer[
                    "weighted_coverage_absM"
                ].values,

            "energy_transfer_coverage":
                transfer[
                    "energy_coverage_M2"
                ].values,
        }
    ).to_csv(
        OUT
        / "PA14_programme_reliability_flags.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 12. Audit gates
    # ----------------------------------------------------------

    checks = {
        "projection_gene_count_5286":
            len(
                projection_map
            )
            == 5286,

        "model_gene_coverage_ge_94pct":
            model_coverage
            >= 0.94,

        "planktonic_expression_replicate_r_ge_075":
            plank_expression_r
            >= 0.75,

        "biofilm_expression_replicate_r_ge_075":
            biofilm_expression_r
            >= 0.75,

        "planktonic_reference_centered":
            max_activity_reference_residual
            < 1e-8,

        "finite_projection":
            bool(
                np.isfinite(
                    A_array
                ).all()
            ),

        "projection_corr_ge_040":
            projection_corr
            >= 0.40,

        "biofilm_activity_replicate_r_ge_075":
            biofilm_activity_r
            >= 0.75,

        "robust_transfer_programmes_ge_90":
            len(
                robust_programmes
            )
            >= 90,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
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
        "# CF-State Phase 3 — Audit 02g",
        "",
        "## Final GSE191103 WT PA14 cross-strain iModulon projection",
        "",
        "Projection used the frozen 5,286-gene PA14 projection-compatible "
        "universe accepted in Audit 02f.",
        "",
        "Raw unique-gene counts were length-normalized and scaled within "
        "the identical 5,286-gene model-compatible universe for every "
        "sample. These values are therefore projection-compatible TPM, "
        "not whole-transcriptome TPM.",
        "",
        "- Projection genes: "
        + str(
            len(
                projection_map
            )
        ),
        "- PAO1 model coverage: "
        + format(
            model_coverage,
            ".2%",
        ),
        "- Planktonic expression replicate r: "
        + format(
            plank_expression_r,
            ".4f",
        ),
        "- Biofilm expression replicate r: "
        + format(
            biofilm_expression_r,
            ".4f",
        ),
        "- Projection correlation: "
        + format(
            projection_corr,
            ".4f",
        ),
        "- Relative projection error: "
        + format(
            relative_projection_error,
            ".4f",
        ),
        "- External/training activity SD ratio: "
        + format(
            scale_ratio,
            ".4f",
        ),
        "- Biofilm activity replicate r: "
        + format(
            biofilm_activity_r,
            ".4f",
        ),
        "- Robust cross-strain sensitivity programmes: "
        + str(
            len(
                robust_programmes
            )
        ),
        "",
        "Robust programmes were pre-defined as retaining at least 90% "
        "absolute-M weight and at least 80% squared-M energy.",
        "",
        "## Decision",
        "",
        "PASS: "
        + str(
            passed
        ),
    ]

    (
        OUT
        / "AUDIT_02G_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Phase 3 Audit 02g:",
        "PASS"
        if passed
        else "FAIL",
    )
    print(
        "Summary:",
        OUT
        / "AUDIT_02G_SUMMARY.md",
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
