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
    / "results/phase3/audit_03c_GSE55197_composite_ID_resolution"
    / "GSE55197_stationary_exponential_normalized_locus_counts.csv"
)

PROJECTION_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

TRANSFER = (
    ROOT
    / "results/phase3/audit_03c_GSE55197_composite_ID_resolution"
    / "GSE55197_exact_imodulon_transferability.csv"
)

GSE120602 = (
    ROOT
    / "results/phase2/audit_04_controlled_state_anchoring_GSE120602"
    / "controlled_contrast_vectors.csv"
)

PHASE2_ROOT = (
    ROOT
    / "results/phase2"
)

OUT = (
    ROOT
    / "results/phase3/audit_03d_GSE55197_stationary_crossstrain_projection"
)


EXPONENTIAL = [
    "PA14_EXP_1",
    "PA14_EXP_2",
]

STATIONARY = [
    "PA14_STAT_1",
    "PA14_STAT_2",
]

ALL_SAMPLES = (
    EXPONENTIAL
    + STATIONARY
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


def pair_r(matrix, a, b):
    return float(
        matrix[
            [
                a,
                b,
            ]
        ]
        .corr()
        .iloc[0, 1]
    )


def read_imodulon_vector(
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
            + ". Available: "
            + ", ".join(
                str(x)
                for x in table.columns
            )
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
    )

    vector = vector.reindex(
        imodulons
    )

    if vector.isna().any():
        raise RuntimeError(
            "Comparator does not contain all 116 iModulons: "
            + str(path)
        )

    return vector


def discover_GSE152295_stationary(
    imodulons,
):
    """
    Find the frozen Phase-2 GSE152295 stationary axis without
    assuming a brittle filename.

    Only files whose path contains GSE152295 are considered.
    """

    column_priority = [
        "stationary",
        "stationary_vs_ctrl",
        "stationary_vs_control",
        "stationary_vs_Ctrl",
        "stationary_minus_ctrl",
        "stationary_minus_control",
    ]

    candidates = []

    for path in PHASE2_ROOT.rglob(
        "*.csv"
    ):
        if "GSE152295" not in str(path):
            continue

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
            table[index_column]
            .astype(str)
            .str.strip()
        )

        overlap = len(
            observed.intersection(
                set(imodulons)
            )
        )

        if overlap < 110:
            continue

        for column in column_priority:
            if column in table.columns:
                candidates.append(
                    (
                        path,
                        column,
                    )
                )

    if not candidates:
        return (
            None,
            None,
            None,
        )

    # Prefer files from the final corrected/frozen audit.
    candidates = sorted(
        candidates,
        key=lambda item: (
            "10c" not in str(
                item[0]
            ).lower(),
            "10" not in str(
                item[0]
            ).lower(),
            len(
                str(
                    item[0]
                )
            ),
        ),
    )

    path, column = candidates[0]

    vector = read_imodulon_vector(
        path,
        column,
        imodulons,
    )

    return (
        vector,
        path,
        column,
    )


def permutation_alignment_p(
    a,
    b,
    n_perm=10000,
    seed=20261007,
):
    """
    Programme-label permutation diagnostic.

    Tests whether observed cosine alignment is larger than expected
    after randomly permuting programme labels of the comparator.

    This is a geometry diagnostic, not an independence-based
    biological hypothesis test.
    """

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
                null
                >= observed
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


def compare_vectors(
    target,
    reference,
    reference_name,
    programme_index,
):
    a = target.reindex(
        programme_index
    ).to_numpy(
        dtype=float
    )

    b = reference.reindex(
        programme_index
    ).to_numpy(
        dtype=float
    )

    observed_cosine, perm_p = (
        permutation_alignment_p(
            a,
            b,
        )
    )

    return {
        "reference":
            reference_name,

        "programme_set":
            programme_index.name,

        "n_programmes":
            len(
                programme_index
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
            observed_cosine,

        "sign_agreement":
            sign_agreement(
                a,
                b,
            ),

        "programme_label_permutation_p":
            perm_p,
    }


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 03d")
    print("GSE55197 PA14 stationary cross-strain iModulon validation")
    print("=" * 78)

    for path in [
        MODEL,
        COUNTS,
        PROJECTION_MAP,
        TRANSFER,
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
    # 1. Frozen PAO1 iModulon model
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

    A_train = json_df(
        raw["A"]
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
            "Unexpected M shape: "
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
        M.shape[0],
    )
    print(
        "  iModulons:",
        M.shape[1],
    )

    # ----------------------------------------------------------
    # 2. GSE55197 normalized locus counts
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

    for sample in ALL_SAMPLES:
        if sample not in counts.columns:
            raise RuntimeError(
                "Missing sample: "
                + sample
            )

    counts = counts[
        ALL_SAMPLES
    ]

    # ----------------------------------------------------------
    # 3. Frozen 5286-gene PA14 projection universe
    # ----------------------------------------------------------

    projection = pd.read_csv(
        PROJECTION_MAP
    )

    for column in [
        "pa14_gene",
        "pao1_gene",
    ]:
        projection[column] = (
            projection[column]
            .astype(str)
            .str.strip()
        )

    projection["length_bp"] = pd.to_numeric(
        projection["length_bp"],
        errors="raise",
    )

    if len(
        projection
    ) != 5286:
        raise RuntimeError(
            "Expected frozen 5286-gene projection map."
        )

    missing = [
        gene
        for gene in projection[
            "pa14_gene"
        ]
        if gene not in counts.index
    ]

    if missing:
        raise RuntimeError(
            "GSE55197 missing frozen projection genes: "
            + str(
                len(
                    missing
                )
            )
        )

    pa14_order = list(
        projection[
            "pa14_gene"
        ]
    )

    pao1_order = list(
        projection[
            "pao1_gene"
        ]
    )

    projection_counts = counts.reindex(
        pa14_order
    )

    lengths = pd.Series(
        projection[
            "length_bp"
        ].to_numpy(
            dtype=float
        ),
        index=pa14_order,
    )

    # ----------------------------------------------------------
    # 4. Projection-compatible TPM
    # ----------------------------------------------------------

    rpk = projection_counts.divide(
        lengths
        / 1000.0,
        axis=0,
    )

    denominators = rpk.sum(
        axis=0
    )

    if (
        denominators
        <= 0
    ).any():
        raise RuntimeError(
            "Invalid RPK denominator."
        )

    tpm = rpk.divide(
        denominators,
        axis=1,
    ) * 1_000_000.0

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    exp_expression_r = pair_r(
        log_tpm,
        "PA14_EXP_1",
        "PA14_EXP_2",
    )

    stat_expression_r = pair_r(
        log_tpm,
        "PA14_STAT_1",
        "PA14_STAT_2",
    )

    print()
    print("Projection-compatible expression")
    print(
        "  genes:",
        log_tpm.shape[0],
    )
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
    print(
        "  exponential replicate r:",
        round(
            exp_expression_r,
            4,
        ),
    )
    print(
        "  stationary replicate r:",
        round(
            stat_expression_r,
            4,
        ),
    )

    # Rename into PAO1 model space.
    log_tpm.index = (
        pao1_order
    )

    tpm.index = (
        pao1_order
    )

    projection_counts.index = (
        pao1_order
    )

    M_shared = M.reindex(
        pao1_order
    )

    if M_shared.isna().any().any():
        raise RuntimeError(
            "Missing M values after PAO1 alignment."
        )

    # ----------------------------------------------------------
    # 5. Exponential reference centering
    # ----------------------------------------------------------

    exp_reference = log_tpm[
        EXPONENTIAL
    ].mean(
        axis=1
    )

    X_centered = log_tpm.subtract(
        exp_reference,
        axis=0,
    )

    max_exp_gene_residual = float(
        X_centered[
            EXPONENTIAL
        ]
        .mean(
            axis=1
        )
        .abs()
        .max()
    )

    # ----------------------------------------------------------
    # 6. PA14 -> PAO1 iModulon projection
    # ----------------------------------------------------------

    M_array = M_shared.to_numpy(
        dtype=float
    )

    X_array = X_centered.to_numpy(
        dtype=float
    )

    pinv_M = np.linalg.pinv(
        M_array
    )

    A_array = (
        pinv_M
        @ X_array
    )

    activities = pd.DataFrame(
        A_array,
        index=M.columns,
        columns=X_centered.columns,
    )

    reconstruction = (
        M_array
        @ A_array
    )

    full_projection_corr = vector_corr(
        X_array.ravel(),
        reconstruction.ravel(),
    )

    full_relative_error = float(
        np.linalg.norm(
            X_array
            - reconstruction
        )
        / np.linalg.norm(
            X_array
        )
    )

    stat_activity_r = pair_r(
        activities,
        "PA14_STAT_1",
        "PA14_STAT_2",
    )

    # Exponential activity r is not meaningful as a QC gate
    # because two replicates centered around their own mean are
    # constrained to be oppositely signed.

    max_exp_activity_residual = float(
        activities[
            EXPONENTIAL
        ]
        .mean(
            axis=1
        )
        .abs()
        .max()
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

    print()
    print("Projection QC")
    print(
        "  stationary activity replicate r:",
        round(
            stat_activity_r,
            4,
        ),
    )
    print(
        "  max |mean exponential gene residual|:",
        format(
            max_exp_gene_residual,
            ".3e",
        ),
    )
    print(
        "  max |mean exponential activity|:",
        format(
            max_exp_activity_residual,
            ".3e",
        ),
    )
    print(
        "  full centered projection correlation:",
        round(
            full_projection_corr,
            4,
        ),
    )
    print(
        "  full relative projection error:",
        round(
            full_relative_error,
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
    # 7. The scientific target:
    #    stationary minus exponential group contrast
    # ----------------------------------------------------------

    delta_x = (
        log_tpm[
            STATIONARY
        ].mean(
            axis=1
        )
        - log_tpm[
            EXPONENTIAL
        ].mean(
            axis=1
        )
    )

    delta_a_direct = (
        pinv_M
        @ delta_x.to_numpy(
            dtype=float
        )
    )

    delta_a_samples = (
        activities[
            STATIONARY
        ].mean(
            axis=1
        )
        - activities[
            EXPONENTIAL
        ].mean(
            axis=1
        )
    )

    direct_sample_difference = float(
        np.max(
            np.abs(
                delta_a_direct
                - delta_a_samples.to_numpy(
                    dtype=float
                )
            )
        )
    )

    delta_reconstruction = (
        M_array
        @ delta_a_direct
    )

    contrast_projection_corr = vector_corr(
        delta_x.to_numpy(
            dtype=float
        ),
        delta_reconstruction,
    )

    contrast_relative_error = float(
        np.linalg.norm(
            delta_x.to_numpy(
                dtype=float
            )
            - delta_reconstruction
        )
        / np.linalg.norm(
            delta_x.to_numpy(
                dtype=float
            )
        )
    )

    contrast_energy_fraction = float(
        np.linalg.norm(
            delta_reconstruction
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

    pa14_stationary = pd.Series(
        delta_a_direct,
        index=M.columns,
        name="stationary_vs_exponential",
    )

    print()
    print("Stationary-minus-exponential contrast projection")
    print(
        "  corr(delta X, M @ delta A):",
        round(
            contrast_projection_corr,
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
        "  direct-vs-sample delta activity max difference:",
        format(
            direct_sample_difference,
            ".3e",
        ),
    )

    # ----------------------------------------------------------
    # 8. Programme transfer reliability
    # ----------------------------------------------------------

    transfer = pd.read_csv(
        TRANSFER
    )

    transfer["imodulon"] = (
        transfer["imodulon"]
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
            "Missing programme transfer values."
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

    robust_programmes = pd.Index(
        transfer.index[
            robust_mask
        ],
        name="ROBUST_TRANSFER",
    )

    print()
    print("Cross-strain programme sets")
    print(
        "  all programmes:",
        len(
            all_programmes
        ),
    )
    print(
        "  robust programmes:",
        len(
            robust_programmes
        ),
    )

    # ----------------------------------------------------------
    # 9. Largest PA14 stationary programme changes
    # ----------------------------------------------------------

    programme_table = pd.DataFrame(
        {
            "imodulon":
                imodulons,

            "stationary_vs_exponential":
                pa14_stationary.reindex(
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

    programme_table[
        "abs_delta"
    ] = programme_table[
        "stationary_vs_exponential"
    ].abs()

    programme_table = (
        programme_table
        .sort_values(
            "abs_delta",
            ascending=False,
        )
    )

    print()
    print("Largest PA14 stationary programme changes")
    print(
        programme_table[
            [
                "imodulon",
                "stationary_vs_exponential",
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
    # 10. Frozen PAO1 stationary anchors
    # ----------------------------------------------------------

    references = {}

    gse120602 = read_imodulon_vector(
        GSE120602,
        "stationary_vs_planktonic",
        imodulons,
    )

    references[
        "GSE120602_stationary_vs_planktonic"
    ] = gse120602

    (
        gse152295,
        gse152295_path,
        gse152295_column,
    ) = discover_GSE152295_stationary(
        imodulons
    )

    print()
    print("PAO1 stationary reference vectors")
    print(
        "  GSE120602:",
        GSE120602,
    )

    if gse152295 is not None:
        references[
            "GSE152295_stationary_vs_control"
        ] = gse152295

        print(
            "  GSE152295:",
            gse152295_path,
        )
        print(
            "  GSE152295 column:",
            gse152295_column,
        )
    else:
        print(
            "  WARNING: GSE152295 stationary vector "
            "not automatically found; comparison skipped."
        )

    # ----------------------------------------------------------
    # 11. Cross-strain stationary agreement
    # ----------------------------------------------------------

    comparison_rows = []

    for name, reference in references.items():
        comparison_rows.append(
            compare_vectors(
                pa14_stationary,
                reference,
                name,
                all_programmes,
            )
        )

        comparison_rows.append(
            compare_vectors(
                pa14_stationary,
                reference,
                name,
                robust_programmes,
            )
        )

    comparisons = pd.DataFrame(
        comparison_rows
    )

    print()
    print("Cross-strain stationary agreement")
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
    # 12. PAO1 anchor-to-anchor agreement
    #
    # If both independent PAO1 stationary anchors are available,
    # this gives useful context for how much agreement should
    # realistically be expected even within the same strain.
    # ----------------------------------------------------------

    pao1_anchor_rows = []

    if gse152295 is not None:
        for programme_set in [
            all_programmes,
            robust_programmes,
        ]:
            a = gse120602.reindex(
                programme_set
            ).to_numpy(
                dtype=float
            )

            b = gse152295.reindex(
                programme_set
            ).to_numpy(
                dtype=float
            )

            pao1_anchor_rows.append(
                {
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
            )

        pao1_anchor_comparison = pd.DataFrame(
            pao1_anchor_rows
        )

        print()
        print("PAO1 stationary anchor-to-anchor agreement")
        print(
            pao1_anchor_comparison
            .round(
                4
            )
            .to_string(
                index=False
            )
        )

    else:
        pao1_anchor_comparison = pd.DataFrame()

    # ----------------------------------------------------------
    # 13. Save outputs
    # ----------------------------------------------------------

    projection_counts.to_csv(
        OUT
        / "GSE55197_projection_gene_counts_PAO1_space.csv"
    )

    tpm.to_csv(
        OUT
        / "GSE55197_projection_compatible_TPM_PAO1_space.csv"
    )

    log_tpm.to_csv(
        OUT
        / "GSE55197_projection_log2TPM_plus1_PAO1_space.csv"
    )

    activities.to_csv(
        OUT
        / "GSE55197_PA14_imodulon_activities.csv"
    )

    programme_table.drop(
        columns=[
            "abs_delta"
        ]
    ).to_csv(
        OUT
        / "GSE55197_PA14_stationary_programme_vector.csv",
        index=False,
    )

    comparisons.to_csv(
        OUT
        / "GSE55197_crossstrain_stationary_agreement.csv",
        index=False,
    )

    if not pao1_anchor_comparison.empty:
        pao1_anchor_comparison.to_csv(
            OUT
            / "PAO1_stationary_anchor_agreement.csv",
            index=False,
        )

    # ----------------------------------------------------------
    # 14. Technical acceptance gates
    #
    # IMPORTANT:
    # Cross-strain agreement is NOT a gate. It is the scientific
    # result being tested.
    # ----------------------------------------------------------

    checks = {
        "projection_gene_count_5286":
            len(
                projection
            )
            == 5286,

        "exponential_expression_r_ge_090":
            exp_expression_r
            >= 0.90,

        "stationary_expression_r_ge_090":
            stat_expression_r
            >= 0.90,

        "stationary_activity_r_ge_075":
            stat_activity_r
            >= 0.75,

        "exponential_reference_centered":
            max_exp_activity_residual
            < 1e-8,

        "finite_projection":
            bool(
                np.isfinite(
                    A_array
                ).all()
            ),

        "contrast_projection_corr_ge_040":
            contrast_projection_corr
            >= 0.40,

        "contrast_energy_fraction_ge_015":
            contrast_energy_fraction
            >= 0.15,

        "robust_programmes_ge_90":
            len(
                robust_programmes
            )
            >= 90,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Technical audit checks")

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
        "GSE55197_STATIONARY_CROSSSTRAIN_PROJECTION_ACCEPTED"
        if passed
        else "GSE55197_STATIONARY_PROJECTION_REVIEW_REQUIRED"
    )

    print()
    print(
        "TECHNICAL STATUS:",
        status,
    )

    print()
    print(
        "NOTE: cross-strain stationary correlations are "
        "scientific outputs, not acceptance gates."
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
        "# CF-State Phase 3 — Audit 03d",
        "",
        "## GSE55197 PA14 stationary cross-strain validation",
        "",
        "Contrast:",
        "",
        "**PA14 stationary minus PA14 exponential**",
        "",
        "The experiment was projected through the frozen "
        "5,286-gene PA14↔PAO1 orthologue universe into the "
        "116-programme PAO1 aeruPRECISE representation.",
        "",
        "Technical projection acceptance is evaluated separately "
        "from the biological cross-strain agreement.",
        "",
        "- Exponential expression replicate r: "
        + format(
            exp_expression_r,
            ".4f",
        ),
        "- Stationary expression replicate r: "
        + format(
            stat_expression_r,
            ".4f",
        ),
        "- Stationary activity replicate r: "
        + format(
            stat_activity_r,
            ".4f",
        ),
        "- Stationary contrast projection correlation: "
        + format(
            contrast_projection_corr,
            ".4f",
        ),
        "- Stationary contrast energy fraction: "
        + format(
            contrast_energy_fraction,
            ".4f",
        ),
        "- Robust transferable programmes: "
        + str(
            len(
                robust_programmes
            )
        ),
        "",
        "## Technical decision",
        "",
        status,
        "",
        "Cross-strain stationary agreement is reported in "
        "`GSE55197_crossstrain_stationary_agreement.csv` and is "
        "not used to determine technical acceptance.",
    ]

    (
        OUT
        / "AUDIT_03D_SUMMARY.md"
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
            "Failed technical checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
