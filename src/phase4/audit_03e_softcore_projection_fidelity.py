from pathlib import Path
from io import StringIO
import json

import numpy as np
import pandas as pd


ROOT = Path.cwd()

MODEL = (
    ROOT
    / "external/aeruprecise/precise411"
    / "paeru_116_iModulon.json"
)

REPLICATION_MAP = (
    ROOT
    / "results/phase4/audit_03d_PRJEB24688_published_count_intake"
    / "PRJEB24688_frozen_replication_gene_map.csv"
)

PHASE3_PRIMARY = (
    ROOT
    / "results/phase3/audit_04_freeze_crossstrain_state_framework"
    / "phase3_robust_transferable_imodulons.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_03e_softcore_projection_fidelity"
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


def fit_linear_calibration(x, y):
    """
    Fit y = intercept + slope * x.

    x = masked activity
    y = full-model activity
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    design = np.column_stack(
        [
            np.ones(
                len(x)
            ),
            x,
        ]
    )

    beta = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )[0]

    intercept = float(
        beta[
            0
        ]
    )

    slope = float(
        beta[
            1
        ]
    )

    return (
        intercept,
        slope,
    )


def deterministic_folds(n_samples, n_folds=5):
    """
    Deterministic fold assignment.

    No random seed dependence.
    """

    folds = np.arange(
        n_samples
    ) % n_folds

    return folds


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 03e")
    print("Empirical fidelity of the PRJEB24688 soft-core projection mask")
    print("=" * 78)

    for path in [
        MODEL,
        REPLICATION_MAP,
        PHASE3_PRIMARY,
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
    # 1. Load full aeruPRECISE model
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

    X = json_df(
        raw[
            "X"
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

    X.index = (
        X.index
        .astype(str)
        .str.strip()
    )

    X.columns = (
        X.columns
        .astype(str)
        .str.strip()
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

    if X.shape[
        0
    ] != 5572:
        raise RuntimeError(
            "Unexpected X gene dimension: "
            + str(
                X.shape
            )
        )

    if list(
        M.index
    ) != list(
        X.index
    ):
        X = X.reindex(
            M.index
        )

    if X.isna().any().any():
        raise RuntimeError(
            "X could not be aligned exactly to M."
        )

    imodulons = list(
        M.columns
    )

    samples = list(
        X.columns
    )

    print()
    print("Full aeruPRECISE model")
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
    print(
        "  transcriptomes:",
        X.shape[
            1
        ],
    )

    # ----------------------------------------------------------
    # 2. Frozen PRJEB24688 mask
    # ----------------------------------------------------------

    replication_map = pd.read_csv(
        REPLICATION_MAP
    )

    if "pao1_gene" not in replication_map.columns:
        raise RuntimeError(
            "Replication map missing pao1_gene."
        )

    mask_genes = (
        replication_map[
            "pao1_gene"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(
        mask_genes
    ) != len(
        set(
            mask_genes
        )
    ):
        raise RuntimeError(
            "Replication mask contains duplicate PAO1 genes."
        )

    missing_from_model = [
        gene
        for gene in mask_genes
        if gene not in M.index
    ]

    if missing_from_model:
        raise RuntimeError(
            "Replication genes absent from aeruPRECISE model: "
            + ", ".join(
                missing_from_model[
                    :20
                ]
            )
        )

    mask_genes = [
        gene
        for gene in M.index
        if gene in set(
            mask_genes
        )
    ]

    print()
    print("PRJEB24688 soft-core mask")
    print(
        "  genes:",
        len(
            mask_genes
        ),
    )
    print(
        "  fraction of full model:",
        format(
            len(
                mask_genes
            )
            / len(
                M.index
            ),
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # 3. Frozen Phase-3 primary programme set
    # ----------------------------------------------------------

    primary_table = pd.read_csv(
        PHASE3_PRIMARY
    )

    if "imodulon" not in primary_table.columns:
        raise RuntimeError(
            "Phase-3 primary file missing imodulon column."
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

    sensitivity = [
        imodulon
        for imodulon in imodulons
        if imodulon not in primary_set
    ]

    if len(
        primary
    ) != 111:
        raise RuntimeError(
            "Expected 111 Phase-3 primary programmes."
        )

    print()
    print("Interpretation hierarchy")
    print(
        "  primary:",
        len(
            primary
        ),
    )
    print(
        "  sensitivity-only:",
        len(
            sensitivity
        ),
    )

    # ----------------------------------------------------------
    # 4. Full versus masked projection operators
    #
    # We compare:
    #
    #   pinv(M_full)   @ X_full
    #
    # against
    #
    #   pinv(M_masked) @ X_masked
    #
    # across all 411 aeruPRECISE transcriptomes.
    #
    # This directly measures how much the gene mask changes the
    # projection coordinates.
    # ----------------------------------------------------------

    M_full = M.to_numpy(
        dtype=float
    )

    X_full = X.to_numpy(
        dtype=float
    )

    M_mask = M.loc[
        mask_genes
    ].to_numpy(
        dtype=float
    )

    X_mask = X.loc[
        mask_genes
    ].to_numpy(
        dtype=float
    )

    rank_full = int(
        np.linalg.matrix_rank(
            M_full
        )
    )

    rank_mask = int(
        np.linalg.matrix_rank(
            M_mask
        )
    )

    print()
    print("Projection matrix rank")
    print(
        "  full:",
        rank_full,
    )
    print(
        "  masked:",
        rank_mask,
    )

    full_activity = (
        np.linalg.pinv(
            M_full
        )
        @ X_full
    )

    masked_activity = (
        np.linalg.pinv(
            M_mask
        )
        @ X_mask
    )

    if full_activity.shape != (
        116,
        len(
            samples
        ),
    ):
        raise RuntimeError(
            "Unexpected full activity shape."
        )

    if masked_activity.shape != full_activity.shape:
        raise RuntimeError(
            "Masked activity shape mismatch."
        )

    # ----------------------------------------------------------
    # 5. Per-programme fidelity before calibration
    # ----------------------------------------------------------

    programme_rows = []

    for i, imodulon in enumerate(
        imodulons
    ):
        full_vector = full_activity[
            i,
            :
        ]

        masked_vector = masked_activity[
            i,
            :
        ]

        full_centered = (
            full_vector
            - np.mean(
                full_vector
            )
        )

        masked_centered = (
            masked_vector
            - np.mean(
                masked_vector
            )
        )

        r = pearson(
            full_vector,
            masked_vector,
        )

        full_sd = float(
            np.std(
                full_centered,
                ddof=1,
            )
        )

        raw_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        full_centered
                        - masked_centered
                    )
                    ** 2
                )
            )
        )

        relative_rmse = (
            raw_rmse
            / full_sd
            if full_sd > 0
            else np.nan
        )

        intercept, slope = fit_linear_calibration(
            masked_vector,
            full_vector,
        )

        calibrated = (
            intercept
            + slope
            * masked_vector
        )

        calibrated_centered = (
            calibrated
            - np.mean(
                calibrated
            )
        )

        calibrated_rmse = float(
            np.sqrt(
                np.mean(
                    (
                        full_centered
                        - calibrated_centered
                    )
                    ** 2
                )
            )
        )

        calibrated_relative_rmse = (
            calibrated_rmse
            / full_sd
            if full_sd > 0
            else np.nan
        )

        programme_rows.append(
            {
                "imodulon":
                    imodulon,

                "phase3_primary":
                    imodulon
                    in primary_set,

                "pearson_full_vs_masked":
                    r,

                "raw_centered_relative_RMSE":
                    relative_rmse,

                "calibration_intercept":
                    intercept,

                "calibration_slope":
                    slope,

                "calibrated_centered_relative_RMSE":
                    calibrated_relative_rmse,
            }
        )

    programme = pd.DataFrame(
        programme_rows
    )

    # ----------------------------------------------------------
    # 6. Five-fold out-of-fold calibration
    #
    # This tests whether any scale correction generalizes rather
    # than merely fitting the same 411 transcriptomes.
    # ----------------------------------------------------------

    n_samples = len(
        samples
    )

    folds = deterministic_folds(
        n_samples,
        n_folds=5,
    )

    oof_calibrated = np.full_like(
        masked_activity,
        np.nan,
        dtype=float,
    )

    fold_rows = []

    for fold in range(
        5
    ):
        train_mask = (
            folds
            != fold
        )

        test_mask = (
            folds
            == fold
        )

        n_train = int(
            train_mask.sum()
        )

        n_test = int(
            test_mask.sum()
        )

        for i, imodulon in enumerate(
            imodulons
        ):
            full_train = full_activity[
                i,
                train_mask
            ]

            masked_train = masked_activity[
                i,
                train_mask
            ]

            intercept, slope = fit_linear_calibration(
                masked_train,
                full_train,
            )

            masked_test = masked_activity[
                i,
                test_mask
            ]

            predictions = (
                intercept
                + slope
                * masked_test
            )

            oof_calibrated[
                i,
                test_mask
            ] = predictions

        fold_rows.append(
            {
                "fold":
                    fold,
                "n_train":
                    n_train,
                "n_test":
                    n_test,
            }
        )

    if np.isnan(
        oof_calibrated
    ).any():
        raise RuntimeError(
            "Out-of-fold calibration contains missing predictions."
        )

    oof_rows = []

    for i, imodulon in enumerate(
        imodulons
    ):
        full_vector = full_activity[
            i,
            :
        ]

        predicted = oof_calibrated[
            i,
            :
        ]

        full_centered = (
            full_vector
            - np.mean(
                full_vector
            )
        )

        predicted_centered = (
            predicted
            - np.mean(
                predicted
            )
        )

        full_sd = float(
            np.std(
                full_centered,
                ddof=1,
            )
        )

        rmse = float(
            np.sqrt(
                np.mean(
                    (
                        full_centered
                        - predicted_centered
                    )
                    ** 2
                )
            )
        )

        relative_rmse = (
            rmse
            / full_sd
            if full_sd > 0
            else np.nan
        )

        oof_rows.append(
            {
                "imodulon":
                    imodulon,

                "phase3_primary":
                    imodulon
                    in primary_set,

                "OOF_pearson":
                    pearson(
                        full_vector,
                        predicted,
                    ),

                "OOF_centered_relative_RMSE":
                    relative_rmse,
            }
        )

    oof = pd.DataFrame(
        oof_rows
    )

    programme = programme.merge(
        oof,
        on=[
            "imodulon",
            "phase3_primary",
        ],
        how="left",
        validate="one_to_one",
    )

    # ----------------------------------------------------------
    # 7. Sample-wise activity-vector fidelity
    #
    # Since downstream CF-State analyses use programme-vector
    # direction, compare each sample's centred 116-dimensional
    # vector between full and masked projections.
    #
    # Center each iModulon across transcriptomes first so constant
    # mask-dependent offsets do not influence contrast geometry.
    # ----------------------------------------------------------

    full_centered = (
        full_activity
        - full_activity.mean(
            axis=1,
            keepdims=True,
        )
    )

    masked_centered = (
        masked_activity
        - masked_activity.mean(
            axis=1,
            keepdims=True,
        )
    )

    oof_centered = (
        oof_calibrated
        - oof_calibrated.mean(
            axis=1,
            keepdims=True,
        )
    )

    sample_rows = []

    primary_indices = [
        imodulons.index(
            imodulon
        )
        for imodulon in primary
    ]

    for j, sample in enumerate(
        samples
    ):
        full_all = full_centered[
            :,
            j
        ]

        masked_all = masked_centered[
            :,
            j
        ]

        oof_all = oof_centered[
            :,
            j
        ]

        full_primary = full_centered[
            primary_indices,
            j,
        ]

        masked_primary = masked_centered[
            primary_indices,
            j,
        ]

        oof_primary = oof_centered[
            primary_indices,
            j,
        ]

        sample_rows.append(
            {
                "sample":
                    sample,

                "raw_cosine_ALL116":
                    cosine(
                        full_all,
                        masked_all,
                    ),

                "OOF_calibrated_cosine_ALL116":
                    cosine(
                        full_all,
                        oof_all,
                    ),

                "raw_cosine_PRIMARY111":
                    cosine(
                        full_primary,
                        masked_primary,
                    ),

                "OOF_calibrated_cosine_PRIMARY111":
                    cosine(
                        full_primary,
                        oof_primary,
                    ),

                "raw_pearson_PRIMARY111":
                    pearson(
                        full_primary,
                        masked_primary,
                    ),

                "OOF_calibrated_pearson_PRIMARY111":
                    pearson(
                        full_primary,
                        oof_primary,
                    ),
            }
        )

    sample_fidelity = pd.DataFrame(
        sample_rows
    )

    # ----------------------------------------------------------
    # 8. Global contrast-space fidelity
    #
    # Flatten centred activities across all transcriptomes.
    # This asks whether the full contrast geometry is retained.
    # ----------------------------------------------------------

    global_raw_all = pearson(
        full_centered.ravel(),
        masked_centered.ravel(),
    )

    global_oof_all = pearson(
        full_centered.ravel(),
        oof_centered.ravel(),
    )

    global_raw_primary = pearson(
        full_centered[
            primary_indices,
            :
        ].ravel(),
        masked_centered[
            primary_indices,
            :
        ].ravel(),
    )

    global_oof_primary = pearson(
        full_centered[
            primary_indices,
            :
        ].ravel(),
        oof_centered[
            primary_indices,
            :
        ].ravel(),
    )

    # ----------------------------------------------------------
    # 9. Empirically stable programmes
    #
    # This does NOT replace the Phase-3 111-programme hierarchy.
    #
    # It creates a second mask-specific flag:
    #
    #   Phase-3 primary
    #   AND
    #   faithfully recoverable through the Rossi soft-core mask.
    # ----------------------------------------------------------

    programme[
        "softcore_empirically_stable"
    ] = (
        (
            programme[
                "OOF_pearson"
            ]
            >= 0.90
        )
        & (
            programme[
                "OOF_centered_relative_RMSE"
            ]
            <= 0.50
        )
    )

    programme[
        "primary_and_softcore_stable"
    ] = (
        programme[
            "phase3_primary"
        ]
        & programme[
            "softcore_empirically_stable"
        ]
    )

    primary_table_fidelity = programme.loc[
        programme[
            "phase3_primary"
        ]
    ].copy()

    n_stable_all = int(
        programme[
            "softcore_empirically_stable"
        ].sum()
    )

    n_stable_primary = int(
        programme[
            "primary_and_softcore_stable"
        ].sum()
    )

    median_primary_r = float(
        primary_table_fidelity[
            "OOF_pearson"
        ].median()
    )

    tenth_primary_r = float(
        primary_table_fidelity[
            "OOF_pearson"
        ].quantile(
            0.10
        )
    )

    minimum_primary_r = float(
        primary_table_fidelity[
            "OOF_pearson"
        ].min()
    )

    median_primary_rmse = float(
        primary_table_fidelity[
            "OOF_centered_relative_RMSE"
        ].median()
    )

    median_sample_cosine = float(
        sample_fidelity[
            "OOF_calibrated_cosine_PRIMARY111"
        ].median()
    )

    fifth_sample_cosine = float(
        sample_fidelity[
            "OOF_calibrated_cosine_PRIMARY111"
        ].quantile(
            0.05
        )
    )

    minimum_sample_cosine = float(
        sample_fidelity[
            "OOF_calibrated_cosine_PRIMARY111"
        ].min()
    )

    print()
    print("Empirical programme fidelity")
    print(
        "  global raw centred corr ALL116:",
        round(
            global_raw_all,
            4,
        ),
    )
    print(
        "  global OOF calibrated centred corr ALL116:",
        round(
            global_oof_all,
            4,
        ),
    )
    print(
        "  global raw centred corr PRIMARY111:",
        round(
            global_raw_primary,
            4,
        ),
    )
    print(
        "  global OOF calibrated centred corr PRIMARY111:",
        round(
            global_oof_primary,
            4,
        ),
    )
    print()
    print(
        "  median primary programme OOF r:",
        round(
            median_primary_r,
            4,
        ),
    )
    print(
        "  10th percentile primary OOF r:",
        round(
            tenth_primary_r,
            4,
        ),
    )
    print(
        "  minimum primary OOF r:",
        round(
            minimum_primary_r,
            4,
        ),
    )
    print(
        "  median primary OOF relative RMSE:",
        round(
            median_primary_rmse,
            4,
        ),
    )
    print()
    print(
        "  empirically stable programmes ALL:",
        n_stable_all,
        "/ 116",
    )
    print(
        "  empirically stable Phase-3 primary:",
        n_stable_primary,
        "/ 111",
    )

    print()
    print("Sample-vector fidelity")
    print(
        "  median PRIMARY111 OOF cosine:",
        round(
            median_sample_cosine,
            4,
        ),
    )
    print(
        "  5th percentile PRIMARY111 OOF cosine:",
        round(
            fifth_sample_cosine,
            4,
        ),
    )
    print(
        "  minimum PRIMARY111 OOF cosine:",
        round(
            minimum_sample_cosine,
            4,
        ),
    )

    print()
    print("Lowest-fidelity Phase-3 primary programmes")
    print(
        primary_table_fidelity
        .sort_values(
            "OOF_pearson"
        )[
            [
                "imodulon",
                "OOF_pearson",
                "OOF_centered_relative_RMSE",
                "calibration_slope",
                "primary_and_softcore_stable",
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

    print()
    print("Lowest sample-vector fidelity")
    print(
        sample_fidelity
        .sort_values(
            "OOF_calibrated_cosine_PRIMARY111"
        )
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
    # 10. Final all-data calibration coefficients
    #
    # These are saved for later use when translating masked
    # PRJEB24688 activity coordinates back toward the full-model
    # coordinate scale.
    #
    # For expression CONTRASTS, the intercept cancels. The slope
    # is therefore the important scale correction.
    # ----------------------------------------------------------

    calibration_rows = []

    for i, imodulon in enumerate(
        imodulons
    ):
        intercept, slope = fit_linear_calibration(
            masked_activity[
                i,
                :
            ],
            full_activity[
                i,
                :
            ],
        )

        calibration_rows.append(
            {
                "imodulon":
                    imodulon,

                "calibration_intercept":
                    intercept,

                "calibration_slope":
                    slope,

                "phase3_primary":
                    imodulon
                    in primary_set,

                "softcore_empirically_stable":
                    bool(
                        programme.loc[
                            programme[
                                "imodulon"
                            ]
                            == imodulon,
                            "softcore_empirically_stable",
                        ].iloc[
                            0
                        ]
                    ),
            }
        )

    calibration = pd.DataFrame(
        calibration_rows
    )

    # ----------------------------------------------------------
    # 11. Audit decision
    #
    # These are NEW empirical mask-fidelity gates.
    #
    # They do not overwrite Audit 03d.
    # Audit 03d remains a formal structural-coverage FAIL.
    #
    # This audit asks a distinct question:
    #
    # "Despite that structural loss, does the masked projection
    #  recover the full-model activity geometry empirically?"
    # ----------------------------------------------------------

    checks = {
        "full_M_rank_116":
            rank_full
            == 116,

        "masked_M_rank_116":
            rank_mask
            == 116,

        "global_primary_OOF_corr_ge_095":
            global_oof_primary
            >= 0.95,

        "median_primary_programme_OOF_r_ge_095":
            median_primary_r
            >= 0.95,

        "10th_percentile_primary_OOF_r_ge_090":
            tenth_primary_r
            >= 0.90,

        "at_least_100_of_111_primary_programmes_stable":
            n_stable_primary
            >= 100,

        "median_sample_vector_OOF_cosine_ge_095":
            median_sample_cosine
            >= 0.95,

        "5th_percentile_sample_vector_OOF_cosine_ge_090":
            fifth_sample_cosine
            >= 0.90,

        "finite_programme_metrics":
            bool(
                np.isfinite(
                    programme[
                        [
                            "pearson_full_vs_masked",
                            "raw_centered_relative_RMSE",
                            "calibration_slope",
                            "calibrated_centered_relative_RMSE",
                            "OOF_pearson",
                            "OOF_centered_relative_RMSE",
                        ]
                    ].to_numpy(
                        dtype=float
                    )
                ).all()
            ),

        "finite_sample_metrics":
            bool(
                np.isfinite(
                    sample_fidelity[
                        [
                            "raw_cosine_ALL116",
                            "OOF_calibrated_cosine_ALL116",
                            "raw_cosine_PRIMARY111",
                            "OOF_calibrated_cosine_PRIMARY111",
                            "raw_pearson_PRIMARY111",
                            "OOF_calibrated_pearson_PRIMARY111",
                        ]
                    ].to_numpy(
                        dtype=float
                    )
                ).all()
            ),
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 03e empirical soft-core fidelity checks")

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
        "PRJEB24688_SOFTCORE_PROJECTION_EMPIRICALLY_VALIDATED"
        if passed
        else "PRJEB24688_SOFTCORE_PROJECTION_NOT_SUFFICIENTLY_FAITHFUL"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: Audit 03d remains REVIEW_REQUIRED under the "
        "original structural loading-coverage gates. Audit 03e is "
        "an independent empirical validation of the actual projection "
        "distortion caused by the Rossi species soft-core mask."
    )

    # ----------------------------------------------------------
    # 12. Save
    # ----------------------------------------------------------

    programme.to_csv(
        OUT
        / "softcore_programme_projection_fidelity.csv",
        index=False,
    )

    sample_fidelity.to_csv(
        OUT
        / "softcore_sample_vector_fidelity.csv",
        index=False,
    )

    calibration.to_csv(
        OUT
        / "softcore_to_full_activity_calibration.csv",
        index=False,
    )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        OUT
        / "cross_validation_folds.csv",
        index=False,
    )

    stable_primary = programme.loc[
        programme[
            "primary_and_softcore_stable"
        ]
    ].copy()

    stable_primary.to_csv(
        OUT
        / "PRJEB24688_empirically_stable_primary_programmes.csv",
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
        / "AUDIT_03E_STATUS.txt"
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
            "Failed Audit 03e checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
