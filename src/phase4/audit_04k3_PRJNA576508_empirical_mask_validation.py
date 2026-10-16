from pathlib import Path
from io import StringIO
import hashlib
import json
import traceback

import numpy as np
import pandas as pd


ROOT = Path.cwd()
P4 = ROOT / "results/phase4"

MODEL = ROOT / (
    "external/aeruprecise/precise411/"
    "paeru_116_iModulon.json"
)

ORIGINAL_MASK = P4 / (
    "audit_03d_PRJEB24688_published_count_intake/"
    "PRJEB24688_frozen_replication_gene_map.csv"
)

COMMON_MASK = P4 / (
    "audit_04k2_PRJNA576508_gene_universe_diagnostic/"
    "DIAGNOSTIC_common_gene_order.csv"
)

AUDIT_04K2_STATUS = P4 / (
    "audit_04k2_PRJNA576508_gene_universe_diagnostic/"
    "AUDIT_04K2_STATUS.txt"
)

PREVIOUS_FIDELITY = P4 / (
    "audit_03e_softcore_projection_fidelity/"
    "softcore_programme_projection_fidelity.csv"
)

STABLE = P4 / (
    "audit_03e_softcore_projection_fidelity/"
    "PRJEB24688_empirically_stable_primary_programmes.csv"
)

CALIBRATION = P4 / (
    "audit_03e_softcore_projection_fidelity/"
    "softcore_to_full_activity_calibration.csv"
)

OUT = P4 / (
    "audit_04k3_PRJNA576508_empirical_mask_validation"
)

checks = []


def require(name, condition):
    passed = bool(condition)
    checks.append({"check": name, "pass": passed})
    print(name, "PASS" if passed else "FAIL", flush=True)

    if not passed:
        raise RuntimeError("Failed: " + name)


def json_df(value):
    if isinstance(value, str):
        return pd.read_json(StringIO(value))

    if isinstance(value, dict):
        return pd.DataFrame(value)

    raise RuntimeError("Unsupported model matrix format")


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def vector_pearson(a, b):
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)

    x = x - x.mean()
    y = y - y.mean()

    denominator = (
        np.linalg.norm(x) * np.linalg.norm(y)
    )

    if denominator <= 1e-12:
        return float("nan")

    return float(np.dot(x, y) / denominator)


def cosine(a, b):
    denominator = (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if denominator <= 1e-12:
        return float("nan")

    return float(np.dot(a, b) / denominator)


def main():
    print("=" * 76)
    print("CF-State Phase 4 — Audit 04k3")
    print("Empirical 4925 vs 4923 gene-mask validation")
    print("=" * 76)

    sources = {
        "model": MODEL,
        "original_mask": ORIGINAL_MASK,
        "common_mask": COMMON_MASK,
        "04k2_status": AUDIT_04K2_STATUS,
        "previous_fidelity": PREVIOUS_FIDELITY,
        "stable_108": STABLE,
        "calibration": CALIBRATION,
        "script": Path(__file__)
    }

    for name, path in sources.items():
        require(name + "_exists", path.is_file())

    require(
        "04k2_diagnostic_complete",
        AUDIT_04K2_STATUS.read_text().strip()
        == "PRJNA576508_04K2_GENE_UNIVERSE_DIAGNOSTIC_COMPLETE"
    )

    print()
    print("========== LOAD ORIGINAL MODEL ==========")

    with MODEL.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    require(
        "model_has_M_and_X",
        "M" in payload and "X" in payload
    )

    # Same serialized model representation used by Audit 03e.
    M = json_df(payload["M"])
    X = json_df(payload["X"])

    M.index = M.index.astype(str).str.strip()
    M.columns = M.columns.astype(str).str.strip()

    X.index = X.index.astype(str).str.strip()
    X.columns = X.columns.astype(str)

    require(
        "M_shape_5572_116",
        M.shape == (5572, 116)
    )

    require(
        "X_shape_5572_411",
        X.shape == (5572, 411)
    )

    require(
        "model_gene_universe_agrees",
        M.index.is_unique
        and X.index.is_unique
        and set(M.index) == set(X.index)
    )

    require(
        "model_sample_identifiers_unique",
        X.columns.is_unique
    )

    M = M.apply(pd.to_numeric, errors="raise")
    X = X.reindex(M.index)
    X = X.apply(pd.to_numeric, errors="raise")

    require(
        "model_matrices_finite",
        np.isfinite(M.to_numpy()).all()
        and np.isfinite(X.to_numpy()).all()
    )

    print()
    print("========== FROZEN MASKS ==========")

    old_table = pd.read_csv(ORIGINAL_MASK)
    new_table = pd.read_csv(COMMON_MASK)

    require(
        "original_mask_column",
        "pao1_gene" in old_table.columns
    )

    require(
        "common_mask_column",
        "GeneID" in new_table.columns
    )

    original_genes = (
        old_table["pao1_gene"]
        .astype(str).str.strip().tolist()
    )

    common_genes = (
        new_table["GeneID"]
        .astype(str).str.strip().tolist()
    )

    require(
        "original_4925_unique",
        len(original_genes) == 4925
        and len(set(original_genes)) == 4925
    )

    require(
        "common_4923_unique",
        len(common_genes) == 4923
        and len(set(common_genes)) == 4923
    )

    missing = set(original_genes) - set(common_genes)

    require(
        "exact_two_missing_genes",
        missing == {"PA0195.1", "PA4724.1"}
        and set(common_genes).issubset(set(original_genes))
    )

    original_set = set(original_genes)
    common_set = set(common_genes)

    original_order = [
        g for g in M.index if g in original_set
    ]

    common_order = [
        g for g in M.index if g in common_set
    ]

    require(
        "common_order_matches_04k2",
        common_order == common_genes
    )

    require(
        "mask_lengths_correct",
        len(original_order) == 4925
        and len(common_order) == 4923
    )

    print()
    print("========== PROJECT ALL 411 TRANSCRIPTOMES ==========")

    matrix_full = M.to_numpy(dtype=float)
    matrix_old = M.loc[original_order].to_numpy(dtype=float)
    matrix_new = M.loc[common_order].to_numpy(dtype=float)

    expression_full = X.to_numpy(dtype=float)
    expression_old = X.loc[original_order].to_numpy(dtype=float)
    expression_new = X.loc[common_order].to_numpy(dtype=float)

    rank_old = int(np.linalg.matrix_rank(matrix_old))
    rank_new = int(np.linalg.matrix_rank(matrix_new))

    require(
        "both_masks_full_rank",
        rank_old == 116 and rank_new == 116
    )

    activities_full = (
        np.linalg.pinv(matrix_full) @ expression_full
    )

    activities_old = (
        np.linalg.pinv(matrix_old) @ expression_old
    )

    activities_new = (
        np.linalg.pinv(matrix_new) @ expression_new
    )

    require(
        "all_activity_shapes",
        activities_full.shape == (116, 411)
        and activities_old.shape == (116, 411)
        and activities_new.shape == (116, 411)
    )

    require(
        "all_activities_finite",
        np.isfinite(activities_full).all()
        and np.isfinite(activities_old).all()
        and np.isfinite(activities_new).all()
    )

    programmes = list(M.columns)

    print()
    print("========== VERIFY ORIGINAL AUDIT 03e ==========")

    previous = pd.read_csv(PREVIOUS_FIDELITY)

    previous["imodulon"] = (
        previous["imodulon"].astype(str).str.strip()
    )

    require(
        "previous_116_programmes",
        len(previous) == 116
        and previous["imodulon"].is_unique
        and set(previous["imodulon"]) == set(programmes)
    )

    previous = previous.set_index("imodulon").reindex(programmes)

    programme_rows = []

    for i, programme in enumerate(programmes):
        full = activities_full[i]
        old = activities_old[i]
        new = activities_new[i]

        old_centered = old - old.mean()
        new_centered = new - new.mean()

        old_norm = np.linalg.norm(old_centered)

        if old_norm <= 1e-12:
            raise RuntimeError(
                "Zero original activity variance: " + programme
            )

        programme_rows.append({
            "imodulon": programme,
            "pearson_old_vs_full":
                vector_pearson(old, full),
            "pearson_new_vs_full":
                vector_pearson(new, full),
            "pearson_new_vs_old":
                vector_pearson(new, old),
            "centered_relative_RMSE_new_vs_old":
                float(
                    np.linalg.norm(
                        new_centered - old_centered
                    ) / old_norm
                ),
            "direct_relative_L2_new_vs_old":
                float(
                    np.linalg.norm(new - old)
                    / max(np.linalg.norm(old), 1e-12)
                ),
            "max_absolute_activity_difference":
                float(np.max(np.abs(new - old)))
        })

    programme_qc = pd.DataFrame(programme_rows)

    previous_corr = previous[
        "pearson_full_vs_masked"
    ].to_numpy(dtype=float)

    observed_corr = programme_qc[
        "pearson_old_vs_full"
    ].to_numpy(dtype=float)

    reproduction_error = float(
        np.max(np.abs(previous_corr - observed_corr))
    )

    print(
        "Maximum Audit 03e reproduction discrepancy:",
        reproduction_error
    )

    require(
        "original_03e_projection_reproduced",
        np.isfinite(reproduction_error)
        and reproduction_error <= 1e-6
    )

    print()
    print("========== FROZEN 108 PROGRAMMES ==========")

    stable = pd.read_csv(STABLE)
    stable["imodulon"] = (
        stable["imodulon"].astype(str).str.strip()
    )

    selected = set(stable["imodulon"])

    require(
        "108_stable_programmes",
        len(stable) == 108
        and len(selected) == 108
        and selected.issubset(set(programmes))
    )

    programme_qc["stable_108"] = (
        programme_qc["imodulon"].isin(selected)
    )

    stable_qc = programme_qc.loc[
        programme_qc["stable_108"]
    ]

    print()
    print("Programme correlations: new vs original mask")

    print(
        stable_qc[
            [
                "pearson_new_vs_old",
                "centered_relative_RMSE_new_vs_old",
                "direct_relative_L2_new_vs_old"
            ]
        ].describe().round(7).to_string()
    )

    print()
    print("Lowest-agreement stable programmes:")

    print(
        stable_qc.nsmallest(
            10, "pearson_new_vs_old"
        )[
            [
                "imodulon",
                "pearson_new_vs_old",
                "centered_relative_RMSE_new_vs_old",
                "pearson_old_vs_full",
                "pearson_new_vs_full"
            ]
        ].round(7).to_string(index=False)
    )

    print()
    print("========== FROZEN CALIBRATION SLOPES ==========")

    calibration = pd.read_csv(CALIBRATION)

    calibration["imodulon"] = (
        calibration["imodulon"].astype(str).str.strip()
    )

    require(
        "calibration_116_unique",
        len(calibration) == 116
        and calibration["imodulon"].is_unique
    )

    slopes = pd.to_numeric(
        calibration.set_index("imodulon")
        .reindex(programmes)["calibration_slope"],
        errors="raise"
    ).to_numpy(dtype=float)

    require(
        "all_original_slopes_finite",
        np.isfinite(slopes).all()
    )

    # Preserve the original 03e slopes.
    # Do not refit or select programmes using Atlanta.
    calibrated_old = activities_old * slopes[:, None]
    calibrated_new = activities_new * slopes[:, None]

    selected_indices = [
        i for i, p in enumerate(programmes)
        if p in selected
    ]

    print()
    print("========== SAMPLE-VECTOR AGREEMENT ==========")

    sample_rows = []

    for j, sample in enumerate(X.columns):
        a = calibrated_old[selected_indices, j]
        b = calibrated_new[selected_indices, j]

        sample_rows.append({
            "aeruprecise_sample": sample,
            "cosine_108": cosine(a, b),
            "pearson_108": vector_pearson(a, b),
            "relative_L2_change_108":
                float(
                    np.linalg.norm(b - a)
                    / max(np.linalg.norm(a), 1e-12)
                )
        })

    sample_qc = pd.DataFrame(sample_rows)

    require(
        "411_sample_comparisons",
        len(sample_qc) == 411
    )

    require(
        "comparison_metrics_finite",
        np.isfinite(
            programme_qc.select_dtypes(
                include="number"
            ).to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            sample_qc.select_dtypes(
                include="number"
            ).to_numpy(dtype=float)
        ).all()
    )

    print(
        sample_qc[
            [
                "cosine_108",
                "pearson_108",
                "relative_L2_change_108"
            ]
        ].describe().round(7).to_string()
    )

    print()
    print("========== CONDITIONING ==========")

    cond_old = float(np.linalg.cond(matrix_old))
    cond_new = float(np.linalg.cond(matrix_new))

    print("Original 4925 condition number:", cond_old)
    print("Common 4923 condition number:", cond_new)

    require(
        "both_condition_numbers_finite",
        np.isfinite(cond_old)
        and np.isfinite(cond_new)
    )

    print()
    print("========== SAVE DIAGNOSTIC RESULTS ==========")

    programme_qc.to_csv(
        OUT / "programme_4925_vs_4923_empirical_fidelity.csv",
        index=False
    )

    sample_qc.to_csv(
        OUT / "sample_4925_vs_4923_empirical_fidelity.csv",
        index=False
    )

    summary = {
        "reference_transcriptomes": 411,
        "original_mask_genes": 4925,
        "common_mask_genes": 4923,
        "programmes_fit": 116,
        "stable_programmes": 108,
        "original_rank": rank_old,
        "common_rank": rank_new,
        "condition_number_old": cond_old,
        "condition_number_new": cond_new,
        "prior_03e_reproduction_max_error":
            reproduction_error,
        "min_108_programme_Pearson_new_vs_old":
            float(stable_qc["pearson_new_vs_old"].min()),
        "median_108_programme_Pearson_new_vs_old":
            float(stable_qc["pearson_new_vs_old"].median()),
        "max_108_programme_centered_relative_RMSE":
            float(
                stable_qc[
                    "centered_relative_RMSE_new_vs_old"
                ].max()
            ),
        "min_411_sample_cosine_108":
            float(sample_qc["cosine_108"].min()),
        "median_411_sample_cosine_108":
            float(sample_qc["cosine_108"].median()),
        "max_411_sample_relative_L2_change":
            float(
                sample_qc[
                    "relative_L2_change_108"
                ].max()
            )
    }

    pd.DataFrame([summary]).to_csv(
        OUT / "audit_04k3_empirical_summary.csv",
        index=False
    )

    pd.DataFrame(checks).to_csv(
        OUT / "audit_pass_checks.csv",
        index=False
    )

    pd.DataFrame(
        [
            {
                "source": name,
                "path": str(path),
                "sha256": sha256(path)
            }
            for name, path in sources.items()
        ]
    ).to_csv(
        OUT / "input_provenance_sha256.csv",
        index=False
    )

    status = (
        "PRJNA576508_4923_MASK_EMPIRICAL_"
        "COMPARISON_COMPLETE"
    )

    (OUT / "AUDIT_04K3_STATUS.txt").write_text(
        status + "\n"
    )

    print()
    print("=" * 76)
    print("STATUS:", status)
    print("=" * 76)
    print("Original 4925-gene mask unchanged.")
    print("Frozen calibration slopes unchanged.")
    print("No Atlanta expression used.")
    print("No Audit-03h candidate accessed.")
    print("Empirical comparison complete; approval pending.")


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 04k3 already exists. Refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()
    except Exception:
        (OUT / "AUDIT_04K3_STATUS.txt").write_text(
            "PRJNA576508_04K3_REVIEW_REQUIRED\n"
        )

        (OUT / "AUDIT_04K3_FAILURE.txt").write_text(
            traceback.format_exc()
        )

        pd.DataFrame(checks).to_csv(
            OUT / "audit_pass_checks.csv",
            index=False
        )

        raise
