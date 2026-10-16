from pathlib import Path
from io import StringIO
import hashlib
import json
import traceback

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / "results/phase4"

K4 = P4 / "audit_04k4_PRJNA576508_blinded_softcore_projection"
K3 = P4 / "audit_04k3_PRJNA576508_empirical_mask_validation"
J2 = P4 / "audit_04j2_PRJNA576508_Atlanta_expression_normalisation"

OUT = P4 / "audit_04l_PRJNA576508_blinded_projection_QC_lock"

MODEL = ROOT / (
    "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

CAL = P4 / (
    "audit_03e_softcore_projection_fidelity/"
    "softcore_to_full_activity_calibration.csv"
)

STABLE = P4 / (
    "audit_03e_softcore_projection_fidelity/"
    "PRJEB24688_empirically_stable_primary_programmes.csv"
)

MASK = K4 / "FROZEN_softcore_gene_order_4923.csv"

PRIMARY_CONTRAST = J2 / (
    "FROZEN_Atlanta_13_minus_PRIMARY_CFBE_ALL4_log2TPM.csv"
)

FIT = K4 / "FROZEN_projection_fit_QC_78.csv"
STABILITY = K4 / "FROZEN_108_programme_sensitivity_91.csv"

NORMS = ["FROZEN_5567_TPM", "SOFTCORE_4923_TPM"]

REFS = [
    "PRIMARY_CFBE_ALL4",
    "SENSITIVITY_CFBE_C_PAIR",
    "SENSITIVITY_CFBE_R_PAIR"
]

SAMPLES = [
    "EM6", "EM13", "EM24", "EM30", "EM33",
    "EM35", "EM36", "EM37", "EM43",
    "EM48", "EM49", "EM53", "EM54"
]

CHECKS = []


def require(label, valid):
    passed = bool(valid)
    CHECKS.append({"check": label, "pass": passed})

    print(
        label,
        "PASS" if passed else "FAIL",
        flush=True
    )

    if not passed:
        raise RuntimeError("Audit 04l failed: " + label)


def digest(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for part in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            h.update(part)

    return h.hexdigest()


def cosine(a, b):
    denom = float(
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if denom <= 1e-12:
        return float("nan")

    return float(np.dot(a, b) / denom)


def pearson(a, b):
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def activity_table(path, expected_rows):
    df = pd.read_csv(path)

    require(
        path.name + "_columns",
        "imodulon" in df
        and set(df.columns) == set(SAMPLES) | {"imodulon"}
    )

    require(
        path.name + "_rows",
        len(df) == expected_rows
        and df["imodulon"].is_unique
    )

    df = df.set_index("imodulon").reindex(columns=SAMPLES)

    require(
        path.name + "_finite",
        np.isfinite(
            df.to_numpy(dtype=float)
        ).all()
    )

    return df


def main():
    print("=" * 72)
    print("CF-State Phase 4 — Audit 04l")
    print("Blinded projection QC and evaluation-design lock")
    print("=" * 72)

    sources = {
        "04k4_status": K4 / "AUDIT_04K4_STATUS.txt",
        "04k3_status": K3 / "AUDIT_04K3_STATUS.txt",
        "04j2_status": J2 / "AUDIT_04J2_STATUS.txt",
        "projection_fit": FIT,
        "projection_stability": STABILITY,
        "mask": MASK,
        "model": MODEL,
        "calibration": CAL,
        "stable_108": STABLE,
        "primary_contrast": PRIMARY_CONTRAST,
        "script": Path(__file__)
    }

    for key, path in sources.items():
        require("input_" + key, path.is_file())

    expected_statuses = {
        "04k4_status":
            "PRJNA576508_ATLANTA_13_4923_MASK_BLINDED_SOFTCORE_PROJECTIONS_FROZEN",
        "04k3_status":
            "PRJNA576508_4923_MASK_EMPIRICAL_COMPARISON_COMPLETE",
        "04j2_status":
            "PRJNA576508_ATLANTA_13_LOG2TPM_EXTERNAL_CONTRASTS_FROZEN"
    }

    for key, expected in expected_statuses.items():
        require(
            key + "_correct",
            sources[key].read_text().strip() == expected
        )

    stable = pd.read_csv(STABLE)

    selected = (
        stable["imodulon"]
        .astype(str).str.strip().tolist()
    )

    require(
        "exact_108_previous_programmes",
        len(selected) == len(set(selected)) == 108
    )

    cal = pd.read_csv(CAL)

    cal["imodulon"] = (
        cal["imodulon"].astype(str).str.strip()
    )

    require(
        "exact_116_calibration_slopes",
        len(cal) == 116
        and cal["imodulon"].is_unique
    )

    slopes = pd.to_numeric(
        cal.set_index("imodulon")["calibration_slope"],
        errors="raise"
    )

    require(
        "all_slopes_finite",
        np.isfinite(slopes.to_numpy()).all()
    )

    require(
        "stable_programmes_have_slopes",
        set(selected).issubset(set(slopes.index))
    )

    print()
    print("========== SIX ACTIVITY MATRICES ==========")

    raw = {}
    calibrated = {}

    for norm in NORMS:
        for ref in REFS:
            suffix = "__" + norm + "__" + ref + ".csv"

            raw_path = K4 / (
                "FROZEN_116_raw_activities" + suffix
            )

            cal_path = K4 / (
                "FROZEN_108_activities" + suffix
            )

            require(
                "files_exist_" + norm + "_" + ref,
                raw_path.is_file() and cal_path.is_file()
            )

            r = activity_table(raw_path, 116)
            c = activity_table(cal_path, 108)

            require(
                "raw_programmes_match_" + norm + "_" + ref,
                set(r.index) == set(slopes.index)
            )

            require(
                "selected_programmes_match_" + norm + "_" + ref,
                set(c.index) == set(selected)
            )

            r = r.reindex(slopes.index)
            c = c.reindex(selected)

            recovered = r.reindex(selected).mul(
                slopes.reindex(selected),
                axis=0
            )

            require(
                "unchanged_calibration_" + norm + "_" + ref,
                np.allclose(
                    c.to_numpy(),
                    recovered.to_numpy(),
                    rtol=1e-10,
                    atol=1e-8
                )
            )

            raw[(norm, ref)] = r
            calibrated[(norm, ref)] = c

    print()
    print("========== INDEPENDENT PRIMARY REPRODUCTION ==========")

    with MODEL.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    obj = payload["M"]

    if isinstance(obj, str):
        M = pd.read_json(StringIO(obj))
    else:
        M = pd.DataFrame(obj)

    M.index = M.index.astype(str).str.strip()
    M.columns = M.columns.astype(str).str.strip()

    mask = (
        pd.read_csv(MASK)["GeneID"]
        .astype(str).tolist()
    )

    require(
        "common_mask_4923_unique",
        len(mask) == 4923
        and len(set(mask)) == 4923
    )

    require(
        "model_contains_mask",
        set(mask).issubset(set(M.index))
    )

    M_array = M.reindex(mask).to_numpy(dtype=float)

    require(
        "model_rank_116",
        np.linalg.matrix_rank(M_array) == 116
    )

    delta = (
        pd.read_csv(PRIMARY_CONTRAST)
        .set_index("GeneID")
        .reindex(index=mask, columns=SAMPLES)
    )

    require(
        "primary_contrast_all_genes_finite",
        np.isfinite(
            delta.to_numpy(dtype=float)
        ).all()
    )

    a = np.linalg.pinv(M_array) @ delta.to_numpy(dtype=float)

    observed = (
        raw[(NORMS[0], REFS[0])]
        .reindex(M.columns)
        .to_numpy(dtype=float)
    )

    require(
        "independently_reproduced_primary_116",
        np.allclose(
            a, observed,
            rtol=1e-10,
            atol=1e-8
        )
    )

    print()
    print("========== RECONSTRUCTION QC ==========")

    fit = pd.read_csv(FIT)

    require(
        "78_unique_fits",
        len(fit) == 78
        and not fit.duplicated(
            ["normalisation", "reference", "sample"]
        ).any()
    )

    observed_keys = set(
        fit[
            ["normalisation", "reference", "sample"]
        ].itertuples(index=False, name=None)
    )

    expected_keys = {
        (n, r, s)
        for n in NORMS
        for r in REFS
        for s in SAMPLES
    }

    require(
        "all_13_samples_times_six_variants",
        observed_keys == expected_keys
    )

    fit_columns = [
        "projection_pearson",
        "projection_energy_fraction",
        "relative_reconstruction_error",
        "input_contrast_l2",
        "raw_activity_l2"
    ]

    require(
        "projection_fit_finite",
        np.isfinite(
            fit[fit_columns].to_numpy(dtype=float)
        ).all()
    )

    energy = fit[
        "projection_energy_fraction"
    ].to_numpy(dtype=float)

    error = fit[
        "relative_reconstruction_error"
    ].to_numpy(dtype=float)

    require(
        "projection_identity_verified",
        np.allclose(
            energy + error ** 2,
            1,
            atol=1e-7,
            rtol=0
        )
    )

    require(
        "energy_between_zero_and_one",
        np.all(
            (energy >= 0)
            & (energy <= 1 + 1e-9)
        )
    )

    primary_fit = fit.loc[
        (fit["normalisation"] == NORMS[0])
        & (fit["reference"] == REFS[0])
    ].copy()

    require(
        "primary_retains_13",
        len(primary_fit) == 13
        and set(primary_fit["sample"]) == set(SAMPLES)
    )

    print()
    print("========== RECOMPUTE SENSITIVITY ==========")

    stability = pd.read_csv(STABILITY)

    fields = [
        "sample", "comparison",
        "normalisation", "alternative"
    ]

    require(
        "91_unique_sensitivities",
        len(stability) == 91
        and not stability.duplicated(fields).any()
    )

    observed_stability = stability.set_index(fields)

    recomputed = []

    for sample in SAMPLES:
        for norm in NORMS:
            first = calibrated[
                (norm, REFS[0])
            ][sample].to_numpy(dtype=float)

            for ref in REFS[1:]:
                second = calibrated[
                    (norm, ref)
                ][sample].to_numpy(dtype=float)

                recomputed.append(
                    (
                        sample, "reference",
                        norm, ref, first, second
                    )
                )

        for ref in REFS:
            first = calibrated[
                (NORMS[0], ref)
            ][sample].to_numpy(dtype=float)

            second = calibrated[
                (NORMS[1], ref)
            ][sample].to_numpy(dtype=float)

            recomputed.append(
                (
                    sample, "normalisation",
                    NORMS[0],
                    NORMS[1] + "__" + ref,
                    first, second
                )
            )

    require(
        "91_recomputed",
        len(recomputed) == 91
    )

    missing_keys = []
    mismatched_metrics = []

    for sample, comparison, norm, alternative, first, second in recomputed:
        key = (
            sample, comparison,
            norm, alternative
        )

        if key not in observed_stability.index:
            missing_keys.append(key)
            continue

        saved = observed_stability.loc[key]

        comparisons = {
            "cosine_108": cosine(first, second),
            "pearson_108": pearson(first, second),
            "relative_change_108":
                float(
                    np.linalg.norm(second - first)
                    / max(np.linalg.norm(first), 1e-12)
                )
        }

        for metric, value in comparisons.items():
            if (
                not np.isfinite(value)
                or abs(value - float(saved[metric])) > 1e-9
            ):
                mismatched_metrics.append(
                    (key, metric)
                )

    require(
        "all_91_stability_keys_present",
        not missing_keys
    )

    require(
        "all_273_stability_metrics_reproduced",
        not mismatched_metrics
    )

    print()
    print("========== BLINDED SAMPLE-LEVEL QC ==========")

    sample_rows = []

    for sample in SAMPLES:
        f = (
            primary_fit
            .set_index("sample")
            .loc[sample]
        )

        ss = stability.loc[
            stability["sample"] == sample
        ]

        ref_ss = ss.loc[
            ss["comparison"] == "reference"
        ]

        norm_ss = ss.loc[
            ss["comparison"] == "normalisation"
        ]

        sample_rows.append({
            "sample": sample,
            "primary_projection_energy":
                float(f["projection_energy_fraction"]),
            "primary_projection_pearson":
                float(f["projection_pearson"]),
            "primary_relative_error":
                float(f["relative_reconstruction_error"]),
            "min_reference_cosine_108":
                float(ref_ss["cosine_108"].min()),
            "min_normalisation_cosine_108":
                float(norm_ss["cosine_108"].min()),
            "max_reference_relative_change_108":
                float(ref_ss["relative_change_108"].max()),
            "max_normalisation_relative_change_108":
                float(norm_ss["relative_change_108"].max()),
            "retained_in_primary": True
        })

    qc = pd.DataFrame(sample_rows)

    require(
        "blinded_QC_complete",
        len(qc) == 13
        and np.isfinite(
            qc.select_dtypes(
                include="number"
            ).to_numpy()
        ).all()
    )

    print(
        qc.round(5).to_string(index=False)
    )

    print()
    print("Primary energy min / median / max:")

    print(
        qc["primary_projection_energy"].min(),
        qc["primary_projection_energy"].median(),
        qc["primary_projection_energy"].max()
    )

    print()
    print("Reference cosine min / median:")

    print(
        stability.loc[
            stability["comparison"] == "reference",
            "cosine_108"
        ].min(),
        stability.loc[
            stability["comparison"] == "reference",
            "cosine_108"
        ].median()
    )

    print()
    print("Normalisation cosine min / median:")

    print(
        stability.loc[
            stability["comparison"] == "normalisation",
            "cosine_108"
        ].min(),
        stability.loc[
            stability["comparison"] == "normalisation",
            "cosine_108"
        ].median()
    )

    print()
    print("========== FREEZE BLINDED QC ==========")

    qc.to_csv(
        OUT / "FROZEN_blinded_13_sample_QC.csv",
        index=False
    )

    pd.DataFrame(CHECKS).to_csv(
        OUT / "audit_pass_checks.csv",
        index=False
    )

    pd.DataFrame([
        {
            "input": label,
            "path": str(path),
            "sha256": digest(path)
        }
        for label, path in sources.items()
    ]).to_csv(
        OUT / "input_sha256_provenance.csv",
        index=False
    )

    protocol = (
        "CF-State prospective third-cohort evaluation lock (blinded)\n"
        "Primary: 13 frozen Atlanta samples; FROZEN_5567_TPM; "
        "PRIMARY_CFBE_ALL4; 108 previously stable calibrated programmes.\n"
        "No sample excluded based on Phase-4 projection energy "
        "or programme similarity.\n"
        "Sensitivity: C-pair and R-pair references, "
        "and SOFTCORE_4923_TPM normalisation; retain all 13 samples.\n"
        "Comparison target: original frozen Audit-03h CF-airway "
        "residual programme candidate, unchanged.\n"
        "Before examining candidate values, reproduce the established "
        "controlled-state residualisation and align programme identities.\n"
        "Do not compare an unresidualised Atlanta vector directly "
        "with a residual candidate.\n"
        "Report per-sample directional agreement, cohort-level summaries "
        "and all predefined sensitivity contrasts.\n"
        "Inferential testing must respect donor-level independence, "
        "not yet established for the 13 sample labels.\n"
        "Low projection energy limits interpretation even if "
        "programme-space replication is positive.\n"
        "External CFBE culture is not a patient-matched baseline.\n"
        "This audit verifies technical reproducibility, "
        "not biological validity or statistical significance.\n"
        "No Audit-03h candidate accessed or modified.\n"
    )

    (
        OUT / "FROZEN_BLINDED_EVALUATION_DESIGN.txt"
    ).write_text(
        protocol,
        encoding="utf-8"
    )

    status = (
        "PRJNA576508_ATLANTA_BLINDED_"
        "PROJECTION_QC_AND_DESIGN_LOCK_FROZEN"
    )

    (OUT / "AUDIT_04L_STATUS.txt").write_text(
        status + "\n",
        encoding="utf-8"
    )

    print()
    print("STATUS:", status)
    print("No candidate unblinding performed.")


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 04l output exists. Refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()

    except Exception:
        (
            OUT / "AUDIT_04L_STATUS.txt"
        ).write_text(
            "PRJNA576508_AUDIT_04L_REVIEW_REQUIRED\n"
        )

        (
            OUT / "AUDIT_04L_FAILURE.txt"
        ).write_text(
            traceback.format_exc()
        )

        pd.DataFrame(CHECKS).to_csv(
            OUT / "audit_pass_checks.csv",
            index=False
        )

        raise
