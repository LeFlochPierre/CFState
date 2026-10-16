from pathlib import Path
from io import StringIO
import hashlib
import json
import traceback

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / "results/phase4"

A04J = P4 / "audit_04j2_PRJNA576508_Atlanta_expression_normalisation"
A04I = P4 / "audit_04i3_PRJNA576508_CFBE_reference_metadata_repair"
A04G = P4 / "audit_04g_PRJNA576508_Atlanta_full_cohort_processing"
A03E = P4 / "audit_03e_softcore_projection_fidelity"

OUT = P4 / "audit_04k4_PRJNA576508_blinded_softcore_projection"

MODEL = ROOT / (
    "external/aeruprecise/precise411/paeru_116_iModulon.json"
)
MASK = P4 / (
    "audit_04k2_PRJNA576508_gene_universe_diagnostic/"
    "DIAGNOSTIC_common_gene_order.csv"
)
STABLE = A03E / "PRJEB24688_empirically_stable_primary_programmes.csv"
CAL = A03E / "softcore_to_full_activity_calibration.csv"
SAF = ROOT / (
    "references/phase4/PRJNA576508_cornforth/"
    "Cornforth_PAO1_no_decimal_locus_tags.saf"
)

ATL_COUNTS = A04G / "FROZEN_Atlanta_13_raw_PAO1_gene_counts.csv"
CFBE_COUNTS = A04I / "FROZEN_CFBE_four_raw_gene_counts.csv"
ATL_LOG = A04J / "FROZEN_Atlanta_13_log2TPM_expression.csv"
REF_FULL = A04I / "FROZEN_CFBE_log2TPM_reference_profiles.csv"

SAMPLES = [
    "EM6", "EM13", "EM24", "EM30", "EM33", "EM35",
    "EM36", "EM37", "EM43", "EM48", "EM49",
    "EM53", "EM54"
]

CFBE_SAMPLES = ["C1", "C2", "R1", "R2"]

REFS = [
    "PRIMARY_CFBE_ALL4",
    "SENSITIVITY_CFBE_C_PAIR",
    "SENSITIVITY_CFBE_R_PAIR"
]

NORMS = [
    "FROZEN_5567_TPM",
    "SOFTCORE_4923_TPM"
]

CHECKS = []


def require(label, condition):
    okay = bool(condition)
    CHECKS.append({"check": label, "pass": okay})
    print(label, "PASS" if okay else "FAIL", flush=True)
    if not okay:
        raise RuntimeError("Failed check: " + label)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(4 * 1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def gene_table(path, separator=","):
    df = pd.read_csv(
        path, sep=separator, dtype={"GeneID": str}
    )
    require(
        path.name + "_GeneID",
        "GeneID" in df.columns
    )
    require(
        path.name + "_unique_genes",
        df["GeneID"].notna().all()
        and df["GeneID"].is_unique
    )
    return df.set_index("GeneID")


def numerical(df, name):
    x = df.apply(pd.to_numeric, errors="raise")
    require(
        name + "_finite",
        np.isfinite(x.to_numpy(dtype=float)).all()
    )
    return x


def counts_to_logtpm(counts, lengths):
    rates = counts.div(lengths / 1000.0, axis=0)
    denom = rates.sum(axis=0)

    if (denom <= 0).any() or not np.isfinite(denom).all():
        raise RuntimeError("TPM denominator invalid")

    tpm = rates.div(denom, axis=1) * 1e6

    if not np.allclose(
        tpm.sum(axis=0), 1e6, atol=0.01, rtol=0
    ):
        raise RuntimeError("TPM column sums invalid")

    return np.log2(tpm + 1.0)


def cosine(a, b):
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm <= 1e-14:
        return float("nan")
    return float(np.dot(a, b) / norm)


def pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    print("=" * 76)
    print("CF-State Phase 4 — Audit 04k4")
    print("Blinded Atlanta soft-core iModulon projection")
    print("=" * 76)

    inputs = {
        "model": MODEL,
        "mask": MASK,
        "stable_108": STABLE,
        "calibration": CAL,
        "PAO1_SAF": SAF,
        "Atlanta_counts": ATL_COUNTS,
        "CFBE_counts": CFBE_COUNTS,
        "Atlanta_log2TPM": ATL_LOG,
        "CFBE_reference": REF_FULL,
        "04g_status": A04G / "AUDIT_04G_STATUS.txt",
        "04i3_status": A04I / "AUDIT_04I3_STATUS.txt",
        "04j2_status": A04J / "AUDIT_04J2_STATUS.txt",
        "03e_status": A03E / "AUDIT_03E_STATUS.txt",
        "04k3_status": P4 / (
            "audit_04k3_PRJNA576508_empirical_mask_validation/"
            "AUDIT_04K3_STATUS.txt"
        ),
        "script": Path(__file__)
    }

    for name, path in inputs.items():
        require(
            "input_" + name + "_exists",
            path.is_file()
        )

    expected_status = {
        "04g_status":
            "PRJNA576508_ATLANTA_13_PUBLISHED_PIPELINE_TECHNICALLY_REPLICATED",
        "04i3_status":
            "PRJNA576508_CFBE_ALL4_LOG2TPM_EXTERNAL_REFERENCE_FROZEN",
        "04j2_status":
            "PRJNA576508_ATLANTA_13_LOG2TPM_EXTERNAL_CONTRASTS_FROZEN",
        "03e_status":
            "PRJEB24688_SOFTCORE_PROJECTION_EMPIRICALLY_VALIDATED",
        "04k3_status":
            "PRJNA576508_4923_MASK_EMPIRICAL_COMPARISON_COMPLETE"
    }

    for name, expected in expected_status.items():
        require(
            name + "_frozen",
            inputs[name].read_text().strip() == expected
        )

    print()
    print("========== FROZEN iMODULON SPACE ==========")

    with MODEL.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    require(
        "model_contains_M",
        isinstance(payload, dict) and "M" in payload
    )

    value = payload["M"]

    if isinstance(value, str):
        M = pd.read_json(StringIO(value))
    elif isinstance(value, dict):
        M = pd.DataFrame(value)
    else:
        raise RuntimeError(
            "Unexpected M representation in frozen model"
        )

    M.index = M.index.astype(str).str.strip()
    M.columns = M.columns.astype(str).str.strip()

    require(
        "model_5572_by_116",
        M.shape == (5572, 116)
    )
    require(
        "M_gene_and_programme_unique",
        M.index.is_unique and M.columns.is_unique
    )

    M = numerical(M, "M")
    programmes = list(M.columns)

    map_df = pd.read_csv(
        MASK, dtype={"GeneID": str}
    )

    require(
        "softcore_map_GeneID",
        "GeneID" in map_df.columns
    )

    original_genes = (
        map_df["GeneID"].str.strip().tolist()
    )

    require(
        "mask_4923_unique",
        len(original_genes) == 4923
        and len(set(original_genes)) == 4923
    )

    mask_set = set(original_genes)

    require(
        "mask_contained_in_M",
        mask_set.issubset(set(M.index))
    )

    # Match the frozen Audit 03e ordering.
    mask_genes = [
        g for g in M.index if g in mask_set
    ]

    require(
        "mask_gene_order_4923",
        len(mask_genes) == 4923
    )

    Mmask = M.loc[mask_genes]
    matrix = Mmask.to_numpy(dtype=float)

    require(
        "Mmask_full_rank_116",
        np.linalg.matrix_rank(matrix) == 116
    )

    operator = np.linalg.pinv(matrix)

    require(
        "projection_operator_shape",
        operator.shape == (116, 4923)
    )

    calibration = pd.read_csv(CAL)
    calibration["imodulon"] = (
        calibration["imodulon"].astype(str).str.strip()
    )

    require(
        "calibration_116_unique",
        len(calibration) == 116
        and calibration["imodulon"].is_unique
    )

    slopes = pd.to_numeric(
        calibration.set_index("imodulon")[
            "calibration_slope"
        ].reindex(programmes),
        errors="raise"
    )

    require(
        "calibration_slopes_valid",
        slopes.notna().all()
        and np.isfinite(slopes.to_numpy()).all()
    )

    stable = pd.read_csv(STABLE)
    stable["imodulon"] = (
        stable["imodulon"].astype(str).str.strip()
    )

    require(
        "stable_108_unique",
        len(stable) == 108
        and stable["imodulon"].is_unique
    )

    require(
        "stable_all_previously_qualified",
        stable["primary_and_softcore_stable"]
        .astype(str).str.lower().eq("true").all()
    )

    selected_set = set(stable["imodulon"])

    selected = [
        p for p in programmes if p in selected_set
    ]

    require(
        "stable_108_in_model",
        len(selected) == 108
    )

    print()
    print("========== ATLANTA / CFBE EXPRESSION ==========")

    atl_counts = numerical(
        gene_table(ATL_COUNTS), "Atlanta_counts"
    )

    cfbe_counts = numerical(
        gene_table(CFBE_COUNTS), "CFBE_counts"
    )

    atl_full = numerical(
        gene_table(ATL_LOG), "Atlanta_log2TPM"
    )

    ref_full = numerical(
        gene_table(REF_FULL), "CFBE_reference"
    )

    require(
        "Atlanta_exact_13_samples",
        len(atl_counts.columns) == 13
        and set(atl_counts.columns) == set(SAMPLES)
    )

    require(
        "CFBE_exact_4_samples",
        len(cfbe_counts.columns) == 4
        and set(cfbe_counts.columns) == set(CFBE_SAMPLES)
    )

    require(
        "reference_exact_3",
        list(ref_full.columns) == REFS
    )

    require(
        "Atlanta_expression_sample_match",
        set(atl_full.columns) == set(SAMPLES)
    )

    for name, table in [
        ("Atlanta_counts", atl_counts),
        ("CFBE_counts", cfbe_counts),
        ("Atlanta_expression", atl_full),
        ("CFBE_reference", ref_full)
    ]:
        require(
            name + "_has_mask_genes",
            mask_set.issubset(set(table.index))
        )

    atl_values = atl_counts.to_numpy()
    cfbe_values = cfbe_counts.to_numpy()

    require(
        "Atlanta_and_CFBE_counts_nonnegative_integer",
        (atl_values >= 0).all()
        and (cfbe_values >= 0).all()
        and np.all(atl_values == np.floor(atl_values))
        and np.all(cfbe_values == np.floor(cfbe_values))
    )

    atl_full = atl_full.reindex(columns=SAMPLES)

    saf = gene_table(SAF, separator="\t")

    require(
        "SAF_has_mask_genes",
        mask_set.issubset(set(saf.index))
    )

    saf = saf.reindex(mask_genes)

    lengths = (
        pd.to_numeric(saf["End"])
        - pd.to_numeric(saf["Start"])
        + 1
    ).astype(float)

    require(
        "SAF_mask_lengths_positive",
        np.isfinite(lengths.to_numpy()).all()
        and (lengths > 0).all()
    )

    print()
    print("========== SOFT-CORE TPM SENSITIVITY ==========")

    # Recompute the 4923-gene TPM denominator for each
    # Atlanta and CFBE sample separately.
    atl_mask_log = counts_to_logtpm(
        atl_counts.reindex(mask_genes).reindex(
            columns=SAMPLES
        ),
        lengths
    )

    cfbe_mask_log = counts_to_logtpm(
        cfbe_counts.reindex(mask_genes).reindex(
            columns=CFBE_SAMPLES
        ),
        lengths
    )

    refs_mask = pd.DataFrame({
        "PRIMARY_CFBE_ALL4":
            cfbe_mask_log[CFBE_SAMPLES].mean(axis=1),

        "SENSITIVITY_CFBE_C_PAIR":
            cfbe_mask_log[["C1", "C2"]].mean(axis=1),

        "SENSITIVITY_CFBE_R_PAIR":
            cfbe_mask_log[["R1", "R2"]].mean(axis=1)
    })

    require(
        "masked_reference_4923_by_3",
        refs_mask.shape == (4923, 3)
    )

    print()
    print("========== BLINDED PROJECTION ==========")

    raw_results = {}
    cal_results = {}
    fit_rows = []

    for normalisation in NORMS:

        if normalisation == NORMS[0]:
            expr = atl_full.reindex(mask_genes)
            reference = ref_full.reindex(mask_genes)
        else:
            expr = atl_mask_log
            reference = refs_mask

        for ref_name in REFS:
            key = (normalisation, ref_name)

            delta = expr.sub(
                reference[ref_name], axis=0
            )

            d = delta.to_numpy(dtype=float)

            require(
                normalisation + "_" + ref_name
                + "_finite_contrast",
                np.isfinite(d).all()
            )

            # Same 116-programme operator as Audit 03f.
            a = operator @ d
            reconstruction = matrix @ a

            # The intercept cancels for expression contrasts.
            calibrated = (
                a * slopes.to_numpy(dtype=float)[:, None]
            )

            require(
                normalisation + "_" + ref_name
                + "_finite_activity",
                np.isfinite(calibrated).all()
            )

            raw_results[key] = pd.DataFrame(
                a, index=programmes, columns=SAMPLES
            )

            cal_results[key] = pd.DataFrame(
                calibrated,
                index=programmes,
                columns=SAMPLES
            )

            for j, sample in enumerate(SAMPLES):
                observed = d[:, j]
                predicted = reconstruction[:, j]

                total = float(
                    np.dot(observed, observed)
                )

                fitted = float(
                    np.dot(predicted, predicted)
                )

                residual = float(
                    np.dot(
                        observed - predicted,
                        observed - predicted
                    )
                )

                if total <= 1e-15:
                    raise RuntimeError(
                        "Zero contrast energy: "
                        + sample + " " + str(key)
                    )

                energy = fitted / total
                relative_error = np.sqrt(
                    residual / total
                )

                require(
                    "orthogonal_projection_identity_"
                    + normalisation + "_" + ref_name
                    + "_" + sample,
                    abs(
                        energy
                        + relative_error ** 2
                        - 1
                    ) <= 1e-7
                )

                fit_rows.append({
                    "normalisation": normalisation,
                    "reference": ref_name,
                    "sample": sample,
                    "projection_pearson":
                        pearson(observed, predicted),
                    "projection_energy_fraction":
                        energy,
                    "relative_reconstruction_error":
                        relative_error,
                    "input_contrast_l2":
                        np.sqrt(total),
                    "raw_activity_l2":
                        float(np.linalg.norm(a[:, j]))
                })

            print(
                normalisation, ref_name, "DONE",
                flush=True
            )

    fit = pd.DataFrame(fit_rows)

    require(
        "all_78_projection_fits",
        len(fit) == 78
    )

    require(
        "all_QC_metrics_finite",
        np.isfinite(
            fit.select_dtypes(
                include="number"
            ).to_numpy()
        ).all()
    )

    require(
        "projection_energy_in_unit_interval",
        fit["projection_energy_fraction"].between(
            -1e-9, 1 + 1e-9
        ).all()
    )

    print()
    print("========== SENSITIVITY COMPARISONS ==========")

    stability_rows = []

    for sample in SAMPLES:

        # Compare the C and R references with ALL4.
        for norm in NORMS:
            primary = cal_results[
                (norm, REFS[0])
            ].loc[selected, sample].to_numpy()

            for reference in REFS[1:]:
                other = cal_results[
                    (norm, reference)
                ].loc[selected, sample].to_numpy()

                stability_rows.append({
                    "sample": sample,
                    "comparison": "reference",
                    "normalisation": norm,
                    "alternative": reference,
                    "cosine_108":
                        cosine(primary, other),
                    "pearson_108":
                        pearson(primary, other),
                    "relative_change_108":
                        float(
                            np.linalg.norm(other - primary)
                            / max(np.linalg.norm(primary), 1e-12)
                        )
                })

        # Compare normalisation over 5567 and 4923 genes.
        for reference in REFS:
            primary = cal_results[
                (NORMS[0], reference)
            ].loc[selected, sample].to_numpy()

            other = cal_results[
                (NORMS[1], reference)
            ].loc[selected, sample].to_numpy()

            stability_rows.append({
                "sample": sample,
                "comparison": "normalisation",
                "normalisation": NORMS[0],
                "alternative": NORMS[1] + "__" + reference,
                "cosine_108":
                    cosine(primary, other),
                "pearson_108":
                    pearson(primary, other),
                "relative_change_108":
                    float(
                        np.linalg.norm(other - primary)
                        / max(np.linalg.norm(primary), 1e-12)
                    )
            })

    stability = pd.DataFrame(stability_rows)

    require(
        "stability_91_rows",
        len(stability) == 91
    )

    require(
        "stability_metrics_finite",
        np.isfinite(
            stability.select_dtypes(
                include="number"
            ).to_numpy()
        ).all()
    )

    print()
    print("========== PRIMARY PROJECTION QC ==========")

    primary_fit = fit.loc[
        (fit["normalisation"] == NORMS[0])
        & (fit["reference"] == REFS[0])
    ]

    print(
        primary_fit[
            [
                "sample",
                "projection_pearson",
                "projection_energy_fraction",
                "relative_reconstruction_error"
            ]
        ].round(5).to_string(index=False)
    )

    print()
    print("========== PROGRAMME STABILITY ==========")

    stability_summary = (
        stability.groupby("comparison")[
            ["cosine_108", "relative_change_108"]
        ].agg(["min", "median", "max"])
    )

    print(stability_summary.round(4).to_string())

    print()
    print("========== FREEZE TECHNICAL OUTPUTS ==========")

    for key, activities in cal_results.items():
        normalisation, ref_name = key

        # Retain raw coordinates for technical traceability.
        raw_results[key].rename_axis(
            "imodulon"
        ).reset_index().to_csv(
            OUT / (
                "FROZEN_116_raw_activities__"
                + normalisation + "__" + ref_name + ".csv"
            ),
            index=False,
            float_format="%.16g"
        )

        # Primary downstream 108-programme coordinates.
        activities.loc[selected].rename_axis(
            "imodulon"
        ).reset_index().to_csv(
            OUT / (
                "FROZEN_108_activities__"
                + normalisation + "__" + ref_name + ".csv"
            ),
            index=False,
            float_format="%.16g"
        )

    fit.to_csv(
        OUT / "FROZEN_projection_fit_QC_78.csv",
        index=False
    )

    stability.to_csv(
        OUT / "FROZEN_108_programme_sensitivity_91.csv",
        index=False
    )

    pd.DataFrame({
        "GeneID": mask_genes
    }).to_csv(
        OUT / "FROZEN_softcore_gene_order_4923.csv",
        index=False
    )

    pd.DataFrame({
        "imodulon": programmes,
        "calibration_slope": slopes.to_numpy(),
        "selected_108": [
            p in selected_set for p in programmes
        ]
    }).to_csv(
        OUT / "FROZEN_projection_programme_manifest.csv",
        index=False
    )

    refs_mask.rename_axis(
        "GeneID"
    ).reset_index().to_csv(
        OUT / "FROZEN_CFBE_softcore_4923_log2TPM_references.csv",
        index=False
    )

    atl_mask_log.rename_axis(
        "GeneID"
    ).reset_index().to_csv(
        OUT / "FROZEN_Atlanta_softcore_4923_log2TPM.csv",
        index=False
    )

    pd.DataFrame([
        {
            "source": name,
            "path": str(path),
            "sha256": sha256(path)
        }
        for name, path in inputs.items()
    ]).to_csv(
        OUT / "input_sha256_provenance.csv",
        index=False
    )

    pd.DataFrame(CHECKS).to_csv(
        OUT / "audit_pass_checks.csv",
        index=False
    )

    notes = (
        "Audit 04k: blinded technical projection.\n"
        "116-programme aeruPRECISE pseudoinverse.\n"
        "Audit 04k2 shared mask: 4923 genes, derived from the original frozen 4925-gene mask.\n"
        "Frozen 03e calibration slopes applied to contrasts.\n"
        "108 stable programmes selected after 116-programme fit.\n"
        "Primary: frozen 5567-gene TPM contrasts restricted to mask.\n"
        "Sensitivity: TPM denominator recomputed within 4923 genes.\n"
        "Primary reference: CFBE ALL4.\n"
        "Reference sensitivities: CFBE C-pair and R-pair.\n"
        "Projection reconstruction QC uses raw 116 activities.\n"
        "No new sample exclusion or projection-fit thresholds.\n"
        "No centring between Atlanta samples.\n"
        "C/R provenance does not establish RSV infection status.\n"
        "No Audit-03h candidate accessed.\n"
        "Independent validation NOT yet performed.\n"
    )

    (OUT / "AUDIT_04K4_METHODS.txt").write_text(
        notes, encoding="utf-8"
    )

    status = (
        "PRJNA576508_ATLANTA_13_4923_MASK_"
        "BLINDED_SOFTCORE_PROJECTIONS_FROZEN"
    )

    (OUT / "AUDIT_04K4_STATUS.txt").write_text(
        status + "\n"
    )

    print()
    print("STATUS:", status)
    print(
        "NEXT: blinded assessment of projection QC, "
        "not candidate unblinding."
    )


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "04k4 output already exists. "
            "Refusing to overwrite audit history."
        )

    OUT.mkdir(parents=True)

    try:
        main()
    except Exception:
        (OUT / "AUDIT_04K4_STATUS.txt").write_text(
            "PRJNA576508_AUDIT_04K4_REVIEW_REQUIRED\n"
        )

        (OUT / "AUDIT_04K4_FAILURE.txt").write_text(
            traceback.format_exc()
        )

        pd.DataFrame(CHECKS).to_csv(
            OUT / "audit_pass_checks.csv",
            index=False
        )

        raise
