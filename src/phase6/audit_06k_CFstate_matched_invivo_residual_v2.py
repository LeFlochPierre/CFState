#!/usr/bin/env python3
"""CF-State Audit 06k: first supervised-by-matched-contrast, reference-anchored axis.

Training: four PRJEB24688 CF patients with matched in-vivo sputum and
in-vitro cultures. Equal EARLY/LATE laboratory-baseline contrast per patient.
Evaluation: leave-one-PRJEB-patient-out; one GSE119356 patient held out.
Neither Atlanta nor Cornforth is used to train, fit or select this axis.
This is an exploratory CF in-vivo-versus-laboratory signal, not CF specificity.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / "results/phase4"
OUT = ROOT / "results/phase6/audit_06k_CFstate_matched_invivo_residual_v2"
FILES = {
    "gse_expression": P4 / "audit_02a_GSE119356_matched_projection_QC/GSE119356_projection_log2TPM_plus1.csv",
    "prjeb_sputum": P4 / "audit_03f_PRJEB24688_multipatient_CF_replication/PRJEB24688_PRIMARY_sputum_TPM.csv",
    "prjeb_lab": P4 / "audit_03f_PRJEB24688_multipatient_CF_replication/PRJEB24688_matched_lab_TPM.csv",
    "prjeb_lab_metadata": P4 / "audit_03f_PRJEB24688_multipatient_CF_replication/matched_lab_sample_metadata.csv",
    "strict_design": P4 / "audit_03h_freeze_replicated_CF_airway_candidate/frozen_PRJEB24688_strict_patient_design.csv",
    "mask": P4 / "audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv",
    "stable": P4 / "audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv",
    "modulome": ROOT / "external/aeruprecise/precise411/paeru_116_iModulon.json",
    "script": Path(__file__).resolve(),
}
REGIONS = ("BR", "ULP", "ULC", "LLP", "LLC")
BASELINES = ("EARLY", "LATE")
EXPECTED_PATIENTS = {"P24M1", "P30M0", "P73F1", "P77F1"}


def check(label, condition):
    if not bool(condition):
        raise RuntimeError("FAILED: " + label)
    print(label + " PASS", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_gene_table(path, idcol):
    df = pd.read_csv(path)
    check("identifier_" + path.stem, idcol in df.columns)
    df[idcol] = df[idcol].astype(str)
    check("unique_ids_" + path.stem, df[idcol].is_unique)
    return df.set_index(idcol)


def load_reference_model(path, genes, programmes):
    obj = json.loads(path.read_text())["M"]
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and {"index", "columns", "data"}.issubset(obj):
        m = pd.DataFrame(obj["data"], index=obj["index"], columns=obj["columns"])
    else:
        m = pd.DataFrame(obj)
    m.index = m.index.map(str)
    m.columns = m.columns.map(str)
    if set(genes).issubset(m.columns) and set(programmes).issubset(m.index):
        m = m.T
    check("reference_identifiers", m.index.is_unique and m.columns.is_unique
          and set(genes).issubset(m.index) and set(programmes).issubset(m.columns))
    x = m.loc[genes, programmes].apply(pd.to_numeric, errors="raise").to_numpy(float, copy=True)
    check("reference_finite", np.isfinite(x).all())
    q, r = np.linalg.qr(x, mode="reduced")
    check("reference_108_rank", q.shape[1] == 108 and np.min(np.abs(np.diag(r))) > 1e-9)
    return q


def resid(vector, q):
    return vector - q @ (q.T @ vector)


def unit(vector):
    n = np.linalg.norm(vector)
    if n <= 1e-10:
        raise RuntimeError("Zero or near-zero fitted residual direction")
    return vector / n


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 1e-10 or nb <= 1e-10:
        raise RuntimeError("Undefined cosine")
    return float(a @ b / (na * nb))


def design_samples(s):
    return list(dict.fromkeys(re.findall(r"P\d+[MF]\d+_S\d+", str(s))))


def main():
    print("=" * 76)
    print("CF-State Phase 6 — Audit 06k: matched clinical/lab residual model")
    print("=" * 76)
    check("new_audit_directory", not OUT.exists())
    for label, path in FILES.items():
        check("input_" + label, path.is_file())

    mask = pd.read_csv(FILES["mask"])["GeneID"].astype(str).tolist()
    check("mask_4923_unique", len(mask) == 4923 and len(set(mask)) == 4923)
    gse = read_gene_table(FILES["gse_expression"], "Unnamed: 0")
    sputum = read_gene_table(FILES["prjeb_sputum"], "pao1_gene")
    lab = read_gene_table(FILES["prjeb_lab"], "pao1_gene")
    lab_md = pd.read_csv(FILES["prjeb_lab_metadata"], dtype=str).fillna("")
    design = pd.read_csv(FILES["strict_design"], dtype=str).fillna("")

    check("lab_metadata_fields", {"sample", "patient", "baseline"}.issubset(lab_md.columns))
    check("strict_design_fields", {"patient", "baseline", "n_sputum_samples", "samples"}.issubset(design.columns))
    check("lab_samples_exact", lab_md["sample"].is_unique and set(lab_md["sample"]) == set(lab.columns))
    check("design_eight_rows", len(design) == 8 and design[["patient", "baseline"]].drop_duplicates().shape[0] == 8)
    check("design_patients_four", set(design.patient) == EXPECTED_PATIENTS)
    check("design_two_baselines", set(design.baseline) == set(BASELINES))
    check("lab_four_patients", set(lab_md.patient) == EXPECTED_PATIENTS)
    check("lab_two_baselines", set(lab_md.baseline) == set(BASELINES))
    check("all_study_genes_unique", all(x.index.is_unique for x in (gse, sputum, lab)))

    genes = [g for g in mask if g in gse.index and g in sputum.index and g in lab.index]
    check("shared_gene_universe_sufficient_for_reference", len(genes) >= 108)
    print("Shared original masked genes:", len(genes), "/", len(mask))
    print("No missing gene is assigned a fabricated expression value.")

    for prefix, df in (("GSE", gse), ("PRJEB_sputum", sputum), ("PRJEB_lab", lab)):
        z = df.loc[genes].apply(pd.to_numeric, errors="raise").to_numpy(float)
        check("finite_nonnegative_" + prefix, np.isfinite(z).all() and (z >= 0).all())

    stable = pd.read_csv(FILES["stable"])["imodulon"].astype(str).tolist()
    check("stable_108", len(stable) == 108 and len(set(stable)) == 108)
    q = load_reference_model(FILES["modulome"], genes, stable)

    # GSE119356 is one patient across FIVE regions, not five independent donors.
    region_contrasts = {}
    for region in REGIONS:
        in_vivo, in_vitro = "invivo_" + region, "invitro_" + region
        check("gse_region_" + region, in_vivo in gse.columns and in_vitro in gse.columns)
        region_contrasts[region] = (gse.loc[genes, in_vivo] - gse.loc[genes, in_vitro]).to_numpy(float)
    gse_mean = np.mean(list(region_contrasts.values()), axis=0)
    gse_residual = resid(gse_mean, q)

    # PRJEB: PRIMARY = equal weight of EARLY (3H lab) and LATE (24H lab).
    # The SAME sputum sample may occur under both baselines; never duplicate donors.
    log_sputum = np.log2(sputum.loc[genes] + 1.0)
    log_lab = np.log2(lab.loc[genes] + 1.0)
    patient_baselines = {}
    design_records = []
    for patient in sorted(EXPECTED_PATIENTS):
        patient_baselines[patient] = {}
        for baseline in BASELINES:
            row = design.loc[(design.patient == patient) & (design.baseline == baseline)]
            check("one_design_row_" + patient + "_" + baseline, len(row) == 1)
            samples = design_samples(row.iloc[0]["samples"])
            check("design_sample_count_" + patient + "_" + baseline,
                  len(samples) == int(row.iloc[0]["n_sputum_samples"]))
            check("design_samples_in_matrix_" + patient + "_" + baseline,
                  len(samples) > 0 and set(samples).issubset(set(sputum.columns)))
            ref_samples = lab_md.loc[(lab_md.patient == patient) & (lab_md.baseline == baseline), "sample"].tolist()
            check("lab_samples_for_" + patient + "_" + baseline,
                  len(ref_samples) >= 1 and set(ref_samples).issubset(set(lab.columns)))
            diff = (log_sputum[samples].mean(axis=1) - log_lab[ref_samples].mean(axis=1)).to_numpy(float)
            patient_baselines[patient][baseline] = diff
            design_records.append({"patient": patient, "baseline": baseline,
                                   "n_sputum_samples": len(samples), "sputum_samples": ";".join(samples),
                                   "n_lab_profiles": len(ref_samples), "lab_profiles": ";".join(ref_samples)})

    patients = sorted(EXPECTED_PATIENTS)
    prjeb_primary = {}
    for patient in patients:
        prjeb_primary[patient] = np.mean([patient_baselines[patient][b] for b in BASELINES], axis=0)
    patient_residuals = {p: resid(prjeb_primary[p], q) for p in patients}

    # First v2 model: one prespecified patient-balanced directional axis.
    # GSE is held out completely from training and model selection.
    axis = unit(np.mean([patient_residuals[p] for p in patients], axis=0))
    gse_cos = cosine(axis, gse_residual)
    print("\n========== PRIMARY CROSS-STUDY HELD-OUT TEST ==========")
    print("Training: four PRJEB24688 patients; GSE119356 excluded")
    print("GSE119356 held-out patient cosine:", round(gse_cos, 5))
    print("GSE119356 residual variation along axis (%):", round(100 * gse_cos ** 2, 3))
    print("Signed projection (positive means concordant):", round(float(axis @ gse_residual), 5))

    loo = []
    for patient in patients:
        others = [p for p in patients if p != patient]
        train_axis = unit(np.mean([patient_residuals[p] for p in others], axis=0))
        score = cosine(train_axis, patient_residuals[patient])
        loo.append({"heldout_PRJEB_patient": patient, "signed_cosine": score,
                    "fraction_of_heldout_residual": score ** 2,
                    "concordant": score > 0})
    loo_df = pd.DataFrame(loo)
    print("\n========== PRJEB LEAVE-ONE-PATIENT-OUT ==========")
    print(loo_df.round(5).to_string(index=False))

    sensitivity = []
    for baseline in BASELINES:
        base_res = {p: resid(patient_baselines[p][baseline], q) for p in patients}
        base_axis = unit(np.mean([base_res[p] for p in patients], axis=0))
        sensitivity.append({"training_baseline": baseline,
                            "axis_cosine_vs_primary": cosine(base_axis, axis),
                            "heldout_GSE119356_cosine": cosine(base_axis, gse_residual),
                            "all4_PRJEB_positive_vs_primary": all(cosine(axis, base_res[p]) > 0 for p in patients)})
    sensitivity_df = pd.DataFrame(sensitivity)
    print("\n========== LABORATORY BASELINE SENSITIVITY ==========")
    print(sensitivity_df.round(5).to_string(index=False))

    region_rows = []
    for region, vec in region_contrasts.items():
        rv = resid(vec, q)
        region_rows.append({"GSE119356_region_same_patient": region,
                            "cosine_vs_fitted_axis": cosine(rv, axis)})
    region_df = pd.DataFrame(region_rows)
    print("\n========== GSE WITHIN-PATIENT REGION CONSISTENCY ==========")
    print(region_df.round(5).to_string(index=False))

    # These are fit QC, not biological pass/fail thresholds.
    check("axis_orthogonal_to_existing_108", np.max(np.abs(q.T @ axis)) < 1e-8)
    check("axis_unit_norm", abs(float(np.linalg.norm(axis)) - 1) < 1e-8)
    check("five_unique_CF_patients_total", len(patients) + 1 == 5)
    check("all_outputs_finite", np.isfinite(gse_cos) and np.isfinite(loo_df.signed_cosine).all())

    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(design_records).to_csv(OUT / "FROZEN_PRJEB_patient_baseline_pairing.csv", index=False)
    pd.DataFrame({"GeneID": genes, **prjeb_primary,
                  "GSE119356_5regions_1patient": gse_mean}).to_csv(
                      OUT / "FROZEN_five_patient_matched_gene_contrasts.csv", index=False)
    pd.DataFrame({"GeneID": genes, **patient_residuals,
                  "GSE119356_heldout": gse_residual}).to_csv(
                      OUT / "FROZEN_five_patient_reference_residuals.csv", index=False)
    pd.DataFrame({"GeneID": genes, "CF_invivo_minus_lab_residual_axis": axis}).to_csv(
        OUT / "FITTED_four_patient_CF_invivo_matched_residual_axis.csv", index=False)
    loo_df.to_csv(OUT / "PRJEB_four_patient_LOPO.csv", index=False)
    sensitivity_df.to_csv(OUT / "EARLY_LATE_reference_sensitivity.csv", index=False)
    region_df.to_csv(OUT / "GSE_five_regions_one_donor_sensitivity.csv", index=False)
    np.savez_compressed(OUT / "FITTED_CFstate_v2_candidate_arrays.npz",
                        genes=np.array(genes), programmes=np.array(stable),
                        reference_Q=q, added_direction=axis,
                        training_patients=np.array(patients))
    pd.DataFrame([{"input": key, "path": str(path), "sha256": sha256(path)}
                  for key, path in FILES.items()]).to_csv(OUT / "input_sha256_provenance.csv", index=False)
    (OUT / "DESIGN_AND_LIMITATIONS.txt").write_text(
        "Four PRJEB CF patients train one prespecified mean residual contrast direction.\n"
        "EARLY/LATE are alternative laboratory baselines, NOT independent clinical specimens.\n"
        "PRJEB samples from frozen strict design are averaged per patient and baseline.\n"
        "One GSE119356 patient is held out, with five regions averaged BEFORE evaluating.\n"
        "The model is in the shared intersection of the frozen 4923-gene mask.\n"
        "Orthogonal reference subspace uses 108 stable aeruPRECISE programmes.\n"
        "No model selection uses the GSE held-out patient.\n"
        "CF in-vivo versus matched laboratory culture does not equal CF specificity.\n"
        "GSE and PRJEB previously informed CF-State research; NOT pristine independent validation.\n"
        "Only four training patients; LOPO stability is exploratory.\n"
        "Projection/contrast alignment is not unseen-gene or physiological-state prediction.\n"
    )
    status = "CFSTATE_PHASE6_06K_CF_INVIVO_MATCHED_CONTRAST_MODEL_FITTED_EXPLORATORY"
    (OUT / "AUDIT_06K_STATUS.txt").write_text(status + "\n")
    print("\nSTATUS:", status)


if __name__ == "__main__":
    main()
