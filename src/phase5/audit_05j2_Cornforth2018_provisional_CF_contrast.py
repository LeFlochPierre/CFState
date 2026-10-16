from pathlib import Path
import hashlib
import json
import traceback

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / "results/phase4"
P5 = ROOT / "results/phase5"
OUT = P5 / "audit_05j2_Cornforth2018_provisional_CF_contrast"

INPUTS = {
    "05i_status": P5 / "audit_05i_Cornforth2018_preprojection_QC/AUDIT_05I_STATUS.txt",
    "05h_status": P5 / "audit_05h_Cornforth2018_full_depth_baseline/AUDIT_05H_STATUS.txt",
    "counts": P5 / "audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv",
    "manifest": P5 / "audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv",
    "saf": ROOT / "references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf",
    "mask": P4 / "audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv",
    "modulome": ROOT / "external/aeruprecise/precise411/paeru_116_iModulon.json",
    "calibration": P4 / "audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv",
    "stable": P4 / "audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv",
    "library": P4 / "audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv",
    "candidate": P4 / "audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv",
    "registry18": P4 / "audit_03h_freeze_replicated_CF_airway_candidate/conservatively_replicated_programmes.csv",
    "script": Path(__file__).resolve(),
}


def require(name, test):
    passed = bool(test)
    print(name, "PASS" if passed else "FAIL", flush=True)
    if not passed:
        raise RuntimeError("Failed: " + name)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_m(path, genes, imodulons):
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict) or "M" not in obj:
        raise RuntimeError("Reference JSON does not have an M entry; inspect 04k4 loader")
    obj = obj["M"]
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and all(k in obj for k in ("index", "columns", "data")):
        m = pd.DataFrame(obj["data"], index=obj["index"], columns=obj["columns"])
    else:
        m = pd.DataFrame(obj)
    m.index = m.index.map(str)
    m.columns = m.columns.map(str)
    required_g = set(genes)
    required_i = set(imodulons)
    if required_g.issubset(m.columns) and required_i.issubset(m.index):
        m = m.T
    require("modulome_gene_and_imodulon_ids", required_g.issubset(m.index) and required_i.issubset(m.columns))
    require("modulome_identifiers_unique", m.index.is_unique and m.columns.is_unique)
    m = m.reindex(index=genes, columns=imodulons).apply(pd.to_numeric, errors="raise")
    require("modulome_finite", np.isfinite(m.to_numpy(dtype=float)).all())
    return m


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        raise RuntimeError("Undefined cosine for zero-norm vector")
    return float(np.dot(a, b) / denom)


def log2_tpm(counts, lengths):
    rates = counts.div(lengths, axis=0)
    denominator = rates.sum(axis=0)
    require("TPM_denominators_positive", np.isfinite(denominator).all() and (denominator > 0).all())
    tpm = rates.div(denominator, axis=1) * 1e6
    require("TPM_columns_sum_to_1e6", np.allclose(tpm.sum(axis=0), 1e6, atol=.01, rtol=0))
    return np.log2(tpm + 1)


def main():
    print("=" * 74)
    print("CF-State Phase 5 — Audit 05j2: provisional clinical-group projection")
    print("=" * 74)
    for name, path in INPUTS.items():
        require("input_" + name, path.is_file())
    require("05i_design_locked", INPUTS["05i_status"].read_text().strip() == "CFSTATE_PHASE5_05I_PREPROJECTION_QC_AND_DESIGN_LOCKED")
    require("05h_counts_frozen", INPUTS["05h_status"].read_text().strip() == "CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN")

    counts = pd.read_csv(INPUTS["counts"], index_col=0)
    counts.index = counts.index.map(str)
    manifest = pd.read_csv(INPUTS["manifest"], dtype=str).fillna("")
    require("counts_5567_by_13", counts.shape == (5567, 13) and counts.index.is_unique and counts.columns.is_unique)
    require("manifest_13_and_matching", len(manifest) == 13 and set(manifest.published_sample) == set(counts.columns) and manifest.published_sample.is_unique)
    require("groups_7_CF_4_wound_2_burn", (manifest.group == "CF_sputum").sum() == 7 and (manifest.group == "nonCF_chronic_wound").sum() == 4 and (manifest.group == "nonCF_burn").sum() == 2)
    require("12_unique_donors", manifest.published_donor_group.nunique() == 12)
    counts = counts.loc[:, manifest.published_sample.tolist()].apply(pd.to_numeric, errors="raise")
    require("nonnegative_finite_integer_counts", np.isfinite(counts.to_numpy()).all() and (counts.to_numpy() >= 0).all() and np.allclose(counts.to_numpy(), np.round(counts.to_numpy())))

    saf = pd.read_csv(INPUTS["saf"], sep="\t")
    require("SAF_columns", {"GeneID", "Start", "End"}.issubset(saf.columns))
    require("SAF_gene_universe", saf.GeneID.is_unique and set(saf.GeneID) == set(counts.index))
    saf = saf.set_index("GeneID").loc[counts.index]
    lengths = (pd.to_numeric(saf.End) - pd.to_numeric(saf.Start) + 1).astype(float)
    require("SAF_positive_lengths", np.isfinite(lengths).all() and (lengths > 0).all())

    mask = pd.read_csv(INPUTS["mask"])["GeneID"].astype(str).tolist()
    require("frozen_mask_4923", len(mask) == 4923 and len(set(mask)) == 4923 and set(mask).issubset(set(counts.index)))
    cal = pd.read_csv(INPUTS["calibration"]).set_index("imodulon")
    stable = pd.read_csv(INPUTS["stable"])["imodulon"].astype(str).tolist()
    require("116_calibrations_108_stable", cal.shape[0] == 116 and cal.index.is_unique and len(stable) == 108 and len(set(stable)) == 108 and set(stable).issubset(cal.index))
    model_names = cal.index.tolist()
    slopes = pd.to_numeric(cal.loc[model_names, "calibration_slope"], errors="raise").to_numpy(float)
    require("slopes_finite_nonzero", np.isfinite(slopes).all() and (np.abs(slopes) > 1e-12).all())

    M = load_m(INPUTS["modulome"], mask, model_names)
    design = M.to_numpy(float)
    require("operator_shape_rank", design.shape == (4923, 116) and np.linalg.matrix_rank(design) == 116)
    pinv = np.linalg.pinv(design)
    print("M shape:", design.shape, "condition number:", round(float(np.linalg.cond(design)), 5))

    library = pd.read_csv(INPUTS["library"], index_col=0)
    library.index = library.index.astype(str).str.strip()
    require("controlled_state_library", library.index.is_unique and set(stable).issubset(library.index) and library.shape[1] == 11)
    B = library.loc[stable].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, copy=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    require("known_state_matrix_rank", np.isfinite(B).all() and np.linalg.matrix_rank(B) == 11)

    candidate = pd.read_csv(INPUTS["candidate"], index_col=0)
    candidate.index = candidate.index.astype(str).str.strip()
    require("candidate_exact_same_108", len(candidate) == 108 and candidate.index.is_unique and set(candidate.index) == set(stable))
    c = pd.to_numeric(candidate.loc[stable, "two_cohort_consensus_unit"], errors="raise").to_numpy(float)
    require("candidate_unit_norm", np.isfinite(c).all() and abs(np.linalg.norm(c) - 1) < 1e-6)
    frozen18 = pd.read_csv(INPUTS["registry18"])
    chosen18 = frozen18.imodulon.astype(str).tolist()
    require("frozen_18_names", len(chosen18) == 18 and len(set(chosen18)) == 18 and set(chosen18).issubset(set(stable)))

    records = []
    loo = []
    all_activities = {}
    all_residuals = {}
    for mode in ["FROZEN_5567_TPM", "SOFTCORE_4923_TPM"]:
        source = counts if mode == "FROZEN_5567_TPM" else counts.loc[mask]
        expr = log2_tpm(source, lengths.loc[source.index]).loc[mask]
        donors = []
        groups = {}
        donor_expr = {}
        for donor, group in manifest.groupby("published_donor_group", sort=False):
            samples = group.published_sample.tolist()
            donors.append(donor)
            types = set(group.group)
            require("donor_consistent_group_" + donor, len(types) == 1)
            groups[donor] = "CF" if group.group.iloc[0] == "CF_sputum" else "nonCF"
            donor_expr[donor] = expr[samples].mean(axis=1)
        D = pd.DataFrame(donor_expr).loc[mask, donors]
        cf = [d for d in donors if groups[d] == "CF"]
        noncf = [d for d in donors if groups[d] == "nonCF"]
        require("donor_groups_7_vs_5", len(cf) == 7 and len(noncf) == 5)
        delta = D[cf].mean(axis=1).to_numpy(float) - D[noncf].mean(axis=1).to_numpy(float)
        anchor = D.mean(axis=1).to_numpy(float)
        centered = D.to_numpy(float) - anchor[:, None]
        raw = pinv @ centered
        a116 = raw * slopes[:, None]
        a = pd.DataFrame(a116, index=model_names, columns=donors).loc[stable]
        beta = np.linalg.lstsq(B, a.to_numpy(float), rcond=None)[0]
        R = a.to_numpy(float) - B @ beta
        require("orthogonal_residuals_" + mode, np.allclose(B.T @ R, 0, atol=1e-7))
        residual = pd.DataFrame(R, index=stable, columns=donors)
        delta_raw = (pinv @ delta) * slopes
        delta_a = pd.Series(delta_raw, index=model_names).loc[stable].to_numpy(float)
        delta_r = delta_a - B @ np.linalg.lstsq(B, delta_a, rcond=None)[0]
        empirical_delta = residual[cf].mean(axis=1).to_numpy(float) - residual[noncf].mean(axis=1).to_numpy(float)
        require("group_difference_reference_invariant_" + mode, np.allclose(delta_r, empirical_delta, atol=1e-7))
        require("nonzero_group_contrast_" + mode, np.linalg.norm(delta_r) > 1e-12)
        reconstruction = design @ (pinv @ delta)
        fit = float(np.dot(reconstruction, reconstruction) / np.dot(delta, delta))
        require("group_gene_projection_energy_valid_" + mode, -1e-9 <= fit <= 1 + 1e-9)
        result = {
            "mode": mode, "n_CF_donors": 7, "n_nonCF_donors": 5,
            "CF_minus_nonCF_cosine_108": cosine(delta_r, c),
            "CF_minus_nonCF_residual_norm_108": float(np.linalg.norm(delta_r)),
            "contrast_gene_space_energy_fraction": fit,
            "frozen_18_direction_matches": int((np.sign(pd.Series(delta_r, index=stable).loc[chosen18].to_numpy()) == np.sign(pd.Series(c, index=stable).loc[chosen18].to_numpy())).sum()),
        }
        records.append(result)
        all_activities[mode] = a
        all_residuals[mode] = residual
        for d in donors:
            c2 = [x for x in cf if x != d]
            n2 = [x for x in noncf if x != d]
            r2 = residual[c2].mean(axis=1).to_numpy(float) - residual[n2].mean(axis=1).to_numpy(float)
            loo.append({"mode": mode, "excluded_donor": d, "excluded_group": groups[d], "cosine_108": cosine(r2, c), "n_CF": len(c2), "n_nonCF": len(n2)})
        pd.DataFrame({"imodulon": stable, "CF_minus_nonCF_residual": delta_r, "frozen_consensus": c}).to_csv(OUT / ("DESCRIPTIVE_108_programme_contrast__" + mode + ".csv"), index=False)
        a.to_csv(OUT / ("DESCRIPTIVE_108_anchored_donor_activities__" + mode + ".csv"), index_label="imodulon")
        residual.to_csv(OUT / ("DESCRIPTIVE_108_anchored_donor_residuals__" + mode + ".csv"), index_label="imodulon")

    pd.DataFrame(records).to_csv(OUT / "DESCRIPTIVE_primary_and_TPM_sensitivity.csv", index=False)
    pd.DataFrame(loo).to_csv(OUT / "DESCRIPTIVE_24_leave_one_donor_out_checks.csv", index=False)
    pd.DataFrame([{"input": name, "path": str(path), "sha256": sha256(path)} for name, path in INPUTS.items()]).to_csv(OUT / "input_sha256_provenance.csv", index=False)
    print("\n=== PROVISIONAL GROUP CONTRAST ===")
    print(pd.DataFrame(records).round(5).to_string(index=False))
    print("\n=== DESCRIPTIVE LEAVE-ONE-DONOR-OUT RANGE ===")
    print(pd.DataFrame(loo).groupby("mode").cosine_108.agg(["min", "median", "max"]).round(5).to_string())
    print("\nCAVEATS: PAO1-only mapping; wound/CF sequencing depth differs;")
    print("CF vs non-CF is confounded by tissue and collection site;")
    print("2018 CF donors may overlap published cohorts; no p-values.")
    print("No novel programme discovery and no model fitting.")
    status = "CFSTATE_PHASE5_05J2_PROVISIONAL_CF_NONCF_PROJECTION_COMPLETE"
    (OUT / "AUDIT_05J2_STATUS.txt").write_text(status + "\n")
    print("\nSTATUS:", status)


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError("Audit 05j2 exists; refusing overwrite")
    OUT.mkdir(parents=True)
    try:
        main()
    except Exception:
        (OUT / "AUDIT_05J2_STATUS.txt").write_text("CFSTATE_PHASE5_05J2_REVIEW_REQUIRED\n")
        (OUT / "AUDIT_05J_FAILURE.txt").write_text(traceback.format_exc())
        raise
