"""CF-State Phase 5 Audit 05k: post-hoc technical count-depth sensitivity.

This is an exploratory robustness analysis, not independent biological replication.
The 05j2 primary analysis and candidate are not modified.
"""
from pathlib import Path
import hashlib
import json
import traceback
import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / "results/phase4"
P5 = ROOT / "results/phase5"
PREV = P5 / "audit_05j2_Cornforth2018_provisional_CF_contrast"
OUT = P5 / "audit_05k_Cornforth2018_depth_matched_sensitivity"

INPUTS = {
    "05j2_status": PREV / "AUDIT_05J2_STATUS.txt",
    "counts": P5 / "audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv",
    "manifest": P5 / "audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv",
    "saf": ROOT / "references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf",
    "mask": P4 / "audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv",
    "modulome": ROOT / "external/aeruprecise/precise411/paeru_116_iModulon.json",
    "calibration": P4 / "audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv",
    "stable": P4 / "audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv",
    "library": P4 / "audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv",
    "candidate": P4 / "audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv",
    "05j2_baselines": PREV / "DESCRIPTIVE_primary_and_TPM_sensitivity.csv",
    "script": Path(__file__).resolve(),
}
MODES = ["FROZEN_5567_TPM", "SOFTCORE_4923_TPM"]
SEED = 20261013
REPLICATES = 48  # Descriptive Monte Carlo draws, not independent patients


def require(label, condition):
    if not bool(condition):
        raise RuntimeError("Failed check: " + label)
    print(label, "PASS", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_m(path, gene_ids, component_ids):
    obj = json.loads(path.read_text())["M"]
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and set(["index", "columns", "data"]).issubset(obj):
        matrix = pd.DataFrame(obj["data"], index=obj["index"], columns=obj["columns"])
    else:
        matrix = pd.DataFrame(obj)
    matrix.index = matrix.index.map(str)
    matrix.columns = matrix.columns.map(str)
    if set(gene_ids).issubset(matrix.columns) and set(component_ids).issubset(matrix.index):
        matrix = matrix.T
    require("modulome_compatible", set(gene_ids).issubset(matrix.index) and set(component_ids).issubset(matrix.columns))
    return matrix.loc[gene_ids, component_ids].to_numpy(dtype=float)


def main():
    print("=" * 76)
    print("CF-State Phase 5 — Audit 05k")
    print("Equal assigned-count depth technical sensitivity (post hoc)")
    print("=" * 76)
    for name, path in INPUTS.items():
        require("input_" + name, path.is_file())
    require(
        "05j2_status_frozen",
        INPUTS["05j2_status"].read_text().strip()
        == "CFSTATE_PHASE5_05J2_PROVISIONAL_CF_NONCF_PROJECTION_COMPLETE",
    )
    counts = pd.read_csv(INPUTS["counts"], index_col=0)
    counts.index = counts.index.astype(str)
    manifest = pd.read_csv(INPUTS["manifest"], dtype=str).fillna("")
    require("13_samples", counts.shape == (5567, 13) and len(manifest) == 13)
    require("matching_samples", set(manifest.published_sample) == set(counts.columns))
    counts = counts.loc[:, manifest.published_sample.tolist()].apply(pd.to_numeric, errors="raise")
    original = counts.to_numpy(dtype=np.int64)
    require("integer_nonnegative_counts", np.array_equal(original, counts.to_numpy()) and (original >= 0).all())
    total = original.sum(axis=0)
    require("all_samples_positive_depth", (total > 0).all())
    target = int(total.min())
    require("min_sample_CW21", manifest.published_sample.iloc[int(np.argmin(total))] == "Lubbock_CW_21")
    print("Minimum assigned reads (matched count depth):", target)

    saf = pd.read_csv(INPUTS["saf"], sep="\t").set_index("GeneID")
    require("SAF_gene_universe", saf.index.is_unique and set(counts.index) == set(saf.index))
    lengths = (saf.loc[counts.index, "End"] - saf.loc[counts.index, "Start"] + 1).to_numpy(float)
    require("gene_lengths_positive", np.isfinite(lengths).all() and (lengths > 0).all())
    mask = pd.read_csv(INPUTS["mask"])["GeneID"].astype(str).tolist()
    require("mask_4923", len(mask) == 4923 and len(set(mask)) == 4923 and set(mask).issubset(set(counts.index)))
    mask_idx = counts.index.get_indexer(mask)

    calibration = pd.read_csv(INPUTS["calibration"]).set_index("imodulon")
    model_names = calibration.index.tolist()
    stable = pd.read_csv(INPUTS["stable"])["imodulon"].astype(str).tolist()
    require("116_and_108", len(model_names) == 116 and len(stable) == 108 and set(stable).issubset(set(model_names)))
    stable_idx = np.array([model_names.index(x) for x in stable], dtype=int)
    slopes = calibration.loc[model_names, "calibration_slope"].to_numpy(float)
    require("slopes_finite", np.isfinite(slopes).all())
    M = load_m(INPUTS["modulome"], mask, model_names)
    require("M_shape_and_rank", M.shape == (4923, 116) and np.linalg.matrix_rank(M) == 116)
    op = np.linalg.pinv(M)

    state = pd.read_csv(INPUTS["library"], index_col=0)
    state.index = state.index.astype(str).str.strip()
    require("reference_11", len(state.columns) == 11 and set(stable).issubset(set(state.index)))
    B = state.loc[stable].to_numpy(dtype=float, copy=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    require("controlled_basis_rank", np.linalg.matrix_rank(B) == 11)
    # Residual operator: exactly the same least-squares subspace as 05j2.
    residual_operator = np.eye(108) - B @ np.linalg.pinv(B)
    candidate = pd.read_csv(INPUTS["candidate"], index_col=0)
    candidate.index = candidate.index.astype(str).str.strip()
    v = candidate.loc[stable, "two_cohort_consensus_unit"].to_numpy(float)
    require("candidate_unit", abs(float(np.linalg.norm(v)) - 1) < 1e-6)

    groupmap = manifest.set_index("published_sample")
    donor_order = groupmap.published_donor_group.drop_duplicates().tolist()
    donor_samples = [
        [manifest.published_sample.tolist().index(s) for s in groupmap.index[groupmap.published_donor_group == donor]]
        for donor in donor_order
    ]
    cf_idx = [i for i, donor in enumerate(donor_order) if (groupmap.loc[groupmap.published_donor_group == donor, "group"] == "CF_sputum").all()]
    noncf_idx = [i for i in range(len(donor_order)) if i not in cf_idx]
    require("donors_7_CF_5_nonCF", len(donor_order) == 12 and len(cf_idx) == 7 and len(noncf_idx) == 5)

    def cosine_from_gene_counts(raw_counts, mode):
        ix = np.arange(5567) if mode == MODES[0] else mask_idx
        rates = raw_counts[ix].astype(float) / lengths[ix, None]
        denom = rates.sum(axis=0)
        if not np.isfinite(denom).all() or (denom <= 0).any():
            raise RuntimeError("TPM denominator failed")
        expr = np.log2(rates / denom[None, :] * 1e6 + 1.0)
        if mode == MODES[0]:
            expr = expr[mask_idx, :]
        d = np.column_stack([expr[:, indices].mean(axis=1) for indices in donor_samples])
        difference = d[:, cf_idx].mean(axis=1) - d[:, noncf_idx].mean(axis=1)
        activities = (op @ difference) * slopes
        r = residual_operator @ activities[stable_idx]
        nr = np.linalg.norm(r)
        if nr <= 1e-12:
            raise RuntimeError("Zero residual contrast")
        return float(np.dot(r, v) / (nr * np.linalg.norm(v)))

    previous = pd.read_csv(INPUTS["05j2_baselines"]).set_index("mode")
    for mode in MODES:
        observed = cosine_from_gene_counts(original, mode)
        recorded = float(previous.loc[mode, "CF_minus_nonCF_cosine_108"])
        require("reproduce_frozen_05j2_" + mode, np.isclose(observed, recorded, atol=1e-9, rtol=0))
        print("Frozen", mode, "cosine:", round(observed, 6))

    print("\n========== MATCHED-DEPTH RESAMPLING ==========")
    rng = np.random.default_rng(SEED)
    rows = []
    for iteration in range(REPLICATES):
        rarefied = np.column_stack([
            rng.multivariate_hypergeometric(original[:, j], target)
            for j in range(original.shape[1])
        ])
        if not (rarefied.sum(axis=0) == target).all():
            raise RuntimeError("Depth matching failed")
        for mode in MODES:
            rows.append({
                "replicate": iteration + 1,
                "normalisation": mode,
                "matched_assigned_reads_per_specimen": target,
                "cosine_CF_minus_nonCF_vs_frozen_consensus": cosine_from_gene_counts(rarefied, mode),
            })
    results = pd.DataFrame(rows)
    summary = (results.groupby("normalisation")["cosine_CF_minus_nonCF_vs_frozen_consensus"]
               .agg(["min", "median", "max", lambda s: int((s > 0).sum())]).reset_index())
    summary.columns = ["normalisation", "min_cosine", "median_cosine", "max_cosine", "n_positive_out_of_48"]
    print(summary.round(5).to_string(index=False))
    require("exact_96_comparisons", len(results) == 2 * REPLICATES)

    results.to_csv(OUT / "POSTHOC_96_equal_assigned_depth_resampling.csv", index=False)
    summary.to_csv(OUT / "POSTHOC_equal_assigned_depth_summary.csv", index=False)
    pd.DataFrame([{"source": k, "path": str(p), "sha256": sha256(p)} for k, p in INPUTS.items()]).to_csv(
        OUT / "input_provenance_sha256.csv", index=False
    )
    (OUT / "AUDIT_05K_METHODS_AND_CAVEATS.txt").write_text(
        "Post hoc diagnostic only. Frozen 05j2 results are unchanged.\n"
        "For each of 48 Monte Carlo draws (seed 20261013), all 13 bacterial gene-count profiles\n"
        "are independently rarefied WITHOUT replacement to the lowest observed assigned gene\n"
        "count (Lubbock_CW_21). Retain all 12 donor groups; average the two burn specimens\n"
        "within their donor. Recompute TPM, fixed 116->108 projection, original slopes,\n"
        "11-state residual, and frozen-consensus CF-minus-nonCF cosine.\n"
        "Monte Carlo draws are not biological replicates or a confidence interval.\n"
        "Gene-count rarefaction does not simulate read trimming, mapping bias,\n"
        "strain divergence, library composition, sample site, or geographical confounding.\n"
        "Do not replace the frozen 05j2 primary result with this diagnostic.\n"
    )
    status = "CFSTATE_PHASE5_05K_POSTHOC_COUNT_DEPTH_SENSITIVITY_COMPLETE"
    (OUT / "AUDIT_05K_STATUS.txt").write_text(status + "\n")
    print("\nSTATUS:", status)


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError("Audit 05k exists; refusing overwrite")
    OUT.mkdir(parents=True)
    try:
        main()
    except Exception:
        (OUT / "AUDIT_05K_STATUS.txt").write_text("CFSTATE_PHASE5_05K_REVIEW_REQUIRED\n")
        (OUT / "AUDIT_05K_FAILURE.txt").write_text(traceback.format_exc())
        raise
