#!/usr/bin/env python3
"""CF-State Phase 6, Audit 06i: fit a variable-rank reference-anchored model.

Use only Atlanta expression for hyperparameter selection, nested specimen-held-out
checks, and final fitting. Frozen 108 aeruPRECISE gene-space span is unchanged.
This is an exploratory specimen-level model: donor identities are unresolved.
Prior Cornforth analyses are not independent validation and are not used here.

Model: project centered log2(TPM+1) to orthogonal complement of frozen
reference gene space, estimate residual PCA on training specimens, truncate
selected gene-readout weights, then re-project to the reference complement.
The effective reconstruction directions are dense: gene readouts are sparse.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/phase6/audit_06i_CFstate_data_driven_rank_sparse_v1"
ATL = ROOT / "results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_Atlanta_softcore_4923_log2TPM.csv"
MASK = ROOT / "results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv"
MODULOME = ROOT / "external/aeruprecise/precise411/paeru_116_iModulon.json"
CAL = ROOT / "results/phase4/audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv"
STABLE = ROOT / "results/phase4/audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv"
PRIOR = ROOT / "results/phase6/audit_06h_CFstate_sparse_anchored_v0/AUDIT_06H_STATUS.txt"
SCRIPT = Path(__file__).resolve()
INPUTS = {"Atlanta": ATL, "mask": MASK, "modulome": MODULOME, "calibration": CAL, "stable": STABLE, "06h_status": PRIOR, "script": SCRIPT}

# Fixed before running this audit: no evaluation on Cornforth for choosing K or B.
K_VALUES = (1, 2, 3, 4, 5, 6)
GENE_BUDGETS = (50, 100, 200, 500, 1000, 2000)
FRACTION_EPS = 1e-10


def require(message, valid):
    if not bool(valid):
        raise RuntimeError("FAILED: " + message)
    print(message + " PASS", flush=True)


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1048576), b""):
            digest.update(part)
    return digest.hexdigest()


def read_modulome(path, genes, names):
    obj = json.loads(path.read_text())["M"]
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and {"index", "columns", "data"}.issubset(obj):
        mat = pd.DataFrame(obj["data"], index=obj["index"], columns=obj["columns"])
    else:
        mat = pd.DataFrame(obj)
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    if set(genes).issubset(mat.columns) and set(names).issubset(mat.index):
        mat = mat.T
    require("modulome_identifiers", mat.index.is_unique and mat.columns.is_unique and set(genes).issubset(mat.index) and set(names).issubset(mat.columns))
    m = mat.loc[genes, names].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float, copy=True)
    require("finite_modulome", np.isfinite(m).all())
    return m


def residualise(x, reference_q):
    return x - reference_q @ (reference_q.T @ x)


def left_pcs(residual_training, max_k):
    """SVD using the small sample Gram matrix to avoid large repeated SVDs."""
    gram = residual_training.T @ residual_training
    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    top_vals = eigvals[order[:max_k]]
    if len(top_vals) < max_k or np.min(top_vals) <= 1e-10:
        raise RuntimeError("Insufficient independent residual training directions")
    basis = residual_training @ eigvecs[:, order[:max_k]]
    basis /= np.sqrt(top_vals)[None, :]
    return basis


def sparse_seeds_from_pcs(pcs, budget):
    if budget is None:
        seeds = pcs.copy()
    else:
        seeds = np.zeros_like(pcs)
        for column in range(pcs.shape[1]):
            selected = np.argpartition(np.abs(pcs[:, column]), -budget)[-budget:]
            seeds[selected, column] = pcs[selected, column]
        seeds /= np.linalg.norm(seeds, axis=0, keepdims=True)
    return seeds


def effective_axes(seeds, ref_q):
    directions = residualise(seeds, ref_q)
    gram = directions.T @ directions
    vals = np.linalg.eigvalsh(gram)
    if vals[0] <= 1e-9 * vals[-1]:
        raise RuntimeError("Residual extension directions rank deficient")
    q, _ = np.linalg.qr(directions, mode="reduced")
    return q


def candidate_scores(residual_training, residual_test, ref_q):
    """Evaluate all K/budget choices without refitting the reference."""
    pcs = left_pcs(residual_training, max(K_VALUES))
    t = residual_test[:, 0]
    denominator = float(t @ t)
    if denominator <= FRACTION_EPS:
        raise RuntimeError("Zero held-out residual energy")
    records = []
    for budget in GENE_BUDGETS:
        seeds = sparse_seeds_from_pcs(pcs, budget)
        e = residualise(seeds, ref_q)
        gram = e.T @ e
        dots = e.T @ t
        for k in K_VALUES:
            g = gram[:k, :k]
            b = dots[:k]
            try:
                numerator = float(b @ np.linalg.solve(g, b))
            except np.linalg.LinAlgError as exc:
                raise RuntimeError("Singular basis at K={}, budget={}".format(k, budget)) from exc
            score = float(np.clip(numerator / denominator, 0.0, 1.0))
            records.append((k, budget, score))
    return records


def cv_scores(expression, names, ref_q):
    """Returns one row per candidate for every leave-one-specimen-out fold."""
    records = []
    for j, heldout in enumerate(names):
        training = np.delete(expression, j, axis=1)
        train_mean = training.mean(axis=1, keepdims=True)
        r_train = residualise(training - train_mean, ref_q)
        r_test = residualise(expression[:, j:j + 1] - train_mean, ref_q)
        for k, budget, score in candidate_scores(r_train, r_test, ref_q):
            records.append({"heldout_specimen": heldout, "k": k, "gene_budget": budget, "fraction_explained": score})
    return pd.DataFrame.from_records(records)


def choose_one_se(scores):
    """Heuristic one-SE rule; folds are correlated, NOT a formal confidence bound."""
    summary = scores.groupby(["k", "gene_budget"], as_index=False).agg(
        mean_fraction=("fraction_explained", "mean"),
        median_fraction=("fraction_explained", "median"),
        sd_fraction=("fraction_explained", "std"),
        n_folds=("fraction_explained", "size"),
    )
    require("all_candidate_folds", summary.n_folds.eq(scores.heldout_specimen.nunique()).all())
    best = summary.sort_values(["mean_fraction", "k", "gene_budget"], ascending=[False, True, True]).iloc[0]
    best_mean = float(best.mean_fraction)
    one_se = float(best.sd_fraction / np.sqrt(best.n_folds))
    threshold = best_mean - one_se
    # Prefer fewer latent directions, then fewer gene readouts, among near-best models.
    eligible = summary[summary.mean_fraction >= threshold - 1e-12].sort_values(["k", "gene_budget"])
    chosen = eligible.iloc[0]
    return int(chosen.k), int(chosen.gene_budget), summary, best_mean, one_se


def fit_final(expression, ref_q, k, budget):
    mean = expression.mean(axis=1, keepdims=True)
    r = residualise(expression - mean, ref_q)
    pcs = left_pcs(r, max(k, 1))[:, :k]
    seeds = sparse_seeds_from_pcs(pcs, budget)
    axes = effective_axes(seeds, ref_q)
    return mean[:, 0], seeds, axes


def explained_for_one_fold(expression, heldout_index, ref_q, k, budget):
    training = np.delete(expression, heldout_index, axis=1)
    mean, seeds, axes = fit_final(training, ref_q, k, budget)
    target = residualise(expression[:, heldout_index] - mean, ref_q)
    denominator = float(target @ target)
    return float(np.clip(np.sum((axes.T @ target)**2) / denominator, 0, 1))


def main():
    print("=" * 78)
    print("CF-State Phase 6 — Audit 06i: data-driven rank and gene-budget model")
    print("=" * 78)
    require("new_output_directory", not OUT.exists())
    for key, path in INPUTS.items():
        require("input_" + key, path.is_file())
    require("06h_fitted", PRIOR.read_text().strip() == "CFSTATE_PHASE6_06H_SPARSE_ANCHORED_V0_FITTED_EXPLORATORY")
    genes = pd.read_csv(MASK)["GeneID"].astype(str).tolist()
    require("mask_4923", len(genes) == 4923 and len(set(genes)) == len(genes))
    atl = pd.read_csv(ATL).set_index("GeneID")
    require("Atlanta_13_expression_profiles", atl.shape == (4923, 13) and set(atl.index) == set(genes) and atl.columns.is_unique)
    sample_names = atl.columns.astype(str).tolist()
    x = atl.loc[genes].to_numpy(dtype=float, copy=True)
    require("Atlanta_finite_expression", np.isfinite(x).all())
    calib = pd.read_csv(CAL)
    stable = pd.read_csv(STABLE)["imodulon"].astype(str).tolist()
    require("reference_registry_108", len(stable) == 108 and len(set(stable)) == 108 and set(stable).issubset(set(calib.imodulon.astype(str))))
    m = read_modulome(MODULOME, genes, stable)
    ref_q, ref_r = np.linalg.qr(m, mode="reduced")
    require("reference_108_full_rank", np.min(np.abs(np.diag(ref_r))) > 1e-8)

    print("\nNested Atlanta leave-one-SPECIMEN-out evaluation (no Cornforth selection)", flush=True)
    nested = []
    for outer_index, heldout in enumerate(sample_names):
        inside = [i for i in range(len(sample_names)) if i != outer_index]
        inner_names = [sample_names[i] for i in inside]
        inner_cv = cv_scores(x[:, inside], inner_names, ref_q)
        k, budget, _summary, _best, _se = choose_one_se(inner_cv)
        external_score = explained_for_one_fold(x, outer_index, ref_q, k, budget)
        dense_score = explained_for_one_fold(x, outer_index, ref_q, k, None)
        dense3_score = explained_for_one_fold(x, outer_index, ref_q, 3, None)
        nested.append({
            "heldout_Atlanta_specimen": heldout,
            "inner_selected_k": k,
            "inner_selected_gene_budget": budget,
            "outer_sparse_fraction_explained": external_score,
            "outer_dense_matched_rank_fraction": dense_score,
            "outer_dense_rank3_fraction": dense3_score,
        })
        print("Outer {:2d}/13: {:6s} K={} budget={} sparse={:.4f} dense_K={:.4f}".format(outer_index + 1, heldout, k, budget, external_score, dense_score), flush=True)

    print("\nFinal choice: all-Atlanta specimen-held-out model selection", flush=True)
    full_cv = cv_scores(x, sample_names, ref_q)
    final_k, final_budget, summary, best_mean, one_se = choose_one_se(full_cv)
    mean, seeds, axes = fit_final(x, ref_q, final_k, final_budget)
    require("sparse_gene_seed_count", all(np.count_nonzero(seeds[:, j]) == final_budget for j in range(final_k)))
    require("effective_basis_orthogonal_to_reference", np.max(np.abs(ref_q.T @ axes)) < 1e-8)
    require("effective_basis_orthonormal", np.allclose(axes.T @ axes, np.eye(final_k), atol=1e-8))
    require("fitted_arrays_finite", np.isfinite(seeds).all() and np.isfinite(axes).all() and np.isfinite(mean).all())

    OUT.mkdir(parents=True, exist_ok=False)
    nested_df = pd.DataFrame(nested)
    nested_df.to_csv(OUT / "NESTED_13_Atlanta_specimen_heldout_results.csv", index=False)
    full_cv.to_csv(OUT / "ATLANTA_ONLY_13fold_all_rank_budget_scores.csv", index=False)
    summary.to_csv(OUT / "ATLANTA_ONLY_rank_budget_selection_grid.csv", index=False)
    pd.DataFrame({"GeneID": genes, **{"sparse_seed_{}".format(j+1): seeds[:, j] for j in range(final_k)}}).to_csv(
        OUT / "FITTED_CFstate_v1_sparse_gene_readouts.csv", index=False)
    np.savez_compressed(OUT / "FITTED_CFstate_v1_arrays.npz", genes=np.array(genes), programmes=np.array(stable),
        Atlanta_gene_mean=mean, reference_Q=ref_q, sparse_gene_readouts=seeds,
        effective_residual_directions=axes, chosen_new_dimensions=np.array(final_k),
        chosen_gene_budget=np.array(final_budget))
    pd.DataFrame([{"input": key, "path": str(path), "sha256": checksum(path)} for key, path in INPUTS.items()]).to_csv(
        OUT / "input_sha256_provenance.csv", index=False)
    (OUT / "DESIGN_AND_LIMITATIONS.txt").write_text(
        "CF-State reference-anchored v1 exploratory fit; original 108 iModulon span fixed.\n"
        "K in 1..6, sparse gene budgets 50,100,200,500,1000,2000.\n"
        "Rank and budget chosen by one-SE heuristic on Atlanta specimen-held-out reconstruction.\n"
        "Nested specimen-held-out scores assess model-selection optimism but NOT independent-patient generalization.\n"
        "Some Atlanta samples may share donors; no verified patient IDs.\n"
        "Previously explored Cornforth cohort deliberately not used for selection or evaluation in this audit.\n"
        "Saved sparse gene readouts define EFFECTIVE DENSE residual-orthogonal reconstruction directions.\n"
        "Test coefficients are inferred from the full test expression vector: this is projection/reconstruction, not unseen-gene prediction.\n"
        "No claim of CF-specific biology, independent validation, pathway coherence, or patient utility.\n"
        "Reference-mapping and clinical site/geography confounding remain unresolved.\n"
    )
    status = "CFSTATE_PHASE6_06I_DATA_DRIVEN_SPARSE_V1_FITTED_EXPLORATORY"
    (OUT / "AUDIT_06I_STATUS.txt").write_text(status + "\n")
    print("\n========== NESTED SPECIMEN-HELD-OUT SUMMARY ==========")
    print(nested_df[["outer_sparse_fraction_explained", "outer_dense_matched_rank_fraction", "outer_dense_rank3_fraction"]].agg(["min", "median", "max"]).round(5).to_string())
    print("\nInner-model selection frequency:")
    print(nested_df.groupby(["inner_selected_k", "inner_selected_gene_budget"]).size().to_string())
    print("\n========== FINAL MODEL ==========")
    print("Rank K:", final_k)
    print("Genes per sparse readout:", final_budget)
    print("Best full CV mean:", round(best_mean, 5), "one-SE margin:", round(one_se, 5))
    print("Chosen model mean CV:", round(float(summary[(summary.k == final_k) & (summary.gene_budget == final_budget)].mean_fraction.iloc[0]), 5))
    print("Model arrays and provenance saved.")
    print("STATUS:", status)


if __name__ == "__main__":
    main()
