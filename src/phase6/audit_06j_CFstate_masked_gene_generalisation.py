#!/usr/bin/env python3
"""CF-State Phase 6 — Audit 06j: genuinely held-out GENE reconstruction.

Frozen Atlanta-trained v1 is assessed on already-explored Cornforth 2018 donor
profiles. 20% of genes are hidden before estimating each donor's activities;
only the other 80% are used to estimate coefficients. Hidden-gene predictions
are compared for the reference-108, CF-State v1 (108+3), and dense 108+3.

EXPLORATORY only: Cornforth was used in earlier model-design discussions, and
CF status is confounded with infection site/geography. This is NOT a clinical
state forecast or a pristine independent validation set.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / 'results/phase6/audit_06j_CFstate_masked_gene_generalisation'
NPZ = ROOT / 'results/phase6/audit_06i_CFstate_data_driven_rank_sparse_v1/FITTED_CFstate_v1_arrays.npz'
STATUS06I = ROOT / 'results/phase6/audit_06i_CFstate_data_driven_rank_sparse_v1/AUDIT_06I_STATUS.txt'
ATL = ROOT / 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_Atlanta_softcore_4923_log2TPM.csv'
COUNTS = ROOT / 'results/phase5/audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv'
STATUS05H = ROOT / 'results/phase5/audit_05h_Cornforth2018_full_depth_baseline/AUDIT_05H_STATUS.txt'
SAF = ROOT / 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf'
SCRIPT = Path(__file__).resolve()
N_REPEATS = 20
HIDDEN_FRAC = 0.20
RIDGE = 0.001   # Fixed beforehand; same for every model, donor and repeat
SEED = 20261014


def check(statement: str, condition: bool) -> None:
    if not bool(condition):
        raise RuntimeError('FAILED ' + statement)
    print(statement + ' PASS', flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def construct_dense_atlanta_axes(atl: np.ndarray, mean: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    residual = atl - mean[:, None]
    residual -= q @ (q.T @ residual)
    gram = residual.T @ residual
    vals, vecs = np.linalg.eigh(gram)
    ix = np.argsort(vals)[::-1][:k]
    check('dense_atlanta_rank', len(ix) == k and np.min(vals[ix]) > 1e-9)
    axes = residual @ (vecs[:, ix] / np.sqrt(vals[ix]))
    check('dense_axes_orthogonal', np.allclose(axes.T @ axes, np.eye(k), atol=1e-7)
          and np.max(np.abs(q.T @ axes)) < 1e-7)
    return axes


def reconstruct_on_unseen_genes(
    expression: np.ndarray, mean: np.ndarray, basis: np.ndarray,
    observed: np.ndarray, hidden: np.ndarray,
) -> tuple[float, float]:
    """Fit activity from observed genes only; compare at hidden genes only."""
    bo = basis[observed]
    delta = expression[observed] - mean[observed]
    gram = bo.T @ bo
    b = bo.T @ delta
    a = np.linalg.solve(gram + RIDGE * np.eye(basis.shape[1]), b)
    prediction = mean[hidden] + basis[hidden] @ a
    residual = expression[hidden] - prediction
    return float(residual @ residual), float(np.mean(residual**2))


def main() -> None:
    print('=' * 80)
    print('CF-State — Audit 06j: masked-GENE external reconstruction challenge')
    print('=' * 80)
    check('new_output_directory', not OUT.exists())
    for key, path in {'06i_model': NPZ, '06i_status': STATUS06I, 'Atlanta': ATL,
                      '05h_counts': COUNTS, '05h_status': STATUS05H,
                      'SAF': SAF, 'script': SCRIPT}.items():
        check('input_' + key, path.is_file())
    check('06i_model_frozen', STATUS06I.read_text().strip()
          == 'CFSTATE_PHASE6_06I_DATA_DRIVEN_SPARSE_V1_FITTED_EXPLORATORY')
    check('05h_baseline_frozen', STATUS05H.read_text().strip()
          == 'CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN')

    with np.load(NPZ, allow_pickle=False) as fit:
        genes = fit['genes'].astype(str).tolist()
        mu = fit['Atlanta_gene_mean'].astype(float)
        q = fit['reference_Q'].astype(float)
        new_axes = fit['effective_residual_directions'].astype(float)
        rank = int(fit['chosen_new_dimensions'])
        budget = int(fit['chosen_gene_budget'])
    check('frozen_model_dimensions', len(genes) == 4923 and len(set(genes)) == 4923
          and mu.shape == (4923,) and q.shape == (4923, 108)
          and new_axes.shape == (4923, rank) and rank == 3 and budget == 2000)
    check('model_orthogonality', np.allclose(q.T @ q, np.eye(108), atol=1e-7)
          and np.allclose(new_axes.T @ new_axes, np.eye(rank), atol=1e-7)
          and np.max(np.abs(q.T @ new_axes)) < 1e-7)

    atl = pd.read_csv(ATL).set_index('GeneID')
    check('Atlanta_expression_order', len(atl.columns) == 13
          and set(genes) == set(atl.index))
    x_atl = atl.loc[genes].to_numpy(dtype=float)
    dense_axes = construct_dense_atlanta_axes(x_atl, mu, q, rank)
    bases = {
        'reference_108': q,
        'CFstate_v1_sparse_readouts_111': np.column_stack((q, new_axes)),
        'Atlanta_dense_PC3_111': np.column_stack((q, dense_axes)),
    }

    counts = pd.read_csv(COUNTS).set_index('Geneid')
    saf = pd.read_csv(SAF, sep='\t').set_index('GeneID')
    check('count_gene_universe', counts.shape == (5567, 13)
          and set(counts.index) == set(saf.index)
          and set(genes).issubset(counts.index))
    lengths = (saf.loc[counts.index, 'End'] - saf.loc[counts.index, 'Start'] + 1).to_numpy(dtype=float)
    check('positive_gene_lengths', np.all(lengths > 0))
    arr = counts.to_numpy(dtype=float)
    check('nonnegative_finite_counts', np.isfinite(arr).all() and np.min(arr) >= 0)
    rate = arr / lengths[:, None]
    denom = rate.sum(axis=0)
    check('positive_TPM_denominators', np.all(denom > 0))
    logtpm = np.log2(rate / denom[None, :] * 1e6 + 1.0)
    full = pd.DataFrame(logtpm, index=counts.index, columns=counts.columns).loc[genes]
    check('all_Cornforth_13_present', full.shape == (4923, 13))

    cf = sorted(name for name in full.columns if name.startswith('Human_Sputum_'))
    cw = sorted(name for name in full.columns if name.startswith('Lubbock_CW_'))
    burn = sorted(name for name in full.columns if name.startswith('Lubbock_Burn_'))
    check('donor_group_design', len(cf) == 7 and len(cw) == 4 and len(burn) == 2)
    samples = [(name, 'CF', full[name].to_numpy(dtype=float)) for name in cf]
    samples += [(name, 'nonCF', full[name].to_numpy(dtype=float)) for name in cw]
    samples.append(('Lubbock_Burn_shared_donor', 'nonCF', full[burn].mean(axis=1).to_numpy(dtype=float)))
    check('7_CF_and_5_nonCF_donors', len(samples) == 12)

    rng = np.random.default_rng(SEED)
    rows = []
    for rep in range(N_REPEATS):
        hidden = np.sort(rng.choice(len(genes), size=round(HIDDEN_FRAC * len(genes)), replace=False))
        observed = np.setdiff1d(np.arange(len(genes)), hidden, assume_unique=True)
        for donor, group, expression in samples:
            centered_err = expression[hidden] - mu[hidden]
            mean_sse = float(centered_err @ centered_err)
            for model_name, basis in bases.items():
                sse, mse = reconstruct_on_unseen_genes(expression, mu, basis, observed, hidden)
                rows.append({'repeat': rep, 'donor': donor, 'group': group,
                             'model': model_name, 'hidden_genes': len(hidden),
                             'sse_hidden': sse, 'mse_hidden': mse,
                             'mean_baseline_sse_hidden': mean_sse})
    detail = pd.DataFrame(rows)
    check('complete_masked_gene_evaluation', len(detail) == N_REPEATS * 12 * 3
          and detail.sse_hidden.ge(0).all() and np.isfinite(detail.sse_hidden).all())
    donor_sum = detail.groupby(['group', 'donor', 'model'], as_index=False).agg(
        sse_hidden=('sse_hidden', 'sum'), mean_baseline_sse_hidden=('mean_baseline_sse_hidden', 'sum'))
    donor_sum['relative_to_mean_error'] = donor_sum.sse_hidden / donor_sum.mean_baseline_sse_hidden
    wide = donor_sum.pivot(index=['group', 'donor'], columns='model', values='sse_hidden')
    mean_baseline = donor_sum[donor_sum['model'] == 'reference_108'].set_index(
        ['group', 'donor'])['mean_baseline_sse_hidden']
    wide['reference_vs_mean_RMSE_change_pct'] = 100 * (1 - np.sqrt(
        wide['reference_108'] / mean_baseline))
    wide['sparse_vs_mean_RMSE_change_pct'] = 100 * (1 - np.sqrt(
        wide['CFstate_v1_sparse_readouts_111'] / mean_baseline))
    wide['sparse_vs_reference_RMSE_change_pct'] = 100 * (1 - np.sqrt(
        wide['CFstate_v1_sparse_readouts_111'] / wide['reference_108']))
    wide['dense_vs_reference_RMSE_change_pct'] = 100 * (1 - np.sqrt(
        wide['Atlanta_dense_PC3_111'] / wide['reference_108']))
    wide = wide.reset_index()
    summary = wide.groupby('group', as_index=False).agg(
        donors=('donor', 'size'),
        reference_median_RMSE_vs_mean_pct=('reference_vs_mean_RMSE_change_pct', 'median'),
        sparse_median_RMSE_vs_mean_pct=('sparse_vs_mean_RMSE_change_pct', 'median'),
        sparse_median_RMSE_improvement_pct=('sparse_vs_reference_RMSE_change_pct', 'median'),
        sparse_min_RMSE_improvement_pct=('sparse_vs_reference_RMSE_change_pct', 'min'),
        sparse_max_RMSE_improvement_pct=('sparse_vs_reference_RMSE_change_pct', 'max'),
        dense_median_RMSE_improvement_pct=('dense_vs_reference_RMSE_change_pct', 'median'),
        sparse_donors_improved=('sparse_vs_reference_RMSE_change_pct', lambda s: int((s > 0).sum())),
    )
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / 'FROZEN_20_mask_repeats_by_donor_and_model.csv', index=False)
    wide.to_csv(OUT / 'DESCRIPTIVE_12_donor_masked_gene_RMSE_comparison.csv', index=False)
    summary.to_csv(OUT / 'DESCRIPTIVE_CF_nonCF_masked_gene_summary.csv', index=False)
    inputs = [NPZ, STATUS06I, ATL, COUNTS, STATUS05H, SAF, SCRIPT]
    pd.DataFrame([{'path': str(p), 'sha256': sha256(p)} for p in inputs]).to_csv(
        OUT / 'input_sha256_provenance.csv', index=False)
    (OUT / 'LIMITATIONS.txt').write_text(
        'Fixed frozen Atlanta-trained v1; rank=3, 2000 gene readouts.\n'
        'Evaluate external Cornforth 2018, already examined in earlier audits; NOT untouched external validation.\n'
        'Twenty random held-out gene subsets of 20%; same masks across donors.\n'
        'Activity coefficients estimated ONLY from measured 80% genes; remaining 20% predicted.\n'
        'Same prespecified ridge regularisation for reference, v1 and dense comparator.\n'
        'Atlanta mean is shared across models, no Cornforth-specific centring/refitting.\n'
        'Both burn specimens combined at one donor; donor-level summaries.\n'
        'PAO1-only mapping, possible cohort overlap, collection site/geographic confounding remain.\n'
        'Repeated gene masks are technical replicates, not new donors.\n'
        'This tests held-out gene reconstruction, not physiological-state prediction or CF specificity.\n'
    )
    status = 'CFSTATE_PHASE6_06J_EXPLORATORY_MASKED_GENE_RECONSTRUCTION_COMPLETE'
    (OUT / 'AUDIT_06J_STATUS.txt').write_text(status + '\n')
    print('\n========== MASKED-GENE PREDICTION (20% HELD OUT) ==========')
    print(summary.round(4).to_string(index=False))
    print('\n========== INDIVIDUAL DONORS: CF-STATE V1 VS REFERENCE ==========')
    print(wide[['group', 'donor', 'reference_vs_mean_RMSE_change_pct',
                'sparse_vs_reference_RMSE_change_pct',
                'dense_vs_reference_RMSE_change_pct']].round(4).to_string(index=False))
    print('\nPOSITIVE RMSE IMPROVEMENT = LOWER PREDICTION ERROR.')
    print('NEGATIVE RMSE IMPROVEMENT = MODEL MADE PREDICTION WORSE.')
    print('No fitted parameters or hyperparameters changed in this audit.')
    print('STATUS:', status)


if __name__ == '__main__':
    main()
