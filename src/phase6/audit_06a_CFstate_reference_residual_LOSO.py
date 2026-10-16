#!/usr/bin/env python3
"""CF-State Phase 6 audit 06a: exploratory reference-anchored residual benchmark.

This does NOT discover a CF-specific biological programme. Thirteen Atlanta
specimens have unresolved donor identities, so LOSO is not patient validation.
"""
from pathlib import Path
import hashlib
import json
import traceback

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / 'results/phase4'
OUT = ROOT / 'results/phase6/audit_06a_Atlanta_reference_residual_LOSO'
INPUTS = {
    'atlanta': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_Atlanta_softcore_4923_log2TPM.csv',
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'script': Path(__file__).resolve(),
}
SEED = 20261014
N_RANDOM_BASES = 32
K_VALUES = (1, 2, 3)


def check(name, condition):
    if not condition:
        raise RuntimeError('FAILED: ' + name)
    print(name, 'PASS', flush=True)


def read_modulome(path, genes, names):
    obj = json.loads(path.read_text())
    check('modulome_M_key', isinstance(obj, dict) and 'M' in obj)
    obj = obj['M']
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and all(key in obj for key in ('index', 'columns', 'data')):
        matrix = pd.DataFrame(obj['data'], index=obj['index'], columns=obj['columns'])
    else:
        matrix = pd.DataFrame(obj)
    matrix.index = matrix.index.map(str)
    matrix.columns = matrix.columns.map(str)
    if set(genes).issubset(matrix.columns) and set(names).issubset(matrix.index):
        matrix = matrix.T
    check('modulome_ids', set(genes).issubset(matrix.index) and set(names).issubset(matrix.columns))
    check('modulome_unique_ids', matrix.index.is_unique and matrix.columns.is_unique)
    matrix = matrix.reindex(index=genes, columns=names).apply(pd.to_numeric, errors='raise')
    check('modulome_finite', np.isfinite(matrix.to_numpy(dtype=float)).all())
    return matrix


def orthogonal_residual(x, basis, pinv):
    return x - basis @ (pinv @ x)


def orthonormal_columns(a):
    q, _ = np.linalg.qr(a, mode='reduced')
    return q


def experiment(expression, basis, stable_or_full, rng):
    n_genes, n_samples = expression.shape
    pinv = np.linalg.pinv(basis)
    check('reference_rank_' + stable_or_full, np.linalg.matrix_rank(basis) == basis.shape[1])
    results = []
    for heldout in range(n_samples):
        train_idx = [i for i in range(n_samples) if i != heldout]
        training = expression[:, train_idx]
        train_mean = training.mean(axis=1, keepdims=True)
        centered_train = training - train_mean
        centered_test = expression[:, heldout:heldout + 1] - train_mean
        r_train = orthogonal_residual(centered_train, basis, pinv)
        r_test = orthogonal_residual(centered_test, basis, pinv)
        base_error = float(np.sum(r_test ** 2))
        original_total = float(np.sum(centered_test ** 2))
        check('positive_residual_' + stable_or_full + '_' + str(heldout + 1), base_error > 1e-10 and original_total > 1e-10)

        # PCA is trained on the other 12 specimens. Only the test projection
        # coefficients (not the gene loadings) are inferred from the test specimen.
        u, s, _ = np.linalg.svd(r_train, full_matrices=False)
        check('svd_finite_' + stable_or_full + '_' + str(heldout + 1), np.isfinite(s).all())
        test = r_test[:, 0]
        observed = {}
        for k in K_VALUES:
            explained = np.sum((u[:, :k].T @ test) ** 2) / base_error
            observed[k] = float(np.clip(explained, 0, 1))

        null = {k: [] for k in K_VALUES}
        # Random subspaces are sampled from the SAME training residual span:
        # a stronger comparator than random directions in 4,923-gene space.
        for _ in range(N_RANDOM_BASES):
            coeff = rng.standard_normal((r_train.shape[1], max(K_VALUES)))
            random_basis = orthonormal_columns(r_train @ coeff)
            for k in K_VALUES:
                explained = np.sum((random_basis[:, :k].T @ test) ** 2) / base_error
                null[k].append(float(np.clip(explained, 0, 1)))

        for k in K_VALUES:
            results.append({
                'baseline': stable_or_full,
                'heldout_sample': heldout,
                'k_new_dimensions': k,
                'baseline_residual_fraction_of_centered_gene_energy': base_error / original_total,
                'heldout_residual_fraction_explained_by_learned_k': observed[k],
                'random_subspace_median_fraction': float(np.median(null[k])),
                'random_subspace_95pct_fraction': float(np.quantile(null[k], .95)),
                'learned_exceeds_random_95pct': bool(observed[k] > np.quantile(null[k], .95)),
            })
    return pd.DataFrame(results)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def main():
    print('=' * 75)
    print('CF-State Phase 6 — Audit 06a: Atlanta residual-model feasibility')
    print('=' * 75)
    check('new_output_directory', not OUT.exists())
    for name, path in INPUTS.items():
        check('input_' + name, path.is_file())

    mask = pd.read_csv(INPUTS['mask'])['GeneID'].astype(str).tolist()
    check('mask_4923_unique', len(mask) == 4923 and len(set(mask)) == 4923)
    exp = pd.read_csv(INPUTS['atlanta']).set_index('GeneID')
    exp.index = exp.index.map(str)
    exp.columns = exp.columns.map(str)
    check('Atlanta_4923_by_13', exp.shape == (4923, 13) and exp.index.is_unique and exp.columns.is_unique and set(exp.index) == set(mask))
    exp = exp.loc[mask].apply(pd.to_numeric, errors='raise')
    X = exp.to_numpy(dtype=float, copy=True)
    check('Atlanta_finite_log2TPM', np.isfinite(X).all() and np.min(X) >= 0)
    cal = pd.read_csv(INPUTS['calibration'])
    stable = pd.read_csv(INPUTS['stable'])['imodulon'].astype(str).tolist()
    names = cal['imodulon'].astype(str).tolist()
    check('116_and_108_identities', len(names) == 116 and len(set(names)) == 116 and len(stable) == 108 and len(set(stable)) == 108 and set(stable).issubset(names))
    M = read_modulome(INPUTS['modulome'], mask, names)

    rng = np.random.default_rng(SEED)
    result_parts = []
    for label, names_here in [('STABLE_108', stable), ('FULL_116', names)]:
        print('\nRunning sample-held-out benchmark:', label, flush=True)
        component = experiment(X, M[names_here].to_numpy(dtype=float, copy=True), label, rng)
        component['heldout_sample'] = component['heldout_sample'].map(dict(enumerate(exp.columns)))
        result_parts.append(component)
    per_holdout = pd.concat(result_parts, ignore_index=True)
    check('78_actual_holdout_results', len(per_holdout) == 2 * 13 * len(K_VALUES))
    check('all_fraction_metrics_finite', np.isfinite(per_holdout.select_dtypes(include='number').to_numpy()).all())

    summary = per_holdout.groupby(['baseline', 'k_new_dimensions'], sort=True).agg(
        median_baseline_residual_energy_fraction=('baseline_residual_fraction_of_centered_gene_energy', 'median'),
        median_new_fraction_of_residual=('heldout_residual_fraction_explained_by_learned_k', 'median'),
        min_new_fraction_of_residual=('heldout_residual_fraction_explained_by_learned_k', 'min'),
        median_random_subspace_fraction=('random_subspace_median_fraction', 'median'),
        holdouts_beating_random95=('learned_exceeds_random_95pct', 'sum'),
    ).reset_index()

    OUT.mkdir(parents=True, exist_ok=False)
    per_holdout.to_csv(OUT / 'FROZEN_exploratory_LOSO_78_reconstruction_checks.csv', index=False)
    summary.to_csv(OUT / 'FROZEN_exploratory_residual_model_summary.csv', index=False)

    # Reference-projected residual across all Atlanta samples, with the cohort
    # mean removed: programmes reflect *within-Atlanta variation*, not a CF effect.
    centered = X - X.mean(axis=1, keepdims=True)
    stable_basis = M[stable].to_numpy(dtype=float, copy=True)
    residual = orthogonal_residual(centered, stable_basis, np.linalg.pinv(stable_basis))
    u, s, _ = np.linalg.svd(residual, full_matrices=False)
    pd.DataFrame({'GeneID': mask, 'exploratory_PC1_weight': u[:, 0]}).to_csv(
        OUT / 'EXPLORATORY_NOT_VALIDATED_Atlanta_residual_PC1_gene_weights.csv', index=False
    )
    pd.DataFrame({'component': np.arange(1, len(s) + 1),
                  'singular_value': s,
                  'fraction_of_training_residual_energy': s ** 2 / np.sum(s ** 2)}).to_csv(
        OUT / 'EXPLORATORY_Atlanta_residual_spectrum.csv', index=False
    )
    pd.DataFrame([{'path': str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                   'sha256': digest(path)} for path in INPUTS.values()]).to_csv(
        OUT / 'FROZEN_input_sha256.csv', index=False
    )
    (OUT / 'DESIGN_AND_LIMITATIONS.txt').write_text(
        'CF-State Phase 6 Audit 06a: method-feasibility only.\n'
        'Fixed reference basis: 108 validated or full 116 aeruPRECISE iModulons.\n'
        'Expression: 13 Atlanta log2TPM clinical specimens; centered ONLY within Atlanta.\n'
        'No CFBE contrast needed: shared reference cancels when centering within Atlanta.\n'
        'For each left-out specimen: fit residual PCA on other 12, project the left-out expression.\n'
        'Evaluate k=1,2,3 added dimensions vs same-span random rank-k subspaces (32 draws).\n'
        'This is out-of-sample gene-basis evaluation, NOT unseen-gene prediction.\n'
        'Donor identities for Atlanta are not verified. Not patient-independent CV.\n'
        'Components may reflect technical effects, within-airway heterogeneity or strain effects.\n'
        'No novel programme is biologically validated or asserted CF-specific.\n'
        'Do not use this same Atlanta set for claimed external validation of learned loadings.\n'
    )
    status = 'CFSTATE_PHASE6_06A_EXPLORATORY_REFERENCE_RESIDUAL_BENCHMARK_COMPLETE'
    (OUT / 'AUDIT_06A_STATUS.txt').write_text(status + '\n')
    print('\n========== LO-SAMPLE-OUT MODEL BENCHMARK ==========')
    print(summary.round(4).to_string(index=False))
    print('\nSTATUS:', status, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
