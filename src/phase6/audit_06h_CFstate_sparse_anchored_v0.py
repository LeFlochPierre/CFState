#!/usr/bin/env python3
"""CF-State Phase 6 Audit 06h: fit first sparse CF-airway-adapted latent model.

Train only on 13 Atlanta specimens. Existing 108 aeruPRECISE axes are frozen.
Select gene support size using Atlanta leave-one-specimen-out scores only.
Fit 3 sparse residual readouts, residual-orthogonal to the reference subspace.
Compare to rank-matched dense residual PCA; then evaluate fixed model once on
previously explored Cornforth 2018 (CF/non-CF within-group variation).

Exploratory: not donor-independent validation, not CF specificity, no new
biologically validated iModulons or new-patient clinical utility claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / 'results/phase4'
P5 = ROOT / 'results/phase5'
P6 = ROOT / 'results/phase6'
OUT = P6 / 'audit_06h_CFstate_sparse_anchored_v0'

PATHS = {
    'atlanta': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_Atlanta_softcore_4923_log2TPM.csv',
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    '06b_status': P6 / 'audit_06b_Cornforth2018_external_residual_transfer/AUDIT_06B_STATUS.txt',
    '06b_summary': P6 / 'audit_06b_Cornforth2018_external_residual_transfer/FROZEN_24_external_Cornforth_residual_transfer_checks.csv',
    '05h_status': P5 / 'audit_05h_Cornforth2018_full_depth_baseline/AUDIT_05H_STATUS.txt',
    'counts': P5 / 'audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv',
    'manifest': P5 / 'audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv',
    'saf': ROOT / 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf',
    'script': Path(__file__).resolve(),
}

K = 3
GENE_BUDGETS = (200, 500, 1000, 2000)
SELECTION_FRACTION_OF_BEST_CV = 0.95


def require(label, condition):
    if not bool(condition):
        raise RuntimeError('FAILED: ' + label)
    print(label + ' PASS', flush=True)


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_modulome(path, genes, programmes):
    obj = json.loads(path.read_text())['M']
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and all(k in obj for k in ('index', 'columns', 'data')):
        m = pd.DataFrame(obj['data'], index=obj['index'], columns=obj['columns'])
    else:
        m = pd.DataFrame(obj)
    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)
    if set(genes).issubset(m.columns) and set(programmes).issubset(m.index):
        m = m.T
    require('reference_gene_and_programme_ids', set(genes).issubset(m.index) and set(programmes).issubset(m.columns))
    require('reference_identifiers_unique', m.index.is_unique and m.columns.is_unique)
    m = m.loc[genes, programmes].apply(pd.to_numeric, errors='raise')
    require('reference_values_finite', np.isfinite(m.to_numpy(dtype=float)).all())
    return m.to_numpy(dtype=float, copy=True)


def residualise(x, reference_q):
    return x - reference_q @ (reference_q.T @ x)


def fit_sparse_axes(residual_training, reference_q, budget, raw_left_vectors=None):
    """Sparse seeds define readouts of REF-residualised expression.

    The effective reconstruction directions after reference orthogonalisation
    are dense; they must NOT be misdescribed as sparse gene-level effects.
    """
    if raw_left_vectors is None:
        u, singular_values, _ = np.linalg.svd(residual_training, full_matrices=False)
        if singular_values[K - 1] <= 1e-8:
            raise RuntimeError('Three nonzero training components required')
        raw = u[:, :K]
    else:
        raw = raw_left_vectors
        if raw.shape != (residual_training.shape[0], K):
            raise RuntimeError('Unexpected PCA basis dimensions')
    if budget is None:
        seeds = raw.copy()
    else:
        seeds = np.zeros_like(raw)
        for j in range(K):
            selected = np.argpartition(np.abs(raw[:, j]), -budget)[-budget:]
            seeds[selected, j] = raw[selected, j]
        seeds /= np.linalg.norm(seeds, axis=0, keepdims=True)
    effective = residualise(seeds, reference_q)
    q, r = np.linalg.qr(effective, mode='reduced')
    if np.min(np.abs(np.diag(r))) < 1e-8:
        raise RuntimeError('Sparse extension is rank deficient')
    return seeds, q


def fraction_explained(effective_axes, residual_target):
    denominator = float(np.sum(residual_target ** 2))
    if denominator <= 1e-9:
        raise RuntimeError('Empty evaluation residual')
    numerator = float(np.sum((effective_axes.T @ residual_target) ** 2))
    frac = numerator / denominator
    if not -1e-8 <= frac <= 1 + 1e-8:
        raise RuntimeError('Invalid energy fraction')
    return float(np.clip(frac, 0, 1))


def lo_sample_out_cv(X, ref_q, sample_names):
    rows = []
    for idx, heldout in enumerate(sample_names):
        train = np.delete(X, idx, axis=1)
        mean = train.mean(axis=1, keepdims=True)
        train_r = residualise(train - mean, ref_q)
        test_r = residualise(X[:, idx:idx + 1] - mean, ref_q)
        u, sing, _ = np.linalg.svd(train_r, full_matrices=False)
        if sing[K - 1] <= 1e-8:
            raise RuntimeError('Insufficient training residual rank')
        raw = u[:, :K]
        _, dense = fit_sparse_axes(train_r, ref_q, None, raw)
        rows.append({'heldout_specimen': heldout, 'model': 'DENSE_PC3',
                     'genes_per_sparse_seed': X.shape[0],
                     'fraction_explained': fraction_explained(dense, test_r)})
        for budget in GENE_BUDGETS:
            _, sparse = fit_sparse_axes(train_r, ref_q, budget, raw)
            rows.append({'heldout_specimen': heldout, 'model': 'SPARSE_REF3',
                         'genes_per_sparse_seed': budget,
                         'fraction_explained': fraction_explained(sparse, test_r)})
    return pd.DataFrame(rows)


def read_external(genes):
    counts = pd.read_csv(PATHS['counts'], index_col=0)
    counts.index = counts.index.astype(str)
    manifest = pd.read_csv(PATHS['manifest'], dtype=str).fillna('')
    require('external_13_samples', counts.shape == (5567, 13) and len(manifest) == 13 and set(counts.columns) == set(manifest.published_sample))
    require('external_12_unique_donors', manifest.published_donor_group.nunique() == 12)
    require('external_CF7_wound4_burn2', (manifest.group == 'CF_sputum').sum() == 7 and (manifest.group == 'nonCF_chronic_wound').sum() == 4 and (manifest.group == 'nonCF_burn').sum() == 2)
    saf = pd.read_csv(PATHS['saf'], sep='\t').set_index('GeneID')
    require('external_saf_genes', saf.index.is_unique and set(saf.index) == set(counts.index))
    lengths = (saf.loc[counts.index, 'End'] - saf.loc[counts.index, 'Start'] + 1).astype(float)
    require('external_gene_lengths_positive', (lengths > 0).all())
    require('external_mask_covered', set(genes).issubset(counts.index))
    groups = {}
    for mode in ('FROZEN_5567_TPM', 'SOFTCORE_4923_TPM'):
        cs = counts if mode == 'FROZEN_5567_TPM' else counts.loc[genes]
        rates = cs.div(lengths.loc[cs.index], axis=0)
        denom = rates.sum(axis=0)
        require('external_TPM_denominator_' + mode, (denom > 0).all())
        expr = np.log2(rates.div(denom, axis=1) * 1e6 + 1).loc[genes]
        donors = {}
        donor_groups = {}
        for donor, frame in manifest.groupby('published_donor_group', sort=False):
            require('donor_group_consistent_' + donor, frame.group.nunique() == 1)
            donors[donor] = expr[frame.published_sample.tolist()].mean(axis=1)
            donor_groups[donor] = 'CF' if (frame.group == 'CF_sputum').all() else 'nonCF'
        donor_matrix = pd.DataFrame(donors)
        groups[mode] = {}
        for group in ('CF', 'nonCF'):
            selected = [donor for donor in donor_matrix.columns if donor_groups[donor] == group]
            expected = 7 if group == 'CF' else 5
            require('external_donor_count_' + mode + '_' + group, len(selected) == expected)
            x = donor_matrix[selected].to_numpy(dtype=float, copy=True)
            groups[mode][group] = x - x.mean(axis=1, keepdims=True)
    return groups


def main():
    print('=' * 76)
    print('CF-State Phase 6 — Audit 06h: FIT CF-State sparse anchored v0')
    print('=' * 76)
    require('new_output_directory', not OUT.exists())
    for label, path in PATHS.items():
        require('input_' + label, path.is_file())
    require('06b_completed', PATHS['06b_status'].read_text().strip() == 'CFSTATE_PHASE6_06B_EXPLORATORY_EXTERNAL_RESIDUAL_TRANSFER_COMPLETE')
    require('05h_completed', PATHS['05h_status'].read_text().strip() == 'CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN')
    genes = pd.read_csv(PATHS['mask'])['GeneID'].astype(str).tolist()
    require('frozen_gene_mask', len(genes) == 4923 and len(set(genes)) == 4923)
    df = pd.read_csv(PATHS['atlanta']).set_index('GeneID')
    require('Atlanta_matrix', df.shape == (4923, 13) and set(df.index) == set(genes))
    X = df.loc[genes].to_numpy(dtype=float, copy=True)
    require('Atlanta_values_finite', np.isfinite(X).all())
    cal = pd.read_csv(PATHS['calibration'])
    names = cal.imodulon.astype(str).tolist()
    stable = pd.read_csv(PATHS['stable']).imodulon.astype(str).tolist()
    require('frozen_108_registry', len(names) == 116 and len(stable) == 108 and set(stable).issubset(names) and len(set(stable)) == 108)
    m = load_modulome(PATHS['modulome'], genes, stable)
    ref_q, ref_r = np.linalg.qr(m, mode='reduced')
    require('reference_108_full_rank', np.min(np.abs(np.diag(ref_r))) > 1e-7)
    print('\nChoosing sparsity using ATLANTA ONLY (13 leave-one-specimen-out folds)...', flush=True)
    cv = lo_sample_out_cv(X, ref_q, df.columns.tolist())
    summary = cv.groupby(['model', 'genes_per_sparse_seed'], as_index=False).agg(
        median_heldout_fraction=('fraction_explained', 'median'),
        min_heldout_fraction=('fraction_explained', 'min'),
        mean_heldout_fraction=('fraction_explained', 'mean'))
    candidates = summary.loc[summary.model == 'SPARSE_REF3'].copy()
    best_score = float(candidates.median_heldout_fraction.max())
    require('positive_CV_score', best_score > 0)
    eligible = candidates.loc[candidates.median_heldout_fraction >= SELECTION_FRACTION_OF_BEST_CV * best_score]
    chosen_budget = int(eligible.genes_per_sparse_seed.min())
    print('\n========== ATLANTA-ONLY MODEL SELECTION ==========', flush=True)
    print(summary.round(5).to_string(index=False), flush=True)
    print('Chosen sparse genes per seed:', chosen_budget, flush=True)

    mean_train = X.mean(axis=1, keepdims=True)
    r_train = residualise(X - mean_train, ref_q)
    u_train, singular_values, _ = np.linalg.svd(r_train, full_matrices=False)
    require('final_training_rank', singular_values[K - 1] > 1e-8)
    raw_train = u_train[:, :K]
    sparse_seeds, sparse_eff = fit_sparse_axes(r_train, ref_q, chosen_budget, raw_train)
    _, dense_eff = fit_sparse_axes(r_train, ref_q, None, raw_train)
    require('reference_orthogonality', np.max(np.abs(ref_q.T @ sparse_eff)) < 1e-8)
    require('model_orthonormality', np.allclose(sparse_eff.T @ sparse_eff, np.eye(3), atol=1e-8))
    require('seed_sparsity_exact', all(np.count_nonzero(sparse_seeds[:, i]) == chosen_budget for i in range(K)))

    external = read_external(genes)
    frozen_baselines = pd.read_csv(PATHS['06b_summary'])
    external_rows = []
    for mode, group_mats in external.items():
        for group, centered in group_mats.items():
            r = residualise(centered, ref_q)
            dense_frac = fraction_explained(dense_eff, r)
            sparse_frac = fraction_explained(sparse_eff, r)
            row = frozen_baselines[
                (frozen_baselines.reference_baseline == 'STABLE_108') &
                (frozen_baselines.normalisation == mode) &
                (frozen_baselines.external_group == group) &
                (frozen_baselines.new_dimensions == 3)
            ]
            require('prior_dense_transfer_row_' + mode + '_' + group, len(row) == 1)
            prior = float(row.external_residual_fraction_explained.iloc[0])
            require('reproduce_frozen_dense_external_' + mode + '_' + group,
                    abs(dense_frac - prior) < 2e-5)
            external_rows.append({
                'normalisation': mode, 'external_group': group,
                'n_external_donors': centered.shape[1],
                'chosen_genes_per_sparse_seed': chosen_budget,
                'frozen_reference_only_fraction_of_residual_captured': 0.0,
                'dense_PC3_fraction_of_remaining_residual': dense_frac,
                'sparse_CFstate_v0_fraction_of_remaining_residual': sparse_frac,
                'sparse_to_dense_capture_ratio': sparse_frac / dense_frac,
            })

    OUT.mkdir(parents=True, exist_ok=False)
    cv.to_csv(OUT / 'EXPLORATORY_Atlanta_LOSO_model_comparisons.csv', index=False)
    summary.to_csv(OUT / 'EXPLORATORY_Atlanta_model_selection_summary.csv', index=False)
    pd.DataFrame(external_rows).to_csv(OUT / 'EXPLORATORY_fixed_CFstate_v0_external_transfer.csv', index=False)
    loading = pd.DataFrame({'GeneID': genes})
    for i in range(K):
        loading['CFstate_v0_sparse_seed_' + str(i + 1)] = sparse_seeds[:, i]
    loading.to_csv(OUT / 'FROZEN_CFstate_v0_3_sparse_gene_readouts.csv', index=False)
    np.savez_compressed(
        OUT / 'FROZEN_CFstate_v0_model_arrays.npz',
        genes=np.array(genes), programmes=np.array(stable),
        Atlanta_gene_mean=mean_train[:, 0],
        aeruPRECISE_reference_Q=ref_q,
        sparse_gene_readouts=sparse_seeds,
        effective_residual_directions=sparse_eff,
        chosen_gene_budget=np.array(chosen_budget),
    )
    pd.DataFrame([{'input': name, 'path': str(path), 'sha256': checksum(path)}
                  for name, path in PATHS.items()]).to_csv(OUT / 'input_sha256_provenance.csv', index=False)
    (OUT / 'DESIGN_AND_LIMITATIONS.txt').write_text(
        'First fitted CF-State v0: 108 fixed aeruPRECISE basis dimensions plus THREE Atlanta-trained sparse residual readouts.\n'
        'The residual-orthogonal reconstruction directions are dense; the saved sparse seeds are gene-level readouts.\n'
        'Sparsity selected using Atlanta leave-one-SPECIMEN-out only; Atlanta independent patient IDs NOT VERIFIED.\n'
        'Reference 108 programme activities are not retrained; orthonormal basis of their original column span is saved.\n'
        'This exploratory model is fitted on within-Atlanta variation; no proof of CF specificity.\n'
        'Previously examined Cornforth 2018 was reused for TRANSFER CHARACTERISATION, NOT fresh independent validation.\n'
        'External mean centering within each group is transductive and not a deployable prediction protocol.\n'
        'Low coverage, PAO1 strain bias, clinical site/geographic confounding remain unresolved.\n'
        'Sparse and dense models have the same 3 added dimensions; sparse may capture less raw residual energy.\n'
        'Future development: independent-donor CF cohorts, prospective group-agnostic centering, functional validation.\n'
    )
    status = 'CFSTATE_PHASE6_06H_SPARSE_ANCHORED_V0_FITTED_EXPLORATORY'
    (OUT / 'AUDIT_06H_STATUS.txt').write_text(status + '\n')
    print('\n========== FIXED MODEL EXTERNAL TRANSFER ==========', flush=True)
    print(pd.DataFrame(external_rows).round(5).to_string(index=False), flush=True)
    print('\nModel weights and inference arrays saved.', flush=True)
    print('STATUS:', status, flush=True)


if __name__ == '__main__':
    main()
