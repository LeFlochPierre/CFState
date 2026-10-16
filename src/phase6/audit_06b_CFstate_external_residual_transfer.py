#!/usr/bin/env python3
"""CF-State Phase 6 Audit 06b: fixed Atlanta residual PCs on Cornforth 2018.

Exploratory cross-cohort subspace transfer, not a new CF programme validation.
All gene loadings are learned from Atlanta only. External profiles are centered
within their own clinical group to assess within-group variation; this does not
predict an individual patient's expression from a standalone clinical profile.
"""
from pathlib import Path
import hashlib
import json
import traceback

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / 'results/phase4'
P5 = ROOT / 'results/phase5'
OUT = ROOT / 'results/phase6/audit_06b_Cornforth2018_external_residual_transfer'
INPUTS = {
    '06a_status': ROOT / 'results/phase6/audit_06a_Atlanta_reference_residual_LOSO/AUDIT_06A_STATUS.txt',
    '06a_pc1': ROOT / 'results/phase6/audit_06a_Atlanta_reference_residual_LOSO/EXPLORATORY_NOT_VALIDATED_Atlanta_residual_PC1_gene_weights.csv',
    'atlanta': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_Atlanta_softcore_4923_log2TPM.csv',
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    '05h_status': P5 / 'audit_05h_Cornforth2018_full_depth_baseline/AUDIT_05H_STATUS.txt',
    'counts': P5 / 'audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv',
    'manifest': P5 / 'audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv',
    'saf': ROOT / 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf',
    'script': Path(__file__).resolve(),
}
SEED = 20261014
N_RANDOM_BASES = 256
K_VALUES = (1, 2, 3)


def require(name, condition):
    if not bool(condition):
        raise RuntimeError('FAILED: ' + name)
    print(name, 'PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load_m(path, genes, programmes):
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
    require('modulome_gene_programme_match', set(genes).issubset(m.index) and set(programmes).issubset(m.columns))
    require('modulome_unique_ids', m.index.is_unique and m.columns.is_unique)
    m = m.loc[genes, programmes].apply(pd.to_numeric, errors='raise')
    require('modulome_finite', np.isfinite(m.to_numpy(dtype=float)).all())
    return m


def logtpm(counts, lengths):
    rates = counts.div(lengths, axis=0)
    denominators = rates.sum(axis=0)
    require('TPM_denominators_positive', (denominators > 0).all() and np.isfinite(denominators).all())
    tpm = rates.div(denominators, axis=1) * 1e6
    require('TPM_sum_1e6', np.allclose(tpm.sum(axis=0), 1e6, rtol=0, atol=.01))
    return np.log2(tpm + 1)


def orth_residual(x, basis, pinv):
    return x - basis @ (pinv @ x)


def compute_comparisons(U, r_train, group_matrices, reference_baseline, mode, rng):
    # A single random subspace per draw is evaluated on BOTH external groups,
    # and every random basis is derived from the same frozen Atlanta residual span.
    out = []
    obs_by_group = {}
    total_by_group = {}
    for group, r in group_matrices.items():
        total = float(np.sum(r * r))
        require('external_residual_energy_' + reference_baseline + '_' + mode + '_' + group, total > 1e-10)
        total_by_group[group] = total
        obs_by_group[group] = {
            k: float(np.sum((U[:, :k].T @ r) ** 2) / total)
            for k in K_VALUES
        }
    null = {group: {k: [] for k in K_VALUES} for group in group_matrices}
    for _ in range(N_RANDOM_BASES):
        coeff = rng.standard_normal((r_train.shape[1], max(K_VALUES)))
        q, _ = np.linalg.qr(r_train @ coeff, mode='reduced')
        for group, r in group_matrices.items():
            projections = q.T @ r
            cumulative = np.cumsum(np.sum(projections ** 2, axis=1))
            for k in K_VALUES:
                null[group][k].append(float(cumulative[k - 1] / total_by_group[group]))
    for group in group_matrices:
        for k in K_VALUES:
            values = np.asarray(null[group][k], dtype=float)
            obs = obs_by_group[group][k]
            out.append({
                'reference_baseline': reference_baseline,
                'normalisation': mode,
                'external_group': group,
                'n_external_donors': group_matrices[group].shape[1],
                'new_dimensions': k,
                'external_residual_fraction_explained': obs,
                'random_subspace_median_fraction': float(np.median(values)),
                'random_subspace_95pct_fraction': float(np.quantile(values, .95)),
                'exceeds_random_95pct': bool(obs > np.quantile(values, .95)),
                'null_draws': len(values),
                'empirical_random_tail_fraction': float((1 + np.sum(values >= obs)) / (len(values) + 1)),
            })
    return out


def main():
    print('=' * 78)
    print('CF-State Phase 6 — Audit 06b: external transfer of Atlanta residual PCs')
    print('=' * 78)
    require('new_output_directory', not OUT.exists())
    for name, path in INPUTS.items():
        require('input_' + name, path.is_file())
    require('06a_status_frozen', INPUTS['06a_status'].read_text().strip() == 'CFSTATE_PHASE6_06A_EXPLORATORY_REFERENCE_RESIDUAL_BENCHMARK_COMPLETE')
    require('05h_status_frozen', INPUTS['05h_status'].read_text().strip() == 'CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN')

    genes = pd.read_csv(INPUTS['mask'])['GeneID'].astype(str).tolist()
    require('frozen_4923_gene_mask', len(genes) == 4923 and len(set(genes)) == 4923)
    atlanta = pd.read_csv(INPUTS['atlanta']).set_index('GeneID')
    require('atlanta_4923_by_13', atlanta.shape == (4923, 13) and set(atlanta.index) == set(genes) and atlanta.columns.is_unique)
    X_atl = atlanta.loc[genes].to_numpy(dtype=float, copy=True)
    require('atlanta_expression_finite', np.isfinite(X_atl).all())

    cal = pd.read_csv(INPUTS['calibration'])
    names = cal['imodulon'].astype(str).tolist()
    stable = pd.read_csv(INPUTS['stable'])['imodulon'].astype(str).tolist()
    require('programme_registry_116_108', len(names) == 116 and len(set(names)) == 116 and len(stable) == 108 and len(set(stable)) == 108 and set(stable).issubset(names))
    M = load_m(INPUTS['modulome'], genes, names)

    counts = pd.read_csv(INPUTS['counts'], index_col=0)
    counts.index = counts.index.astype(str)
    manifest = pd.read_csv(INPUTS['manifest'], dtype=str).fillna('')
    require('external_5567_by_13', counts.shape == (5567, 13) and counts.index.is_unique and counts.columns.is_unique)
    require('external_manifest_matches', len(manifest) == 13 and manifest.published_donor_group.nunique() == 12 and set(manifest.published_sample) == set(counts.columns))
    require('external_groups_expected', (manifest.group == 'CF_sputum').sum() == 7 and (manifest.group == 'nonCF_chronic_wound').sum() == 4 and (manifest.group == 'nonCF_burn').sum() == 2)
    require('external_counts_valid', np.isfinite(counts.to_numpy(dtype=float)).all() and (counts.to_numpy(dtype=float) >= 0).all())
    saf = pd.read_csv(INPUTS['saf'], sep='\t').set_index('GeneID')
    require('saf_gene_universe', saf.index.is_unique and set(saf.index) == set(counts.index))
    lengths = (saf.loc[counts.index, 'End'] - saf.loc[counts.index, 'Start'] + 1).astype(float)
    require('saf_positive_lengths', (lengths > 0).all() and np.isfinite(lengths).all())
    require('mask_within_external', set(genes).issubset(counts.index))

    rng = np.random.default_rng(SEED)
    records = []
    sample_groups = {}
    donor_columns = []
    for donor, frame in manifest.groupby('published_donor_group', sort=False):
        sample_groups[donor] = 'CF' if (frame.group == 'CF_sputum').all() else 'nonCF'
        require('donor_group_consistent_' + donor, frame.group.nunique() == 1)
        donor_columns.append((donor, frame.published_sample.tolist()))
    require('external_7_CF_5_nonCF_donors', sum(g == 'CF' for g in sample_groups.values()) == 7 and sum(g == 'nonCF' for g in sample_groups.values()) == 5)

    for mode in ('FROZEN_5567_TPM', 'SOFTCORE_4923_TPM'):
        expr_counts = counts if mode == 'FROZEN_5567_TPM' else counts.loc[genes]
        expr = logtpm(expr_counts, lengths.loc[expr_counts.index]).loc[genes]
        D = pd.DataFrame({donor: expr[samples].mean(axis=1) for donor, samples in donor_columns})
        group_centered = {}
        for group in ('CF', 'nonCF'):
            donors = [donor for donor in D.columns if sample_groups[donor] == group]
            subgroup = D[donors].to_numpy(dtype=float, copy=True)
            group_centered[group] = subgroup - subgroup.mean(axis=1, keepdims=True)
            require('nonzero_external_group_' + mode + '_' + group, np.sum(group_centered[group] ** 2) > 1e-10)

        for baseline, programmes in [('STABLE_108', stable), ('FULL_116', names)]:
            print('\nEvaluating:', mode, baseline, flush=True)
            basis = M[programmes].to_numpy(dtype=float, copy=True)
            require('baseline_full_rank_' + baseline, np.linalg.matrix_rank(basis) == len(programmes))
            pinv = np.linalg.pinv(basis)
            train_centered = X_atl - X_atl.mean(axis=1, keepdims=True)
            r_train = orth_residual(train_centered, basis, pinv)
            u, s, _ = np.linalg.svd(r_train, full_matrices=False)
            require('training_PCs_finite_' + baseline, np.isfinite(u).all() and np.isfinite(s).all())
            require('train_residual_nonzero_' + baseline, s[2] > 1e-9)
            if baseline == 'STABLE_108':
                pc1 = pd.read_csv(INPUTS['06a_pc1']).set_index('GeneID').loc[genes, 'exploratory_PC1_weight'].to_numpy(dtype=float)
                require('reproduce_06a_frozen_PC1', abs(float(np.dot(pc1, u[:, 0]))) > 0.999999)
            groups = {group: orth_residual(group_centered[group], basis, pinv) for group in group_centered}
            records.extend(compute_comparisons(u, r_train, groups, baseline, mode, rng))

    summary = pd.DataFrame(records)
    require('24_evaluations', summary.shape[0] == 24)
    require('fractions_valid', np.isfinite(summary.select_dtypes(include='number').to_numpy()).all() and summary.external_residual_fraction_explained.between(0, 1).all())
    OUT.mkdir(parents=True, exist_ok=False)
    summary.to_csv(OUT / 'FROZEN_24_external_Cornforth_residual_transfer_checks.csv', index=False)
    pd.DataFrame([{'input': key, 'path': str(path), 'sha256': sha256(path)} for key, path in INPUTS.items()]).to_csv(OUT / 'input_sha256_provenance.csv', index=False)
    (OUT / 'DESIGN_AND_LIMITATIONS.txt').write_text(
        'All residual PC gene loadings fitted on Atlanta 13 only; no Cornforth fitting.\n'
        'Cornforth 2018 is processed using the frozen PAO1-only count baseline; strain bias unresolved.\n'
        'For the external cohort, expression is centered within CF (7 donors) and nonCF (5 donors) groups separately.\n'
        'The two burn specimens are averaged at the donor level before centering.\n'
        'The metric tests transfer of WITHIN-GROUP transcriptional variation, not CF-versus-nonCF enrichment.\n'
        'Test coefficients are projected from full test-gene expression; this is NOT unseen-gene prediction.\n'
        'Random baseline: 256 random k-spaces inside the same Atlanta residual training span.\n'
        'Unmatched collection sites/geographies and unequal RNA coverage prevent CF-specific conclusions.\n'
        'Atlanta donor identities unverified. No external learned programme is claimed validated.\n'
        'No thresholds or exclusions applied; reported empirical random tails are descriptive only.\n'
    )
    status = 'CFSTATE_PHASE6_06B_EXPLORATORY_EXTERNAL_RESIDUAL_TRANSFER_COMPLETE'
    (OUT / 'AUDIT_06B_STATUS.txt').write_text(status + '\n')
    show_cols = ['reference_baseline', 'normalisation', 'external_group', 'n_external_donors', 'new_dimensions', 'external_residual_fraction_explained', 'random_subspace_median_fraction', 'random_subspace_95pct_fraction', 'exceeds_random_95pct']
    print('\n========== EXTERNAL RESIDUAL TRANSFER ==========', flush=True)
    print(summary[show_cols].round(5).to_string(index=False), flush=True)
    print('\nNOTE: within-group variation, not CF enrichment; no patient-level p-value.', flush=True)
    print('STATUS:', status, flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
