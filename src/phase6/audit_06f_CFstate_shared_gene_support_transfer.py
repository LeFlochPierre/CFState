#!/usr/bin/env python3
"""CF-State Phase 6 / Audit 06f.

Post hoc sensitivity: does external clinical residual transfer of the frozen
Atlanta 3-PC subspace persist when analysed in genes measured in BOTH the
CF and non-CF Cornforth groups?

The support mask is derived from external data, and therefore the resulting
comparison is *not* independent validation. Do not infer CF specificity.
No changes to prior audits or existing learnt programme weights.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / 'results/phase6/audit_06f_shared_gene_support_transfer_sensitivity'
BASE6 = ROOT / 'results/phase6'
BASE4 = ROOT / 'results/phase4'
BASE5 = ROOT / 'results/phase5'
INPUTS = {
    '06e_status': BASE6 / 'audit_06e_PAO1_functional_annotation/AUDIT_06E_STATUS.txt',
    '06c_status': BASE6 / 'audit_06c_residual_component_stability_diagnostics/AUDIT_06C_STATUS.txt',
    '06b_status': BASE6 / 'audit_06b_Cornforth2018_external_residual_transfer/AUDIT_06B_STATUS.txt',
    '06c_loadings': BASE6 / 'audit_06c_residual_component_stability_diagnostics/EXPLORATORY_NOT_VALIDATED_Atlanta_residual_PC1_PC2_PC3_gene_weights.csv',
    '06b_baselines': BASE6 / 'audit_06b_Cornforth2018_external_residual_transfer/FROZEN_24_external_Cornforth_residual_transfer_checks.csv',
    'mask': BASE4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'stable': BASE4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'counts': BASE5 / 'audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv',
    'manifest': BASE5 / 'audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv',
    'saf': ROOT / 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf',
    'script': Path(__file__).resolve(),
}
SEED = 20261014
N_RANDOM = 256


def require(test, label):
    if not bool(test):
        raise RuntimeError('FAILED: ' + label)
    print(label + ' PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_modulome(path, genes, names):
    obj = json.loads(path.read_text())['M']
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and all(k in obj for k in ('index', 'columns', 'data')):
        m = pd.DataFrame(obj['data'], index=obj['index'], columns=obj['columns'])
    else:
        m = pd.DataFrame(obj)
    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)
    if set(genes).issubset(m.columns) and set(names).issubset(m.index):
        m = m.T
    require(m.index.is_unique and m.columns.is_unique, 'modulome_unique_ids')
    require(set(genes).issubset(m.index) and set(names).issubset(m.columns), 'modulome_ids')
    numeric = m.loc[genes, names].apply(pd.to_numeric, errors='raise')
    require(np.isfinite(numeric.to_numpy(dtype=float)).all(), 'modulome_finite')
    return numeric.to_numpy(dtype=float, copy=True)


def log_tpm(counts, lengths):
    rates = counts.div(lengths, axis=0)
    denom = rates.sum(axis=0)
    require((denom > 0).all() and np.isfinite(denom).all(), 'positive_TPM_denominators')
    tpm = rates.div(denom, axis=1) * 1e6
    require(np.allclose(tpm.sum(axis=0), 1e6, atol=.01, rtol=0), 'TPM_columns_sum_to_1e6')
    return np.log2(tpm + 1.0)


def project_residual(x, basis):
    return x - basis @ (np.linalg.pinv(basis) @ x)


def evaluate_subset(keep, m_full, u_full, grouped_full, atl_centered, rng):
    # Freeze original Atlanta loadings; do not relearn any PC from external data.
    # On a subset of genes, reorthogonalise against the restricted reference
    # iModulons so the added subspace cannot simply duplicate the reference.
    m = m_full[keep, :]
    require(m.shape[0] > 108 and np.linalg.matrix_rank(m) == 108,
            'masked_reference_rank_108')
    train = project_residual(atl_centered[keep, :], m)
    fixed = project_residual(u_full[keep, :], m)
    q, r = np.linalg.qr(fixed, mode='reduced')
    require(np.linalg.matrix_rank(r) == 3, 'three_fixed_axes_non_degenerate')
    require(np.allclose(q.T @ q, np.eye(3), atol=1e-8), 'fixed_axes_orthonormal')
    external = {
        group: project_residual(centered[keep, :], m)
        for group, centered in grouped_full.items()
    }
    totals = {key: float(np.sum(v ** 2)) for key, v in external.items()}
    require(all(v > 1e-10 for v in totals.values()), 'external_variation_positive')
    actual = {
        group: float(np.sum((q.T @ matrix) ** 2) / totals[group])
        for group, matrix in external.items()
    }
    random_values = {group: [] for group in external}
    for _ in range(N_RANDOM):
        coeff = rng.standard_normal((train.shape[1], 3))
        null_q, _ = np.linalg.qr(train @ coeff, mode='reduced')
        for group, matrix in external.items():
            fraction = float(np.sum((null_q.T @ matrix) ** 2) / totals[group])
            random_values[group].append(fraction)
    results = {}
    for group in external:
        vals = np.array(random_values[group])
        results[group] = {
            'fraction_explained': actual[group],
            'random_median': float(np.median(vals)),
            'random_95th_percentile': float(np.quantile(vals, .95)),
            'above_random_95pct': bool(actual[group] > np.quantile(vals, .95)),
        }
    return results


def main():
    print('=' * 77)
    print('CF-State Phase 6 — Audit 06f: shared-gene detection sensitivity')
    print('=' * 77)
    require(not OUT.exists(), 'new_output_directory')
    for label, path in INPUTS.items():
        require(path.is_file(), 'input_' + label)
    expected_statuses = {
        '06e_status': 'CFSTATE_PHASE6_06E_DESCRIPTIVE_GENE_ANNOTATION_COMPLETE',
        '06c_status': 'CFSTATE_PHASE6_06C_RESIDUAL_TRANSFER_STABILITY_DIAGNOSTICS_COMPLETE',
        '06b_status': 'CFSTATE_PHASE6_06B_EXPLORATORY_EXTERNAL_RESIDUAL_TRANSFER_COMPLETE',
    }
    for label, expected in expected_statuses.items():
        require(INPUTS[label].read_text().strip() == expected, 'frozen_' + label)

    genes = pd.read_csv(INPUTS['mask'])['GeneID'].astype(str).tolist()
    require(len(genes) == 4923 and len(set(genes)) == 4923, 'frozen_gene_universe')
    stable = pd.read_csv(INPUTS['stable'])['imodulon'].astype(str).tolist()
    require(len(stable) == 108 and len(set(stable)) == 108, '108_stable_programmes')
    m_full = load_modulome(INPUTS['modulome'], genes, stable)
    u_frame = pd.read_csv(INPUTS['06c_loadings']).set_index('GeneID').loc[genes]
    axes = ['exploratory_PC1_weight', 'exploratory_PC2_weight', 'exploratory_PC3_weight']
    u_full = u_frame[axes].to_numpy(dtype=float, copy=True)
    require(u_full.shape == (4923, 3) and np.isfinite(u_full).all(), 'three_frozen_pc_gene_weights')
    require(np.allclose(u_full.T @ u_full, np.eye(3), atol=1e-8), 'frozen_pc_orthonormal')

    counts = pd.read_csv(INPUTS['counts'], index_col=0)
    counts.index = counts.index.astype(str)
    manifest = pd.read_csv(INPUTS['manifest'], dtype=str).fillna('')
    require(counts.shape == (5567, 13) and counts.index.is_unique, 'external_counts_5567_by_13')
    require(set(counts.columns) == set(manifest['published_sample']), 'samples_match_manifest')
    require((counts.to_numpy(dtype=float) >= 0).all(), 'nonnegative_counts')
    saf = pd.read_csv(INPUTS['saf'], sep='\t').set_index('GeneID')
    require(set(saf.index) == set(counts.index), 'SAF_matches_counts')
    lengths = (saf.loc[counts.index, 'End'] - saf.loc[counts.index, 'Start'] + 1).astype(float)
    require((lengths > 0).all(), 'positive_gene_lengths')

    donor_columns = []
    donor_group = {}
    for donor, frame in manifest.groupby('published_donor_group', sort=False):
        require(frame.group.nunique() == 1, 'consistent_donor_group_' + donor)
        donor_columns.append((donor, frame.published_sample.tolist()))
        donor_group[donor] = 'CF' if frame.group.iloc[0] == 'CF_sputum' else 'nonCF'
    require(len(donor_columns) == 12, '12_distinct_donors')
    require(sum(v == 'CF' for v in donor_group.values()) == 7, '7_CF_donors')
    require(sum(v == 'nonCF' for v in donor_group.values()) == 5, '5_nonCF_donors')

    # One burn patient has two specimens; gene support is presence in either.
    donor_counts = pd.DataFrame({d: counts.loc[genes, samples].max(axis=1)
                                 for d, samples in donor_columns})
    cf_cols = [d for d in donor_counts if donor_group[d] == 'CF']
    other_cols = [d for d in donor_counts if donor_group[d] == 'nonCF']
    require(len(cf_cols) == 7 and len(other_cols) == 5, 'two_correct_donor_groups')
    masks = {
        'ALL_4923': np.ones(len(genes), dtype=bool),
        'SHARED_GE1_CF6_NC4': ((donor_counts[cf_cols] >= 1).sum(axis=1) >= 6).to_numpy()
                             & ((donor_counts[other_cols] >= 1).sum(axis=1) >= 4).to_numpy(),
        'SHARED_GE3_CF5_NC4': ((donor_counts[cf_cols] >= 3).sum(axis=1) >= 5).to_numpy()
                             & ((donor_counts[other_cols] >= 3).sum(axis=1) >= 4).to_numpy(),
        'SHARED_GE10_CF5_NC4': ((donor_counts[cf_cols] >= 10).sum(axis=1) >= 5).to_numpy()
                              & ((donor_counts[other_cols] >= 10).sum(axis=1) >= 4).to_numpy(),
    }
    for name, keep in masks.items():
        require(keep.sum() >= 150, 'sufficient_genes_' + name)
        print(name + ': ' + str(int(keep.sum())) + '/4923 genes')

    baseline = pd.read_csv(INPUTS['06b_baselines'])
    results = []
    mask_stats = []
    for name, keep in masks.items():
        energy = np.sum(u_full[keep, :] ** 2, axis=0)
        mask_stats.append({
            'gene_subset': name,
            'n_genes': int(keep.sum()),
            'PC1_original_weight_energy_retained': float(energy[0]),
            'PC2_original_weight_energy_retained': float(energy[1]),
            'PC3_original_weight_energy_retained': float(energy[2]),
        })
    rng = np.random.default_rng(SEED)
    # Atlanta PCs were originally learnt in 4923-gene log2TPM space;
    # centre Atlanta only to construct random training-span comparators.
    atl = pd.read_csv(BASE4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_Atlanta_softcore_4923_log2TPM.csv').set_index('GeneID')
    require(atl.shape == (4923, 13) and set(atl.index) == set(genes), 'Atlanta_training_4923_by_13')
    x_train = atl.loc[genes].to_numpy(dtype=float, copy=True)
    x_train = x_train - x_train.mean(axis=1, keepdims=True)
    for mode in ('FROZEN_5567_TPM', 'SOFTCORE_4923_TPM'):
        source = counts if mode == 'FROZEN_5567_TPM' else counts.loc[genes]
        expr = log_tpm(source, lengths.loc[source.index]).loc[genes]
        donor_expr = pd.DataFrame({d: expr[samples].mean(axis=1)
                                   for d, samples in donor_columns})
        group_matrices = {}
        for group in ('CF', 'nonCF'):
            cols = [d for d in donor_expr if donor_group[d] == group]
            x = donor_expr[cols].to_numpy(dtype=float, copy=True)
            group_matrices[group] = x - x.mean(axis=1, keepdims=True)
        for name, keep in masks.items():
            outcome = evaluate_subset(keep, m_full, u_full, group_matrices, x_train, rng)
            for group, values in outcome.items():
                row = {'normalisation': mode, 'gene_subset': name,
                       'external_group': group, 'n_genes': int(keep.sum())}
                row.update(values)
                if name == 'ALL_4923':
                    previous = baseline.loc[
                        (baseline.reference_baseline == 'STABLE_108')
                        & (baseline.normalisation == mode)
                        & (baseline.external_group == group)
                        & (baseline.new_dimensions == 3),
                        'external_residual_fraction_explained'
                    ]
                    require(len(previous) == 1 and
                            np.isclose(row['fraction_explained'], float(previous.iloc[0]), atol=1e-8, rtol=0),
                            'reproduce_frozen_06b_' + mode + '_' + group)
                results.append(row)

    df = pd.DataFrame(results)
    require(len(df) == 2 * 4 * 2, 'expected_16_results')
    require(df['fraction_explained'].between(0, 1).all(), 'fractions_between_0_and_1')
    OUT.mkdir(parents=True, exist_ok=False)
    df.to_csv(OUT / 'DESCRIPTIVE_16_shared_gene_external_transfer_checks.csv', index=False)
    pd.DataFrame(mask_stats).to_csv(OUT / 'DESCRIPTIVE_gene_support_masks_and_PC_weight_retention.csv', index=False)
    for name, keep in masks.items():
        pd.DataFrame({'GeneID': np.asarray(genes)[keep]}).to_csv(
            OUT / ('DESCRIPTIVE_gene_mask_' + name + '.csv'), index=False)
    pd.DataFrame([{'input': name, 'path': str(path), 'sha256': sha256(path)}
                  for name, path in INPUTS.items()]).to_csv(
                      OUT / 'input_provenance_sha256.csv', index=False)
    (OUT / 'DESIGN_AND_LIMITATIONS.txt').write_text(
        'FIXED original Atlanta gene loadings; no new CF model fitted.\n'
        'Previously frozen 108 iModulon basis; reorthogonalised for each gene subset.\n'
        'Shared-support subsets are derived POST HOC from Cornforth expression counts,\n'
        'including group membership; therefore NO independent held-out validation.\n'
        'CF and nonCF are confounded by anatomical site, geography, sequencing and strain.\n'
        'External group means removed; metric is WITHIN-group residual expression\n'
        'variation explained, NOT CF enrichment, not gene holdout prediction.\n'
        'Random subspaces sampled from same Atlanta within-cohort training residual span.\n'
        'Fractions have different gene-set denominators: do not directly interpret an\n'
        'increase with filtering as stronger biological transfer.\n'
        'Top-loading genes may be strain-specific, non-mappable or low coverage.\n'
        'The 3-component dimensionality was previously explored; no p-values.\n'
    )
    status = 'CFSTATE_PHASE6_06F_SHARED_GENE_SUPPORT_SENSITIVITY_COMPLETE'
    (OUT / 'AUDIT_06F_STATUS.txt').write_text(status + '\n')
    print('\n========== SHARED-GENE SENSITIVITY (FIXED THREE ATLANTA PCS) ==========')
    display = df[['normalisation', 'gene_subset', 'external_group', 'n_genes',
                  'fraction_explained', 'random_95th_percentile', 'above_random_95pct']]
    print(display.round(5).to_string(index=False))
    print('\n========== FROZEN PC WEIGHT ENERGY RETAINED BY MASK ==========')
    print(pd.DataFrame(mask_stats).round(5).to_string(index=False))
    print('\nAll gene-mask selections are post hoc. No inference of CF specificity.')
    print('STATUS:', status)


if __name__ == '__main__':
    main()
