#!/usr/bin/env python3
"""CF-State Phase 6 / Audit 06c.

Diagnostic of stability of three Atlanta-trained residual gene axes on the
Cornforth 2018 clinical expression cohort. This is NOT CF-specific inference.
Does not change pre-existing audits; writes only a new output directory.
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
P6 = ROOT / 'results/phase6'
OUT = P6 / 'audit_06c_residual_component_stability_diagnostics'
FILES = {
    '06b_status': P6 / 'audit_06b_Cornforth2018_external_residual_transfer/AUDIT_06B_STATUS.txt',
    '06b_summary': P6 / 'audit_06b_Cornforth2018_external_residual_transfer/FROZEN_24_external_Cornforth_residual_transfer_checks.csv',
    '06a_pc1': P6 / 'audit_06a_Atlanta_reference_residual_LOSO/EXPLORATORY_NOT_VALIDATED_Atlanta_residual_PC1_gene_weights.csv',
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

def check(label, value):
    if not bool(value):
        raise RuntimeError('FAILED: ' + label)
    print(label + ' PASS', flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_modulome(path, genes, programmes):
    obj = json.loads(path.read_text())['M']
    if isinstance(obj, str):
        obj = json.loads(obj)
    if isinstance(obj, dict) and all(k in obj for k in ('index', 'columns', 'data')):
        frame = pd.DataFrame(obj['data'], index=obj['index'], columns=obj['columns'])
    else:
        frame = pd.DataFrame(obj)
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    if set(genes).issubset(frame.columns) and set(programmes).issubset(frame.index):
        frame = frame.T
    check('modulome_gene_programme_ids', set(genes).issubset(frame.index) and set(programmes).issubset(frame.columns))
    check('modulome_unique_gene_programme_ids', frame.index.is_unique and frame.columns.is_unique)
    result = frame.loc[genes, programmes].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float, copy=True)
    check('modulome_finite', np.isfinite(result).all())
    return result


def log_tpm(counts, lengths):
    rates = counts.div(lengths, axis=0)
    denom = rates.sum(axis=0)
    check('TPM_denominators_positive', (denom > 0).all() and np.isfinite(denom).all())
    tpm = rates.div(denom, axis=1) * 1e6
    check('TPM_columns_sum_to_1e6', np.allclose(tpm.sum(axis=0), 1e6, rtol=0, atol=.01))
    return np.log2(tpm + 1.0)


def residual(x, basis, operator):
    return x - basis @ (operator @ x)


def learn_pcs(x, basis, operator):
    centered = x - x.mean(axis=1, keepdims=True)
    r = residual(centered, basis, operator)
    u, s, _ = np.linalg.svd(r, full_matrices=False)
    check('learnt_axes_finite', np.isfinite(u).all() and np.isfinite(s).all() and s[2] > 1e-9)
    return u[:, :3]


def fractions(u, residual_matrix):
    total = float(np.sum(residual_matrix ** 2))
    if not np.isfinite(total) or total <= 1e-10:
        raise RuntimeError('Degenerate external within-group residual')
    energy_by_pc = np.sum((u.T @ residual_matrix) ** 2, axis=1) / total
    return energy_by_pc, float(np.sum(energy_by_pc))


def main():
    print('=' * 76)
    print('CF-State Phase 6 — Audit 06c: external transfer stability diagnostics')
    print('=' * 76)
    check('new_output_directory', not OUT.exists())
    for name, path in FILES.items():
        check('input_' + name, path.is_file())
    check('06b_frozen', FILES['06b_status'].read_text().strip() == 'CFSTATE_PHASE6_06B_EXPLORATORY_EXTERNAL_RESIDUAL_TRANSFER_COMPLETE')
    check('05h_frozen', FILES['05h_status'].read_text().strip() == 'CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN')
    mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
    check('4923_unique_mask', len(mask) == 4923 and len(set(mask)) == 4923)
    atl = pd.read_csv(FILES['atlanta']).set_index('GeneID')
    check('Atlanta_4923_by_13', atl.shape == (4923, 13) and atl.index.is_unique and set(atl.index) == set(mask))
    X = atl.loc[mask].to_numpy(dtype=float, copy=True)
    check('Atlanta_finite', np.isfinite(X).all())
    all_names = pd.read_csv(FILES['calibration'])['imodulon'].astype(str).tolist()
    stable = pd.read_csv(FILES['stable'])['imodulon'].astype(str).tolist()
    check('108_stable_of_116', len(stable) == 108 and len(set(stable)) == 108 and len(all_names) == 116 and set(stable).issubset(all_names))
    basis = load_modulome(FILES['modulome'], mask, stable)
    check('reference_rank_108', np.linalg.matrix_rank(basis) == 108)
    operator = np.linalg.pinv(basis)
    U = learn_pcs(X, basis, operator)
    old_pc1 = pd.read_csv(FILES['06a_pc1']).set_index('GeneID').loc[mask, 'exploratory_PC1_weight'].to_numpy(dtype=float)
    check('reproduce_frozen_06a_PC1', abs(float(np.dot(old_pc1, U[:, 0]))) > .999999)

    counts = pd.read_csv(FILES['counts'], index_col=0)
    counts.index = counts.index.astype(str)
    manifest = pd.read_csv(FILES['manifest'], dtype=str).fillna('')
    check('counts_5567_by_13', counts.shape == (5567, 13) and counts.index.is_unique and set(counts.columns) == set(manifest['published_sample']))
    check('12_donors', len(manifest) == 13 and manifest['published_donor_group'].nunique() == 12)
    check('valid_counts', np.isfinite(counts.to_numpy(dtype=float)).all() and (counts.to_numpy(dtype=float) >= 0).all())
    saf = pd.read_csv(FILES['saf'], sep='\t').set_index('GeneID')
    check('SAF_matches_counts', saf.index.is_unique and set(saf.index) == set(counts.index))
    lengths = (saf.loc[counts.index, 'End'] - saf.loc[counts.index, 'Start'] + 1).astype(float)
    check('gene_lengths_positive', np.isfinite(lengths).all() and (lengths > 0).all())
    donors = []
    group_of = {}
    for donor, block in manifest.groupby('published_donor_group', sort=False):
        check('group_consistent_' + donor, block.group.nunique() == 1)
        group_of[donor] = 'CF' if block.group.iloc[0] == 'CF_sputum' else 'nonCF'
        donors.append((donor, block.published_sample.tolist()))
    check('group_donors_7_5', sum(x == 'CF' for x in group_of.values()) == 7 and sum(x == 'nonCF' for x in group_of.values()) == 5)

    previous = pd.read_csv(FILES['06b_summary'])
    check('06b_24_rows', len(previous) == 24)
    training_rows = []
    donor_rows = []
    pc_rows = []
    for norm in ('FROZEN_5567_TPM', 'SOFTCORE_4923_TPM'):
        expr_counts = counts if norm == 'FROZEN_5567_TPM' else counts.loc[mask]
        expression = log_tpm(expr_counts, lengths.loc[expr_counts.index]).loc[mask]
        averaged = pd.DataFrame({donor: expression[samples].mean(axis=1) for donor, samples in donors})
        for group in ('CF', 'nonCF'):
            names = [d for d in averaged.columns if group_of[d] == group]
            G = averaged[names].to_numpy(dtype=float, copy=True)
            R = residual(G - G.mean(axis=1, keepdims=True), basis, operator)
            component, whole = fractions(U, R)
            prior = previous.loc[
                (previous.reference_baseline == 'STABLE_108')
                & (previous.normalisation == norm)
                & (previous.external_group == group)
                & (previous.new_dimensions == 3),
                'external_residual_fraction_explained'
            ]
            check('reproduce_06b_' + norm + '_' + group,
                  len(prior) == 1 and np.isclose(whole, float(prior.iloc[0]), atol=1e-9, rtol=0))
            pc_rows.append({
                'normalisation': norm, 'group': group, 'n_donors': len(names),
                'PC1_fraction': component[0], 'PC2_fraction': component[1],
                'PC3_fraction': component[2], 'all3_fraction': whole
            })
            for i, omit in enumerate(atl.columns):
                idx = [j for j in range(X.shape[1]) if j != i]
                U_retrained = learn_pcs(X[:, idx], basis, operator)
                _, explained = fractions(U_retrained, R)
                overlap = float(np.linalg.norm(U.T @ U_retrained, ord='fro') ** 2 / 3.0)
                training_rows.append({
                    'normalisation': norm, 'external_group': group,
                    'omitted_Atlanta_specimen': omit,
                    'external_fraction_with_3_retrained_PCs': explained,
                    'PC_subspace_overlap_with_full_Atlanta': overlap
                })
            for omit in names:
                remaining = [d for d in names if d != omit]
                G2 = averaged[remaining].to_numpy(dtype=float, copy=True)
                R2 = residual(G2 - G2.mean(axis=1, keepdims=True), basis, operator)
                _, explained = fractions(U, R2)
                donor_rows.append({
                    'normalisation': norm, 'external_group': group,
                    'omitted_external_donor': omit, 'remaining_donors': len(remaining),
                    'fraction_with_3_frozen_Atlanta_PCs': explained
                })

    pc = pd.DataFrame(pc_rows)
    train = pd.DataFrame(training_rows)
    donor = pd.DataFrame(donor_rows)
    check('expected_4_PC_rows', len(pc) == 4)
    check('expected_52_training_checks', len(train) == 52)
    check('expected_24_external_donor_checks', len(donor) == 24)
    numeric = [pc.select_dtypes(include='number'), train.select_dtypes(include='number'), donor.select_dtypes(include='number')]
    check('all_outputs_finite', all(np.isfinite(x.to_numpy(dtype=float)).all() for x in numeric))
    check('valid_overlap', train.PC_subspace_overlap_with_full_Atlanta.between(0, 1.000001).all())

    OUT.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({'GeneID': mask, 'exploratory_PC1_weight': U[:, 0],
                  'exploratory_PC2_weight': U[:, 1],
                  'exploratory_PC3_weight': U[:, 2]}).to_csv(
        OUT / 'EXPLORATORY_NOT_VALIDATED_Atlanta_residual_PC1_PC2_PC3_gene_weights.csv', index=False)
    pc.to_csv(OUT / 'FROZEN_external_PC1_PC2_PC3_contributions.csv', index=False)
    train.to_csv(OUT / 'FROZEN_Atlanta_leave_one_specimen_out_retraining.csv', index=False)
    donor.to_csv(OUT / 'FROZEN_external_leave_one_donor_out_checks.csv', index=False)
    pd.DataFrame([{'input': name, 'path': str(path), 'sha256': digest(path)} for name, path in FILES.items()]).to_csv(OUT / 'input_provenance_sha256.csv', index=False)
    (OUT / 'DESIGN_AND_LIMITATIONS.txt').write_text(
        'Diagnostic of 06b residual subspace generalisation; no new programmes are validated.\n'
        'Atlanta training: 13 specimens, donor identities not verified; 108 frozen iModulons.\n'
        'Retraining check: leave 1 Atlanta specimen out; compare 3-PC subspace and external explained fractions.\n'
        'External check: leave 1 donor out WITHIN CF or nonCF; recompute group-centered variation.\n'
        'Burn replicates aggregated per patient. Gene projection coefficients fitted on test gene expression.\n'
        'Gene loadings and rank-3 selection remain exploratory; k=3 was inspected in 06b.\n'
        'Group-centered expression cannot establish CF specificity, state prediction or unseen-gene prediction.\n'
        'CF/nonCF groups confounded by body site and collection location; mapping differences unresolved.\n'
        'No patient-level significance test; no technical QC exclusions; prior audits unchanged.\n'
    )
    status = 'CFSTATE_PHASE6_06C_RESIDUAL_TRANSFER_STABILITY_DIAGNOSTICS_COMPLETE'
    (OUT / 'AUDIT_06C_STATUS.txt').write_text(status + '\n')
    print('\n========== COMPONENT CONTRIBUTIONS (STABLE 108) ==========')
    print(pc.round(5).to_string(index=False))
    print('\n========== LEAVE-ONE-ATLANTA-OUT TRAINING STABILITY ==========')
    print(train.groupby(['normalisation', 'external_group']).agg(
        external_fraction_min=('external_fraction_with_3_retrained_PCs', 'min'),
        external_fraction_median=('external_fraction_with_3_retrained_PCs', 'median'),
        external_fraction_max=('external_fraction_with_3_retrained_PCs', 'max'),
        subspace_overlap_min=('PC_subspace_overlap_with_full_Atlanta', 'min'),
        subspace_overlap_median=('PC_subspace_overlap_with_full_Atlanta', 'median')
    ).round(5).to_string())
    print('\n========== LEAVE-ONE-EXTERNAL-DONOR-OUT ==========')
    print(donor.groupby(['normalisation', 'external_group']).agg(
        external_fraction_min=('fraction_with_3_frozen_Atlanta_PCs', 'min'),
        external_fraction_median=('fraction_with_3_frozen_Atlanta_PCs', 'median'),
        external_fraction_max=('fraction_with_3_frozen_Atlanta_PCs', 'max')
    ).round(5).to_string())
    print('\nInterpretation: subspace overlap near 1 is more stable; not a CF specificity test.')
    print('STATUS:', status)

if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
