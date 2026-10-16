#!/usr/bin/env python3
"""Audit 06l: apply frozen 06k in-vivo-vs-lab axis without refitting.

Descriptive, non-confirmatory transfer into:
  (a) Atlanta CF sputum relative to external CFBE culture references;
  (b) Cornforth 2018 CF-vs-non-CF patient-balanced clinical contrast.
Neither test estimates CF specificity; the datasets informed prior analyses.
"""
from pathlib import Path
import hashlib
import json
import traceback

import numpy as np
import pandas as pd

ROOT = Path.cwd()
BASE = ROOT / 'results/phase6/audit_06k_CFstate_matched_invivo_residual_v2'
OUT = ROOT / 'results/phase6/audit_06l_CFstate_v2_frozen_clinical_transfer'
A4 = ROOT / 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection'
P5 = ROOT / 'results/phase5'
FILES = {
    'model_status': BASE / 'AUDIT_06K_STATUS.txt',
    'frozen_model': BASE / 'FITTED_CFstate_v2_candidate_arrays.npz',
    'Atlanta_log2TPM': A4 / 'FROZEN_Atlanta_softcore_4923_log2TPM.csv',
    'CFBE_log2TPM': A4 / 'FROZEN_CFBE_softcore_4923_log2TPM_references.csv',
    'Cornforth_counts': P5 / 'audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv',
    'Cornforth_manifest': P5 / 'audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv',
    'Cornforth_SAF': ROOT / 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf',
    'Cornforth_status': P5 / 'audit_05h_Cornforth2018_full_depth_baseline/AUDIT_05H_STATUS.txt',
    'script': Path(__file__).resolve(),
}


def check(name, yes):
    if not bool(yes):
        raise RuntimeError('FAILED: ' + name)
    print(name + ' PASS', flush=True)


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_expression(path, key):
    df = pd.read_csv(path)
    check(path.stem + '_gene_column', key in df.columns)
    check(path.stem + '_genes_unique', df[key].astype(str).is_unique)
    return df.set_index(key).apply(pd.to_numeric, errors='raise')


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def axis_stats(delta, axis, Q):
    r = delta - Q @ (Q.T @ delta)
    n = np.linalg.norm(r)
    if not np.isfinite(n) or n < 1e-10:
        raise RuntimeError('Undefined residual norm')
    score = float(axis @ r)
    return {
        'signed_axis_projection': score,
        'signed_residual_cosine': cosine(r, axis),
        'residual_energy_fraction_along_axis': (score / n) ** 2,
        'residual_l2_norm': float(n),
    }


def tpm_full(counts, lengths):
    rates = counts.div(lengths, axis=0)
    denom = rates.sum(axis=0)
    check('Cornforth_positive_TPM_denominators', (denom > 0).all() and np.isfinite(denom.to_numpy()).all())
    expr = np.log2(1 + rates.div(denom, axis=1) * 1e6)
    check('Cornforth_finite_log2TPM', np.isfinite(expr.to_numpy()).all())
    return expr


def main():
    print('=' * 75)
    print('CF-State Phase 6 — Audit 06l: frozen v2 clinical transfer')
    print('=' * 75)
    check('new_output_directory', not OUT.exists())
    for name, path in FILES.items():
        check('input_' + name, path.is_file())
    check('06k_fitted', FILES['model_status'].read_text().strip() ==
          'CFSTATE_PHASE6_06K_CF_INVIVO_MATCHED_CONTRAST_MODEL_FITTED_EXPLORATORY')
    check('05h_frozen', FILES['Cornforth_status'].read_text().strip() ==
          'CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN')
    with np.load(FILES['frozen_model'], allow_pickle=False) as z:
        genes = z['genes'].astype(str).tolist()
        Q = np.array(z['reference_Q'], dtype=float, copy=True)
        axis = np.array(z['added_direction'], dtype=float, copy=True)
        train_patients = z['training_patients'].astype(str).tolist()
    check('unique_model_genes', len(genes) > 4000 and len(set(genes)) == len(genes))
    check('reference_shape', Q.shape == (len(genes), 108))
    check('axis_shape', axis.shape == (len(genes),))
    check('train_four_unique_PRJEB_patients', len(train_patients) == 4 and len(set(train_patients)) == 4)
    check('unit_axis_and_orthogonality', abs(np.linalg.norm(axis) - 1) < 1e-8 and
          np.max(np.abs(Q.T @ axis)) < 1e-8 and
          np.allclose(Q.T @ Q, np.eye(108), atol=1e-8))

    # Atlanta 13 clinical specimens against three *shared*, external CFBE references.
    atl = read_expression(FILES['Atlanta_log2TPM'], 'GeneID')
    cfbe = read_expression(FILES['CFBE_log2TPM'], 'GeneID')
    check('Atlanta_genes_present', set(genes).issubset(atl.index) and set(genes).issubset(cfbe.index))
    check('Atlanta_thirteen_profiles', atl.shape[1] == 13 and atl.columns.is_unique)
    check('reference_three_columns', set([
        'PRIMARY_CFBE_ALL4', 'SENSITIVITY_CFBE_C_PAIR',
        'SENSITIVITY_CFBE_R_PAIR']).issubset(cfbe.columns))
    A = atl.loc[genes]
    C = cfbe.loc[genes]
    check('Atlanta_CFBE_all_finite', np.isfinite(A.to_numpy()).all() and np.isfinite(C.to_numpy()).all())
    arows = []
    for reference in ['PRIMARY_CFBE_ALL4', 'SENSITIVITY_CFBE_C_PAIR', 'SENSITIVITY_CFBE_R_PAIR']:
        center = C[reference].to_numpy()
        for specimen in A.columns:
            values = axis_stats(A[specimen].to_numpy() - center, axis, Q)
            arows.append({'reference': reference, 'specimen': specimen, **values})
    atl_df = pd.DataFrame(arows)
    atl_summary = (atl_df.groupby('reference', sort=False)
                   .agg(n_specimens=('specimen', 'size'),
                        n_positive=('signed_residual_cosine', lambda v: int((v > 0).sum())),
                        min_cosine=('signed_residual_cosine', 'min'),
                        median_cosine=('signed_residual_cosine', 'median'),
                        max_cosine=('signed_residual_cosine', 'max'))
                   .reset_index())
    print('\n========== ATLANTA CF AGAINST SHARED CFBE LAB REFERENCE ==========')
    print(atl_summary.round(5).to_string(index=False))
    print('Specimens may not represent 13 independent donors.')
    print('A shared CFBE reference can induce common shifts across specimens.')

    # Cornforth: patient-balanced CF-vs-non-CF difference; no matched in-vitro baseline.
    counts = pd.read_csv(FILES['Cornforth_counts'], index_col=0)
    counts.index = counts.index.astype(str)
    md = pd.read_csv(FILES['Cornforth_manifest'], dtype=str).fillna('')
    check('Cornforth_counts_5567_by_13', counts.shape == (5567, 13) and counts.index.is_unique)
    check('Cornforth_manifest_matching', len(md) == 13 and set(md.published_sample) == set(counts.columns))
    check('Cornforth_donor_count', md.published_donor_group.nunique() == 12)
    check('Cornforth_group_counts', (md.group == 'CF_sputum').sum() == 7 and
          (md.group == 'nonCF_chronic_wound').sum() == 4 and (md.group == 'nonCF_burn').sum() == 2)
    saf = pd.read_csv(FILES['Cornforth_SAF'], sep='\t')
    check('SAF_gene_matches', saf.GeneID.is_unique and set(saf.GeneID) == set(counts.index))
    sf = saf.set_index('GeneID').loc[counts.index]
    lengths = (sf.End.astype(float) - sf.Start.astype(float) + 1)
    check('SAF_valid_lengths', (lengths > 0).all())
    logtpm = tpm_full(counts.apply(pd.to_numeric, errors='raise'), lengths)
    check('Cornforth_model_genes_present', set(genes).issubset(logtpm.index))
    logtpm = logtpm.loc[genes]
    donors = {}
    types = {}
    for donor, rows in md.groupby('published_donor_group', sort=False):
        check('one_group_' + donor, rows.group.nunique() == 1)
        samples = rows.published_sample.tolist()
        donors[donor] = logtpm[samples].mean(axis=1).to_numpy()
        types[donor] = 'CF' if rows.group.iloc[0] == 'CF_sputum' else 'nonCF'
    cf = [d for d in donors if types[d] == 'CF']
    noncf = [d for d in donors if types[d] == 'nonCF']
    check('donors_seven_vs_five', len(cf) == 7 and len(noncf) == 5)
    d = np.mean([donors[v] for v in cf], axis=0) - np.mean([donors[v] for v in noncf], axis=0)
    crow = {'comparison': 'CF_minus_nonCF_patient_balanced', 'n_CF_donors': len(cf),
            'n_nonCF_donors': len(noncf), **axis_stats(d, axis, Q)}
    print('\n========== CORNFORTH CF MINUS NON-CF CLINICAL CONTRAST ==========')
    print(pd.DataFrame([crow]).round(5).to_string(index=False))
    print('This has no matched-lab comparator and cannot isolate CF from tissue/geography.')

    OUT.mkdir(parents=True, exist_ok=False)
    atl_df.to_csv(OUT / 'DESCRIPTIVE_Atlanta_13_CFBE_axis_scores.csv', index=False)
    atl_summary.to_csv(OUT / 'DESCRIPTIVE_Atlanta_CFBE_reference_sensitivity.csv', index=False)
    pd.DataFrame([crow]).to_csv(OUT / 'DESCRIPTIVE_Cornforth_7vs5_patient_balanced_axis_contrast.csv', index=False)
    pd.DataFrame([{'file': name, 'path': str(path), 'sha256': file_hash(path)}
                  for name, path in FILES.items()]).to_csv(OUT / 'input_sha256.csv', index=False)
    (OUT / 'DESIGN_LIMITATIONS.txt').write_text(
        'Frozen 06k PRJEB-trained gene-space axis. No training or model selection here.\n'
        'Atlanta contrasts compare CF sputum against SHARED external CFBE cultures.\n'
        'Atlanta 13 specimens are not verified to be 13 independent patients.\n'
        'Cornforth patient-balanced group contrast: 7 CF donors vs 5 non-CF donors.\n'
        'Two burn specimens grouped as one patient.\n'
        'Cornforth CF/non-CF is confounded by tissue and geography, and lacks matched cultures.\n'
        'The original datasets informed prior analyses; this is not untouched confirmatory validation.\n'
        'PAO1-only mapping of Cornforth reads remains a technical limitation.\n'
        'A projection sign or cosine does NOT validate CF specificity or physiological-state inference.\n'
    )
    status = 'CFSTATE_PHASE6_06L_FROZEN_V2_CLINICAL_TRANSFER_DESCRIPTIVE_COMPLETE'
    (OUT / 'AUDIT_06L_STATUS.txt').write_text(status + '\n')
    print('\nSTATUS:', status)


if __name__ == '__main__':
    main()
