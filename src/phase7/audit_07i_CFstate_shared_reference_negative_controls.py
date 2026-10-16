#!/usr/bin/env python3
"""CF-State 07i: reference-matched NON-CF negative-control challenge.

DESCRIPTIVE, post-hoc: no fitting or phenotype-specific thresholds. Calculate
frozen 03h residual-consensus alignment for Cornforth 2018 CF/non-CF donors and
new CF01/CF146 specimens using the SAME external CFBE reference. This asks
whether positivity against CFBE is equally common among non-CF clinical RNA.
It cannot isolate CF from tissue, geography, platform or strain effects.

Needs completed frozen Audits 05h, 05j2, 07c, 07g, and 07h.
Run from the CFstate project root.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P4 = ROOT / 'results/phase4'
P5 = ROOT / 'results/phase5'
P7 = ROOT / 'results/phase7'
A7H = P7 / 'audit_07h_frozen_CF_signature_two_new_patients'
A7C = P7 / 'audit_07c_SRP173673_D7_D8_PAO1_feasibility'
A7G = P7 / 'audit_07g_GSE139943_CF146_D33_D35_PAO1_feasibility'
A5H = P5 / 'audit_05h_Cornforth2018_full_depth_baseline'
A5J2 = P5 / 'audit_05j2_Cornforth2018_provisional_CF_contrast'
A4K = P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection'
OUT = P7 / 'audit_07i_shared_CFBE_reference_negative_controls'

FILES = {
    '07h_status': A7H / 'AUDIT_07H_STATUS.txt',
    '07h_results': A7H / 'DESCRIPTIVE_new_CF_patients_reference_sensitivity.csv',
    '05h_status': A5H / 'AUDIT_05H_STATUS.txt',
    '05h_counts': A5H / 'FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv',
    '05j2_status': A5J2 / 'AUDIT_05J2_STATUS.txt',
    '05j2_contrast': A5J2 / 'DESCRIPTIVE_108_programme_contrast__FROZEN_5567_TPM.csv',
    'manifest': P5 / 'audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv',
    'SAF': ROOT / 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf',
    'mask': A4K / 'FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'states': P4 / 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'CFBE': A4K / 'FROZEN_CFBE_softcore_4923_log2TPM_references.csv',
    'script': Path(__file__).resolve(),
}
RUNS = {
    'SRR8334098': ('CF01', 'D-8', A7C),
    'SRR8334097': ('CF01', 'D-7', A7C),
    'SRR10401171': ('CF146', 'day33', A7G),
    'SRR10401172': ('CF146', 'day35', A7G),
}
REFS = ('PRIMARY_CFBE_ALL4', 'SENSITIVITY_CFBE_C_PAIR', 'SENSITIVITY_CFBE_R_PAIR')
STATUS = 'CFSTATE_PHASE7_07I_SHARED_REFERENCE_NEGATIVE_CONTROLS_COMPLETE_REVIEW_REQUIRED'


def ensure(label, test):
    if not bool(test):
        raise RuntimeError('FAILED ' + label)
    print(label + ' PASS', flush=True)


def get_modulome(path, mask, names):
    raw = json.loads(path.read_text())['M']
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and {'index', 'columns', 'data'}.issubset(raw):
        frame = pd.DataFrame(raw['data'], index=raw['index'], columns=raw['columns'])
    else:
        frame = pd.DataFrame(raw)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if set(mask).issubset(frame.columns) and set(names).issubset(frame.index):
        frame = frame.T
    ensure('modulome_IDs', set(mask).issubset(frame.index) and set(names).issubset(frame.columns))
    return frame.loc[mask, names].apply(pd.to_numeric, errors='raise').to_numpy(float)


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        raise RuntimeError('zero vector in cosine')
    return float(a @ b / denom)


def get_new_counts(run, folder):
    if folder == A7G:
        path = folder / run / 'PAO1_5567_assigned_counts.csv'
        frame = pd.read_csv(path, index_col=0)
        ensure('one_new_count_column_' + run, frame.shape[1] == 1)
        s = frame.iloc[:, 0]
    else:
        path = folder / run / 'PAO1_featureCounts.txt'
        frame = pd.read_csv(path, sep='\t', comment='#')
        ensure('new_featureCounts_Geneid_' + run, 'Geneid' in frame.columns)
        s = pd.Series(frame.iloc[:, -1].to_numpy(), index=frame['Geneid'].astype(str))
    s.index = s.index.astype(str)
    s = pd.to_numeric(s, errors='raise')
    ensure('new_count_5567_' + run, len(s) == 5567 and s.index.is_unique)
    ensure('new_count_finite_' + run, np.isfinite(s.to_numpy(float)).all() and (s >= 0).all())
    return s


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda: f.read(1048576), b''):
            h.update(x)
    return h.hexdigest()


def main():
    print('=' * 77)
    print('CF-State 07i: shared CFBE reference NEGATIVE-CONTROL comparison')
    print('=' * 77, flush=True)
    ensure('new_output_directory', not OUT.exists())
    for name, path in FILES.items():
        ensure('input_' + name, path.is_file())
    ensure('07h_frozen', FILES['07h_status'].read_text().strip() ==
           'CFSTATE_PHASE7_07H_FROZEN_03H_NEW_PATIENTS_DESCRIPTIVE_PROJECTION_COMPLETE_REVIEW_REQUIRED')
    ensure('05h_frozen', FILES['05h_status'].read_text().strip() ==
           'CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN')
    ensure('05j2_frozen', FILES['05j2_status'].read_text().strip() ==
           'CFSTATE_PHASE5_05J2_PROVISIONAL_CF_NONCF_PROJECTION_COMPLETE')

    saf = pd.read_csv(FILES['SAF'], sep='\t').set_index('GeneID')
    ensure('SAF_5567', len(saf) == 5567 and saf.index.is_unique)
    lengths = (pd.to_numeric(saf['End']) - pd.to_numeric(saf['Start']) + 1).astype(float)
    ensure('gene_lengths', np.isfinite(lengths.to_numpy()).all() and (lengths > 0).all())
    mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
    ensure('mask_4923', len(mask) == 4923 and len(set(mask)) == 4923 and set(mask).issubset(saf.index))

    cal = pd.read_csv(FILES['calibration']).set_index('imodulon')
    names = cal.index.astype(str).tolist()
    stable = pd.read_csv(FILES['stable'])['imodulon'].astype(str).tolist()
    ensure('116_108', len(names) == 116 and len(stable) == 108 and set(stable).issubset(names))
    slopes = pd.to_numeric(cal.loc[names, 'calibration_slope']).to_numpy(float)
    M = get_modulome(FILES['modulome'], mask, names)
    pinv = np.linalg.pinv(M)
    ensure('M_rank', np.linalg.matrix_rank(M) == 116)
    state = pd.read_csv(FILES['states'], index_col=0)
    state.index = state.index.astype(str).str.strip()
    B = state.loc[stable].apply(pd.to_numeric, errors='raise').to_numpy(float, copy=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    ensure('11_states', B.shape == (108, 11) and np.linalg.matrix_rank(B) == 11)
    candidate_df = pd.read_csv(FILES['candidate'], index_col=0)
    candidate_df.index = candidate_df.index.astype(str).str.strip()
    c = candidate_df.loc[stable, 'two_cohort_consensus_unit'].to_numpy(float)
    ensure('consensus_unit', np.isclose(np.linalg.norm(c), 1.0, atol=1e-6))
    cfbe = pd.read_csv(FILES['CFBE']).set_index('GeneID').loc[mask, list(REFS)]
    ensure('CFBE_finite', np.isfinite(cfbe.to_numpy(float)).all())

    def project(d):
        raw = (pinv @ d) * slopes
        a = pd.Series(raw, index=names).loc[stable].to_numpy(float)
        return a - B @ np.linalg.lstsq(B, a, rcond=None)[0]

    # Unify the counting/TPM procedure for Cornforth and both new patients.
    counts = pd.read_csv(FILES['05h_counts'], index_col=0)
    counts.index = counts.index.astype(str)
    manifest = pd.read_csv(FILES['manifest'], dtype=str).fillna('')
    ensure('Cornforth_13', counts.shape == (5567, 13) and set(counts.columns) == set(manifest.published_sample))
    ensure('Cornforth_donor_groups_12', manifest.published_donor_group.nunique() == 12)
    counts = counts.reindex(index=saf.index)
    for run, (patient, time, folder) in RUNS.items():
        s = get_new_counts(run, folder)
        ensure('new_gene_ids_' + run, set(s.index) == set(saf.index))
        counts[run] = s.reindex(saf.index).to_numpy(float)
    ensure('combined_5567_by_17', counts.shape == (5567, 17) and np.isfinite(counts.to_numpy(float)).all())
    rates = counts.div(lengths, axis=0)
    denom = rates.sum(axis=0)
    ensure('TPM_denom_positive', (denom > 0).all())
    expr = np.log2(1.0 + rates.div(denom, axis=1) * 1e6).loc[mask]

    prior07h = pd.read_csv(FILES['07h_results'])
    records = []
    # New CF patient specimens and controls share precisely the same CFBE offset.
    for ref in REFS:
        r0 = cfbe[ref].to_numpy(float)
        for col in counts.columns:
            x = expr[col].to_numpy(float)
            residual = project(x - r0)
            if col in RUNS:
                patient, time, _ = RUNS[col]
                cohort = 'NEW_CF'
                donor = patient
                sample = time
            else:
                row = manifest.loc[manifest.published_sample == col].iloc[0]
                cohort = 'CORNFORTH_CF' if row.group == 'CF_sputum' else 'CORNFORTH_NONCF'
                donor = row.published_donor_group
                sample = col
            records.append(dict(
                reference=ref, cohort=cohort, donor=donor, specimen=sample, run_or_sample=col,
                cosine_frozen_consensus=cosine(residual, c),
                signed_consensus_projection=float(residual @ c),
                residual_norm=float(np.linalg.norm(residual)),
            ))
    specimens = pd.DataFrame(records)
    # Exact independent reproduction of previous 07h four-sample outputs.
    for run, (patient, time, _) in RUNS.items():
        for ref in REFS:
            old = prior07h.loc[(prior07h.run == run) & (prior07h.reference == ref)]
            new = specimens.loc[(specimens.run_or_sample == run) & (specimens.reference == ref)]
            ensure('reproduce_07h_' + run + '_' + ref,
                   len(old) == 1 and len(new) == 1 and
                   np.isclose(old.cosine_frozen_consensus.iloc[0], new.cosine_frozen_consensus.iloc[0], atol=1e-5) and
                   np.isclose(old.signed_consensus_projection.iloc[0], new.signed_consensus_projection.iloc[0], atol=1e-5))

    # Each donor contributes once: burn pair is averaged, as are the new timepoints.
    # For cosine, average residual vectors before calculating the cosine; computing
    # cosine of the average specimen cosine would NOT be equivalent.
    records = []
    for ref in REFS:
        r0 = cfbe[ref].to_numpy(float)
        donor_groups = []
        for donor, group in manifest.groupby('published_donor_group', sort=False):
            group_class = ('CORNFORTH_CF' if (group.group == 'CF_sputum').all()
                           else 'CORNFORTH_NONCF')
            ensure('donor_type_consistent_' + donor, group.group.nunique() == 1)
            donor_groups.append((group_class, donor, group.published_sample.tolist()))
        for patient in ('CF01', 'CF146'):
            patient_runs = [run for run, desc in RUNS.items() if desc[0] == patient]
            donor_groups.append(('NEW_CF', patient, patient_runs))
        ensure('14_donor_groups', len(donor_groups) == 14)
        for cohort, donor, sample_cols in donor_groups:
            donor_expr = expr[sample_cols].mean(axis=1).to_numpy(float)
            residual = project(donor_expr - r0)
            records.append(dict(reference=ref, cohort=cohort, donor=donor,
                                n_specimens=len(sample_cols),
                                signed_consensus_projection=float(residual @ c),
                                cosine_frozen_consensus=cosine(residual, c),
                                residual_norm=float(np.linalg.norm(residual))))
    donor_df = pd.DataFrame(records)

    # Reproduce locked Cornforth CF/non-CF *reference invariant difference*.
    contrast = pd.read_csv(FILES['05j2_contrast']).set_index('imodulon')
    contrast = contrast.loc[stable]
    expected_delta = contrast['CF_minus_nonCF_residual'].to_numpy(float)
    ensure('05j2_contrast_consensus', np.allclose(contrast['frozen_consensus'].to_numpy(float), c))
    primary_donor = donor_df.loc[donor_df.reference == REFS[0]]
    cf_donors = manifest.loc[manifest.group == 'CF_sputum', 'published_donor_group'].drop_duplicates().tolist()
    other_donors = manifest.loc[manifest.group != 'CF_sputum', 'published_donor_group'].drop_duplicates().tolist()
    donor_to_samples = {d: group.published_sample.tolist() for d, group in manifest.groupby('published_donor_group')}
    cf_expr = pd.concat([expr[donor_to_samples[d]].mean(axis=1) for d in cf_donors], axis=1).mean(axis=1)
    noncf_expr = pd.concat([expr[donor_to_samples[d]].mean(axis=1) for d in other_donors], axis=1).mean(axis=1)
    new_expr = pd.concat([
        expr[[run for run, desc in RUNS.items() if desc[0] == patient]].mean(axis=1)
        for patient in ('CF01', 'CF146')], axis=1).mean(axis=1)
    delta = project((cf_expr - noncf_expr).to_numpy(float))
    ensure('reproduce_05j2_vector', np.allclose(delta, expected_delta, atol=1e-6))

    contrasts = []
    for name, lhs, rhs in (
        ('Cornforth_CF_minus_nonCF_7v5', cf_expr, noncf_expr),
        ('New_CF_minus_Cornforth_nonCF_2v5', new_expr, noncf_expr),
        ('New_CF_minus_Cornforth_CF_2v7', new_expr, cf_expr),
    ):
        vec = project((lhs - rhs).to_numpy(float))
        contrasts.append(dict(contrast=name, signed_projection=float(vec @ c),
                              cosine_frozen_consensus=cosine(vec, c),
                              residual_l2_norm=float(np.linalg.norm(vec))))
    contrast_df = pd.DataFrame(contrasts)

    print('\n========== ALL CLINICAL GROUPS AGAINST IDENTICAL CFBE REFERENCE ==========', flush=True)
    tab = primary_donor.groupby('cohort').agg(
        independent_donors=('donor', 'nunique'),
        positive=('signed_consensus_projection', lambda s: int((s > 0).sum())),
        median_signed_projection=('signed_consensus_projection', 'median'),
        median_cosine=('cosine_frozen_consensus', 'median'),
        min_cosine=('cosine_frozen_consensus', 'min'),
        max_cosine=('cosine_frozen_consensus', 'max'),
    )
    print(tab.round(5).to_string())
    print('\n========== PATIENT-BALANCED REFERENCE-INVARIANT CONTRASTS ==========', flush=True)
    print(contrast_df.round(5).to_string(index=False))
    print('\n========== REFERENCE-SENSITIVITY OF POSITIVE DONORS ==========', flush=True)
    counts_positive = donor_df.groupby(['reference', 'cohort']).agg(
        n_donors=('donor', 'nunique'),
        n_positive=('signed_consensus_projection', lambda s: int((s > 0).sum())),
        median_cosine=('cosine_frozen_consensus', 'median'),
    )
    print(counts_positive.round(5).to_string())

    OUT.mkdir(parents=True, exist_ok=False)
    specimens.to_csv(OUT / 'DESCRIPTIVE_17_specimens_shared_CFBE_consensus.csv', index=False)
    donor_df.to_csv(OUT / 'DESCRIPTIVE_14_donors_shared_CFBE_consensus.csv', index=False)
    contrast_df.to_csv(OUT / 'DESCRIPTIVE_reference_invariant_group_contrasts.csv', index=False)
    pd.DataFrame([{'input': name, 'path': str(path), 'sha256': sha(path)}
                  for name, path in FILES.items()]).to_csv(OUT / 'input_sha256.csv', index=False)
    (OUT / 'DESIGN_LIMITATIONS.txt').write_text(
        'EXPLORATORY POST-HOC NEGATIVE CONTROL. No fitting or refitting.\n'
        'Two new independent CF donors, each with two longitudinal samples.\n'
        'Cornforth 7 CF / 5 nonCF donors, repeated burn averaged.\n'
        'Same CFBE external reference, same gene universe / projection for all groups.\n'
        'Absolute positive alignment against CFBE is susceptible to shared-reference effects.\n'
        'Group-difference residuals are reference-invariant, but tissue, country, study and batch confounded.\n'
        'PAO1-only mapping and bacterial identity / gene coverage are unresolved limitations.\n'
        'All cohorts were previously explored; not independent confirmatory testing.\n'
        'No p-values; no new parameter selection; do not infer CF-specific physiology.\n'
    )
    (OUT / 'AUDIT_07I_STATUS.txt').write_text(STATUS + '\n')
    print('\nSTATUS:', STATUS)


if __name__ == '__main__':
    main()
