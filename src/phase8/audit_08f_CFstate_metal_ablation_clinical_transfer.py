#!/usr/bin/env python3
"""CF-State Audit 08f: does clinical CF-associated enrichment survive metal-axis ablation?

This is a POST-HOC, DESCRIPTIVE cross-dataset diagnostic, not fresh validation.
It uses the original frozen 108-coordinate CF consensus and original clinical
projections, plus the already completed 17-pair Neff 2024 metal perturbation.
No model fitting, no re-normalisation of an ablated consensus, no new data.

Ablations evaluated:
  ALL_108: frozen original consensus.
  WITHOUT_ZUR2: zero only the Zur-2 coefficient.
  WITHOUT_ZUR2_PVDS_CZCR: zero Zur-2, PvdS, CzcR coefficients.

Crucially, ablation is of the SIGNED READOUT, not a re-projection or a
re-residualisation. Positive and negative programme effects may cancel.

Run from the CFstate project root, with the completed 05j2, 07i and 08e2
results and the original src/phase7/audit_07i*.py present.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P4 = ROOT / 'results/phase4'
P5 = ROOT / 'results/phase5'
P7 = ROOT / 'results/phase7'
P8 = ROOT / 'results/phase8'
A08E = P8 / 'audit_08e2_Neff2024_paired_CF_sputum_metal_supplementation'
A08F = P8 / 'audit_08f_frozen_metal_axis_ablation_clinical_transfer'
A07I = P7 / 'audit_07i_shared_CFBE_reference_negative_controls'
PRIOR_SCRIPT = ROOT / 'src/phase7/audit_07i_CFstate_shared_reference_negative_controls.py'
EXPECTED_08E = 'CFSTATE_PHASE8_08E2_17_DONOR_COMBINED_METALS_DESCRIPTIVE_COMPLETE_REVIEW_REQUIRED'
EXPECTED_07I = 'CFSTATE_PHASE7_07I_SHARED_REFERENCE_NEGATIVE_CONTROLS_COMPLETE_REVIEW_REQUIRED'
STATUS = 'CFSTATE_PHASE8_08F_METAL_ABLATION_CLINICAL_TRANSFER_DESCRIPTIVE_COMPLETE_REVIEW_REQUIRED'


def require(label, value):
    if not bool(value):
        raise RuntimeError('CHECK FAILED: ' + label)
    print(label + ' PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 1e-12 else np.nan


def build_readouts(stable, consensus):
    require('consensus_norm_one', np.isclose(np.linalg.norm(consensus), 1.0, atol=1e-6))
    out = {}
    for label, drop in [
        ('ALL_108', []),
        ('WITHOUT_ZUR2', ['Zur-2']),
        ('WITHOUT_ZUR2_PVDS_CZCR', ['Zur-2', 'PvdS', 'CzcR']),
    ]:
        require(label + '_programmes_known', set(drop).issubset(set(stable)))
        v = np.array(consensus, dtype=float, copy=True)
        for name in drop:
            v[stable.index(name)] = 0.0
        require(label + '_nonzero_readout', np.linalg.norm(v) > 0.1)
        out[label] = v
    return out


def self_test():
    names = ['Zur-2', 'PvdS', 'CzcR', 'Other']
    weights = np.array([.3, .4, -.2, .5])
    weights /= np.linalg.norm(weights)
    ab = build_readouts(names, weights)
    change = np.array([-2., -3., 7., 4.])
    base = change @ weights
    assert np.isclose(base - change @ ab['WITHOUT_ZUR2'], weights[0] * change[0])
    assert np.isclose(base - change @ ab['WITHOUT_ZUR2_PVDS_CZCR'],
                      weights[0] * change[0] + weights[1] * change[1] + weights[2] * change[2])
    assert np.isclose(np.linalg.norm(ab['ALL_108']), 1)
    assert not np.isclose(np.linalg.norm(ab['WITHOUT_ZUR2']), 1)
    print('AUDIT_08F_SELF_TEST_PASS')


def main():
    print('=' * 84)
    print('CF-State Phase 8 — Audit 08f: frozen metal-axis readout ablation')
    print('Original clinical donor comparisons + Neff 17-donor intervention')
    print('POST-HOC DESCRIPTIVE ONLY — do not claim CF specificity')
    print('=' * 84, flush=True)
    require('CFstate_repository_root', P4.is_dir())
    require('new_output_directory', not A08F.exists())
    require('original_07i_script_available', PRIOR_SCRIPT.is_file())
    require('08e2_status', (A08E / 'AUDIT_08E2_STATUS.txt').read_text().strip() == EXPECTED_08E)
    require('07i_status', (A07I / 'AUDIT_07I_STATUS.txt').read_text().strip() == EXPECTED_07I)

    # The exact previously run 07i code is re-used as a read-only helper.
    sys.path.insert(0, str(PRIOR_SCRIPT.parent))
    import audit_07i_CFstate_shared_reference_negative_controls as previous
    require('07i_source_points_to_same_project', previous.ROOT == ROOT)
    for label, path in previous.FILES.items():
        require('input_07i_' + label, path.is_file())

    saf = pd.read_csv(previous.FILES['SAF'], sep='\t').set_index('GeneID')
    require('SAF_unambiguous', len(saf) == 5567 and saf.index.is_unique)
    lengths = (pd.to_numeric(saf['End']) - pd.to_numeric(saf['Start']) + 1).astype(float)
    mask = pd.read_csv(previous.FILES['mask'])['GeneID'].astype(str).tolist()
    require('frozen_gene_mask', len(mask) == 4923 and len(set(mask)) == 4923 and set(mask).issubset(saf.index))
    cal = pd.read_csv(previous.FILES['calibration']).set_index('imodulon')
    names = cal.index.astype(str).tolist()
    stable = pd.read_csv(previous.FILES['stable'])['imodulon'].astype(str).tolist()
    slopes = pd.to_numeric(cal.loc[names, 'calibration_slope']).to_numpy(float)
    require('frozen_116_108', len(names) == 116 and len(stable) == 108)
    M = previous.get_modulome(previous.FILES['modulome'], mask, names)
    require('full_rank_model', np.linalg.matrix_rank(M) == 116)
    pinv = np.linalg.pinv(M)
    states = pd.read_csv(previous.FILES['states'], index_col=0)
    states.index = states.index.astype(str).str.strip()
    B = states.loc[stable].apply(pd.to_numeric, errors='raise').to_numpy(float, copy=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    require('11_control_directions', B.shape == (108, 11) and np.linalg.matrix_rank(B) == 11)
    cand = pd.read_csv(previous.FILES['candidate'], index_col=0)
    cand.index = cand.index.astype(str).str.strip()
    c = cand.loc[stable, 'two_cohort_consensus_unit'].to_numpy(float)
    readouts = build_readouts(stable, c)

    def project(x):
        a = pd.Series((pinv @ np.asarray(x, dtype=float)) * slopes, index=names)
        s = a.loc[stable].to_numpy(float)
        r = s - B @ np.linalg.lstsq(B, s, rcond=None)[0]
        require_control = np.max(np.abs(B.T @ r)) < 1e-5
        if not require_control:
            raise RuntimeError('Controlled programme residual is not orthogonal')
        return r

    counts = pd.read_csv(previous.FILES['05h_counts'], index_col=0)
    counts.index = counts.index.astype(str)
    manifest = pd.read_csv(previous.FILES['manifest'], dtype=str).fillna('')
    require('Cornforth_13_5567', counts.shape == (5567, 13) and set(counts.columns) == set(manifest.published_sample))
    counts = counts.reindex(index=saf.index)
    for run, (patient, time, folder) in previous.RUNS.items():
        s = previous.get_new_counts(run, folder)
        counts[run] = s.reindex(saf.index).to_numpy(float)
    require('clinical_17_expression_profiles', counts.shape == (5567, 17) and
            np.isfinite(counts.to_numpy(float)).all())
    rates = counts.div(lengths, axis=0)
    denominators = rates.sum(axis=0)
    require('TPM_denominators_positive', (denominators > 0).all())
    expr = np.log2(1 + rates.div(denominators, axis=1) * 1e6).loc[mask]
    ref = pd.read_csv(previous.FILES['CFBE']).set_index('GeneID').loc[mask, 'PRIMARY_CFBE_ALL4'].to_numpy(float)

    donor_rows = []
    donor_expr = {}
    for donor, group in manifest.groupby('published_donor_group', sort=False):
        cohort = 'CORNFORTH_CF' if (group.group == 'CF_sputum').all() else 'CORNFORTH_NONCF'
        require('donor_cohort_consistent_' + donor, group.group.nunique() == 1)
        donor_expr[donor] = (cohort, expr[group.published_sample.tolist()].mean(axis=1).to_numpy(float))
    for patient in ('CF01', 'CF146'):
        runs = [run for run, (p, _, _) in previous.RUNS.items() if p == patient]
        require('two_samples_' + patient, len(runs) == 2)
        donor_expr[patient] = ('NEW_CF', expr[runs].mean(axis=1).to_numpy(float))
    require('independent_donor_groups_14', len(donor_expr) == 14)

    for donor, (cohort, expression) in donor_expr.items():
        vec = project(expression - ref)
        for label, weights in readouts.items():
            donor_rows.append({'cohort': cohort, 'donor': donor, 'readout': label,
                               'signed_projection_vs_shared_CFBE': float(vec @ weights),
                               'cosine_vs_shared_CFBE': cosine(vec, weights),
                               'residual_norm': float(np.linalg.norm(vec))})
    donor_df = pd.DataFrame(donor_rows)
    primary_old = pd.read_csv(A07I / 'DESCRIPTIVE_14_donors_shared_CFBE_consensus.csv')
    primary_old = primary_old.loc[primary_old.reference == 'PRIMARY_CFBE_ALL4']
    check = donor_df.loc[donor_df.readout == 'ALL_108'].merge(
        primary_old[['cohort', 'donor', 'signed_consensus_projection', 'cosine_frozen_consensus']],
        on=['cohort', 'donor'], how='inner', validate='one_to_one')
    require('reproduces_original_07i_all_14_donors', len(check) == 14 and
            np.allclose(check.signed_projection_vs_shared_CFBE, check.signed_consensus_projection, atol=1e-5) and
            np.allclose(check.cosine_vs_shared_CFBE, check.cosine_frozen_consensus, atol=1e-5))

    def mean_for(cohort):
        vals = [v for _, (group, v) in donor_expr.items() if group == cohort]
        return np.array(vals).mean(axis=0)

    cf = mean_for('CORNFORTH_CF')
    noncf = mean_for('CORNFORTH_NONCF')
    new = mean_for('NEW_CF')
    require('cohort_independence_7_5_2',
            sum(v[0] == 'CORNFORTH_CF' for v in donor_expr.values()) == 7 and
            sum(v[0] == 'CORNFORTH_NONCF' for v in donor_expr.values()) == 5 and
            sum(v[0] == 'NEW_CF' for v in donor_expr.values()) == 2)
    contrast_05j2 = pd.read_csv(previous.FILES['05j2_contrast']).set_index('imodulon').loc[stable]
    vec05j2 = contrast_05j2['CF_minus_nonCF_residual'].to_numpy(float)
    require('reproduce_frozen_05j2_CF_minus_nonCF_vector',
            np.allclose(project(cf - noncf), vec05j2, atol=1e-6))
    original07i = pd.read_csv(A07I / 'DESCRIPTIVE_reference_invariant_group_contrasts.csv')
    contrast_vectors = {
        'Cornforth_CF_minus_nonCF_7v5': project(cf - noncf),
        'New_CF_minus_Cornforth_nonCF_2v5': project(new - noncf),
        'New_CF_minus_Cornforth_CF_2v7': project(new - cf),
    }
    contrast_rows, contribution_rows = [], []
    for name, vec in contrast_vectors.items():
        for label, weights in readouts.items():
            contrast_rows.append({
                'contrast': name, 'readout': label,
                'signed_projection': float(vec @ weights),
                'cosine_with_readout': cosine(vec, weights),
                'unablated_residual_l2_norm': float(np.linalg.norm(vec)),
                'readout_l2_norm': float(np.linalg.norm(weights)),
            })
        for g in ['Zur-2', 'PvdS', 'CzcR']:
            ix = stable.index(g)
            contribution_rows.append({'contrast': name, 'imodulon': g,
                                      'frozen_consensus_weight': float(c[ix]),
                                      'controlled_programme_difference': float(vec[ix]),
                                      'signed_projection_contribution': float(vec[ix] * c[ix])})
    contrast_df = pd.DataFrame(contrast_rows)
    for _, old in original07i.iterrows():
        row = contrast_df.loc[(contrast_df.contrast == old['contrast']) & (contrast_df.readout == 'ALL_108')]
        require('reproduce_07i_' + str(old['contrast']), len(row) == 1 and
                np.isclose(row.iloc[0].signed_projection, old['signed_projection'], atol=1e-5) and
                np.isclose(row.iloc[0].cosine_with_readout, old['cosine_frozen_consensus'], atol=1e-5))

    # Numerical bridge to the earlier intervention: read-only output from 08e2.
    prog = pd.read_csv(A08E / 'DESCRIPTIVE_108_programme_mean_paired_changes.csv').set_index('imodulon')
    effect = pd.read_csv(A08E / 'DESCRIPTIVE_paired_bootstrap_and_exact_signflip.csv').set_index('metric')
    require('08e2_same_108_programmes', set(prog.index) == set(stable) and len(prog) == 108)
    require('08e2_same_frozen_weights',
            np.allclose(prog.loc[stable, 'frozen_CF_weight'].to_numpy(float), c, atol=1e-6))
    neff_deltas = prog.loc[stable, 'mean_controlled_change'].to_numpy(float)
    neff_rows = [{'readout': label, 'mean_metals_minus_untreated_signed_change': float(neff_deltas @ w),
                  'comparison': '17 paired PAO1 spike-ins; combined Zn+Fe+Mn'}
                 for label, w in readouts.items()]
    neff_df = pd.DataFrame(neff_rows)
    require('reproduce_08e2_full_signed_mean', np.isclose(
        neff_df.loc[neff_df.readout == 'ALL_108', 'mean_metals_minus_untreated_signed_change'].iloc[0],
        effect.loc['FROZEN_CF_SIGNED_SCORE_METALS_MINUS_UNTREATED', 'mean_change'], atol=1e-5))
    require('reproduce_08e2_without_Zur2_signed_mean', np.isclose(
        neff_df.loc[neff_df.readout == 'WITHOUT_ZUR2', 'mean_metals_minus_untreated_signed_change'].iloc[0],
        effect.loc['FROZEN_CF_SCORE_WITHOUT_ZUR2_CONTRIBUTION', 'mean_change'], atol=1e-5))

    summary = donor_df.groupby(['readout', 'cohort'], sort=False).agg(
        n_donors=('donor', 'nunique'),
        n_positive=('signed_projection_vs_shared_CFBE', lambda s: int((s > 0).sum())),
        median_signed_projection=('signed_projection_vs_shared_CFBE', 'median'),
        median_cosine=('cosine_vs_shared_CFBE', 'median'),
    ).reset_index()

    A08F.mkdir(parents=True, exist_ok=False)
    (A08F / 'AUDIT_08F_STATUS.txt').write_text('INCOMPLETE_PENDING_EXPORT\n')
    donor_df.to_csv(A08F / 'DESCRIPTIVE_14_donor_ablation_scores.csv', index=False)
    contrast_df.to_csv(A08F / 'DESCRIPTIVE_reference_invariant_clinical_ablation_contrasts.csv', index=False)
    pd.DataFrame(contribution_rows).to_csv(A08F / 'DESCRIPTIVE_metal_programme_clinical_contrast_contributions.csv', index=False)
    neff_df.to_csv(A08F / 'DESCRIPTIVE_17_donor_metal_response_ablation_means.csv', index=False)
    summary.to_csv(A08F / 'DESCRIPTIVE_donor_cohort_ablation_summary.csv', index=False)
    pd.DataFrame([{'path': str(p), 'sha256': sha256(p)} for p in [
        PRIOR_SCRIPT, previous.FILES['candidate'], previous.FILES['05j2_contrast'],
        A07I / 'DESCRIPTIVE_14_donors_shared_CFBE_consensus.csv',
        A08E / 'DESCRIPTIVE_108_programme_mean_paired_changes.csv',
        Path(__file__).resolve()]]).to_csv(A08F / 'input_sha256.csv', index=False)
    (A08F / 'DESIGN_LIMITATIONS.txt').write_text(
        'POST-HOC EXPLORATORY. Original clinical comparison has CF vs wound/burn confounding.\n'
        'Same shared CFBE reference for donor positivity. Reference cancels for group differences.\n'
        'Two new CF patients are longitudinally sampled; patient-level means used.\n'
        'Ablations zero selected readout coefficients with NO renormalisation or retraining.\n'
        'Neither the ablations nor the 17-donor ex vivo PAO1 experiment establish CF specificity.\n'
        'The three metal programmes were chosen in light of observed perturbation results, so\n'
        'the three-programme analysis is post-hoc and must not be presented as pre-registered.\n'
        'Cross-experiment numbers are relative *within each experiment*, not comparable in absolute\n'
        'units: clinical uses log2TPM; metal perturbation uses normalised log2(1+Salmon counts).\n'
        'Do not infer free Zn/Fe/Mn levels or clinical severity from any transcriptome projection.\n'
    )
    print('\n========== CLINICAL CF VS NONCF: FROZEN READOUT ABLATIONS ==========', flush=True)
    print(contrast_df.round(5).to_string(index=False), flush=True)
    print('\n========== CLINICAL DONOR GROUPS AGAINST SHARED CFBE ==========', flush=True)
    print(summary.round(5).to_string(index=False), flush=True)
    print('\n========== METAL PROGRAMMES: CONTRIBUTIONS TO CLINICAL CONTRASTS ==========', flush=True)
    print(pd.DataFrame(contribution_rows).round(5).to_string(index=False), flush=True)
    print('\n========== NEFF 17-DONOR RESPONSE: MATCHED FROZEN READOUTS ==========', flush=True)
    print(neff_df.round(5).to_string(index=False), flush=True)
    (A08F / 'AUDIT_08F_STATUS.txt').write_text(STATUS + '\n')
    print('\nSTATUS:', STATUS, flush=True)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    else:
        main()
