#!/usr/bin/env python3
"""CF-State Phase 8 Audit 08k: joint perturbation-anchored decomposition.

Read-only integration of two already completed experiments:
  (1) Neff et al. 17-donor Zn+Fe+Mn vs untreated sputum, Audit 08g.
  (2) Hammond et al. PAO1/J215 WT vs Δanr under 1% O2, Audit 08j.

DO NOT label the Anr knockout axis as an oxygen-concentration calibration.
Orthogonal residual is relative to these TWO PARTICULAR intervention directions,
not a proof of CF specificity or of non-metal/non-Anr biology. Comparisons share
Cornforth wound controls and are confounded by tissue/strain/study differences.
The 108 programme coordinates, weights and model parameters are never refitted.

Run from CFstate root:
    python src/phase8/audit_08k_two_experimental_axes_clinical_decomposition.py --self-test
    python -u src/phase8/audit_08k_two_experimental_axes_clinical_decomposition.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P8 = ROOT / 'results/phase8'
A08G = P8 / 'audit_08g_perturbation_anchored_clinical_decomposition'
A08J = P8 / 'audit_08j_GSE68534_frozen_Anr_regulator_response'
OUT = P8 / 'audit_08k_two_experimental_axes_clinical_decomposition'
STATUS = 'CFSTATE_PHASE8_08K_TWO_EXPERIMENTAL_AXES_EXPLORATORY_COMPLETE'

INPUTS = {
    'metal_axis': A08G / 'FROZEN_derived_108_programme_metal_axis.csv',
    'clinical_108': A08G / 'DESCRIPTIVE_324_programme_contributions_to_clinical_decomposition.csv',
    'previous_clinical': A08G / 'EXPLORATORY_3_clinical_contrast_parallel_orthogonal.csv',
    'previous_status': A08G / 'AUDIT_08G_STATUS.txt',
    'anr_PA01': A08J / 'DESCRIPTIVE_108_programme_changes_PAO1.csv',
    'anr_J215': A08J / 'DESCRIPTIVE_108_programme_changes_J215.csv',
    'anr_summary': A08J / 'DESCRIPTIVE_CF_signature_delta_anr_minus_WT.csv',
    'script': Path(__file__).resolve(),
}
EXPECTED_G = 'CFSTATE_PHASE8_08G_PERTURBATION_ANCHORED_DECOMPOSITION_EXPLORATORY_COMPLETE'
CONTRASTS = [
    'Cornforth_CF_minus_nonCF_7v5',
    'New_CF_minus_Cornforth_nonCF_2v5',
    'New_CF_minus_Cornforth_CF_2v7',
]


def check(name: str, ok: bool):
    if not bool(ok):
        raise RuntimeError('CHECK FAILED: ' + name)
    print(name + ' PASS', flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def unit(v):
    v = np.asarray(v, dtype=float)
    check('unit_direction_finite', v.ndim == 1 and np.isfinite(v).all())
    norm = np.linalg.norm(v)
    check('unit_direction_nonzero', norm > 1e-8)
    return v / norm


def cosine(a, b):
    return float(unit_quiet(a) @ unit_quiet(b))


def unit_quiet(x):
    z = np.asarray(x, float)
    n = np.linalg.norm(z)
    if z.ndim != 1 or n <= 1e-12 or not np.isfinite(z).all():
        raise ValueError('Nonfinite or zero-length vector')
    return z / n


def projections(d, c, m, a):
    """Decompose d over joint 2D span and calculate order-averaged signed shares.

    Both axes are unit vectors. The two Shapley score shares average two orders
    of fitting the 1D axes. Their sum is the JOINT 2D signed score; the shares
    have mathematical, not causal, meaning, especially for correlated axes.
    """
    d, c, m, a = [np.asarray(x, float) for x in (d, c, m, a)]
    U = np.stack((m, a), axis=1)
    # QR makes a stable orthonormal basis for the 2D subspace.
    Q, R = np.linalg.qr(U, mode='reduced')
    if np.min(np.abs(np.diag(R))) < 1e-4:
        raise RuntimeError('Metal and Anr axes are too nearly collinear')
    p_joint = Q @ (Q.T @ d)
    remainder = d - p_joint
    p_m = m * (d @ m)
    p_a = a * (d @ a)
    s_full = float(c @ d)
    s_m = float(c @ p_m)
    s_a = float(c @ p_a)
    s_joint = float(c @ p_joint)
    s_rest = float(c @ remainder)
    metal_shapley = (s_m + s_joint - s_a) / 2
    anr_shapley = (s_a + s_joint - s_m) / 2
    if not (np.allclose(d, p_joint + remainder, atol=1e-10)
            and np.allclose(U.T @ remainder, 0, atol=1e-8)
            and np.isclose(s_full, s_joint + s_rest, atol=1e-8)
            and np.isclose(metal_shapley + anr_shapley, s_joint, atol=1e-8)):
        raise RuntimeError('Two-axis accounting is inconsistent')
    energy = float(p_joint @ p_joint / (d @ d)) if d @ d > 1e-12 else float('nan')
    return {
        'full_score': s_full,
        'metal_axis_alone_score': s_m,
        'Anr_axis_alone_score': s_a,
        'joint_two_axis_score': s_joint,
        'score_orthogonal_to_both_axes': s_rest,
        'Shapley_metal_score_component': metal_shapley,
        'Shapley_Anr_score_component': anr_shapley,
        'fraction_108D_vector_energy_in_joint_span': energy,
        'cosine_clinical_vs_metal_addition': cosine(d, m),
        'cosine_clinical_vs_Anr_intact': cosine(d, a),
        'vector_norm': float(np.linalg.norm(d)),
        'orthogonal_vector_norm': float(np.linalg.norm(remainder)),
        'orthogonal': remainder,
        'parallel': p_joint,
    }


def self_test():
    rng = np.random.default_rng(208)
    d = rng.normal(size=108)
    c = unit_quiet(rng.normal(size=108))
    m = unit_quiet(rng.normal(size=108))
    a = unit_quiet(rng.normal(size=108) + m * 2)
    z = projections(d, c, m, a)
    assert np.isclose(z['full_score'], z['joint_two_axis_score'] + z['score_orthogonal_to_both_axes'])
    assert 0 <= z['fraction_108D_vector_energy_in_joint_span'] <= 1 + 1e-12
    assert np.isclose(z['Shapley_metal_score_component'] + z['Shapley_Anr_score_component'], z['joint_two_axis_score'])
    # Must be invariant to swapping the input axis order (within numerical tolerance).
    reverse = projections(d, c, a, m)
    assert np.isclose(z['joint_two_axis_score'], reverse['joint_two_axis_score'])
    assert np.isclose(z['score_orthogonal_to_both_axes'], reverse['score_orthogonal_to_both_axes'])
    assert np.isclose(z['Shapley_metal_score_component'], reverse['Shapley_Anr_score_component'])
    # A vector in the span of the axes has zero 108D joint residual.
    close = projections(m * 2 + a * 4, c, m, a)
    assert close['orthogonal_vector_norm'] < 1e-10
    # A vector perpendicular to both axes should be completely retained.
    orth = z['orthogonal']
    check('self_test_orthogonal_residual', abs(orth @ m) < 1e-9 and abs(orth @ a) < 1e-9)
    print('AUDIT_08K_SELF_TEST_PASS', flush=True)


def read_unique(path, key, expected_n=108):
    df = pd.read_csv(path)
    check(path.name + '_key_unique', key in df.columns and df[key].is_unique and len(df) == expected_n)
    return df.set_index(key)


def main():
    print('=' * 85)
    print('CF-State Phase 8 — Audit 08k: TWO experimental axes and clinical comparisons')
    print('Metal supplementation vs Anr regulator knockout; FROZEN MODEL; exploratory geometry')
    print('=' * 85, flush=True)
    check('workdir_CFstate', (ROOT / 'results/phase4').is_dir())
    check('new_output_dir', not OUT.exists())
    for name, path in INPUTS.items():
        check('input_' + name, path.is_file())
    check('08g_completed', INPUTS['previous_status'].read_text().strip() == EXPECTED_G)
    metal_df = read_unique(INPUTS['metal_axis'], 'imodulon')
    program_order = metal_df.index.tolist()
    check('frozen_108', len(program_order) == 108)
    m0 = metal_df['unit_mean_metal_direction_108'].to_numpy(float)
    c = metal_df['frozen_CF_signature_unit_108'].to_numpy(float)
    check('metal_direction_unit', abs(np.linalg.norm(m0) - 1) < 1e-6)
    check('candidate_unit', abs(np.linalg.norm(c) - 1) < 1e-6)
    m = unit_quiet(m0)

    raw_anr = {}
    for strain, key in (('PAO1', 'anr_PA01'), ('J215', 'anr_J215')):
        d = read_unique(INPUTS[key], 'imodulon')
        check('Anr_' + strain + '_identical_108_set', set(d.index) == set(program_order))
        check('Anr_' + strain + '_weights_match', np.allclose(
            d.loc[program_order, 'frozen_candidate_weight'].to_numpy(float), c, atol=1e-6))
        raw_anr[strain] = d.loc[program_order, 'controlled_residual_programme_change'].to_numpy(float)
        check('Anr_' + strain + '_finite', np.isfinite(raw_anr[strain]).all())
    summary = pd.read_csv(INPUTS['anr_summary']).set_index('strain')
    check('Anr_summary_two_strains', set(summary.index) == {'PAO1','J215'})
    for strain in ('PAO1','J215'):
        check('Anr_signed_score_matches_08j_' + strain, np.isclose(
            raw_anr[strain] @ c, float(summary.loc[strain, 'signed_delta_CF_candidate']), atol=1e-5))
    concordance = cosine(raw_anr['PAO1'], raw_anr['J215'])
    check('Anr_knockout_strain_concordance_reproduces_08j', np.isclose(concordance, .8118, atol=.00015))

    # Positive Anr axis points from deletion mutant back toward Anr-intact wild type.
    anr_wt = -0.5 * (raw_anr['PAO1'] + raw_anr['J215'])
    a = unit_quiet(anr_wt)
    check('two_axes_noncollinear', abs(m @ a) < .98)
    check('Anr_intact_expected_positive_CF_score', anr_wt @ c > 0)

    clinical = pd.read_csv(INPUTS['clinical_108'])
    prev = pd.read_csv(INPUTS['previous_clinical']).set_index('contrast')
    check('3_clinical_contrast_names', set(clinical['contrast']) == set(CONTRASTS) and set(prev.index) == set(CONTRASTS))
    check('three_times_108', len(clinical) == 324)

    results = []
    gene_contributions = []
    for name in CONTRASTS:
        group = clinical.loc[clinical.contrast == name].set_index('imodulon')
        check('clinical_' + name + '_all_108', group.index.is_unique and set(group.index) == set(program_order))
        check('clinical_' + name + '_frozen_weights', np.allclose(
            group.loc[program_order, 'frozen_CF_signature_weight'].to_numpy(float), c, atol=1e-6))
        d = group.loc[program_order, 'clinical_full_programme_delta'].to_numpy(float)
        z = projections(d, c, m, a)
        check('clinical_' + name + '_full_reproduce_08g', np.isclose(
            z['full_score'], prev.loc[name, 'clinical_full_signed_CF_score'], atol=1e-5))
        check('clinical_' + name + '_metal_reproduce_08g', np.isclose(
            z['metal_axis_alone_score'], prev.loc[name, 'clinical_parallel_signed_CF_score'], atol=1e-5))
        row = {'contrast': name}
        row.update({key: value for key, value in z.items() if not isinstance(value, np.ndarray)})
        results.append(row)
        for j, p in enumerate(program_order):
            gene_contributions.append({
                'contrast': name,
                'imodulon': p,
                'frozen_weight': float(c[j]),
                'clinical_full_programme_delta': float(d[j]),
                'clinical_joint_orthogonal_programme_delta': float(z['orthogonal'][j]),
                'clinical_score_contribution_joint_orthogonal': float(c[j] * z['orthogonal'][j]),
            })

    result = pd.DataFrame(results)
    contrib = pd.DataFrame(gene_contributions)
    for name in CONTRASTS:
        rows = contrib[contrib.contrast == name]
        calc = rows['clinical_score_contribution_joint_orthogonal'].sum()
        expected = result.set_index('contrast').loc[name, 'score_orthogonal_to_both_axes']
        check('orthogonal_contributions_sum_' + name, np.isclose(calc, expected, atol=1e-8))

    # Swap the overall Anr axis for each strain-specific response: sensitivity only.
    sensitivity = []
    for label, direction in [
        ('PAO1_ONLY_Anr_intact', unit_quiet(-raw_anr['PAO1'])),
        ('J215_ONLY_Anr_intact', unit_quiet(-raw_anr['J215'])),
        ('BOTH_STRAINS_Anr_intact', a),
    ]:
        check('sensitivity_' + label + '_noncollinear', abs(m @ direction) < .98)
        for name in CONTRASTS:
            group = clinical[clinical.contrast == name].set_index('imodulon')
            d = group.loc[program_order, 'clinical_full_programme_delta'].to_numpy(float)
            z = projections(d, c, m, direction)
            sensitivity.append({
                'Anr_axis_source': label,
                'contrast': name,
                'cosine_metal_vs_Anr_axis': float(m @ direction),
                'joint_two_axis_score': z['joint_two_axis_score'],
                'score_orthogonal_to_both_axes': z['score_orthogonal_to_both_axes'],
                'fraction_108D_vector_energy_in_joint_span': z['fraction_108D_vector_energy_in_joint_span'],
            })
    axis_data = pd.DataFrame({
        'imodulon': program_order,
        'frozen_CF_weight': c,
        'METALS_minus_UNTREATED_mean_vector_unit': m,
        'Anr_intact_minus_deletion_mean_vector_unit': a,
        'Anr_deletion_minus_WT_PA01_raw_vector': raw_anr['PAO1'],
        'Anr_deletion_minus_WT_J215_raw_vector': raw_anr['J215'],
    })
    axes = pd.DataFrame([{
        'Anr_PA01_vs_J215_knockout_cosine': concordance,
        'METALS_addition_vs_Anr_intact_cosine': float(m @ a),
        'METALS_addition_vs_Anr_deletion_cosine': float(-(m @ a)),
        'CF_signature_cosine_METALS_addition': float(c @ m),
        'CF_signature_cosine_Anr_intact': float(c @ a),
        'CF_signature_fraction_energy_in_two_axis_span': float(
            np.linalg.norm(np.linalg.qr(np.stack([m, a], axis=1), mode='reduced')[0].T @ c) ** 2),
    }])
    check('scores_finite', np.isfinite(result.select_dtypes('number').to_numpy()).all())
    check('one_complete_result_per_clinical_comparison', len(result) == 3)

    OUT.mkdir(parents=True, exist_ok=False)
    axes.to_csv(OUT / 'DESCRIPTIVE_two_axis_geometry_summary.csv', index=False)
    axis_data.to_csv(OUT / 'DESCRIPTIVE_two_frozen_108_programme_intervention_directions.csv', index=False)
    result.to_csv(OUT / 'EXPLORATORY_three_clinical_joint_two_axis_decompositions.csv', index=False)
    contrib.to_csv(OUT / 'DESCRIPTIVE_joint_orthogonal_programme_contributions.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(OUT / 'SENSITIVITY_Anr_strain_specific_axes.csv', index=False)
    provenance = {'audit': '08k', 'scientific_status': STATUS,
                  'inputs': {k: {'path': str(v), 'sha256': sha256(v)} for k,v in INPUTS.items()}}
    (OUT / 'INPUT_SHA256_PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (OUT / 'INTERPRETATION_LIMITS.txt').write_text(
        'NO NEW TRAINING, NO MODEL UPDATE. These are geometrical post-hoc projections.\n'
        'Anr axis is WT MINUS deletion under 1% O2, NOT an oxygen-dose or respiration-rate axis.\n'
        'Metal axis is combined Zn+Fe+Mn addition to PAO1 in CF sputum, not a zinc-only axis.\n'
        'Anr knockout has two replicates per genotype per strain, and two strain backgrounds.\n'
        'CF sputum PAO1 spike-ins are not native patient Pseudomonas expression.\n'
        'Both CF-vs-nonCF contrasts use the SAME nonCF wound/burn donor comparator.\n'
        'Study/tissue/strain/depth differences confound the clinical comparisons.\n'
        'Projection coordinates depend on frozen iModulon calibration and scaling.\n'
        'Shared response of two genetic/experimental axes cannot be partitioned causally.\n'
        'Shapley score allocations are descriptive symmetrised geometry, NOT effects.\n'
        'Remaining 108D signal orthogonal to these directions does not imply nonmetal or non-Anr biology.\n'
        'Aeruprecise training overlap not independently ruled out.\n'
    )
    (OUT / 'AUDIT_08K_STATUS.txt').write_text(STATUS + '\n')

    print('\n========== TWO EXPERIMENTAL AXES: GEOMETRY ==========', flush=True)
    print(axes.round(5).to_string(index=False), flush=True)
    print('\n========== CLINICAL CONTRASTS: TWO-AXIS DECOMPOSITION ==========', flush=True)
    print(result[[
        'contrast', 'full_score', 'metal_axis_alone_score', 'Anr_axis_alone_score',
        'joint_two_axis_score', 'score_orthogonal_to_both_axes',
        'fraction_108D_vector_energy_in_joint_span',
        'cosine_clinical_vs_metal_addition', 'cosine_clinical_vs_Anr_intact'
    ]].round(5).to_string(index=False), flush=True)
    print('\n========== SENSITIVITY: ANR AXIS FROM EACH STRAIN ==========', flush=True)
    print(pd.DataFrame(sensitivity).round(5).to_string(index=False), flush=True)
    print('\n========== TOP JOINT-ORTHOGONAL SIGNED CONTRIBUTORS ==========', flush=True)
    for name in CONTRASTS:
        g = contrib[contrib.contrast == name].copy()
        order = g['clinical_score_contribution_joint_orthogonal'].abs().sort_values(ascending=False).index
        print('\n' + name, flush=True)
        print(g.loc[order, ['imodulon', 'clinical_score_contribution_joint_orthogonal']].head(12)
              .round(5).to_string(index=False), flush=True)
    print('\nSTATUS:', STATUS, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
