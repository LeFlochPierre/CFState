#!/usr/bin/env python3
"""CF-State Audit 08g: perturbation-anchored decomposition of clinical CF contrasts.

Freeze every existing model component. Use the 17 donor-paired PAO1 spike-in
metal perturbations (Neff 2024, Zn+Fe+Mn together) to define ONE empirical
108-dimensional intervention direction (METALS minus UNTREATED). Then split
three pre-existing clinical contrasts into parallel and orthogonal parts.

IMPORTANT: the orthogonal remainder is *not* necessarily non-metal biology;
these data do not establish CF specificity, zinc-specific causation, or
independent clinical validation. The model and all previous audits are read-only.

Prerequisites: completed audits 07i, 08a, 08e2, 08f; scripts 07i and 08e2.
Run from ~/projects/CFstate with environment CFSTATE_ENV:
    python src/phase8/audit_08g_perturbation_anchored_clinical_decomposition.py --self-test
    python -u src/phase8/audit_08g_perturbation_anchored_clinical_decomposition.py
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P4 = ROOT / 'results/phase4'
P8 = ROOT / 'results/phase8'
A07I = ROOT / 'results/phase7/audit_07i_shared_CFBE_reference_negative_controls'
A08E = P8 / 'audit_08e2_Neff2024_paired_CF_sputum_metal_supplementation'
A08F = P8 / 'audit_08f_frozen_metal_axis_ablation_clinical_transfer'
A08G = P8 / 'audit_08g_perturbation_anchored_clinical_decomposition'
SRC07I = ROOT / 'src/phase7/audit_07i_CFstate_shared_reference_negative_controls.py'
SRC08E = ROOT / 'src/phase8/audit_08e2_CF_sputum_17_donor_metal_perturbation.py'
GFF = ROOT / 'references/phase6/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz'
SEED = 20261014
BOOT = 2000
STATUS = 'CFSTATE_PHASE8_08G_PERTURBATION_ANCHORED_DECOMPOSITION_EXPLORATORY_COMPLETE'
EXPECT = {
    A07I / 'AUDIT_07I_STATUS.txt': 'CFSTATE_PHASE7_07I_SHARED_REFERENCE_NEGATIVE_CONTROLS_COMPLETE_REVIEW_REQUIRED',
    A08E / 'AUDIT_08E2_STATUS.txt': 'CFSTATE_PHASE8_08E2_17_DONOR_COMBINED_METALS_DESCRIPTIVE_COMPLETE_REVIEW_REQUIRED',
    A08F / 'AUDIT_08F_STATUS.txt': 'CFSTATE_PHASE8_08F_METAL_ABLATION_CLINICAL_TRANSFER_DESCRIPTIVE_COMPLETE_REVIEW_REQUIRED',
}


def require(label, ok):
    if not bool(ok):
        print(label + ' FAIL', flush=True)
        raise RuntimeError('CHECK FAILED: ' + label)
    print(label + ' PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def cosine(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float((x @ y) / den) if den > 1e-12 else np.nan


def unit(x):
    z = np.asarray(x, float)
    norm = float(np.linalg.norm(z))
    if z.ndim != 1 or not np.isfinite(z).all() or norm < 1e-9:
        raise RuntimeError('Cannot construct a unit direction from nonfinite/nearzero vector')
    return z / norm


def decompose(d, u, c):
    """Orthogonal decomposition of clinical programme delta w.r.t. unit metal axis."""
    d = np.asarray(d, float)
    u = np.asarray(u, float)
    c = np.asarray(c, float)
    parallel = float(d @ u) * u
    remainder = d - parallel
    require_local = np.allclose(d, parallel + remainder, atol=1e-10)
    require_local &= abs(float(remainder @ u)) < 1e-8
    require_local &= abs(float(parallel @ remainder)) < 1e-8
    if not require_local:
        raise RuntimeError('Non-orthogonal vector decomposition')
    normsq = float(d @ d)
    return {
        'parallel': parallel,
        'orthogonal': remainder,
        'signed_full': float(d @ c),
        'signed_parallel': float(parallel @ c),
        'signed_orthogonal': float(remainder @ c),
        'vector_energy_fraction_parallel': float(parallel @ parallel / normsq) if normsq > 1e-12 else np.nan,
    }


def gff_gene_annotations(path):
    """PAO1 RefSeq descriptions for gene-membership hints, not causal annotation."""
    result = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] not in ('gene', 'CDS'):
                continue
            attrs = {}
            for part in fields[8].split(';'):
                if '=' in part:
                    k, v = part.split('=', 1)
                    attrs[k] = unquote(v)
            tag = attrs.get('locus_tag', '') or attrs.get('old_locus_tag', '')
            if not re.fullmatch(r'PA\d{4}(?:\.\d+)?', tag, re.I):
                continue
            tag = tag[:6].upper()
            row = (attrs.get('gene', ''), attrs.get('product', ''), fields[2])
            old = result.get(tag)
            if old is None or (fields[2] == 'CDS' and old[2] != 'CDS'):
                result[tag] = row
    return result


def self_test():
    rng = np.random.default_rng(48)
    c = unit(rng.normal(size=108))
    m = rng.normal(size=108)
    u = unit(m)
    d = rng.normal(size=108)
    x = decompose(d, u, c)
    assert np.allclose(d, x['parallel'] + x['orthogonal'])
    assert np.isclose(x['signed_full'], x['signed_parallel'] + x['signed_orthogonal'])
    assert 0 <= x['vector_energy_fraction_parallel'] <= 1
    assert np.allclose(decompose(m, u, c)['orthogonal'], 0, atol=1e-10)
    donor = np.stack([m + rng.normal(scale=.1, size=108) for _ in range(17)])
    left_out = np.stack([unit((donor.sum(axis=0) - row) / 16) for row in donor])
    assert all(cosine(row, u) > .9 for row in left_out)
    assert np.allclose(d @ (c - (c @ u) * u), x['signed_orthogonal'])
    print('AUDIT_08G_SELF_TEST_PASS', flush=True)


def main():
    print('=' * 87)
    print('CF-State Phase 8 — Audit 08g: perturbation-anchored clinical decomposition')
    print('Frozen 108 programmes; combined Zn+Fe+Mn intervention; post-hoc exploratory')
    print('=' * 87, flush=True)
    require('working_directory_CFstate', P4.is_dir())
    require('new_output_directory', not A08G.exists())
    for path, expected in EXPECT.items():
        require('completed_input_' + path.parent.name, path.is_file() and path.read_text().strip() == expected)
    for path in [SRC07I, SRC08E, GFF]:
        require('required_source_' + path.name, path.is_file())

    # Original read-only modules: import functions; do not invoke their main functions.
    sys.path.insert(0, str(SRC08E.parent))
    sys.path.insert(0, str(SRC07I.parent))
    import audit_08e2_CF_sputum_17_donor_metal_perturbation as neff
    import audit_07i_CFstate_shared_reference_negative_controls as clinical
    require('08e2_source_same_project', neff.ROOT == ROOT)
    require('07i_source_same_project', clinical.ROOT == ROOT)

    # 1. Recover the 17 *full* paired 108-vectors; 08e2 CSV saved only their mean.
    for label, path in neff.MODEL.items():
        require('frozen_model_' + label, path.is_file())
    cnt_path = neff.RAW / neff.COUNT_NAME
    require('existing_public_Neff_counts', cnt_path.is_file())
    counts, manifest, excluded, ambiguous = neff.parse_counts(cnt_path)
    require('Neff_pairing_17', len(manifest) == 34 and manifest['donor_id'].nunique() == 17)
    mask = pd.read_csv(neff.MODEL['mask'])['GeneID'].astype(str).tolist()
    require('full_4923_Neff_genes', len(mask) == 4923 and set(mask).issubset(counts.index))
    frozen = neff.load_frozen(mask)
    names = frozen['all']
    stable = frozen['stable']
    candidate = frozen['candidate']
    require('frozen_candidate_unit', abs(np.linalg.norm(candidate) - 1) < 1e-6)
    matrix = counts.loc[mask, manifest['sample'].tolist()].to_numpy(float)
    factors, _ = neff.size_factors(matrix)
    norm = np.log2(1 + matrix / factors[None, :])
    pinv = np.linalg.pinv(frozen['m'])
    all_activities = (pinv @ norm) * frozen['slopes'][:, None]
    positions = [names.index(s) for s in stable]
    calibrated = all_activities[positions, :]
    B = frozen['b']
    residual = calibrated - B @ np.linalg.lstsq(B, calibrated, rcond=None)[0]
    require('Neff_controls_orthogonal', np.max(np.abs(B.T @ residual)) < 1e-6)
    donors, paired_programmes = neff.paired_values(residual, manifest)
    require('17x108_donor_responses', paired_programmes.shape == (17, 108) and
            len(set(donors)) == 17 and np.isfinite(paired_programmes).all())
    metal_mean = paired_programmes.mean(axis=0)

    # Strong bridge to frozen 08e2 all-programme means and signed response.
    prev_prog = pd.read_csv(A08E / 'DESCRIPTIVE_108_programme_mean_paired_changes.csv').set_index('imodulon')
    require('08e2_stable_108_match', set(prev_prog.index) == set(stable))
    require('08e2_reproduce_all_108_mean_changes', np.allclose(
        metal_mean, prev_prog.loc[stable, 'mean_controlled_change'].to_numpy(float), atol=1e-5))
    require('08e2_reproduce_all_108_weights', np.allclose(
        candidate, prev_prog.loc[stable, 'frozen_CF_weight'].to_numpy(float), atol=1e-6))
    prev_metrics = pd.read_csv(A08E / 'DESCRIPTIVE_paired_bootstrap_and_exact_signflip.csv').set_index('metric')
    require('08e2_reproduce_frozen_score_delta', np.isclose(
        metal_mean @ candidate,
        prev_metrics.loc['FROZEN_CF_SIGNED_SCORE_METALS_MINUS_UNTREATED', 'mean_change'], atol=1e-5))

    # Direction is metal addition, *not* a uniquely zinc-related / CF-specific axis.
    u = unit(metal_mean)
    loo = np.stack([unit((paired_programmes.sum(axis=0) - row) / 16)
                    for row in paired_programmes])
    loo_cos = np.array([cosine(v, u) for v in loo])
    individual_cos = np.array([cosine(row, u) for row in paired_programmes])
    require('leave_one_donor_out_positive_stability', np.isfinite(loo_cos).all() and (loo_cos > 0).all())
    require('individual_donor_cosines_finite', np.isfinite(individual_cos).all())
    jack = pd.DataFrame({
        'donor_id': donors,
        'individual_response_cosine_to_full_mean': individual_cos,
        'leave_one_out_direction_cosine_to_full_mean': loo_cos,
        'individual_signed_CF_score_metals_minus_untreated': paired_programmes @ candidate,
        'individual_magnitude_108': np.linalg.norm(paired_programmes, axis=1),
    })
    rng = np.random.default_rng(SEED)
    bs_idx = rng.integers(0, 17, size=(BOOT, 17))
    bs_means = paired_programmes[bs_idx].mean(axis=1)
    bs_u = bs_means / np.linalg.norm(bs_means, axis=1)[:, None]
    require('bootstrap_metal_unit_directions', np.isfinite(bs_u).all())
    bootstrap_cos = bs_u @ u

    # 2. Recreate precisely the three previous CLINICAL contrasts, separately
    #    from the experimental intervention (different measurement scale).
    for label, path in clinical.FILES.items():
        require('clinical_file_' + label, path.is_file())
    require('shared_model_mask', mask == pd.read_csv(clinical.FILES['mask'])['GeneID'].astype(str).tolist())
    saf = pd.read_csv(clinical.FILES['SAF'], sep='\t').set_index('GeneID')
    require('SAF_5567_unique', len(saf) == 5567 and saf.index.is_unique)
    length = (pd.to_numeric(saf['End']) - pd.to_numeric(saf['Start']) + 1).astype(float)
    co = pd.read_csv(clinical.FILES['05h_counts'], index_col=0)
    co.index = co.index.astype(str)
    cm = pd.read_csv(clinical.FILES['manifest'], dtype=str).fillna('')
    require('13_Cornforth_columns', co.shape == (5567, 13) and set(co.columns) == set(cm.published_sample))
    co = co.reindex(index=saf.index)
    for run, (patient, time, folder) in clinical.RUNS.items():
        s = clinical.get_new_counts(run, folder)
        co[run] = s.reindex(saf.index).to_numpy(float)
    require('17_clinical_sample_count_columns', co.shape == (5567, 17) and np.isfinite(co.to_numpy(float)).all())
    rates = co.div(length, axis=0)
    denom = rates.sum(axis=0)
    require('clinical_TPM_denominators_positive', (denom > 0).all())
    clinical_expression = np.log2(1 + rates.div(denom, axis=1) * 1e6).loc[mask]

    donor_expression = {}
    for donor, group in cm.groupby('published_donor_group', sort=False):
        require('clinical_one_cohort_' + donor, group.group.nunique() == 1)
        label = 'CORNFORTH_CF' if (group.group == 'CF_sputum').all() else 'CORNFORTH_NONCF'
        donor_expression[donor] = (label, clinical_expression[group.published_sample.tolist()].mean(axis=1).to_numpy(float))
    for patient in ('CF01', 'CF146'):
        runs = [run for run, (p, _, _) in clinical.RUNS.items() if p == patient]
        require('two_specimens_' + patient, len(runs) == 2)
        donor_expression[patient] = ('NEW_CF', clinical_expression[runs].mean(axis=1).to_numpy(float))
    require('14_independent_clinical_donors', len(donor_expression) == 14)
    by_cohort = {}
    for group in ('CORNFORTH_CF', 'CORNFORTH_NONCF', 'NEW_CF'):
        vals = [x for label, x in donor_expression.values() if label == group]
        by_cohort[group] = np.mean(vals, axis=0)
    require('cohorts_have_7_5_2_donors',
            [sum(label == group for label, _ in donor_expression.values())
             for group in ('CORNFORTH_CF', 'CORNFORTH_NONCF', 'NEW_CF')] == [7, 5, 2])

    def project_expression(x):
        raw116 = (pinv @ np.asarray(x, float)) * frozen['slopes']
        raw108 = raw116[positions]
        result = raw108 - B @ np.linalg.lstsq(B, raw108, rcond=None)[0]
        if np.max(np.abs(B.T @ result)) >= 1e-5:
            raise RuntimeError('Clinical projection outside controlled residual space')
        return result

    a = by_cohort
    contrasts = {
        'Cornforth_CF_minus_nonCF_7v5': project_expression(a['CORNFORTH_CF'] - a['CORNFORTH_NONCF']),
        'New_CF_minus_Cornforth_nonCF_2v5': project_expression(a['NEW_CF'] - a['CORNFORTH_NONCF']),
        'New_CF_minus_Cornforth_CF_2v7': project_expression(a['NEW_CF'] - a['CORNFORTH_CF']),
    }
    prev_08f = pd.read_csv(A08F / 'DESCRIPTIVE_reference_invariant_clinical_ablation_contrasts.csv')
    prev_07i = pd.read_csv(A07I / 'DESCRIPTIVE_reference_invariant_group_contrasts.csv')
    prev_05j = pd.read_csv(clinical.FILES['05j2_contrast']).set_index('imodulon')
    require('frozen_05j2_contrast_reproduced', np.allclose(
        contrasts['Cornforth_CF_minus_nonCF_7v5'],
        prev_05j.loc[stable, 'CF_minus_nonCF_residual'].to_numpy(float), atol=1e-6))
    for name, x in contrasts.items():
        old = prev_08f.loc[(prev_08f.contrast == name) & (prev_08f.readout == 'ALL_108')]
        old07 = prev_07i.loc[prev_07i.contrast == name]
        require('08f_07i_reproduce_' + name,
                len(old) == 1 and len(old07) == 1 and
                np.isclose(x @ candidate, float(old.iloc[0].signed_projection), atol=1e-5) and
                np.isclose(cosine(x, candidate), float(old.iloc[0].cosine_with_readout), atol=1e-5) and
                np.isclose(x @ candidate, float(old07.iloc[0].signed_projection), atol=1e-5))

    # 3. Reproducible empirical metal axis: 1-dimensional decomposition.
    rows, program_rows, gene_candidates = [], [], set()
    selected = frozen['selected']
    for name, d in contrasts.items():
        v = decompose(d, u, candidate)
        require('clinical_score_accounting_' + name, np.isclose(
            v['signed_full'], v['signed_parallel'] + v['signed_orthogonal'], atol=1e-8))
        assert np.allclose(v['orthogonal'] @ u, 0, atol=1e-8)
        bs_proj = bs_u @ d
        bs_parallel_scores = bs_proj * (bs_u @ candidate)
        jack_parallel = np.array([(d @ direction) * (candidate @ direction) for direction in loo])
        rows.append({
            'contrast': name,
            'clinical_full_signed_CF_score': v['signed_full'],
            'clinical_parallel_signed_CF_score': v['signed_parallel'],
            'clinical_orthogonal_signed_CF_score': v['signed_orthogonal'],
            'fraction_clinical_CF_score_parallel_signed': (
                v['signed_parallel'] / v['signed_full'] if abs(v['signed_full']) > 1e-10 else np.nan),
            'cosine_clinical_vs_metal_addition': cosine(d, u),
            'cosine_clinical_vs_unsupplemented_like': cosine(d, -u),
            'fraction_clinical_vector_energy_parallel': v['vector_energy_fraction_parallel'],
            'clinical_full_residual_norm': float(np.linalg.norm(d)),
            'clinical_orthogonal_residual_norm': float(np.linalg.norm(v['orthogonal'])),
            'LODO_min_clinical_parallel_score': float(jack_parallel.min()),
            'LODO_max_clinical_parallel_score': float(jack_parallel.max()),
            'donor_bootstrap_parallel_score_q025': float(np.quantile(bs_parallel_scores, .025)),
            'donor_bootstrap_parallel_score_q975': float(np.quantile(bs_parallel_scores, .975)),
            'donor_bootstrap_cosine_to_metal_q025': float(np.quantile(
                bs_proj / np.linalg.norm(d), .025)),
            'donor_bootstrap_cosine_to_metal_q975': float(np.quantile(
                bs_proj / np.linalg.norm(d), .975)),
        })
        for k, gene in enumerate(stable):
            program_rows.append({
                'contrast': name,
                'imodulon': gene,
                'selected_in_original18': gene in selected,
                'frozen_CF_signature_weight': float(candidate[k]),
                'mean_experimental_metal_change': float(metal_mean[k]),
                'clinical_full_programme_delta': float(d[k]),
                'clinical_metal_parallel_programme_delta': float(v['parallel'][k]),
                'clinical_metal_orthogonal_programme_delta': float(v['orthogonal'][k]),
                'signed_CF_score_full_programme_contribution': float(candidate[k] * d[k]),
                'signed_CF_score_parallel_programme_contribution': float(candidate[k] * v['parallel'][k]),
                'signed_CF_score_orthogonal_programme_contribution': float(candidate[k] * v['orthogonal'][k]),
            })
        ranked = sorted(range(108), key=lambda k: abs(candidate[k] * v['orthogonal'][k]), reverse=True)
        gene_candidates.update(stable[k] for k in ranked[:12])
    contrasts_df = pd.DataFrame(rows)
    program_df = pd.DataFrame(program_rows)

    # Gene members are described by original reference iModulon loadings, not
    # new DE tests; never label them directly as experimentally responsive genes.
    m = frozen['m'][:, positions]
    annot = gff_gene_annotations(GFF)
    gene_rows = []
    for programme in sorted(gene_candidates):
        idx = stable.index(programme)
        loadings = m[:, idx]
        top = np.argsort(np.abs(loadings))[::-1]
        top = [int(k) for k in top if abs(loadings[k]) > 1e-12][:8]
        for rank, k in enumerate(top, 1):
            gene = mask[k]
            sym, product, _ = annot.get(gene, ('', '', ''))
            gene_rows.append({'imodulon': programme, 'rank_by_absolute_reference_loading': rank,
                              'PAO1_GeneID': gene, 'gene_symbol': sym, 'annotated_product': product,
                              'reference_iModulon_loading': float(loadings[k]),
                              'interpretation': 'REFERENCE_IMODULON_GENE_MEMBERSHIP_NOT_DE_PROOF'})

    stability = pd.DataFrame([{
        'n_donor_pairs': 17,
        'mean_metal_intervention_norm': float(np.linalg.norm(metal_mean)),
        'cosine_mean_metal_intervention_with_frozen_CF_signature': cosine(u, candidate),
        'signed_CF_score_of_mean_metal_intervention': float(metal_mean @ candidate),
        'median_individual_donor_cosine_to_mean': float(np.median(individual_cos)),
        'min_individual_donor_cosine_to_mean': float(individual_cos.min()),
        'median_LODO_cosine_to_full': float(np.median(loo_cos)),
        'min_LODO_cosine_to_full': float(loo_cos.min()),
        'donor_bootstrap_mean_direction_cosine_q025': float(np.quantile(bootstrap_cos, .025)),
        'donor_bootstrap_mean_direction_cosine_q975': float(np.quantile(bootstrap_cos, .975)),
        'frozen_signature_fraction_vector_energy_along_mean_metal_axis': float((candidate @ u) ** 2),
    }])

    # Preflight all output calculations before writing any results.
    require('three_clinical_contrasts', len(contrasts_df) == 3)
    require('three_times_108_programme_contributions', len(program_df) == 324)
    require('candidate_score_component_sums_all_three', all(
        np.isclose(group['signed_CF_score_orthogonal_programme_contribution'].sum(),
                   float(contrasts_df.loc[contrasts_df.contrast == name,
                                         'clinical_orthogonal_signed_CF_score'].iloc[0]), atol=1e-8)
        for name, group in program_df.groupby('contrast')))
    require('all_export_numeric_finite', np.isfinite(
        contrasts_df.select_dtypes('number').to_numpy(float)).all())

    A08G.mkdir(parents=True, exist_ok=False)
    (A08G / 'AUDIT_08G_STATUS.txt').write_text('INCOMPLETE_PENDING_EXPORT\n')
    stability.to_csv(A08G / 'DESCRIPTIVE_metal_intervention_direction_stability.csv', index=False)
    jack.to_csv(A08G / 'DESCRIPTIVE_17_donor_response_and_LODO_stability.csv', index=False)
    pd.DataFrame({'imodulon': stable,
                  'mean_METALS_minus_UNTREATED_108': metal_mean,
                  'unit_mean_metal_direction_108': u,
                  'frozen_CF_signature_unit_108': candidate,
                  'original18': [s in selected for s in stable]}).to_csv(
        A08G / 'FROZEN_derived_108_programme_metal_axis.csv', index=False)
    contrasts_df.to_csv(A08G / 'EXPLORATORY_3_clinical_contrast_parallel_orthogonal.csv', index=False)
    program_df.to_csv(A08G / 'DESCRIPTIVE_324_programme_contributions_to_clinical_decomposition.csv', index=False)
    pd.DataFrame(gene_rows).to_csv(A08G / 'EXPLORATORY_reference_genes_for_top_orthogonal_programmes.csv', index=False)
    sources = [SRC07I, SRC08E, GFF, neff.MODEL['mask'], neff.MODEL['modulome'],
               neff.MODEL['calibration'], neff.MODEL['stable'], neff.MODEL['controls'],
               neff.MODEL['candidate'], cnt_path, clinical.FILES['05h_counts'],
               clinical.FILES['manifest'], clinical.FILES['05j2_contrast'],
               A08E / 'DESCRIPTIVE_108_programme_mean_paired_changes.csv',
               A08F / 'DESCRIPTIVE_reference_invariant_clinical_ablation_contrasts.csv',
               Path(__file__).resolve()]
    pd.DataFrame([{'path': str(p), 'sha256': sha256(p)} for p in sources]).to_csv(
        A08G / 'FROZEN_input_SHA256_provenance.csv', index=False)
    (A08G / 'DESIGN_LIMITATIONS.txt').write_text(
        'Exploratory POST-HOC decomposition. No fitting or model updates.\n'
        'Metal direction is mean within-donor METALS minus UNTREATED PAO1-spike-in response.\n'
        'Combined Zn+Fe+Mn cannot attribute individual metal effects.\n'
        'Clinical comparisons are CF sputum vs wound/burn: tissue/study/site confounded.\n'
        'Cornforth Danish CF overlaps historical discovery cohorts; no new independent proof.\n'
        'The two newer CF patients are two independent donors, not four donors.\n'
        'Ex-vivo PAO1 counts and patient log2TPM are different quantifications; compare\n'
        'directions only, not magnitudes or calibration across the two datasets.\n'
        'One experimental direction captures only MEAN RESPONSE to this intervention.\n'
        'Orthogonal remainder does NOT mean non-metal biology or a causal CF factor.\n'
        'LODO and bootstrap are WITHIN THIS ONE EXPERIMENT, not independent cohorts.\n'
        'Top iModulon reference genes are gene MEMBERSHIP hints, not validated DE drivers.\n'
        'Any overlap with model-discovery/reference datasets limits independent validation.\n')

    print('\n========== 17-DONOR METAL DIRECTION: STABILITY ==========', flush=True)
    print(stability.round(5).to_string(index=False), flush=True)
    print('\n========== CLINICAL CF CONTRASTS: PARALLEL VS ORTHOGONAL ==========', flush=True)
    print(contrasts_df.round(5).to_string(index=False), flush=True)
    print('\n========== TOP CLINICAL ORTHOGONAL SIGNED CONTRIBUTORS ==========', flush=True)
    for name, grp in program_df.groupby('contrast', sort=False):
        print('\n' + name, flush=True)
        out = grp.reindex(grp['signed_CF_score_orthogonal_programme_contribution'].abs().sort_values(
            ascending=False).index)
        print(out[['imodulon', 'selected_in_original18', 'mean_experimental_metal_change',
                   'clinical_metal_orthogonal_programme_delta',
                   'signed_CF_score_orthogonal_programme_contribution']].head(12).round(5).to_string(index=False), flush=True)
    print('\n========== INDIVIDUAL METAL-DONOR RESPONSE STABILITY ==========', flush=True)
    print(jack.round(5).to_string(index=False), flush=True)
    (A08G / 'AUDIT_08G_STATUS.txt').write_text(STATUS + '\n')
    print('\nSTATUS:', STATUS, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        main()
