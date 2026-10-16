#!/usr/bin/env python3
"""CF-State Phase 8 / Audit 08b — public 24-sample perturbation test.

Projects within-strain, within-experiment contrasts from GSE280261
(PAO1 and FRD1; WT and delta_glpR; untreated and 5 ug/mL tobramycin)
into the existing frozen 116 iModulon / 108 stable / 11-state-residual
representation and compares contrast directions to frozen CF-airway 03h.

IMPORTANT: Does not retrain the signature, does not compare absolute activity
between strains or to patient samples, and does NOT establish CF specificity.
Processed counts aligned against PAO1 may be biased for FRD1; harvest timing
also varies across treatment/strain. n=3 replicates per experimental cell.

Run from ~/projects/CFstate using CFSTATE_ENV. --self-test exercises parsers,
median-ratio normalisation and contrast directions without project files.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P4 = ROOT / 'results/phase4'
P8 = ROOT / 'results/phase8'
A8A = P8 / 'audit_08a_frozen_CF_biology_GSE280261_preflight'
RAW = ROOT / 'data/raw/GSE280261_public_processed'
OUT = P8 / 'audit_08b_GSE280261_frozen_within_strain_perturbations'
FILES = {
    '08a_manifest': A8A / 'GSE280261_FROZEN_24_processed_sample_manifest_and_schema.csv',
    '08a_status': A8A / 'AUDIT_08A_STATUS.txt',
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'controlled_states': P4 / 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'registry18': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/conservatively_replicated_programmes.csv',
    '08a_programmes': A8A / 'FROZEN_108_programme_signed_contributions.csv',
    'script': Path(__file__).resolve(),
}
STATUS = 'CFSTATE_PHASE8_08B_PUBLIC_PERTURBATION_FROZEN_PROGRAMMES_DESCRIPTIVE_REVIEW_REQUIRED'
SEED = 20261014
N_BOOT = 1200


def check(label, cond):
    if not bool(cond):
        print(label + ' FAIL', flush=True)
        raise RuntimeError('CHECK FAILED: ' + label)
    print(label + ' PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_pao1_count_file(path):
    """Return unambiguous PAO1 gene counts and duplicated IDs.

    Duplicate loci are never silently summed. Count-table row identifiers must
    explicitly encode PAO1 locus tags. Unrecognised entries are documented.
    """
    frame = pd.read_csv(path, sep='\t', compression='gzip', dtype={0: str})
    if frame.shape[1] != 2:
        raise RuntimeError('Expected two-column processed counts: ' + str(path))
    if frame.columns[0].strip().lower() not in ('geneid', 'gene_id', 'gene'):
        raise RuntimeError('Unrecognised gene identifier column in ' + str(path))
    raw_ids = frame.iloc[:, 0].astype(str).str.strip()
    numbers = pd.to_numeric(frame.iloc[:, 1], errors='coerce')
    if numbers.isna().any() or not np.isfinite(numbers.to_numpy(float)).all():
        raise RuntimeError('Invalid/nonfinite count value in ' + str(path))
    if (numbers < 0).any() or not np.allclose(numbers, np.round(numbers)):
        raise RuntimeError('Noninteger/negative count value in ' + str(path))
    ids = raw_ids.str.extract(r'^(PA\d{4})(?:\.\d+)?$', flags=re.I, expand=False)
    ids = ids.str.upper()
    known = pd.DataFrame({'GeneID': ids, 'count': numbers.to_numpy(float)})
    known = known.dropna(subset=['GeneID'])
    collisions = set(known.loc[known['GeneID'].duplicated(keep=False), 'GeneID'])
    good = known.loc[~known['GeneID'].isin(collisions)].set_index('GeneID')['count']
    return good, collisions, {
        'raw_gene_rows': len(frame),
        'recognised_gene_rows': len(known),
        'unknown_gene_rows': int(ids.isna().sum()),
        'duplicate_gene_ids': len(collisions),
        'total_assigned_counts_all_rows': float(numbers.sum()),
    }


def load_modulome(path, genes, names):
    obj = json.loads(path.read_text())
    check('modulome_has_M', isinstance(obj, dict) and 'M' in obj)
    raw = obj['M']
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and {'index', 'columns', 'data'}.issubset(raw):
        frame = pd.DataFrame(raw['data'], index=raw['index'], columns=raw['columns'])
    else:
        frame = pd.DataFrame(raw)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if set(genes).issubset(frame.columns) and set(names).issubset(frame.index):
        frame = frame.T
    check('modulome_IDs', frame.index.is_unique and frame.columns.is_unique and
          set(genes).issubset(frame.index) and set(names).issubset(frame.columns))
    m = frame.loc[genes, names].apply(pd.to_numeric, errors='raise').to_numpy(float)
    check('modulome_finite', np.isfinite(m).all())
    return m


def size_factors_median_ratio(counts):
    """DESeq-style median ratio on positive-in-every-sample genes; no inference."""
    vals = np.asarray(counts, dtype=float)
    if vals.ndim != 2 or (vals < 0).any() or not np.isfinite(vals).all():
        raise RuntimeError('Invalid counts for median-ratio normalisation')
    valid = (vals > 0).all(axis=1)
    if valid.sum() < 1:
        raise RuntimeError('No positive-across-samples genes for median-ratio factors')
    geo = np.exp(np.mean(np.log(vals[valid, :]), axis=1))
    factors = np.median(vals[valid, :] / geo[:, None], axis=0)
    if (factors <= 0).any() or not np.isfinite(factors).all():
        raise RuntimeError('Invalid median-ratio size factors')
    factors /= np.exp(np.log(factors).mean())
    return factors, int(valid.sum())


def group_key(strain, genotype, treated):
    return (str(strain), str(genotype), str(treated))


def build_contrasts():
    """Every group is internal to one strain, no cross-strain mean contrast."""
    out = []
    for strain in ('PAO1', 'FRD1'):
        wt_no = group_key(strain, 'WT', 'NO')
        wt_yes = group_key(strain, 'WT', 'YES')
        dl_no = group_key(strain, 'delta_glpR', 'NO')
        dl_yes = group_key(strain, 'delta_glpR', 'YES')
        out.extend([
            (strain + '_WT_tobramycin', {wt_yes: 1, wt_no: -1}),
            (strain + '_delta_glpR_tobramycin', {dl_yes: 1, dl_no: -1}),
            (strain + '_glpR_deletion_untreated', {dl_no: 1, wt_no: -1}),
            (strain + '_glpR_deletion_treated', {dl_yes: 1, wt_yes: -1}),
            (strain + '_tobramycin_by_glpR_interaction',
             {dl_yes: 1, dl_no: -1, wt_yes: -1, wt_no: 1}),
        ])
    return out


def project_signatures(m, slopes, names, stable, b, expression):
    """Exactly the historical frozen linear projection and control residual."""
    operator = np.linalg.pinv(m)
    raw = (operator @ expression) * slopes[:, None]
    index = {name: i for i, name in enumerate(names)}
    stable_activity = raw[[index[n] for n in stable], :]
    residual = stable_activity - b @ np.linalg.lstsq(b, stable_activity, rcond=None)[0]
    if np.max(np.abs(b.T @ residual)) > 1e-6:
        raise RuntimeError('Residual not orthogonal to the frozen 11 controls')
    return stable_activity, residual


def contrast_vector(resid, groups, weights):
    result = np.zeros(resid.shape[0], dtype=float)
    for key, weight in weights.items():
        result += float(weight) * resid[:, groups[key]].mean(axis=1)
    return result


def safe_cos(x, unit):
    norm = float(np.linalg.norm(x))
    return float(x @ unit / norm) if norm > 1e-10 else float('nan')


def bootstrap_contrast(resid, groups, weights, candidate, rng, n_boot=N_BOOT):
    vals = np.empty(n_boot, dtype=float)
    cosines = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        vec = np.zeros(resid.shape[0], dtype=float)
        for key, coeff in weights.items():
            ix = groups[key]
            drawn = rng.choice(ix, size=len(ix), replace=True)
            vec += float(coeff) * resid[:, drawn].mean(axis=1)
        vals[b] = float(vec @ candidate)
        cosines[b] = safe_cos(vec, candidate)
    return {
        'bootstrap_signed_ci025': float(np.quantile(vals, 0.025)),
        'bootstrap_signed_ci975': float(np.quantile(vals, 0.975)),
        'bootstrap_signed_fraction_positive': float(np.mean(vals > 0)),
        'bootstrap_cosine_ci025': float(np.nanquantile(cosines, 0.025)),
        'bootstrap_cosine_ci975': float(np.nanquantile(cosines, 0.975)),
    }


def self_test():
    with tempfile.TemporaryDirectory() as td:
        file = Path(td) / 'toy.tabular.gz'
        with gzip.open(file, 'wt') as f:
            f.write('Geneid\tCounts\nPA0001\t3\nPA0002\t10\nPA0002.1\t8\n_bad\t2\n')
        series, collision, qc = parse_pao1_count_file(file)
        assert set(series.index) == {'PA0001'}
        assert collision == {'PA0002'}
        assert qc['unknown_gene_rows'] == 1
        assert qc['total_assigned_counts_all_rows'] == 23
    counts = np.array([[2, 4, 6], [10, 20, 30], [0, 0, 0]], float)
    factors, used = size_factors_median_ratio(counts)
    assert used == 2 and np.allclose(factors / factors[0], [1, 2, 3])
    groups = {('PAO1', 'WT', 'NO'): np.array([0, 1, 2]),
              ('PAO1', 'WT', 'YES'): np.array([3, 4, 5])}
    residual = np.array([[0, 0, 0, 2, 2, 2], [0, 0, 0, -1, -1, -1]], float)
    weights = {('PAO1', 'WT', 'YES'): 1, ('PAO1', 'WT', 'NO'): -1}
    assert np.allclose(contrast_vector(residual, groups, weights), [2, -1])
    assert len(build_contrasts()) == 10
    assert safe_cos(np.array([2, -1]), np.array([1, 0])) > 0
    print('AUDIT_08B_SELF_TEST_PASS')


def main():
    print('=' * 78)
    print('CF-State Phase 8 — Audit 08b: GSE280261 frozen perturbation responses')
    print('WITHIN-STRAIN contrasts; no CF-specificity or clinical validation claim')
    print('=' * 78, flush=True)
    check('working_directory_CFstate', (ROOT / 'results/phase4').is_dir())
    check('new_output_directory', not OUT.exists())
    for label, path in FILES.items():
        check('input_' + label, path.is_file())
    check('08a_status_valid', FILES['08a_status'].read_text().strip() ==
          'CFSTATE_PHASE8_08A_FROZEN_BIOLOGY_AND_PUBLIC_GEO_PREFLIGHT_REVIEW_REQUIRED')

    manifest = pd.read_csv(FILES['08a_manifest'])
    needed = {'GSM', 'strain', 'genotype', 'tobramycin_5ug_ml', 'processed_file',
              'processed_file_sha256'}
    check('manifest_columns', needed.issubset(manifest.columns))
    manifest = manifest.sort_values('GSM').reset_index(drop=True)
    check('manifest_24_unique', len(manifest) == 24 and manifest['GSM'].is_unique)
    checks = manifest.groupby(['strain', 'genotype', 'tobramycin_5ug_ml']).size()
    check('eight_groups_three_replicates', len(checks) == 8 and (checks == 3).all())
    check('manifest_strains', set(manifest['strain']) == {'PAO1', 'FRD1'})
    check('manifest_genotypes', set(manifest['genotype']) == {'WT', 'delta_glpR'})
    check('manifest_treatment', set(manifest['tobramycin_5ug_ml']) == {'YES', 'NO'})

    mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
    check('4923_unique_mask', len(mask) == 4923 and len(set(mask)) == 4923)
    cal = pd.read_csv(FILES['calibration']).set_index('imodulon')
    names = cal.index.astype(str).tolist()
    stable = pd.read_csv(FILES['stable'])['imodulon'].astype(str).tolist()
    check('116_108_programmes', len(names) == 116 and len(stable) == 108 and
          len(set(stable)) == 108 and set(stable).issubset(names))
    slopes = pd.to_numeric(cal.loc[names, 'calibration_slope'], errors='raise').to_numpy(float)
    check('calibration_finite_nonzero', np.isfinite(slopes).all() and (np.abs(slopes) > 1e-12).all())
    candidate_df = pd.read_csv(FILES['candidate'], index_col=0)
    candidate_df.index = candidate_df.index.astype(str).str.strip()
    check('candidate_108_names', candidate_df.index.is_unique and set(candidate_df.index) == set(stable))
    c = candidate_df.loc[stable, 'two_cohort_consensus_unit'].to_numpy(float)
    check('frozen_candidate_unit', np.isfinite(c).all() and np.isclose(np.linalg.norm(c), 1, atol=1e-6))
    selected18 = set(pd.read_csv(FILES['registry18'])['imodulon'].astype(str))
    check('18_original_programmes', len(selected18) == 18 and selected18.issubset(stable))

    loaded = {}
    qcs = []
    collision_union = set()
    inputs = []
    for record in manifest.itertuples(index=False):
        file = RAW / str(record.processed_file)
        check('file_' + record.GSM, file.is_file())
        check('archive_sha256_' + record.GSM,
              sha256(file) == str(record.processed_file_sha256))
        series, collisions, qc = parse_pao1_count_file(file)
        loaded[str(record.GSM)] = series
        collision_union.update(collisions)
        qcs.append({'GSM': record.GSM, **qc})
        inputs.append({'input': 'processed_' + record.GSM,
                       'path': str(file.relative_to(ROOT)), 'sha256': sha256(file)})
    # Require a consistent genuinely observed feature set in every sample.
    common = set(mask) - collision_union
    for series in loaded.values():
        common.intersection_update(series.index)
    genes = [gene for gene in mask if gene in common]
    check('minimum_common_4400_genes', len(genes) >= 4400)
    print('Common unambiguous frozen genes:', len(genes), '/4923',
          'globally ambiguous:', len(collision_union), flush=True)

    C = pd.DataFrame({gsm: loaded[gsm].reindex(genes).to_numpy(float)
                      for gsm in manifest['GSM']}, index=genes)
    check('count_matrix_24_nonnegative_finite', C.shape == (len(genes), 24) and
          np.isfinite(C.to_numpy()).all() and (C.to_numpy() >= 0).all())

    groups = {}
    for key, frame in manifest.groupby(['strain', 'genotype', 'tobramycin_5ug_ml'], sort=True):
        groups[key] = frame.index.to_numpy(dtype=int)
    check('group_index_8x3', len(groups) == 8 and all(len(v) == 3 for v in groups.values()))

    # Within-strain DESeq-style median-ratio normalisation; avoid comparing
    # absolute PAO1 vs FRD1 levels after PAO1-reference mapping.
    norm = np.zeros(C.shape, dtype=float)
    factors = np.zeros(C.shape[1], dtype=float)
    normalisation_qc = {}
    for strain in ('PAO1', 'FRD1'):
        ix = np.flatnonzero(manifest['strain'].to_numpy() == strain)
        f, genes_ref = size_factors_median_ratio(C.to_numpy()[:, ix])
        check('size_factor_genes_' + strain, genes_ref >= 500)
        factors[ix] = f
        norm[:, ix] = np.log2(1.0 + C.to_numpy()[:, ix] / f[None, :])
        normalisation_qc[strain] = genes_ref
    check('log2norm_finite', np.isfinite(norm).all())
    sample_qc = manifest[['GSM', 'strain', 'genotype', 'tobramycin_5ug_ml',
                          'processed_file']].copy()
    sample_qc = sample_qc.merge(pd.DataFrame(qcs), on='GSM', validate='1:1')
    sample_qc['usable_mask_genes'] = len(genes)
    sample_qc['detected_genes_GE1'] = (C.to_numpy() >= 1).sum(axis=0)
    sample_qc['detected_genes_GE10'] = (C.to_numpy() >= 10).sum(axis=0)
    sample_qc['assigned_counts_usable_mask'] = C.sum(axis=0).to_numpy()
    sample_qc['within_strain_median_ratio_size_factor'] = factors
    sample_qc['median_ratio_reference_genes'] = sample_qc['strain'].map(normalisation_qc)

    # QC replicate agreement is descriptive, not a biological effect test.
    corr = np.corrcoef(norm.T)
    group_qc = []
    for key, ix in groups.items():
        pair = corr[np.ix_(ix, ix)]
        values = pair[np.triu_indices(len(ix), 1)]
        group_qc.append({'strain': key[0], 'genotype': key[1],
                         'tobramycin_5ug_ml': key[2], 'replicates': len(ix),
                         'min_replicate_correlation': float(np.nanmin(values)),
                         'median_replicate_correlation': float(np.nanmedian(values))})
    group_qc_df = pd.DataFrame(group_qc)
    check('replicate_correlations_finite', np.isfinite(group_qc_df['min_replicate_correlation']).all())
    # PCA strictly for diagnosing outlier samples, never choosing model features.
    centered = norm.T - norm.mean(axis=1)
    gram = centered @ centered.T
    eigval, eigvec = np.linalg.eigh(gram)
    order = np.argsort(eigval)[::-1]
    eigval = np.clip(eigval[order], 0, None)
    eigvec = eigvec[:, order]
    pca = manifest[['GSM', 'strain', 'genotype', 'tobramycin_5ug_ml']].copy()
    for k in range(2):
        pca['PC' + str(k + 1)] = eigvec[:, k] * np.sqrt(eigval[k])
    explain = eigval[:2] / eigval.sum()

    M = load_modulome(FILES['modulome'], genes, names)
    check('rank116_on_observed_genes', np.linalg.matrix_rank(M) == 116)
    ref = pd.read_csv(FILES['controlled_states'], index_col=0)
    ref.index = ref.index.astype(str).str.strip()
    check('11_controls_108_stable', ref.shape[1] == 11 and set(stable).issubset(ref.index))
    B = ref.loc[stable].apply(pd.to_numeric, errors='raise').to_numpy(float)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    check('control_matrix_rank11', np.linalg.matrix_rank(B) == 11)
    raw_activities, resid = project_signatures(M, slopes, names, stable, B, norm)
    check('projected_activities_finite', np.isfinite(resid).all())
    c = c / np.linalg.norm(c)

    contrasts = []
    shifts = []
    rng = np.random.default_rng(SEED)
    for label, weights in build_contrasts():
        vector = contrast_vector(resid, groups, weights)
        signed = float(vector @ c)
        similarity = safe_cos(vector, c)
        record = {'contrast': label, 'contrast_formula': ' + '.join(
                      ('+' if coeff == 1 else '-') + str(key) for key, coeff in weights.items()),
                  'strain': label.split('_')[0],
                  'genes_used': len(genes), 'n_replicates_per_condition': 3,
                  'signed_projection_onto_frozen_03h': signed,
                  'cosine_vs_frozen_03h': similarity,
                  'residual_contrast_norm': float(np.linalg.norm(vector)),
                  'status': 'EXPLORATORY_CONTROLLED_CULTURE_NOT_CF_SPECIFICITY'}
        record.update(bootstrap_contrast(resid, groups, weights, c, rng))
        contrasts.append(record)
        for i, name in enumerate(stable):
            shifts.append({'contrast': label, 'imodulon': name,
                           'selected_original18': bool(name in selected18),
                           'residual_programme_difference': float(vector[i]),
                           'candidate_weight': float(c[i]),
                           'candidate_projection_contribution': float(vector[i] * c[i])})
    contrasts_df = pd.DataFrame(contrasts)
    shifts_df = pd.DataFrame(shifts)
    samples_df = pd.DataFrame(resid, index=stable, columns=manifest['GSM']).T
    samples_df.index.name = 'GSM'

    OUT.mkdir(parents=True, exist_ok=False)
    sample_qc.to_csv(OUT / 'DESCRIPTIVE_24_sample_counts_normalisation_QC.csv', index=False)
    group_qc_df.to_csv(OUT / 'DESCRIPTIVE_8_group_replicate_correlation_QC.csv', index=False)
    pca.to_csv(OUT / 'DESCRIPTIVE_24_sample_transcriptome_PCA_QC.csv', index=False)
    pd.DataFrame({'GeneID': mask,
                  'usable_across_all_samples': [g in common for g in mask],
                  'globally_duplicated_source_tag': [g in collision_union for g in mask]}
                 ).to_csv(OUT / 'FROZEN_4923_gene_coverage_inventory.csv', index=False)
    samples_df.to_csv(OUT / 'DESCRIPTIVE_24_sample_frozen108_residual_activities.csv')
    contrasts_df.to_csv(OUT / 'DESCRIPTIVE_within_strain_10_perturbation_contrasts.csv', index=False)
    shifts_df.to_csv(OUT / 'DESCRIPTIVE_all108_programme_shifts_by_contrast.csv', index=False)
    provenance = [{'input': key, 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
                  for key, path in FILES.items()]
    pd.DataFrame(provenance + inputs).to_csv(OUT / 'INPUT_PROVENANCE_SHA256.csv', index=False)
    (OUT / 'DESIGN_AND_INTERPRETATION_LIMITS.txt').write_text(
        'Frozen model unchanged. No trained/adapted signature or additional latent components.\n'
        'These are 24 laboratory-culture samples, NOT 24 patients. One PAO1 background and one FRD1 background.\n'
        'Only within-strain contrasts are used: four simple contrasts plus an interaction per strain.\n'
        'Three replicates per cell: bootstrap intervals are descriptive and can understate uncertainty.\n'
        'Treatment and harvest time may be confounded. FRD1 counts aligned to PAO1 reference may incur mapping bias.\n'
        'Median-ratio-normalised log2(count+1) differences are NOT clinical log2TPM and are NOT DESeq2 LFC.\n'
        'The frozen linear projection/pseudoinverse is recalculated on the COMMON observed gene mask.\n'
        'Sample-level coordinates and signed projection are NOT comparable in scale to phase7 clinical comparisons.\n'
        'No BH or formal differential-expression claims; the perturbation results are descriptive.\n'
        'Consistent signed responses would indicate assay responsiveness, NOT CF-environment specificity.\n'
        'No group/sample selection based on outcome; all 24 are reported, without QC exclusions.\n'
        'FeatureCounts processed counts and source group labels derive from GEO GSE280261.\n'
    )
    (OUT / 'AUDIT_08B_STATUS.txt').write_text(STATUS + '\n')

    print('\n========== PUBLIC PERTURBATION SAMPLE QC ==========', flush=True)
    cols = ['strain', 'genotype', 'tobramycin_5ug_ml',
            'min_replicate_correlation', 'median_replicate_correlation']
    print(group_qc_df[cols].round(4).to_string(index=False))
    print('All 24 samples; common mask', len(genes), '/4923',
          '; PCA variance PC1/PC2:', np.round(explain, 4).tolist(), flush=True)
    print('\n========== FROZEN CF SIGNATURE: WITHIN-STRAIN PERTURBATION TEST ==========', flush=True)
    cols = ['contrast', 'signed_projection_onto_frozen_03h', 'cosine_vs_frozen_03h',
            'bootstrap_signed_ci025', 'bootstrap_signed_ci975',
            'bootstrap_signed_fraction_positive']
    print(contrasts_df[cols].round(4).to_string(index=False))
    print('\n========== TOP PROGRAMME CONTRIBUTIONS FOR EACH STRAIN WT TOBRAMYCIN ==========', flush=True)
    for label in ('PAO1_WT_tobramycin', 'FRD1_WT_tobramycin'):
        df = shifts_df.loc[shifts_df['contrast'] == label].copy()
        df['abs_contribution'] = df['candidate_projection_contribution'].abs()
        print(label)
        cols = ['imodulon', 'residual_programme_difference',
                'candidate_weight', 'candidate_projection_contribution', 'selected_original18']
        print(df.nlargest(12, 'abs_contribution')[cols].round(4).to_string(index=False))
    print('\nINTERPRETATION: culture perturbations and treatment/genotype responses, not CF specificity.')
    print('Within-strain normalised log-count contrasts are not directly comparable to patient log2TPM.')
    print('STATUS:', STATUS, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--self-test':
        self_test()
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit('Usage: python audit_08b_GSE280261_frozen_programme_perturbations.py [--self-test]')
