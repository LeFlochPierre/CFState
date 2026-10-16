#!/usr/bin/env python3
"""CF-State Phase 8 / Audit 08d: frozen iModulon response to zinc-acquisition perturbations.

INPUTS (all previously frozen/downloaded):
  GSE310904: PA14 WT versus delta-cntL, three RNA-seq replicates per group.
  GSE60177: PAO1 WT versus delta-znuA, biological material pooled into ONE
            sequencing library per group. No uncertainty estimable from GEO RNA.
  Phase3 frozen PA14->PAO1 map (never rebuilt or inferred using gene symbols).
  Phase4 frozen 116-iModulon, 108-stable, 11-controls, 03h CF-airway vector.

Outputs descriptive within-study perturbation comparisons ONLY. This does not
validate specificity to CF, classify patients, or retrain a model. PA14 and PAO1
contrast magnitudes are NOT directly comparable due to different gene subsets.

Run from ~/projects/CFstate. `--self-test` exercises parsers and mapping logic.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P3 = ROOT / 'results/phase3'
P4 = ROOT / 'results/phase4'
P8 = ROOT / 'results/phase8'
A8C = P8 / 'audit_08c_public_zinc_perturbation_preflight'
RAW = ROOT / 'data/raw/phase8_public_zinc_processed'
OUT = P8 / 'audit_08d_frozen_zinc_perturbation_projection'
STATUS = 'CFSTATE_PHASE8_08D_FROZEN_ZINC_PERTURBATION_DESCRIPTIVE_REVIEW_REQUIRED'
FILES = {
    '08c_manifest': A8C / 'PUBLIC_8_processed_sample_manifest_and_schema.csv',
    '08c_status': A8C / 'AUDIT_08C_STATUS.txt',
    'pa14_frozen_map': P3 / 'audit_02e_PA14_projection_universe/frozen_PA14_projection_map.csv',
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'controls': P4 / 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'selected18': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/conservatively_replicated_programmes.csv',
    'script': Path(__file__).resolve(),
}
METAL_PROGRAMMES = ['Zur-2', 'Zur-1', 'CzcR', 'PvdS', 'Iron acquisition', 'FoxR', 'FpvR', 'PmrA', 'CysB-2']
SEED = 20261014
N_BOOT = 2500


def check(name, condition):
    if not bool(condition):
        print(name + ' FAIL', flush=True)
        raise RuntimeError('CHECK FAILED: ' + name)
    print(name + ' PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def get_id(value, kind):
    """Extract only explicit locus tags, never match gene symbols by similarity."""
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if kind == 'PAO1':
        m = re.search(r'(?<![A-Z0-9])PA\d{4}(?![A-Z0-9])', s)
    else:
        m = re.search(r'(?<![A-Z0-9])PA14[_-]\d{4,5}(?![A-Z0-9])', s)
    return (m.group(0).replace('-', '_') if m else None)


def parse_count_file(path, strain):
    """Return unique locus count series + QC; never sum duplicate source rows."""
    frame = pd.read_csv(path, sep='\t', compression='gzip', comment='#', low_memory=False)
    if strain == 'PA14':
        if len(frame.columns) < 7:
            raise RuntimeError('PA14 featureCounts table lacks expected columns: ' + str(frame.columns.tolist()))
        id_col = 'Geneid' if 'Geneid' in frame.columns else frame.columns[0]
        count_col = frame.columns[-1]
        tags = frame[id_col].map(lambda x: get_id(x, 'PA14'))
    elif strain == 'PAO1':
        typed = next((c for c in frame.columns if c.strip().lower() == 'type'), None)
        locus_col = next((c for c in frame.columns if c.strip().lower() in ('locus_tag', 'locustag')), None)
        count_col = next((c for c in frame.columns if c.strip().lower() == 'counts'), None)
        if typed is None or locus_col is None or count_col is None:
            raise RuntimeError('PAO1 processed table expected Type, Locus_tag, Counts; observed: ' + str(frame.columns.tolist()))
        # GEO table also contains CDS/transcript features; select ONE gene feature per locus.
        frame = frame.loc[frame[typed].astype(str).str.strip().str.lower() == 'gene'].copy()
        tags = frame[locus_col].map(lambda x: get_id(x, 'PAO1'))
    else:
        raise ValueError('Unexpected strain: ' + strain)
    numbers = pd.to_numeric(frame[count_col], errors='coerce')
    known = pd.DataFrame({'GeneID': tags, 'count': numbers}).dropna(subset=['GeneID'])
    if (known['count'].isna().any() or not np.isfinite(known['count'].to_numpy(float)).all()
            or (known['count'] < 0).any()
            or not np.allclose(known['count'], np.rint(known['count']))):
        raise RuntimeError('Nonfinite, missing, negative or noninteger processed count: ' + str(path))
    dup = set(known.loc[known['GeneID'].duplicated(keep=False), 'GeneID'])
    unique = known.loc[~known['GeneID'].isin(dup)].set_index('GeneID')['count']
    return unique, dup, {
        'raw_rows_selected': len(frame),
        'recognized_locus_rows': len(known),
        'duplicate_locus_ids': len(dup),
        'unique_valid_loci': len(unique),
        'assigned_counts_selected_gene_rows': float(known['count'].sum()),
        'unrecognized_locus_rows': int(tags.isna().sum()),
    }


def best_map_columns(map_table, mask, pa14_source_ids):
    """Choose source and target columns by actual ID content and names; fail on ties.

    The frozen cross-strain map is taken as supplied. We never regenerate a map.
    """
    print('FROZEN PA14 MAP COLUMNS:', list(map_table.columns), flush=True)
    print('FROZEN PA14 MAP FIRST ROWS:')
    print(map_table.head(3).to_string(index=False, max_cols=12), flush=True)
    found = {}
    for kind, ref in [('PA14', pa14_source_ids), ('PAO1', mask)]:
        options = []
        for col in map_table.columns:
            vals = map_table[col].map(lambda x: get_id(x, kind))
            matched = vals.dropna()
            hit = int(matched.isin(ref).sum())
            uniq = int(matched[matched.isin(ref)].nunique())
            if hit < min(100, max(1, len(map_table) // 10)):
                continue
            column_name = str(col).lower()
            label_score = (int('pa14' in column_name) if kind == 'PA14'
                           else int('pao1' in column_name))
            options.append((uniq, hit, label_score, str(col), vals))
        options.sort(key=lambda v: (v[0], v[1], v[2]), reverse=True)
        print('MAP COLUMN CANDIDATES', kind,
              [(x[3], x[0], x[1], x[2]) for x in options[:6]], flush=True)
        if not options:
            raise RuntimeError('Could not locate valid ' + kind + ' locus-tag column in frozen map')
        if len(options) > 1 and options[0][:3] == options[1][:3]:
            if not options[0][4].equals(options[1][4]):
                raise RuntimeError('Ambiguous ' + kind + ' frozen map columns: ' +
                                   options[0][3] + ', ' + options[1][3])
        found[kind] = options[0]
    if found['PA14'][3] == found['PAO1'][3]:
        raise RuntimeError('Source and target mapping columns are identical')
    frame = pd.DataFrame({'PAO1_GeneID': found['PAO1'][4],
                          'PA14_GeneID': found['PA14'][4]})
    frame = frame.dropna().copy()
    frame = frame.loc[frame['PAO1_GeneID'].isin(mask) & frame['PA14_GeneID'].isin(pa14_source_ids)]
    dup1 = frame['PAO1_GeneID'].duplicated(keep=False)
    dup2 = frame['PA14_GeneID'].duplicated(keep=False)
    ambiguous = frame.loc[dup1 | dup2]
    clean = frame.loc[~(dup1 | dup2)].copy()
    clean = clean.sort_values('PAO1_GeneID').reset_index(drop=True)
    check('frozen_map_one_to_one', clean['PAO1_GeneID'].is_unique and clean['PA14_GeneID'].is_unique)
    report = {'PAO1_map_column': found['PAO1'][3], 'PA14_map_column': found['PA14'][3],
              'mapped_pair_rows': len(frame), 'ambiguous_pair_rows_excluded': len(ambiguous),
              'strict_unique_PAO1_PA14_pairs': len(clean)}
    print('FROZEN MAP SUMMARY:', report, flush=True)
    return clean, ambiguous, report


def load_model(genes):
    mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
    check('original_4923_gene_mask', len(mask) == 4923 and len(set(mask)) == 4923)
    cal = pd.read_csv(FILES['calibration']).set_index('imodulon')
    all_names = cal.index.astype(str).tolist()
    stable = pd.read_csv(FILES['stable'])['imodulon'].astype(str).tolist()
    check('frozen_116_108_programmes', len(all_names) == 116 and len(stable) == 108 and
          len(set(stable)) == 108 and set(stable).issubset(all_names))
    slopes = pd.to_numeric(cal.loc[all_names, 'calibration_slope'], errors='raise').to_numpy(float)
    check('frozen_calibration', np.isfinite(slopes).all() and (np.abs(slopes) > 1e-12).all())
    cf = pd.read_csv(FILES['candidate'], index_col=0)
    cf.index = cf.index.astype(str).str.strip()
    check('frozen_CF_candidate', cf.index.is_unique and set(stable) == set(cf.index))
    c = cf.loc[stable, 'two_cohort_consensus_unit'].to_numpy(float)
    check('frozen_CF_candidate_unit_norm', np.isfinite(c).all() and np.isclose(np.linalg.norm(c), 1, atol=1e-6))
    selected18 = set(pd.read_csv(FILES['selected18'])['imodulon'].astype(str))
    check('frozen_18_registry', len(selected18) == 18 and selected18.issubset(stable))
    bdf = pd.read_csv(FILES['controls'], index_col=0)
    bdf.index = bdf.index.astype(str).str.strip()
    check('frozen_11_controls', bdf.shape[1] == 11 and set(stable).issubset(bdf.index))
    b = bdf.loc[stable].apply(pd.to_numeric, errors='raise').to_numpy(float)
    b = b / np.linalg.norm(b, axis=0, keepdims=True)
    check('frozen_control_rank11', np.linalg.matrix_rank(b) == 11)
    obj = json.loads(FILES['modulome'].read_text())
    check('frozen_modulome_M', isinstance(obj, dict) and 'M' in obj)
    raw = obj['M']
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and {'index', 'columns', 'data'}.issubset(raw):
        frame = pd.DataFrame(raw['data'], index=raw['index'], columns=raw['columns'])
    else:
        frame = pd.DataFrame(raw)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if set(genes).issubset(frame.columns) and set(all_names).issubset(frame.index):
        frame = frame.T
    check('frozen_modulome_identifiers', frame.index.is_unique and frame.columns.is_unique and
          set(genes).issubset(frame.index) and set(all_names).issubset(frame.columns))
    m = frame.loc[genes, all_names].apply(pd.to_numeric, errors='raise').to_numpy(float)
    check('frozen_modulome_finite', np.isfinite(m).all())
    check('frozen_projection_full_rank116', np.linalg.matrix_rank(m) == 116)
    return all_names, stable, slopes, c, selected18, b, m


def size_factors(counts):
    counts = np.asarray(counts, dtype=float)
    check('nonnegative_finite_count_matrix', np.isfinite(counts).all() and (counts >= 0).all())
    valid = (counts > 0).all(axis=1)
    if valid.sum() < 500:
        raise RuntimeError('Too few all-positive genes for normalisation: ' + str(valid.sum()))
    geo = np.exp(np.log(counts[valid]).mean(axis=1))
    f = np.median(counts[valid] / geo[:, None], axis=0)
    if not np.isfinite(f).all() or (f <= 0).any():
        raise RuntimeError('Invalid median-ratio size factor')
    f = f / np.exp(np.log(f).mean())
    return f, int(valid.sum())


def cos_safe(vec, unit):
    n = float(np.linalg.norm(vec))
    return float(vec @ unit / n) if n > 1e-10 else float('nan')


def evaluate_study(label, sample_manifest, source_counts, genes, source_gene_for_pao1, rng):
    """Project one study on its frozen gene subset; do not mix PA14/PAO1 scale."""
    order = sample_manifest.sort_values(['genotype', 'GSM']).reset_index(drop=True)
    source_ids = [source_gene_for_pao1[g] for g in genes]
    matrix = np.column_stack([source_counts[row.GSM].reindex(source_ids).to_numpy(float)
                              for row in order.itertuples(index=False)])
    check(label + '_counts_nonnegative_finite', np.isfinite(matrix).all() and (matrix >= 0).all())
    factors, genes_ref = size_factors(matrix)
    lognorm = np.log2(1 + matrix / factors[None, :])
    check(label + '_normalised_expression_finite', np.isfinite(lognorm).all())
    all_names, stable, slopes, candidate, selected18, b, m = load_model(genes)
    calibrated = (np.linalg.pinv(m) @ lognorm) * slopes[:, None]
    idx = {name: i for i, name in enumerate(all_names)}
    stable_idx = [idx[name] for name in stable]
    raw_stable = calibrated[stable_idx]
    residual = raw_stable - b @ np.linalg.lstsq(b, raw_stable, rcond=None)[0]
    check(label + '_controlled_residual_orthogonal', np.max(np.abs(b.T @ residual)) < 1e-5)
    groups = {g: np.flatnonzero(order['genotype'].to_numpy() == g)
              for g in ('WT', 'delta_cntL' if label == 'PA14' else 'delta_znuA')}
    wt = groups['WT']
    mutant_label = [key for key in groups if key != 'WT'][0]
    mut = groups[mutant_label]
    check(label + '_design_n_replicates', len(wt) == (3 if label == 'PA14' else 1) and
          len(mut) == len(wt))
    shift_raw = raw_stable[:, mut].mean(axis=1) - raw_stable[:, wt].mean(axis=1)
    shift_res = residual[:, mut].mean(axis=1) - residual[:, wt].mean(axis=1)
    signed = float(shift_res @ candidate)
    cosine = cos_safe(shift_res, candidate)
    boot_ci = {'bootstrap_signed_ci025': np.nan, 'bootstrap_signed_ci975': np.nan,
               'bootstrap_signed_fraction_positive': np.nan,
               'bootstrap_Zur2_raw_ci025': np.nan, 'bootstrap_Zur2_raw_ci975': np.nan,
               'bootstrap_Zur2_residual_ci025': np.nan, 'bootstrap_Zur2_residual_ci975': np.nan}
    if label == 'PA14':
        vals = np.empty(N_BOOT)
        zr = np.empty(N_BOOT)
        zs = np.empty(N_BOOT)
        if 'Zur-2' not in stable:
            raise RuntimeError('Missing frozen Zur-2')
        z = stable.index('Zur-2')
        for i in range(N_BOOT):
            mi = rng.choice(mut, len(mut), replace=True)
            wi = rng.choice(wt, len(wt), replace=True)
            dv = residual[:, mi].mean(axis=1) - residual[:, wi].mean(axis=1)
            vals[i] = float(dv @ candidate)
            zr[i] = float(raw_stable[z, mi].mean() - raw_stable[z, wi].mean())
            zs[i] = float(dv[z])
        boot_ci = {
            'bootstrap_signed_ci025': float(np.quantile(vals, .025)),
            'bootstrap_signed_ci975': float(np.quantile(vals, .975)),
            'bootstrap_signed_fraction_positive': float((vals > 0).mean()),
            'bootstrap_Zur2_raw_ci025': float(np.quantile(zr, .025)),
            'bootstrap_Zur2_raw_ci975': float(np.quantile(zr, .975)),
            'bootstrap_Zur2_residual_ci025': float(np.quantile(zs, .025)),
            'bootstrap_Zur2_residual_ci975': float(np.quantile(zs, .975)),
        }
    stats = {
        'dataset': 'GSE310904' if label == 'PA14' else 'GSE60177', 'strain': label,
        'contrast': mutant_label + '_minus_WT', 'n_libraries_per_condition': len(wt),
        'independent_RNA_replicates_per_condition': 3 if label == 'PA14' else 0,
        'gene_count_model_overlap': len(genes),
        'size_factor_reference_gene_count': genes_ref,
        'signed_projection_onto_frozen_CF03h': signed,
        'cosine_vs_frozen_CF03h': cosine,
        'status': 'REPLICATED_DESCRIPTIVE_LAB_PERTURBATION' if label == 'PA14'
                  else 'POOLED_ONE_LIBRARY_NO_BIOLOGICAL_UNCERTAINTY',
        **boot_ci,
    }
    programme = pd.DataFrame({
        'dataset': stats['dataset'], 'strain': label, 'contrast': stats['contrast'],
        'imodulon': stable, 'selected_original18': [x in selected18 for x in stable],
        'raw_calibrated_programme_change': shift_raw,
        'controlled_residual_programme_change': shift_res,
        'candidate_weight': candidate,
        'candidate_signed_score_contribution': shift_res * candidate,
    })
    qc = order[['GSM', 'strain', 'genotype', 'replicate']].copy()
    qc['usable_model_genes'] = len(genes)
    qc['size_factor'] = factors
    qc['counts_in_usable_model_genes'] = matrix.sum(axis=0)
    qc['genes_detected_GE1'] = (matrix >= 1).sum(axis=0)
    qc['genes_detected_GE10'] = (matrix >= 10).sum(axis=0)
    if label == 'PA14':
        corr = np.corrcoef(lognorm.T)
        values = []
        for j in range(len(qc)):
            others = [k for k in groups[order.loc[j, 'genotype']]
                      if k != j]
            values.append(float(np.median(corr[j, others])))
        qc['within_group_median_replicate_correlation'] = values
    else:
        qc['within_group_median_replicate_correlation'] = np.nan
    return stats, programme, qc


def self_test():
    with tempfile.TemporaryDirectory() as work:
        base = Path(work)
        pa14 = base / 'toy_pa14.bed.gz'
        pao1 = base / 'toy_pao1.txt.gz'
        import gzip
        with gzip.open(pa14, 'wt') as f:
            f.write('# featureCounts header\nGeneid\tChr\tStart\tEnd\tStrand\tLength\tname\tcount\n')
            f.write('PA14_00010\tchr\t1\t10\t+\t10\tx\t20\n')
            f.write('PA14_00020\tchr\t10\t20\t+\t10\ty\t10\n')
            f.write('PA14_00020\tchr\t30\t40\t+\t10\ty\t12\n')
        with gzip.open(pao1, 'wt') as f:
            f.write('Type\tLocus_tag\tCounts\n')
            f.write('gene\tlocus_tag=PA0001\t13\nCDS\tlocus_tag=PA0001\t14\n')
            f.write('gene\tlocus_tag=PA0002\t21\n')
        a, d1, q1 = parse_count_file(pa14, 'PA14')
        b, d2, q2 = parse_count_file(pao1, 'PAO1')
        assert list(a.index) == ['PA14_00010'] and d1 == {'PA14_00020'}
        assert set(b.index) == {'PA0001', 'PA0002'} and not d2
        assert q1['duplicate_locus_ids'] == 1 and q2['unique_valid_loci'] == 2
        toy = pd.DataFrame({'PAO1_tag': ['PA0001', 'PA0002', 'PA0002'],
                            'PA14_tag': ['PA14_00010', 'PA14_00020', 'PA14_00030']})
        mapped, amb, rep = best_map_columns(toy, {'PA0001', 'PA0002'},
                                            {'PA14_00010', 'PA14_00020', 'PA14_00030'})
        assert len(mapped) == 1 and len(amb) == 2
        assert mapped.iloc[0]['PAO1_GeneID'] == 'PA0001'
        counts = np.array([[10, 20], [5, 10], [0, 0]], dtype=float)
        # tiny example can't pass 500-positive-gene size-factor QC, by design.
        assert np.isfinite(counts).all()
    print('AUDIT_08D_SELF_TEST_PASS')


def main():
    print('=' * 80)
    print('CF-State Phase 8 — Audit 08d: frozen zinc-perturbation mechanistic test')
    print('PA14 delta-cntL and PAO1 delta-znuA; both analyses stay separate')
    print('=' * 80, flush=True)
    check('working_directory_CFstate', (ROOT / 'results/phase4').is_dir())
    check('new_output_directory', not OUT.exists())
    for label, path in FILES.items():
        check('input_' + label, path.is_file())
    check('08c_status_valid', FILES['08c_status'].read_text().strip() ==
          'CFSTATE_PHASE8_08C_ZINC_PUBLIC_PROCESSED_PREFLIGHT_COMPLETE_REVIEW_REQUIRED')
    manifest = pd.read_csv(FILES['08c_manifest']).sort_values(['series', 'GSM']).reset_index(drop=True)
    need = {'GSM', 'series', 'strain', 'genotype', 'replicate', 'path', 'processed_sha256'}
    check('08c_manifest_schema', need.issubset(manifest.columns))
    check('08c_manifest_8_unique_GSM', len(manifest) == 8 and manifest['GSM'].is_unique)
    check('08c_PA14_three_plus_three',
          manifest.query("strain == 'PA14'").groupby('genotype').size().to_dict() ==
          {'WT': 3, 'delta_cntL': 3})
    check('08c_PAO1_one_plus_one',
          manifest.query("strain == 'PAO1'").groupby('genotype').size().to_dict() ==
          {'WT': 1, 'delta_znuA': 1})
    mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
    check('4923_frozen_mask', len(mask) == 4923 and len(set(mask)) == 4923)
    observed = {'PA14': {}, 'PAO1': {}}
    qc_records = []
    collisions = {'PA14': set(), 'PAO1': set()}
    input_hashes = []
    for row in manifest.itertuples(index=False):
        sample_path = ROOT / str(row.path)
        check('08c_processed_file_' + row.GSM, sample_path.is_file())
        check('08c_processed_hash_' + row.GSM, sha256(sample_path) == row.processed_sha256)
        series, dup, qc = parse_count_file(sample_path, row.strain)
        observed[row.strain][row.GSM] = series
        collisions[row.strain].update(dup)
        qc_records.append({'GSM': row.GSM, 'strain': row.strain, **qc})
        input_hashes.append({'input': 'processed_' + row.GSM,
                             'path': str(sample_path.relative_to(ROOT)), 'sha256': sha256(sample_path)})
    check('all_eight_gene_tables', len(observed['PA14']) == 6 and len(observed['PAO1']) == 2)
    common_pa14 = set.intersection(*(set(s.index) for s in observed['PA14'].values()))
    common_pa14 -= collisions['PA14']
    common_pao1 = set.intersection(*(set(s.index) for s in observed['PAO1'].values()))
    common_pao1 -= collisions['PAO1']
    print('Unique common PA14 source genes:', len(common_pa14),
          'and PAO1 source genes:', len(common_pao1), flush=True)
    frozen_map = pd.read_csv(FILES['pa14_frozen_map'], low_memory=False)
    clean_map, ambiguous_map, map_qc = best_map_columns(frozen_map, set(mask), common_pa14)
    sub_pa14 = clean_map.loc[clean_map['PAO1_GeneID'].isin(mask)].copy()
    sub_pa14 = sub_pa14.loc[sub_pa14['PA14_GeneID'].isin(common_pa14)]
    pa14_lookup = dict(zip(sub_pa14['PAO1_GeneID'], sub_pa14['PA14_GeneID']))
    genes_pa14 = [g for g in mask if g in pa14_lookup]
    genes_pao1 = [g for g in mask if g in common_pao1]
    print('FROZEN PA14 ORTHOLOGUE GENE COVERAGE:', len(genes_pa14), '/4923', flush=True)
    print('DIRECT PAO1 GENE COVERAGE:', len(genes_pao1), '/4923', flush=True)
    # PA14 is an orthologue subset and MUST NOT be judged by the direct PAO1 4400-gene gate.
    check('PA14_frozen_orthologue_support_at_least_3500', len(genes_pa14) >= 3500)
    check('PAO1_direct_support_at_least_4400', len(genes_pao1) >= 4400)
    rng = np.random.default_rng(SEED)
    summaries = []
    programmes = []
    all_sample_qc = []
    for strain, genes, lookup in [('PA14', genes_pa14, pa14_lookup),
                                  ('PAO1', genes_pao1, {g: g for g in genes_pao1})]:
        group = manifest.loc[manifest['strain'] == strain].copy()
        summary, prog, qc = evaluate_study(strain, group, observed[strain], genes, lookup, rng)
        summaries.append(summary)
        programmes.append(prog)
        all_sample_qc.append(qc)
    summary_df = pd.DataFrame(summaries)
    programme_df = pd.concat(programmes, ignore_index=True)
    sample_df = pd.concat(all_sample_qc, ignore_index=True)
    metal = programme_df.loc[programme_df['imodulon'].isin(METAL_PROGRAMMES)].copy()
    cross = metal.pivot(index='imodulon', columns='strain',
                        values='raw_calibrated_programme_change').reset_index()
    cross['same_direction_raw_if_nonzero'] = np.sign(cross['PA14']) == np.sign(cross['PAO1'])
    cross['interpretation'] = 'DIFFERENT_GENETIC_PERTURBATIONS_NOT_A_REPLICATION_OF_IDENTICAL_EFFECT'
    OUT.mkdir(parents=True, exist_ok=False)
    summary_df.to_csv(OUT / 'DESCRIPTIVE_two_zinc_perturbation_contrasts.csv', index=False)
    programme_df.to_csv(OUT / 'DESCRIPTIVE_all108_zinc_programme_changes.csv', index=False)
    sample_df.to_csv(OUT / 'DESCRIPTIVE_eight_zinc_RNA_sample_QC.csv', index=False)
    metal.to_csv(OUT / 'DESCRIPTIVE_metal_related_programmes.csv', index=False)
    cross.to_csv(OUT / 'DESCRIPTIVE_cross_experiment_metal_direction_check.csv', index=False)
    pd.DataFrame({'PAO1_GeneID': genes_pa14,
                  'PA14_GeneID': [pa14_lookup[g] for g in genes_pa14]}).to_csv(
        OUT / 'FROZEN_PA14_to_PAO1_mapped_genes_used.csv', index=False)
    ambiguous_map.to_csv(OUT / 'DIAGNOSTIC_excluded_ambiguous_PA14_map_pairs.csv', index=False)
    pd.DataFrame({'input': key, 'path': str(path.relative_to(ROOT)), 'sha256': sha256(path)}
                 for key, path in FILES.items()).to_csv(OUT / 'FROZEN_input_SHA256.csv', index=False)
    pd.DataFrame(input_hashes).to_csv(OUT / 'PUBLIC_processed_sample_SHA256.csv', index=False)
    (OUT / 'FROZEN_map_column_selection_and_QC.json').write_text(json.dumps(map_qc, indent=2) + '\n')
    (OUT / 'INTERPRETATION_LIMITS.txt').write_text(
        'No retraining or re-estimation of the model or CF candidate. Pseudoinverse on observed genes only.\n'
        'The PA14 study uses the existing frozen Phase3 one-to-one PA14->PAO1 map.\n'
        'PA14 (GSE310904): three replicates/group; bootstrap CIs describe sampling within ONE study.\n'
        'PAO1 (GSE60177): ONE pooled RNA library per group; biological replicate variance UNAVAILABLE.\n'
        'Genotype contrasts are mutant minus WT, never directly CF minus non-CF.\n'
        'No assumption of sign of Zur, CzcR or PvdS response; direction is empirical.\n'
        'Difference in means of median-ratio log2(count+1) is NOT a DESeq2 log-fold-change.\n'
        'Raw calibrated programmes are separated from 11-control residualised programmes.\n'
        'Cross-strain effect sizes/scales are not comparable because gene subsets and conditions differ.\n'
        'These genetic perturbations probe zinc uptake and related pleiotropic physiology, NOT CF specificity.\n'
        'Potential overlap with original aeruPRECISE training reference not independently excluded.\n'
    )
    (OUT / 'AUDIT_08D_STATUS.txt').write_text(STATUS + '\n')
    print('\n========== PUBLIC ZINC PERTURBATION SAMPLE QC ==========', flush=True)
    print(sample_df[['GSM', 'strain', 'genotype', 'usable_model_genes',
                     'genes_detected_GE1', 'size_factor',
                     'within_group_median_replicate_correlation']].round(4).to_string(index=False))
    print('\n========== ZINC: FROZEN CF SIGNATURE RESPONSE ==========', flush=True)
    print(summary_df[['dataset', 'contrast', 'gene_count_model_overlap',
                      'signed_projection_onto_frozen_CF03h', 'cosine_vs_frozen_CF03h',
                      'bootstrap_signed_ci025', 'bootstrap_signed_ci975']].round(4).to_string(index=False))
    print('\n========== ZINC: FROZEN METAL PROGRAMME RESPONSES ==========', flush=True)
    cols = ['strain', 'imodulon', 'raw_calibrated_programme_change',
            'controlled_residual_programme_change', 'candidate_weight',
            'candidate_signed_score_contribution']
    print(metal[cols].round(4).to_string(index=False))
    print('\n========== TOP CONTRIBUTORS TO ZINC PERTURBATION SCORES ==========', flush=True)
    for strain in ('PA14', 'PAO1'):
        ranked = programme_df.loc[programme_df['strain'] == strain].copy()
        ranked['absolute_contribution'] = ranked['candidate_signed_score_contribution'].abs()
        print(strain)
        print(ranked.nlargest(10, 'absolute_contribution')[
            ['imodulon', 'raw_calibrated_programme_change',
             'controlled_residual_programme_change', 'candidate_signed_score_contribution']
        ].round(4).to_string(index=False), flush=True)
    print('\nSTATUS:', STATUS, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--self-test':
        self_test()
    elif len(sys.argv) == 1:
        main()
    else:
        raise SystemExit('Usage: python audit_08d_frozen_zinc_perturbation_projection.py [--self-test]')
