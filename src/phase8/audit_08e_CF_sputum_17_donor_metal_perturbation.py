#!/usr/bin/env python3
"""CF-State / Audit 08e: PAO1 ex-vivo CF sputum, paired combined-metal perturbation.

Neff et al. (2024), Microbiology Spectrum 12:e03157-23, PRJNA1006084.
Source: https://github.com/samlo777/pa-sputum-transcriptome

Uses public AUTHOR-PROVIDED gene count matrix (Salmon) and metadata, NOT raw reads.
Uses a strictly frozen, previously established 116-iModulon/108-stable/
11-controlled-state operator and the original 03h consensus. No fitting.

CRITICAL: 17 different donor sputum environments; bacteria are PAO1 spike-ins,
NOT transcriptomes of native bacteria from those 17 patients. The intervention
supplied Zn+Fe+Mn together, so no single-metal causal attribution is possible.

Run from ~/projects/CFstate, after placing in src/phase8/:
  python -u src/phase8/audit_08e_CF_sputum_17_donor_metal_perturbation.py
  python src/phase8/audit_08e_CF_sputum_17_donor_metal_perturbation.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path.cwd().resolve()
P4 = ROOT / 'results/phase4'
P8 = ROOT / 'results/phase8'
RAW = ROOT / 'data/raw/phase8_Neff2024_exvivo_CF_sputum'
OUT = P8 / 'audit_08e_Neff2024_paired_CF_sputum_metal_supplementation'
BASE = ('https://raw.githubusercontent.com/samlo777/pa-sputum-transcriptome/'
        'main/Figure%20Production%20Code/Data_Files/')
COUNT_NAME = 'RNAseq__sputum_metals_counts.csv'
META_NAME = 'Donor_Metadata_V2.csv'
SAMPLE_RX = re.compile(r'^Spu(\d{3})(_M)?_S\d+_R2_001\.salmon$')
GENE_RX = re.compile(r'^PA\d{4}$')
SEED = 20261014
BOOT = 20000
STATUS = 'CFSTATE_PHASE8_08E_17_DONOR_COMBINED_METALS_DESCRIPTIVE_COMPLETE_REVIEW_REQUIRED'
MODEL = {
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'controls': P4 / 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'selected18': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/conservatively_replicated_programmes.csv',
}
FOCUS = ['Zur-2', 'Zur-1', 'CzcR', 'PvdS', 'Iron acquisition', 'FoxR', 'FpvR', 'PmrA', 'CysB-2']


def require(name, ok):
    if not bool(ok):
        print(name + ' FAIL', flush=True)
        raise RuntimeError('CHECK FAILED: ' + name)
    print(name + ' PASS', flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def fetch(name):
    """Download exact authors' public file once, using an atomic rename."""
    dest = RAW / name
    if dest.is_file() and dest.stat().st_size > 100:
        print('Reusing verified nonempty local file:', dest, flush=True)
        return dest
    RAW.mkdir(parents=True, exist_ok=True)
    tmp = RAW / (name + '.incomplete')
    if tmp.exists():
        tmp.unlink()
    request = urllib.request.Request(BASE + name,
                                     headers={'User-Agent': 'CF-State-public-research-audit/1.0'})
    print('Downloading authors\' public data:', name, flush=True)
    try:
        with urllib.request.urlopen(request, timeout=90) as response, tmp.open('wb') as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        require('download_nonempty_' + name, tmp.stat().st_size > 100)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


def donor_manifest(columns):
    entries = []
    for name in columns:
        mat = SAMPLE_RX.fullmatch(str(name))
        if mat:
            entries.append({'sample': str(name), 'donor_id': str(int(mat.group(1))),
                            'condition': 'METALS' if mat.group(2) else 'UNTREATED'})
        elif str(name).startswith('Spu'):
            raise RuntimeError('Unrecognised sputum sample label: ' + str(name))
    x = pd.DataFrame(entries)
    require('all_17_sputum_donor_pairs',
            len(x) == 35 and x['sample'].is_unique and
            (x.groupby('donor_id')['condition'].apply(lambda a: sorted(a.tolist()))
             .drop('105', errors='ignore').apply(lambda a: a == ['METALS', 'UNTREATED']).all())
            and (x.loc[x['donor_id'] == '105', 'condition'].tolist() == ['UNTREATED']))
    excluded = x.loc[x['donor_id'] == '105'].copy()
    x = x.loc[x['donor_id'] != '105'].copy()
    require('17_donors_34_paired_libraries', len(x) == 34 and
            x['donor_id'].nunique() == 17 and
            (x.groupby(['donor_id', 'condition']).size() == 1).all())
    ids = sorted(x['donor_id'].unique(), key=int)
    x['donor_id'] = pd.Categorical(x['donor_id'], categories=ids, ordered=True)
    x['condition'] = pd.Categorical(x['condition'], categories=['UNTREATED', 'METALS'], ordered=True)
    return x.sort_values(['donor_id', 'condition']).reset_index(drop=True), excluded


def parse_counts(path):
    df = pd.read_csv(path, low_memory=False, encoding='utf-8-sig')
    require('authors_count_table_gene_name_column', 'Name' in df.columns)
    require('authors_count_table_nontrivial', len(df) >= 5300 and len(df.columns) >= 40)
    man, excluded = donor_manifest(df.columns)
    ids = df['Name'].astype('string').str.strip().str.upper()
    valid_id = ids.str.fullmatch(GENE_RX).fillna(False)
    hit = ids[valid_id]
    dup = set(hit[hit.duplicated(keep=False)].dropna())
    if dup:
        print('Globally ambiguous gene IDs excluded:', len(dup), sorted(dup)[:15], flush=True)
    kept = valid_id & ~ids.isin(dup)
    x = df.loc[kept, ['Name'] + man['sample'].tolist()].copy()
    x['Name'] = ids[kept].to_numpy()
    require('unambiguous_5600_gene_annotation', len(x) >= 5000 and x['Name'].is_unique)
    values = x[man['sample'].tolist()].apply(pd.to_numeric, errors='coerce')
    bad = ~np.isfinite(values.to_numpy(float)) | (values.to_numpy(float) < 0)
    require('processed_counts_no_missing_negative_nonfinite', not bad.any())
    # Salmon estimated gene counts may be fractional; do not round them.
    x = values.set_axis(x['Name'].tolist(), axis='index')
    require('input_sample_label_order_consistent', x.columns.tolist() == man['sample'].tolist())
    return x, man, excluded, sorted(dup)


def load_frozen(genes):
    mask = pd.read_csv(MODEL['mask'])['GeneID'].astype(str).tolist()
    require('mask_4923_unique', len(mask) == 4923 and len(set(mask)) == 4923)
    require('observed_frozen_mask_coverage_at_least_4400', len(genes) >= 4400)
    require('observed_gene_subset_of_frozen_mask', set(genes).issubset(set(mask)))
    cal = pd.read_csv(MODEL['calibration']).set_index('imodulon')
    all_names = cal.index.astype(str).tolist()
    stable = pd.read_csv(MODEL['stable'])['imodulon'].astype(str).tolist()
    require('frozen_116_108', len(all_names) == 116 and len(stable) == 108 and
            len(set(all_names)) == 116 and len(set(stable)) == 108 and
            set(stable).issubset(all_names))
    slopes = pd.to_numeric(cal.loc[all_names, 'calibration_slope'], errors='raise').to_numpy(float)
    require('calibration_finite_nonzero', np.isfinite(slopes).all() and (np.abs(slopes) > 1e-12).all())
    candidate_df = pd.read_csv(MODEL['candidate'], index_col=0)
    candidate_df.index = candidate_df.index.astype(str).str.strip()
    require('frozen_consensus_index', set(stable) == set(candidate_df.index))
    candidate = candidate_df.loc[stable, 'two_cohort_consensus_unit'].to_numpy(float)
    require('frozen_candidate_108_unit', np.isfinite(candidate).all() and
            np.isclose(np.linalg.norm(candidate), 1, atol=1e-6))
    selected = set(pd.read_csv(MODEL['selected18'])['imodulon'].astype(str))
    require('frozen_original_selected18', len(selected) == 18 and selected.issubset(set(stable)))
    bframe = pd.read_csv(MODEL['controls'], index_col=0)
    bframe.index = bframe.index.astype(str).str.strip()
    require('frozen_11_control_directions', bframe.shape[1] == 11 and set(stable).issubset(bframe.index))
    b = bframe.loc[stable].apply(pd.to_numeric, errors='raise').to_numpy(float)
    b /= np.linalg.norm(b, axis=0)[None, :]
    require('controlled_reference_rank11', np.linalg.matrix_rank(b) == 11)
    mod = json.loads(MODEL['modulome'].read_text())['M']
    if isinstance(mod, str):
        mod = json.loads(mod)
    if isinstance(mod, dict) and {'index','columns','data'}.issubset(mod):
        f = pd.DataFrame(mod['data'], index=mod['index'], columns=mod['columns'])
    else:
        f = pd.DataFrame(mod)
    f.index = f.index.astype(str)
    f.columns = f.columns.astype(str)
    if set(genes).issubset(f.columns) and set(all_names).issubset(f.index):
        f = f.T
    require('modulome_index_and_programmes', f.index.is_unique and f.columns.is_unique and
            set(genes).issubset(f.index) and set(all_names).issubset(f.columns))
    m = f.loc[genes, all_names].apply(pd.to_numeric, errors='raise').to_numpy(float)
    require('116_projection_full_rank', np.isfinite(m).all() and np.linalg.matrix_rank(m) == 116)
    return {'all': all_names, 'stable': stable, 'slopes': slopes,
            'candidate': candidate, 'selected': selected, 'b': b, 'm': m}


def size_factors(counts):
    """Median-ratio, with identical gene universe for every library."""
    x = np.asarray(counts, dtype=float)
    require('normalisation_nonnegative_finite', np.isfinite(x).all() and (x >= 0).all())
    all_positive = (x > 0).all(axis=1)
    require('normalisation_at_least_500_all_positive_genes', all_positive.sum() >= 500)
    geomean = np.exp(np.log(x[all_positive]).mean(axis=1))
    factors = np.median(x[all_positive] / geomean[:, None], axis=0)
    require('normalisation_factors_valid', np.isfinite(factors).all() and (factors > 0).all())
    factors /= np.exp(np.log(factors).mean())
    return factors, int(all_positive.sum())


def paired_values(values, manifest):
    """Return DONOR x programme METALS-minus-UNTREATED differences."""
    # values is [programme, sample] with columns ordered exactly as manifest.
    groups = manifest['donor_id'].astype(str).tolist()
    cond = manifest['condition'].astype(str).tolist()
    a = {d: {} for d in sorted(set(groups), key=int)}
    for j, (d, g) in enumerate(zip(groups, cond)):
        if g in a[d]:
            raise RuntimeError('Duplicate donor condition ' + d + ', ' + g)
        a[d][g] = j
    diffs = np.array([values[:, v['METALS']] - values[:, v['UNTREATED']]
                      for _, v in a.items()], dtype=float)
    return list(a), diffs


def exact_signflip_paired_mean(vals):
    """Exact two-sided random-sign test on paired MEAN, 2^17 sign patterns."""
    vals = np.asarray(vals, dtype=float)
    if vals.ndim != 1 or not np.isfinite(vals).all() or not (1 <= len(vals) <= 20):
        raise ValueError('Invalid paired effects')
    obs = abs(float(vals.mean()))
    n = len(vals)
    extreme = 0
    total = 1 << n
    bits = 1 << np.arange(n, dtype=np.uint32)
    for start in range(0, total, 4096):
        inds = np.arange(start, min(total, start + 4096), dtype=np.uint32)
        signs = np.where((inds[:, None] & bits[None, :]) != 0, 1.0, -1.0)
        m = np.abs(signs @ vals / n)
        extreme += int((m >= obs - 1e-10).sum())
    return extreme / total


def summarise_effect(name, values, rng):
    z = np.asarray(values, dtype=float)
    require('paired_effect_finite_' + name, len(z) == 17 and np.isfinite(z).all())
    sampled = z[rng.integers(0, len(z), size=(BOOT, len(z)))]
    bootmeans = sampled.mean(axis=1)
    return {'metric': name, 'n_donors': int(len(z)), 'mean_change': float(z.mean()),
            'median_change': float(np.median(z)), 'n_positive': int((z > 0).sum()),
            'n_negative': int((z < 0).sum()),
            'bootstrap_mean_ci025': float(np.quantile(bootmeans, .025)),
            'bootstrap_mean_ci975': float(np.quantile(bootmeans, .975)),
            'exact_signflip_two_sided_p_exploratory': exact_signflip_paired_mean(z)}


def test():
    sample_cols = [f'Spu{d}{m}_S{j}_R2_001.salmon'
                   for j, d in enumerate([102, 106, 108], 1)
                   for m in ('_M', '')] + ['Spu105_S25_R2_001.salmon']
    # Full 17 required in preflight, but individual regex should be exact.
    assert SAMPLE_RX.fullmatch(sample_cols[0]).group(1) == '102'
    assert SAMPLE_RX.fullmatch('Spu102_S29_R2_001.salmon').group(2) is None
    assert SAMPLE_RX.fullmatch('Spu105_S1_R2_001.salmon')
    assert not SAMPLE_RX.fullmatch('Spu102_unknown')
    assert exact_signflip_paired_mean([1., 1., 1.]) == 0.25
    # Check projection/algebra of removing a frozen signature coordinate.
    rng = np.random.default_rng(4)
    b = rng.normal(size=(5, 2))
    x = rng.normal(size=(5, 7))
    resid = x - b @ np.linalg.lstsq(b, x, rcond=None)[0]
    assert np.max(np.abs(b.T @ resid)) < 1e-11
    c = rng.normal(size=5)
    n = c @ resid
    assert np.allclose(n - c[3] * resid[3], np.delete(c, 3) @ np.delete(resid, 3, axis=0))
    print('AUDIT_08E_SELF_TEST_PASS')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        test()
        return

    print('=' * 84)
    print('CF-State Phase 8 — Audit 08e: 17 donor-matched CF sputum metal supplementation')
    print('PAO1 spike-in; metals = Zn + Fe + Mn; frozen model, zero retraining')
    print('=' * 84, flush=True)
    require('working_directory_is_CFstate', P4.is_dir())
    require('new_output_directory', not OUT.exists())
    for label, path in MODEL.items():
        require('frozen_input_' + label, path.is_file())
    count_path = fetch(COUNT_NAME)
    metadata_path = fetch(META_NAME)
    print('Public input SHA256:', COUNT_NAME, digest(count_path), flush=True)
    print('Public input SHA256:', META_NAME, digest(metadata_path), flush=True)
    counts, man, low_depth, duplicate_ids = parse_counts(count_path)
    print('\n========== PAIRED 17 CF SPUTUM SAMPLES ==========', flush=True)
    print(man[['donor_id', 'condition', 'sample']].to_string(index=False), flush=True)
    print('Excluded low-depth donor 105 library:', low_depth['sample'].tolist(), flush=True)

    meta = pd.read_csv(metadata_path, encoding='utf-8-sig', dtype=str)
    require('metadata_donor_ID_column', 'Donor ID' in meta.columns)
    mids = set(meta['Donor ID'].str.strip())
    seqids = set(man['donor_id'].astype(str))
    donor_mismatch = {
        'sequenced_donors_without_metadata': sorted(seqids - mids),
        'metadata_donors_without_sequencing': sorted(mids - seqids),
    }
    print('\n========== DONOR METADATA ID AUDIT ==========', flush=True)
    print(donor_mismatch, flush=True)
    # No inferred cross-ID mapping, and no phenotype/metal regression in 08e.

    mask = pd.read_csv(MODEL['mask'])['GeneID'].astype(str).tolist()
    included = [g for g in mask if g in counts.index]
    print('\nFROZEN PAO1 GENE COVERAGE:', len(included), '/ 4923', flush=True)
    frozen = load_frozen(included)
    matrix = counts.loc[included, man['sample'].tolist()].to_numpy(float)
    factors, nref = size_factors(matrix)
    norm = np.log2(1 + matrix / factors[None, :])
    require('normalised_log2_counts_finite', np.isfinite(norm).all())
    all_activities = (np.linalg.pinv(frozen['m']) @ norm) * frozen['slopes'][:, None]
    index = {g: j for j, g in enumerate(frozen['all'])}
    raw = all_activities[[index[g] for g in frozen['stable']], :]
    b = frozen['b']
    residual = raw - b @ np.linalg.lstsq(b, raw, rcond=None)[0]
    require('residual_orthogonal_to_frozen_controls', np.max(np.abs(b.T @ residual)) < 1e-6)
    donors, perprog = paired_values(residual, man)
    donors_raw, perprog_raw = paired_values(raw, man)
    require('donor_order_same_raw_and_residual', donors == donors_raw)
    c = frozen['candidate']
    zind = frozen['stable'].index('Zur-2')
    pvd = frozen['stable'].index('PvdS')
    czc = frozen['stable'].index('CzcR')
    delta = perprog @ c
    zurcontrib = c[zind] * perprog[:, zind]
    no_zur = delta - zurcontrib  # NO renormalisation: accounting identity
    require('Zur_ablation_accounting_identity', np.allclose(delta, zurcontrib + no_zur))
    rng = np.random.default_rng(SEED)
    metrics = {
        'FROZEN_CF_SIGNED_SCORE_METALS_MINUS_UNTREATED': delta,
        'ZUR2_CONTROLLED_PROGRAMME_METALS_MINUS_UNTREATED': perprog[:, zind],
        'ZUR2_RAW_CALIBRATED_PROGRAMME_METALS_MINUS_UNTREATED': perprog_raw[:, zind],
        'PvdS_CONTROLLED_PROGRAMME_METALS_MINUS_UNTREATED': perprog[:, pvd],
        'CzcR_CONTROLLED_PROGRAMME_METALS_MINUS_UNTREATED': perprog[:, czc],
        'ZUR2_CONTRIBUTION_TO_CF_SCORE': zurcontrib,
        'FROZEN_CF_SCORE_WITHOUT_ZUR2_CONTRIBUTION': no_zur,
    }
    summary = pd.DataFrame([summarise_effect(k, v, rng) for k, v in metrics.items()])
    paired = pd.DataFrame({'donor_id': donors, 'signed_score_change': delta,
                           'cosine_change_vs_CF_frozen': [float(row @ c / np.linalg.norm(row))
                            if np.linalg.norm(row) > 1e-10 else np.nan for row in perprog],
                           'Zur2_controlled_change': perprog[:, zind],
                           'Zur2_raw_change': perprog_raw[:, zind],
                           'PvdS_controlled_change': perprog[:, pvd],
                           'CzcR_controlled_change': perprog[:, czc],
                           'Zur2_signed_contribution': zurcontrib,
                           'CF_score_change_without_Zur2': no_zur})
    prog = pd.DataFrame({'imodulon': frozen['stable'],
                         'original_18_selected': [g in frozen['selected'] for g in frozen['stable']],
                         'frozen_CF_weight': c,
                         'mean_controlled_change': perprog.mean(axis=0),
                         'median_controlled_change': np.median(perprog, axis=0),
                         'mean_raw_calibrated_change': perprog_raw.mean(axis=0),
                         'mean_CF_score_contribution': c * perprog.mean(axis=0),
                         'n_donors_controlled_change_positive': (perprog > 0).sum(axis=0)})
    require('programme_contributions_sum_to_score',
            np.isclose(prog['mean_CF_score_contribution'].sum(), delta.mean(), atol=1e-6))
    prog = prog.sort_values('mean_CF_score_contribution', key=lambda a: a.abs(), ascending=False)
    qc = man.copy()
    qc['size_factor'] = factors
    qc['model_genes'] = len(included)
    qc['genes_detected_GE1'] = (matrix >= 1).sum(axis=0)
    qc['library_size_on_model_genes'] = matrix.sum(axis=0)

    # Create result directory only after core inputs are known to be valid.
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'AUDIT_08E_STATUS.txt').write_text('INCOMPLETE_PENDING_EXPORT\n')
    paired.to_csv(OUT / 'FROZEN_17_donor_paired_CF_score_and_metal_programme_changes.csv', index=False)
    summary.to_csv(OUT / 'DESCRIPTIVE_paired_bootstrap_and_exact_signflip.csv', index=False)
    prog.to_csv(OUT / 'DESCRIPTIVE_108_programme_mean_paired_changes.csv', index=False)
    qc.to_csv(OUT / 'FROZEN_34_sputum_sample_QC_and_pairing.csv', index=False)
    pd.DataFrame({'GeneID': included}).to_csv(OUT / 'FROZEN_08e_gene_support.csv', index=False)
    pd.DataFrame([{'source': k, 'path': str(v), 'sha256': digest(v)}
                  for k, v in {**MODEL, 'public_counts': count_path,
                               'public_donor_metadata': metadata_path,
                               'script': Path(__file__).resolve()}.items()]).to_csv(
        OUT / 'FROZEN_input_SHA256_provenance.csv', index=False)
    (OUT / 'DONOR_METADATA_JOIN_WARNINGS.json').write_text(json.dumps(donor_mismatch, indent=2) + '\n')
    (OUT / 'EXCLUDED_low_depth_or_ambiguous_features.json').write_text(json.dumps({
        'excluded_low_depth_sample': low_depth['sample'].tolist(),
        'ambiguous_PAO1_gene_IDs': duplicate_ids,
        'unmatched_metadata': donor_mismatch,
        'normalisation': 'log2(1 + Salmon counts / median-ratio size factors from 34 sputum libraries)',
        'number_all_positive_reference_genes': nref,
        'no_Zur2_ablation_definition': 'Drop Zur-2 signed contribution without refitting or renormalising'
    }, indent=2) + '\n')

    print('\n========== FROZEN CF-STATE: METAL SUPPLEMENTATION ==========', flush=True)
    print(summary.loc[summary['metric'].isin([
        'FROZEN_CF_SIGNED_SCORE_METALS_MINUS_UNTREATED',
        'FROZEN_CF_SCORE_WITHOUT_ZUR2_CONTRIBUTION'])].round(5).to_string(index=False), flush=True)
    print('\n========== METAL PROGRAMMES: PAIRED RESPONSE ==========', flush=True)
    print(summary.loc[summary['metric'].isin([
        'ZUR2_CONTROLLED_PROGRAMME_METALS_MINUS_UNTREATED',
        'ZUR2_RAW_CALIBRATED_PROGRAMME_METALS_MINUS_UNTREATED',
        'PvdS_CONTROLLED_PROGRAMME_METALS_MINUS_UNTREATED',
        'CzcR_CONTROLLED_PROGRAMME_METALS_MINUS_UNTREATED'])].round(5).to_string(index=False), flush=True)
    print('\n========== TOP 108 PROGRAMME CONTRIBUTIONS ==========', flush=True)
    print(prog.head(16).round(5).to_string(index=False), flush=True)
    print('\n========== DONOR-PAIRED EFFECTS ==========', flush=True)
    print(paired.round(4).to_string(index=False), flush=True)
    print('\nINTERPRETATION LIMITS: combined Zn+Fe+Mn, not individual-metal effects; PAO1',
          'spiked into patient sputum; ex vivo, not native clinical PA expression; donor-',
          'pairing within study, no independent treatment replication per donor; no model',
          'refitting; normalised Salmon gene counts, not patient log2TPM.', sep='\n', flush=True)
    (OUT / 'AUDIT_08E_STATUS.txt').write_text(STATUS + '\n')
    print('\nSTATUS:', STATUS, flush=True)


if __name__ == '__main__':
    main()
