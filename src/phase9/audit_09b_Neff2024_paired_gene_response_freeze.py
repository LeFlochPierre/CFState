#!/usr/bin/env python3
"""CF-State Phase 09b: freeze measured paired CF-sputum metal responses in gene space.

Reconstructs donor-paired PAO1 gene-level responses from PRJNA1006084 Salmon
estimated counts and *frozen* Audit 08e2 sample pairing/size factors.
No network is trained; no iModulon projection, CF specificity or Zn/Fe/Mn
separation is inferred. Fails closed on missing/ambiguous sample and gene IDs.

Run:
    python src/phase9/audit_09b_Neff2024_paired_gene_response_freeze.py --inspect-input
    python src/phase9/audit_09b_Neff2024_paired_gene_response_freeze.py --self-test
    python src/phase9/audit_09b_Neff2024_paired_gene_response_freeze.py

Requirements: Python 3.10+, numpy, pandas.
Output: results/phase9/audit_09b_Neff2024_paired_gene_response_freeze (new only).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('results/phase9/audit_09b_Neff2024_paired_gene_response_freeze')
MASK = Path('results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv')
PAIR = Path('results/phase8/audit_08e2_Neff2024_paired_CF_sputum_metal_supplementation/FROZEN_34_sputum_sample_QC_and_pairing.csv')
GENES = Path('results/phase8/audit_08e2_Neff2024_paired_CF_sputum_metal_supplementation/FROZEN_08e_gene_support.csv')
COUNTS = Path('data/raw/phase8_Neff2024_exvivo_CF_sputum/RNAseq__sputum_metals_counts.csv')
PREVIOUS = Path('results/phase9/audit_09a_perturbation_evidence_preflight/AUDIT_09A_STATUS.txt')
EXPECTED_STATUS = 'PHASE09A_EVIDENCE_PREFLIGHT_COMPLETE_NOT_MODEL_FITTED'


def need(ok, msg):
    if not ok:
        raise RuntimeError('AUDIT_09B_CHECK_FAILED: ' + str(msg))


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def samplekey(value):
    # Whitespace and leading directory components are formatting only. Do NOT
    # infer donors, remove SRA run tokens or fuzzy-match sample identifiers.
    v = str(value).strip().replace('\\', '/').split('/')[-1]
    if v.lower().endswith('.salmon'):
        v = v[:-7]
    return v


def scalar_gene_check(values, expected):
    labels = pd.Series(values, dtype='string').str.strip()
    return (len(labels) == len(expected) and labels.is_unique and
            labels.notna().all() and set(labels) == set(expected))


def load_pairing(root):
    p = root / PAIR
    need(p.is_file(), 'missing frozen 08e2 sample QC and donor pairing: ' + str(p))
    q = pd.read_csv(p, dtype={'sample': str, 'donor_id': str, 'condition': str})
    cols = ['sample', 'donor_id', 'condition', 'size_factor', 'model_genes',
            'genes_detected_GE1', 'library_size_on_model_genes']
    need(set(cols).issubset(q.columns), 'unexpected 08e2 QC columns: ' + str(list(q.columns)))
    need(len(q) == 34, 'expected exactly 34 frozen specimens')
    need(q[cols[:3]].notna().all().all(), 'missing sample, donor or condition')
    q['condition'] = q['condition'].str.strip().str.upper()
    q['donor_id'] = q['donor_id'].str.strip()
    q['sample'] = q['sample'].str.strip()
    q['sample_match_key'] = q['sample'].map(samplekey)
    need(q.sample_match_key.is_unique and q['sample'].is_unique,
         'ambiguous duplicate specimen IDs')
    need(set(q.condition) == {'UNTREATED', 'METALS'},
         'unexpected condition in paired QC: ' + str(q.condition.unique()))
    need(q.donor_id.nunique() == 17 and '105' not in set(q.donor_id),
         'expected exactly 17 donors excluding 105; never invent 102/201 linkage')
    need(not q.donor_id.isin(['', 'nan', 'None']).any(), 'invalid donor label')
    need(all((g.condition.value_counts().reindex(['UNTREATED', 'METALS'], fill_value=0) == 1).all()
             for _, g in q.groupby('donor_id')),
         'each donor must have exactly one untreated and one METALS specimen')
    for col in ['size_factor', 'library_size_on_model_genes', 'model_genes']:
        q[col] = pd.to_numeric(q[col], errors='raise')
    need(np.isfinite(q.size_factor.to_numpy(float)).all() and (q.size_factor > 0).all(),
         'frozen size factors must be finite and positive')
    need((q.model_genes == 4923).all(), '08e2 QC expected 4923 model genes')
    need(np.isfinite(q.library_size_on_model_genes).all() and
         (q.library_size_on_model_genes > 0).all(), 'invalid frozen library sizes')
    return q


def read_mask(root):
    m = pd.read_csv(root / MASK, dtype=str)
    need('GeneID' in m and len(m) == 4923 and m.GeneID.is_unique,
         'frozen 4923-gene mask changed')
    ids = m.GeneID.str.strip().tolist()
    need(all(re.fullmatch(r'PA\d{4}', a) for a in ids), 'unexpected mask gene identifier')
    support = pd.read_csv(root / GENES, dtype=str)
    need(list(support.columns) == ['GeneID'] and scalar_gene_check(support.GeneID, ids),
         '08e2 gene-support set differs from frozen mask')
    return ids


def load_counts(root, genes, qc):
    """Read strictly matched Salmon gene count matrix in wide, long, or transposed form.

    No aggregation, duplicate repair, approximate gene mapping or sample-name
    inference. A detailed failure provides headers for manual schema diagnosis.
    """
    p = root / COUNTS
    need(p.is_file(), 'missing original processed Salmon count matrix: ' + str(p))
    raw = pd.read_csv(p, low_memory=False)
    need(raw.shape[0] > 0 and raw.shape[1] >= 3,
         'unexpected count table shape ' + str(raw.shape))
    all_samples = qc.sample_match_key.tolist()
    expected_samples = set(all_samples)
    possible_ids = []
    for col in raw.columns[:min(7, len(raw.columns))]:
        vals = raw[col].astype('string').str.strip()
        if vals.notna().all() and vals.is_unique:
            overlap = len(set(vals).intersection(genes))
            if overlap >= 4923:
                possible_ids.append(col)
    # Wide: one gene per row, columns are QC samples.
    if len(possible_ids) == 1:
        gene_col = possible_ids[0]
        g = raw[gene_col].astype(str).str.strip()
        need(g.is_unique and g.notna().all(), 'duplicate/missing gene IDs in raw count matrix')
        # Ambiguous sample identifiers must fail rather than be guessed.
        mapped = [(col, samplekey(col)) for col in raw.columns if col != gene_col]
        detected = [(col, key) for col, key in mapped if key in expected_samples]
        keys = [key for _, key in detected]
        need(len(detected) == 34 and len(set(keys)) == 34 and set(keys) == expected_samples,
             'wide raw counts cannot be matched exactly to frozen 34 specimen IDs; '
             'matched=' + str(len(detected)) + '; missing=' +
             str(sorted(expected_samples - set(keys))[:6]) + '; headers=' +
             str(list(raw.columns[:8])))
        need(set(genes).issubset(set(g)),
             'raw wide count matrix missing some of the 4923 frozen genes')
        raw = raw.set_index(g)
        columns = {key: col for col, key in detected}
        aligned = raw.loc[genes, [columns[s] for s in all_samples]].copy()
        aligned.columns = all_samples
        layout = 'wide_genes_by_samples'
    else:
        # Transposed: each row is a specimen, with 4923 PAxxxx columns.
        gene_columns = [c for c in raw.columns if c in set(genes)]
        if len(gene_columns) == 4923:
            id_candidates = []
            for col in raw.columns[:min(7, len(raw.columns))]:
                if col in genes:
                    continue
                skeys = raw[col].astype(str).map(samplekey)
                if len(skeys) == 34 and skeys.is_unique and set(skeys) == expected_samples:
                    id_candidates.append(col)
            need(len(id_candidates) == 1,
                 'transposed sample ID column ambiguous/missing: ' + str(id_candidates))
            c = id_candidates[0]
            rr = raw.assign(_sample_match_key=raw[c].map(samplekey)).set_index('_sample_match_key')
            aligned = rr.loc[all_samples, genes].T.copy()
            layout = 'transposed_samples_by_genes'
        else:
            # Strict long table: one row per (GeneID, sample), count as value.
            names = {str(col).strip().lower(): col for col in raw.columns}
            gname = next((names[x] for x in ['geneid', 'geneid', 'gene', 'gene_id', 'geneid_pa'] if x in names), None)
            sname = next((names[x] for x in ['sample', 'sample_id', 'specimen'] if x in names), None)
            cname = next((names[x] for x in ['count', 'counts', 'numreads', 'est_counts'] if x in names), None)
            need(gname is not None and sname is not None and cname is not None,
                 'unsupported raw count matrix schema; shape=' + str(raw.shape) +
                 '; first columns=' + str(list(raw.columns[:12])))
            rows = raw[[gname, sname, cname]].copy()
            rows[gname] = rows[gname].astype(str).str.strip()
            rows[sname] = rows[sname].map(samplekey)
            rows = rows[rows[gname].isin(genes) & rows[sname].isin(expected_samples)]
            need(not rows.duplicated([gname, sname]).any(), 'duplicate gene/sample in long table')
            need(len(rows) == 4923 * 34,
                 'long count table missing gene/sample cells; got ' + str(len(rows)))
            aligned = rows.pivot(index=gname, columns=sname, values=cname).loc[genes, all_samples]
            layout = 'long_gene_sample_count'
    need(aligned.shape == (4923, 34), 'unexpected aligned count dimensions')
    aligned = aligned.apply(pd.to_numeric, errors='raise')
    vals = aligned.to_numpy(dtype=float)
    need(np.isfinite(vals).all() and (vals >= 0).all(),
         'negative, missing or nonfinite counts in original Salmon estimates')
    need((vals.sum(axis=0) > 0).all(), 'empty specimen count column')
    return aligned, layout, raw.shape


def sign_agreement(a, b):
    agree = np.sign(a) == np.sign(b)
    return float(agree.mean())


def spearman(a, b):
    # Avoid scipy dependency; gene-wise tie-adjusted ranks by pandas.
    ra = pd.Series(a).rank(method='average').to_numpy(dtype=float)
    rb = pd.Series(b).rank(method='average').to_numpy(dtype=float)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def produce(root, output, bootstrap=2000, seed=20261015, inspect_only=False):
    need(bootstrap >= 200 and bootstrap <= 20000, 'bootstrap must lie within 200..20000')
    need((root / PREVIOUS).is_file() and
         (root / PREVIOUS).read_text().strip() == EXPECTED_STATUS,
         'Audit 09a frozen evidence preflight missing/incomplete')
    mask = read_mask(root)
    q = load_pairing(root)
    if inspect_only:
        path = root / COUNTS
        need(path.is_file(), 'raw count file missing')
        raw_header = pd.read_csv(path, nrows=3)
        print('RAW:', path)
        print('RAW file bytes:', path.stat().st_size)
        print('RAW columns:', list(raw_header.columns[:15]))
        print('RAW first rows (max 7 columns):')
        print(raw_header.iloc[:, :7].to_string(index=False))
        print('FROZEN QC sample IDs (first 5):', q['sample'].head().tolist())
        counts, layout, shape = load_counts(root, mask, q)
        print('DETECTED layout:', layout, '| raw shape:', shape)
        print('ALIGNED shape:', counts.shape)
        return
    need(not output.exists(), 'non-overwrite: result directory already exists: ' + str(output))
    counts, layout, raw_shape = load_counts(root, mask, q)
    q = q.set_index('sample_match_key', verify_integrity=True)
    s_keys = q.index.tolist()
    model_sums = counts[s_keys].sum(axis=0).to_numpy(dtype=float)
    qlibs = q.loc[s_keys, 'library_size_on_model_genes'].to_numpy(dtype=float)
    rel_err = np.abs(model_sums - qlibs) / np.maximum(1., qlibs)
    need(np.isfinite(rel_err).all() and rel_err.max() < 0.05,
         'original frozen 08e2 mask library sizes disagree with counts; '
         'max relative difference=' + f'{rel_err.max():.6f}' +
         '. Check sample identifiers/count units; do not silently continue')
    sf = q.loc[s_keys, 'size_factor'].to_numpy(dtype=float)
    normal = counts[s_keys].to_numpy(dtype=float) / sf[None, :]
    log2_norm_plus1 = np.log2(normal + 1.)
    donors = sorted(q.donor_id.unique(), key=lambda x: int(x) if x.isdigit() else x)
    diff_rows = []
    paired_rows = []
    untreated_counts = []
    metals_counts = []
    for donor in donors:
        subset = q[q.donor_id == donor]
        u = subset[subset.condition == 'UNTREATED'].index[0]
        m = subset[subset.condition == 'METALS'].index[0]
        ui, mi = s_keys.index(u), s_keys.index(m)
        diff_rows.append(log2_norm_plus1[:, mi] - log2_norm_plus1[:, ui])
        untreated_counts.append(normal[:, ui])
        metals_counts.append(normal[:, mi])
        paired_rows.append({
            'donor_id': donor,
            'untreated_sample': q.loc[u, 'sample'],
            'metals_sample': q.loc[m, 'sample'],
            'untreated_size_factor_frozen_08e2': float(sf[ui]),
            'metals_size_factor_frozen_08e2': float(sf[mi]),
            'untreated_raw_model_sum': float(model_sums[ui]),
            'metals_raw_model_sum': float(model_sums[mi]),
            'comparison': 'METALS_MINUS_UNTREATED',
            'independent_unit': 'CF_sputum_donor',
        })
    effect = np.stack(diff_rows, axis=0)  # donor x gene
    un = np.stack(untreated_counts, axis=0)
    mt = np.stack(metals_counts, axis=0)
    mean = effect.mean(axis=0)
    median = np.median(effect, axis=0)
    n_up = (effect > 0).sum(axis=0)
    n_down = (effect < 0).sum(axis=0)
    n_zero = (effect == 0).sum(axis=0)
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(len(donors), [1./len(donors)] * len(donors), size=bootstrap)
    bs = weights @ effect / len(donors)
    ci = np.quantile(bs, [.025, .975], axis=0)
    gene_summary = pd.DataFrame({
        'GeneID': mask,
        'mean_paired_log2_normalized_count_plus1_change': mean,
        'median_paired_log2_normalized_count_plus1_change': median,
        'bootstrap_mean_q025': ci[0],
        'bootstrap_mean_q975': ci[1],
        'n_donors_increased': n_up,
        'n_donors_decreased': n_down,
        'n_donors_unchanged': n_zero,
        'fraction_17_donors_matching_majority_sign': np.maximum(n_up, n_down) / len(donors),
        'median_untreated_frozen_sizefactor_normalized_count': np.median(un, axis=0),
        'median_metals_frozen_sizefactor_normalized_count': np.median(mt, axis=0),
        'n_untreated_donors_nonzero': (un > 0).sum(axis=0),
        'n_metals_donors_nonzero': (mt > 0).sum(axis=0),
        'ci_excludes_zero_descriptive': (ci[0] > 0) | (ci[1] < 0),
        'measured_edge_type': 'combined_Zn_Fe_Mn_addition_to_PAO1_gene_expression',
        'measurement_unit': 'mean within-donor log2(sizefactor-normalized estimated count + 1) difference',
        'direct_regulator_gene_binding_proven': False,
        'CF_specific_edge_proven': False,
    })
    effects_wide = pd.DataFrame(effect.T, columns=['donor_' + str(d) + '_METALS_minus_UNTREATED' for d in donors])
    effects_wide.insert(0, 'GeneID', mask)
    donor_stability = []
    for i, donor in enumerate(donors):
        other_mean = np.delete(effect, i, axis=0).mean(axis=0)
        x = effect[i]
        denom = np.linalg.norm(x) * np.linalg.norm(other_mean)
        donor_stability.append({
            'donor_id': donor,
            'cosine_individual_vs_other16_gene_effects': float(np.dot(x, other_mean)/denom) if denom > 0 else float('nan'),
            'spearman_individual_vs_other16_gene_effects': spearman(x, other_mean),
            'fraction_matching_other16_gene_sign': sign_agreement(x, other_mean),
            'l2_norm_individual_gene_effect': float(np.linalg.norm(x)),
            'mean_gene_effect': float(x.mean()),
            'note': 'Donor-level directional heterogeneity, not clinical disease phenotype',
        })
    comparisons = []
    for field, prev in [(MASK,'frozen_gene_mask'), (PAIR,'08e2_frozen_pairing'),
                        (GENES,'08e2_frozen_gene_support'), (COUNTS,'original_salmon_counts'),
                        (PREVIOUS,'09a_completed_preflight')]:
        comparisons.append({'role': prev, 'path': str(field), 'sha256': digest(root/field)})
    qc_json = {
        'status':'PHASE09B_MEASURED_GENE_RESPONSE_FREEZE_COMPLETE_NOT_NETWORK_TRAINED',
        'n_genes':len(mask), 'n_donors':len(donors), 'n_samples':len(q),
        'raw_table_shape':list(raw_shape), 'parsed_format':layout,
        'max_relative_library_size_disagreement_vs_08e2':float(rel_err.max()),
        'n_bootstrap':bootstrap, 'random_seed':seed,
        'mean_effect_positive_genes':int((mean > 0).sum()),
        'mean_effect_negative_genes':int((mean < 0).sum()),
        'n_genes_14_of_17_or_more_in_same_direction':int((np.maximum(n_up,n_down) >= 14).sum()),
        'n_genes_bootstrap_mean_interval_excludes_zero_descriptive':int(((ci[0] > 0)|(ci[1] < 0)).sum()),
        'donors':donors,
        'design':'17 CF sputum donor pairs; laboratory PAO1 spike-in; combined Zn+Fe+Mn, not separate elements',
        'transformation':'log2(original Salmon estimated counts / original frozen 08e2 size factor + 1)',
        'bootstrap':'descriptive donor resampling of mean gene effects, not independent treatment replication within each donor',
    }
    interpretation = '''# Phase 09b — Measured CF-sputum combined-metal gene responses

This audit freezes signed *within-donor* gene responses for the laboratory PAO1
strain exposed to 17 CF donors' sputum, each untreated versus combined Zn+Fe+Mn.
These are proposed **intervention -> gene response evidence**, not a fitted network,
regulator-target interactions, or evidence of CF specificity.

- Donors, paired sample identities and size factors are reused without repair
  from frozen 08e2. Excluded donor 105 remains excluded. No fabricated donor
  102/201 metadata mapping is introduced.
- Contrast = METALS minus UNTREATED. Gene effect = per-donor difference of
  log2(Salmon estimated count / frozen 08e2 size factor + 1). This **is not** a
  DESeq2 shrinkage log-fold change or a quantified molecular effect per metal.
- Each donor supplies one pair, not independently replicated cultures for each
  condition. Bootstrap intervals assess between-donor descriptive variability.
- All genes are retained without training-data-based edge threshold selection.
  Majority-sign fraction and descriptive intervals are *evidence annotations*.
- Cross-study microarray effects remain in their **own native measurement scale**
  in Phase 09a. Do not pool raw magnitudes or perform global PCA on uncalibrated
  cross-platform matrices.
- This same public metal study was interrogated during Phase 08; results here
  cannot become an untouched external validation. Donor hold-out tests will
  measure within-study reproducibility, not cross-study generalisation.
- A network must separate intervention-response links, curated regulatory
  interactions, clinical gene-expression transfer, and independent validation.

Next decision: examine the measured gene-level effects and donor heterogeneity,
then specify an evaluation split and network-edge construction rules *before*
using these measurements to fit the first CF-relevant response network.
'''
    # Reserve output atomically and fail without overwriting an older audit.
    output.mkdir(parents=True, exist_ok=False)
    effects_wide.to_csv(output / 'FROZEN_17_donors_x_4923_genes_paired_effects.csv', index=False)
    gene_summary.to_csv(output / 'FROZEN_4923_gene_combined_metals_response_summary.csv', index=False)
    pd.DataFrame(paired_rows).to_csv(output / 'FROZEN_17_donor_pairing_and_sizefactors.csv', index=False)
    pd.DataFrame(donor_stability).to_csv(output / 'EXPLORATORY_donor_leave_one_out_gene_response_stability.csv', index=False)
    pd.DataFrame(comparisons).to_csv(output / 'INPUT_SHA256_provenance.csv', index=False)
    (output/'QC_INPUT_AND_EFFECTS.json').write_text(json.dumps(qc_json, indent=2) + '\n')
    (output/'INTERPRETATION_LIMITS.md').write_text(interpretation)
    (output/'AUDIT_09B_STATUS.txt').write_text(qc_json['status']+'\n')
    print('AUDIT 09B completed:', output)
    print('Input table format:', layout, '| raw shape:', raw_shape)
    print('Aligned genes:',len(mask),'| frozen donor pairs:',len(donors),'| samples:',len(q))
    print('QC maximum difference from frozen 08e2 library sums:',f'{rel_err.max()*100:.4f}%')
    print('Gene effects: positive mean=',int((mean>0).sum()),
          'negative mean=',int((mean<0).sum()))
    print('>=14/17 donors same sign:',qc_json['n_genes_14_of_17_or_more_in_same_direction'])
    print('Descriptive bootstrap CI excludes zero:',
          qc_json['n_genes_bootstrap_mean_interval_excludes_zero_descriptive'])
    print('No model fitted, no causal regulator-gene edges inferred.')


def self_test():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        g = [f'PA{i:04d}' for i in range(1,4924)]
        for x in [MASK, GENES, PAIR, COUNTS, PREVIOUS]:
            (root/x).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'GeneID': g}).to_csv(root/MASK,index=False)
        pd.DataFrame({'GeneID': g[::-1]}).to_csv(root/GENES,index=False)
        (root/PREVIOUS).write_text(EXPECTED_STATUS+'\n')
        samples = []
        table = {'Geneid':g + ['PA9999']}
        for d in range(101,119):
            if d == 105:
                continue
            for cond in ['UNTREATED','METALS']:
                name = f'Spu{d}_{cond}.salmon'
                # A small known response; the frozen 08e2 library sums are
                # calculated on precisely the same synthetic mask counts.
                vals = 5 + (np.arange(4924) % 17).astype(float)
                if cond == 'METALS':
                    vals[:100] *= 2
                table[name] = vals
                samples.append({'sample':name, 'donor_id':str(d), 'condition':cond,
                                'size_factor':1., 'model_genes':4923,
                                'genes_detected_GE1':4923,
                                'library_size_on_model_genes':vals[:4923].sum()})
        pd.DataFrame(samples).to_csv(root/PAIR,index=False)
        pd.DataFrame(table).to_csv(root/COUNTS,index=False)
        produce(root, root/OUTPUT, bootstrap=200, seed=1)
        o = root/OUTPUT
        a = pd.read_csv(o/'FROZEN_4923_gene_combined_metals_response_summary.csv')
        need(len(a) == 4923 and a.iloc[0]['n_donors_increased'] == 17,
             'synthetic measured response not recovered')
        need(a.iloc[200]['n_donors_unchanged'] == 17,
             'synthetic nonresponding gene not recovered')
        need((pd.read_csv(o/'FROZEN_17_donors_x_4923_genes_paired_effects.csv').shape == (4923,18)),
             'synthetic donor x gene effects wrong shape')
        try:
            produce(root, root/OUTPUT, bootstrap=200)
        except RuntimeError as exc:
            need('non-overwrite' in str(exc), 'wrong repeat-run rejection')
        else:
            raise AssertionError('output overwrite was not rejected')
        # Fail closed if the frozen sample IDs do not match a raw count header.
        raw_file = root/COUNTS
        raw = pd.read_csv(raw_file)
        old = raw.columns[1]
        raw = raw.rename(columns={old:'unrelated_salmon_column'})
        raw.to_csv(raw_file,index=False)
        try:
            load_counts(root,g,load_pairing(root))
        except RuntimeError as exc:
            need('cannot be matched' in str(exc), 'wrong missing sample rejection')
        else:
            raise AssertionError('missing sample must fail')
    print('SELF_TEST_PASSED: 4923 genes, 17 donor pairs, frozen size factors,')
    print('  deterministic response, descriptive bootstrap, SHA and no-overwrite;')
    print('  mismatched source sample identifiers fail closed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--inspect-input', action='store_true', help='Read-only raw matrix header/layout check; no audit output')
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        produce(args.root.resolve(), args.root.resolve()/OUTPUT,
                bootstrap=args.bootstrap, seed=args.seed, inspect_only=args.inspect_input)


if __name__ == '__main__':
    main()
