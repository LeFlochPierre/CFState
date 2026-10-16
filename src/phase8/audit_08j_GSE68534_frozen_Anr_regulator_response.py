#!/usr/bin/env python3
"""CF-State Phase 8 / Audit 08j: frozen Anr knockout response in GSE68534.

Reads GSE68534's *raw read values* worksheet already downloaded by Audit 08i.
Conservatively reconciles PAO1 locus tags and uniquely annotated gene symbols,
then computes PAO1/J215 *within-strain* Delta-anr-minus-WT contrasts. The frozen
116-iModulon projection, 108 stable programmes, 11 controlled-state directions,
and original CF candidate are unchanged. All interpretation is descriptive:
only two biological replicates per group; Anr deletion under 1% oxygen is not
an aerobic-versus-anaerobic experiment or a CF-specificity validation.

Run from CFstate project root.  Self-test: python SCRIPT --self-test
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd
import openpyxl

ROOT = Path.cwd().resolve()
P4 = ROOT / 'results/phase4'
P8 = ROOT / 'results/phase8'
A8I = P8 / 'audit_08i_GSE68534_Anr_low_oxygen_preflight'
OUT = P8 / 'audit_08j_GSE68534_frozen_Anr_regulator_response'
FILES = {
    '08i_provenance': A8I / 'DATA_PROVENANCE.json',
    'raw_workbook': ROOT / 'data/raw/phase8_public_anr_processed/GSE68534/GSE68534_ProcessedDataMatrices.xlsx.gz',
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'gff': ROOT / 'references/phase6/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'controls': P4 / 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'registry18': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/conservatively_replicated_programmes.csv',
    'script': Path(__file__).resolve(),
}
EXPECTED = [
    ('PAO1', 'WT', 1), ('PAO1', 'WT', 2),
    ('PAO1', 'delta_anr', 1), ('PAO1', 'delta_anr', 2),
    ('J215', 'WT', 1), ('J215', 'WT', 2),
    ('J215', 'delta_anr', 1), ('J215', 'delta_anr', 2),
]
MARKER_SYMBOLS = ('anr', 'narG', 'narH', 'nirS', 'nirM', 'norB', 'nosZ',
                  'hcnA', 'hemN', 'rhlA', 'lasB', 'pvdS')
PA_TAG = re.compile(r'(?<![A-Za-z0-9])(PA\d{4})(?:\.\d+)?(?![A-Za-z0-9])', re.I)


def check(name, condition):
    if not bool(condition):
        raise RuntimeError('CHECK FAILED: ' + name)
    print(name + ' PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def normalized_header(value):
    x = '' if value is None else str(value).strip().lower()
    x = x.replace('∆', 'delta').replace('Δ', 'delta').replace('δ', 'delta')
    x = x.replace(' ', '').replace('-', '_')
    x = re.sub(r'[^a-z0-9_]', '', x)
    m = re.fullmatch(r'(pao1|j215)_(wt|delta_?anr)_(\d+)', x)
    if not m:
        return None
    return (m.group(1).upper(), 'WT' if m.group(2) == 'wt' else 'delta_anr', int(m.group(3)))


def parse_gff_symbols(path):
    alias_loci = defaultdict(set)
    loci = set()
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line or line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] not in ('gene', 'CDS'):
                continue
            attrs = {}
            for item in fields[8].split(';'):
                if '=' in item:
                    k, v = item.split('=', 1)
                    attrs[k] = unquote(v).strip()
            tags = [v for v in (attrs.get('locus_tag'), attrs.get('old_locus_tag')) if v]
            ids = set()
            for tag in tags:
                m = re.fullmatch(r'(PA\d{4})(?:\.\d+)?', tag, re.I)
                if m:
                    ids.add(m.group(1).upper())
            # If the same annotated feature provides contradictory PAO1 IDs, skip it.
            if len(ids) != 1:
                continue
            locus = next(iter(ids))
            loci.add(locus)
            symbol = attrs.get('gene', '').strip()
            if symbol and not PA_TAG.fullmatch(symbol):
                alias_loci[symbol.casefold()].add(locus)
    unique = {s: next(iter(v)) for s, v in alias_loci.items() if len(v) == 1}
    ambiguous = {s: sorted(v) for s, v in alias_loci.items() if len(v) > 1}
    print('GFF PAO1 loci:', len(loci), 'unique aliases:', len(unique),
          'ambiguous aliases:', len(ambiguous), flush=True)
    check('gff_annotated_loci_ge4500', len(loci) >= 4500)
    return unique, ambiguous


def reconcile_gene(raw, aliases, ambiguous):
    s = '' if raw is None else str(raw).strip()
    if not s:
        return None, 'empty_identifier'
    tags = {m.upper() for m in PA_TAG.findall(s)}
    if len(tags) > 1:
        return None, 'multiple_locus_tokens'
    if len(tags) == 1:
        return next(iter(tags)), 'direct_PAO1_locus'
    if s.casefold() in ambiguous:
        return None, 'ambiguous_gene_symbol'
    if s.casefold() in aliases:
        return aliases[s.casefold()], 'unique_GFF_gene_symbol'
    return None, 'unmapped_identifier'


def parse_raw_workbook(workbook, aliases, ambiguous):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unknown extension is not supported*')
        book = openpyxl.load_workbook(io.BytesIO(workbook), read_only=True, data_only=True)
    try:
        check('raw_values_sheet_found', 'Raw Read Values' in book.sheetnames)
        ws = book['Raw Read Values']
        rows = ws.iter_rows(values_only=True)
        header = next(rows)
        positions = {}
        for j, cell in enumerate(header):
            key = normalized_header(cell)
            if key is not None:
                check('header_no_duplicate_' + '_'.join(map(str, key)), key not in positions)
                positions[key] = j
        check('eight_expected_sample_columns', set(positions) == set(EXPECTED))
        raw_ids, mapped, count_rows, flagged = [], defaultdict(list), [], []
        invalid_loci = set()
        method_counts = Counter()
        sample_keys = EXPECTED[:]
        for excel_row, row in enumerate(rows, start=2):
            if not row or row[0] is None:
                continue
            raw = str(row[0]).strip()
            if excel_row == 2 and raw.casefold() == 'feature id':
                continue
            gene, method = reconcile_gene(raw, aliases, ambiguous)
            method_counts[method] += 1
            if gene is None:
                flagged.append({'excel_row': excel_row, 'raw_feature_id': raw,
                                'resolved_GeneID': '', 'reason': method})
                continue
            values = []
            invalid_reason = ''
            for key in sample_keys:
                j = positions[key]
                cell = row[j] if j < len(row) else None
                try:
                    if cell is None or str(cell).strip() == '':
                        raise ValueError('missing')
                    val = float(cell)
                    if not np.isfinite(val) or val < 0 or not val.is_integer():
                        raise ValueError('invalid_numeric_count')
                    values.append(val)
                except (ValueError, TypeError, OverflowError):
                    invalid_reason = 'invalid_or_missing_count_in_' + '_'.join(map(str, key))
                    break
            if invalid_reason:
                invalid_loci.add(gene)
                flagged.append({'excel_row': excel_row, 'raw_feature_id': raw,
                                'resolved_GeneID': gene, 'reason': invalid_reason})
                continue
            item = {'GeneID': gene, 'raw_feature_id': raw, 'excel_row': excel_row,
                    'mapping_method': method, 'counts': np.array(values, dtype=float)}
            mapped[gene].append(item)
            raw_ids.append(raw)
            count_rows.append(item)
        repeated = {gene for gene, recs in mapped.items() if len(recs) > 1}
        repeated.update(invalid_loci)
        for gene in sorted(repeated):
            for item in mapped.get(gene, []):
                flagged.append({'excel_row': item['excel_row'],
                                'raw_feature_id': item['raw_feature_id'],
                                'resolved_GeneID': gene,
                                'reason': 'duplicate_or_invalid_resolved_GeneID'})
        kept = {gene: recs[0]['counts'] for gene, recs in mapped.items() if len(recs) == 1 and gene not in repeated}
        mapping = [{'GeneID': gene, 'raw_feature_id': recs[0]['raw_feature_id'],
                    'excel_row': recs[0]['excel_row'],
                    'mapping_method': recs[0]['mapping_method']}
                   for gene, recs in sorted(mapped.items()) if len(recs) == 1 and gene not in repeated]
        qc = {'total_recognised_rows': len(count_rows), 'unique_resolved_genes': len(kept),
              'excluded_duplicated_or_invalid_genes': len(repeated), 'unresolved_or_invalid_rows': len(flagged),
              **{'rows_' + label: int(n) for label, n in method_counts.items()}}
        return kept, flagged, mapping, qc, sample_keys
    finally:
        book.close()


def median_ratio_factors(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim != 2 or not np.isfinite(arr).all() or (arr < 0).any():
        raise RuntimeError('Cannot normalise invalid count matrix')
    positive = (arr > 0).all(axis=1)
    if positive.sum() < 1000:
        raise RuntimeError('Not enough positive-across-samples genes to normalise')
    geometric = np.exp(np.mean(np.log(arr[positive]), axis=1))
    factors = np.median(arr[positive] / geometric[:, None], axis=0)
    if not np.isfinite(factors).all() or np.any(factors <= 0):
        raise RuntimeError('Invalid sample size factors')
    factors = factors / np.exp(np.mean(np.log(factors)))
    return factors, int(positive.sum())


def load_modulome(path, genes, names):
    obj = json.loads(path.read_text())
    check('modulome_has_M', isinstance(obj, dict) and 'M' in obj)
    data = obj['M']
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and {'index', 'columns', 'data'}.issubset(data):
        table = pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])
    else:
        table = pd.DataFrame(data)
    table.index = table.index.map(str)
    table.columns = table.columns.map(str)
    if set(genes).issubset(table.columns) and set(names).issubset(table.index):
        table = table.T
    check('modulome_gene_programme_IDs',
          table.index.is_unique and table.columns.is_unique and
          set(genes).issubset(table.index) and set(names).issubset(table.columns))
    M = table.loc[genes, names].apply(pd.to_numeric, errors='raise').to_numpy(float)
    check('modulome_entries_finite', np.isfinite(M).all())
    return M


def project_frozen(M, slopes, names, stable, controls, expr):
    raw = (np.linalg.pinv(M) @ expr) * slopes[:, None]
    idx = {k: i for i, k in enumerate(names)}
    st = raw[[idx[p] for p in stable], :]
    residual = st - controls @ np.linalg.lstsq(controls, st, rcond=None)[0]
    check('residual_orthogonal_to_11_controls', np.max(np.abs(controls.T @ residual)) < 1e-5)
    return st, residual


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 1e-12 else float('nan')


def write_table(path, rows, columns=None):
    if isinstance(rows, pd.DataFrame):
        rows.to_csv(path, index=False)
        return
    if not rows:
        (path).write_text('no_rows\n')
        return
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def self_test():
    assert normalized_header('PAO1_∆anr_2') == ('PAO1', 'delta_anr', 2)
    assert normalized_header('J215_Δanr_1') == ('J215', 'delta_anr', 1)
    assert normalized_header('PAO1_WT_2') == ('PAO1', 'WT', 2)
    aliases = {'dnaa': 'PA0001', 'dnan': 'PA0002'}
    assert reconcile_gene('dnaA', aliases, {}) == ('PA0001', 'unique_GFF_gene_symbol')
    assert reconcile_gene('PA0001', aliases, {}) == ('PA0001', 'direct_PAO1_locus')
    assert reconcile_gene('foo', aliases, {})[0] is None
    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = 'Raw Read Values'
    sheet.append([''] + [k[0]+'_'+('WT' if k[1]=='WT' else '∆anr')+'_'+str(k[2]) for k in EXPECTED])
    sheet.append(['Feature ID'] + ['Unique gene reads']*8)
    sheet.append(['dnaA'] + [10]*8)
    sheet.append(['PA0002'] + [25]*8)
    sheet.append(['dnaN'] + [28]*8)  # duplicate PA0002 must be excluded
    sheet.append(['unknown'] + [15]*8)
    mem = io.BytesIO()
    wb.save(mem)
    vals, flagged, mapping, qc, keys = parse_raw_workbook(mem.getvalue(), aliases, {})
    assert keys == EXPECTED and list(vals) == ['PA0001']
    assert qc['excluded_duplicated_or_invalid_genes'] == 1
    assert len([v for v in flagged if v['reason']=='duplicate_or_invalid_resolved_GeneID']) == 2
    arr = np.array([[100, 200, 150, 120], [30, 40, 50, 80], [15, 20, 25, 30]], float)
    # Small matrix test uses higher-level elementary normalisation equation.
    g = np.exp(np.mean(np.log(arr), axis=1))
    factors = np.median(arr/g[:, None], axis=0)
    assert (factors > 0).all()
    B = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    R = np.array([[1., 2.], [2., 3.], [3., 4.], [4., 5.]])
    out = R - B @ np.linalg.lstsq(B, R, rcond=None)[0]
    assert np.allclose(B.T @ out, 0)
    print('AUDIT_08J_SELF_TEST_PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    print('=' * 78)
    print('CF-State Phase 8 — Audit 08j: frozen Anr knockout response')
    print('PAO1/J215 WT versus delta-anr in 1%-oxygen biofilms; 2 replicates/group')
    print('GENE SYMBOL RECONCILIATION FIRST / NO MODEL FITTING / NOT CF VALIDATION')
    print('=' * 78, flush=True)
    check('CFstate_results_present', P4.is_dir() and P8.is_dir())
    check('new_output_directory', not OUT.exists())
    for label, path in FILES.items():
        check('input_' + label, path.is_file())
    OUT.mkdir(parents=True)
    try:
        provenance = json.loads(FILES['08i_provenance'].read_text())
        check('08i_source_GSE68534', provenance['series'] == 'GSE68534')
        check('same_compressed_workbook_as_08i', provenance['compressed_sha256'] == sha256(FILES['raw_workbook']))
        mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
        check('frozen_mask_4923_unique', len(mask) == 4923 and len(set(mask)) == 4923)
        aliases, ambiguous = parse_gff_symbols(FILES['gff'])
        with gzip.open(FILES['raw_workbook'], 'rb') as stream:
            data = stream.read()
        check('decompressed_workbook_matches_08i',
              hashlib.sha256(data).hexdigest() == provenance['xlsx_sha256'])
        mapped, problems, mapping, qc, keys = parse_raw_workbook(data, aliases, ambiguous)
        write_table(OUT/'DIAGNOSTIC_source_gene_mapping.csv', mapping)
        write_table(OUT/'DIAGNOSTIC_unmapped_duplicate_or_bad_rows.csv', problems)
        (OUT/'DIAGNOSTIC_parser_summary.json').write_text(json.dumps(qc, indent=2)+'\n')
        print('\n========== GSE68534 GENE IDENTIFIER RECONCILIATION ==========', flush=True)
        for k, v in qc.items():
            print(k, v, flush=True)
        genes = [g for g in mask if g in mapped]
        print('USABLE FROZEN GENE COVERAGE:', len(genes), '/4923', flush=True)
        check('common_unambiguous_genes_at_least_4400', len(genes) >= 4400)
        raw_counts = np.stack([mapped[gene] for gene in genes], axis=0)
        check('count_matrix_8_valid', raw_counts.shape == (len(genes), 8) and
              np.isfinite(raw_counts).all() and (raw_counts >= 0).all())
        manifest = pd.DataFrame(keys, columns=['strain','genotype','replicate'])
        check('eight_sample_design', len(manifest) == 8 and
              manifest.groupby(['strain','genotype']).size().eq(2).all())
        norm = np.zeros_like(raw_counts, dtype=float)
        factors = np.zeros(8, dtype=float)
        refs = {}
        for strain in ('PAO1','J215'):
            ix = np.flatnonzero(manifest['strain'].to_numpy() == strain)
            sf, n = median_ratio_factors(raw_counts[:, ix])
            factors[ix] = sf
            norm[:, ix] = np.log2(1. + raw_counts[:, ix] / sf[None,:])
            refs[strain] = n
            print('NORMALISATION:', strain, 'positive-all-gene reference:', n,
                  'size factors:', np.round(sf,4).tolist(), flush=True)
        check('log2_normalised_expression_finite', np.isfinite(norm).all())
        sample_qc = manifest.copy()
        sample_qc['detected_genes_GE1'] = (raw_counts>=1).sum(axis=0)
        sample_qc['assigned_counts_usable_genes'] = raw_counts.sum(axis=0)
        sample_qc['within_strain_size_factor'] = factors
        sample_qc['median_ratio_ref_genes'] = sample_qc['strain'].map(refs)
        cors = np.corrcoef(norm.T)
        for (strain, genotype), sub in manifest.groupby(['strain','genotype']):
            ix = sub.index.to_numpy()
            value = float(cors[ix[0],ix[1]])
            sample_qc.loc[ix, 'replicate_pair_correlation'] = value
        check('replicate_QC_correlations_finite', np.isfinite(sample_qc['replicate_pair_correlation']).all())
        write_table(OUT/'FROZEN_GSE68534_8_sample_QC.csv', sample_qc)
        print('\n========== GSE68534: PUBLIC SAMPLE QC ==========', flush=True)
        print(sample_qc.to_string(index=False, float_format=lambda v: f'{v:.4f}'))

        cal = pd.read_csv(FILES['calibration']).set_index('imodulon')
        names = cal.index.astype(str).tolist()
        stable = pd.read_csv(FILES['stable'])['imodulon'].astype(str).tolist()
        check('frozen_116_108', len(names)==116 and len(stable)==108 and
              len(set(stable))==108 and set(stable).issubset(set(names)))
        slopes = pd.to_numeric(cal.loc[names,'calibration_slope'], errors='raise').to_numpy(float)
        check('slopes_finite_nonzero', np.isfinite(slopes).all() and (np.abs(slopes)>1e-12).all())
        M = load_modulome(FILES['modulome'], genes, names)
        check('observed_116_projection_full_rank', np.linalg.matrix_rank(M)==116)
        ref = pd.read_csv(FILES['controls'], index_col=0)
        ref.index = ref.index.astype(str).str.strip()
        check('11_frozen_control_directions', ref.shape[1]==11 and set(stable).issubset(ref.index))
        B = ref.loc[stable].apply(pd.to_numeric, errors='raise').to_numpy(float)
        B = np.array(B, copy=True)
        B = B / np.linalg.norm(B, axis=0, keepdims=True)
        check('11_control_rank', np.linalg.matrix_rank(B)==11)
        candidate_frame = pd.read_csv(FILES['candidate'], index_col=0)
        candidate_frame.index = candidate_frame.index.astype(str).str.strip()
        check('frozen_CF_108_identifiers', candidate_frame.index.is_unique and
              set(stable)==set(candidate_frame.index))
        c = candidate_frame.loc[stable,'two_cohort_consensus_unit'].to_numpy(float)
        check('frozen_CF_candidate_unit', np.isfinite(c).all() and
              abs(np.linalg.norm(c)-1)<1e-6)
        selected18 = set(pd.read_csv(FILES['registry18'])['imodulon'].astype(str))
        check('18_original_programmes', len(selected18)==18 and selected18.issubset(stable))
        raw_activity, residual = project_frozen(M,slopes,names,stable,B,norm)
        groups = {(strain,genotype): sub.index.to_numpy(dtype=int)
                  for (strain,genotype),sub in manifest.groupby(['strain','genotype'])}
        contrasts = []
        changes = {}
        for strain in ('PAO1','J215'):
            wi = groups[(strain,'WT')]
            di = groups[(strain,'delta_anr')]
            diff = residual[:,di].mean(axis=1) - residual[:,wi].mean(axis=1)
            rawdiff = raw_activity[:,di].mean(axis=1)-raw_activity[:,wi].mean(axis=1)
            changes[strain] = diff
            score = float(diff @ c)
            pair_scores = [float((residual[:,d]-residual[:,w])@c) for d in di for w in wi]
            contrasts.append({'strain':strain,'contrast':'delta_anr_minus_WT',
                  'n_WT':2,'n_delta_anr':2,
                  'signed_delta_CF_candidate':score,
                  'cosine_delta_vs_frozen_CF_candidate':cosine(diff,c),
                  'descriptive_pairwise_min_signed_score':min(pair_scores),
                  'descriptive_pairwise_max_signed_score':max(pair_scores),
                  'descriptive_pairwise_fraction_positive':np.mean(np.array(pair_scores)>0),
                  'controlled_programme_difference_norm':float(np.linalg.norm(diff))})
            rows = []
            for j,p in enumerate(stable):
                rows.append({'strain':strain,'contrast':'delta_anr_minus_WT','imodulon':p,
                             'raw_calibrated_programme_change':float(rawdiff[j]),
                             'controlled_residual_programme_change':float(diff[j]),
                             'frozen_candidate_weight':float(c[j]),
                             'signed_CF_score_contribution':float(c[j]*diff[j]),
                             'selected_original18':p in selected18})
            table = pd.DataFrame(rows)
            table.to_csv(OUT/('DESCRIPTIVE_108_programme_changes_'+strain+'.csv'),index=False)
        outcome = pd.DataFrame(contrasts)
        write_table(OUT/'DESCRIPTIVE_CF_signature_delta_anr_minus_WT.csv',outcome)
        print('\n========== FROZEN CF SIGNATURE: ANR KNOCKOUT ==========',flush=True)
        print(outcome.to_string(index=False, float_format=lambda v:f'{v:.5f}'))
        anr_names = [p for p in stable if re.fullmatch(r'Anr(?:-\d+)?',p,re.I)]
        check('Anr_programmes_found', len(anr_names)>=2)
        rows = []
        for strain in ('PAO1','J215'):
            table = pd.read_csv(OUT/('DESCRIPTIVE_108_programme_changes_'+strain+'.csv'))
            rows.extend(table[table['imodulon'].isin(anr_names)].to_dict('records'))
        anr_df = pd.DataFrame(rows)
        write_table(OUT/'DESCRIPTIVE_Anr_programme_changes.csv',anr_df)
        print('\n========== ANR PROGRAMMES: WITHIN-STRAIN DELETION RESPONSE ==========',flush=True)
        print(anr_df[['strain','imodulon','raw_calibrated_programme_change',
                      'controlled_residual_programme_change','signed_CF_score_contribution']]
              .to_string(index=False,float_format=lambda v:f'{v:.5f}'))
        print('\n========== OTHER TOP CONTRIBUTIONS TO CF SCORE CHANGE ==========',flush=True)
        for strain in ('PAO1','J215'):
            table = pd.read_csv(OUT/('DESCRIPTIVE_108_programme_changes_'+strain+'.csv'))
            top = table.iloc[np.argsort(-np.abs(table['signed_CF_score_contribution'].to_numpy()))[:12]]
            print(strain)
            print(top[['imodulon','controlled_residual_programme_change',
                       'signed_CF_score_contribution','selected_original18']]
                  .to_string(index=False,float_format=lambda v:f'{v:.4f}'))
        # Pre-registered biological interpretation hints, not post-hoc selected hits.
        marker_rows = []
        for symbol in MARKER_SYMBOLS:
            gene = aliases.get(symbol.casefold())
            if gene not in genes:
                marker_rows.append({'gene_symbol':symbol,'PAO1_locus':gene or '',
                                    'strain':'','delta_anr_minus_WT_log2norm':'',
                                    'status':'marker_unmapped_or_excluded'})
                continue
            gi = genes.index(gene)
            for strain in ('PAO1','J215'):
                wi=groups[(strain,'WT')]
                di=groups[(strain,'delta_anr')]
                marker_rows.append({'gene_symbol':symbol,'PAO1_locus':gene,
                                    'strain':strain,
                                    'delta_anr_minus_WT_log2norm':float(norm[gi,di].mean()-norm[gi,wi].mean()),
                                    'status':'descriptive_two_replicates'})
        write_table(OUT/'DESCRIPTIVE_predefined_Anr_related_gene_markers.csv',marker_rows)
        print('\n========== PREDEFINED ANR-RELATED GENE MARKERS ==========',flush=True)
        print(pd.DataFrame(marker_rows).to_string(index=False))
        concordance=cosine(changes['PAO1'],changes['J215'])
        (OUT/'SUMMARY_LIMITATIONS.txt').write_text(
            'GSE68534 is an Anr knockout within 1%-O2 biofilms, not an O2-dose perturbation.\n'
            'Two biological replicates per group; no inferential significance or disease-specificity claim.\n'
            'PAO1 and J215 are independent strain backgrounds but NOT independent patient airway transcriptomes.\n'
            'Both sample sets are mapped in the supplied PAO1 gene-reference matrix.\n'
            'Exact unique GFF3 symbols only; ambiguous/duplicate/invalid rows excluded.\n'
            'DESeq-style log2(1+count/size-factor) differs from clinical log2TPM.\n'
            'Full 108-D signature is frozen. This analysis does not fit an oxygen axis.\n'
            'Overlap with original aeruPRECISE reference training not independently resolved.\n')
        inputs=[{'input':k,'path':str(v),'sha256':sha256(v)} for k,v in FILES.items()]
        (OUT/'INPUT_SHA256.json').write_text(json.dumps(inputs,indent=2)+'\n')
        print('\nANR DELETION RESPONSE: cosine(PAO1 108-D,J215 108-D)=',round(concordance,5))
        print('Do not interpret as direct oxygen-level response or clinical CF specificity.')
        print('STATUS: CFSTATE_PHASE8_08J_ANR_GENE_RECONCILED_FROZEN_PERTURBATION_DESCRIPTIVE_COMPLETE')
    except Exception:
        print('STATUS: CFSTATE_PHASE8_08J_INCOMPLETE_DIAGNOSTICS_PRESERVED',file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
