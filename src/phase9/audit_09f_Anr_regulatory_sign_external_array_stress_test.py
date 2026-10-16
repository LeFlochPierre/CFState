#!/usr/bin/env python3
"""CF-State Audit 09f: retrospective cross-study Anr KO sign-concordance stress test.

Reads FROZEN 09e4 curated 2011 regulator-target evidence and 09a frozen
GSE67006 array gene effects; all published sign, strain and PMID metadata are
preserved. Comparators include exhaustive genome-position circular shifts,
sign-composition matched random gene controls and genomic-neighborhood summaries.

This is RETROSPECTIVE evaluation: GSE67006 was previously examined in Phase08.
No model fitting, causal TF binding, independent *clinical* validation or CF
specificity established. Strong/weak-effect splits are post-hoc sensitivities.

  python audit_09f_Anr_regulatory_sign_external_array_stress_test.py --self-test
  python audit_09f_Anr_regulatory_sign_external_array_stress_test.py \\
    --gff references/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
import tempfile
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REG_REL = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/FROZEN_CURATED_REGULATORY_ASSOCIATIONS_09E4.csv')
REG_QC_REL = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/QC_09E4.json')
ARR_REL = Path('results/phase9/audit_09a_perturbation_evidence_preflight/FROZEN_08m_native_array_gene_response_evidence.csv')
OUT_REL = Path('results/phase9/audit_09f_Anr_regulatory_sign_external_array_stress_test')
EXPECTED_GFF_SHA = 'e6b8bc796af95342acd5c170ecf5003e75135c894b1c22f8c4b46c51c4bdf7f0'
CHROM = 'NC_002516.2'
ID_PATTERN = re.compile(r'PA\d{4}$')


def need(ok, reason):
    if not ok:
        raise RuntimeError('AUDIT_09F_FAILED: ' + reason)


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def gff_order(path, strict=True):
    need(path.is_file(), 'PAO1 GFF3 not found: ' + str(path))
    if strict:
        need(sha256(path) == EXPECTED_GFF_SHA, 'reference GFF checksum differs from the frozen 09e4 verified file')
    op = gzip.open if str(path).endswith('.gz') else open
    rows = []
    seq_seen = False
    with op(path, 'rt', encoding='utf8', errors='replace') as f:
        for line in f:
            if line.startswith('##sequence-region ' + CHROM + ' '):
                seq_seen = True
            if line.startswith('#'):
                continue
            bits = line.rstrip('\n').split('\t')
            if len(bits) != 9 or bits[0] != CHROM or bits[2].lower() != 'gene':
                continue
            at = {}
            for part in bits[8].split(';'):
                if '=' in part:
                    key, value = part.split('=', 1)
                    at[key] = unquote(value)
            gid = at.get('locus_tag', '').upper()
            if not ID_PATTERN.fullmatch(gid):
                continue  # Dotted loci have different node identity; array excludes them.
            rows.append((gid, int(bits[3]), int(bits[4]), bits[6]))
    need(seq_seen, 'expected chromosome sequence header absent')
    res = pd.DataFrame(rows, columns=['GeneID', 'start', 'end', 'strand'])
    need(res.GeneID.is_unique, 'duplicate reference loci')
    need(len(res) >= (4900 if strict else 30), 'incomplete genomic annotation')
    return res.sort_values(['start', 'GeneID']).reset_index(drop=True)


def load_inputs(root, gff, strict=True):
    rp, ap, qp = root/REG_REL, root/ARR_REL, root/REG_QC_REL
    for p in (rp, ap, qp):
        need(p.is_file(), 'required frozen input missing: ' + str(p))
    qc = json.loads(qp.read_text())
    r = pd.read_csv(rp, dtype=str, keep_default_na=False)
    a = pd.read_csv(ap, dtype={'GeneID': str})
    rneeded = {'regulator_GeneID', 'target_GeneID', 'raw_target',
               'raw_mode_of_regulation', 'original_strain_includes_PAO1',
               'original_strain', 'raw_experimental_evidence', 'original_PMID',
               'source_excel_row', 'raw_operon_annotation'}
    aneeded = {'GeneID', 'contrast_id', 'effect_value_native', 'dataset_id'}
    need(rneeded.issubset(r.columns), '09e4 regulatory schema differs: ' + str(sorted(rneeded-set(r.columns))))
    need(aneeded.issubset(a.columns), '09a array schema differs: ' + str(sorted(aneeded-set(a.columns))))
    if strict:
        need(int(qc['mapped_09e4']) == len(r) == 806, 'expected 806 regulatory evidence rows')
    need(r[['source_excel_row']].drop_duplicates().shape[0] == len(r), 'duplicate source row IDs')
    a = a[a.contrast_id == 'delta_anr_minus_WT_CFBE'].copy()
    need(a.GeneID.is_unique, 'duplicate Anr array gene identifiers')
    a['effect_value_native'] = pd.to_numeric(a['effect_value_native'], errors='raise')
    need(np.isfinite(a.effect_value_native).all(), 'nonfinite array effects')
    if strict:
        need(len(a) == 4920 and set(a.dataset_id) == {'GSE67006'}, 'unexpected frozen GSE67006 dataset')
    coord = gff_order(gff, strict=strict)
    need(set(a.GeneID).issubset(set(coord.GeneID)), 'array gene missing genomic coordinate mapping')
    return r, a, coord, [rp, ap, qp, gff]


def targets_with_signs(reg, arr):
    r = reg[(reg.regulator_GeneID == 'PA1544') &
            (reg.original_strain_includes_PAO1.str.lower() == 'true') &
            (reg.raw_mode_of_regulation.isin(['+', '-'])) &
            (reg.target_GeneID.map(lambda x: bool(ID_PATTERN.fullmatch(x))))].copy()
    r['expected_KO_sign'] = r.raw_mode_of_regulation.map({'+': -1, '-': 1}).astype(int)
    conflict = r.groupby('target_GeneID').expected_KO_sign.nunique()
    bad = set(conflict[conflict != 1].index)
    r = r[~r.target_GeneID.isin(bad)].copy()
    r = r.sort_values('source_excel_row', key=lambda s: pd.to_numeric(s))
    # Collapse multiple literature reports to one testing unit per gene, but
    # preserve PMID evidence in pipe-delimited strings.
    def joinunique(x):
        return '|'.join(sorted({str(v) for v in x if str(v).strip()}))
    t = (r.groupby('target_GeneID', as_index=False)
           .agg(expected_KO_sign=('expected_KO_sign', 'first'),
                raw_target=('raw_target', joinunique),
                literature_mode=('raw_mode_of_regulation', joinunique),
                PMIDs=('original_PMID', joinunique),
                evidence=('raw_experimental_evidence', joinunique),
                source_rows=('source_excel_row', joinunique),
                original_operon=('raw_operon_annotation', joinunique),
                n_literature_rows=('source_excel_row', 'size')))
    m = t.merge(arr[['GeneID', 'effect_value_native']], left_on='target_GeneID',
                right_on='GeneID', how='inner', validate='one_to_one')
    m['observed_KO_sign'] = np.sign(m.effect_value_native).astype(int)
    m['sign_match'] = (m.observed_KO_sign == m.expected_KO_sign)
    m['is_anr_self_target'] = m.target_GeneID == 'PA1544'
    return t, m, sorted(bad)


def background_expectation(subset, signs):
    signs = np.asarray(signs)
    signs = signs[signs != 0]
    if not len(subset) or len(signs) == 0:
        return np.nan
    return float(np.mean([np.mean(signs == pred) for pred in subset.expected_KO_sign]))


def stat(m, background, label):
    return {'subset': label, 'n_genes': len(m),
            'n_matching': int(m.sign_match.sum()),
            'fraction_match': round(float(m.sign_match.mean()), 6) if len(m) else None,
            'background_sign_composition_expectation':
            round(background_expectation(m, background), 6) if len(m) else None,
            'note': 'POST_HOC_SENSITIVITY_NOT_NEW_VALIDATION' if label != 'PRIMARY_ALL_TARGETS'
                    else 'PREVIOUSLY_INSPECTED_RETROSPECTIVE_EXTERNAL_STUDY'}


def neighborhood_groups(m, coord, max_intervening=1):
    # Define a transparent genomic proximity group; these are NOT validated operons.
    genome = coord.sort_values('start').reset_index(drop=True)
    genome['order'] = np.arange(len(genome))
    s = m.merge(genome[['GeneID','start','strand','order']],
                left_on='target_GeneID', right_on='GeneID', validate='one_to_one')
    s = s.sort_values(['order', 'target_GeneID']).reset_index(drop=True)
    groups = []
    groupid = 0
    last_order, last_strand = None, None
    for r in s.itertuples():
        if last_order is None or r.order-last_order > max_intervening+1 or r.strand != last_strand:
            groupid += 1
        groups.append('NH%03d' % groupid)
        last_order, last_strand = r.order, r.strand
    s['genomic_neighborhood_id'] = groups
    s['always_DOWN_correct'] = (s.observed_KO_sign == -1)
    agg = (s.groupby('genomic_neighborhood_id', as_index=False)
           .agg(n_genes=('target_GeneID','size'),
                genes=('target_GeneID', lambda x:'|'.join(x)),
                fractional_agreement=('sign_match','mean'),
                all_DOWN_fraction=('always_DOWN_correct','mean'),
                mean_native_effect=('effect_value_native','mean')))
    return s, agg


def circular_shift_null(m, arr, coord):
    # Exhaustive shifts across measured PAO1 genes in true GFF3 order. A single
    # shift applies to the entire array; contiguous blocks remain correlated.
    ranked = coord[coord.GeneID.isin(arr.GeneID)].sort_values('start').reset_index(drop=True)
    need(len(ranked) == len(arr), 'position/array count mismatch')
    ranks = ranked.GeneID.to_numpy()
    sign_map = dict(zip(arr.GeneID, np.sign(arr.effect_value_native).astype(int)))
    vals = np.array([sign_map[x] for x in ranks], dtype=np.int8)
    positions = {name:i for i,name in enumerate(ranks)}
    ix = np.array([positions[name] for name in m.target_GeneID], dtype=int)
    expected = m.expected_KO_sign.to_numpy(dtype=np.int8)
    n = len(ranks)
    all_shifts = (ix[None,:] + np.arange(n)[:,None]) % n
    null = (vals[all_shifts] == expected[None,:]).mean(axis=1)
    observed = float(m.sign_match.mean())
    need(abs(null[0]-observed)<1e-12, 'zero-shift fails to reproduce actual gene sign agreement')
    ctrl = null[1:]
    p_emp = (1 + int((ctrl >= observed - 1e-12).sum())) / n
    out = {'n_available_offsets_including_true_alignment': n,
           'n_alternative_offsets': n-1,
           'observed_accuracy': observed,
           'shift_null_mean': float(ctrl.mean()),
           'shift_null_median': float(np.median(ctrl)),
           'shift_null_p95': float(np.quantile(ctrl, 0.95)),
           'shift_null_p975': float(np.quantile(ctrl, 0.975)),
           'n_shifts_at_least_observed': int((ctrl >= observed-1e-12).sum()),
           'empirical_upper_tail_p_descriptive': float(p_emp),
           'null_meaning': 'genome-position circular shift of the complete gene-effect vector; diagnostic not independent biological p-value'}
    return out, pd.DataFrame({'shift_index':np.arange(1,n),
                              'fraction_matching':ctrl})


def random_gene_null(m, a, seed=20261015, reps=10000):
    # Matches the FIXED counts of expected signs, compares against sampled
    # genes chosen without replacement from the full array universe.
    rng = np.random.default_rng(seed)
    background = np.sign(a.effect_value_native.to_numpy()).astype(np.int8)
    exp = m.expected_KO_sign.to_numpy(np.int8)
    scores = np.empty(reps)
    for j in range(reps):
        draw = rng.choice(len(background), len(exp), replace=False)
        scores[j] = np.mean(background[draw] == exp)
    return {'n_random_gene_draws':reps, 'mean':float(scores.mean()),
            'p975':float(np.quantile(scores,0.975)),
            'empirical_upper_tail_p_descriptive':float((1+sum(scores >= float(m.sign_match.mean())-1e-12))/(reps+1)),
            'control': 'random measured gene labels without replacement, fixed regulatory predictions; no operon preservation'}, scores


def run(root, gff, dest, strict=True):
    need(not dest.exists(), 'output exists (never overwrite): ' + str(dest))
    reg, arr, coord, sources = load_inputs(root, gff, strict)
    universe, match, conflicting = targets_with_signs(reg,arr)
    need(len(match) > 0 and len(universe) > 0, 'no signed Anr targets overlapping array')
    if strict:
        need(len(universe) == 37 and len(match) == 34,
             'Anr signed target counts differ from verified feasibility output: ' + str((len(universe),len(match))))
    overall = [stat(match, np.sign(arr.effect_value_native), 'PRIMARY_ALL_TARGETS')]
    without = match[~match.is_anr_self_target].copy()
    overall.append(stat(without, np.sign(arr.effect_value_native), 'WITHOUT_ANR_SELF_TARGET'))
    for cutoff in (0.1, 0.25, 0.5, 1.0):
        sel = match[match.effect_value_native.abs() >= cutoff]
        overall.append(stat(sel, np.sign(arr.effect_value_native),
                            'POST_HOC_ABS_NATIVE_EFFECT_GE_' + str(cutoff)))
    tested, groups = neighborhood_groups(match, coord)
    shift_stats, shift_table = circular_shift_null(match, arr, coord)
    control_stats, control_vals = random_gene_null(match, arr)
    observed = float(match.sign_match.mean())
    # Essential trivial baselines on the IDENTICAL target set; an apparent
    # 80% network agreement can be entirely explained by a DOWN-skewed set.
    always_down_n = int((match.observed_KO_sign == -1).sum())
    always_up_n = int((match.observed_KO_sign == 1).sum())
    all_down_correct = match.observed_KO_sign == -1
    network_only_wins = int((match.sign_match & ~all_down_correct).sum())
    naive_only_wins = int((~match.sign_match & all_down_correct).sum())
    by_mode = (match.groupby('literature_mode', as_index=False)
               .agg(n_targets=('target_GeneID', 'size'),
                    n_matching=('sign_match', 'sum'),
                    n_observed_down=('observed_KO_sign', lambda x: int((x == -1).sum())),
                    n_observed_up=('observed_KO_sign', lambda x: int((x == 1).sum()))))
    by_mode['fraction_match'] = by_mode.n_matching / by_mode.n_targets
    summary = {'status':'PHASE09F_RETROSPECTIVE_ANR_KO_REGULATORY_SIGN_STRESS_TEST_NO_FITTING',
               'n_curated_signed_PAO1_Anr_targets':len(universe),
               'n_measured_targets':len(match),
               'n_sign_agreement':int(match.sign_match.sum()),
               'observed_fraction':observed,
               'n_conflicting_sign_targets_excluded':len(conflicting),
               'conflicting_sign_targets':conflicting,
               'always_predict_DOWN_fraction':always_down_n / len(match),
               'always_predict_DOWN_n_correct':always_down_n,
               'always_predict_UP_fraction':always_up_n / len(match),
               'network_extra_correct_vs_all_DOWN':network_only_wins,
               'all_DOWN_extra_correct_vs_network':naive_only_wins,
               'net_correct_advantage_over_all_DOWN':int(match.sign_match.sum())-always_down_n,
               'n_nonself_measured':len(without),
               'n_nonself_matching':int(without.sign_match.sum()),
               'n_genomic_neighborhood_groups':len(groups),
               'equal_neighborhood_weighted_fraction_match':float(groups.fractional_agreement.mean()),
               'equal_neighborhood_weighted_all_DOWN_fraction':float(groups.all_DOWN_fraction.mean()),
               'genomic_neighborhoods_not_verifiable_operons':True,
               'background_sign_expectation':background_expectation(match,np.sign(arr.effect_value_native)),
               'random_gene_label_control':control_stats,
               'circular_shift_control':shift_stats,
               'prior_GSE67006_exploration_means_not_prospectively_blinded':True,
               '2011_literature_reported_regulatory_sign_not_direct_binding_proof':True,
               'frozen_external_study_array_no_model_fitted':True,
               'CF_specificity_not_established':True}
    dest.mkdir(parents=True,exist_ok=False)
    match.to_csv(dest/'FROZEN_ANR_TARGETS_VS_GSE67006_GENE_EFFECTS.csv',index=False)
    by_mode.to_csv(dest/'EVALUATION_BY_2011_REPORTED_SIGN.csv',index=False)
    pd.DataFrame(overall).to_csv(dest/'SENSITIVITY_PRIMARY_AND_EXPLORATORY_THRESHOLDS.csv',index=False)
    tested.to_csv(dest/'GENOMIC_NEIGHBORHOOD_ASSIGNMENT_NOT_OPERON_INFERENCE.csv',index=False)
    groups.to_csv(dest/'GENOMIC_NEIGHBORHOOD_EQUAL_WEIGHT_SUMMARY.csv',index=False)
    shift_table.to_csv(dest/'DIAGNOSTIC_FULL_GENOME_CIRCULAR_SHIFT_NULL.csv',index=False)
    pd.DataFrame({'draw_index':np.arange(len(control_vals)),
                  'fraction_matching':control_vals}).to_csv(dest/'DIAGNOSTIC_RANDOM_GENE_CONTROLS.csv',index=False)
    (dest/'QC_09F.json').write_text(json.dumps(summary,indent=2)+'\n')
    pd.DataFrame([{'path':str(p.resolve()), 'sha256':sha256(p)} for p in sources]).to_csv(dest/'INPUT_SHA256_PROVENANCE.csv',index=False)
    notes = '''# Phase 09f — Retrospective regulator-sign comparison\n\nThe 2011 curated Anr +/- target signs were projected to expected delta-anr minus WT expression signs and compared to the frozen GSE67006 microarray response, with exact PAO1 locus IDs and preserved source evidence. 09a and 09e4 inputs were never changed.\n\nGSE67006 is a distinct experimental dataset from the literature source, but was used in earlier CF-State Phase 08 analyses: this is not prospectively blinded validation. Literature evidence itself includes heterogeneous assays, not exclusively direct-binding studies. Neighboring genes and operons are correlated.\n\nThe exhaustive circular-shift null preserves chromosomal adjacency of observed gene effects but does not reproduce regulatory-target annotation biases, gene properties or sample-level variation. Its p-value is diagnostic, not donor- or biological-replicate-based. Equal-neighborhood-weight accuracy is a sensitivity metric, not an independence-corrected significance test.\n\nAll effect-magnitude cutoffs were explored after observing the primary result: post-hoc only, not feature-selection or a higher-performing preregistered test. Array effect units are within-study transformed GPL84 differences, not RNA-seq fold changes. No model fitting, direct TF-binding inference, clinical validation or CF specificity is claimed.\n'''
    (dest/'INTERPRETATION_LIMITS.md').write_text(notes)
    (dest/'AUDIT_09F_STATUS.txt').write_text(summary['status']+'\n')
    print('\n'+'='*80)
    print('CF-STATE AUDIT 09F | ANR CURATED NETWORK vs INDEPENDENT EXPERIMENT')
    print('='*80)
    print('2011 signed PAO1 Anr targets:',len(universe), '| measured GSE67006:',len(match), '| conflicting modes excluded:',len(conflicting))
    print('Primary target agreement: %d/%d = %.2f%%' % (int(match.sign_match.sum()),len(match),100*observed))
    print('Genome-wide background expected agreement: %.2f%%' % (100*summary['background_sign_expectation']))
    print('CRITICAL all-DOWN baseline on SAME targets: %d/%d = %.2f%%' % (always_down_n, len(match),100*always_down_n/len(match)))
    print('Always-UP baseline on SAME targets: %d/%d = %.2f%%' % (always_up_n,len(match),100*always_up_n/len(match)))
    print('Network extra wins vs always-DOWN:',network_only_wins,
          '| always-DOWN extra wins:',naive_only_wins,
          '| NET gain in correct target signs:',int(match.sign_match.sum())-always_down_n)
    print('EVALUATION BY LITERATURE REPORTED SIGN (+ activation, - repression):')
    print(by_mode.to_string(index=False))
    print('Excluding Anr self target: %d/%d = %.2f%%' % (int(without.sign_match.sum()),len(without),100*without.sign_match.mean()))
    print('Genomic neighborhoods (NOT verified operons):',len(groups),
          '| equal-neighborhood network=%.2f%%' % (100*groups.fractional_agreement.mean()),
          '| equal-neighborhood all-DOWN=%.2f%%' % (100*groups.all_DOWN_fraction.mean()))
    print('Random gene-label null: mean=%.2f%% upper97.5%%=%.2f%% diagnostic p=%.5f' % (
        100*control_stats['mean'],100*control_stats['p975'],control_stats['empirical_upper_tail_p_descriptive']))
    print('Chromosomal circular-shift null: mean=%.2f%% upper97.5%%=%.2f%% diagnostic p=%.5f' % (
        100*shift_stats['shift_null_mean'],100*shift_stats['shift_null_p975'],shift_stats['empirical_upper_tail_p_descriptive']))
    print('\nPOST-HOC EFFECT-MAGNITUDE SENSITIVITY (not predeclared significance cutoffs):')
    print(pd.DataFrame(overall)[['subset','n_genes','n_matching','fraction_match','background_sign_composition_expectation']].to_string(index=False))
    print('\nGENOMIC NEIGHBORHOODS (partial listing, ordered genomic positions):')
    print(groups.to_string(index=False))
    print('\nALL MEASURED TARGETS (effect is native array signal, not log fold change):')
    fullcols=['target_GeneID','raw_target','literature_mode','expected_KO_sign','effect_value_native','observed_KO_sign','sign_match','PMIDs']
    print(match[fullcols].sort_values('target_GeneID').to_string(index=False))
    print('\nMISMATCHED LITERATURE TARGETS:')
    cols=['target_GeneID','raw_target','expected_KO_sign','effect_value_native','PMIDs']
    mistakes=match.loc[~match.sign_match,cols].sort_values('target_GeneID')
    print(mistakes.to_string(index=False) if len(mistakes) else '  none')
    print('\nREVIEW: cross-study expression sign check only, no model fitting, not independent clinical or blind validation.')
    print('Source files unchanged. Outputs:',dest)
    return summary


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        p=root/REG_REL
        p.parent.mkdir(parents=True)
        records=[]
        for i in range(1,38):
            records.append({'regulator_GeneID':'PA1544','target_GeneID':f'PA{i:04d}',
                            'raw_target':f't{i}','raw_mode_of_regulation': '+' if i%3 else '-',
                            'original_strain_includes_PAO1':'True',
                            'original_strain':'PAO1','raw_experimental_evidence':'RT-PCR',
                            'original_PMID':'12345678','source_excel_row':str(i),
                            'raw_operon_annotation':''})
        pd.DataFrame(records).to_csv(p,index=False)
        (root/REG_QC_REL).write_text(json.dumps({'mapped_09e4':37})+'\n')
        a=root/ARR_REL
        a.parent.mkdir(parents=True)
        # 100 reference genes, 37 signed targets; PA1544 is a regulator outside
        # the synthetic response targets and hence not a self-target test here.
        ar=[]
        for i in range(1,101):
            sign=1 if i%3==0 else -1
            if i%13==0: sign=-sign
            ar.append({'GeneID':f'PA{i:04d}','contrast_id':'delta_anr_minus_WT_CFBE',
                       'dataset_id':'GSE67006','effect_value_native':sign*(0.1+i/100)})
        pd.DataFrame(ar).to_csv(a,index=False)
        g=root/'reference.gff'
        with g.open('w') as f:
            f.write('##gff-version 3\n##sequence-region NC_002516.2 1 1000000\n')
            for i in range(1,101):
                f.write('NC_002516.2\tRefSeq\tgene\t%d\t%d\t.\t+\t.\tID=gene-PA%04d;locus_tag=PA%04d;gene=g%d\n' % (i*200,i*200+99,i,i,i))
        q=run(root,g,root/OUT_REL,strict=False)
        need(q['n_curated_signed_PAO1_Anr_targets']==37 and q['n_measured_targets']==37,
             'synthetic target count invalid')
        need(q['n_genomic_neighborhood_groups']>=1,'no genomic neighborhoods')
        need(q['circular_shift_control']['n_alternative_offsets']==99,'shift null not complete')
        need(q['n_sign_agreement'] > 0,'synthetic test missing matches')
        need(q['net_correct_advantage_over_all_DOWN'] == q['n_sign_agreement'] - q['always_predict_DOWN_n_correct'],
             'trivial baseline not computed consistently')
        try:
            run(root,g,root/OUT_REL,strict=False)
            raise AssertionError('overwrite protection failed')
        except RuntimeError as e:
            need('output exists' in str(e), 'wrong no-overwrite failure')
        print('SELF_TEST_PASSED: input preservation, sign mapping, independence controls,')
        print('  100 circular shifts, genomic neighborhoods, output and no-overwrite gate.')


def main():
    pa=argparse.ArgumentParser()
    pa.add_argument('--gff',default='references/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz')
    pa.add_argument('--self-test',action='store_true')
    args=pa.parse_args()
    if args.self_test:
        self_test()
    else:
        g=Path(args.gff)
        if not g.is_absolute():
            g=ROOT/g
        run(ROOT,g,ROOT/OUT_REL,strict=True)

if __name__=='__main__':
    main()
