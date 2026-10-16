#!/usr/bin/env python3
"""CF-State 09o: non-positional sample-name/experimental-design group reconciliation.

This is a deliberately narrow *candidate* verifier; no expression contrasts are frozen.
Known study-specific naming rules are explicit and auditable. A parser cannot establish
independent biological replicate provenance and is not independent GEO verification.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

DESIGN_DIR = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Design_Matrices')
COUNT_DIR = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Count_Tables')
QC_FILE = Path('results/phase9/audit_09n_EPathDash_sample_count_alignment_preflight/STUDY_COUNT_DESIGN_IDENTITY_QC.csv')
CAND_FILE = Path('results/phase9/audit_09m_multi_perturbation_contrast_census/CONTRAST_CANDIDATES_REVIEW_ONLY.csv')
OUT = Path('results/phase9/audit_09o_EPathDash_semantic_sample_group_reconciliation')
# These rules are based on visible examples of upstream count sample names, not design row order.
# All returns use the EXACT design category names; unknowns must stay unknown.
STUDIES = ['GSE179150', 'GSE167137', 'GSE190194', 'GSE199323', 'GSE223663',
           'GSE138731', 'GSE178622', 'GSE182853', 'GSE192340', 'GSE201738']


def need(ok, msg):
    if not ok:
        raise RuntimeError('AUDIT_09O_FAILED: ' + str(msg))


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def read_count_columns(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        cols = next(csv.reader(f))
    need(len(cols) > 1, 'missing count sample columns: ' + str(path))
    vals = [x.strip() for x in cols[1:]]
    need(all(vals) and len(vals) == len(set(vals)), 'empty or duplicated count headers: ' + str(path))
    return vals


def norm(x):
    return re.sub(r'[^a-z0-9]+', '', str(x).strip().lower())


def get_case_level(design, col, expected):
    """Use exact or punctuation/case-normalized design level, no fuzzy or positional assignment."""
    if col not in design.columns:
        return None
    levels = list(dict.fromkeys(design[col].astype(str).str.strip()))
    matches = [s for s in levels if norm(s) == norm(expected)]
    return matches[0] if len(matches) == 1 else None


def known_group(study, name, design):
    """Return (factor dictionary, independent parsed replicate token, rule ID) or unknown."""
    x = name.strip()
    m = None
    if study == 'GSE179150':
        m = re.fullmatch(r'PA2504_p([0-9]+)_htseq', x, flags=re.I)
        if m:
            return {'Genotype': get_case_level(design, 'Genotype', 'Delta-PA2504')}, m[1], 'PA2504_MUTANT'
        m = re.fullmatch(r'PAO1161_p([0-9]+)_htseq', x, flags=re.I)
        if m:
            return {'Genotype': get_case_level(design, 'Genotype', 'WT')}, m[1], 'PAO1161_PARENT'
    elif study == 'GSE167137':
        m = re.fullmatch(r'PA-([05])([MC])-([A-Za-z0-9]+)', x, flags=re.I)
        if m:
            medium = 'Monoculture' if m[2].upper() == 'M' else 'Coculture'
            treat = 'Control' if m[1] == '0' else 'meropenem_5mg'
            return {'Medium': get_case_level(design, 'Medium', medium),
                    'Treatment': get_case_level(design, 'Treatment', treat)}, '', 'MEROPENEM_MEDIUM'
    elif study == 'GSE190194':
        m = re.fullmatch(r'(ALE|COL|CTRL|CTL|CON|CONTROL|UNT|WT)__?([0-9]+)', x, flags=re.I)
        if m:
            group = {'ALE': 'alexidine', 'COL': 'colistin'}.get(m[1].upper(), 'control')
            return {'Treatment': get_case_level(design, 'Treatment', group)}, m[2], 'DRUG_PREFIX'
    elif study == 'GSE199323':
        patterns = [
            (r'LawnControl([0-9]+)', 'Control', 'LAWN_CONTROL'),
            (r'PhoenixColony([0-9]+)', 'Tobramycin-Pheonix', 'PHOENIX'),
            (r'Edge(?:Colony)?([0-9]+)', 'Tobramycin-Edge', 'EDGE'),
            (r'Outer(?:Colony)?([0-9]+)', 'Tobramycin-Outer', 'OUTER'),
            (r'VBNC(?:Colony)?([0-9]+)', 'Tobramycin-VBNC', 'VBNC'),
        ]
        for pat, level, rule in patterns:
            m = re.fullmatch(pat, x, flags=re.I)
            if m:
                return {'Treatment': get_case_level(design, 'Treatment', level)}, m[1], rule
    elif study == 'GSE223663':
        m = re.fullmatch(r'BF0285BR([0-9]+)_S[0-9]+', x, flags=re.I)
        if m:
            return {'Genotype': get_case_level(design, 'Genotype', 'deltaPA0285'),
                    'Growth': get_case_level(design, 'Growth', 'Biofilm')}, m[1], 'BIOFILM_MUTANT'
        m = re.fullmatch(r'BioFWTBR([0-9]+)_S[0-9]+', x, flags=re.I)
        if m:
            return {'Genotype': get_case_level(design, 'Genotype', 'WT'),
                    'Growth': get_case_level(design, 'Growth', 'Biofilm')}, m[1], 'BIOFILM_WT'
    # The other studies intentionally have no decoding rules until their actual names
    # and original GEO samples are independently examined. Do not guess abbreviations.
    return None, '', 'UNKNOWN_NAME'


def factors(design):
    return [c for c in design.columns if c.strip().lower() not in
            {'replicate', 'rep', 'sample', 'sample_id', 'id', 'gsm', 'srr', 'run', 'batch'}]


def key_for(dic, dimensions):
    return tuple(str(dic[c]).strip() for c in dimensions)


def process_study(root, study, previous_qc):
    dpath = root / DESIGN_DIR / (study + '_design.csv')
    cpath = root / COUNT_DIR / (study + '_counts.csv')
    need(dpath.is_file() and cpath.is_file(), 'missing design/count pair ' + study)
    d = pd.read_csv(dpath, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    d.columns = [x.strip() for x in d.columns]
    for c in d.columns:
        d[c] = d[c].astype(str).str.strip()
    names = read_count_columns(cpath)
    dims = factors(d)
    need(len(dims) > 0, 'no experimental factor columns ' + study)
    design_group_sizes = Counter(key_for(row, dims) for row in d.to_dict('records'))
    recognised = Counter()
    sample_rows = []
    for col_idx, name in enumerate(names):
        parsed, rep, rule = known_group(study, name, d)
        valid = parsed is not None and all(parsed.get(c) is not None for c in dims)
        # Explicit design-factor matching across EVERY factor: no hidden extrapolation.
        if valid:
            group = key_for(parsed, dims)
            valid = group in design_group_sizes
        else:
            group = tuple()
        if valid:
            recognised[group] += 1
        row = {'study': study, 'count_sample': name, 'count_column_1based': col_idx+2,
               'rule': rule, 'replicate_token_from_name': rep,
               'name_based_group_status': 'NAME_DERIVED_GROUP' if valid else 'UNRESOLVED_NAME',
               'full_factor_group': ' | '.join(f'{c}={v}' for c,v in zip(dims,group)) if valid else '',
               'corresponding_design_row': 'NOT_INFERRED',
               'independent_GEO_identity_verified': False,
               'training_approved': False}
        for c in dims:
            row['factor_' + c] = parsed.get(c, '') if parsed is not None else ''
        sample_rows.append(row)
    # Does every named count sample populate an experimentally valid group, with
    # exactly the same MULTISET of group sizes, independent of column ordering?
    no_unresolved = all(r['name_based_group_status'] == 'NAME_DERIVED_GROUP' for r in sample_rows)
    sizes_match = recognised == design_group_sizes and len(names) == len(d)
    if no_unresolved and sizes_match:
        status = 'NAME_GROUPS_RECONCILED_GEO_CONFIRMATION_PENDING'
    elif recognised:
        status = 'PARTIAL_NAME_GROUPS_DO_NOT_FREEZE'
    else:
        status = 'NO_INDEPENDENT_NAME_GROUP_RECONCILIATION'
    p = previous_qc.loc[study]
    problems = []
    if int(p.n_duplicate_gene_ids) > 0:
        problems.append('DUPLICATE_GENE_IDENTIFIERS')
    if float(p.missing_fraction) > 0.01:
        problems.append('EXCESSIVE_MISSING_COUNTS')
    if int(p.negative_values) > 0 or int(p.zero_library_samples) > 0:
        problems.append('INVALID_COUNT_VALUES')
    if str(p.gene_id_type).startswith('MOSTLY_PA14'):
        problems.append('PA14_REQUIRES_ORTHOLOG_PROJECTION')
    if problems:
        status += '__QC_REVIEW_REQUIRED'
    summary = {'study': study, 'n_design':len(d),'n_count_samples':len(names),
               'n_group_assigned_from_names':len(names)-sum(r['name_based_group_status']=='UNRESOLVED_NAME' for r in sample_rows),
               'n_unresolved':sum(r['name_based_group_status']=='UNRESOLVED_NAME' for r in sample_rows),
               'n_factor_groups_in_design':len(design_group_sizes),
               'n_groups_reconstructed':len(recognised),
               'group_multiset_matches':bool(sizes_match),
               'nonpositional_reconciliation_status':status,
               'qc_flags': '|'.join(problems) or 'NONE',
               'source_study_processed_previously':study in {'GSE223663'},
               'approved_training_contrasts':0}
    group_rows=[]
    for group in sorted(set(design_group_sizes) | set(recognised)):
        group_rows.append({'study':study,'factor_group':' | '.join(f'{c}={v}' for c,v in zip(dims,group)),
                           'n_design':design_group_sizes.get(group,0),
                           'n_from_sample_names':recognised.get(group,0),
                           'counts_equal':design_group_sizes.get(group,0)==recognised.get(group,0)})
    return summary, sample_rows, group_rows


def run(root, out):
    need(not out.exists(), 'refusing to overwrite existing directory ' + str(out))
    need((root / QC_FILE).is_file(), 'missing 09n QC file')
    need((root / CAND_FILE).is_file(), 'missing 09m contrast candidates')
    q = pd.read_csv(root / QC_FILE).set_index('study')
    cand = pd.read_csv(root / CAND_FILE)
    need(len(q) >= len(STUDIES), 'missing prior 09n study QC')
    totals, samples, groups, hashes = [], [], [], []
    for study in STUDIES:
        need(study in q.index, 'study not found in 09n QC: ' + study)
        s, r, g = process_study(root, study, q)
        totals.append(s); samples.extend(r); groups.extend(g)
        for p in (root / DESIGN_DIR / (study+'_design.csv'), root / COUNT_DIR / (study+'_counts.csv')):
            hashes.append({'path':str(p.relative_to(root)),'sha256':sha256(p)})
    for p in (root / QC_FILE, root / CAND_FILE):
        hashes.append({'path':str(p.relative_to(root)),'sha256':sha256(p)})
    tab = pd.DataFrame(totals)
    samp = pd.DataFrame(samples)
    grp = pd.DataFrame(groups)
    eligible = cand[cand.study.isin(STUDIES)].copy()
    eligible = eligible.merge(tab[['study','nonpositional_reconciliation_status','qc_flags']],on='study',how='left',validate='many_to_one')
    eligible['contrast_training_status']='NOT_APPROVED_GEO_SAMPLE_PROVENANCE_PENDING'
    print('\n'+'='*91)
    print('CF-STATE 09O | EPathDash COUNT NAME → DESIGN GROUP CROSS-CHECK')
    print('='*91)
    print('Candidate studies:',len(tab),'| NOT TRAINING DATA: NO CONTRASTS COMPUTED')
    print('Completed group-name reconciliations:',sum(tab.nonpositional_reconciliation_status=='NAME_GROUPS_RECONCILED_GEO_CONFIRMATION_PENDING'))
    print('Partially matched studies:',sum(tab.nonpositional_reconciliation_status.str.startswith('PARTIAL_NAME')))
    print('Unresolved studies:',sum(tab.nonpositional_reconciliation_status.str.startswith('NO_INDEPENDENT')))
    print('Full GEO-verified sample identities:',0,'| approved training contrasts:',0)
    print('\nSTUDY LEVEL — count groups matched independently of row order:')
    print(tab[['study','n_count_samples','n_group_assigned_from_names','n_unresolved','group_multiset_matches',
               'nonpositional_reconciliation_status','qc_flags']].to_string(index=False,max_colwidth=38))
    print('\nDESIGN GROUP COUNT COMPARISON (all groups):')
    print(grp.to_string(index=False,max_colwidth=75))
    for study in STUDIES[:5]:
        print('\nSAMPLE LABEL PREVIEW |',study)
        print(samp[samp.study==study][['count_sample','rule','full_factor_group','replicate_token_from_name','name_based_group_status']].head(20).to_string(index=False,max_colwidth=72))
    print('\nRELEASE GATE: Original GEO accession/sample table must corroborate names,')
    print('genotypes/conditions and independent replicates; exact same study must not be counted twice.')
    print('GSE199323 groups are treated colony phenotypes, not automatically a homogeneous antibiotic-dose effect.')
    print('No positional mapping used, no count columns reordered, no differential-expression analysis.')
    out.mkdir(parents=True,exist_ok=False)
    tab.to_csv(out/'STUDY_NAME_GROUP_RECONCILIATION.csv',index=False)
    samp.to_csv(out/'SAMPLE_GROUP_NAME_EVIDENCE_NOT_TRAINING.csv',index=False)
    grp.to_csv(out/'DESIGN_VS_NAME_GROUP_COUNTS.csv',index=False)
    eligible.to_csv(out/'CONTRAST_CANDIDATES_PROVENANCE_PENDING.csv',index=False)
    pd.DataFrame(hashes).to_csv(out/'INPUT_SHA256_PROVENANCE.csv',index=False)
    (out/'QC_STATUS.json').write_text(json.dumps({'status':'09O_NONPOSITIONAL_NAME_GROUP_PREFLIGHT',
        'n_studies':len(tab),'n_fully_reconciled':int(sum(tab.nonpositional_reconciliation_status=='NAME_GROUPS_RECONCILED_GEO_CONFIRMATION_PENDING')),
        'n_approved_training_contrasts':0,'mapping_requires_external_sample_confirmation':True},indent=2)+'\n')
    print('\nSAVED WITHOUT OVERWRITE:',out)
    return tab, samp


def self_test():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for x in (DESIGN_DIR, COUNT_DIR, QC_FILE.parent, CAND_FILE.parent):
            (root/x).mkdir(parents=True,exist_ok=True)
        groups = {'GSE179150': ([{'Genotype':'Delta-PA2504','Replicate':str(i)} for i in (1,2,3)]+
                                 [{'Genotype':'WT','Replicate':str(i)} for i in (1,2,3)],
                                 [f'PAO1161_p{i}_htseq' for i in (3,1,2)]+[f'PA2504_p{i}_htseq' for i in (2,3,1)]),
                  'GSE167137': ([{'Medium':med,'Treatment':t,'Replicate':str(i)} for med in ('Monoculture','Coculture') for t in ('Control','meropenem_5mg') for i in (1,2)],
                                 [f'PA-{v}{m}-{r}' for v in (5,0) for m in ('C','M') for r in ('7','4')]),
                  'GSE199323': ([{'Treatment':t,'Replicate':str(i)} for t in ('Control','Tobramycin-Pheonix') for i in (1,2,3)],
                                 [f'PhoenixColony{i}' for i in (3,1,2)]+[f'LawnControl{i}' for i in (2,3,1)])}
        prior=[]
        for study in STUDIES:
            if study in groups:
                rows,names=groups[study]
                if study=='GSE167137':
                    # group count: 2/group = 8, reversed group order
                    pass
            else:
                rows=[{'Treatment':'Control','Replicate':str(i)} for i in (1,2,3)]+[{'Treatment':'Drug','Replicate':str(i)} for i in (1,2,3)]
                names=[f'unknown-{i}' for i in range(6)]
            pd.DataFrame(rows).to_csv(root/DESIGN_DIR/(study+'_design.csv'),index=False)
            with (root/COUNT_DIR/(study+'_counts.csv')).open('w') as f:
                f.write('GeneID,'+','.join(names)+'\n')
                f.write('PA0001,'+','.join(['1']*len(names))+'\n')
            prior.append({'study':study,'n_duplicate_gene_ids':0,'missing_fraction':0,
                          'negative_values':0,'zero_library_samples':0,'gene_id_type':'MOSTLY_BARE_PAO1_LOCUS'})
        pd.DataFrame(prior).to_csv(root/QC_FILE,index=False)
        pd.DataFrame([{'study':s,'changing_factor':'Treatment','level_a':'Control','level_b':'Drug'} for s in STUDIES]).to_csv(root/CAND_FILE,index=False)
        tab,samp=run(root,root/'results/phase9/test09o')
        verdict=tab.set_index('study')
        for st in ('GSE179150','GSE167137','GSE199323'):
            need(verdict.loc[st,'group_multiset_matches'], 'shuffled columns must reconcile: '+st)
            need(verdict.loc[st,'n_unresolved']==0,'sample names should be decoded: '+st)
        need(verdict.loc['GSE190194','n_unresolved']==6,'no guesses for unknown names')
        need(not samp.training_approved.any(),'must NEVER approve training without external confirmation')
        try:
            run(root,root/'results/phase9/test09o')
        except RuntimeError as e:
            need('overwrite' in str(e),'must reject overwrite')
        else:
            raise AssertionError('rerun must reject overwrite')
        print('SELF_TEST_PASSED: row-order shuffled groups, strict unknown quarantine, no training approval, no-overwrite.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:
        self_test()
    else:
        r=a.root.resolve()
        run(r,r/OUT)

if __name__=='__main__':
    main()
