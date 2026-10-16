#!/usr/bin/env python3
"""CF-State 09n: Audit EPathDash design/count identity before training.

No differential expression, no inferred sample assignment, no training contrasts.
Only explicitly shared sample identifiers verify mapping. Equal row/column counts
are necessary but do NOT prove identical sample order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

DESIGN = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Design_Matrices')
COUNTS = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Count_Tables')
META = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Metadata')
CANDIDATES = Path('results/phase9/audit_09m_multi_perturbation_contrast_census/CONTRAST_CANDIDATES_REVIEW_ONLY.csv')
OUT = Path('results/phase9/audit_09n_EPathDash_sample_count_alignment_preflight')
ID_COLUMN = re.compile(r'^(?:sample[_ -]?(?:id|name|accession)|gsm|srr|srs|run[_ -]?id|library[_ -]?id|biosample)$', re.I)
BARE_PA = re.compile(r'^PA\d{4}$')
PA_VARIANT = re.compile(r'^PA\d{4}(?:[.][0-9]+|[a-zA-Z])$')
PA14 = re.compile(r'^PA14[_-]?\d+', re.I)
CONTROL_RE = re.compile(r'^(control|ctrl|wt|wild[ -]?type|untreated|vehicle|mock|parental)$', re.I)


def require(condition, msg):
    if not condition:
        raise RuntimeError('AUDIT_09N_FAILED: '+str(msg))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def trim(v):
    return str(v).strip().lstrip('\ufeff')


def norm_id(x):
    return re.sub(r'\s+', '', trim(x)).lower()


def unique_raw_headers(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        header = next(csv.reader(f))
    require(len(header) >= 2, 'bad count header: '+str(path))
    labels = [trim(x) for x in header[1:]]
    return len(labels) == len(set(labels)), labels


def identify_map(design, count_samples):
    """Return row index of count sample only for a proven identical ID set."""
    norm_counts = [norm_id(x) for x in count_samples]
    candidate_cols = [x for x in design.columns if ID_COLUMN.fullmatch(str(x).strip())]
    if len(set(norm_counts)) != len(norm_counts):
        return 'DUPLICATE_COUNT_ID', '', [], 0
    best_overlap, best_col = 0, ''
    for col in candidate_cols:
        vals = [norm_id(x) for x in design[col].tolist()]
        if any(not x or x in {'nan', 'none', 'missing'} for x in vals):
            continue
        if len(vals) != len(set(vals)):
            continue
        overlap = len(set(vals) & set(norm_counts))
        if overlap > best_overlap:
            best_overlap, best_col = overlap, str(col)
        if len(vals) == len(norm_counts) and set(vals) == set(norm_counts):
            lookup = {s:i for i,s in enumerate(vals)}
            return 'EXPLICIT_EXACT_SAMPLE_ID_SET', str(col), [lookup[s] for s in norm_counts], overlap
    if best_overlap:
        return 'PARTIAL_ID_OVERLAP_DO_NOT_ASSIGN', best_col, [], best_overlap
    if len(design) == len(count_samples):
        return 'COUNT_MATCH_ONLY_ORDER_UNVERIFIED', '', [], 0
    return 'SAMPLE_NUMBER_MISMATCH', '', [], 0


def gene_id_distribution(ids):
    x = [trim(s) for s in ids]
    n = len(x)
    direct = sum(bool(BARE_PA.fullmatch(y)) for y in x)
    variant = sum(bool(PA_VARIANT.fullmatch(y)) for y in x)
    pa14 = sum(bool(PA14.match(y)) for y in x)
    if n == 0:
        return 'EMPTY', direct, variant, pa14, 0
    if direct / n >= 0.80:
        kind = 'MOSTLY_BARE_PAO1_LOCUS'
    elif pa14 / n >= 0.50:
        kind = 'MOSTLY_PA14_IDENTIFIERS'
    elif direct / n >= 0.20:
        kind = 'MIXED_WITH_PAO1_LOCUS'
    else:
        kind = 'GENE_SYMBOLS_OR_OTHER_REQUIRES_MAPPING'
    duplicates = n - len(set(x))
    return kind, direct, variant, pa14, duplicates


def process_study(root, study, design_path, count_path, meta_path):
    design = pd.read_csv(design_path, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    design.columns = [trim(k) for k in design.columns]
    design = design.apply(lambda c: c.map(trim))
    metadata = pd.read_csv(meta_path, dtype=str, encoding='utf-8-sig') if meta_path.exists() else pd.DataFrame()
    strain = 'UNKNOWN'
    if not metadata.empty:
        for k in metadata.columns:
            if trim(k).lower() == 'strain':
                strain = str(metadata[k].iloc[0]).strip()
                break
    result = {'study':study,'strain_metadata':strain,'n_design_rows':len(design),
              'design_columns':' | '.join(design.columns),'count_file_exists':count_path.is_file(),
              'count_path':str(count_path.relative_to(root)),'count_read_error':'',
              'n_count_samples':0,'n_gene_rows':0,'sample_alignment_status':'NO_COUNT_FILE',
              'explicit_id_column':'','n_id_overlap':0,'gene_id_type':'NOT_CHECKED',
              'n_PA_bare':0,'n_PA_suffix_or_variant':0,'n_PA14_like':0,'n_duplicate_gene_ids':0,
              'missing_fraction':np.nan,'negative_values':0,'fractional_values':0,
              'zero_library_samples':0,'min_sample_sum':np.nan,'max_sample_sum':np.nan,
              'column_names_preview':'','decision':'NOT_TRAINING_READY'}
    previews = []
    if not count_path.is_file():
        return result, previews
    try:
        unique, raw_samples = unique_raw_headers(count_path)
        if not unique:
            result['sample_alignment_status'] = 'DUPLICATE_COUNT_COLUMN_HEADERS'
        df = pd.read_csv(count_path, dtype=str, encoding='utf-8-sig', low_memory=False)
        require(len(df.columns) == len(raw_samples)+1, 'header was altered during read '+study)
        require(not df.empty, 'no count genes: '+study)
        gene_col = df.columns[0]
        gene_ids = [trim(s) for s in df[gene_col].fillna('')]
        sample_cols = list(df.columns[1:])
        result['n_count_samples'] = len(sample_cols)
        result['n_gene_rows'] = len(df)
        result['column_names_preview'] = ' | '.join(raw_samples[:4])[:220]
        kind, bare, variant, pa14, duplicates = gene_id_distribution(gene_ids)
        result.update(gene_id_type=kind, n_PA_bare=bare, n_PA_suffix_or_variant=variant,
                      n_PA14_like=pa14, n_duplicate_gene_ids=duplicates)
        numbers = df[sample_cols].apply(pd.to_numeric, errors='coerce')
        arr = numbers.to_numpy(dtype=float)
        result['missing_fraction'] = float(np.mean(~np.isfinite(arr)))
        result['negative_values'] = int(np.sum(arr[np.isfinite(arr)] < 0))
        result['fractional_values'] = int(np.sum(np.abs(arr[np.isfinite(arr)] - np.round(arr[np.isfinite(arr)])) > 1e-6))
        totals = numbers.sum(axis=0, skipna=True).to_numpy(dtype=float)
        result['zero_library_samples'] = int(np.sum(~np.isfinite(totals) | (totals <= 0)))
        result['min_sample_sum'] = float(np.min(totals))
        result['max_sample_sum'] = float(np.max(totals))
        status, id_col, row_order, overlap = identify_map(design, raw_samples)
        if not unique:
            status = 'DUPLICATE_COUNT_COLUMN_HEADERS'
            row_order = []
        result['sample_alignment_status'] = status
        result['explicit_id_column'] = id_col
        result['n_id_overlap'] = overlap
        numeric_ok = (result['missing_fraction'] <= .01 and result['negative_values'] == 0
                      and result['zero_library_samples'] == 0 and len(df) >= 1000)
        if status == 'EXPLICIT_EXACT_SAMPLE_ID_SET' and numeric_ok:
            result['decision'] = 'EXPLICIT_SAMPLE_MATCH_COUNTS_PASS_BIOLOGY_REVIEW_PENDING'
        elif status == 'COUNT_MATCH_ONLY_ORDER_UNVERIFIED' and numeric_ok:
            result['decision'] = 'ORDER_PROVENANCE_REQUIRED_BEFORE_CONTRAST'
        else:
            result['decision'] = 'BLOCK_OR_MANUAL_REPAIR'
        for i in range(min(6, len(raw_samples))):
            d_idx = row_order[i] if row_order else (i if i < len(design) else None)
            row_fields = ' | '.join(f'{c}={design.iloc[d_idx][c]}' for c in design.columns[:7]) if d_idx is not None else 'NO_CORRESPONDING_DESIGN_ROW'
            previews.append({'study':study,'count_column_1based':i+2,'count_sample':raw_samples[i],
                             'design_row_1based':d_idx+2 if d_idx is not None else None,
                             'design_factors_IF_POSITIONAL_UNVERIFIED':row_fields,
                             'mapping_evidence':status})
    except Exception as exc:
        result['count_read_error'] = str(exc)[:500]
        result['decision'] = 'COUNT_READ_ERROR'
    return result, previews


def run(root, out, write=True):
    design_dir = root / DESIGN
    ensure_files = sorted(design_dir.glob('GSE*_design.csv'))
    require(ensure_files, 'no design files in '+str(design_dir))
    if write:
        require(not out.exists(), 'refusing to overwrite '+str(out))
    rows, preview, provenance = [], [], []
    for des in ensure_files:
        study = des.name.split('_design.csv')[0]
        cnt = root/COUNTS/f'{study}_counts.csv'
        met = root/META/f'{study}_metadata.csv'
        report, pre = process_study(root, study, des, cnt, met)
        rows.append(report)
        preview.extend(pre)
        for p in (des,cnt,met):
            if p.is_file():
                provenance.append({'source':str(p.relative_to(root)),'sha256':sha256(p)})
    audit = pd.DataFrame(rows)
    summaries = pd.DataFrame(preview)
    require((root/CANDIDATES).is_file(), '09m review-only candidate ledger missing: '+str(root/CANDIDATES))
    candidates = pd.read_csv(root/CANDIDATES)
    require({'study','comparison_type','design_priority','level_a','level_b','n_a','n_b'}.issubset(candidates.columns),
            'unexpected 09m candidate schema')
    candidates = candidates.merge(audit[['study','sample_alignment_status','decision','gene_id_type','n_gene_rows','n_count_samples','n_duplicate_gene_ids','missing_fraction']],on='study',how='left',validate='many_to_one')
    candidates['contrast_use_status'] = 'NOT_VALIDATED_FOR_MODEL_TRAINING'
    candidates['readiness_note'] = np.where(candidates['decision']=='ORDER_PROVENANCE_REQUIRED_BEFORE_CONTRAST',
        'Equal sample counts; establish column-to-design row identity from external sample metadata',
        np.where(candidates['decision']=='EXPLICIT_SAMPLE_MATCH_COUNTS_PASS_BIOLOGY_REVIEW_PENDING',
        'Sample IDs exact; review controls, independence, gene mapping and replicates',
        'Fix sample/count problem before training'))
    candidates['is_possible_control'] = candidates['control_hint'].astype(str).str.startswith('POSSIBLE_CONTROL=')
    candidates['replicate_min'] = candidates[['n_a','n_b']].min(axis=1)
    candidates['rank_score'] = candidates['design_priority'] + 5*candidates.is_possible_control.astype(int) + 4*(candidates.replicate_min>=3).astype(int)
    candidates = candidates.sort_values(['rank_score','study'], ascending=[False,True])
    rank = candidates[(candidates['comparison_type'].isin(['GENETIC_INTERVENTION_CANDIDATE','ENVIRONMENT_OR_TREATMENT_CANDIDATE'])) & (candidates['is_possible_control'])].copy()
    rank = rank[rank['decision'].isin(['ORDER_PROVENANCE_REQUIRED_BEFORE_CONTRAST','EXPLICIT_SAMPLE_MATCH_COUNTS_PASS_BIOLOGY_REVIEW_PENDING'])]
    rank_pao1 = rank[rank['strain_metadata'].fillna('').str.contains('PAO1',case=False,regex=False)]
    priority_studies = rank_pao1.groupby('study',as_index=False).agg(n_candidate_groups=('study','size'), top_rank=('rank_score','max'), n_replicates_min=('replicate_min','max')).sort_values(['top_rank','n_candidate_groups'],ascending=False)
    print('\n'+'='*88)
    print('CF-STATE 09N | EPathDash SAMPLE/COUNT IDENTITY — NOT TRAINING CONTRAST FREEZE')
    print('='*88)
    print('Study designs:',len(audit),'| review-only contrast candidates:',len(candidates))
    print('Count tables located:',int(audit.count_file_exists.sum()))
    print('EXPLICIT matched sample ID studies:',int((audit.sample_alignment_status=='EXPLICIT_EXACT_SAMPLE_ID_SET').sum()))
    print('Equal count but ORDER UNVERIFIED:',int((audit.sample_alignment_status=='COUNT_MATCH_ONLY_ORDER_UNVERIFIED').sum()))
    print('Mismatched, missing, ambiguous:',int((~audit.sample_alignment_status.isin(['EXPLICIT_EXACT_SAMPLE_ID_SET','COUNT_MATCH_ONLY_ORDER_UNVERIFIED'])).sum()))
    print('Important: equal sample numbers do NOT establish sample identity or biological replicate independence.')
    print('\nSAMPLE AND GENE-ID STATUS, ALL STUDIES:')
    print(audit[['study','strain_metadata','n_design_rows','n_count_samples','n_gene_rows','gene_id_type','sample_alignment_status','decision']].to_string(index=False,max_colwidth=32))
    print('\nINPUT COUNT QUALITY ISSUES (if any):')
    bad=audit[(audit['count_read_error'].astype(bool))|(audit['n_duplicate_gene_ids']>0)|(audit['negative_values']>0)|(audit['missing_fraction']>.01)|(audit['zero_library_samples']>0)]
    if len(bad):
        print(bad[['study','count_read_error','n_duplicate_gene_ids','negative_values','missing_fraction','zero_library_samples']].to_string(index=False,max_colwidth=38))
    else:
        print('No gross count-matrix QC issues detected (does NOT imply biological validity).')
    print('\nTOP PAO1 INTERVENTION STUDIES FOR MANUAL SAMPLE-MAP REVIEW:')
    if len(priority_studies):
        print(priority_studies.head(15).to_string(index=False))
    else:
        print('None with clear control hints and usable count-table shape.')
    print('\nTOP PAO1 CONTRAST CANDIDATES (still NOT VALIDATED):')
    cols=['study','changing_factor','level_a','n_a','level_b','n_b','controlled_recorded_factors','sample_alignment_status']
    if len(rank_pao1):
        print(rank_pao1.head(23)[cols].to_string(index=False,max_colwidth=30))
    else:
        print('None')
    print('\nCOUNT COLUMN ↔ DESIGN ROW POSITIONAL PREVIEW (UNVERIFIED unless exact ID):')
    if len(summaries):
        show_studies = list(priority_studies.study.head(5)) if len(priority_studies) else list(audit.study.head(3))
        print(summaries[summaries.study.isin(show_studies)].to_string(index=False,max_colwidth=83))
    print('\nNEXT GATE: Establish independent sample identity and control orientation for a SMALL study subset,')
    print('then harmonise gene IDs and only then freeze supervised training contrasts.')
    print('NO differential expression fitted. NO 09n candidates promoted to training by this script.')
    if write:
        out.mkdir(parents=True,exist_ok=False)
        audit.to_csv(out/'STUDY_COUNT_DESIGN_IDENTITY_QC.csv',index=False)
        candidates.to_csv(out/'CONTRAST_CANDIDATES_WITH_ALIGNMENT_STATUS.csv',index=False)
        priority_studies.to_csv(out/'PRIORITY_PAO1_STUDIES_FOR_MANUAL_VERIFICATION.csv',index=False)
        summaries.to_csv(out/'SAMPLE_ORDER_PREVIEWS_UNVERIFIED.csv',index=False)
        pd.DataFrame(provenance).to_csv(out/'INPUT_SHA256_PROVENANCE.csv',index=False)
        status=dict(status='ALIGNMENT_PREFLIGHT_ONLY_NO_CONTRASTS_FROZEN',
                    n_studies=len(audit),n_design_candidate_pairs=len(candidates),
                    n_explicit_id_matched_studies=int((audit.sample_alignment_status=='EXPLICIT_EXACT_SAMPLE_ID_SET').sum()),
                    n_positional_only_unverified_studies=int((audit.sample_alignment_status=='COUNT_MATCH_ONLY_ORDER_UNVERIFIED').sum()),
                    n_training_contrasts=0,
                    caveat='no sample row assignment from count-order without explicit ID mapping')
        (out/'QC_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
        print('\nSAVED WITHOUT OVERWRITE:',out)
    return audit,candidates


def self_test():
    with tempfile.TemporaryDirectory() as td:
        r=Path(td)
        (r/DESIGN).mkdir(parents=True)
        (r/COUNTS).mkdir(parents=True)
        (r/META).mkdir(parents=True)
        (r/CANDIDATES).parent.mkdir(parents=True)
        for study,with_id,bad in [('GSE100001',True,False),('GSE100002',False,False),('GSE100003',False,True)]:
            design = {'Replicate':['1','2','3','1','2','3'],'Treatment':['Control']*3+['Drug']*3}
            names=['a','b','c','d','e','f']
            if with_id:
                design['Sample_ID']=names
            pd.DataFrame(design).to_csv(r/DESIGN/f'{study}_design.csv',index=False)
            pd.DataFrame({'Strain':['PAO1']}).to_csv(r/META/f'{study}_metadata.csv',index=False)
            sample_names = names if with_id else [f'unknown-{j}' for j in range(6 if not bad else 5)]
            mat=pd.DataFrame({'gene_id':['PA%04d'%k for k in range(1200)]})
            for i,n in enumerate(sample_names):
                mat[n]=np.arange(1200)+i+1
            mat.to_csv(r/COUNTS/f'{study}_counts.csv',index=False)
        pd.DataFrame([{'study':s,'comparison_type':'ENVIRONMENT_OR_TREATMENT_CANDIDATE',
                       'changing_factor':'Treatment','level_a':'Control','level_b':'Drug',
                       'n_a':3,'n_b':3,'control_hint':'POSSIBLE_CONTROL=Control','design_priority':65,
                       'strain_metadata':'PAO1','controlled_recorded_factors':'NONE'}
                      for s in ['GSE100001','GSE100002','GSE100003']]).to_csv(r/CANDIDATES,index=False)
        a,c=run(r,r/'results/phase9/test09n',write=True)
        status=dict(zip(a.study,a.sample_alignment_status))
        require(status['GSE100001']=='EXPLICIT_EXACT_SAMPLE_ID_SET','explicit match should pass')
        require(status['GSE100002']=='COUNT_MATCH_ONLY_ORDER_UNVERIFIED','position cannot verify')
        require(status['GSE100003']=='SAMPLE_NUMBER_MISMATCH','mismatch should block')
        require((c.contrast_use_status=='NOT_VALIDATED_FOR_MODEL_TRAINING').all(),'never release unverified contrasts')
        try:
            run(r,r/'results/phase9/test09n',write=True)
        except RuntimeError as exc:
            require('overwrite' in str(exc),'no-overwrite test')
        else:
            raise AssertionError('overwrite not blocked')
        print('SELF_TEST_PASSED: explicit sample ID vs positional-only vs mismatch; gene QC, candidates unvalidated, provenance, no-overwrite')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        root=args.root.resolve()
        run(root,root/OUT,write=True)

if __name__=='__main__':
    main()
