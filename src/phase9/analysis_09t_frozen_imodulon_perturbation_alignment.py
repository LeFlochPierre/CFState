#!/usr/bin/env python3
"""CF-State 09t: descriptive alignment of frozen 09p/09r perturbation responses
with unmodified aeruPRECISE411 iModulon loadings.

Crucial: This is NOT calibrated iModulon activity inference, a trained response
predictor, a prospective external validation, or an experiment-level p-value.
GSE138731 is reference-included; the other studies are only candidates for
external analysis pending cross-dataset sample provenance checks.
No curated regulatory graph is fitted; no holdout outcome used to train M.

All results printed at study/contrast level, and exported with provenance.
Synthetic --self-test tests orientation, missingness, sample overlap, ranking,
non-destructive save, and rejection of inconsistent reference matrices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REF = Path('external/aeruprecise/precise411/data_files')
P = Path('results/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze')
R = Path('results/phase9/audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze')
OUT = Path('results/phase9/analysis_09t_frozen_imodulon_perturbation_alignment')
PRIMARY = ('GSE179150', 'GSE167137', 'GSE138731', 'GSE192340')
SECONDARY = ('GSE190194', 'GSE223663')
PA = re.compile(r'^PA\d{4}$')
REG_GSM = re.compile(r'GSM\d+', re.IGNORECASE)
REG_GSE = re.compile(r'GSE\d+', re.IGNORECASE)


def need(condition, message):
    if not bool(condition):
        raise RuntimeError('09T_CHECK_FAILED: '+str(message))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def clean_matrix_csv(path, axis_name):
    data = pd.read_csv(path, index_col=0, low_memory=False)
    data.index = data.index.map(lambda x: str(x).strip().lstrip('\ufeff'))
    data.columns = data.columns.map(lambda x: str(x).strip())
    need(data.index.is_unique, str(path)+' duplicate '+axis_name+' index')
    need(data.columns.is_unique, str(path)+' duplicate columns')
    data = data.apply(pd.to_numeric, errors='coerce')
    return data


def load_reference(root, strict=True):
    mp = root/REF/'M.csv'
    ap = root/REF/'A.csv'
    sp = root/REF/'sample_table.csv'
    for p in (mp, ap, sp):
        need(p.is_file(), 'missing frozen reference '+str(p))
    m_all = clean_matrix_csv(mp, 'gene')
    a = clean_matrix_csv(ap, 'iModulon')
    genes = m_all.index.to_series().str.fullmatch(r'PA\d{4}').to_numpy()
    m = m_all.iloc[np.flatnonzero(genes)].copy()
    m = m.loc[m.notna().all(axis=1)]
    need(len(m)>=500 and m.shape[1]>=2, 'reference PA-locus or iModulon coverage unexpectedly small; inspect M.csv orientation')
    same = set(m.columns).intersection(set(a.index))
    trans = set(m.columns).intersection(set(a.columns))
    if len(trans)>len(same):
        a = a.T
        same = set(m.columns).intersection(set(a.index))
    need(len(same)>=max(2, int(0.75*m.shape[1])), 'M/A component identities do not correspond; inspect labels')
    if strict:
        need(m.shape[1]==116, 'expected frozen 116 iModulon columns; got '+str(m.shape[1]))
        need(a.shape[1]==411, 'expected 411 reference samples in A; got '+str(a.shape[1]))
        need(len(same)==116, 'expected all 116 M/A component names to match')
    m = m.loc[:, sorted(same)]
    need(np.isfinite(m.to_numpy(float)).all(), 'reference M retains non-finite values')
    samples = pd.read_csv(sp, dtype=str, keep_default_na=False)
    if strict:
        need(len(samples)==411, 'reference sample table not 411 samples')
    return m, a, samples, [mp, ap, sp], len(m_all)


def load_responses(root, strict=True):
    paths = [root/P/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv',
             root/P/'CONTRAST_SOURCE_DESIGN_AND_QC.csv',
             root/P/'SAMPLE_GEO_SERIES_TITLE_CROSSWALK.csv',
             root/P/'QC_STATUS.json',
             root/R/'PROVISIONAL_ADDITIONAL_GENE_RESPONSES.csv',
             root/R/'PROVISIONAL_ADDITIONAL_CONTRASTS.csv',
             root/R/'QC_COUNT_COLUMN_TO_GEO_SERIES_TITLE.csv',
             root/R/'QC_STATUS.json']
    for p in paths:
        need(p.is_file(), 'missing frozen 09p/09r input '+str(p))
    p_wide = clean_matrix_csv(paths[0], 'gene')
    p_meta = pd.read_csv(paths[1], dtype=str, keep_default_na=False)
    p_samples = pd.read_csv(paths[2], dtype=str, keep_default_na=False)
    p_qc = json.loads(paths[3].read_text())
    r_long = pd.read_csv(paths[4], dtype={'GeneID':str, 'contrast_id':str}, low_memory=False)
    r_meta = pd.read_csv(paths[5], dtype=str, keep_default_na=False)
    r_samples = pd.read_csv(paths[6], dtype=str, keep_default_na=False)
    r_qc = json.loads(paths[7].read_text())
    need(p_qc.get('status')=='09P_PROVISIONAL_MULTISTUDY_RESPONSE_CANDIDATES_FROZEN', '09p QC status')
    need(r_qc.get('status')=='09R_THREE_STUDY_ADDITIONAL_CONTRAST_QC_COMPLETE', '09r QC status')
    need(p_wide.columns.is_unique, 'duplicate 09p contrasts')
    need(not r_long.duplicated(['GeneID','contrast_id']).any(), 'duplicate 09r response row')
    need({'GeneID','contrast_id','effect_log2_count_over_median_ratio_plus1'}.issubset(r_long.columns), '09r missing response columns')
    need({'study','contrast_id'}.issubset(p_meta.columns) and {'study','contrast_id'}.issubset(r_meta.columns), 'missing contrast metadata')
    need(set(p_wide.columns)==set(p_meta.contrast_id), '09p contrasts and response matrix mismatch')
    need(set(r_long.contrast_id)==set(r_meta.contrast_id), '09r contrasts and response rows mismatch')
    need(set(r_meta.study)=={'GSE138731','GSE192340'}, '09r expected eligible studies; do not load quarantined DATS')
    need(set(p_meta.study)=={'GSE179150','GSE167137','GSE190194','GSE223663'}, '09p study list unexpected')
    need(p_wide.index.to_series().str.fullmatch(r'PA\d{4}').all(), '09p non-PA GeneID')
    need(r_long.GeneID.str.fullmatch(r'PA\d{4}').all(), '09r non-PA GeneID')
    for p in (p_wide,):
        need(np.isfinite(p.to_numpy(float)).all(), 'non-finite 09p response')
    r_wide = r_long.pivot(index='GeneID',columns='contrast_id',values='effect_log2_count_over_median_ratio_plus1').astype(float)
    need(np.isfinite(pd.to_numeric(r_long['effect_log2_count_over_median_ratio_plus1'], errors='coerce')).all(), 'non-finite 09r response rows')
    need(not np.isinf(r_wide.to_numpy(float)).any(), 'infinite 09r response matrix')
    meta = pd.concat([p_meta[['study','contrast_id']],r_meta[['study','contrast_id']]], ignore_index=True)
    need(meta.contrast_id.is_unique, 'contrast ID collision')
    need(len(meta)==10 and meta.study.nunique()==6, 'expected 10 contrasts / 6 study groups including secondary')
    if strict:
        need(p_wide.shape == (3314,8), '09p frozen 3314x8 dimensions changed')
        need(r_wide.shape[1] == 2, 'expected two 09r eligible responses')
    response = pd.concat([p_wide,r_wide],axis=1)
    response = response.loc[:,meta.contrast_id]
    sample_crosswalk = pd.concat([p_samples,r_samples], ignore_index=True, sort=False)
    need('GSM' in sample_crosswalk.columns and 'study' in sample_crosswalk.columns, 'crosswalk study/GSM missing')
    return response, meta, sample_crosswalk, paths


def overlap_report(meta, samples, sample_crosswalk):
    upper = samples.astype(str).apply(lambda col: col.str.upper())
    gsm_rows = []
    for i, row in upper.iterrows():
        gsm_rows.append(set(REG_GSM.findall(' '.join(row.values.tolist()))))
    data=[]
    for study in sorted(meta.study.unique()):
        relevant=sample_crosswalk[sample_crosswalk.study==study]
        gsms=set(relevant.GSM.astype(str).str.upper())
        n_gsm=int(sum(bool(s & gsms) for s in gsm_rows))
        # Search all reference metadata columns, including GEO series and aliases.
        series_hit=upper.apply(lambda col:col.str.contains(study,regex=False)).any(axis=1)
        n_series=int(series_hit.sum())
        n_union=int(sum(series_hit.to_numpy()[i] or bool(s & gsms) for i,s in enumerate(gsm_rows)))
        classification=('REFERENCE_INCLUDED_NOT_EXTERNAL' if n_union else
                        'NO_ID_MATCH_CANDIDATE_EXTERNAL_NOT_VERIFIED')
        data.append(dict(study=study, n_source_GSM=len(gsms),
                         reference_rows_with_study_accession=n_series,
                         reference_rows_with_exact_GSM=n_gsm,
                         union_matching_reference_rows=n_union,
                         reference_overlap_class=classification))
    return pd.DataFrame(data)


def centre_standardise_1d(x):
    x=np.asarray(x,float)
    z=x-x.mean()
    n=np.sqrt(np.dot(z,z))
    if n<1e-13:
        return np.zeros_like(z)
    return z/n


def score_one(gene_ids, y, m, n_shift=101, seed=0):
    common = sorted(set(gene_ids).intersection(m.index),key=lambda s:int(s[2:]))
    need(len(common)>=150, 'too few PA gene overlaps with reference')
    v=y.reindex(common).to_numpy(float)
    w=m.loc[common].to_numpy(float)
    need(np.isfinite(v).all() and np.isfinite(w).all(), 'nonfinite aligned response/loading')
    # Gene-centered Pearson loading-response correlations.
    wz=w-w.mean(axis=0,keepdims=True)
    norm=np.sqrt((wz*wz).sum(axis=0))
    norm[norm<1e-13]=np.inf
    wn=wz/norm
    vy=centre_standardise_1d(v)
    corr=vy @ wn
    # Rank concordance: robust descriptive statistic, not calibrated TF activity.
    ry=centre_standardise_1d(pd.Series(v).rank(method='average').to_numpy())
    ranked=pd.DataFrame(w).rank(axis=0,method='average').to_numpy()
    wr=ranked-ranked.mean(axis=0,keepdims=True)
    wrnorm=np.sqrt((wr*wr).sum(axis=0))
    wrnorm[wrnorm<1e-13]=np.inf
    spear=ry @ (wr/wrnorm)
    # Diagnostic null: circular shift entire gene-ordered response, preserving
    # local autocorrelation approximately (NOT a formal exchangeable-null p).
    rng=np.random.default_rng(seed)
    n=len(v)
    viable=np.arange(min(25,n//4),n-min(25,n//4),dtype=int)
    if len(viable)<1:
        viable=np.arange(1,n)
    shifts=rng.choice(viable,size=min(n_shift,len(viable)),replace=False)
    null=np.array([float(np.max(np.abs(np.roll(vy,int(k))@wn))) for k in shifts])
    absmax=float(np.max(np.abs(corr)))
    diag=(1+int((null>=absmax).sum()))/(len(null)+1)
    return pd.DataFrame({'imodulon':list(m.columns), 'pearson_alignment':corr,
                         'spearman_alignment':spear}), {
        'n_gene_overlap':n,'n_exact_zero_response':int((v==0).sum()),
        'median_abs_response':float(np.median(np.abs(v))),
        'max_abs_pearson_alignment':absmax,
        'shift_null_max_abs_p_DIAGNOSTIC_ONLY':diag,
        'n_shift_nulls':len(null),
        'shift_null_95th_percentile':float(np.percentile(null,95))}, pd.Series(v,index=common)


def compare_primary_pairs(response,meta,overlaps,aligned):
    rows=[]
    index_map=overlaps.set_index('study').reference_overlap_class.to_dict()
    contrasts=meta[meta.study.isin(PRIMARY)]
    for a,b in combinations(contrasts.itertuples(index=False),2):
        ya=response[a.contrast_id].dropna()
        yb=response[b.contrast_id].dropna()
        genes=sorted(set(ya.index)&set(yb.index),key=lambda g:int(g[2:]))
        gene_spear=float(pd.Series(ya.loc[genes]).corr(pd.Series(yb.loc[genes]),method='spearman'))
        sa=aligned[a.contrast_id].set_index('imodulon').pearson_alignment
        sb=aligned[b.contrast_id].set_index('imodulon').pearson_alignment
        program_pcc=float(sa.corr(sb))
        no_reference_overlap=(index_map[a.study].startswith('NO_ID_MATCH') and
                              index_map[b.study].startswith('NO_ID_MATCH'))
        rows.append({'study_a':a.study,'study_b':b.study,
                     'contrast_a':a.contrast_id,'contrast_b':b.contrast_id,
                     'same_GEO_study':a.study==b.study,
                     'both_candidate_external':bool(no_reference_overlap),
                     'n_common_genes_for_gene_spearman':len(genes),
                     'gene_response_spearman':gene_spear,
                     'frozen_imodulon_alignment_profile_pearson':program_pcc})
    return pd.DataFrame(rows)


def run(root,out,strict=True,n_shift=101):
    need(not out.exists(), 'refusing to overwrite existing '+str(out))
    m,a,samples,ref_paths,n_M_raw=load_reference(root,strict)
    response, meta, crosswalk, source_paths=load_responses(root,strict)
    overlaps=overlap_report(meta,samples,crosswalk)
    if strict:
        f=overlaps.set_index('study')
        need(f.loc['GSE138731','reference_rows_with_study_accession']==6,
             'known GSE138731 reference overlap no longer 6; investigate')
        for s in ('GSE167137','GSE179150','GSE192340'):
            need(int(f.loc[s,'union_matching_reference_rows'])==0,
                 'unexpected primary reference sample overlap '+s)
    provenance=[dict(path=str(p.relative_to(root)),sha256=sha256(p))
                for p in ref_paths+source_paths]
    align=[]; qc=[]; by_id={}
    for row in meta.itertuples(index=False):
        cid=row.contrast_id
        available=response[cid].dropna()
        z,check,effect=score_one(available.index,available,m,n_shift=n_shift,
                                  seed=1000+sum(ord(c) for c in cid))
        z.insert(0,'contrast_id',cid)
        z.insert(0,'study',row.study)
        check.update({'study':row.study,'contrast_id':cid,
                      'panel':('PRIMARY' if row.study in PRIMARY else 'SECONDARY'),
                      'reference_overlap_class':overlaps.set_index('study').loc[row.study,'reference_overlap_class']})
        align.append(z); qc.append(check); by_id[cid]=z.copy()
    all_align=pd.concat(align,ignore_index=True)
    summary=pd.DataFrame(qc)
    top=(all_align.assign(abs_pearson=lambda x:x.pearson_alignment.abs())
         .sort_values(['contrast_id','abs_pearson'],ascending=[True,False])
         .groupby('contrast_id',sort=False).head(10).copy())
    pairs=compare_primary_pairs(response,meta,overlaps,by_id)
    print('\n'+'='*92)
    print('CF-STATE 09T | FROZEN AERUPRECISE411 PROGRAMME ALIGNMENTS — NO REFITTING')
    print('='*92)
    print('Reference M:',n_M_raw,'original rows,',len(m),'usable PA genes,',m.shape[1],'iModulons')
    print('Reference A:',a.shape[0],'components x',a.shape[1],'samples')
    print('Frozen perturbation data:',meta.study.nunique(),'studies,',len(meta),'contrasts; primary',len(meta[meta.study.isin(PRIMARY)]))
    print('Measured response to M correlation is NOT a calibrated iModulon activity change.')
    print('GSE178622 DATS excluded (09r count matrix quarantined).')
    print('\nREFERENCE OVERLAP: accession AND exact GEO sample IDs')
    print(overlaps.to_string(index=False))
    print('\nCONTRAST COVERAGE AND MAX-CORRELATION SHIFT-NULL DIAGNOSTICS:')
    print(summary[['study','contrast_id','panel','n_gene_overlap','n_exact_zero_response',
                   'median_abs_response','max_abs_pearson_alignment',
                   'shift_null_max_abs_p_DIAGNOSTIC_ONLY']].round(4).to_string(index=False))
    for row in meta.itertuples(index=False):
        cid=row.contrast_id
        status=overlaps.set_index('study').loc[row.study,'reference_overlap_class']
        zz=top[top.contrast_id==cid]
        print('\nTOP FROZEN iMODULON ALIGNMENTS:',cid,'|',status)
        print(zz[['imodulon','pearson_alignment','spearman_alignment']].head(8).round(4).to_string(index=False))
    print('\nPRIMARY CONTRAST PAIRS: response and reference-profile similarities')
    print(pairs[['study_a','study_b','same_GEO_study','both_candidate_external',
                 'n_common_genes_for_gene_spearman','gene_response_spearman',
                 'frozen_imodulon_alignment_profile_pearson']].round(4).to_string(index=False))
    print('\nINTERPRETATION: frozen loadings are pre-trained on 411 profiles, including GSE138731.')
    print('Other studies are candidate external only: absence by accession/GSM is NOT full raw-read deduplication.')
    print('Loading-response correlations and shift-null maxima are descriptive, not biological sample-level inference.')
    print('Gene effects are normalized log(count+1) group differences, not harmonised DE logFC.')
    print('Programme signs follow the frozen M orientation; do not claim activation or CF specificity.')
    out.mkdir(parents=True,exist_ok=False)
    overlaps.to_csv(out/'REFERENCE_411_GEO_GSM_OVERLAP.csv',index=False)
    summary.to_csv(out/'CONTRAST_AND_NULL_DIAGNOSTICS.csv',index=False)
    all_align.to_csv(out/'ALL_FROZEN_IMODULON_LOADINGS_RESPONSE_ALIGNMENTS.csv',index=False)
    top.to_csv(out/'TOP_10_IMODULON_ALIGNMENTS_PER_CONTRAST.csv',index=False)
    pairs.to_csv(out/'PRIMARY_PAIR_ALIGNMENT_SIMILARITIES.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'SOURCE_SHA256.csv',index=False)
    (out/'QC_STATUS.json').write_text(json.dumps({
        'status':'09T_DESCRIPTIVE_FROZEN_IMODULON_ALIGNMENT_COMPLETE',
        'reference_iModulons':len(m.columns),'n_studies':meta.study.nunique(),
        'n_contrasts':len(meta),'reference_overlapping_primary':['GSE138731'],
        'external_confirmation':'ACCESSION_AND_GSM_ABSENCE_ONLY',
        'calibrated_iModulon_activity':False,'trained_predictor':False,
        'formal_sample_level_significance':False,'n_shift_nulls':n_shift
        },indent=2)+'\n')
    print('\nSUCCESS — saved without overwrite:',out)
    return top,summary,overlaps


def self_test():
    rng=np.random.default_rng(91016)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        for p in [root/REF,root/P,root/R]:p.mkdir(parents=True,exist_ok=True)
        genes=[f'PA{i:04d}' for i in range(1,601)]
        mods=['M0','M1','M2','M3']
        M=rng.normal(size=(600,4))
        pd.DataFrame(M,index=genes,columns=mods).to_csv(root/REF/'M.csv')
        pd.DataFrame(rng.normal(size=(4,8)),index=mods,columns=[f'S{i}' for i in range(8)]).to_csv(root/REF/'A.csv')
        pd.DataFrame({'GEO Series':['GSE138731']*6+['GSEOTHER']*2,
                      'GEO Sample':[f'GSM{4117161+i}' for i in range(6)]+['GSM999','GSM998'],
                      'project':['farnesol']*6+['other']*2}).to_csv(root/REF/'sample_table.csv',index=False)
        p_rows=[]
        cids=[]
        for s,cases in [('GSE179150',['KO']),('GSE167137',['MONO','CO']),
                        ('GSE190194',['ALE','COL','NV']),('GSE223663',['BIO','PLANK'])]:
            for short in cases:
                cid=s+'__'+short
                cids.append(cid);p_rows.append({'study':s,'contrast_id':cid})
        vals=rng.normal(size=(600,len(cids)))
        vals[:,0]=1.6*M[:,0]+0.1*rng.normal(size=600)
        pd.DataFrame(vals,index=genes,columns=cids).to_csv(root/P/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv')
        pd.DataFrame(p_rows).to_csv(root/P/'CONTRAST_SOURCE_DESIGN_AND_QC.csv',index=False)
        pd.DataFrame([{'study':s,'GSM':f'GSM{i+9000000}'} for i,s in enumerate(
             ['GSE179150']*6+['GSE167137']*16+['GSE190194']*11+['GSE223663']*12)]).to_csv(
            root/P/'SAMPLE_GEO_SERIES_TITLE_CROSSWALK.csv',index=False)
        (root/P/'QC_STATUS.json').write_text(json.dumps({'status':'09P_PROVISIONAL_MULTISTUDY_RESPONSE_CANDIDATES_FROZEN'}))
        rr=[];rm=[]
        for s,short in [('GSE138731','FARNESOL'),('GSE192340','DDDY')]:
            cid=s+'__'+short
            rm.append({'study':s,'contrast_id':cid})
            y=rng.normal(size=600)
            for g,v in zip(genes,y):rr.append({'GeneID':g,'contrast_id':cid,'effect_log2_count_over_median_ratio_plus1':v})
        pd.DataFrame(rr).to_csv(root/R/'PROVISIONAL_ADDITIONAL_GENE_RESPONSES.csv',index=False)
        pd.DataFrame(rm).to_csv(root/R/'PROVISIONAL_ADDITIONAL_CONTRASTS.csv',index=False)
        pd.DataFrame([{'study':s,'GSM':('GSM'+str(4117161+i) if s=='GSE138731' else 'GSM'+str(5745072+i))}
           for s in ('GSE138731','GSE192340') for i in range(6)]).to_csv(
               root/R/'QC_COUNT_COLUMN_TO_GEO_SERIES_TITLE.csv',index=False)
        (root/R/'QC_STATUS.json').write_text(json.dumps({'status':'09R_THREE_STUDY_ADDITIONAL_CONTRAST_QC_COMPLETE'}))
        out=root/'results/test_09t'
        top,summary,overlaps=run(root,out,strict=False,n_shift=17)
        need((out/'QC_STATUS.json').exists(),'expected status output')
        need(len(top)==40,'10 contrasts x 4 top matches in synthetic reference')
        need(overlaps.set_index('study').loc['GSE138731','reference_rows_with_study_accession']==6,
             'farnesol included overlap failed')
        need(overlaps.set_index('study').loc['GSE179150','union_matching_reference_rows']==0,
             'external candidate falsely marked overlapping')
        z=top[top.contrast_id=='GSE179150__KO'].iloc[0]
        need(z.imodulon=='M0' and z.pearson_alignment>0.9,'imprinted iModulon not top')
        try:run(root,out,strict=False,n_shift=3)
        except RuntimeError as e:need('refusing to overwrite' in str(e),'overwrite guard mismatch')
        else:raise AssertionError('overwriting was not prevented')
        # Reject wrong A component labels.
        a_path=root/REF/'A.csv'
        bad=pd.read_csv(a_path,index_col=0)
        bad.index=['Q0','Q1','Q2','Q3']
        bad.to_csv(a_path)
        try:load_reference(root,strict=False)
        except RuntimeError as e:need('M/A component identities' in str(e),'reference error incorrectly classified')
        else:raise AssertionError('bad M/A orientation not blocked')
    print('\nSELF_TEST_PASSED 09T: M/A component consistency, induced programme ranking,')
    print('  GEO/GSM overlap, no DATS inclusion, output provenance, non-overwrite, tampering rejection.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--n-shift',type=int,default=101)
    args=p.parse_args()
    if args.self_test:self_test()
    else:
        root=args.root.resolve()
        out=args.out if args.out.is_absolute() else root/args.out
        need(10<=args.n_shift<=500,'n-shift must be 10 to 500')
        run(root,out,n_shift=args.n_shift)


if __name__=='__main__':
    main()
