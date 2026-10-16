#!/usr/bin/env python3
"""CF-State 09r: conservative 3-study add-on contrast candidate freeze.

Studies: GSE138731 farnesol; GSE178622 DATS; GSE192340 DDDY.
GEO sample titles are used to corroborate count-column group/replicate labels,
never to claim original run-to-processed-count provenance. No row-order mapping.
Reject entire study as count-scale-ineligible if any numeric retained PA-locus
row contains a fractional value; missing rows are quarantined, not imputed.
Per-study size factors and descriptive group mean log2(count/sf+1) differences;
NOT DESeq2 log fold changes, not independent validation, not directly comparable
across studies. Existing audits are untouched.
"""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import tempfile
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
COUNT_DIR = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Count_Tables')
DESIGN_DIR = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Design_Matrices')
MASK = Path('results/phase9/audit_09b_Neff2024_paired_gene_response_freeze/FROZEN_4923_gene_combined_metals_response_summary.csv')
OUT = Path('results/phase9/audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze')
STUDIES = ('GSE138731','GSE178622','GSE192340')
BARE = re.compile(r'^PA\d{4}$')


def need(ok, msg):
    if not bool(ok):
        raise RuntimeError('AUDIT_09R_FAILED: '+str(msg))


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20), b''):
            h.update(b)
    return h.hexdigest()


def sample_map(study):
    z={}
    if study=='GSE138731':
        for i in (1,2,3):
            for p,group,base in [('0','Control',4117160),('6','Farnesol',4117163)]:
                # GEO titles 'RNA of 0-1F_L1' and 'RNA of 6-1F_L1'
                key=f'{p}-{i}F'
                z[key]={'GSM':f'GSM{base+i}', 'GEO_title':f'RNA of {key}_L1',
                        'group':group, 'replicate':i}
    elif study=='GSE178622':
        for i in (1,2,3):
            z[f'Control{i}']={'GSM':f'GSM{5394596+i}', 'GEO_title':f'RNA of control{i} (DATS control)',
                             'group':'Control','replicate':i}
            z[f'DTS{i}']={'GSM':f'GSM{5394599+i}', 'GEO_title':f'RNA of DATS{i}',
                         'group':'DATS','replicate':i}
    elif study=='GSE192340':
        for i in (1,2,3,4,5):
            z[f'P.CK{i}']={'GSM':f'GSM{5745071+i}','GEO_title':f'P_CK{i}',
                            'group':'Control','replicate':i}
            z[f'P.DDDY{i}']={'GSM':f'GSM{5745076+i}','GEO_title':f'P_DDDY{i}',
                             'group':'DDDY','replicate':i}
    return z


def resolve(study, name):
    # Derive group and replicate independently of count column position.
    raw=name.strip().lstrip('\ufeff')
    if study=='GSE138731':
        m=re.fullmatch(r'([06])-([123])F_mapped_reads\(\d+\)',raw)
        if m: return f'{m.group(1)}-{m.group(2)}F'
    if study=='GSE178622':
        m=re.fullmatch(r'(Control|DTS)([123])_count',raw)
        if m: return m.group(1)+m.group(2)
    if study=='GSE192340':
        m=re.fullmatch(r'P\.(CK|DDDY)([12345])',raw)
        if m: return f'P.{m.group(1)}{m.group(2)}'
    return None


def design_check(study,names,design):
    need({'Treatment','Replicate'}.issubset(design.columns),study+' missing design Treatment/Replicate')
    lut=sample_map(study)
    refs=[resolve(study,x) for x in names]
    need(None not in refs, study+' unmatched count column(s): '+repr([x for x,r in zip(names,refs) if r is None]))
    need(len(refs)==len(set(refs)),study+' duplicate count sample tokens')
    need(set(refs)==set(lut),study+' GEO sample token mismatch: missing='+repr(set(lut)-set(refs)))
    groups=Counter((lut[r]['group'],lut[r]['replicate']) for r in refs)
    groups_design=Counter((str(z['Treatment']).strip(),int(z['Replicate'])) for z in design.to_dict('records'))
    need(groups==groups_design,study+' GEO and EPathDash design group/replicate multiset mismatch: '+repr(groups)+' vs '+repr(groups_design))
    records=[]
    for name,token in zip(names,refs):
        r=lut[token]
        records.append({'study':study,'count_column':name,'name_derived_token':token,
                        'GSM':r['GSM'],'GEO_series_sample_title':r['GEO_title'],
                        'treatment_group':r['group'],'replicate_token':r['replicate'],
                        'mapping_evidence':'GEO_SERIES_TITLE_AND_COUNT_COLUMN_TOKEN',
                        'SRR_to_processed_column_proven':False})
    return pd.DataFrame(records)


def calculate_counts(study,path,mask,min_eligible):
    z=pd.read_csv(path,low_memory=False)
    need(z.shape[1]>=4,study+' malformed count table')
    ids=z.iloc[:,0].astype(str).str.strip().str.lstrip('\ufeff')
    bare=ids.str.fullmatch(r'PA\d{4}')
    valid_mask=bare & ids.isin(mask)
    selected=z.loc[valid_mask].copy()
    selected.iloc[:,0]=ids.loc[valid_mask]
    need(selected.iloc[:,0].is_unique,study+' duplicated bare PA loci in mask')
    selected=selected.set_index(z.columns[0])
    numbers=selected.apply(pd.to_numeric,errors='coerce')
    X=numbers.to_numpy(float)
    bad=np.any(~np.isfinite(X),axis=1)
    negative=np.any(np.isfinite(X)&(X<0),axis=1)
    fractional=np.any(np.isfinite(X)&(np.abs(X-np.round(X))>1e-7),axis=1)
    eligible=(~bad)&(~negative)&(~fractional)
    qc={'study':study,'n_raw_gene_rows':int(len(z)),'n_bare_mask_rows':int(len(selected)),
        'n_missing_value_rows':int(bad.sum()),'n_negative_rows':int(negative.sum()),
        'n_fractional_numeric_rows':int(fractional.sum()),'n_integer_complete_eligible_rows':int(eligible.sum()),
        'fraction_bad_or_missing':float(np.mean(~eligible)) if len(eligible) else 1.0}
    # Mixed fractional/nonfractional 'counts' cannot be silently rescued by discarding affected genes.
    # Entire dataset is quarantined pending independent measurement-unit verification.
    if fractional.any():
        qc['status']='QUARANTINE_FRACTIONAL_COUNT_MATRIX_UNITS_UNVERIFIED'
        return None,qc,selected.columns.tolist()
    if int(eligible.sum())<min_eligible or float(np.mean(~eligible))>0.08:
        qc['status']='QUARANTINE_INSUFFICIENT_COMPLETE_INTEGER_GENE_ROWS'
        return None,qc,selected.columns.tolist()
    q=numbers.iloc[np.flatnonzero(eligible)].copy()
    need((q.to_numpy()>=0).all(),study+' negative after filter')
    need(np.min(q.sum(axis=0))>500,study+' inadequate sample library sum')
    qc['status']='COUNT_SCALE_ELIGIBLE_PROVISIONAL'
    return q,qc,q.columns.tolist()


def normalize(q,study):
    x=q.to_numpy(float)
    ref=(x>0).all(axis=1)
    need(ref.sum()>=250,study+' insufficient strictly positive genes for median ratio')
    gm=np.exp(np.log(x[ref]).mean(axis=1))
    sf=np.median(x[ref]/gm[:,None],axis=0)
    need(np.isfinite(sf).all() and (sf>0).all(),study+' invalid size factors')
    sf/=np.exp(np.mean(np.log(sf)))
    log=np.log2(x/sf[None,:]+1)
    return pd.DataFrame(log,index=q.index,columns=q.columns),pd.Series(sf,index=q.columns),int(ref.sum())


def execute(root,out,min_eligible=1500):
    need(not out.exists(),'refusing overwrite: '+str(out))
    need((root/MASK).is_file(),'09b mask missing')
    mask=set(pd.read_csv(root/MASK,dtype={'GeneID':str}).GeneID.dropna().astype(str))
    need(len(mask)>300,'invalid frozen mask')
    samples=[]; studyqc=[]; contrasts=[]; responses=[]; provenance=[]
    for study in STUDIES:
        cp=root/COUNT_DIR/(study+'_counts.csv')
        dp=root/DESIGN_DIR/(study+'_design.csv')
        need(cp.is_file() and dp.is_file(),'missing EPathDash sources for '+study)
        for p in (cp,dp): provenance.append({'study':study,'path':str(p.resolve()),'sha256':sha(p)})
        design=pd.read_csv(dp,encoding='utf-8-sig')
        q,qc,names=calculate_counts(study,cp,mask,min_eligible)
        cross=design_check(study,names,design)
        samples.append(cross)
        qc['n_samples']=len(names)
        qc['n_count_name_GEO_token_matches']=len(cross)
        qc['GEO_group_count_corroborated']=True
        if q is None:
            studyqc.append(qc)
            continue
        log,sf,nref=normalize(q,study)
        cross['median_ratio_sf']=cross.count_column.map(sf)
        samples[-1]=cross
        qc['n_sf_ref_genes']=nref
        qc['min_sf']=float(sf.min()); qc['max_sf']=float(sf.max())
        studyqc.append(qc)
        refs=sample_map(study)
        treated={'GSE138731':'Farnesol','GSE178622':'DATS','GSE192340':'DDDY'}[study]
        mapping={row.count_column:row.treatment_group for row in cross.itertuples()}
        ccols=[x for x in log.columns if mapping[x]=='Control']
        tcols=[x for x in log.columns if mapping[x]==treated]
        need(len(ccols)==len(tcols) and len(ccols)>=3,study+' group sample counts differ')
        y=log[tcols].mean(axis=1)-log[ccols].mean(axis=1)
        cid=study+'__'+treated.upper()+'_vs_CONTROL'
        contrasts.append({'study':study,'contrast_id':cid,'treatment':treated,'control':'Control',
                          'n_treatment_samples':len(tcols),'n_control_samples':len(ccols),
                          'n_eligible_genes':len(y),'median_abs_effect':float(y.abs().median()),
                          'n_up':int((y>0).sum()),'n_down':int((y<0).sum()),
                          'release_status':'PROVISIONAL_GEO_TITLE_CORROBORATED_COUNT_SCALE_ELIGIBLE',
                          'raw_run_processed_provenance_confirmed':False})
        responses.extend({'study':study,'contrast_id':cid,'GeneID':gid,
                          'effect_log2_count_over_median_ratio_plus1':float(val)} for gid,val in y.items())
    qcs=pd.DataFrame(studyqc)
    xr=pd.concat(samples,ignore_index=True)
    cs=pd.DataFrame(contrasts)
    rr=pd.DataFrame(responses)
    print('\n'+'='*90)
    print('CF-STATE 09R | THREE ADDITIONAL PAO1 TREATMENT STUDIES | QC & CANDIDATE FREEZE')
    print('='*90)
    print('Expected GEO series count samples:',sum(len(sample_map(s)) for s in STUDIES),
          '| actual count-to-GEO sample-token matches:',len(xr))
    print('Raw SRR -> processed count-column provenance remains unproven.')
    print('Count units must be integer; any fractional values quarantine WHOLE study.')
    print('\nPER-STUDY COUNT ELIGIBILITY:')
    print(qcs.to_string(index=False))
    print('\nRECONCILED SAMPLE GROUP COUNTS:')
    print(xr.groupby(['study','treatment_group'],as_index=False).agg(n=('count_column','count'),
          unique_replicates=('replicate_token','nunique')).to_string(index=False))
    print('\nSTUDY RESPONSE CANDIDATES (not differential-expression inference):')
    if len(cs):
        print(cs.to_string(index=False))
        for cid in cs.contrast_id:
            z=rr[rr.contrast_id==cid].copy()
            z['absolute']=z.effect_log2_count_over_median_ratio_plus1.abs()
            print('\nTOP 8 |effect|:',cid)
            print(z.nlargest(8,'absolute')[['GeneID','effect_log2_count_over_median_ratio_plus1']].to_string(index=False))
    else:
        print('NONE. All studies quarantined. Existing 09p results unchanged.')
    print('\nPRIMARY GATE: Do not fit to quarantined studies. Candidate controls and GEO titles')
    print('corroborated but full count processing provenance not independently proven.')
    print('No samples or gene responses from pre-existing 09p reused here.')
    out.mkdir(parents=True,exist_ok=False)
    qcs.to_csv(out/'QC_STUDY_COUNT_SCALE_AND_COMPLETENESS.csv',index=False)
    xr.to_csv(out/'QC_COUNT_COLUMN_TO_GEO_SERIES_TITLE.csv',index=False)
    cs.to_csv(out/'PROVISIONAL_ADDITIONAL_CONTRASTS.csv',index=False)
    rr.to_csv(out/'PROVISIONAL_ADDITIONAL_GENE_RESPONSES.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'INPUT_SHA256_PROVENANCE.csv',index=False)
    status={'status':'09R_THREE_STUDY_ADDITIONAL_CONTRAST_QC_COMPLETE',
            'n_studies_inspected':len(STUDIES),'n_eligible_studies':len(cs),
            'n_quarantined_studies':len(STUDIES)-len(cs),
            'n_unique_GEO_title_tokens':len(xr),'count_to_SRR_proven':False,
            'source_09p_unchanged':True,'n_studies_with_fractional_values':int((qcs.n_fractional_numeric_rows>0).sum()),
            'is_formal_DE_analysis':False}
    (out/'QC_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    print('\nSUCCESS | saved (non-overwriting):',out)
    return qcs,cs,xr,rr


def self_test():
    rng=np.random.default_rng(20261016)
    with tempfile.TemporaryDirectory() as tmp:
        rt=Path(tmp)
        (rt/MASK.parent).mkdir(parents=True,exist_ok=True)
        pd.DataFrame({'GeneID':[f'PA{i:04d}' for i in range(1,1801)]}).to_csv(rt/MASK,index=False)
        for study in STUDIES:
            cp=rt/COUNT_DIR/(study+'_counts.csv')
            dp=rt/DESIGN_DIR/(study+'_design.csv')
            cp.parent.mkdir(parents=True,exist_ok=True);dp.parent.mkdir(parents=True,exist_ok=True)
            lut=sample_map(study)
            z=[]
            for k in lut:
                if study=='GSE138731':name=k+'_mapped_reads(1234567)'
                elif study=='GSE178622':name=k+'_count'
                else:name=k
                z.append((name,k))
            z=z[::-1]
            dat=pd.DataFrame({'GeneID':[f'PA{i:04d}' for i in range(1,1801)]})
            for name,k in z:
                factor=2 if lut[k]['group']!='Control' else 1
                dat[name]=rng.poisson(70*factor,size=1800)
            if study=='GSE138731':
                # pandas >= 2.2 rejects inserting a string in an integer column.
                # This intentionally simulates a source CSV missing-value marker.
                value_col = dat.columns[1]
                dat[value_col] = dat[value_col].astype(object)
                dat.at[dat.index[0], value_col] = '-'
            if study=='GSE178622':
                value_col = dat.columns[1]
                dat[value_col] = dat[value_col].astype(float)
                dat.at[dat.index[0], value_col] = 25.5  # quarantine entire mixed-scale table
            dat.to_csv(cp,index=False)
            design=pd.DataFrame([{'Treatment':v['group'],'Replicate':v['replicate']} for v in lut.values()])
            design.sample(frac=1,random_state=34).to_csv(dp,index=False)
        out=rt/'results/09r_test'
        q,c,x,r=execute(rt,out,min_eligible=1500)
        need(set(c.study)=={'GSE138731','GSE192340'},'expected synthetic eligible studies incorrect')
        need(q.set_index('study').loc['GSE178622','status'].startswith('QUARANTINE'),
             'fractional values not quarantined')
        need(len(x)==22,'synthetic GEO token count expected 6+6+10')
        need(r.GeneID.nunique()>=1799,'unexpected gene loss after one missing row')
        try:execute(rt,out,min_eligible=1500)
        except RuntimeError as e:need('refusing overwrite' in str(e),'wrong overwrite exception')
        else:raise AssertionError('overwrite not blocked')
        print('SELF_TEST_PASSED 09R: shuffled sample columns and design rows, GEO token checks,')
        print('  missing-value quarantine by gene, whole-study mixed-fraction quarantine,')
        print('  normalized responses, provenance, non-overwrite.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        root=a.root.resolve()
        out=a.out if a.out.is_absolute() else root/a.out
        execute(root,out)

if __name__=='__main__':main()
