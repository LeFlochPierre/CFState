#!/usr/bin/env python3
"""CF-State 09p: source-corroborated EPathDash multi-study response candidates.

Uses GEO SERIES SAMPLE TITLES checked 2026-10-15, local EPathDash design/count
matrices, and the previously frozen CF-State 4923-gene mask. Refuses to use row
order as proof of sample identity. Generates descriptive, unpaired group mean
log2(normalized count+1) differences, NOT inferential DE log fold changes.

Provenance limitation: count-column to GEO-title correspondence is verified by
explicit experimental group and replicate tokens, not by an original-run GSM/SRR
identifier embedded in the processed count column. These are provisional
GEO-title-corroborated contrasts, not prospectively independent test samples.

Original GEO: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE179150
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE167137
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190194
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE223663
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

D = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Design_Matrices')
C = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Count_Tables')
M = Path('results/phase9/audit_09b_Neff2024_paired_gene_response_freeze/FROZEN_4923_gene_combined_metals_response_summary.csv')
OUT = Path('results/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze')
STUDIES = ('GSE179150','GSE167137','GSE190194','GSE223663')
BARE = re.compile(r'^PA\d{4}$')


def check(ok, message):
    if not bool(ok):
        raise RuntimeError('AUDIT_09P_FAILED: ' + str(message))


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def expected_samples(study):
    """Hand-transcribed original GEO series sample design and count-token crosswalk.

    Returns for EACH count header a (GSM, original GEO sample title,
    experimental factors, replicate identifier, identity match class).
    """
    rows = []
    def add(name, gsm, title, factors, replicate, match):
        rows.append(dict(count_sample=name,GSM='GSM'+str(gsm),geo_sample_title=title,
                         factors=factors,replicate_token=str(replicate),geo_crosswalk_method=match))
    if study == 'GSE179150':
        for i in (1,2,3):
            add(f'PAO1161_p{i}_htseq',5410639+i,f'PAO1161 WT_{i} RNA-seq',
                {'Genotype':'WT'},i,'GEO_TITLE_GENOTYPE_AND_REPLICATE_TOKEN')
            add(f'PA2504_p{i}_htseq',5410642+i,f'PAO1161 2504_{i} RNA-seq',
                {'Genotype':'Delta-PA2504'},i,'GEO_TITLE_GENOTYPE_AND_REPLICATE_TOKEN')
    elif study == 'GSE167137':
        for j,rep in enumerate((4,6,7,8)):
            base = 5097904+6*j
            for prefix,treated,offset,medium in (
                ('PA',False,0,'Monoculture'),('PACA',False,2,'Coculture'),
                ('PA',True,3,'Monoculture'),('PACA',True,5,'Coculture')):
                token='5' if treated else '0'
                name=f'{prefix}-{token}M-{rep}'
                add(name,base+offset,name,{'Medium':medium,
                    'Treatment':'meropenem_5mg' if treated else 'Control'},rep,
                    'EXACT_GEO_SERIES_SAMPLE_TITLE')
    elif study == 'GSE190194':
        for prefix,title,group,gsm_start,rep_n in (
            ('ctrl','P. aeruginosa PAO1 (WT) non-treatment_rep','control',5717979,3),
            ('COL','P. aeruginosa PAO1 (WT) treated by colistin_rep','colistin',5717982,3),
            ('ALE','P. aeruginosa PAO1 (WT) treated by alexidine_rep','alexidine',5717985,3),
            ('NV716','P. aeruginosa PAO1 (WT) treated by NV716_rep','NV716',5717988,2)):
            for i in range(1,rep_n+1):
                add(f'{prefix}__{i}',gsm_start+i-1,f'{title} {i}',
                    {'Treatment':group},i,'GEO_TITLE_TREATMENT_AND_REPLICATE_TOKEN')
    elif study == 'GSE223663':
        for group,pattern,title,gsm0 in (
            ('B_WT','BioFWTBR{rep}_S{run}','Biofilm_WT_BR',6970256),
            ('P_WT','PlankWTBR{rep}_S{run}','Planktonic_WT_BR',6970259),
            ('B_MUT','BF0285BR{rep}_S{run}','Biofilm_0285_BR',6970262),
            ('P_MUT','Pla0285BR{rep}_S{run}','Planktonic_0285_BR',6970265)):
            for i in (1,2,3):
                run = {'B_WT':9+i,'P_WT':i,'B_MUT':15+i,'P_MUT':6+i}[group]
                name=pattern.format(rep=i,run=run)
                add(name,gsm0+i-1,f'{title}{i}',
                    {'Genotype':'WT' if 'WT' in group else 'deltaPA0285',
                     'Growth':'Biofilm' if group.startswith('B_') else 'Planktonic'},
                    i,'GEO_TITLE_GROUP_REPLICATE_AND_SOURCE_SAMPLE_TOKEN')
    check(len({r['count_sample'] for r in rows}) == len(rows), 'duplicate GEO count-name crosswalk')
    return {r['count_sample']:r for r in rows}


def validate_design(study, count_names, design):
    check(len(count_names)==len(set(count_names)),study+' duplicate count column')
    lookup = expected_samples(study)
    miss = sorted(set(count_names) - set(lookup))
    surplus = sorted(set(lookup) - set(count_names))
    check(not miss and not surplus,study+' unexpected count/GEO sample headers: missing='+str(miss)+', surplus='+str(surplus))
    if 'Replicate' in design.columns:
        factors = [c for c in design.columns if c!='Replicate']
    else:
        factors = list(design.columns)
    check(bool(factors),study+' missing design factors')
    check(all(f in lookup[next(iter(lookup))]['factors'] for f in factors),study+' unknown design factor')
    count_groups=Counter(tuple(lookup[n]['factors'][k] for k in factors) for n in count_names)
    design_groups=Counter(tuple(str(row[k]).strip() for k in factors) for row in design.to_dict('records'))
    check(count_groups==design_groups,study+' group multiplicities disagree with GEO sample-title crosswalk: '+str(count_groups)+' vs '+str(design_groups))
    check(len(count_names)==len(design),study+' design rows/sample counts mismatch')
    # Optional 'Replicate' column: demand that its multiset matches original GEO
    # sample replicate tokens within each experimental group, even if row order differs.
    if 'Replicate' in design.columns:
        # Sample suffixes are NOT necessarily the same labels as design replicate
        # numbers. Example: GSE167137 design reps 1-4, GEO sample suffixes
        # 4/6/7/8. Compare replicate GROUP STRUCTURES without aligning either
        # row order or arbitrary replicate-label identities.
        def replicate_signatures(pairs):
            by_rep={}
            for group,token in pairs:
                by_rep.setdefault(token,Counter())[group]+=1
            return Counter(tuple(sorted(v.items())) for v in by_rep.values())
        design_pairs=[(tuple(str(row[k]).strip() for k in factors),str(row['Replicate']).strip()) for row in design.to_dict('records')]
        geo_pairs=[(tuple(lookup[n]['factors'][k] for k in factors),lookup[n]['replicate_token']) for n in count_names]
        check(replicate_signatures(design_pairs)==replicate_signatures(geo_pairs),
              study+' biological-replicate group structure conflicts with GEO titles')
    return lookup


def load_counts(path, study, gene_mask, min_overlap):
    z=pd.read_csv(path,low_memory=False)
    check(z.shape[1]>=5,study+' count table has too few columns')
    gene_col=z.columns[0]
    ids=z.iloc[:,0].astype(str).str.strip()
    keep=ids.map(lambda x:bool(BARE.fullmatch(x)))
    selected=z.loc[keep].copy()
    selected.iloc[:,0]=ids.loc[keep]
    check(selected.iloc[:,0].is_unique,study+' duplicated PAO1 bare loci; do not aggregate silently')
    selected=selected.set_index(gene_col)
    check(len(selected)>min_overlap,study+' insufficient PAO1-ID mapping')
    check(len(set(selected.index)&gene_mask)>=min_overlap,
          study+' insufficient overlap with frozen 4923 gene mask')
    nums=selected.apply(pd.to_numeric,errors='coerce')
    counts=nums.to_numpy(float)
    check(np.isfinite(counts).all(),study+' NA/non-numeric counts in mapped PA genes')
    check((counts>=0).all(),study+' negative count')
    check(np.max(np.abs(counts-np.round(counts)))<1e-6,study+' noninteger read-count values')
    check(np.min(counts.sum(axis=0))>100,study+' zero or extremely small PA gene count library')
    return nums


def normalize(counts):
    """Within-study DESeq-style median-ratio factors, descriptive log2(count/sf+1)."""
    x=counts.to_numpy(dtype=float)
    ok=(x>0).all(axis=1)
    check(ok.sum()>=200,'fewer than 200 all-positive PAO1 genes for median ratios')
    gm=np.exp(np.log(x[ok]).mean(axis=1))
    sf=np.median(x[ok]/gm[:,None],axis=0)
    check(np.isfinite(sf).all() and np.min(sf)>0,'invalid median-ratio size factor')
    sf/=np.exp(np.mean(np.log(sf)))
    lognorm=np.log2(x/sf[None,:]+1)
    return pd.DataFrame(lognorm,index=counts.index,columns=counts.columns),pd.Series(sf,index=counts.columns),int(ok.sum())


def candidate_contrasts(study, lookup):
    """Predeclared treatment minus control; no choosing contrasts from outcomes."""
    if study=='GSE179150':
        return [('KO_PA2504_vs_WT',{'Genotype':'Delta-PA2504'},{'Genotype':'WT'},'GENETIC_KO','PAO1161_derivative')]
    if study=='GSE167137':
        return [
          ('MEROPENEM_MONOCULTURE',{'Medium':'Monoculture','Treatment':'meropenem_5mg'},
           {'Medium':'Monoculture','Treatment':'Control'},'ANTIBIOTIC_ACUTE','PAO1'),
          ('MEROPENEM_COCULTURE',{'Medium':'Coculture','Treatment':'meropenem_5mg'},
           {'Medium':'Coculture','Treatment':'Control'},'ANTIBIOTIC_ACUTE_COCULTURE','PAO1_Candida_co_culture')]
    if study=='GSE190194':
        return [(g.upper()+'_LONG_PASSAGE',{'Treatment':g},{'Treatment':'control'},
                 'LONG_TERM_ADAPTATION','PAO1') for g in ('alexidine','colistin','NV716')]
    if study=='GSE223663':
        return [('KO_PA0285_BIOFILM',{'Genotype':'deltaPA0285','Growth':'Biofilm'},
                 {'Genotype':'WT','Growth':'Biofilm'},'GENETIC_KO_BIOFILM','PAO1'),
                ('KO_PA0285_PLANKTONIC',{'Genotype':'deltaPA0285','Growth':'Planktonic'},
                 {'Genotype':'WT','Growth':'Planktonic'},'GENETIC_KO_PLANKTONIC','PAO1')]
    return []


def group_samples(lookup, group):
    return sorted([name for name,entry in lookup.items() if all(entry['factors'].get(k)==v for k,v in group.items())])


def run(root, out, min_overlap=2500):
    check(not out.exists(),'refusing to overwrite '+str(out))
    mask_file=root/M
    check(mask_file.is_file(),'missing frozen 09b metal mask: '+str(mask_file))
    mask=pd.read_csv(mask_file,usecols=['GeneID'])
    check(mask.GeneID.is_unique,'duplicate mask IDs')
    frozen=set(mask.GeneID.astype(str))
    provenance=[{'source_path':str(M),'sha256':sha256(mask_file)}]
    responses=[]; crosswalk=[]; qc=[]; contrast_summary=[]
    normed_all={}
    for study in STUDIES:
        despath=root/D/(study+'_design.csv')
        counts_path=root/C/(study+'_counts.csv')
        check(despath.is_file() and counts_path.is_file(),'missing '+study+' count/design source')
        design=pd.read_csv(despath,dtype=str,keep_default_na=False)
        design.columns=[str(k).strip() for k in design.columns]
        design=design.map(lambda v:str(v).strip())
        counts=load_counts(counts_path,study,frozen,min_overlap)
        lookup=validate_design(study,list(counts.columns),design)
        lognorm,sf,nref=normalize(counts)
        normed_all[study]=lognorm
        for path in (despath,counts_path):
            provenance.append({'source_path':str(path.relative_to(root)),'sha256':sha256(path)})
        overlap=set(counts.index)&frozen
        qc.append({'study':study,'n_count_columns':len(counts.columns),'n_GEO_titles_checked':len(lookup),
                   'n_bare_PAO1_genes':len(counts),'n_shared_frozen_4923':len(overlap),
                   'n_all_positive_size_factor_ref_genes':nref,
                   'min_size_factor':float(sf.min()),'max_size_factor':float(sf.max()),
                   'n_biological_rep_tokens':len(set(x['replicate_token'] for x in lookup.values())),
                   'mapping_status':'GEO_SERIES_TITLE_TOKEN_CORROBORATED',
                   'exact_raw_SRR_to_count_column_verification':False,
                   'historical_CFSTATE_study_overlap':study=='GSE223663'})
        for name in counts.columns:
            entry=lookup[name]
            crosswalk.append({'study':study,'count_sample':name,'GSM':entry['GSM'],
                              'GEO_series_title':entry['geo_sample_title'],
                              'experimental_factors':json.dumps(entry['factors'],sort_keys=True),
                              'replicate_token':entry['replicate_token'],
                              'sample_label_match':entry['geo_crosswalk_method'],
                              'GEO_url':'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc='+entry['GSM'],
                              'within_study_median_ratio_size_factor':float(sf[name]),
                              'count_sum_PA_bare':float(counts[name].sum())})
        for name,treat,control,kind,strain in candidate_contrasts(study,lookup):
            tcols=group_samples(lookup,treat)
            ccols=group_samples(lookup,control)
            check(len(tcols)>=2 and len(ccols)>=2,study+'/'+name+' inadequate group size')
            check(not set(tcols)&set(ccols),'treatment/control group overlap')
            common=sorted(overlap)
            y=(lognorm.loc[common,tcols].mean(axis=1)-lognorm.loc[common,ccols].mean(axis=1))
            rep_t={lookup[n]['replicate_token'] for n in tcols}
            rep_c={lookup[n]['replicate_token'] for n in ccols}
            # The paired-replicate structure is metadata only; summary is
            # difference of group means, not a paired inferential analysis.
            identical_reps=rep_t==rep_c
            for gene,effect in y.items():
                responses.append({'study':study,'contrast_id':study+'__'+name,'GeneID':gene,
                     'effect_log2_within_study_normalized_count_plus1_difference':float(effect),
                     'perturbation_class':kind,'background':strain,
                     'historical_CFSTATE_overlap':study=='GSE223663'})
            contrast_summary.append({'study':study,'contrast_id':study+'__'+name,
                 'treatment_group':json.dumps(treat,sort_keys=True),
                 'control_group':json.dumps(control,sort_keys=True),
                 'n_treatment_samples':len(tcols),'n_control_samples':len(ccols),
                 'n_mapped_response_genes':len(common),
                 'n_up':int((y>0).sum()),'n_down':int((y<0).sum()),
                 'mean_abs_effect':float(np.abs(y).mean()),
                 'median_abs_effect':float(np.abs(y).median()),
                 'matched_replicate_tokens_across_groups':bool(identical_reps),
                 'intervention_category':kind,'background':strain,
                 'historical_CFSTATE_overlap':study=='GSE223663',
                 'primary_modelling_status':
                     'SECONDARY_INTERNAL_OVERLAP' if study=='GSE223663' else
                     'SECONDARY_LONG_TERM_ADAPTATION' if study=='GSE190194' else
                     'PROVISIONAL_PRIMARY_GEO_TITLE_CORROBORATED',
                 'raw_SRR_to_count_column_proven':False,
                 'effect_scale':'within-study log2(raw count / median-ratio SF + 1) group mean difference'})
    r=pd.DataFrame(responses)
    x=pd.DataFrame(crosswalk)
    q=pd.DataFrame(qc)
    c=pd.DataFrame(contrast_summary)
    check(not r.duplicated(['contrast_id','GeneID']).any(),'duplicate response gene key')
    check(q.n_count_columns.sum()==len(x),'sample map total mismatch')
    common=sorted(set.intersection(frozen,*(set(mat.index) for mat in normed_all.values())))
    check(len(common)>=min_overlap,'too few genes shared across all four studies')
    wide=(r[r.GeneID.isin(common)].pivot(index='GeneID',columns='contrast_id',
            values='effect_log2_within_study_normalized_count_plus1_difference')
          .loc[common, c.contrast_id.tolist()])
    check(wide.notna().sum().min()==len(common),'missing common gene-contrast value')
    print('\n'+'='*96)
    print('CF-STATE 09P | GEO-TITLE-CORROBORATED MULTISTUDY RESPONSE CANDIDATE FREEZE')
    print('='*96)
    print('Studies:',len(q),'| contrasts:',len(c),'| unique studies still =',len(q))
    print('Common genes in ALL four studies + frozen response mask:',len(common))
    print('GEO series sample-title crosswalks checked:',len(x))
    print('NOTE: Counts are locally available; raw-run GSM/SRR->processed-column provenance remains unproven.')
    print('NOT cross-study comparable DE log fold changes; no inferential DE tests.')
    print('\nSTUDY AND SAMPLE QC:')
    print(q.to_string(index=False))
    print('\nCONTRASTS (treatment minus control, with predeclared context):')
    print(c[['contrast_id','n_treatment_samples','n_control_samples','n_mapped_response_genes',
             'n_up','n_down','median_abs_effect','matched_replicate_tokens_across_groups',
             'primary_modelling_status']].to_string(index=False))
    print('\nSAMPLE-to-GEO CROSSWALK (first 4/sample context):')
    print(x.groupby('study',sort=False).head(4)[['study','count_sample','GSM','GEO_series_title',
                   'replicate_token','within_study_median_ratio_size_factor']].to_string(index=False))
    print('\nLARGEST RESPONSES (top 5 |effect| for each contrast; DESCRIPTIVE):')
    for cid in c.contrast_id:
        top=r[r.contrast_id==cid].copy()
        top['abs_effect']=top.effect_log2_within_study_normalized_count_plus1_difference.abs()
        top=top.nlargest(5,'abs_effect')
        print('\n '+cid)
        print(top[['GeneID','effect_log2_within_study_normalized_count_plus1_difference']].to_string(index=False))
    print('\nRELEASE LIMITS: Candidate response matrix, not a biological validity or blind validation claim.')
    print('GSE167137 coculture contains Candida; GSE190194 is ~50-passage adaptation (NV716 n=2);')
    print('GSE223663 previously encountered in CF-State; GSE199323 excluded (colony-state confounding).')
    print('Whole GEO study is the evaluation unit; no treating genes/contrasts as independent studies.')
    out.mkdir(parents=True,exist_ok=False)
    r.to_csv(out/'FROZEN_PROVISIONAL_GEO_CORROBORATED_GENE_RESPONSES.csv',index=False)
    wide.to_csv(out/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv')
    c.to_csv(out/'CONTRAST_SOURCE_DESIGN_AND_QC.csv',index=False)
    x.to_csv(out/'SAMPLE_GEO_SERIES_TITLE_CROSSWALK.csv',index=False)
    q.to_csv(out/'SAMPLE_COUNT_ID_AND_MAPPING_QC.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'INPUT_SHA256_PROVENANCE.csv',index=False)
    state={'status':'09P_PROVISIONAL_MULTISTUDY_RESPONSE_CANDIDATES_FROZEN',
           'studies':len(q),'contrasts':len(c),'n_common_response_genes':len(common),
           'count_GEO_series_sample_titles_corroborated':len(x),
           'raw_run_to_count_link_proven':False,'heldout_performance_measured':False,
           'has_long_passage_and_internal_overlap_flags':True,
           'gene_scale':'within-study normalized log count difference; not DE log fold change',
           'no_overwrite':True}
    (out/'QC_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    print('\nSUCCESS | saved without overwrite:',out)
    return q,c,x,r


def self_test():
    # Synthetic fixtures with REVERSED count-column order, so no positional shortcut.
    rng=np.random.default_rng(20261015)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for sub in (D,C,M.parent):
            (root/sub).mkdir(parents=True,exist_ok=True)
        genes=[f'PA{i:04d}' for i in range(1,501)]
        pd.DataFrame({'GeneID':genes}).to_csv(root/M,index=False)
        for study in STUDIES:
            mapping=expected_samples(study)
            names=list(reversed(list(mapping)))
            dims=list(mapping[names[0]]['factors'])
            vals=pd.DataFrame({'GeneID':genes})
            for name in names:
                g=mapping[name]['factors']
                shift=1.2 if ('Delta' in str(g) or 'meropenem' in str(g) or 'alexidine' in str(g) or 'deltaPA' in str(g)) else 0
                vals[name]=rng.poisson(100*np.exp(shift),size=len(genes))
            vals.to_csv(root/C/(study+'_counts.csv'),index=False)
            design=[]
            for name in names:
                z=mapping[name]
                rep=z['replicate_token']
                if study=='GSE167137':
                    rep={'4':'1','6':'2','7':'3','8':'4'}[rep]
                design.append(dict(z['factors'],Replicate=rep))
            pd.DataFrame(design).sample(frac=1,random_state=3).to_csv(root/D/(study+'_design.csv'),index=False)
        q,c,x,r=run(root,root/'results/test09p',min_overlap=200)
        check(len(q)==4 and len(c)==8,'bad synthetic study/contrast number')
        check(len(x)==6+16+11+12,'GEO sample title crosswalk fixture mismatch')
        check((c.n_mapped_response_genes==500).all(),'lost fixture genes')
        check((c.n_treatment_samples>=2).all(),'bad treatment sample sizes')
        check((c.n_control_samples>=2).all(),'bad control sample sizes')
        try:
            run(root,root/'results/test09p',min_overlap=200)
        except RuntimeError as ex:
            check('refusing to overwrite' in str(ex),'wrong non-overwrite failure')
        else:
            raise AssertionError('failed to protect existing output')
        # Deliberately tampered sample identity must fail, even with same sample count.
        bad=root/C/'GSE179150_counts.csv'
        dd=pd.read_csv(bad)
        dd=dd.rename(columns={'PA2504_p1_htseq':'UNRECOGNISED_SAMPLE'})
        dd.to_csv(bad,index=False)
        try:
            run(root,root/'results/test09p_bad',min_overlap=200)
        except RuntimeError as ex:
            check('unexpected count/GEO sample headers' in str(ex),'wrong tampering failure')
        else:
            raise AssertionError('sample identity tamper not caught')
        print('SELF_TEST_PASSED: 4 GEO-title-crosswalked studies, 8 contrasts, shuffled columns/design,')
        print('  within-study SF normalization, sample identity tampering rejection, no-overwrite.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        root=args.root.resolve()
        dest=args.out if args.out.is_absolute() else root/args.out
        run(root,dest)

if __name__=='__main__':
    main()
