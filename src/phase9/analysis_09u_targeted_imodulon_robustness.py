#!/usr/bin/env python3
"""CF-State 09u: targeted, exploratory gene-influence and module-annotation checks.

This does NOT discover new biological mechanisms, estimate iModulon activities,
or produce sample-level inference. Assesses 09t correlations with frozen aeruPRECISE411
loadings, their dependence on influential genes/genomic blocks, and threshold-defined
module membership. Reuses frozen 09p and 09r gene responses only.

Run --self-test first. Prints all decisive findings. Refuses output overwrite.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
REF=Path('external/aeruprecise/precise411/data_files')
P=Path('results/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze')
R=Path('results/phase9/audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze')
T=Path('results/phase9/analysis_09t_frozen_imodulon_perturbation_alignment')
OUT=Path('results/phase9/analysis_09u_targeted_imodulon_robustness')

CASES=[
 ('GSE179150__KO_PA2504_vs_WT','64','CysB-2','candidate_external'),
 ('GSE167137__MEROPENEM_MONOCULTURE','19','TagR1','candidate_external'),
 ('GSE167137__MEROPENEM_COCULTURE','19','TagR1','candidate_external'),
 ('GSE138731__FARNESOL_vs_CONTROL','39','AtuR','reference_included'),
 ('GSE192340__DDDY_vs_CONTROL','85','RpoH','candidate_external'),
]


def require(v,message):
    if not bool(v):
        raise RuntimeError('09U_CHECK_FAILED: '+message)


def checksum(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def numeric_csv(path):
    d=pd.read_csv(path,index_col=0,low_memory=False)
    d.index=d.index.astype(str).str.strip()
    d.columns=d.columns.astype(str).str.strip()
    require(d.index.is_unique and d.columns.is_unique, str(path)+' duplicate IDs')
    return d.apply(pd.to_numeric,errors='coerce')


def corr(x,y):
    x=np.asarray(x,dtype=float)
    y=np.asarray(y,dtype=float)
    if len(x)<3 or np.std(x)<1e-13 or np.std(y)<1e-13:
        return float('nan')
    return float(np.corrcoef(x,y)[0,1])


def all_corr(y,W):
    vy=np.asarray(y,float); w=np.asarray(W,float)
    yy=vy-vy.mean(); ww=w-w.mean(axis=0,keepdims=True)
    yn=np.linalg.norm(yy); wn=np.linalg.norm(ww,axis=0)
    with np.errstate(invalid='ignore',divide='ignore'):
        cc=(yy @ ww)/(yn*wn)
    return np.where(np.isfinite(cc),cc,0.0)


def rank_target(y,W,target):
    cc=np.abs(all_corr(y,W))
    idx=W.columns.get_loc(target)
    # 1 = most aligned by |Pearson|, ties get minimum rank
    return int(1+np.sum(cc > cc[idx]+1e-12))


def load_data(root):
    files={
      'M':root/REF/'M.csv',
      'table':root/REF/'iM_table.csv',
      'thresholds':root/REF/'M_thresholds.csv',
      'p_response':root/P/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv',
      'r_response':root/R/'PROVISIONAL_ADDITIONAL_GENE_RESPONSES.csv',
    }
    for label,path in files.items():
        require(path.is_file(),'missing '+label+': '+str(path))
    M=numeric_csv(files['M'])
    M=M.loc[M.index.to_series().str.fullmatch(r'PA\d{4}')]
    M=M.dropna(axis=0,how='any')
    table=pd.read_csv(files['table'],dtype=str,keep_default_na=False)
    require({'k','name'}.issubset(table.columns),'iM_table.csv must provide k,name')
    table['k']=table['k'].astype(str).str.strip()
    table=table.set_index('k')
    require(table.index.is_unique,'duplicate iModulon annotations')
    thresholds=pd.read_csv(files['thresholds'],index_col=0)
    thresholds.index=thresholds.index.astype(str).str.strip()
    require(thresholds.shape[1]>=1,'empty threshold table')
    thresholds=pd.to_numeric(thresholds.iloc[:,0],errors='coerce')
    require(thresholds.notna().all(),'non-numeric iModulon thresholds')
    p=numeric_csv(files['p_response'])
    r=pd.read_csv(files['r_response'],dtype={'GeneID':str,'contrast_id':str})
    require({'GeneID','contrast_id','effect_log2_count_over_median_ratio_plus1'}.issubset(r.columns),'09r long response columns missing')
    require(not r.duplicated(['GeneID','contrast_id']).any(),'duplicate 09r gene responses')
    r['effect_log2_count_over_median_ratio_plus1']=pd.to_numeric(r['effect_log2_count_over_median_ratio_plus1'],errors='raise')
    r=r.pivot(index='GeneID',columns='contrast_id',values='effect_log2_count_over_median_ratio_plus1')
    require(set(M.columns).issubset(set(table.index)),'M components missing annotations')
    require(set(M.columns).issubset(set(thresholds.index)),'M components missing thresholds')
    require(len(M)>=300,'too few PA gene loadings')
    require(M.shape[1]>=4,'too few iModulons')
    require(np.isfinite(M.to_numpy(float)).all(),'invalid M loadings')
    rows=[]
    for cid,k,expected_name,status in CASES:
        require(k in M.columns, 'missing iModulon '+k)
        actual=table.loc[k,'name']
        require(actual==expected_name, 'unexpected annotation for '+k+': '+actual+' vs '+expected_name)
        available=p if cid in p.columns else r
        require(cid in available.columns,'missing frozen response '+cid)
        y=available[cid].dropna()
        common=M.index.intersection(y.index)
        require(len(common)>=250,'too few overlapping genes '+cid)
        yy=y.loc[common].astype(float)
        ww=M.loc[common].astype(float)
        require(np.isfinite(yy.to_numpy()).all(),'nonfinite response '+cid)
        rows.append((cid,k,status,yy,ww))
    return files,M,table,thresholds,rows


def process_case(cid,k,category,y,W,threshold,block_size=25):
    # Use same centred Pearson as frozen 09t, recomputing after each removal.
    m=W[k].to_numpy(float)
    v=y.to_numpy(float)
    genes=W.index.to_numpy(str)
    n=len(genes)
    baseline=corr(v,m)
    spearman=float(pd.Series(v).corr(pd.Series(m),method='spearman'))
    centred_m=m-m.mean(); centred_y=v-v.mean()
    nm=np.linalg.norm(centred_m); ny=np.linalg.norm(centred_y)
    require(nm>1e-12 and ny>1e-12,'degenerate target profiles '+cid)
    contrib=(centred_m/nm)*(centred_y/ny)
    order=np.argsort(-np.abs(contrib))
    pa=np.array([int(g[2:]) for g in genes],dtype=int)
    blocks=pa//int(block_size)
    ranks=[]

    def add(label,kept):
        ii=np.flatnonzero(kept)
        if len(ii)<30:
            return
        co=corr(v[ii],m[ii])
        target_rank=rank_target(v[ii],W.iloc[ii,:],k)
        ranks.append(dict(contrast_id=cid,module_id=k,scenario=label,n_genes=len(ii),
                          pearson=co,top_module_rank_abs_pearson=target_rank,
                          retained_sign=bool(np.isfinite(co) and np.sign(co)==np.sign(baseline))))
    add('all_genes',np.ones(n,dtype=bool))
    for K in (1,5,10,25):
        mask=np.ones(n,dtype=bool); mask[order[:K]]=False
        add('remove_top_'+str(K)+'_covariance_genes',mask)
    block_sum={b:float(np.sum(np.abs(contrib[blocks==b]))) for b in np.unique(blocks)}
    block_order=sorted(block_sum, key=lambda b:(-block_sum[b],int(b)))
    for K in (1,2,3):
        mask=~np.isin(blocks,block_order[:K])
        add('remove_top_'+str(K)+'_25_locus_blocks',mask)
    core=(np.abs(m)>float(threshold))
    add('only_frozen_iModulon_members',core)
    add('exclude_frozen_iModulon_members',~core)
    sr=pd.DataFrame({'GeneID':genes,'module_id':k,'contrast_id':cid,
                     'response':v,'frozen_loading':m,'pearson_covariance_contribution':contrib,
                     'abs_covariance_contribution':np.abs(contrib),
                     'frozen_iModulon_member':core,
                     'genomic_25_locus_block':blocks})
    sr=sr.sort_values('abs_covariance_contribution',ascending=False)
    ab=float(np.abs(contrib).sum())
    metrics=dict(contrast_id=cid,module_id=k,reference_status=category,
                 n_overlap=n,n_frozen_core_genes=int(core.sum()),
                 threshold=float(threshold),pearson=baseline,spearman=spearman,
                 top_1_abs_covariance_fraction=float(np.abs(contrib[order[:1]]).sum()/ab),
                 top_5_abs_covariance_fraction=float(np.abs(contrib[order[:5]]).sum()/ab),
                 top_10_abs_covariance_fraction=float(np.abs(contrib[order[:10]]).sum()/ab),
                 top_block_abs_covariance_fraction=float(block_sum[block_order[0]]/ab),
                 top_block_locus_begin=int(block_order[0])*int(block_size),
                 top_block_locus_end=(int(block_order[0])+1)*int(block_size)-1)
    blocksout=pd.DataFrame([{'contrast_id':cid,'module_id':k,'genomic_25_locus_block':int(b),
                             'PA_tag_start':f'PA{int(b)*block_size:04d}',
                             'PA_tag_end':f'PA{(int(b)+1)*block_size-1:04d}',
                             'block_n_genes':int((blocks==b).sum()),
                             'sum_abs_covariance_contribution':block_sum[b]} for b in block_order[:10]])
    return metrics,pd.DataFrame(ranks),sr,blocksout


def run(root,out,strict=False):
    require(not out.exists(),'refusing to overwrite '+str(out))
    files,M,table,thresholds,rows=load_data(root)
    summaries=[]; tests=[]; topgenes=[]; blocks=[]
    for cid,k,status,y,W in rows:
        met,ranks,genes,bl=process_case(cid,k,status,y,W,float(thresholds.loc[k]))
        met.update({'module_name':table.loc[k,'name'],
                    'function':table.loc[k,'function'] if 'function' in table.columns else '',
                    'category':table.loc[k,'category'] if 'category' in table.columns else ''})
        summaries.append(met);tests.append(ranks);topgenes.append(genes.head(30));blocks.append(bl)
    summary=pd.DataFrame(summaries)
    checks=pd.concat(tests,ignore_index=True)
    top=pd.concat(topgenes,ignore_index=True)
    blockdf=pd.concat(blocks,ignore_index=True)
    # Alignments must be numerically consistent with the earlier immutable 09t outputs.
    previous=root/T/'ALL_FROZEN_IMODULON_LOADINGS_RESPONSE_ALIGNMENTS.csv'
    if previous.is_file():
        hist=pd.read_csv(previous,dtype={'imodulon':str,'contrast_id':str})
        for row in summary.itertuples(index=False):
            match=hist[(hist.contrast_id==row.contrast_id)&(hist.imodulon==row.module_id)]
            require(len(match)==1,'09t does not contain exact contrast/module '+row.contrast_id+'/'+row.module_id)
            require(abs(float(match.iloc[0].pearson_alignment)-row.pearson)<5e-7,
                    '09u/09t original Pearson mismatch: '+row.contrast_id)
    elif strict:
        raise RuntimeError('09U_CHECK_FAILED: 09t expected previous alignment CSV missing')

    print('\n'+'='*100)
    print('CF-STATE 09U | TARGETED FROZEN iMODULON INFLUENCE CHECKS (DESCRIPTIVE)')
    print('='*100)
    print('Published frozen module annotations; 09p+09r responses. NO new decomposition / TF activity estimates.')
    print('Modules tested: TagR1/H1-T6SS (19), AtuR/terpenes (39), CysB-2/sulfur (64), RpoH/heat shock (85).')
    print('Reference overlap: GSE138731 included; other studies only candidate external (GSE/GSM IDs).')
    print('Reference module membership: |loading| > frozen threshold. PA tag blocks of',25,'genes are NOT verified operons.')
    cols=['contrast_id','module_id','module_name','n_overlap','n_frozen_core_genes','pearson','spearman',
          'top_5_abs_covariance_fraction','top_block_abs_covariance_fraction']
    print('\nPROGRAMME ALIGNMENT AND INFLUENCE:')
    print(summary[cols].round(4).to_string(index=False))
    pivot=checks.pivot(index='contrast_id',columns='scenario',values='pearson')
    ordered=['all_genes','remove_top_5_covariance_genes','remove_top_10_covariance_genes',
             'remove_top_1_25_locus_blocks','remove_top_2_25_locus_blocks',
             'only_frozen_iModulon_members','exclude_frozen_iModulon_members']
    existing=[c for c in ordered if c in pivot.columns]
    print('\nPEARSON AFTER REMOVING HIGH-LEVERAGE GENES / LOCUS BLOCKS:')
    print(pivot[existing].round(4).to_string())
    for _,one in summary.iterrows():
        cid=one.contrast_id;k=one.module_id
        print('\nMOST INFLUENTIAL GENES:',cid,'module',k,one.module_name)
        print(top[top.contrast_id==cid][['GeneID','response','frozen_loading',
                  'pearson_covariance_contribution','frozen_iModulon_member']].head(10).round(5).to_string(index=False))
        print('  Highest-contribution genomic 25-locus block:',
              f"PA{one.top_block_locus_begin:04d}..PA{one.top_block_locus_end:04d}")
    print('\nINTERPRETATION: 09t correlation alone is not a calibrated activity score or a new biology claim.')
    print('Removing genes/blocks is a sensitivity check chosen using the same data, not held-out statistical inference.')
    print('GSE167137 TagR1/meropenem and GSE138731 AtuR/terpene signatures have established literature context.')
    print('GSE138731 is PRECISE411-INCLUDED. Other studies require raw-run provenance for strict external status.')

    out.mkdir(parents=True,exist_ok=False)
    summary.to_csv(out/'TARGET_MODULE_ROBUSTNESS_SUMMARY.csv',index=False)
    checks.to_csv(out/'LEAVE_GENE_AND_BLOCK_OUT.csv',index=False)
    top.to_csv(out/'TOP_30_GENE_CONTRIBUTORS_PER_CONTRAST.csv',index=False)
    blockdf.to_csv(out/'TOP_10_GENOMIC_BLOCK_CONTRIBUTIONS.csv',index=False)
    paths=list(files.values())+([previous] if previous.is_file() else [])
    pd.DataFrame([dict(path=str(p.relative_to(root)),sha256=checksum(p)) for p in paths]).to_csv(
        out/'SOURCE_SHA256.csv',index=False)
    (out/'QC_STATUS.json').write_text(json.dumps({
        'status':'09U_TARGETED_PROGRAMME_ROBUSTNESS_COMPLETE',
        'n_contrasts':len(rows),'no_refitting':True,'clinical_association_claim':False,
        'activity_inference':False,'statistical_p_values':False,
        'reference_included':['GSE138731'],'positional_blocks_are_operons':False},indent=2)+'\n')
    print('\nSUCCESS — saved without overwrite:',out)
    return summary,checks,top


def self_test():
    rng=np.random.default_rng(901701)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        for path in (root/REF,root/P,root/R):path.mkdir(parents=True,exist_ok=True)
        genes=[f'PA{x:04d}' for x in range(600)]
        mods=[str(x) for x in range(116)]
        M=rng.normal(size=(600,116))*0.005
        kmap={k:mods.index(k) for k in ('19','39','64','85')}
        for k in kmap:
            M[:,kmap[k]]=rng.normal(size=600)*0.02
        pd.DataFrame(M,index=genes,columns=mods).to_csv(root/REF/'M.csv')
        pd.DataFrame({'k':mods,'name':[{'19':'TagR1','39':'AtuR','64':'CysB-2','85':'RpoH'}.get(k,'other') for k in mods],
                      'function':['test']*116,'category':['test']*116}).to_csv(root/REF/'iM_table.csv',index=False)
        pd.DataFrame({'0':[0.012]*116},index=mods).to_csv(root/REF/'M_thresholds.csv')
        noise=lambda:rng.normal(size=600)*0.16
        p=pd.DataFrame(index=genes)
        p['GSE179150__KO_PA2504_vs_WT']=-2*M[:,kmap['64']]+noise()
        p['GSE167137__MEROPENEM_MONOCULTURE']=2*M[:,kmap['19']]+noise()
        p['GSE167137__MEROPENEM_COCULTURE']=2.5*M[:,kmap['19']]+noise()
        p.to_csv(root/P/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv')
        r=pd.DataFrame({'GeneID':genes*2,
                        'contrast_id':['GSE138731__FARNESOL_vs_CONTROL']*600+['GSE192340__DDDY_vs_CONTROL']*600,
                        'effect_log2_count_over_median_ratio_plus1':
                        np.r_[4*M[:,kmap['39']]+noise(),noise()]})
        r.to_csv(root/R/'PROVISIONAL_ADDITIONAL_GENE_RESPONSES.csv',index=False)
        out=root/'results'/'test_09u'
        summary,checks,top=run(root,out)
        require(len(summary)==5 and len(top)==150,'missing rows in synthetic results')
        require(int(checks.loc[(checks.contrast_id=='GSE167137__MEROPENEM_MONOCULTURE')&
                  (checks.scenario=='all_genes'),'top_module_rank_abs_pearson'].iloc[0])==1,
                'spiked TagR1 should rank first')
        try:run(root,out)
        except RuntimeError as e:
            require('overwrite' in str(e),'unexpected rerun failure')
        else:raise RuntimeError('nonoverwrite check failed')
        # Tampered identity test
        bad=pd.read_csv(root/REF/'iM_table.csv')
        bad.loc[bad.k==19,'name']='wrong'
        bad.to_csv(root/REF/'iM_table.csv',index=False)
        try:run(root,root/'results'/'test_09u_bad')
        except RuntimeError as e:
            require('unexpected annotation' in str(e),'unexpected identity failure')
        else:raise RuntimeError('tampered annotation was accepted')
        print('SELF_TEST_PASSED 09U: full 5-contrast run, frozen membership, gene/block influence,')
        print('  target rank, source SHA, overwrite rejection and annotation tampering rejection.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,default=None)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        root=args.root.resolve()
        out=args.out.resolve() if args.out is not None else root/OUT
        run(root,out,strict=True)


if __name__=='__main__':
    main()
