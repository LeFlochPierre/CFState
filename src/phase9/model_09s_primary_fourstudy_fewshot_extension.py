#!/usr/bin/env python3
"""CF-State 09s: extend the frozen 09q few-shot benchmark with 09r studies.

Primary only: GSE179150 (PA2504 KO), GSE167137 (meropenem 2 contexts),
GSE138731 (farnesol), GSE192340 (DDDY): 4 independent GEO studies,
5 contexts. Leave all contexts from one study out (3 training studies).
Secondary-augmented sensitivity: add 09p GSE190194 (long adaptation),
GSE223663 (historical overlap) only to TRAINING while always testing on the
four primary studies. The 09q architecture/hyperparameters/anchor blocks are
unchanged; the purpose is to measure consequences of broader training data.

Requires adjacent src/phase9/model_09q_multistudy_fewshot_regulon_completion.py.
No model or hyperparameter tuning on held-out outcomes. Retrospective only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import model_09q_multistudy_fewshot_regulon_completion as q

ROOT = Path(__file__).resolve().parents[2]
P = Path('results/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze')
R = Path('results/phase9/audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze')
OUT = Path('results/phase9/model_09s_primary_fourstudy_fewshot_extension')
PRIMARY = ('GSE167137', 'GSE179150', 'GSE138731', 'GSE192340')
SECONDARY = ('GSE190194', 'GSE223663')
N_REWIRES = 20
N_SPLITS = 5
BUDGETS = (0.10, 0.25)


def need(ok, msg):
    if not bool(ok):
        raise RuntimeError('MODEL_09S_FAILED: ' + str(msg))


def hash_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_inputs(root, strict=True):
    paths = [
        root/P/'QC_STATUS.json',
        root/P/'CONTRAST_SOURCE_DESIGN_AND_QC.csv',
        root/P/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv',
        root/R/'QC_STATUS.json',
        root/R/'QC_STUDY_COUNT_SCALE_AND_COMPLETENESS.csv',
        root/R/'PROVISIONAL_ADDITIONAL_CONTRASTS.csv',
        root/R/'PROVISIONAL_ADDITIONAL_GENE_RESPONSES.csv',
        root/q.G_PRIOR,
        root/q.G_QC,
    ]
    for p in paths:
        need(p.is_file(), 'missing required frozen input '+str(p))
    p_qc = json.loads(paths[0].read_text())
    r_qc = json.loads(paths[3].read_text())
    need(p_qc.get('status') == '09P_PROVISIONAL_MULTISTUDY_RESPONSE_CANDIDATES_FROZEN', '09p status incorrect')
    need(r_qc.get('status') == '09R_THREE_STUDY_ADDITIONAL_CONTRAST_QC_COMPLETE', '09r status incorrect')
    need(p_qc.get('raw_run_to_count_link_proven') is False, 'unexpected 09p provenance flag')
    need(r_qc.get('count_to_SRR_proven') is False, 'unexpected 09r provenance flag')
    p_meta = pd.read_csv(paths[1], keep_default_na=False)
    p_wide = pd.read_csv(paths[2], dtype={'GeneID':str}).set_index('GeneID')
    r_study_qc = pd.read_csv(paths[4], keep_default_na=False)
    r_meta = pd.read_csv(paths[5], keep_default_na=False)
    r_long = pd.read_csv(paths[6], dtype={'GeneID':str}, keep_default_na=False)
    prior = pd.read_csv(paths[7], dtype=str, keep_default_na=False)
    need(p_wide.index.is_unique and p_wide.columns.is_unique, '09p duplicate response IDs')
    need(p_meta.contrast_id.is_unique and r_meta.contrast_id.is_unique, 'duplicate contrast metadata')
    need(set(p_wide.columns)==set(p_meta.contrast_id), '09p wide/meta contrast mismatch')
    need(not r_long.duplicated(['GeneID','contrast_id']).any(), '09r duplicated gene/contrast')
    need(set(r_long.contrast_id)==set(r_meta.contrast_id), '09r long/meta contrast mismatch')
    need(set(r_meta.study)=={'GSE138731','GSE192340'}, '09r eligible studies unexpected; DATS must stay quarantined')
    status = r_study_qc.set_index('study')['status'].to_dict()
    need('QUARANTINE' in status.get('GSE178622',''), 'DATS quarantine absent')
    for s in ('GSE138731','GSE192340'):
        need(status.get(s)=='COUNT_SCALE_ELIGIBLE_PROVISIONAL',s+' not provisionally eligible')
    need(set(p_meta.study)==set(('GSE179150','GSE167137')+SECONDARY), '09p study set unexpected')
    need(p_meta[p_meta.study.isin(PRIMARY)].primary_modelling_status.eq('PROVISIONAL_PRIMARY_GEO_TITLE_CORROBORATED').all(), '09p primary tag mismatch')
    need(p_meta[p_meta.study.isin(SECONDARY)].primary_modelling_status.str.startswith('SECONDARY').all(), '09p secondary tag mismatch')
    need(set(p_meta.contrast_id).isdisjoint(set(r_meta.contrast_id)), '09p/09r contrast ID collision')
    if strict:
        need(p_wide.shape == (3314,8), '09p real-data dimensions changed')
        need(len(r_meta)==2 and len(r_long)>9000, '09r real-data size changed')
        need(int(json.loads(paths[8].read_text())['mapped_09e4'])==len(prior)==806, '09e4 source changed')
    r_wide = r_long.pivot(index='GeneID',columns='contrast_id',
       values='effect_log2_count_over_median_ratio_plus1')
    common = sorted(set(p_wide.index).intersection(*(set(r_wide[c].dropna().index) for c in r_meta.contrast_id)),
                   key=lambda a:int(a[2:]))
    need(len(common)>400, 'too few genes shared by provisional inputs')
    need(all(str(g).startswith('PA') and len(g)==6 and str(g)[2:].isdigit() for g in common), 'nonbare PA gene')
    if strict:
        need(3300<=len(common)<=3314, 'unexpected real 09p/09r gene intersection')
    p_meta = p_meta.copy()
    r_meta = r_meta.copy()
    r_meta['intervention_category'] = r_meta['treatment'].astype(str).str.upper()+'_TREATMENT'
    r_meta['primary_modelling_status'] = 'PROVISIONAL_PRIMARY_GEO_TITLE_CORROBORATED'
    r_meta['historical_CFSTATE_overlap'] = False
    meta = pd.concat([p_meta,r_meta],ignore_index=True,sort=False)
    meta = meta.sort_values(['study','contrast_id']).reset_index(drop=True)
    need(meta.contrast_id.is_unique, 'merged duplicated contrast')
    vals = pd.concat([p_wide.loc[common],r_wide.loc[common]],axis=1)
    vals = vals.loc[common,meta.contrast_id.tolist()].astype(float)
    need(np.isfinite(vals.to_numpy()).all(), 'nonfinite merged contrast matrix')
    need(len(set(meta.study))==6 and len(meta)==10, 'expected 6 studies / 10 contrasts (including secondary)')
    primary_meta = meta[meta.study.isin(PRIMARY)].reset_index(drop=True)
    need(len(primary_meta)==5 and primary_meta.study.nunique()==4, 'expected four PRIMARY studies and five contexts')
    primary_wide = vals[primary_meta.contrast_id]
    return vals, meta, primary_wide, primary_meta, prior, paths, status


def score_primary_holdouts(wide, meta, prior, *, training_mode='PRIMARY_ONLY', n_rewires=N_REWIRES, n_splits=N_SPLITS, budgets=BUDGETS, strict=True):
    """Build fixed test-study-only 09q predictions, no response leakage from test genes."""
    genes=wide.index.to_numpy(dtype=str)
    Y=wide.to_numpy(float)
    G, regs, covered, conflicts=q.make_prior(genes,prior,strict=strict)
    nulls=list(q.null_graphs(G,n=n_rewires,seed=q.SEED))
    rows=[]
    for held in PRIMARY:
        testidx=np.flatnonzero(meta.study.to_numpy()==held)
        trainidx=np.flatnonzero((meta.study.to_numpy()!=held)&(
            (meta.study.isin(PRIMARY).to_numpy()) if training_mode=='PRIMARY_ONLY' else np.ones(len(meta),dtype=bool)))
        need(len(testidx)>0 and len(trainidx)>=3, 'invalid training/heldout split: '+held)
        need(len(set(meta.iloc[trainidx].study))>=3, 'not enough independent source studies')
        designs={}
        for method,mode in [('NO_GRAPH','none'),('CURATED_SIGNED','signed'),('CURATED_UNSIGNED','unsigned')]:
            designs[method]=q.train_design(Y,meta,G,trainidx,mode)[:2]
        rwdesigns=[q.train_design(Y,meta,H,trainidx,'signed')[:2] for H in nulls]
        for idx in testidx:
            y=Y[:,idx]
            for frac in budgets:
                for split in range(n_splits):
                    known=q.block_anchors(genes,frac,split)
                    basepred=np.full(len(y),float(np.mean(y[known])))
                    pred={k:q.ridge_predict(des,y,known,n) for k,(des,n) in designs.items()}
                    pred['ANCHOR_MEAN']=basepred
                    rew_preds=[q.ridge_predict(des,y,known,n) for des,n in rwdesigns]
                    for panel,masked in [('ALL_COMMON',np.ones(len(genes),dtype=bool)),('CURATED_TARGETS',covered)]:
                        score_anchor=known | (~masked)
                        need((~score_anchor).sum()>20, 'too few test genes')
                        rw_maes=np.array([q.metrics(p,y,score_anchor)['mae'] for p in rew_preds])
                        for method,v in pred.items():
                            m=q.metrics(v,y,score_anchor)
                            rows.append({'regime':training_mode,'heldout_study':held,'contrast_id':meta.iloc[idx].contrast_id,
                                'panel':panel,'budget':frac,'split':split,'method':method,
                                'n_train_studies':int(meta.iloc[trainidx].study.nunique()),
                                'n_train_contrasts':int(len(trainidx)),
                                'n_anchor_genes':int(known.sum()),'n_anchor_regulator_genes':int((known&covered).sum()),
                                'rewire_mean_mae':float(rw_maes.mean()),
                                'curated_beats_single_rewires':int(np.sum(rw_maes>m['mae'])) if method=='CURATED_SIGNED' else np.nan,
                                **m})
                        rwms=[q.metrics(p,y,score_anchor) for p in rew_preds]
                        rows.append({'regime':training_mode,'heldout_study':held,'contrast_id':meta.iloc[idx].contrast_id,
                            'panel':panel,'budget':frac,'split':split,'method':'DEGREE_MATCHED_REWIRE_MEAN',
                            'n_train_studies':int(meta.iloc[trainidx].study.nunique()),
                            'n_train_contrasts':int(len(trainidx)),
                            'n_anchor_genes':int(known.sum()),'n_anchor_regulator_genes':int((known&covered).sum()),
                            'rewire_mean_mae':float(rw_maes.mean()),'curated_beats_single_rewires':np.nan,
                            **{k:float(np.mean([z[k] for z in rwms])) if k!='n_test_genes' else rwms[0][k] for k in rwms[0]}})
    return pd.DataFrame(rows),regs,covered,conflicts


def run(root,out,strict=True,n_rewires=N_REWIRES,n_splits=N_SPLITS,budgets=BUDGETS):
    need(not out.exists(), 'refusing to overwrite '+str(out))
    all_wide,all_meta,primary_wide,primary_meta,prior,paths,status=load_inputs(root,strict=strict)
    rows=[]; selected_regs=[]; coverage=[]
    for mode in ('PRIMARY_ONLY','SECONDARY_AUGMENTED_TRAINING'):
        wide = primary_wide if mode=='PRIMARY_ONLY' else all_wide
        meta = primary_meta if mode=='PRIMARY_ONLY' else all_meta
        z,regs,covered,conflicts=score_primary_holdouts(wide,meta,prior,training_mode=mode,
          n_rewires=n_rewires,n_splits=n_splits,budgets=budgets,strict=strict)
        rows.append(z)
        selected_regs=regs
        coverage.append(int(covered.sum()))
    results=pd.concat(rows,ignore_index=True)
    summaries=(results.groupby(['regime','heldout_study','panel','budget','method'],as_index=False)
        .agg(n_contrasts=('contrast_id','nunique'),n_rows=('mae','size'),
            mae=('mae','mean'),sign_accuracy=('sign_accuracy','mean'),
            balanced_accuracy=('balanced_sign_accuracy','mean'),pearson=('pearson','mean'),
            n_test_genes=('n_test_genes','mean')))
    print('\n'+'='*98)
    print('CF-STATE MODEL 09S | FOUR PRIMARY STUDIES: FIVE-SHOT CONTEXTS, 09Q ARCHITECTURE FROZEN')
    print('='*98)
    print('09p + 09r provisional primary studies:', len(PRIMARY), '| primary contexts:',len(primary_meta))
    print('Primary studies: '+', '.join(PRIMARY))
    print('09p secondary studies used only for sensitivity TRAINING, never as held-out primary performance claims.')
    print('Common nonmissing genes:',len(primary_wide), '| regulators:',len(selected_regs),
          '| graph-covered genes:',coverage[0])
    print('DATS GSE178622 remains QUARANTINED; no inferred DATS intervention response.')
    print('Frozen 09q methods: source-trained PCA, signed/unsigned curated regulators,',n_rewires,'degree-preserved rewires')
    print('Anchor proportions:',budgets,'| locus-block repeats:',n_splits)
    print('All models observe ONLY anchor gene responses from held-out contrasts; NOT unseen-intervention forecasting.')
    print('MAE is within the held-out study effect scale; never pool effect magnitudes as equivalent DE logFC.')
    print('\nEXACT-ZERO EFFECTS (sign metrics exclude zeros; MAE includes zeros):')
    for cid in primary_meta.contrast_id:
        y=primary_wide[cid].to_numpy(float)
        print(' ',cid,'zeros=',int(np.sum(y==0)), '/',len(y), 'median_abs_effect=',f'{np.median(np.abs(y)):.5f}')
    for mode in ('PRIMARY_ONLY','SECONDARY_AUGMENTED_TRAINING'):
        print('\n'+'-'*82)
        print('TRAINING MODE:',mode)
        for panel in ('ALL_COMMON','CURATED_TARGETS'):
            z=summaries[(summaries.regime==mode)&(summaries.panel==panel)&(summaries.budget==max(budgets))]
            print('\nHELD-OUT PRIMARY STUDIES |',panel,'|',f'{max(budgets):.0%}','anchors')
            print(z[['heldout_study','n_contrasts','method','mae','sign_accuracy','balanced_accuracy','pearson']].to_string(index=False,float_format=lambda a:f'{a:.4f}'))
            wide=z.pivot(index='heldout_study',columns='method',values='mae')
            print('\nREGULATORY MAE minus baselines (negative means network improves):')
            for name,row in wide.iterrows():
                d0=row['CURATED_SIGNED']-row['NO_GRAPH']
                dn=row['CURATED_SIGNED']-row['DEGREE_MATCHED_REWIRE_MEAN']
                print(' ',name,'vs no-graph:',f'{d0:+.5f}',' vs rewires:',f'{dn:+.5f}')
    print('\nCAUTIONS: retrospective development, 4 independent provisional primary GEO studies;')
    print('processed counts are GEO-title corroborated but not original SRR-to-count-column verified.')
    print('Zero-response gene signs not evaluated. Target-study outcome use restricted to anchor genes.')
    print('Curated regulatory signs not proof of direct binding or perturbation direction.')
    print('Secondary studies include long passage adaptation and known historical dataset overlap.')
    out.mkdir(parents=True,exist_ok=False)
    results.to_csv(out/'ALL_HELDOUT_PRIMARY_STUDY_SPLIT_METRICS.csv',index=False)
    summaries.to_csv(out/'SUMMARY_HELDOUT_PRIMARY_STUDY_METRICS.csv',index=False)
    all_meta.to_csv(out/'FULL_PROVISIONAL_CONTEXT_LEDGER.csv',index=False)
    primary_wide.rename_axis('GeneID').to_csv(out/'FROZEN_PRIMARY_COMMON_GENE_RESPONSE_MATRIX.csv')
    pd.DataFrame({'regulator_GeneID':selected_regs}).to_csv(out/'CURATED_REGULATORS.csv',index=False)
    pd.DataFrame({'source_file':[str(p) for p in paths], 'sha256':[hash_file(p) for p in paths]}).to_csv(out/'INPUT_SHA256_PROVENANCE.csv',index=False)
    qc={'status':'09S_PRIMARY_FOUR_STUDY_FROZEN_ARCHITECTURE_RETROSPECTIVE_BENCHMARK',
        'primary_studies':list(PRIMARY),'n_primary_contexts':len(primary_meta),
        'n_primary_common_genes':len(primary_wide),'n_curated_covered_genes':coverage[0],
        'n_degree_preserving_nulls':n_rewires,'budgets':list(budgets),'n_splits':n_splits,
        'DATS_quarantined':True,'trained_model_zero_shot':False,
        'SRR_processing_provenance_verified':False,'primary_evaluation_studies_retrospectively_analysed':True}
    (out/'QC_STATUS.json').write_text(json.dumps(qc,indent=2)+'\n')
    print('\nSUCCESS: '+str(out))
    return results,summaries


def self_test():
    rng=np.random.default_rng(26)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        for d in (root/P,root/R,(root/q.G_PRIOR).parent): d.mkdir(parents=True,exist_ok=True)
        genes=[f'PA{i:04d}' for i in range(1,601)]
        pids=['GSE167137__A','GSE167137__B','GSE179150__KO','GSE190194__A','GSE190194__B','GSE190194__C','GSE223663__A','GSE223663__B']
        pst=[x.split('__')[0] for x in pids]
        pm=pd.DataFrame({'contrast_id':pids,'study':pst,
          'intervention_category':['ACUTE']*2+['KO']+['LONG_ADAPT']*3+['KO']*2,
          'primary_modelling_status':['PROVISIONAL_PRIMARY_GEO_TITLE_CORROBORATED']*3+['SECONDARY_LONG_TERM_ADAPTATION']*3+['SECONDARY_INTERNAL_OVERLAP']*2,
          'historical_CFSTATE_overlap':[False]*6+[True]*2})
        pm.to_csv(root/P/'CONTRAST_SOURCE_DESIGN_AND_QC.csv',index=False)
        pY=rng.normal(0,1,(600,8))
        pd.DataFrame(pY,index=genes,columns=pids).rename_axis('GeneID').to_csv(root/P/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv')
        (root/P/'QC_STATUS.json').write_text(json.dumps({'status':'09P_PROVISIONAL_MULTISTUDY_RESPONSE_CANDIDATES_FROZEN','raw_run_to_count_link_proven':False}))
        rms=['GSE138731','GSE192340']
        rm=pd.DataFrame({'study':rms,'contrast_id':[x+'__RESP' for x in rms], 'treatment':['Farnesol','DDDY'], 'control':['Control']*2})
        rm.to_csv(root/R/'PROVISIONAL_ADDITIONAL_CONTRASTS.csv',index=False)
        rg=[{'study':study,'contrast_id':study+'__RESP','GeneID':gene,'effect_log2_count_over_median_ratio_plus1':float(rng.normal())}
            for study in rms for gene in genes]
        pd.DataFrame(rg).to_csv(root/R/'PROVISIONAL_ADDITIONAL_GENE_RESPONSES.csv',index=False)
        pd.DataFrame({'study':rms+['GSE178622'],'status':['COUNT_SCALE_ELIGIBLE_PROVISIONAL']*2+['QUARANTINE_FRACTIONAL_COUNT_MATRIX_UNITS_UNVERIFIED']}).to_csv(root/R/'QC_STUDY_COUNT_SCALE_AND_COMPLETENESS.csv',index=False)
        (root/R/'QC_STATUS.json').write_text(json.dumps({'status':'09R_THREE_STUDY_ADDITIONAL_CONTRAST_QC_COMPLETE','count_to_SRR_proven':False}))
        prior=[]
        for reg,lo,hi in [('PA0701',30,90),('PA0702',65,120),('PA0703',105,170)]:
            for j in range(lo,hi):
                prior.append({'regulator_GeneID':reg,'target_GeneID':genes[j],
                              'original_strain_includes_PAO1':'True','raw_mode_of_regulation':'+' if j%2 else '-'})
        pd.DataFrame(prior).to_csv(root/q.G_PRIOR,index=False)
        (root/q.G_QC).write_text(json.dumps({'mapped_09e4':len(prior)}))
        _,_,pw,pm,_,_,_=load_inputs(root,strict=False)
        need(len(pw)==600 and len(pm)==5 and pm.study.nunique()==4,'wrong primary fixture')
        result,summary=run(root,root/'results/09s_synthetic',strict=False,n_rewires=3,n_splits=2,budgets=(0.10,0.25))
        need(len(result)>0 and summary.heldout_study.nunique()==4,'missing synthetic output')
        need(set(summary.regime)=={'PRIMARY_ONLY','SECONDARY_AUGMENTED_TRAINING'},'missing sensitivity')
        try:
            run(root,root/'results/09s_synthetic',strict=False,n_rewires=3,n_splits=2,budgets=(0.10,0.25))
        except RuntimeError as e: need('refusing to overwrite' in str(e),'incorrect overwrite fail')
        else: raise AssertionError('overwrite permitted')
        # Unscored genes of heldout contrast cannot enter source-trained features or anchor fit.
        wide,meta,_,_,prior,_,_=load_inputs(root,strict=False)
        train=meta.study.ne('GSE179150').to_numpy()
        train=train & meta.study.isin(PRIMARY).to_numpy()
        G,regs,cov,nconf=q.make_prior(wide.index.to_numpy(dtype=str),prior,strict=False)
        design,nreg,_=q.train_design(wide.to_numpy(float),meta,G,np.flatnonzero(train),'signed')
        a=q.block_anchors(wide.index.to_numpy(dtype=str),0.25,0)
        target=wide['GSE179150__KO'].to_numpy(float)
        pred1=q.ridge_predict(design,target,a,nreg)
        edited=target.copy(); edited[~a]+=500
        pred2=q.ridge_predict(design,edited,a,nreg)
        need(np.allclose(pred1,pred2,atol=1e-10),'test gene response leaked into prediction')
        print('SELF_TEST_PASSED 09S: 09p+09r merge; DATS quarantine; 4 study holdouts;')
        print('  primary-only versus secondary-augmented training; intact 09q architecture;')
        print('  non-overwrite; source hashes; zero heldout test-gene leakage.')


def main():
    a=argparse.ArgumentParser()
    a.add_argument('--root',type=Path,default=ROOT)
    a.add_argument('--out',type=Path,default=OUT)
    a.add_argument('--self-test',action='store_true')
    v=a.parse_args()
    if v.self_test: self_test()
    else:
        rt=v.root.resolve()
        out=v.out if v.out.is_absolute() else rt/v.out
        run(rt,out)

if __name__=='__main__': main()
