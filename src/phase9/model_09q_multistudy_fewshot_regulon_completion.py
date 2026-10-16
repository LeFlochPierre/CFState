#!/usr/bin/env python3
"""CF-State model 09q: multi-study regulator-informed few-shot contrast completion.

PROTOCOL (retrospective development, NOT zero-shot intervention forecasting):
  * Leave all contrasts of one GEO STUDY out when learning response templates.
  * Learn a low-rank response basis from OTHER studies' normalized contrasts.
  * Learn modest regulator reliability weights from OTHER studies' responses.
  * For each held-out contrast, observe fixed, locus-block-selected anchor genes;
    infer coefficients from ONLY those response values; score disjoint test genes.
  * Baselines: anchor mean, training-only low-rank basis, signed/unsigned
    curated network plus that basis, 20 target-degree-preserving rewired priors.
  * Equal weighting across studies in training. All eight provisional contrasts
    remain flagged; main interpretation is two primary studies only.

Not a completely unseen-intervention predictor: anchor gene responses from test
study are required. No absolute effect sizes are compared across studies.
All inputs frozen. Outputs are descriptive, never formal DE or causal effects.

Run:
  python -m py_compile src/phase9/model_09q_multistudy_fewshot_regulon_completion.py
  python src/phase9/model_09q_multistudy_fewshot_regulon_completion.py --self-test
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u src/phase9/model_09q_multistudy_fewshot_regulon_completion.py \
    2>&1 | tee logs/model_09q_multistudy_fewshot_regulon_completion.log
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
P09 = Path('results/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze')
G_PRIOR = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/FROZEN_CURATED_REGULATORY_ASSOCIATIONS_09E4.csv')
G_QC = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/QC_09E4.json')
OUT = Path('results/phase9/model_09q_multistudy_fewshot_regulon_completion')
PRIMARY = {'GSE179150','GSE167137'}
SEED = 20261016
N_REWIRES = 20
N_SPLITS = 5
BUDGETS = (0.10, 0.25)
N_PCA = 2
N_REGULATORS = 12
RIDGE_BASE = 0.12
RIDGE_REG = 0.40
RIDGE_REG_FIT = 0.30
LOCUS_BLOCK_WIDTH = 25


def need(ok, msg):
    if not bool(ok):
        raise RuntimeError('MODEL_09Q_FAILED: ' + str(msg))


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def parse_data(root, strict=True):
    paths = [root/P09/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv',
             root/P09/'CONTRAST_SOURCE_DESIGN_AND_QC.csv',
             root/P09/'QC_STATUS.json',root/G_PRIOR,root/G_QC]
    for p in paths:
        need(p.is_file(),'required frozen input absent: '+str(p))
    qc=json.loads(paths[2].read_text())
    need(qc.get('status')=='09P_PROVISIONAL_MULTISTUDY_RESPONSE_CANDIDATES_FROZEN', '09p QC status inconsistent')
    wide=pd.read_csv(paths[0],dtype={'GeneID':str}).set_index('GeneID')
    meta=pd.read_csv(paths[1])
    prior=pd.read_csv(paths[3],dtype=str,keep_default_na=False)
    need(wide.index.is_unique and not wide.columns.duplicated().any(), 'duplicate genes/contrasts')
    need(meta.contrast_id.is_unique, 'duplicate contrast ID in metadata')
    need(set(meta.contrast_id)==set(wide.columns), 'contrast metadata mismatch')
    need({'study','contrast_id','primary_modelling_status','historical_CFSTATE_overlap','intervention_category'}.issubset(meta.columns), '09p metadata schema mismatch')
    need({'regulator_GeneID','target_GeneID','raw_mode_of_regulation','original_strain_includes_PAO1'}.issubset(prior.columns), 'curated prior schema mismatch')
    studies=meta.study.unique().tolist()
    if strict:
        need(wide.shape==(3314,8), 'expected exact real 09p 3314 genes, 8 contrasts')
        need(len(studies)==4,'expected 4 distinct GEO studies')
        need(qc.get('raw_run_to_count_link_proven') is False, 'unexpected raw-run provenance change')
        p_qc=json.loads(paths[4].read_text())
        need(int(p_qc['mapped_09e4'])==len(prior)==806, 'curated prior integrity mismatch')
    wide=wide.loc[:,meta.contrast_id.to_list()]
    Y=wide.to_numpy(dtype=float)
    need(np.isfinite(Y).all(), 'nonfinite response values')
    need((np.abs(Y)>1e-9).sum()>100, 'almost no response signal')
    return wide,meta,prior,paths


def make_prior(genes,prior,max_regs=N_REGULATORS,strict=True):
    z=prior[(prior.original_strain_includes_PAO1.str.lower()=='true') &
            prior.raw_mode_of_regulation.isin(['+','-']) &
            prior.regulator_GeneID.str.fullmatch(r'PA\d{4}') &
            prior.target_GeneID.str.fullmatch(r'PA\d{4}') &
            prior.target_GeneID.isin(genes)].copy()
    z['sign']=z.raw_mode_of_regulation.map({'+':1.,'-':-1.})
    contradictory=z.groupby(['regulator_GeneID','target_GeneID']).sign.nunique()
    conflict=set(contradictory.index[contradictory>1])
    if conflict:
        z=z[~z[['regulator_GeneID','target_GeneID']].apply(tuple,axis=1).isin(conflict)]
    z=z.drop_duplicates(['regulator_GeneID','target_GeneID'])
    degrees=z.groupby('regulator_GeneID').target_GeneID.nunique()
    regs=sorted([x for x in degrees.index if degrees[x]>=5],key=lambda x:(-int(degrees[x]),x))[:max_regs]
    need(len(regs)>=2, 'fewer than two curated multi-target regulators')
    z=z[z.regulator_GeneID.isin(regs)].copy()
    G=np.zeros((len(genes),len(regs)),dtype=float)
    gi={g:i for i,g in enumerate(genes)}
    ri={g:i for i,g in enumerate(regs)}
    for r in z.itertuples():
        G[gi[r.target_GeneID],ri[r.regulator_GeneID]]=float(r.sign)
    degree=(G!=0).sum(axis=1)
    covered=degree>0
    if strict:
        need(covered.sum()>80, 'too few curated targets in 09p gene intersection')
    return G,regs,covered,len(conflict)


def null_graphs(G,n=N_REWIRES,seed=SEED):
    """Shuffle COMPLETE signed rows within original target-degree classes.
    Keeps each gene's degree and each regulator's signed degree unchanged.
    """
    rng=np.random.default_rng(seed)
    d=(G!=0).sum(axis=1)
    groups=[np.flatnonzero(d==k) for k in np.unique(d) if k>0 and np.sum(d==k)>1]
    need(groups,'cannot build degree matched graph controls')
    for _ in range(n):
        perm=np.arange(len(d))
        for g in groups:
            perm[g]=rng.permutation(g)
        H=G[perm]
        need(np.array_equal((H!=0).sum(axis=1),d),'rewire changed target degree')
        need(np.array_equal((H==1).sum(axis=0),(G==1).sum(axis=0)),'rewire changed positive in-degree')
        need(np.array_equal((H==-1).sum(axis=0),(G==-1).sum(axis=0)),'rewire changed negative in-degree')
        need(not np.array_equal(H,G),'rewire unchanged')
        yield H


def rms_scale(X):
    X=np.asarray(X,float)
    scale=np.sqrt(np.mean(X*X,axis=0))
    keep=scale>1e-10
    need(bool(np.any(keep)), 'basis has no nonzero feature')
    return X[:,keep]/scale[keep],keep


def train_design(Y,meta,G,train_col_indices,reg_mode='signed'):
    """Train-only low-rank gene response dictionary, and soft TF feature reliability.
    All training contrasts are rescaled on TRAINING outcomes and weighted so each
    source STUDY contributes equal total squared energy to the PCA.
    """
    Yt=Y[:,train_col_indices].copy()
    sub=meta.iloc[train_col_indices]
    scales=np.median(np.abs(Yt),axis=0)
    need(np.all(scales>1e-5),'noninformative training contrast with median effect ~0')
    Yt=Yt/scales[None,:]
    weights=np.array([1/np.sqrt((sub.study==s).sum()) for s in sub.study],float)
    Yw=Yt*weights[None,:]
    centroid=Yw.mean(axis=1,keepdims=True)
    centered=Yw-centroid
    U,s,_=np.linalg.svd(centered,full_matrices=False)
    n_components=min(N_PCA,max(0,int(np.sum(s>1e-8))))
    candidate=np.column_stack([centroid,U[:,:n_components]]) if n_components else centroid
    base, _=rms_scale(candidate)
    # Always add intercept. Train-only response mean/PCA are the transfer baseline.
    B=np.column_stack([np.ones(len(Y)),base])
    if reg_mode=='none':
        return B,0,{}
    H=np.asarray(G,float).copy()
    if reg_mode=='unsigned':
        H=np.abs(H)
    H,good=rms_scale(H)
    # Fit a TRAINING-only empirical prior on regulator coefficients after
    # projecting out the low-rank response representation.
    cov=B.T@B+RIDGE_BASE*len(Y)*np.eye(B.shape[1])
    residual=Yw-B@np.linalg.solve(cov,B.T@Yw)
    rcoef=np.linalg.solve(H.T@H+RIDGE_REG_FIT*len(Y)*np.eye(H.shape[1]),H.T@residual)
    activity=np.sqrt(np.mean(rcoef*rcoef,axis=1))
    pos=activity[activity>1e-10]
    median=np.median(pos) if len(pos) else 1.0
    # Strong shrinkage towards equal coefficient prior, because only a few
    # distinct training studies are available. Prevent one large training
    # response from dominating the regulator prior.
    reliability=np.clip(0.8+0.2*(activity/median),0.6,1.5)
    # All FOLD-specific feature scales are trained without target study outcome.
    H=H*reliability[None,:]
    return np.column_stack([B,H]),H.shape[1],{'regulator_reliability':reliability.tolist(),
        'train_contrast_scales':scales.tolist(),'n_train_studies':len(set(sub.study)),
        'n_train_contexts':len(sub)}


def ridge_predict(design,y,anchor,reg_features):
    """Infer held-out study coefficients ONLY from known anchor gene outcomes."""
    anchor=np.asarray(anchor,dtype=bool)
    need(10<anchor.sum()<len(y)-10,'bad anchor mask')
    X=design[anchor,:]
    observed=y[anchor]
    n=len(observed)
    pen=np.full(X.shape[1],RIDGE_BASE*n)
    pen[0]=1e-8  # intercept unpenalized
    if reg_features:
        pen[-reg_features:]=RIDGE_REG*n
    gram=X.T@X+np.diag(pen)
    beta=np.linalg.solve(gram,X.T@observed)
    return design@beta


def block_anchors(genes,frac,repeat):
    """Pre-fixed random subsets of locus-tag blocks (NOT verified operons)."""
    # Input genes are bare PAO1 tags, sorted in numerical PA locus order.
    blocks=np.array([int(g[2:])//LOCUS_BLOCK_WIDTH for g in genes],dtype=int)
    unique=np.unique(blocks)
    n=max(2,min(len(unique)-2,round(frac*len(unique))))
    rng=np.random.default_rng(SEED+int(1000*frac)*1000+repeat)
    sampled=rng.choice(unique,size=n,replace=False)
    return np.isin(blocks,sampled)


def metrics(yhat,y,observed):
    ok=(~observed) & np.isfinite(y) & np.isfinite(yhat)
    n=int(ok.sum())
    need(n>20,'too few test genes')
    actual=y[ok]
    pred=yhat[ok]
    valid=(np.abs(actual)>1e-12) & (np.abs(pred)>1e-12)
    agree=(np.sign(actual[valid])==np.sign(pred[valid]))
    u=actual[valid]>0; d=actual[valid]<0
    balanced=float((np.mean(pred[valid][u]>0)+np.mean(pred[valid][d]<0))/2) if u.any() and d.any() else np.nan
    corr=float(np.corrcoef(pred,actual)[0,1]) if np.std(pred)>1e-10 and np.std(actual)>1e-10 else np.nan
    return {'n_test_genes':n,'mae':float(np.mean(np.abs(pred-actual))),
       'sign_accuracy':float(agree.mean()) if valid.any() else np.nan,
       'balanced_sign_accuracy':balanced, 'pearson':corr}


def benchmark(wide,meta,prior,n_rewires=N_REWIRES,n_splits=N_SPLITS,budgets=BUDGETS,strict=True):
    genes=wide.index.to_numpy(dtype=str)
    need(all(g.startswith('PA') and len(g)==6 and g[2:].isdigit() for g in genes),'non-bare PA loci')
    Y=wide.to_numpy(float)
    G,regulators,covered,nconflict=make_prior(genes,prior,strict=strict)
    nulls=list(null_graphs(G,n=n_rewires))
    rows=[]; predictions=[]; reliability=[]
    studies=meta.study.unique().tolist()
    for study in studies:
        train_indices=np.flatnonzero(meta.study.to_numpy()!=study)
        held_indices=np.flatnonzero(meta.study.to_numpy()==study)
        need(len(train_indices)>1 and len(held_indices)>=1,'study split invalid')
        designs={}
        for method,H,mode in [('NO_GRAPH',G,'none'),('CURATED_SIGNED',G,'signed'),('CURATED_UNSIGNED',G,'unsigned')]:
            des,nreg,info=train_design(Y,meta,G if method=='NO_GRAPH' else H,train_indices,mode)
            designs[method]=(des,nreg)
            if method=='CURATED_SIGNED':
                for regulator,val in zip(regulators,info['regulator_reliability']):
                    reliability.append({'heldout_study':study,'regulator_GeneID':regulator,
                                        'train_only_reliability':val,'n_train_studies':info['n_train_studies']})
        null_designs=[train_design(Y,meta,H,train_indices,'signed')[:2] for H in nulls]
        for cid_idx in held_indices:
            record=meta.iloc[cid_idx]
            y=Y[:,cid_idx]
            for frac in budgets:
                for repeat in range(n_splits):
                    known=block_anchors(genes,frac,repeat)
                    coverage_n=int(np.sum(known & covered))
                    for panel,scoring_genes in [('ALL_COMMON',np.ones(len(Y),dtype=bool)),('CURATED_TARGETS',covered)]:
                        score_anchor=known | ~scoring_genes
                        # Make sure there are enough unseen regulator-covered genes.
                        if (~score_anchor).sum()<25: continue
                        models={
                            'ANCHOR_MEAN':np.full(len(Y),float(np.mean(y[known]))),
                            'ANCHOR_MAJORITY_SIGN':np.full(len(Y),(1. if np.mean(y[known]>0)>=.5 else -1.) * max(float(np.median(np.abs(y[known]))),1e-8))}
                        for method,(des,nreg) in designs.items():
                            models[method]=ridge_predict(des,y,known,nreg)
                        null_preds=[ridge_predict(des,y,known,nreg) for des,nreg in null_designs]
                        # Compare each individual rewired network, not only the ensemble.
                        null_maes=[]
                        for npred in null_preds:
                            null_maes.append(metrics(npred,y,score_anchor)['mae'])
                        for method,pred in models.items():
                            res=metrics(pred,y,score_anchor)
                            rows.append({'heldout_study':study,'contrast_id':record.contrast_id,
                                   'intervention_category':record.intervention_category,
                                   'primary_modelling_status':record.primary_modelling_status,
                                   'panel':panel,'anchor_block_fraction':frac,'anchor_split':repeat,
                                   'n_anchor_genes':int(known.sum()),'n_anchor_curated_genes':coverage_n,
                                   'method':method,**res,
                                   'mean_single_rewire_MAE':float(np.mean(null_maes)),
                                   'real_better_than_nulls_n':int(np.sum(np.array(null_maes)>res['mae'])) if method=='CURATED_SIGNED' else np.nan,
                                   'n_rewires':len(null_maes)})
                            if repeat==0 and frac==max(budgets) and method in ('NO_GRAPH','CURATED_SIGNED'):
                                for i,g in enumerate(genes):
                                    if scoring_genes[i] and not known[i]:
                                        predictions.append({'heldout_study':study,'contrast_id':record.contrast_id,'GeneID':g,
                                        'panel':panel,'method':method,'actual_response':float(y[i]),
                                        'pred_response':float(pred[i]),'is_regulator_covered':bool(covered[i])})
                        # Important: produce null as a *single-model average of metrics*;
                        # an ensemble has different statistical properties and is not fair.
                        null_metrics=[metrics(npred,y,score_anchor) for npred in null_preds]
                        rows.append({'heldout_study':study,'contrast_id':record.contrast_id,
                                    'intervention_category':record.intervention_category,
                                    'primary_modelling_status':record.primary_modelling_status,
                                    'panel':panel,'anchor_block_fraction':frac,'anchor_split':repeat,
                                    'n_anchor_genes':int(known.sum()),'n_anchor_curated_genes':coverage_n,
                                    'method':'DEGREE_MATCHED_REWIRE_MEAN',
                                    **{k:float(np.mean([z[k] for z in null_metrics])) if k!='n_test_genes' else null_metrics[0][k]
                                       for k in null_metrics[0]},
                                    'mean_single_rewire_MAE':float(np.mean(null_maes)),
                                    'real_better_than_nulls_n':np.nan,'n_rewires':len(null_maes)})
    out=pd.DataFrame(rows)
    need(not out.empty,'no metrics produced')
    return out,pd.DataFrame(predictions),pd.DataFrame(reliability),regulators,covered,nconflict


def run(root,out,strict=True,n_rewires=N_REWIRES,n_splits=N_SPLITS,budgets=BUDGETS):
    need(not out.exists(),'refusing to overwrite existing audit '+str(out))
    wide,meta,prior,paths=parse_data(root,strict=strict)
    results,preds,rels,regs,covered,conf=benchmark(wide,meta,prior,
                    n_rewires=n_rewires,n_splits=n_splits,budgets=budgets,strict=strict)
    def summary(df):
        return (df.groupby(['heldout_study','panel','anchor_block_fraction','method'],as_index=False)
                  .agg(n_contrasts=('contrast_id','nunique'),n_repeats=('mae','size'),
                       mae=('mae','mean'),sign_accuracy=('sign_accuracy','mean'),
                       balanced_accuracy=('balanced_sign_accuracy','mean'),pearson=('pearson','mean'),
                       mean_single_rewire_MAE=('mean_single_rewire_MAE','mean')))
    s=summary(results)
    print('\n'+'='*91)
    print('CF-STATE MODEL 09Q | MULTI-STUDY FEW-SHOT REGULATORY RESPONSE COMPLETION')
    print('='*91)
    print('Training: 09p provisional response contrasts only; leave WHOLE GEO study out')
    print('A target contrast contributes ONLY its ANCHOR gene responses to coefficient inference.')
    print('The scored test genes are disjoint from the anchor genes. NOT ZERO-SHOT.')
    print('Raw-run processed count-column provenance not independently established.')
    print('Studies:',meta.study.nunique(),'| contexts:',len(meta),'| common genes:',len(wide))
    print('Provisional PRIMARY studies:',len(PRIMARY),'| primary contrasts:',int(meta.study.isin(PRIMARY).sum()))
    print('Curated regulators:',len(regs),'| curated-covered genes:',int(covered.sum()),'| conflicting signed links:',conf)
    print('Anchor gene blocks of',LOCUS_BLOCK_WIDTH,'PA tags, not verified operons; budgets:',budgets,
          '| splits per budget:',n_splits,'| independent rewire maps:',n_rewires)
    print('Comparison: fixed source-trained PCA vs curated signs vs unsigned vs degree-matched rewires.')
    print('Reported MAEs are WITHIN held-out study effect scale; do not pool raw response magnitudes.')
    print('\nHELD-OUT STUDY RESULTS | 25% anchor gene blocks | ALL COMMON GENES:')
    z=s[(s.panel=='ALL_COMMON')&(s.anchor_block_fraction==max(budgets))]
    print(z[['heldout_study','n_contrasts','method','mae','sign_accuracy','balanced_accuracy','pearson']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\nHELD-OUT STUDY RESULTS | 25% anchor blocks | CURATED TARGETS:')
    z=s[(s.panel=='CURATED_TARGETS')&(s.anchor_block_fraction==max(budgets))]
    print(z[['heldout_study','n_contrasts','method','mae','sign_accuracy','balanced_accuracy','pearson']].to_string(index=False,float_format=lambda v:f'{v:.4f}'))
    print('\nCURATED GRAPH MINUS NO GRAPH (negative is better), study-level:')
    for frac in budgets:
        pivot=s[(s.panel=='CURATED_TARGETS')&(s.anchor_block_fraction==frac)].pivot(
            index='heldout_study',columns='method',values='mae')
        print(f'  Anchor block budget: {frac:.0%}')
        for study,row in pivot.iterrows():
            print('   ',study,'curated-minus-no-graph=',f"{row['CURATED_SIGNED']-row['NO_GRAPH']:+.5f}",
                  'curated-minus-rewires=',f"{row['CURATED_SIGNED']-row['DEGREE_MATCHED_REWIRE_MEAN']:+.5f}")
    print('\nCAUTION: two primary studies, different intervention families, non-blind development.')
    print('Other studies include LONG-TERM ADAPTATION and KNOWN PRIOR CF-State sample overlap.')
    print('Study-level 4-fold evaluations do not constitute four independent primary validation studies.')
    print('Graph sign is literature regulation, NOT signed response to perturbation.')
    print('The gene blocks reduce local leakage but are NOT independent biological observations.')
    print('No target test-gene responses used for dictionary training, coefficient fit or anchor selection.')
    out.mkdir(parents=True,exist_ok=False)
    results.to_csv(out/'HELDOUT_GENE_SPLIT_REPEATED_CONTRAST_METRICS.csv',index=False)
    s.to_csv(out/'SUMMARY_BY_HELDOUT_STUDY.csv',index=False)
    preds.to_csv(out/'HELDOUT_GENE_RESPONSE_PREDICTIONS_FIRST_SPLIT.csv',index=False)
    rels.to_csv(out/'TRAIN_ONLY_REGULATOR_RELIABILITY_BY_HELDOUT_STUDY.csv',index=False)
    meta.to_csv(out/'INHERITED_PROVISIONAL_CONTRAST_DESIGN.csv',index=False)
    pd.DataFrame({'regulator_GeneID':regs}).to_csv(out/'CURATED_REGULATORS.csv',index=False)
    pd.DataFrame({'path':[str(p.relative_to(root)) if p.is_relative_to(root) else str(p) for p in paths],
                  'sha256':[sha256(p) for p in paths]}).to_csv(out/'INPUT_SHA256_PROVENANCE.csv',index=False)
    (out/'QC_STATUS.json').write_text(json.dumps({'status':'09Q_RETROSPECTIVE_FEWSHOT_STUDY_HOLDOUT_MODEL_COMPLETE',
           'n_studies':int(meta.study.nunique()),'n_contrasts':int(len(meta)),'n_genes':int(len(wide)),
           'n_curated_targets':int(covered.sum()),'n_nulls':n_rewires,'anchor_split_repeats':n_splits,
           'leakage_guard':'heldout outcomes used for anchors only; all other genes untouched until metrics',
           'zero_shot_unseen_intervention':False,'primary_studies_only_two':True,
           'raw_run_provenance_complete':False,'evidence_status':'PROVISIONAL_RETROSPECTIVE_DEVELOPMENT'},indent=2)+'\n')
    print('\nSUCCESS | saved without overwrite:',out)
    return s


def synthetic_test():
    rng=np.random.default_rng(1347)
    with tempfile.TemporaryDirectory() as t:
        root=Path(t)
        p=root/P09; p.mkdir(parents=True)
        (root/G_PRIOR).parent.mkdir(parents=True)
        (root/G_QC).write_text(json.dumps({'mapped_09e4':50}))
        names=[f'PA{i:04d}' for i in range(1,501)]
        contexts=['S1__KO','S2__A','S2__B','S3__A','S3__B','S3__C','S4__A','S4__B']
        studies=['S1','S2','S2','S3','S3','S3','S4','S4']
        Y=rng.normal(0,1,(len(names),len(contexts)))
        for j in range(len(contexts)):
            Y[80:160,j]+=j/3
        pd.DataFrame(Y,index=names,columns=contexts).rename_axis('GeneID').to_csv(p/'FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv')
        pd.DataFrame({'contrast_id':contexts,'study':studies,
           'intervention_category':['GENETIC_KO']+['ANTIBIOTIC_ACUTE']*2+['LONG_TERM_ADAPTATION']*3+['GENETIC_KO_BIOFILM']*2,
           'primary_modelling_status':['PROVISIONAL_PRIMARY_GEO_TITLE_CORROBORATED']*3+['SECONDARY_LONG_TERM_ADAPTATION']*3+['SECONDARY_INTERNAL_OVERLAP']*2,
           'historical_CFSTATE_overlap':[False]*6+[True]*2}).to_csv(p/'CONTRAST_SOURCE_DESIGN_AND_QC.csv',index=False)
        (p/'QC_STATUS.json').write_text(json.dumps({'status':'09P_PROVISIONAL_MULTISTUDY_RESPONSE_CANDIDATES_FROZEN','raw_run_to_count_link_proven':False}))
        rows=[]
        for ri,reg in enumerate(['PA0501','PA0502','PA0503']):
            for i in range(40+ri*5,80+ri*5):
                rows.append({'regulator_GeneID':reg,'target_GeneID':names[i],
                  'original_strain_includes_PAO1':'True','raw_mode_of_regulation':'+' if i%3 else '-'})
        z=pd.DataFrame(rows)
        z.to_csv(root/G_PRIOR,index=False)
        wide,meta,prior,paths=parse_data(root,strict=False)
        result=run(root,root/'results/synthetic_09q',strict=False,n_rewires=3,n_splits=2,budgets=(0.10,0.25))
        need(len(result)>0,'synthetic summary absent')
        need((root/'results/synthetic_09q/QC_STATUS.json').exists(),'status not saved')
        try:
            run(root,root/'results/synthetic_09q',strict=False,n_rewires=3,n_splits=2,budgets=(0.10,0.25))
        except RuntimeError as e:
            need('refusing to overwrite' in str(e),'unexpected non-overwrite error')
        else:
            raise AssertionError('overwriting old results allowed')
        # Sentinel: held-out prediction cannot depend on *unscored target genes*.
        train_idx=np.flatnonzero(meta.study.to_numpy()!='S1')
        known=block_anchors(wide.index.to_numpy(dtype=str),0.25,0)
        GG,_,_,_=make_prior(wide.index.to_numpy(dtype=str),prior,strict=False)
        B,ns,_=train_design(wide.to_numpy(float),meta,GG,train_idx,'signed')
        y=wide.iloc[:,0].to_numpy(float)
        pred1=ridge_predict(B,y,known,ns)
        altered=y.copy(); altered[~known]+=1234.
        pred2=ridge_predict(B,altered,known,ns)
        need(np.allclose(pred1,pred2,atol=1e-10),'unobserved held-out gene outcome leaked')
        print('SELF_TEST_PASSED 09Q: 4-study leave-one-GEO-out, curated/unsigned/rewired,')
        print('  train-only response basis, no test-gene leakage, locus-block anchors, save/no-overwrite.')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--self-test',action='store_true')
    a=parser.parse_args()
    if a.self_test:
        synthetic_test()
    else:
        root=a.root.resolve()
        out=a.out if a.out.is_absolute() else root/a.out
        run(root,out)

if __name__=='__main__':
    main()
