#!/usr/bin/env python3
"""CF-State 09l | multi-perturbation signed-regulon response prototype.

TRAIN (frozen): GSE67006 Delta-anr array, GSE6741 oxygen gradient (three
correlated contrasts), Neff 2024 combined Zn+Fe+Mn (17 donors, collapsed to
one study-level contrast). HELD OUT: complete GSE68534 study (PAO1 AND J215).

Model is joint regularized regression over intervention-specific response scores:
    Y[g,c] ~= sum_k F[c,k] * (B[g,k] + sum_r M[g,r]*A[r,k])
where B is gene-specific, M is literature signed regulator incidence, and A
is a learnt perturbation-specific regulon-activity contribution.  Different
interventions have separate A columns; NO assumption that a + literature
activation edge means metals increase target expression.  The loss is convex,
solved with block coordinate descent. Fixed hyperparameters, diagnostic only.

NO target-study outcomes used for fitting, selection, normalization, or nulls.
The GSE68534 study was previously examined; this is retrospective, not blind.
RNAseq-vs-array magnitudes are NOT compared: ONLY predicted directions scored.

Run:
  python -m py_compile src/phase9/model_09l_multiperturbation_regulator_activity_transfer.py
  python src/phase9/model_09l_multiperturbation_regulator_activity_transfer.py --self-test
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -u src/phase9/model_09l_multiperturbation_regulator_activity_transfer.py
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
ARRAY = Path('results/phase9/audit_09a_perturbation_evidence_preflight/FROZEN_08m_native_array_gene_response_evidence.csv')
METAL = Path('results/phase9/audit_09b_Neff2024_paired_gene_response_freeze/FROZEN_4923_gene_combined_metals_response_summary.csv')
KO = Path('results/phase9/audit_09j2_GSE68534_usable_gene_QC_repair/FROZEN_GSE68534_PAO1_J215_delta_anr_minus_WT_gene_responses.csv')
KO_QC = Path('results/phase9/audit_09j2_GSE68534_usable_gene_QC_repair/QC_STATUS.json')
REG = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/FROZEN_CURATED_REGULATORY_ASSOCIATIONS_09E4.csv')
REG_QC = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/QC_09E4.json')
OUT = Path('results/phase9/model_09l_multiperturbation_regulator_activity_transfer')

CONTRASTS = [
    ('GSE67006', 'delta_anr_minus_WT_CFBE', 'ANR_KNOCKOUT'),
    ('GSE6741', 'O2_2_minus_20', 'OXYGEN_LOW'),
    ('GSE6741', 'O2_0p4_minus_20', 'OXYGEN_LOW'),
    ('GSE6741', 'anoxic_nitrate_minus_20', 'ANOXIC_PLUS_NITRATE'),
]
CONTEXTS = ['GSE67006_DELTA_ANR', 'GSE6741_O2_2', 'GSE6741_O2_0p4',
            'GSE6741_ANOXIC_NITRATE', 'NEFF_COMBINED_METALS']
FEATURES = ['common_gene_response', 'ANR_KNOCKOUT', 'OXYGEN_LOW',
            'ADDED_NITRATE_WITH_ANOXIA', 'COMBINED_METALS']
F = np.array([
    [1, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 0, 0, 0, 1],
], dtype=float)
# Three GSE6741 contrasts share one study. Total weight by study = 1 each.
W = np.array([1., 1./3, 1./3, 1./3, 1.], dtype=float)
GENE_RIDGE = 2.0
COMMON_RIDGE = 0.8
REGULON_RIDGE = 16.0
N_REGULATORS = 12
N_REWIRES = 30
SEED = 20261015


def need(ok, message):
    if not bool(ok):
        raise RuntimeError('MODEL_09L_FAILED: ' + str(message))


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def to_score(x):
    """Within-contrast effect scaling; preserve sign, avoid pooling measurement units."""
    x = np.asarray(x, dtype=float)
    need(np.isfinite(x).all(), 'nonfinite response')
    nz = np.abs(x[np.abs(x) > 1e-12])
    need(nz.size > 10, 'almost entirely zero contrast')
    scale = float(np.median(nz))
    need(scale > 0, 'invalid response scale')
    return np.tanh(x / scale), scale


def load(root, strict):
    inputs = [root / p for p in [ARRAY, METAL, KO, KO_QC, REG, REG_QC]]
    for p in inputs:
        need(p.is_file(), 'missing frozen source: ' + str(p))
    a = pd.read_csv(root / ARRAY, dtype={'GeneID':str})
    m = pd.read_csv(root / METAL, dtype={'GeneID':str})
    k = pd.read_csv(root / KO, dtype={'GeneID':str})
    r = pd.read_csv(root / REG, dtype=str, keep_default_na=False)
    need({'GeneID','dataset_id','contrast_id','effect_value_native'}.issubset(a.columns), '09a schema')
    need({'GeneID','mean_paired_log2_normalized_count_plus1_change'}.issubset(m.columns), '09b schema')
    need({'GeneID','strain','delta_anr_minus_WT_log2norm'}.issubset(k.columns), '09j2 schema')
    need({'regulator_GeneID','target_GeneID','raw_mode_of_regulation','original_strain_includes_PAO1'}.issubset(r.columns), '09e4 schema')
    need(m.GeneID.is_unique, 'metal gene duplicate')
    need(k[['strain','GeneID']].duplicated().sum()==0, 'KO strain-gene duplicate')
    if strict:
        need(len(m)==4923, 'expected 4923 metal genes')
        need(len(a)==19680, 'expected four 4920-gene GPL84 contrasts')
        need(len(k)==10898, 'expected 5449 genes in each KO strain')
        q = json.loads((root / KO_QC).read_text())
        need(q['sample_count_QC_max_relative_difference'] < 1e-8, '09j2 QC was not passed')
        qr = json.loads((root / REG_QC).read_text())
        need(int(qr['mapped_09e4'])==len(r)==806, '09e4 prior QC mismatch')
    frames = []
    for dataset, contrast, _ in CONTRASTS:
        z = a[(a.dataset_id==dataset)&(a.contrast_id==contrast)][['GeneID','effect_value_native']].copy()
        need(z.GeneID.is_unique and len(z)>= (4900 if strict else 40), 'missing contrast ' + contrast)
        z = z.rename(columns={'effect_value_native':contrast})
        frames.append(z.set_index('GeneID'))
    metal_col = 'mean_paired_log2_normalized_count_plus1_change'
    frames.append(m[['GeneID',metal_col]].rename(columns={metal_col:'METALS_MEAN_17'}).set_index('GeneID'))
    merged = frames[0]
    for z in frames[1:]:
        merged = merged.join(z,how='inner')
    need(merged.index.is_unique, 'duplicate training genes')
    merged = merged.sort_index()
    if strict:
        need(len(merged)>=4800, 'fewer than 4800 shared training genes')
    genes = merged.index.to_numpy(dtype=str)
    raw_y = merged.to_numpy(float)
    need(raw_y.shape==(len(genes),5), 'training matrix shape wrong')
    scales = []
    ys = []
    for col in range(5):
        s, scale = to_score(raw_y[:,col])
        ys.append(s)
        scales.append(scale)
    Y = np.column_stack(ys)
    # Curated signed edges: signs describe TF -> target regulation, NOT direction
    # of a particular environmental intervention's target-gene response.
    r = r[r.original_strain_includes_PAO1.str.lower().eq('true') &
          r.raw_mode_of_regulation.isin(['+','-']) &
          r.regulator_GeneID.str.fullmatch(r'PA\d{4}') &
          r.target_GeneID.str.fullmatch(r'PA\d{4}') &
          r.target_GeneID.isin(genes)].copy()
    r['s'] = r.raw_mode_of_regulation.map({'+':1.,'-':-1.})
    ct = r.groupby(['regulator_GeneID','target_GeneID']).s.nunique()
    conflicting = set(ct[ct>1].index)
    if conflicting:
        r = r[~r[['regulator_GeneID','target_GeneID']].apply(tuple,axis=1).isin(conflicting)]
    r = r.drop_duplicates(['regulator_GeneID','target_GeneID'])
    nr = r.groupby('regulator_GeneID').target_GeneID.nunique().sort_values(ascending=False)
    regulators = sorted([x for x in nr.index if nr[x]>=5],key=lambda x:(-int(nr[x]),x))[:N_REGULATORS]
    need(len(regulators)>=2, 'not enough multi-target regulators')
    r = r[r.regulator_GeneID.isin(regulators)].copy()
    idx = {g:i for i,g in enumerate(genes)}
    ri = {g:i for i,g in enumerate(regulators)}
    mat = np.zeros((len(genes),len(regulators)),dtype=float)
    for record in r.itertuples():
        mat[idx[record.target_GeneID],ri[record.regulator_GeneID]] = float(record.s)
    degrees = np.count_nonzero(mat,axis=1)
    # Normalize per gene to keep genes with multiple edges on comparable scale.
    M = mat / np.sqrt(np.maximum(degrees[:,None], 1.))
    covered = degrees > 0
    need(covered.sum()>= (250 if strict else 10), 'too few regulator-covered genes')
    outcomes = {}
    for strain in ['PAO1','J215']:
        sub = k[k.strain.eq(strain)].copy()
        sub['delta_anr_minus_WT_log2norm'] = pd.to_numeric(sub.delta_anr_minus_WT_log2norm,errors='raise')
        need(sub.GeneID.is_unique, 'duplicated target KO gene')
        outcomes[strain] = sub.set_index('GeneID')['delta_anr_minus_WT_log2norm']
    info = pd.DataFrame({'context':CONTEXTS,'study_id':['GSE67006','GSE6741','GSE6741','GSE6741','Neff2024'],
                         'intervention':['ANR_KNOCKOUT','OXYGEN_LOW','OXYGEN_LOW','ANOXIA_PLUS_NITRATE','Zn_Fe_Mn_COMBINED'],
                         'study_weight':W,'source_native_abs_median':scales,
                         'n_genes':len(genes)})
    return genes, Y, M, regulators, covered, outcomes, info, inputs, len(conflicting), r


def fit_joint(y, M, f=F, weights=W, use_network=True,
              gene_ridge=GENE_RIDGE,common_ridge=COMMON_RIDGE,
              regulon_ridge=REGULON_RIDGE, max_iter=150, tol=1e-10):
    """Joint strictly convex ridge fit; gene-level and regulon terms."""
    n,c = y.shape
    need(f.shape[0]==c and len(weights)==c and M.shape[0]==n, 'bad fit dimensions')
    K = f.shape[1]
    fw = f * weights[:,None]
    gram = f.T @ fw + np.diag([common_ridge]+[gene_ridge]*(K-1))
    inv_gram = np.linalg.inv(gram)
    yfw = y @ fw
    B = yfw @ inv_gram
    A = np.zeros((M.shape[1],K),dtype=float)
    if use_network and M.shape[1]:
        Q = M.T @ M
        H = f.T @ fw
        mat = np.kron(H,Q) + regulon_ridge * np.eye(M.shape[1]*K)
        for iteration in range(max_iter):
            rhs = M.T @ (yfw - B @ H)
            new_A = np.linalg.solve(mat,rhs.reshape(-1,order='F')).reshape(A.shape,order='F')
            new_B = (yfw - (M @ new_A) @ H) @ inv_gram
            difference = max(float(np.max(np.abs(new_B-B))),float(np.max(np.abs(new_A-A))))
            B,A = new_B,new_A
            if difference<tol:
                break
        else:
            need(False,'joint training did not converge')
    z = f[0]  # intervention descriptor: ANR knockout, shared across study environments
    pred = B @ z + M @ (A @ z)
    need(np.isfinite(pred).all(), 'nonfinite predictions')
    return pred,A,B


def rewired_matrices(M, n_null=N_REWIRES, seed=SEED):
    """Degree-preserving target-row swaps: regulator signed counts unchanged."""
    rng = np.random.default_rng(seed)
    degree = np.count_nonzero(M,axis=1)
    eligible = np.flatnonzero(degree>0)
    groups = [np.flatnonzero(degree==d) for d in np.unique(degree[eligible])]
    groups = [g for g in groups if len(g)>1]
    need(groups, 'cannot construct graph rewires')
    for i in range(n_null):
        permutation = np.arange(len(M))
        for group in groups:
            permutation[group] = rng.permutation(group)
        need(not np.array_equal(permutation[eligible],eligible), 'null identity permutation')
        new = M[permutation]
        need(np.array_equal(np.count_nonzero(new,axis=1),degree), 'null changed gene degree')
        need(np.array_equal(np.count_nonzero(new,axis=0),np.count_nonzero(M,axis=0)), 'null changed regulator degree')
        yield new


def metric(sign_pred, actual):
    valid = (np.sign(actual)!=0)&(np.sign(sign_pred)!=0)
    y = np.sign(actual[valid]); p = np.sign(sign_pred[valid])
    if not valid.any():
        return dict(n_scored=0,n_correct=0,accuracy=np.nan,balanced_accuracy=np.nan,
                    fraction_predicted_up=np.nan,n_observed_up=0,n_observed_down=0)
    acc = float(np.mean(y==p))
    ups = y==1
    downs = y==-1
    ba = float((np.mean(p[ups]==y[ups])+np.mean(p[downs]==y[downs]))/2) if ups.any() and downs.any() else np.nan
    return dict(n_scored=int(valid.sum()),n_correct=int((y==p).sum()),
                accuracy=acc,balanced_accuracy=ba,fraction_predicted_up=float((p==1).mean()),
                n_observed_up=int(ups.sum()),n_observed_down=int(downs.sum()))


def run(root, dest, strict=True, n_rewires=N_REWIRES):
    need(not dest.exists(), 'refusing to overwrite existing output: ' + str(dest))
    genes,Y,M,regs,covered,outcomes,info,inputs,n_conflicts,links = load(root,strict)
    print('\n'+'='*87)
    print('CF-STATE MODEL 09L | FIRST TRAINED MULTI-PERTURBATION REGULON RESPONSE MODEL')
    print('='*87)
    print('TRAIN: GSE67006 Anr KO + 3 GSE6741 oxygen/nitrate contrasts + 17-donor metal MEAN')
    print('HELD-OUT STUDY: GSE68534 (PAO1 and J215); neither strain is used for training')
    print('Study weights: GSE67006=1, GSE6741=1 TOTAL, Neff combined-metals=1 TOTAL')
    print('Effects transformed within each SOURCE contrast by tanh(effect/median_abs_effect).')
    print('Holdout RNAseq values are used ONLY for response-direction evaluation.')
    print('Gene universe:',len(genes),'| curated regulators:',len(regs),'| graph-covered genes:',int(covered.sum()))
    print('Fixed gene ridge:',GENE_RIDGE,'common ridge:',COMMON_RIDGE,'regulon ridge:',REGULON_RIDGE)
    print('Rewired networks:',n_rewires,'| conflicting prior sign pairs excluded:',n_conflicts)
    print('\nINPUT CONTEXT DESIGN:')
    print(info.to_string(index=False,float_format=lambda v:f'{v:.5f}'))
    print('\nFROZEN REGULATORS AND EDGE COUNTS:')
    print(pd.DataFrame({'regulator_GeneID':regs,'n_covered_targets':np.count_nonzero(M,axis=0)}).to_string(index=False))
    pred_gene,_,_ = fit_joint(Y,np.zeros((len(genes),0)))
    pred_net,A,_ = fit_joint(Y,M)
    pred_no_aux,_,_ = fit_joint(Y[:,:1],M,f=F[:1],weights=np.array([1.]))
    # NOTE: full and no-aux share the same fixed architecture; only other
    # intervention responses are removed in the ablation.
    M_unsigned = np.abs(M)
    pred_unsigned,_,_ = fit_joint(Y,M_unsigned)
    null_preds=[]
    for perm in rewired_matrices(M,n_null=n_rewires):
        z,_,_ = fit_joint(Y,perm)
        null_preds.append(z)
    nulls=np.stack(null_preds)
    # Exact source sign used only as a predictor, never as target-study outcome.
    source = Y[:,0]
    # Prior-only knockout signs for individually curated Anr links.
    anr_prior = np.full(len(genes),np.nan)
    anr_rows = links[links.regulator_GeneID.eq('PA1544')]
    for r in anr_rows.itertuples():
        gi=np.where(genes==r.target_GeneID)[0]
        if len(gi):
            anr_prior[gi[0]]=-float(r.s)
    columns={'ALWAYS_DOWN':-np.ones(len(genes)),
             'SOURCE_ANR_GENE_SIGNATURE':source,
             'MULTIPERTURB_NO_GRAPH':pred_gene,
             'MULTIPERTURB_CURATED_SIGNED':pred_net,
             'MULTIPERTURB_CURATED_UNSIGNED':pred_unsigned,
             'ANR_SOURCE_ONLY_CURATED_GRAPH':pred_no_aux,
             'CURATED_ANR_SIGN_ONLY':anr_prior}
    for i,v in enumerate(nulls):
        columns['REWIRED_%02d'%(i+1)] = v
    rows=[]
    all_predictions=[]
    for strain, effects in outcomes.items():
        lookup = effects.reindex(genes)
        exists = lookup.notna().to_numpy()
        if strict:
            need(exists.sum()>=4800, 'too few target outcome genes for '+strain)
        actual=lookup.to_numpy(float)
        predtab = pd.DataFrame({'GeneID':genes,'strain':strain,
                                'source_anr_response_score':source,
                                'target_anr_log2_effect_EVALUATION_ONLY':actual,
                                'network_covered':covered})
        for label, values in columns.items():
            if label.startswith('REWIRED_'):
                continue
            predtab['score_'+label]=values
            predtab['sign_'+label]=np.sign(values)
        # All models use identical test-observed genes within each comparison panel.
        top = np.zeros(len(genes),dtype=bool)
        top[np.argsort(-np.abs(source))[:min(500,len(genes))]]=True
        masks={'ALL_SHARED':np.ones(len(genes),dtype=bool),
               'SOURCE_TOP500':top,
               'CURATED_TARGETS':covered,
               'CURATED_ANR_TARGETS':np.isfinite(anr_prior)}
        for panel, pmask in masks.items():
            mask=pmask & exists & (actual!=0)
            if not mask.any():
                continue
            for method, values in columns.items():
                if method.startswith('REWIRED_') and panel not in ('CURATED_TARGETS','ALL_SHARED'):
                    continue
                valid=mask & np.isfinite(values) & (values!=0)
                if not valid.any():
                    continue
                met=metric(values[valid],actual[valid])
                rows.append(dict(strain=strain,panel=panel,method=method,**met))
        all_predictions.append(predtab[predtab.target_anr_log2_effect_EVALUATION_ONLY.notna()])
    met = pd.DataFrame(rows)
    pt = pd.concat(all_predictions,ignore_index=True)
    metric_keep=['SOURCE_ANR_GENE_SIGNATURE','MULTIPERTURB_NO_GRAPH',
                 'MULTIPERTURB_CURATED_SIGNED','MULTIPERTURB_CURATED_UNSIGNED',
                 'ANR_SOURCE_ONLY_CURATED_GRAPH','ALWAYS_DOWN','CURATED_ANR_SIGN_ONLY']
    print('\nRESULTS: source signature vs fitted models. DIRECTIONS ONLY.')
    for strain in ['PAO1','J215']:
        print('\n--- HELD-OUT GSE68534:',strain,'---')
        for panel in ['ALL_SHARED','SOURCE_TOP500','CURATED_TARGETS','CURATED_ANR_TARGETS']:
            tab=met[(met.strain==strain)&(met.panel==panel)&met.method.isin(metric_keep)].copy()
            if tab.empty: continue
            tab['accuracy']=tab.accuracy.map(lambda x: f'{x:.4f}')
            tab['balanced_accuracy']=tab.balanced_accuracy.map(lambda x:f'{x:.4f}' if pd.notna(x) else '-')
            print('\n'+panel)
            print(tab[['method','n_scored','accuracy','balanced_accuracy','fraction_predicted_up']].to_string(index=False))
            q=met[(met.strain==strain)&(met.panel==panel)&met.method.str.startswith('REWIRED_')]
            if len(q):
                real=met[(met.strain==strain)&(met.panel==panel)&met.method.eq('MULTIPERTURB_CURATED_SIGNED')].accuracy.iloc[0]
                print('  degree-matched individual rewires: mean=',round(float(q.accuracy.mean()),4),
                      'range=',[round(float(q.accuracy.min()),4),round(float(q.accuracy.max()),4)],
                      'curated beats=',int((real>q.accuracy).sum()),'/',len(q))
    effect_out=pd.DataFrame({'regulator_GeneID':regs})
    for j,col in enumerate(FEATURES):
        effect_out['coefficient_'+col]=A[:,j]
    print('\nLEARNED CONTEXT-DEPENDENT REGULON CONTRIBUTIONS (training only):')
    print(effect_out.to_string(index=False,float_format=lambda x:f'{x:+.4f}'))
    print('\nCAUTION: EVALUATION IS RETROSPECTIVE; GSE68534 ALREADY EXPLORED IN PHASE 08.')
    print('Interventions differ; this model can extrapolate to a different study of a KNOWN intervention,')
    print('not an unseen perturbation category. Metal donor effects collapsed to one study-level contrast.')
    print('Regulon coefficients are statistical, ridge-dependent components, NOT measured TF activity.')
    print('2011 signed edges are associations, not necessarily direct binding or CF-specific regulation.')
    print('The test context (low-O2 biofilm) is NOT explicitly encoded beyond its known Anr knockout.')
    print('Within-source tanh scaling is an exploratory comparability choice, not validated calibration.')
    print('No cross-platform effect-size comparison, no TF-activity identification from held-out outcomes.')
    print('No hyperparameter tuning on GSE68534. 30 rewires are DESCRIPTIVE, not a p-value.')
    # Only write once all data validation and model fitting has succeeded.
    dest.mkdir(parents=True,exist_ok=False)
    met.to_csv(dest/'HELDOUT_STUDY_SIGN_COMPARISONS.csv',index=False)
    pt.to_csv(dest/'HELDOUT_STUDY_PER_GENE_PREDICTIONS.csv',index=False)
    info.to_csv(dest/'TRAIN_ONLY_INTERVENTION_DESIGN.csv',index=False)
    effect_out.to_csv(dest/'LEARNED_TRAIN_ONLY_REGULON_CONTEXT_COEFFICIENTS.csv',index=False)
    pd.DataFrame([{'source_file':str(p.resolve()),'sha256':checksum(p)} for p in inputs]).to_csv(dest/'INPUT_SHA256_PROVENANCE.csv',index=False)
    state={'status':'PHASE09L_MULTI_PERTURBATION_PROTOTYPE_COMPLETE',
           'n_training_genes':len(genes),'n_covered_genes':int(covered.sum()),
           'n_frozen_regulators':len(regs),'regulators':regs,
           'n_correlated_GSE6741_contrasts':3,'n_metal_studies':1,
           'n_GSE68534_strains_held_out_together':2,
           'n_rewired_graphs':n_rewires,'fixed_gene_ridge':GENE_RIDGE,
           'fixed_common_ridge':COMMON_RIDGE,'fixed_regulon_ridge':REGULON_RIDGE,
           'no_target_study_used_for_training_or_hyperparameter_selection':True,
           'not_blind_external_validation':True,'not_unseen_intervention_prediction':True}
    (dest/'QC_STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
    print('\nSUCCESS | saved without overwrite:',dest,flush=True)
    return met,state


def self_test():
    rng=np.random.default_rng(93)
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        for p in [ARRAY,METAL,KO,KO_QC,REG,REG_QC]:
            (root/p).parent.mkdir(parents=True,exist_ok=True)
        genes=[f'PA{i:04d}' for i in range(1,101)]
        rec=[]
        for ds,contrast,_ in CONTRASTS:
            for i,g in enumerate(genes):
                v=rng.normal(size=1)[0]+0.25*np.sin(i/11)
                rec.append(dict(GeneID=g,dataset_id=ds,contrast_id=contrast,effect_value_native=v))
        pd.DataFrame(rec).to_csv(root/ARRAY,index=False)
        pd.DataFrame({'GeneID':genes,'mean_paired_log2_normalized_count_plus1_change':rng.normal(size=100)}).to_csv(root/METAL,index=False)
        ko=[]
        for strain in ('PAO1','J215'):
            for i,g in enumerate(genes):
                ko.append(dict(strain=strain,GeneID=g,delta_anr_minus_WT_log2norm=rng.normal()+0.15*i/100))
        pd.DataFrame(ko).to_csv(root/KO,index=False)
        (root/KO_QC).write_text(json.dumps({'sample_count_QC_max_relative_difference':0.0}))
        edges=[]
        for j,reg in enumerate(['PA0001','PA0002','PA0003']):
            for idx in range(j*15,(j+2)*15):
                edges.append(dict(regulator_GeneID=reg,target_GeneID=genes[idx],
                                  raw_mode_of_regulation='+' if (idx+j)%3 else '-',
                                  original_strain_includes_PAO1='true'))
        pd.DataFrame(edges).to_csv(root/REG,index=False)
        (root/REG_QC).write_text(json.dumps({'mapped_09e4':len(edges)}))
        y=rng.normal(size=(100,5))
        MM=np.zeros((100,3))
        for j in range(3): MM[j*15:(j+2)*15,j]=1
        p,a,b=fit_joint(y,MM)
        p0,_,_=fit_joint(y,np.zeros((100,0)))
        need(np.isfinite(p).all() and np.isfinite(p0).all() and a.shape==(3,5),'bad synthetic model')
        p2,_,_=fit_joint(y[:,:1],MM,f=F[:1],weights=np.array([1.]))
        need(p2.shape==(100,), 'single-context ablation broken')
        permutations=list(rewired_matrices(MM,5))
        need(len(permutations)==5,'null generation failed')
        output=root/OUT
        metrics, status=run(root,output,strict=False,n_rewires=5)
        need(output.is_dir() and (output/'QC_STATUS.json').is_file(),'output not saved')
        need(set(metrics.strain)=={'PAO1','J215'}, 'missing target strains')
        need(status['n_frozen_regulators']==3,'wrong regulators')
        try:
            run(root,output,strict=False,n_rewires=5)
        except RuntimeError as e:
            need('refusing to overwrite' in str(e),'wrong no-overwrite error')
        else:
            need(False,'unsafe overwrite of frozen output')
        print('SELF_TEST_PASSED: study weighting, signed multitask training, same-architecture rewires,')
        print('  whole-study heldout targets, no-overwrite outputs, source provenance, terminal summary.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        root=args.root.resolve()
        output=(args.output.resolve() if args.output else root/OUT)
        run(root,output,strict=True)

if __name__=='__main__':
    main()
