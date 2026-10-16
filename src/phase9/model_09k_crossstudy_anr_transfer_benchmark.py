#!/usr/bin/env python3
"""CF-State 09k: fixed retrospective study-held-out Anr sign-prediction challenge.

Prediction sources (not retrained on GSE68534):
  GSE67006 GPL84 Δanr - WT expression sign, curated 2011 Anr reported +/-,
  always-down and source-majority direction baselines.
Targets: separate GSE68534 PAO1 and J215 RNA-seq Δanr - WT contrasts.

Only signs are transferred between platforms, not magnitudes. There is no
parameter fitting on the held-out experiment. Phase 08 already explored the
GSE68534 study; therefore this is retrospective, NOT a blind validation.
Outputs print at terminal and are saved to an immutable-by-contract directory.

Run:
 python src/phase9/model_09k_crossstudy_anr_transfer_benchmark.py --self-test
 python -u src/phase9/model_09k_crossstudy_anr_transfer_benchmark.py
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
ARRAY = Path('results/phase9/audit_09a_perturbation_evidence_preflight/FROZEN_08m_native_array_gene_response_evidence.csv')
KO = Path('results/phase9/audit_09j2_GSE68534_usable_gene_QC_repair/FROZEN_GSE68534_PAO1_J215_delta_anr_minus_WT_gene_responses.csv')
KO_QC = Path('results/phase9/audit_09j2_GSE68534_usable_gene_QC_repair/QC_STATUS.json')
REG = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/FROZEN_CURATED_REGULATORY_ASSOCIATIONS_09E4.csv')
REG_QC = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/QC_09E4.json')
GFF = Path('references/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz')
OUT = Path('results/phase9/model_09k_crossstudy_anr_transfer_benchmark')
CHROM = 'NC_002516.2'
EXPECTED_GFF_SHA = 'e6b8bc796af95342acd5c170ecf5003e75135c894b1c22f8c4b46c51c4bdf7f0'
GENE = re.compile(r'^PA\d{4}$')


def need(cond, msg):
    if not bool(cond):
        raise RuntimeError('MODEL_09K_FAILED: ' + str(msg))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for x in iter(lambda:f.read(1024*1024), b''):
            h.update(x)
    return h.hexdigest()


def genome_table(path, strict):
    need(path.is_file(), 'missing independent NCBI GFF3 ' + str(path))
    if strict:
        need(sha256(path) == EXPECTED_GFF_SHA, 'NCBI GFF checksum differs from verified input')
    open_file = gzip.open if path.suffix == '.gz' else open
    rows = []
    seen = False
    with open_file(path, 'rt', encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('##sequence-region ' + CHROM + ' '):
                seen = True
            if line.startswith('#'):
                continue
            q = line.rstrip('\n').split('\t')
            if len(q) != 9 or q[0] != CHROM or q[2].lower() != 'gene':
                continue
            a = {}
            for z in q[8].split(';'):
                if '=' in z:
                    k, v = z.split('=', 1)
                    a[k] = unquote(v)
            gid = a.get('locus_tag', '').upper()
            if GENE.fullmatch(gid):
                rows.append((gid,int(q[3]),q[6]))
    need(seen and len(rows) >= (4900 if strict else 12), 'incorrect or incomplete PAO1 GFF3')
    tab = pd.DataFrame(rows,columns=['GeneID','start','strand']).sort_values(['start','GeneID'])
    need(tab.GeneID.is_unique,'duplicate genome feature IDs')
    tab['genome_order'] = np.arange(len(tab))
    return tab


def load_inputs(root, strict=True):
    paths = [root/p for p in [ARRAY,KO,KO_QC,REG,REG_QC,GFF]]
    for p in paths:
        need(p.is_file(), 'missing frozen source: ' + str(p))
    a = pd.read_csv(root/ARRAY, dtype={'GeneID':str})
    k = pd.read_csv(root/KO, dtype={'GeneID':str})
    r = pd.read_csv(root/REG, dtype=str, keep_default_na=False)
    kqc = json.loads((root/KO_QC).read_text())
    rqc = json.loads((root/REG_QC).read_text())
    aneed = {'contrast_id','GeneID','effect_value_native'}
    kneed = {'strain','GeneID','delta_anr_minus_WT_log2norm','contrast_id'}
    rneed = {'regulator_GeneID','target_GeneID','raw_mode_of_regulation','original_strain_includes_PAO1','source_excel_row','original_PMID'}
    need(aneed.issubset(a.columns),'GSE67006 array schema changed')
    need(kneed.issubset(k.columns),'GSE68534 KO schema changed')
    need(rneed.issubset(r.columns),'2011 curated schema changed')
    a = a.loc[a.contrast_id.eq('delta_anr_minus_WT_CFBE')].copy()
    need(a.GeneID.is_unique,'duplicate GSE67006 gene')
    a['source_native_effect'] = pd.to_numeric(a.effect_value_native, errors='raise')
    need(np.isfinite(a.source_native_effect).all(),'GSE67006 nonfinite')
    a = a[['GeneID','source_native_effect']]
    need(k[['strain','GeneID']].duplicated().sum()==0,'duplicate KO strain+gene')
    k['target_log2_effect'] = pd.to_numeric(k.delta_anr_minus_WT_log2norm, errors='raise')
    need(np.isfinite(k.target_log2_effect).all(), 'nonfinite GSE68534 KO effects')
    need(set(k.strain)=={'PAO1','J215'},'missing strain in GSE68534')
    if strict:
        need(len(a)==4920, 'expected 4920 frozen GSE67006 genes')
        need(len(k)==10898 and all((k.strain==s).sum()==5449 for s in ('PAO1','J215')),
             'expected 5449 genes for each GSE68534 strain')
        need(int(kqc['n_source_mapped_genes'])==5449 and kqc['sample_count_QC_max_relative_difference']<1e-8,
             '09j2 read QC not passed')
        need(int(rqc['mapped_09e4'])==len(r)==806, '09e4 curator QC not passed')
    coord = genome_table(root/GFF,strict)
    need(set(a.GeneID).issubset(set(coord.GeneID)), 'GSE67006 locus outside PAO1 GFF')
    # Curated Anr modes +: Anr activates, so knockout predicts down; -: knockout predicts up.
    q = r.loc[r.regulator_GeneID.eq('PA1544')
              & r.original_strain_includes_PAO1.str.lower().eq('true')
              & r.raw_mode_of_regulation.isin(['+','-'])
              & r.target_GeneID.map(lambda z:bool(GENE.fullmatch(z)))].copy()
    q['curated_pred_sign'] = q.raw_mode_of_regulation.map({'+':-1,'-':1}).astype(int)
    nmode = q.groupby('target_GeneID').curated_pred_sign.nunique()
    contradictory = set(nmode[nmode>1].index)
    q = q[~q.target_GeneID.isin(contradictory)].copy()
    def joinu(v):
        return '|'.join(sorted({str(x) for x in v if str(x).strip()}))
    q = q.groupby('target_GeneID',as_index=False).agg(
        curated_pred_sign=('curated_pred_sign','first'),
        curated_PMIDs=('original_PMID',joinu),n_curated_rows=('source_excel_row','size'))
    q = q.rename(columns={'target_GeneID':'GeneID'})
    need(q.GeneID.is_unique, 'curated row collapse failed')
    if strict:
        need(len(q)>=30, 'unexpectedly few curated Anr targets')
    return a,k,q,coord,paths,contradictory


def with_neighborhoods(g, coord):
    g = g.merge(coord,on='GeneID',validate='one_to_one',how='left')
    need(g.genome_order.notna().all(), 'missing gene coordinate')
    g = g.sort_values(['genome_order','GeneID']).reset_index(drop=True)
    groups = []
    prev = None
    n = 0
    for row in g.itertuples():
        if prev is None or int(row.genome_order) - int(prev.genome_order) > 2 or row.strand != prev.strand:
            n+=1
        groups.append('NH%05d' % n)
        prev=row
    g['genomic_neighborhood_id'] = groups
    return g


def metrics(data, method, pred_col):
    z = data.loc[data.target_sign.ne(0) & data[pred_col].notna()].copy()
    if len(z)==0:
        return dict(method=method,n_genes=0,n_neighborhoods=0,accuracy=np.nan,
                    balanced_accuracy=np.nan,macro_neighborhood_accuracy=np.nan,
                    n_correct=0,n_up=0,n_down=0)
    need(set(z[pred_col].astype(int)).issubset({-1,1}), 'bad model sign output')
    z['correct'] = z[pred_col].astype(int).eq(z.target_sign)
    ups = z[z.target_sign==1]
    downs = z[z.target_sign==-1]
    sens1 = float(ups.correct.mean()) if len(ups) else np.nan
    sens2 = float(downs.correct.mean()) if len(downs) else np.nan
    balanced = float(np.mean([sens1,sens2])) if np.isfinite(sens1) and np.isfinite(sens2) else np.nan
    return dict(method=method,n_genes=len(z),
                n_neighborhoods=z.genomic_neighborhood_id.nunique(),
                accuracy=float(z.correct.mean()),balanced_accuracy=balanced,
                macro_neighborhood_accuracy=float(z.groupby('genomic_neighborhood_id').correct.mean().mean()),
                n_correct=int(z.correct.sum()),n_up=int(len(ups)),n_down=int(len(downs)))


def neighborhood_bootstrap_diff(t, pred_a, pred_b, B=1200, seed=90210):
    z = t[t.target_sign.ne(0) & t[pred_a].notna() & t[pred_b].notna()].copy()
    if z.empty:
        return {'n_genes':0, 'n_neighborhoods':0,'delta_accuracy':None,'CI95':[None,None]}
    z['delta'] = z[pred_a].astype(int).eq(z.target_sign).astype(float) - z[pred_b].astype(int).eq(z.target_sign).astype(float)
    groups = z.groupby('genomic_neighborhood_id',sort=True).delta.agg(['sum','size'])
    v = groups['sum'].to_numpy(float)
    nn = groups['size'].to_numpy(float)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0,len(groups),size=(B,len(groups)))
    diffs = v[sample].sum(axis=1)/nn[sample].sum(axis=1)
    return {'n_genes':len(z),'n_neighborhoods':len(groups),
            'delta_accuracy':float(z.delta.mean()),
            'CI95':[float(x) for x in np.quantile(diffs,[0.025,0.975])]}


def run(root, dest, strict=True):
    need(not dest.exists(),'refusing to overwrite existing audit: ' + str(dest))
    a,k,q,coord,inputs,contradictory = load_inputs(root,strict)
    # Only SOURCE study determines source gene-response signature and majority sign.
    a = a.copy()
    a['source_sign'] = np.sign(a.source_native_effect).astype(int)
    nonzero = a.loc[a.source_sign.ne(0),'source_sign']
    need(len(nonzero)>0,'no source direction labels')
    nup=int((nonzero==1).sum())
    ndown=int((nonzero==-1).sum())
    # Freeze source-majority prediction from GSE67006 alone, break ties toward DOWN.
    majority = 1 if nup>ndown else -1
    a['source_top_250'] = False
    a['source_top_500'] = False
    sorted_source = a.assign(abs_source=a.source_native_effect.abs()).sort_values(
        ['abs_source','GeneID'],ascending=[False,True])
    for top in [250,500]:
        ids = set(sorted_source.head(top).GeneID)
        a['source_top_'+str(top)] = a.GeneID.isin(ids)
    all_tests=[]
    rows=[]
    bootstrap=[]
    panel_counts=[]
    method_def=[('ALWAYS_DOWN','always_down'),('SOURCE_MAJORITY','source_majority'),
                ('GSE67006_GENE_SIGNATURE','source_sign'),('CURATED_ANR_REGULATORY_SIGN','curated_pred_sign')]
    for strain in ['PAO1','J215']:
        target=k.loc[k.strain.eq(strain),['GeneID','target_log2_effect']].copy()
        target=target.merge(a,on='GeneID',how='inner',validate='one_to_one')
        target=target.merge(q,on='GeneID',how='left',validate='one_to_one')
        target['strain']=strain
        target['target_sign']=np.sign(target.target_log2_effect).astype(int)
        target['always_down']=-1
        target['source_majority']=majority
        target.loc[target.source_sign.eq(0),'source_sign']=np.nan
        target['has_curated_ANR_prior']=target.curated_pred_sign.notna()
        target['source_prior_agree']=target.has_curated_ANR_prior & target.source_sign.eq(target.curated_pred_sign)
        target['source_prior_disagree']=target.has_curated_ANR_prior & target.source_sign.notna() & ~target.source_prior_agree
        target=with_neighborhoods(target,coord)
        # The panel definitions are decided without GSE68534 KO effects.
        panel_defs = [
            ('ALL_SHARED',target.GeneID.notna()),
            ('SOURCE_TOP250',target.source_top_250),
            ('SOURCE_TOP500',target.source_top_500),
            ('CURATED_ANR_TARGETS',target.has_curated_ANR_prior),
            ('CURATED_ANR_NO_SELF',target.has_curated_ANR_prior & target.GeneID.ne('PA1544')),
            ('CURATED_AND_SOURCE_AGREE',target.source_prior_agree),
            ('CURATED_AND_SOURCE_DISAGREE',target.source_prior_disagree),
        ]
        for panel,mask in panel_defs:
            sub=with_neighborhoods(target.loc[mask].drop(columns=['genomic_neighborhood_id','genome_order','start','strand']).copy(),coord)
            panel_counts.append(dict(strain=strain,panel=panel,n_selected_without_target_response=len(sub),
                                     n_nonzero_target=int(sub.target_sign.ne(0).sum())))
            for method,col in method_def:
                v=metrics(sub,method,col)
                rows.append(dict(strain=strain,panel=panel,**v))
            if panel in ['ALL_SHARED','SOURCE_TOP500','CURATED_ANR_TARGETS','CURATED_ANR_NO_SELF']:
                for a_name,b_name in [('GSE67006_GENE_SIGNATURE','ALWAYS_DOWN'),
                                      ('GSE67006_GENE_SIGNATURE','SOURCE_MAJORITY'),
                                      ('CURATED_ANR_REGULATORY_SIGN','ALWAYS_DOWN'),
                                      ('CURATED_ANR_REGULATORY_SIGN','GSE67006_GENE_SIGNATURE')]:
                    cols=dict(method_def)
                    res=neighborhood_bootstrap_diff(sub,cols[a_name],cols[b_name],B=1200)
                    bootstrap.append(dict(strain=strain,panel=panel,compare=a_name+' MINUS '+b_name,
                                          **res))
        all_tests.append(target)
    pred = pd.concat(all_tests,ignore_index=True)
    met=pd.DataFrame(rows)
    boots=pd.DataFrame(bootstrap)
    panel_df=pd.DataFrame(panel_counts)
    if strict:
        need(pred.groupby('strain').size().min() >= 4800,'too few GSE67006 to GSE68534 gene overlaps')
        need(len(met)==2*7*4,'incomplete benchmark results')
    print('\n' + '='*88)
    print('CF-STATE MODEL CHALLENGE 09K | ANR RESPONSE TRANSFER — NO HELD-OUT FITTING')
    print('='*88)
    print('SOURCE (fixed): GSE67006 PAO1 CFBE Δanr GPL84 array; genes=',len(a))
    print('TARGET (untouched for selection): GSE68534 RNA-seq, PAO1 and J215, 2 WT + 2 KO per strain')
    print('Literature prior: curated 2011 signed Anr targets, only PAO1-including original strains')
    print('Training-source sign composition: UP=',nup,' DOWN=',ndown,
          '| majority predictor=',('UP' if majority==1 else 'DOWN'))
    print('Curated Anr unambiguous targets:',len(q),' | conflicting sign targets quarantined:',len(contradictory))
    print('All cross-platform metrics use DIRECTION only, never raw effect magnitude comparisons.')
    for strain in ['PAO1','J215']:
        print('\n'+'-'*88)
        print('HELD-OUT EXPERIMENT: GSE68534', strain)
        show=met[(met.strain.eq(strain)) & met.panel.isin(['ALL_SHARED','SOURCE_TOP500','CURATED_ANR_TARGETS','CURATED_ANR_NO_SELF'])].copy()
        show['accuracy']=show.accuracy.map(lambda v:f'{v:.4f}' if np.isfinite(v) else '-')
        show['balanced_accuracy']=show.balanced_accuracy.map(lambda v:f'{v:.4f}' if np.isfinite(v) else '-')
        show['macro_neighborhood_accuracy']=show.macro_neighborhood_accuracy.map(lambda v:f'{v:.4f}' if np.isfinite(v) else '-')
        print(show[['panel','method','n_genes','n_neighborhoods','accuracy','balanced_accuracy',
                    'macro_neighborhood_accuracy']].to_string(index=False))
        print('\nNEIGHBORHOOD-RESAMPLED DIFFERENCE IN ACCURACY (descriptive only):')
        for x in boots[(boots.strain.eq(strain)) & boots.panel.isin(['ALL_SHARED','CURATED_ANR_TARGETS'])].to_dict('records'):
            if x['n_genes']:
                print(' ',x['panel'], '|', x['compare'], ':', f"{x['delta_accuracy']:+.4f}",
                      ' descriptive CI',f"[{x['CI95'][0]:+.4f}, {x['CI95'][1]:+.4f}]",
                      'n_regions=',x['n_neighborhoods'])
        sub=pred[(pred.strain.eq(strain)) & pred.has_curated_ANR_prior].copy()
        sub['source_correct']=sub.source_sign.eq(sub.target_sign)
        sub['prior_correct']=sub.curated_pred_sign.eq(sub.target_sign)
        print('\nCURATED Anr target details (source / prior / observed):')
        fmt=sub[['GeneID','source_native_effect','target_log2_effect','source_sign','curated_pred_sign',
                 'target_sign','source_correct','prior_correct','genomic_neighborhood_id']].copy()
        print(fmt.to_string(index=False,float_format=lambda x:f'{x:+.3f}'))
    print('\nINTERPRETATION:')
    print(' * GSE68534 is a different study but was already explored in earlier CF-State phases: RETROSPECTIVE.')
    print(' * PAO1 and J215 targets share one experiment; do not count them as independent studies.')
    print(' * Neighboring genes are correlated; equal-neighborhood scores and bootstrap CIs are descriptive.')
    print(' * All-down and source-majority are mandatory, same-panel baselines.')
    print(' * Literature signs describe Anr-related regulation, not direct ChIP binding or a universal intervention response.')
    print(' * Different interventions (combined Zn+Fe+Mn vs Δanr) are NOT pooled or used as training replicates.')
    print(' * This benchmarks fixed predictions, not a trained cross-context predictor; no CF specificity established.')
    dest.mkdir(parents=True,exist_ok=False)
    pred.to_csv(dest/'FROZEN_PER_GENE_CROSSSTUDY_PREDICTIONS_AND_OUTCOMES.csv',index=False)
    met.to_csv(dest/'SUMMARY_SIGN_PREDICTION_BASELINE_COMPARISONS.csv',index=False)
    boots.to_csv(dest/'DESCRIPTIVE_NEIGHBORHOOD_BOOTSTRAP_DIFFERENCES.csv',index=False)
    panel_df.to_csv(dest/'QC_SOURCE_DEFINED_GENE_PANELS.csv',index=False)
    pd.DataFrame([{'source_file':str(p.resolve()),'sha256':sha256(p)} for p in inputs]).to_csv(
        dest/'INPUT_SHA256_PROVENANCE.csv',index=False)
    status={'status':'PHASE09K_RETROSPECTIVE_ANR_CROSSSTUDY_SIGN_BASELINES_COMPLETE',
            'n_GSE67006_array_genes':len(a),'n_Anr_curated_prior_targets':len(q),
            'n_excluded_conflicting_Anr_prior_targets':len(contradictory),
            'n_shared_genes_by_strain':{s:int((pred.strain==s).sum()) for s in ['PAO1','J215']},
            'no_target_outcomes_used_to_fit_or_select':True,'no_model_fitted':True,
            'training_intervention':'delta_anr','not_blind_external_validation':True,
            'no_cross_platform_effect_scale_calibration':True}
    (dest/'QC_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    (dest/'DESIGN_AND_INTERPRETATION.md').write_text(
        '# 09k cross-study Anr sign transfer\n\nSource GSE67006 knockout on CFBE array; GSE68534 PAO1/J215 low oxygen RNA-seq targets. '
        'Gene panels are selected from source and literature only. NO target expression is used to select source_top_250, '
        'source_top_500, curated Anr targets, or source–prior agreement panels. Nonzero target effects are required '
        'to score a directional outcome and their exclusion is reported.\n\n'
        'The reference baseline is always-DOWN and training-source majority, plus GSE67006 gene signature and '
        'signed curated Anr response-direction prior. Literature curated edges come from diverse experiments and '
        'do NOT prove direct binding. Gene-neighborhood grouping uses one intervening GFF gene and same strand; '
        'it is NOT operon annotation.\n\n'
        'Two replicates per group, different experimental conditions and platforms, prior Phase 08 exposure, '
        'and one shared GSE68534 study limit inference. Not independent clinical CF specificity or model superiority.\n')
    print('\nSAVED without overwrite:',dest)
    return status


def self_test():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        for path in [ARRAY,KO,KO_QC,REG,REG_QC,GFF]:
            (root/path).parent.mkdir(parents=True,exist_ok=True)
        genes=[f'PA{i:04d}' for i in range(1,121)]
        # Gene order and strand explicitly available for neighborhood controls.
        lines=['##gff-version 3','##sequence-region NC_002516.2 1 200000']
        for j,g in enumerate(genes):
            lines.append(f'NC_002516.2\tRefSeq\tgene\t{j*100+1}\t{j*100+90}\t.\t+\t.\tlocus_tag={g}')
        with gzip.open(root/GFF,'wt') as f:
            f.write('\n'.join(lines)+'\n')
        a=pd.DataFrame({'GeneID':genes[:105],
                        'contrast_id':'delta_anr_minus_WT_CFBE',
                        'effect_value_native':[(-1)**j * (1+j/110) for j in range(105)]})
        a.to_csv(root/ARRAY,index=False)
        arr=[]
        for strain in ['PAO1','J215']:
            for j,g in enumerate(genes[1:]):
                arr.append({'GeneID':g,'strain':strain,
                            'contrast_id':f'GSE68534_{strain}_delta_anr_minus_WT',
                            'delta_anr_minus_WT_log2norm':((-1)**j)*(.2+j/110)})
        pd.DataFrame(arr).to_csv(root/KO,index=False)
        (root/KO_QC).write_text(json.dumps({'n_source_mapped_genes':119,
                                              'sample_count_QC_max_relative_difference':0.0}))
        r=[]
        for j in range(17):
            r.append({'regulator_GeneID':'PA1544','target_GeneID':genes[j+2],
                      'raw_mode_of_regulation':'+' if j%3 else '-',
                      'original_strain_includes_PAO1':'True','source_excel_row':j+4,
                      'original_PMID':'18974177'})
        pd.DataFrame(r).to_csv(root/REG,index=False)
        (root/REG_QC).write_text(json.dumps({'mapped_09e4':17}))
        out=root/OUT
        run(root,out,strict=False)
        need((out/'SUMMARY_SIGN_PREDICTION_BASELINE_COMPARISONS.csv').is_file(), 'missing output metrics')
        m=pd.read_csv(out/'SUMMARY_SIGN_PREDICTION_BASELINE_COMPARISONS.csv')
        need(len(m)==2*7*4 and set(m.strain)=={'PAO1','J215'}, 'missing test panels')
        try:
            run(root,out,strict=False)
            raise AssertionError('repeated run overwrote frozen results')
        except RuntimeError as e:
            need('overwrite' in str(e), 'wrong repeated-run failure: ' + str(e))
        print('SELF_TEST_PASSED: independent array-to-RNAseq studies, literature prior, source-only')
        print('  panels, sign-majority baselines, locus-neighborhood scoring, provenance, no-overwrite.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        run(ROOT,ROOT/OUT,strict=True)

if __name__=='__main__':
    main()
