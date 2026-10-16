#!/usr/bin/env python3
"""CF-State Phase 8 / Audit 08m: oxygen and CFBE Anr gene-response audit.

Uses the already downloaded GSE6741 and GSE67006 GEO series matrices and
GPL84 probe-to-PAO1 mapping frozen in Audit 08l2. Performs within-study
comparisons only; NEVER applies the RNA-seq iModulon pseudoinverse to array data.

Outputs: array-QC, gene-level within-study changes, predeclared Anr marker panel,
Anr iModulon top-loading gene-response checks, and cross-study concordance.

Important: GSE6741 uses positive Affymetrix signal intensities (log2(1+x)
transformation), whereas GSE67006 uses RMA log2 signals (no additional log).
GSE6741 anoxia also has 100 mM nitrate; treat it as a separate perturbation,
not a pure member of the oxygen-dose series.

Run from CFstate project root:
  python src/phase8/audit_08m_GPL84_oxygen_CFBE_Anr_gene_response.py --self-test
  python -u src/phase8/audit_08m_GPL84_oxygen_CFBE_Anr_gene_response.py
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P8 = ROOT / 'results/phase8'
PRE = P8 / 'audit_08l2_public_oxygen_CFBE_array_preflight'
A08J = P8 / 'audit_08j_GSE68534_frozen_Anr_regulator_response'
OUT = P8 / 'audit_08m_GPL84_oxygen_CFBE_Anr_gene_response'
RAW = ROOT / 'data/raw/phase8_public_oxygen_CFBE_arrays'
MASK = ROOT / 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'
MODULOME = ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json'
STABLE = ROOT / 'results/phase4/audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv'

GROUPS = {
    'GSE6741': {
        'anoxic_100mM_nitrate': ['GSM155231','GSM155239'],
        'O2_0p4': ['GSM155240','GSM155241'],
        'O2_2': ['GSM155242','GSM155243'],
        'O2_20': ['GSM155244','GSM155245'],
    },
    'GSE67006': {
        'WT_CFBE': ['GSM1635895','GSM1635896','GSM1635897'],
        'delta_anr_CFBE': ['GSM1635898','GSM1635899','GSM1635900'],
    },
}
CONTRASTS = {
    'GSE6741': [
        ('O2_2_minus_20','O2_2','O2_20'),
        ('O2_0p4_minus_20','O2_0p4','O2_20'),
        ('O2_0p4_minus_2','O2_0p4','O2_2'),
        ('anoxic_nitrate_minus_20','anoxic_100mM_nitrate','O2_20'),
    ],
    'GSE67006': [('delta_anr_minus_WT_CFBE','delta_anr_CFBE','WT_CFBE')],
}
# Declared before observing results; these loci were identified in Audit 08j.
MARKERS = {
    'anr':'PA1544', 'narG':'PA3875', 'narH':'PA3874',
    'nirS':'PA0519', 'nirM':'PA0518', 'norB':'PA0524',
    'nosZ':'PA3392', 'hemN':'PA1546',
}
RESPIRATORY = ['narG','narH','nirS','nirM','norB','nosZ','hemN']
TOP_LOADING_N = 40


def check(label, condition):
    if not bool(condition):
        raise RuntimeError('CHECK FAILED: '+label)
    print(label+' PASS',flush=True)


def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def parse_geo_matrix_gz(path):
    """Return (GSM columns, probe => numerical sample values); fail on duplicate probes."""
    samples=None
    values={}
    inside=False
    with gzip.open(path,'rt',encoding='utf-8-sig',errors='replace',newline='') as stream:
        for line in stream:
            line=line.rstrip('\r\n')
            if line.startswith('!series_matrix_table_begin'):
                inside=True
                continue
            if line.startswith('!series_matrix_table_end'):
                break
            if not inside:
                continue
            row=next(csv.reader([line],delimiter='\t',quotechar='"'))
            if samples is None:
                samples=[x.strip() for x in row[1:]]
                continue
            if len(row)!=len(samples)+1:
                raise RuntimeError(f'Bad GEO row width in {path}: {row[:3]}')
            probe=row[0].strip()
            if probe in values:
                raise RuntimeError(f'Duplicate GEO probe ID: {probe}')
            try:
                array=np.array([float(x) for x in row[1:]],dtype=float)
            except ValueError:
                array=np.full(len(samples),np.nan)
            values[probe]=array
    if samples is None:
        raise RuntimeError(f'Missing GEO matrix table in {path}')
    return samples, values


def as_gene_matrix(samples, probe_values, mapping):
    d={}
    invalid=[]
    for probe, gene in mapping.items():
        if probe not in probe_values:
            continue
        arr=probe_values[probe]
        if not np.isfinite(arr).all() or (arr < 0).any():
            invalid.append((probe,gene))
            continue
        if gene in d:
            raise RuntimeError('Unexpected duplicated unambiguous PAO1 locus: '+gene)
        d[gene]=arr
    return d, invalid


def transform_array(data, dataset):
    """GSE6741 MAS5-like raw signal; GSE67006 already log2 RMA."""
    data=np.asarray(data,dtype=float)
    if dataset=='GSE6741':
        if not (np.median(data)>10 and np.percentile(data,90)>30):
            raise RuntimeError('GSE6741 value scale unexpected: inspect source before transformation')
        return np.log2(1.+data),'log2(1+GEO_signal_intensity)'
    if dataset=='GSE67006':
        if not (2<np.median(data)<15 and np.percentile(data,90)<25):
            raise RuntimeError('GSE67006 not evidently log2 RMA: inspect source before transformation')
        return data.copy(),'GEO_RMA_log2_signal_unchanged'
    raise ValueError(dataset)


def pair_cosines(v1,v2):
    x=np.asarray(v1,dtype=float); y=np.asarray(v2,dtype=float)
    denom=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/denom) if denom>1e-15 else float('nan')


def spearman(a,b):
    a=pd.Series(a).rank().to_numpy(dtype=float)
    b=pd.Series(b).rank().to_numpy(dtype=float)
    return float(np.corrcoef(a,b)[0,1])


def group_effects(X,genes,samples,group_dict,contrasts):
    ix={v:i for i,v in enumerate(samples)}
    out={}
    rows=[]
    for name,treat,reference in contrasts:
        ii=[ix[g] for g in group_dict[treat]]
        jj=[ix[g] for g in group_dict[reference]]
        vec=X[:,ii].mean(axis=1)-X[:,jj].mean(axis=1)
        out[name]=dict(zip(genes,vec))
        for gene,value in zip(genes,vec):
            rows.append({'contrast':name,'GeneID':gene,'gene_signal_change':float(value),
                         'n_treatment':len(ii),'n_reference':len(jj)})
    return out,pd.DataFrame(rows)


def sample_qc(X, genes, samples, dataset, group_dict, transform_name):
    group={gsm:k for k,ids in group_dict.items() for gsm in ids}
    if set(group)!=set(samples):
        raise RuntimeError('Sample/group mismatch for '+dataset)
    cor=np.corrcoef(X.T)
    rows=[]
    for gsm in samples:
        k=group[gsm]; mates=[x for x in group_dict[k] if x!=gsm]
        ix=samples.index(gsm)
        rows.append({'dataset':dataset,'GSM':gsm,'group':k,'transformation':transform_name,
                     'median_log2_signal':float(np.median(X[:,ix])),
                     'replicate_min_correlation':float(min(cor[ix,samples.index(other)] for other in mates)),
                     'n_genes':len(genes)})
    return pd.DataFrame(rows)


def load_modulome(path, genes, names):
    obj=json.loads(path.read_text())
    if not isinstance(obj,dict) or 'M' not in obj:
        raise RuntimeError('Frozen modulome JSON lacks M')
    data=obj['M']
    if isinstance(data,str): data=json.loads(data)
    if isinstance(data,dict) and {'index','columns','data'}.issubset(data):
        frame=pd.DataFrame(data['data'],index=data['index'],columns=data['columns'])
    else:
        frame=pd.DataFrame(data)
    frame.index=frame.index.astype(str)
    frame.columns=frame.columns.astype(str)
    if set(genes).issubset(frame.columns) and set(names).issubset(frame.index):
        frame=frame.T
    check('modulome_gene_programme_identifiers',
          frame.index.is_unique and frame.columns.is_unique and
          set(genes).issubset(frame.index) and set(names).issubset(frame.columns))
    M=frame.loc[genes,names].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    check('modulome_finite',np.isfinite(M).all())
    return M


def signed_loading_checks(M, full_genes, programmes, contrasts, contrast_groups, sample_by_study, n=TOP_LOADING_N):
    rows=[]; members=[]
    for j,name in enumerate(programmes):
        col=M[:,j]
        # Fix reference membership BEFORE observing array-expression contrasts.
        order=np.argsort(-np.abs(col),kind='stable')[:n]
        selected=[(full_genes[i],col[i]) for i in order if abs(col[i])>1e-12]
        for rank,(gene,loading) in enumerate(selected,start=1):
            members.append({'imodulon':name,'rank_abs_loading':rank,'GeneID':gene,'frozen_loading':float(loading)})
        for dataset, cname in contrast_groups:
            effects=contrasts[(dataset,cname)]
            present=[(gene,l, effects[gene]) for gene,l in selected if gene in effects]
            if len(present)<int(0.8*len(selected)):
                raise RuntimeError(f'Insufficient frozen gene-set overlap: {name}, {dataset}, {cname}')
            load=np.array([t[1] for t in present],float)
            eff=np.array([t[2] for t in present],float)
            signed=float(np.dot(load,eff)/np.sum(np.abs(load)))
            ranks=float(spearman(load,eff)) if np.std(load)>1e-12 and np.std(eff)>1e-12 else float('nan')
            rows.append({'dataset':dataset,'contrast':cname,'imodulon':name,
                         'top_reference_genes':len(selected),'available_top_genes':len(present),
                         'signed_loading_weighted_response':signed,
                         'loading_effect_spearman':ranks,
                         'measurement':'within-study array contrast; not iModulon activity'})
    return pd.DataFrame(rows),pd.DataFrame(members)


def self_test():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'demo.txt.gz'
        content='!Sample_title\t"A"\t"B"\n!series_matrix_table_begin\n"ID_REF"\t"A"\t"B"\n"probe1"\t"99"\t"84"\n"probe2"\t"80"\t"62"\n!series_matrix_table_end\n'
        with gzip.open(p,'wt') as f:f.write(content)
        sample,probes=parse_geo_matrix_gz(p)
        assert sample==['A','B'] and len(probes)==2
        vals,invalid=as_gene_matrix(sample,probes,{'probe1':'PA0001','probe2':'PA0002'})
        assert len(vals)==2 and not invalid
        raw=np.stack([vals['PA0001'],vals['PA0002']])
        transformed,_=transform_array(raw,'GSE6741')
        assert abs(transformed[0,0]-np.log2(100))<1e-9
        g={'x':['A'],'y':['B']}
        effects,df=group_effects(transformed,['PA0001','PA0002'],sample,g,[('x_minus_y','x','y')])
        assert len(df)==2 and effects['x_minus_y']['PA0001']>0
        assert abs(pair_cosines([1,0],[0,1]))<1e-10
        assert abs(spearman([1,2,3],[3,2,1])+1)<1e-10
        cfbe=np.array([[4.,4.1],[10.,12.],[6.,7.]])
        x,n=transform_array(cfbe,'GSE67006')
        assert n=='GEO_RMA_log2_signal_unchanged' and np.array_equal(x,cfbe)
    print('AUDIT_08M_SELF_TEST_PASS')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test(); return
    print('='*78)
    print('CF-State Phase 8 — Audit 08m: controlled oxygen + CFBE Anr gene responses')
    print('GENE-LEVEL ARRAYS ONLY / NO PSEUDOINVERSE / NO CF-SPECIFICITY CLAIM')
    print('='*78,flush=True)
    check('project_CFstate_root',MASK.is_file() and P8.is_dir())
    check('08l2_completed',(PRE/'FROZEN_two_study_GEO_sample_manifest.csv').is_file())
    check('08l2_provenance',(PRE/'PUBLIC_INPUT_PROVENANCE.json').is_file())
    check('08j_prior_gene_marker_file',(A08J/'DESCRIPTIVE_predefined_Anr_related_gene_markers.csv').is_file())
    check('frozen_modulome',MODULOME.is_file())
    check('frozen_stable',STABLE.is_file())
    check('new_output_directory',not OUT.exists())
    OUT.mkdir(parents=True)
    try:
        mask=pd.read_csv(MASK)['GeneID'].astype(str).tolist()
        check('frozen_4923_gene_mask',len(mask)==4923 and len(set(mask))==4923)
        provenance=json.loads((PRE/'PUBLIC_INPUT_PROVENANCE.json').read_text())
        used_files=[MASK, MODULOME, STABLE, PRE/'PUBLIC_INPUT_PROVENANCE.json',
                    PRE/'FROZEN_two_study_GEO_sample_manifest.csv',
                    A08J/'DESCRIPTIVE_predefined_Anr_related_gene_markers.csv']
        changes={}; raw_changes={}; gene_sets={}; groups_by_study={}
        qc_tables=[]; gene_effect_tables=[]; all_markers=[]
        for dataset in ('GSE6741','GSE67006'):
            raw_path=RAW/f'{dataset}_series_matrix.txt.gz'
            map_path=PRE/f'DIAGNOSTIC_{dataset}_probe_to_PA_locus.csv'
            check('input_'+dataset+'_series_matrix',raw_path.is_file())
            check('input_'+dataset+'_frozen_probe_map',map_path.is_file())
            used_files.extend([raw_path,map_path])
            expected_hash=provenance[dataset]['compressed_sha256']
            check(dataset+'_sha256_matches_08l2',file_sha(raw_path)==expected_hash)
            mapping=pd.read_csv(map_path,dtype=str)
            check(dataset+'_mapping_schema',set(('probe','PAO1_gene')).issubset(mapping.columns))
            check(dataset+'_unambiguous_probe_map',mapping['probe'].is_unique and mapping['PAO1_gene'].is_unique)
            mapping_dict=dict(zip(mapping['probe'],mapping['PAO1_gene']))
            samples,probes=parse_geo_matrix_gz(raw_path)
            required=[k for ids in GROUPS[dataset].values() for k in ids]
            check(dataset+'_sample_accessions_match',len(samples)==len(required) and set(samples)==set(required))
            gene_data,invalid=as_gene_matrix(samples,probes,mapping_dict)
            usable=[g for g in mask if g in gene_data]
            check(dataset+'_at_least_4400_frozen_genes',len(usable)>=4400)
            X_raw=np.stack([gene_data[g] for g in usable])
            X,transform_name=transform_array(X_raw,dataset)
            check(dataset+'_transformed_finite',np.isfinite(X).all())
            print('\n'+dataset+' DATA SCALE: raw_median',float(np.median(X_raw)),
                  'transformed_median',float(np.median(X)),
                  'genes',len(usable),'invalid_mapping_rows',len(invalid),flush=True)
            primary,part=group_effects(X,usable,samples,GROUPS[dataset],CONTRASTS[dataset])
            centered=X-np.median(X,axis=0,keepdims=True)
            sensitivity,_=group_effects(centered,usable,samples,GROUPS[dataset],CONTRASTS[dataset])
            for cname in primary:
                changes[(dataset,cname)]=primary[cname]
                raw_changes[(dataset,cname)]=sensitivity[cname]
            part.insert(0,'dataset',dataset)
            gene_effect_tables.append(part)
            sample_q=sample_qc(X,usable,samples,dataset,GROUPS[dataset],transform_name)
            check(dataset+'_replicate_correlations_ge_0p80',float(sample_q['replicate_min_correlation'].min())>=0.80)
            qc_tables.append(sample_q)
            gene_sets[dataset]=set(usable)
            groups_by_study[dataset]=CONTRASTS[dataset]
        all_effects=pd.concat(gene_effect_tables,ignore_index=True)
        all_effects.to_csv(OUT/'DESCRIPTIVE_all_five_array_gene_contrasts.csv',index=False)
        pd.concat(qc_tables,ignore_index=True).to_csv(OUT/'FROZEN_GPL84_14_sample_QC.csv',index=False)
        common=[g for g in mask if all(g in gene_sets[d] for d in ('GSE6741','GSE67006'))]
        check('cross_study_common_genes_at_least_4400',len(common)>=4400)
        print('\n========== PUBLIC GPL84 ARRAY SAMPLE QC ==========',flush=True)
        for q in qc_tables:
            print(q[['dataset','GSM','group','n_genes','replicate_min_correlation']]
                  .to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
        print('Cross-study common unambiguous frozen genes:',len(common),'/4923',flush=True)

        # Compare array datasets at gene-level only, using within-study contrasts.
        comparison_rows=[]
        cname_c='delta_anr_minus_WT_CFBE'
        for oxygen in ('O2_2_minus_20','O2_0p4_minus_20','anoxic_nitrate_minus_20'):
            a=np.array([changes[('GSE6741',oxygen)][g] for g in common])
            b=np.array([changes[('GSE67006',cname_c)][g] for g in common])
            comparison_rows.append({'oxygen_contrast':oxygen,'CFBE_contrast':cname_c,
                                    'shared_genes':len(common),'spearman_gene_changes':spearman(a,b),
                                    'pearson_gene_changes':float(np.corrcoef(a,b)[0,1]),
                                    'cosine_gene_changes':pair_cosines(a,b),
                                    'expected_if_general_Anr_regulation':'opposing expression directions, not universal'})
        pd.DataFrame(comparison_rows).to_csv(OUT/'DESCRIPTIVE_oxygen_vs_CFBE_gene_concordance.csv',index=False)

        # Reference Anr marker effects were calculated independently in Audit 08j.
        j=pd.read_csv(A08J/'DESCRIPTIVE_predefined_Anr_related_gene_markers.csv')
        j=j[(j['gene_symbol'].isin(MARKERS)) & (j['strain'].isin(['PAO1','J215']))]
        ref_j={(r.gene_symbol,r.strain):float(r.delta_anr_minus_WT_log2norm)
               for r in j.itertuples(index=False)
               if pd.notna(r.delta_anr_minus_WT_log2norm)}
        mark=[]
        for symbol,gene in MARKERS.items():
            item={'gene_symbol':symbol,'GeneID':gene}
            for dataset,cname,column in [
                ('GSE6741','O2_2_minus_20','O2_2_minus_20'),
                ('GSE6741','O2_0p4_minus_20','O2_0p4_minus_20'),
                ('GSE6741','anoxic_nitrate_minus_20','anoxic_nitrate_minus_20'),
                ('GSE67006','delta_anr_minus_WT_CFBE','CFBE_delta_anr_minus_WT')]:
                item[column]=changes[(dataset,cname)].get(gene,np.nan)
                item[column+'_median_centered']=raw_changes[(dataset,cname)].get(gene,np.nan)
            item['08j_PAO1_delta_anr_minus_WT']=ref_j.get((symbol,'PAO1'),np.nan)
            item['08j_J215_delta_anr_minus_WT']=ref_j.get((symbol,'J215'),np.nan)
            mark.append(item)
        marker_df=pd.DataFrame(mark)
        marker_df.to_csv(OUT/'DESCRIPTIVE_predefined_Anr_marker_cross_experiment.csv',index=False)
        # Compare direction against prior independent Anr-deletion experiments,
        # excluding anr itself since knockout removes the target by construction.
        concordance_rows=[]
        for refstrain in ('PAO1','J215'):
            usable_markers=marker_df[marker_df['gene_symbol'].isin(RESPIRATORY)].dropna(
                subset=['CFBE_delta_anr_minus_WT',f'08j_{"PAO1" if refstrain=="PAO1" else "J215"}_delta_anr_minus_WT'])
            refcol='08j_PAO1_delta_anr_minus_WT' if refstrain=='PAO1' else '08j_J215_delta_anr_minus_WT'
            a=usable_markers['CFBE_delta_anr_minus_WT'].to_numpy(float)
            b=usable_markers[refcol].to_numpy(float)
            concordance_rows.append({'reference_strain':refstrain,'comparison':'CFBE_delta_anr_vs_08j_delta_anr',
                                     'n_predeclared_respiratory_genes':len(a),
                                     'n_matching_effect_sign':int((np.sign(a)==np.sign(b)).sum()),
                                     'fraction_matching_sign':float((np.sign(a)==np.sign(b)).mean()) if len(a) else np.nan,
                                     'pearson':float(np.corrcoef(a,b)[0,1]) if len(a)>=3 else np.nan})
        pd.DataFrame(concordance_rows).to_csv(OUT/'DESCRIPTIVE_Anr_marker_directional_replication.csv',index=False)
        print('\n========== PREDECLARED ANR RESPIRATORY MARKER PANEL ==========',flush=True)
        print(marker_df[['gene_symbol','GeneID','O2_2_minus_20','O2_0p4_minus_20',
                         'anoxic_nitrate_minus_20','CFBE_delta_anr_minus_WT',
                         '08j_PAO1_delta_anr_minus_WT','08j_J215_delta_anr_minus_WT']]
              .to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
        print('\n========== ANR MARKER DIRECTIONAL REPLICATION ==========',flush=True)
        print(pd.DataFrame(concordance_rows).to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)

        stable=pd.read_csv(STABLE)['imodulon'].astype(str).tolist()
        anr_programmes=[p for p in stable if re.fullmatch(r'Anr(?:-\d+)?',p,re.I)]
        controls=[p for p in ('Zur-2','PvdS','ExsB','RiPP') if p in stable]
        programmes=anr_programmes+controls
        check('four_Anr_programmes_in_frozen_representation',len(anr_programmes)==4)
        M=load_modulome(MODULOME,mask,programmes)
        contrast_names=[(d,c) for d in ('GSE6741','GSE67006') for c,_,_ in CONTRASTS[d]]
        prog,members=signed_loading_checks(M,mask,programmes,changes,contrast_names,None)
        prog.to_csv(OUT/'DESCRIPTIVE_frozen_iModulon_top40_gene_loading_responses.csv',index=False)
        members.to_csv(OUT/'FROZEN_Anr_and_control_top40_gene_membership.csv',index=False)
        print('\n========== FROZEN ANR PROGRAMMES: ARRAY GENE-LOADING RESPONSE ==========',flush=True)
        print(prog[prog['imodulon'].isin(anr_programmes)][[
              'contrast','imodulon','available_top_genes','signed_loading_weighted_response','loading_effect_spearman']]
              .to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
        print('\n========== CROSS-STUDY GENE-LEVEL CONCORDANCE ==========',flush=True)
        print(pd.DataFrame(comparison_rows).to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
        (OUT/'INTERPRETATION_LIMITS.txt').write_text(
            'GSE6741 positive GEO microarray signals converted with log2(1+x); GSE67006 GEO RMA already log2.\n'
            'All contrasts are within the SAME microarray study, not comparable in absolute scale across studies.\n'
            'Original log2-scale and sample-median-centred contrasts saved for sensitivity in Anr marker CSV.\n'
            'GSE6741 0% oxygen also includes 100 mM nitrate; do not treat as a pure oxygen dose.\n'
            'GSE67006 CFBE experiment is Anr knockout, not CF-vs-non-CF host comparison.\n'
            'Frozen Anr gene-loading scores use only 40 preselected strongest frozen membership genes;\n'
            'they are NOT standard RNA-seq-calibrated iModulon activities or clinical CF-State scores.\n'
            'Small n (two replicates per oxygen condition; three per CFBE group). All results descriptive.\n'
            'Training atlas overlap and platform-related bias are not fully audited.\n'
            'These tests cannot establish a CF-specific oxygen or Anr phenotype.\n'
        )
        (OUT/'INPUT_SHA256.json').write_text(json.dumps(
            [{'path':str(p),'sha256':file_sha(p)} for p in used_files],indent=2)+'\n')
        print('\nSTATUS: CFSTATE_PHASE8_08M_GPL84_OXYGEN_CFBE_ANR_GENE_VALIDATION_DESCRIPTIVE_COMPLETE')
    except Exception:
        print('STATUS: CFSTATE_PHASE8_08M_INCOMPLETE_DIAGNOSTICS_PRESERVED',file=sys.stderr)
        raise


if __name__=='__main__':
    main()
