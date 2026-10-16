#!/usr/bin/env python3
"""CF-State Audit 09j: reconstruct GSE68534 RNA-seq gene-level Δanr contrasts.

Reads the original published 8-sample processed count workbook, frozen 08j
feature-to-PAO1 gene map and frozen within-strain size factors. Verifies
pre-existing 08j marker effects before writing a new independent-study ledger.

NO network training, no mixing array/RNA-seq effect magnitudes, no overwrite.

Run:
 python src/phase9/audit_09j_GSE68534_gene_level_delta_anr_benchmark_freeze.py --self-test
 python -u src/phase9/audit_09j_GSE68534_gene_level_delta_anr_benchmark_freeze.py
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path('data/raw/phase8_public_anr_processed/GSE68534/GSE68534_ProcessedDataMatrices.xlsx.gz')
OLD = Path('results/phase8/audit_08j_GSE68534_frozen_Anr_regulator_response')
MAP = OLD / 'DIAGNOSTIC_source_gene_mapping.csv'
QC = OLD / 'FROZEN_GSE68534_8_sample_QC.csv'
MARKERS = OLD / 'DESCRIPTIVE_predefined_Anr_related_gene_markers.csv'
ARRAY = Path('results/phase9/audit_09a_perturbation_evidence_preflight/FROZEN_08m_native_array_gene_response_evidence.csv')
OUT = Path('results/phase9/audit_09j_GSE68534_gene_level_delta_anr_benchmark_freeze')
SAMPLES = [
    ('PAO1', 'WT', 1, 'PAO1_WT_1'),
    ('PAO1', 'WT', 2, 'PAO1_WT_2'),
    ('PAO1', 'delta_anr', 1, 'PAO1_∆anr_1'),
    ('PAO1', 'delta_anr', 2, 'PAO1_∆anr_2'),
    ('J215', 'WT', 1, 'J215_WT_1'),
    ('J215', 'WT', 2, 'J215_WT_2'),
    ('J215', 'delta_anr', 1, 'J215_∆anr_1'),
    ('J215', 'delta_anr', 2, 'J215_∆anr_2'),
]
SAMPLE_NAMES = [q[3] for q in SAMPLES]


def need(ok, message):
    if not bool(ok):
        raise RuntimeError('AUDIT_09J_FAILED: ' + str(message))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def fmt_num(v, decimals=5):
    return 'nan' if not np.isfinite(v) else f'{v:.{decimals}f}'


def read_original_xlsx_gz(path):
    need(path.is_file(), 'missing original GSE68534 workbook: ' + str(path))
    with gzip.open(path, 'rb') as f:
        contents = f.read()
    try:
        xl = pd.ExcelFile(io.BytesIO(contents))
    except Exception as e:
        raise RuntimeError('AUDIT_09J_FAILED: invalid xlsx inside gzip: ' + str(e)) from e
    required_sheets = {'Raw Read Values', 'Quantile Normalized Read Values', 'Fold Changes and Significance'}
    need(required_sheets.issubset(set(xl.sheet_names)), 'unexpected source sheet names: ' + str(xl.sheet_names))
    raw = pd.read_excel(xl, sheet_name='Raw Read Values', header=None, dtype=object)
    need(raw.shape[0] > 10 and raw.shape[1] >= 9, 'unexpected raw read-value shape ' + str(raw.shape))
    names = [str(raw.iat[0, c]).strip() for c in range(1, 9)]
    need(names == SAMPLE_NAMES, 'sample order/name mismatch: ' + str(names))
    need(str(raw.iat[1, 0]).strip() == 'Feature ID', 'missing second-row Feature ID label')
    for col in range(1, 9):
        need('gene reads' in str(raw.iat[1,col]).lower(), 'not a raw gene-read column: ' + SAMPLE_NAMES[col-1])
    return raw, xl.sheet_names


def as_numeric_matrix(raw, map_df):
    # IMPORTANT: original 08j mapping stores 1-based EXCEL ROW indices.
    # It is safer than attempting gene-symbol remapping in a new pipeline.
    ref_rows = pd.to_numeric(map_df.excel_row, errors='coerce')
    need(ref_rows.notna().all(), 'non-numeric frozen mapping excel_row')
    ref_rows = ref_rows.astype(int).to_numpy()
    need(np.all((ref_rows >= 3) & (ref_rows <= len(raw))), 'mapped row outside original worksheet')
    need(len(np.unique(ref_rows)) == len(ref_rows), 'mapping includes duplicate Excel rows')
    raw_features = raw.iloc[ref_rows - 1, 0].fillna('').astype(str).str.strip().to_numpy()
    mapped_features = map_df.raw_feature_id.fillna('').astype(str).str.strip().to_numpy()
    need(np.array_equal(raw_features, mapped_features),
         'frozen 08j raw_feature_id != workbook feature ID at recorded Excel row; first differences: '
         + str([(int(ref_rows[i]), mapped_features[i], raw_features[i])
                for i in np.flatnonzero(raw_features != mapped_features)[:8]]))
    sub = raw.iloc[ref_rows - 1, 1:9]
    val = sub.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    need(np.isfinite(val).all(), 'mapped raw count matrix has NaN/non-finite')
    need((val >= 0).all(), 'negative mapped raw counts')
    need(np.max(abs(val - np.round(val))) < 1e-6, 'raw read matrix has non-integer counts')
    ids = map_df.GeneID.astype(str).to_numpy()
    return pd.DataFrame(val, index=pd.Index(ids, name='GeneID'), columns=SAMPLE_NAMES)


def validate_frozen_08j_qc(counts, qc, strict):
    want = {'strain','genotype','replicate','assigned_counts_usable_genes',
            'within_strain_size_factor','detected_genes_GE1'}
    need(want.issubset(qc.columns), '08j QC required fields missing: ' + str(want - set(qc.columns)))
    need(len(qc) == 8, 'frozen 08j QC does not have 8 rows')
    rows = []
    size_factors = {}
    for strain, genotype, rep, key in SAMPLES:
        q = qc.loc[(qc.strain.astype(str) == strain)
                   & (qc.genotype.astype(str) == genotype)
                   & (pd.to_numeric(qc.replicate,errors='raise') == rep)]
        need(len(q) == 1, 'missing/duplicated frozen sample QC: ' + key)
        rr = q.iloc[0]
        sf = float(rr.within_strain_size_factor)
        need(np.isfinite(sf) and sf > 0, 'invalid sample size factor: ' + key)
        size_factors[key] = sf
        total = float(counts[key].sum())
        detected = int((counts[key] >= 1).sum())
        frozen_total = float(rr.assigned_counts_usable_genes)
        frozen_detected = int(rr.detected_genes_GE1)
        rel = abs(total - frozen_total) / max(1.,frozen_total)
        rows.append(dict(sample=key,strain=strain,genotype=genotype,replicate=rep,
                         frozen_08j_size_factor=sf,raw_mapped_sum=total,
                         frozen_08j_usable_sum=frozen_total,relative_sum_difference=rel,
                         raw_mapped_detected_GE1=detected,frozen_08j_detected_GE1=frozen_detected,
                         detected_difference=detected - frozen_detected))
    result = pd.DataFrame(rows)
    # Some 08j source rows may have been excluded after summing the source's
    # usable read rows. Report this transparently; markers are the stricter
    # validation of the actual response reconstruction.
    if strict:
        need(result.relative_sum_difference.max() < 0.05,
             '08j read-sum mismatch >5%: inspect QC table; do NOT silently change normalization')
    return result, size_factors


def make_contrasts(counts, size_factors):
    logdata = pd.DataFrame({key: np.log2(1 + counts[key] / size_factors[key]) for key in SAMPLE_NAMES})
    logdata.index.name = 'GeneID'
    effects = []
    for strain in ['PAO1','J215']:
        wt = [f'{strain}_WT_{i}' for i in [1,2]]
        ko = [f'{strain}_∆anr_{i}' for i in [1,2]]
        out = pd.DataFrame({
            'GeneID': logdata.index.to_numpy(), 'study_id': 'GSE68534',
            'strain': strain, 'contrast_id': f'GSE68534_{strain}_delta_anr_minus_WT',
            'perturbation': 'knockout_anr',
            'n_WT_biological_replicates': 2, 'n_KO_biological_replicates': 2,
            'WT_mean_log2_sizefactor_normalized_count_plus1': logdata[wt].mean(axis=1).to_numpy(),
            'KO_mean_log2_sizefactor_normalized_count_plus1': logdata[ko].mean(axis=1).to_numpy(),
            'WT_replicate_sd_log2': logdata[wt].std(axis=1,ddof=1).to_numpy(),
            'KO_replicate_sd_log2': logdata[ko].std(axis=1,ddof=1).to_numpy(),
            'delta_anr_minus_WT_log2norm': (logdata[ko].mean(axis=1)-logdata[wt].mean(axis=1)).to_numpy(),
            'evidence_kind': 'within_strain_knockout_expression_response_NOT_TF_binding',
            'effect_unit': 'mean_of_sample_log2(raw_reads/frozen_08j_within_strain_sizefactor+1)',
            'source_08j_mapping_reused': True,
        })
        out['effect_sign'] = np.where(out.delta_anr_minus_WT_log2norm > 0,'UP',
                                       np.where(out.delta_anr_minus_WT_log2norm < 0,'DOWN','ZERO'))
        effects.append(out)
    return logdata, pd.concat(effects, ignore_index=True)


def validate_markers(effects, markers, strict):
    need({'PAO1_locus','strain','status','delta_anr_minus_WT_log2norm'}.issubset(markers.columns),
         'frozen 08j marker schema changed')
    selected = markers.loc[markers.status.eq('descriptive_two_replicates')
                           & markers.strain.isin(['PAO1','J215'])].copy()
    need(not selected.duplicated(['PAO1_locus','strain']).any(), 'duplicate frozen marker/strain')
    ck = selected.merge(effects[['GeneID','strain','delta_anr_minus_WT_log2norm']],
                        how='left',left_on=['PAO1_locus','strain'],right_on=['GeneID','strain'],
                        suffixes=('_frozen08j','_reconstructed'), validate='one_to_one')
    need(ck['GeneID'].notna().all(), '08j frozen marker absent from reconstructed matrix')
    a = pd.to_numeric(ck.delta_anr_minus_WT_log2norm_frozen08j,errors='raise')
    b = pd.to_numeric(ck.delta_anr_minus_WT_log2norm_reconstructed,errors='raise')
    need(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite 08j marker effects')
    ck['absolute_reconstruction_difference'] = (a-b).abs()
    maxdiff = float(ck.absolute_reconstruction_difference.max()) if len(ck) else float('nan')
    if strict:
        need(len(ck) >= 16, 'fewer than 16 frozen marker-by-strain verifications')
        need(maxdiff <= 0.005,
             f'08j marker replication FAILED max absolute difference {maxdiff:.5f} >0.005; '
             'inspect both effects; do not freeze a different normalization')
    return ck, maxdiff


def sign_agreement(a,b):
    idx = (a!=0) & (b!=0) & np.isfinite(a) & np.isfinite(b)
    return (int(idx.sum()), float(np.mean(np.sign(a[idx]) == np.sign(b[idx]))) if idx.any() else float('nan'))


def study_concordance(effects, root, strict):
    piv = effects.pivot(index='GeneID',columns='strain',values='delta_anr_minus_WT_log2norm')
    need(not piv.isna().any().any() and set(piv.columns)=={'PAO1','J215'}, 'missing strain contrasts')
    mask = (piv['PAO1']!=0)&(piv['J215']!=0)
    cross = piv.rename(columns={'PAO1':'PAO1_delta_anr_minus_WT','J215':'J215_delta_anr_minus_WT'}).reset_index()
    cross['same_effect_sign'] = np.sign(cross.PAO1_delta_anr_minus_WT) == np.sign(cross.J215_delta_anr_minus_WT)
    cross['abs_effect_difference'] = abs(cross.PAO1_delta_anr_minus_WT - cross.J215_delta_anr_minus_WT)
    stats = []
    ar = root / ARRAY
    if ar.is_file():
        ds = pd.read_csv(ar)
        need({'contrast_id','GeneID','effect_value_native'}.issubset(ds.columns), '09a array schema incompatible')
        a = ds.loc[ds.contrast_id.eq('delta_anr_minus_WT_CFBE'),['GeneID','effect_value_native']].copy()
        need(a.GeneID.is_unique and len(a)>0, 'duplicate or missing GSE67006 array effects')
        a.effect_value_native = pd.to_numeric(a.effect_value_native,errors='raise')
        if strict:
            need(len(a)==4920, '09a expected 4,920 GPL84 CFBE effects')
        tab = cross.merge(a,how='inner',on='GeneID',validate='one_to_one')
        for strain in ['PAO1','J215']:
            x=tab[f'{strain}_delta_anr_minus_WT'].to_numpy(float)
            y=tab.effect_value_native.to_numpy(float)
            n, agree=sign_agreement(x,y)
            stats.append(dict(comparison=f'GSE68534_{strain}_RNAseq_vs_GSE67006_PAO1_CFBE_array',
                              n_shared_genes=len(tab),n_nonzero_both=n,
                              same_effect_sign_fraction=agree,
                              spearman_rho=float(pd.Series(x).corr(pd.Series(y),method='spearman')),
                              inference='DESCRIPTIVE_NONBLIND_cross_study_and_cross_platform_SIGN_AND_RANK_ONLY'))
    a=cross.PAO1_delta_anr_minus_WT.to_numpy(float)
    b=cross.J215_delta_anr_minus_WT.to_numpy(float)
    n,agree=sign_agreement(a,b)
    stats.insert(0,dict(comparison='GSE68534_PAO1_vs_J215_within_study',
                        n_shared_genes=len(cross),n_nonzero_both=n,same_effect_sign_fraction=agree,
                        spearman_rho=float(pd.Series(a).corr(pd.Series(b),method='spearman')),
                        inference='DESCRIPTIVE_cross_background_shared_study_not_independent_study_validation'))
    return cross,pd.DataFrame(stats)


def main_compute(root, output, strict=True):
    paths = [SOURCE,MAP,QC,MARKERS]
    for p in paths:
        need((root/p).is_file(), 'missing required source: ' + str(root/p))
    need(not output.exists(), 'output already exists; never overwrite frozen audit: ' + str(output))
    raw, sheets = read_original_xlsx_gz(root/SOURCE)
    gene_map = pd.read_csv(root/MAP, dtype={'GeneID':str,'raw_feature_id':str})
    need({'GeneID','raw_feature_id','excel_row'}.issubset(gene_map.columns),'08j mapping schema')
    need(gene_map.GeneID.is_unique and gene_map.GeneID.notna().all(),'frozen map has duplicated/null loci')
    if strict:
        need(len(gene_map)==5449,'expected 5449 mapped distinct PAO1 genes from frozen 08j')
    counts=as_numeric_matrix(raw,gene_map)
    sample_qc,size_factors=validate_frozen_08j_qc(counts,pd.read_csv(root/QC),strict)
    logdata,effects=make_contrasts(counts,size_factors)
    checks,maxdiff=validate_markers(effects,pd.read_csv(root/MARKERS),strict)
    cross, concordance=study_concordance(effects,root,strict)
    n=int(len(counts))
    if strict:
        need(n==5449 and len(effects)==2*n, 'unexpected contrast sizes')
    # Print everything relevant BEFORE output is written, so failures remain inspectable.
    print('\n'+'='*83)
    print('CF-STATE AUDIT 09J | GSE68534 FULL GENE-LEVEL DELTA-ANR BENCHMARK')
    print('='*83)
    print('Source workbook:',root/SOURCE)
    print('Sheets:',', '.join(sheets))
    print('Frozen feature mappings:',n,'| sample libraries:',len(SAMPLES),'| strain contrasts:',2)
    print('Orientation: delta_anr MINUS WT in each strain; original FOLD CHANGE sheet NOT used')
    print('Scale: mean sample log2(raw_count / FROZEN within-strain size factor + 1)')
    print('08j marker comparisons:',len(checks),'| max absolute deviation:',fmt_num(maxdiff,7))
    print('\nFROZEN 08J SAMPLE QC RECONSTRUCTION:')
    print(sample_qc[['sample','frozen_08j_size_factor','raw_mapped_sum','frozen_08j_usable_sum',
                     'relative_sum_difference','raw_mapped_detected_GE1','frozen_08j_detected_GE1']].to_string(index=False))
    print('\n08J PREDECLARED MARKER CHECKS (old vs reconstructed):')
    print(checks[['PAO1_locus','strain','delta_anr_minus_WT_log2norm_frozen08j',
                  'delta_anr_minus_WT_log2norm_reconstructed','absolute_reconstruction_difference']]
          .to_string(index=False))
    print('\nRESPONSE DISTRIBUTION BY STRAIN (descriptive; n=2 per group):')
    for strain in ['PAO1','J215']:
        z=effects.loc[effects.strain.eq(strain),'delta_anr_minus_WT_log2norm'].to_numpy(float)
        print(f' {strain}: n_genes={len(z)} increased={(z>0).sum()} decreased={(z<0).sum()} '
              f'median={np.median(z):.4f} |effect|>=1={(abs(z)>=1).sum()}')
    print('\nCROSS-EXPERIMENT/STRAIN RESPONSE CONCORDANCE (DESCRIPTIVE):')
    print(concordance.to_string(index=False))
    print('\nLARGEST-MAGNITUDE GENES PER STRAIN (20 each):')
    for strain in ['PAO1','J215']:
        sub=effects.loc[effects.strain.eq(strain)].copy()
        sub['absolute_effect']=abs(sub.delta_anr_minus_WT_log2norm)
        print('\n',strain)
        print(sub.nlargest(20,'absolute_effect')[['GeneID','delta_anr_minus_WT_log2norm']].to_string(index=False))
    print('\nINTERPRETATION: two replicate libraries/genotype/strain; no reliable gene-level DE inference.')
    print('GSE67006 is GPL84 ARRAY; compare signs/ranks only, not absolute effect magnitudes.')
    print('GSE68534 was used in prior Phase 08: retrospective transfer benchmark, NOT blind validation.')
    print('No network trained; no direct transcriptional binding or CF specificity inferred.')
    # All validation complete: save immutable-by-contract outputs.
    output.mkdir(parents=True,exist_ok=False)
    counts.reset_index().to_csv(output/'FROZEN_GSE68534_5449_genes_8_raw_counts.csv',index=False)
    logdata.reset_index().to_csv(output/'FROZEN_GSE68534_5449_genes_8_log2_normalized.csv',index=False)
    effects.to_csv(output/'FROZEN_GSE68534_PAO1_J215_delta_anr_minus_WT_gene_responses.csv',index=False)
    sample_qc.to_csv(output/'QC_RECONSTRUCTED_vs_FROZEN_08J_sample_counts.csv',index=False)
    checks.to_csv(output/'QC_RECONSTRUCTED_vs_FROZEN_08J_markers.csv',index=False)
    cross.to_csv(output/'DESCRIPTIVE_PAO1_vs_J215_knockout_gene_comparison.csv',index=False)
    concordance.to_csv(output/'DESCRIPTIVE_GSE68534_strain_and_GSE67006_cross_study_concordance.csv',index=False)
    provenance=[]
    for p in paths + ([ARRAY] if (root/ARRAY).is_file() else []):
        provenance.append({'source_path':str(p),'sha256':sha(root/p),'evidence_role':'read-only source'})
    pd.DataFrame(provenance).to_csv(output/'INPUT_SHA256_PROVENANCE.csv',index=False)
    summary={'status':'PHASE09J_GSE68534_GENE_RESPONSE_RECONSTRUCTION_COMPLETE_NO_FITTING',
             'n_source_mapped_genes':n,'n_sample_libraries':8,'n_strain_contrasts':2,
             'n_08j_marker_checks':len(checks),'max_abs_08j_marker_deviation':maxdiff,
             'sample_count_QC_max_relative_difference':float(sample_qc.relative_sum_difference.max()),
             'GSE67006_cross_platform_available':bool((root/ARRAY).is_file()),
             'previous_08j_source_unmodified':True,'network_trained':False}
    (output/'QC_STATUS.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'INTERPRETATION_AND_VALIDATION_LIMITS.md').write_text(
        '# CF-State audit 09j | Original GSE68534 gene-level response evidence\n\n'
        '- Two group contrasts: PAO1 and J215 delta-anr minus their own WT, 2 versus 2 each.\n'
        '- Feature-to-locus alignment uses PREVIOUS frozen 08j `excel_row` positions and must match original workbook identifiers exactly.\n'
        '- Original raw read values; frozen 08j within-strain size factors. Quantile-normalised worksheet and publication\'s signed fold-change convention are NOT inputs to the reconstructed effects.\n'
        '- Each sample is converted to log2(raw count / frozen size factor + 1), then within-genotype replicate means are subtracted.\n'
        '- Pre-existing 08j marker effects checked numerically BEFORE the new files are frozen. Technical replication is not biological validation.\n'
        '- Cross-platform GSE67006 array effects and RNA-seq are compared only by signs/ranks within matched genes, not magnitude or pooled calibration.\n'
        '- GSE68534 and GSE67006 were inspected in Phase 08: cross-study comparison is retrospective and is NOT an untouched test set.\n'
        '- Two biological replicates per cell and adjacent genes/operons limit uncertainty claims. No DE p values, causal TF binding or CF specificity inferred.\n'
        '- This is gene-response DATASET preparation, not fitting or demonstrating transfer of a metal-trained network.\n')
    (output/'AUDIT_09J_STATUS.txt').write_text(summary['status']+'\n')
    print('\nSUCCESS | saved:',output)
    return summary


def make_fixture(root):
    # Fully synthetic xlsx.gz fixture with the same two-line original header.
    rng=np.random.default_rng(113)
    genes=['PA%04d'%i for i in range(1,81)]
    features=['sym%04d'%i for i in range(1,81)]
    x=rng.poisson(lam=100,size=(80,8)).astype(float)
    # explicit easy-to-detect synthetic effects
    x[0,2:4]*=4
    x[1,6:8]=np.rint(x[1,6:8]*0.25)
    rows=[[None]+SAMPLE_NAMES+['Chromosome region start'],
          ['Feature ID']+['Unique gene reads']*8+['Chromosome region start']]
    for i,f in enumerate(features):
        rows.append([f]+x[i,:].tolist()+[i*100])
    src=root/SOURCE;src.parent.mkdir(parents=True,exist_ok=True)
    out=io.BytesIO()
    with pd.ExcelWriter(out,engine='openpyxl') as w:
        pd.DataFrame(rows).to_excel(w,sheet_name='Raw Read Values',header=False,index=False)
        pd.DataFrame(rows).to_excel(w,sheet_name='Quantile Normalized Read Values',header=False,index=False)
        pd.DataFrame(rows).to_excel(w,sheet_name='Fold Changes and Significance',header=False,index=False)
    src.write_bytes(gzip.compress(out.getvalue()))
    m=pd.DataFrame({'GeneID':genes,'raw_feature_id':features,'excel_row':np.arange(3,83)})
    (root/MAP).parent.mkdir(parents=True,exist_ok=True);m.to_csv(root/MAP,index=False)
    qc=[]
    for i,(strain,genotype,rep,key) in enumerate(SAMPLES):
        qc.append({'strain':strain,'genotype':genotype,'replicate':rep,'detected_genes_GE1':int((x[:,i]>=1).sum()),
                   'assigned_counts_usable_genes':float(x[:,i].sum()),'within_strain_size_factor':1.,'median_ratio_ref_genes':75})
    pd.DataFrame(qc).to_csv(root/QC,index=False)
    effects=np.log2(1+x)
    mr=[]
    for j in range(10):
        for strain,lo in [('PAO1',0),('J215',4)]:
            val=float(np.mean(effects[j,[lo+2,lo+3]])-np.mean(effects[j,[lo,lo+1]]))
            mr.append({'gene_symbol':features[j],'PAO1_locus':genes[j], 'strain':strain,
                       'delta_anr_minus_WT_log2norm':val,'status':'descriptive_two_replicates'})
    pd.DataFrame(mr).to_csv(root/MARKERS,index=False)
    aa=pd.DataFrame({'contrast_id':['delta_anr_minus_WT_CFBE']*20,'GeneID':genes[:20],
                     'effect_value_native':rng.normal(size=20)})
    (root/ARRAY).parent.mkdir(parents=True,exist_ok=True);aa.to_csv(root/ARRAY,index=False)


def self_test():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);make_fixture(root)
        out=root/OUT
        summary=main_compute(root,out,strict=False)
        need(summary['n_source_mapped_genes']==80 and summary['max_abs_08j_marker_deviation']<1e-10,
             'synthetic 08j exact reconstruction failed')
        need(len(pd.read_csv(out/'FROZEN_GSE68534_PAO1_J215_delta_anr_minus_WT_gene_responses.csv'))==160,
             'synthetic full response rows missing')
        # Old marker discordance must be caught before accepting the new freeze.
        test_effects = pd.read_csv(out/'FROZEN_GSE68534_PAO1_J215_delta_anr_minus_WT_gene_responses.csv')
        tampered_markers = pd.read_csv(root/MARKERS)
        tampered_markers.loc[0,'delta_anr_minus_WT_log2norm'] += 2.0
        try:
            validate_markers(test_effects,tampered_markers,strict=True)
        except RuntimeError as e:
            need('marker replication FAILED' in str(e),'old-marker discrepancy not checked')
        else:
            raise AssertionError('tampered frozen 08j marker accepted')
        try:
            main_compute(root,out,strict=False)
        except RuntimeError as e:
            need('already exists' in str(e),'expected no-overwrite failure')
        else:
            raise AssertionError('existing outputs overwritten')
        # Wrong recorded mapping position must fail rather than aligning by a similar gene name.
        m=pd.read_csv(root/MAP);m.loc[0,'excel_row']=4;m.to_csv(root/MAP,index=False)
        try:
            main_compute(root,root/'new_out',strict=False)
        except RuntimeError as e:
            need('duplicate Excel rows' in str(e) or 'feature ID' in str(e),
                 'wrong row position not detected')
        else:
            raise AssertionError('tampered 08j mapping accepted')
    print('SELF_TEST_PASSED: gzip Excel parser; strict frozen Excel-row identity; eight samples;')
    print('  frozen size-factor contrasts; marker replication; descriptive cross-study ledger;')
    print('  source provenance, no overwrite and tampered source mapping rejected.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--root',default=str(ROOT))
    args=p.parse_args()
    if args.self_test:
        self_test();return
    root=Path(args.root).expanduser().resolve()
    main_compute(root,root/OUT,strict=True)


if __name__=='__main__':
    main()
