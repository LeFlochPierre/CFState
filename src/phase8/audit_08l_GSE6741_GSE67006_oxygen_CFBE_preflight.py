#!/usr/bin/env python3
"""CF-State Phase 8 Audit 08l: public oxygen and CFBE Anr microarray preflight.

Read-only check of two independent experimental designs on platform GPL84:
  * GSE6741: PAO1 at 20%, 2%, 0.4% O2 or anoxia with nitrate (two replicates).
  * GSE67006: PAO1 WT and delta-anr on CFBE cells (three replicates per group).

Only downloads public GEO SERIES MATRIX and GPL84 platform annotation. Resolves
unique PAO1 locus IDs and reports probe coverage and sample metadata; it does
NOT project array intensities into RNA-seq-calibrated iModulons or fit anything.

Run from ~/projects/CFstate:
 python src/phase8/audit_08l_GSE6741_GSE67006_oxygen_CFBE_preflight.py --self-test
 python -u src/phase8/audit_08l_GSE6741_GSE67006_oxygen_CFBE_preflight.py
"""
from __future__ import annotations
import argparse
import csv
import gzip
import hashlib
import io
import json
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT=Path.cwd().resolve()
OUTPUT=ROOT/'results/phase8/audit_08l_public_oxygen_CFBE_array_preflight'
RAW=ROOT/'data/raw/phase8_public_oxygen_CFBE_arrays'
MASK=ROOT/'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'
SERIES={
 'GSE6741': {
  'url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE6nnn/GSE6741/matrix/GSE6741_series_matrix.txt.gz',
  'n':8,
  'purpose':'Controlled oxygen availability in PAO1; 20%, 2%, 0.4% O2 and 0% plus nitrate'
 },
 'GSE67006': {
  'url':'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE67nnn/GSE67006/matrix/GSE67006_series_matrix.txt.gz',
  'n':6,
  'purpose':'PAO1 WT vs delta-anr biofilms on CFBE cells; 3 biological replicates each'
 },
}
GPL84_URL='https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL84/soft/GPL84_family.soft.gz'
PA_TAG=re.compile(r'(?<![A-Za-z0-9])PA\d{4}(?!\d)', re.IGNORECASE)
PA_FROM_PROBE=re.compile(r'(?:^|_)PA\d{4}(?:_|$)', re.IGNORECASE)


def require(label, ok):
    if not bool(ok):
        raise RuntimeError('CHECK FAILED: '+label)
    print(label, 'PASS', flush=True)


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def get_gz(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size>1000:
        blob=dest.read_bytes()
        print('Cached:',dest,flush=True)
    else:
        print('Downloading:',url,flush=True)
        req=urllib.request.Request(url,headers={'User-Agent':'CFstate-academic-GEO-preflight/08l'})
        with urllib.request.urlopen(req,timeout=120) as handle:
            blob=handle.read()
        part=dest.with_name(dest.name+'.partial')
        part.write_bytes(blob)
        part.replace(dest)
    require(dest.name+'_archive_nonempty',len(blob)>1000)
    try:
        body=gzip.decompress(blob).decode('utf-8-sig',errors='replace')
    except Exception as exc:
        raise RuntimeError('Invalid gzip GEO download '+url) from exc
    require(dest.name+'_decompressed_nonempty',len(body)>1000)
    return blob,body


def read_tab_row(line):
    return next(csv.reader([line],delimiter='\t',quotechar='"'))


def parse_matrix(text):
    metadata={}
    headers=[]
    first_rows=[]
    ndata=0
    in_table=False
    cols=None
    unique_probes=set()
    repeated_probes=set()
    complete_nonempty=0
    value_examples=[]
    for ln in text.splitlines():
        if ln.startswith('!series_matrix_table_begin'):
            in_table=True
            continue
        if ln.startswith('!series_matrix_table_end'):
            break
        if not in_table:
            if ln.startswith('!Sample_'):
                row=read_tab_row(ln)
                key=row[0][1:]
                metadata.setdefault(key,[]).append(row[1:])
            continue
        row=read_tab_row(ln)
        if cols is None:
            cols=row
            headers=row[1:]
            continue
        if len(row)!=len(cols):
            raise RuntimeError('Matrix row width mismatch: '+str(row[:3]))
        probe=row[0].strip()
        if probe in unique_probes:
            repeated_probes.add(probe)
        unique_probes.add(probe)
        ndata+=1
        if all(x.strip() not in {'','null','NA'} for x in row[1:]):
            complete_nonempty+=1
        if len(first_rows)<10:
            first_rows.append({'probe':probe,'first_sample_value':row[1] if len(row)>1 else ''})
        if len(value_examples)<8:
            value_examples.extend(row[1:3])
    if cols is None:
        raise RuntimeError('No series_matrix_table_begin or matrix header found')
    return {'metadata':metadata,'headers':headers,'n_rows':ndata,
            'probes':unique_probes,'duplicates':repeated_probes,
            'complete_rows':complete_nonempty,'first_rows':first_rows,
            'value_examples':value_examples}


def parse_platform(text):
    in_table=False
    columns=None
    probe_to_genes={}
    nrows=0
    for ln in text.splitlines():
        if ln.startswith('!platform_table_begin'):
            in_table=True
            continue
        if ln.startswith('!platform_table_end'):
            break
        if not in_table or not ln or ln.startswith('#'):
            continue
        row=read_tab_row(ln)
        if columns is None:
            columns=row
            continue
        if len(row)!=len(columns):
            continue
        dat=dict(zip(columns,row))
        probe=dat.get('ID','').strip()
        if not probe:
            continue
        orf=dat.get('ORF','')
        genes={g.upper() for g in PA_TAG.findall(orf)}
        probe_to_genes[probe]=genes
        nrows+=1
    if columns is None:
        raise RuntimeError('GPL84 platform table missing')
    if not {'ID','ORF'}.issubset(columns):
        raise RuntimeError('GPL84 missing ID or ORF; columns='+repr(columns))
    return probe_to_genes,columns,nrows


def gene_overlap(probes,probe_to_genes,mask):
    valid={p:next(iter(probe_to_genes[p])) for p in probes
           if p in probe_to_genes and len(probe_to_genes[p])==1
           and next(iter(probe_to_genes[p])) in mask}
    npergene=Counter(valid.values())
    unique_genes={g for g,n in npergene.items() if n==1}
    multi_genes={g for g,n in npergene.items() if n>1}
    return {
      'probes_mapped_to_one_frozen_gene':len(valid),
      'distinct_frozen_genes_with_probes':len(npergene),
      'genes_with_one_probe_only':len(unique_genes),
      'genes_with_multiple_probes':len(multi_genes),
      'probes_missing_or_ambiguous_GPL84':len(probes)-len(valid),
      'example_multi_probe_genes':','.join(sorted(multi_genes)[:20]),
      'example_missing_mask_genes':','.join(sorted(mask-set(npergene))[:20]),
    },valid


def write_csv(p,records):
    if not records:
        p.write_text('empty\n')
        return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)


def sample_rows(dataset,obj):
    md=obj['metadata']
    names=obj['headers']
    # Sample metadata fields generally contain one row of N sample values.
    # Some characteristics fields occur multiple times; preserve them all.
    expected=len(names)
    result=[]
    for i,gsm in enumerate(names):
        d={'dataset':dataset,'GSM':gsm,'matrix_column':i+2}
        for key,records in md.items():
            if all(len(vals)==expected for vals in records):
                texts=[record[i].strip() for record in records]
                if key in ['Sample_title','Sample_source_name_ch1','Sample_characteristics_ch1',
                           'Sample_description','Sample_treatment_protocol_ch1',
                           'Sample_growth_protocol_ch1','Sample_geo_accession','Sample_data_processing']:
                    d[key]=' || '.join(texts)[:1200]
        result.append(d)
    return result


def self_test():
    fake='!Sample_title\t"20% oxygen"\t"0.4% oxygen"\n!Sample_geo_accession\t"GSMa"\t"GSMb"\n!series_matrix_table_begin\n"ID_REF"\t"GSMa"\t"GSMb"\n"PA0001_dnaA_at"\t"123.5"\t"100.7"\n"PA0002_dnaN_at"\t"8"\t"9"\n!series_matrix_table_end\n'
    parsed=parse_matrix(fake)
    assert parsed['headers']==['GSMa','GSMb']
    assert parsed['n_rows']==2 and parsed['complete_rows']==2
    assert sample_rows('demo',parsed)[1]['Sample_title']=='0.4% oxygen'
    platform='!platform_table_begin\nID\tORF\tSPOT_ID\nPA0001_dnaA_at\tPA0001\t.\nPA0002_dnaN_at\tPA0002\t.\n!platform_table_end\n'
    annot,_,n=parse_platform(platform)
    assert n==2
    cov,m=gene_overlap(parsed['probes'],annot,{'PA0001','PA0002','PA0003'})
    assert cov['genes_with_one_probe_only']==2
    assert len(m)==2
    print('SELF TEST PASSED')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
        return
    print('='*80)
    print('CF-State Phase 8 — Audit 08l: oxygen & CFBE Anr PUBLIC ARRAY PREFLIGHT')
    print('NO iModulon projection, NO retraining, NO clinical specificity claim')
    print('='*80)
    require('CFstate_root',MASK.is_file() and (ROOT/'results/phase8').is_dir())
    require('new_output_directory',not OUTPUT.exists())
    OUTPUT.mkdir(parents=True)
    try:
        import pandas as pd
        ids=pd.read_csv(MASK)['GeneID'].astype(str).tolist()
        mask=set(ids)
        require('frozen_mask_4923_unique',len(ids)==4923 and len(mask)==4923)
        gpl_blob,gpl_text=get_gz(GPL84_URL,RAW/'GPL84_family.soft.gz')
        annotations,platcols,nplat=parse_platform(gpl_text)
        require('GPL84_probe_annotations',nplat>=5000)
        require('GPL84_some_strict_PA_locus_tags',sum(bool(x) for x in annotations.values())>=4500)
        summaries=[]
        all_metadata=[]
        hashes={'GPL84':{'url':GPL84_URL,'compressed_sha256':sha256(gpl_blob),'n_platform_probe_rows':nplat}}
        print('\n========== GPL84 PLATFORM ANNOTATION ==========',flush=True)
        print('Rows:',nplat,'annotated probe sets:',sum(bool(x) for x in annotations.values()))
        for dataset,config in SERIES.items():
            blob,text=get_gz(config['url'],RAW/(dataset+'_series_matrix.txt.gz'))
            obj=parse_matrix(text)
            require(dataset+'_expected_samples',len(obj['headers'])==config['n'])
            require(dataset+'_unique_GSM',len(set(obj['headers']))==config['n'])
            require(dataset+'_matrix_gene_probe_rows',obj['n_rows']>=5000)
            cov,mapped=gene_overlap(obj['probes'],annotations,mask)
            require(dataset+'_some_genes_mappable',cov['distinct_frozen_genes_with_probes']>=4000)
            summary={'dataset':dataset,'n_samples':len(obj['headers']),
                     'n_probes':obj['n_rows'],'unique_probe_ids':len(obj['probes']),
                     'duplicate_probe_ids':len(obj['duplicates']),
                     'complete_value_rows':obj['complete_rows'],**cov,
                     'platform':'GPL84','experiment':config['purpose'],
                     'measurement':'Affymetrix array signal; not RNA-seq counts'}
            summaries.append(summary)
            sm=sample_rows(dataset,obj)
            all_metadata.extend(sm)
            write_csv(OUTPUT/('DIAGNOSTIC_'+dataset+'_top_probe_values.csv'),obj['first_rows'])
            write_csv(OUTPUT/('DIAGNOSTIC_'+dataset+'_probe_to_PA_locus.csv'),[
                {'probe':p,'PAO1_gene':g} for p,g in sorted(mapped.items())])
            hashes[dataset]={'url':config['url'],'compressed_sha256':sha256(blob),
                              'n_samples':len(obj['headers']),'n_probes':obj['n_rows']}
            print('\n==========',dataset,'SAMPLE METADATA ==========',flush=True)
            for r in sm:
                print(r['GSM'],r.get('Sample_title','<missing-title>'),
                    '|',r.get('Sample_characteristics_ch1','')[:150],flush=True)
            print('GENE COVERAGE:',cov,flush=True)
        write_csv(OUTPUT/'FROZEN_two_study_GEO_sample_manifest.csv',all_metadata)
        write_csv(OUTPUT/'DIAGNOSTIC_two_study_probe_gene_coverage.csv',summaries)
        (OUTPUT/'PUBLIC_INPUT_PROVENANCE.json').write_text(json.dumps(hashes,indent=2)+'\n')
        print('\n========== PUBLIC OXYGEN AND CFBE ARRAY PREFLIGHT ==========',flush=True)
        for item in summaries:
            print(item['dataset'],'samples',item['n_samples'],
                  'distinct frozen-mask genes',item['distinct_frozen_genes_with_probes'],
                  'single-probe frozen genes',item['genes_with_one_probe_only'],flush=True)
        print('STATUS: CFSTATE_PHASE8_08L_OXYGEN_CFBE_GPL84_PREFLIGHT_COMPLETE_REVIEW_REQUIRED')
        print('NEXT: Confirm condition assignments and atlas overlap; validate gene-set contrasts')
    except Exception:
        print('STATUS: CFSTATE_PHASE8_08L_INCOMPLETE_DIAGNOSTICS_PRESERVED',file=sys.stderr)
        raise


if __name__=='__main__':
    main()
