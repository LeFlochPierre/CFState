#!/usr/bin/env python3
"""CF-State Phase 7 Audit 07g: CF146 day 33/day 35 read-level PAO1 feasibility pilot.

Usage: run from CFstate repository root within CFSTATE_ENV.
Only two frozen GSE139943 run accessions; no other study reads downloaded.
Paired libraries are processed as pairs (Bowtie2, featureCounts), not as independent mates.
This is a feasibility screen with the existing 104-decoy/PAO1 reference, NOT strain-aware quantification.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P7 = ROOT / 'results/phase7'
A07F = P7 / 'audit_07f_GSE139943_run_preflight'
A07E = P7 / 'audit_07e_GSE139943_taxon_screen'
OUT = P7 / 'audit_07g_GSE139943_CF146_D33_D35_PAO1_feasibility'
RAW = ROOT / 'data/raw/GSE139943_CF146_D33_D35'
REF = ROOT / 'references/phase4/PRJNA576508_cornforth'
DECOY = REF / 'bowtie2_decoy/Cornforth_104_decoys'
PAO1 = REF / 'bowtie2_PAO1/PAO1_NC_002516.2'
SAF = REF / 'Cornforth_PAO1_no_decimal_locus_tags.saf'
MASK = ROOT / 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'
RUNS = {'SRR10401171': 33, 'SRR10401172': 35}
STATUS = 'CFSTATE_PHASE7_07G_CF146_D33_D35_PAO1_FEASIBILITY_COMPLETE_REVIEW_REQUIRED'
ADAPTER = 'AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC'


def check(cond: bool, label: str):
    if not cond:
        raise RuntimeError('FAIL ' + label)
    print(label + ' PASS', flush=True)


def run_cmd(command, log: Path, cwd=ROOT):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as stream:
        proc = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        tail = log.read_text(errors='replace')[-2500:]
        raise RuntimeError(f'Command failed ({proc.returncode}): {command!r}\n{tail}')


def run_pipeline(cmd: str, log: Path):
    # Bash pipefail ensures mapper, samtools view and sort failures propagate.
    with log.open('w') as stream:
        proc = subprocess.run(['bash', '-o', 'pipefail', '-c', cmd], cwd=ROOT,
                              stdout=stream, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(f'PAO1 mapping failed: {log}\n' +
                           log.read_text(errors='replace')[-2500:])


def fastq_files(row: dict, run: str):
    paths = row.get('fastq_ftp', '').split(';')
    md5s = row.get('fastq_md5', '').split(';')
    sizes = row.get('fastq_bytes', '').split(';')
    if not paths or not all(paths) or len(paths) != len(md5s) or len(paths) != len(sizes):
        raise RuntimeError(f'Missing or inconsistent ENA FASTQ fields for {run}')
    layout = row.get('library_layout', '').upper()
    if layout not in {'SINGLE', 'PAIRED'}:
        raise RuntimeError(f'Unexpected layout {layout!r} for {run}')
    if len(paths) != (1 if layout == 'SINGLE' else 2):
        raise RuntimeError(f'FASTQ file count/layout mismatch for {run}: {layout}, {paths}')
    result = []
    for path, md5, size in zip(paths, md5s, sizes):
        if not path.startswith('ftp.sra.ebi.ac.uk/'):
            raise RuntimeError('Unexpected ENA FTP host: ' + path)
        url = 'https://' + path
        name = Path(urlparse(url).path).name
        if not name.endswith('.fastq.gz') or run not in name:
            raise RuntimeError('Unexpected FASTQ name: ' + name)
        if not re.fullmatch('[a-fA-F0-9]{32}', md5):
            raise RuntimeError('Invalid ENA MD5: ' + md5)
        result.append((url, md5.lower(), int(size), RAW / name))
    return layout, result


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open('rb') as handle:
        for buf in iter(lambda: handle.read(2 * 1024 * 1024), b''):
            digest.update(buf)
    return digest.hexdigest()


def download_file(url: str, md5: str, size: int, dst: Path):
    if dst.exists():
        check(dst.stat().st_size == size and md5_file(dst) == md5,
              'existing_download_MD5_' + dst.name)
        return
    tmp = dst.with_name(dst.name + '.partial')
    print('Downloading', dst.name, f'({size / 1e6:.1f} MB)', flush=True)
    # curl resume is safe because final size/MD5 are checked; retry partial bytes if needed.
    run_cmd(['curl', '-fL', '-C', '-', '--retry', '3', '--connect-timeout', '30',
             '-o', str(tmp), url], OUT / ('curl_' + dst.name + '.log'))
    check(tmp.stat().st_size == size, 'download_size_' + dst.name)
    check(md5_file(tmp) == md5, 'download_MD5_' + dst.name)
    run_cmd(['gzip', '-t', str(tmp)], OUT / ('gzip_' + dst.name + '.log'))
    tmp.rename(dst)


def align_rate(path: Path) -> float:
    text = path.read_text(errors='replace')
    found = re.search(r'([0-9.]+)% overall alignment rate', text)
    if not found:
        raise RuntimeError('Bowtie2 alignment percentage absent: ' + str(path))
    return float(found.group(1))


def run_single(run: str, layout: str, files: list[Path], work: Path):
    work.mkdir(exist_ok=False)
    trimmed = [work / ('trimmed_' + str(i + 1) + '.fastq.gz')
               for i in range(len(files))]
    nondecoy = [work / ('nondecoy_' + str(i + 1) + '.fastq.gz')
                for i in range(len(files))]
    if layout == 'SINGLE':
        trim_cmd = ['cutadapt', '-a', ADAPTER, '-m', '25', '-j', '4',
                    '-o', str(trimmed[0]), str(files[0])]
        decoy_reads = ['-U', str(trimmed[0])]
        decoy_out = ['--un-gz', str(nondecoy[0])]
        pao_reads = ['-U', str(nondecoy[0])]
    else:
        trim_cmd = ['cutadapt', '-a', ADAPTER, '-A', ADAPTER, '-m', '25',
                    '-j', '4', '-o', str(trimmed[0]), '-p', str(trimmed[1]),
                    str(files[0]), str(files[1])]
        decoy_reads = ['-1', str(trimmed[0]), '-2', str(trimmed[1])]
        decoy_out = ['--un-conc-gz', str(work / 'nondecoy_%.fastq.gz')]
        pao_reads = ['-1', str(nondecoy[0]), '-2', str(nondecoy[1])]

    print(' ', run, '[1/4] trimming', flush=True)
    run_cmd(trim_cmd, work / 'cutadapt.log')
    print(' ', run, '[2/4] decoy alignment', flush=True)
    cmd = ['bowtie2', '--very-sensitive', '-p', '6', '-x', str(DECOY)] + \
          decoy_reads + decoy_out + ['-S', '/dev/null']
    run_cmd(cmd, work / 'decoy_bowtie2.log')
    for nd in nondecoy:
        check(nd.is_file(), 'nondecoy_FASTQ_' + nd.name)

    print(' ', run, '[3/4] PAO1 alignment', flush=True)
    bam = work / 'PAO1_mapped.sorted.bam'
    parts = ['bowtie2', '--very-sensitive', '-p', '6', '-x', str(PAO1)] + pao_reads
    left = ' '.join(shlex.quote(str(x)) for x in parts)
    pipeline = (left + ' 2> ' + shlex.quote(str(work / 'PAO1_bowtie2.log')) +
                ' | samtools view -@ 2 -u -F 4 -' +
                ' | samtools sort -@ 3 -m 512M -o ' + shlex.quote(str(bam)) + ' -')
    run_pipeline(pipeline, work / 'PAO1_pipeline.log')
    run_cmd(['samtools', 'quickcheck', str(bam)], work / 'BAM_quickcheck.log')
    run_cmd(['samtools', 'index', str(bam)], work / 'BAM_index.log')
    print(' ', run, '[4/4] featureCounts', flush=True)
    fc = ['featureCounts', '-T', '6', '-F', 'SAF', '-s', '0', '-a', str(SAF),
          '-o', str(work / 'PAO1_featureCounts.txt')]
    if layout == 'PAIRED':
        fc += ['-p', '--countReadPairs']
    fc += [str(bam)]
    run_cmd(fc, work / 'featureCounts.log')
    return bam


def count_bam(path: Path) -> int:
    return int(subprocess.check_output(
        ['samtools', 'view', '-c', '-F', '4', str(path)], text=True).strip())


def main():
    print('=' * 76)
    print('CF-State Phase 7 — Audit 07g: CF146 day 33/day 35 feasibility')
    print('=' * 76, flush=True)
    check(not OUT.exists(), 'new_07g_output_directory')
    check((A07F / 'AUDIT_07F_STATUS.txt').read_text().strip() ==
          'CFSTATE_PHASE7_07F_RUN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED', '07f_frozen')
    check((A07E / 'AUDIT_07E_STATUS.txt').read_text().strip() ==
          'CFSTATE_PHASE7_07E_REFERENCE_COUNT_SCREEN_REVIEW_REQUIRED', '07e_frozen')
    for exe in ['curl', 'gzip', 'cutadapt', 'bowtie2', 'samtools', 'featureCounts']:
        check(shutil.which(exe) is not None, 'executable_' + exe)
    for path in [SAF, MASK, Path(str(DECOY) + '.1.bt2'),
                 Path(str(PAO1) + '.1.bt2')]:
        check(path.is_file(), 'reference_' + path.name)

    frozen_file = A07F / 'FROZEN_GSE139943_candidate_run_metadata.csv'
    frozen = pd.read_csv(frozen_file, dtype=str).fillna('')
    check(frozen['run_accession'].is_unique, 'unique_frozen_07f_runs')
    selected = frozen[frozen['run_accession'].isin(RUNS)].copy()
    check(len(selected) == 2 and set(selected['run_accession']) == set(RUNS),
          'frozen_selected_runs_33_35')
    for _, rec in selected.iterrows():
        check(int(rec['CF146_day']) == RUNS[rec['run_accession']],
              'day_run_mapping_' + rec['run_accession'])
        check(rec['library_strategy'] == 'RNA-Seq' and
              rec['library_source'] == 'TRANSCRIPTOMIC' and
              rec['library_layout'] == 'SINGLE',
              'single_end_RNA_metadata_' + rec['run_accession'])

    geotable = pd.read_csv(A07E / 'FROZEN_9_reference_count_screen.csv')
    check(set(RUNS.values()).issubset(set(geotable['day'].astype(int))),
          'GEO_days_33_35_reference_screen')
    mask = pd.read_csv(MASK)['GeneID'].astype(str).tolist()
    check(len(mask) == 4923 and len(set(mask)) == 4923, '4923_gene_mask')

    OUT.mkdir(parents=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (OUT / 'AUDIT_07G_STATUS.txt').write_text('CFSTATE_PHASE7_07G_REVIEW_REQUIRED\n')
    manifest = []
    summary = []
    for run, day in RUNS.items():
        print('\nRUN', run, 'CF146 day', day, flush=True)
        metadata = selected.loc[selected['run_accession'] == run].iloc[0]
        row = {key: str(value) for key, value in metadata.items()}
        layout, records = fastq_files(row, run)
        check(layout == 'SINGLE', 'single_end_layout_' + run)
        manifest.extend({'run': run, 'day': day, 'layout': layout,
                         'url': url, 'MD5': md5, 'bytes': byte_count,
                         'raw_path': str(dst)}
                        for url, md5, byte_count, dst in records)
        for url, md5, byte_count, dst in records:
            download_file(url, md5, byte_count, dst)
        work = OUT / run
        bam = run_single(run, layout, [x[-1] for x in records], work)
        table = pd.read_csv(work / 'PAO1_featureCounts.txt', sep='\t', comment='#')
        check(len(table) == 5567 and table['Geneid'].astype(str).is_unique,
              '5567_PAO1_gene_counts_' + run)
        counts = pd.Series(table.iloc[:, -1].to_numpy(dtype=np.int64),
                           index=table['Geneid'].astype(str))
        check(set(mask).issubset(counts.index), '4923_gene_mask_' + run)
        check((counts >= 0).all(), 'nonnegative_counts_' + run)
        assigned = int(counts.sum())
        shared = counts.reindex(mask)
        geo_count = int(geotable.loc[geotable['day'] == day,
                                     'PAO1_reference_count'].iloc[0])
        summary.append({
            'run_accession': run, 'CF146_day': day, 'layout': layout,
            'ENA_read_count': row.get('read_count', ''),
            'GEO_processed_PAO1_reference_count': geo_count,
            'decoy_alignment_percent': align_rate(work / 'decoy_bowtie2.log'),
            'PAO1_alignment_percent_of_nondecoy': align_rate(work / 'PAO1_bowtie2.log'),
            'PAO1_mapped_BAM_records': count_bam(bam),
            'featureCounts_assigned_units': assigned,
            'genes_detected_common_4923_GE1': int((shared >= 1).sum()),
            'genes_detected_common_4923_GE3': int((shared >= 3).sum()),
            'genes_detected_common_4923_GE10': int((shared >= 10).sum()),
            'gene_detection_fraction_GE1': float((shared >= 1).mean()),
            'RNA_feasibility_status': 'TECHNICAL_QC_ONLY',
        })
        counts.rename('assigned_count').to_csv(
            work / 'PAO1_5567_assigned_counts.csv', index_label='Geneid')
        print(' ', run, 'assigned', assigned,
              'genes >=1', int((shared >= 1).sum()), flush=True)

    pd.DataFrame(manifest).to_csv(OUT / 'FROZEN_ENA_FASTQ_MD5_manifest.csv', index=False)
    output = pd.DataFrame(summary)
    output.to_csv(OUT / 'FROZEN_D33_D35_PAO1_gene_coverage_QC.csv', index=False)
    (OUT / 'METHODS_AND_LIMITATIONS.txt').write_text(
        'Two timepoints from ONE CF patient CF146: NOT two independent donors.\n'
        'Processing uses 104-sequence decoy then PAO1 reference alignment.\n'
        'Reference-level GEO counts and current featureCounts are NOT directly equivalent.\n'
        'Mapping and 4923-gene detection are technical QC only; taxonomic specificity, '
        'clinical strain divergence and library-batch effects remain unresolved.\n'
        'No CF candidate projected or model parameters refitted.\n'
        'Inputs 07e/07f frozen; no other runs downloaded.\n')
    (OUT / 'INPUT_SHA256.json').write_text(json.dumps({
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [frozen_file, A07E / 'FROZEN_9_reference_count_screen.csv', SAF, MASK]
    }, indent=2) + '\n')
    (OUT / 'AUDIT_07G_STATUS.txt').write_text(STATUS + '\n')
    print('\n========== CF146 D33/D35 PAO1 FEASIBILITY ==========', flush=True)
    print(output.round(4).to_string(index=False))
    print('\nONE patient CF146. TECHNICAL FEASIBILITY ONLY.')
    print('STATUS:', STATUS)


if __name__ == '__main__':
    main()
