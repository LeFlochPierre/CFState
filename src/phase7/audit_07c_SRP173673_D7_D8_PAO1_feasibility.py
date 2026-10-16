#!/usr/bin/env python3
"""CF-State Phase 7 Audit 07c: D-7/D-8 CF01 read-level PAO1 feasibility pilot.

Usage: run from CFstate repository root within CFSTATE_ENV.
Only two frozen run accessions; no other study reads downloaded.
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
A07A = P7 / 'audit_07a_external_CF_RNA_metadata_census'
A07B = P7 / 'audit_07b_candidate_RNA_run_triage'
OUT = P7 / 'audit_07c_SRP173673_D7_D8_PAO1_feasibility'
RAW = ROOT / 'data/raw/SRP173673_CF01_D7_D8'
REF = ROOT / 'references/phase4/PRJNA576508_cornforth'
DECOY = REF / 'bowtie2_decoy/Cornforth_104_decoys'
PAO1 = REF / 'bowtie2_PAO1/PAO1_NC_002516.2'
SAF = REF / 'Cornforth_PAO1_no_decimal_locus_tags.saf'
MASK = ROOT / 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'
RUNS = {'SRR8334097': 'D-7', 'SRR8334098': 'D-8'}
STATUS = 'CFSTATE_PHASE7_07C_D7_D8_PAO1_READ_LEVEL_FEASIBILITY_COMPLETE_REVIEW_REQUIRED'
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


def fetch_ena(run: str) -> dict[str, str]:
    fields = 'run_accession,library_layout,fastq_ftp,fastq_md5,fastq_bytes,read_count'
    query = urlencode({'accession': run, 'result': 'read_run', 'fields': fields, 'format': 'tsv'})
    url = 'https://www.ebi.ac.uk/ena/portal/api/filereport?' + query
    last = None
    for attempt in range(3):
        try:
            req = Request(url, headers={'User-Agent': 'CFstate-Audit-07c/academic-research'})
            with urlopen(req, timeout=50) as handle:
                content = handle.read(200000).decode('utf-8-sig')
            rows = list(csv.DictReader(io.StringIO(content), delimiter='\t'))
            if len(rows) != 1 or rows[0]['run_accession'] != run:
                raise ValueError('Unexpected ENA run record: ' + content[:250])
            return rows[0]
        except Exception as exc:
            last = exc
            time.sleep(attempt + 1)
    raise RuntimeError(f'ENA retrieval failed for {run}: {last}')


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
    print('CF-State Phase 7 — Audit 07c: CF01 D-7/D-8 PAO1 feasibility')
    print('=' * 76, flush=True)
    check((A07A / 'AUDIT_07A_STATUS.txt').read_text().strip() ==
          'CFSTATE_PHASE7_07A_METADATA_CENSUS_COMPLETE_REVIEW_REQUIRED',
          '07a_frozen')
    check((A07B / 'AUDIT_07B_STATUS.txt').read_text().strip() ==
          'CFSTATE_PHASE7_07B_CANDIDATE_RNA_METADATA_TRIAGE_COMPLETE_REVIEW_REQUIRED',
          '07b_frozen')
    check(not OUT.exists(), 'new_07c_output_directory')
    for exe in ['curl', 'gzip', 'cutadapt', 'bowtie2', 'samtools', 'featureCounts']:
        check(shutil.which(exe) is not None, 'executable_' + exe)
    for path in [SAF, MASK, Path(str(DECOY) + '.1.bt2'),
                 Path(str(PAO1) + '.1.bt2')]:
        check(path.is_file(), 'reference_' + path.name)
    frozen = pd.read_csv(A07A / 'FROZEN_ENA_all_run_metadata.csv', dtype=str).fillna('')
    selected = frozen[frozen['run_accession'].isin(RUNS)]
    check(len(selected) == 2 and set(selected['requested_accession']) == {'SRP173673'},
          'frozen_study_and_runs')
    check((selected['metadata_RNA_class'] ==
          'CANDIDATE_RNA_REVIEW_SPECIES_AND_COVERAGE').all(), 'RNA_assay_label')
    mask = pd.read_csv(MASK)['GeneID'].astype(str).tolist()
    check(len(mask) == 4923 and len(set(mask)) == 4923, '4923_gene_mask')

    OUT.mkdir(parents=True)
    RAW.mkdir(parents=True, exist_ok=True)
    (OUT / 'AUDIT_07C_STATUS.txt').write_text('CFSTATE_PHASE7_07C_REVIEW_REQUIRED\n')
    manifest = []
    summary = []
    for run, day in RUNS.items():
        print('\nRUN', run, day, flush=True)
        row = fetch_ena(run)
        layout, records = fastq_files(row, run)
        metadata_layout = selected.loc[selected['run_accession'] == run, 'library_layout'].iloc[0]
        if metadata_layout:
            check(layout == metadata_layout, 'layout_matches_07a_' + run)
        manifest.extend({'run': run, 'day': day, 'layout': layout,
                         'url': url, 'MD5': md5, 'bytes': byte_count,
                         'raw_path': str(dst)}
                        for url, md5, byte_count, dst in records)
        for url, md5, byte_count, dst in records:
            download_file(url, md5, byte_count, dst)
        work = OUT / run
        bam = run_single(run, layout, [x[-1] for x in records], work)
        feature_file = work / 'PAO1_featureCounts.txt'
        table = pd.read_csv(feature_file, sep='\t', comment='#')
        check(len(table) == 5567 and table['Geneid'].is_unique,
              '5567_PAO1_gene_counts_' + run)
        counts = pd.Series(table.iloc[:, -1].to_numpy(dtype=np.int64),
                           index=table['Geneid'].astype(str))
        check(set(mask).issubset(set(counts.index)), '4923_gene_mask_covered_' + run)
        assigned = int(counts.sum())
        shared = counts.reindex(mask)
        summary.append({
            'run_accession': run, 'timepoint': day, 'layout': layout,
            'ENA_read_count': row.get('read_count', ''),
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
        print(' ', run, 'assigned', assigned,
              'mask genes >=1', int((shared >= 1).sum()), flush=True)

    pd.DataFrame(manifest).to_csv(OUT / 'FROZEN_ENA_FASTQ_MD5_manifest.csv', index=False)
    output = pd.DataFrame(summary)
    output.to_csv(OUT / 'FROZEN_D7_D8_PAO1_mapping_and_gene_coverage_QC.csv', index=False)
    notes = (
        'Two RNA runs from the SAME CF patient CF01. One biological donor, two timepoints.\n'
        'Reads processed with the previously frozen 104-decoy + PAO1 alignment references.\n'
        'PAO1 alignments are feasibility indicators, NOT species-confirmed counts.\n'
        'No strain-aware or multi-strain mapping; no diagnostic conclusions.\n'
        'Technical count coverage is not sufficient by itself to establish valid iModulon activities.\n'
        'Paired reads, if present, are aligned and counted as pairs, NOT independent reads.\n'
        'Before interpreting Pseudomonas-specific expression, independently validate taxonomic specificity and reference bias.\n'
        'Runs D279/D303 excluded from pilot due to very low ENA read counts; no permanent exclusion.\n'
    )
    (OUT / 'METHODS_AND_LIMITATIONS.txt').write_text(notes)
    (OUT / 'INPUT_SHA256.json').write_text(json.dumps({
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [A07A / 'FROZEN_ENA_all_run_metadata.csv', SAF, MASK]
    }, indent=2) + '\n')
    (OUT / 'AUDIT_07C_STATUS.txt').write_text(STATUS + '\n')
    print('\n========== D7/D8 PAO1 FEASIBILITY ==========', flush=True)
    print(output.round(4).to_string(index=False))
    print('\nSAME DONOR: CF01. No new independent patients from these runs.')
    print('STATUS:', STATUS)


if __name__ == '__main__':
    main()
