#!/usr/bin/env python3
"""CF-State Phase 8, Audit 08c: public zinc-perturbation processed-data preflight.

Two independent laboratory studies:
  * GSE310904 (PA14 WT vs delta-cntL, 3 biological replicates each), and
  * GSE60177 (PAO1 WT vs delta-znuA, one pooled RNA library per group).

Downloads ONLY small public GEO processed files, not FASTQ/SRA. Documents
sample/gene identifier formats, incomplete annotations, and whether the old
frozen PA14->PAO1 transfer resources exist. Does not re-fit or project the model.

Run from ~/projects/CFstate (with CFSTATE_ENV). --self-test exercises a synthetic
archive; no downloaded data or project inputs required for that mode.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

ROOT = Path.cwd().resolve()
OUT = ROOT / 'results/phase8/audit_08c_public_zinc_perturbation_preflight'
RAW = ROOT / 'data/raw/phase8_public_zinc_processed'
MASK = ROOT / 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'
SCRIPT = Path(__file__).resolve()
SERIES = {
    'GSE310904': {
        'archive': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE310nnn/GSE310904/suppl/GSE310904_RAW.tar',
        'expected': {'GSM9311958', 'GSM9311959', 'GSM9311960', 'GSM9311961', 'GSM9311962', 'GSM9311963'},
        'prefix': 'PA14',
    },
    'GSE60177': {
        'archive': 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60177/suppl/GSE60177_RAW.tar',
        'expected': {'GSM1466898', 'GSM1466899'},
        'prefix': 'PAO1',
    },
}
OPTIONAL_DESEQ = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE60nnn/GSE60177/suppl/GSE60177_PseudomonasDESeq.txt.gz'
STATUS = 'CFSTATE_PHASE8_08C_ZINC_PUBLIC_PROCESSED_PREFLIGHT_COMPLETE_REVIEW_REQUIRED'
MAX_ARCHIVE_BYTES = 15 * 1024 * 1024
MAX_MEMBER_BYTES = 4 * 1024 * 1024


def require(label: str, value: bool) -> None:
    if not bool(value):
        print(label + ' FAIL', flush=True)
        raise RuntimeError('CHECK FAILED: ' + label)
    print(label + ' PASS', flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_small(url: str, target: Path, limit: int = MAX_ARCHIVE_BYTES) -> Path:
    if target.exists():
        require('existing_download_nonempty_' + target.name, 100 < target.stat().st_size <= limit)
        print('  Reusing:', target, flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + '.part')
    error = None
    for attempt in range(3):
        try:
            request = Request(url, headers={'User-Agent': 'CFState-Phase8-Public-GEO-Research/1.0'})
            total = 0
            with urlopen(request, timeout=70) as response, tmp.open('wb') as dest:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > limit:
                        raise RuntimeError('Exceeded download limit: ' + url)
                    dest.write(chunk)
            require('download_nonempty_' + target.name, 100 < total <= limit)
            tmp.rename(target)
            return target
        except Exception as exc:
            error = exc
            if tmp.exists():
                tmp.unlink()
            if attempt < 2:
                time.sleep(attempt + 2)
    raise RuntimeError('Unable to retrieve public GEO supplementary file: ' + str(error))


def safe_extract(archive: Path, series: str, destination: Path) -> list[dict]:
    """Extract only expected GSM processed data into series-scoped folder."""
    accepted = []
    expected = SERIES[series]['expected']
    with tarfile.open(archive, 'r:*') as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            filename = Path(member.name).name
            if ('..' in Path(member.name).parts or member.name.startswith('/')
                    or not re.match(r'^GSM\d+_.+\.(?:gz|txt|bed|csv|tsv)$', filename, re.I)):
                print('  Skipping unrecognised tar member:', member.name, flush=True)
                continue
            match = re.match(r'^(GSM\d+)_', filename, re.I)
            gsm = match.group(1).upper()
            if gsm not in expected:
                print('  Skipping unexpected GSM:', filename, flush=True)
                continue
            if member.size > MAX_MEMBER_BYTES:
                raise RuntimeError('Processed sample file too large: ' + filename)
            target = destination / filename
            stream = tar.extractfile(member)
            if stream is None:
                raise RuntimeError('Could not read tar member: ' + filename)
            payload = stream.read()
            if len(payload) != member.size:
                raise RuntimeError('Short read in tar member: ' + filename)
            checksum = hashlib.sha256(payload).hexdigest()
            if target.exists():
                if sha256(target) != checksum:
                    raise RuntimeError('Existing processed file differs; refusing overwrite: ' + str(target))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('xb') as dest:
                    dest.write(payload)
            accepted.append({'series': series, 'GSM': gsm, 'processed_file': filename,
                             'processed_sha256': checksum, 'bytes': member.size,
                             'path': str(target.relative_to(ROOT)) if target.is_relative_to(ROOT) else str(target)})
    return accepted


def parse_schema(path: Path, mask: set[str]) -> dict:
    """Identify gene-ID columns without assuming PA14 and PAO1 IDs are interchangeable."""
    opener = gzip.open if path.suffix.lower() == '.gz' else open
    first = []
    widths = Counter()
    n_rows = 0
    potential_pao1: set[str] = set()
    potential_pa14: set[str] = set()
    first_field_examples = []
    numeric_last = 0
    with opener(path, 'rt', encoding='utf-8-sig', errors='replace') as reader:
        for line in reader:
            line = line.rstrip('\r\n')
            if not line.strip():
                continue
            if len(first) < 6:
                first.append(line[:250])
            cols = line.split('\t')
            widths[len(cols)] += 1
            n_rows += 1
            if len(first_field_examples) < 12:
                first_field_examples.append(cols[0][:70])
            if cols and re.fullmatch(r'[+\-]?(?:\d+\.?\d*|\.\d+)', cols[-1].strip()):
                numeric_last += 1
            # A PAO1 tag may occur in the original unmodified file.
            for tag in re.findall(r'(?<![A-Za-z0-9])PA\d{4}(?![A-Za-z0-9])', line, re.I):
                potential_pao1.add(tag.upper())
            for tag in re.findall(r'(?<![A-Za-z0-9])PA14[_-]\d{4,5}(?![A-Za-z0-9])', line, re.I):
                potential_pa14.add(tag.upper())
            if n_rows > 200_000:
                raise RuntimeError('Unexpectedly large gene table: ' + str(path))
    return {
        'nonempty_lines': n_rows,
        'modal_tab_fields': widths.most_common(1)[0][0] if widths else 0,
        'numeric_last_column_fraction': round(numeric_last / n_rows, 4) if n_rows else 0.,
        'PAO1_locus_ids_seen': len(potential_pao1),
        'PAO1_ids_in_4923_mask': len(potential_pao1 & mask),
        'PA14_locus_ids_seen': len(potential_pa14),
        'first_field_preview': '|'.join(first_field_examples),
        'first_lines': ' || '.join(first),
    }


def group_for(gsm: str) -> dict:
    id_number = int(gsm[3:])
    if 9311958 <= id_number <= 9311963:
        idx = id_number - 9311958
        return {'strain': 'PA14', 'genotype': 'WT' if idx < 3 else 'delta_cntL',
                'replicate': idx % 3 + 1, 'replicate_note': 'THREE_BIOLOGICAL_REPLICATES',
                'experimental_intervention': 'PSEUDOPALINE_BIOSYNTHESIS_GENE_DELETION'}
    if id_number in (1466898, 1466899):
        return {'strain': 'PAO1', 'genotype': 'WT' if id_number == 1466898 else 'delta_znuA',
                'replicate': 1, 'replicate_note': 'POOLED_BIOLOGICAL_REPLICATES_NO_VARIANCE_ESTIMATE',
                'experimental_intervention': 'ZINC_IMPORT_GENE_DELETION'}
    raise ValueError('Unexpected GSM: ' + gsm)


def pa14_map_candidates() -> list[dict]:
    """Report possible frozen PA14-to-PAO1 mappings, but do not guess their schema."""
    records = []
    phase3 = ROOT / 'results/phase3'
    if not phase3.exists():
        return records
    for file in phase3.rglob('*'):
        if not file.is_file() or file.suffix.lower() not in {'.csv', '.tsv', '.txt'}:
            continue
        name = file.name.lower()
        if any(key in name for key in ['pa14', 'ortholog', 'projection_map', 'transfer_map']):
            records.append({'path': str(file.relative_to(ROOT)),
                            'file_bytes': file.stat().st_size,
                            'sha256': sha256(file)})
    return sorted(records, key=lambda x: x['path'])[:150]


def self_test():
    assert group_for('GSM9311958')['genotype'] == 'WT'
    assert group_for('GSM9311963')['genotype'] == 'delta_cntL'
    assert group_for('GSM1466899')['genotype'] == 'delta_znuA'
    with tempfile.TemporaryDirectory() as work:
        location = Path(work)
        toy = location / 'toy.bed.gz'
        with gzip.open(toy, 'wt') as stream:
            stream.write('gene\tcount\nPA14_00120\t10\nPA14_00130\t12\n')
        schema = parse_schema(toy, {'PA0001'})
        assert schema['PA14_locus_ids_seen'] == 2
        assert schema['numeric_last_column_fraction'] > .6
        toy2 = location / 'toy2.txt.gz'
        with gzip.open(toy2, 'wt') as stream:
            stream.write('Geneid\tCount\nPA0001\t8\nPA0002\t13\n')
        assert parse_schema(toy2, {'PA0001'})['PAO1_ids_in_4923_mask'] == 1
        toy_archive = location / 'toy.tar'
        raw = gzip.compress(b'Geneid\tCount\nPA0001\t8\n')
        with tarfile.open(toy_archive, 'w') as tar:
            info = tarfile.TarInfo('GSM1466898_CountsRNA6.txt.gz')
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
        extracted = safe_extract(toy_archive, 'GSE60177', location / 'extract')
        assert len(extracted) == 1 and extracted[0]['GSM'] == 'GSM1466898'
    print('AUDIT_08C_SELF_TEST_PASS')


def main():
    print('=' * 78)
    print('CF-State Phase 8 — Audit 08c: targeted public zinc-perturbation preflight')
    print('Processed GEO data only. No raw SRA, no model fitting or projection.')
    print('=' * 78, flush=True)
    require('working_directory_CFstate', (ROOT / 'results/phase4').is_dir())
    require('new_output_directory', not OUT.exists())
    require('input_softcore_mask', MASK.is_file())
    require('input_script', SCRIPT.is_file())
    mask = set(pd.read_csv(MASK)['GeneID'].astype(str))
    require('mask_4923', len(mask) == 4923)

    records = []
    archives = []
    for accession, cfg in SERIES.items():
        print('\nPUBLIC DATASET:', accession, flush=True)
        tar_path = RAW / (accession + '_RAW.tar')
        fetch_small(cfg['archive'], tar_path)
        with tarfile.open(tar_path, 'r:*') as tar:
            require(accession + '_archive_readable', len(tar.getmembers()) >= len(cfg['expected']))
        archives.append({'dataset': accession, 'archive_sha256': sha256(tar_path),
                         'archive_bytes': tar_path.stat().st_size,
                         'archive_path': str(tar_path.relative_to(ROOT)),
                         'url': cfg['archive']})
        rows = safe_extract(tar_path, accession, RAW / accession)
        require(accession + '_expected_six_or_two_GSM',
                set(r['GSM'] for r in rows) == cfg['expected'] and len(rows) == len(cfg['expected']))
        for row in rows:
            row.update(group_for(row['GSM']))
            row.update(parse_schema(ROOT / row['path'], mask))
            records.append(row)
        print('  Processed samples inspected:', len(rows), flush=True)
        for row in rows[:2]:
            print('  ', row['GSM'], 'columns:', row['modal_tab_fields'],
                  'PA14 tags:', row['PA14_locus_ids_seen'],
                  'PAO1 tags:', row['PAO1_locus_ids_seen'], flush=True)
            print('  Preview:', row['first_lines'][:300], flush=True)

    frame = pd.DataFrame(records)
    require('eight_unique_GSM', len(frame) == 8 and frame['GSM'].is_unique)
    require('PA14_3x2_design', frame.query("strain == 'PA14'").groupby('genotype').size().to_dict()
            == {'WT': 3, 'delta_cntL': 3})
    require('PAO1_pooled_1x2_design', frame.query("strain == 'PAO1'").groupby('genotype').size().to_dict()
            == {'WT': 1, 'delta_znuA': 1})
    require('processed_tables_nonempty', (frame['nonempty_lines'] > 1000).all())
    # Report probable transfer mapping files, without inferring that PA14 IDs are PAO1 IDs.
    maps = pa14_map_candidates()
    print('\n========== PUBLIC ZINC PERTURBATION PREFLIGHT ==========', flush=True)
    print(frame[['series', 'GSM', 'strain', 'genotype', 'replicate', 'nonempty_lines',
                 'modal_tab_fields', 'PA14_locus_ids_seen', 'PAO1_locus_ids_seen',
                 'PAO1_ids_in_4923_mask']].to_string(index=False), flush=True)
    print('\n========== PA14 FROZEN PROJECTION-MAP CANDIDATES ==========', flush=True)
    for row in maps[:15]:
        print(row['path'], flush=True)
    if not maps:
        print('No candidate names found in results/phase3. Map may be elsewhere; do NOT use identity mapping.', flush=True)

    # Store outputs only once all critical checks have succeeded.
    OUT.mkdir(parents=True, exist_ok=False)
    frame.sort_values(['series', 'GSM']).to_csv(OUT / 'PUBLIC_8_processed_sample_manifest_and_schema.csv', index=False)
    pd.DataFrame(archives).to_csv(OUT / 'PUBLIC_GEO_archive_provenance.csv', index=False)
    pd.DataFrame(maps, columns=['path', 'file_bytes', 'sha256']).to_csv(
        OUT / 'PA14_frozen_projection_map_candidate_paths.csv', index=False)
    provenance = {'audit': '08c', 'status': STATUS,
                  'mask_sha256': sha256(MASK), 'script_sha256': sha256(SCRIPT),
                  'GSE310904_note': 'PA14 WT vs delta-cntL, 3 replicates/group; ID conversion required',
                  'GSE60177_note': 'PAO1 WT vs delta-znuA; biological replicates pooled; no uncertainty estimate',
                  'warning': 'Training overlap with original aeruPRECISE model has not been assessed; preflight only.'}
    (OUT / 'PROVENANCE.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (OUT / 'AUDIT_08C_STATUS.txt').write_text(STATUS + '\n')
    print('\nSTATUS:', STATUS, flush=True)
    print('NEXT: reconcile PA14 locus tags with the existing frozen PA14-PAO1 map;', flush=True)
    print('then evaluate Zur-2 and related programmes in genotype contrasts, without retraining.', flush=True)


if __name__ == '__main__':
    if '--self-test' in sys.argv:
        self_test()
    else:
        try:
            main()
        except Exception as exc:
            print('AUDIT 08c INCOMPLETE:', exc, flush=True)
            raise
