#!/usr/bin/env python3
"""CF-State Audit 07d: inspect one GEO BacRefSeq processed table, no inference.

Downloads GSM4150367 only and writes a read-only-style schema/identifier
report to a new Phase 7 audit directory. Never modifies prior audits.
"""
from pathlib import Path
from collections import Counter
from urllib.request import urlopen, Request
import hashlib
import gzip
import json
import re
import shutil
import sys

BASE = Path('results/phase7/audit_07d_GSE139943_processed_table_preflight')
FILE = 'GSM4150367_CF146_day16_mt_cs_BacRefSeq.tab.gz'
URL = ('https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4150nnn/'
       'GSM4150367/suppl/' + FILE)
TARGET = BASE / FILE
STATUS = BASE / 'AUDIT_07D_STATUS.txt'

if STATUS.exists():
    raise SystemExit('Audit 07d status already exists; preserving frozen output. Inspect: ' + str(BASE))
BASE.mkdir(parents=True, exist_ok=True)

print('=== CF-State Phase 7, Audit 07d ===', flush=True)
print('Source accession: GSM4150367 (CF146 day 16)', flush=True)
print('Purpose: identify the true processed-table schema, not infer gene counts', flush=True)

if not TARGET.exists():
    part = BASE / (FILE + '.part')
    if part.exists():
        raise SystemExit('An incomplete .part file exists; inspect it before retrying: ' + str(part))
    print('Downloading only one GEO supplementary table...', flush=True)
    try:
        req = Request(URL, headers={'User-Agent': 'CFState-research/1.0'})
        with urlopen(req, timeout=90) as source, part.open('wb') as dest:
            shutil.copyfileobj(source, dest)
        with gzip.open(part, 'rb') as handle:
            handle.read(1024)  # Detect invalid gzip immediately
        part.rename(TARGET)
    except Exception:
        part.unlink(missing_ok=True)
        print('Download failed; no completed audit status was written.', file=sys.stderr)
        raise

hasher = hashlib.sha256()
with TARGET.open('rb') as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b''):
        hasher.update(chunk)
sha256 = hasher.hexdigest()

# Inspect decompressed content without assuming that it is PAO1 gene counts.
preview = []
seen = 0
pa_locus_lines = 0
pao1_refseq_lines = 0
column_counts = Counter()
first_row_columns = []
max_rows = 50000

with gzip.open(TARGET, 'rt', encoding='utf-8', errors='replace') as stream:
    for line in stream:
        if not line.strip():
            continue
        seen += 1
        line = line.rstrip('\r\n')
        if len(preview) < 12:
            preview.append(line[:350])
        fields = line.split('\t')
        column_counts[len(fields)] += 1
        if seen == 1:
            first_row_columns = [f[:100] for f in fields[:16]]
        if re.search(r'(?<![A-Za-z0-9])PA\d{4}(?!\d)', line):
            pa_locus_lines += 1
        if 'NC_002516' in line:
            pao1_refseq_lines += 1
        if seen >= max_rows:
            break

report = {
    'source_accession': 'GSM4150367',
    'source_timepoint': 'CF146_day16_mt',
    'source_url': URL,
    'local_file': str(TARGET),
    'compressed_bytes': TARGET.stat().st_size,
    'sha256': sha256,
    'nonempty_lines_inspected': seen,
    'maximum_lines_scanned': max_rows,
    'scanned_entire_file': seen < max_rows,
    'tab_field_count_frequency': dict(column_counts),
    'first_row_fields': first_row_columns,
    'lines_containing_PA_four_digit_locus': pa_locus_lines,
    'lines_containing_NC_002516': pao1_refseq_lines,
    'preview_lines': preview,
    'interpretation': 'REQUIRES_MANUAL_SCHEMA_REVIEW; NO_PA_GENE_COUNTS_ASSUMED',
}
(BASE / 'FROZEN_source_and_schema_inspection.json').write_text(json.dumps(report, indent=2) + '\n')

print('\n===== RAW TABLE PREVIEW (first 12 nonempty lines) =====')
for line in preview:
    print(line)
print('\n===== SCHEMA / IDENTIFIER SUMMARY =====')
print('Compressed bytes:', report['compressed_bytes'])
print('Nonempty lines inspected:', seen)
print('Scanned entire file:', report['scanned_entire_file'])
print('Tab-separated column counts:', dict(column_counts))
print('Lines mentioning PA####:', pa_locus_lines)
print('Lines mentioning PAO1 NC_002516:', pao1_refseq_lines)
print('SHA256:', sha256)

STATUS.write_text('CFSTATE_PHASE7_07D_GEO_TABLE_SCHEMA_RETRIEVED_REVIEW_REQUIRED\n')
print('\nSTATUS:', STATUS.read_text().strip())
