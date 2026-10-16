#!/usr/bin/env python3
"""CF-State Phase 8 Audit 08i: schema-only preflight of public Anr deletion RNA-seq.

GSE68534: PAO1 WT/delta-anr and J215 WT/delta-anr, 2 replicates per group,
colony biofilms at 1% oxygen (Hammond et al., J Bacteriol, 2015).

No projection, modelling, or reinterpretation of processed measurements is done.
Dependencies: Python standard library, pandas, openpyxl.
Run from CFstate project root. Self-test: --self-test.
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
import urllib.request
from collections import Counter
from pathlib import Path

import openpyxl

SERIES = 'GSE68534'
URL = ('https://ftp.ncbi.nlm.nih.gov/geo/series/GSE68nnn/GSE68534/suppl/'
       'GSE68534_ProcessedDataMatrices.xlsx.gz')
OUTPUT_NAME = 'audit_08i_GSE68534_Anr_low_oxygen_preflight'
SAMPLE_MANIFEST = [
    ('GSM1674725', 'PAO1', 'WT', '1'),
    ('GSM1674726', 'PAO1', 'WT', '2'),
    ('GSM1674727', 'PAO1', 'delta_anr', '1'),
    ('GSM1674728', 'PAO1', 'delta_anr', '2'),
    ('GSM1674729', 'J215', 'WT', '1'),
    ('GSM1674730', 'J215', 'WT', '2'),
    ('GSM1674731', 'J215', 'delta_anr', '1'),
    ('GSM1674732', 'J215', 'delta_anr', '2'),
]
TAG = re.compile(r'(?<![A-Za-z0-9])PA\d{4}(?!\d)', re.IGNORECASE)
SOURCE_EXT = '.xlsx.gz'


def require(label, condition):
    if not condition:
        raise RuntimeError('CHECK FAILED: ' + label)
    print(label, 'PASS', flush=True)


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def download_gz(dest):
    if dest.is_file() and dest.stat().st_size > 5000:
        print('Using existing downloaded archive:', dest, flush=True)
        compressed = dest.read_bytes()
    else:
        req = urllib.request.Request(URL, headers={'User-Agent': 'CFstate-public-GEO-audit/08i (academic research)'})
        print('Downloading public GEO processed workbook (~1.1 MB compressed)', flush=True)
        with urllib.request.urlopen(req, timeout=120) as response:
            compressed = response.read()
        dest.parent.mkdir(parents=True, exist_ok=True)
        temp = dest.with_name(dest.name + '.partial')
        temp.write_bytes(compressed)
        temp.replace(dest)
    require('compressed_archive_nonempty', len(compressed) > 5000)
    xlsx = gzip.decompress(compressed)
    require('xlsx_zip_signature', xlsx.startswith(b'PK\x03\x04'))
    return compressed, xlsx


def load_mask(root):
    path = root / 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'
    require('input_frozen_gene_mask', path.is_file())
    with path.open(newline='') as handle:
        reader = csv.DictReader(handle)
        require('gene_mask_GeneID_column', 'GeneID' in (reader.fieldnames or []))
        ids = [str(row['GeneID']).strip() for row in reader]
    require('frozen_mask_4923_unique', len(ids) == 4923 and len(set(ids)) == 4923)
    return set(ids)


def short_value(value, limit=160):
    if value is None:
        return ''
    s = str(value).replace('\n', ' ').replace('\r', ' ').strip()
    return s if len(s) <= limit else s[:limit] + '...'


def candidate_id(value):
    if value is None:
        return None
    match = TAG.search(str(value))
    return match.group(0).upper() if match else None


def inspect_workbook(xlsx, out, mask):
    wb = openpyxl.load_workbook(io.BytesIO(xlsx), read_only=True, data_only=True)
    require('processed_workbook_has_worksheets', len(wb.sheetnames) > 0)
    print('\n========== PUBLIC ANR WORKBOOK SHEETS ==========', flush=True)
    results = []
    candidates = []
    for name in wb.sheetnames:
        ws = wb[name]
        print('SHEET', repr(name), 'rows', ws.max_row, 'columns', ws.max_column, flush=True)
        # Collect up to 30 rows of preview and a bounded set of locus candidates
        max_scan_rows = min(ws.max_row or 0, 13000)
        max_scan_cols = min(ws.max_column or 0, 24)
        sampled_col_tags = [set() for _ in range(max_scan_cols)]
        sampled_col_valid = [0 for _ in range(max_scan_cols)]
        first_rows = []
        for i, row in enumerate(ws.iter_rows(min_row=1, max_row=max_scan_rows,
                                            max_col=max_scan_cols, values_only=True), start=1):
            if i <= 12:
                first_rows.append([short_value(v) for v in row])
            for c, v in enumerate(row):
                if v is not None and str(v).strip():
                    sampled_col_valid[c] += 1
                tag = candidate_id(v)
                if tag:
                    sampled_col_tags[c].add(tag)
        name_safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)[:70] or 'Sheet'
        preview_path = out / ('PREVIEW_' + name_safe + '.csv')
        with preview_path.open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerows(first_rows)
        print('  FIRST-ROW PREVIEW:')
        for i, row in enumerate(first_rows[:5], start=1):
            print('   ', i, ' | '.join(row[:9]), flush=True)
        best_col = max(range(max_scan_cols), key=lambda k: len(sampled_col_tags[k]), default=None)
        best_overlap = len(sampled_col_tags[best_col] & mask) if best_col is not None else 0
        print('  BEST PAO1 ID COLUMN:', (best_col + 1) if best_col is not None else '-',
              'unique PA IDs', len(sampled_col_tags[best_col]) if best_col is not None else 0,
              'frozen-mask IDs', best_overlap, flush=True)
        for c in range(max_scan_cols):
            row = {'sheet': name, 'excel_col_1based': c + 1,
                   'first_row_value': first_rows[0][c] if first_rows else '',
                   'second_row_value': first_rows[1][c] if len(first_rows) > 1 else '',
                   'nonempty_cells_scanned': sampled_col_valid[c],
                   'distinct_PA_locus_tags': len(sampled_col_tags[c]),
                   'distinct_frozen_mask_tags': len(sampled_col_tags[c] & mask)}
            candidates.append(row)
        results.append({'sheet': name, 'rows': ws.max_row, 'columns': ws.max_column,
                        'scanned_rows': max_scan_rows, 'scanned_columns': max_scan_cols,
                        'best_PA_column_1based': None if best_col is None else best_col+1,
                        'best_unique_PA_ID_count': 0 if best_col is None else len(sampled_col_tags[best_col]),
                        'best_frozen_mask_overlap': best_overlap})
    wb.close()
    return results, candidates


def write_csv(path, records):
    if not records:
        path.write_text('no_data\n')
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def self_test():
    assert candidate_id('PA0001') == 'PA0001'
    assert candidate_id('prefix PA4340 annotation') == 'PA4340'
    assert candidate_id('PA14_00010') is None
    assert candidate_id('non-locus') is None
    blob = io.BytesIO()
    from openpyxl import Workbook
    book = Workbook()
    s = book.active
    s.title = 'gene counts'
    s.append(['gene', 'GSM1674725', 'GSM1674726'])
    s.append(['PA0001', 10, 20])
    s.append(['PA0002', 5, 6])
    book.save(blob)
    assert blob.getvalue().startswith(b'PK\x03\x04')
    assert len(SAMPLE_MANIFEST) == 8
    assert len({s[0] for s in SAMPLE_MANIFEST}) == 8
    print('SELF TEST PASSED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root = Path.cwd()
    print('=' * 78)
    print('CF-State Phase 8 — Audit 08i: GSE68534 Anr low-oxygen RNA-seq preflight')
    print('PUBLIC PROCESSED WORKBOOK ONLY — NO BIOLOGICAL PROJECTION')
    print('=' * 78)
    require('working_directory_CFstate', (root/'results/phase8').is_dir())
    out = root/'results/phase8'/OUTPUT_NAME
    require('new_output_directory', not out.exists())
    out.mkdir(parents=True)
    try:
        mask = load_mask(root)
        raw = root/'data/raw/phase8_public_anr_processed/GSE68534'
        compressed, workbook_data = download_gz(raw/'GSE68534_ProcessedDataMatrices.xlsx.gz')
        require('manifest_8_unique_GSM', len(SAMPLE_MANIFEST) == 8 and
                len({r[0] for r in SAMPLE_MANIFEST}) == 8)
        manifest_rows = [{'GSM': gsm, 'strain': strain, 'genotype': geno,
                          'replicate': rep, 'oxygen': '1_percent',
                          'culture': '12h_colony_biofilm'}
                         for gsm, strain, geno, rep in SAMPLE_MANIFEST]
        write_csv(out/'FROZEN_GEO_8_sample_design.csv', manifest_rows)
        summaries, columns = inspect_workbook(workbook_data, out, mask)
        write_csv(out/'DIAGNOSTIC_workbook_sheet_inventory.csv', summaries)
        write_csv(out/'DIAGNOSTIC_workbook_column_geneID_candidates.csv', columns)
        metadata = {'source_url': URL, 'series': SERIES,
                    'compressed_sha256': sha256_bytes(compressed),
                    'xlsx_sha256': sha256_bytes(workbook_data),
                    'compressed_bytes': len(compressed),
                    'xlsx_bytes': len(workbook_data),
                    'n_sheets': len(summaries),
                    'explicit_donor_or_patient_comparison': False,
                    'independent_biological_replicates_per_group': 2,
                    'data_scope': 'public processed RNA-seq, experimental biofilms',
                    'no_projection_or_model_fitting': True}
        (out/'DATA_PROVENANCE.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print('\n========== AUDIT 08i SUMMARY ==========', flush=True)
        for r in summaries:
            print(r['sheet'], ':', r['rows'], 'rows,',r['columns'],
                  'columns; top PAO1 ID overlap', r['best_frozen_mask_overlap'])
        print('Files saved:', out)
        print('NEXT: inspect worksheet/sample column semantics before frozen projection')
        print('STATUS: CFSTATE_PHASE8_08I_ANR_PUBLIC_MATRIX_PREFLIGHT_COMPLETE_REVIEW_REQUIRED')
    except Exception:
        print('STATUS: CFSTATE_PHASE8_08I_INCOMPLETE_DIAGNOSTICS_PRESERVED', file=sys.stderr)
        raise


if __name__ == '__main__':
    main()
