#!/usr/bin/env python3
"""CF-State Phase 7 audit 07j: public supplementary XLSX metadata only.

No controlled-access data, no retraining, no claims about human donor counts.
"""
from __future__ import annotations

import csv
import hashlib
import io
import posixpath
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile

BASE = Path('results/phase7/audit_07j_airway_organoid_public_supplements')
DATA = Path('data/raw/airway_organoid_2025_public_supplements')
STATUS = BASE / 'AUDIT_07J_STATUS.txt'
DOI_KEY = '41598_2024_82500'
X_NS = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
R_NS = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
P_NS = 'http://schemas.openxmlformats.org/package/2006/relationships'
X = '{' + X_NS + '}'


def download_supplement(num: int) -> tuple[Path, str]:
    name = f'{DOI_KEY}_MOESM{num}_ESM.xlsx'
    target = DATA / name
    urls = [
        ('https://static-content.springer.com/esm/'
         'art%3A10.1038%2Fs41598-024-82500-w/MediaObjects/' + name),
        ('https://pmc.ncbi.nlm.nih.gov/articles/instance/'
         '11742674/bin/' + name),
    ]
    if target.exists():
        with target.open('rb') as fh:
            if fh.read(4) == b'PK\x03\x04':
                with ZipFile(target) as z:
                    if z.testzip() is None:
                        return target, 'PRESENT_PREVIOUSLY'
        raise RuntimeError('Existing file is invalid; refusing overwrite: ' + str(target))

    errors = []
    for url in urls:
        try:
            req = Request(url, headers={'User-Agent': 'CFState-academic-metadata-audit/1.0'})
            with urlopen(req, timeout=90) as response:
                payload = response.read()
            if not payload.startswith(b'PK\x03\x04'):
                raise ValueError('Response is not a ZIP-based .xlsx')
            with ZipFile(io.BytesIO(payload)) as z:
                if 'xl/workbook.xml' not in z.namelist() or z.testzip():
                    raise ValueError('Not a valid XLSX workbook')
            target.write_bytes(payload)
            return target, url
        except Exception as exc:
            errors.append(str(exc)[:160])
    raise RuntimeError(f'MOESM{num}: both sources failed: ' + ' / '.join(errors))


def texts_from_shared_strings(z: ZipFile) -> list[str]:
    if 'xl/sharedStrings.xml' not in z.namelist():
        return []
    shared = ET.fromstring(z.read('xl/sharedStrings.xml'))
    return [
        ''.join(t.text or '' for t in si.iter(X + 't'))
        for si in shared.findall(X + 'si')
    ]


def preview_rows(z: ZipFile, path: str, shared: list[str]) -> tuple[str, list[str]]:
    dimension = 'UNKNOWN'
    preview = []
    with z.open(path) as handle:
        for event, elem in ET.iterparse(handle, events=('start', 'end')):
            if event == 'start' and elem.tag == X + 'dimension':
                dimension = elem.get('ref', 'UNKNOWN')
            if event != 'end' or elem.tag != X + 'row':
                continue
            cells = []
            for cell in list(elem):
                if cell.tag != X + 'c':
                    continue
                v = cell.find(X + 'v')
                raw = '' if v is None or v.text is None else v.text
                if cell.get('t') == 's' and raw:
                    idx = int(raw)
                    raw = shared[idx] if idx < len(shared) else 'BAD_STRING_INDEX'
                elif cell.get('t') == 'inlineStr':
                    raw = ''.join(n.text or '' for n in cell.iter(X + 't'))
                cells.append(f"{cell.get('r', '?')}={raw[:65]}")
                if len(cells) >= 7:
                    break
            preview.append(' | '.join(cells)[:400])
            elem.clear()
            if len(preview) >= 2:
                break
    return dimension, preview


def inspect_workbook(path: Path, source: str) -> list[dict]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    result = []
    with ZipFile(path) as z:
        workbook = ET.fromstring(z.read('xl/workbook.xml'))
        rels = ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        lookup = {r.get('Id'): r.get('Target') for r in rels.findall('{' + P_NS + '}Relationship')}
        shared = texts_from_shared_strings(z)
        sheets = workbook.find(X + 'sheets')
        if sheets is None:
            raise ValueError('No worksheets: ' + str(path))
        for sheet in sheets.findall(X + 'sheet'):
            rid = sheet.get('{' + R_NS + '}id')
            relative = lookup.get(rid)
            if not relative:
                raise ValueError('Worksheet relationship missing: ' + str(rid))
            filename = (
                relative.lstrip('/') if relative.startswith('/')
                else posixpath.normpath(posixpath.join('xl', relative))
            )
            dimension, preview = preview_rows(z, filename, shared)
            row = {
                'file': path.name,
                'size_mb': round(path.stat().st_size / 1e6, 3),
                'source_url_or_status': source,
                'sha256': digest,
                'worksheet': sheet.get('name', ''),
                'dimension': dimension,
                'first_row': preview[0] if preview else '',
                'second_row': preview[1] if len(preview) > 1 else '',
            }
            result.append(row)
    return result


def main():
    if STATUS.exists():
        raise RuntimeError('Audit 07j already frozen; refusing to overwrite')
    DATA.mkdir(parents=True, exist_ok=True)
    BASE.mkdir(parents=True, exist_ok=True)
    output = []
    failed = []
    for i in (3, 4, 5, 6):
        print(f'Checking public Supplementary Material {i}', flush=True)
        try:
            file, source = download_supplement(i)
            records = inspect_workbook(file, source)
            output.extend(records)
            for rec in records:
                print('  SHEET:', rec['worksheet'], 'DIM:', rec['dimension'])
                print('    HEADER:', rec['first_row'][:240])
        except Exception as exc:
            failed.append(f'MOESM{i}: {exc}')
            print('  FAILED:', exc)
    if output:
        target = BASE / 'PUBLIC_SUPPLEMENT_WORKBOOK_SCHEMA.csv'
        with target.open('w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    print('\n=== AUDIT 07j SUMMARY ===')
    print('Files inspected:', len(set(r['file'] for r in output)))
    print('Worksheets:', len(output))
    for error in failed:
        print('BLOCKER:', error)
    status = (
        'CFSTATE_PHASE7_07J_PUBLIC_ORGANOID_SUPPLEMENTS_SCHEMA_REVIEW_REQUIRED'
        if not failed else
        'CFSTATE_PHASE7_07J_PARTIAL_RETRIEVAL_REVIEW_REQUIRED'
    )
    STATUS.write_text(status + '\n')
    print('STATUS:', status)
    print('No controlled-access human-containing reads downloaded.')


if __name__ == '__main__':
    main()
