#!/usr/bin/env python3
"""Read-only input/schema inspection before CF-State Audit 08n.

Usage (from ~/projects/CFstate):
    python src/phase8/inspect_08n_inputs.py --self-test
    python src/phase8/inspect_08n_inputs.py > notes/phase8_08n_input_inventory.txt

This does not create an audit directory or edit existing data. It produces a
concise manifest of earlier source scripts and existing output schemas so that
Audit 08n can use verified inputs rather than guessed paths or formats.
"""

import argparse
import csv
import gzip
import os
from pathlib import Path
import re
import sys

SCRIPT_NAMES = [
    'audit_08m_GPL84_oxygen_CFBE_Anr_gene_response.py',
    'audit_08j_GSE68534_frozen_Anr_regulator_response.py',
    'audit_08k_two_experimental_axes_clinical_decomposition.py',
    'audit_08g_perturbation_anchored_clinical_decomposition.py',
    'audit_08l2_GSE6741_GSE67006_oxygen_CFBE_preflight.py',
    'audit_08h_orthogonal_programme_concordance.py',
    'audit_08f_CFstate_metal_ablation_clinical_transfer.py',
]

# Exact known outputs first; later parts scan prior clinical audits for relevant files.
PRIORITY_RESULT_PREFIXES = (
    'audit_08m_', 'audit_08j_', 'audit_08k_', 'audit_08g_', 'audit_08l2_',
    'audit_03h_', 'audit_07i_',
)

KEYWORDS = (
    'cornforth', 'cf01', 'cf146', 'sputum', 'noncf', 'non_cf', 'wound',
    'clinical', 'gene', 'log2tpm', 'expression', 'contrast', 'mask',
    'gse6741', 'gse67006', 'oxygen', 'anr', 'respirat', 'donor',
)

EXTENSIONS = ('.csv', '.tsv', '.csv.gz', '.tsv.gz')
MAX_TOTAL_CANDIDATES = 100
MAX_PREVIEW_FILES = 55
MAX_SCRIPT_LINES = 32


def norm_ext(path):
    name = path.name.lower()
    if name.endswith('.tsv.gz') or name.endswith('.csv.gz'):
        return name[-6:]
    return path.suffix.lower()


def readable_table(path):
    return path.name.lower().endswith(EXTENSIONS)


def relative(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def preview_table(path):
    """Only reads header + 2 rows, bounded to 500 KiB per logical line."""
    opener = gzip.open if path.name.endswith('.gz') else open
    delimiter = '\t' if '.tsv' in path.name.lower() else ','
    try:
        with opener(path, mode='rt', encoding='utf-8-sig', errors='replace', newline='') as stream:
            # Some expression tables contain extremely wide header/records.
            reader = csv.reader(stream, delimiter=delimiter)
            header = next(reader, [])
            samples = []
            for _ in range(2):
                line = next(reader, None)
                if line is None:
                    break
                samples.append(line)
        def small(value):
            s = str(value).replace('\n', ' ').replace('\r', ' ')
            return s[:76] + ('…' if len(s) > 76 else '')
        cols = [small(x) for x in header[:18]]
        if len(header) > 18:
            cols.append('…(+' + str(len(header) - 18) + ' columns)')
        sample_repr = []
        for line in samples:
            fields = [small(x) for x in line[:8]]
            if len(line) > 8:
                fields.append('…(+' + str(len(line) - 8) + ' fields)')
            sample_repr.append(fields)
        return 'ncols=' + str(len(header)) + ' header=' + repr(cols) + ' first_rows=' + repr(sample_repr)
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        return 'PREVIEW_ERROR ' + type(exc).__name__ + ': ' + str(exc)[:150]


def audit_dirs(root):
    dirs = []
    for phase in ('phase4', 'phase5', 'phase7', 'phase8'):
        basedir = root / 'results' / phase
        if basedir.is_dir():
            dirs.extend(p for p in basedir.iterdir() if p.is_dir())
    return dirs


def score(path, root):
    name = str(path.relative_to(root)).lower()
    score_value = sum(2 if k in path.name.lower() else 1 for k in KEYWORDS if k in name)
    if any(prefix in name for prefix in PRIORITY_RESULT_PREFIXES):
        score_value += 8
    if ('cornforth' in name or 'cf01' in name or 'cf146' in name):
        score_value += 8
    if ('sample' in name or 'sample_level' in name) and 'expression' in name:
        score_value += 8
    return score_value


def find_candidates(root):
    candidates = []
    for audit_dir in audit_dirs(root):
        phase = audit_dir.parent.name
        important = (phase == 'phase8' and audit_dir.name.startswith(PRIORITY_RESULT_PREFIXES))
        important = important or audit_dir.name.startswith('audit_03h_')
        # Avoid reading old unrelated audit outputs unless filename clearly relevant.
        if not important and phase == 'phase8':
            continue
        for dirname, _, files in os.walk(audit_dir):
            for filename in files:
                path = Path(dirname) / filename
                if not path.is_file() or not readable_table(path):
                    continue
                path_str = str(path.relative_to(root)).lower()
                if important or any(word in path_str for word in KEYWORDS):
                    candidates.append(path)
    candidates = list(set(candidates))
    candidates.sort(key=lambda p: (-score(p, root), str(p)))
    return candidates[:MAX_TOTAL_CANDIDATES]


def scan_scripts(root, emit):
    emit('=== TARGET SCRIPT PATHS & FILE I/O ===')
    for script_name in SCRIPT_NAMES:
        path = root / 'src' / 'phase8' / script_name
        emit('SCRIPT ' + relative(path, root) + (' FOUND' if path.is_file() else ' MISSING'))
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        except OSError as exc:
            emit('  READ_ERROR ' + str(exc))
            continue
        io_lines = []
        rx = re.compile(r'(read_csv|to_csv|read_table|read_parquet|to_parquet|read_text|glob\(|rglob\(|\.csv|\.tsv|\.parquet|results/phase|data/processed|FROZEN_|Cornforth|CF01|CF146)', re.I)
        for i, line in enumerate(lines, 1):
            if rx.search(line):
                io_lines.append((i, line.strip()))
        emit('  source_lines=' + str(len(lines)) + ' matching_path_or_io_lines=' + str(len(io_lines)))
        for lineno, line in io_lines[:MAX_SCRIPT_LINES]:
            emit('  L' + str(lineno) + ': ' + line[:235])
        if len(io_lines) > MAX_SCRIPT_LINES:
            emit('  ... ' + str(len(io_lines) - MAX_SCRIPT_LINES) + ' more relevant lines (open original script)')
    emit('')


def inspect_known_outputs(root, emit):
    emit('=== PRIORITY OUTPUT DIRECTORIES ===')
    all_dirs = audit_dirs(root)
    for path in all_dirs:
        if path.name.startswith(PRIORITY_RESULT_PREFIXES):
            emit('DIR ' + relative(path, root))
            for item in sorted(path.iterdir())[:35]:
                if item.is_file():
                    emit('  ' + item.name + '  bytes=' + str(item.stat().st_size))
                else:
                    emit('  ' + item.name + '/')
    emit('')


def inspect_candidates(root, emit):
    emit('=== HIGH-RELEVANCE PRIOR OUTPUT TABLE SCHEMAS ===')
    candidates = find_candidates(root)
    emit('showing=' + str(len(candidates)) + ' max=' + str(MAX_TOTAL_CANDIDATES))
    for index, path in enumerate(candidates):
        emit('TABLE ' + relative(path, root) + '  bytes=' + str(path.stat().st_size))
        if index < MAX_PREVIEW_FILES:
            emit('  ' + preview_table(path))
        else:
            emit('  [preview omitted: cap reached]')
    emit('')
    emit('=== CLINICAL SOURCE HINTS ===')
    for phase in ('phase4', 'phase5', 'phase7'):
        src_dir = root / 'src' / phase
        if not src_dir.is_dir():
            emit('MISSING ' + relative(src_dir, root))
            continue
        hints = []
        for path in sorted(src_dir.glob('*.py')):
            if any(k in path.name.lower() for k in ('cornforth', 'clinical', 'cf01', 'cf146', 'sputum', '07i', '03h', 'patient')):
                hints.append(path)
        emit('PHASE ' + phase + ': ' + str(len(hints)) + ' relevant source names')
        for path in hints[:40]:
            emit('  ' + relative(path, root))
    emit('')


def get_report(root):
    lines = []
    def emit(value):
        lines.append(str(value))
    emit('CFSTATE_PHASE8_08N_INPUT_INVENTORY_READ_ONLY')
    emit('ROOT ' + str(root))
    emit('VERIFY This report does not assert clinical comparisons can be built from the files shown.')
    emit('VERIFY No arrays, calibration or CF specificity claims should be inferred from table headers alone.')
    emit('')
    scan_scripts(root, emit)
    inspect_known_outputs(root, emit)
    inspect_candidates(root, emit)
    emit('END_INVENTORY')
    return '\n'.join(lines) + '\n'


def self_test():
    assert norm_ext(Path('a.csv.gz')) == 'csv.gz'
    assert readable_table(Path('a.tsv.gz'))
    assert not readable_table(Path('a.py'))
    assert 'clinical' in KEYWORDS
    print('CFSTATE_08N_INPUT_INVENTORY_SELF_TEST_PASS')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='CFstate project root (default: cwd)')
    parser.add_argument('--save', type=Path, help='Write output to NEW file; fails if it already exists')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    root = args.root.expanduser().resolve()
    if not (root / 'src' / 'phase8').is_dir():
        parser.error('Not a CFstate project root: src/phase8 missing at ' + str(root))
    report = get_report(root)
    if args.save:
        savepath = args.save.expanduser()
        savepath.parent.mkdir(parents=True, exist_ok=True)
        # x means exclusive creation, so no existing report is overwritten.
        with savepath.open('x', encoding='utf-8') as stream:
            stream.write(report)
        print('INVENTORY_SAVED ' + str(savepath))
    else:
        sys.stdout.write(report)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
