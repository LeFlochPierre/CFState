#!/usr/bin/env python3
"""CF-State housekeeping: a read-only inventory and actionable tidy-up plan.

Does NOT delete, rename, move, patch, or overwrite ANY project source,
data, external repository, result or existing log. It writes a uniquely named
report folder under maintenance/cleanup_runs/ only.

Run: python tools/organize_cfstate_inventory.py --root .
Test: python tools/organize_cfstate_inventory.py --self-test
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile

SCAN_EXCLUDE = {'.git', '.hg', '.svn', '__pycache__', '.ipynb_checkpoints', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
CODE_EXT = {'.py', '.sh', '.r', '.rmd', '.R', '.Rmd', '.ipynb'}
PATH_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(?:data|external|references|results|src|logs)/[A-Za-z0-9_./+\-]+")
STUDIES = [
    ('aeruPRECISE411', 'REFERENCE', 'Frozen 411-sample, 116-iModulon lab reference; includes GSE138731', 'FROZEN_CORE'),
    ('GSE138731', 'PERTURBATION', 'Farnesol; overlap with aeruPRECISE411; not external iModulon validation', 'PROVISIONAL_PRIMARY_REFERENCE_OVERLAP'),
    ('GSE167137', 'PERTURBATION', 'Meropenem mono/co-culture; no accession/GSM match to reference', 'PROVISIONAL_PRIMARY'),
    ('GSE179150', 'PERTURBATION', 'PA2504 knockout vs WT; no accession/GSM match to reference', 'PROVISIONAL_PRIMARY'),
    ('GSE192340', 'PERTURBATION', 'DDDY peptide; 785 exact-zero effects in 09t full-gene response', 'PROVISIONAL_PRIMARY'),
    ('GSE190194', 'PERTURBATION', 'Long-passage adaptation, not acute antibiotic effects', 'SECONDARY'),
    ('GSE223663', 'PERTURBATION', 'Biofilm/planktonic PA0285 knockout; overlaps earlier CF-State data', 'SECONDARY_HISTORICAL_OVERLAP'),
    ('GSE178622', 'PERTURBATION', 'DATS matrix contains fractional counts; measurement units unresolved', 'QUARANTINED'),
    ('GSE119356', 'CLINICAL', 'CF explanted lung-region samples; one person across multiple regions', 'CLINICAL_DISCOVERY'),
    ('PRJEB24688', 'CLINICAL', 'Danish CF airway patients and culture comparisons; patient units matter', 'CLINICAL_DISCOVERY'),
    ('PRJNA576508', 'CLINICAL', 'Atlanta CF airway transcriptomes; independent-patient identities unresolved', 'CLINICAL_VALIDATION_CANDIDATE'),
    ('SRP135669', 'CLINICAL_COMPARATOR', 'Cornforth CF sputum versus non-CF wound/burn; site confounded', 'CLINICAL_COMPARATOR_WITH_CONFOUNDING'),
    ('SRP173673', 'CLINICAL', 'CF01 longitudinal samples, one patient', 'CLINICAL_FOLLOWUP'),
    ('GSE139943', 'CLINICAL', 'CF146 longitudinal samples, one patient', 'CLINICAL_FOLLOWUP'),
    ('PRJNA1006084', 'EX_VIVO', 'Neff metal supplementation; 17 paired donors, lab PAO1 spiked into sputum', 'CONTROLLED_PERTURBATION'),
]
FROZEN_PATHS = [
 'external/aeruprecise/precise411/data_files/M.csv',
 'external/aeruprecise/precise411/data_files/A.csv',
 'external/aeruprecise/precise411/data_files/sample_table.csv',
 'results/phase4/audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
 'results/phase4/audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
 'results/phase3/audit_02e_PA14_projection_universe/frozen_PA14_projection_map.csv',
 'results/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze',
 'results/phase9/audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze',
 'results/phase9/model_09s_primary_fourstudy_fewshot_extension',
 'results/phase9/analysis_09t_frozen_imodulon_perturbation_alignment',
 'results/phase9/analysis_09u_targeted_imodulon_robustness',
]


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(piece)
    return h.hexdigest()


def csv_write(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf8') as handle:
        wr = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)


def role(rel: str) -> str:
    if rel.startswith('external/'):
        return 'VENDORED_OR_EXTERNAL_DO_NOT_MOVE'
    if rel.startswith('data/raw/'):
        return 'RAW_IMMUTABLE'
    if rel.startswith('data/processed/'):
        return 'DERIVED_DATA_CHECK_PROVENANCE'
    if rel.startswith('data/metadata/'):
        return 'METADATA_KEEP'
    if rel.startswith('references/'):
        return 'REFERENCE_KEEP'
    if rel.startswith('results/'):
        return 'RESULT_DO_NOT_REMOVE_WITHOUT_REPRODUCING'
    if rel.startswith('logs/'):
        return 'LOG_REVIEW_FOR_ARCHIVE'
    if rel.startswith('src/audit/'):
        return 'LEGACY_PHASE1_CODE_KEEP_UNTIL_ARCHIVE'
    if rel.startswith('src/phase'):
        return 'HISTORICAL_PHASE_CODE_REVIEW'
    if rel.startswith('src/'):
        return 'SOURCE_CODE_REVIEW'
    if rel.startswith('notes/'):
        return 'NOTE_REVIEW'
    return 'ROOT_OR_OTHER_REVIEW'


def source_family(rel: str) -> str:
    stem = Path(rel).stem.lower()
    stem = re.sub(r'fixed\d*$', '', stem)
    stem = re.sub(r'(?<=\d)(?:b|c|d|e|f)(?=_)', '', stem)
    return str(Path(rel).parent / stem)


def scan(root: Path):
    files = []
    code = []
    cache_dirs = []
    skipped_subtrees = []
    broken_symlinks = []
    for cur, dirs, names in os.walk(root, topdown=True, followlinks=False):
        base = Path(cur)
        relative_base = base.relative_to(root)
        dkeep = []
        for name in sorted(dirs):
            sub = base / name
            rp = sub.relative_to(root).as_posix()
            if sub.is_symlink():
                skipped_subtrees.append(rp+' [symlink]')
            elif name in SCAN_EXCLUDE:
                if name != '.git':
                    cache_dirs.append(rp)
                skipped_subtrees.append(rp)
            elif (relative_base == Path('maintenance') and name == 'cleanup_runs') or rp == 'maintenance/cleanup_runs':
                skipped_subtrees.append(rp+' [previous maintenance output]')
            else:
                dkeep.append(name)
        dirs[:] = dkeep
        for filename in sorted(names):
            p = base / filename
            rel = p.relative_to(root).as_posix()
            if rel.startswith('maintenance/cleanup_runs/'):
                continue
            try:
                st = p.lstat()
            except OSError:
                continue
            symlink = p.is_symlink()
            if symlink and not p.exists():
                broken_symlinks.append(rel)
            category = rel.split('/', 1)[0]
            ext = p.suffix.lower()
            is_code = category == 'src' and ext in {'.py', '.sh', '.r', '.rmd', '.ipynb'}
            sha = ''
            mentions = []
            if is_code and not symlink and st.st_size <= 4*1024*1024:
                try:
                    sha = checksum(p)
                    content = p.read_text(encoding='utf8', errors='replace')
                    mentions = sorted(set(m.group().rstrip('.-+/') for m in PATH_PATTERN.finditer(content)))
                except (OSError, UnicodeError):
                    pass
            row = {
                'path': rel, 'size_bytes': 0 if symlink else st.st_size,
                'area': category, 'extension': ext, 'role': role(rel),
                'is_symlink': symlink,
            }
            files.append(row)
            if is_code:
                code.append({**row, 'sha256': sha,
                             'path_mentions_count': len(mentions),
                             'path_mentions': mentions,
                             'version_or_repair_name': bool(re.search(r'(fixed\d*|repair|backup|_old|copy)', filename, re.I)),
                             'family': source_family(rel)})
    return files, code, cache_dirs, skipped_subtrees, broken_symlinks


def format_bytes(size: int) -> str:
    x = float(size)
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB']:
        if x < 1024 or unit == 'TiB':
            return f'{x:,.1f} {unit}'
        x /= 1024
    return str(size)


def guide(stats, checks, code_by_phase, refs, duplicates, cache_dirs):
    lines = [
        '# CF-State — project map and safe cleanup plan',
        '',
        '> Generated from a read-only filesystem inventory; the scientific decisions below are a curated starting point, not a claim that every local output was inspected.',
        '',
        '## CURRENT SCIENTIFIC TARGET',
        '',
        'Characterise reproducible, interpretable P. aeruginosa physiological programmes in direct CF airway bacterial transcriptomes using frozen aeruPRECISE iModulons, independent clinical cohorts and controlled perturbation evidence.',
        '',
        '## THREE KEEP RULES',
        '',
        '1. **ACTIVE**: frozen programme reference, clinical projections/patient metadata, and forthcoming programme reproducibility/uncertainty analyses.',
        '2. **FROZEN EVIDENCE**: completed phase outputs, raw processing provenance, PA14 mapping, controlled laboratory data and perturbation-response matrices. Never overwrite.',
        '3. **LEGACY**: previous biofilm classifiers, residual-axis experiments, graph-prediction models and repaired/abandoned scripts. Archive only after dependency/path checks; do not delete.',
        '',
        '## RESEARCH DECISIONS ALREADY MADE',
        '',
        '- Phase 1: a universal biofilm probability is not the main scientific target.',
        '- Phases 2–3: frozen aeruPRECISE411 reference (116 iModulons), approximately 108 robust programmes, PA14 projection map.',
        '- Phases 4–8: clinical CF airway transcriptome processing and conservative CF-associated programme candidate; explicit patient/study overlaps and confounding.',
        '- Phase 9: curated regulatory graph models 09l/09q/09s did **not** consistently beat simple no-network baselines.',
        '- Phase 9t/u: descriptive programme-response alignment. Meropenem–TagR1 is a robust known-response positive control within one study; farnesol–AtuR overlaps reference training; DDDY alignment weak.',
        '- Current next milestone: leakage-controlled independent-patient clinical programme reproducibility and uncertainty analysis, not another 09q-style graph predictor.',
        '',
        '## CANONICAL ASSETS TO PROTECT (presence as scanned)',
        '',
    ]
    for r in checks:
        lines.append(f"- **{'FOUND' if r['exists'] else 'NOT LOCATED'}** `{r['path']}`")
    lines.extend(['', '## PROJECT SNAPSHOT', '',
                  f"- Files inventoried: **{stats['n_files']:,}** (excluding .git and cache directory internals)",
                  f"- Apparent non-cache bytes inventoried: **{format_bytes(stats['total_bytes'])}** (not filesystem allocated size)",
                  f"- Source scripts inventoried: **{stats['n_code']}**",
                  f"- Byte-identical script duplicate groups: **{len(duplicates)}**",
                  f"- Cache directories to review: **{len(cache_dirs)}**",
                  '', '### Source scripts per group', ''])
    for key, val in sorted(code_by_phase.items()):
        lines.append(f'- `{key}`: {val} script(s)')
    lines.extend(['', '## SUGGESTED FUTURE PRESENTATION (DO NOT MOVE EXISTING FILES YET)', '',
                  '```text',
                  'CFstate/',
                  '  README.md                        # one-page project entrypoint',
                  '  docs/                            # objectives, evidence, decisions, data registry',
                  '  src/                             # old paths preserved until verified',
                  '  data/raw/                        # immutable original inputs',
                  '  data/processed/                  # reproducible derived products',
                  '  external/                        # pinned reference repos; never hand-edit',
                  '  references/                      # genomes, mappings, annotations',
                  '  results/                         # frozen scientific evidence',
                  '  logs/                            # audit/debug provenance',
                  '  maintenance/cleanup_runs/        # this one-time housekeeping report',
                  '```',
                  '', '## FOUR-STAGE ACTUAL CLEANUP', '',
                  '1. **Pin**: commit source/config/metadata (if Git available), record the frozen input manifests, ensure raw data is backed up. Don\'t commit giant raw data or secrets.',
                  '2. **Select**: identify each active clinical entrypoint and its exact dependencies. Read `SOURCE_PATH_MENTIONS.csv` and validate imported modules and dynamically built paths.',
                  '3. **Archive**: move legacy *code* only after a tested compatibility plan (Git `git mv` where tracked); preserve a versioned original path or wrapper if downstream scripts refer to it.',
                  '4. **Verify**: compare frozen file hashes/checksums; rerun a small set of reproducibility/smoke tests. Only then consider log/cache pruning.',
                  '', '## WHAT NOT TO DO', '',
                  '- Do not `rm -rf results`, `rm -rf data/raw`, or delete `src/audit` in bulk.',
                  '- Do not merge all study CSVs by taking their intersection without provenance/units.',
                  '- Do not rename phase output directories while executable scripts still use them.',
                  '- Do not presume suffixes `_repair` or `_FIXED` mean the file is obsolete; some are **the** authoritative version.',
                  '- Do not automatically treat repeated specimens/regions as independent patients.',
                  '', '## RELEVANT REPORTS', '',
                  '`FILE_INVENTORY.csv`, `SOURCE_SCRIPTS.csv`, `SOURCE_PATH_MENTIONS.csv`, `EXACT_SCRIPT_DUPLICATES.csv`, `REVIEW_CANDIDATES.csv`, `DATASET_REGISTER.csv`, `SIZE_BY_AREA.csv`, `QC_STATUS.json`.',
                  '', '## DATASET IDENTITIES', '',
                  'See `DATASET_REGISTER.csv`. It includes status labels based on completed CF-State analyses; path matches are filename/path evidence, not proof of independent participant identities.',
                  '', '## FIRST CLEANUP DECISION TO TAKE', '',
                  '**Do not reorganise all nine phases.** Make one clear current clinical workflow index, and move only tested/dependency-free legacy scripts after inspecting the inventory. Keep the frozen data and results at their current paths initially.',
                  ''])
    return '\n'.join(lines)


def inventory(root: Path, output_parent: Path):
    if not root.is_dir():
        raise ValueError('Root is not a directory: '+str(root))
    files, code, cache_dirs, skips, broken = scan(root)
    now = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = output_parent / ('cleanup_'+now)
    output.mkdir(parents=True, exist_ok=False)

    csv_write(output/'FILE_INVENTORY.csv', files, ['path','size_bytes','area','extension','role','is_symlink'])
    csv_write(output/'SOURCE_SCRIPTS.csv', code, ['path','size_bytes','role','sha256','path_mentions_count','version_or_repair_name','family'])
    file_set = {i['path'] for i in files}
    refs = [{'path': x, 'exists': (root/x).exists(), 'kind':'directory' if (root/x).is_dir() else 'file'} for x in FROZEN_PATHS]
    csv_write(output/'FROZEN_ASSET_CHECKS.csv', refs, ['path','exists','kind'])

    refs_rows = []
    for entry in code:
        for mentioned in entry['path_mentions']:
            refs_rows.append({'source_script':entry['path'], 'mentioned_path':mentioned,
                              'exists_as_exact_path':(root/mentioned).exists(),
                              'note':'lexical reference only; dynamic paths and imports not resolved'})
    csv_write(output/'SOURCE_PATH_MENTIONS.csv', refs_rows,
              ['source_script','mentioned_path','exists_as_exact_path','note'])

    sha_groups = defaultdict(list)
    for item in code:
        if item['sha256']:
            sha_groups[item['sha256']].append(item['path'])
    duplicate_groups = [sorted(group) for group in sha_groups.values() if len(group)>1]
    duplicate_rows = [{'sha256':digest, 'path':p, 'n_identical_files':len(paths)}
                      for digest,paths in sha_groups.items() if len(paths)>1 for p in sorted(paths)]
    csv_write(output/'EXACT_SCRIPT_DUPLICATES.csv', duplicate_rows,
              ['sha256','path','n_identical_files'])

    review = []
    for entry in code:
        if entry['version_or_repair_name']:
            review.append({'path':entry['path'], 'category':'VERSION_OR_REPAIR_REVIEW',
                           'advice':'Check lineage and downstream uses; DO NOT auto-delete'})
    for path in cache_dirs:
        review.append({'path':path,'category':'CACHE_DIRECTORY',
                       'advice':'Usually regenerable; clear only after review and backup'})
    for path in broken:
        review.append({'path':path,'category':'BROKEN_SYMLINK',
                       'advice':'Check external dependency or old path'})
    csv_write(output/'REVIEW_CANDIDATES.csv', review, ['path','category','advice'])

    byarea = defaultdict(lambda: [0,0])
    phases = Counter()
    for f in files:
        byarea[f['area']][0] += 1
        byarea[f['area']][1] += int(f['size_bytes'])
    for c in code:
        key = str(Path(c['path']).parent)
        phases[key] += 1
    size_rows = [{'area':k,'n_files':n,'size_bytes':s,'size_readable':format_bytes(s)}
                 for k,(n,s) in sorted(byarea.items(), key=lambda x:-x[1][1])]
    csv_write(output/'SIZE_BY_AREA.csv',size_rows,['area','n_files','size_bytes','size_readable'])

    dataset_rows = []
    for dataset,layer,note,status in STUDIES:
        found = [f['path'] for f in files if dataset.lower() in f['path'].lower()]
        dataset_rows.append({'dataset_id':dataset,'layer':layer,'status_from_prior_research':status,
                             'interpretation_and_caution':note, 'n_path_matches':len(found),
                             'example_matching_paths':' | '.join(found[:3]),
                             'data_identity_independently_verified_by_this_scan':False})
    csv_write(output/'DATASET_REGISTER.csv',dataset_rows,
              ['dataset_id','layer','status_from_prior_research','interpretation_and_caution',
               'n_path_matches','example_matching_paths','data_identity_independently_verified_by_this_scan'])
    stats = {'n_files':len(files), 'total_bytes':sum(f['size_bytes'] for f in files),
             'n_code':len(code), 'n_path_mentions':len(refs_rows),
             'n_duplicate_script_groups':len(duplicate_groups),
             'n_cache_directories':len(cache_dirs),
             'n_broken_symlinks':len(broken),
             'n_frozen_asset_checks_present':sum(i['exists'] for i in refs),
             'n_frozen_asset_checks_total':len(refs),
             'mutation_policy':'NO_EXISTING_PROJECT_FILES_CHANGED'}
    (output/'QC_STATUS.json').write_text(json.dumps(stats,indent=2)+'\n',encoding='utf8')
    (output/'PROJECT_MAP_AND_CLEANUP_PLAN.md').write_text(guide(stats, refs, phases, refs_rows, duplicate_groups, cache_dirs),encoding='utf8')

    print('='*82)
    print('CF-STATE REPOSITORY CLEANUP | READ-ONLY INVENTORY AND NO-DELETION PLAN')
    print('='*82)
    print('Root:                ',root)
    print('Files:               ',f"{stats['n_files']:,}")
    print('Apparent bytes:      ',format_bytes(stats['total_bytes']))
    print('Source scripts:      ',stats['n_code'])
    print('Path mentions:       ',stats['n_path_mentions'],'(lexical; not guaranteed dependencies)')
    print('Exact code duplicates:',stats['n_duplicate_script_groups'],'groups')
    print('Cache directories:   ',stats['n_cache_directories'],'(only listed, untouched)')
    print('Frozen paths found:  ',str(stats['n_frozen_asset_checks_present'])+'/'+str(stats['n_frozen_asset_checks_total']))
    print('\nCODE DISTRIBUTION:')
    for k,v in sorted(phases.items()):
        print(' ',f'{k:24}',v)
    print('\nLARGEST TOP-LEVEL AREAS:')
    for x in size_rows[:10]:
        print(' ',f"{x['area']:16}",f"{x['n_files']:>7,}",format_bytes(x['size_bytes']))
    if duplicate_groups:
        print('\nEXACT SOURCE DUPLICATES (up to five groups):')
        for group in duplicate_groups[:5]:
            print(' ', ' | '.join(group[:4]))
    print('\nDATASET REGISTRY:')
    for row in dataset_rows:
        print(' ',f"{row['dataset_id']:18}", f"{row['status_from_prior_research']:42}",f"matching paths={row['n_path_matches']}")
    print('\nREPORTS:',output)
    print('OPEN FIRST: PROJECT_MAP_AND_CLEANUP_PLAN.md')
    print('NO PROJECT FILES MOVED, RENAMED, DELETED OR MODIFIED.')
    return output, stats


def self_test():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)/'CFstate'
        (root/'src/audit').mkdir(parents=True)
        (root/'src/phase9').mkdir(parents=True)
        (root/'data/raw').mkdir(parents=True)
        (root/'results/phase9').mkdir(parents=True)
        (root/'src/phase9/__pycache__').mkdir()
        (root/'src/audit/test.py').write_text('from pathlib import Path\nx = "results/phase9/test.csv"\n',encoding='utf8')
        (root/'src/phase9/copy.py').write_text('from pathlib import Path\nx = "results/phase9/test.csv"\n',encoding='utf8')
        (root/'src/phase9/copy_FIXED.py').write_text('from pathlib import Path\nx = "results/phase9/test.csv"\n',encoding='utf8')
        (root/'data/raw/a.csv').write_text('gene,s1\nPA0001,1\n',encoding='utf8')
        (root/'results/phase9/test.csv').write_text('ok\n',encoding='utf8')
        before = {p.relative_to(root).as_posix():checksum(p) for p in root.rglob('*') if p.is_file()}
        out, stats = inventory(root,root/'maintenance/cleanup_runs')
        after = {p.relative_to(root).as_posix():checksum(p) for p in root.rglob('*') if p.is_file() and 'maintenance/' not in p.relative_to(root).as_posix()}
        assert before == after, 'input sources changed'
        assert stats['n_code'] == 3
        assert stats['n_duplicate_script_groups'] == 1
        assert stats['n_path_mentions'] == 3
        assert stats['n_cache_directories'] == 1
        assert (out/'PROJECT_MAP_AND_CLEANUP_PLAN.md').is_file()
        assert (out/'DATASET_REGISTER.csv').is_file()
        out2, stats2 = inventory(root,root/'maintenance/cleanup_runs')
        assert out2 != out, 'second report overwrote first'
        assert stats2['n_files'] == stats['n_files'], 'previous reports should not pollute scan'
        print('\nSELF_TEST_PASSED: no source mutation, exact duplicate detection, path hints, cache listing, unique reports.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('.'),help='CF-State repository root')
    parser.add_argument('--output-parent',type=Path,default=None,help='default: ROOT/maintenance/cleanup_runs')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    root=args.root.resolve()
    parent=args.output_parent.resolve() if args.output_parent else root/'maintenance/cleanup_runs'
    try:
        inventory(root,parent)
    except Exception as exc:
        print('CLEANUP_INVENTORY_FAILED:',exc,file=sys.stderr)
        raise


if __name__=='__main__':
    main()
