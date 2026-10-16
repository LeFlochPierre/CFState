#!/usr/bin/env python3
"""Create a clean, documented entry point to the existing CF-State project.

Consumes the latest inventory from organize_cfstate_inventory.py.
It NEVER moves/deletes/rewrites source, data, references, external files, or results.
All generated files are new (exclusive-create); existing documents are not overwritten.

Usage:
  python tools/build_cfstate_clean_frontdoor.py --self-test
  python tools/build_cfstate_clean_frontdoor.py --root .          # preview
  python tools/build_cfstate_clean_frontdoor.py --root . --apply  # create docs
"""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import date
import io
import json
from pathlib import Path
import tempfile

REQUIRED = (
    'QC_STATUS.json', 'FILE_INVENTORY.csv', 'SOURCE_SCRIPTS.csv',
    'FROZEN_ASSET_CHECKS.csv', 'DATASET_REGISTER.csv',
)


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def csv_text(rows, fields):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fields, lineterminator='\n', extrasaction='ignore')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def find_inventory(root: Path, requested: str | None):
    if requested:
        directory = Path(requested).resolve()
    else:
        parent = root / 'maintenance' / 'cleanup_runs'
        candidates = sorted(parent.glob('cleanup_*'), reverse=True) if parent.is_dir() else []
        directory = next((p for p in candidates if p.is_dir() and all((p / n).is_file() for n in REQUIRED)), None)
        if directory is None:
            raise FileNotFoundError('No complete cleanup inventory in maintenance/cleanup_runs')
    missing = [n for n in REQUIRED if not (directory / n).is_file()]
    if missing:
        raise FileNotFoundError('Inventory is incomplete: ' + ', '.join(missing))
    return directory


def category(path):
    p = path.lower()
    if p.startswith('src/phase9/analysis_09t_') or p.startswith('src/phase9/analysis_09u_'):
        return 'RECENT_PROGRAMME_EVIDENCE', 'Frozen iModulon interpretation; not yet clinical validation'
    if p.startswith('src/phase9/audit_09p_') or p.startswith('src/phase9/audit_09r_'):
        return 'PERTURBATION_INPUT_PROVENANCE', 'Check the exact version used before treating as canonical'
    if p.startswith('src/phase9/model_09l_') or p.startswith('src/phase9/model_09q_') or p.startswith('src/phase9/model_09s_'):
        return 'HISTORICAL_NEGATIVE_BENCHMARK', 'Completed exploratory model; retain for reproducibility'
    if p.startswith('src/phase2/'):
        return 'REFERENCE_PROVENANCE', 'Frozen reference development; do not rename blindly'
    if p.startswith('src/phase3/'):
        return 'PA14_TRANSFER_PROVENANCE', 'Gene mapping and transferability history'
    if p.startswith('src/phase4/'):
        return 'CLINICAL_PROVENANCE_REVIEW', 'Clinical processing/projection scripts; identify active entrypoints manually'
    if p.startswith('src/audit/'):
        return 'INITIAL_BIOFILM_HISTORY', 'Legacy code candidate, dependent downstream work may exist'
    if p.startswith('src/phase9/'):
        return 'PHASE9_HISTORICAL_OR_SUPPORT', 'Do not determine canonical status by filename alone'
    return 'HISTORICAL_OR_SUPPORT_REVIEW', 'Not yet assigned an authoritative workflow role'


def markdown_table(rows, columns):
    def safe(x):
        return str(x).replace('|', '\\|').replace('\n', ' ')
    out = ['| ' + ' | '.join(columns) + ' |', '| ' + ' | '.join('---' for _ in columns) + ' |']
    for row in rows:
        out.append('| ' + ' | '.join(safe(row.get(c, '')) for c in columns) + ' |')
    return '\n'.join(out)


def make_documents(root, report):
    meta = json.loads((report / 'QC_STATUS.json').read_text(encoding='utf-8'))
    files = read_csv(report / 'FILE_INVENTORY.csv')
    scripts = read_csv(report / 'SOURCE_SCRIPTS.csv')
    frozen = read_csv(report / 'FROZEN_ASSET_CHECKS.csv')
    datasets = read_csv(report / 'DATASET_REGISTER.csv')
    if len(files) != meta.get('n_files') or len(scripts) != meta.get('n_code'):
        raise ValueError('Inventory CSV rows do not match its QC summary')
    script_index = []
    for r in scripts:
        label, note = category(r['path'])
        script_index.append({
            'path': r['path'], 'category': label, 'note': note,
            'version_or_repair_name': r.get('version_or_repair_name', ''),
            'path_mentions_count': r.get('path_mentions_count', ''),
            'sha256': r.get('sha256', ''),
        })
    script_index.sort(key=lambda x: (x['category'], x['path']))
    sizes = sorted(files, key=lambda r: int(r.get('size_bytes') or 0), reverse=True)
    size_review = []
    for r in sizes[:60]:
        size = int(r.get('size_bytes') or 0)
        size_review.append({'path': r['path'], 'size_GiB': f'{size/1024**3:.4f}',
                            'size_bytes': size, 'role': r.get('role', ''),
                            'action': 'REVIEW_ONLY_NEVER_AUTO_DELETE'})
    phases = Counter(r['path'].rsplit('/', 1)[0] for r in scripts)
    categories = Counter(r['category'] for r in script_index)
    existing = {r['path'] for r in files}
    created_at = date.today().isoformat()
    output = {}

    output['README.md'] = f'''# CF-State — Physiological programmes in CF airway *Pseudomonas aeruginosa*

> Project navigator created {created_at}. This is a **navigation layer**, not a rebuilt pipeline. Prior data, results, and source directories remain at their original locations.

## Current research question

Which interpretable transcriptional programmes are reliably observable in direct clinical CF airway bacterial transcriptomes, how reproducible are they across independent patients/cohorts, and what do controlled perturbations tell us about their biological meaning?

## Start here

- [Current project status](docs/PROJECT_STATUS.md) — aims, evidence, unanswered questions.
- [Workflow and code map](docs/WORKFLOWS.md) — where the usable results are; historical scripts remain traceable.
- [Frozen assets](docs/FROZEN_ASSETS.md) — reference inputs and outputs that must not be silently changed.
- [Dataset registry](docs/DATASET_REGISTRY.csv) — per-study role, caveats and provenance status.
- [Research decisions](docs/RESEARCH_HISTORY.md) — tested alternatives, including negative findings.
- [Cleanup and storage plan](docs/CLEANUP_PLAN.md) — the safe way to consolidate files without losing reproducibility.
- [Full script index](docs/CODE_INDEX.csv) — all {len(scripts)} indexed scripts and review labels.
- [Largest-file review list](docs/STORAGE_REVIEW.csv) — space-consuming files, **not deletion candidates**.

## Repository conventions

- `external/`: published or third-party resources; treat as immutable.
- `references/`: genomes and other mapping references.
- `data/raw/`: original source data, preserve provenance.
- `data/processed/`: intermediate and frozen derived matrices; status varies.
- `results/`: frozen audits and experiments; do not overwrite.
- `src/`: historical and current research code, **paths intentionally unchanged**.
- `logs/`: recorded runs.
- `maintenance/cleanup_runs/`: inventories (latest input: `{report.relative_to(root).as_posix() if report.is_relative_to(root) else str(report)}`).

## Next scientific milestone

Re-evaluate the frozen clinical CF programme profiles across independent patients and cohorts, including detection/missingness uncertainty and non-CF specificity. Use the controlled-perturbation evidence as *interpretation/positive controls*, not as proof of CF causality.
'''

    output['docs/PROJECT_STATUS.md'] = '''# Current research status — CF-State

## Scientific objective

Develop a reproducible, interpretable, uncertainty-aware approach to characterising the physiological programmes of *Pseudomonas aeruginosa* in direct CF airway transcriptomes.

## Primary evidence layers

1. **Frozen aeruPRECISE411:** 411 reference samples, 116 iModulons. Existing PA14 transfer/mapping sensitivity work; do not silently recompute or retrain the reference.
2. **Clinical CF airway:** earlier Denmark and Atlanta processing, sample-level projections and patient-aware comparisons. Consult exact provenance before claiming patient independence or CF-specificity.
3. **Controlled perturbations:** frozen oxygen, metal, genetic and antimicrobial response data; 09p/09r provide recent candidate treatment-control contrasts with GEO-title corroboration, not complete raw-read processing validation.

## What currently stands up

- The published iModulon representation contains interpretable known biological programmes.
- 09t/09u recover meropenem-associated TagR1 / H1-T6SS across two conditions **within one experiment**, with sensitivity to influential genes checked.
- Farnesol / AtuR provides a positive-control association, but GSE138731 samples are part of aeruPRECISE411 and are **not independent reference validation**.
- Cross-study perturbation resources have been carefully indexed and candidate sample group assignments reconciled.

## Negative / inconclusive findings

- A universal biofilm-probability model was not the appropriate transferable target.
- Curated-sign regulatory-response models (09l/09q/09s) did not consistently beat simple no-network / anchor-mean baselines.
- A 09t alignment correlation is **not** an iModulon activity change, direct causal effect, or clinical CF-specific programme.
- The limited clinical datasets cannot establish universal CF-specific bacterial states without independent-patient and matched-comparator controls.

## Next locked question (not yet answered)

Which frozen physiological programme profiles reproducibly describe direct CF airway bacteria across genuinely independent patients and cohorts, accounting for sequencing depth, missing genes, strain, sampling site, and study effects?

**No new clinical validity, novel causal mechanism or prediction gain is claimed by this housekeeping document.**
'''

    counts = '\n'.join(f'- `{phase}`: {n} scripts' for phase, n in sorted(phases.items()))
    cat_counts = '\n'.join(f'- **{key}**: {n}' for key, n in sorted(categories.items()))
    landmarks = [
        ('reference', 'external/aeruprecise/precise411/data_files/M.csv'),
        ('reference', 'external/aeruprecise/precise411/data_files/A.csv'),
        ('clinical', 'results/phase4/audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv'),
        ('clinical', 'results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'),
        ('perturbation', 'results/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze'),
        ('perturbation', 'results/phase9/audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze'),
        ('interpretation', 'results/phase9/analysis_09t_frozen_imodulon_perturbation_alignment'),
        ('interpretation', 'results/phase9/analysis_09u_targeted_imodulon_robustness'),
    ]
    landmark_lines = '\n'.join(f'- **{layer}:** `{path}` — ' + ('present' if (root / path).exists() else 'NOT FOUND') for layer, path in landmarks)
    output['docs/WORKFLOWS.md'] = f'''# Workflow navigation (existing files are unchanged)

## Where to look first

{landmark_lines}

These landmarks identify previously used outputs, **not** a fully audited or automatically executable pipeline. Always read the related QC/status JSON and the generating script before reusing the contents.

## Script distribution

{counts}

## Code role labels (review only)

{cat_counts}

See [CODE_INDEX.csv](CODE_INDEX.csv) for all {len(scripts)} scripts. Filename-based role labels are a starting point; `*_repair*` and `*_FIXED*` can be the correct production versions, not clutter.

## Reproduction sequence for the next clinical analysis

1. Load the frozen reference and canonical gene mappings; validate the expected shapes and hashes.
2. Resolve patient/study identities and existing clinical freeze outputs; never count technical replicates as independent patients.
3. Run the clinical projection and uncertainty checks using existing scripts or new versioned scripts as necessary.
4. Evaluate transfer across held-out patients and studies, avoiding training/reference overlap.
5. Only then join lab perturbation programme interpretations; maintain acute/long-term and known overlap flags.

No historic scripts were moved, rewritten, or designated officially obsolete by this step.
'''

    freeze_lines = []
    for item in frozen:
        ok = str(item.get('exists', '')).lower() == 'true'
        freeze_lines.append(f"- [{'x' if ok else ' '}] `{item['path']}` — {'located' if ok else 'MISSING / RECHECK'}")
    output['docs/FROZEN_ASSETS.md'] = '# Protected frozen assets\n\nAll paths below come from the prior filesystem inventory. Presence does not verify file contents or scientific correctness. **Do not overwrite or move without versioning and dependency tests.**\n\n' + '\n'.join(freeze_lines) + '\n'

    output['docs/RESEARCH_HISTORY.md'] = '''# Research history and decisions

This is a concise *decision record*, not a bibliography or a comprehensive reproduction protocol.

| Workstream | Established | Current disposition |
|---|---|---|
| Initial biofilm/planktonic modelling | Large differences across experiments; universal biofilm score not established | Historical evidence |
| Frozen aeruPRECISE + PA14 mapping | Existing 116-iModulon reference; transfer/mapping audits | Preserve and reuse |
| Direct clinical CF profiling | Denmark, Atlanta and other clinical processing and comparisons; clinical reproducibility/specificity unresolved | Main active question |
| Curated regulation / metal / Δanr experiments | Useful priors and controlled evidence; negative predictive comparisons | Retain as negative-model benchmarks |
| Expanded EPathDash perturbations 09m–09r | 34 studies inventoried, new reconciled contrasts; DATS fractional-count QC quarantine | Frozen evidence, not automatically approved |
| 09q / 09s response completion | Signed curated graph did not consistently outperform no graph or simple anchor mean | Do not optimise this formulation further without new hypothesis |
| 09t / 09u iModulon alignment | TagR1/meropenem survives influential-gene removal; AtuR/farnesol overlaps pretraining; DDDY weak | Biological positive controls and interpretation |

## Evidence limits

- GSE138731 farnesol **was included** in the aeruPRECISE411 reference.
- Meropenem monoculture and coculture are **within one GEO study**, not independent replications.
- GSE178622 fractional-count matrix is quarantined pending verification of units.
- Correlations between gene responses and loadings are not calibrated TF/iModulon activity changes.
- Published biology recovered by 09t does not by itself establish novelty.
- Earlier clinical processing outcomes are not the same as a validated multi-patient CF-specific physiological state.
'''

    output['docs/CLEANUP_PLAN.md'] = f'''# Safe repository cleanup — step-by-step

## Baseline from latest inventory

- Files: **{meta['n_files']:,}**; source scripts: **{meta['n_code']}**.
- Apparent project size: approximately **121.8 GiB** in the supplied real inventory; actual numbers may vary after future runs.
- `data/` and `results/` contain almost all occupied bytes. Moving Python scripts will not materially recover storage space.
- Inventory found **{meta['n_duplicate_script_groups']} exact script duplicate group(s)**, **{meta['n_path_mentions']} lexical path mentions**, **{meta['n_cache_directories']} cache directories**.
- Protected outputs: {meta['n_frozen_asset_checks_present']}/{meta['n_frozen_asset_checks_total']} paths located.

## Step 1 — COMPLETE: create a clean entry point

This README, documentation and indexes are new files. Original `src`, `data`, `results`, `external`, `references`, `notes`, and `logs` are not moved or modified.

## Step 2 — Safeguard before any destructive or path-changing operation

- Check `git status --short` and identify whether `src/` is tracked; avoid accidentally committing raw data.
- Make a version-controlled source/config snapshot and confirm an independent raw-data backup exists.
- Hash frozen files and record expected source/result provenance for any assets about to move.

## Step 3 — Name canonical outputs, not just scripts

- Inspect QC JSON / known processing manifests for the current Denmark, Atlanta and reference workflows.
- Resolve competing `_repair`, `_FIXED`, numbered audit variants by following **which file produced the frozen output**, not latest timestamp alone.
- Treat `docs/CODE_INDEX.csv` as a review queue, not an approved removal list.

## Step 4 — Archive legacy CODE separately from data

- Make a reviewed, explicit file manifest of obsolete scripts, with dependency tests and rollback steps.
- Prefer Git-tracked moves or an archival Git branch; keep compatibility where imports or paths are referenced.
- After each small move, compile/import and run smoke tests for relevant active workflows.

## Step 5 — Storage consolidation (distinct and slower task)

- Inspect `docs/STORAGE_REVIEW.csv` and verify raw/intermediate/frozen status for large files.
- Do not deduplicate by filename or equal size. Compare checksums, processing provenance, and active references.
- `data/raw`, protected frozen results, and original external reference snapshots are **not deletion candidates**.
- Only after backup and byte-for-byte verification should identical, regenerable intermediates be proposed for relocation/removal.

## Manual review queues

- [File sizes](STORAGE_REVIEW.csv) — top 60 largest paths, not proven duplicates.
- [Full code index](CODE_INDEX.csv) — historical scripts and status labels.
- Exact byte-identical code duplicates: see `{report.relative_to(root).as_posix() if report.is_relative_to(root) else str(report)}/EXACT_SCRIPT_DUPLICATES.csv`.
- Lexical path hints: same report's `SOURCE_PATH_MENTIONS.csv`.

**No deletion/move was performed by the front-door utility.**
'''

    output['docs/CODE_INDEX.csv'] = csv_text(script_index, ['path','category','note','version_or_repair_name','path_mentions_count','sha256'])
    output['docs/STORAGE_REVIEW.csv'] = csv_text(size_review, ['path','size_GiB','size_bytes','role','action'])
    output['docs/DATASET_REGISTRY.csv'] = csv_text(datasets, list(datasets[0]) if datasets else ['dataset_id','layer','status_from_prior_research'])
    return output, meta, size_review


def run(root, inv_path, apply):
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError('Root not found: ' + str(root))
    inventory = find_inventory(root, inv_path)
    output, meta, largest = make_documents(root, inventory)
    print('=' * 79)
    print('CF-STATE | CLEAN REPOSITORY FRONT DOOR')
    print('=' * 79)
    print('Root:', root)
    print('Inventory:', inventory)
    print('Inventory files:', meta['n_files'], '| scripts:', meta['n_code'])
    print('Mode:', 'APPLY: create NEW documentation ONLY' if apply else 'DRY RUN: no file changes')
    print('\nDOCUMENTS:')
    for rel, content in output.items():
        path = root / rel
        if path.exists():
            print('  EXISTS / UNCHANGED:', rel)
            continue
        if not apply:
            print('  WOULD CREATE:', rel, f'({len(content):,} chars)')
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='') as stream:
            stream.write(content)
        print('  CREATED:', rel)
    print('\nLARGEST FIVE FILES (REVIEW ONLY):')
    for r in largest[:5]:
        print(' ', r['size_GiB'] + ' GiB', r['path'])
    print('\nNO SOURCE, DATA, REFERENCES, EXTERNAL FILES OR RESULTS MOVED/DELETED.')
    return output


def self_test():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)/'CFstate'
        src = root/'src/phase9'
        src.mkdir(parents=True)
        (src/'analysis_09t_example.py').write_text('print(1)\n')
        (root/'data/raw').mkdir(parents=True)
        raw = root/'data/raw/original.tsv'
        raw.write_text('untouched data\n')
        rep = root/'maintenance/cleanup_runs/cleanup_20261016_1'
        rep.mkdir(parents=True)
        (rep/'QC_STATUS.json').write_text(json.dumps({'n_files':2, 'n_code':1, 'n_duplicate_script_groups':0,
               'n_path_mentions':0, 'n_cache_directories':0, 'n_frozen_asset_checks_present':1,
               'n_frozen_asset_checks_total':1}))
        (rep/'FILE_INVENTORY.csv').write_text(csv_text([
            {'path':'src/phase9/analysis_09t_example.py','size_bytes':9,'role':'SOURCE'},
            {'path':'data/raw/original.tsv','size_bytes':15,'role':'RAW_IMMUTABLE'}],
            ['path','size_bytes','role']))
        (rep/'SOURCE_SCRIPTS.csv').write_text(csv_text([
            {'path':'src/phase9/analysis_09t_example.py', 'sha256':'abc',
             'version_or_repair_name':False, 'path_mentions_count':0}],
            ['path','sha256','version_or_repair_name','path_mentions_count']))
        (rep/'FROZEN_ASSET_CHECKS.csv').write_text(csv_text([
            {'path':'data/raw/original.tsv','exists':'True','kind':'file'}],['path','exists','kind']))
        (rep/'DATASET_REGISTER.csv').write_text(csv_text([
            {'dataset_id':'Test','layer':'CLINICAL','status_from_prior_research':'TEST'}],
            ['dataset_id','layer','status_from_prior_research']))
        before = raw.read_bytes()
        run(root, None, False)
        assert not (root/'README.md').exists(), 'dry run modified repo'
        run(root, None, True)
        assert (root/'README.md').is_file()
        assert (root/'docs/CODE_INDEX.csv').is_file()
        assert 'RECENT_PROGRAMME_EVIDENCE' in (root/'docs/CODE_INDEX.csv').read_text()
        assert 'data/raw/original.tsv' in (root/'docs/FROZEN_ASSETS.md').read_text()
        assert raw.read_bytes() == before, 'raw file modified'
        (root/'README.md').write_text('my custom README\n')
        run(root, None, True)
        assert (root/'README.md').read_text() == 'my custom README\n', 'rewrote existing file'
        assert raw.read_bytes() == before, 'source modified on second run'
    print('\nSELF_TEST_PASSED: dry-run, documentation generation, inventory integrity, no overwrite, no source/data mutation')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--inventory', type=str, default=None)
    p.add_argument('--apply', action='store_true', help='Create missing docs only, never overwrite')
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args.root, args.inventory, args.apply)


if __name__ == '__main__':
    main()
