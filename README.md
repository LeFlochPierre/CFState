# CF-State — Physiological programmes in CF airway *Pseudomonas aeruginosa*

> Project navigator created 2026-10-16. This is a **navigation layer**, not a rebuilt pipeline. Prior data, results, and source directories remain at their original locations.

## Current research question

Which interpretable transcriptional programmes are reliably observable in direct clinical CF airway bacterial transcriptomes, how reproducible are they across independent patients/cohorts, and what do controlled perturbations tell us about their biological meaning?

## Run the maintained checks

```bash
# Offline numerical + historical-source SHA256 checks
python workflows/verify.py

# Full read-only local verification (needs existing frozen datasets and results)
python workflows/verify.py --full
```

**[Active code and protected historical workflow](docs/ACTIVE_PIPELINE.md)** is
the guide to what to edit, what to reuse, and what must remain frozen.
The source-only tests also run in GitHub Actions; real frozen datasets remain
local and are never uploaded by that workflow.

## Start here

- [Current project status](docs/PROJECT_STATUS.md) — aims, evidence, unanswered questions.
- [Workflow and code map](docs/WORKFLOWS.md) — where the usable results are; historical scripts remain traceable.
- [Frozen assets](docs/FROZEN_ASSETS.md) — reference inputs and outputs that must not be silently changed.
- [Dataset registry](docs/DATASET_REGISTRY.csv) — per-study role, caveats and provenance status.
- [Research decisions](docs/RESEARCH_HISTORY.md) — tested alternatives, including negative findings.
- [Cleanup and storage plan](docs/CLEANUP_PLAN.md) — the safe way to consolidate files without losing reproducibility.
- [Full script index](docs/CODE_INDEX.csv) — all 249 indexed scripts and review labels.
- [Largest-file review list](docs/STORAGE_REVIEW.csv) — space-consuming files, **not deletion candidates**.

## Repository conventions

- `external/`: published or third-party resources; treat as immutable.
- `references/`: genomes and other mapping references.
- `data/raw/`: original source data, preserve provenance.
- `data/processed/`: intermediate and frozen derived matrices; status varies.
- `results/`: frozen audits and experiments; do not overwrite.
- `src/`: historical and current research code, **paths intentionally unchanged**.
- `logs/`: recorded runs.
- `maintenance/cleanup_runs/`: inventories (latest input: `maintenance/cleanup_runs/cleanup_20261016_155654_208861`).

## Next scientific milestone

Re-evaluate the frozen clinical CF programme profiles across independent patients and cohorts, including detection/missingness uncertainty and non-CF specificity. Use the controlled-perturbation evidence as *interpretation/positive controls*, not as proof of CF causality.
