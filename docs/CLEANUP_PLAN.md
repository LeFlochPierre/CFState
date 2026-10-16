# Safe repository cleanup — step-by-step

## Baseline from latest inventory

- Files: **3,020**; source scripts: **249**.
- Apparent project size: approximately **121.8 GiB** in the supplied real inventory; actual numbers may vary after future runs.
- `data/` and `results/` contain almost all occupied bytes. Moving Python scripts will not materially recover storage space.
- Inventory found **1 exact script duplicate group(s)**, **967 lexical path mentions**, **8 cache directories**.
- Protected outputs: 12/12 paths located.

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
- Exact byte-identical code duplicates: see `maintenance/cleanup_runs/cleanup_20261016_155654_208861/EXACT_SCRIPT_DUPLICATES.csv`.
- Lexical path hints: same report's `SOURCE_PATH_MENTIONS.csv`.

**No deletion/move was performed by the front-door utility.**
