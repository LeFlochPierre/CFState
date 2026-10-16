# CF-State: Clean workflow entry points

This folder provides **stable navigation and verification** around existing frozen analyses.
It does not replace their computational provenance. Existing scripts under `src/` and
results under `results/` remain unchanged.

## Clinical chain

```bash
python workflows/clinical.py --show
python workflows/clinical.py --self-test
python workflows/clinical.py --root . --check
python workflows/clinical.py --root . --check --include-optional
```

The default check visits these stages in dependency order:

1. Frozen aeruPRECISE411 reference.
2. Denmark softcore fidelity (`03e`).
3. Frozen Denmark/discovery candidate (`03h`).
4. Atlanta softcore projection (`04k4`).
5. Atlanta projection QC lock (`04l`).
6. Atlanta residual/protocol freeze (`04m2`).
7. Independent frozen-candidate comparison (`04n`).

The additional-patient descriptive transfer (`07h`) is optional.

**Scope of check:** existence and nonempty size of selected outputs, and exact
frozen status strings transcribed from their generating scripts. With `--hash`,
the checker also calculates SHA256 digests. It does **not** inspect the full
data matrices, reproduce inference, establish sample independence, or certify
biological validity. A failed check does not automatically mean the underlying
science failed: it may indicate a missing file or a stale path in this registry.

This is the first incremental refactor. Next, we will reconcile any paths
reported by the checker against actual local outputs and only then extract
shared functions with regression tests. **Do not run the old scripts simply to
satisfy a missing-file check; some involve large sequencing processes.**

No existing research file has been moved or deleted.


## Perturbation evidence chain

```bash
python workflows/perturbations.py --self-test
python workflows/perturbations.py --show
python workflows/perturbations.py --root . --check --include-optional
```

The second entry point covers the **frozen read-only contracts** for 09p
(provisional multi-study response), 09r (additional studies and DATS
quarantine), 09t (loading-to-response alignment), and 09u (robustness).
The failed graph-predictor comparison 09s is retained as an **optional
historical negative benchmark**, not part of the active scientific model.

Both `clinical.py` and `perturbations.py` now reuse `_frozen.py`
for file presence, optional SHA256 and frozen status validation. Status
files ending in `.json` are read as JSON, while `.txt` status files
are checked as exact strings.

These scripts **do not run the generating audits or modify any datasets**.

## Scientific functions (first extracted module)

The `cfstate/imodulons.py` package now centralises three previously repeated
numerical operations **without changing their formulas**:

- `project_calibrated(M, expression_contrasts, slopes)`: frozen pseudoinverse
  projection, then per-component slope calibration. Inputs must already be
  position-aligned, and expression inputs are contrasts (intercepts cancel).
- `residualize_controls(activities, B)`: same `np.linalg.lstsq(..., rcond=None)`
  residualisation used by clinical and perturbation audits.
- `counts_to_logtpm(counts, lengths_bp)`: 04k4 panel-specific gene-length
  TPM normalisation, then log2(TPM+1). Never silently substitute softcore TPM
  for full-gene TPM.

Regression commands (from the project root):

```bash
python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python workflows/projection_regression.py --self-test
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python workflows/projection_regression.py --check
```

The real-data regression reads **the original frozen 04k4 inputs** and independently
recalculates the published 108-programme × 13-sample Atlanta primary reference
activity matrix. It checks it against
`FROZEN_108_activities__FROZEN_5567_TPM__PRIMARY_CFBE_ALL4.csv`.
It refuses discrepancies greater than the specified numerical tolerance.

Historical source scripts are intentionally unchanged in this first extraction;
their outputs provide the regression ground truth. We will only replace their
internal duplicated calculations once the real-data check passes.


## Phase 7 and Phase 8 scientific replays (provenance-safe refactor)

The original generating scripts, including `src/phase7/audit_07h_...` and
`src/phase8/audit_08g_...`, are intentionally **not modified**: their
archived outputs include script SHA256 provenance. We instead expose shared
science in `cfstate/imodulons.py`, and verify its behaviour against frozen
outputs using the following read-only replays:

```bash
# Always run from repository root in CFSTATE_ENV
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python workflows/projection_regression.py --check
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python workflows/followup_regression.py --check
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python workflows/metal_axis_regression.py --check
```

- `followup_regression.py` independently replays the frozen **07h** two
  patients × two specimens × three CFBE references (12 comparisons) and the
  two within-patient temporal differences. Uses the new shared
  `project_residual` and `counts_to_logtpm` functions.
- `metal_axis_regression.py` independently reconstructs the **08g** mean
  combined Zn+Fe+Mn response direction from the 17 paired public
  PAO1-spike-in sputum environments. It **imports only the original 08e2
  input-parsing helpers**, not its main or any output-writing pipeline, and
  uses the new shared `project_residual` function. Run it from the project
  root so the original helpers resolve their paths correctly.
- `tests/test_legacy_source_hashes.py` checks all 249 original generating
  scripts against the hashes captured in `docs/CODE_INDEX.csv`.
  Any historical source change must be treated as an explicit, separately
  versioned provenance decision rather than a silent refactor.

The scripts will **fail closed** if dimensions, required inputs, frozen QC
status, sample identities, or numerical results disagree (tolerance
`atol=1e-7`, `rtol=1e-9`). Do not execute a legacy generating audit to
fix a missing file or a failing replay. Instead inspect the diagnosis and
the upstream provenance. Successful numerical reproduction is not biological
validation or evidence of CF-specific bacterial state.
