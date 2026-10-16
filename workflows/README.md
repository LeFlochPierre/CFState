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
