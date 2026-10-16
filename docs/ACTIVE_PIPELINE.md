# CF-State — active code and preserved research history

This document is the canonical navigation point for day-to-day work. It
does **not** relabel original audit source as unused or delete frozen evidence.

## One verification command

From the repository root with `CFSTATE_ENV` active:

```bash
# Fast offline tests: mathematical regressions + byte-for-byte archive checks
python workflows/verify.py

# Read-only checks and numerical replays using your local frozen datasets
python workflows/verify.py --full
```

The offline command also runs in GitHub Actions and does not need `data/`,
`results/`, `external/` or `references/`. The full mode remains local:
it validates the clinical chain and Phase 9 response evidence, then checks
frozen outputs for Atlanta 04k4, clinical follow-up 07h and metal-axis 08g.

The full run **does not include historical model 09s by default**. Verify
that separately via `python workflows/perturbations.py --include-optional`
if working on the older negative model experiments.

All these checks are **read-only** and must never be used as an excuse to
regenerate or overwrite an accepted frozen output.

## Which code should I edit?

| Role | Preferred location | Notes |
| --- | --- | --- |
| Shared numerical science | `cfstate/` | For *new* analyses; add numerical regression tests |
| Numerical and provenance tests | `tests/` | Offline checks can run in GitHub Actions |
| New analysis entry points | `workflows/` | Stable, documented workflow names |
| Frozen generating code | `src/phase2/` … `src/phase9/` | Preserve original SHA256 and input/output paths |
| Initial biofilm-model history | `src/audit/` | Historical negative exploration, retained for provenance |
| Historical source registry | `docs/CODE_INDEX.csv` | Initial inventory of all 249 source scripts |
| Scientific study caveats | `docs/PROJECT_STATUS.md` and `docs/DATASET_REGISTRY.csv` | Avoid incorrect independence/CF-specificity claims |

### The main scientific pathways

**Reference / cross-strain:** `src/phase2/audit_01_aeruprecise_resource_freeze.py`
and `src/phase3/audit_04_freeze_crossstrain_state_framework.py`
define parts of the original reference freeze. Retain the original
published iModulon matrices and projection order.

**Clinical programme discovery and transfer:**
`src/phase4/audit_03e_softcore_projection_fidelity.py`
→ `src/phase4/audit_03h_freeze_replicated_CF_airway_candidate.py`;
independently, `04k4` → `04l` → `04m2`; then `04n`.
`src/phase7/audit_07h_CFstate_frozen_signature_new_CF_patients.py`
supports descriptive follow-up for two additional patients. See
`workflows/clinical.py --show` for full paths and order.

**Perturbation interpretation:** Source outputs 09p and 09r
→ iModulon loading-to-response alignment 09t → sensitivity 09u.
These are *descriptive positive-control associations*, not independently
proven CF-specific mechanisms. The 09s curated regulatory network
comparison belongs to the preserved negative-results history.

**Combined-metal response:** `src/phase8/audit_08g_perturbation_anchored_clinical_decomposition.py`
preserves the original post-hoc 17-donor-pair metal-response decomposition.
It uses PAO1 in sputum exposures and a combined Zn+Fe+Mn intervention, not
native bacterial transcriptomes or individual-metal attribution.

## What not to change yet

- Do not move or edit any original 249 scripts under `src/` without a new,
  separately reviewed provenance/migration plan.
- Do not merge `main` until the new offline and full checks pass.
- Do not track the 121.8 GiB of excluded data and frozen results in GitHub.
- Do not treat equal output file counts as proof of independent patients,
  or a passing projection regression as clinical validation.

## Next **real** scientific-development milestone

New work should use the `cfstate/` package and current `workflows/`
entry points. The next analysis needs donor-resolved clinical metadata,
independent-cohort/held-out evaluation, and uncertainty due to incomplete
gene coverage. Maintain known laboratory-reference and sample overlap flags.
