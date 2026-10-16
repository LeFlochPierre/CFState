# Current research status — CF-State

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
