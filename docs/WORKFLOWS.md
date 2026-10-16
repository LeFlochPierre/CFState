# Workflow navigation (existing files are unchanged)

## Where to look first

- **reference:** `external/aeruprecise/precise411/data_files/M.csv` — present
- **reference:** `external/aeruprecise/precise411/data_files/A.csv` — present
- **clinical:** `results/phase4/audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv` — present
- **clinical:** `results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv` — present
- **perturbation:** `results/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze` — present
- **perturbation:** `results/phase9/audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze` — present
- **interpretation:** `results/phase9/analysis_09t_frozen_imodulon_perturbation_alignment` — present
- **interpretation:** `results/phase9/analysis_09u_targeted_imodulon_robustness` — present

These landmarks identify previously used outputs, **not** a fully audited or automatically executable pipeline. Always read the related QC/status JSON and the generating script before reusing the contents.

## Script distribution

- `src/audit`: 73 scripts
- `src/phase2`: 22 scripts
- `src/phase3`: 22 scripts
- `src/phase4`: 44 scripts
- `src/phase5`: 16 scripts
- `src/phase6`: 12 scripts
- `src/phase7`: 13 scripts
- `src/phase8`: 20 scripts
- `src/phase9`: 27 scripts

## Code role labels (review only)

- **CLINICAL_PROVENANCE_REVIEW**: 44
- **HISTORICAL_NEGATIVE_BENCHMARK**: 3
- **HISTORICAL_OR_SUPPORT_REVIEW**: 61
- **INITIAL_BIOFILM_HISTORY**: 73
- **PA14_TRANSFER_PROVENANCE**: 22
- **PERTURBATION_INPUT_PROVENANCE**: 4
- **PHASE9_HISTORICAL_OR_SUPPORT**: 18
- **RECENT_PROGRAMME_EVIDENCE**: 2
- **REFERENCE_PROVENANCE**: 22

See [CODE_INDEX.csv](CODE_INDEX.csv) for all 249 scripts. Filename-based role labels are a starting point; `*_repair*` and `*_FIXED*` can be the correct production versions, not clutter.

## Reproduction sequence for the next clinical analysis

1. Load the frozen reference and canonical gene mappings; validate the expected shapes and hashes.
2. Resolve patient/study identities and existing clinical freeze outputs; never count technical replicates as independent patients.
3. Run the clinical projection and uncertainty checks using existing scripts or new versioned scripts as necessary.
4. Evaluate transfer across held-out patients and studies, avoiding training/reference overlap.
5. Only then join lab perturbation programme interpretations; maintain acute/long-term and known overlap flags.

No historic scripts were moved, rewritten, or designated officially obsolete by this step.
