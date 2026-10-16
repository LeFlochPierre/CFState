#!/usr/bin/env python3
"""Read-only clinical frozen-output verifier for CF-State.

Run from anywhere: python workflows/clinical.py --root . --check
No historical audit is imported or executed. No scientific result is recomputed.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from _frozen import Stage, check_stages, run_cli, validate_definition as validate_stages


P4 = 'results/phase4/'
STAGES = (
    Stage('reference', 'aeruPRECISE411 reference',
          'src/phase2/audit_01_aeruprecise_resource_freeze.py',
          ('external/aeruprecise/precise411/data_files/M.csv',
           'external/aeruprecise/precise411/data_files/A.csv',
           'external/aeruprecise/precise411/data_files/sample_table.csv')),
    Stage('softcore', 'Denmark softcore fidelity',
          'src/phase4/audit_03e_softcore_projection_fidelity.py',
          (P4+'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
           P4+'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv'),
          (P4+'audit_03e_softcore_projection_fidelity/AUDIT_03E_STATUS.txt',
           'PRJEB24688_SOFTCORE_PROJECTION_EMPIRICALLY_VALIDATED'), ('reference',)),
    Stage('candidate', 'Frozen Denmark / discovery candidate',
          'src/phase4/audit_03h_freeze_replicated_CF_airway_candidate.py',
          (P4+'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',),
          (P4+'audit_03h_freeze_replicated_CF_airway_candidate/AUDIT_03H_STATUS.txt',
           'REPLICATED_CF_AIRWAY_PROGRAMME_CANDIDATE_FROZEN'), ('softcore',)),
    Stage('projection', 'Atlanta blinded softcore projection',
          'src/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection.py',
          (P4+'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
           P4+'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_Atlanta_softcore_4923_log2TPM.csv',
           P4+'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_CFBE_softcore_4923_log2TPM_references.csv'),
          (P4+'audit_04k4_PRJNA576508_blinded_softcore_projection/AUDIT_04K4_STATUS.txt',
           'PRJNA576508_ATLANTA_13_4923_MASK_BLINDED_SOFTCORE_PROJECTIONS_FROZEN'), ('softcore',)),
    Stage('qc_lock', 'Atlanta blinded QC lock',
          'src/phase4/audit_04l_PRJNA576508_blinded_projection_QC_lock.py',
          (P4+'audit_04l_PRJNA576508_blinded_projection_QC_lock/FROZEN_blinded_13_sample_QC.csv',),
          (P4+'audit_04l_PRJNA576508_blinded_projection_QC_lock/AUDIT_04L_STATUS.txt',
           'PRJNA576508_ATLANTA_BLINDED_PROJECTION_QC_AND_DESIGN_LOCK_FROZEN'), ('projection',)),
    Stage('residual', 'Atlanta residual and protocol freeze',
          'src/phase4/audit_04m2_PRJNA576508_blinded_residual_freeze.py',
          (P4+'audit_04m2_PRJNA576508_blinded_residual_freeze/FROZEN_PREUNBLINDING_EVALUATION_PROTOCOL.txt',
           P4+'audit_04m2_PRJNA576508_blinded_residual_freeze/FROZEN_Atlanta_13_sample_provenance.csv',
           P4+'audit_04m2_PRJNA576508_blinded_residual_freeze/DONOR_PROVENANCE_STATUS.txt'),
          (P4+'audit_04m2_PRJNA576508_blinded_residual_freeze/AUDIT_04M2_STATUS.txt',
           'PRJNA576508_ATLANTA_13_BLINDED_CONTROLLED_STATE_RESIDUALS_FROZEN'), ('qc_lock',)),
    Stage('comparison', 'Independent Atlanta versus frozen 03h',
          'src/phase4/audit_04n_PRJNA576508_independent_03h_comparison.py',
          (P4+'audit_04n_PRJNA576508_independent_03h_comparison/FROZEN_13_primary_candidate_comparisons.csv',
           P4+'audit_04n_PRJNA576508_independent_03h_comparison/FROZEN_18_directional_programme_summary.csv'),
          (P4+'audit_04n_PRJNA576508_independent_03h_comparison/AUDIT_04N_STATUS.txt',
           'PRJNA576508_INDEPENDENT_03H_DESCRIPTIVE_COMPARISON_FROZEN'),
          ('candidate', 'residual')),
    Stage('new_patients', 'Additional CF-patient descriptive checks',
          'src/phase7/audit_07h_CFstate_frozen_signature_new_CF_patients.py',
          ('results/phase7/audit_07h_frozen_CF_signature_two_new_patients/DESCRIPTIVE_new_CF_patients_reference_sensitivity.csv',),
          ('results/phase7/audit_07h_frozen_CF_signature_two_new_patients/AUDIT_07H_STATUS.txt',
           'CFSTATE_PHASE7_07H_FROZEN_03H_NEW_PATIENTS_DESCRIPTIVE_PROJECTION_COMPLETE_REVIEW_REQUIRED'),
          ('candidate', 'projection'), True),
)


def validate_definition() -> None:
    validate_stages(STAGES)


def check(root: Path, include_optional: bool = False, hashes: bool = False) -> list[dict]:
    return check_stages(root, STAGES, include_optional, hashes)


def self_test() -> None:
    validate_definition()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for stage in STAGES:
            for relative in (stage.script, *stage.files):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('test\n', encoding='utf-8')
            if stage.status:
                relative, expected = stage.status
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(expected + '\n', encoding='utf-8')
        assert all(x['ok'] for x in check(root))
        assert len(check(root)) == len(STAGES) - 1
        assert all(x['ok'] for x in check(root, True, True))
        status_path = root / STAGES[3].status[0]
        status_path.write_text('WRONG\n', encoding='utf-8')
        bad = check(root)
        assert len([x for x in bad if not x['ok']]) == 1
        assert bad[3]['key'] == 'projection'
        status_path.write_text(STAGES[3].status[1] + '\n', encoding='utf-8')
        source_path = root / STAGES[5].files[0]
        source_path.unlink()
        bad = check(root)
        assert not bad[5]['ok']
        assert all(x['ok'] for i, x in enumerate(bad) if i != 5)
    print('SELF_TEST_PASSED: full chain, optional stage, hashes, status mismatch and missing input')


def main() -> int:
    return run_cli(
        STAGES,
        'CF-STATE | CLINICAL FROZEN WORKFLOW CHECK (READ ONLY)',
        'File/status checks do not prove biological validity or cohort independence.',
        self_test=self_test,
    )


if __name__ == '__main__':
    raise SystemExit(main())
