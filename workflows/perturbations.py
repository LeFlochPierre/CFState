#!/usr/bin/env python3
"""Read-only Phase 9 perturbation evidence inventory and freeze checker."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from _frozen import Stage, check_stages, run_cli, validate_definition

P9 = 'results/phase9/'
STAGES = (
    Stage('imodulon', 'Frozen aeruPRECISE411 loadings',
          'src/phase2/audit_01_aeruprecise_resource_freeze.py',
          ('external/aeruprecise/precise411/data_files/M.csv',
           'external/aeruprecise/precise411/data_files/A.csv')),
    Stage('responses_09p', '09p reconstructed multi-study contrasts',
          'src/phase9/audit_09p_GEO_corroborated_multistudy_response_freeze.py',
          (P9+'audit_09p_GEO_corroborated_multistudy_response_freeze/FROZEN_COMMON_GENE_BY_CONTRAST_RESPONSE_MATRIX.csv',
           P9+'audit_09p_GEO_corroborated_multistudy_response_freeze/CONTRAST_SOURCE_DESIGN_AND_QC.csv'),
          (P9+'audit_09p_GEO_corroborated_multistudy_response_freeze/QC_STATUS.json',
           '09P_PROVISIONAL_MULTISTUDY_RESPONSE_CANDIDATES_FROZEN')),
    Stage('responses_09r', '09r additional responses and DATS quarantine',
          'src/phase9/audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze_FIXED2.py',
          (P9+'audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze/PROVISIONAL_ADDITIONAL_GENE_RESPONSES.csv',
           P9+'audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze/QC_STUDY_COUNT_SCALE_AND_COMPLETENESS.csv'),
          (P9+'audit_09r_three_PAO1_perturbations_GEO_sample_QC_freeze/QC_STATUS.json',
           '09R_THREE_STUDY_ADDITIONAL_CONTRAST_QC_COMPLETE')),
    Stage('alignment_09t', '09t frozen iModulon alignment',
          'src/phase9/analysis_09t_frozen_imodulon_perturbation_alignment.py',
          (P9+'analysis_09t_frozen_imodulon_perturbation_alignment/ALL_FROZEN_IMODULON_LOADINGS_RESPONSE_ALIGNMENTS.csv',
           P9+'analysis_09t_frozen_imodulon_perturbation_alignment/REFERENCE_411_GEO_GSM_OVERLAP.csv'),
          (P9+'analysis_09t_frozen_imodulon_perturbation_alignment/QC_STATUS.json',
           '09T_DESCRIPTIVE_FROZEN_IMODULON_ALIGNMENT_COMPLETE'),
          ('imodulon', 'responses_09p', 'responses_09r')),
    Stage('robustness_09u', '09u targeted alignment robustness',
          'src/phase9/analysis_09u_targeted_imodulon_robustness.py',
          (P9+'analysis_09u_targeted_imodulon_robustness/TARGET_MODULE_ROBUSTNESS_SUMMARY.csv',
           P9+'analysis_09u_targeted_imodulon_robustness/LEAVE_GENE_AND_BLOCK_OUT.csv'),
          (P9+'analysis_09u_targeted_imodulon_robustness/QC_STATUS.json',
           '09U_TARGETED_PROGRAMME_ROBUSTNESS_COMPLETE'), ('alignment_09t',)),
    Stage('negative_benchmark_09s', '09s retrospective graph baseline (historical, optional)',
          'src/phase9/model_09s_primary_fourstudy_fewshot_extension.py',
          (P9+'model_09s_primary_fourstudy_fewshot_extension/SUMMARY_HELDOUT_PRIMARY_STUDY_METRICS.csv',),
          (P9+'model_09s_primary_fourstudy_fewshot_extension/QC_STATUS.json',
           '09S_PRIMARY_FOUR_STUDY_FROZEN_ARCHITECTURE_RETROSPECTIVE_BENCHMARK'),
          ('responses_09p', 'responses_09r'), True),
)


def self_test() -> None:
    validate_definition(STAGES)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for stage in STAGES:
            for relative in (stage.script, *stage.files):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture\n', encoding='utf-8')
            if stage.status:
                relative, expected = stage.status
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({'status': expected}), encoding='utf-8')
        assert len(check_stages(root, STAGES)) == len(STAGES) - 1
        assert all(row['ok'] for row in check_stages(root, STAGES, True, True))
        relative = STAGES[2].status[0]
        (root / relative).write_text('not JSON', encoding='utf-8')
        rows = check_stages(root, STAGES, True)
        assert sum(not row['ok'] for row in rows) == 1
        assert not rows[2]['ok']
    print('SELF_TEST_PASSED: Phase 9 stages, JSON status, optional benchmark and malformed QC')


def main() -> int:
    return run_cli(
        STAGES,
        'CF-STATE | FROZEN PERTURBATION EVIDENCE CHECK (READ ONLY)',
        'Checks do not verify experimental independence or biological significance.',
        self_test=self_test,
    )


if __name__ == '__main__':
    raise SystemExit(main())
