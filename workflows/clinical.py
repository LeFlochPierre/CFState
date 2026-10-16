#!/usr/bin/env python3
"""Read-only clinical frozen-output verifier for CF-State.

Run from anywhere: python workflows/clinical.py --root . --check
No historical audit is imported or executed. No scientific result is recomputed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    key: str
    name: str
    script: str
    files: tuple[str, ...]
    status: tuple[str, str] | None = None
    depends: tuple[str, ...] = ()
    optional: bool = False


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def validate_definition() -> None:
    names = [s.key for s in STAGES]
    if len(set(names)) != len(names):
        raise ValueError('Duplicate stage key')
    previous = set()
    for stage in STAGES:
        if any(dep not in previous for dep in stage.depends):
            raise ValueError('Stages are not in dependency order: ' + stage.key)
        previous.add(stage.key)


def check(root: Path, include_optional: bool = False, hashes: bool = False) -> list[dict]:
    validate_definition()
    results = []
    for stage in STAGES:
        if stage.optional and not include_optional:
            continue
        issues = []
        found = []
        for relative in (stage.script, *stage.files):
            path = root / relative
            if not path.is_file() or path.stat().st_size == 0:
                issues.append('missing or empty: ' + relative)
            else:
                item = {'path': relative, 'bytes': path.stat().st_size}
                if hashes:
                    item['sha256'] = sha256(path)
                found.append(item)
        if stage.status:
            rel, expected = stage.status
            path = root / rel
            if not path.is_file():
                issues.append('missing status: ' + rel)
            else:
                actual = path.read_text(encoding='utf-8').strip()
                if actual != expected:
                    issues.append('unexpected status: ' + rel + ' (' + repr(actual) + ')')
                elif hashes:
                    found.append({'path': rel, 'sha256': sha256(path), 'bytes': path.stat().st_size})
        results.append({'key': stage.key, 'name': stage.name,
                        'depends': list(stage.depends), 'ok': not issues,
                        'issues': issues, 'checked': found})
    return results


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--show', action='store_true', help='Show ordered stages without touching data')
    parser.add_argument('--check', action='store_true', help='Check source and frozen assets (default)')
    parser.add_argument('--include-optional', action='store_true', help='Include follow-up clinical stage')
    parser.add_argument('--hash', action='store_true', help='Also compute SHA256 for checked files')
    parser.add_argument('--json', action='store_true', help='Print machine-readable JSON instead of table')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.show:
        for i, stage in enumerate(STAGES, 1):
            optional = ' (optional)' if stage.optional else ''
            print(f'{i:02d}. {stage.key}: {stage.name}{optional}')
            print(f'    source: {stage.script}')
            print(f'    depends: {", ".join(stage.depends) or "none"}')
        return 0
    results = check(args.root.resolve(), args.include_optional, args.hash)
    if args.json:
        print(json.dumps({'root': str(args.root.resolve()), 'stages': results}, indent=2))
    else:
        print('CF-STATE | CLINICAL FROZEN WORKFLOW CHECK (READ ONLY)')
        print('Root:', args.root.resolve())
        for row in results:
            label = 'PASS' if row['ok'] else 'FAIL'
            print(f'[{label}] {row["key"]}: {row["name"]}')
            for issue in row['issues']:
                print('       -', issue)
        total = sum(x['ok'] for x in results)
        print(f'Checked stages: {total}/{len(results)} PASS')
        print('File/status checks do not prove biological validity or cohort independence.')
        print('No existing data, frozen results or legacy scripts were changed.')
    return 0 if all(row['ok'] for row in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())
