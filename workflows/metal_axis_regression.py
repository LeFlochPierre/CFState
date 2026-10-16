#!/usr/bin/env python3
"""Read-only Phase 8 08g metal-axis regression via shared frozen projection.

Imports the original 08e2 parsing helpers without calling their main routine.
Does not re-download data, overwrite results, or change archived source hashes.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from cfstate.imodulons import project_residual  # noqa: E402

SOURCE = 'src/phase8/audit_08e2_CF_sputum_17_donor_metal_perturbation.py'
FREEZE = 'results/phase8/audit_08g_perturbation_anchored_clinical_decomposition/'
STATUS = 'CFSTATE_PHASE8_08G_PERTURBATION_ANCHORED_DECOMPOSITION_EXPLORATORY_COMPLETE'


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError('METAL_AXIS_REGRESSION_FAILED: ' + message)


def check(root: Path) -> float:
    # Original helper constants rely on Path.cwd(); do not silently resolve
    # them in a different project if this script is invoked elsewhere.
    require(Path.cwd().resolve() == root.resolve(),
            'run from the requested CFstate project root')
    status_path = root / FREEZE / 'AUDIT_08G_STATUS.txt'
    require(status_path.is_file() and status_path.read_text().strip() == STATUS,
            '08g frozen status missing or unexpected')
    script = root / SOURCE
    require(script.is_file(), '08e2 source unavailable')
    spec = importlib.util.spec_from_file_location('cfstate_legacy_neff08e2', script)
    require(spec is not None and spec.loader is not None, 'cannot load legacy 08e2 helpers')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Import only; not module.main()
    require(module.ROOT == root, 'legacy 08e2 root mismatch')
    require(all(path.is_file() for path in module.MODEL.values()),
            'some frozen 08e2 projection inputs are missing')
    count_path = module.RAW / module.COUNT_NAME
    require(count_path.is_file(), 'public Neff count table missing; no download attempted')
    counts, manifest, _excluded, _ambiguous = module.parse_counts(count_path)
    require(len(manifest) == 34 and manifest['donor_id'].nunique() == 17,
            'expected 17 paired sputum environments')
    mask = pd.read_csv(module.MODEL['mask'])['GeneID'].astype(str).tolist()
    require(len(mask) == 4923 and set(mask).issubset(counts.index),
            'frozen gene mask mismatch')
    frozen = module.load_frozen(mask)
    stable = frozen['stable']
    indices = np.array([frozen['all'].index(name) for name in stable], dtype=int)
    ordered = manifest['sample'].tolist()
    count_matrix = counts.loc[mask, ordered].to_numpy(float)
    factors, _ = module.size_factors(count_matrix)
    normalized = np.log2(1.0 + count_matrix / factors[None, :])
    operator = np.linalg.pinv(frozen['m'])
    residual = project_residual(operator, normalized, frozen['slopes'],
                                indices, frozen['b'])
    require(residual.shape == (108, 34), 'unexpected projected response dimensions')
    require(float(np.max(np.abs(frozen['b'].T @ residual))) < 1e-6,
            'metal projection not orthogonal to frozen controls')
    donors, paired = module.paired_values(residual, manifest)
    require(len(donors) == 17 and paired.shape == (17, 108),
            'paired donor-response dimensions changed')
    mean_metal = paired.mean(axis=0)
    mean_norm = float(np.linalg.norm(mean_metal))
    require(mean_norm > 1e-12 and np.isfinite(mean_norm),
            'zero/nonfinite average combined-metal direction')
    unit = mean_metal / mean_norm
    target = pd.read_csv(root / FREEZE / 'FROZEN_derived_108_programme_metal_axis.csv')
    required = {'imodulon', 'mean_METALS_minus_UNTREATED_108',
                'unit_mean_metal_direction_108', 'frozen_CF_signature_unit_108'}
    require(required.issubset(target.columns) and target['imodulon'].is_unique
            and set(target['imodulon']) == set(stable) and len(target) == 108,
            'frozen 08g programme axis shape/labels wrong')
    target = target.set_index('imodulon').loc[stable]
    expected_mean = target['mean_METALS_minus_UNTREATED_108'].to_numpy(float)
    expected_unit = target['unit_mean_metal_direction_108'].to_numpy(float)
    expected_cf = target['frozen_CF_signature_unit_108'].to_numpy(float)
    errors = {
        '17-donor mean response': float(np.max(np.abs(mean_metal - expected_mean))),
        'normalised metal direction': float(np.max(np.abs(unit - expected_unit))),
        'frozen CF candidate': float(np.max(np.abs(frozen['candidate'] - expected_cf))),
    }
    print('CF-STATE | READ-ONLY PHASE 8 08G METAL-AXIS REGRESSION')
    print('Donor pairs:', len(donors), '| stable programmes:', len(stable))
    for name, value in errors.items():
        print(name + ' maximum absolute error:', f'{value:.10g}')
    for name, fresh, old in (
        ('mean metal direction', mean_metal, expected_mean),
        ('unit metal direction', unit, expected_unit),
        ('frozen CF candidate', frozen['candidate'], expected_cf)):
        require(np.allclose(fresh, old, rtol=1e-9, atol=1e-7),
                name + ' changed compared with frozen 08g')
    print('PASS — 08g 17-donor combined-metal programme direction reproduced')
    print('This is a within-study numerical check, NOT independent CF specificity evidence.')
    print('Original Phase 8 scripts and frozen results remain untouched.')
    return max(errors.values())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    check(args.root.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
