#!/usr/bin/env python3
"""Read-only numerical regression: reproduce a frozen 04k4 Atlanta activity table.

Run from the repository root: python workflows/projection_regression.py --check
Does not execute any audit or modify any result. Uses the original 04k4
full-gene TPM contrast, published frozen loadings and frozen slope calibration.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cfstate.imodulons import project_calibrated  # noqa: E402

P4 = 'results/phase4/'
FILES = {
    'model': 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'mask': P4 + 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'calibration': P4 + 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 + 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'atlanta': P4 + 'audit_04j2_PRJNA576508_Atlanta_expression_normalisation/FROZEN_Atlanta_13_log2TPM_expression.csv',
    'reference': P4 + 'audit_04i3_PRJNA576508_CFBE_reference_metadata_repair/FROZEN_CFBE_log2TPM_reference_profiles.csv',
    'expected': P4 + 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_108_activities__FROZEN_5567_TPM__PRIMARY_CFBE_ALL4.csv',
}


def fail(msg: str) -> None:
    raise RuntimeError('PROJECTION_REGRESSION_FAILED: ' + msg)


def load_indexed(path: Path, key: str) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={key: str})
    if key not in frame.columns or frame[key].duplicated().any():
        fail(str(path) + ' missing/duplicate ' + key)
    return frame.set_index(key)


def load_m(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or 'M' not in payload:
        fail('frozen model missing M')
    data = payload['M']
    if isinstance(data, str):
        frame = pd.read_json(StringIO(data))
    elif isinstance(data, dict):
        frame = pd.DataFrame(data)
    else:
        fail('unrecognized frozen M format')
    frame.index = frame.index.astype(str).str.strip()
    frame.columns = frame.columns.astype(str).str.strip()
    if not frame.index.is_unique or not frame.columns.is_unique:
        fail('frozen M labels not unique')
    return frame


def check(root: Path, expect_real: bool = True) -> tuple[float, int, int]:
    paths = {key: root / rel for key, rel in FILES.items()}
    for key, path in paths.items():
        if not path.is_file():
            fail('missing ' + key + ' ' + str(path))
    m = load_m(paths['model'])
    mask = load_indexed(paths['mask'], 'GeneID').index.tolist()
    cal = load_indexed(paths['calibration'], 'imodulon')
    stable = load_indexed(paths['stable'], 'imodulon').index.tolist()
    atl = load_indexed(paths['atlanta'], 'GeneID')
    ref = load_indexed(paths['reference'], 'GeneID')
    expected = load_indexed(paths['expected'], 'imodulon')
    if expect_real and (len(mask) != 4923 or m.shape[1] != 116 or len(stable) != 108 or atl.shape[1] != 13):
        fail('unexpected real frozen mask/component/sample dimensions')
    if len(set(mask)) != len(mask) or not set(mask).issubset(m.index):
        fail('mask does not identify unique frozen M genes')
    if not set(mask).issubset(atl.index) or not set(mask).issubset(ref.index):
        fail('expression missing masked genes')
    if not set(m.columns).issubset(cal.index) or not set(stable).issubset(m.columns):
        fail('calibration/stable labels do not match frozen M')
    if 'calibration_slope' not in cal or 'PRIMARY_CFBE_ALL4' not in ref:
        fail('missing calibration slope or primary CFBE reference')
    if set(expected.columns) != set(atl.columns):
        fail('frozen samples differ from full-gene Atlanta expression')
    if set(expected.index) != set(stable):
        fail('frozen component labels differ from stable reference')
    # Match 04k4's frozen programme order and output column order exactly.
    matrix = m.loc[mask].to_numpy(float)
    slopes = cal.loc[m.columns, 'calibration_slope'].to_numpy(float)
    samples = expected.columns.tolist()
    contrasts = atl.loc[mask, samples].to_numpy(float) - ref.loc[mask, 'PRIMARY_CFBE_ALL4'].to_numpy(float)[:, None]
    if not np.isfinite(matrix).all() or not np.isfinite(contrasts).all() or not np.isfinite(slopes).all():
        fail('nonfinite reference, expression or slope values')
    predicted = project_calibrated(matrix, contrasts, slopes)
    indices = [m.columns.get_loc(p) for p in expected.index]
    predicted = predicted[indices, :]
    old = expected.loc[:, samples].to_numpy(float)
    if not np.isfinite(old).all():
        fail('frozen activity contains nonfinite values')
    difference = np.abs(predicted - old)
    worst = float(np.max(difference))
    print('CF-STATE | FROZEN ATLANTA 04K4 NUMERICAL REGRESSION')
    print('Frozen panel:', len(mask), 'genes,', m.shape[1], 'iModulons; reference:', len(expected.index), 'stable programmes')
    print('Atlanta samples:', len(samples))
    print('Maximum absolute error:', f'{worst:.10g}')
    if not np.allclose(predicted, old, atol=1e-7, rtol=1e-9):
        fail('new shared projection does not reproduce frozen 04k4 activities')
    print('PASS — shared projection matches frozen 04k4 activities; no existing files changed')
    return worst, len(mask), len(samples)


def self_test() -> None:
    rng = np.random.default_rng(42)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        m = pd.DataFrame(rng.normal(size=(14, 4)), index=[f'PA{i:04d}' for i in range(14)], columns=[f'M{i}' for i in range(4)])
        mask = list(m.index[2:12])
        stable = ['M0', 'M2', 'M3']
        samples = ['S2', 'S1']
        atl = pd.DataFrame(rng.normal(size=(14, 2)), index=m.index, columns=samples)
        ref = pd.DataFrame({'PRIMARY_CFBE_ALL4':rng.normal(size=14)}, index=m.index)
        slopes = pd.Series([1.2, 0.7, 1.5, 0.9], index=m.columns)
        contrast = atl.loc[mask, samples].to_numpy() - ref.loc[mask, 'PRIMARY_CFBE_ALL4'].to_numpy()[:, None]
        expected = ((np.linalg.pinv(m.loc[mask].to_numpy()) @ contrast) * slopes.to_numpy()[:, None])
        frozen = pd.DataFrame(expected, index=m.columns, columns=samples).loc[stable]
        contents = {
            'model': json.dumps({'M':m.to_json()}),
        }
        for key, rel in FILES.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if key in contents:
                path.write_text(contents[key])
            elif key == 'mask':
                pd.DataFrame({'GeneID':mask}).to_csv(path,index=False)
            elif key == 'calibration':
                pd.DataFrame({'imodulon':slopes.index,'calibration_slope':slopes.to_numpy()}).to_csv(path,index=False)
            elif key == 'stable':
                pd.DataFrame({'imodulon':stable}).to_csv(path,index=False)
            elif key == 'atlanta':
                atl.rename_axis('GeneID').reset_index().to_csv(path,index=False)
            elif key == 'reference':
                ref.rename_axis('GeneID').reset_index().to_csv(path,index=False)
            elif key == 'expected':
                frozen.rename_axis('imodulon').reset_index().to_csv(path,index=False)
        worst, _, _ = check(root, expect_real=False)
        assert worst < 1e-10
        bad = root / FILES['expected']
        corrupt = pd.read_csv(bad)
        corrupt.loc[0, 'S2'] += 0.5
        corrupt.to_csv(bad, index=False)
        try:
            check(root, expect_real=False)
        except RuntimeError as exc:
            assert 'does not reproduce' in str(exc)
        else:
            raise AssertionError('tampered frozen output passed')
    print('SELF_TEST_PASSED: synthetic frozen regression and corrupted-output rejection')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        check(args.root.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
