#!/usr/bin/env python3
"""Read-only replay of Phase 7 audit 07h using shared CF-State projection maths.

Reads published local frozen inputs and checks saved outputs. It neither
imports/reruns the original audit nor changes any frozen results.
"""
from __future__ import annotations

import argparse
import json
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from cfstate.imodulons import counts_to_logtpm, project_residual  # noqa: E402

P4 = 'results/phase4/'
P7 = 'results/phase7/'
MODEL = 'external/aeruprecise/precise411/paeru_116_iModulon.json'
SAF = 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf'
MASK = P4 + 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'
CAL = P4 + 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv'
STABLE = P4 + 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv'
CONTROLS = P4 + 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv'
CANDIDATE = P4 + 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv'
CFBE = P4 + 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_CFBE_softcore_4923_log2TPM_references.csv'
A7C = P7 + 'audit_07c_SRP173673_D7_D8_PAO1_feasibility/'
A7G = P7 + 'audit_07g_GSE139943_CF146_D33_D35_PAO1_feasibility/'
A7H = P7 + 'audit_07h_frozen_CF_signature_two_new_patients/'
RUNS = (
    ('SRR8334098', 'CF01', 'D-8', A7C, 'PAO1_featureCounts.txt'),
    ('SRR8334097', 'CF01', 'D-7', A7C, 'PAO1_featureCounts.txt'),
    ('SRR10401171', 'CF146', 'day33', A7G, 'PAO1_5567_assigned_counts.csv'),
    ('SRR10401172', 'CF146', 'day35', A7G, 'PAO1_5567_assigned_counts.csv'),
)
REFERENCES = ('PRIMARY_CFBE_ALL4', 'SENSITIVITY_CFBE_C_PAIR', 'SENSITIVITY_CFBE_R_PAIR')
STATUS = 'CFSTATE_PHASE7_07H_FROZEN_03H_NEW_PATIENTS_DESCRIPTIVE_PROJECTION_COMPLETE_REVIEW_REQUIRED'


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError('FOLLOWUP_REGRESSION_FAILED: ' + message)


def indexed(path: Path, index: str, sep: str = ',') -> pd.DataFrame:
    tab = pd.read_csv(path, sep=sep, dtype={index: str})
    require(index in tab.columns and tab[index].notna().all()
            and tab[index].is_unique, 'invalid unique index: ' + str(path))
    return tab.set_index(index)


def model_m(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding='utf-8'))['M']
    if isinstance(payload, str):
        frame = pd.read_json(StringIO(payload))
    else:
        frame = pd.DataFrame(payload)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    require(frame.index.is_unique and frame.columns.is_unique, 'duplicate frozen modulome labels')
    return frame


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    require(den >= 1e-12, 'undefined cosine')
    return float(x @ y / den)


def check(root: Path) -> float:
    status_path = root / A7H / 'AUDIT_07H_STATUS.txt'
    require(status_path.is_file() and status_path.read_text().strip() == STATUS,
            '07h frozen status unavailable or incorrect')
    mask = indexed(root / MASK, 'GeneID').index.tolist()
    cal = indexed(root / CAL, 'imodulon')
    stable = indexed(root / STABLE, 'imodulon').index.tolist()
    names = cal.index.tolist()
    m = model_m(root / MODEL)
    if set(mask).issubset(m.columns) and set(names).issubset(m.index):
        m = m.T
    require(len(mask) == 4923 and len(set(mask)) == 4923
            and len(names) == 116 and len(stable) == 108
            and set(mask).issubset(m.index) and set(names).issubset(m.columns),
            'frozen projection gene/programme mismatch')
    matrix = m.loc[mask, names].to_numpy(float)
    require(np.isfinite(matrix).all(), 'invalid frozen loading matrix')
    operator = np.linalg.pinv(matrix)
    slopes = cal.loc[names, 'calibration_slope'].to_numpy(float)
    indices = np.array([names.index(p) for p in stable], dtype=int)
    controls = pd.read_csv(root / CONTROLS, index_col=0)
    controls.index = controls.index.astype(str).str.strip()
    b = controls.loc[stable].to_numpy(float, copy=True)
    b /= np.linalg.norm(b, axis=0, keepdims=True)
    candidate = pd.read_csv(root / CANDIDATE, index_col=0)
    candidate.index = candidate.index.astype(str).str.strip()
    target = candidate.loc[stable, 'two_cohort_consensus_unit'].to_numpy(float)
    refs = indexed(root / CFBE, 'GeneID').loc[mask, list(REFERENCES)]
    saf = indexed(root / SAF, 'GeneID', '\t')
    require(len(saf) == 5567 and set(mask).issubset(saf.index),
            'incorrect full-gene SAF or gene mask')
    lengths = (pd.to_numeric(saf['End']) - pd.to_numeric(saf['Start']) + 1).to_numpy(float)

    counts = {}
    for run, patient, timepoint, folder, filename in RUNS:
        file = root / folder / run / filename
        require(file.is_file(), 'missing count file: ' + str(file))
        if filename.endswith('featureCounts.txt'):
            table = pd.read_csv(file, sep='\t', comment='#', dtype={'Geneid':str})
            require(len(table) == 5567 and 'Geneid' in table, 'invalid 07c gene count file')
            count = pd.Series(pd.to_numeric(table.iloc[:, -1]).to_numpy(),
                              index=table['Geneid'].astype(str))
        else:
            table = pd.read_csv(file, index_col=0)
            require(table.shape[1] == 1, 'invalid 07g count file')
            count = pd.to_numeric(table.iloc[:, 0])
            count.index = count.index.astype(str)
        require(len(count) == 5567 and count.index.is_unique
                and set(count.index) == set(saf.index), 'incorrect gene count universe: ' + run)
        counts[run] = count.loc[saf.index].to_numpy(float)

    runs = [entry[0] for entry in RUNS]
    count_matrix = np.column_stack([counts[r] for r in runs])
    require(np.isfinite(count_matrix).all() and (count_matrix >= 0).all(),
            'invalid sequencing counts')
    expression = counts_to_logtpm(count_matrix, lengths)
    genes = saf.index.tolist()
    gene_indices = [genes.index(gene) for gene in mask]
    x = expression[gene_indices, :]
    actual = []
    for j, (run, patient, timepoint, _, _) in enumerate(RUNS):
        for ref in REFERENCES:
            d = x[:, j] - refs[ref].to_numpy(float)
            result = project_residual(operator, d, slopes, indices, b)
            similarity = cosine(result, target)
            actual.append({'patient':patient, 'timepoint':timepoint, 'run':run,
                           'reference':ref, 'cosine_frozen_consensus':similarity,
                           'signed_consensus_projection':float(result @ target),
                           'residual_activity_norm':float(np.linalg.norm(result)),
                           'positive_alignment':bool(similarity > 0)})
    result = pd.DataFrame(actual)
    frozen = pd.read_csv(root / A7H / 'DESCRIPTIVE_new_CF_patients_reference_sensitivity.csv')
    keys = ['run', 'reference']
    require(len(frozen) == 12 and not frozen.duplicated(keys).any(),
            'unexpected frozen sample-by-reference table')
    merged = result.merge(frozen, on=keys, suffixes=('_new', '_frozen'), how='outer',
                          validate='one_to_one', indicator=True)
    require(len(merged) == 12 and merged['_merge'].eq('both').all(), 'replay sample/reference mismatch')
    maxima = []
    for column in ('cosine_frozen_consensus', 'signed_consensus_projection', 'residual_activity_norm'):
        a = merged[column + '_new'].to_numpy(float)
        f = merged[column + '_frozen'].to_numpy(float)
        error = float(np.max(np.abs(a-f)))
        maxima.append(error)
        print(f'  {column}: maximum absolute error {error:.10g}', flush=True)
        require(np.allclose(a, f, rtol=1e-9, atol=1e-7),
                'frozen mismatch in ' + column)
    require((merged['positive_alignment_new'].astype(str).str.lower()
             == merged['positive_alignment_frozen'].astype(str).str.lower()).all(),
            'positive alignment flags changed')

    # Within-patient changes cancel the shared CFBE reference.
    changes = []
    for patient, early, late, label in (
        ('CF01', 'SRR8334098', 'SRR8334097', 'D-8_to_D-7'),
        ('CF146', 'SRR10401171', 'SRR10401172', 'day33_to_day35')):
        dx = x[:, runs.index(late)] - x[:, runs.index(early)]
        r = project_residual(operator, dx, slopes, indices, b)
        changes.append({'patient':patient, 'comparison':label,
                        'signed_projection_change':float(r @ target),
                        'change_cosine_vs_consensus':cosine(r, target),
                        'change_residual_norm':float(np.linalg.norm(r))})
    fresh = pd.DataFrame(changes)
    original = pd.read_csv(root / A7H / 'DESCRIPTIVE_within_patient_changes_reference_free.csv')
    merged = fresh.merge(original, on=['patient', 'comparison'], suffixes=('_new','_frozen'),
                         validate='one_to_one', indicator=True)
    require(len(merged) == 2 and merged['_merge'].eq('both').all(),
            'within-patient change rows differ')
    for column in ('signed_projection_change','change_cosine_vs_consensus','change_residual_norm'):
        v = merged[column+'_new'].to_numpy(float)
        w = merged[column+'_frozen'].to_numpy(float)
        error = float(np.max(np.abs(v-w)))
        maxima.append(error)
        print(f'  {column}: maximum absolute error {error:.10g}', flush=True)
        require(np.allclose(v, w, atol=1e-7, rtol=1e-9),
                'frozen within-patient difference mismatch in ' + column)
    overall = max(maxima)
    print('PASS — Phase 7 07h 12 reference contrasts + 2 within-patient changes reproduced')
    print('Largest observed absolute difference:', f'{overall:.10g}')
    print('Original source, provenance records and frozen outputs remain unchanged.')
    print('Clinical validity and CF specificity are NOT established by this numerical test.')
    return overall


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=PROJECT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    print('CF-STATE | READ-ONLY PHASE 7 07H FROZEN REPLAY')
    check(args.root.resolve())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
