#!/usr/bin/env python3
"""CF-State Audit 07h: descriptive transfer of the frozen 03h CF-airway
108-iModulon direction to two newly recovered longitudinal CF donors.

No fitting or parameter selection. Uses the EXACT historical 4923-gene
soft-core projection, 116 calibration slopes, 108 stable programmes,
11-state residualisation and frozen 03h consensus. Reference contrasts
are against an external SHARED CFBE lab baseline: these are NOT matched
patient cultures and hence do not validate CF specificity.

Run from the CFstate project root: python -u src/phase7/audit_07h_....py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P4 = ROOT / 'results/phase4'
P7 = ROOT / 'results/phase7'
A7C = P7 / 'audit_07c_SRP173673_D7_D8_PAO1_feasibility'
A7G = P7 / 'audit_07g_GSE139943_CF146_D33_D35_PAO1_feasibility'
A4K = P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection'
OUT = P7 / 'audit_07h_frozen_CF_signature_two_new_patients'
FILES = {
    '07c_status': A7C / 'AUDIT_07C_STATUS.txt',
    '07g_status': A7G / 'AUDIT_07G_STATUS.txt',
    'SAF': ROOT / 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf',
    'mask': A4K / 'FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'controlled_states': P4 / 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'CFBE': A4K / 'FROZEN_CFBE_softcore_4923_log2TPM_references.csv',
    'Atlanta': A4K / 'FROZEN_Atlanta_softcore_4923_log2TPM.csv',
    'script': Path(__file__).resolve(),
}
RUNS = {
    'SRR8334098': ('CF01', 'D-8', -8, A7C),
    'SRR8334097': ('CF01', 'D-7', -7, A7C),
    'SRR10401171': ('CF146', 'day33', 33, A7G),
    'SRR10401172': ('CF146', 'day35', 35, A7G),
}
STATUS = 'CFSTATE_PHASE7_07H_FROZEN_03H_NEW_PATIENTS_DESCRIPTIVE_PROJECTION_COMPLETE_REVIEW_REQUIRED'


def check(label, cond):
    if not bool(cond):
        raise RuntimeError('FAILED: ' + label)
    print(label + ' PASS', flush=True)


def filehash(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def load_modulome(path, genes, imodulons):
    obj = json.loads(path.read_text())
    check('modulome_has_M', isinstance(obj, dict) and 'M' in obj)
    data = obj['M']
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and all(k in data for k in ('index', 'columns', 'data')):
        frame = pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])
    else:
        frame = pd.DataFrame(data)
    frame.index = frame.index.map(str)
    frame.columns = frame.columns.map(str)
    if set(genes).issubset(frame.columns) and set(imodulons).issubset(frame.index):
        frame = frame.T
    check('modulome_IDs', set(genes).issubset(frame.index) and
          set(imodulons).issubset(frame.columns) and frame.index.is_unique and frame.columns.is_unique)
    frame = frame.loc[genes, imodulons].apply(pd.to_numeric, errors='raise')
    matrix = frame.to_numpy(float)
    check('modulome_shape_finite', matrix.shape == (4923, 116) and np.isfinite(matrix).all())
    return matrix


def cos(x, y):
    d = np.linalg.norm(x) * np.linalg.norm(y)
    if d < 1e-12:
        raise RuntimeError('Undefined cosine (zero-norm vector)')
    return float(x @ y / d)


def main():
    print('=' * 78)
    print('CF-State Phase 7 — Audit 07h: two new longitudinal CF patients')
    print('FROZEN 03h signature, no model fitting')
    print('=' * 78, flush=True)
    check('new_output_directory', not OUT.exists())
    for key, value in FILES.items():
        check('input_' + key, value.is_file())
    check('07c_frozen', FILES['07c_status'].read_text().strip() ==
          'CFSTATE_PHASE7_07C_D7_D8_PAO1_READ_LEVEL_FEASIBILITY_COMPLETE_REVIEW_REQUIRED')
    check('07g_frozen', FILES['07g_status'].read_text().strip() ==
          'CFSTATE_PHASE7_07G_CF146_D33_D35_PAO1_FEASIBILITY_COMPLETE_REVIEW_REQUIRED')

    saf = pd.read_csv(FILES['SAF'], sep='\t').set_index('GeneID')
    check('SAF_5567_unique', len(saf) == 5567 and saf.index.is_unique)
    lengths = (pd.to_numeric(saf['End']) - pd.to_numeric(saf['Start']) + 1).astype(float)
    check('gene_lengths_positive', (lengths > 0).all() and np.isfinite(lengths.to_numpy()).all())
    mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
    check('frozen_mask_4923', len(mask) == 4923 and len(set(mask)) == 4923 and set(mask).issubset(saf.index))

    cal = pd.read_csv(FILES['calibration']).set_index('imodulon')
    names = cal.index.astype(str).tolist()
    stable = pd.read_csv(FILES['stable'])['imodulon'].astype(str).tolist()
    check('frozen_116_108', len(names) == 116 and len(stable) == 108 and
          len(set(stable)) == 108 and set(stable).issubset(names))
    slopes = pd.to_numeric(cal.loc[names, 'calibration_slope']).to_numpy(float)
    check('valid_frozen_slopes', np.isfinite(slopes).all() and (np.abs(slopes) > 1e-12).all())
    M = load_modulome(FILES['modulome'], mask, names)
    operator = np.linalg.pinv(M)
    check('projection_rank', np.linalg.matrix_rank(M) == 116)

    lib = pd.read_csv(FILES['controlled_states'], index_col=0)
    lib.index = lib.index.astype(str).str.strip()
    check('11_frozen_states', set(stable).issubset(lib.index) and lib.shape[1] == 11)
    B = lib.loc[stable].apply(pd.to_numeric, errors='raise').to_numpy(float, copy=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    check('state_rank_11', np.linalg.matrix_rank(B) == 11)

    cand = pd.read_csv(FILES['candidate'], index_col=0)
    cand.index = cand.index.astype(str).str.strip()
    candidate = cand.loc[stable, 'two_cohort_consensus_unit'].to_numpy(float)
    check('frozen_consensus_unit', abs(np.linalg.norm(candidate) - 1) < 1e-6)

    cfbe = pd.read_csv(FILES['CFBE']).set_index('GeneID')
    references = ['PRIMARY_CFBE_ALL4', 'SENSITIVITY_CFBE_C_PAIR', 'SENSITIVITY_CFBE_R_PAIR']
    check('CFBE_reference_columns', set(references).issubset(cfbe.columns) and
          set(mask).issubset(cfbe.index))
    cfbe = cfbe.loc[mask, references].apply(pd.to_numeric, errors='raise')
    check('CFBE_finite', np.isfinite(cfbe.to_numpy()).all())

    # Projection is identical to the historical 05j/04k4 linear softcore
    # operator; calibration intercepts cancel in expression *contrasts*.
    def project_contrast(x):
        raw = (operator @ x) * slopes
        stable_activities = pd.Series(raw, index=names).loc[stable].to_numpy(float)
        controlled_coeff = np.linalg.lstsq(B, stable_activities, rcond=None)[0]
        residual = stable_activities - B @ controlled_coeff
        check('residual_orthogonal_to_controls',
              np.allclose(B.T @ residual, np.zeros(11), atol=1e-7))
        return residual

    # Reproduce primary Atlanta comparison as a procedural positive control.
    # This does NOT establish that CFBE is a valid patient-matched reference.
    atl = pd.read_csv(FILES['Atlanta']).set_index('GeneID')
    check('Atlanta_13', atl.shape[1] == 13 and set(mask).issubset(atl.index))
    atl = atl.loc[mask].apply(pd.to_numeric, errors='raise')
    baseline = cfbe['PRIMARY_CFBE_ALL4'].to_numpy(float)
    old = []
    for specimen in atl.columns:
        delta = atl[specimen].to_numpy(float) - baseline
        old.append(cos(project_contrast(delta), candidate))
    old_median = float(np.median(old))
    print('\nREFERENCE PROCEDURE CHECK (previously evaluated Atlanta specimens):')
    print('Atlanta positives:', sum(v > 0 for v in old), '/13; median cosine:', round(old_median, 5))
    check('reproduces_previous_Atlanta_median_within_0p03',
          len(old) == 13 and sum(v > 0 for v in old) == 13 and abs(old_median - 0.32478) < 0.03)

    counts = {}
    qc = []
    for run, (patient, timepoint, day, folder) in RUNS.items():
        if folder == A7G:
            file = folder / run / 'PAO1_5567_assigned_counts.csv'
            table = pd.read_csv(file, index_col=0)
            check('07g_count_column_' + run, table.shape[1] == 1)
            col = table.columns[0]
            count = pd.to_numeric(table[col], errors='raise')
        else:
            file = folder / run / 'PAO1_featureCounts.txt'
            table = pd.read_csv(file, sep='\t', comment='#')
            check('07c_count_columns_' + run, len(table) == 5567 and 'Geneid' in table)
            count = pd.Series(pd.to_numeric(table.iloc[:, -1], errors='raise').to_numpy(),
                              index=table['Geneid'].astype(str))
        count.index = count.index.astype(str)
        check('5567_count_ids_' + run, len(count) == 5567 and count.index.is_unique and
              set(count.index) == set(saf.index))
        count = count.reindex(saf.index).astype(float)
        check('nonnegative_integer_counts_' + run, np.isfinite(count.to_numpy()).all() and
              (count >= 0).all() and np.allclose(count.to_numpy(), np.round(count.to_numpy())))
        total = int(count.sum())
        detected = int((count.loc[mask] > 0).sum())
        qc.append({'patient':patient, 'timepoint':timepoint, 'run':run,
                   'assigned_counts':total, 'genes_detected_4923':detected,
                   'detection_fraction':detected / len(mask), 'count_file':str(file.relative_to(ROOT))})
        counts[run] = count

    C = pd.DataFrame(counts)
    rates = C.div(lengths, axis=0)
    denom = rates.sum(axis=0)
    check('all_TPM_denominators_positive', (denom > 0).all())
    X = np.log2(1 + rates.div(denom, axis=1) * 1e6).loc[mask]
    check('all_log2TPM_finite', np.isfinite(X.to_numpy()).all())

    records = []
    residuals = {}
    for run, (patient, timepoint, day, _) in RUNS.items():
        x = X[run].to_numpy(float)
        for reference in references:
            r = project_contrast(x - cfbe[reference].to_numpy(float))
            similarity = cos(r, candidate)
            records.append({'patient':patient, 'timepoint':timepoint, 'run':run,
                            'reference':reference, 'cosine_frozen_consensus':similarity,
                            'signed_consensus_projection':float(r @ candidate),
                            'residual_activity_norm':float(np.linalg.norm(r)),
                            'positive_alignment':bool(similarity > 0)})
            if reference == 'PRIMARY_CFBE_ALL4':
                residuals[run] = r

    main_df = pd.DataFrame(records)
    print('\n========== NEW PATIENTS: FROZEN CF SIGNATURE ==========', flush=True)
    print(main_df[['patient','timepoint','reference','cosine_frozen_consensus',
                   'signed_consensus_projection','positive_alignment']].round(5).to_string(index=False))
    print('\nTwo independent patients, four longitudinal specimens; all against a SHARED CFBE lab reference.')

    # Within-individual changes cancel the CFBE reference exactly. This
    # asks whether the *change* aligns with the existing CF candidate,
    # not whether that patient possesses a CF-specific state.
    changes = []
    for patient, earlier, later, label in [
        ('CF01', 'SRR8334098', 'SRR8334097', 'D-8_to_D-7'),
        ('CF146', 'SRR10401171', 'SRR10401172', 'day33_to_day35'),
    ]:
        delta = X[later].to_numpy(float) - X[earlier].to_numpy(float)
        r = project_contrast(delta)
        changes.append({'patient':patient, 'comparison':label,
                        'signed_projection_change':float(r @ candidate),
                        'change_cosine_vs_consensus':cos(r, candidate),
                        'change_residual_norm':float(np.linalg.norm(r))})
    change_df = pd.DataFrame(changes)
    print('\n========== WITHIN-PATIENT CHANGES (REFERENCE CANCELS) ==========', flush=True)
    print(change_df.round(5).to_string(index=False))
    print('These changes are NOT independent patients or evidence of CF specificity.')

    OUT.mkdir(parents=True, exist_ok=False)
    main_df.to_csv(OUT / 'DESCRIPTIVE_new_CF_patients_reference_sensitivity.csv', index=False)
    change_df.to_csv(OUT / 'DESCRIPTIVE_within_patient_changes_reference_free.csv', index=False)
    pd.DataFrame(qc).to_csv(OUT / 'FROZEN_four_specimen_input_coverage_QC.csv', index=False)
    pd.DataFrame([{'input':key,'path':str(path),'sha256':filehash(path)}
                  for key, path in FILES.items()]).to_csv(OUT / 'input_sha256.csv', index=False)
    (OUT / 'DESIGN_LIMITATIONS.txt').write_text(
        'No learning, model selection or updates to the frozen 03h candidate.\n'
        'Four RNA profiles from TWO donors: CF01 and CF146. NOT four independent patients.\n'
        'Patient references are NOT matched laboratory isolates: CFBE is shared and external.\n'
        'The shared CFBE reference can induce common positive shifts; absolute similarities are descriptive.\n'
        'Within-patient differences cancel shared CFBE reference, but are not CF specificity tests.\n'
        'Reference 104-decoy + PAO1 mapping is preliminary; strain/taxonomic biases unquantified.\n'
        'CF01 and CF146 samples require independent read-taxonomy/strain validation.\n'
        'TPM computed with 5567 PAO1 SAF lengths, then limited to the frozen 4923 mask.\n'
        'No patient-level hypothesis testing; two timepoints per patient are correlated.\n'
        'Atlanta procedural reproduction is a positive pipeline check, NOT fresh validation.\n'
    )
    (OUT / 'AUDIT_07H_STATUS.txt').write_text(STATUS + '\n')
    print('\nSTATUS:', STATUS)


if __name__ == '__main__':
    main()
