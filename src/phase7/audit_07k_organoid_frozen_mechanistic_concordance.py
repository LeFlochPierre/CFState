#!/usr/bin/env python3
"""CF-State Phase 7 Audit 07k: public bacterial DESeq2 contrasts versus frozen 03h direction.

EXPLORATORY MECHANISTIC CONCORDANCE, NOT CF-SPECIFIC VALIDATION.
The published in-vivo contrast reuses earlier CF clinical cohorts. The
co-culture and in-vivo comparisons use DESeq2 log2 fold changes, not
per-sample log2TPM. No expression matrix or model is refitted.
"""
from pathlib import Path
import json
import hashlib
import re
import sys
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
RAW = ROOT / 'data/raw/airway_organoid_2025_public_supplements'
P4 = ROOT / 'results/phase4'
OUT = ROOT / 'results/phase7/audit_07k_public_organoid_mechanistic_concordance'
FILES = {
    'supplement_4': RAW / '41598_2024_82500_MOESM4_ESM.xlsx',
    'supplement_6': RAW / '41598_2024_82500_MOESM6_ESM.xlsx',
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'states': P4 / 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'script': Path(__file__).resolve(),
}
COMPARISONS = [
    ('SUP4_COCULT_vs_LAB', 'supplement_4', 'res_cocult_vs_invitro_Fig4b'),
    ('SUP6_COCULT_vs_LAB', 'supplement_6', 'res_cocult_vs_invitro_apeglm'),
    ('SUP6_INVIVO_vs_LAB', 'supplement_6', 'res_invivo_vs_invitro_apeglm'),
]


def require(name, test):
    if not bool(test):
        raise RuntimeError('CHECK FAILED: ' + name)
    print(name + ' PASS', flush=True)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def read_deseq2(path, sheet):
    from openpyxl import load_workbook
    book = load_workbook(path, read_only=True, data_only=True)
    try:
        require('worksheet_exists_' + sheet, sheet in book.sheetnames)
        rows = book[sheet].iter_rows(values_only=True)
        header = None
        for row in rows:
            labels = [str(x).strip() if x is not None else '' for x in row]
            if 'log2FoldChange' in labels and 'baseMean' in labels:
                header = labels
                break
        if header is None:
            raise RuntimeError('DESeq2 header missing: ' + sheet)
        lfc_idx = header.index('log2FoldChange')
        id_idx = 0  # Published DESeq2 rownames in first column
        result = {}
        raw_ids = []
        finite_lfc = 0
        for row in rows:
            if len(row) <= max(lfc_idx, id_idx):
                continue
            raw_id = '' if row[id_idx] is None else str(row[id_idx]).strip()
            if len(raw_ids) < 5 and raw_id:
                raw_ids.append(raw_id)
            match = re.search(r'(?<![A-Za-z0-9])(PA\d{4})(?:\.\d+)?(?!\d)', raw_id, re.I)
            if not match:
                continue
            gene = match.group(1).upper()
            raw_value = row[lfc_idx]
            if raw_value is None or raw_value == '':
                continue
            try:
                value = float(raw_value)
            except (ValueError, TypeError):
                continue
            if not np.isfinite(value):
                continue
            finite_lfc += 1
            if gene in result:
                raise RuntimeError(f'Duplicate PAO1 ID {gene} in {sheet}')
            result[gene] = value
        if len(result) < 1000:
            raise RuntimeError(f'Only {len(result)} PAO1 genes parsed in {sheet}; first IDs: {raw_ids!r}')
        return pd.Series(result, name=sheet, dtype=float)
    finally:
        book.close()


def load_modulome(path, genes, names):
    obj = json.loads(path.read_text())
    require('modulome_has_M', isinstance(obj, dict) and 'M' in obj)
    data = obj['M']
    if isinstance(data, str):
        data = json.loads(data)
    if isinstance(data, dict) and all(k in data for k in ('index', 'columns', 'data')):
        table = pd.DataFrame(data['data'], index=data['index'], columns=data['columns'])
    else:
        table = pd.DataFrame(data)
    table.index = table.index.map(str)
    table.columns = table.columns.map(str)
    if set(genes).issubset(table.columns) and set(names).issubset(table.index):
        table = table.T
    require('model_gene_IDs', set(genes).issubset(table.index))
    require('model_programme_IDs', set(names).issubset(table.columns))
    m = table.loc[genes, names].apply(pd.to_numeric, errors='raise').to_numpy(float)
    require('model_finite', np.isfinite(m).all())
    return m


def cosine(x, y):
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 1e-12 else float('nan')


def main():
    print('=' * 76)
    print('CF-State Phase 7 — Audit 07k: public organoid mechanistic concordance')
    print('=' * 76, flush=True)
    require('new_output_directory', not OUT.exists())
    for label, path in FILES.items():
        require('input_' + label, path.is_file())

    mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
    require('mask_4923_unique', len(mask) == 4923 and len(set(mask)) == 4923)
    calibration = pd.read_csv(FILES['calibration']).set_index('imodulon')
    names = calibration.index.astype(str).tolist()
    stable = pd.read_csv(FILES['stable'])['imodulon'].astype(str).tolist()
    require('116_108_programmes', len(names) == 116 and len(stable) == 108 and set(stable).issubset(names))
    slopes = pd.to_numeric(calibration.loc[names, 'calibration_slope']).to_numpy(float)
    require('calibration_valid', np.isfinite(slopes).all() and np.all(np.abs(slopes) > 1e-12))
    M = load_modulome(FILES['modulome'], mask, names)

    lib = pd.read_csv(FILES['states'], index_col=0)
    lib.index = lib.index.astype(str).str.strip()
    require('controlled_state_11', lib.shape[1] == 11 and set(stable).issubset(lib.index))
    B = lib.loc[stable].apply(pd.to_numeric, errors='raise').to_numpy(float, copy=True)
    B = B / np.linalg.norm(B, axis=0, keepdims=True)
    require('controlled_basis_rank', np.linalg.matrix_rank(B) == 11)
    cand = pd.read_csv(FILES['candidate'], index_col=0)
    cand.index = cand.index.astype(str).str.strip()
    consensus = cand.loc[stable, 'two_cohort_consensus_unit'].to_numpy(float)
    require('candidate_108_unit', abs(np.linalg.norm(consensus) - 1) < 1e-5)

    frames = {}
    for label, file_key, sheet in COMPARISONS:
        series = read_deseq2(FILES[file_key], sheet)
        common = [g for g in mask if g in series.index]
        print(f'{label}: {len(series)} PAO1 IDs with finite LFC; {len(common)}/4923 frozen genes', flush=True)
        require(label + '_adequate_gene_coverage', len(common) >= 4400)
        frames[label] = series

    records = []
    activities = {}
    for label, series in frames.items():
        # Only use observed finite DESeq2 coefficients; never zero-fill missing genes.
        common = [g for g in mask if g in series.index]
        mask_positions = {gene: i for i, gene in enumerate(mask)}
        indices = [mask_positions[g] for g in common]
        M_sub = M[indices]
        require(label + '_model_rank', np.linalg.matrix_rank(M_sub) == 116)
        x = series.loc[common].to_numpy(float)
        raw = np.linalg.pinv(M_sub) @ x
        calibrated = pd.Series(raw * slopes, index=names).loc[stable].to_numpy(float)
        controlled = calibrated - B @ np.linalg.lstsq(B, calibrated, rcond=None)[0]
        score = cosine(controlled, consensus)
        projection = float(controlled @ consensus)
        activities[label] = controlled
        records.append({
            'contrast': label,
            'mapped_PAO1_genes': len(series),
            'frozen_mask_genes_used': len(common),
            'cosine_vs_frozen_03h_consensus': score,
            'signed_projection': projection,
            'residual_activity_norm': float(np.linalg.norm(controlled)),
            'interpretation_scope': 'EXPLORATORY_DESEQ2_LOG2FC_PROXY_NOT_CLINICAL_VALIDATION',
        })

    result = pd.DataFrame(records)
    print('\n========== PUBLIC ORGANOID GENE-LEVEL MECHANISTIC COMPARISON ==========', flush=True)
    print(result.round(5).to_string(index=False))

    pairs = []
    labels = list(frames)
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            genes = frames[a].index.intersection(frames[b].index)
            x = frames[a].loc[genes]
            y = frames[b].loc[genes]
            pairs.append({
                'comparison_A': a, 'comparison_B': b, 'shared_genes': len(genes),
                'spearman_gene_logFC': float(x.corr(y, method='spearman')),
                'pearson_gene_logFC': float(x.corr(y)),
                'cosine_108_controlled_residual': cosine(activities[a], activities[b]),
            })
    concordance = pd.DataFrame(pairs)
    print('\n========== ORGANOID vs PUBLISHED INVIVO LFC CONCORDANCE ==========', flush=True)
    print(concordance.round(5).to_string(index=False))

    print('\nINTERPRETATION LIMITS:')
    print('  These are shrinkage-adjusted DESeq2 logFC contrasts, NOT per-sample log2TPM.')
    print('  Pseudoinverse is recalculated on each contrast\'s available PAO1 genes.')
    print('  SUP4 and SUP6 co-culture analyses come from the SAME study, not independent cohorts.')
    print('  SUP6 INVIVO contrast uses prior published clinical cohorts, potentially overlapping CF-State discovery.')
    print('  Neither co-culture contrast isolates CF organoids from healthy organoids.')
    print('  This test assesses mechanistic directional agreement, NOT CF specificity or new patient validation.')

    OUT.mkdir(parents=True, exist_ok=False)
    result.to_csv(OUT / 'DESCRIPTIVE_frozen_03h_projection_of_public_DESeq2_logFC.csv', index=False)
    concordance.to_csv(OUT / 'DESCRIPTIVE_between_published_logFC_concordance.csv', index=False)
    pd.DataFrame([{'input': key, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
                  for key, path in FILES.items()]).to_csv(OUT / 'input_sha256.csv', index=False)
    status = 'CFSTATE_PHASE7_07K_PUBLIC_ORGANOID_MECHANISTIC_CONCORDANCE_REVIEW_REQUIRED'
    (OUT / 'AUDIT_07K_STATUS.txt').write_text(status + '\n')
    print('\nSTATUS:', status)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('AUDIT 07k INCOMPLETE:', str(exc), file=sys.stderr)
        raise
