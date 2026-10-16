#!/usr/bin/env python3
"""CF-State Phase 7 Audit 07k4: PAO1 gene-symbol rescue and exploratory bacterial DESeq2 contrasts versus frozen 03h direction.

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
OUT = ROOT / 'results/phase7/audit_07k4_organoid_gene_symbol_reconciliation'
FILES = {
    'supplement_4': RAW / '41598_2024_82500_MOESM4_ESM.xlsx',
    'supplement_6': RAW / '41598_2024_82500_MOESM6_ESM.xlsx',
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'states': P4 / 'audit_02b_GSE119356_CF_state_interpretation/controlled_state_reference_library.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'gff': ROOT / 'references/phase6/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz',
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


def gene_symbol_lookup(gff_path):
    """Create conservative, unique RefSeq symbol -> PAO1 locus-tag mapping.

    A symbol must match a GFF3 gene/CDS `gene` attribute exactly (ignoring
    case). Multi-locus symbols are deliberately left unresolved.
    """
    import gzip
    from collections import defaultdict
    from urllib.parse import unquote

    symbol_to_loci = defaultdict(set)
    loci_seen = set()
    with gzip.open(gff_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] not in ('gene', 'CDS'):
                continue
            attrs = {}
            for entry in fields[8].split(';'):
                if '=' in entry:
                    k, v = entry.split('=', 1)
                    attrs[k] = unquote(v).strip()
            tag = attrs.get('locus_tag', '') or attrs.get('old_locus_tag', '')
            matched = re.fullmatch(r'PA\d{4}(?:\.\d+)?', tag, re.I)
            if not matched:
                continue
            locus = tag[:6].upper()
            loci_seen.add(locus)
            for field in ('gene',):
                symbol = attrs.get(field, '').strip()
                if symbol and not re.fullmatch(r'PA\d{4}(?:\.\d+)?', symbol, re.I):
                    symbol_to_loci[symbol.casefold()].add(locus)

    unique = {sym: next(iter(ids)) for sym, ids in symbol_to_loci.items()
              if len(ids) == 1}
    ambiguous = {sym: sorted(ids) for sym, ids in symbol_to_loci.items()
                 if len(ids) > 1}
    print('GFF3 reference loci:', len(loci_seen))
    print('Unique gene-symbol aliases:', len(unique))
    print('Ambiguous reference gene symbols:', len(ambiguous))
    require('GFF3_gene_annotation_resolved', len(loci_seen) >= 4500)
    return unique, ambiguous


def read_deseq2(path, sheet, symbols, ambiguous_symbols):
    """Map PAO1 loci and exact, unique GFF3 gene symbols without collapsing rows."""
    from collections import defaultdict, Counter
    from openpyxl import load_workbook

    book = load_workbook(path, read_only=True, data_only=True)
    try:
        require('worksheet_exists_' + sheet, sheet in book.sheetnames)
        rows = book[sheet].iter_rows(values_only=True)
        header = None
        header_row = None
        for rownum, row in enumerate(rows, 1):
            labels = [str(v).strip() if v is not None else '' for v in row]
            if 'log2FoldChange' in labels and 'baseMean' in labels:
                header, header_row = labels, rownum
                break
        if header is None:
            raise RuntimeError('DESeq2 header missing: ' + sheet)

        lfc_idx = header.index('log2FoldChange')
        base_idx = header.index('baseMean')
        padj_idx = header.index('padj') if 'padj' in header else None
        observations = defaultdict(list)
        unresolved = []
        stats = Counter()

        for rownum, row in enumerate(rows, start=header_row + 1):
            if len(row) <= lfc_idx:
                continue
            raw = '' if row[0] is None else str(row[0]).strip()
            if not raw:
                continue
            loci = sorted(set(re.findall(
                r'(?<![A-Za-z0-9])(PA\d{4})(?:\.\d+)?(?!\d)', raw, re.I
            )))
            loci = [v.upper() for v in loci]
            method = ''
            gene = None
            if len(loci) == 1:
                gene, method = loci[0], 'PAO1_locus_tag'
            elif len(loci) > 1:
                method = 'multiple_PAO1_locus_tokens'
            elif raw.casefold() in ambiguous_symbols:
                method = 'ambiguous_GFF_gene_symbol'
            elif raw.casefold() in symbols:
                gene, method = symbols[raw.casefold()], 'unique_GFF_gene_symbol'
            else:
                method = 'unmapped_identifier'
            stats[method] += 1
            if gene is None:
                unresolved.append({
                    'sheet': sheet, 'excel_row': rownum,
                    'original_row_identifier': raw, 'mapping_status': method,
                    'candidate_loci': '|'.join(loci) if loci else
                         '|'.join(ambiguous_symbols.get(raw.casefold(), [])),
                })
                continue
            raw_value = row[lfc_idx]
            try:
                value = float(raw_value)
                if not np.isfinite(value):
                    value = None
            except (TypeError, ValueError):
                value = None
            observations[gene].append({
                'sheet': sheet, 'excel_row': rownum,
                'normalized_PAO1_gene': gene,
                'original_row_identifier': raw,
                'mapping_method': method,
                'log2FoldChange': value,
                'baseMean': str(row[base_idx]) if base_idx < len(row) and row[base_idx] is not None else '',
                'padj': str(row[padj_idx]) if padj_idx is not None and padj_idx < len(row) and row[padj_idx] is not None else '',
            })

        bad = {gene for gene, entries in observations.items() if len(entries) > 1}
        duplicates = [
            {**entry, 'n_rows_mapping_to_PAO1_gene': len(observations[gene])}
            for gene in sorted(bad) for entry in observations[gene]
        ]
        result = {
            gene: entries[0]['log2FoldChange']
            for gene, entries in observations.items()
            if gene not in bad and entries[0]['log2FoldChange'] is not None
        }
        print(f'{sheet}: {len(result)} unique finite-LFC genes; '
              f'{len(bad)} ambiguous duplicate genes excluded')
        print('  Resolution counts:', dict(stats))
        print('  Unresolved examples:', [r['original_row_identifier'] for r in unresolved[:15]])
        print('  Duplicate examples:', sorted(bad)[:12])
        require('minimum_parsed_genes_' + sheet, len(result) >= 3000)
        return pd.Series(result, name=sheet, dtype=float), duplicates, bad, unresolved, stats
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
    print('CF-State Phase 7 — Audit 07k4: gene symbol reconciliation and projection')
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

    symbol_map, multi_symbols = gene_symbol_lookup(FILES['gff'])

    # Parse all contrasts before choosing the common identifier universe.
    # Exclude every PAO1 gene with ambiguous duplicate rows in ANY source
    # comparison to keep the gene universe consistent across projections.
    frames = {}
    all_duplicates = []
    all_unresolved = []
    summary_mapping = []
    ambiguous_across_comparisons = set()
    for label, file_key, sheet in COMPARISONS:
        series, duplicates, bad, unresolved, stats = read_deseq2(
            FILES[file_key], sheet, symbol_map, multi_symbols
        )
        all_unresolved.extend(unresolved)
        summary_mapping.append({'contrast': label, 'sheet': sheet, **dict(stats),
                                'unresolved_rows': len(unresolved), 'duplicated_PAO1_genes': len(bad),
                                'unique_finite_lfc_genes': len(series)})
        frames[label] = series
        all_duplicates.extend(duplicates)
        ambiguous_across_comparisons.update(bad)

    OUT.mkdir(parents=True, exist_ok=False)
    dup_columns = [
        'sheet', 'excel_row', 'normalized_PAO1_gene',
        'original_row_identifier', 'log2FoldChange', 'baseMean', 'padj',
        'n_rows_mapping_to_PAO1_gene'
    ]
    dup_columns.insert(4, 'mapping_method')
    pd.DataFrame(all_unresolved, columns=['sheet', 'excel_row', 'original_row_identifier',
                'mapping_status', 'candidate_loci']).to_csv(
        OUT / 'QC_unresolved_source_identifiers.csv', index=False
    )
    pd.DataFrame(summary_mapping).fillna(0).to_csv(
        OUT / 'QC_gene_identifier_mapping_counts.csv', index=False
    )
    pd.DataFrame(all_duplicates, columns=dup_columns).to_csv(
        OUT / 'QC_all_duplicate_PAO1_identifier_rows.csv', index=False
    )
    pd.DataFrame({'excluded_PAO1_gene': sorted(ambiguous_across_comparisons)}).to_csv(
        OUT / 'QC_ambiguous_PAO1_genes_excluded_in_all_comparisons.csv', index=False
    )
    print(
        f'GLOBAL AMBIGUOUS ID FILTER: {len(ambiguous_across_comparisons)} '
        'PAO1 identifiers removed from every contrast', flush=True
    )
    adequate = True
    for label in frames:
        frames[label] = frames[label].drop(
            list(ambiguous_across_comparisons), errors='ignore'
        )
        common = [g for g in mask if g in frames[label].index]
        print(
            f'{label}: {len(frames[label])} unambiguous finite-LFC PAO1 IDs; '
            f'{len(common)}/4923 frozen genes', flush=True
        )
        ratio = len(common) / len(mask)
        print(f'  USABLE FROZEN-MASK COVERAGE: {len(common)}/{len(mask)} = {ratio:.1%}')
        print('  ORIGINAL 07k2 COVERAGE REQUIREMENT: >=4400/4923 (89.4%)')
        if len(common) < 4400:
            print('  QC WARNING: ORIGINAL 07k2 COVERAGE REQUIREMENT NOT MET')
        if len(common) < 4400:
            adequate = False
        # Ten genes per model column is a guard for numerical exploration,
        # NOT a validated biological coverage threshold.
        require(label + '_minimum_numerical_support_only', len(common) >= 116 * 10)

    if not adequate:
        status = 'CFSTATE_PHASE7_07K4_SYMBOL_RECONCILIATION_INCOMPLETE_NO_PROJECTION'
        (OUT / 'AUDIT_07K4_STATUS.txt').write_text(status + '\n')
        print('Some contrasts still fail the original 4400-gene coverage gate.')
        print('Identifier QC files saved. Projection intentionally not performed.')
        print('STATUS:', status)
        return

    records = []
    activities = {}
    programme_qc = []
    gene_dropout_qc = []
    rng = np.random.default_rng(20261014)
    for label, series in frames.items():
        # Only use observed finite DESeq2 coefficients; never zero-fill missing genes.
        common = [g for g in mask if g in series.index]
        mask_positions = {gene: i for i, gene in enumerate(mask)}
        indices = [mask_positions[g] for g in common]
        M_sub = M[indices]
        weight_fraction = (
            (M_sub * M_sub).sum(axis=0) / np.maximum((M * M).sum(axis=0), 1e-20)
        )
        for p, fraction in zip(names, weight_fraction):
            programme_qc.append({
                'contrast': label, 'imodulon': p,
                'fraction_reference_loading_energy_on_observed_genes': float(fraction),
            })
        print(
            label + ' reference-loading energy retained (all 116): ' +
            'min={:.3f}, 10th_pct={:.3f}, median={:.3f}'.format(
                np.min(weight_fraction), np.percentile(weight_fraction, 10),
                np.median(weight_fraction)
            ), flush=True
        )
        require(label + '_model_rank', np.linalg.matrix_rank(M_sub) == 116)
        x = series.loc[common].to_numpy(float)
        raw = np.linalg.pinv(M_sub) @ x
        calibrated = pd.Series(raw * slopes, index=names).loc[stable].to_numpy(float)
        controlled = calibrated - B @ np.linalg.lstsq(B, calibrated, rcond=None)[0]
        score = cosine(controlled, consensus)
        projection = float(controlled @ consensus)
        activities[label] = controlled
        # Deliberately weaker-than-original numerical stability check:
        # remove 20% of already available genes, then recompute the
        # pseudoinverse. This DOES NOT address systematic missingness.
        for iteration in range(1, 13):
            keep = np.sort(rng.choice(len(common), size=int(len(common)*0.8), replace=False))
            M_drop = M_sub[keep, :]
            x_drop = x[keep]
            drop_raw = np.linalg.lstsq(M_drop, x_drop, rcond=None)[0]
            drop_calibrated = pd.Series(drop_raw * slopes, index=names).loc[stable].to_numpy(float)
            drop_residual = drop_calibrated - B @ np.linalg.lstsq(B, drop_calibrated, rcond=None)[0]
            gene_dropout_qc.append({
                'contrast': label, 'iteration': iteration,
                'n_genes': int(len(keep)),
                'cosine_vs_frozen_03h_consensus': cosine(drop_residual, consensus),
                'cosine_vs_full_available_gene_projection': cosine(drop_residual, controlled),
            })
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
    weight_qc = pd.DataFrame(programme_qc)
    dropout = pd.DataFrame(gene_dropout_qc)
    print('\n========== GENE-COVERAGE SENSITIVITY: EXPLORATORY ONLY ==========', flush=True)
    print(dropout.groupby('contrast').agg(
        min_cosine_vs_candidate=('cosine_vs_frozen_03h_consensus', 'min'),
        median_cosine_vs_candidate=('cosine_vs_frozen_03h_consensus', 'median'),
        max_cosine_vs_candidate=('cosine_vs_frozen_03h_consensus', 'max'),
        min_cosine_vs_full_subset=('cosine_vs_full_available_gene_projection', 'min'),
    ).round(5).to_string())
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
    print('  PAO1 IDs with multiple published source rows excluded from all projections.')
    print('  Unique gene-symbol aliases recovered via GFF3; ambiguous symbols remain excluded.')
    print('  This test assesses mechanistic directional agreement, NOT CF specificity or new patient validation.')

    weight_qc.to_csv(OUT / 'QC_programme_gene_loading_energy_retained.csv', index=False)
    dropout.to_csv(OUT / 'QC_random_20pct_additional_gene_dropout.csv', index=False)
    result.to_csv(OUT / 'DESCRIPTIVE_frozen_03h_projection_of_public_DESeq2_logFC.csv', index=False)
    concordance.to_csv(OUT / 'DESCRIPTIVE_between_published_logFC_concordance.csv', index=False)
    pd.DataFrame([{'input': key, 'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
                  for key, path in FILES.items()]).to_csv(OUT / 'input_sha256.csv', index=False)
    print('  The original 4400-gene coverage gate is required for this exploratory projection.')
    print('  Random gene dropout does not correct systematic gene-identifier omissions.')
    status = 'CFSTATE_PHASE7_07K4_SYMBOL_RECONCILED_EXPLORATORY_NOT_VALIDATION'
    (OUT / 'AUDIT_07K4_STATUS.txt').write_text(status + '\n')
    print('\nSTATUS:', status)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('AUDIT 07k4 INCOMPLETE:', str(exc), file=sys.stderr)
        raise
