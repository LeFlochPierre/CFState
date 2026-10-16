#!/usr/bin/env python3
"""CF-State Phase 6 / Audit 06e: reproducible PAO1 gene-annotation triage.

Annotates existing exploratory Atlanta residual loadings. No new model fitting,
CF-specific programme claims, differential expression tests or enrichment p-values.
Requires the PAO1 GCF_000006765.1 RefSeq genomic GFF3 to have been downloaded.
"""
from __future__ import annotations

import gzip
import hashlib
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd


def require(condition, label):
    if not condition:
        raise RuntimeError(label)
    print(label, 'PASS')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def parse_attributes(text):
    values = {}
    for part in text.split(';'):
        if '=' in part:
            key, value = part.split('=', 1)
            values[key] = unquote(value)
    return values


def read_gff(gff_path):
    rows = {}
    with gzip.open(gff_path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line or line.startswith('#'):
                continue
            fields = line.rstrip('\n').split('\t')
            if len(fields) != 9 or fields[2] not in ('gene', 'CDS'):
                continue
            attrs = parse_attributes(fields[8])
            tags = ' '.join([attrs.get('locus_tag', ''), attrs.get('old_locus_tag', '')])
            gene_ids = re.findall(r'(?<![A-Za-z0-9])PA\d{4}(?![0-9])', tags)
            if not gene_ids:
                continue
            for gene_id in gene_ids:
                candidate = {
                    'GeneID': gene_id,
                    'gene_symbol': attrs.get('gene', ''),
                    'product': attrs.get('product', ''),
                    'annotation_feature': fields[2],
                    'reference_accession': fields[0],
                    'start': int(fields[3]),
                    'end': int(fields[4]),
                }
                previous = rows.get(gene_id)
                # Prefer protein/CDS annotation for product description.
                if previous is None or (fields[2] == 'CDS' and previous['annotation_feature'] != 'CDS'):
                    rows[gene_id] = candidate
                elif previous is not None and not previous['product'] and candidate['product']:
                    rows[gene_id] = candidate
    return pd.DataFrame(list(rows.values())).set_index('GeneID')


def main():
    root = Path.cwd()
    prior = root / 'results/phase6/audit_06d_component_gene_annotation_triage'
    out = root / 'results/phase6/audit_06e_PAO1_functional_annotation'
    gff = Path(sys.argv[1]) if len(sys.argv) == 2 else (
        root / 'references/phase6/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz'
    )

    print('=== CF-State Phase 6 — Audit 06e: functional annotation ===')
    require(not out.exists(), 'new_output_directory')
    require(prior.is_dir(), 'prior_06d_dir')
    require(
        (prior / 'AUDIT_06D_STATUS.txt').read_text().strip() ==
        'CFSTATE_PHASE6_06D_EXPLORATORY_GENE_TRIAGE_COMPLETE',
        'prior_06d_frozen'
    )
    require(gff.is_file() and gff.stat().st_size > 50000, 'reference_GFF3_present')

    loadings_path = prior / 'EXPLORATORY_ALL_4923_PC_LOADINGS_AND_COVERAGE.csv'
    loadings = pd.read_csv(loadings_path).set_index('GeneID')
    axes = [f'exploratory_PC{k}_weight' for k in (1, 2, 3)]
    require(loadings.shape[0] == 4923 and loadings.index.is_unique, 'gene_universe_4923')
    require(all(col in loadings.columns for col in axes), 'three_existing_components')
    require(np.isfinite(loadings[axes].to_numpy(dtype=float)).all(), 'finite_loadings')
    for col in ['Cornforth_detected_frac_all13',
                'Cornforth_detected_frac_CF7',
                'Cornforth_detected_frac_nonCF6']:
        require(col in loadings.columns and loadings[col].between(0, 1).all(), col)

    annotations = read_gff(gff)
    require(len(annotations) > 4500, 'GFF3_contains_PAO1_gene_annotations')
    shared = loadings.index.intersection(annotations.index)
    require(len(shared) >= 4400, 'GFF3_locus_tag_match_at_least_4400')
    print(f'Annotation coverage: {len(shared)}/4923')
    merged = loadings.join(annotations, how='left')
    merged['product'] = merged['product'].fillna('')
    merged['gene_symbol'] = merged['gene_symbol'].fillna('')
    merged['description_unresolved'] = merged['product'].str.lower().str.contains(
        r'hypothetical|uncharacterized|unknown function|^$', regex=True
    )
    merged['nonCF_minus_CF_detection_fraction'] = (
        merged['Cornforth_detected_frac_nonCF6'] - merged['Cornforth_detected_frac_CF7']
    )

    report_rows = []
    for pc in (1, 2, 3):
        column = f'exploratory_PC{pc}_weight'
        for direction, ascending in [('positive', False), ('negative', True)]:
            order = merged.sort_values(column, ascending=ascending, kind='stable').head(20)
            for rank, (gene, record) in enumerate(order.iterrows(), start=1):
                report_rows.append({
                    'component': f'PC{pc}',
                    'loading_direction': direction,
                    'rank': rank,
                    'GeneID': gene,
                    'signed_loading': float(record[column]),
                    'gene_symbol': record['gene_symbol'],
                    'product': record['product'],
                    'Atlanta_sd_log2TPM': record['Atlanta_sd_log2TPM'],
                    'Cornforth_detected_frac_CF7': record['Cornforth_detected_frac_CF7'],
                    'Cornforth_detected_frac_nonCF6': record['Cornforth_detected_frac_nonCF6'],
                })
    tops = pd.DataFrame(report_rows)

    summaries = []
    for pc in (1, 2, 3):
        column = f'exploratory_PC{pc}_weight'
        largest = merged.loc[merged[column].abs().nlargest(50).index]
        summaries.append({
            'component': f'PC{pc}',
            'top50_mean_Cornforth_detection_CF':
                largest['Cornforth_detected_frac_CF7'].mean(),
            'top50_mean_Cornforth_detection_nonCF':
                largest['Cornforth_detected_frac_nonCF6'].mean(),
            'top50_fraction_function_unresolved':
                largest['description_unresolved'].mean(),
            'reference_function_unresolved_fraction':
                merged['description_unresolved'].mean(),
        })
    summary = pd.DataFrame(summaries)

    out.mkdir(parents=True, exist_ok=False)
    merged.to_csv(out / 'EXPLORATORY_ALL4923_LOADINGS_WITH_REFSEQ_ANNOTATION.csv', index_label='GeneID')
    tops.to_csv(out / 'EXPLORATORY_TOP20_SIGNED_GENES_PER_PC_ANNOTATED.csv', index=False)
    summary.to_csv(out / 'DESCRIPTIVE_GENE_ANNOTATION_AND_COVERAGE_SUMMARY.csv', index=False)
    (out / 'REFERENCE_PROVENANCE.txt').write_text(
        'GFF3: ' + str(gff.resolve()) + '\n' +
        'GFF3 SHA256: ' + sha256(gff) + '\n' +
        'Prior 06d SHA256: ' + sha256(loadings_path) + '\n' +
        'Reference genome: PAO1 GCF_000006765.1 (NC_002516.2)\n'
    )
    (out / 'INTERPRETATION_LIMITS.txt').write_text(
        'Annotation-only triage; does NOT perform statistical pathway enrichment.\n'
        'Component sign is arbitrary and individual PCs may rotate after retraining.\n'
        'Reference gene products are annotations, not experimental validations.\n'
        'Group differences in detection are confounded by depth and PAO1 mapping.\n'
        'Top genes were selected from Atlanta expression and cannot validate CF specificity.\n'
        'Genomic islands/strain absence and technical batch effects remain untested.\n'
    )
    (out / 'AUDIT_06E_STATUS.txt').write_text(
        'CFSTATE_PHASE6_06E_DESCRIPTIVE_GENE_ANNOTATION_COMPLETE\n'
    )
    for pc in ('PC2', 'PC3'):
        print(f'\n=== {pc} — top 10 POSITIVE and NEGATIVE genes ===')
        part = tops[(tops.component == pc) & (tops['rank'] <= 10)]
        display = part[['loading_direction', 'rank', 'GeneID', 'signed_loading',
                        'gene_symbol', 'product', 'Cornforth_detected_frac_nonCF6']].copy()
        display['product'] = display['product'].str.slice(0, 65)
        print(display.round(3).to_string(index=False))
    print('\n=== DESCRIPTIVE QC (not a test of enrichment) ===')
    print(summary.round(4).to_string(index=False))
    print('\nSTATUS: CFSTATE_PHASE6_06E_DESCRIPTIVE_GENE_ANNOTATION_COMPLETE')


if __name__ == '__main__':
    main()
