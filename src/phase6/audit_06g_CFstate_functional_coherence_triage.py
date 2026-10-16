#!/usr/bin/env python3
"""CF-State / Phase 6 / Audit 06g: hypothesis-generating PC functional coherence.

Uses frozen 06e RefSeq annotation and existing three PC loadings; no refitting.
Tests signed top-100 gene lists against exploratory keyword annotations and
spatial-neighbourhood clustering using an Atlanta-SD/external-detection matched
null. This is NOT curated GO enrichment, CF specificity, or independent validation.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
BASE = ROOT / 'results/phase6'
SOURCE = BASE / 'audit_06e_PAO1_functional_annotation'
SOURCE_TABLE = SOURCE / 'EXPLORATORY_ALL4923_LOADINGS_WITH_REFSEQ_ANNOTATION.csv'
STATUS_06F = (BASE / 'audit_06f_shared_gene_support_transfer_sensitivity'
              / 'AUDIT_06F_STATUS.txt')
OUT = BASE / 'audit_06g_functional_coherence_triage'
SEED = 20261014
N_NULL = 999
TOP_N = 100
DISTANCE_BP = 10000

# Exploratory, broad text-derived categories, fixed before inspecting this audit.
# Categories overlap, and missing function annotations reduce interpretability.
THEMES = {
    'flagella_and_motility': r'flagell|chemotax|motility|twitch|\bpil(?:us|i)\b',
    'type_III_secretion': r'type[ -]?(?:iii|3) secretion|injectisome|\b(?:psc|pcr|pop|exs)[a-z0-9]+\b',
    'type_VI_secretion': r'type[ -]?(?:vi|6) secretion|\b(?:tss|hsi|hcp|vgrg|clpv)[a-z0-9]*\b',
    'iron_and_siderophores': r'\biron\b|siderophore|pyoverdin|pyochelin|\bheme\b|ferric|ferrous|\b(?:pvd|pch)[a-z0-9]+\b',
    'respiration_and_nitrogen': r'respirat|cytochrome|denitrif|nitrate|nitrite|anaerob|\boxidase\b',
    'quorum_and_biofilm': r'quorum|rhamnolipid|biofilm|alginate|exopolysaccharide|phenazine|pyocyanin|\b(?:pel|psl|rhl|las)[a-z0-9]+\b',
    'ribosome_and_translation': r'ribosom|translation|elongation factor|aminoacyl.trna|trna synthetase',
    'envelope_and_efflux': r'outer membrane|efflux|lipopolysaccharide|peptidoglycan|\bporin\b|\bmex[a-z0-9]+\b',
    'stress_and_DNA_repair': r'oxidative stress|superoxide|peroxide|chaperon|dna repair|heat shock|\blexa\b|\breca\b',
    'urease_and_amino_acid_metabolism': r'urease|amino acid|aminotransferase|glutamate|glutamine|histidine',
    'unresolved_function': r'^$',  # special mask assigned below
}


def check(condition, label):
    if not condition:
        raise RuntimeError('FAILED: ' + label)
    print(label + ' PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def bh_fdr(pvalues):
    x = np.asarray(pvalues, dtype=float)
    result = np.empty(len(x), dtype=float)
    order = np.argsort(x)
    ranked = x[order] * len(x) / np.arange(1, len(x) + 1)
    result[order] = np.minimum(1, np.minimum.accumulate(ranked[::-1])[::-1])
    return result


def nearby_pairs(selected_indices, coordinates, chromosomes, radius=DISTANCE_BP):
    good = np.isfinite(coordinates[selected_indices])
    idx = selected_indices[good]
    if len(idx) < 2:
        return 0
    pairs = 0
    for chrom in np.unique(chromosomes[idx]):
        positions = np.sort(coordinates[idx[chromosomes[idx] == chrom]])
        pairs += int(np.sum(np.diff(positions) <= radius))
    return pairs


def matched_samples(observed_idx, strata, n_replicates, rng):
    # Samples uniformly within the same SD/detection strata as observed genes.
    unique, n = np.unique(strata[observed_idx], return_counts=True)
    pools = [np.flatnonzero(strata == stratum) for stratum in unique]
    check(all(len(pool) >= required for pool, required in zip(pools, n)),
          'matched_strata_have_sufficient_genes')
    draws = []
    for _ in range(n_replicates):
        picks = [rng.choice(pool, size=int(k), replace=False)
                 for pool, k in zip(pools, n)]
        draw = np.concatenate(picks)
        check_unique = len(np.unique(draw)) == len(observed_idx)
        if not check_unique:
            raise RuntimeError('Duplicate within one matched null draw')
        draws.append(draw)
    return draws


def main():
    print('=' * 76)
    print('CF-State Phase 6 — Audit 06g: exploratory functional coherence')
    print('=' * 76)
    check(not OUT.exists(), 'new_output_directory')
    check(SOURCE_TABLE.is_file(), '06e_annotated_loadings')
    check((SOURCE / 'AUDIT_06E_STATUS.txt').read_text().strip() ==
          'CFSTATE_PHASE6_06E_DESCRIPTIVE_GENE_ANNOTATION_COMPLETE',
          '06e_status')
    check(STATUS_06F.read_text().strip() ==
          'CFSTATE_PHASE6_06F_SHARED_GENE_SUPPORT_SENSITIVITY_COMPLETE',
          '06f_status')

    df = pd.read_csv(SOURCE_TABLE, dtype={'GeneID': str}).set_index('GeneID')
    check(df.shape[0] == 4923 and df.index.is_unique, '4923_unique_PAO1_genes')
    axes = [f'exploratory_PC{k}_weight' for k in (1, 2, 3)]
    needed = axes + ['Atlanta_sd_log2TPM', 'Cornforth_detected_frac_all13',
                     'product', 'gene_symbol', 'start', 'reference_accession']
    check(all(col in df.columns for col in needed), 'required_columns')
    for col in axes + ['Atlanta_sd_log2TPM', 'Cornforth_detected_frac_all13']:
        check(np.isfinite(pd.to_numeric(df[col]).to_numpy()).all(),
              'finite_' + col)
    check(df['Cornforth_detected_frac_all13'].between(0, 1).all(),
          'external_detection_bounded')
    check((df['Atlanta_sd_log2TPM'] >= 0).all(), 'Atlanta_sd_nonnegative')

    # Annotation categories from existing frozen RefSeq gene products.
    product = df['product'].fillna('').astype(str).str.lower()
    symbol = df['gene_symbol'].fillna('').astype(str).str.lower()
    descriptions = product + ' ' + symbol
    unresolved = product.str.contains(r'^\s*$|hypothetical|uncharacterized|unknown function',
                                      case=False, regex=True).to_numpy()
    category_masks = {}
    for name, pattern in THEMES.items():
        if name == 'unresolved_function':
            category_masks[name] = unresolved
        else:
            category_masks[name] = descriptions.str.contains(
                pattern, regex=True, case=False).to_numpy()
    print('\nKeyword annotation coverage:')
    for name, mask in category_masks.items():
        print(' ', name, int(mask.sum()), 'of 4923')

    # Null strata: quintiles of within-Atlanta SD, plus external detection bands.
    sd_bin = pd.qcut(df['Atlanta_sd_log2TPM'].rank(method='first'),
                     5, labels=False).to_numpy(dtype=int)
    coverage = df['Cornforth_detected_frac_all13'].to_numpy(dtype=float)
    coverage_bin = np.digitize(coverage, bins=[0.5, 0.85], right=True)
    strata = sd_bin * 3 + coverage_bin
    starts = pd.to_numeric(df['start'], errors='coerce').to_numpy(dtype=float)
    chromosomes = df['reference_accession'].fillna('UNMAPPED').to_numpy(dtype=str)
    rng = np.random.default_rng(SEED)

    results = []
    genomic = []
    top_genes = []
    for pc in (1, 2, 3):
        name = f'PC{pc}'
        weights = df[f'exploratory_PC{pc}_weight'].to_numpy(dtype=float)
        for direction, selected in (
            ('positive', np.argsort(weights)[-TOP_N:]),
            ('negative', np.argsort(weights)[:TOP_N]),
        ):
            null_sets = matched_samples(selected, strata, N_NULL, rng)
            for index in selected:
                top_genes.append({
                    'component': name, 'direction': direction,
                    'GeneID': df.index[index], 'weight': float(weights[index]),
                    'gene_symbol': str(df['gene_symbol'].iloc[index]),
                    'product': str(df['product'].iloc[index]),
                })
            for category, mask in category_masks.items():
                observed = int(mask[selected].sum())
                null_counts = np.array([int(mask[idx].sum()) for idx in null_sets])
                results.append({
                    'component': name, 'direction': direction,
                    'category': category,
                    'total_category_genes': int(mask.sum()),
                    'top100_category_genes': observed,
                    'matched_null_median': float(np.median(null_counts)),
                    'matched_null_95th': float(np.quantile(null_counts, .95)),
                    'empirical_p_enrichment': (1 + int(np.sum(null_counts >= observed))) / (N_NULL + 1),
                })
            observed_pairs = nearby_pairs(selected, starts, chromosomes)
            null_pairs = np.array([nearby_pairs(idx, starts, chromosomes)
                                   for idx in null_sets])
            genomic.append({
                'component': name, 'direction': direction,
                'top100_neighbour_pairs_within_10kb': observed_pairs,
                'matched_null_median_pairs': float(np.median(null_pairs)),
                'matched_null_95th_pairs': float(np.quantile(null_pairs, .95)),
                'empirical_p_enrichment': (1 + int(np.sum(null_pairs >= observed_pairs))) / (N_NULL + 1),
            })

    res = pd.DataFrame(results)
    res['exploratory_BH_q'] = bh_fdr(res['empirical_p_enrichment'])
    res = res.sort_values(['exploratory_BH_q', 'empirical_p_enrichment', 'component'])
    pairs = pd.DataFrame(genomic)
    pairs['exploratory_BH_q'] = bh_fdr(pairs['empirical_p_enrichment'])
    check(len(res) == 3 * 2 * len(THEMES), 'all_theme_tests')
    check(len(pairs) == 6, 'six_genomic_cluster_tests')

    OUT.mkdir(parents=True, exist_ok=False)
    res.to_csv(OUT / 'EXPLORATORY_SIGNED_KEYWORD_COHERENCE_MATCHED_NULL.csv', index=False)
    pairs.to_csv(OUT / 'EXPLORATORY_SIGNED_GENOMIC_NEIGHBOURHOOD_MATCHED_NULL.csv', index=False)
    pd.DataFrame(top_genes).to_csv(OUT / 'EXPLORATORY_TOP100_EACH_PC_DIRECTION.csv', index=False)
    (OUT / 'PROVENANCE.txt').write_text(
        f'Input file SHA256: {sha256(SOURCE_TABLE)}\n'
        f'Script SHA256: {sha256(Path(__file__).resolve())}\n'
        f'Random seed: {SEED}\nNull replicates: {N_NULL}\n'
    )
    (OUT / 'INTERPRETATION_LIMITS.txt').write_text(
        'EXPLORATORY ONLY: keyword text matches from RefSeq GFF3 are not curated GO/pathways.\n'
        'Numerous gene annotations are hypothetical or missing. Categories overlap.\n'
        'PC signs are arbitrary, and PCs may rotate under training changes.\n'
        'Null is matched on Atlanta expression SD and external Cornforth gene detection.\n'
        'External coverage matching is post hoc; it is not independent biological evidence.\n'
        'Genomic proximity can reflect operons OR genomic islands / strain-mapping artefacts.\n'
        'Do not interpret BH-q as CF specificity or validated biological programme identity.\n'
    )
    status = 'CFSTATE_PHASE6_06G_EXPLORATORY_FUNCTIONAL_COHERENCE_TRIAGE_COMPLETE'
    (OUT / 'AUDIT_06G_STATUS.txt').write_text(status + '\n')
    print('\n========== MOST ENRICHED KEYWORD ANNOTATIONS (EXPLORATORY) ==========')
    print(res.head(18).round(4).to_string(index=False))
    print('\n========== GENOMIC NEIGHBOURHOOD CLUSTERING ==========')
    print(pairs.round(4).to_string(index=False))
    print('\nSTATUS:', status)


if __name__ == '__main__':
    main()
