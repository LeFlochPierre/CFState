#!/usr/bin/env python3
"""CF-State Phase 09a: freeze a study-level evidence ledger and inspect gene-level response inputs.

NO model fitting; no network-edge inference, no combining microarray/RNA-seq
magnitudes, no contamination of existing audit outputs. Source work is read-only.

Run from any directory:
  python src/phase9/audit_09a_perturbation_evidence_preflight.py --self-test
  python src/phase9/audit_09a_perturbation_evidence_preflight.py

Outputs under results/phase9/audit_09a_perturbation_evidence_preflight/.
Requires only pandas and Python standard library.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_REL = Path('results/phase9/audit_09a_perturbation_evidence_preflight')
MASK = Path('results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv')
ARRAY = Path('results/phase8/audit_08m_GPL84_oxygen_CFBE_Anr_gene_response/DESCRIPTIVE_all_five_array_gene_contrasts.csv')
STUDIES = [
    ('PRJNA1006084', 'combined_Zn_Fe_Mn', 'PAO1 added to sputum from 17 CF donors', 'RNA-seq', '17 paired CF sputum donors', 'CF-context response, not native patient bacterial RNA', 'audit_08e2*', 'BUILD_CANDIDATE', 'One combined intervention; cannot resolve individual Zn, Fe, Mn responses'),
    ('GSE6741', 'oxygen_gradient', 'PAO1 O2: 20%, 2%, 0.4%, anoxia plus nitrate', 'GPL84 microarray', '2 biological replicates per condition', 'laboratory response', 'audit_08m*', 'BUILD_CANDIDATE', 'Anoxia includes 100 mM nitrate; not a pure O2 effect'),
    ('GSE67006', 'delta_anr_CFBE', 'PAO1 WT vs delta-anr grown on CF bronchial epithelial cells', 'GPL84 microarray', '3 biological replicates per condition', 'laboratory regulatory perturbation', 'audit_08m*', 'RESERVE_FOR_PHASE09_VALIDATION', 'Knockout response can be indirect; reserve before selecting edges'),
    ('GSE68534', 'delta_anr_biofilm', 'PAO1 and J215 WT vs delta-anr in low-O2 biofilm', 'RNA-seq', '2 biological replicates per group and strain', 'cross-background regulatory perturbation', 'audit_08j*', 'RESERVE_FOR_PHASE09_VALIDATION', 'Phase 08 previously examined this study; not fully blind hypothesis validation'),
    ('GSE280261', 'tobramycin_glpR', 'PAO1 and FRD1 WT/delta-glpR x tobramycin', 'RNA-seq', '3 biological replicates per cell', 'antibiotic response', 'audit_08b*', 'CANDIDATE_LATER', 'Strain/growth timing/regulator interactions need explicit factorial contrasts'),
    ('GSE310904', 'delta_cntL', 'PA14 delta-cntL vs WT pseudopaline', 'RNA-seq', '3 biological replicates per group', 'zinc acquisition mechanism', 'audit_08d2*', 'EXPLORATORY_CHALLENGE', 'Different zinc intervention from combined metal addition'),
    ('GSE60177', 'delta_znuA', 'PAO1 delta-znuA vs WT', 'RNA-seq', '1 pooled library per condition', 'zinc acquisition mechanism', 'audit_08d2*', 'LOW_CONFIDENCE_SENSITIVITY', 'No independently estimable biological variance; 3 ambiguous rows excluded previously'),
]
SOURCE_COLUMNS = [
    'dataset', 'contrast', 'GeneID', 'gene_signal_change', 'n_treatment', 'n_reference'
]
NATIVE = {
    'O2_2_minus_20': ('GSE6741', 2, 2, '2% O2 minus 20% O2', 'OXYGEN_ONLY'),
    'O2_0p4_minus_20': ('GSE6741', 2, 2, '0.4% O2 minus 20% O2', 'OXYGEN_ONLY'),
    'anoxic_nitrate_minus_20': ('GSE6741', 2, 2, '0% O2 + nitrate minus 20% O2', 'OXYGEN_PLUS_NITRATE'),
    'delta_anr_minus_WT_CFBE': ('GSE67006', 3, 3, 'delta-anr minus WT in CFBE coculture', 'REGULATOR_KNOCKOUT'),
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError('AUDIT_09A_FAILED: ' + message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def frozen_array_effects(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask_file, array_file = root / MASK, root / ARRAY
    require(mask_file.is_file(), 'missing frozen 4,923-gene mask: ' + str(mask_file))
    require(array_file.is_file(), 'missing frozen 08m within-study array effects: ' + str(array_file))
    mask = pd.read_csv(mask_file, dtype=str)
    require('GeneID' in mask.columns and len(mask) == 4923 and mask.GeneID.is_unique,
            'expected 4,923 distinct frozen GeneID identifiers')
    raw = pd.read_csv(array_file, dtype={'GeneID': str})
    require(set(SOURCE_COLUMNS).issubset(raw.columns), 'unexpected frozen 08m effect schema')
    require(not raw.duplicated(['dataset', 'contrast', 'GeneID']).any(), 'duplicated 08m gene effect rows')
    require(raw.GeneID.isin(mask.GeneID).all(), '08m contains gene IDs outside frozen mask')
    for field in ['gene_signal_change', 'n_treatment', 'n_reference']:
        raw[field] = pd.to_numeric(raw[field], errors='raise')
    require(raw.gene_signal_change.notna().all() and raw.gene_signal_change.map(lambda x: abs(x) < float('inf')).all(),
            'nonfinite 08m effect')
    rows, qc = [], []
    for contrast, (ds, nt, nr, meaning, family) in NATIVE.items():
        selected = raw.loc[(raw.dataset == ds) & (raw.contrast == contrast)].copy()
        require(len(selected) == 4920 and selected.GeneID.is_unique,
                '08m expected 4,920 mapped genes for ' + contrast)
        require((selected.n_treatment == nt).all() and (selected.n_reference == nr).all(),
                '08m replicate counts changed for ' + contrast)
        selected['dataset_id'] = ds
        selected['contrast_id'] = contrast
        selected['condition_minus_control'] = meaning
        selected['perturbation_family'] = family
        selected['effect_value_native'] = selected['gene_signal_change']
        selected['effect_unit'] = 'within-study transformed GPL84 array signal difference; NOT RNA-seq log-fold change'
        selected['effect_direction'] = selected['effect_value_native'].map(lambda x: 'UP' if x > 0 else ('DOWN' if x < 0 else 'ZERO'))
        selected['n_biological_treatment'] = nt
        selected['n_biological_reference'] = nr
        selected['evidence_type'] = 'measured_expression_response_NOT_direct_TF_binding'
        selected['direct_regulatory_edge_proven'] = False
        selected['phase09_status'] = 'EXPERIMENTAL_RESPONSE_EVIDENCE_ONLY'
        rows.append(selected[['dataset_id', 'contrast_id', 'GeneID', 'condition_minus_control',
                              'perturbation_family', 'effect_value_native', 'effect_unit',
                              'effect_direction', 'n_biological_treatment', 'n_biological_reference',
                              'evidence_type', 'direct_regulatory_edge_proven', 'phase09_status']])
        qc.append({'dataset_id': ds, 'contrast_id': contrast, 'n_unique_mask_genes': len(selected),
                   'replicates_treatment': nt, 'replicates_reference': nr,
                   'median_native_effect': float(selected.effect_value_native.median()),
                   'fraction_effect_positive': float((selected.effect_value_native > 0).mean()),
                   'use_for': 'native sign/within-study response evidence; do NOT pool magnitudes'})
    # Important: 08m includes an additional exploratory contrast. Inventory, do not silently ingest.
    ignored = raw[['dataset', 'contrast']].drop_duplicates()
    ignored = ignored.loc[~ignored.apply(lambda r: (r['contrast'] in NATIVE and NATIVE[r['contrast']][0] == r['dataset']), axis=1)].copy()
    if len(ignored):
        for r in ignored.itertuples(index=False):
            qc.append({'dataset_id': r.dataset, 'contrast_id': r.contrast,
                       'n_unique_mask_genes': '', 'replicates_treatment': '',
                       'replicates_reference': '', 'median_native_effect': '',
                       'fraction_effect_positive': '',
                       'use_for': 'UNREVIEWED_08M_CONTRAST_EXCLUDED_FROM_THIS_FREEZE'})
    return pd.concat(rows, ignore_index=True), pd.DataFrame(qc)


def file_candidates(root: Path) -> pd.DataFrame:
    p8 = root / 'results/phase8'
    out = []
    for ds, family, desc, assay, replicates, source_type, dir_pattern, role, limitations in STUDIES:
        dirs = sorted(p8.glob(dir_pattern)) if p8.exists() else []
        dirs = [d for d in dirs if d.is_dir()]
        for directory in dirs:
            # Capture lightweight CSV/TSV file schema, not the source data itself.
            files = sorted(f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in {'.csv', '.tsv'})
            for f in files:
                if f.stat().st_size > 50_000_000:
                    continue
                try:
                    delim = '\t' if f.suffix.lower() == '.tsv' else ','
                    cols = list(pd.read_csv(f, sep=delim, nrows=0).columns)
                except Exception:
                    cols = ['<ERROR_READING_HEADER>']
                hinted = any('gene' in c.lower() or 'locus' in c.lower() for c in cols)
                out.append({'study_id': ds, 'source_family': family, 'audit_directory': str(directory.relative_to(root)),
                            'filename': f.name, 'size_bytes': f.stat().st_size,
                            'n_columns': len(cols), 'columns_preview': ' | '.join(cols[:16]),
                            'gene_level_possible': hinted,
                            'inspection_priority': 'REVIEW' if hinted and family == 'combined_Zn_Fe_Mn' else 'SECONDARY',
                            'warning': 'Headers only: gene-level usability and pairing NOT VERIFIED'})
        if not dirs:
            out.append({'study_id': ds, 'source_family': family, 'audit_directory': '<NO_MATCH>',
                        'filename': '', 'size_bytes': '', 'n_columns': 0, 'columns_preview': '',
                        'gene_level_possible': False, 'inspection_priority': 'REVIEW',
                        'warning': 'Audit directory missing or named differently; do not invent a file path'})
    return pd.DataFrame(out)


def plan_text(found: pd.DataFrame) -> str:
    rows = found.loc[(found.study_id == 'PRJNA1006084') & (found.gene_level_possible == True)]
    count = len(rows)
    return f'''# Phase 09a — Evidence freeze and first network design

**Scope:** read-only Phase 08 output inventory and structured native microarray gene responses. This is a preflight, **NOT a trained network**.

## Proposed scientific contribution
A perturbation-anchored, CF-*relevant* bacterial physiological-response representation. CF-*specific* regulatory links require independent CF versus matched non-CF airway validation and are NOT demonstrated by the existing wound/burn comparison.

## Network layers
1. Intervention-to-gene response edges: measured, signed, within-study effects with dose/time/strain and uncertainty metadata. An effect edge means *responds under an intervention*, not direct causal molecular interaction.
2. Regulator-to-gene prior edges: curated signed activation/repression where supported, plus binding-only evidence separately. Do NOT infer direct binding from delta-anr gene expression.
3. Clinical expression and readout: infer strength/direction of experimentally defined response signatures only after appropriate study-specific reference calibration. Clinical validation data MUST NOT be used to choose the network or tune it in advance.

## What is already frozen here
- Four predefined GPL84 microarray contrasts from Audit 08m, each with exactly 4,920 PAO1-aligned genes and within-study signs. Raw effect magnitudes remain on the transformed array scale.
- Separate candidate datasets catalogued for Zn+Fe+Mn sputum, delta-anr biofilm, tobramycin, delta-cntL and delta-znuA. Their gene-level measurement schemas require review before combining.
- Potential gene-level CSV tables found in the 08e2 directory by header scan: **{count}**. This count is a discovery aid, not evidence that any table contains valid 17-donor paired gene effects.

## Critical split and leakage commitments
- Record data overlap with aeruPRECISE training, previous phases, authors' reused samples and donor/specimen identity.
- Assign distinct roles to (a) build/evidence calibration, (b) tuning, and (c) evaluation *before* selecting network edges.
- The 17-donor combined-metal study can support paired, donor-held-out tests; it is NOT an entirely new external validation source after repeated prior use. A held-out study or orthogonal intervention is additionally needed.
- GSE67006 and GSE68534 were already analysed in Phase 08; reserving them for Phase 09 method fitting is possible, but they are NOT prospectively blind evidence of new biology.
- Cornforth CF vs wound/burn may be used for pre-specified exploratory transfer assessment. Because controls differ in anatomical site, it is not a clean CF-specificity test.
- Cross-platform microarray and RNA-seq native magnitudes must NOT be stacked into one matrix without explicit platform-aware normalisation/calibration validated on training data. Start with per-study signs, ranks, and uncertainty, *not* a naive joint PCA/graph NN.

## 09a completion gate before 09b
1. Locate and **verify** the 17-donor paired gene-level files for PRJNA1006084 and the donor metadata. Derive metal effects per donor as (metals minus untreated) if not already frozen; never reconstruct missing donor mappings by guessing.
2. Record exact orientation, technology, independent replicate unit and effect uncertainty for each candidate intervention.
3. Obtain a curated regulatory-prior edge list with per-edge evidence categories; separately freeze that source's version and identifier reconciliation.
4. State one primary held-out prediction task, at least one simple baseline (raw signed gene signature / curated regulon) and the frozen 108-iModulon baseline.
5. If donor/strain independence or appropriate input-level effects cannot be established, STOP. Do not fit a purported CF-specific network.

## Interpretation
This evidence freeze does NOT identify a CF-specific regulatory network, predict oxygen concentrations, separate zinc from iron/manganese, or confer independent validation to studies inspected during Phase 08.
'''


def run(root: Path, out: Path) -> None:
    require((root / 'results/phase8').is_dir(), 'project root does not contain results/phase8')
    require(not out.exists(), 'non-overwrite rule: already exists: ' + str(out))
    edges, effect_qc = frozen_array_effects(root)
    candidates = file_candidates(root)
    study = pd.DataFrame([{'study_id': a, 'perturbation_family': b, 'description': c,
                           'measurement': d, 'biological_units': e, 'context': f,
                           'phase8_directory_pattern': g, 'provisional_phase09_role': h,
                           'limitations': i} for a,b,c,d,e,f,g,h,i in STUDIES])
    provenance = [{'input': str(MASK), 'sha256': sha256(root / MASK)},
                  {'input': str(ARRAY), 'sha256': sha256(root / ARRAY)}]
    out.mkdir(parents=True, exist_ok=False)
    edges.to_csv(out / 'FROZEN_08m_native_array_gene_response_evidence.csv', index=False)
    effect_qc.to_csv(out / 'QC_08m_contrasts_and_exclusions.csv', index=False)
    study.to_csv(out / 'PROVISIONAL_perturbation_study_registry.csv', index=False)
    candidates.to_csv(out / 'REVIEW_existing_phase8_gene_effect_file_candidates.csv', index=False)
    pd.DataFrame(provenance).to_csv(out / 'INPUT_SHA256_provenance.csv', index=False)
    (out / 'PHASE09a_NETWORK_DESIGN_AND_GATES.md').write_text(plan_text(candidates), encoding='utf-8')
    (out / 'AUDIT_09A_STATUS.txt').write_text('PHASE09A_EVIDENCE_PREFLIGHT_COMPLETE_NOT_MODEL_FITTED\n', encoding='utf-8')
    print('AUDIT 09A completed:', out)
    print('Frozen native-array gene-response rows:', len(edges))
    print('Native 08m contrasts:', edges.contrast_id.nunique())
    print('Candidate source files catalogued:', len(candidates))
    print('Potential metal-study gene-level CSVs:', len(candidates.loc[(candidates.study_id == 'PRJNA1006084') & (candidates.gene_level_possible == True)]))
    print('NEXT: inspect candidate metal gene-response files; no network was fitted.')


def self_test() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mask_path = root / MASK
        array_path = root / ARRAY
        mask_path.parent.mkdir(parents=True)
        array_path.parent.mkdir(parents=True)
        genes = [f'PA{i:04d}' for i in range(1, 4924)]
        pd.DataFrame({'GeneID': genes}).to_csv(mask_path, index=False)
        records = []
        for key, (ds, nt, nr, _, _) in NATIVE.items():
            for i, gene in enumerate(genes[:4920]):
                records.append((ds, key, gene, (i % 13 - 6) / 10, nt, nr))
        pd.DataFrame(records, columns=SOURCE_COLUMNS).to_csv(array_path, index=False)
        (root / 'results/phase8').mkdir(parents=True, exist_ok=True)
        out = root / OUT_REL
        run(root, out)
        assert len(pd.read_csv(out / 'FROZEN_08m_native_array_gene_response_evidence.csv')) == 19680
        try:
            run(root, out)
        except RuntimeError as error:
            assert 'non-overwrite' in str(error)
        else:
            raise AssertionError('No-overwrite must reject an existing audit')
    print('SELF_TEST_PASSED: synthetic effect columns, 4 x 4,920 genes, SHA, and non-overwrite')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        root = args.root.expanduser().resolve()
        run(root, root / OUT_REL)


if __name__ == '__main__':
    main()
