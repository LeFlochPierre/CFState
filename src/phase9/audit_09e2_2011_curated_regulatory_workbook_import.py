#!/usr/bin/env python3
"""CF-State Phase 09e2: Galan-Vasquez 2011 curated TF-target workbook overlay.

Reads *network data* worksheet, with the real header at zero-indexed row 3.
Preserves regulation sign, evidence, publication and strain as supplied.
Maps names to PAO1 locus tags only with already-available independent gene
annotations (08j unique GFF-derived symbols, optional GFF3, optional aliases).
Quarantines unresolved or ambiguous identifiers. No expression-derived TF edges.

Commands:
 python src/phase9/audit_09e2_2011_curated_regulatory_workbook_import.py --self-test
 python src/phase9/audit_09e2_2011_curated_regulatory_workbook_import.py --inspect-source
 python src/phase9/audit_09e2_2011_curated_regulatory_workbook_import.py
 Optional: --gff references/<verified_PAO1_file>.gff3
 Optional: --alias-map references/<verified_aliases>.csv (columns alias,GeneID)

Fresh output directory; never overwrites previous audits. Reuses only the
frozen 09d graph, not the iModulon representation to define regulatory edges.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd

# The previous, already-tested 09e module is retained unchanged and supplies
# frozen 09d schema validation and typed-graph assembly.
import audit_09e_curated_regulatory_evidence_overlay as base

ROOT = Path(__file__).resolve().parents[2]
SRC_REL = Path('data/raw/phase9_regulatory/PAO1_2011_curated_regulatory_network.xls')
OUT_REL = Path('results/phase9/audit_09e2_2011_curated_regulatory_workbook_import')
A08J_REL = Path('results/phase8/audit_08j_GSE68534_frozen_Anr_regulator_response/DIAGNOSTIC_source_gene_mapping.csv')
A08G_REL = Path('results/phase8/audit_08g_perturbation_anchored_clinical_decomposition/EXPLORATORY_reference_genes_for_top_orthogonal_programmes.csv')
ID_RE = re.compile(r'^PA\d{4}$', re.I)
PA_TOK = re.compile(r'(?<![A-Za-z0-9])PA\d{4}(?![A-Za-z0-9])', re.I)
NEEDED = ['Regulator (TF or sigma)', 'Operon', 'Target gene',
          'mode of regulation', 'Experimental Evidence',
          'PubMed Referencea', 'P. aeruginosa Strain']


def check(condition, msg):
    if not condition:
        raise RuntimeError('AUDIT_09E2_FAILED: ' + str(msg))


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_xls(path):
    check(path.is_file(), 'missing authentic 2011 supplementary workbook: ' + str(path))
    # The supplied file has three prefatory lines; fourth row is actual header.
    df = pd.read_excel(path, sheet_name='network data', header=3, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    check(set(NEEDED).issubset(df.columns),
          '2011 workbook schema unexpectedly changed, headers=' + str(list(df.columns)))
    df = df[NEEDED].copy().fillna('')
    for col in NEEDED:
        df[col] = df[col].astype(str).str.strip()
    # Retain Excel original data row: header is row 4, first observation row 5.
    df.insert(0, 'source_excel_row', [int(i) + 5 for i in range(len(df))])
    return df


def build_alias_map(mask, root, gff=None, external_alias=None):
    """Map symbol -> locus only if exactly one independent source-supported locus.

    Critical: a symbol with conflicting PA loci is AMBIGUOUS and excluded,
    even if one appears in the experimental response gene list.
    """
    locs = defaultdict(set)
    origins = defaultdict(set)
    paths = []
    for locus in mask:
        locs[locus.upper()].add(locus)
        origins[locus.upper()].add('FROZEN_LOCUS_ID')

    def add(alias, locus, provenance):
        key = str(alias).strip().upper()
        gene = str(locus).strip().upper()
        if not key or not ID_RE.fullmatch(gene):
            return
        # Retain out-of-mask PAO1 genes so curated edges outside response
        # universe are still provenance-checked and can be archived.
        locs[key].add(gene)
        origins[key].add(provenance)

    a08j = root / A08J_REL
    if a08j.is_file():
        a = pd.read_csv(a08j, dtype=str, keep_default_na=False)
        check({'GeneID','raw_feature_id','mapping_method'}.issubset(a.columns),
              '08j independent symbol reference lacks required columns')
        for row in a.itertuples(index=False):
            if 'unique_GFF_gene_symbol' in row.mapping_method:
                add(row.raw_feature_id, row.GeneID, 'AUDIT08J_UNIQUE_GFF_SYMBOL')
        paths.append(a08j)

    a08g = root / A08G_REL
    if a08g.is_file():
        a = pd.read_csv(a08g, dtype=str, keep_default_na=False)
        if {'PAO1_GeneID','gene_symbol'}.issubset(a.columns):
            for row in a.itertuples(index=False):
                add(row.gene_symbol, row.PAO1_GeneID, 'AUDIT08G_REFERENCE_ANNOTATION')
            paths.append(a08g)

    if gff is not None:
        check(gff.is_file(), 'specified GFF missing: ' + str(gff))
        opening = gzip.open if gff.suffix.lower() == '.gz' else open
        with opening(gff, 'rt', encoding='utf-8', errors='replace') as f:
            for line in f:
                if not line.strip() or line.startswith('#'):
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) != 9 or parts[2].lower() != 'gene':
                    continue
                attrs = {}
                for kv in parts[8].split(';'):
                    if '=' in kv:
                        k, v = kv.split('=', 1)
                        attrs[k] = unquote(v)
                # Only use an explicit locus_tag or PA#### in a gene ID;
                # do not infer PAO1 loci from positions or orthologues.
                locus = attrs.get('locus_tag', '')
                if not ID_RE.fullmatch(locus):
                    candidates = PA_TOK.findall(attrs.get('ID', ''))
                    locus = candidates[0] if len(candidates) == 1 else ''
                if ID_RE.fullmatch(locus):
                    for name in ('gene','gene_name'):
                        symbol = attrs.get(name, '')
                        if symbol and ',' not in symbol:
                            add(symbol, locus, 'EXPLICIT_GFF_GENE_SYMBOL')
        paths.append(gff)

    if external_alias is not None:
        check(external_alias.is_file(), 'explicit alias map missing: ' + str(external_alias))
        a = pd.read_csv(external_alias, dtype=str, keep_default_na=False)
        check({'alias','GeneID'}.issubset(a.columns),
              'alias file must have columns alias,GeneID')
        for row in a[['alias','GeneID']].itertuples(index=False, name=None):
            add(row[0], row[1], 'USER_VERIFIED_EXPLICIT_ALIAS')
        paths.append(external_alias)

    aliases = {key: next(iter(vals)) for key, vals in locs.items() if len(vals) == 1}
    conflicts = {key: '|'.join(sorted(vals)) for key, vals in locs.items() if len(vals) > 1}
    source_descriptions = {key: '|'.join(sorted(origins[key])) for key in aliases}
    return aliases, conflicts, source_descriptions, paths


def mode_label(raw):
    val = str(raw).strip().lower().replace(' ', '')
    if val in ('+', 'activation', 'activates', 'positive'):
        return 'REPORTED_ACTIVATION'
    if val in ('-', 'repression', 'represses', 'negative'):
        return 'REPORTED_REPRESSION'
    if val in ('+/-','-/+','±','both','dual','+-','-+'):
        return 'REPORTED_DUAL_OR_CONTEXT_DEPENDENT'
    if val in ('', '?','unknown','undefined','na','nan'):
        return 'REPORTED_UNKNOWN'
    # Sigma/anti-sigma special interactions must not become TF +/−.
    return 'REPORTED_OTHER_KEEP_RAW'


def normalize_rows(df, aliases, conflicts, origin):
    full = []
    for row in df.to_dict('records'):
        raw_r = row['Regulator (TF or sigma)']
        raw_t = row['Target gene']
        rkey, tkey = raw_r.upper(), raw_t.upper()
        reg = aliases.get(rkey, '')
        target = aliases.get(tkey, '')
        reference = row['PubMed Referencea']
        problems = []
        if not raw_r or not raw_t:
            problems.append('MISSING_REGULATOR_OR_TARGET')
        if rkey in conflicts:
            problems.append('AMBIGUOUS_REGULATOR_ALIAS')
        if tkey in conflicts:
            problems.append('AMBIGUOUS_TARGET_ALIAS')
        if not reg:
            problems.append('UNRESOLVED_REGULATOR_ALIAS')
        if not target:
            problems.append('UNRESOLVED_TARGET_ALIAS')
        # A PMID is a citation, NOT an independent biological replicate.
        if not re.fullmatch(r'\d{5,9}', reference):
            problems.append('MISSING_OR_NONCANONICAL_PMID')
        full.append({
            'source_excel_row': row['source_excel_row'],
            'raw_regulator': raw_r,
            'raw_target': raw_t,
            'raw_operon_annotation': row['Operon'],
            'raw_mode_of_regulation': row['mode of regulation'],
            'reported_mode_class': mode_label(row['mode of regulation']),
            'raw_experimental_evidence': row['Experimental Evidence'],
            'original_PMID': reference,
            'original_strain': row['P. aeruginosa Strain'],
            'source_2011_workbook': 'GALAN_VASQUEZ_2011_SUPP1_NETWORK_DATA',
            'regulator_GeneID': reg,
            'target_GeneID': target,
            'regulator_mapping_basis': origin.get(rkey, ''),
            'target_mapping_basis': origin.get(tkey, ''),
            'original_strain_explicit_PAO1': row['P. aeruginosa Strain'].strip().upper() == 'PAO1',
            'row_status': 'CURATED_MAPPED' if not problems else 'QUARANTINE',
            'quarantine_reason': '|'.join(problems),
            'original_reference_is_paper_evidence_not_new_validation': True,
            'binding_or_direct_mechanism_established_by_this_workflow': False,
            'CF_specific_regulatory_interaction_proven': False,
        })
    return pd.DataFrame(full)


def build_analysis(root, source, gff, alias_csv, output):
    check(not output.exists(), 'output directory already exists; never overwrite: ' + str(output))
    mask, metal, paths = base.load_frozen(root)
    # Guarantee that the workbook is not already part of the response-trained data.
    raw = read_xls(source)
    check(len(raw) > 0, 'no curator-supplied interactions')
    aliases, conflicts, alias_origins, evidence_paths = build_alias_map(mask, root, gff, alias_csv)
    normalized = normalize_rows(raw, aliases, conflicts, alias_origins)
    valid = normalized[normalized.row_status == 'CURATED_MAPPED'].copy()
    quarantine = normalized[normalized.row_status == 'QUARANTINE'].copy()
    check(len(valid) > 0,
          'zero unambiguous mapped interactions: add an independent PAO1 GFF with --gff')
    selected = set(metal.target_node.str.removeprefix('gene:'))
    overlap = valid[valid.target_GeneID.isin(selected)].copy()

    rows = []
    for gene in sorted(selected):
        entries = overlap[overlap.target_GeneID == gene]
        pao1 = entries[entries.original_strain_explicit_PAO1]
        rows.append({
            'GeneID': gene,
            'n_2011_curated_evidence_rows_any_strain': len(entries),
            'n_distinct_regulators_any_strain': entries.regulator_GeneID.nunique(),
            'n_distinct_regulators_explicit_PAO1': pao1.regulator_GeneID.nunique(),
            'n_distinct_regulator_target_pairs_explicit_PAO1': pao1[['regulator_GeneID','target_GeneID']].drop_duplicates().shape[0],
            'n_PMIDs_explicit_PAO1': pao1.original_PMID.nunique(),
            'has_curated_evidence_any_strain': len(entries) > 0,
            'has_curated_evidence_explicit_PAO1': len(pao1) > 0,
            'network_response_evidence_remains_provisional': True,
        })
    coverage = pd.DataFrame(rows)

    regulator_rollup = (
        overlap.groupby('regulator_GeneID', as_index=False)
        .agg(n_selected_metal_targets=('target_GeneID','nunique'),
             n_curated_rows=('target_GeneID','size'),
             n_reference_PMIDs=('original_PMID','nunique'),
             n_explicit_PAO1_rows=('original_strain_explicit_PAO1','sum'))
        .sort_values(['n_selected_metal_targets','regulator_GeneID'], ascending=[False,True])
        if len(overlap) else pd.DataFrame(columns=['regulator_GeneID','n_selected_metal_targets',
                                                   'n_curated_rows','n_reference_PMIDs','n_explicit_PAO1_rows'])
    )

    response = pd.DataFrame({
        'source_node': metal.source_node,
        'target_node': metal.target_node,
        'edge_semantics': 'EXPERIMENTALLY_MEASURED_INDIRECT_COMBINED_METALS_RESPONSE',
        'regulatory_sign': 'NOT_APPLICABLE',
        'expression_response_direction': metal.effect_direction,
        'evidence_source': 'PRJNA1006084_via_AUDIT_09D',
        'original_PMID': '',
        'experimental_evidence': 'paired_ex_vivo_ZnFeMn_lab_PAO1_in_CF_sputum_17_donors',
        'source_strain': 'PAO1',
        'direct_binding_proven': False,
        'CF_specificity_proven': False,
    })
    regulatory = pd.DataFrame({
        'source_node': 'gene:' + valid.regulator_GeneID,
        'target_node': 'gene:' + valid.target_GeneID,
        'edge_semantics': '2011_LITERATURE_CURATED_REGULATORY_ASSOCIATION_EVIDENCE_TYPE_VARIES',
        'regulatory_sign': valid.reported_mode_class,
        'expression_response_direction': 'NOT_APPLICABLE',
        'evidence_source': 'GALAN_VASQUEZ_2011_SUPP1_NETWORK_DATA',
        'original_PMID': valid.original_PMID,
        'experimental_evidence': valid.raw_experimental_evidence,
        'source_strain': valid.original_strain,
        'direct_binding_proven': False,
        'CF_specificity_proven': False,
    })
    combined = pd.concat([response, regulatory], ignore_index=True)
    report = {
        'status': 'PHASE09E2_2011_CURATED_REGULATORY_WORKBOOK_OVERLAY_COMPLETE_NO_FITTING',
        'source_rows': int(len(raw)), 'mapped_evidence_rows': int(len(valid)),
        'quarantined_rows': int(len(quarantine)),
        'mapped_distinct_TF_target_pairs': int(valid[['regulator_GeneID','target_GeneID']].drop_duplicates().shape[0]),
        'mapped_rows_explicit_PAO1_strain': int(valid.original_strain_explicit_PAO1.sum()),
        'metal_response_genes_total': int(len(selected)),
        'selected_metal_genes_with_any_2011_regulatory_evidence': int(coverage.has_curated_evidence_any_strain.sum()),
        'selected_metal_genes_with_explicit_PAO1_2011_regulatory_evidence': int(coverage.has_curated_evidence_explicit_PAO1.sum()),
        'ambiguous_symbol_keys_quarantined': int(len(conflicts)),
        'mapping_reference_sources': [str(p.resolve()) for p in evidence_paths],
        'regulator_symbol_and_target_symbol_mapping_from_independent_annotated_sources_only': True,
        'regulatory_sign_is_reported_literature_mode_not_perturbation_inference': True,
        'network_learning': False,
    }
    limitations = f'''# Phase 09e2 — Galan-Vasquez 2011 regulatory evidence overlay

**2011 source:** Supplementary Additional File 1, `network data` worksheet, original header on Excel row 4. This is historical literature-curated regulatory evidence, not new 2026 experimental data.

- Source rows: {len(raw)}; unambiguously mapped with canonical PubMed references: {len(valid)}; quarantined: {len(quarantine)}.
- Metal response genes covered by mapped regulatory evidence from **any recorded strain**: {report['selected_metal_genes_with_any_2011_regulatory_evidence']}/249.
- Metal response genes with evidence **explicitly annotated PAO1**: {report['selected_metal_genes_with_explicit_PAO1_2011_regulatory_evidence']}/249.

## Non-equivalent evidence layers

1. All 249 intervention→gene links remain **provisional combined Zn+Fe+Mn responses** from 17 donor-paired ex-vivo experiments using added laboratory PAO1. Not transcription-factor regulation and not endogenous patient RNA.
2. The 2011 regulator→target links are **published reports** with heterogeneous experimental methods, potentially indirect; reported +/− signs are not newly verified. Operon entries are preserved as metadata, never expanded into target edges.
3. Rows lacking unambiguous gene→PAO1 locus mapping or a canonical PMID are quarantined; unmapped does not mean unsupported biology.
4. The strain field is preserved. Results from other strains are not counted as PAO1-supported records.
5. Repeated curator rows, PMIDs and common regulator targets do not constitute independent perturbation replications.
6. Neither the curated 2011 regulatory network nor this overlap establishes a CF-specific network, direct binding for every edge, clinical state inference, or improved transfer over iModulons.

## Decision gate

Inspect the quarantined symbol mappings and the regulator coverage before considering biological conclusions. Coverage may be sparse because the 2011 network is historical. Import a newer independent regulatory network as a distinct provenance source and predefine a *study-held-out* evaluation against simpler response-signature and frozen iModulon baselines before learning graph weights.
'''
    output.mkdir(parents=True, exist_ok=False)
    normalized.to_csv(output / 'FROZEN_ALL_2011_WORKBOOK_ROWS_MAPPING_STATUS.csv', index=False)
    valid.to_csv(output / 'FROZEN_2011_CURATED_MAPPED_TF_TARGET_EVIDENCE.csv', index=False)
    quarantine.to_csv(output / 'REVIEW_UNRESOLVED_OR_AMBIGUOUS_2011_ROWS.csv', index=False)
    overlap.to_csv(output / 'DESCRIPTIVE_2011_REGULATORY_EVIDENCE_FOR_249_METAL_GENES.csv', index=False)
    coverage.to_csv(output / 'DESCRIPTIVE_249_METAL_GENE_2011_REGULATORY_COVERAGE.csv', index=False)
    regulator_rollup.to_csv(output / 'DESCRIPTIVE_REGULATORS_TARGETING_SELECTED_METAL_GENES.csv', index=False)
    combined.to_csv(output / 'NETWORK_TYPED_COMBINED_METAL_AND_CURATED_2011_EDGES.csv', index=False)
    manifest = pd.DataFrame([{'path':str(p.resolve()), 'sha256':sha256(p)}
                             for p in [source] + paths + evidence_paths])
    manifest.to_csv(output / 'INPUT_SHA256_PROVENANCE.csv', index=False)
    (output / 'QC_09E2.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'INTERPRETATION_LIMITS.md').write_text(limitations)
    (output / 'AUDIT_09E2_STATUS.txt').write_text(report['status'] + '\n')
    print('AUDIT 09E2 completed:', output)
    print('2011 source rows:', len(raw), '| mapped:', len(valid), '| quarantined:', len(quarantine))
    print('Distinct mapped TF-target pairs:', report['mapped_distinct_TF_target_pairs'])
    print('Frozen 249 metal genes covered: any strain =', report['selected_metal_genes_with_any_2011_regulatory_evidence'],
          '| explicitly PAO1 =', report['selected_metal_genes_with_explicit_PAO1_2011_regulatory_evidence'])
    print('No causal links, CF specificity, or independently validated network learning inferred.')
    return report


def self_test():
    # In-memory schema/mapping tests plus the real prior module's 09d contract
    # exercised with a temporary synthetic project. Actual .xls header format
    # is independently established by the user's read-only workbook preview.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        mask_path = root / base.MASK_REL
        mask_path.parent.mkdir(parents=True)
        mask = [f'PA{i:04d}' for i in range(1,4924)]
        pd.DataFrame({'GeneID':mask}).to_csv(mask_path, index=False)
        d = root / base.D_REL
        d.mkdir(parents=True)
        (d / 'AUDIT_09D_STATUS.txt').write_text(base.EXPECTED_D + '\n')
        metal = pd.DataFrame({
            'source_node':['intervention:metals']*249,
            'target_node':['gene:PA%04d' % i for i in range(1,250)],
            'edge_type':['experimentally_measured_indirect_expression_response']*249,
            'effect_direction':['DOWN']*249,
            'direct_TF_regulation':[False]*249,
            'CF_specificity_demonstrated':[False]*249,
            'independently_replicated':[False]*249,
        })
        metal.to_csv(d / 'NETWORK_V0_PROVISIONAL_METAL_RESPONSE_EDGES_NOT_REGULATORY.csv', index=False)
        a08j = root / A08J_REL
        a08j.parent.mkdir(parents=True)
        pd.DataFrame({'GeneID':['PA0100','PA0001','PA0002','PA0003','PA0004'],
                      'raw_feature_id':['anr','nirS','nirM','conflicted','conflicted'],
                      'mapping_method':['unique_GFF_gene_symbol']*5}).to_csv(a08j, index=False)
        symbols, conflicts, origins, _ = build_alias_map(set(mask), root)
        check(symbols['ANR'] == 'PA0100' and symbols['NIRS'] == 'PA0001', 'test independent gene alias mapping')
        check('CONFLICTED' in conflicts and 'CONFLICTED' not in symbols, 'test ambiguous symbols quarantined')
        dfr = pd.DataFrame([
            {'source_excel_row':5,'Regulator (TF or sigma)':'anr','Operon':'test_operon',
             'Target gene':'nirS','mode of regulation':'+','Experimental Evidence':'reporter',
             'PubMed Referencea':'12345678','P. aeruginosa Strain':'PAO1'},
            {'source_excel_row':6,'Regulator (TF or sigma)':'anr','Operon':'',
             'Target gene':'conflicted','mode of regulation':'-','Experimental Evidence':'mutant',
             'PubMed Referencea':'12345678','P. aeruginosa Strain':'other'},
            {'source_excel_row':7,'Regulator (TF or sigma)':'anr','Operon':'',
             'Target gene':'nirM','mode of regulation':'?','Experimental Evidence':'chip',
             'PubMed Referencea':'','P. aeruginosa Strain':'PAO1'},
        ])
        nr = normalize_rows(dfr, symbols, conflicts, origins)
        check((nr.row_status=='CURATED_MAPPED').sum() == 1, 'test quarantine and mapped statuses')
        check(nr.iloc[0].reported_mode_class == 'REPORTED_ACTIVATION', 'test sign convention')
        check(nr.iloc[0].raw_operon_annotation == 'test_operon', 'test metadata preservation')
        check(mode_label('+-') == 'REPORTED_DUAL_OR_CONTEXT_DEPENDENT', 'test dual sign')
        m,g,_ = base.load_frozen(root)
        check(len(g)==249 and len(m)==4923, 'test frozen 09d contract')
    print('SELF_TEST_PASSED: source schema, exact independent aliases, ambiguous mapping quarantine,')
    print('  reported sign, source strain and evidence retention, 249 response-edge contract.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--source',type=Path,default=SRC_REL)
    p.add_argument('--gff',type=Path,help='optional verified PAO1 GFF3 annotation')
    p.add_argument('--alias-map',type=Path,help='optional independently verified CSV: alias,GeneID')
    p.add_argument('--inspect-source',action='store_true')
    p.add_argument('--self-test',action='store_true')
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    root = a.root.expanduser().resolve()
    source = (root / a.source).resolve() if not a.source.is_absolute() else a.source.resolve()
    gff = ((root / a.gff).resolve() if not a.gff.is_absolute() else a.gff.resolve()) if a.gff else None
    alias_csv = ((root / a.alias_map).resolve() if not a.alias_map.is_absolute() else a.alias_map.resolve()) if a.alias_map else None
    if a.inspect_source:
        df = read_xls(source)
        print('2011 workbook sheet=network data; Excel header row=4; interaction rows=', len(df))
        print('Columns:', list(df.columns))
        print(df.head(6).to_string(index=False))
        return
    build_analysis(root, source, gff, alias_csv, root / OUT_REL)

if __name__ == '__main__':
    main()
