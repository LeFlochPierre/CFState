#!/usr/bin/env python3
"""CF-State Phase 09e: strictly provenance-typed external TF-target evidence overlay.

This audit imports an EXTERNAL curator-supplied regulatory/binding table and
intersects its PAO1 targets with the frozen 249 response edges from Audit 09d.
No regulatory links are inferred from expression or genomic proximity.

First acquire an authentic downloaded regulator-target table (e.g. RegulomePA).
Inspect it before choosing exact header columns:
    python src/phase9/audit_09e_curated_regulatory_evidence_overlay.py \
      --source data/raw/phase9_regulatory/RegulomePA_interactions.txt --inspect-source
Then run, specifying the actual headers if automatic header detection fails:
    python src/phase9/audit_09e_curated_regulatory_evidence_overlay.py \
      --source data/raw/phase9_regulatory/RegulomePA_interactions.txt \
      --source-name RegulomePA --source-kind curated_regulation \
      --regulator-col <real_column> --target-col <real_column> \
      --reference-col <real_column> --relation-col <real_column>
If source identifiers are symbols rather than PA####, provide an independently
verified 2-column CSV via --gene-map (alias,GeneID). Ambiguous mappings are
excluded. Binding-only datasets must use --source-kind binding_only.

No network training. No CF specificity, direct causation, or regulatory sign
can be inferred from perturbation responses. Existing outputs never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MASK_REL = Path('results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv')
D_REL = Path('results/phase9/audit_09d_CFstate_evidence_network_v0')
OUT_REL = Path('results/phase9/audit_09e_curated_regulatory_evidence_overlay')
EXPECTED_D = 'PHASE09D_PROVENANCE_TYPED_METAL_RESPONSE_GRAPH_V0_NO_TF_NETWORK'
ID_RE = re.compile(r'^PA\d{4}$', re.I)

COL_SYNONYMS = {
    'regulator': ['regulator', 'regulator_gene', 'regulator_id', 'tf', 'tf_gene', 'tf_id', 'source_gene', 'source'],
    'target': ['target', 'target_gene', 'target_id', 'regulated_gene', 'gene_target', 'regulated', 'target_gene_id'],
    'relation': ['regulation', 'regulation_type', 'type_of_regulation', 'interaction_type', 'effect', 'sign', 'relation', 'type'],
    'reference': ['reference', 'references', 'pubmed', 'pmid', 'publication', 'citation', 'source_reference'],
    'evidence': ['evidence', 'experimental_evidence', 'evidence_type', 'method'],
}


def need(condition, message):
    if not condition:
        raise RuntimeError('AUDIT_09E_FAILED: ' + str(message))


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for part in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read_source(source: Path, sheet=0):
    need(source.is_file(), 'missing external source: ' + str(source))
    suf = source.suffix.lower()
    if suf in ('.xlsx', '.xls'):
        df = pd.read_excel(source, sheet_name=sheet, dtype=str).fillna('')
    elif suf in ('.csv', '.tsv', '.tab', '.txt'):
        # Whitespace-delimited exports should be converted to TSV explicitly;
        # whitespace inside gene names or citation fields is meaningful.
        df = pd.read_csv(source, sep=None, engine='python', dtype=str,
                         keep_default_na=False, on_bad_lines='error')
    else:
        raise RuntimeError('AUDIT_09E_FAILED: unsupported source extension ' + suf)
    df.columns = [str(c).strip() for c in df.columns]
    need(df.columns.is_unique and len(df) > 0, 'empty table / duplicate columns')
    return df.fillna('')


def find_col(df, kind, specified=None, mandatory=True):
    if specified:
        need(specified in df.columns, 'requested column absent for ' + kind + ': ' + specified)
        return specified
    norm = {c: re.sub(r'[^a-z0-9]+', '_', c.lower()).strip('_') for c in df.columns}
    found = [c for c, n in norm.items() if n in COL_SYNONYMS[kind]]
    if len(found) == 1:
        return found[0]
    if not mandatory and not found:
        return None
    raise RuntimeError('AUDIT_09E_FAILED: cannot uniquely detect ' + kind +
                       ' column; candidates=' + repr(found) +
                       ' headers=' + repr(list(df.columns)) +
                       '; supply --' + kind + '-col')


def load_aliases(gene_map, mask):
    # Exact PA identifiers always map without a symbol dictionary.
    alias = {g.upper(): g for g in mask}
    conflicts = set()
    if gene_map is None:
        return alias, conflicts
    need(gene_map.is_file(), 'missing alias mapping file ' + str(gene_map))
    df = pd.read_csv(gene_map, dtype=str, keep_default_na=False)
    need({'alias', 'GeneID'}.issubset(df.columns),
         '--gene-map must contain literal columns alias,GeneID')
    for row in df.itertuples(index=False):
        name = str(getattr(row, 'alias')).strip().casefold()
        locus = str(getattr(row, 'GeneID')).strip().upper()
        if not name or not ID_RE.fullmatch(locus):
            continue
        key = name.upper()
        if key in alias and alias[key] != locus:
            conflicts.add(key)
        else:
            alias[key] = locus
    for key in conflicts:
        alias.pop(key, None)
    return alias, conflicts


def normalise_relation(raw, kind):
    if kind == 'binding_only':
        return 'UNKNOWN_BINDING_ONLY'
    val = re.sub(r'\s+', ' ', str(raw).strip().lower())
    if val in ('activation', 'activator', 'positive', '+', 'positive regulation', 'activates', 'upregulation'):
        return 'ACTIVATION'
    if val in ('repression', 'repressor', 'negative', '-', 'negative regulation', 'represses', 'downregulation'):
        return 'REPRESSION'
    return 'UNKNOWN_OR_COMPLEX'


def load_frozen(root):
    mask_path = root / MASK_REL
    d = root / D_REL
    status = d / 'AUDIT_09D_STATUS.txt'
    graph_file = d / 'NETWORK_V0_PROVISIONAL_METAL_RESPONSE_EDGES_NOT_REGULATORY.csv'
    for p in (mask_path, status, graph_file):
        need(p.is_file(), 'required 09d input missing: ' + str(p))
    need(status.read_text().strip() == EXPECTED_D, 'Audit 09d status differs from frozen schema')
    mask_df = pd.read_csv(mask_path, dtype=str)
    need('GeneID' in mask_df.columns and len(mask_df) == 4923 and mask_df.GeneID.is_unique,
         'frozen gene mask invalid')
    mask = set(mask_df.GeneID)
    graph = pd.read_csv(graph_file, dtype=str, keep_default_na=False)
    required = {'source_node', 'target_node', 'edge_type', 'effect_direction',
                'direct_TF_regulation', 'CF_specificity_demonstrated', 'independently_replicated'}
    need(required.issubset(graph.columns) and len(graph) == 249,
         'Audit 09d graph unexpectedly changed (expected 249 original response edges)')
    need(graph.target_node.is_unique and graph.target_node.str.startswith('gene:PA').all(),
         'response target loci invalid / duplicate')
    need(all(x.removeprefix('gene:') in mask for x in graph.target_node),
         'response gene missing from frozen mask')
    for flag in ('direct_TF_regulation', 'CF_specificity_demonstrated', 'independently_replicated'):
        need((graph[flag].str.lower() == 'false').all(), '09d flag unexpectedly true: ' + flag)
    need((graph.edge_type == 'experimentally_measured_indirect_expression_response').all(),
         '09d response edge semantics unexpectedly changed')
    return mask, graph, [mask_path, status, graph_file]


def normalise_table(df, args, aliases, conflicts, mask):
    regulator_col = find_col(df, 'regulator', args.regulator_col)
    target_col = find_col(df, 'target', args.target_col)
    relation_col = find_col(df, 'relation', args.relation_col, mandatory=False)
    ref_col = find_col(df, 'reference', args.reference_col,
                       mandatory=(args.source_kind == 'curated_regulation'))
    evidence_col = find_col(df, 'evidence', args.evidence_col, mandatory=False)
    good, bad = [], []
    for idx, row in enumerate(df.to_dict('records'), start=2):
        raw_reg, raw_target = str(row[regulator_col]).strip(), str(row[target_col]).strip()
        reg_key, target_key = raw_reg.upper(), raw_target.upper()
        regulator = aliases.get(reg_key, '')
        target = aliases.get(target_key, '')
        problems = []
        if reg_key in conflicts or target_key in conflicts:
            problems.append('AMBIGUOUS_ALIAS')
        if not regulator:
            problems.append('UNRESOLVED_REGULATOR_ID')
        if not target:
            problems.append('UNRESOLVED_TARGET_ID')
        reference = str(row[ref_col]).strip() if ref_col else ''
        if args.source_kind == 'curated_regulation' and not reference:
            problems.append('MISSING_REFERENCE')
        raw_sign = str(row[relation_col]).strip() if relation_col else ''
        item = {
            'external_source_name': args.source_name,
            'external_source_kind': args.source_kind,
            'source_row_number': idx,
            'raw_regulator': raw_reg,
            'raw_target': raw_target,
            'regulator_GeneID': regulator,
            'target_GeneID': target,
            'raw_regulatory_relation': raw_sign,
            'regulatory_sign_evidence': normalise_relation(raw_sign, args.source_kind),
            'raw_evidence_type': str(row[evidence_col]).strip() if evidence_col else '',
            'original_reference': reference,
            'target_in_frozen_4923_mask': bool(target in mask),
            'regulator_in_frozen_4923_mask': bool(regulator in mask),
            'evidence_semantics': ('LITERATURE_CURATED_REPORTED_REGULATORY_INTERACTION' if
                                    args.source_kind == 'curated_regulation' else 'BINDING_ONLY_NO_FUNCTIONAL_SIGN'),
        }
        if problems:
            bad.append({**item, 'quarantine_reasons': '|'.join(problems)})
        else:
            good.append(item)
    return pd.DataFrame(good), pd.DataFrame(bad), {
        'regulator_col': regulator_col, 'target_col': target_col,
        'relation_col': relation_col, 'reference_col': ref_col,
        'evidence_col': evidence_col,
    }


def combine_layers(valid, graph):
    # The response and regulatory layers are combined only in a typed ledger.
    old = pd.DataFrame({
        'source_node': graph.source_node,
        'target_node': graph.target_node,
        'edge_type': 'EXPERIMENTALLY_OBSERVED_PERTURBATION_GENE_RESPONSE',
        'regulatory_sign': 'NOT_APPLICABLE',
        'expression_response_direction': graph.effect_direction,
        'source_name': 'PRJNA1006084_via_frozen_Audit09d',
        'external_reference': '',
        'measurement_or_evidence': 'paired_ex_vivo_combined_ZnFeMn_17_donors',
        'CF_specificity_proven': False,
    })
    reg = pd.DataFrame({
        'source_node': 'gene:' + valid.regulator_GeneID,
        'target_node': 'gene:' + valid.target_GeneID,
        'edge_type': np.where(valid.external_source_kind == 'binding_only',
                              'EXTERNAL_TF_DNA_BINDING_ONLY', 'EXTERNAL_CURATED_REGULATION'),
        'regulatory_sign': valid.regulatory_sign_evidence,
        'expression_response_direction': 'NOT_APPLICABLE',
        'source_name': valid.external_source_name,
        'external_reference': valid.original_reference,
        'measurement_or_evidence': valid.raw_evidence_type,
        'CF_specificity_proven': False,
    })
    return pd.concat([old, reg], ignore_index=True)


def run(root, source, args):
    out = root / OUT_REL
    need(not out.exists(), 'output exists: ' + str(out) + ' (will not overwrite)')
    mask, graph, frozen_paths = load_frozen(root)
    original = read_source(source, args.sheet)
    aliases, conflicts = load_aliases(args.gene_map, mask)
    valid, quarantine, mappings = normalise_table(original, args, aliases, conflicts, mask)
    need(len(valid) > 0, 'no valid curated edges with resolvable PAO1 IDs and references; '
         'inspect source columns and provide an independently verified --gene-map if necessary')
    need((valid.regulator_GeneID.map(lambda x: bool(ID_RE.fullmatch(x)))).all() and
         (valid.target_GeneID.map(lambda x: bool(ID_RE.fullmatch(x)))).all(),
         'unexpected locus mapping')
    selected_genes = set(graph.target_node.str.removeprefix('gene:'))
    subset = valid[valid.target_GeneID.isin(selected_genes)].copy()
    cov = []
    for g in graph.target_node.str.removeprefix('gene:'):
        sub = subset[subset.target_GeneID == g]
        cov.append({
            'GeneID': g,
            'n_curated_evidence_rows': len(sub),
            'n_unique_regulators': sub.regulator_GeneID.nunique(),
            'n_reported_activation_rows': int((sub.regulatory_sign_evidence == 'ACTIVATION').sum()),
            'n_reported_repression_rows': int((sub.regulatory_sign_evidence == 'REPRESSION').sum()),
            'n_unspecified_or_binding_rows': int(sub.regulatory_sign_evidence.isin(
                ('UNKNOWN_BINDING_ONLY','UNKNOWN_OR_COMPLEX')).sum()),
            'has_curated_TF_target_evidence': bool(len(sub)),
            'still_provisional_metal_response_only': True,
            'CF_specific_edge': False,
        })
    coverage = pd.DataFrame(cov)
    ledger = combine_layers(valid, graph)
    paths = frozen_paths + [source]
    if args.gene_map is not None:
        paths.append(args.gene_map)
    report = {
        'status': 'PHASE09E_CURATED_REGULATORY_EVIDENCE_OVERLAY_COMPLETE_NO_NETWORK_FITTING',
        'source': str(source.resolve()),
        'source_kind': args.source_kind,
        'source_name': args.source_name,
        'source_rows': len(original),
        'curated_rows_with_exact_or_UNAMBIGUOUS_verified_alias_ids': len(valid),
        'quarantined_rows': len(quarantine),
        'distinct_regulator_target_pairs': int(valid[['regulator_GeneID','target_GeneID']].drop_duplicates().shape[0]),
        'n_249_metal_genes_with_curated_regulatory_rows': int(coverage.has_curated_TF_target_evidence.sum()),
        'n_249_metal_genes_without_curated_regulatory_rows': int((~coverage.has_curated_TF_target_evidence).sum()),
        'reported_regulation_ACTIVATION_rows': int((valid.regulatory_sign_evidence == 'ACTIVATION').sum()),
        'reported_regulation_REPRESSION_rows': int((valid.regulatory_sign_evidence == 'REPRESSION').sum()),
        'mapping_columns': mappings,
        'unambiguous_aliases_available': int(len(aliases)),
        'conflicting_aliases_excluded': int(len(conflicts)),
        'non_overwriting': True,
        'network_training': False,
        'edge_source_types_kept_distinct': True,
    }
    explanation = '''# Audit 09e: Externally curated TF-target evidence overlay\n\n'''
    explanation += 'This audit preserves 249 frozen metal-response edges from 09d and overlays external regulator-target evidence in a **separate typed ledger**.\n\n'
    explanation += f"- External source: **{args.source_name}** ({args.source_kind}).\n"
    explanation += f"- Valid external source rows: **{len(valid)}**, quarantined rows: **{len(quarantine)}**.\n"
    explanation += f"- Metal-response genes with at least one external TF-target row: **{int(coverage.has_curated_TF_target_evidence.sum())}/249**.\n\n"
    explanation += ('**Not established:** CF-specific regulation, context-specific binding, direction of any indirect Anr-mediated effects, '
                    'independence of articles underlying curated interactions, clinical outcome prediction, or superiority to iModulons.\n\n'
                    'Original reference strings and regulatory-sign labels are retained; an unknown regulatory sign is NOT filled from transcriptional responses. '
                    'Duplicated literature reports are rows of evidence, not independent TF-target interactions. '
                    'Loci outside the frozen mask are retained in curated source rows but are not part of the 249-gene overlap.\n\n'
                    '## Next decision gate\n\n'
                    'Check known regulator coverage and evidence conflicts, then design a priori comparative prediction tasks (e.g., study-held-out perturbation sign recovery) against raw signatures and frozen iModulons. '
                    'Do not fit graph weights without an independent validation design.\n')
    out.mkdir(parents=True, exist_ok=False)
    valid.to_csv(out / 'FROZEN_EXTERNAL_CURATED_TF_TARGET_EVIDENCE.csv', index=False)
    quarantine.to_csv(out / 'REVIEW_UNMAPPED_AMBIGUOUS_OR_UNREFERENCED_ROWS.csv', index=False)
    subset.to_csv(out / 'DESCRIPTIVE_EXTERNAL_TF_TARGET_ROWS_FOR_249_METAL_GENES.csv', index=False)
    coverage.to_csv(out / 'DESCRIPTIVE_249_METAL_GENE_REGULATORY_EVIDENCE_COVERAGE.csv', index=False)
    ledger.to_csv(out / 'NETWORK_V0_TYPED_RESPONSE_AND_EXTERNAL_REGULATORY_EVIDENCE.csv', index=False)
    pd.DataFrame([{'role': 'frozen_or_external_input', 'path': str(p.resolve()), 'sha256': digest(p)}
                  for p in paths]).to_csv(out / 'INPUT_SHA256_PROVENANCE.csv', index=False)
    (out / 'QC_09E.json').write_text(json.dumps(report, indent=2) + '\n')
    (out / 'NETWORK_INTERPRETATION_AND_NEXT_GATES.md').write_text(explanation)
    (out / 'AUDIT_09E_STATUS.txt').write_text(report['status'] + '\n')
    print('AUDIT 09E completed:', out)
    print('Curated external evidence rows:', len(valid), '| quarantined:', len(quarantine))
    print('Frozen metal gene targets with regulator evidence:',
          int(coverage.has_curated_TF_target_evidence.sum()), '/249')
    print('No CF specificity, causality from perturbation, or network learning inferred.')
    return report


def self_test():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        mask_file = root / MASK_REL
        mask_file.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'GeneID': [f'PA{x:04d}' for x in range(1, 4924)]}).to_csv(mask_file, index=False)
        d = root / D_REL
        d.mkdir(parents=True, exist_ok=True)
        (d / 'AUDIT_09D_STATUS.txt').write_text(EXPECTED_D + '\n')
        g = pd.DataFrame({
            'source_node': ['intervention:metals'] * 249,
            'target_node': ['gene:PA%04d' % x for x in range(1, 250)],
            'edge_type': ['experimentally_measured_indirect_expression_response']*249,
            'effect_direction': ['UP']*249,
            'direct_TF_regulation': [False]*249,
            'CF_specificity_demonstrated': [False]*249,
            'independently_replicated': [False]*249,
        })
        g.to_csv(d / 'NETWORK_V0_PROVISIONAL_METAL_RESPONSE_EDGES_NOT_REGULATORY.csv', index=False)
        source = root / 'external.tsv'
        pd.DataFrame({
            'Regulator': ['PA1544', 'zur', 'PA9999', 'PA1544', 'PA1544', 'PA1544'],
            'Target': ['PA0001', 'PA0002', 'PA0003', 'PA0004', 'PA0005', 'PA0006'],
            'Regulation': ['activation', 'repression', '?', 'activation', 'unknown', 'activation'],
            'Reference': ['PMID:100', 'PMID:101', 'PMID:102', '', 'PMID:103', 'PMID:104'],
        }).to_csv(source, sep='\t', index=False)
        gene_map = root / 'alias.csv'
        pd.DataFrame({'alias':['zur','ZUR','pseudonym','pseudonym'],
                      'GeneID':['PA0756','PA0756','PA0005','PA0006']}).to_csv(gene_map,index=False)
        args = argparse.Namespace(source_kind='curated_regulation',source_name='synthetic',
                                  regulator_col=None,target_col=None,relation_col=None,
                                  reference_col=None,evidence_col=None,sheet=0,gene_map=gene_map)
        result = run(root, source, args)
        need(result['source_rows'] == 6 and result['quarantined_rows'] == 2,
             'quarantine should catch missing reference and invalid regulator')
        need(result['n_249_metal_genes_with_curated_regulatory_rows'] == 4,
             'expected four covered metal genes')
        output = pd.read_csv(root / OUT_REL / 'NETWORK_V0_TYPED_RESPONSE_AND_EXTERNAL_REGULATORY_EVIDENCE.csv')
        need(len(output) == 249 + 4 and
             (output.edge_type == 'EXPERIMENTALLY_OBSERVED_PERTURBATION_GENE_RESPONSE').sum() == 249,
             'typed evidence layers unexpectedly collapsed')
        try:
            run(root, source, args)
        except RuntimeError as exc:
            need('output exists' in str(exc), 'overwrite safeguard raised wrong error')
        else:
            raise AssertionError('existing output was overwritten')
        # In binding-only mode no regulatory sign should be extrapolated.
        args.source_kind = 'binding_only'
        df = read_source(source)
        aliases, conflicts = load_aliases(gene_map, set(f'PA{x:04d}' for x in range(1, 4924)))
        v, _, _ = normalise_table(df, args, aliases, conflicts,
                                  set(f'PA{x:04d}' for x in range(1, 4924)))
        need((v.regulatory_sign_evidence == 'UNKNOWN_BINDING_ONLY').all(),
             'binding-only source was assigned regulatory sign')
    print('SELF_TEST_PASSED: 249 frozen response edges, external curated signed/unknown edges,')
    print('  exact PAO1 and independently specified alias mapping, quarantine, SHA and no-overwrite')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--source', type=Path, help='Authentic downloaded external TF-target table, CSV/TSV/TXT/XLSX')
    ap.add_argument('--source-name', default='RegulomePA')
    ap.add_argument('--source-kind', choices=['curated_regulation', 'binding_only'], default='curated_regulation')
    ap.add_argument('--regulator-col')
    ap.add_argument('--target-col')
    ap.add_argument('--relation-col')
    ap.add_argument('--reference-col')
    ap.add_argument('--evidence-col')
    ap.add_argument('--gene-map', type=Path, help='Explicit independently verified CSV columns alias,GeneID')
    ap.add_argument('--sheet', default=0, help='XLSX sheet name or zero-based integer index')
    ap.add_argument('--inspect-source', action='store_true', help='Only report external file headers/preview; write nothing')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    need(args.source is not None, 'provide --source PATH to an authentic external source table')
    root = args.root.expanduser().resolve()
    source = args.source.expanduser()
    if not source.is_absolute():
        source = (root / source).resolve()
    if args.gene_map is not None and not args.gene_map.is_absolute():
        args.gene_map = (root / args.gene_map).resolve()
    if str(args.sheet).isdigit():
        args.sheet = int(args.sheet)
    if args.inspect_source:
        df = read_source(source, args.sheet)
        print('EXTERNAL REGULATORY TABLE PREVIEW (NO WRITES):', source)
        print('Rows:', len(df), 'Columns:', len(df.columns))
        print('Headers:', list(df.columns))
        print(df.head(8).to_string(index=False))
        print('Next: confirm regulator, target, regulation, reference headers and PA#### identifiers.')
        return
    run(root, source, args)


if __name__ == '__main__':
    main()
