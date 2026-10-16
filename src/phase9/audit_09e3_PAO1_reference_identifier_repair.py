#!/usr/bin/env python3
"""CF-State Audit 09e3: conservative PAO1 regulatory identifier repair.

Independent NCBI PAO1 reference genome GFF3 is used only to resolve the
identifiers of EXISTING 2011 literature-curated regulator-target rows. No
new regulatory interactions, operon expansion, sign inference, or fitted
network is introduced.

Requires the already-frozen 09e2 and 09d outputs and both preceding scripts
(09e and 09e2) in the same src/phase9 directory.

Usage:
  python audit_09e3_PAO1_reference_identifier_repair.py --self-test
  python audit_09e3_PAO1_reference_identifier_repair.py \\
    --gff references/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz

Keeps 09e2 unchanged. Default new 09e3 results directory fails if present.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

import audit_09e2_2011_curated_regulatory_workbook_import as prior
import audit_09e_curated_regulatory_evidence_overlay as base

ROOT = Path(__file__).resolve().parents[2]
PRIOR_REL = prior.OUT_REL
OUT_REL = Path('results/phase9/audit_09e3_PAO1_reference_identifier_repair')
EXPECTED_ASSEMBLY = 'GCF_000006765.1'
EXPECTED_ACCESSION = 'NC_002516.2'
ID_RE = re.compile(r'^PA\d{4}$', re.IGNORECASE)
GFF_PA_RE = re.compile(r'(?<![A-Za-z0-9])PA\d{4}(?![A-Za-z0-9])', re.IGNORECASE)
PAO1_STRAIN_RE = re.compile(r'(?<![A-Za-z0-9])PAO1(?![A-Za-z0-9])', re.I)


def need(test, msg):
    if not test:
        raise RuntimeError('AUDIT_09E3_FAILED: ' + msg)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def gff_open(path):
    return gzip.open(path, 'rt', encoding='utf-8', errors='replace') if str(path).endswith('.gz') else open(path, 'rt', encoding='utf-8', errors='replace')


def parse_attributes(raw):
    attrs = {}
    for part in raw.split(';'):
        if '=' in part:
            key, val = part.split('=', 1)
            attrs[key] = unquote(val)
    return attrs


def read_reference_annotations(path, strict=True):
    """Read PAO1 gene records only; avoid CDS duplication and inferred aliases."""
    need(path.is_file(), 'verified independent PAO1 GFF3 missing: ' + str(path))
    genes = []
    header_lines = []
    accessions = set()
    with gff_open(path) as f:
        for line in f:
            if line.startswith('#'):
                if len(header_lines) < 500:
                    header_lines.append(line.rstrip())
                continue
            if not line.strip():
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 9 or parts[2].lower() != 'gene':
                continue
            accessions.add(parts[0])
            attrs = parse_attributes(parts[8])
            locus = str(attrs.get('locus_tag', '')).strip().upper()
            if not ID_RE.fullmatch(locus):
                matches = GFF_PA_RE.findall(attrs.get('ID', ''))
                locus = matches[0].upper() if len(matches) == 1 else ''
            if not ID_RE.fullmatch(locus):
                continue
            symbols = []
            for field in ('gene', 'gene_name'):
                val = attrs.get(field, '').strip()
                if val and ',' not in val and '|' not in val:
                    symbols.append((val, 'NCBI_GFF_EXPLICIT_GENE_SYMBOL'))
            # Optional synonyms are used ONLY when explicitly bound to the
            # same PA locus in the gene feature. Ambiguity across loci fails.
            for field in ('gene_synonym', 'gene_synonyms'):
                raw_syn = attrs.get(field, '').strip()
                if raw_syn:
                    for name in re.split(r'[,|]', raw_syn):
                        if name.strip():
                            symbols.append((name.strip(), 'NCBI_GFF_EXPLICIT_GENE_SYNONYM'))
            genes.append({'GeneID': locus, 'seqid': parts[0], 'symbols': symbols})
    need(len(genes) > 0, 'no PA#### gene features in supplied GFF')
    # Explicitly verify PAO1 accession, not only user-supplied file name.
    if strict:
        need(EXPECTED_ACCESSION in accessions or any(EXPECTED_ACCESSION in s for s in header_lines),
             'GFF does not demonstrate reference accession NC_002516.2; reject to avoid cross-strain aliases')
        need(len(set(x['GeneID'] for x in genes)) >= 4900,
             'too few distinct PA#### gene records for a complete PAO1 annotation')
    return genes, sorted(accessions)


def augmented_aliases(mask, root, gff, explicit_alias=None, strict=True):
    annotations, accessions = read_reference_annotations(gff, strict=strict)
    # Use precisely the original 09e2 mapping rules and independent sources.
    original, conflicts, origins, source_paths = prior.build_alias_map(
        mask, root, gff=gff, external_alias=explicit_alias)
    mapping_sets = defaultdict(set)
    mapping_origins = defaultdict(set)
    for name, locus in original.items():
        mapping_sets[name].add(locus)
        mapping_origins[name].update(origins.get(name, '').split('|'))
    for name, vals in conflicts.items():
        mapping_sets[name].update(vals.split('|'))
        # Preserve ambiguity and never overwrite with new GFF annotations.
        mapping_origins[name].add('PREVIOUS_SOURCE_CONFLICT')
    for row in annotations:
        # Crucial: resolve exact PA#### loci even when outside the frozen
        # 4,923-gene response mask. Do not restrict regulatory source nodes.
        mapping_sets[row['GeneID']].add(row['GeneID'])
        mapping_origins[row['GeneID']].add('NCBI_GFF_EXPLICIT_LOCUS_TAG')
        for symbol, provenance in row['symbols']:
            key = symbol.upper().strip()
            if key:
                mapping_sets[key].add(row['GeneID'])
                mapping_origins[key].add(provenance)
    unique = {name: next(iter(gids)) for name, gids in mapping_sets.items() if len(gids) == 1}
    ambiguous = {name: '|'.join(sorted(gids)) for name, gids in mapping_sets.items() if len(gids) > 1}
    sources = {key: '|'.join(sorted(v for v in mapping_origins[key] if v)) for key in unique}
    return unique, ambiguous, sources, source_paths, annotations, accessions


def reconstruct_input(baseline):
    remap = {
        'source_excel_row': 'source_excel_row',
        'raw_regulator': 'Regulator (TF or sigma)',
        'raw_operon_annotation': 'Operon',
        'raw_target': 'Target gene',
        'raw_mode_of_regulation': 'mode of regulation',
        'raw_experimental_evidence': 'Experimental Evidence',
        'original_PMID': 'PubMed Referencea',
        'original_strain': 'P. aeruginosa Strain',
    }
    need(set(remap).issubset(baseline.columns), '09e2 frozen 2011 row ledger schema is incomplete')
    raw = baseline[list(remap)].rename(columns=remap).copy()
    raw['source_excel_row'] = pd.to_numeric(raw['source_excel_row'], errors='raise').astype(int)
    need(raw.source_excel_row.is_unique, 'duplicate 2011 source workbook row IDs')
    return raw


def most_common_missing(quarantine, field, n=12):
    counts = quarantine.loc[quarantine.quarantine_reason.str.contains(field, na=False)]
    if field == 'REGULATOR':
        col = 'raw_regulator'
    else:
        col = 'raw_target'
    return counts[col].value_counts().head(n)


def compute(root, gff, explicit_alias, out, strict_ref=True):
    need(not out.exists(), 'output directory already exists (refuse overwrite): ' + str(out))
    olddir = root / PRIOR_REL
    needed = [olddir / 'FROZEN_ALL_2011_WORKBOOK_ROWS_MAPPING_STATUS.csv',
              olddir / 'QC_09E2.json',
              olddir / 'AUDIT_09E2_STATUS.txt']
    for p in needed:
        need(p.is_file(), 'frozen 09e2 prerequisite missing: ' + str(p))
    mask, metal, frozen_paths = base.load_frozen(root)
    need(len(metal) == 249, 'frozen 09d selected-metal response count changed')
    baseline = pd.read_csv(needed[0], dtype=str, keep_default_na=False)
    oldqc = json.loads(needed[1].read_text())
    need(oldqc.get('status', '').startswith('PHASE09E2_2011_CURATED_REGULATORY_WORKBOOK_OVERLAY_COMPLETE'),
         'unexpected 09e2 status')
    need(len(baseline) == int(oldqc['source_rows']), '09e2 ledger row count does not match frozen QC')
    raw = reconstruct_input(baseline)
    aliases, conflicts, origins, reference_paths, gene_rows, accessions = augmented_aliases(
        mask, root, gff, explicit_alias, strict=strict_ref)
    updated = prior.normalize_rows(raw, aliases, conflicts, origins)
    need(len(updated) == len(baseline), 'lost source rows during mapping repair')
    baseline['source_excel_row'] = pd.to_numeric(baseline['source_excel_row'], errors='raise').astype(int)
    old = baseline.set_index('source_excel_row')
    new = updated.set_index('source_excel_row')
    need(set(old.index) == set(new.index), 'changed source Excel record identities')
    compare_cols = ['raw_regulator', 'raw_target', 'raw_mode_of_regulation',
                    'raw_experimental_evidence', 'original_PMID', 'original_strain']
    for col in compare_cols:
        need(old[col].equals(new[col]), 'frozen 09e2 original evidence changed in column ' + col)
    oldmapped = old.row_status.eq('CURATED_MAPPED')
    newmapped = new.row_status.eq('CURATED_MAPPED')
    oldregs = old.regulator_GeneID
    oldtargets = old.target_GeneID
    changed = oldmapped & newmapped & ((oldregs != new.regulator_GeneID) | (oldtargets != new.target_GeneID))
    need(not changed.any(),
         'formerly mapped TF/target changed locus assignment; manual review required at rows ' +
         ','.join(changed[changed].index.astype(str).tolist()[:15]))
    rescued = ~oldmapped & newmapped
    regress = oldmapped & ~newmapped
    preserved = oldmapped & newmapped
    updated['mapping_change_vs_09e2'] = updated['source_excel_row'].map(
        {int(idx): ('NEWLY_RESOLVED' if rescued.loc[idx] else
                    'QUARANTINED_PREVIOUSLY_MAPPED' if regress.loc[idx] else
                    'PRESERVED_MAPPED' if preserved.loc[idx] else 'STILL_QUARANTINED')
         for idx in old.index})
    updated['original_strain_includes_PAO1'] = updated.original_strain.apply(
        lambda x: bool(PAO1_STRAIN_RE.search(str(x))))
    updated['original_strain_multistrain_including_PAO1'] = (
        updated.original_strain_includes_PAO1 & ~updated.original_strain_explicit_PAO1)
    mapped = updated[updated.row_status.eq('CURATED_MAPPED')].copy()
    quarantine = updated[updated.row_status.eq('QUARANTINE')].copy()
    selected = set(metal.target_node.str.removeprefix('gene:'))
    overlap = mapped[mapped.target_GeneID.isin(selected)].copy()
    cover_rows = []
    for gene in sorted(selected):
        r = overlap[overlap.target_GeneID.eq(gene)]
        paexact = r[r.original_strain_explicit_PAO1]
        paincl = r[r.original_strain_includes_PAO1]
        cover_rows.append({
            'GeneID': gene,
            'n_curated_rows_any_strain': len(r),
            'n_distinct_regulators_any_strain': r.regulator_GeneID.nunique(),
            'n_curated_rows_PAO1_exact': len(paexact),
            'n_curated_rows_PAO1_including_mixed': len(paincl),
            'n_regulators_PAO1_including_mixed': paincl.regulator_GeneID.nunique(),
            'has_any_regulator_evidence': len(r) > 0,
            'has_exact_PAO1_strain_evidence': len(paexact) > 0,
            'has_PAO1_including_mixed_strain_evidence': len(paincl) > 0,
            'metal_response_remains_provisional': True,
        })
    coverage = pd.DataFrame(cover_rows)
    grouped = (overlap.groupby(['regulator_GeneID','raw_regulator'], as_index=False)
               .agg(n_selected_metal_targets=('target_GeneID','nunique'),
                    n_rows=('target_GeneID','size'),
                    n_PMIDs=('original_PMID','nunique'),
                    n_PAO1_including_mixed=('original_strain_includes_PAO1','sum'))
               .sort_values(['n_selected_metal_targets','regulator_GeneID'], ascending=[False, True]))
    # Rows with same TF/target but contradictory reported modes are flagged.
    conflicting_modes = (mapped.groupby(['regulator_GeneID','target_GeneID'])
                         .reported_mode_class.nunique())
    conflicting_modes = conflicting_modes[conflicting_modes > 1]
    conflicts_sign_df = (mapped.set_index(['regulator_GeneID','target_GeneID'])
                         .loc[conflicting_modes.index].reset_index() if len(conflicting_modes)
                         else mapped.head(0).copy())
    n_prev = int(oldmapped.sum())
    n_now = len(mapped)
    n_rescued = int(rescued.sum())
    n_regress = int(regress.sum())
    n_gff_loci = len(set(x['GeneID'] for x in gene_rows))
    result = {
        'status': 'PHASE09E3_INDEPENDENT_PAO1_IDENTIFIER_REPAIR_COMPLETE_NO_NETWORK_FITTING',
        'source_2011_rows': len(updated),
        'reference': 'NCBI_PAO1_GCF_000006765.1_ASM676v1',
        'accessions_in_gene_features': accessions,
        'reference_distinct_PA_locus_gene_rows': n_gff_loci,
        'reference_gene_records': len(gene_rows),
        'original_mapped_09e2': n_prev,
        'newly_resolved_source_rows': n_rescued,
        'previously_mapped_now_quarantined': n_regress,
        'mapped_09e3': n_now,
        'still_quarantined': len(quarantine),
        'new_distinct_regulator_target_pairs': int(mapped[['regulator_GeneID','target_GeneID']].drop_duplicates().shape[0]),
        'previous_metal_genes_any_strain': int(oldqc['selected_metal_genes_with_any_2011_regulatory_evidence']),
        'metal_genes_any_strain': int(coverage.has_any_regulator_evidence.sum()),
        'previous_metal_genes_exact_PAO1': int(oldqc['selected_metal_genes_with_explicit_PAO1_2011_regulatory_evidence']),
        'metal_genes_exact_PAO1': int(coverage.has_exact_PAO1_strain_evidence.sum()),
        'metal_genes_PAO1_including_mixed': int(coverage.has_PAO1_including_mixed_strain_evidence.sum()),
        'mapped_rows_exact_PAO1': int(mapped.original_strain_explicit_PAO1.sum()),
        'mapped_rows_PAO1_including_mixed': int(mapped.original_strain_includes_PAO1.sum()),
        'n_distinct_symbol_conflicts': len(conflicts),
        'mapped_pairs_with_conflicting_reported_modes': len(conflicting_modes),
        'requires_mapping_review_before_promotion': n_regress > 0,
        'reference_sha256': file_sha256(gff),
        'all_published_modes_and_PubMed_identifiers_preserved': True,
        'new_edges_inferred': False,
        'direct_binding_newly_established': False,
        'CF_specificity_established': False,
        'network_weights_fitted': False,
    }
    # Collect signs and strain info separately, never as a new discovered edge.
    regulatory = mapped.copy()
    regulatory['source_node'] = 'gene:' + regulatory.regulator_GeneID
    regulatory['target_node'] = 'gene:' + regulatory.target_GeneID
    regulatory['edge_semantics'] = '2011_LITERATURE_CURATED_HETEROGENEOUS_REGULATORY_EVIDENCE'
    report = '''# Phase 09e3 — Regulatory identifier repair (2011 curated literature)

Independent reference: NCBI *Pseudomonas aeruginosa* PAO1 GCF_000006765.1,
chromosome NC_002516.2; original GFF3 and SHA256 retained.

The original 2011 curated regulatory associations were not changed or expanded.
Only name-to-locus assignment was updated using explicit PAO1 GFF3 gene symbols
and gene synonyms, combined with already-frozen independent annotations.
Ambiguous identifiers and still-unresolved complexes and multi-gene names remain
quarantined. No operon expansion, inferred TF binding or learned edge weights.

- Old mapped interactions: {old}
- Newly resolved source rows: {rescue}
- Formerly mapped rows now quarantined: {regress}
- Final mapped source rows: {mapped}
- Still quarantined: {quarantine}
- Selected metal-response genes with any mapped curated regulator: {metal_any}/249
- With exact PAO1 strain records: {metal_exact}/249
- Including mixed-strain labels that explicitly mention PAO1: {metal_include}/249

CAVEATS:
- Dataset remains a historical and heterogeneous literature-curated network.
- PMID count is not the number of independent biological experiments.
- A recorded '+' or '-' is the paper's curator-reported mode, not a sign
  independently inferred from 2024 CF-sputum perturbation responses.
- Cross-strain mapping follows PAO1 locus annotations, not established functional
  conservation of regulation between strains.
- A node's high degree can reflect biased historical coverage/operons.
- The original 249 metal responses remain single-study provisional links.
- No CF specificity, direct physical TF binding, independent intervention
  replication, improved held-out transfer, or clinical phenotype prediction.
- If any old row became quarantined, resolve conflicts manually before
  promoting the changed overlay as the frozen regulatory backbone.
'''.format(old=n_prev, rescue=n_rescued, regress=n_regress, mapped=n_now,
           quarantine=len(quarantine), metal_any=result['metal_genes_any_strain'],
           metal_exact=result['metal_genes_exact_PAO1'],
           metal_include=result['metal_genes_PAO1_including_mixed'])
    provenance_paths = [needed[0], needed[1], needed[2], gff] + frozen_paths + reference_paths
    if explicit_alias is not None:
        provenance_paths.append(explicit_alias)
    provenance_paths = sorted({p.resolve() for p in provenance_paths})
    out.mkdir(parents=True, exist_ok=False)
    # Freeze the actual alias mappings, including out-of-mask PA loci, for review.
    pd.DataFrame([{'alias':k, 'GeneID':v, 'mapping_source': origins.get(k,'')}
                  for k,v in sorted(aliases.items())]).to_csv(
        out / 'FROZEN_INDEPENDENT_PAO1_ALIAS_TO_LOCUS_MAP.csv', index=False)
    pd.DataFrame([{'alias':k, 'conflicting_PA_loci':v}
                  for k,v in sorted(conflicts.items())],
                 columns=['alias','conflicting_PA_loci']).to_csv(
        out / 'REVIEW_AMBIGUOUS_REFERENCE_ALIASES.csv',index=False)
    updated.to_csv(out / 'FROZEN_REVIEWED_2011_ROWS_PAO1_MAPPING_09E3.csv', index=False)
    mapped.to_csv(out / 'FROZEN_MAPPED_2011_CURATED_REGULATORY_EVIDENCE_09E3.csv', index=False)
    quarantine.to_csv(out / 'REVIEW_REMAINING_UNRESOLVED_2011_ROWS_09E3.csv', index=False)
    updated[updated.mapping_change_vs_09e2.eq('NEWLY_RESOLVED')].to_csv(
        out / 'REVIEW_NEWLY_RESOLVED_ROWS_VS_09E2.csv', index=False)
    updated[updated.mapping_change_vs_09e2.eq('QUARANTINED_PREVIOUSLY_MAPPED')].to_csv(
        out / 'REVIEW_PREVIOUS_MAPPED_NOW_CONFLICTED.csv', index=False)
    coverage.to_csv(out / 'DESCRIPTIVE_249_METAL_GENE_REGULATOR_COVERAGE_09E3.csv', index=False)
    overlap.to_csv(out / 'DESCRIPTIVE_CURATED_EDGES_TO_249_METAL_GENES_09E3.csv', index=False)
    grouped.to_csv(out / 'DESCRIPTIVE_REGULATOR_RANKING_09E3.csv', index=False)
    regulatory.to_csv(out / 'REGULATORY_EVIDENCE_LAYER_09E3_NOT_INFERRED.csv', index=False)
    conflicts_sign_df.to_csv(out / 'REVIEW_PAIRS_WITH_CONFLICTING_REPORTED_MODES.csv', index=False)
    pd.DataFrame([{'path': str(p), 'sha256': file_sha256(p)} for p in provenance_paths]).to_csv(
        out / 'INPUT_SHA256_PROVENANCE.csv', index=False)
    (out / 'QC_09E3.json').write_text(json.dumps(result, indent=2) + '\n')
    (out / 'INTERPRETATION_LIMITS.md').write_text(report)
    (out / 'AUDIT_09E3_STATUS.txt').write_text(result['status'] + '\n')
    # All key outputs echoed to terminal, intentionally short and tee-friendly.
    print('\n' + '=' * 69)
    print('CF-STATE AUDIT 09E3 | PAO1 REGULATORY IDENTIFIER REPAIR')
    print('=' * 69)
    print('Source 2011 interactions:', len(updated))
    print('Independent GFF3 PA loci:', n_gff_loci, '| chromosome accession:', ', '.join(accessions))
    print('PREVIOUS mapped:', n_prev, '| quarantined:', len(baseline) - n_prev)
    print('NEWLY mapped rows:', n_rescued, '| mapping regressions:', n_regress)
    print('CURRENT mapped:', n_now, '| still quarantined:', len(quarantine))
    print('Current distinct TF-target pairs:', result['new_distinct_regulator_target_pairs'])
    print('Metal genes with curator evidence (any strain):', result['metal_genes_any_strain'], '/249',
          '| previously:', result['previous_metal_genes_any_strain'])
    print('Metal genes exact PAO1:', result['metal_genes_exact_PAO1'], '/249',
          '| previously:', result['previous_metal_genes_exact_PAO1'])
    print('Metal genes including mixed-strain PAO1:', result['metal_genes_PAO1_including_mixed'], '/249')
    print('Ambiguous symbol keys:', len(conflicts))
    print('\nPRIORITY REGULATOR RESCUE (2011 source rows)')
    for key in ['dnr','ihf','cbrB','agmR','rcdB','psdR','bqsR','mexZ']:
        o = baseline.raw_regulator.str.casefold().eq(key.casefold())
        newk = updated.raw_regulator.str.casefold().eq(key.casefold())
        rescued_key = newk & updated.mapping_change_vs_09e2.eq('NEWLY_RESOLVED')
        still_key = newk & updated.row_status.eq('QUARANTINE')
        print(f'  {key}: total={int(o.sum())} newly_resolved={int(rescued_key.sum())} still_quarantined={int(still_key.sum())}')
    print('\nTOP 15 REGULATORS OF METAL-RESPONSE GENES')
    print(grouped.head(15).to_string(index=False) if len(grouped) else 'none')
    print('\nTOP REMAINING UNRESOLVED REGULATORS')
    print(most_common_missing(quarantine, 'REGULATOR').to_string() if len(quarantine) else 'none')
    print('\nTOP REMAINING UNRESOLVED TARGETS')
    print(most_common_missing(quarantine, 'TARGET').to_string() if len(quarantine) else 'none')
    print('\nRECOVERED EXAMPLES (first 15 by original Excel row)')
    examples = updated[updated.mapping_change_vs_09e2.eq('NEWLY_RESOLVED')]
    print(examples[['source_excel_row','raw_regulator','regulator_GeneID',
                    'raw_target','target_GeneID','original_strain']].head(15).to_string(index=False)
          if len(examples) else 'none')
    print('\nREVIEW GATE:', 'REVIEW_CONFLICTS_BEFORE_FREEZE' if n_regress else 'NO_PREVIOUS_MAPPED_REGRESSIONS')
    print('No new regulatory relationships, direct binding, CF specificity or learned network parameters inferred.')
    print('Outputs:', out)
    return result


def self_test():
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        maskpath = root / base.MASK_REL
        maskpath.parent.mkdir(parents=True)
        pd.DataFrame({'GeneID': [f'PA{i:04d}' for i in range(1, 4924)]}).to_csv(maskpath,index=False)
        ddir = root / base.D_REL
        ddir.mkdir(parents=True)
        (ddir/'AUDIT_09D_STATUS.txt').write_text(base.EXPECTED_D + '\n')
        metal = pd.DataFrame({'source_node': ['intervention:metals']*249,
                              'target_node': [f'gene:PA{i:04d}' for i in range(1,250)],
                              'edge_type':['experimentally_measured_indirect_expression_response']*249,
                              'effect_direction':['UP']*249,
                              'direct_TF_regulation':[False]*249,
                              'CF_specificity_demonstrated':[False]*249,
                              'independently_replicated':[False]*249})
        metal.to_csv(ddir/'NETWORK_V0_PROVISIONAL_METAL_RESPONSE_EDGES_NOT_REGULATORY.csv',index=False)
        olddir = root / PRIOR_REL
        olddir.mkdir(parents=True)
        raw = pd.DataFrame([
            {'source_excel_row':5,'Regulator (TF or sigma)':'known','Target gene':'targetA',
             'Operon':'','mode of regulation':'+','Experimental Evidence':'reporter',
             'PubMed Referencea':'12345678','P. aeruginosa Strain':'PAO1'},
            {'source_excel_row':6,'Regulator (TF or sigma)':'dnr','Target gene':'targetB',
             'Operon':'','mode of regulation':'-','Experimental Evidence':'RT-PCR',
             'PubMed Referencea':'23456789','P. aeruginosa Strain':'PAK and PAO1'},
            {'source_excel_row':7,'Regulator (TF or sigma)':'ihf','Target gene':'targetC',
             'Operon':'','mode of regulation':'?','Experimental Evidence':'chip',
             'PubMed Referencea':'34567890','P. aeruginosa Strain':'PA14'},
        ])
        genes = [
            ('PA0001','known',''),('PA0002','targetA',''),('PA0003','dnr',''),
            ('PA0004','targetB',''),('PA0005','targetC',''),('PA5500','outsideMask','')]
        gff = root/'references/test_pao1.gff3'
        gff.parent.mkdir(parents=True)
        lines = ['##gff-version 3','##sequence-region NC_002516.2 1 6264404']
        for i,(gene,sym,syn) in enumerate(genes,1):
            extra = ';gene_synonym='+syn if syn else ''
            lines.append(f'NC_002516.2\tRefSeq\tgene\t{i*100}\t{i*100+90}\t.\t+\t.\tID=gene-{gene};locus_tag={gene};gene={sym}{extra}')
        gff.write_text('\n'.join(lines)+'\n')
        mask = set(pd.read_csv(maskpath).GeneID)
        existing = {'KNOWN':'PA0001','TARGETA':'PA0002', 'PA0001':'PA0001','PA0002':'PA0002'}
        baseline = prior.normalize_rows(raw, existing, {}, {'KNOWN':'SYNTH_OLD', 'TARGETA':'SYNTH_OLD'})
        baseline.to_csv(olddir/'FROZEN_ALL_2011_WORKBOOK_ROWS_MAPPING_STATUS.csv',index=False)
        qc = {'status':'PHASE09E2_2011_CURATED_REGULATORY_WORKBOOK_OVERLAY_COMPLETE_NO_FITTING',
              'source_rows':3, 'selected_metal_genes_with_any_2011_regulatory_evidence':1,
              'selected_metal_genes_with_explicit_PAO1_2011_regulatory_evidence':1}
        (olddir/'QC_09E2.json').write_text(json.dumps(qc)+'\n')
        (olddir/'AUDIT_09E2_STATUS.txt').write_text(qc['status']+'\n')
        dest = root/OUT_REL
        result = compute(root,gff,None,dest,strict_ref=False)
        need(result['newly_resolved_source_rows']==1, 'test only exact new symbol resolves')
        need(result['mapped_09e3']==2, 'test one complex quarantined')
        need(result['previously_mapped_now_quarantined']==0,'test no regression')
        need(result['metal_genes_PAO1_including_mixed']==2,'test mixed-strain PAO1 identification')
        need(result['metal_genes_exact_PAO1']==1,'test exact strain distinction')
        try:
            compute(root,gff,None,dest,strict_ref=False)
        except RuntimeError as e:
            need('refuse overwrite' in str(e), 'test no-overwrite error correct')
        else:
            raise AssertionError('test must refuse overwrite')
        badgff = root/'references/non_pao1.gff3'
        badgff.write_text('##gff-version 3\nX\tRefSeq\tgene\t1\t2\t.\t+\t.\tID=gene-PA0001;locus_tag=PA0001;gene=foo\n')
        try:
            read_reference_annotations(badgff,strict=True)
        except RuntimeError as e:
            need('NC_002516.2' in str(e),'test non-PAO1 fail-closed')
        else:
            raise AssertionError('non PAO1 annotation should fail')
        _,conflicts,_,_,_,_ = augmented_aliases(mask,root,gff,strict=False)
        unique, conflicts, _, _, _, _ = augmented_aliases(mask,root,gff,strict=False)
        need('KNOWN' not in conflicts,'test unique established symbol')
        need(unique['PA5500']=='PA5500', 'test canonical source loci outside 4923 mask')
    print('SELF_TEST_PASSED: independent PAO1 annotation, existing source row preservation,')
    print('  conservative recovered symbol, unresolved complex, mixed-strain labels,')
    print('  provenance, no overwrite, chromosome accession guard and 249 edge contract.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--gff', type=Path, help='Required verified NCBI PAO1 NC_002516.2 GFF3, .gz supported')
    ap.add_argument('--alias-map', type=Path, help='Optional independently verified alias,GeneID CSV')
    ap.add_argument('--self-test', action='store_true')
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    need(a.gff is not None, '--gff required (never silently infer/download reference data)')
    root = a.root.expanduser().resolve()
    gff = (root/a.gff).resolve() if not a.gff.is_absolute() else a.gff.resolve()
    alias = (root/a.alias_map).resolve() if a.alias_map and not a.alias_map.is_absolute() else a.alias_map
    if alias is not None:
        alias = alias.resolve()
    compute(root,gff,alias,root/OUT_REL,strict_ref=True)


if __name__ == '__main__':
    main()
