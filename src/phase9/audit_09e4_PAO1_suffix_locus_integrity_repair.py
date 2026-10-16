#!/usr/bin/env python3
"""CF-State audit 09e4: protect PA####.n loci in historical regulatory edges.

Repair a verified parser defect in 09e3 that truncated dotted PAO1 loci, e.g.
RsmY PA0527.1 -> Dnr PA0527, and CcoQ1 PA1552.1 -> CcoP1 PA1552.
No new published interaction is inferred. The frozen 09e2 and 09e3 ledgers
are compared row-by-row; both remain unchanged.

Commands:
  python audit_09e4_PAO1_suffix_locus_integrity_repair.py --self-test
  python audit_09e4_PAO1_suffix_locus_integrity_repair.py \\
      --gff references/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import audit_09e2_2011_curated_regulatory_workbook_import as prior
import audit_09e_curated_regulatory_evidence_overlay as base

ROOT = Path(__file__).resolve().parents[2]
OLD2 = prior.OUT_REL
OLD3 = Path('results/phase9/audit_09e3_PAO1_reference_identifier_repair')
DEST = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair')
PA_ID = re.compile(r'PA\d{4}(?:\.\d+)?', re.I)
BARE_PA = re.compile(r'PA\d{4}', re.I)
EXACT_CHR = 'NC_002516.2'
EXPECTED_SHA256 = 'e6b8bc796af95342acd5c170ecf5003e75135c894b1c22f8c4b46c51c4bdf7f0'
STRAIN_PAO1 = re.compile(r'(?<![A-Za-z0-9])PAO1(?![A-Za-z0-9])', re.I)


def require(ok, message):
    if not ok:
        raise RuntimeError('AUDIT_09E4_FAILED: ' + message)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def attributes(raw):
    result = {}
    for piece in raw.split(';'):
        if '=' in piece:
            k, v = piece.split('=', 1)
            result[k] = unquote(v)
    return result


def annotation_rows(gff, strict=True):
    require(Path(gff).is_file(), 'reference GFF3 missing: ' + str(gff))
    opening = gzip.open if str(gff).endswith('.gz') else open
    genes = []
    seen_chromosome = False
    with opening(gff, 'rt', encoding='utf-8', errors='replace') as f:
        for raw in f:
            if raw.startswith('##sequence-region ' + EXACT_CHR + ' '):
                seen_chromosome = True
            if raw.startswith('#'):
                continue
            bits = raw.rstrip('\n').split('\t')
            if len(bits) != 9 or bits[2].lower() != 'gene':
                continue
            if bits[0] != EXACT_CHR:
                continue
            a = attributes(bits[8])
            locus = a.get('locus_tag', '').strip().upper()
            if not PA_ID.fullmatch(locus):
                # Crucial: FULL ID matching. Never allow "PA0527.1" -> "PA0527".
                match = re.fullmatch(r'gene-(PA\d{4}(?:\.\d+)?)', a.get('ID', ''), re.I)
                locus = match.group(1).upper() if match else ''
            if not PA_ID.fullmatch(locus):
                continue
            symbols = []
            for field in ('gene', 'gene_name'):
                name = a.get(field, '').strip()
                if name and not any(c in name for c in ',|'):
                    symbols.append((name, 'NCBI_GFF_' + field.upper()))
            for field in ('gene_synonym', 'gene_synonyms'):
                for name in re.split('[,|]', a.get(field, '')):
                    if name.strip():
                        symbols.append((name.strip(), 'NCBI_GFF_' + field.upper()))
            genes.append({'GeneID': locus, 'symbols': symbols,
                          'biotype': a.get('gene_biotype', ''),
                          'gff_id': a.get('ID', ''), 'start': bits[3], 'end': bits[4]})
    if strict:
        require(seen_chromosome, 'missing exact PAO1 chromosome accession ' + EXACT_CHR)
        require(len(set(x['GeneID'] for x in genes if BARE_PA.fullmatch(x['GeneID']))) >= 4900,
                'too few PAO1 gene loci: wrong or incomplete annotation')
        require(sha256(gff) == EXPECTED_SHA256,
                'GFF3 SHA256 differs from user-verified reference; investigate before import')
    require(len(genes) > 0, 'no valid PAO1 reference gene features')
    locus_to_type = {r['GeneID']: r['biotype'] for r in genes}
    return genes, locus_to_type


def make_aliases(root, mask, genes):
    # 09e2 native reference sources: no GFF arg, so its dot-truncating fallback
    # can never run. Freeze older evidence rather than silently adding synonyms.
    old, previous_conflicts, old_origins, source_paths = prior.build_alias_map(
        mask, root, gff=None, external_alias=None)
    candidates = defaultdict(set)
    provenance = defaultdict(set)
    for sym, locus in old.items():
        candidates[sym.upper()].add(locus.upper())
        provenance[sym.upper()].update(filter(None, old_origins.get(sym, '').split('|')))
    for sym, val in previous_conflicts.items():
        candidates[sym.upper()].update(val.upper().split('|'))
        provenance[sym.upper()].add('PREEXISTING_MAPPING_CONFLICT')
    for gene in genes:
        locus = gene['GeneID']
        candidates[locus].add(locus)
        provenance[locus].add('NCBI_EXACT_FULL_LOCUS_TAG')
        for symbol, source in gene['symbols']:
            sym = symbol.upper().strip()
            if sym:
                candidates[sym].add(locus)
                provenance[sym].add(source)
    unique = {k: next(iter(v)) for k, v in candidates.items() if len(v) == 1}
    conflicts = {k: '|'.join(sorted(v)) for k, v in candidates.items() if len(v) > 1}
    origin = {k: '|'.join(sorted(provenance[k])) for k in candidates}
    return unique, conflicts, origin, source_paths


def previous_inputs(root):
    bdir = root / OLD2
    odir = root / OLD3
    p2 = bdir / 'FROZEN_ALL_2011_WORKBOOK_ROWS_MAPPING_STATUS.csv'
    q2 = bdir / 'QC_09E2.json'
    p3 = odir / 'FROZEN_REVIEWED_2011_ROWS_PAO1_MAPPING_09E3.csv'
    q3 = odir / 'QC_09E3.json'
    for p in (p2, q2, p3, q3):
        require(p.is_file(), 'frozen 09e2/09e3 prerequisite absent: ' + str(p))
    old2 = pd.read_csv(p2, dtype=str, keep_default_na=False)
    old3 = pd.read_csv(p3, dtype=str, keep_default_na=False)
    qc2 = json.loads(q2.read_text())
    qc3 = json.loads(q3.read_text())
    require(len(old2) == len(old3) == int(qc2['source_rows']) == int(qc3['source_2011_rows']),
            'frozen 2011 source row count mismatch')
    for df in (old2, old3):
        df['source_excel_row'] = pd.to_numeric(df['source_excel_row'], errors='raise').astype(int)
        require(df.source_excel_row.is_unique, 'duplicate source Excel row identifiers')
    old2 = old2.sort_values('source_excel_row').reset_index(drop=True)
    old3 = old3.sort_values('source_excel_row').reset_index(drop=True)
    require(old2.source_excel_row.equals(old3.source_excel_row), '09e2/09e3 record IDs differ')
    preserve = ['raw_regulator', 'raw_target', 'raw_mode_of_regulation',
                'raw_experimental_evidence', 'original_PMID', 'original_strain']
    for name in preserve:
        require(old2[name].equals(old3[name]), '09e2 and 09e3 disagree on original ' + name)
    raw_cols = {'source_excel_row':'source_excel_row',
                'raw_regulator':'Regulator (TF or sigma)',
                'raw_target':'Target gene', 'raw_operon_annotation':'Operon',
                'raw_mode_of_regulation':'mode of regulation',
                'raw_experimental_evidence':'Experimental Evidence',
                'original_PMID':'PubMed Referencea',
                'original_strain':'P. aeruginosa Strain'}
    require(set(raw_cols).issubset(old2.columns), 'old 2011 ledger missing original evidence fields')
    raw = old2[list(raw_cols)].rename(columns=raw_cols)
    return old2, old3, raw, qc2, qc3, [p2, q2, p3, q3]


def run(root, gff, dest, strict=True):
    require(not dest.exists(), 'existing output (no overwrite): ' + str(dest))
    old2, old3, raw, qc2, qc3, sources = previous_inputs(root)
    mask, metal, frozen_sources = base.load_frozen(root)
    require(len(metal) == 249, 'expected 249 frozen metal-response edges')
    genes, biotypes = annotation_rows(gff, strict=strict)
    aliases, conflicts, origins, alias_sources = make_aliases(root, mask, genes)
    repaired = prior.normalize_rows(raw, aliases, conflicts, origins)
    repaired = repaired.sort_values('source_excel_row').reset_index(drop=True)
    require(repaired.source_excel_row.equals(old3.source_excel_row), 'repaired record IDs shifted')
    for x in ('raw_regulator', 'raw_target', 'raw_mode_of_regulation',
              'raw_experimental_evidence', 'original_PMID', 'original_strain'):
        require(repaired[x].equals(old3[x]), 'original published evidence mutated: ' + x)
    for side in ('regulator', 'target'):
        col = side + '_GeneID'
        repaired[side + '_reference_biotype'] = repaired[col].map(biotypes).fillna('')
        repaired[side + '_has_dotted_locus'] = repaired[col].str.contains(r'^PA\d{4}\.\d+$', regex=True)
        # Force any raw GFF dotted symbol to retain its complete locus. If it
        # conflicts with an existing annotation, quarantine rather than collapse.
        dot_symbols = {sym.upper(): record['GeneID']
                       for record in genes if '.' in record['GeneID']
                       for sym, _ in record['symbols']}
        wrong = repaired['raw_' + side].str.upper().isin(dot_symbols)
        bad = wrong & repaired[col].ne(repaired['raw_' + side].str.upper().map(dot_symbols)) \
            & repaired.row_status.eq('CURATED_MAPPED')
        require(not bad.any(), 'dotted PAO1 gene mapped to different locus in repaired rows: ' +
                ','.join(repaired.loc[bad, 'source_excel_row'].astype(str).tolist()[:10]))
    statuses = []
    for a, b in zip(old3.to_dict('records'), repaired.to_dict('records')):
        was = a['row_status'] == 'CURATED_MAPPED'
        now = b['row_status'] == 'CURATED_MAPPED'
        old_ids = (a['regulator_GeneID'], a['target_GeneID'])
        new_ids = (b['regulator_GeneID'], b['target_GeneID'])
        if was and now and old_ids != new_ids:
            statuses.append('REPAIRED_PREVIOUSLY_MAPPED_IDENTITY')
        elif was and not now:
            statuses.append('PREVIOUSLY_MAPPED_NOW_QUARANTINED')
        elif not was and now:
            statuses.append('NEWLY_MAPPED_09E4')
        elif was and now:
            statuses.append('PRESERVED_MAPPED')
        else:
            statuses.append('STILL_QUARANTINED')
    repaired['mapping_change_vs_09e3'] = statuses
    repaired['original_strain_includes_PAO1'] = repaired.original_strain.apply(
        lambda v: bool(STRAIN_PAO1.search(v)))
    mapped = repaired[repaired.row_status.eq('CURATED_MAPPED')].copy()
    unresolved = repaired[repaired.row_status.eq('QUARANTINE')].copy()
    changed = repaired[repaired.mapping_change_vs_09e3.isin(
        ['REPAIRED_PREVIOUSLY_MAPPED_IDENTITY', 'PREVIOUSLY_MAPPED_NOW_QUARANTINED',
         'NEWLY_MAPPED_09E4'])].copy()
    original = old3[['source_excel_row', 'regulator_GeneID', 'target_GeneID', 'row_status']].rename(
        columns={k:'09e3_' + k for k in ('regulator_GeneID', 'target_GeneID', 'row_status')})
    changed = changed.merge(original, on='source_excel_row', validate='one_to_one')
    selected = set(metal.target_node.str.replace('gene:', '', regex=False))
    overlap = mapped[mapped.target_GeneID.isin(selected)].copy()
    covered = set(overlap.target_GeneID)
    exact = set(overlap.loc[overlap.original_strain_explicit_PAO1, 'target_GeneID'])
    mixed = set(overlap.loc[overlap.original_strain_includes_PAO1, 'target_GeneID'])
    coverage = pd.DataFrame([{
        'GeneID': g,
        'n_2011_regulatory_rows': int((overlap.target_GeneID == g).sum()),
        'n_distinct_regulators': overlap.loc[overlap.target_GeneID == g, 'regulator_GeneID'].nunique(),
        'any_strain_evidence': g in covered,
        'exact_PAO1_evidence': g in exact,
        'PAO1_including_mixed_evidence': g in mixed,
    } for g in sorted(selected)])
    ranks = (overlap.groupby(['regulator_GeneID','raw_regulator'], as_index=False)
             .agg(n_selected_metal_targets=('target_GeneID','nunique'),
                  n_source_rows=('target_GeneID','size'),
                  n_PMIDs=('original_PMID','nunique'))
             .sort_values(['n_selected_metal_targets','regulator_GeneID'], ascending=[False,True]))
    # Explicitly test known problematic distinctions whenever all symbols exist.
    must_be_distinct = [('RSMY', 'DNR'), ('RSMZ', 'PA3621'),
                        ('CCOQ1', 'CCOP1'), ('CCOQ2', 'CCOP2')]
    for a, b in must_be_distinct:
        if a in aliases and b in aliases:
            require(aliases[a] != aliases[b], a + ' and ' + b + ' incorrectly collapsed')
    # Validate ALL suffix-gene symbol mappings, not just the four known cases.
    bad_examples = []
    for gene in genes:
        if '.' not in gene['GeneID']:
            continue
        for symbol, _ in gene['symbols']:
            key = symbol.upper().strip()
            if key in aliases and aliases[key] != gene['GeneID']:
                bad_examples.append((key, aliases[key], gene['GeneID']))
    require(not bad_examples, 'dotted reference locus aliases still truncated: ' + str(bad_examples[:5]))
    qc = {
        'status': 'PHASE09E4_SUFFIX_SAFE_PA01_MAPPING_INTEGRITY_REPAIR_COMPLETE_NO_FITTING',
        'source_rows': len(repaired),
        'mapped_09e3_reported': int((old3.row_status == 'CURATED_MAPPED').sum()),
        'mapped_09e4': len(mapped),
        'quarantined_09e4': len(unresolved),
        'distinct_regulator_target_pairs': int(mapped[['regulator_GeneID','target_GeneID']].drop_duplicates().shape[0]),
        'n_reference_genes_including_dotted_loci': len(set(x['GeneID'] for x in genes)),
        'n_reference_dotted_loci': len(set(x['GeneID'] for x in genes if '.' in x['GeneID'])),
        'n_mapped_edges_to_dotted_locus': int((mapped.regulator_has_dotted_locus | mapped.target_has_dotted_locus).sum()),
        'n_corrected_previously_mapped_ids': int((repaired.mapping_change_vs_09e3 == 'REPAIRED_PREVIOUSLY_MAPPED_IDENTITY').sum()),
        'n_prior_mapped_now_quarantined': int((repaired.mapping_change_vs_09e3 == 'PREVIOUSLY_MAPPED_NOW_QUARANTINED').sum()),
        'n_newly_mapped': int((repaired.mapping_change_vs_09e3 == 'NEWLY_MAPPED_09E4').sum()),
        'n_selected_metal_genes_any_strain': len(covered),
        'n_selected_metal_genes_exact_PAO1': len(exact),
        'n_selected_metal_genes_PAO1_including_mixed': len(mixed),
        'n_ambiguous_aliases': len(conflicts),
        'reference_sha256': sha256(gff),
        'no_network_fitted': True,
        'no_direct_binding_inferred': True,
        'no_CF_specificity_claimed': True,
    }
    report = (
        '# Phase 09e4 — PAO1 suffix-safe locus integrity repair\n\n'
        'Reprocessed all 2011 source rows using independent NCBI PAO1 GFF3; '
        'retained PA0527.1, PA3621.1, PA1552.1, etc. as distinct '
        'reference loci rather than truncating to PA####.\n\n'
        'All prior 09e2 and 09e3 outputs remain unchanged. Source papers, '
        'strain labels, reported signs and evidence are not modified.\n\n'
        'Dots in PAO1 locus names identify distinct features including ncRNAs '
        'and small proteins; they must never be coerced into the four-digit '
        'protein-coding gene mask. Dotted-locus links have regulatory evidence '
        'only and are not measurements in the frozen 4,923-gene response space.\n\n'
        'The 2011 network is heterogeneous and historical. An association or '
        'reported +/- mode does not prove direct TF binding, a mechanistic '
        'perturbation response, CF specificity, or generalisation.\n'
    )
    dest.mkdir(parents=True, exist_ok=False)
    repaired.to_csv(dest / 'FROZEN_ALL_2011_ROWS_SUFFIX_SAFE_09E4.csv', index=False)
    mapped.to_csv(dest / 'FROZEN_CURATED_REGULATORY_ASSOCIATIONS_09E4.csv', index=False)
    unresolved.to_csv(dest / 'REVIEW_REMAINING_QUARANTINED_09E4.csv', index=False)
    changed.to_csv(dest / 'REVIEW_ALL_CHANGED_ROWS_VS_09E3.csv', index=False)
    coverage.to_csv(dest / 'DESCRIPTIVE_249_METAL_GENE_COVERAGE_09E4.csv', index=False)
    ranks.to_csv(dest / 'DESCRIPTIVE_REGULATORS_OF_249_METAL_GENES_09E4.csv', index=False)
    pd.DataFrame([{'alias':key, 'mapped_gene_id':val, 'mapping_basis':origins.get(key, '')}
                  for key,val in sorted(aliases.items())]).to_csv(dest / 'FROZEN_SUFFIX_SAFE_ALIASES.csv',index=False)
    pd.DataFrame([{'alias':key, 'conflicting_loci':val} for key,val in sorted(conflicts.items())],
                 columns=['alias','conflicting_loci']).to_csv(dest / 'REVIEW_AMBIGUOUS_ALIASES.csv',index=False)
    mapped.assign(source_node='gene:' + mapped.regulator_GeneID,
                  target_node='gene:' + mapped.target_GeneID,
                  edge_semantics='2011_LITERATURE_CURATED_HETEROGENEOUS_REGULATORY_EVIDENCE').to_csv(
                      dest / 'REGULATORY_EVIDENCE_LAYER_09E4.csv', index=False)
    used = sorted(set([gff] + sources + frozen_sources + alias_sources), key=str)
    pd.DataFrame([{'path':str(p.resolve()), 'sha256':sha256(p)} for p in used]).to_csv(
        dest / 'INPUT_SHA256_PROVENANCE.csv', index=False)
    (dest / 'QC_09E4.json').write_text(json.dumps(qc, indent=2) + '\n')
    (dest / 'INTERPRETATION_LIMITS.md').write_text(report)
    (dest / 'AUDIT_09E4_STATUS.txt').write_text(qc['status'] + '\n')

    print('\n' + '='*78)
    print('CF-STATE AUDIT 09E4 | FULL PAO1 LOCUS ID INTEGRITY')
    print('='*78)
    print('Source 2011 interactions:', len(repaired))
    print('09e3 mapped:', qc['mapped_09e3_reported'], '| 09e4 corrected mapped:', len(mapped),
          '| quarantined:', len(unresolved))
    print('Distinct curated pairs:', qc['distinct_regulator_target_pairs'])
    print('Reference gene loci:', qc['n_reference_genes_including_dotted_loci'],
          '| distinct dotted loci:', qc['n_reference_dotted_loci'])
    print('Mapped evidence rows involving dotted loci:', qc['n_mapped_edges_to_dotted_locus'])
    print('PREVIOUS MAPPED IDENTITIES CORRECTED:', qc['n_corrected_previously_mapped_ids'])
    print('PREVIOUS MAPPED NOW QUARANTINED:', qc['n_prior_mapped_now_quarantined'])
    print('NEW MAPPED ROWS:', qc['n_newly_mapped'])
    print('Selected metal-response genes with regulator evidence: any strain =', len(covered),
          '/249 | exact PAO1 =', len(exact), '| including mixed =', len(mixed))
    print('\nCRITICAL GFF ALIAS IDENTITIES:')
    for symbol in ('DNR','RSMY','RSMZ','CCOP1','CCOQ1','CCOP2','CCOQ2'):
        print('  %-7s -> %-12s %s' % (symbol,
              aliases.get(symbol, 'QUARANTINED/UNRESOLVED'),
              ('AMBIGUOUS ' + conflicts[symbol]) if symbol in conflicts else ''))
    print('\nALL PREVIOUSLY MAPPED IDENTITY CHANGES (max 40):')
    special = changed[changed.mapping_change_vs_09e3.eq('REPAIRED_PREVIOUSLY_MAPPED_IDENTITY')]
    if len(special):
        display = ['source_excel_row','raw_regulator','raw_target',
                   '09e3_regulator_GeneID','regulator_GeneID',
                   '09e3_target_GeneID','target_GeneID']
        print(special[display].head(40).to_string(index=False))
    else:
        print('  none')
    print('\nOTHER CHANGED ROWS (max 20):')
    other = changed[~changed.mapping_change_vs_09e3.eq('REPAIRED_PREVIOUSLY_MAPPED_IDENTITY')]
    print(other[['source_excel_row','raw_regulator','raw_target','mapping_change_vs_09e3',
                 'quarantine_reason']].head(20).to_string(index=False) if len(other) else '  none')
    print('\nTOP REGULATORS OF THE 249 METAL-RESPONSE GENES:')
    print(ranks.head(15).to_string(index=False) if len(ranks) else '  none')
    print('\nQUARANTINE REASONS:')
    print(unresolved.quarantine_reason.value_counts().head(10).to_string() if len(unresolved) else '  none')
    print('\nNo new regulation inferred. No model fitted. No CF specificity established.')
    print('Output:', dest)
    return qc


def self_test():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        mpath = root / base.MASK_REL
        mpath.parent.mkdir(parents=True)
        pd.DataFrame({'GeneID':[f'PA{i:04d}' for i in range(1,4924)]}).to_csv(mpath, index=False)
        ddir = root / base.D_REL
        ddir.mkdir(parents=True)
        (ddir/'AUDIT_09D_STATUS.txt').write_text(base.EXPECTED_D + '\n')
        pd.DataFrame({'source_node':['intervention:metals']*249,
                      'target_node':[f'gene:PA{i:04d}' for i in range(1,250)],
                      'edge_type':['experimentally_measured_indirect_expression_response']*249,
                      'effect_direction':['UP']*249,
                      'direct_TF_regulation':[False]*249,
                      'CF_specificity_demonstrated':[False]*249,
                      'independently_replicated':[False]*249}).to_csv(
            ddir/'NETWORK_V0_PROVISIONAL_METAL_RESPONSE_EDGES_NOT_REGULATORY.csv',index=False)
        p2 = root/OLD2
        p3 = root/OLD3
        p2.mkdir(parents=True)
        p3.mkdir(parents=True)
        raw = pd.DataFrame([
            {'source_excel_row':5,'Regulator (TF or sigma)':'gacA', 'Target gene':'RsmY',
             'Operon':'','mode of regulation':'+','Experimental Evidence':'reporter',
             'PubMed Referencea':'12345678','P. aeruginosa Strain':'PAO1'},
            {'source_excel_row':6,'Regulator (TF or sigma)':'dnr','Target gene':'RsmZ',
             'Operon':'','mode of regulation':'-','Experimental Evidence':'RT-PCR',
             'PubMed Referencea':'12345678','P. aeruginosa Strain':'PAK and PAO1'},
            {'source_excel_row':7,'Regulator (TF or sigma)':'roxR','Target gene':'ccoQ1',
             'Operon':'','mode of regulation':'+','Experimental Evidence':'microarray',
             'PubMed Referencea':'34567890','P. aeruginosa Strain':'PAO1'},
            {'source_excel_row':8,'Regulator (TF or sigma)':'roxR','Target gene':'ccoP1',
             'Operon':'','mode of regulation':'+','Experimental Evidence':'microarray',
             'PubMed Referencea':'34567890','P. aeruginosa Strain':'PAO1'},
        ])
        baseids = {'GACA':'PA0001','DNR':'PA0003','ROXR':'PA0007','CCOP1':'PA0010'}
        old2 = prior.normalize_rows(raw, baseids, {}, {k:'SYNTH' for k in baseids})
        old2.to_csv(p2/'FROZEN_ALL_2011_WORKBOOK_ROWS_MAPPING_STATUS.csv', index=False)
        map3 = dict(baseids, RSMY='PA0003', RSMZ='PA0005', CCOQ1='PA0010')
        old3 = prior.normalize_rows(raw, map3, {}, {k:'SYNTH' for k in map3})
        old3.to_csv(p3/'FROZEN_REVIEWED_2011_ROWS_PAO1_MAPPING_09E3.csv',index=False)
        (p2/'QC_09E2.json').write_text(json.dumps({'source_rows':4}) + '\n')
        (p3/'QC_09E3.json').write_text(json.dumps({'source_2011_rows':4}) + '\n')
        gff = root/'references/synthetic.gff3'
        gff.parent.mkdir(parents=True)
        features = [('PA0001','gacA','protein_coding'),('PA0003','dnr','protein_coding'),
                    ('PA0003.1','rsmY','ncRNA'),('PA0005','nearby','protein_coding'),
                    ('PA0005.1','rsmZ','ncRNA'),('PA0007','roxR','protein_coding'),
                    ('PA0010','ccoP1','protein_coding'),('PA0010.1','ccoQ1','protein_coding')]
        lines = ['##gff-version 3', '##sequence-region NC_002516.2 1 6264404']
        for i, (locus,name,typ) in enumerate(features,1):
            lines.append(f'NC_002516.2\tRefSeq\tgene\t{i*100}\t{i*100+80}\t.\t+\t.\tID=gene-{locus};locus_tag={locus};gene={name};gene_biotype={typ}')
        gff.write_text('\n'.join(lines)+'\n')
        out = root/DEST
        result = run(root,gff,out,strict=False)
        require(result['mapped_09e4'] == 4, 'test mapped all four synthetic pairs')
        require(result['n_corrected_previously_mapped_ids'] == 3,
                'test corrected all three false PA#### truncations')
        corrected = pd.read_csv(out/'FROZEN_ALL_2011_ROWS_SUFFIX_SAFE_09E4.csv')
        assert list(corrected.target_GeneID) == ['PA0003.1','PA0005.1','PA0010.1','PA0010']
        require(len(set(corrected.target_GeneID)) == 4, 'distinct dot and base loci collapsed')
        require(result['n_mapped_edges_to_dotted_locus'] == 3, 'test dotted node classification')
        try:
            run(root,gff,out,strict=False)
        except RuntimeError as e:
            require('no overwrite' in str(e), 'no overwrite guard absent')
        else:
            raise AssertionError('already-frozen output should not be overwritten')
        # A GFF ID fallback that contains an extra suffix MUST NOT be chopped.
        bad = root/'references/test_id_no_locus.gff3'
        bad.write_text('##sequence-region NC_002516.2 1 6264404\n'
                       'NC_002516.2\tRefSeq\tgene\t1\t10\t.\t+\t.\tID=gene-PA0003.2;gene=ncTest\n')
        features2,_ = annotation_rows(bad,strict=False)
        assert features2[0]['GeneID'] == 'PA0003.2'
        print('SELF_TEST_PASSED: distinct PA#### and PA####.1 features, sRNA and ccoQ correction,')
        print('  original row preservation, changed-row ledger, no overwrite, exact dot-safe ID fallback.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--gff', type=Path)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    if args.self_test:
        self_test()
        return
    require(args.gff is not None, '--gff required (never download implicitly)')
    root = args.root.expanduser().resolve()
    gff = args.gff if args.gff.is_absolute() else (root/args.gff)
    run(root,gff.resolve(),root/DEST,strict=True)

if __name__ == '__main__':
    main()
