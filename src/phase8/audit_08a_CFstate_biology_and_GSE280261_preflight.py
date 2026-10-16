#!/usr/bin/env python3
"""CF-State Phase 8, Audit 08a.

Reproducible first pass of biological interpretation of the FROZEN 03h
CF-airway candidate, plus a small public processed-RNA count data acquisition
and schema preflight for GEO GSE280261. No retraining, no sample-level
projection, and no claims of independent CF specificity.

Run from the CFstate project root in CFSTATE_ENV. This audit creates a new
results/phase8 directory and downloads only the ~540 KB GEO processed-count
archive (not SRA reads). Original phase4/phase5/phase7 files are read-only.

Use --self-test to run parser and numerical unit tests without project files.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import math
import re
import sys
import tarfile
import tempfile
import time
from collections import Counter
from pathlib import Path
from urllib.parse import unquote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
P4 = ROOT / 'results/phase4'
OUT = ROOT / 'results/phase8/audit_08a_frozen_CF_biology_GSE280261_preflight'
RAW = ROOT / 'data/raw/GSE280261_public_processed'
GEO_TAR = 'https://ftp.ncbi.nlm.nih.gov/geo/series/GSE280nnn/GSE280261/suppl/GSE280261_RAW.tar'
TAR_PATH = RAW / 'GSE280261_RAW.tar'
STATUS = 'CFSTATE_PHASE8_08A_FROZEN_BIOLOGY_AND_PUBLIC_GEO_PREFLIGHT_REVIEW_REQUIRED'

FILES = {
    'mask': P4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv',
    'modulome': ROOT / 'external/aeruprecise/precise411/paeru_116_iModulon.json',
    'calibration': P4 / 'audit_03e_softcore_projection_fidelity/softcore_to_full_activity_calibration.csv',
    'stable': P4 / 'audit_03e_softcore_projection_fidelity/PRJEB24688_empirically_stable_primary_programmes.csv',
    'candidate': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/FROZEN_CF_airway_candidate_vectors.csv',
    'registry18': P4 / 'audit_03h_freeze_replicated_CF_airway_candidate/conservatively_replicated_programmes.csv',
    'gff': ROOT / 'references/phase6/PAO1_GCF_000006765.1_ASM676v1_genomic.gff.gz',
    'script': Path(__file__).resolve(),
}

# Hypothesis-generating text categories fixed BEFORE inspecting the output.
# Their enrichment cannot establish an experimentally verified mechanism.
THEMES = {
    'oxygen_denitrification': r'denitrif|nitrate|nitrite|anaerob|nitric oxide|nitrous oxide|\bnar[a-z0-9]*\b|\bnir[a-z0-9]*\b|\bnor[a-z0-9]*\b|\banr\b|\bdnr\b',
    'iron_siderophore_haem': r'\biron\b|siderophore|pyoverdin|pyochelin|\bha?em(e)?\b|ferric|ferrous|\bpvd[a-z0-9]*\b|\bpch[a-z0-9]*\b|\bfurs?\b',
    'efflux_and_drug_resistance': r'efflux|antibiotic resistance|drug resistance|multidrug|aminoglycoside|\bmex[a-z0-9]*\b|\bmexr\b',
    'alginate_Pel_Psl_biofilm': r'biofilm|alginate|exopolysaccharide|polysaccharide biosynthesis|\balg[a-z0-9]*\b|\bpel[a-z0-9]*\b|\bpsl[a-z0-9]*\b',
    'motility_and_chemotaxis': r'flagell|chemotax|motility|twitch|type iv pilus|\bpil[a-z0-9]*\b|\bfli[a-z0-9]*\b|\bflg[a-z0-9]*\b',
    'quorum_phenazines': r'quorum|phenazine|pyocyanin|rhamnolipid|\b(?:las|rhl|pqs|phz)[a-z0-9]+\b',
    'stress_and_DNA_repair': r'oxidative stress|superoxide|peroxide|chaperon|heat shock|dna repair|\blexa\b|\breca\b|\bkat[a-z0-9]*\b',
    'ribosome_and_translation': r'ribosom|translation|aminoacyl.trna|trna synthetase|elongation factor|\brpl[a-z0-9]*\b|\brps[a-z0-9]*\b',
    'glycerol_and_carbon_utilisation': r'glycerol|carbon catabol|glycolysis|gluconeogen|\bglp[a-z0-9]*\b',
    'sulfur_and_amino_acids': r'\bsulf(?:ur|ate|ide)\b|cysteine|methionine|transsulfuration|aminotransferase',
}


def check(name, condition):
    if not bool(condition):
        print(name + ' FAIL', flush=True)
        raise RuntimeError('CHECK FAILED: ' + name)
    print(name + ' PASS', flush=True)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_gff(path):
    """Read PAO1 locus annotations without treating ambiguous symbols as IDs."""
    records = {}
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if not line or line.startswith('#'):
                continue
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 9 or parts[2] not in ('gene', 'CDS'):
                continue
            attrs = {}
            for bit in parts[8].split(';'):
                if '=' in bit:
                    k, v = bit.split('=', 1)
                    attrs[k] = unquote(v)
            locus = attrs.get('locus_tag', '') or attrs.get('old_locus_tag', '')
            if not re.fullmatch(r'PA\d{4}(?:\.\d+)?', locus, flags=re.I):
                continue
            locus = locus[:6].upper()
            entry = {
                'GeneID': locus,
                'gene_symbol': attrs.get('gene', ''),
                'product': attrs.get('product', ''),
                'feature': parts[2],
            }
            old = records.get(locus)
            if old is None or (parts[2] == 'CDS' and old['feature'] != 'CDS'):
                records[locus] = entry
            elif old is not None and not old['product'] and entry['product']:
                records[locus] = entry
    if not records:
        raise RuntimeError('No PAO1 GFF gene IDs found')
    return pd.DataFrame(records.values()).set_index('GeneID')


def load_m(path, genes, names):
    obj = json.loads(path.read_text())
    check('modulome_has_M', isinstance(obj, dict) and 'M' in obj)
    raw = obj['M']
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, dict) and {'index', 'columns', 'data'}.issubset(raw):
        m = pd.DataFrame(raw['data'], index=raw['index'], columns=raw['columns'])
    else:
        m = pd.DataFrame(raw)
    m.index = m.index.map(str)
    m.columns = m.columns.map(str)
    if set(genes).issubset(m.columns) and set(names).issubset(m.index):
        m = m.T
    check('M_identifiers', m.index.is_unique and m.columns.is_unique and
          set(genes).issubset(m.index) and set(names).issubset(m.columns))
    m = m.loc[genes, names].apply(pd.to_numeric, errors='raise')
    check('M_finite', np.isfinite(m.to_numpy(float)).all())
    return m


def hypergeom_enrichment(total, positives, selected, observed):
    """Exact one-sided hypergeometric enrichment, implemented without scipy."""
    if observed <= 0:
        return 1.0
    if positives <= 0:
        return 1.0
    max_k = min(positives, selected)
    lo = max(observed, selected - (total - positives))
    if lo > max_k:
        return 0.0
    def lc(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    denom = lc(total, selected)
    logs = [lc(positives, k) + lc(total - positives, selected - k) - denom
            for k in range(lo, max_k + 1)]
    mx = max(logs)
    return float(min(1.0, math.exp(mx) * sum(math.exp(v - mx) for v in logs)))


def bh_adjust(values):
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    scaled = arr[order] * len(arr) / np.arange(1, len(arr) + 1)
    result = np.empty_like(arr)
    result[order] = np.minimum(1, np.minimum.accumulate(scaled[::-1])[::-1])
    return result


def manifest_for_gsm(number):
    """Eight GEO sample groups x three replicates, verified from GSE280261."""
    idx = number - 8593305
    if not 0 <= idx < 24:
        raise ValueError('Unexpected GSM number: ' + str(number))
    block, within = divmod(idx, 3)
    return {
        'GSM': 'GSM' + str(number),
        'strain': 'PAO1' if block < 4 else 'FRD1',
        'genotype': 'WT' if block % 4 < 2 else 'delta_glpR',
        'tobramycin_5ug_ml': 'NO' if block % 2 == 0 else 'YES',
        'biological_replicate': within + 1,
        'model_relevance': 'CONTROLLED_LAB_PERTURBATION_NOT_INDEPENDENT_CF_PATIENT',
    }


def get_public_tar():
    RAW.mkdir(parents=True, exist_ok=True)
    if TAR_PATH.is_file():
        check('GEO_existing_tar_nonempty', TAR_PATH.stat().st_size > 10000)
        print('Reusing existing GEO processed-count archive:', TAR_PATH, flush=True)
        return TAR_PATH
    print('Downloading PUBLIC processed gene-count archive (~540 KB), NOT raw reads.', flush=True)
    tmp = TAR_PATH.with_suffix('.tar.part')
    try:
        error = None
        for attempt in range(3):
            try:
                request = Request(GEO_TAR, headers={'User-Agent': 'CFstate-audit-08a-public-RNA/1.0'})
                with urlopen(request, timeout=65) as resp, tmp.open('wb') as dst:
                    transferred = 0
                    while True:
                        block = resp.read(1048576)
                        if not block:
                            break
                        transferred += len(block)
                        if transferred > 20 * 1024 * 1024:
                            raise RuntimeError('Unexpectedly large GEO archive: aborted')
                        dst.write(block)
                check('GEO_tar_size', 10000 < transferred < 20 * 1024 * 1024)
                with tarfile.open(tmp, mode='r:*') as tar:
                    check('GEO_tar_readable', len(tar.getmembers()) >= 24)
                tmp.replace(TAR_PATH)
                return TAR_PATH
            except Exception as exc:
                error = exc
                if tmp.exists():
                    tmp.unlink()
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
        raise RuntimeError('Unable to download public GEO processed archive: ' + str(error))
    finally:
        if tmp.exists():
            tmp.unlink()


def inspect_public_counts(archive):
    records = []
    gsm_seen = set()
    with tarfile.open(archive, mode='r:*') as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        for member in members:
            parts = Path(member.name).parts
            if '..' in parts or member.name.startswith('/'):
                raise RuntimeError('Unsafe member in GEO tar: ' + member.name)
            name = Path(member.name).name
            match = re.match(r'^(GSM\d+)_.*\.tabular\.gz$', name, re.I)
            if not match:
                print('  SKIP unknown GEO archive member:', name)
                continue
            gsm = match.group(1).upper()
            if gsm in gsm_seen:
                raise RuntimeError('Duplicate GEO GSM in archive: ' + gsm)
            if member.size > 3 * 1024 * 1024:
                raise RuntimeError('GEO per-sample file exceeds 3 MB: ' + name)
            gsm_seen.add(gsm)
            dst = RAW / name
            if dst.exists():
                if dst.stat().st_size != member.size:
                    raise RuntimeError('Existing GEO file size mismatch, refusing overwrite: ' + str(dst))
            else:
                input_stream = tar.extractfile(member)
                if input_stream is None:
                    raise RuntimeError('Cannot open archive member ' + name)
                with dst.open('xb') as out:
                    out.write(input_stream.read())
            # This is deliberately only schema inspection, NOT featureCounts modelling.
            with gzip.open(dst, 'rt', encoding='utf-8-sig', errors='replace') as source:
                preview = []
                sample_ids = []
                header = ''
                lines = 0
                widths = Counter()
                for line in source:
                    raw = line.rstrip('\n\r')
                    if not raw.strip():
                        continue
                    if not header:
                        header = raw[:200]
                    cols = raw.split('\t')
                    widths[len(cols)] += 1
                    lines += 1
                    if len(preview) < 5:
                        preview.append(raw[:180])
                    if len(sample_ids) < 30:
                        sample_ids.append(cols[0])
            rec = manifest_for_gsm(int(gsm[3:]))
            rec.update({
                'processed_file': name,
                'processed_file_sha256': sha256(dst),
                'nonempty_lines': lines,
                'modal_tab_fields': widths.most_common(1)[0][0] if widths else 0,
                'first_line': header,
                'first_column_preview': '|'.join(sample_ids[:12]),
                'first_rows_preview': ' || '.join(preview[:3]),
                'direct_PAO1_tag_in_first_30': sum(bool(re.search(r'\bPA\d{4}\b', x)) for x in sample_ids),
            })
            records.append(rec)
    frame = pd.DataFrame(records).sort_values('GSM').reset_index(drop=True)
    check('GEO_all_24_GSM', len(frame) == 24 and len(gsm_seen) == 24 and
          set(gsm_seen) == {'GSM' + str(i) for i in range(8593305, 8593329)})
    groups = frame.groupby(['strain', 'genotype', 'tobramycin_5ug_ml']).size()
    check('GEO_2x2x2_three_replicates', len(groups) == 8 and (groups == 3).all())
    check('GEO_each_file_nonempty', (frame.nonempty_lines > 1000).all())
    return frame


def self_test():
    assert np.allclose(bh_adjust([0.01, 0.5, 0.02]), [0.03, 0.5, 0.03])
    assert abs(hypergeom_enrichment(20, 5, 4, 3) - 0.03199174406604747) < 1e-8
    assert len({tuple(manifest_for_gsm(k)[x] for x in
                ('strain', 'genotype', 'tobramycin_5ug_ml'))
                for k in range(8593305, 8593329)}) == 8
    assert manifest_for_gsm(8593328)['biological_replicate'] == 3
    # A miniature GFF annotation, including a CDS preferred over gene.
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / 'toy.gff.gz'
        with gzip.open(p, 'wt') as f:
            f.write('##gff-version 3\n')
            f.write('NC_002516.2\tRefSeq\tgene\t10\t99\t.\t+\t.\tID=g1;locus_tag=PA0001;gene=glpR\n')
            f.write('NC_002516.2\tRefSeq\tCDS\t10\t99\t.\t+\t0\tID=c1;locus_tag=PA0001;gene=glpR;product=glycerol metabolism regulator\n')
        anno = read_gff(p)
        assert anno.loc['PA0001', 'product'] == 'glycerol metabolism regulator'

        # Exercise the actual GEO tar extraction/manifest parser using 24 toy gz files.
        global RAW
        old_raw = RAW
        try:
            RAW = Path(td) / 'processed'
            RAW.mkdir()
            archive = Path(td) / 'test_GEO.tar'
            with tarfile.open(archive, 'w') as tar:
                for number in range(8593305, 8593329):
                    sample = 'GSM' + str(number)
                    name = sample + '_toy.tabular.gz'
                    payload = ('GeneID\tCount\n' +
                               ''.join('PA' + str(i).zfill(4) + '\t' + str(i) + '\n'
                                       for i in range(1100))).encode()
                    zipped = gzip.compress(payload)
                    info = tarfile.TarInfo(name)
                    info.size = len(zipped)
                    tar.addfile(info, io.BytesIO(zipped))
            preflight = inspect_public_counts(archive)
            assert len(preflight) == 24 and preflight['nonempty_lines'].min() == 1101
        finally:
            RAW = old_raw
    print('AUDIT_08A_SELF_TEST_PASS')


def main():
    print('=' * 78)
    print('CF-State Phase 8 — Audit 08a: frozen CF biology + GSE280261 preflight')
    print('=' * 78, flush=True)
    check('working_directory_has_CFstate', (ROOT / 'results/phase4').is_dir())
    check('new_output_directory', not OUT.exists())
    for key, path in FILES.items():
        check('input_' + key, path.is_file())
    mask = pd.read_csv(FILES['mask'])['GeneID'].astype(str).tolist()
    check('mask_4923_unique', len(mask) == 4923 and len(set(mask)) == 4923)
    cal = pd.read_csv(FILES['calibration']).set_index('imodulon')
    model_names = cal.index.astype(str).tolist()
    stable = pd.read_csv(FILES['stable'])['imodulon'].astype(str).tolist()
    check('116_108_frozen', len(model_names) == 116 and len(stable) == 108 and
          len(set(stable)) == 108 and set(stable).issubset(model_names))
    slopes = pd.to_numeric(cal.loc[stable, 'calibration_slope'], errors='raise').to_numpy(float)
    check('calibration_valid', np.isfinite(slopes).all() and np.all(np.abs(slopes) > 1e-12))
    registry18 = pd.read_csv(FILES['registry18'])
    check('registry_18', 'imodulon' in registry18.columns and len(registry18) == 18)
    chosen18 = registry18['imodulon'].astype(str).tolist()
    check('18_unique_stable', len(set(chosen18)) == 18 and set(chosen18).issubset(stable))
    cand = pd.read_csv(FILES['candidate'], index_col=0)
    cand.index = cand.index.astype(str).str.strip()
    check('candidate_frozen_108', cand.index.is_unique and set(cand.index) == set(stable) and
          'two_cohort_consensus_unit' in cand.columns)
    c = pd.to_numeric(cand.loc[stable, 'two_cohort_consensus_unit'], errors='raise').to_numpy(float)
    check('candidate_norm1', np.isfinite(c).all() and np.isclose(np.linalg.norm(c), 1, atol=1e-6))
    m = load_m(FILES['modulome'], mask, model_names)
    annot = read_gff(FILES['gff']).reindex(mask)
    check('annotation_coverage_ge4400', int(annot['product'].notna().sum()) >= 4400)
    annot['gene_symbol'] = annot['gene_symbol'].fillna('')
    annot['product'] = annot['product'].fillna('')
    print('RefSeq annotated genes:', int((annot['product'] != '').sum()), '/4923')

    # IMPORTANT: c is in CALIBRATED programme activity coordinates.
    # A least-squares linear inverse-loading fingerprint in gene space is
    # M[:,stable] @ (c / slopes[stable]). It is neither observed LFC nor
    # an estimate of CF-specific causal gene effects.
    effects = pd.Series(c / slopes, index=stable, name='inverse_calibration_coefficient')
    proxy = pd.Series(m.loc[:, stable].to_numpy(float) @ effects.to_numpy(float),
                      index=mask, name='linear_inverse_loading_fingerprint')
    check('gene_proxy_valid', np.isfinite(proxy.to_numpy()).all() and proxy.abs().max() > 0)

    programmes = pd.DataFrame({
        'imodulon': stable,
        'frozen_candidate_weight': c,
        'absolute_weight': np.abs(c),
        'squared_weight_fraction': c ** 2,
        'selected_in_frozen_18': [name in chosen18 for name in stable],
        'inverse_calibration_gene_coefficient': effects.to_numpy(float),
    }).sort_values('absolute_weight', ascending=False)
    programme_top = []
    for imodulon in chosen18:
        value = float(c[stable.index(imodulon)])
        contribution = m[imodulon] * float(effects[imodulon])
        for direction in ('positive', 'negative'):
            leading = contribution.nlargest(12) if direction == 'positive' else contribution.nsmallest(12)
            for rank, (gene, weight) in enumerate(leading.items(), 1):
                programme_top.append({
                    'imodulon': imodulon,
                    'frozen_candidate_weight': value,
                    'contribution_direction': direction,
                    'rank': rank,
                    'GeneID': gene,
                    'contribution_to_inverse_loading_fingerprint': float(weight),
                    'gene_symbol': annot.loc[gene, 'gene_symbol'],
                    'product': annot.loc[gene, 'product'],
                })

    genes = annot.copy()
    genes['linear_inverse_loading_fingerprint'] = proxy
    genes['absolute_fingerprint'] = proxy.abs()
    top_genes = []
    top_n = 150
    for direction in ('positive', 'negative'):
        selection = proxy.nlargest(top_n) if direction == 'positive' else proxy.nsmallest(top_n)
        for rank, (gene, value) in enumerate(selection.items(), 1):
            top_genes.append({
                'direction': direction, 'rank': rank, 'GeneID': gene,
                'gene_symbol': annot.loc[gene, 'gene_symbol'],
                'product': annot.loc[gene, 'product'],
                'inverse_loading_fingerprint': float(value),
            })

    text_field = (annot['gene_symbol'].astype(str) + ' ' + annot['product'].astype(str)).str.lower()
    all_theme_hits = {}
    for label, regex in THEMES.items():
        # Text annotations (not curated pathways); keyword matching may be imperfect.
        all_theme_hits[label] = text_field.str.contains(regex, regex=True, case=False).astype(bool)
    reports = []
    for direction in ('positive', 'negative'):
        selected = proxy.nlargest(top_n).index if direction == 'positive' else proxy.nsmallest(top_n).index
        for label, matching in all_theme_hits.items():
            n_all = int(matching.sum())
            n_sel = int(matching.loc[selected].sum())
            reports.append({
                'direction': direction, 'theme': label,
                'background_genes': len(mask), 'genes_annotated_with_theme': n_all,
                'selected_top150_genes': top_n, 'selected_matching_genes': n_sel,
                'fraction_selected': n_sel / top_n,
                'fraction_background': n_all / len(mask),
                'hypergeom_p_one_sided': hypergeom_enrichment(len(mask), n_all, top_n, n_sel),
            })
    themes = pd.DataFrame(reports)
    themes['exploratory_BH_q'] = bh_adjust(themes['hypergeom_p_one_sided'])
    themes = themes.sort_values(['exploratory_BH_q', 'hypergeom_p_one_sided'])

    OUT.mkdir(parents=True, exist_ok=False)
    programmes.to_csv(OUT / 'FROZEN_108_programme_signed_contributions.csv', index=False)
    pd.DataFrame(programme_top).to_csv(OUT / 'DESCRIPTIVE_frozen18_top_gene_contributions.csv', index=False)
    genes.to_csv(OUT / 'EXPLORATORY_gene_space_inverse_loading_fingerprint_4923.csv', index_label='GeneID')
    pd.DataFrame(top_genes).to_csv(OUT / 'DESCRIPTIVE_top150_positive_negative_gene_fingerprint.csv', index=False)
    themes.to_csv(OUT / 'EXPLORATORY_annotated_theme_enrichment.csv', index=False)

    print('\n========== FROZEN CF SIGNATURE: TOP PROGRAMMES ==========')
    print(programmes.head(18)[['imodulon', 'frozen_candidate_weight',
                               'squared_weight_fraction', 'selected_in_frozen_18']].round(4).to_string(index=False))
    print('\n========== EXPLORATORY BIOLOGICAL ANNOTATION ==========')
    show = ['direction', 'theme', 'selected_matching_genes', 'genes_annotated_with_theme',
            'hypergeom_p_one_sided', 'exploratory_BH_q']
    print(themes[show].head(16).round(5).to_string(index=False))
    print('\nBiological annotation is descriptive / hypothesis-generating, not causal or CF-specific validation.')

    (OUT / 'PROVENANCE_INPUT_SHA256.csv').write_text(
        pd.DataFrame([{'input': key, 'relative_path': str(path.relative_to(ROOT)),
                       'sha256': sha256(path)} for key, path in FILES.items()]).to_csv(index=False)
    )
    (OUT / 'INTERPRETATION_LIMITS.txt').write_text(
        'Phase 8 Audit 08a: no model refitting. Original phase4/phase5/phase7 frozen files unchanged.\n'
        'Original frozen candidate is 108-dimensional; 18 previously selected replicated programmes are flagged, not reselected.\n'
        'RefSeq product keywords are NOT curated mechanistic/pathway validation. FDR assumes a simplified hypergeometric null.\n'
        'Gene fingerprint is M[:,stable] @ (frozen_consensus / calibration_slope): an inverse-loading proxy, NOT measured differential expression.\n'
        'Signs refer to this frozen reference orientation, not inherently stimulatory/inhibitory regulation.\n'
        'The GSE280261 design contains FRD1 and PAO1, each with WT/Delta-glpR and +/- tobramycin, n=3 per cell.\n'
        'These are two strain backgrounds in culture, not clinical CF and nonCF patient cohorts.\n'
        'Time to mid-exponential differs between strain/treatment groups (see GEO), confounding cross-strain mean contrasts.\n'
        'Processed-data download is only schema/inventory, not yet batch/strain-safe gene expression projection.\n'
    )

    download_ok = False
    try:
        archive = get_public_tar()
        manifest = inspect_public_counts(archive)
        manifest.to_csv(OUT / 'GSE280261_FROZEN_24_processed_sample_manifest_and_schema.csv', index=False)
        (OUT / 'GSE280261_PUBLIC_ARCHIVE_SHA256.txt').write_text(
            'URL: ' + GEO_TAR + '\nSHA256: ' + sha256(archive) + '\n'
        )
        download_ok = True
        print('\n========== GSE280261 PUBLIC PROCESSED-COUNT PREFLIGHT ==========')
        print(manifest.groupby(['strain', 'genotype', 'tobramycin_5ug_ml']).size().to_string())
        print('First two processed count-file previews:')
        print(manifest[['GSM', 'processed_file', 'nonempty_lines',
                        'modal_tab_fields', 'first_column_preview']].head(2).to_string(index=False))
        print('24 public processed samples retrieved. No SRA raw reads downloaded.')
    except Exception as exc:
        # Preserve the separately valid frozen-signature biological interpretation;
        # do not turn a transient GEO network or file issue into false results.
        (OUT / 'GSE280261_PUBLIC_PREFLIGHT_INCOMPLETE.txt').write_text(
            str(exc) + '\nNo GSE280261 projection performed.\n'
        )
        print('GSE280261 PREFLIGHT INCOMPLETE:', str(exc), flush=True)

    (OUT / 'AUDIT_08A_STATUS.txt').write_text(STATUS + '\n')
    print('\n========== AUDIT 08a SUMMARY ==========')
    print('108 frozen programmes, 18 original candidate programmes flagged, 4923 gene fingerprints annotated.')
    print('Processed GEO dataset:', '24_SAMPLES_PREFLIGHT_COMPLETE' if download_ok else 'DOWNLOAD_OR_SCHEMA_REVIEW_REQUIRED')
    print('NEXT: inspect GEO gene identifiers and replicate/strain QC before any programme-activity contrasts.')
    print('STATUS:', STATUS, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == '--self-test':
        self_test()
    elif len(sys.argv) != 1:
        raise SystemExit('Usage: python audit_08a_CFstate_biology_and_GSE280261_preflight.py [--self-test]')
    else:
        main()
