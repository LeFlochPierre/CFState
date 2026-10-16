#!/usr/bin/env python3
"""CF-State Phase 7 Audit 07b: offline triage of frozen 07a run and GEO metadata.

No network calls, no read downloads, no patient IDs inferred.
Run from the CFstate repository root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path('results/phase7/audit_07a_external_CF_RNA_metadata_census')
DEFAULT_OUTPUT = Path('results/phase7/audit_07b_candidate_RNA_run_triage')
EXPECTED_STATUS = 'CFSTATE_PHASE7_07A_METADATA_CENSUS_COMPLETE_REVIEW_REQUIRED'
STATUS = 'CFSTATE_PHASE7_07B_CANDIDATE_RNA_METADATA_TRIAGE_COMPLETE_REVIEW_REQUIRED'


def short(value, maximum=65):
    value = str(value or '').replace('\n', ' ').strip()
    return value if len(value) <= maximum else value[:maximum - 3] + '...'


def fastq_bytes_total(value):
    values = [piece.strip() for piece in str(value or '').split(';') if piece.strip()]
    try:
        return sum(int(piece) for piece in values) if values else None
    except ValueError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    src, out = args.source, args.output
    if out.exists():
        raise RuntimeError('Immutable Audit 07b output already exists: ' + str(out))
    if (src / 'AUDIT_07A_STATUS.txt').read_text().strip() != EXPECTED_STATUS:
        raise RuntimeError('07a metadata census did not pass expected retrieval state')

    runfile = src / 'FROZEN_ENA_all_run_metadata.csv'
    geofile = src / 'FROZEN_GSE139943_sample_metadata.csv'
    runs = pd.read_csv(runfile, dtype=str, keep_default_na=False)
    geo = pd.read_csv(geofile, dtype=str, keep_default_na=False)

    required = ['requested_accession','run_accession','sample_accession',
                'library_strategy','library_source','metadata_RNA_class',
                'potential_overlap_existing_05e']
    missing = [col for col in required if col not in runs.columns]
    if missing:
        raise RuntimeError('Missing ENA metadata fields: ' + repr(missing))
    if 'GEO_accession' not in geo.columns:
        raise RuntimeError('GEO metadata lacks GSM accession')
    if runs.run_accession.duplicated().any():
        raise RuntimeError('Duplicate run accession in frozen 07a inventory')
    if geo.GEO_accession.duplicated().any():
        raise RuntimeError('Duplicate GSM accession')
    if set(runs.requested_accession) != {'SRP173673','PRJNA71831'}:
        raise RuntimeError('Unexpected studies in frozen ENA metadata')

    # Compact but lossless study+run review: retain raw metadata in source.
    review = pd.DataFrame({
        'study': runs.requested_accession,
        'run': runs.run_accession,
        'ENA_sample_accession': runs.sample_accession,
        'experiment': runs.get('experiment_accession', pd.Series('', index=runs.index)),
        'RNA_class_from_07a': runs.metadata_RNA_class,
        'strategy': runs.library_strategy,
        'source': runs.library_source,
        'selection': runs.get('library_selection', pd.Series('', index=runs.index)),
        'layout': runs.get('library_layout', pd.Series('', index=runs.index)),
        'read_count': pd.to_numeric(runs.get('read_count', pd.Series('', index=runs.index)), errors='coerce'),
        'fastq_bytes': runs.get('fastq_bytes', pd.Series('', index=runs.index)),
        'sample_title': runs.get('sample_title', pd.Series('', index=runs.index)),
        'sample_alias': runs.get('sample_alias', pd.Series('', index=runs.index)),
        'experiment_title': runs.get('experiment_title', pd.Series('', index=runs.index)),
        'library_name': runs.get('library_name', pd.Series('', index=runs.index)),
        'fastq_ftp': runs.get('fastq_ftp', pd.Series('', index=runs.index)),
        'exact_run_overlap_05e': runs.potential_overlap_existing_05e,
    })
    review['FASTQ_GB_from_ENA'] = review.fastq_bytes.map(fastq_bytes_total).apply(
        lambda val: round(val / 1e9, 3) if val is not None else None
    )
    review['preliminary_priority'] = review.RNA_class_from_07a.map({
        'CANDIDATE_RNA_REVIEW_SPECIES_AND_COVERAGE': 'PILOT_CANDIDATE_CHECK_BACTERIAL_RNA',
        'POSSIBLE_RNA_MANUAL_REVIEW': 'MANUAL_ASSAY_REVIEW',
        'LIKELY_DNA_NOT_EXPRESSION': 'NOT_RNA_CANDIDATE',
    }).fillna('MANUAL_ASSAY_REVIEW')
    review['independent_donor_established'] = 'NO'
    review['P_aeruginosa_mRNA_coverage_established'] = 'NO'

    assay_summary = (
        review.groupby(['study','RNA_class_from_07a','strategy','source'],
                       dropna=False).size().reset_index(name='n_runs')
    )
    n_candidates = int((review.preliminary_priority ==
                        'PILOT_CANDIDATE_CHECK_BACTERIAL_RNA').sum())
    # Fail loudly if this is not the user's already-frozen 07a inventory.
    if len(review) != 52 or n_candidates != 14 or len(geo) != 9:
        raise RuntimeError(
            f'Unexpected metadata totals: ENA runs={len(review)}, '
            f'RNA candidates={n_candidates}, GEO samples={len(geo)}'
        )

    out.mkdir(parents=True, exist_ok=False)
    review.to_csv(out / 'FROZEN_52_run_assay_review.csv', index=False)
    assay_summary.to_csv(out / 'FROZEN_study_assay_summary.csv', index=False)
    review[review.preliminary_priority != 'NOT_RNA_CANDIDATE'].to_csv(
        out / 'FROZEN_RNA_and_ambiguous_run_shortlist.csv', index=False)
    geo.to_csv(out / 'FROZEN_GSE139943_GSM_manual_review.csv', index=False)
    provenance = {}
    for name, path in [('07a_runs',runfile),('07a_GEO',geofile),
                       ('07a_status',src/'AUDIT_07A_STATUS.txt')]:
        provenance[name] = {'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / 'INPUT_SHA256.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (out / 'METHODS_AND_LIMITATIONS.txt').write_text(
        'Offline metadata-only review; no sequencing reads inspected or downloaded.\n'
        'Candidate RNA status derives from study library metadata, not bacterial mRNA abundance.\n'
        'ENA distinct sample_accession != verified distinct BioSamples or patients.\n'
        'NCBI PRJNA71831 overview reports 9 linked BioSamples whereas this ENA report\n'
        'shows 31 distinct sample accessions; reconcile namespaces and scopes.\n'
        'Exact-run non-overlap does not establish donor or dataset independence.\n'
        'GSE139943 GEO GSM records have not been linked to RNA FASTQ runs.\n'
        'All candidates require actual P. aeruginosa expression coverage verification.\n'
    )
    (out / 'AUDIT_07B_STATUS.txt').write_text(STATUS + '\n')

    print('===== STUDY × ASSAY CENSUS =====')
    print(assay_summary.to_string(index=False))
    print('\n===== RNA CANDIDATES / AMBIGUOUS RUNS =====')
    shown = review[review.preliminary_priority != 'NOT_RNA_CANDIDATE'].copy()
    for col in ['sample_title','experiment_title','library_name']:
        shown[col] = shown[col].map(lambda value: short(value, 48))
    columns = ['study','run','ENA_sample_accession','strategy','source',
               'selection','read_count','FASTQ_GB_from_ENA','sample_title',
               'library_name','RNA_class_from_07a','exact_run_overlap_05e']
    print(shown[columns].to_string(index=False))
    print('\n===== GEO 9 SAMPLE TITLES =====')
    title_col = 'title' if 'title' in geo else None
    for row in geo.itertuples(index=False):
        title = str(getattr(row, title_col, '')) if title_col else ''
        print(str(row.GEO_accession).ljust(14), short(title, 85))
    print('\nNOTE: all donor identities and P. aeruginosa RNA coverage unverified.')
    print('STATUS:', STATUS)


if __name__ == '__main__':
    main()
