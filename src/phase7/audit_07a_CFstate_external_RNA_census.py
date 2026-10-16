#!/usr/bin/env python3
"""CF-State Phase 7, Audit 07a: metadata-only census of candidate CF RNA studies.

Run from the CFstate project root. Does NOT download sequence reads.
Requires Python 3 standard library only. Immutable output directory.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

OUT = Path("results/phase7/audit_07a_external_CF_RNA_metadata_census")
ENA_API = "https://www.ebi.ac.uk/ena/portal/api"
GEO_ACC = "GSE139943"
STUDIES = ("SRP173673", "PRJNA71831")
PREFERRED_FIELDS = [
    "run_accession", "study_accession", "secondary_study_accession",
    "sample_accession", "secondary_sample_accession", "experiment_accession",
    "sample_title", "sample_alias", "experiment_title", "library_name",
    "library_strategy", "library_source", "library_selection", "library_layout",
    "instrument_platform", "scientific_name", "read_count", "base_count",
    "fastq_bytes", "fastq_ftp",
]
CORE_FIELDS = [
    "run_accession", "sample_accession", "experiment_accession",
    "library_strategy", "library_source", "library_selection", "library_layout",
    "read_count", "fastq_bytes",
]


def fetch(url: str, attempts: int = 3) -> bytes:
    last = None
    for trial in range(attempts):
        try:
            request = Request(url, headers={
                "User-Agent": "CFstate-metadata-audit/07a academic research",
                "Accept": "text/tab-separated-values,text/plain,*/*",
            })
            with urlopen(request, timeout=55) as response:
                return response.read(28 * 1024 * 1024 + 1)
        except (HTTPError, URLError, TimeoutError) as err:
            last = err
            if trial + 1 < attempts:
                time.sleep(2 * (trial + 1))
    raise RuntimeError(f"Cannot fetch {url}: {last}")


def parse_tsv(data: bytes) -> list[dict[str, str]]:
    decoded = data.decode("utf-8-sig", errors="replace")
    if decoded.lstrip().lower().startswith("<html"):
        raise RuntimeError("HTML returned where ENA TSV was expected")
    reader = csv.DictReader(io.StringIO(decoded), delimiter="\t")
    if not reader.fieldnames or "run_accession" not in reader.fieldnames:
        raise RuntimeError(f"Not an ENA run report: {decoded[:250]!r}")
    return [{str(k): str(v or "") for k, v in row.items()}
            for row in reader if row.get("run_accession")]


def classify(row: dict[str, str]) -> str:
    strategy = row.get("library_strategy", "").lower()
    source = row.get("library_source", "").lower()
    selection = row.get("library_selection", "").lower()
    if "metatranscript" in source or "transcriptomic" in source or "rna-seq" in strategy:
        return "CANDIDATE_RNA_REVIEW_SPECIES_AND_COVERAGE"
    if "metagenomic" in source or "genomic" in source or strategy in {
        "wgs", "wga", "amplicon", "targeted-capture", "16s",
    }:
        return "LIKELY_DNA_NOT_EXPRESSION"
    if "cdna" in selection or "rrna" in selection or "rna" in strategy:
        return "POSSIBLE_RNA_MANUAL_REVIEW"
    return "UNKNOWN_ASSAY_MANUAL_REVIEW"


def test_parsers() -> None:
    toy = b"run_accession\tsample_accession\tlibrary_strategy\tlibrary_source\nSRR1\tSAMN1\tRNA-Seq\tMETATRANSCRIPTOMIC\n"
    r = parse_tsv(toy)
    assert len(r) == 1 and r[0]["run_accession"] == "SRR1"
    assert classify(r[0]) == "CANDIDATE_RNA_REVIEW_SPECIES_AND_COVERAGE"
    assert classify({"library_strategy": "WGS", "library_source": "METAGENOMIC"}) == "LIKELY_DNA_NOT_EXPRESSION"
    test_geo = b'!Sample_geo_accession\t"GSM1"\t"GSM2"\n!Sample_title\t"CF sample 1"\t"CF sample 2"\n!series_matrix_table_begin\n'
    rows, meta = parse_geo_header(test_geo)
    assert len(rows) == 2 and rows[1]["GEO_accession"] == "GSM2"
    assert "!Sample_title" in meta
    print("07a parser self-test PASS")


def ena_report(accession: str, available: set[str] | None):
    fields = [x for x in PREFERRED_FIELDS if available is None or x in available]
    if not all(x in fields for x in ("run_accession", "sample_accession")):
        fields = CORE_FIELDS
    query = urlencode({
        "accession": accession, "result": "read_run",
        "fields": ",".join(fields), "format": "tsv",
    })
    url = ENA_API + "/filereport?" + query
    try:
        blob = fetch(url)
        parsed = parse_tsv(blob)
    except Exception as first_err:
        fallback = urlencode({
            "accession": accession, "result": "read_run",
            "fields": ",".join(CORE_FIELDS), "format": "tsv",
        })
        url = ENA_API + "/filereport?" + fallback
        try:
            blob = fetch(url)
            parsed = parse_tsv(blob)
        except Exception as second_err:
            raise RuntimeError(f"preferred: {first_err}; fallback: {second_err}")
    if not parsed:
        raise RuntimeError("ENA returned no run records; inspect accession and availability")
    for row in parsed:
        row["requested_accession"] = accession
        row["metadata_RNA_class"] = classify(row)
        row["potential_overlap_existing_05e"] = "NOT_YET_CHECKED"
        row["patient_ID_confirmed"] = "NO"
        row["usable_PAO1_expression_confirmed"] = "NO"
    return parsed, blob, url


def geo_matrix_url(acc: str) -> str:
    num = int(acc[3:])
    series_group = "GSE" + str(num // 1000) + "nnn"
    return ("https://ftp.ncbi.nlm.nih.gov/geo/series/" + series_group +
            "/" + acc + "/matrix/" + acc + "_series_matrix.txt.gz")


def parse_geo_header(data: bytes) -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    txt = data.decode("utf-8-sig", errors="replace")
    metadata: dict[str, list[str]] = {}
    for line in txt.splitlines():
        if line.startswith("!series_matrix_table_begin") or line.startswith('"ID_REF"'):
            break
        if not line.startswith("!Sample_"):
            continue
        cells = next(csv.reader([line], delimiter="\t"))
        if len(cells) > 1:
            metadata.setdefault(cells[0], []).append(cells[1:])
    ids = metadata.get("!Sample_geo_accession", [[]])[0]
    titles = metadata.get("!Sample_title", [[]])[0]
    rows = []
    for i, acc in enumerate(ids):
        row = {
            "GEO_accession": acc,
            "title": titles[i] if i < len(titles) else "",
            "patient_ID_confirmed": "NO",
            "direct_Pseudomonas_RNA_confirmed": "NO",
        }
        for name, arrays in metadata.items():
            if name in {"!Sample_geo_accession", "!Sample_title"}:
                continue
            values = [arr[i] for arr in arrays if i < len(arr) and arr[i]]
            row[name.removeprefix("!Sample_")] = " | ".join(values)
        rows.append(row)
    return rows, metadata


def read_geo_header(url: str):
    request = Request(url, headers={"User-Agent": "CFstate-07a-research"})
    with urlopen(request, timeout=65) as response:
        # Header only: stop at the expression table. No RNA-seq or large matrix download.
        with gzip.GzipFile(fileobj=response) as zipped:
            chunks = []
            total = 0
            for line in zipped:
                total += len(line)
                if total > 6 * 1024 * 1024:
                    raise RuntimeError("GEO metadata header over 6 MiB; stop safely")
                chunks.append(line)
                if line.startswith(b"!series_matrix_table_begin"):
                    break
    header = b"".join(chunks)
    rows, metadata = parse_geo_header(header)
    if not rows:
        raise RuntimeError("Could not resolve GEO GSM sample list from matrix header")
    return rows, metadata, header


def save_csv(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None):
    fields = fields or sorted(set().union(*(set(row) for row in rows)))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    test_parsers()
    if not Path("src/phase4").exists() or not Path("results/phase5").exists():
        raise RuntimeError("Run from ~/projects/CFstate project root")
    if OUT.exists():
        raise RuntimeError(f"Immutable Audit 07a output exists: {OUT}; do not overwrite")
    errors = []
    payloads = []
    all_runs = []
    summaries = []
    available = None
    try:
        return_fields_url = ENA_API + "/returnFields?result=read_run"
        return_fields = fetch(return_fields_url)
        available = {line.split("\t")[0].strip() for line in return_fields.decode().splitlines()[1:]}
        if "run_accession" not in available:
            available = None
    except Exception as exc:
        errors.append("Optional ENA returnFields lookup failed: " + str(exc))

    for acc in STUDIES:
        print("\nENA METADATA:", acc, flush=True)
        try:
            rows, blob, url = ena_report(acc, available)
            all_runs.extend(rows)
            payloads.append((f"SOURCE_ENA_{acc}.tsv", blob, url))
            n_runs = len({r["run_accession"] for r in rows})
            n_samples = len({r["sample_accession"] for r in rows if r.get("sample_accession")})
            n_rna = sum(r["metadata_RNA_class"] == "CANDIDATE_RNA_REVIEW_SPECIES_AND_COVERAGE" for r in rows)
            summaries.append({"study": acc, "status": "ENA_RETRIEVED", "runs": n_runs,
                              "unique_biosamples": n_samples, "candidate_RNA_runs": n_rna,
                              "independent_patients": "NOT_VERIFIED"})
            print("run records:", n_runs, "unique BioSamples:", n_samples,
                  "RNA metadata candidates:", n_rna, flush=True)
        except Exception as exc:
            err = str(exc)
            errors.append(f"REQUIRED {acc}: {err}")
            summaries.append({"study": acc, "status": "RETRIEVAL_FAILED", "runs": "",
                              "unique_biosamples": "", "candidate_RNA_runs": "",
                              "independent_patients": "NOT_VERIFIED"})
            print("ERROR:", err, flush=True)

    print("\nGEO METADATA:", GEO_ACC, flush=True)
    geo_rows = []
    try:
        url = geo_matrix_url(GEO_ACC)
        geo_rows, geo_meta, header = read_geo_header(url)
        payloads.append((f"SOURCE_GEO_{GEO_ACC}_header.txt", header, url))
        summaries.append({"study": GEO_ACC, "status": "GEO_METADATA_RETRIEVED",
                          "runs": "NOT_ESTABLISHED", "unique_biosamples": len(geo_rows),
                          "candidate_RNA_runs": "NOT_ESTABLISHED", "independent_patients": "1_IN_PUBLISHED_CASE_STUDY"})
        print("GEO samples:", len(geo_rows), flush=True)
    except Exception as exc:
        errors.append(f"REQUIRED {GEO_ACC}: {exc}")
        summaries.append({"study": GEO_ACC, "status": "RETRIEVAL_FAILED",
                          "runs": "", "unique_biosamples": "", "candidate_RNA_runs": "",
                          "independent_patients": "1_IN_PUBLISHED_CASE_STUDY"})
        print("ERROR:", str(exc), flush=True)

    # Compare against existing archived run IDs, but no patient-level linkage can be inferred.
    existing = Path("results/phase5/audit_05e_Cornforth2018_download_manifest_lock/FROZEN_15_clinical_archival_mapping.csv")
    known = set()
    if existing.is_file():
        with existing.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                known.update(re.findall(r"(?:SRR|ERR|DRR)\d+", " ".join(str(x) for x in row.values())))
    for r in all_runs:
        r["potential_overlap_existing_05e"] = "EXACT_RUN_MATCH" if r["run_accession"] in known else "NO_EXACT_RUN_MATCH_CHECKED"

    OUT.mkdir(parents=True, exist_ok=False)
    response_manifest = []
    for name, blob, url in payloads:
        (OUT / name).write_bytes(blob)
        response_manifest.append({"source_file": name, "source_URL": url,
                                  "sha256": hashlib.sha256(blob).hexdigest(),
                                  "size_bytes": str(len(blob))})
    if all_runs:
        save_csv(OUT / "FROZEN_ENA_all_run_metadata.csv", all_runs)
        save_csv(OUT / "DESCRIPTIVE_RNA_candidate_runs.csv",
                 [r for r in all_runs if r["metadata_RNA_class"] != "LIKELY_DNA_NOT_EXPRESSION"])
    if geo_rows:
        save_csv(OUT / f"FROZEN_{GEO_ACC}_sample_metadata.csv", geo_rows)
    save_csv(OUT / "FROZEN_study_level_census.csv", summaries)
    if response_manifest:
        save_csv(OUT / "SOURCE_metadata_sha256.csv", response_manifest)
    (OUT / "AUDIT_07A_ERRORS.txt").write_text("\n".join(errors) + "\n" if errors else "NONE\n")
    substantive_errors = [e for e in errors if e.startswith("REQUIRED")]
    status = ("CFSTATE_PHASE7_07A_METADATA_CENSUS_COMPLETE_REVIEW_REQUIRED"
              if not substantive_errors else
              "CFSTATE_PHASE7_07A_INCOMPLETE_METADATA_RETRIEVAL_REVIEW_REQUIRED")
    (OUT / "AUDIT_07A_STATUS.txt").write_text(status + "\n")
    (OUT / "METHODS_AND_LIMITATIONS.txt").write_text(
        "Audit 07a is metadata-only: no reads downloaded and no alignment performed.\n"
        "Accession/run/BioSample counts are not independent patient counts.\n"
        "ENA library metadata may misclassify legacy RNA/DNA assays; manual review required.\n"
        "GEO matrix header is metadata only. Confirm P. aeruginosa gene-level coverage before inclusion.\n"
        "Published case-study patient counts (GSE139943=one, SRP173673=one) are contextual,\n"
        "not independently resolved from run metadata.\n"
        "Current 05e exact-run overlap does not establish donor independence or lack thereof.\n"
        "Studies explored earlier cannot be declared independent prospective validation.\n"
        "No patient-level modelling or inclusion decisions made.\n"
    )
    print("\n========== AUDIT 07a METADATA CENSUS ==========")
    for row in summaries:
        print(row)
    print("\nMetadata output:", OUT)
    print("STATUS:", status)
    if substantive_errors:
        print("Some required metadata could not be retrieved; inspect AUDIT_07A_ERRORS.txt")


if __name__ == "__main__":
    main()
