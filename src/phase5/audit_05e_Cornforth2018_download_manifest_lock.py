from pathlib import Path
import hashlib
import traceback
import re

import pandas as pd

ROOT = Path.cwd()
BASE = ROOT / "results/phase5"

C2 = BASE / "audit_05c2_Cornforth2018_metadata_fields_repair"
D = BASE / "audit_05d_Cornforth2018_sample_mapping"
OUT = BASE / "audit_05e_Cornforth2018_download_manifest_lock"

ENA = C2 / "SOURCE_ENA_SRP135669_read_run.tsv"
MAPPING = D / "FROZEN_15_published_specimen_mapping_audit.csv"

STATUS = "CFSTATE_PHASE5_05E_CLINICAL_MAPPING_AND_DOWNLOAD_PLAN_LOCKED"


def check(label, condition):
    passed = bool(condition)
    print(label, "PASS" if passed else "FAIL", flush=True)
    if not passed:
        raise RuntimeError("Failed check: " + label)


def normalise(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def split_field(value):
    return [
        part.strip()
        for part in str(value).split(";")
        if part.strip()
    ]


def main():
    print("=" * 74)
    print("CF-State Phase 5 — Audit 05e")
    print("Cornforth 2018 publication-to-ENA mapping lock")
    print("=" * 74)

    inputs = {
        "05c2_status": C2 / "AUDIT_05C2_STATUS.txt",
        "05d_status": D / "AUDIT_05D_STATUS.txt",
        "ENA_manifest": ENA,
        "05d_mapping": MAPPING,
        "script": Path(__file__)
    }

    for label, path in inputs.items():
        check("input_" + label, path.is_file())

    check(
        "05c2_complete",
        inputs["05c2_status"].read_text().strip()
        == "CFSTATE_PHASE5_05C2_NEGATIVE_CONTROL_METADATA_PREFLIGHT_COMPLETE"
    )

    check(
        "05d_complete",
        inputs["05d_status"].read_text().strip()
        == "CFSTATE_PHASE5_05D_SAMPLE_MAPPING_DIAGNOSTIC_FROZEN"
    )

    mapping = pd.read_csv(MAPPING, dtype=str).fillna("")
    ena = pd.read_csv(ENA, sep="\t", dtype=str).fillna("")

    check(
        "15_unique_published_specimens",
        len(mapping) == 15
        and mapping["published_sample"].is_unique
    )

    check(
        "ENA_unique_run_accessions",
        ena["run_accession"].is_unique
    )

    required = [
        "run_accession",
        "sample_accession",
        "fastq_ftp",
        "fastq_bytes",
        "fastq_md5"
    ]

    check(
        "required_ENA_fields",
        set(required).issubset(ena.columns)
    )

    ena = ena.set_index("run_accession")

    records = []
    files = []

    print()
    print("========== PUBLICATION-TO-ENA MAPPING ==========")

    for row in mapping.to_dict("records"):
        name = row["published_sample"]
        runs = split_field(row["candidate_runs"])
        samples = split_field(
            row["candidate_sample_accessions"]
        )

        check(
            name + "_one_run_one_sample",
            len(runs) == 1
            and len(samples) == 1
            and row["mapping_status"]
            == "PROVISIONAL_UNIQUE_SAMPLE_ACCESSION"
        )

        run = runs[0]

        check(
            name + "_run_in_archive",
            run in ena.index
        )

        record = ena.loc[run]

        check(
            name + "_biosample_matches",
            record["sample_accession"] == samples[0]
        )

        name_fields = [
            field for field in [
                "sample_alias",
                "sample_title",
                "library_name",
                "experiment_title"
            ]
            if field in record.index
        ]

        evidence = [
            field
            for field in name_fields
            if normalise(name) in normalise(record[field])
        ]

        check(
            name + "_archival_name_evidence",
            len(evidence) >= 1
        )

        urls = split_field(record["fastq_ftp"])
        sizes = split_field(record["fastq_bytes"])
        checksums = split_field(record["fastq_md5"])

        check(
            name + "_FASTQ_metadata_complete",
            len(urls) >= 1
            and len(urls) == len(sizes)
            and len(urls) == len(checksums)
        )

        try:
            numeric_sizes = [int(size) for size in sizes]
        except ValueError:
            raise RuntimeError(
                "Invalid FASTQ size for " + name
            )

        check(
            name + "_FASTQ_sizes_positive",
            all(size > 0 for size in numeric_sizes)
        )

        check(
            name + "_FASTQ_md5_valid",
            all(
                re.fullmatch(r"[0-9a-fA-F]{32}", md5)
                for md5 in checksums
            )
        )

        check(
            name + "_FASTQ_urls_unique",
            len(urls) == len(set(urls))
        )

        is_low = (
            row["analysis_role"] == "low_count_separate"
        )

        if name.startswith("Human_Sputum_"):
            group = "CF_sputum"
        elif is_low:
            group = "nonCF_wound_low_coverage"
        elif "Burn" in name:
            group = "nonCF_burn"
        else:
            group = "nonCF_chronic_wound"

        donor = (
            "shared_Lubbock_burn_patient"
            if "Lubbock_Burn_2_" in name
            else "distinct_patient_" + name
        )

        role = (
            "coverage_limited_supplementary_2"
            if is_low
            else "main_clinical_13"
        )

        read_count = int(
            row["published_PAO1_ortholog_mapped_reads"]
        )

        records.append({
            "published_sample": name,
            "group": group,
            "analysis_role": role,
            "run_accession": run,
            "sample_accession": samples[0],
            "published_mapped_PAO1_ortholog_reads":
                read_count,
            "published_donor_group": donor,
            "FASTQ_n_files": len(urls),
            "FASTQ_total_bytes": sum(numeric_sizes),
            "FASTQ_GB": sum(numeric_sizes) / 1e9,
            "archive_evidence_fields": ";".join(evidence),
            "mapping_evidence":
                "PUBLICATION_NAME_AND_ENA_ALIAS_CONCORDANT"
        })

        for index, (url, size, md5) in enumerate(
            zip(urls, numeric_sizes, checksums), start=1
        ):
            files.append({
                "published_sample": name,
                "group": group,
                "analysis_role": role,
                "run_accession": run,
                "FASTQ_index": index,
                "fastq_ftp": url,
                "fastq_https":
                    url if url.startswith("https://")
                    else "https://" + url.lstrip("/"),
                "expected_bytes": size,
                "expected_md5": md5
            })

    manifest = pd.DataFrame(records)
    fastqs = pd.DataFrame(files)

    check(
        "15_distinct_runs",
        manifest["run_accession"].nunique() == 15
    )

    check(
        "15_distinct_biosamples",
        manifest["sample_accession"].nunique() == 15
    )

    main = manifest.loc[
        manifest["analysis_role"] == "main_clinical_13"
    ].copy()

    supplementary = manifest.loc[
        manifest["analysis_role"]
        == "coverage_limited_supplementary_2"
    ].copy()

    check(
        "main_13_expected_groups",
        len(main) == 13
        and (main["group"] == "CF_sputum").sum() == 7
        and (main["group"] == "nonCF_chronic_wound").sum() == 4
        and (main["group"] == "nonCF_burn").sum() == 2
    )

    check(
        "supplementary_two_low_coverage",
        len(supplementary) == 2
        and (
            supplementary["group"]
            == "nonCF_wound_low_coverage"
        ).all()
    )

    check(
        "main_12_distinct_donor_groups",
        main["published_donor_group"].nunique() == 12
    )

    print()
    print("========== LOCKED MAIN CLINICAL DESIGN ==========")

    print(
        main[
            [
                "published_sample",
                "group",
                "run_accession",
                "sample_accession",
                "published_mapped_PAO1_ortholog_reads",
                "FASTQ_GB"
            ]
        ].round(3).to_string(index=False)
    )

    print()
    print("========== DOWNLOAD FEASIBILITY ==========")

    print("Main clinical specimens:", len(main))
    print("Main unique patients:", 12)
    print(
        "Main compressed FASTQ size (GB):",
        round(float(main["FASTQ_GB"].sum()), 3)
    )
    print(
        "Additional low-coverage specimens (GB):",
        round(
            float(supplementary["FASTQ_GB"].sum()), 3
        )
    )
    print(
        "Total available clinical specimens:", len(manifest)
    )
    print(
        "Total clinical FASTQ files:", len(fastqs)
    )

    print()
    print("========== WRITE FROZEN MANIFESTS ==========")

    manifest.to_csv(
        OUT / "FROZEN_15_clinical_archival_mapping.csv",
        index=False
    )

    main.to_csv(
        OUT / "FROZEN_13_main_clinical_sample_manifest.csv",
        index=False
    )

    supplementary.to_csv(
        OUT / "FROZEN_2_low_coverage_exploratory_samples.csv",
        index=False
    )

    fastqs.to_csv(
        OUT / "FROZEN_individual_FASTQ_download_manifest.csv",
        index=False
    )

    pd.DataFrame([
        {
            "input": key,
            "path": str(path),
            "sha256": digest(path)
        }
        for key, path in inputs.items()
    ]).to_csv(
        OUT / "input_sha256_provenance.csv",
        index=False
    )

    notes = """
CF-State Phase 5 — Cornforth 2018 specificity challenge.

Source: Cornforth et al., PNAS 2018.
DOI: 10.1073/pnas.1717525115.
Study: SRP135669.

Publication-to-ENA sample names and accessions are concordant.
Supplementary workbook cross-check is not yet performed.
Mapping is operationally locked for download planning.

Primary biological groups:
7 CF sputum specimens from Denmark.
4 chronic-wound specimens from Texas.
2 burn specimens from Texas (one patient).

Total primary: 13 specimens from 12 patients.
Two Danish chronic-wound samples are retained as
predefined low-coverage supplementary material.

CRITICAL LIMITATION:
In the main 13-sample comparison, infection site and
geographical recruitment site are confounded.
Differences cannot independently establish CF specificity.

Technical requirements before biological comparison:
1. Verify FASTQ integrity against archived MD5 checksums.
2. Characterise library layout, adapters and read quality.
3. Use a common and documented bacterial mapping strategy.
4. Assess PAO1 ortholog and 4923-gene mask coverage.
5. Preserve all samples in feasibility reporting.
6. Reuse frozen 108-programme projection definitions,
   frozen controlled-state axes and frozen 03h candidate.
7. Compare clinical groups with a common reference.
8. Report low-depth and strain-mapping limitations.
9. Do not infer 13 statistically independent patients.
10. No new programmes or candidate thresholds selected.

This audit downloads no sequencing reads.
"""

    (
        OUT / "FROZEN_PREPROCESSING_AND_DESIGN_CONSTRAINTS.txt"
    ).write_text(notes.strip() + "\n")

    (
        OUT / "AUDIT_05E_STATUS.txt"
    ).write_text(STATUS + "\n")

    print()
    print("=" * 74)
    print("STATUS:", STATUS)
    print("=" * 74)


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 05e already exists; refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()
    except Exception:
        (
            OUT / "AUDIT_05E_STATUS.txt"
        ).write_text(
            "CFSTATE_PHASE5_05E_REVIEW_REQUIRED\n"
        )
        (
            OUT / "AUDIT_05E_FAILURE.txt"
        ).write_text(traceback.format_exc())
        raise
