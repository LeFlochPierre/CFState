from pathlib import Path
from io import BytesIO
from urllib.parse import urlencode
from urllib.request import urlopen, Request
import hashlib
import json
import re
import traceback

import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/phase5/audit_05c_Cornforth2018_specificity_preflight"

ACCESSION = "SRP135669"
API = "https://www.ebi.ac.uk/ena/portal/api/filereport"

# Labels and roles taken from the published human-sample table.
# These are NOT yet mapped to run accessions.
PUBLISHED = [
    ("Human_Sputum_A", "CF_sputum_Denmark", "primary", ""),
    ("Human_Sputum_B", "CF_sputum_Denmark", "primary", ""),
    ("Human_Sputum_C", "CF_sputum_Denmark", "primary", ""),
    ("Human_Sputum_D", "CF_sputum_Denmark", "primary", ""),
    ("Human_Sputum_E", "CF_sputum_Denmark", "primary", ""),
    ("Human_Sputum_F", "CF_sputum_Denmark", "primary", ""),
    ("Human_Sputum_G", "CF_sputum_Denmark", "primary", ""),
    ("Lubbock_CW_15", "nonCF_chronic_wound_US", "primary", ""),
    ("Lubbock_CW_16", "nonCF_chronic_wound_US", "primary", ""),
    ("Lubbock_CW_18", "nonCF_chronic_wound_US", "primary", ""),
    ("Lubbock_CW_21", "nonCF_chronic_wound_US", "primary", ""),
    ("Lubbock_Burn_2_3", "nonCF_burn_US", "primary", "shared_burn_patient"),
    ("Lubbock_Burn_2_4", "nonCF_burn_US", "primary", "shared_burn_patient"),
    ("Danish_Chronic_Wound_1", "nonCF_chronic_wound_Denmark",
     "low_count_separate", ""),
    ("Danish_Chronic_Wound_2", "nonCF_chronic_wound_Denmark",
     "low_count_separate", ""),
]


def clean(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def main():
    if OUT.exists():
        raise RuntimeError(
            "Audit 05c output exists. Refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        print("=" * 72)
        print("CF-State Phase 5 — Audit 05c")
        print("Cornforth 2018 non-CF infection specificity preflight")
        print("=" * 72)

        expected = pd.DataFrame(
            PUBLISHED,
            columns=[
                "published_sample",
                "infection_context",
                "published_analysis_role",
                "donor_dependency_note"
            ]
        )

        expected.to_csv(
            OUT / "FROZEN_published_2018_sample_plan.csv",
            index=False
        )

        params = urlencode({
            "accession": ACCESSION,
            "result": "read_run",
            "format": "tsv"
        })

        url = API + "?" + params

        print()
        print("Query:", ACCESSION)
        print("Downloading metadata only — NO FASTQ files")

        request = Request(
            url,
            headers={
                "User-Agent": "CF-State-audit05c-metadata/1.0"
            }
        )

        with urlopen(request, timeout=45) as response:
            content = response.read(10_000_001)

        if len(content) > 10_000_000:
            raise RuntimeError(
                "Unexpected metadata response larger than 10 MB"
            )

        if (
            not content.strip()
            or content.lstrip().lower().startswith(b"<html")
        ):
            raise RuntimeError(
                "ENA did not return a TSV manifest"
            )

        manifest = pd.read_csv(
            BytesIO(content),
            sep="\t",
            dtype=str
        ).fillna("")

        required = {"run_accession", "sample_accession"}

        if not required.issubset(manifest.columns):
            raise RuntimeError(
                "ENA manifest missing required columns: "
                + str(manifest.columns.tolist())
            )

        if manifest.empty:
            raise RuntimeError("ENA returned no runs")

        if manifest["run_accession"].duplicated().any():
            raise RuntimeError("Duplicate ENA run accession")

        (
            OUT / "SOURCE_ENA_SRP135669_read_run.tsv"
        ).write_bytes(content)

        name_fields = [
            c for c in manifest.columns
            if any(
                token in c.lower()
                for token in [
                    "sample_alias",
                    "sample_title",
                    "experiment_title",
                    "library_name",
                    "sample_description",
                    "sample_name"
                ]
            )
        ]

        matches = []

        # Text matches are evidence to inspect, never
        # automatically accepted sample mappings.
        for _, record in manifest.iterrows():
            for _, published in expected.iterrows():
                evidence = [
                    column for column in name_fields
                    if clean(published["published_sample"])
                    in clean(record[column])
                ]

                if evidence:
                    matches.append({
                        "published_sample": published["published_sample"],
                        "infection_context": published["infection_context"],
                        "run_accession": record["run_accession"],
                        "sample_accession": record["sample_accession"],
                        "matched_fields": ";".join(evidence),
                        "mapping_status":
                            "PROVISIONAL_TEXT_MATCH_NOT_VERIFIED"
                    })

        candidates = pd.DataFrame(
            matches,
            columns=[
                "published_sample",
                "infection_context",
                "run_accession",
                "sample_accession",
                "matched_fields",
                "mapping_status"
            ]
        )

        candidates.to_csv(
            OUT / "PROVISIONAL_run_to_published_sample_matches.csv",
            index=False
        )

        display_columns = [
            column for column in [
                "run_accession",
                "sample_accession",
                "sample_alias",
                "sample_title",
                "experiment_title",
                "library_name",
                "library_layout",
                "read_count",
                "base_count",
                "fastq_bytes"
            ]
            if column in manifest.columns
        ]

        print()
        print("========== DATASET METADATA ==========")

        print("ENA run records:", len(manifest))
        print("Descriptive fields:", name_fields)
        print("Provisional mappings:", len(candidates))
        print("Published specimens:", len(expected))

        print()
        print("========== FIRST 25 ENA RECORDS ==========")

        print(
            manifest[display_columns].head(25).to_string(
                index=False,
                max_colwidth=45
            )
        )

        print()
        print("========== PROVISIONAL SAMPLE MAPPING ==========")

        if candidates.empty:
            print(
                "No exact specimen-name matches identified. "
                "Published supplementary mapping is required."
            )
        else:
            print(candidates.to_string(index=False))

        unmatched = sorted(
            set(expected["published_sample"])
            - set(candidates["published_sample"])
        )

        print()
        print("========== UNMATCHED PUBLISHED SPECIMENS ==========")

        print(
            ", ".join(unmatched)
            if unmatched else "None"
        )

        summary = {
            "accession": ACCESSION,
            "source_publication_doi": "10.1073/pnas.1717525115",
            "ena_request_url": url,
            "number_of_runs": len(manifest),
            "provisional_matches": len(candidates),
            "unmatched_published_specimens": unmatched,
            "raw_reads_downloaded": False,
            "mapping_verified": False,
            "ena_metadata_sha256":
                hashlib.sha256(content).hexdigest()
        }

        (
            OUT / "preflight_summary.json"
        ).write_text(json.dumps(summary, indent=2))

        status = (
            "CFSTATE_PHASE5_05C_"
            "NEGATIVE_CONTROL_METADATA_PREFLIGHT_COMPLETE"
        )

        (
            OUT / "AUDIT_05C_STATUS.txt"
        ).write_text(status + "\n")

        print()
        print("STATUS:", status)
        print("Run-to-specimen mapping remains unverified.")
        print("No raw reads downloaded.")

    except Exception:
        (
            OUT / "AUDIT_05C_STATUS.txt"
        ).write_text(
            "CFSTATE_PHASE5_05C_REVIEW_REQUIRED\n"
        )

        (
            OUT / "AUDIT_05C_FAILURE.txt"
        ).write_text(traceback.format_exc())

        raise


if __name__ == "__main__":
    main()
