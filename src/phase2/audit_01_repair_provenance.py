from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
OUT = ROOT / "results" / "phase2" / "audit_01_aeruprecise_resource_freeze"
SAMPLE_TABLE = ROOT / "external" / "aeruprecise" / "precise411" / "sample_table.csv"

CFSTATE_STUDIES = [
    ("GSE136111", "binary_discovery"),
    ("GSE203348", "binary_discovery"),
    ("GSE87213", "binary_discovery"),
    ("GSE299861", "binary_later"),
    ("GSE191103", "binary_later"),
    ("GSE120602", "binary_later"),
    ("GSE12207", "binary_later"),
    ("GSE223663", "related_duplicate_excluded"),
    ("SRP009257", "specificity_control"),
]

PREFERRED_FIELDS = [
    "GEO Series",
    "SRAStudy",
    "BioProject",
    "Run",
    "GEO Sample",
    "BioSample",
    "SRS_Sample",
    "SRA ID",
]


def norm(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def token_match(value: str, accession: str) -> bool:
    """
    Exact identifier-token matching.
    Avoids broad substring matching while still tolerating cells containing
    multiple IDs separated by punctuation/whitespace.
    """
    value = norm(value)
    if not value:
        return False
    if value == accession:
        return True

    tokens = [
        token
        for token in re.split(r"[\s,;|/]+", value)
        if token
    ]
    return accession in tokens


def main() -> None:
    if not SAMPLE_TABLE.exists():
        raise FileNotFoundError(
            f"Missing frozen sample table: {SAMPLE_TABLE}\n"
            "Run Phase 2 Audit 01 first."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    with SAMPLE_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise RuntimeError("Frozen aeruPRECISE-411 sample table is empty.")

    print("=" * 78)
    print("CF-State Phase 2 - Audit 01 provenance repair")
    print("=" * 78)
    print(f"Rows in frozen sample table: {len(rows)}")
    print(f"Columns: {len(fieldnames)}")

    overlap_rows = []
    detail_rows = []

    for accession, cfstate_role in CFSTATE_STUDIES:
        matched = []

        for row in rows:
            matched_fields = []

            # First search identifier-bearing columns.
            for field in PREFERRED_FIELDS:
                if field in row and token_match(row.get(field, ""), accession):
                    matched_fields.append(field)

            # Defensive fallback: exact-token search across all columns.
            if not matched_fields:
                for field, value in row.items():
                    if token_match(value, accession):
                        matched_fields.append(field)

            if matched_fields:
                matched.append((row, sorted(set(matched_fields))))

        n_matches = len(matched)
        training_overlap = n_matches > 0

        if cfstate_role == "related_duplicate_excluded":
            phase2_status = "NOT_INDEPENDENT_CFSTATE_DUPLICATE"
        elif training_overlap:
            phase2_status = "TRAINING_OVERLAP_INTERNAL_ONLY"
        elif cfstate_role == "specificity_control":
            phase2_status = "CANDIDATE_EXTERNAL_CONTROL"
        else:
            phase2_status = "CANDIDATE_EXTERNAL"

        overlap_rows.append(
            {
                "accession": accession,
                "cfstate_role": cfstate_role,
                "precise411_accession_match": training_overlap,
                "n_matching_rows": n_matches,
                "matched_fields": ";".join(
                    sorted(
                        {
                            field
                            for _, fields in matched
                            for field in fields
                        }
                    )
                ),
                "phase2_status": phase2_status,
                "caveat": (
                    "Confirmed identifier match in frozen aeruPRECISE-411 "
                    "sample metadata"
                    if training_overlap
                    else
                    "No direct identifier-token match in frozen sample table; "
                    "BioProject/run alias checks remain desirable before final "
                    "external-validation claims"
                ),
            }
        )

        for row, matched_fields in matched:
            detail = {
                "queried_accession": accession,
                "matched_fields": ";".join(matched_fields),
            }
            for field in [
                "Run",
                "SRAStudy",
                "BioProject",
                "GEO Series",
                "GEO Sample",
                "BioSample",
                "biosample_strain",
                "strain_final",
                "biosample_genotype",
                "project",
                "condition",
                "refrence",
                "sample",
                "doi",
            ]:
                if field in row:
                    detail[field] = row.get(field, "")
            detail_rows.append(detail)

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(
        OUT / "cfstate_aeruprecise411_overlap.csv",
        index=False,
    )

    details = pd.DataFrame(detail_rows)
    details.to_csv(
        OUT / "cfstate_aeruprecise411_overlap_details.csv",
        index=False,
    )

    # Hard scientific sanity check: this accession is known to be represented
    # in the frozen 411 sample metadata. Never silently report zero again.
    gse136 = overlap.loc[overlap["accession"] == "GSE136111"].iloc[0]
    if int(gse136["n_matching_rows"]) < 5:
        raise RuntimeError(
            "PROVENANCE CHECK FAILED: expected at least 5 retained "
            "GSE136111 profiles in the frozen aeruPRECISE-411 metadata, "
            f"but found {int(gse136['n_matching_rows'])}. "
            "Do not proceed to Audit 02."
        )

    confirmed = overlap.loc[
        overlap["precise411_accession_match"], "accession"
    ].tolist()

    candidate_external = overlap.loc[
        overlap["phase2_status"].isin(
            ["CANDIDATE_EXTERNAL", "CANDIDATE_EXTERNAL_CONTROL"]
        ),
        "accession",
    ].tolist()

    report = [
        "# CF-State Phase 2 - Audit 01 provenance correction",
        "",
        "The original Audit 01 accession-matching routine produced a false "
        "negative. This repair re-scanned the frozen aeruPRECISE-411 sample "
        "metadata using identifier-token matching and added a hard sanity "
        "check for GSE136111.",
        "",
        "## Corrected result",
        f"- Confirmed training-overlap accessions: "
        f"{', '.join(confirmed) if confirmed else 'none'}",
        f"- GSE136111 retained profiles found: "
        f"{int(gse136['n_matching_rows'])}",
        f"- Candidate external accessions after this screen: "
        f"{', '.join(candidate_external) if candidate_external else 'none'}",
        "",
        "## Phase-2 rule",
        "- Any accession represented in aeruPRECISE-411 is internal/"
        "training-overlap evidence and cannot be counted as independent "
        "external validation of the frozen representation.",
        "- Absence from this exact-accession/token screen is necessary but "
        "not by itself sufficient proof of total independence; later "
        "provenance work can additionally compare BioProject/SRR/sample "
        "aliases where needed.",
        "",
        "## Audit status",
        "- Audit 01 is considered PASS only if GSE136111 is detected.",
        "- Proceed to Audit 02 only after this script finishes successfully.",
    ]

    (OUT / "AUDIT_01_PROVENANCE_CORRECTION.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print()
    print("Corrected provenance screen:")
    print(overlap.to_string(index=False))
    print()
    print(
        "Confirmed training overlap: "
        + (", ".join(confirmed) if confirmed else "none")
    )
    print(
        f"GSE136111 retained profiles detected: "
        f"{int(gse136['n_matching_rows'])}"
    )
    print()
    print("Audit 01 provenance repair: PASS")
    print(
        "Updated: "
        + str(OUT / "cfstate_aeruprecise411_overlap.csv")
    )
    print(
        "Updated: "
        + str(OUT / "cfstate_aeruprecise411_overlap_details.csv")
    )
    print(
        "Created: "
        + str(OUT / "AUDIT_01_PROVENANCE_CORRECTION.md")
    )


if __name__ == "__main__":
    main()
