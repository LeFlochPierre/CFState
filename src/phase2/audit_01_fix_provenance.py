from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path.cwd()
SAMPLE_TABLE = ROOT / "external" / "aeruprecise" / "precise411" / "sample_table.csv"
OUT = ROOT / "results" / "phase2" / "audit_01_aeruprecise_resource_freeze"

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


def clean(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().strip('"').strip("'")


def main() -> None:
    if not SAMPLE_TABLE.exists():
        raise FileNotFoundError(
            f"Missing frozen sample table: {SAMPLE_TABLE}\n"
            "Run Phase 2 Audit 01 first."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    # Use Python's CSV reader directly so identifier matching is not affected
    # by pandas dtype inference, missing-value conversion, or index handling.
    with SAMPLE_TABLE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    print("=" * 78)
    print("CF-State Phase 2 - Audit 01 provenance correction")
    print("=" * 78)
    print(f"Frozen sample-table rows: {len(rows)}")
    print(f"Columns: {len(fieldnames)}")

    results = []
    detail_rows = []

    for accession, cfstate_role in CFSTATE_STUDIES:
        exact_hits = []
        substring_hits = []

        for row_i, row in enumerate(rows, start=1):
            values = {key: clean(value) for key, value in row.items()}

            exact_fields = [
                key for key, value in values.items()
                if value == accession
            ]
            substring_fields = [
                key for key, value in values.items()
                if accession in value and value != accession
            ]

            if exact_fields:
                exact_hits.append((row_i, row, exact_fields))
            elif substring_fields:
                substring_hits.append((row_i, row, substring_fields))

        n_exact = len(exact_hits)
        n_substring = len(substring_hits)
        training_overlap = n_exact > 0

        if cfstate_role == "related_duplicate_excluded":
            status = "NOT_INDEPENDENT_CFSTATE_DUPLICATE"
        elif training_overlap:
            status = "TRAINING_OVERLAP_INTERNAL_ONLY"
        elif cfstate_role == "specificity_control":
            status = "CANDIDATE_EXTERNAL_CONTROL"
        else:
            status = "CANDIDATE_EXTERNAL"

        matched_fields = sorted(
            {
                field
                for _, _, fields in exact_hits
                for field in fields
            }
        )

        results.append(
            {
                "accession": accession,
                "cfstate_role": cfstate_role,
                "precise411_exact_accession_match": training_overlap,
                "n_exact_matching_rows": n_exact,
                "n_substring_only_rows": n_substring,
                "exact_matched_fields": ";".join(matched_fields),
                "phase2_status": status,
                "caveat": (
                    "Confirmed exact identifier overlap with frozen "
                    "aeruPRECISE-411 sample metadata."
                    if training_overlap
                    else
                    "No exact identifier match in frozen sample table. "
                    "Absence here is not yet a complete BioProject/run/alias-level "
                    "proof of independence."
                ),
            }
        )

        for row_i, row, fields in exact_hits:
            detail_rows.append(
                {
                    "queried_accession": accession,
                    "csv_row_number": row_i,
                    "matched_fields": ";".join(fields),
                    "Run": clean(row.get("Run")),
                    "SRAStudy": clean(row.get("SRAStudy")),
                    "BioProject": clean(row.get("BioProject")),
                    "BioSample": clean(row.get("BioSample")),
                    "GEO Series": clean(row.get("GEO Series")),
                    "GEO Sample": clean(row.get("GEO Sample")),
                    "biosample_strain": clean(row.get("biosample_strain")),
                    "strain_final": clean(row.get("strain_final")),
                    "biosample_genotype": clean(row.get("biosample_genotype")),
                    "project": clean(row.get("project")),
                    "condition": clean(row.get("condition")),
                    "sample": clean(row.get("sample")),
                }
            )

        print(
            f"{accession:12s} exact={n_exact:2d} "
            f"substring_only={n_substring:2d} status={status}"
        )

    result_df = pd.DataFrame(results)
    detail_df = pd.DataFrame(detail_rows)

    overlap_path = OUT / "cfstate_aeruprecise411_overlap.csv"
    details_path = OUT / "cfstate_aeruprecise411_overlap_details.csv"

    result_df.to_csv(overlap_path, index=False)
    detail_df.to_csv(details_path, index=False)

    # Hard scientific sanity check: the frozen published sample table is known
    # to contain retained GSE136111 profiles. If this fails, stop rather than
    # silently carrying a provenance error into later audits.
    gse = result_df.loc[result_df["accession"] == "GSE136111"].iloc[0]
    if int(gse["n_exact_matching_rows"]) != 5:
        raise RuntimeError(
            "Expected exactly 5 retained GSE136111 rows in the frozen "
            f"aeruPRECISE-411 sample table, found "
            f"{int(gse['n_exact_matching_rows'])}. "
            "Do not continue to Audit 02 until this is understood."
        )

    gse_detail = detail_df.loc[
        detail_df["queried_accession"] == "GSE136111"
    ].copy()

    condition_counts = (
        gse_detail["condition"]
        .value_counts(dropna=False)
        .rename_axis("condition")
        .reset_index(name="n")
    )
    condition_counts.to_csv(
        OUT / "GSE136111_training_overlap_conditions.csv",
        index=False,
    )

    correction = [
        "# CF-State Phase 2 - Audit 01 provenance correction",
        "",
        "The original Audit 01 pandas-based accession screen produced a false "
        "negative for GSE136111. This correction uses direct CSV identifier "
        "matching against the already frozen aeruPRECISE-411 sample table.",
        "",
        "## Corrected result",
        f"- Frozen sample-table rows inspected: {len(rows)}",
        "- GSE136111 exact matching rows: 5",
        "- GSE136111 status: `TRAINING_OVERLAP_INTERNAL_ONLY`",
        "- The five retained profiles comprise three planktonic and two "
        "biofilm total-RNA profiles in the frozen metadata.",
        "",
        "## Consequence",
        "GSE136111 must not be treated as independent external validation of "
        "aeruPRECISE-411. The other current CF-State accessions remain only "
        "candidate external datasets until deeper alias/BioProject/run-level "
        "provenance checks are completed.",
    ]

    (OUT / "AUDIT_01_PROVENANCE_CORRECTION.md").write_text(
        "\n".join(correction) + "\n",
        encoding="utf-8",
    )

    print()
    print("PASS: GSE136111 recovered as 5 retained training-overlap profiles.")
    print(f"Updated: {overlap_path}")
    print(f"Updated: {details_path}")
    print(
        f"Written: {OUT / 'AUDIT_01_PROVENANCE_CORRECTION.md'}"
    )


if __name__ == "__main__":
    main()
