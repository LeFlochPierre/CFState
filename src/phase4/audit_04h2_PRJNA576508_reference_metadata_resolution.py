from pathlib import Path
from io import StringIO
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import pandas as pd


ROOT = Path.cwd()

MANIFEST = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "PRJNA576508_full_ENA_manifest.csv"
)

SUPPLEMENT = (
    ROOT
    / "data/raw/phase4/PRJNA576508/published_supplement"
    / "mbio.03042-19-sd001.xlsx"
)

OUT = (
    ROOT
    / "results/phase4/audit_04h2_PRJNA576508_reference_metadata_resolution"
)

RUNS = [
    "SRR10267767",
    "SRR10267768",
    "SRR10267775",
    "SRR10267776",
]

SEARCH_TERMS = [
    "SRR10267767",
    "SRR10267768",
    "SRR10267775",
    "SRR10267776",
    "SAMN12995559",
    "SAMN12995558",
    "SAMN12995543",
    "SAMN12995542",
    "RSV",
    "CFBE",
    "airway epithelial",
    "BEGM",
    "SingleQuots",
]


def fetch_ena_run(run):
    base = "https://www.ebi.ac.uk/ena/portal/api/filereport"

    fields = [
        "run_accession",
        "sample_accession",
        "sample_title",
        "experiment_accession",
        "experiment_title",
        "study_accession",
        "study_title",
        "scientific_name",
        "library_name",
        "library_strategy",
        "library_source",
        "library_selection",
        "library_layout",
        "instrument_model",
    ]

    query = urlencode(
        {
            "accession": run,
            "result": "read_run",
            "fields": ",".join(fields),
            "format": "tsv",
        }
    )

    url = base + "?" + query

    request = Request(
        url,
        headers={
            "User-Agent":
                "CF-State metadata audit"
        },
    )

    try:
        with urlopen(
            request,
            timeout=60,
        ) as response:
            text = response.read().decode(
                "utf-8"
            )

        table = pd.read_csv(
            StringIO(text),
            sep="\t",
        )

        return table

    except Exception as error:
        return pd.DataFrame(
            [
                {
                    "run_accession":
                        run,

                    "ENA_query_error":
                        (
                            type(error).__name__
                            + ": "
                            + str(error)
                        ),
                }
            ]
        )


def normalize_text(value):
    return (
        str(value)
        .strip()
        .lower()
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04h2")
    print("Resolve ambiguous Cornforth CFBE/RSV reference metadata")
    print("=" * 78)

    for path in [
        MANIFEST,
        SUPPLEMENT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Local frozen manifest
    # ----------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST
    )

    local = manifest.loc[
        manifest[
            "run_accession"
        ]
        .astype(str)
        .isin(
            RUNS
        )
    ].copy()

    local = local.sort_values(
        "run_accession"
    ).reset_index(
        drop=True
    )

    print()
    print("Local frozen manifest rows")

    columns_to_show = [
        column
        for column in local.columns
        if (
            column
            in [
                "run_accession",
                "sample_accession",
                "sample_title",
                "experiment_accession",
                "experiment_title",
                "study_accession",
                "study_title",
                "scientific_name",
                "library_name",
                "library_strategy",
                "library_source",
                "library_selection",
                "library_layout",
                "instrument_model",
            ]
        )
    ]

    print(
        local[
            columns_to_show
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 2. Explicit title logic
    # ----------------------------------------------------------

    title_rows = []

    for _, row in local.iterrows():
        run = str(
            row[
                "run_accession"
            ]
        )

        title = str(
            row[
                "sample_title"
            ]
        )

        lower = normalize_text(
            title
        )

        contains_rsv_infected_phrase = (
            "rsv-infected"
            in lower
        )

        contains_without_rsv_phrase = (
            "without rsv infection"
            in lower
        )

        contradictory = (
            contains_rsv_infected_phrase
            and contains_without_rsv_phrase
        )

        title_rows.append(
            {
                "run_accession":
                    run,

                "sample_title":
                    title,

                "contains_RSV_infected_phrase":
                    contains_rsv_infected_phrase,

                "contains_without_RSV_phrase":
                    contains_without_rsv_phrase,

                "internally_contradictory_title":
                    contradictory,
            }
        )

    title_audit = pd.DataFrame(
        title_rows
    )

    print()
    print("Explicit title audit")
    print(
        title_audit.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 3. Query ENA live metadata
    # ----------------------------------------------------------

    ena_tables = []

    for run in RUNS:
        table = fetch_ena_run(
            run
        )

        ena_tables.append(
            table
        )

    ena = pd.concat(
        ena_tables,
        ignore_index=True,
        sort=False,
    )

    print()
    print("ENA live metadata")
    print(
        ena.fillna(
            ""
        ).to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. Search every sheet of original Data Set S1
    # ----------------------------------------------------------

    workbook = pd.ExcelFile(
        SUPPLEMENT
    )

    supplement_hits = []

    print()
    print("Original Data Set S1 sheets")
    print(
        workbook.sheet_names
    )

    for sheet in workbook.sheet_names:
        table = pd.read_excel(
            SUPPLEMENT,
            sheet_name=sheet,
            dtype=str,
        )

        table = table.fillna(
            ""
        )

        for index, row in table.iterrows():
            row_values = [
                str(value)
                for value in row.tolist()
            ]

            combined = " | ".join(
                row_values
            )

            lower = combined.lower()

            matched_terms = [
                term
                for term in SEARCH_TERMS
                if term.lower()
                in lower
            ]

            if not matched_terms:
                continue

            supplement_hits.append(
                {
                    "sheet":
                        sheet,

                    "excel_row_approx":
                        int(index)
                        + 2,

                    "matched_terms":
                        "; ".join(
                            matched_terms
                        ),

                    "row_text":
                        combined,
                }
            )

    supplement_hits = pd.DataFrame(
        supplement_hits
    )

    print()
    print("Supplement metadata hits")

    if supplement_hits.empty:
        print(
            "  none"
        )

    else:
        with pd.option_context(
            "display.max_colwidth",
            250,
        ):
            print(
                supplement_hits.to_string(
                    index=False
                )
            )

    # ----------------------------------------------------------
    # 5. Conservative interpretation
    #
    # We do NOT infer that "RSV-infected" wins over the explicit
    # parenthetical "(without RSV infection)".
    #
    # Until metadata proves otherwise, runs 67/68 are marked
    # ambiguous rather than RSV-positive.
    # ----------------------------------------------------------

    resolution_rows = []

    for _, row in title_audit.iterrows():
        run = row[
            "run_accession"
        ]

        if row[
            "internally_contradictory_title"
        ]:
            resolved = (
                "AMBIGUOUS_CONTROL_FROM_RSV_EXPERIMENT"
            )

        elif row[
            "contains_without_RSV_phrase"
        ]:
            resolved = (
                "NO_RSV"
            )

        elif row[
            "contains_RSV_infected_phrase"
        ]:
            resolved = (
                "RSV_EXPOSED"
            )

        else:
            resolved = (
                "UNRESOLVED"
            )

        resolution_rows.append(
            {
                "run_accession":
                    run,

                "conservative_metadata_class":
                    resolved,
            }
        )

    resolution = pd.DataFrame(
        resolution_rows
    )

    print()
    print("Conservative metadata classification")
    print(
        resolution.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Checks
    # ----------------------------------------------------------

    run67_68 = title_audit.loc[
        title_audit[
            "run_accession"
        ].isin(
            [
                "SRR10267767",
                "SRR10267768",
            ]
        )
    ]

    run75_76 = title_audit.loc[
        title_audit[
            "run_accession"
        ].isin(
            [
                "SRR10267775",
                "SRR10267776",
            ]
        )
    ]

    checks = {
        "all_4_reference_runs_recovered":
            len(
                local
            )
            == 4,

        "all_4_titles_explicitly_say_without_RSV_infection":
            bool(
                title_audit[
                    "contains_without_RSV_phrase"
                ].all()
            ),

        "SRR10267767_68_titles_are_internally_contradictory":
            bool(
                run67_68[
                    "internally_contradictory_title"
                ].all()
            ),

        "SRR10267775_76_titles_are_not_internally_contradictory":
            bool(
                (
                    ~run75_76[
                        "internally_contradictory_title"
                    ]
                ).all()
            ),

        "original_supplement_searched":
            len(
                workbook.sheet_names
            )
            >= 3,

        "no_expression_or_CF_candidate_used":
            True,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04h2 checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    if passed:
        status = (
            "PRJNA576508_CFBE_REFERENCE_METADATA_AMBIGUITY_DOCUMENTED"
        )

    else:
        status = (
            "PRJNA576508_CFBE_REFERENCE_METADATA_REVIEW_REQUIRED"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: SRR10267767/68 must NOT be labelled as "
        "RSV-exposed solely because their titles contain the phrase "
        "'RSV-infected'. The same titles explicitly state "
        "'without RSV infection'."
    )

    print()
    print(
        "No expression values, iModulon activities, or Audit-03h "
        "CF-airway candidate values were used."
    )

    # ----------------------------------------------------------
    # 7. Save
    # ----------------------------------------------------------

    local.to_csv(
        OUT
        / "four_reference_runs_local_metadata.csv",
        index=False,
    )

    ena.to_csv(
        OUT
        / "four_reference_runs_ENA_live_metadata.csv",
        index=False,
    )

    title_audit.to_csv(
        OUT
        / "four_reference_runs_title_logic.csv",
        index=False,
    )

    supplement_hits.to_csv(
        OUT
        / "DataSetS1_reference_metadata_hits.csv",
        index=False,
    )

    resolution.to_csv(
        OUT
        / "conservative_reference_metadata_classification.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "check":
                    key,

                "pass":
                    value,
            }
            for key, value
            in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    (
        OUT
        / "AUDIT_04H2_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
