from pathlib import Path
from urllib.request import urlopen, Request
from urllib.parse import urlencode
from io import StringIO
import json
import re

import pandas as pd


ROOT = Path.cwd()

OUT = (
    ROOT
    / "results/phase4/audit_03a_PRJEB24688_replication_preflight"
)

ENA_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport"

ZENODO_API = "https://zenodo.org/api/records/1162703"


KNOWN_NON_RRNA_DEPLETED = [
    "P24M1_S1",
    "P30M0_S1",
    "P11F2",
]


def fetch_text(url):
    request = Request(
        url,
        headers={
            "User-Agent":
                "CF-State Phase4 dataset audit"
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as handle:
        return handle.read().decode(
            "utf-8"
        )


def fetch_json(url):
    return json.loads(
        fetch_text(
            url
        )
    )


def extract_patient(value):
    text = str(
        value
    )

    match = re.search(
        r"(P\d+[MF]\d+)",
        text,
        flags=re.IGNORECASE,
    )

    if match is None:
        return None

    return match.group(
        1
    ).upper()


def looks_like_invivo(value):
    text = str(
        value
    )

    if re.search(
        r"P\d+[MF]\d+_S\d+",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    if re.search(
        r"\bP11F2\b",
        text,
        flags=re.IGNORECASE,
    ):
        return True

    return False


def looks_like_invitro(value):
    text = str(
        value
    ).lower()

    patterns = [
        "_24h",
        "_48h",
        "stationary",
        "_stat",
        "exponential",
        "_exp",
        "pa14",
    ]

    return any(
        pattern in text
        for pattern in patterns
    )


def pilot_status(value):
    text = str(
        value
    )

    for sample in KNOWN_NON_RRNA_DEPLETED:
        if sample.lower() in text.lower():
            return "NON_RRNA_DEPLETED_PILOT"

    return "STANDARD_OR_UNKNOWN"


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 03a")
    print("PRJEB24688 multi-patient CF sputum replication preflight")
    print("=" * 78)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. ENA run-level project metadata
    # ----------------------------------------------------------

    fields = [
        "run_accession",
        "experiment_accession",
        "sample_accession",
        "sample_title",
        "scientific_name",
        "library_strategy",
        "library_source",
        "library_selection",
        "library_layout",
        "instrument_model",
        "read_count",
        "base_count",
        "fastq_ftp",
        "fastq_md5",
        "fastq_bytes",
    ]

    params = {
        "accession":
            "PRJEB24688",
        "result":
            "read_run",
        "fields":
            ",".join(
                fields
            ),
        "format":
            "tsv",
        "download":
            "true",
    }

    ena_query = (
        ENA_URL
        + "?"
        + urlencode(
            params
        )
    )

    print()
    print("Retrieving ENA project metadata")

    ena_text = fetch_text(
        ena_query
    )

    ena = pd.read_csv(
        StringIO(
            ena_text
        ),
        sep="\t",
    )

    if ena.empty:
        raise RuntimeError(
            "ENA returned no runs for PRJEB24688."
        )

    print(
        "  runs:",
        len(
            ena
        ),
    )

    print(
        "  columns:",
        len(
            ena.columns
        ),
    )

    # ----------------------------------------------------------
    # 2. Derive human-readable sample structure
    # ----------------------------------------------------------

    ena[
        "sample_title"
    ] = (
        ena[
            "sample_title"
        ]
        .fillna("")
        .astype(str)
    )

    ena[
        "patient_guess"
    ] = ena[
        "sample_title"
    ].map(
        extract_patient
    )

    ena[
        "looks_invivo"
    ] = ena[
        "sample_title"
    ].map(
        looks_like_invivo
    )

    ena[
        "looks_invitro"
    ] = ena[
        "sample_title"
    ].map(
        looks_like_invitro
    )

    ena[
        "pilot_status"
    ] = ena[
        "sample_title"
    ].map(
        pilot_status
    )

    def classify(row):
        if (
            row[
                "looks_invivo"
            ]
            and not row[
                "looks_invitro"
            ]
        ):
            return "IN_VIVO_CANDIDATE"

        if row[
            "looks_invitro"
        ]:
            return "IN_VITRO_CANDIDATE"

        return "UNRESOLVED"

    ena[
        "sample_class_guess"
    ] = ena.apply(
        classify,
        axis=1,
    )

    # ----------------------------------------------------------
    # 3. Convert numeric metadata
    # ----------------------------------------------------------

    for column in [
        "read_count",
        "base_count",
    ]:
        ena[
            column
        ] = pd.to_numeric(
            ena[
                column
            ],
            errors="coerce",
        )

    # fastq_bytes may contain semicolon-separated values.
    def total_fastq_bytes(value):
        if pd.isna(
            value
        ):
            return 0

        total = 0

        for item in str(
            value
        ).split(
            ";"
        ):
            item = item.strip()

            if not item:
                continue

            try:
                total += int(
                    item
                )
            except ValueError:
                pass

        return total

    ena[
        "total_fastq_bytes"
    ] = ena[
        "fastq_bytes"
    ].map(
        total_fastq_bytes
    )

    ena[
        "total_fastq_GB"
    ] = (
        ena[
            "total_fastq_bytes"
        ]
        / 1e9
    )

    # ----------------------------------------------------------
    # 4. Print project manifest
    # ----------------------------------------------------------

    display_columns = [
        "run_accession",
        "sample_accession",
        "sample_title",
        "patient_guess",
        "sample_class_guess",
        "pilot_status",
        "library_selection",
        "library_layout",
        "read_count",
        "total_fastq_GB",
    ]

    print()
    print("ENA run manifest")
    print(
        ena[
            display_columns
        ]
        .sort_values(
            [
                "patient_guess",
                "sample_title",
                "run_accession",
            ],
            na_position="last",
        )
        .round(
            {
                "total_fastq_GB":
                    3
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Project-level summary
    # ----------------------------------------------------------

    patient_ids = sorted(
        {
            patient
            for patient in ena[
                "patient_guess"
            ]
            if isinstance(
                patient,
                str
            )
            and patient
        }
    )

    invivo = ena.loc[
        ena[
            "sample_class_guess"
        ]
        == "IN_VIVO_CANDIDATE"
    ].copy()

    invitro = ena.loc[
        ena[
            "sample_class_guess"
        ]
        == "IN_VITRO_CANDIDATE"
    ].copy()

    unresolved = ena.loc[
        ena[
            "sample_class_guess"
        ]
        == "UNRESOLVED"
    ].copy()

    pilot = ena.loc[
        ena[
            "pilot_status"
        ]
        == "NON_RRNA_DEPLETED_PILOT"
    ].copy()

    standard_invivo = invivo.loc[
        invivo[
            "pilot_status"
        ]
        != "NON_RRNA_DEPLETED_PILOT"
    ].copy()

    print()
    print("Project structure")
    print(
        "  inferred patient IDs:",
        len(
            patient_ids
        ),
        patient_ids,
    )
    print(
        "  in-vivo candidate runs:",
        len(
            invivo
        ),
    )
    print(
        "  standard/rRNA-depleted in-vivo candidate runs:",
        len(
            standard_invivo
        ),
    )
    print(
        "  pilot/non-rRNA-depleted candidate runs:",
        len(
            pilot
        ),
    )
    print(
        "  in-vitro candidate runs:",
        len(
            invitro
        ),
    )
    print(
        "  unresolved runs:",
        len(
            unresolved
        ),
    )

    print()
    print(
        "  total FASTQ volume:",
        round(
            float(
                ena[
                    "total_fastq_GB"
                ].sum()
            ),
            2,
        ),
        "GB",
    )

    print(
        "  candidate standard in-vivo FASTQ volume:",
        round(
            float(
                standard_invivo[
                    "total_fastq_GB"
                ].sum()
            ),
            2,
        ),
        "GB",
    )

    # ----------------------------------------------------------
    # 6. Technical/library metadata
    # ----------------------------------------------------------

    print()
    print("Library selections")
    print(
        ena[
            "library_selection"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("Library layouts")
    print(
        ena[
            "library_layout"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("Instrument models")
    print(
        ena[
            "instrument_model"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 7. Zenodo analysis repository
    #
    # Before downloading huge FASTQs, inspect the authors'
    # published analysis archive for processed sample-level
    # count matrices.
    # ----------------------------------------------------------

    print()
    print("Retrieving Zenodo analysis repository metadata")

    zenodo = fetch_json(
        ZENODO_API
    )

    files = zenodo.get(
        "files",
        []
    )

    zenodo_rows = []

    keywords = [
        "count",
        "counts",
        "expression",
        "expr",
        "matrix",
        "rlog",
        "normalized",
        "htseq",
        "transcript",
    ]

    for item in files:
        key = str(
            item.get(
                "key",
                ""
            )
        )

        size = item.get(
            "size",
            0
        )

        links = item.get(
            "links",
            {}
        )

        content_url = links.get(
            "self",
            links.get(
                "content",
                "",
            ),
        )

        lower = key.lower()

        possible_expression_file = any(
            keyword in lower
            for keyword in keywords
        )

        zenodo_rows.append(
            {
                "filename":
                    key,

                "size_bytes":
                    size,

                "size_MB":
                    (
                        float(
                            size
                        )
                        / 1e6
                        if size is not None
                        else np.nan
                    ),

                "possible_expression_file":
                    possible_expression_file,

                "download_url":
                    content_url,
            }
        )

    zenodo_table = pd.DataFrame(
        zenodo_rows
    )

    print()
    print(
        "Zenodo record title:",
        zenodo.get(
            "metadata",
            {}
        ).get(
            "title",
            "",
        ),
    )

    print(
        "  files:",
        len(
            zenodo_table
        ),
    )

    if not zenodo_table.empty:
        print()
        print("Zenodo files")
        print(
            zenodo_table[
                [
                    "filename",
                    "size_MB",
                    "possible_expression_file",
                ]
            ]
            .sort_values(
                [
                    "possible_expression_file",
                    "filename",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .round(
                {
                    "size_MB":
                        3
                }
            )
            .to_string(
                index=False
            )
        )

    expression_candidates = (
        zenodo_table.loc[
            zenodo_table[
                "possible_expression_file"
            ]
        ]
        if not zenodo_table.empty
        else pd.DataFrame()
    )

    # ----------------------------------------------------------
    # 8. Known pilot sample recovery
    # ----------------------------------------------------------

    recovered_pilots = []

    for known in KNOWN_NON_RRNA_DEPLETED:
        found = bool(
            ena[
                "sample_title"
            ]
            .str.contains(
                known,
                case=False,
                regex=False,
            )
            .any()
        )

        recovered_pilots.append(
            {
                "known_pilot_sample":
                    known,
                "found_in_ENA_titles":
                    found,
            }
        )

    pilot_check = pd.DataFrame(
        recovered_pilots
    )

    print()
    print("Known non-rRNA-depleted pilot samples")
    print(
        pilot_check.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Save preflight products
    # ----------------------------------------------------------

    ena.to_csv(
        OUT
        / "PRJEB24688_ENA_run_manifest.csv",
        index=False,
    )

    invivo.to_csv(
        OUT
        / "PRJEB24688_invivo_candidate_runs.csv",
        index=False,
    )

    standard_invivo.to_csv(
        OUT
        / "PRJEB24688_standard_invivo_candidate_runs.csv",
        index=False,
    )

    invitro.to_csv(
        OUT
        / "PRJEB24688_invitro_candidate_runs.csv",
        index=False,
    )

    unresolved.to_csv(
        OUT
        / "PRJEB24688_unresolved_runs.csv",
        index=False,
    )

    pilot_check.to_csv(
        OUT
        / "known_pilot_sample_recovery.csv",
        index=False,
    )

    zenodo_table.to_csv(
        OUT
        / "PRJEB24688_Zenodo_file_manifest.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 10. Preflight decision
    #
    # This does NOT approve expression analysis yet.
    # It only asks whether the external multi-patient cohort and
    # its data resources are sufficiently identifiable to proceed.
    # ----------------------------------------------------------

    fastq_present_fraction = float(
        ena[
            "fastq_ftp"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(
            0
        )
        .mean()
    )

    pilots_found = int(
        pilot_check[
            "found_in_ENA_titles"
        ].sum()
    )

    checks = {
        "ENA_project_runs_ge_12":
            len(
                ena
            )
            >= 12,

        "inferred_patient_IDs_ge_5":
            len(
                patient_ids
            )
            >= 5,

        "invivo_candidate_runs_ge_9":
            len(
                invivo
            )
            >= 9,

        "known_pilot_samples_found_ge_2":
            pilots_found
            >= 2,

        "FASTQ_links_present_ge_95pct":
            fastq_present_fraction
            >= 0.95,

        "Zenodo_record_accessible":
            bool(
                zenodo.get(
                    "id"
                )
            ),

        "finite_read_metadata":
            bool(
                ena[
                    "read_count"
                ]
                .dropna()
                .ge(
                    0
                )
                .all()
            ),
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 03a preflight checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    status = (
        "PRJEB24688_MULTIPATIENT_REPLICATION_PREFLIGHT_READY"
        if passed
        else "PRJEB24688_REPLICATION_PREFLIGHT_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: sample_class_guess is heuristic only. "
        "The exact in-vivo/in-vitro/sample-replicate structure "
        "must be frozen from study metadata before expression "
        "processing."
    )

    print()
    print(
        "If Zenodo contains a full sample-level count matrix, "
        "prefer auditing that published matrix before committing "
        "to large-scale raw-read reprocessing."
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
        / "AUDIT_03A_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
