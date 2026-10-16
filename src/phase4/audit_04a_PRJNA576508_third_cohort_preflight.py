from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from io import StringIO
import re

import pandas as pd


ROOT = Path.cwd()

OUT = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
)

ENA_URL = "https://www.ebi.ac.uk/ena/portal/api/filereport"


def fetch_text(url):
    request = Request(
        url,
        headers={
            "User-Agent":
                "CF-State Phase4 third-cohort audit"
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as handle:
        return handle.read().decode(
            "utf-8"
        )


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


def classify_title(title, scientific_name):
    title_text = str(
        title
    ).upper()

    organism_text = str(
        scientific_name
    ).upper()

    positive_terms = [
        "SPUTUM",
        "HUMANSPUTUM",
        "EXPECTORATED",
    ]

    model_terms = [
        "CFBE",
        "EPITHELIAL",
        "CELL CULTURE",
        "IN VITRO",
        "INVITRO",
        "MODEL",
    ]

    positive = any(
        term in title_text
        for term in positive_terms
    )

    model_like = any(
        term in title_text
        for term in model_terms
    )

    lung_metagenome = (
        "HUMAN LUNG METAGENOME"
        in organism_text
    )

    if (
        positive
        and not model_like
    ):
        return "ATLANTA_CF_SPUTUM_CANDIDATE"

    if (
        lung_metagenome
        and not model_like
    ):
        return "ATLANTA_CF_SPUTUM_CANDIDATE"

    return "OTHER_PROJECT_SAMPLE"


def extract_short_sample_label(title):
    text = str(
        title
    )

    patterns = [
        r"(EM\d+)",
        r"(HumanSputum[_-]?[A-Za-z0-9]+)",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if match is not None:
            return match.group(
                1
            )

    return text


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04a")
    print("PRJNA576508 independent Atlanta CF sputum cohort preflight")
    print("=" * 78)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. ENA metadata
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
            "PRJNA576508",

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

    url = (
        ENA_URL
        + "?"
        + urlencode(
            params
        )
    )

    print()
    print("Retrieving PRJNA576508 ENA metadata")

    text = fetch_text(
        url
    )

    manifest = pd.read_csv(
        StringIO(
            text
        ),
        sep="\t",
    )

    if manifest.empty:
        raise RuntimeError(
            "ENA returned no PRJNA576508 runs."
        )

    manifest[
        "sample_title"
    ] = (
        manifest[
            "sample_title"
        ]
        .fillna("")
        .astype(str)
    )

    manifest[
        "scientific_name"
    ] = (
        manifest[
            "scientific_name"
        ]
        .fillna("")
        .astype(str)
    )

    manifest[
        "sample_class"
    ] = [
        classify_title(
            title,
            organism,
        )
        for title, organism
        in zip(
            manifest[
                "sample_title"
            ],
            manifest[
                "scientific_name"
            ],
        )
    ]

    manifest[
        "short_sample_label"
    ] = manifest[
        "sample_title"
    ].map(
        extract_short_sample_label
    )

    for column in [
        "read_count",
        "base_count",
    ]:
        manifest[
            column
        ] = pd.to_numeric(
            manifest[
                column
            ],
            errors="coerce",
        )

    manifest[
        "total_fastq_bytes"
    ] = manifest[
        "fastq_bytes"
    ].map(
        total_fastq_bytes
    )

    manifest[
        "total_fastq_GB"
    ] = (
        manifest[
            "total_fastq_bytes"
        ]
        / 1e9
    )

    # ----------------------------------------------------------
    # 2. Print full project structure
    # ----------------------------------------------------------

    display_columns = [
        "run_accession",
        "experiment_accession",
        "sample_accession",
        "short_sample_label",
        "sample_title",
        "scientific_name",
        "sample_class",
        "library_strategy",
        "library_source",
        "library_selection",
        "library_layout",
        "instrument_model",
        "read_count",
        "total_fastq_GB",
    ]

    print()
    print("Full PRJNA576508 manifest")

    print(
        manifest[
            display_columns
        ]
        .sort_values(
            [
                "sample_class",
                "short_sample_label",
            ]
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
    # 3. Freeze Atlanta sputum subset
    # ----------------------------------------------------------

    sputum = manifest.loc[
        manifest[
            "sample_class"
        ]
        == "ATLANTA_CF_SPUTUM_CANDIDATE"
    ].copy()

    other = manifest.loc[
        manifest[
            "sample_class"
        ]
        != "ATLANTA_CF_SPUTUM_CANDIDATE"
    ].copy()

    n_runs = len(
        manifest
    )

    n_experiments = manifest[
        "experiment_accession"
    ].nunique()

    n_samples = manifest[
        "sample_accession"
    ].nunique()

    n_sputum_runs = len(
        sputum
    )

    n_sputum_experiments = sputum[
        "experiment_accession"
    ].nunique()

    n_sputum_samples = sputum[
        "sample_accession"
    ].nunique()

    print()
    print("Project structure")
    print(
        "  total ENA runs:",
        n_runs,
    )
    print(
        "  unique experiments:",
        n_experiments,
    )
    print(
        "  unique BioSamples:",
        n_samples,
    )
    print(
        "  Atlanta sputum candidate runs:",
        n_sputum_runs,
    )
    print(
        "  Atlanta sputum candidate experiments:",
        n_sputum_experiments,
    )
    print(
        "  Atlanta sputum candidate BioSamples:",
        n_sputum_samples,
    )
    print(
        "  other project runs:",
        len(
            other
        ),
    )

    print()
    print("Candidate independent third-cohort sputum samples")

    print(
        sputum[
            [
                "run_accession",
                "sample_accession",
                "short_sample_label",
                "sample_title",
                "scientific_name",
                "library_layout",
                "instrument_model",
                "read_count",
                "total_fastq_GB",
            ]
        ]
        .sort_values(
            "short_sample_label"
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

    print()
    print("Other PRJNA576508 project samples")

    if other.empty:
        print(
            "  none"
        )
    else:
        print(
            other[
                [
                    "run_accession",
                    "sample_accession",
                    "sample_title",
                    "scientific_name",
                    "library_layout",
                    "instrument_model",
                    "read_count",
                    "total_fastq_GB",
                ]
            ]
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
    # 4. Basic sequencing characteristics
    # ----------------------------------------------------------

    print()
    print("Atlanta sputum library layouts")
    print(
        sputum[
            "library_layout"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("Atlanta sputum instruments")
    print(
        sputum[
            "instrument_model"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print("Atlanta sputum library selections")
    print(
        sputum[
            "library_selection"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print()
    print(
        "Atlanta sputum total FASTQ volume:",
        round(
            float(
                sputum[
                    "total_fastq_GB"
                ].sum()
            ),
            2,
        ),
        "GB",
    )

    # ----------------------------------------------------------
    # 5. Independence guard
    #
    # Frozen discovery/replication samples use GSE119356 region
    # names or PRJEB24688 Pxx patient labels.
    #
    # We do not expect those names in this Atlanta cohort.
    # ----------------------------------------------------------

    previous_patterns = [
        r"\bP24M1\b",
        r"\bP30M0\b",
        r"\bP73F1\b",
        r"\bP77F1\b",
        r"\bP11F2\b",
        r"\bINVIVO\b",
    ]

    overlap_rows = []

    for _, row in sputum.iterrows():
        title = str(
            row[
                "sample_title"
            ]
        )

        matched = [
            pattern
            for pattern in previous_patterns
            if re.search(
                pattern,
                title,
                flags=re.IGNORECASE,
            )
        ]

        if matched:
            overlap_rows.append(
                {
                    "run_accession":
                        row[
                            "run_accession"
                        ],

                    "sample_title":
                        title,

                    "matched_previous_pattern":
                        ",".join(
                            matched
                        ),
                }
            )

    overlap = pd.DataFrame(
        overlap_rows
    )

    print()
    print("Previous-cohort identifier overlap")

    if overlap.empty:
        print(
            "  none detected"
        )
    else:
        print(
            overlap.to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 6. FASTQ accessibility
    # ----------------------------------------------------------

    fastq_present = (
        sputum[
            "fastq_ftp"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(
            0
        )
    )

    fastq_fraction = float(
        fastq_present.mean()
    )

    print()
    print("FASTQ accessibility")
    print(
        "  sputum runs with FASTQ links:",
        int(
            fastq_present.sum()
        ),
        "/",
        len(
            sputum
        ),
    )
    print(
        "  fraction:",
        round(
            fastq_fraction,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 7. Freeze design
    #
    # Do NOT merge or otherwise transform samples here.
    # This audit is metadata-only.
    # ----------------------------------------------------------

    frozen_design = sputum[
        [
            "short_sample_label",
            "sample_accession",
            "experiment_accession",
            "run_accession",
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
            "total_fastq_GB",
        ]
    ].copy()

    # ----------------------------------------------------------
    # 8. Preflight checks
    #
    # Publication expectation:
    # 13 newly collected Atlanta sputum samples.
    #
    # NCBI currently reports 18 experiments in PRJNA576508,
    # therefore the purpose of this audit is explicitly to freeze
    # the 13 clinical sputum samples and separate the remainder.
    # ----------------------------------------------------------

    candidate_titles_unique = (
        sputum[
            "sample_title"
        ].nunique()
        == len(
            sputum
        )
    )

    all_sputum_RNAseq = bool(
        sputum[
            "library_strategy"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "RNA-SEQ"
        )
        .all()
    )

    all_sputum_single_end = bool(
        sputum[
            "library_layout"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "SINGLE"
        )
        .all()
    )

    finite_reads = bool(
        sputum[
            "read_count"
        ]
        .notna()
        .all()
        and (
            sputum[
                "read_count"
            ]
            > 0
        ).all()
    )

    checks = {
        "project_has_18_experiments":
            n_experiments
            == 18,

        "13_Atlanta_sputum_BioSamples":
            n_sputum_samples
            == 13,

        "13_Atlanta_sputum_experiments":
            n_sputum_experiments
            == 13,

        "13_Atlanta_sputum_runs":
            n_sputum_runs
            == 13,

        "candidate_titles_unique":
            candidate_titles_unique,

        "all_candidate_runs_RNAseq":
            all_sputum_RNAseq,

        "all_candidate_runs_single_end":
            all_sputum_single_end,

        "all_candidate_runs_have_positive_read_counts":
            finite_reads,

        "all_candidate_FASTQ_links_present":
            fastq_fraction
            == 1.0,

        "no_previous_cohort_identifier_overlap":
            overlap.empty,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04a preflight checks")

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
        "PRJNA576508_INDEPENDENT_ATLANTA_COHORT_FROZEN"
        if passed
        else "PRJNA576508_THIRD_COHORT_PREFLIGHT_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: this audit freezes only the newly collected "
        "Atlanta CF sputum subset. No PRJNA576508 expression data "
        "have yet been compared with the frozen CF-airway candidate."
    )

    print()
    print(
        "The two-cohort candidate from Audit 03h must remain "
        "completely unchanged during all PRJNA576508 processing "
        "and validation."
    )

    # ----------------------------------------------------------
    # 9. Save
    # ----------------------------------------------------------

    manifest.to_csv(
        OUT
        / "PRJNA576508_full_ENA_manifest.csv",
        index=False,
    )

    sputum.to_csv(
        OUT
        / "PRJNA576508_Atlanta_sputum_manifest.csv",
        index=False,
    )

    other.to_csv(
        OUT
        / "PRJNA576508_other_project_samples.csv",
        index=False,
    )

    frozen_design.to_csv(
        OUT
        / "FROZEN_PRJNA576508_independent_third_cohort.csv",
        index=False,
    )

    overlap.to_csv(
        OUT
        / "previous_cohort_identifier_overlap.csv",
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
        / "AUDIT_04A_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed Audit 04a checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
