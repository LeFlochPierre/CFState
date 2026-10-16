from pathlib import Path
import csv
import io
import json
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd


ROOT = Path.cwd()

PA14_GB = (
    ROOT
    / "references/mappings/NC_008463.1.gb"
)

OUT = (
    ROOT
    / "results/phase3/audit_02i_GSE191103_raw_requantification_preflight"
)


SAMPLES = [
    {
        "gsm": "GSM5739075",
        "sample": "PA14_BF_1",
        "condition": "biofilm",
    },
    {
        "gsm": "GSM5739076",
        "sample": "PA14_BF_2",
        "condition": "biofilm",
    },
    {
        "gsm": "GSM5739077",
        "sample": "PA14_BF_3",
        "condition": "biofilm",
    },
    {
        "gsm": "GSM5739108",
        "sample": "PA14_PL_1",
        "condition": "planktonic",
    },
    {
        "gsm": "GSM5739109",
        "sample": "PA14_PL_2",
        "condition": "planktonic",
    },
    {
        "gsm": "GSM5739110",
        "sample": "PA14_PL_3",
        "condition": "planktonic",
    },
]


def fetch_text(url):
    """
    NCBI/GEO-friendly HTTP fetch with throttling and retry.

    Without an NCBI API key, E-utilities should not be queried
    aggressively. We therefore pause before every request and
    retry transient 429/5xx responses with exponential backoff.
    """

    max_attempts = 6

    # Keep comfortably below the unauthenticated NCBI request limit.
    time.sleep(
        0.5
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "CF-State Phase3 Audit02i"
        },
    )

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:
                return response.read().decode(
                    "utf-8",
                    errors="replace",
                )

        except urllib.error.HTTPError as error:
            retryable = (
                error.code == 429
                or 500 <= error.code < 600
            )

            if (
                not retryable
                or attempt == max_attempts
            ):
                raise

            retry_after = error.headers.get(
                "Retry-After"
            )

            if retry_after is not None:
                try:
                    wait_seconds = float(
                        retry_after
                    )
                except ValueError:
                    wait_seconds = float(
                        2 ** attempt
                    )
            else:
                wait_seconds = float(
                    2 ** attempt
                )

            print(
                "  HTTP",
                error.code,
                "- retrying in",
                round(
                    wait_seconds,
                    1,
                ),
                "seconds",
                "("
                + str(
                    attempt
                )
                + "/"
                + str(
                    max_attempts
                )
                + ")",
            )

            time.sleep(
                wait_seconds
            )

        except urllib.error.URLError:
            if attempt == max_attempts:
                raise

            wait_seconds = float(
                2 ** attempt
            )

            print(
                "  transient network error - retrying in",
                round(
                    wait_seconds,
                    1,
                ),
                "seconds",
                "("
                + str(
                    attempt
                )
                + "/"
                + str(
                    max_attempts
                )
                + ")",
            )

            time.sleep(
                wait_seconds
            )

    raise RuntimeError(
        "HTTP fetch failed unexpectedly."
    )


def resolve_gsm_to_srx(gsm):
    url = (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"
        "?acc="
        + urllib.parse.quote(
            gsm
        )
        + "&targ=self&view=full&form=text"
    )

    text = fetch_text(
        url
    )

    matches = re.findall(
        r"SRX\d+",
        text,
    )

    matches = sorted(
        set(
            matches
        )
    )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            gsm
            + ": expected exactly one SRX accession, found "
            + str(
                matches
            )
        )

    return (
        matches[
            0
        ],
        text,
    )


def sra_search_id(accession):
    query = urllib.parse.urlencode(
        {
            "db": "sra",
            "term":
                accession
                + "[ACCN]",
            "retmode": "json",
        }
    )

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        "esearch.fcgi?"
        + query
    )

    result = json.loads(
        fetch_text(
            url
        )
    )

    ids = result[
        "esearchresult"
    ][
        "idlist"
    ]

    if not ids:
        raise RuntimeError(
            "No SRA record found for "
            + accession
        )

    return ids


def fetch_runinfo(sra_ids):
    query = urllib.parse.urlencode(
        {
            "db": "sra",
            "id":
                ",".join(
                    sra_ids
                ),
            "rettype": "runinfo",
            "retmode": "text",
        }
    )

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        "efetch.fcgi?"
        + query
    )

    text = fetch_text(
        url
    )

    table = pd.read_csv(
        io.StringIO(
            text
        )
    )

    return table


def first_existing_column(
    table,
    candidates,
):
    for column in candidates:
        if column in table.columns:
            return column

    return None


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02i")
    print("GSE191103 WT PA14 raw-requantification preflight")
    print("=" * 78)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PA14_GB.exists():
        raise FileNotFoundError(
            str(
                PA14_GB
            )
        )

    # ----------------------------------------------------------
    # 1. Resolve GEO -> SRA
    # ----------------------------------------------------------

    manifest_rows = []

    print()
    print("Resolving six WT GEO samples")

    for entry in SAMPLES:
        gsm = entry[
            "gsm"
        ]

        sample = entry[
            "sample"
        ]

        condition = entry[
            "condition"
        ]

        srx, geo_text = resolve_gsm_to_srx(
            gsm
        )

        sra_ids = sra_search_id(
            srx
        )

        runinfo = fetch_runinfo(
            sra_ids
        )

        if runinfo.empty:
            raise RuntimeError(
                "No SRA RunInfo returned for "
                + srx
            )

        run_column = first_existing_column(
            runinfo,
            [
                "Run",
                "run_accession",
            ],
        )

        layout_column = first_existing_column(
            runinfo,
            [
                "LibraryLayout",
                "library_layout",
            ],
        )

        spots_column = first_existing_column(
            runinfo,
            [
                "spots",
                "Spots",
            ],
        )

        bases_column = first_existing_column(
            runinfo,
            [
                "bases",
                "Bases",
            ],
        )

        if run_column is None:
            raise RuntimeError(
                "RunInfo has no recognizable run-accession column."
            )

        for _, row in runinfo.iterrows():
            run = str(
                row[
                    run_column
                ]
            ).strip()

            if not run.startswith(
                "SRR"
            ):
                continue

            layout = (
                str(
                    row[
                        layout_column
                    ]
                ).strip()
                if layout_column is not None
                else ""
            )

            spots = (
                int(
                    row[
                        spots_column
                    ]
                )
                if (
                    spots_column is not None
                    and pd.notna(
                        row[
                            spots_column
                        ]
                    )
                )
                else None
            )

            bases = (
                int(
                    row[
                        bases_column
                    ]
                )
                if (
                    bases_column is not None
                    and pd.notna(
                        row[
                            bases_column
                        ]
                    )
                )
                else None
            )

            manifest_rows.append(
                {
                    "gsm":
                        gsm,

                    "sample":
                        sample,

                    "condition":
                        condition,

                    "srx":
                        srx,

                    "srr":
                        run,

                    "library_layout":
                        layout,

                    "spots":
                        spots,

                    "bases":
                        bases,
                }
            )

        print(
            " ",
            gsm,
            sample,
            "->",
            srx,
            "->",
            ", ".join(
                str(
                    value
                )
                for value
                in runinfo[
                    run_column
                ].dropna()
            ),
        )

    manifest = pd.DataFrame(
        manifest_rows
    )

    if manifest.empty:
        raise RuntimeError(
            "No SRR runs resolved."
        )

    manifest = manifest.sort_values(
        [
            "condition",
            "sample",
            "srr",
        ]
    )

    print()
    print("Resolved raw-run manifest")
    print(
        manifest.to_string(
            index=False
        )
    )

    manifest.to_csv(
        OUT
        / "GSE191103_WT_PA14_raw_run_manifest.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 2. Manifest integrity
    # ----------------------------------------------------------

    resolved_samples = set(
        manifest[
            "sample"
        ]
    )

    expected_samples = set(
        entry[
            "sample"
        ]
        for entry
        in SAMPLES
    )

    unique_runs = manifest[
        "srr"
    ].nunique()

    paired_fraction = float(
        manifest[
            "library_layout"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "PAIRED"
        )
        .mean()
    )

    print()
    print("Raw-data structure")
    print(
        "  samples resolved:",
        len(
            resolved_samples
        ),
    )
    print(
        "  unique SRR runs:",
        unique_runs,
    )
    print(
        "  paired-end fraction:",
        format(
            paired_fraction,
            ".2%",
        ),
    )

    if "spots" in manifest.columns:
        numeric_spots = pd.to_numeric(
            manifest[
                "spots"
            ],
            errors="coerce",
        )

        if numeric_spots.notna().any():
            print(
                "  minimum spots:",
                int(
                    numeric_spots.min()
                ),
            )
            print(
                "  maximum spots:",
                int(
                    numeric_spots.max()
                ),
            )

    if "bases" in manifest.columns:
        numeric_bases = pd.to_numeric(
            manifest[
                "bases"
            ],
            errors="coerce",
        )

        if numeric_bases.notna().any():
            print(
                "  minimum bases:",
                int(
                    numeric_bases.min()
                ),
            )
            print(
                "  maximum bases:",
                int(
                    numeric_bases.max()
                ),
            )

    # ----------------------------------------------------------
    # 3. Tool availability
    # ----------------------------------------------------------

    tools = [
        "prefetch",
        "fasterq-dump",
        "cutadapt",
        "bowtie2",
        "bowtie2-build",
        "samtools",
        "featureCounts",
    ]

    tool_rows = []

    print()
    print("Required tools")

    for tool in tools:
        path = shutil.which(
            tool
        )

        available = (
            path is not None
        )

        tool_rows.append(
            {
                "tool":
                    tool,
                "available":
                    available,
                "path":
                    path,
            }
        )

        print(
            " ",
            tool,
            ":",
            path
            if available
            else "MISSING",
        )

    tool_table = pd.DataFrame(
        tool_rows
    )

    tool_table.to_csv(
        OUT
        / "tool_availability.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 4. Canonical PA14 reference integrity
    # ----------------------------------------------------------

    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython is required."
        )

    record = SeqIO.read(
        str(
            PA14_GB
        ),
        "genbank",
    )

    locus_tags = set()

    cds_old_tags = set()

    cds_current_tags = set()

    for feature in record.features:
        current = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        old = feature.qualifiers.get(
            "old_locus_tag",
            [],
        )

        for value in current:
            locus_tags.add(
                str(
                    value
                ).strip()
            )

        if feature.type == "CDS":
            for value in current:
                cds_current_tags.add(
                    str(
                        value
                    ).strip()
                )

            for value in old:
                cds_old_tags.add(
                    str(
                        value
                    ).strip()
                )

    print()
    print("Canonical PA14 reference")
    print(
        "  accession:",
        record.id,
    )
    print(
        "  genome length:",
        len(
            record.seq
        ),
    )
    print(
        "  distinct locus tags:",
        len(
            locus_tags
        ),
    )
    print(
        "  CDS current locus tags:",
        len(
            cds_current_tags
        ),
    )
    print(
        "  CDS old locus tags:",
        len(
            cds_old_tags
        ),
    )

    # ----------------------------------------------------------
    # 5. Formal preflight gates
    # ----------------------------------------------------------

    checks = {
        "six_expected_samples_resolved":
            resolved_samples
            == expected_samples,

        "at_least_six_unique_runs":
            unique_runs
            >= 6,

        "all_runs_paired_end":
            paired_fraction
            == 1.0,

        "all_required_tools_available":
            bool(
                tool_table[
                    "available"
                ].all()
            ),

        "reference_accession_NC_008463_1":
            str(
                record.id
            ).startswith(
                "NC_008463.1"
            ),

        "reference_genome_length_expected":
            len(
                record.seq
            )
            == 6537648,

        "reference_has_many_CDS":
            len(
                cds_current_tags
            )
            >= 5800,

        "reference_has_old_PA14_tags":
            len(
                cds_old_tags
            )
            >= 5500,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Preflight checks")

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
        "READY_FOR_CANONICAL_PA14_RAW_REQUANTIFICATION"
        if passed
        else "RAW_REQUANTIFICATION_PREFLIGHT_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
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
        / "AUDIT_02I_STATUS.txt"
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
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
