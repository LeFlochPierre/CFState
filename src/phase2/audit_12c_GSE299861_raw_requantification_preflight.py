from pathlib import Path
import json
import re
import shutil
import urllib.parse
import urllib.request

import pandas as pd


ROOT = Path.cwd()

OUT = (
    ROOT
    / "results/phase2/audit_12c_GSE299861_raw_requantification_preflight"
)

REFERENCE = (
    ROOT
    / "references/mappings/NC_002516.2.gb"
)

SAMPLES = [
    {
        "gsm": "GSM9047975",
        "condition": "biofilm",
        "replicate": 1,
    },
    {
        "gsm": "GSM9047976",
        "condition": "biofilm",
        "replicate": 2,
    },
    {
        "gsm": "GSM9047977",
        "condition": "biofilm",
        "replicate": 3,
    },
    {
        "gsm": "GSM9047978",
        "condition": "planktonic",
        "replicate": 1,
    },
    {
        "gsm": "GSM9047979",
        "condition": "planktonic",
        "replicate": 2,
    },
    {
        "gsm": "GSM9047980",
        "condition": "planktonic",
        "replicate": 3,
    },
]


def fetch_text(url):
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "CF-State Phase2 Audit12c"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=60,
    ) as response:
        return response.read().decode(
            "utf-8",
            errors="replace",
        )


def resolve_srx(gsm):
    url = (
        "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?"
        + urllib.parse.urlencode(
            {
                "acc": gsm,
                "targ": "self",
                "form": "text",
                "view": "full",
            }
        )
    )

    text = fetch_text(
        url
    )

    matches = sorted(
        set(
            re.findall(
                r"SRX\d+",
                text,
            )
        )
    )

    if len(matches) != 1:
        raise RuntimeError(
            gsm
            + ": expected exactly one SRX, found "
            + str(matches)
        )

    return matches[0]


def resolve_ena_run(srx):
    fields = [
        "run_accession",
        "experiment_accession",
        "sample_accession",
        "library_layout",
        "read_count",
        "base_count",
        "fastq_ftp",
        "fastq_md5",
        "fastq_bytes",
    ]

    url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport?"
        + urllib.parse.urlencode(
            {
                "accession": srx,
                "result": "read_run",
                "fields": ",".join(
                    fields
                ),
                "format": "json",
            }
        )
    )

    text = fetch_text(
        url
    )

    rows = json.loads(
        text
    )

    if len(rows) != 1:
        raise RuntimeError(
            srx
            + ": expected exactly one sequencing run, found "
            + str(
                len(rows)
            )
        )

    return rows[0]


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 12c")
    print("GSE299861 canonical raw-RNA-seq requantification preflight")
    print("=" * 78)

    if not REFERENCE.exists():
        raise FileNotFoundError(
            str(REFERENCE)
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Resolve the six PAO1 raw-data runs
    # ----------------------------------------------------------

    metadata = []

    for sample in SAMPLES:
        gsm = sample[
            "gsm"
        ]

        print()
        print(
            "Resolving",
            gsm,
            "...",
        )

        srx = resolve_srx(
            gsm
        )

        ena = resolve_ena_run(
            srx
        )

        row = {
            **sample,
            "srx":
                srx,
            "srr":
                ena.get(
                    "run_accession",
                    "",
                ),
            "sample_accession":
                ena.get(
                    "sample_accession",
                    "",
                ),
            "library_layout":
                ena.get(
                    "library_layout",
                    "",
                ),
            "read_count":
                ena.get(
                    "read_count",
                    "",
                ),
            "base_count":
                ena.get(
                    "base_count",
                    "",
                ),
            "fastq_ftp":
                ena.get(
                    "fastq_ftp",
                    "",
                ),
            "fastq_md5":
                ena.get(
                    "fastq_md5",
                    "",
                ),
            "fastq_bytes":
                ena.get(
                    "fastq_bytes",
                    "",
                ),
        }

        metadata.append(
            row
        )

        print(
            " ",
            gsm,
            "->",
            srx,
            "->",
            row[
                "srr"
            ],
        )

    metadata = pd.DataFrame(
        metadata
    )

    metadata.to_csv(
        OUT
        / "PAO1_raw_run_manifest.csv",
        index=False,
    )

    print()
    print("Resolved PAO1 runs")
    print(
        metadata[
            [
                "gsm",
                "condition",
                "replicate",
                "srx",
                "srr",
                "library_layout",
                "read_count",
                "base_count",
            ]
        ].to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 2. Validate metadata
    # ----------------------------------------------------------

    unique_srr = (
        metadata[
            "srr"
        ]
        .astype(str)
        .nunique()
    )

    paired = (
        metadata[
            "library_layout"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "PAIRED"
        )
        .all()
    )

    ftp_complete = (
        metadata[
            "fastq_ftp"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(0)
        .all()
    )

    # ----------------------------------------------------------
    # 3. Check local tools
    # ----------------------------------------------------------

    tools = [
        "bowtie2",
        "bowtie2-build",
        "samtools",
        "featureCounts",
        "cutadapt",
    ]

    tool_rows = []

    print()
    print("Required tool availability")

    for tool in tools:
        path = shutil.which(
            tool
        )

        installed = (
            path is not None
        )

        tool_rows.append(
            {
                "tool":
                    tool,
                "installed":
                    installed,
                "path":
                    path or "",
            }
        )

        print(
            " ",
            tool,
            ":",
            path
            if installed
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

    essential_tools = {
        "bowtie2",
        "bowtie2-build",
        "samtools",
        "featureCounts",
    }

    essential_ok = bool(
        tool_table.loc[
            tool_table[
                "tool"
            ].isin(
                essential_tools
            ),
            "installed",
        ].all()
    )

    cutadapt_ok = bool(
        tool_table.loc[
            tool_table[
                "tool"
            ]
            == "cutadapt",
            "installed",
        ].iloc[0]
    )

    # ----------------------------------------------------------
    # 4. Reference sanity check
    # ----------------------------------------------------------

    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython is required."
        )

    record = SeqIO.read(
        str(REFERENCE),
        "genbank",
    )

    locus_tags = set()

    cds_count = 0

    for feature in record.features:
        tags = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if tags:
            locus_tags.add(
                str(
                    tags[0]
                ).strip()
            )

        if feature.type == "CDS":
            cds_count += 1

    print()
    print("Canonical reference")
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
        "  CDS features:",
        cds_count,
    )

    # ----------------------------------------------------------
    # 5. Decision
    # ----------------------------------------------------------

    checks = {
        "six_samples":
            len(
                metadata
            )
            == 6,

        "six_unique_runs":
            unique_srr
            == 6,

        "all_paired_end":
            paired,

        "fastq_locations_resolved":
            ftp_complete,

        "canonical_reference_present":
            REFERENCE.exists(),

        "canonical_reference_large_gene_set":
            len(
                locus_tags
            )
            > 5500,

        "essential_alignment_tools_available":
            essential_ok,
    }

    passed = all(
        checks.values()
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

    print()
    print(
        "Cutadapt available:",
        cutadapt_ok,
    )

    print()
    print(
        "Audit 12c:",
        "PASS"
        if passed
        else "ACTION_REQUIRED",
    )

    if not essential_ok:
        print()
        print(
            "Missing alignment/counting tools."
        )
        print(
            "Install into CFSTATE_ENV with:"
        )
        print()
        print(
            "conda install -c bioconda "
            "bowtie2 samtools subread"
        )

    if not cutadapt_ok:
        print()
        print(
            "Cutadapt is optional for the technical preflight "
            "but should be installed if we reproduce the "
            "authors' trimming step:"
        )
        print()
        print(
            "conda install -c bioconda cutadapt"
        )

    print()
    print(
        "Manifest:",
        OUT
        / "PAO1_raw_run_manifest.csv",
    )


if __name__ == "__main__":
    main()
