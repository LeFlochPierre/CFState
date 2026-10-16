from pathlib import Path
import io
import re
import shutil
import time

import pandas as pd
import requests

from Bio import SeqIO


# ============================================================
# Paths
# ============================================================

SRA_MANIFEST = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "GSE299861_PAO1_sra_manifest.csv"
)

CANONICAL_GB = Path(
    "references/mappings/NC_002516.2.gb"
)

REF_DIR = Path(
    "references/mappings/GSE299861/canonical_PAO1"
)

FASTQ_DIR = Path(
    "data/raw/expanded_biofilm/GSE299861/fastq"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861"
)

REF_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FASTQ_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 25: "
    "prepare GSE299861 PAO1 reprocessing"
)
print("=" * 100)


# ============================================================
# Check required existing inputs
# ============================================================

if not SRA_MANIFEST.exists():

    raise FileNotFoundError(
        f"Missing SRA manifest: {SRA_MANIFEST}"
    )

if not CANONICAL_GB.exists():

    raise FileNotFoundError(
        f"Missing canonical PAO1 GenBank: {CANONICAL_GB}"
    )


manifest = pd.read_csv(
    SRA_MANIFEST
)


print()
print(
    f"SRA manifest rows: "
    f"{len(manifest)}"
)

print(
    f"Canonical GenBank: "
    f"{CANONICAL_GB}"
)


# ============================================================
# Check tools that will be needed later
#
# Not fatal yet. Audit 25 only reports availability.
# ============================================================

TOOLS = [
    "wget",
    "curl",
    "pigz",
    "bowtie2",
    "bowtie2-build",
    "samtools",
    "featureCounts",
]


print()
print("=" * 100)
print("TOOL AVAILABILITY")
print("=" * 100)


tool_records = []


for tool in TOOLS:

    path = shutil.which(
        tool
    )

    available = (
        path is not None
    )

    print(
        f"{tool:15s} "
        f"{'FOUND' if available else 'MISSING'}"
        f"{'  ' + path if path else ''}"
    )

    tool_records.append(
        {
            "tool":
                tool,

            "available":
                available,

            "path":
                path or "",
        }
    )


pd.DataFrame(
    tool_records
).to_csv(
    OUT_DIR
    / "audit25_tool_availability.csv",
    index=False,
)


# ============================================================
# Query ENA for direct FASTQ files
# ============================================================

def ena_run_info(
    srr,
    max_attempts=5,
):

    url = (
        "https://www.ebi.ac.uk/"
        "ena/portal/api/filereport"
    )

    params = {
        "accession":
            srr,

        "result":
            "read_run",

        "fields":
            (
                "run_accession,"
                "library_layout,"
                "read_count,"
                "base_count,"
                "fastq_ftp,"
                "fastq_md5,"
                "fastq_bytes"
            ),

        "format":
            "tsv",
    }


    for attempt in range(
        1,
        max_attempts + 1,
    ):

        response = requests.get(
            url,
            params=params,
            timeout=120,
        )

        if response.status_code in {
            429,
            500,
            502,
            503,
            504,
        }:

            wait = min(
                2 ** attempt,
                30,
            )

            print(
                f"  ENA temporary error "
                f"{response.status_code}; "
                f"waiting {wait}s..."
            )

            time.sleep(
                wait
            )

            continue

        response.raise_for_status()

        result = pd.read_csv(
            io.StringIO(
                response.text
            ),
            sep="\t",
        )

        if result.empty:

            raise RuntimeError(
                f"ENA returned no run data for {srr}"
            )

        return result.iloc[0]


    raise RuntimeError(
        f"Failed to retrieve ENA metadata "
        f"for {srr}"
    )


fastq_records = []


print()
print("=" * 100)
print("ENA FASTQ RESOLUTION")
print("=" * 100)


for _, row in manifest.iterrows():

    srr = str(
        row["SRR"]
    )

    sample = str(
        row["sample"]
    )

    state = str(
        row["state"]
    )


    print()
    print(
        f"Resolving {srr} "
        f"({sample})..."
    )


    ena = ena_run_info(
        srr
    )


    ftp_files = str(
        ena.get(
            "fastq_ftp",
            "",
        )
    ).split(";")

    md5_values = str(
        ena.get(
            "fastq_md5",
            "",
        )
    ).split(";")

    byte_values = str(
        ena.get(
            "fastq_bytes",
            "",
        )
    ).split(";")


    ftp_files = [
        x
        for x in ftp_files
        if x
        and x != "nan"
    ]


    if len(ftp_files) != 2:

        print(
            f"WARNING: expected 2 FASTQ files "
            f"for paired run, found "
            f"{len(ftp_files)}"
        )


    for i, ftp_path in enumerate(
        ftp_files,
        start=1,
    ):

        if ftp_path.startswith(
            "ftp://"
        ):

            url = (
                "https://"
                + ftp_path[
                    len("ftp://"):
                ]
            )

        elif ftp_path.startswith(
            "http"
        ):

            url = ftp_path

        else:

            url = (
                "https://"
                + ftp_path
            )


        md5 = (
            md5_values[
                i - 1
            ]
            if i - 1 < len(
                md5_values
            )
            else ""
        )

        byte_value = (
            byte_values[
                i - 1
            ]
            if i - 1 < len(
                byte_values
            )
            else ""
        )


        try:

            n_bytes = int(
                byte_value
            )

        except Exception:

            n_bytes = None


        filename = (
            f"{sample}_R{i}.fastq.gz"
        )


        fastq_records.append(
            {
                "sample":
                    sample,

                "state":
                    state,

                "SRR":
                    srr,

                "read":
                    i,

                "url":
                    url,

                "filename":
                    filename,

                "md5":
                    md5,

                "bytes":
                    n_bytes,
            }
        )


    time.sleep(
        0.25
    )


fastq_manifest = pd.DataFrame(
    fastq_records
)


fastq_manifest.to_csv(
    OUT_DIR
    / "GSE299861_PAO1_ENA_fastq_manifest.csv",
    index=False,
)


# ============================================================
# FASTQ size summary
# ============================================================

fastq_manifest[
    "size_GB"
] = (
    pd.to_numeric(
        fastq_manifest[
            "bytes"
        ],
        errors="coerce",
    )
    / (1024 ** 3)
)


print()
print("=" * 100)
print("DIRECT FASTQ FILES")
print("=" * 100)


print(
    fastq_manifest[
        [
            "sample",
            "state",
            "read",
            "filename",
            "size_GB",
        ]
    ].to_string(
        index=False
    )
)


total_gb = (
    fastq_manifest[
        "size_GB"
    ].sum()
)


print()
print(
    f"Total compressed FASTQ size: "
    f"{total_gb:.2f} GB"
)


# ============================================================
# Build canonical PAO1 FASTA from the existing GenBank file
# ============================================================

record = SeqIO.read(
    CANONICAL_GB,
    "genbank",
)


reference_fasta = (
    REF_DIR
    / "NC_002516.2.fasta"
)


SeqIO.write(
    record,
    reference_fasta,
    "fasta",
)


print()
print("=" * 100)
print("CANONICAL PAO1 REFERENCE")
print("=" * 100)

print(
    f"Accession: "
    f"{record.id}"
)

print(
    f"Genome length: "
    f"{len(record.seq):,} bp"
)

print(
    f"FASTA written to: "
    f"{reference_fasta}"
)


# ============================================================
# Build SAF annotation for featureCounts
#
# Only canonical PAxxxx gene features are retained.
#
# GenBank:
#   start = 0-based
#   end   = end-exclusive
#
# SAF / featureCounts:
#   Start/End = 1-based inclusive
# ============================================================

saf_records = []


for feature in record.features:

    if feature.type != "gene":
        continue


    locus_tags = (
        feature.qualifiers.get(
            "locus_tag",
            [],
        )
    )


    if not locus_tags:
        continue


    locus_tag = str(
        locus_tags[0]
    )


    if not re.fullmatch(
        r"PA\d{4}",
        locus_tag,
    ):

        continue


    start = (
        int(
            feature.location.start
        )
        + 1
    )

    end = int(
        feature.location.end
    )


    if feature.location.strand == 1:

        strand = "+"

    elif feature.location.strand == -1:

        strand = "-"

    else:

        strand = "."


    saf_records.append(
        {
            "GeneID":
                locus_tag,

            "Chr":
                record.id,

            "Start":
                start,

            "End":
                end,

            "Strand":
                strand,
        }
    )


saf = pd.DataFrame(
    saf_records
)


if saf[
    "GeneID"
].duplicated().any():

    duplicated = saf.loc[
        saf[
            "GeneID"
        ].duplicated(
            keep=False
        ),
        "GeneID",
    ].tolist()

    raise RuntimeError(
        "Duplicate PAO1 locus tags "
        f"in canonical SAF: {duplicated}"
    )


saf_file = (
    REF_DIR
    / "NC_002516.2_PA_locus_tags.saf"
)


saf.to_csv(
    saf_file,
    sep="\t",
    index=False,
)


print()
print(
    f"Canonical PAxxxx gene features: "
    f"{len(saf)}"
)

print(
    f"SAF written to: "
    f"{saf_file}"
)


# ============================================================
# Compare against frozen CF-State clean gene universe
# ============================================================

clean_gene_file = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
)


if clean_gene_file.exists():

    clean = pd.read_csv(
        clean_gene_file
    )

    possible_columns = [
        "canonical_pao1",
        "GeneID",
        "gene",
    ]

    gene_col = None


    for column in possible_columns:

        if column in clean.columns:

            gene_col = column
            break


    if gene_col is None:

        print()
        print(
            "WARNING: clean_gene_universe.csv "
            "was found, but canonical gene "
            "column could not be identified."
        )

    else:

        clean_genes = set(
            clean[
                gene_col
            ]
            .astype(str)
        )

        saf_genes = set(
            saf[
                "GeneID"
            ]
            .astype(str)
        )

        overlap = (
            clean_genes
            & saf_genes
        )


        print()
        print("=" * 100)
        print(
            "CF-STATE GENE UNIVERSE OVERLAP"
        )
        print("=" * 100)

        print(
            f"Frozen clean genes: "
            f"{len(clean_genes)}"
        )

        print(
            f"Found in canonical reference: "
            f"{len(overlap)}"
        )

        print(
            f"Missing: "
            f"{len(clean_genes - saf_genes)}"
        )


# ============================================================
# Create download shell script
# ============================================================

download_script = (
    OUT_DIR
    / "download_GSE299861_PAO1_fastqs.sh"
)


with open(
    download_script,
    "w",
) as handle:

    handle.write(
        "#!/usr/bin/env bash\n"
    )

    handle.write(
        "set -euo pipefail\n\n"
    )

    handle.write(
        f'mkdir -p "{FASTQ_DIR}"\n\n'
    )


    for _, row in (
        fastq_manifest.iterrows()
    ):

        output_file = (
            FASTQ_DIR
            / row[
                "filename"
            ]
        )

        handle.write(
            f'echo "Downloading '
            f'{row["filename"]}..."\n'
        )

        handle.write(
            f'wget -c '
            f'-O "{output_file}" '
            f'"{row["url"]}"\n\n'
        )


    handle.write(
        'echo "FASTQ downloads complete."\n'
    )


download_script.chmod(
    0o755
)


print()
print("=" * 100)
print("AUDIT 25 SUMMARY")
print("=" * 100)

print(
    f"ENA manifest: "
    f"{OUT_DIR / 'GSE299861_PAO1_ENA_fastq_manifest.csv'}"
)

print(
    f"Canonical FASTA: "
    f"{reference_fasta}"
)

print(
    f"Canonical SAF: "
    f"{saf_file}"
)

print(
    f"Download script: "
    f"{download_script}"
)

print()
print(
    "NOTE: FASTQs have NOT been downloaded yet."
)

print(
    "Audit 25 only resolves files, "
    "checks tools, and prepares the "
    "canonical PAO1 reference."
)

print()
