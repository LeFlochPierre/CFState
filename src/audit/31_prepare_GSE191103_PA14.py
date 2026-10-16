from pathlib import Path
import io
import re
import time

import pandas as pd
import requests
from Bio import SeqIO


# ============================================================
# Paths
# ============================================================

SRA_MANIFEST = Path(
    "results/audit/expanded_biofilm/GSE191103/"
    "GSE191103_PA14_sra_manifest.csv"
)

ORTHOLOGY_FILE = Path(
    "references/mappings/pao1_pa14_rbbh.csv"
)

CLEAN_GENE_FILE = Path(
    "data/processed/biofilm/clean_gene_universe.csv"
)

REF_DIR = Path(
    "references/mappings/GSE191103/PA14"
)

FASTQ_DIR = Path(
    "data/raw/expanded_biofilm/GSE191103/fastq"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE191103"
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


PA14_ACCESSION = "NC_008463.1"


print()
print("=" * 100)
print(
    "CF-STATE — Audit 31: "
    "prepare GSE191103 PA14 reprocessing"
)
print("=" * 100)


# ============================================================
# Check inputs
# ============================================================

for path in [
    SRA_MANIFEST,
    ORTHOLOGY_FILE,
    CLEAN_GENE_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(
            "Missing required file: " + str(path)
        )


manifest = pd.read_csv(
    SRA_MANIFEST
)

print()
print(
    "SRA samples: "
    + str(len(manifest))
)


# ============================================================
# Download PA14 GenBank
# ============================================================

gb_file = (
    REF_DIR
    / "NC_008463.1.gb"
)


if not gb_file.exists():

    print()
    print(
        "Downloading PA14 reference "
        + PA14_ACCESSION
    )

    url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/efetch.fcgi"
    )

    params = {
        "db": "nuccore",
        "id": PA14_ACCESSION,
        "rettype": "gbwithparts",
        "retmode": "text",
    }

    response = requests.get(
        url,
        params=params,
        timeout=120,
    )

    response.raise_for_status()

    gb_file.write_text(
        response.text
    )

    print(
        "Saved: "
        + str(gb_file)
    )

else:

    print()
    print(
        "PA14 GenBank already present."
    )


# ============================================================
# Parse PA14 reference
# ============================================================

record = SeqIO.read(
    gb_file,
    "genbank",
)


print()
print("=" * 100)
print("PA14 REFERENCE")
print("=" * 100)

print(
    "Accession: "
    + str(record.id)
)

print(
    "Genome length: "
    + format(
        len(record.seq),
        ",",
    )
    + " bp"
)


# ============================================================
# Write FASTA
# ============================================================

fasta_file = (
    REF_DIR
    / "NC_008463.1.fasta"
)

SeqIO.write(
    record,
    fasta_file,
    "fasta",
)

# ============================================================
# Build PA14 SAF annotation
#
# IMPORTANT:
# Current RefSeq annotation uses locus tags such as:
#
#     PA14_RS00005
#
# Our PAO1 <-> PA14 orthology table uses the historical
# identifiers:
#
#     PA14_00010
#
# These historical identifiers are stored in old_locus_tag.
#
# We therefore use old_locus_tag as GeneID whenever available.
# ============================================================

saf_records = []


for feature in record.features:

    if feature.type not in [
        "gene",
        "CDS",
    ]:
        continue


    old_tags = feature.qualifiers.get(
        "old_locus_tag",
        [],
    )

    current_tags = feature.qualifiers.get(
        "locus_tag",
        [],
    )


    pa14_gene_id = None


    # --------------------------------------------------------
    # Prefer historical PA14 identifiers because these match
    # the orthology mapping already used by CF-State.
    # --------------------------------------------------------

    for value in old_tags:

        value = str(
            value
        ).strip()

        if re.fullmatch(
            r"PA14_\d+",
            value,
        ):
            pa14_gene_id = value
            break


    # --------------------------------------------------------
    # Fallback in case an old-style identifier is stored
    # directly as locus_tag in another annotation version.
    # --------------------------------------------------------

    if pa14_gene_id is None:

        for value in current_tags:

            value = str(
                value
            ).strip()

            if re.fullmatch(
                r"PA14_\d+",
                value,
            ):
                pa14_gene_id = value
                break


    if pa14_gene_id is None:
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


    # Prefer a gene feature over CDS if both exist.
    feature_priority = (
        0
        if feature.type == "gene"
        else 1
    )


    saf_records.append(
        {
            "GeneID":
                pa14_gene_id,

            "Chr":
                record.id,

            "Start":
                start,

            "End":
                end,

            "Strand":
                strand,

            "feature_priority":
                feature_priority,
        }
    )


if len(saf_records) == 0:

    raise RuntimeError(
        "No historical PA14 locus tags were recovered "
        "from the GenBank annotation."
    )


saf = pd.DataFrame(
    saf_records
)


# ============================================================
# gene and CDS features usually describe the same locus.
# Keep the gene feature where both are present.
# ============================================================

saf = (
    saf
    .sort_values(
        [
            "GeneID",
            "feature_priority",
        ]
    )
    .drop_duplicates(
        subset="GeneID",
        keep="first",
    )
    .drop(
        columns="feature_priority"
    )
    .reset_index(
        drop=True
    )
)


if saf[
    "GeneID"
].duplicated().any():

    raise RuntimeError(
        "Duplicate historical PA14 locus tags remain "
        "after annotation cleanup."
    )


# ============================================================
# Basic sanity check
# ============================================================

if len(saf) < 5000:

    raise RuntimeError(
        "Unexpectedly few PA14 genes recovered: "
        + str(len(saf))
    )


saf_file = (
    REF_DIR
    / "NC_008463.1_PA14_locus_tags.saf"
)


saf.to_csv(
    saf_file,
    sep="\t",
    index=False,
)


print(
    "Historical PA14 gene features: "
    + str(len(saf))
)

print(
    "Example PA14 identifiers:"
)

print(
    ", ".join(
        saf[
            "GeneID"
        ]
        .head(10)
        .tolist()
    )
)


print(
    "FASTA: "
    + str(fasta_file)
)

print(
    "SAF:   "
    + str(saf_file)
)

# ============================================================
# Load PAO1 <-> PA14 orthology
# ============================================================

orth = pd.read_csv(
    ORTHOLOGY_FILE
)


query_col = "Locus Tag (Query)"
hit_col = "Locus Tag (Hit)"


if query_col not in orth.columns:
    raise RuntimeError(
        "Missing orthology Query locus column."
    )

if hit_col not in orth.columns:
    raise RuntimeError(
        "Missing orthology Hit locus column."
    )


pair_records = []


for _, row in orth.iterrows():

    query = str(
        row[query_col]
    ).strip()

    hit = str(
        row[hit_col]
    ).strip()


    query_is_pao1 = (
        re.fullmatch(
            r"PA\d{4}",
            query,
        )
        is not None
    )

    hit_is_pao1 = (
        re.fullmatch(
            r"PA\d{4}",
            hit,
        )
        is not None
    )

    query_is_pa14 = (
        re.fullmatch(
            r"PA14_\d+",
            query,
        )
        is not None
    )

    hit_is_pa14 = (
        re.fullmatch(
            r"PA14_\d+",
            hit,
        )
        is not None
    )


    if query_is_pao1 and hit_is_pa14:

        pao1 = query
        pa14 = hit

    elif hit_is_pao1 and query_is_pa14:

        pao1 = hit
        pa14 = query

    else:

        continue


    pair_records.append(
        {
            "canonical_pao1":
                pao1,

            "pa14_locus_tag":
                pa14,
        }
    )


pairs = (
    pd.DataFrame(
        pair_records
    )
    .drop_duplicates()
)


pao1_counts = (
    pairs[
        "canonical_pao1"
    ]
    .value_counts()
)

pa14_counts = (
    pairs[
        "pa14_locus_tag"
    ]
    .value_counts()
)


pairs[
    "pao1_unique"
] = (
    pairs[
        "canonical_pao1"
    ]
    .map(
        pao1_counts
    )
    == 1
)

pairs[
    "pa14_unique"
] = (
    pairs[
        "pa14_locus_tag"
    ]
    .map(
        pa14_counts
    )
    == 1
)


clean_pairs = (
    pairs.loc[
        pairs[
            "pao1_unique"
        ]
        &
        pairs[
            "pa14_unique"
        ],
        [
            "canonical_pao1",
            "pa14_locus_tag",
        ],
    ]
    .copy()
)


clean_pairs.to_csv(
    OUT_DIR
    / "PAO1_PA14_clean_1to1_mapping.csv",
    index=False,
)


print()
print("=" * 100)
print("STRICT PAO1 <-> PA14 ORTHOLOGY")
print("=" * 100)

print(
    "Parsed PAO1/PA14 pairs: "
    + str(len(pairs))
)

print(
    "Clean 1-to-1 pairs: "
    + str(len(clean_pairs))
)


# ============================================================
# Check PA14 tags exist in the reference
# ============================================================

reference_pa14 = set(
    saf[
        "GeneID"
    ].astype(str)
)

mapping_pa14 = set(
    clean_pairs[
        "pa14_locus_tag"
    ].astype(str)
)


mapping_missing_reference = (
    mapping_pa14
    - reference_pa14
)


print(
    "1-to-1 PA14 tags absent from reference: "
    + str(
        len(
            mapping_missing_reference
        )
    )
)


# ============================================================
# Check frozen 4,858-gene universe
# ============================================================

clean = pd.read_csv(
    CLEAN_GENE_FILE
)


gene_col = None


for candidate in [
    "canonical_pao1",
    "GeneID",
    "gene",
    "locus_tag",
]:

    if candidate in clean.columns:
        gene_col = candidate
        break


if gene_col is None:

    if clean.shape[1] == 1:
        gene_col = clean.columns[0]

    else:
        raise RuntimeError(
            "Could not identify frozen gene column."
        )


frozen_genes = set(
    clean[
        gene_col
    ].astype(str)
)

mapped_pao1 = set(
    clean_pairs[
        "canonical_pao1"
    ].astype(str)
)


frozen_mapped = (
    frozen_genes
    & mapped_pao1
)

frozen_missing = (
    frozen_genes
    - mapped_pao1
)


print()
print("=" * 100)
print("FROZEN CF-STATE GENE COVERAGE")
print("=" * 100)

print(
    "Frozen genes: "
    + str(len(frozen_genes))
)

print(
    "With strict PA14 1-to-1 ortholog: "
    + str(len(frozen_mapped))
)

print(
    "Missing strict PA14 ortholog: "
    + str(len(frozen_missing))
)


if len(frozen_missing) > 0:

    missing_file = (
        OUT_DIR
        / "frozen_genes_missing_PA14_ortholog.txt"
    )

    missing_file.write_text(
        "\n".join(
            sorted(
                frozen_missing
            )
        )
        + "\n"
    )


# ============================================================
# Download preparation
#
# ENA did not provide exactly two direct FASTQ files for at
# least one paired run. Do not guess how that run is packaged.
#
# Use SRA Toolkit instead. fasterq-dump --split-files will
# create R1 and R2 explicitly from each paired SRR.
# ============================================================

import shutil


SRA_DIR = Path(
    "data/raw/expanded_biofilm/GSE191103/sra"
)

TMP_DIR = Path(
    "data/raw/expanded_biofilm/GSE191103/fasterq_tmp"
)


SRA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Tool check
# ============================================================

print()
print("=" * 100)
print("DOWNLOAD TOOL CHECK")
print("=" * 100)


tool_names = [
    "prefetch",
    "fasterq-dump",
    "pigz",
]

tool_status = {}


for name in tool_names:

    path = shutil.which(
        name
    )

    tool_status[name] = path

    if path is None:

        print(
            name + ": MISSING"
        )

    else:

        print(
            name + ": " + path
        )


# ============================================================
# Save simple download manifest
# ============================================================

download_columns = [
    "sample",
    "state",
    "SRR",
    "library_layout",
]


for optional in [
    "spots",
    "bases",
    "size_MB",
]:

    if optional in manifest.columns:

        download_columns.append(
            optional
        )


download_manifest = manifest[
    download_columns
].copy()


download_manifest_file = (
    OUT_DIR
    / "GSE191103_PA14_download_manifest.csv"
)


download_manifest.to_csv(
    download_manifest_file,
    index=False,
)


# ============================================================
# Generate SRA Toolkit download + R1/R2 split script
# ============================================================

download_script = (
    OUT_DIR
    / "download_GSE191103_PA14_fastqs.sh"
)


shell = []


shell.append(
    "#!/usr/bin/env bash"
)

shell.append(
    "set -euo pipefail"
)

shell.append(
    ""
)

shell.append(
    'THREADS="${THREADS:-8}"'
)

shell.append(
    'FASTQ_DIR="'
    + str(FASTQ_DIR)
    + '"'
)

shell.append(
    'SRA_DIR="'
    + str(SRA_DIR)
    + '"'
)

shell.append(
    'TMP_DIR="'
    + str(TMP_DIR)
    + '"'
)

shell.append(
    ""
)

shell.append(
    'mkdir -p "$FASTQ_DIR" "$SRA_DIR" "$TMP_DIR"'
)

shell.append(
    ""
)

shell.append(
    'command -v prefetch >/dev/null 2>&1 || '
    '{ echo "ERROR: prefetch not found"; exit 1; }'
)

shell.append(
    'command -v fasterq-dump >/dev/null 2>&1 || '
    '{ echo "ERROR: fasterq-dump not found"; exit 1; }'
)

shell.append(
    'command -v pigz >/dev/null 2>&1 || '
    '{ echo "ERROR: pigz not found"; exit 1; }'
)

shell.append(
    ""
)


for _, row in manifest.iterrows():

    sample = str(
        row["sample"]
    )

    srr = str(
        row["SRR"]
    )

    r1_name = (
        sample
        + "_R1.fastq.gz"
    )

    r2_name = (
        sample
        + "_R2.fastq.gz"
    )


    shell.append(
        "echo"
    )

    shell.append(
        'echo "============================================================"'
    )

    shell.append(
        'echo "Processing '
        + sample
        + ' ('
        + srr
        + ')"'
    )

    shell.append(
        'echo "============================================================"'
    )


    # --------------------------------------------------------
    # Skip if both final files already exist
    # --------------------------------------------------------

    shell.append(
        'if [[ -s "$FASTQ_DIR/'
        + r1_name
        + '" && -s "$FASTQ_DIR/'
        + r2_name
        + '" ]]; then'
    )

    shell.append(
        '    echo "FASTQs already present; skipping."'
    )

    shell.append(
        "else"
    )


    # --------------------------------------------------------
    # Download SRA run
    # --------------------------------------------------------

    shell.append(
        '    prefetch '
        '--max-size u '
        '--output-directory "$SRA_DIR" '
        + srr
    )


    shell.append(
        '    SRA_FILE="$SRA_DIR/'
        + srr
        + '/'
        + srr
        + '.sra"'
    )


    shell.append(
        '    if [[ ! -s "$SRA_FILE" ]]; then'
    )

    shell.append(
        '        echo "ERROR: expected SRA file not found: $SRA_FILE"'
    )

    shell.append(
        "        exit 1"
    )

    shell.append(
        "    fi"
    )


    # --------------------------------------------------------
    # Remove stale temporary FASTQs
    # --------------------------------------------------------

    shell.append(
        '    rm -f '
        '"$TMP_DIR/'
        + srr
        + '_1.fastq" '
        '"$TMP_DIR/'
        + srr
        + '_2.fastq" '
        '"$TMP_DIR/'
        + srr
        + '_1.fastq.gz" '
        '"$TMP_DIR/'
        + srr
        + '_2.fastq.gz"'
    )


    # --------------------------------------------------------
    # Explicit paired-end splitting
    # --------------------------------------------------------

    shell.append(
        '    fasterq-dump '
        '--split-files '
        '-e "$THREADS" '
        '-t "$TMP_DIR" '
        '-O "$TMP_DIR" '
        '"$SRA_FILE"'
    )


    # --------------------------------------------------------
    # Verify both mates exist
    # --------------------------------------------------------

    shell.append(
        '    test -s "$TMP_DIR/'
        + srr
        + '_1.fastq" || '
        '{ echo "ERROR: R1 was not created"; exit 1; }'
    )

    shell.append(
        '    test -s "$TMP_DIR/'
        + srr
        + '_2.fastq" || '
        '{ echo "ERROR: R2 was not created"; exit 1; }'
    )


    # --------------------------------------------------------
    # Compress
    # --------------------------------------------------------

    shell.append(
        '    pigz '
        '-f '
        '-p "$THREADS" '
        '"$TMP_DIR/'
        + srr
        + '_1.fastq" '
        '"$TMP_DIR/'
        + srr
        + '_2.fastq"'
    )


    # --------------------------------------------------------
    # Rename to our CF-State sample names
    # --------------------------------------------------------

    shell.append(
        '    mv "$TMP_DIR/'
        + srr
        + '_1.fastq.gz" '
        '"$FASTQ_DIR/'
        + r1_name
        + '"'
    )

    shell.append(
        '    mv "$TMP_DIR/'
        + srr
        + '_2.fastq.gz" '
        '"$FASTQ_DIR/'
        + r2_name
        + '"'
    )


    shell.append(
        "fi"
    )


    # --------------------------------------------------------
    # Final sanity check
    # --------------------------------------------------------

    shell.append(
        'test -s "$FASTQ_DIR/'
        + r1_name
        + '"'
    )

    shell.append(
        'test -s "$FASTQ_DIR/'
        + r2_name
        + '"'
    )


    shell.append(
        'ls -lh '
        '"$FASTQ_DIR/'
        + r1_name
        + '" '
        '"$FASTQ_DIR/'
        + r2_name
        + '"'
    )

    shell.append(
        ""
    )


download_script.write_text(
    "\n".join(
        shell
    )
    + "\n"
)


download_script.chmod(
    0o755
)


# ============================================================
# Final summary
# ============================================================

mapping_file = (
    OUT_DIR
    / "PAO1_PA14_clean_1to1_mapping.csv"
)


print()
print("=" * 100)
print("AUDIT 31 COMPLETE")
print("=" * 100)


print(
    "PA14 FASTA: "
    + str(fasta_file)
)

print(
    "PA14 SAF: "
    + str(saf_file)
)

print(
    "Orthology map: "
    + str(mapping_file)
)

print(
    "Download manifest: "
    + str(download_manifest_file)
)

print(
    "Download script: "
    + str(download_script)
)


missing_tools = [
    name
    for name in tool_names
    if tool_status[name] is None
]


print()


if len(missing_tools) > 0:

    print(
        "Missing tools: "
        + ", ".join(
            missing_tools
        )
    )

    print(
        "Install them before running "
        "the download script."
    )

else:

    print(
        "All download tools are available."
    )


print()

print(
    "FASTQs have NOT been downloaded yet."
)