#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# Settings
# ============================================================

THREADS=16

FASTQ_DIR="data/raw/expanded_biofilm/GSE191103/fastq"

REF_DIR="references/mappings/GSE191103/PA14"

FASTA="${REF_DIR}/NC_008463.1.fasta"

SAF="${REF_DIR}/NC_008463.1_PA14_locus_tags.saf"

INDEX_DIR="${REF_DIR}/bowtie2_index"

INDEX_PREFIX="${INDEX_DIR}/NC_008463.1"


OUT_DIR="results/audit/expanded_biofilm/GSE191103/reprocessing"

BAM_DIR="${OUT_DIR}/bam"

LOG_DIR="${OUT_DIR}/logs"

COUNT_DIR="${OUT_DIR}/counts"

QC_DIR="${OUT_DIR}/qc"


mkdir -p \
    "${INDEX_DIR}" \
    "${BAM_DIR}" \
    "${LOG_DIR}" \
    "${COUNT_DIR}" \
    "${QC_DIR}"


SAMPLES=(
    "biofilm_PA14_1"
    "biofilm_PA14_2"
    "biofilm_PA14_3"
    "planktonic_PA14_1"
    "planktonic_PA14_2"
    "planktonic_PA14_3"
)


echo
echo "===================================================================================================="
echo "CF-STATE — Audit 33: align and count GSE191103 PA14"
echo "===================================================================================================="
echo


# ============================================================
# Input checks
# ============================================================

if [[ ! -s "${FASTA}" ]]; then
    echo "ERROR: missing FASTA: ${FASTA}"
    exit 1
fi

if [[ ! -s "${SAF}" ]]; then
    echo "ERROR: missing SAF: ${SAF}"
    exit 1
fi


for SAMPLE in "${SAMPLES[@]}"; do

    R1="${FASTQ_DIR}/${SAMPLE}_R1.fastq.gz"
    R2="${FASTQ_DIR}/${SAMPLE}_R2.fastq.gz"

    if [[ ! -s "${R1}" ]]; then
        echo "ERROR: missing ${R1}"
        exit 1
    fi

    if [[ ! -s "${R2}" ]]; then
        echo "ERROR: missing ${R2}"
        exit 1
    fi

done


# ============================================================
# Tool checks
# ============================================================

for TOOL in bowtie2 bowtie2-build samtools featureCounts; do

    if ! command -v "${TOOL}" >/dev/null 2>&1; then
        echo "ERROR: ${TOOL} not found"
        exit 1
    fi

done


# ============================================================
# Build Bowtie2 index
# ============================================================

if [[ ! -f "${INDEX_PREFIX}.1.bt2" && ! -f "${INDEX_PREFIX}.1.bt2l" ]]; then

    echo
    echo "===================================================================================================="
    echo "BUILDING PA14 BOWTIE2 INDEX"
    echo "===================================================================================================="

    bowtie2-build \
        "${FASTA}" \
        "${INDEX_PREFIX}"

else

    echo
    echo "Bowtie2 index already exists."

fi


# ============================================================
# Align samples
# ============================================================

for SAMPLE in "${SAMPLES[@]}"; do

    echo
    echo "===================================================================================================="
    echo "ALIGNING: ${SAMPLE}"
    echo "===================================================================================================="

    R1="${FASTQ_DIR}/${SAMPLE}_R1.fastq.gz"
    R2="${FASTQ_DIR}/${SAMPLE}_R2.fastq.gz"

    BAM="${BAM_DIR}/${SAMPLE}.sorted.bam"

    LOG="${LOG_DIR}/${SAMPLE}.bowtie2.log"

    FLAGSTAT="${QC_DIR}/${SAMPLE}.flagstat.txt"


    if [[ ! -s "${BAM}" ]]; then

        bowtie2 \
            --very-sensitive \
            -p "${THREADS}" \
            -x "${INDEX_PREFIX}" \
            -1 "${R1}" \
            -2 "${R2}" \
            2> "${LOG}" \
        | samtools sort \
            -@ "${THREADS}" \
            -o "${BAM}" \
            -

    else

        echo "BAM already exists; skipping alignment."

    fi


    samtools index \
        -@ "${THREADS}" \
        "${BAM}"


    samtools flagstat \
        -@ "${THREADS}" \
        "${BAM}" \
        > "${FLAGSTAT}"


    echo
    echo "Bowtie2 summary:"

    grep -E \
        "aligned concordantly exactly 1 time|aligned concordantly >1 times|overall alignment rate" \
        "${LOG}" \
        || true

done


# ============================================================
# BAM list
# ============================================================

BAMS=()

for SAMPLE in "${SAMPLES[@]}"; do

    BAMS+=(
        "${BAM_DIR}/${SAMPLE}.sorted.bam"
    )

done


# ============================================================
# featureCounts
#
# Test all three possibilities:
#
# -s 0 = unstranded
# -s 1 = forward stranded
# -s 2 = reverse stranded
# ============================================================

for STRAND in 0 1 2; do

    echo
    echo "===================================================================================================="
    echo "FEATURECOUNTS — STRAND SETTING ${STRAND}"
    echo "===================================================================================================="

    featureCounts \
        -T "${THREADS}" \
        -F SAF \
        -a "${SAF}" \
        -o "${COUNT_DIR}/counts_s${STRAND}.txt" \
        -p \
        --countReadPairs \
        -B \
        -C \
        -s "${STRAND}" \
        "${BAMS[@]}"

done


# ============================================================
# Python QC summary
# ============================================================

python - <<'PY'

from pathlib import Path
import re

import numpy as np
import pandas as pd


OUT_DIR = Path(
    "results/audit/expanded_biofilm/"
    "GSE191103/reprocessing"
)

LOG_DIR = OUT_DIR / "logs"
COUNT_DIR = OUT_DIR / "counts"
QC_DIR = OUT_DIR / "qc"


SAMPLES = [
    "biofilm_PA14_1",
    "biofilm_PA14_2",
    "biofilm_PA14_3",
    "planktonic_PA14_1",
    "planktonic_PA14_2",
    "planktonic_PA14_3",
]


# ============================================================
# Alignment summary
# ============================================================

alignment_records = []


for sample in SAMPLES:

    log_file = LOG_DIR / f"{sample}.bowtie2.log"

    if not log_file.exists():
        raise RuntimeError(
            f"Missing Bowtie2 log: {log_file}"
        )

    text = log_file.read_text()

    match = re.search(
        r"([\d.]+)% overall alignment rate",
        text,
    )

    if match is None:
        alignment_rate = np.nan
    else:
        alignment_rate = float(
            match.group(1)
        )

    state = (
        "biofilm"
        if sample.startswith("biofilm")
        else "planktonic"
    )

    alignment_records.append(
        {
            "sample": sample,
            "state": state,
            "alignment_percent": alignment_rate,
        }
    )


alignment_df = pd.DataFrame(
    alignment_records
)


alignment_df.to_csv(
    QC_DIR / "alignment_summary.csv",
    index=False,
)


# ============================================================
# featureCounts summaries
# ============================================================

assignment_records = []


for strand in [0, 1, 2]:

    summary_file = (
        COUNT_DIR
        / f"counts_s{strand}.txt.summary"
    )

    if not summary_file.exists():
        raise RuntimeError(
            f"Missing featureCounts summary: {summary_file}"
        )

    summary = pd.read_csv(
        summary_file,
        sep="\t",
        index_col=0,
    )

    for column in summary.columns:

        sample = (
            Path(column)
            .name
            .replace(
                ".sorted.bam",
                "",
            )
        )

        total = float(
            summary[column].sum()
        )

        assigned = float(
            summary.loc[
                "Assigned",
                column,
            ]
        )

        if total > 0:
            assigned_fraction = assigned / total
        else:
            assigned_fraction = np.nan

        state = (
            "biofilm"
            if sample.startswith("biofilm")
            else "planktonic"
        )

        assignment_records.append(
            {
                "strand_setting": strand,
                "sample": sample,
                "state": state,
                "total_fragments": int(total),
                "assigned_fragments": int(assigned),
                "assigned_fraction": assigned_fraction,
            }
        )


assignment_df = pd.DataFrame(
    assignment_records
)


assignment_df.to_csv(
    QC_DIR
    / "strand_assignment_summary.csv",
    index=False,
)


# ============================================================
# Aggregate strandedness
# ============================================================

strand_summary = (
    assignment_df
    .groupby(
        "strand_setting"
    )[
        "assigned_fraction"
    ]
    .agg(
        [
            "median",
            "mean",
            "min",
            "max",
        ]
    )
    .reset_index()
)


strand_summary.to_csv(
    QC_DIR
    / "strand_assignment_overall.csv",
    index=False,
)


# ============================================================
# Count table sanity checks
# ============================================================

count_checks = []


for strand in [0, 1, 2]:

    count_file = (
        COUNT_DIR
        / f"counts_s{strand}.txt"
    )

    counts = pd.read_csv(
        count_file,
        sep="\t",
        comment="#",
    )

    if "Geneid" not in counts.columns:
        raise RuntimeError(
            f"Geneid missing from {count_file}"
        )

    n_genes = len(counts)

    n_unique = counts["Geneid"].nunique()

    count_checks.append(
        {
            "strand_setting": strand,
            "genes": n_genes,
            "unique_genes": n_unique,
        }
    )


count_check_df = pd.DataFrame(
    count_checks
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print("ALIGNMENT SUMMARY")
print("=" * 100)

print(
    alignment_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("STRANDEDNESS — PER SAMPLE")
print("=" * 100)


display_df = assignment_df.copy()

display_df[
    "assigned_percent"
] = (
    100.0
    * display_df[
        "assigned_fraction"
    ]
)


print(
    display_df[
        [
            "strand_setting",
            "sample",
            "state",
            "total_fragments",
            "assigned_fragments",
            "assigned_percent",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("STRANDEDNESS — OVERALL")
print("=" * 100)

print(
    strand_summary.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("COUNT TABLE CHECK")
print("=" * 100)

print(
    count_check_df.to_string(
        index=False
    )
)


# ============================================================
# Simple strandedness interpretation
# ============================================================

median_s0 = float(
    strand_summary.loc[
        strand_summary[
            "strand_setting"
        ] == 0,
        "median",
    ].iloc[0]
)

median_s1 = float(
    strand_summary.loc[
        strand_summary[
            "strand_setting"
        ] == 1,
        "median",
    ].iloc[0]
)

median_s2 = float(
    strand_summary.loc[
        strand_summary[
            "strand_setting"
        ] == 2,
        "median",
    ].iloc[0]
)


print()
print("=" * 100)
print("PRELIMINARY STRANDEDNESS INTERPRETATION")
print("=" * 100)

print(
    f"-s 0 median assigned: "
    f"{100.0 * median_s0:.2f}%"
)

print(
    f"-s 1 median assigned: "
    f"{100.0 * median_s1:.2f}%"
)

print(
    f"-s 2 median assigned: "
    f"{100.0 * median_s2:.2f}%"
)


if median_s1 > 3.0 * median_s2:

    print()
    print(
        "Likely forward-stranded library (-s 1)."
    )

elif median_s2 > 3.0 * median_s1:

    print()
    print(
        "Likely reverse-stranded library (-s 2)."
    )

else:

    print()
    print(
        "No strong forward/reverse separation."
    )

    print(
        "The library may be unstranded, or further "
        "inspection may be required."
    )


print()
print(
    "Do not freeze the PAO1 matrix yet."
)

print(
    "First confirm alignment quality and strandedness."
)

PY


echo
echo "===================================================================================================="
echo "AUDIT 33 COMPLETE"
echo "===================================================================================================="
echo

echo "Outputs:"
echo "  ${OUT_DIR}"
echo
