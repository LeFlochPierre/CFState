#!/usr/bin/env bash

set -euo pipefail


# ============================================================
# Settings
# ============================================================

THREADS=16

FASTQ_DIR="data/raw/expanded_biofilm/GSE299861/fastq"

REF_DIR="references/mappings/GSE299861/canonical_PAO1"

FASTA="${REF_DIR}/NC_002516.2.fasta"

SAF="${REF_DIR}/NC_002516.2_PA_locus_tags.saf"

INDEX_DIR="${REF_DIR}/bowtie2_index"

INDEX_PREFIX="${INDEX_DIR}/NC_002516.2"


OUT_DIR="results/audit/expanded_biofilm/GSE299861/reprocessing"

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
    "biofilm_PAO1_1"
    "biofilm_PAO1_2"
    "biofilm_PAO1_3"
    "planktonic_PAO1_1"
    "planktonic_PAO1_2"
    "planktonic_PAO1_3"
)


echo
echo "===================================================================================================="
echo "CF-STATE — Audit 26: align and count GSE299861 PAO1"
echo "===================================================================================================="
echo

echo "Reference:"
echo "  ${FASTA}"

echo

echo "Annotation:"
echo "  ${SAF}"

echo

echo "Threads:"
echo "  ${THREADS}"

echo


# ============================================================
# Build Bowtie2 index
# ============================================================

if [[ ! -f "${INDEX_PREFIX}.1.bt2" && ! -f "${INDEX_PREFIX}.1.bt2l" ]]; then

    echo "===================================================================================================="
    echo "BUILDING BOWTIE2 INDEX"
    echo "===================================================================================================="

    bowtie2-build \
        --threads "${THREADS}" \
        "${FASTA}" \
        "${INDEX_PREFIX}"

else

    echo "Bowtie2 index already exists."

fi


# ============================================================
# Align each sample
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


    if [[ ! -f "${R1}" ]]; then

        echo "ERROR: missing ${R1}"
        exit 1

    fi


    if [[ ! -f "${R2}" ]]; then

        echo "ERROR: missing ${R2}"
        exit 1

    fi


    if [[ ! -f "${BAM}" ]]; then

        bowtie2 \
            --very-sensitive \
            --threads "${THREADS}" \
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
        "overall alignment rate|aligned concordantly exactly 1 time|aligned concordantly >1 times" \
        "${LOG}" \
        || true

done


# ============================================================
# Collect BAM list
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
# Test:
#   -s 0 = unstranded
#   -s 1 = stranded
#   -s 2 = reversely stranded
#
# Paired-end:
#   -p --countReadPairs
#
# -B:
#   both ends must be successfully aligned
#
# -C:
#   exclude chimeric fragments
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
# Python summary
# ============================================================

python - <<'PY'

from pathlib import Path
import re

import numpy as np
import pandas as pd


OUT_DIR = Path(
    "results/audit/expanded_biofilm/"
    "GSE299861/reprocessing"
)

LOG_DIR = OUT_DIR / "logs"

COUNT_DIR = OUT_DIR / "counts"

QC_DIR = OUT_DIR / "qc"


SAMPLES = [
    "biofilm_PAO1_1",
    "biofilm_PAO1_2",
    "biofilm_PAO1_3",
    "planktonic_PAO1_1",
    "planktonic_PAO1_2",
    "planktonic_PAO1_3",
]


# ============================================================
# Bowtie2 alignment summary
# ============================================================

alignment_records = []


for sample in SAMPLES:

    log_file = (
        LOG_DIR
        / f"{sample}.bowtie2.log"
    )

    text = log_file.read_text()


    match = re.search(
        r"([\d.]+)% overall alignment rate",
        text,
    )


    alignment_rate = (
        float(
            match.group(1)
        )
        if match
        else np.nan
    )


    alignment_records.append(
        {
            "sample":
                sample,

            "state":
                (
                    "biofilm"
                    if sample.startswith(
                        "biofilm"
                    )
                    else "planktonic"
                ),

            "bowtie2_alignment_rate_percent":
                alignment_rate,
        }
    )


alignment_df = pd.DataFrame(
    alignment_records
)


alignment_df.to_csv(
    QC_DIR
    / "alignment_summary.csv",
    index=False,
)


# ============================================================
# featureCounts summaries
# ============================================================

strand_records = []


for strand in [
    0,
    1,
    2,
]:

    summary_file = (
        COUNT_DIR
        / f"counts_s{strand}.txt.summary"
    )


    summary = pd.read_csv(
        summary_file,
        sep="\t",
        index_col=0,
    )


    for column in summary.columns:

        sample_name = (
            Path(column)
            .name
            .replace(
                ".sorted.bam",
                "",
            )
        )


        total = float(
            summary[
                column
            ].sum()
        )


        assigned = float(
            summary.loc[
                "Assigned",
                column,
            ]
        )


        assigned_fraction = (
            assigned / total
            if total > 0
            else np.nan
        )


        strand_records.append(
            {
                "strand_setting":
                    strand,

                "sample":
                    sample_name,

                "state":
                    (
                        "biofilm"
                        if sample_name.startswith(
                            "biofilm"
                        )
                        else "planktonic"
                    ),

                "total_fragments":
                    int(total),

                "assigned_fragments":
                    int(assigned),

                "assigned_fraction":
                    assigned_fraction,
            }
        )


strand_df = pd.DataFrame(
    strand_records
)


strand_df.to_csv(
    QC_DIR
    / "strand_assignment_summary.csv",
    index=False,
)


# ============================================================
# Median assignment by strand setting
# ============================================================

median_summary = (
    strand_df
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


median_summary.to_csv(
    QC_DIR
    / "strand_assignment_overall.csv",
    index=False,
)


# ============================================================
# Print results
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

display = (
    strand_df.copy()
)

display[
    "assigned_percent"
] = (
    display[
        "assigned_fraction"
    ]
    * 100
)


print(
    display[
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
    median_summary.to_string(
        index=False
    )
)


# ============================================================
# Basic interpretation
# ============================================================

s1 = float(
    median_summary.loc[
        median_summary[
            "strand_setting"
        ] == 1,
        "median",
    ].iloc[0]
)

s2 = float(
    median_summary.loc[
        median_summary[
            "strand_setting"
        ] == 2,
        "median",
    ].iloc[0]
)


print()
print("=" * 100)
print("PRELIMINARY STRAND ORIENTATION")
print("=" * 100)


if s1 > (
    s2 * 3
):

    print(
        "Strand setting 1 is much higher "
        "than setting 2."
    )

    print(
        "Likely forward-stranded library."
    )


elif s2 > (
    s1 * 3
):

    print(
        "Strand setting 2 is much higher "
        "than setting 1."
    )

    print(
        "Likely reverse-stranded library."
    )


else:

    print(
        "Strand settings 1 and 2 are not "
        "strongly separated."
    )

    print(
        "Library orientation should be "
        "investigated further before choosing."
    )


print()
print(
    "IMPORTANT: do not choose strandedness "
    "simply because -s 0 has the highest "
    "assignment rate."
)

print(
    "The useful comparison is primarily "
    "-s 1 versus -s 2."
)

print()

PY


echo
echo "===================================================================================================="
echo "AUDIT 26 COMPLETE"
echo "===================================================================================================="

echo
echo "Outputs:"
echo "  ${OUT_DIR}"
echo
