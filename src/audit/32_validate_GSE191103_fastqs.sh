#!/usr/bin/env bash

set -euo pipefail

FASTQ_DIR="data/raw/expanded_biofilm/GSE191103/fastq"

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
echo "CF-STATE — Audit 32: validate GSE191103 FASTQs"
echo "===================================================================================================="
echo

for SAMPLE in "${SAMPLES[@]}"; do

    R1="${FASTQ_DIR}/${SAMPLE}_R1.fastq.gz"
    R2="${FASTQ_DIR}/${SAMPLE}_R2.fastq.gz"

    echo "Checking ${SAMPLE}"

    if [[ ! -s "${R1}" ]]; then
        echo "ERROR: missing or empty ${R1}"
        exit 1
    fi

    if [[ ! -s "${R2}" ]]; then
        echo "ERROR: missing or empty ${R2}"
        exit 1
    fi

    gzip -t "${R1}"
    gzip -t "${R2}"

    R1_LINES=$(pigz -dc "${R1}" | wc -l)
    R2_LINES=$(pigz -dc "${R2}" | wc -l)

    if (( R1_LINES % 4 != 0 )); then
        echo "ERROR: R1 line count is not divisible by 4"
        exit 1
    fi

    if (( R2_LINES % 4 != 0 )); then
        echo "ERROR: R2 line count is not divisible by 4"
        exit 1
    fi

    R1_READS=$((R1_LINES / 4))
    R2_READS=$((R2_LINES / 4))

    echo "  R1 reads: ${R1_READS}"
    echo "  R2 reads: ${R2_READS}"

    if [[ "${R1_READS}" -ne "${R2_READS}" ]]; then
        echo "ERROR: R1/R2 read counts differ"
        exit 1
    fi

    echo "  PASS"
    echo

done

echo "===================================================================================================="
echo "AUDIT 32 COMPLETE"
echo "===================================================================================================="
echo
echo "All 12 FASTQ files are valid."
echo "All six R1/R2 pairs have matching read counts."
echo
