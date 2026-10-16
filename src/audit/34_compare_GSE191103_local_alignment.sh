#!/usr/bin/env bash

set -euo pipefail

THREADS=16

FASTQ_DIR="data/raw/expanded_biofilm/GSE191103/fastq"

REF_DIR="references/mappings/GSE191103/PA14"

INDEX_PREFIX="${REF_DIR}/bowtie2_index/NC_008463.1"

OUT_DIR="results/audit/expanded_biofilm/GSE191103/local_alignment_test"

BAM_DIR="${OUT_DIR}/bam"
LOG_DIR="${OUT_DIR}/logs"

mkdir -p "${BAM_DIR}" "${LOG_DIR}"


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
echo "CF-STATE — Audit 34: documented local-alignment test"
echo "===================================================================================================="
echo


for SAMPLE in "${SAMPLES[@]}"; do

    echo
    echo "===================================================================================================="
    echo "ALIGNING: ${SAMPLE}"
    echo "===================================================================================================="

    R1="${FASTQ_DIR}/${SAMPLE}_R1.fastq.gz"
    R2="${FASTQ_DIR}/${SAMPLE}_R2.fastq.gz"

    BAM="${BAM_DIR}/${SAMPLE}.sorted.bam"
    LOG="${LOG_DIR}/${SAMPLE}.bowtie2.log"


    bowtie2 \
        --very-sensitive-local \
        --no-mixed \
        --fr \
        --no-unal \
        -p "${THREADS}" \
        -x "${INDEX_PREFIX}" \
        -1 "${R1}" \
        -2 "${R2}" \
        2> "${LOG}" \
    | samtools sort \
        -@ "${THREADS}" \
        -o "${BAM}" \
        -


    samtools index \
        -@ "${THREADS}" \
        "${BAM}"


    echo
    echo "Bowtie2 summary:"

    grep -E \
        "aligned concordantly exactly 1 time|aligned concordantly >1 times|overall alignment rate" \
        "${LOG}" \
        || true

done


python - <<'PY'

from pathlib import Path
import re
import pandas as pd


OLD_LOG_DIR = Path(
    "results/audit/expanded_biofilm/"
    "GSE191103/reprocessing/logs"
)

NEW_LOG_DIR = Path(
    "results/audit/expanded_biofilm/"
    "GSE191103/local_alignment_test/logs"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/"
    "GSE191103/local_alignment_test"
)


SAMPLES = [
    "biofilm_PA14_1",
    "biofilm_PA14_2",
    "biofilm_PA14_3",
    "planktonic_PA14_1",
    "planktonic_PA14_2",
    "planktonic_PA14_3",
]


def get_rate(path):

    text = path.read_text()

    match = re.search(
        r"([\d.]+)% overall alignment rate",
        text,
    )

    if match is None:
        raise RuntimeError(
            "Could not find alignment rate in "
            + str(path)
        )

    return float(
        match.group(1)
    )


records = []


for sample in SAMPLES:

    old_rate = get_rate(
        OLD_LOG_DIR
        / f"{sample}.bowtie2.log"
    )

    new_rate = get_rate(
        NEW_LOG_DIR
        / f"{sample}.bowtie2.log"
    )

    if sample.startswith("biofilm"):
        state = "biofilm"
    else:
        state = "planktonic"

    improvement = (
        new_rate
        - old_rate
    )

    records.append(
        {
            "sample": sample,
            "state": state,
            "global_alignment_percent": old_rate,
            "local_alignment_percent": new_rate,
            "improvement_percentage_points": improvement,
        }
    )


df = pd.DataFrame(
    records
)


df.to_csv(
    OUT_DIR
    / "global_vs_local_alignment.csv",
    index=False,
)


print()
print("=" * 100)
print("GLOBAL VS DOCUMENTED LOCAL ALIGNMENT")
print("=" * 100)

print(
    df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("STATE SUMMARY")
print("=" * 100)


summary = (
    df
    .groupby("state")[
        [
            "global_alignment_percent",
            "local_alignment_percent",
            "improvement_percentage_points",
        ]
    ]
    .mean()
)


print(
    summary.round(2).to_string()
)


print()
print("=" * 100)
print("AUDIT 34 COMPLETE")
print("=" * 100)
print()

PY
