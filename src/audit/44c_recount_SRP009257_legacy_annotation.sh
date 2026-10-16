et -euo pipefail

THREADS=16

BAM_DIR="results/audit/expanded_biofilm/SRP009257/reprocessing/bam"

SAF="references/mappings/SRP009257/PA14_legacy/CP000438.1_legacy_PA14_locus_tags.saf"

OUT_DIR="results/audit/expanded_biofilm/SRP009257/reprocessing/counts"

OUT_FILE="${OUT_DIR}/counts_legacy_s1.txt"

mkdir -p "${OUT_DIR}"


SAMPLES=(
    "planktonic_P4_1"
    "planktonic_P4_2"
    "planktonic_P12_1"
    "planktonic_P12_2"
    "biofilm_B24_1"
    "biofilm_B24_2"
    "biofilm_B48_1"
    "biofilm_B48_2"
)


BAMS=()

for SAMPLE in "${SAMPLES[@]}"; do

    BAM="${BAM_DIR}/${SAMPLE}.sorted.bam"

    if [[ ! -s "${BAM}" ]]; then
        echo "ERROR: missing BAM: ${BAM}"
        exit 1
    fi

    BAMS+=("${BAM}")

done


if [[ ! -s "${SAF}" ]]; then
    echo "ERROR: missing legacy SAF: ${SAF}"
    exit 1
fi


echo
echo "===================================================================================================="
echo "CF-STATE — Audit 44c: recount SRP009257 using legacy PA14 annotation"
echo "===================================================================================================="
echo


featureCounts \
    -T "${THREADS}" \
    -F SAF \
    -a "${SAF}" \
    -o "${OUT_FILE}" \
    -s 1 \
    --read2pos 5 \
    "${BAMS[@]}"


python - <<'PY'

from pathlib import Path

import pandas as pd


COUNT_FILE = Path(
    "results/audit/expanded_biofilm/SRP009257/"
    "reprocessing/counts/counts_legacy_s1.txt"
)

SUMMARY_FILE = Path(
    str(COUNT_FILE) + ".summary"
)


counts = pd.read_csv(
    COUNT_FILE,
    sep="\t",
    comment="#",
)


if "Geneid" not in counts.columns:
    raise RuntimeError(
        "Geneid missing from legacy count table."
    )


print()
print("=" * 100)
print("LEGACY COUNT TABLE")
print("=" * 100)

print(
    "Genes/features counted: "
    + str(len(counts))
)

print(
    "Unique Geneid values: "
    + str(
        counts["Geneid"].nunique()
    )
)


if counts["Geneid"].duplicated().any():

    raise RuntimeError(
        "Duplicate Geneid values detected."
    )


summary = pd.read_csv(
    SUMMARY_FILE,
    sep="\t",
    index_col=0,
)


print()
print("=" * 100)
print("ASSIGNMENT SUMMARY")
print("=" * 100)


records = []


for column in summary.columns:

    sample = (
        Path(column)
        .name
        .replace(
            ".sorted.bam",
            "",
        )
    )

    total = int(
        summary[column].sum()
    )

    assigned = int(
        summary.loc[
            "Assigned",
            column,
        ]
    )

    fraction = (
        assigned / total
        if total > 0
        else float("nan")
    )

    records.append(
        {
            "sample": sample,
            "total_unique_mapped": total,
            "assigned_legacy_genes": assigned,
            "assigned_percent": 100.0 * fraction,
        }
    )


result = pd.DataFrame(
    records
)


print(
    result.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 44c COMPLETE")
print("=" * 100)

print(
    "Count file: "
    + str(COUNT_FILE)
)

print()
PY
