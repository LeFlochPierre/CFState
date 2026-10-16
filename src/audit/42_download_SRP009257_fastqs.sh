!/usr/bin/env bash

set -euo pipefail

MANIFEST="results/audit/expanded_biofilm/SRP009257/SRP009257_ENA_manifest.tsv"
FASTQ_DIR="data/raw/expanded_biofilm/SRP009257/fastq"
OUT_DIR="results/audit/expanded_biofilm/SRP009257"

mkdir -p "${FASTQ_DIR}" "${OUT_DIR}"

command -v curl >/dev/null 2>&1 || { echo "ERROR: curl not found"; exit 1; }
command -v gzip >/dev/null 2>&1 || { echo "ERROR: gzip not found"; exit 1; }
command -v md5sum >/dev/null 2>&1 || { echo "ERROR: md5sum not found"; exit 1; }

if [[ ! -s "${MANIFEST}" ]]; then
    echo "ERROR: manifest not found: ${MANIFEST}"
    exit 1
fi


python - <<'PY'

from pathlib import Path
import pandas as pd


manifest_file = Path(
    "results/audit/expanded_biofilm/SRP009257/SRP009257_ENA_manifest.tsv"
)

out_file = Path(
    "results/audit/expanded_biofilm/SRP009257/SRP009257_canonical_manifest.tsv"
)


sample_map = {
    "P4-1": ("planktonic_P4_1", "planktonic", "P4"),
    "P4-2": ("planktonic_P4_2", "planktonic", "P4"),
    "P12-1": ("planktonic_P12_1", "planktonic", "P12"),
    "P12-2": ("planktonic_P12_2", "planktonic", "P12"),
    "B24-1": ("biofilm_B24_1", "biofilm", "B24"),
    "B24-2": ("biofilm_B24_2", "biofilm", "B24"),
    "B48-1": ("biofilm_B48_1", "biofilm", "B48"),
    "B48-2": ("biofilm_B48_2", "biofilm", "B48"),
}


manifest = pd.read_csv(
    manifest_file,
    sep="\t",
    dtype=str,
).fillna("")


required = [
    "run_accession",
    "sample_title",
    "library_layout",
    "read_count",
    "base_count",
    "fastq_ftp",
    "fastq_md5",
    "fastq_bytes",
]


for column in required:
    if column not in manifest.columns:
        raise RuntimeError(
            "Missing manifest column: " + column
        )


if len(manifest) != 8:
    raise RuntimeError(
        "Expected exactly 8 runs, found "
        + str(len(manifest))
    )


records = []


for _, row in manifest.iterrows():

    title = row["sample_title"].strip()

    if title not in sample_map:
        raise RuntimeError(
            "Unexpected sample_title: " + title
        )

    if row["library_layout"].strip() != "SINGLE":
        raise RuntimeError(
            "Expected SINGLE layout for " + title
        )

    sample, state, condition = sample_map[title]

    read_count = int(
        row["read_count"]
    )

    base_count = int(
        row["base_count"]
    )

    read_length = (
        base_count
        / read_count
    )

    url = row["fastq_ftp"].strip()

    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url

    records.append(
        {
            "sample": sample,
            "state": state,
            "condition": condition,
            "sample_title": title,
            "run_accession": row["run_accession"].strip(),
            "library_layout": row["library_layout"].strip(),
            "read_count": read_count,
            "base_count": base_count,
            "read_length": read_length,
            "fastq_url": url,
            "fastq_md5": row["fastq_md5"].strip(),
            "fastq_bytes": row["fastq_bytes"].strip(),
        }
    )


canonical = pd.DataFrame(
    records
)


canonical = canonical.sort_values(
    [
        "condition",
        "sample",
    ]
).reset_index(
    drop=True
)


expected_titles = set(
    sample_map
)

observed_titles = set(
    canonical["sample_title"]
)


if observed_titles != expected_titles:
    raise RuntimeError(
        "The eight expected biological samples "
        "were not recovered exactly."
    )


canonical.to_csv(
    out_file,
    sep="\t",
    index=False,
)


print()
print("=" * 100)
print("CANONICAL SAMPLE MANIFEST")
print("=" * 100)

print(
    canonical[
        [
            "sample",
            "state",
            "condition",
            "run_accession",
            "read_count",
            "read_length",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Manifest written to: "
    + str(out_file)
)

PY


CANONICAL_MANIFEST="${OUT_DIR}/SRP009257_canonical_manifest.tsv"


echo
echo "===================================================================================================="
echo "DOWNLOADING AND VALIDATING FASTQS"
echo "===================================================================================================="
echo


while IFS=$'\t' read -r \
    sample \
    state \
    condition \
    sample_title \
    run_accession \
    library_layout \
    read_count \
    base_count \
    read_length \
    fastq_url \
    fastq_md5 \
    fastq_bytes

do

    if [[ "${sample}" == "sample" ]]; then
        continue
    fi

    OUT_FILE="${FASTQ_DIR}/${sample}.fastq.gz"

    echo "${sample} (${run_accession})"


    if [[ ! -s "${OUT_FILE}" ]]; then

        curl \
            -4 \
            -L \
            --fail \
            --retry 3 \
            --retry-delay 2 \
            "${fastq_url}" \
            -o "${OUT_FILE}"

    else

        echo "  FASTQ already exists; skipping download."

    fi


    gzip -t "${OUT_FILE}"


    if [[ -n "${fastq_md5}" ]]; then

        ACTUAL_MD5=$(
            md5sum "${OUT_FILE}" |
            awk '{print $1}'
        )

        if [[ "${ACTUAL_MD5}" != "${fastq_md5}" ]]; then

            echo "ERROR: MD5 mismatch for ${sample}"
            echo "Expected: ${fastq_md5}"
            echo "Actual:   ${ACTUAL_MD5}"
            exit 1

        fi

    fi


    if [[ -n "${fastq_bytes}" ]]; then

        ACTUAL_BYTES=$(
            stat -c%s "${OUT_FILE}"
        )

        if [[ "${ACTUAL_BYTES}" -ne "${fastq_bytes}" ]]; then

            echo "ERROR: byte-size mismatch for ${sample}"
            echo "Expected: ${fastq_bytes}"
            echo "Actual:   ${ACTUAL_BYTES}"
            exit 1

        fi

    fi


    echo "  gzip: PASS"
    echo "  md5:  PASS"
    echo "  size: PASS"
    echo


done < "${CANONICAL_MANIFEST}"


echo "===================================================================================================="
echo "AUDIT 42 COMPLETE"
echo "===================================================================================================="
echo

echo "All 8 single-end FASTQs downloaded and validated."
echo "FASTQ directory: ${FASTQ_DIR}"
echo
