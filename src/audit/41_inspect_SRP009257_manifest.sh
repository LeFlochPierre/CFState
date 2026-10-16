#!/usr/bin/env bash

set -euo pipefail

OUT_DIR="results/audit/expanded_biofilm/SRP009257"

mkdir -p "${OUT_DIR}"

OUT_FILE="${OUT_DIR}/SRP009257_ENA_manifest.tsv"

URL="https://www.ebi.ac.uk/ena/portal/api/filereport?accession=SRP009257&result=read_run&fields=run_accession,experiment_accession,sample_accession,sample_alias,sample_title,experiment_title,library_layout,library_strategy,instrument_model,read_count,base_count,fastq_ftp,fastq_md5,fastq_bytes&format=tsv"


echo
echo "===================================================================================================="
echo "CF-STATE — Audit 41: inspect SRP009257"
echo "===================================================================================================="
echo


curl \
    -L \
    --fail \
    --retry 3 \
    "${URL}" \
    -o "${OUT_FILE}"


if [[ ! -s "${OUT_FILE}" ]]; then
    echo "ERROR: manifest is empty"
    exit 1
fi


python - <<'PY'

from pathlib import Path

import pandas as pd


FILE = Path(
    "results/audit/expanded_biofilm/"
    "SRP009257/SRP009257_ENA_manifest.tsv"
)


df = pd.read_csv(
    FILE,
    sep="\t",
)


print()
print("=" * 100)
print("SRP009257 ENA MANIFEST")
print("=" * 100)

print(
    "Runs: "
    + str(len(df))
)

print()


columns = [
    "run_accession",
    "experiment_accession",
    "sample_alias",
    "sample_title",
    "experiment_title",
    "library_layout",
    "instrument_model",
    "read_count",
    "base_count",
]


available = [
    column
    for column in columns
    if column in df.columns
]


print(
    df[
        available
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("FASTQ AVAILABILITY")
print("=" * 100)


if "fastq_ftp" not in df.columns:

    print(
        "No fastq_ftp column returned."
    )

else:

    for _, row in df.iterrows():

        run = str(
            row[
                "run_accession"
            ]
        )

        fastq = str(
            row[
                "fastq_ftp"
            ]
        )

        print(
            run
            + ": "
            + fastq
        )


print()
print("=" * 100)
print("AUDIT 41 COMPLETE")
print("=" * 100)
print()

PY
