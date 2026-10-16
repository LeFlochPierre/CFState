set -euo pipefail

STUDY="SRP162817"

OUT_DIR="results/audit/expanded_biofilm/GSE120602"

RAW_DIR="data/raw/expanded_biofilm/GSE120602"

mkdir -p "${OUT_DIR}"
mkdir -p "${RAW_DIR}"

ENA_FILE="${OUT_DIR}/GSE120602_ENA_manifest.tsv"


echo
echo "===================================================================================================="
echo "CF-STATE — Audit 46: inspect GSE120602 / SRP162817"
echo "===================================================================================================="
echo


URL="https://www.ebi.ac.uk/ena/portal/api/filereport?accession=${STUDY}&result=read_run&fields=run_accession,sample_accession,secondary_sample_accession,sample_alias,experiment_accession,study_accession,scientific_name,instrument_platform,instrument_model,library_strategy,library_layout,read_count,base_count,fastq_ftp,fastq_md5,fastq_bytes&format=tsv&download=true"


echo "Downloading ENA run manifest..."

curl \
    -4 \
    -L \
    --fail \
    --retry 3 \
    "${URL}" \
    -o "${ENA_FILE}"


if [[ ! -s "${ENA_FILE}" ]]; then
    echo "ERROR: ENA manifest is empty."
    exit 1
fi


python - <<'PY'

from pathlib import Path

import pandas as pd


MANIFEST = Path(
    "results/audit/expanded_biofilm/GSE120602/"
    "GSE120602_ENA_manifest.tsv"
)


df = pd.read_csv(
    MANIFEST,
    sep="\t",
)


print()
print("=" * 100)
print("ENA MANIFEST")
print("=" * 100)

print(
    df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("BASIC CHECKS")
print("=" * 100)

print(
    "Number of runs: "
    + str(
        len(df)
    )
)

print(
    "Unique samples: "
    + str(
        df[
            "sample_accession"
        ].nunique()
    )
)

print(
    "Library layouts:"
)

print(
    df[
        "library_layout"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

print()

print(
    "Instrument models:"
)

print(
    df[
        "instrument_model"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)


if "read_count" in df.columns:

    df[
        "read_count"
    ] = pd.to_numeric(
        df[
            "read_count"
        ],
        errors="coerce",
    )


if "base_count" in df.columns:

    df[
        "base_count"
    ] = pd.to_numeric(
        df[
            "base_count"
        ],
        errors="coerce",
    )


display_columns = [
    column
    for column in [
        "run_accession",
        "sample_alias",
        "secondary_sample_accession",
        "library_layout",
        "read_count",
        "base_count",
    ]
    if column in df.columns
]


print()
print("=" * 100)
print("COMPACT RUN TABLE")
print("=" * 100)

print(
    df[
        display_columns
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("EXPECTED GEO DESIGN")
print("=" * 100)

expected = [
    ("GSM3405609", "persister_ciprofloxacin_1"),
    ("GSM3405610", "persister_ciprofloxacin_2"),
    ("GSM3405611", "stationary_1"),
    ("GSM3405612", "stationary_2"),
    ("GSM3405613", "planktonic_1"),
    ("GSM3405614", "planktonic_2"),
    ("GSM3405615", "biofilm_1"),
    ("GSM3405616", "biofilm_2"),
]

for gsm, label in expected:

    print(
        gsm
        + "\t"
        + label
    )


print()
print("=" * 100)
print("AUDIT 46 COMPLETE")
print("=" * 100)

print(
    "Manifest saved to: "
    + str(MANIFEST)
)

print()

PY
