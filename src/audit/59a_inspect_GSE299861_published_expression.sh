#!/usr/bin/env bash

set -euo pipefail


OUT_DIR="data/raw/expanded_biofilm/GSE299861/published_expression"

FILE="GSE299861_PAO1_all.fpkm.txt.gz"

URL="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE299nnn/GSE299861/suppl/${FILE}"

DEST="${OUT_DIR}/${FILE}"


mkdir -p "${OUT_DIR}"


echo
echo "===================================================================================================="
echo "CF-STATE — Audit 59a: inspect GSE299861 authors' PAO1 expression table"
echo "===================================================================================================="


# ============================================================
# Download
# ============================================================

if [[ -s "${DEST}" ]]; then

    if gzip -t "${DEST}" 2>/dev/null; then

        echo
        echo "Published file already downloaded and valid."

    else

        echo
        echo "Existing file is invalid — redownloading."

        rm -f "${DEST}"

    fi
fi


if [[ ! -s "${DEST}" ]]; then

    echo
    echo "Downloading:"
    echo "${URL}"

    rm -f "${DEST}.part"

    curl \
        -4 \
        --http1.1 \
        -L \
        --fail \
        --retry 5 \
        --retry-delay 2 \
        --connect-timeout 30 \
        "${URL}" \
        -o "${DEST}.part"

    if [[ ! -s "${DEST}.part" ]]; then

        echo "ERROR: downloaded file is empty."
        exit 1

    fi

    if ! gzip -t "${DEST}.part"; then

        echo "ERROR: downloaded file is not a valid gzip."
        exit 1

    fi

    mv \
        "${DEST}.part" \
        "${DEST}"

fi


# ============================================================
# Basic file information
# ============================================================

echo
echo "===================================================================================================="
echo "FILE INFORMATION"
echo "===================================================================================================="

ls -lh "${DEST}"

file "${DEST}"

echo

echo "Uncompressed line count:"

gzip -cd "${DEST}" \
    | wc -l


# ============================================================
# Header
# ============================================================

echo
echo "===================================================================================================="
echo "HEADER"
echo "===================================================================================================="

gzip -cd "${DEST}" \
    | sed -n '1p'


# ============================================================
# First 12 rows
#
# sed is used instead of head so pipefail does not interpret
# a normal SIGPIPE as a script failure.
# ============================================================

echo
echo "===================================================================================================="
echo "FIRST 12 ROWS"
echo "===================================================================================================="

gzip -cd "${DEST}" \
    | sed -n '1,12p'


# ============================================================
# Number of columns in first rows
# ============================================================

echo
echo "===================================================================================================="
echo "COLUMN COUNTS"
echo "===================================================================================================="

gzip -cd "${DEST}" \
    | sed -n '1,10p' \
    | awk -F '\t' '{print NR "\t" NF}'


# ============================================================
# Search for PAO1 locus tags
# ============================================================

echo
echo "===================================================================================================="
echo "PAxxxx IDENTIFIER CHECK"
echo "===================================================================================================="

TMP_FILE="${OUT_DIR}/Audit59a_preview.tmp"

gzip -cd "${DEST}" \
    | sed -n '1,200p' \
    > "${TMP_FILE}"


PA_COUNT=$(
    grep -Eo 'PA[0-9]{4}' "${TMP_FILE}" \
        | sort -u \
        | wc -l
)


echo "Unique PAxxxx strings in first 200 lines: ${PA_COUNT}"

echo

echo "First PAxxxx examples:"

grep -Eo 'PA[0-9]{4}' "${TMP_FILE}" \
    | sort -u \
    | sed -n '1,20p' \
    || true


rm -f "${TMP_FILE}"


echo
echo "===================================================================================================="
echo "AUDIT 59a COMPLETE"
echo "===================================================================================================="

echo
echo "No expression values were transformed or reanalysed."
echo
