set -euo pipefail

OUT_DIR="data/raw/expanded_biofilm/GSE120602/published_counts"

mkdir -p "${OUT_DIR}"

declare -A FILES

FILES["GSM3405609"]="GSM3405609_plusCiproLarge1_readcount.txt.gz"
FILES["GSM3405610"]="GSM3405610_plusCiproLarge2_readcount.txt.gz"
FILES["GSM3405611"]="GSM3405611_minusCiproLarge1_readcount.txt.gz"
FILES["GSM3405612"]="GSM3405612_minusCiproLarge2_readcount.txt.gz"
FILES["GSM3405613"]="GSM3405613_PAPlanktonic1_readcount.txt.gz"
FILES["GSM3405614"]="GSM3405614_PAPlanktonic2_readcount.txt.gz"
FILES["GSM3405615"]="GSM3405615_PABiofilm1_readcount.txt.gz"
FILES["GSM3405616"]="GSM3405616_PABiofilm2_readcount.txt.gz"


echo
echo "===================================================================================================="
echo "CF-STATE — Audit 47a: download GSE120602 published count files"
echo "===================================================================================================="


for GSM in \
    GSM3405609 \
    GSM3405610 \
    GSM3405611 \
    GSM3405612 \
    GSM3405613 \
    GSM3405614 \
    GSM3405615 \
    GSM3405616

do

    FILE="${FILES[$GSM]}"
    DEST="${OUT_DIR}/${FILE}"

    URL="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM3405nnn/${GSM}/suppl/${FILE}"


    echo
    echo "--------------------------------------------------------------------------------"
    echo "${GSM}"
    echo "${FILE}"
    echo "--------------------------------------------------------------------------------"


    if [[ -s "${DEST}" ]]; then

        if gzip -t "${DEST}" 2>/dev/null; then

            echo "Already downloaded and valid."
            continue

        fi

        echo "Existing file invalid — redownloading."
        rm -f "${DEST}"

    fi


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

        echo "ERROR: empty download for ${GSM}"
        exit 1

    fi


    if ! gzip -t "${DEST}.part"; then

        echo "ERROR: gzip validation failed for ${GSM}"
        exit 1

    fi


    mv "${DEST}.part" "${DEST}"

    echo "PASS"

done


echo
echo "===================================================================================================="
echo "DOWNLOADED FILES"
echo "===================================================================================================="

ls -lh "${OUT_DIR}"/*.txt.gz


echo
echo "===================================================================================================="
echo "QUICK CONTENT CHECK"
echo "===================================================================================================="

for GSM in \
    GSM3405609 \
    GSM3405610 \
    GSM3405611 \
    GSM3405612 \
    GSM3405613 \
    GSM3405614 \
    GSM3405615 \
    GSM3405616

do

    FILE="${FILES[$GSM]}"

    echo
    echo "${GSM}"

    zcat "${OUT_DIR}/${FILE}" | head -n 4

done


echo
echo "===================================================================================================="
echo "AUDIT 47a COMPLETE"
echo "===================================================================================================="
echo