set -euo pipefail

THREADS=16

FASTQ_DIR="data/raw/expanded_biofilm/SRP009257/fastq"

MANIFEST="results/audit/expanded_biofilm/SRP009257/SRP009257_canonical_manifest.tsv"

REF_FASTA="references/mappings/GSE191103/PA14/NC_008463.1.fasta"

SAF="references/mappings/GSE191103/PA14/NC_008463.1_PA14_locus_tags.saf"

INDEX_DIR="references/mappings/SRP009257/PA14/bowtie1_index"

INDEX_PREFIX="${INDEX_DIR}/NC_008463.1"

OUT_DIR="results/audit/expanded_biofilm/SRP009257/reprocessing"

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
    "planktonic_P4_1"
    "planktonic_P4_2"
    "planktonic_P12_1"
    "planktonic_P12_2"
    "biofilm_B24_1"
    "biofilm_B24_2"
    "biofilm_B48_1"
    "biofilm_B48_2"
)


echo
echo "===================================================================================================="
echo "CF-STATE — Audit 43: SRP009257 PA14 alignment and strandedness"
echo "===================================================================================================="
echo


# ============================================================
# Tool checks
# ============================================================

for TOOL in bowtie bowtie-build samtools featureCounts pigz; do

    if ! command -v "${TOOL}" >/dev/null 2>&1; then

        echo "ERROR: ${TOOL} not found"

        echo "For Bowtie 1, install with:"
        echo "conda install -c conda-forge -c bioconda bowtie"

        exit 1

    fi

done


# ============================================================
# Input checks
# ============================================================

for FILE in "${MANIFEST}" "${REF_FASTA}" "${SAF}"; do

    if [[ ! -s "${FILE}" ]]; then

        echo "ERROR: missing input: ${FILE}"

        exit 1

    fi

done


for SAMPLE in "${SAMPLES[@]}"; do

    FASTQ="${FASTQ_DIR}/${SAMPLE}.fastq.gz"

    if [[ ! -s "${FASTQ}" ]]; then

        echo "ERROR: missing FASTQ: ${FASTQ}"

        exit 1

    fi

done


# ============================================================
# Build Bowtie 1 index
#
# Note:
# The existing GSE191103 index is a Bowtie 2 index.
# Bowtie 1 requires its own index.
# ============================================================

if [[ ! -f "${INDEX_PREFIX}.1.ebwt" && ! -f "${INDEX_PREFIX}.1.ebwtl" ]]; then

    echo
    echo "Building Bowtie 1 PA14 index..."

    bowtie-build \
        "${REF_FASTA}" \
        "${INDEX_PREFIX}"

else

    echo
    echo "Bowtie 1 index already exists."

fi


# ============================================================
# Align
#
# Reproduce the original paper:
#
#   -m 1
#   --best
#   --strata
#
# Only uniquely mapping reads are therefore retained.
# ============================================================

for SAMPLE in "${SAMPLES[@]}"; do

    echo
    echo "===================================================================================================="
    echo "ALIGNING: ${SAMPLE}"
    echo "===================================================================================================="

    FASTQ="${FASTQ_DIR}/${SAMPLE}.fastq.gz"

    BAM="${BAM_DIR}/${SAMPLE}.sorted.bam"

    LOG="${LOG_DIR}/${SAMPLE}.bowtie.log"


    if [[ ! -s "${BAM}" ]]; then

        pigz -dc "${FASTQ}" \
        | bowtie \
            -q \
            -m 1 \
            --best \
            --strata \
            -p "${THREADS}" \
            -S \
            "${INDEX_PREFIX}" \
            - \
            2> "${LOG}" \
        | samtools view \
            -@ "${THREADS}" \
            -b \
            -F 4 \
            - \
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


    echo "Unique mapped reads retained: $(samtools view -c "${BAM}")"

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
# Test orientation:
#
# -s 0 = unstranded
# -s 1 = read strand = annotated gene strand
# -s 2 = read strand opposite annotated gene strand
#
# --read2pos 5 reproduces the paper's strategy of counting
# the mapped 5' end of each read rather than full-read overlap.
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
        -s "${STRAND}" \
        --read2pos 5 \
        "${BAMS[@]}"

done


# ============================================================
# QC summary
# ============================================================

python - <<'PY'

from pathlib import Path
import subprocess

import pandas as pd


MANIFEST = Path(
    "results/audit/expanded_biofilm/SRP009257/"
    "SRP009257_canonical_manifest.tsv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/SRP009257/reprocessing"
)

BAM_DIR = OUT_DIR / "bam"

COUNT_DIR = OUT_DIR / "counts"

QC_DIR = OUT_DIR / "qc"


manifest = pd.read_csv(
    MANIFEST,
    sep="\t",
)


sample_order = [
    "planktonic_P4_1",
    "planktonic_P4_2",
    "planktonic_P12_1",
    "planktonic_P12_2",
    "biofilm_B24_1",
    "biofilm_B24_2",
    "biofilm_B48_1",
    "biofilm_B48_2",
]


manifest = (
    manifest
    .set_index("sample")
    .loc[sample_order]
    .reset_index()
)


# ============================================================
# Unique mapping
# ============================================================

alignment_records = []


for _, row in manifest.iterrows():

    sample = str(
        row["sample"]
    )

    total_reads = int(
        row["read_count"]
    )

    bam = (
        BAM_DIR
        / f"{sample}.sorted.bam"
    )


    result = subprocess.run(
        [
            "samtools",
            "view",
            "-c",
            str(bam),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


    unique_mapped = int(
        result.stdout.strip()
    )


    unique_fraction = (
        unique_mapped
        / total_reads
    )


    alignment_records.append(
        {
            "sample":
                sample,

            "state":
                str(
                    row["state"]
                ),

            "condition":
                str(
                    row["condition"]
                ),

            "total_reads":
                total_reads,

            "unique_mapped_reads":
                unique_mapped,

            "unique_mapping_fraction":
                unique_fraction,
        }
    )


alignment = pd.DataFrame(
    alignment_records
)


alignment.to_csv(
    QC_DIR
    / "unique_mapping_summary.csv",
    index=False,
)


# ============================================================
# Strandedness
# ============================================================

assignment_records = []


for strand in [0, 1, 2]:

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


        if total > 0:

            fraction = (
                assigned
                / total
            )

        else:

            fraction = float("nan")


        meta = (
            manifest.loc[
                manifest["sample"]
                == sample
            ]
            .iloc[0]
        )


        assignment_records.append(
            {
                "strand_setting":
                    strand,

                "sample":
                    sample,

                "state":
                    str(
                        meta["state"]
                    ),

                "condition":
                    str(
                        meta["condition"]
                    ),

                "mapped_reads_entering_featureCounts":
                    total,

                "assigned_5prime_reads":
                    assigned,

                "assigned_fraction_of_unique_mapped":
                    fraction,
            }
        )


assignment = pd.DataFrame(
    assignment_records
)


assignment.to_csv(
    QC_DIR
    / "strand_assignment_summary.csv",
    index=False,
)


strand_summary = (
    assignment
    .groupby(
        "strand_setting"
    )[
        "assigned_fraction_of_unique_mapped"
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
# Print mapping
# ============================================================

print()
print("=" * 100)
print("UNIQUE BOWTIE 1 MAPPING")
print("=" * 100)


alignment_display = (
    alignment.copy()
)


alignment_display[
    "unique_mapping_percent"
] = (
    100.0
    * alignment_display[
        "unique_mapping_fraction"
    ]
)


print(
    alignment_display[
        [
            "sample",
            "condition",
            "total_reads",
            "unique_mapped_reads",
            "unique_mapping_percent",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Print strandedness
# ============================================================

print()
print("=" * 100)
print("STRANDEDNESS — OVERALL")
print("=" * 100)


strand_display = (
    strand_summary.copy()
)


for column in [
    "median",
    "mean",
    "min",
    "max",
]:

    strand_display[
        column
    ] = (
        100.0
        * strand_display[
            column
        ]
    )


print(
    strand_display.to_string(
        index=False
    )
)


median_s1 = float(
    strand_summary.loc[
        strand_summary[
            "strand_setting"
        ]
        == 1,
        "median",
    ].iloc[0]
)


median_s2 = float(
    strand_summary.loc[
        strand_summary[
            "strand_setting"
        ]
        == 2,
        "median",
    ].iloc[0]
)


# ============================================================
# Orientation interpretation
# ============================================================

print()
print("=" * 100)
print("PRELIMINARY ORIENTATION")
print("=" * 100)


if median_s1 > 3.0 * median_s2:

    print(
        "Likely featureCounts -s 1 "
        "(read strand matches annotated gene strand)."
    )

elif median_s2 > 3.0 * median_s1:

    print(
        "Likely featureCounts -s 2 "
        "(read strand is opposite annotated gene strand)."
    )

else:

    print(
        "Orientation is not cleanly separated; "
        "inspect before finalizing counts."
    )


print()

print(
    "Do not freeze SRP009257 counts yet."
)

print(
    "First confirm unique mapping and strand orientation."
)

print()

PY


echo "===================================================================================================="
echo "AUDIT 43 COMPLETE"
echo "===================================================================================================="

echo "Outputs: ${OUT_DIR}"

echo
