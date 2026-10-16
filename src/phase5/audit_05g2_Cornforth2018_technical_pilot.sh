#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."

ROOT="$PWD"

OUT="$ROOT/results/phase5/audit_05g2_Cornforth2018_technical_pilot"
RAW="$ROOT/data/raw/cornforth2018_SRP135669"

REF="$ROOT/references/phase4/PRJNA576508_cornforth"

DECOY="$REF/bowtie2_decoy/Cornforth_104_decoys"
PAO1="$REF/bowtie2_PAO1/PAO1_NC_002516.2"

SAF="$REF/Cornforth_PAO1_no_decimal_locus_tags.saf"

MASK="$ROOT/results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv"

F_STATUS="$ROOT/results/phase5/audit_05f_download_QC_and_pilot_readiness/AUDIT_05F_STATUS.txt"

PILOT_MANIFEST="$ROOT/results/phase5/audit_05f_download_QC_and_pilot_readiness/FROZEN_two_sample_technical_pilot.csv"

if [[ -e "$OUT" ]]; then
    echo "Audit 05g2 output already exists. Refusing overwrite."
    exit 1
fi

mkdir -p "$OUT"

trap 'echo "CFSTATE_PHASE5_05G2_REVIEW_REQUIRED" > "$OUT/AUDIT_05G2_STATUS.txt"' ERR

echo "============================================================"
echo "CF-State Phase 5 — Audit 05g2"
echo "Two-sample RNA-seq technical feasibility pilot"
echo "============================================================"

python - "$F_STATUS" "$PILOT_MANIFEST" <<'PY'
import sys
import pandas as pd
from pathlib import Path

status = Path(sys.argv[1]).read_text().strip()

assert status == (
    "CFSTATE_PHASE5_05F_DOWNLOAD_AND_PILOT_READINESS_VERIFIED"
)

pilot = pd.read_csv(sys.argv[2], dtype=str)

expected = {
    ("Human_Sputum_G", "SRR6833349"),
    ("Lubbock_CW_15", "SRR6833340")
}

actual = set(zip(
    pilot["published_sample"],
    pilot["run_accession"]
))

assert len(pilot) == 2
assert actual == expected

print("Frozen 05f pilot selection verified PASS")
PY

for tool in seqtk pigz cutadapt bowtie2 samtools featureCounts; do
    command -v "$tool" >/dev/null
done

for path in \
    "$SAF" \
    "$MASK" \
    "${DECOY}.1.bt2" \
    "${PAO1}.1.bt2"; do

    if [[ ! -f "$path" ]]; then
        echo "Missing reference: $path"
        exit 1
    fi
done

echo "All processing tools and references found PASS"

ADAPTER="AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"

N_READS=2000000
SEED=42

for RUN in SRR6833349 SRR6833340; do

    if [[ "$RUN" == "SRR6833349" ]]; then
        SAMPLE="Human_Sputum_G"
        GROUP="CF_sputum"
    else
        SAMPLE="Lubbock_CW_15"
        GROUP="nonCF_chronic_wound"
    fi

    INPUT="$RAW/${RUN}__${RUN}.fastq.gz"
    SAMPLE_OUT="$OUT/$SAMPLE"

    mkdir -p "$SAMPLE_OUT"

    SUBSET="$SAMPLE_OUT/subsample_2M.fastq.gz"
    TRIMMED="$SAMPLE_OUT/trimmed.fastq.gz"
    NONDECOY="$SAMPLE_OUT/nondecoy.fastq.gz"

    BAM="$SAMPLE_OUT/PAO1_mapped.sorted.bam"
    COUNTS="$SAMPLE_OUT/PAO1_featureCounts.txt"

    echo
    echo "============================================================"
    echo "Processing $SAMPLE ($RUN)"
    echo "============================================================"

    if [[ ! -f "$INPUT" ]]; then
        echo "Missing input FASTQ: $INPUT"
        exit 1
    fi

    echo "[1/5] Reproducible two-million-read subsample"

    seqtk sample -s"$SEED" "$INPUT" "$N_READS" \
        | pigz -p 4 > "$SUBSET"

    echo "[2/5] Adapter trimming and minimum length 25"

    cutadapt \
        -a "$ADAPTER" \
        -m 25 \
        -j 4 \
        -o "$TRIMMED" \
        "$SUBSET" \
        > "$SAMPLE_OUT/cutadapt.log" 2>&1

    echo "[3/5] Non-Pseudomonas decoy filtering"

    bowtie2 \
        --very-sensitive \
        -p 6 \
        -x "$DECOY" \
        -U "$TRIMMED" \
        --un-gz "$NONDECOY" \
        -S /dev/null \
        2> "$SAMPLE_OUT/decoy_bowtie2.log"

    echo "[4/5] PAO1 alignment"

    bowtie2 \
        --very-sensitive \
        -p 6 \
        -x "$PAO1" \
        -U "$NONDECOY" \
        2> "$SAMPLE_OUT/PAO1_bowtie2.log" \
        | samtools view -@ 2 -u -F 4 - \
        | samtools sort \
            -@ 4 \
            -m 256M \
            -o "$BAM" -

    samtools index "$BAM"

    samtools flagstat \
        "$BAM" > "$SAMPLE_OUT/PAO1_flagstat.txt"

    echo "[5/5] PAO1 gene counting"

    featureCounts \
        -T 4 \
        -F SAF \
        -a "$SAF" \
        -s 0 \
        -o "$COUNTS" \
        "$BAM" \
        > "$SAMPLE_OUT/featureCounts.log" 2>&1

    echo "$SAMPLE COMPLETE"

done

echo
echo "============================================================"
echo "GENE-COVERAGE AND ALIGNMENT SUMMARY"
echo "============================================================"

python - "$OUT" "$MASK" <<'PY'
from pathlib import Path
import gzip
import re
import subprocess
import sys

import numpy as np
import pandas as pd

out = Path(sys.argv[1])
mask_path = Path(sys.argv[2])

samples = [
    ("Human_Sputum_G", "CF_sputum"),
    ("Lubbock_CW_15", "nonCF_chronic_wound")
]

mask_df = pd.read_csv(mask_path)

results = []

for sample, group in samples:

    folder = out / sample

    counts = pd.read_csv(
        folder / "PAO1_featureCounts.txt",
        sep="\t",
        comment="#"
    )

    if "Geneid" not in counts.columns:
        raise RuntimeError(
            "featureCounts output missing Geneid"
        )

    if counts["Geneid"].duplicated().any():
        raise RuntimeError("Duplicate gene identifiers")

    genes = counts["Geneid"].astype(str)
    values = pd.to_numeric(
        counts.iloc[:, -1],
        errors="raise"
    )

    if not np.isfinite(values).all():
        raise RuntimeError("Nonfinite count values")

    if (values < 0).any():
        raise RuntimeError("Negative gene counts")

    gene_ids = set(genes)

    candidate_columns = list(mask_df.columns)

    overlap = {}

    for column in candidate_columns:
        ids = set(mask_df[column].astype(str))
        overlap[column] = len(ids & gene_ids)

    best_column = max(overlap, key=overlap.get)

    mask_genes = set(
        mask_df[best_column].astype(str)
    )

    if len(mask_genes) != 4923:
        raise RuntimeError(
            "4923-gene mask has unexpected size"
        )

    if len(mask_genes & gene_ids) != 4923:
        raise RuntimeError(
            "Not all frozen mask genes occur in count table"
        )

    count_series = pd.Series(
        values.to_numpy(dtype=float),
        index=genes.to_numpy()
    )

    mapped = subprocess.check_output(
        [
            "samtools",
            "view",
            "-c",
            "-F",
            "4",
            str(
                folder / "PAO1_mapped.sorted.bam"
            )
        ],
        text=True
    )

    mapped_reads = int(mapped.strip())

    decoy_log = (
        folder / "decoy_bowtie2.log"
    ).read_text()

    pao1_log = (
        folder / "PAO1_bowtie2.log"
    ).read_text()

    def alignment_rate(text):
        match = re.search(
            r"([0-9.]+)% overall alignment rate",
            text
        )

        if not match:
            return float("nan")

        return float(match.group(1))

    detected = int(
        (count_series.loc[list(mask_genes)] > 0).sum()
    )

    results.append({
        "sample": sample,
        "group": group,
        "subsample_target_reads": 2000000,
        "decoy_alignment_percent":
            alignment_rate(decoy_log),
        "PAO1_alignment_percent_of_nondecoy":
            alignment_rate(pao1_log),
        "PAO1_mapped_alignments": mapped_reads,
        "featureCounts_assigned_reads":
            int(count_series.sum()),
        "genes_detected_all_5567":
            int((count_series > 0).sum()),
        "genes_detected_common_4923": detected,
        "common_gene_detected_fraction":
            detected / 4923
    })

summary = pd.DataFrame(results)

if len(summary) != 2:
    raise RuntimeError("Expected two pilot specimens")

if summary["PAO1_mapped_alignments"].le(0).any():
    raise RuntimeError("No PAO1 alignments in a pilot specimen")

summary.to_csv(
    out / "FROZEN_two_sample_pilot_QC.csv",
    index=False
)

print(summary.round(5).to_string(index=False))

print()
print("Interpretation:")
print("Two-million-read pilot only.")
print("104-sequence decoy + PAO1 alignment is exploratory.")
print("No sample or gene coverage threshold was optimised.")
print("No biological CF-State prediction was calculated.")
print("Do not use these subsampled counts for CF specificity.")

status = (
    "CFSTATE_PHASE5_05G2_"
    "TWO_SAMPLE_TECHNICAL_PILOT_COMPLETE"
)

(out / "AUDIT_05G2_STATUS.txt").write_text(
    status + "\n"
)

print()
print("STATUS:", status)
PY
