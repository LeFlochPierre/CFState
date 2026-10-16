#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "$0")/../.."
ROOT="$PWD"

OUT="$ROOT/results/phase5/audit_05h_Cornforth2018_full_depth_baseline"
RAW="$ROOT/data/raw/cornforth2018_SRP135669"
REF="$ROOT/references/phase4/PRJNA576508_cornforth"

DECOY="$REF/bowtie2_decoy/Cornforth_104_decoys"
PAO1="$REF/bowtie2_PAO1/PAO1_NC_002516.2"
SAF="$REF/Cornforth_PAO1_no_decimal_locus_tags.saf"

MANIFEST="$ROOT/results/phase5/audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv"
PILOT_STATUS="$ROOT/results/phase5/audit_05g2_Cornforth2018_technical_pilot/AUDIT_05G2_STATUS.txt"

mkdir -p "$OUT/work" "$OUT/samples"

if [[ -f "$OUT/AUDIT_05H_STATUS.txt" ]] &&
   grep -qx 'CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN' \
   "$OUT/AUDIT_05H_STATUS.txt"; then
    echo "Audit 05h already frozen. Refusing to rerun."
    exit 1
fi

trap 'echo CFSTATE_PHASE5_05H_REVIEW_REQUIRED > "$OUT/AUDIT_05H_STATUS.txt"' ERR

echo "=========================================================="
echo "CF-State Phase 5 — Audit 05h"
echo "Full-depth, 13-sample technical PAO1 baseline"
echo "=========================================================="

for tool in python cutadapt bowtie2 samtools featureCounts; do
    command -v "$tool" >/dev/null || {
        echo "Missing tool: $tool"
        exit 1
    }
done

[[ -f "$SAF" ]]
[[ -f "${DECOY}.1.bt2" ]]
[[ -f "${PAO1}.1.bt2" ]]
[[ -f "$MANIFEST" ]]
[[ "$(cat "$PILOT_STATUS")" == \
"CFSTATE_PHASE5_05G2_TWO_SAMPLE_TECHNICAL_PILOT_COMPLETE" ]]

python - "$MANIFEST" "$OUT/FROZEN_processing_order.tsv" <<'PY'
import sys
from pathlib import Path
import pandas as pd

source = Path(sys.argv[1])
destination = Path(sys.argv[2])

df = pd.read_csv(source, dtype=str).fillna("")

expected = {
    "SRR6833346", "SRR6833345", "SRR6833344",
    "SRR6833347", "SRR6833351", "SRR6833350",
    "SRR6833349", "SRR6833340", "SRR6833324",
    "SRR6833323", "SRR6833325", "SRR6833342",
    "SRR6833341"
}

assert len(df) == 13
assert df["run_accession"].is_unique
assert set(df["run_accession"]) == expected
assert df["published_sample"].is_unique
assert (df["analysis_role"] == "main_clinical_13").all()

assert (df["group"] == "CF_sputum").sum() == 7
assert (df["group"] == "nonCF_chronic_wound").sum() == 4
assert (df["group"] == "nonCF_burn").sum() == 2

df[["published_sample", "group", "run_accession"]].to_csv(
    destination,
    sep="\t",
    index=False,
    header=False
)

print("Frozen 13-sample processing design PASS")
PY

ADAPTER="AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"

while IFS=$'\t' read -r SAMPLE GROUP RUN; do

    INPUT="$RAW/${RUN}__${RUN}.fastq.gz"

    FINAL="$OUT/samples/$RUN"
    WORK="$OUT/work/$RUN"

    echo
    echo "=========================================================="
    echo "Sample: $SAMPLE ($RUN)"
    echo "Group: $GROUP"
    echo "=========================================================="

    [[ -f "$INPUT" ]] || {
        echo "Missing FASTQ: $INPUT"
        exit 1
    }

    if [[ -f "$FINAL/PROCESSING_DONE.txt" ]]; then
        echo "Previously completed; checking outputs."

        [[ -s "$FINAL/PAO1_featureCounts.txt" ]]
        [[ -s "$FINAL/PAO1_mapped.sorted.bam" ]]
        [[ -s "$FINAL/nondecoy.fastq.gz" ]]
        samtools quickcheck "$FINAL/PAO1_mapped.sorted.bam"

        echo "Verified, skipping."
        continue
    fi

    if [[ -d "$FINAL" ]]; then
        echo "Unexpected incomplete final sample directory."
        echo "$FINAL"
        exit 1
    fi

    # Only this audit's unfinished working directory is cleared.
    if [[ -d "$WORK" ]]; then
        echo "Restarting unfinished sample from the original FASTQ."
        rm -rf -- "$WORK"
    fi

    mkdir -p "$WORK"

    echo "[1/4] Cutadapt — full-depth reads"

    cutadapt \
        -a "$ADAPTER" \
        -m 25 \
        -j 8 \
        -o "$WORK/trimmed.fastq.gz" \
        "$INPUT" \
        > "$WORK/cutadapt.log" 2>&1

    echo "[2/4] Existing 104-sequence decoy reference"

    bowtie2 \
        --very-sensitive \
        -p 8 \
        -x "$DECOY" \
        -U "$WORK/trimmed.fastq.gz" \
        --un-gz "$WORK/nondecoy.fastq.gz" \
        -S /dev/null \
        2> "$WORK/decoy_bowtie2.log"

    echo "[3/4] PAO1 alignment"

    bowtie2 \
        --very-sensitive \
        -p 8 \
        -x "$PAO1" \
        -U "$WORK/nondecoy.fastq.gz" \
        2> "$WORK/PAO1_bowtie2.log" \
        | samtools view -@ 2 -u -F 4 - \
        | samtools sort \
            -@ 4 \
            -m 512M \
            -o "$WORK/PAO1_mapped.sorted.bam" -

    samtools index "$WORK/PAO1_mapped.sorted.bam"

    samtools quickcheck "$WORK/PAO1_mapped.sorted.bam"

    samtools flagstat "$WORK/PAO1_mapped.sorted.bam" \
        > "$WORK/PAO1_flagstat.txt"

    echo "[4/4] featureCounts — PAO1 genes"

    featureCounts \
        -T 8 \
        -F SAF \
        -a "$SAF" \
        -s 0 \
        -o "$WORK/PAO1_featureCounts.txt" \
        "$WORK/PAO1_mapped.sorted.bam" \
        > "$WORK/featureCounts.log" 2>&1

    [[ -s "$WORK/PAO1_featureCounts.txt" ]] || {
        echo "Missing featureCounts result for $RUN"
        exit 1
    }

    echo "FULL_DEPTH_TECHNICAL_PROCESSING_COMPLETE" \
        > "$WORK/PROCESSING_DONE.txt"

    mv "$WORK" "$FINAL"

    echo "$SAMPLE ($RUN) COMPLETED"

done < "$OUT/FROZEN_processing_order.tsv"

echo
echo "=========================================================="
echo "AGGREGATE FULL-DEPTH GENE COUNTS"
echo "=========================================================="

python - "$ROOT" "$OUT" "$MANIFEST" <<'PY'
from pathlib import Path
import hashlib
import re
import subprocess
import sys

import numpy as np
import pandas as pd

root = Path(sys.argv[1])
out = Path(sys.argv[2])
manifest = Path(sys.argv[3])

metadata = pd.read_csv(manifest)

mask_file = (
    root /
    "results/phase4/"
    "audit_04k4_PRJNA576508_blinded_softcore_projection/"
    "FROZEN_softcore_gene_order_4923.csv"
)

mask_frame = pd.read_csv(mask_file)
counts_by_sample = {}
qc_rows = []
gene_order = None


def alignment_rate(path):
    text = path.read_text()

    match = re.search(
        r"([0-9.]+)% overall alignment rate",
        text
    )

    if match is None:
        raise RuntimeError(
            "Bowtie2 alignment rate missing: " + str(path)
        )

    return float(match.group(1))


for row in metadata.itertuples(index=False):
    run = row.run_accession
    sample = row.published_sample

    directory = out / "samples" / run

    assert (
        directory / "PROCESSING_DONE.txt"
    ).is_file(), run

    path = directory / "PAO1_featureCounts.txt"

    frame = pd.read_csv(
        path, sep="\t", comment="#"
    )

    assert "Geneid" in frame.columns

    genes = frame["Geneid"].astype(str)

    assert genes.is_unique
    assert len(genes) == 5567

    if gene_order is None:
        gene_order = list(genes)
    else:
        assert set(genes) == set(gene_order)

    counts = pd.to_numeric(
        frame.iloc[:, -1], errors="raise"
    )

    assert np.isfinite(counts.to_numpy()).all()
    assert (counts >= 0).all()
    assert np.allclose(counts, np.round(counts))

    series = pd.Series(
        counts.to_numpy(dtype=np.int64),
        index=genes.to_numpy()
    )

    counts_by_sample[sample] = series.reindex(gene_order)

    mapped = int(
        subprocess.check_output(
            [
                "samtools", "view", "-c", "-F", "4",
                str(directory / "PAO1_mapped.sorted.bam")
            ],
            text=True
        ).strip()
    )

    qc_rows.append({
        "published_sample": sample,
        "group": row.group,
        "run_accession": run,
        "published_PAO1_ortholog_mapped_reads":
            row.published_mapped_PAO1_ortholog_reads,
        "PAO1_mapped_alignments": mapped,
        "featureCounts_assigned_reads": int(series.sum()),
        "decoy_alignment_percent":
            alignment_rate(directory / "decoy_bowtie2.log"),
        "PAO1_alignment_percent_of_nondecoy":
            alignment_rate(directory / "PAO1_bowtie2.log"),
        "genes_detected_all_5567":
            int((series > 0).sum())
    })


matrix = pd.DataFrame(counts_by_sample)

assert matrix.shape == (5567, 13)
assert matrix.index.is_unique
assert np.isfinite(matrix.to_numpy()).all()

overlaps = {
    column: len(
        set(mask_frame[column].astype(str))
        & set(matrix.index)
    )
    for column in mask_frame.columns
}

chosen = max(overlaps, key=overlaps.get)

mask = mask_frame[chosen].astype(str).tolist()

assert len(mask) == 4923
assert len(set(mask)) == 4923
assert set(mask).issubset(matrix.index)

qc = pd.DataFrame(qc_rows)

for row in qc_rows:
    sample = row["published_sample"]

    detected = int(
        (matrix.loc[mask, sample] > 0).sum()
    )

    qc.loc[
        qc["published_sample"] == sample,
        "genes_detected_common_4923"
    ] = detected

    qc.loc[
        qc["published_sample"] == sample,
        "common_gene_detected_fraction"
    ] = detected / 4923

assert len(qc) == 13
assert qc["run_accession"].is_unique
assert qc["PAO1_mapped_alignments"].ge(0).all()

matrix.to_csv(
    out / "FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv",
    index_label="Geneid"
)

qc.to_csv(
    out / "FROZEN_13_full_depth_PAO1_mapping_QC.csv",
    index=False
)

provenance = []

for path in [
    manifest,
    mask_file,
    Path(sys.argv[0])
]:
    if not path.is_file():
        continue

    h = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1048576), b""
        ):
            h.update(chunk)

    provenance.append({
        "path": str(path),
        "sha256": h.hexdigest()
    })

pd.DataFrame(provenance).to_csv(
    out / "input_provenance_sha256.csv",
    index=False
)

print()
print("========== FULL-DEPTH QC ==========")

print(qc.round(5).to_string(index=False))

print()
print("Group summary — descriptive only")

print(
    qc.groupby("group")[
        [
            "PAO1_mapped_alignments",
            "featureCounts_assigned_reads",
            "common_gene_detected_fraction"
        ]
    ].agg(["min", "median", "max"]).round(4).to_string()
)

print()
print("ALL 13 SAMPLES RETAINED.")
print("No CF-State candidate evaluated.")
print("PAO1 mapping is a provisional technical baseline.")
print("Multi-strain reference sensitivity is still required.")

status = (
    "CFSTATE_PHASE5_05H_FULL_DEPTH_PAO1_BASELINE_FROZEN"
)

(out / "AUDIT_05H_STATUS.txt").write_text(
    status + "\n"
)

print()
print("STATUS:", status)
PY
