from pathlib import Path
from io import StringIO
import hashlib
import json
import re
import shutil
import subprocess
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path.cwd()

MANIFEST = (
    ROOT
    / "results/phase2/audit_12c_GSE299861_raw_requantification_preflight"
    / "PAO1_raw_run_manifest.csv"
)

GENBANK = (
    ROOT
    / "references/mappings/NC_002516.2.gb"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

RAW_DIR = (
    ROOT
    / "data/raw/phase2/GSE299861/fastq"
)

WORK = (
    ROOT
    / "data/processed/phase2/GSE299861_canonical_requantification"
)

REF_DIR = WORK / "reference"
TRIM_DIR = WORK / "trimmed"
BAM_DIR = WORK / "bam"
LOG_DIR = WORK / "logs"

OUT = (
    ROOT
    / "results/phase2/audit_12d_GSE299861_canonical_requantification"
)

THREADS = 8

ADAPTER_R1 = (
    "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
)

ADAPTER_R2 = (
    "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT"
)


def run(command):
    print()
    print("$", command)

    subprocess.run(
        [
            "bash",
            "-lc",
            "set -o pipefail; " + command,
        ],
        check=True,
    )


def json_df(value):
    if isinstance(value, str):
        return pd.read_json(
            StringIO(value)
        )

    if isinstance(value, dict):
        return pd.DataFrame(
            value
        )

    raise RuntimeError(
        "Unexpected serialized DataFrame."
    )


def md5(path):
    h = hashlib.md5()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def download(url, destination, expected_md5=None):
    if destination.exists():
        if (
            expected_md5 is None
            or md5(destination)
            == expected_md5
        ):
            print(
                "Already downloaded:",
                destination.name,
            )
            return

        print(
            "Existing file has incorrect MD5; "
            "downloading again:",
            destination.name,
        )

        destination.unlink()

    if url.startswith(
        "ftp://"
    ):
        url = (
            "https://"
            + url[
                len("ftp://"):
            ]
        )

    elif not url.startswith(
        "http"
    ):
        url = (
            "https://"
            + url
        )

    temporary = Path(
        str(destination)
        + ".part"
    )

    if temporary.exists():
        temporary.unlink()

    print(
        "Downloading:",
        url,
    )

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "CF-State Phase2 Audit12d"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        with temporary.open(
            "wb"
        ) as handle:
            while True:
                block = response.read(
                    8 * 1024 * 1024
                )

                if not block:
                    break

                handle.write(
                    block
                )

    temporary.rename(
        destination
    )

    if expected_md5 is not None:
        observed = md5(
            destination
        )

        if observed != expected_md5:
            raise RuntimeError(
                "MD5 mismatch for "
                + str(destination)
                + ": expected "
                + expected_md5
                + ", observed "
                + observed
            )


def create_reference():
    from Bio import SeqIO

    REF_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fasta = (
        REF_DIR
        / "NC_002516.2.fasta"
    )

    saf = (
        REF_DIR
        / "NC_002516.2_model_CDS.saf"
    )

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        model_raw = json.load(
            handle
        )

    M = json_df(
        model_raw[
            "M"
        ]
    )

    model_ids = [
        str(x)
        for x in M.index
    ]

    model_set = set(
        model_ids
    )

    record = SeqIO.read(
        str(GENBANK),
        "genbank",
    )

    SeqIO.write(
        record,
        str(fasta),
        "fasta",
    )

    rows = []

    seen = set()

    for feature in record.features:
        if feature.type != "CDS":
            continue

        tags = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if not tags:
            continue

        locus = str(
            tags[0]
        ).strip()

        if locus not in model_set:
            continue

        if locus in seen:
            raise RuntimeError(
                "Duplicate CDS locus tag: "
                + locus
            )

        seen.add(
            locus
        )

        start = (
            int(
                feature.location.start
            )
            + 1
        )

        end = int(
            feature.location.end
        )

        strand = (
            "+"
            if feature.location.strand == 1
            else "-"
        )

        rows.append(
            {
                "GeneID":
                    locus,

                "Chr":
                    record.id,

                "Start":
                    start,

                "End":
                    end,

                "Strand":
                    strand,
            }
        )

    table = pd.DataFrame(
        rows
    )

    table.to_csv(
        saf,
        sep="\t",
        index=False,
    )

    found = set(
        table[
            "GeneID"
        ]
    )

    missing = sorted(
        model_set
        - found
    )

    print()
    print("Canonical counting reference")
    print(
        "  genome:",
        record.id,
    )
    print(
        "  CDS/model genes:",
        len(
            table
        ),
    )
    print(
        "  missing model genes:",
        len(
            missing
        ),
    )

    if len(
        table
    ) != 5572:
        raise RuntimeError(
            "Expected exactly 5572 model CDS features, found "
            + str(
                len(
                    table
                )
            )
        )

    if missing:
        raise RuntimeError(
            "Canonical CDS reference is missing model genes."
        )

    return (
        fasta,
        saf,
        table,
    )


def build_index(fasta):
    prefix = (
        REF_DIR
        / "pao1"
    )

    candidates = [
        Path(
            str(prefix)
            + ".1.bt2"
        ),
        Path(
            str(prefix)
            + ".1.bt2l"
        ),
    ]

    if not any(
        path.exists()
        for path in candidates
    ):
        run(
            "bowtie2-build "
            + str(fasta)
            + " "
            + str(prefix)
        )

    else:
        print()
        print(
            "Bowtie2 index already present."
        )

    return prefix


def parse_alignment_rate(log_path):
    text = log_path.read_text(
        errors="replace"
    )

    matches = re.findall(
        r"([0-9.]+)% overall alignment rate",
        text,
    )

    if not matches:
        return np.nan

    return float(
        matches[-1]
    )


def parse_assigned(summary_path):
    table = pd.read_csv(
        summary_path,
        sep="\t",
        index_col=0,
    )

    if "Assigned" not in table.index:
        raise RuntimeError(
            "Assigned row absent from "
            + str(summary_path)
        )

    return int(
        table.loc[
            "Assigned"
        ].iloc[0]
    )


def featurecounts_single(
    saf,
    bam,
    strand,
    output,
):
    run(
        "featureCounts "
        "-T "
        + str(THREADS)
        + " "
        "-F SAF "
        "-a "
        + str(saf)
        + " "
        "-o "
        + str(output)
        + " "
        "-p --countReadPairs "
        "-B -C "
        "-s "
        + str(strand)
        + " "
        + str(bam)
    )


def mean_pairwise_corr(
    matrix,
    columns,
):
    corr = matrix[
        columns
    ].corr()

    values = []

    for i in range(
        len(columns)
    ):
        for j in range(
            i + 1,
            len(columns),
        ):
            values.append(
                float(
                    corr.iloc[
                        i,
                        j,
                    ]
                )
            )

    return float(
        np.mean(
            values
        )
    )


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 12d")
    print("Canonical raw-read requantification of GSE299861 PAO1")
    print("=" * 78)

    for tool in [
        "cutadapt",
        "bowtie2",
        "bowtie2-build",
        "samtools",
        "featureCounts",
    ]:
        if shutil.which(
            tool
        ) is None:
            raise RuntimeError(
                "Required tool missing: "
                + tool
            )

    for path in [
        MANIFEST,
        GENBANK,
        MODEL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    for directory in [
        RAW_DIR,
        WORK,
        REF_DIR,
        TRIM_DIR,
        BAM_DIR,
        LOG_DIR,
        OUT,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest = pd.read_csv(
        MANIFEST
    )

    if len(
        manifest
    ) != 6:
        raise RuntimeError(
            "Expected six runs."
        )

    # ----------------------------------------------------------
    # 1. Canonical reference
    # ----------------------------------------------------------

    fasta, saf, gene_table = (
        create_reference()
    )

    index_prefix = build_index(
        fasta
    )

    # ----------------------------------------------------------
    # 2. Download FASTQs
    # ----------------------------------------------------------

    sample_rows = []

    for _, row in manifest.iterrows():
        srr = str(
            row[
                "srr"
            ]
        )

        gsm = str(
            row[
                "gsm"
            ]
        )

        condition = str(
            row[
                "condition"
            ]
        )

        replicate = int(
            row[
                "replicate"
            ]
        )

        urls = str(
            row[
                "fastq_ftp"
            ]
        ).split(
            ";"
        )

        md5s = str(
            row[
                "fastq_md5"
            ]
        ).split(
            ";"
        )

        if len(
            urls
        ) != 2:
            raise RuntimeError(
                srr
                + ": expected two paired FASTQ URLs, found "
                + str(
                    urls
                )
            )

        if len(
            md5s
        ) != 2:
            raise RuntimeError(
                srr
                + ": expected two FASTQ MD5 values."
            )

        raw1 = (
            RAW_DIR
            / (
                srr
                + "_1.fastq.gz"
            )
        )

        raw2 = (
            RAW_DIR
            / (
                srr
                + "_2.fastq.gz"
            )
        )

        download(
            urls[0],
            raw1,
            md5s[0],
        )

        download(
            urls[1],
            raw2,
            md5s[1],
        )

        sample_rows.append(
            {
                "gsm":
                    gsm,

                "srr":
                    srr,

                "condition":
                    condition,

                "replicate":
                    replicate,

                "raw_r1":
                    str(
                        raw1
                    ),

                "raw_r2":
                    str(
                        raw2
                    ),
            }
        )

    samples = pd.DataFrame(
        sample_rows
    )

    # ----------------------------------------------------------
    # 3. Adapter/quality trimming
    #
    # GEO specifies Cutadapt but not the exact command.
    # We use explicit standard Illumina adapters, Q20,
    # minimum 75 nt and <=10% Ns.
    # ----------------------------------------------------------

    for _, row in samples.iterrows():
        srr = row[
            "srr"
        ]

        raw1 = Path(
            row[
                "raw_r1"
            ]
        )

        raw2 = Path(
            row[
                "raw_r2"
            ]
        )

        trim1 = (
            TRIM_DIR
            / (
                srr
                + "_1.trim.fastq.gz"
            )
        )

        trim2 = (
            TRIM_DIR
            / (
                srr
                + "_2.trim.fastq.gz"
            )
        )

        trim_log = (
            LOG_DIR
            / (
                srr
                + ".cutadapt.log"
            )
        )

        if not (
            trim1.exists()
            and trim2.exists()
        ):
            command = (
                "cutadapt "
                "-j "
                + str(THREADS)
                + " "
                "-a "
                + ADAPTER_R1
                + " "
                "-A "
                + ADAPTER_R2
                + " "
                "-q 20,20 "
                "-m 75 "
                "--max-n 0.1 "
                "-o "
                + str(trim1)
                + " "
                "-p "
                + str(trim2)
                + " "
                + str(raw1)
                + " "
                + str(raw2)
                + " "
                "> "
                + str(trim_log)
                + " 2>&1"
            )

            run(
                command
            )

        else:
            print()
            print(
                "Trimmed FASTQs already present:",
                srr,
            )

    # ----------------------------------------------------------
    # 4. Align all six libraries to canonical PAO1
    # ----------------------------------------------------------

    alignment_rows = []

    for _, row in samples.iterrows():
        srr = row[
            "srr"
        ]

        trim1 = (
            TRIM_DIR
            / (
                srr
                + "_1.trim.fastq.gz"
            )
        )

        trim2 = (
            TRIM_DIR
            / (
                srr
                + "_2.trim.fastq.gz"
            )
        )

        bam = (
            BAM_DIR
            / (
                srr
                + ".sorted.bam"
            )
        )

        bowtie_log = (
            LOG_DIR
            / (
                srr
                + ".bowtie2.log"
            )
        )

        if not bam.exists():
            command = (
                "bowtie2 "
                "--very-sensitive "
                "-p "
                + str(THREADS)
                + " "
                "-x "
                + str(index_prefix)
                + " "
                "-1 "
                + str(trim1)
                + " "
                "-2 "
                + str(trim2)
                + " "
                "2> "
                + str(bowtie_log)
                + " "
                "| samtools sort "
                "-@ "
                + str(THREADS)
                + " "
                "-o "
                + str(bam)
                + " -"
            )

            run(
                command
            )

            run(
                "samtools index "
                + str(bam)
            )

        else:
            print()
            print(
                "BAM already present:",
                srr,
            )

        rate = parse_alignment_rate(
            bowtie_log
        )

        alignment_rows.append(
            {
                "gsm":
                    row[
                        "gsm"
                    ],

                "srr":
                    srr,

                "condition":
                    row[
                        "condition"
                    ],

                "replicate":
                    row[
                        "replicate"
                    ],

                "bam":
                    str(
                        bam
                    ),

                "overall_alignment_rate_pct":
                    rate,
            }
        )

    alignment_qc = pd.DataFrame(
        alignment_rows
    )

    print()
    print("Alignment QC")
    print(
        alignment_qc[
            [
                "gsm",
                "condition",
                "replicate",
                "srr",
                "overall_alignment_rate_pct",
            ]
        ]
        .round(2)
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Infer strandedness from one representative library
    # ----------------------------------------------------------

    pilot_bam = Path(
        alignment_qc.iloc[
            0
        ][
            "bam"
        ]
    )

    stranded_counts = {}

    for strand in [
        0,
        1,
        2,
    ]:
        output = (
            OUT
            / (
                "strandedness_s"
                + str(strand)
                + ".txt"
            )
        )

        featurecounts_single(
            saf,
            pilot_bam,
            strand,
            output,
        )

        assigned = parse_assigned(
            Path(
                str(output)
                + ".summary"
            )
        )

        stranded_counts[
            strand
        ] = assigned

    unstranded = stranded_counts[
        0
    ]

    forward = stranded_counts[
        1
    ]

    reverse = stranded_counts[
        2
    ]

    if forward >= reverse:
        chosen_strand = 1
        correct = forward
        wrong = reverse

    else:
        chosen_strand = 2
        correct = reverse
        wrong = forward

    strand_ratio = (
        correct
        / max(
            wrong,
            1,
        )
    )

    strand_retention = (
        correct
        / max(
            unstranded,
            1,
        )
    )

    print()
    print("Strandedness inference")
    print(
        "  featureCounts -s 0 assigned:",
        unstranded,
    )
    print(
        "  featureCounts -s 1 assigned:",
        forward,
    )
    print(
        "  featureCounts -s 2 assigned:",
        reverse,
    )
    print(
        "  inferred strand mode:",
        chosen_strand,
    )
    print(
        "  correct/wrong assignment ratio:",
        round(
            strand_ratio,
            3,
        ),
    )
    print(
        "  correct/unstranded retention:",
        round(
            strand_retention,
            3,
        ),
    )

    if (
        strand_ratio < 3.0
        or strand_retention < 0.60
    ):
        raise RuntimeError(
            "Strandedness could not be inferred confidently. "
            "Stop before final counting."
        )

    # ----------------------------------------------------------
    # 6. Count all six libraries using inferred strand mode
    # ----------------------------------------------------------

    all_bams = [
        Path(x)
        for x in alignment_qc[
            "bam"
        ]
    ]

    count_output = (
        OUT
        / "GSE299861_featureCounts.txt"
    )

    command = (
        "featureCounts "
        "-T "
        + str(THREADS)
        + " "
        "-F SAF "
        "-a "
        + str(saf)
        + " "
        "-o "
        + str(count_output)
        + " "
        "-p --countReadPairs "
        "-B -C "
        "-s "
        + str(chosen_strand)
        + " "
        + " ".join(
            str(x)
            for x in all_bams
        )
    )

    run(
        command
    )

    counts_raw = pd.read_csv(
        count_output,
        sep="\t",
        comment="#",
    )

    annotation_cols = [
        "Geneid",
        "Chr",
        "Start",
        "End",
        "Strand",
        "Length",
    ]

    count_cols = [
        col
        for col in counts_raw.columns
        if col not in annotation_cols
    ]

    if len(
        count_cols
    ) != 6:
        raise RuntimeError(
            "Expected six count columns, found "
            + str(
                len(
                    count_cols
                )
            )
        )

    ordered_names = []

    for _, row in alignment_qc.iterrows():
        ordered_names.append(
            str(
                row[
                    "gsm"
                ]
            )
            + "_"
            + str(
                row[
                    "condition"
                ]
            )
            + "_"
            + str(
                int(
                    row[
                        "replicate"
                    ]
                )
            )
        )

    count_matrix = (
        counts_raw[
            count_cols
        ]
        .copy()
    )

    count_matrix.index = (
        counts_raw[
            "Geneid"
        ]
        .astype(str)
        .values
    )

    count_matrix.columns = (
        ordered_names
    )

    count_matrix = (
        count_matrix
        .apply(
            pd.to_numeric,
            errors="raise",
        )
    )

    lengths = pd.Series(
        counts_raw[
            "Length"
        ].to_numpy(
            dtype=float
        ),
        index=count_matrix.index,
    )

    if count_matrix.shape != (
        5572,
        6,
    ):
        raise RuntimeError(
            "Expected canonical count matrix (5572, 6), found "
            + str(
                count_matrix.shape
            )
        )

    # ----------------------------------------------------------
    # 7. Counts -> TPM
    # ----------------------------------------------------------

    length_kb = (
        lengths
        / 1000.0
    )

    rpk = count_matrix.divide(
        length_kb,
        axis=0,
    )

    tpm = rpk.divide(
        rpk.sum(
            axis=0
        ),
        axis=1,
    ) * 1_000_000.0

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    # ----------------------------------------------------------
    # 8. Count assignment QC
    # ----------------------------------------------------------

    summary = pd.read_csv(
        str(
            count_output
        )
        + ".summary",
        sep="\t",
        index_col=0,
    )

    assigned = summary.loc[
        "Assigned"
    ].to_numpy(
        dtype=float
    )

    total = summary.sum(
        axis=0
    ).to_numpy(
        dtype=float
    )

    assignment_fraction = (
        assigned
        / total
    )

    alignment_qc[
        "assigned_fragment_fraction"
    ] = assignment_fraction

    biofilm_cols = [
        col
        for col in log_tpm.columns
        if "_biofilm_" in col
    ]

    planktonic_cols = [
        col
        for col in log_tpm.columns
        if "_planktonic_" in col
    ]

    biofilm_r = mean_pairwise_corr(
        log_tpm,
        biofilm_cols,
    )

    planktonic_r = mean_pairwise_corr(
        log_tpm,
        planktonic_cols,
    )

    print()
    print("Canonical expression QC")
    print(
        "  count matrix:",
        count_matrix.shape,
    )
    print(
        "  TPM column-sum min:",
        round(
            float(
                tpm.sum(
                    axis=0
                ).min()
            ),
            3,
        ),
    )
    print(
        "  TPM column-sum max:",
        round(
            float(
                tpm.sum(
                    axis=0
                ).max()
            ),
            3,
        ),
    )
    print(
        "  planktonic replicate mean r:",
        round(
            planktonic_r,
            4,
        ),
    )
    print(
        "  biofilm replicate mean r:",
        round(
            biofilm_r,
            4,
        ),
    )

    print()
    print("Per-sample QC")
    print(
        alignment_qc[
            [
                "gsm",
                "condition",
                "replicate",
                "overall_alignment_rate_pct",
                "assigned_fragment_fraction",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Save frozen canonical expression matrices
    # ----------------------------------------------------------

    count_matrix.to_csv(
        OUT
        / "GSE299861_canonical_counts.csv"
    )

    tpm.to_csv(
        OUT
        / "GSE299861_canonical_TPM.csv"
    )

    log_tpm.to_csv(
        OUT
        / "GSE299861_canonical_log2TPM_plus1.csv"
    )

    alignment_qc.to_csv(
        OUT
        / "sample_qc.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "featureCounts_strand_mode":
                    chosen_strand,

                "pilot_unstranded_assigned":
                    unstranded,

                "pilot_forward_assigned":
                    forward,

                "pilot_reverse_assigned":
                    reverse,

                "correct_wrong_ratio":
                    strand_ratio,

                "correct_unstranded_retention":
                    strand_retention,

                "planktonic_replicate_mean_r":
                    planktonic_r,

                "biofilm_replicate_mean_r":
                    biofilm_r,
            }
        ]
    ).to_csv(
        OUT
        / "requantification_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 10. Audit gate
    # ----------------------------------------------------------

    min_alignment = float(
        alignment_qc[
            "overall_alignment_rate_pct"
        ].min()
    )

    min_assignment = float(
        alignment_qc[
            "assigned_fragment_fraction"
        ].min()
    )

    checks = {
        "canonical_5572_gene_matrix":
            count_matrix.shape
            == (
                5572,
                6,
            ),

        "six_samples":
            count_matrix.shape[
                1
            ]
            == 6,

        "strandedness_confident":
            (
                strand_ratio
                >= 3.0
                and strand_retention
                >= 0.60
            ),

        "minimum_alignment_ge_80pct":
            min_alignment
            >= 80.0,

        "minimum_assigned_fragment_fraction_ge_50pct":
            min_assignment
            >= 0.50,

        "planktonic_replicate_r_ge_075":
            planktonic_r
            >= 0.75,

        "biofilm_replicate_r_ge_075":
            biofilm_r
            >= 0.75,

        "TPM_sums_to_1e6":
            bool(
                np.allclose(
                    tpm.sum(
                        axis=0
                    ).to_numpy(),
                    1_000_000.0,
                    rtol=1e-8,
                    atol=1e-3,
                )
            ),
    }

    passed = all(
        checks.values()
    )

    pd.DataFrame(
        [
            {
                "check":
                    key,

                "pass":
                    value,
            }
            for key, value
            in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    print()
    print("Audit checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    lines = [
        "# CF-State Phase 2 — Audit 12d",
        "",
        "## Objective",
        "",
        "Re-quantify the six GSE299861 PAO1 RNA-seq libraries "
        "directly against the frozen canonical PAO1 reference "
        "NC_002516.2, bypassing the custom PAO1DT annotation.",
        "",
        "## Processing",
        "",
        "- Paired-end raw FASTQs from ENA/SRA",
        "- Cutadapt adapter/quality trimming",
        "- Bowtie2 alignment to NC_002516.2",
        "- Empirical featureCounts strandedness inference",
        "- Paired-fragment feature counting over the 5572 "
        "aeruPRECISE CDS genes",
        "- Counts converted to TPM using canonical CDS lengths",
        "",
        "## QC",
        "",
        "- Inferred featureCounts strand mode: "
        + str(
            chosen_strand
        ),
        "- Correct/wrong strand ratio: "
        + format(
            strand_ratio,
            ".3f",
        ),
        "- Minimum alignment rate: "
        + format(
            min_alignment,
            ".2f",
        )
        + "%",
        "- Minimum assigned-fragment fraction: "
        + format(
            min_assignment,
            ".2%",
        ),
        "- Planktonic replicate correlation: "
        + format(
            planktonic_r,
            ".4f",
        ),
        "- Biofilm replicate correlation: "
        + format(
            biofilm_r,
            ".4f",
        ),
        "",
        "## Decision",
        "",
        "PASS: "
        + str(
            passed
        ),
        "",
        "No iModulon projection or biological interpretation "
        "was performed in this audit.",
    ]

    (
        OUT
        / "AUDIT_12D_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 12d:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_12D_SUMMARY.md",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
