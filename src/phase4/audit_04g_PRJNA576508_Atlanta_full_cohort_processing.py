from pathlib import Path
import gzip
import hashlib
import re
import shutil
import subprocess

import pandas as pd


ROOT = Path.cwd()

THREADS = 8

ADAPTER = "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
MINIMUM_LENGTH = 25
ERROR_RATE = 0.1
MINIMUM_OVERLAP = 3

MAX_ASSIGNED_RELATIVE_ERROR = 0.25
MAX_DETECTED_RELATIVE_ERROR = 0.05

MANIFEST = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "PRJNA576508_Atlanta_sputum_manifest.csv"
)

PUBLISHED_METADATA = (
    ROOT
    / "results/phase4/audit_04b2_PRJNA576508_supplement_parser_repair"
    / "FROZEN_Cornforth_13_Atlanta_sample_metadata.csv"
)

REFERENCE_DIR = (
    ROOT
    / "references/phase4/PRJNA576508_cornforth"
)

DECOY_INDEX = (
    REFERENCE_DIR
    / "bowtie2_decoy/Cornforth_104_decoys"
)

PAO1_INDEX = (
    REFERENCE_DIR
    / "bowtie2_PAO1/PAO1_NC_002516.2"
)

PAO1_SAF = (
    REFERENCE_DIR
    / "Cornforth_PAO1_no_decimal_locus_tags.saf"
)

RAW_DIR = (
    ROOT
    / "data/raw/phase4/PRJNA576508/Atlanta_13"
)

WORK_DIR = (
    ROOT
    / "data/processed/phase4/PRJNA576508/Atlanta_13"
)

OUT = (
    ROOT
    / "results/phase4/audit_04g_PRJNA576508_Atlanta_full_cohort_processing"
)

COUNT_DIR = (
    OUT
    / "per_sample_counts"
)

SAMPLE_SUMMARY_DIR = (
    OUT
    / "per_sample_summaries"
)

LOG_DIR = (
    OUT
    / "logs"
)


def md5_file(path):
    digest = hashlib.md5()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def normalize_first(value):
    return str(
        value
    ).split(";")[0].strip()


def normalize_url(value):
    text = normalize_first(
        value
    )

    if text.startswith("ftp://"):
        return (
            "https://"
            + text[len("ftp://"):]
        )

    if text.startswith("ftp."):
        return (
            "https://"
            + text
        )

    return text


def run_command(
    command,
    stdout_path=None,
    stderr_path=None,
):
    print()
    print(
        "Running:"
    )
    print(
        " ",
        " ".join(
            str(x)
            for x in command
        )
    )

    stdout_handle = None
    stderr_handle = None

    try:
        if stdout_path is not None:
            stdout_handle = open(
                stdout_path,
                "w",
                encoding="utf-8",
            )

        if stderr_path is not None:
            stderr_handle = open(
                stderr_path,
                "w",
                encoding="utf-8",
            )

        result = subprocess.run(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )

    finally:
        if stdout_handle is not None:
            stdout_handle.close()

        if stderr_handle is not None:
            stderr_handle.close()

    if result.returncode != 0:
        raise RuntimeError(
            "Command failed with return code "
            + str(result.returncode)
        )


def index_exists(prefix):
    prefix = str(
        prefix
    )

    small = [
        prefix + ".1.bt2",
        prefix + ".2.bt2",
        prefix + ".3.bt2",
        prefix + ".4.bt2",
        prefix + ".rev.1.bt2",
        prefix + ".rev.2.bt2",
    ]

    large = [
        prefix + ".1.bt2l",
        prefix + ".2.bt2l",
        prefix + ".3.bt2l",
        prefix + ".4.bt2l",
        prefix + ".rev.1.bt2l",
        prefix + ".rev.2.bt2l",
    ]

    return (
        all(
            Path(x).exists()
            for x in small
        )
        or all(
            Path(x).exists()
            for x in large
        )
    )


def download_fastq(
    url,
    destination,
    expected_md5,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if destination.exists():
        observed = md5_file(
            destination
        )

        if observed == expected_md5:
            print(
                "  existing raw FASTQ MD5 PASS"
            )
            return

        print(
            "  existing raw FASTQ MD5 mismatch; redownloading"
        )

        destination.unlink()

    temporary = Path(
        str(destination)
        + ".part"
    )

    temporary.unlink(
        missing_ok=True
    )

    command = [
        shutil.which("curl"),
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--continue-at",
        "-",
        "--output",
        str(temporary),
        url,
    ]

    run_command(
        command
    )

    observed = md5_file(
        temporary
    )

    if observed != expected_md5:
        raise RuntimeError(
            "Downloaded FASTQ MD5 mismatch."
        )

    temporary.rename(
        destination
    )


def parse_cutadapt_report(text):
    patterns = {
        "input_reads":
            r"Total reads processed:\s*([\d,]+)",

        "adapter_reads":
            r"Reads with adapters:\s*([\d,]+)",

        "too_short":
            r"Reads that were too short:\s*([\d,]+)",

        "written_reads":
            r"Reads written \(passing filters\):\s*([\d,]+)",
    }

    values = {}

    for name, pattern in patterns.items():
        match = re.search(
            pattern,
            text,
        )

        if match is None:
            raise RuntimeError(
                "Could not parse Cutadapt field "
                + name
            )

        values[name] = int(
            match.group(1).replace(
                ",",
                "",
            )
        )

    return values


def parse_bowtie_summary(path):
    text = path.read_text(
        encoding="utf-8"
    )

    patterns = {
        "total":
            r"^\s*([\d,]+)\s+reads;",

        "zero":
            (
                r"^\s*([\d,]+)\s+\([\d.]+%\)"
                r"\s+aligned 0 times"
            ),

        "once":
            (
                r"^\s*([\d,]+)\s+\([\d.]+%\)"
                r"\s+aligned exactly 1 time"
            ),

        "multi":
            (
                r"^\s*([\d,]+)\s+\([\d.]+%\)"
                r"\s+aligned >1 times"
            ),
    }

    values = {}

    for name, pattern in patterns.items():
        match = re.search(
            pattern,
            text,
            flags=re.MULTILINE,
        )

        if match is None:
            raise RuntimeError(
                "Could not parse Bowtie2 field "
                + name
            )

        values[name] = int(
            match.group(1).replace(
                ",",
                "",
            )
        )

    values["mapped"] = (
        values["once"]
        + values["multi"]
    )

    values["mapped_fraction"] = (
        values["mapped"]
        / values["total"]
    )

    return values


def count_fastq_reads(path):
    lines = 0

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
    ) as handle:
        for lines, _ in enumerate(
            handle,
            start=1,
        ):
            pass

    if lines % 4 != 0:
        raise RuntimeError(
            "FASTQ line count not divisible by four."
        )

    return (
        lines
        // 4
    )


def run_featurecounts(
    sample,
    sam,
    count_csv,
    stat_csv,
):
    rscript = shutil.which(
        "Rscript"
    )

    script = (
        OUT
        / (
            "run_Rsubread_"
            + sample
            + ".R"
        )
    )

    code = (
        'suppressPackageStartupMessages(library(Rsubread))\n'
        + 'sam <- "'
        + str(sam)
        + '"\n'
        + 'saf <- "'
        + str(PAO1_SAF)
        + '"\n'
        + 'count_out <- "'
        + str(count_csv)
        + '"\n'
        + 'stat_out <- "'
        + str(stat_csv)
        + '"\n'
        + 'fc <- featureCounts(\n'
        + '  files=sam,\n'
        + '  annot.ext=saf,\n'
        + '  isGTFAnnotationFile=FALSE,\n'
        + '  isPairedEnd=FALSE,\n'
        + '  strandSpecific=0\n'
        + ')\n'
        + 'counts <- data.frame(\n'
        + '  GeneID=rownames(fc$counts),\n'
        + '  count=as.numeric(fc$counts[,1]),\n'
        + '  stringsAsFactors=FALSE\n'
        + ')\n'
        + 'write.csv(\n'
        + '  counts,\n'
        + '  count_out,\n'
        + '  row.names=FALSE,\n'
        + '  quote=FALSE\n'
        + ')\n'
        + 'if ("Status" %in% colnames(fc$stat)) {\n'
        + '  status_values <- as.character(fc$stat$Status)\n'
        + '  count_columns <- setdiff(colnames(fc$stat), "Status")\n'
        + '} else {\n'
        + '  status_values <- rownames(fc$stat)\n'
        + '  count_columns <- colnames(fc$stat)\n'
        + '}\n'
        + 'if (length(count_columns) != 1) {\n'
        + '  stop("Expected one fc$stat count column")\n'
        + '}\n'
        + 'stats <- data.frame(\n'
        + '  Status=status_values,\n'
        + '  count=as.numeric(fc$stat[[count_columns[1]]]),\n'
        + '  stringsAsFactors=FALSE\n'
        + ')\n'
        + 'write.csv(\n'
        + '  stats,\n'
        + '  stat_out,\n'
        + '  row.names=FALSE,\n'
        + '  quote=FALSE\n'
        + ')\n'
    )

    script.write_text(
        code,
        encoding="utf-8",
    )

    run_command(
        [
            rscript,
            str(script),
        ]
    )


def process_sample(
    row,
    published_row,
):
    sample = str(
        row["short_sample_label"]
    )

    run = str(
        row["run_accession"]
    )

    published_assigned = int(
        published_row[
            "published_assigned"
        ]
    )

    published_detected = int(
        published_row[
            "published_detected"
        ]
    )

    sample_summary_path = (
        SAMPLE_SUMMARY_DIR
        / (
            sample
            + "_technical_summary.csv"
        )
    )

    count_csv = (
        COUNT_DIR
        / (
            sample
            + "_gene_counts.csv"
        )
    )

    # ------------------------------------------------------
    # Resume support
    # ------------------------------------------------------

    if (
        sample_summary_path.exists()
        and count_csv.exists()
    ):
        existing = pd.read_csv(
            sample_summary_path
        )

        if (
            len(existing) == 1
            and bool(
                existing.iloc[0][
                    "sample_technical_pass"
                ]
            )
        ):
            print()
            print("=" * 78)
            print(
                sample,
                "already completed and passed; reusing frozen output"
            )
            print("=" * 78)

            return existing.iloc[0].to_dict()

    print()
    print("=" * 78)
    print(
        "PROCESSING",
        sample,
        run,
    )
    print("=" * 78)

    sample_work = (
        WORK_DIR
        / sample
    )

    sample_work.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = (
        RAW_DIR
        / (
            run
            + ".fastq.gz"
        )
    )

    trimmed = (
        sample_work
        / (
            run
            + ".TruSeq_trimmed.min25.fastq.gz"
        )
    )

    decoy_unmapped = (
        sample_work
        / (
            run
            + ".after_decoy_filter.fastq.gz"
        )
    )

    sam = (
        sample_work
        / (
            run
            + ".PAO1.sam"
        )
    )

    cutadapt_log = (
        LOG_DIR
        / (
            sample
            + "_cutadapt.log"
        )
    )

    decoy_log = (
        LOG_DIR
        / (
            sample
            + "_decoy_bowtie2.log"
        )
    )

    pao1_log = (
        LOG_DIR
        / (
            sample
            + "_PAO1_bowtie2.log"
        )
    )

    stat_csv = (
        OUT
        / (
            sample
            + "_Rsubread_stat.csv"
        )
    )

    # ------------------------------------------------------
    # Raw FASTQ
    # ------------------------------------------------------

    expected_md5 = normalize_first(
        row[
            "fastq_md5"
        ]
    ).lower()

    url = normalize_url(
        row[
            "fastq_ftp"
        ]
    )

    download_fastq(
        url,
        raw,
        expected_md5,
    )

    raw_md5 = md5_file(
        raw
    )

    # ------------------------------------------------------
    # Frozen trimming rule
    # ------------------------------------------------------

    trimmed.unlink(
        missing_ok=True
    )

    cutadapt_command = [
        shutil.which(
            "cutadapt"
        ),
        "-a",
        ADAPTER,
        "-m",
        str(
            MINIMUM_LENGTH
        ),
        "-e",
        str(
            ERROR_RATE
        ),
        "-O",
        str(
            MINIMUM_OVERLAP
        ),
        "-j",
        "4",
        "-o",
        str(
            trimmed
        ),
        str(
            raw
        ),
    ]

    print()
    print("Cutadapt")

    result = subprocess.run(
        cutadapt_command,
        capture_output=True,
        text=True,
    )

    cutadapt_text = (
        result.stdout
        + "\n"
        + result.stderr
    )

    cutadapt_log.write_text(
        cutadapt_text,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            sample
            + ": Cutadapt failed."
        )

    cutadapt = parse_cutadapt_report(
        cutadapt_text
    )

    adapter_fraction = (
        cutadapt[
            "adapter_reads"
        ]
        / cutadapt[
            "input_reads"
        ]
    )

    # ------------------------------------------------------
    # Frozen decoy alignment
    # ------------------------------------------------------

    decoy_unmapped.unlink(
        missing_ok=True
    )

    run_command(
        [
            shutil.which(
                "bowtie2"
            ),
            "-x",
            str(
                DECOY_INDEX
            ),
            "-U",
            str(
                trimmed
            ),
            "-p",
            str(
                THREADS
            ),
            "--un-gz",
            str(
                decoy_unmapped
            ),
            "-S",
            "/dev/null",
        ],
        stderr_path=decoy_log,
    )

    decoy = parse_bowtie_summary(
        decoy_log
    )

    retained_after_decoy = count_fastq_reads(
        decoy_unmapped
    )

    if (
        retained_after_decoy
        != decoy["zero"]
    ):
        raise RuntimeError(
            sample
            + ": decoy retained-read mismatch."
        )

    # ------------------------------------------------------
    # Frozen PAO1 alignment
    # ------------------------------------------------------

    sam.unlink(
        missing_ok=True
    )

    run_command(
        [
            shutil.which(
                "bowtie2"
            ),
            "-x",
            str(
                PAO1_INDEX
            ),
            "-U",
            str(
                decoy_unmapped
            ),
            "-p",
            str(
                THREADS
            ),
            "-S",
            str(
                sam
            ),
        ],
        stderr_path=pao1_log,
    )

    pao1 = parse_bowtie_summary(
        pao1_log
    )

    # ------------------------------------------------------
    # Frozen Rsubread counting
    # ------------------------------------------------------

    run_featurecounts(
        sample,
        sam,
        count_csv,
        stat_csv,
    )

    counts = pd.read_csv(
        count_csv
    )

    stats = pd.read_csv(
        stat_csv
    )

    assigned_row = stats.loc[
        stats[
            "Status"
        ]
        .astype(str)
        .eq(
            "Assigned"
        )
    ]

    if len(
        assigned_row
    ) != 1:
        raise RuntimeError(
            sample
            + ": Assigned status not unique."
        )

    assigned = int(
        assigned_row.iloc[0][
            "count"
        ]
    )

    sum_counts = int(
        counts[
            "count"
        ].sum()
    )

    detected = int(
        (
            counts[
                "count"
            ]
            > 0
        ).sum()
    )

    if assigned != sum_counts:
        raise RuntimeError(
            sample
            + ": assigned != sum gene counts."
        )

    assigned_error = abs(
        assigned
        - published_assigned
    ) / published_assigned

    detected_error = abs(
        detected
        - published_detected
    ) / published_detected

    sample_pass = (
        assigned_error
        <= MAX_ASSIGNED_RELATIVE_ERROR
        and detected_error
        <= MAX_DETECTED_RELATIVE_ERROR
    )

    summary = {
        "sample":
            sample,

        "run_accession":
            run,

        "raw_MD5_pass":
            raw_md5
            == expected_md5,

        "raw_reads":
            cutadapt[
                "input_reads"
            ],

        "adapter_reads":
            cutadapt[
                "adapter_reads"
            ],

        "adapter_fraction":
            adapter_fraction,

        "trimmed_retained_reads":
            cutadapt[
                "written_reads"
            ],

        "trimmed_too_short":
            cutadapt[
                "too_short"
            ],

        "decoy_mapped_reads":
            decoy[
                "mapped"
            ],

        "decoy_mapped_fraction":
            decoy[
                "mapped_fraction"
            ],

        "reads_after_decoy_filter":
            retained_after_decoy,

        "PAO1_Bowtie2_mapped_reads":
            pao1[
                "mapped"
            ],

        "PAO1_Bowtie2_mapping_fraction":
            pao1[
                "mapped_fraction"
            ],

        "reconstructed_assigned_reads":
            assigned,

        "published_assigned_reads":
            published_assigned,

        "assigned_ratio":
            assigned
            / published_assigned,

        "assigned_relative_error":
            assigned_error,

        "reconstructed_detected_genes":
            detected,

        "published_detected_genes":
            published_detected,

        "detected_gene_ratio":
            detected
            / published_detected,

        "detected_gene_relative_error":
            detected_error,

        "sample_technical_pass":
            sample_pass,
    }

    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        sample_summary_path,
        index=False,
    )

    print()
    print(
        sample,
        "technical benchmark"
    )

    print(
        "  assigned:",
        assigned,
        "published:",
        published_assigned,
        "relative error:",
        round(
            assigned_error,
            6,
        ),
    )

    print(
        "  detected genes:",
        detected,
        "published:",
        published_detected,
        "relative error:",
        round(
            detected_error,
            6,
        ),
    )

    print(
        "  sample technical status:",
        (
            "PASS"
            if sample_pass
            else "FAIL"
        ),
    )

    # ------------------------------------------------------
    # Remove reproducible large intermediates only when this
    # sample passes. Raw FASTQ and final counts are retained.
    # ------------------------------------------------------

    if sample_pass:
        sam.unlink(
            missing_ok=True
        )

        decoy_unmapped.unlink(
            missing_ok=True
        )

        trimmed.unlink(
            missing_ok=True
        )

    return summary


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04g")
    print("Frozen Cornforth pipeline across all 13 Atlanta sputum samples")
    print("=" * 78)

    for path in [
        MANIFEST,
        PUBLISHED_METADATA,
        PAO1_SAF,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    if not index_exists(
        DECOY_INDEX
    ):
        raise RuntimeError(
            "Frozen decoy Bowtie2 index missing."
        )

    if not index_exists(
        PAO1_INDEX
    ):
        raise RuntimeError(
            "Frozen PAO1 Bowtie2 index missing."
        )

    for executable in [
        "curl",
        "cutadapt",
        "bowtie2",
        "Rscript",
    ]:
        if shutil.which(
            executable
        ) is None:
            raise RuntimeError(
                executable
                + " unavailable."
            )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    COUNT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAMPLE_SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST
    )

    published = pd.read_csv(
        PUBLISHED_METADATA
    )

    assigned_col = [
        column
        for column in published.columns
        if "reads mapping" in column.lower()
        and "pao1" in column.lower()
    ]

    detected_col = [
        column
        for column in published.columns
        if "number of pao1" in column.lower()
        and "genes" in column.lower()
    ]

    if len(
        assigned_col
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify published assigned-read column."
        )

    if len(
        detected_col
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify published detected-gene column."
        )

    published = published.rename(
        columns={
            assigned_col[0]:
                "published_assigned",

            detected_col[0]:
                "published_detected",
        }
    )

    published[
        "published_assigned"
    ] = pd.to_numeric(
        published[
            "published_assigned"
        ],
        errors="raise",
    )

    published[
        "published_detected"
    ] = pd.to_numeric(
        published[
            "published_detected"
        ],
        errors="raise",
    )

    merged = manifest.merge(
        published[
            [
                "normalized_EM_label",
                "published_assigned",
                "published_detected",
            ]
        ],
        left_on="short_sample_label",
        right_on="normalized_EM_label",
        how="left",
        validate="one_to_one",
    )

    if len(
        merged
    ) != 13:
        raise RuntimeError(
            "Expected exactly 13 Atlanta samples."
        )

    if merged[
        [
            "published_assigned",
            "published_detected",
        ]
    ].isna().any().any():
        raise RuntimeError(
            "Missing published benchmark after metadata merge."
        )

    merged = merged.sort_values(
        "short_sample_label"
    ).reset_index(
        drop=True
    )

    print()
    print("Frozen full-cohort technical acceptance criteria")
    print(
        "  samples:",
        len(
            merged
        ),
    )
    print(
        "  assigned-read relative error <=",
        MAX_ASSIGNED_RELATIVE_ERROR,
    )
    print(
        "  detected-gene relative error <=",
        MAX_DETECTED_RELATIVE_ERROR,
    )
    print(
        "  pipeline settings are identical to validated EM36 pilot"
    )

    summaries = []

    for _, row in merged.iterrows():
        published_row = {
            "published_assigned":
                int(
                    row[
                        "published_assigned"
                    ]
                ),

            "published_detected":
                int(
                    row[
                        "published_detected"
                    ]
                ),
        }

        summary = process_sample(
            row,
            published_row,
        )

        summaries.append(
            summary
        )

    summary = pd.DataFrame(
        summaries
    ).sort_values(
        "sample"
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        OUT
        / "Atlanta_13_technical_replication_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Assemble frozen raw count matrix
    # ----------------------------------------------------------

    count_matrix = None

    for sample in summary[
        "sample"
    ]:
        path = (
            COUNT_DIR
            / (
                sample
                + "_gene_counts.csv"
            )
        )

        if not path.exists():
            raise RuntimeError(
                "Missing count file for "
                + sample
            )

        table = pd.read_csv(
            path
        )

        table = table.rename(
            columns={
                "count":
                    sample
            }
        )

        if count_matrix is None:
            count_matrix = table

        else:
            count_matrix = count_matrix.merge(
                table,
                on="GeneID",
                how="outer",
                validate="one_to_one",
            )

    if count_matrix is None:
        raise RuntimeError(
            "No count matrix generated."
        )

    count_matrix = count_matrix.fillna(
        0
    )

    for sample in summary[
        "sample"
    ]:
        count_matrix[
            sample
        ] = pd.to_numeric(
            count_matrix[
                sample
            ],
            errors="raise",
        ).astype(
            int
        )

    count_matrix.to_csv(
        OUT
        / "FROZEN_Atlanta_13_raw_PAO1_gene_counts.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Cohort-level summary
    # ----------------------------------------------------------

    print()
    print("=" * 78)
    print("Atlanta 13 technical replication summary")
    print("=" * 78)

    display_columns = [
        "sample",
        "reconstructed_assigned_reads",
        "published_assigned_reads",
        "assigned_ratio",
        "assigned_relative_error",
        "reconstructed_detected_genes",
        "published_detected_genes",
        "detected_gene_ratio",
        "detected_gene_relative_error",
        "sample_technical_pass",
    ]

    print(
        summary[
            display_columns
        ]
        .round(
            {
                "assigned_ratio":
                    4,

                "assigned_relative_error":
                    4,

                "detected_gene_ratio":
                    4,

                "detected_gene_relative_error":
                    4,
            }
        )
        .to_string(
            index=False
        )
    )

    n_pass = int(
        summary[
            "sample_technical_pass"
        ].sum()
    )

    max_assigned_error = float(
        summary[
            "assigned_relative_error"
        ].max()
    )

    max_detected_error = float(
        summary[
            "detected_gene_relative_error"
        ].max()
    )

    median_assigned_error = float(
        summary[
            "assigned_relative_error"
        ].median()
    )

    median_detected_error = float(
        summary[
            "detected_gene_relative_error"
        ].median()
    )

    print()
    print("Cohort technical QC")
    print(
        "  samples passing both pre-frozen gates:",
        n_pass,
        "/ 13",
    )
    print(
        "  median assigned-read relative error:",
        round(
            median_assigned_error,
            6,
        ),
    )
    print(
        "  maximum assigned-read relative error:",
        round(
            max_assigned_error,
            6,
        ),
    )
    print(
        "  median detected-gene relative error:",
        round(
            median_detected_error,
            6,
        ),
    )
    print(
        "  maximum detected-gene relative error:",
        round(
            max_detected_error,
            6,
        ),
    )
    print(
        "  frozen count-matrix genes:",
        len(
            count_matrix
        ),
    )

    checks = {
        "exactly_13_Atlanta_samples":
            len(
                summary
            )
            == 13,

        "all_13_raw_MD5_pass":
            bool(
                summary[
                    "raw_MD5_pass"
                ].all()
            ),

        "all_13_have_nonzero_assigned_reads":
            bool(
                (
                    summary[
                        "reconstructed_assigned_reads"
                    ]
                    > 0
                ).all()
            ),

        "all_13_have_nonzero_detected_genes":
            bool(
                (
                    summary[
                        "reconstructed_detected_genes"
                    ]
                    > 0
                ).all()
            ),

        "all_13_assigned_reads_within_prefrozen_25pct":
            bool(
                (
                    summary[
                        "assigned_relative_error"
                    ]
                    <= MAX_ASSIGNED_RELATIVE_ERROR
                ).all()
            ),

        "all_13_detected_genes_within_prefrozen_5pct":
            bool(
                (
                    summary[
                        "detected_gene_relative_error"
                    ]
                    <= MAX_DETECTED_RELATIVE_ERROR
                ).all()
            ),

        "all_13_sample_technical_pass":
            bool(
                summary[
                    "sample_technical_pass"
                ].all()
            ),

        "frozen_count_matrix_has_5567_genes":
            len(
                count_matrix
            )
            == 5567,

        "frozen_count_matrix_has_13_sample_columns":
            len(
                [
                    column
                    for column in count_matrix.columns
                    if column != "GeneID"
                ]
            )
            == 13,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04g checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    status = (
        "PRJNA576508_ATLANTA_13_PUBLISHED_PIPELINE_TECHNICALLY_REPLICATED"
        if passed
        else "PRJNA576508_ATLANTA_13_TECHNICAL_REPLICATION_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: the complete Atlanta cohort has only undergone "
        "the processing pipeline validated prospectively on EM36. "
        "No normalization, iModulon projection, controlled-state "
        "decomposition, or comparison with the frozen Audit-03h "
        "CF-airway candidate has been performed."
    )

    pd.DataFrame(
        [
            {
                "check":
                    key,
                "pass":
                    value,
            }
            for key, value in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    (
        OUT
        / "AUDIT_04G_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    if not passed:
        failed = [
            key
            for key, value in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed Audit 04g checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
