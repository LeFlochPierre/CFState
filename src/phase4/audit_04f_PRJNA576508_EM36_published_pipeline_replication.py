from pathlib import Path
from urllib.request import Request, urlopen
import gzip
import hashlib
import re
import shutil
import subprocess

import pandas as pd


ROOT = Path.cwd()

THREADS = 8

TRIMMED_FASTQ = (
    ROOT
    / "data/processed/phase4/PRJNA576508/pilot_EM36"
    / "SRR10267761.TruSeq_trimmed.min25.fastq.gz"
)

REFERENCE_DIR = (
    ROOT
    / "references/phase4/PRJNA576508_cornforth"
)

DECOY_FASTA = (
    REFERENCE_DIR
    / "Cornforth_exact_104_decoy_sequences.fasta"
)

PAO1_FASTA = (
    REFERENCE_DIR
    / "PAO1_NC_002516.2.fasta"
)

PAO1_GFF_GZ = (
    REFERENCE_DIR
    / "GCF_000006765.1_ASM676v1_genomic.gff.gz"
)

PAO1_GFF = (
    REFERENCE_DIR
    / "GCF_000006765.1_ASM676v1_genomic.gff"
)

PAO1_SAF = (
    REFERENCE_DIR
    / "Cornforth_PAO1_no_decimal_locus_tags.saf"
)

DECOY_INDEX_PREFIX = (
    REFERENCE_DIR
    / "bowtie2_decoy"
    / "Cornforth_104_decoys"
)

PAO1_INDEX_PREFIX = (
    REFERENCE_DIR
    / "bowtie2_PAO1"
    / "PAO1_NC_002516.2"
)

WORK = (
    ROOT
    / "data/processed/phase4/PRJNA576508/pilot_EM36/04f_mapping"
)

DECOY_UNMAPPED = (
    WORK
    / "SRR10267761.after_decoy_filter.fastq.gz"
)

PAO1_SAM = (
    WORK
    / "SRR10267761.PAO1.sam"
)

OUT = (
    ROOT
    / "results/phase4/audit_04f_PRJNA576508_EM36_published_pipeline_replication"
)

PILOT = (
    ROOT
    / "results/phase4/audit_04c_PRJNA576508_reference_and_toolchain_freeze"
    / "FROZEN_Atlanta_technical_pilot.csv"
)

TRIM_SUMMARY = (
    ROOT
    / "results/phase4/audit_04e_PRJNA576508_EM36_TruSeq_trim_freeze"
    / "EM36_frozen_TruSeq_trimming_summary.csv"
)

GFF_URL = (
    "https://ftp.ncbi.nlm.nih.gov/genomes/all/"
    "GCF/000/006/765/GCF_000006765.1_ASM676v1/"
    "GCF_000006765.1_ASM676v1_genomic.gff.gz"
)

PUBLISHED_PAO1_GENE_READS = 228345
PUBLISHED_GENES_DETECTED = 5075

MAX_ASSIGNED_READ_RELATIVE_ERROR = 0.25
MAX_DETECTED_GENE_RELATIVE_ERROR = 0.05


def md5_file(path):
    digest = hashlib.md5()

    with path.open(
        "rb"
    ) as handle:
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
            str(
                value
            )
            for value in command
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
            stdout=(
                stdout_handle
                if stdout_handle is not None
                else None
            ),
            stderr=(
                stderr_handle
                if stderr_handle is not None
                else None
            ),
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
            + str(
                result.returncode
            )
        )


def download_file(
    url,
    destination,
):
    if destination.exists():
        print()
        print(
            "Existing file:",
            destination,
        )
        return

    print()
    print(
        "Downloading:",
        url,
    )

    request = Request(
        url,
        headers={
            "User-Agent":
                "CF-State exact annotation retrieval"
        },
    )

    with urlopen(
        request,
        timeout=120,
    ) as response:
        data = response.read()

    destination.write_bytes(
        data
    )

    print(
        "  bytes:",
        destination.stat().st_size,
    )


def ensure_gff():
    download_file(
        GFF_URL,
        PAO1_GFF_GZ,
    )

    if PAO1_GFF.exists():
        print()
        print(
            "Existing uncompressed PAO1 GFF:",
            PAO1_GFF,
        )
        return

    print()
    print(
        "Decompressing PAO1 GFF"
    )

    with gzip.open(
        PAO1_GFF_GZ,
        "rb",
    ) as source:
        with PAO1_GFF.open(
            "wb"
        ) as target:
            shutil.copyfileobj(
                source,
                target,
            )


def parse_attributes(text):
    values = {}

    for item in str(
        text
    ).split(
        ";"
    ):
        item = item.strip()

        if not item:
            continue

        if "=" not in item:
            continue

        key, value = item.split(
            "=",
            1,
        )

        values[
            key
        ] = value

    return values


def build_saf():
    print()
    print(
        "Constructing Cornforth-style PAO1 SAF"
    )

    rows = []

    with PAO1_GFF.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.startswith(
                "#"
            ):
                continue

            fields = line.rstrip(
                "\n"
            ).split(
                "\t"
            )

            if len(
                fields
            ) != 9:
                continue

            seqid = fields[
                0
            ]

            feature_type = fields[
                2
            ]

            start = fields[
                3
            ]

            end = fields[
                4
            ]

            strand = fields[
                6
            ]

            attributes = parse_attributes(
                fields[
                    8
                ]
            )

            if feature_type != "gene":
                continue

            locus_tag = attributes.get(
                "locus_tag",
                "",
            ).strip()

            if not locus_tag:
                continue

            # Exact published rule:
            # all genes WITHOUT decimal points in locus tag.
            if "." in locus_tag:
                continue

            rows.append(
                {
                    "GeneID":
                        locus_tag,

                    "Chr":
                        seqid,

                    "Start":
                        int(
                            start
                        ),

                    "End":
                        int(
                            end
                        ),

                    "Strand":
                        strand,
                }
            )

    saf = pd.DataFrame(
        rows
    )

    if saf.empty:
        raise RuntimeError(
            "No PAO1 SAF rows recovered from GFF."
        )

    duplicate_ids = saf[
        "GeneID"
    ].duplicated(
        keep=False
    )

    if duplicate_ids.any():
        duplicated = sorted(
            saf.loc[
                duplicate_ids,
                "GeneID",
            ].unique()
        )

        raise RuntimeError(
            "Duplicate GeneIDs in SAF: "
            + ", ".join(
                duplicated[
                    :30
                ]
            )
        )

    saf.to_csv(
        PAO1_SAF,
        sep="\t",
        index=False,
    )

    print(
        "  SAF genes:",
        len(
            saf
        ),
    )
    print(
        "  chromosome IDs:",
        sorted(
            saf[
                "Chr"
            ].unique()
        ),
    )
    print(
        "  locus tags containing decimal:",
        int(
            saf[
                "GeneID"
            ]
            .astype(str)
            .str.contains(
                r"\.",
                regex=True,
            )
            .sum()
        ),
    )

    return saf


def index_exists(prefix):
    prefix = str(
        prefix
    )

    small = [
        prefix
        + ".1.bt2",
        prefix
        + ".2.bt2",
        prefix
        + ".3.bt2",
        prefix
        + ".4.bt2",
        prefix
        + ".rev.1.bt2",
        prefix
        + ".rev.2.bt2",
    ]

    large = [
        prefix
        + ".1.bt2l",
        prefix
        + ".2.bt2l",
        prefix
        + ".3.bt2l",
        prefix
        + ".4.bt2l",
        prefix
        + ".rev.1.bt2l",
        prefix
        + ".rev.2.bt2l",
    ]

    return (
        all(
            Path(
                item
            ).exists()
            for item in small
        )
        or all(
            Path(
                item
            ).exists()
            for item in large
        )
    )


def ensure_bowtie_index(
    fasta,
    prefix,
):
    prefix.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if index_exists(
        prefix
    ):
        print()
        print(
            "Existing Bowtie2 index:",
            prefix,
        )
        return

    command = [
        shutil.which(
            "bowtie2-build"
        ),
        "--threads",
        str(
            THREADS
        ),
        str(
            fasta
        ),
        str(
            prefix
        ),
    ]

    run_command(
        command
    )


def parse_bowtie2_summary(path):
    text = path.read_text(
        encoding="utf-8"
    )

    total_match = re.search(
        r"^\s*([\d,]+)\s+reads;",
        text,
        flags=re.MULTILINE,
    )

    zero_match = re.search(
        r"^\s*([\d,]+)\s+\([\d.]+%\)\s+aligned 0 times",
        text,
        flags=re.MULTILINE,
    )

    once_match = re.search(
        r"^\s*([\d,]+)\s+\([\d.]+%\)\s+aligned exactly 1 time",
        text,
        flags=re.MULTILINE,
    )

    multi_match = re.search(
        r"^\s*([\d,]+)\s+\([\d.]+%\)\s+aligned >1 times",
        text,
        flags=re.MULTILINE,
    )

    if (
        total_match is None
        or zero_match is None
        or once_match is None
        or multi_match is None
    ):
        raise RuntimeError(
            "Could not parse Bowtie2 summary:\n"
            + text[
                -3000:
            ]
        )

    def integer(match):
        return int(
            match.group(
                1
            ).replace(
                ",",
                "",
            )
        )

    total = integer(
        total_match
    )

    zero = integer(
        zero_match
    )

    once = integer(
        once_match
    )

    multi = integer(
        multi_match
    )

    mapped = (
        once
        + multi
    )

    return {
        "total_reads":
            total,

        "aligned_zero":
            zero,

        "aligned_once":
            once,

        "aligned_multiple":
            multi,

        "aligned_at_least_once":
            mapped,

        "alignment_fraction":
            mapped
            / total,
    }


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
            "FASTQ line count is not divisible by four."
        )

    return (
        lines
        // 4
    )


def run_featurecounts():
    rscript = shutil.which(
        "Rscript"
    )

    if rscript is None:
        raise RuntimeError(
            "Rscript unavailable."
        )

    script = (
        OUT
        / "run_Rsubread_featureCounts.R"
    )

    count_csv = (
        OUT
        / "EM36_Rsubread_gene_counts.csv"
    )

    stat_csv = (
        OUT
        / "EM36_Rsubread_assignment_stat.csv"
    )

    r_code = (
        'suppressPackageStartupMessages(library(Rsubread))\n'
        + 'sam <- "'
        + str(
            PAO1_SAM
        )
        + '"\n'
        + 'saf <- "'
        + str(
            PAO1_SAF
        )
        + '"\n'
        + 'count_out <- "'
        + str(
            count_csv
        )
        + '"\n'
        + 'stat_out <- "'
        + str(
            stat_csv
        )
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
        + 'write.csv(counts, count_out, row.names=FALSE, quote=FALSE)\n'
        + 'stats <- data.frame(\n'
        + '  Status=rownames(fc$stat),\n'
        + '  count=as.numeric(fc$stat[,1]),\n'
        + '  stringsAsFactors=FALSE\n'
        + ')\n'
        + 'write.csv(stats, stat_out, row.names=FALSE, quote=FALSE)\n'
    )

    script.write_text(
        r_code,
        encoding="utf-8",
    )

    run_command(
        [
            rscript,
            str(
                script
            ),
        ]
    )

    return (
        count_csv,
        stat_csv,
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04f")
    print("EM36 published decoy -> PAO1 -> Rsubread pipeline replication")
    print("=" * 78)

    for path in [
        TRIMMED_FASTQ,
        DECOY_FASTA,
        PAO1_FASTA,
        PILOT,
        TRIM_SUMMARY,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    bowtie2 = shutil.which(
        "bowtie2"
    )

    bowtie2_build = shutil.which(
        "bowtie2-build"
    )

    if bowtie2 is None:
        raise RuntimeError(
            "bowtie2 unavailable."
        )

    if bowtie2_build is None:
        raise RuntimeError(
            "bowtie2-build unavailable."
        )

    WORK.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    pilot = pd.read_csv(
        PILOT
    )

    if len(
        pilot
    ) != 1:
        raise RuntimeError(
            "Expected exactly one frozen pilot."
        )

    pilot_row = pilot.iloc[
        0
    ]

    if str(
        pilot_row[
            "pilot_label"
        ]
    ) != "EM36":
        raise RuntimeError(
            "Frozen pilot is not EM36."
        )

    trim = pd.read_csv(
        TRIM_SUMMARY
    )

    if len(
        trim
    ) != 1:
        raise RuntimeError(
            "Expected exactly one trimming summary row."
        )

    trimmed_input_reads = int(
        trim.iloc[
            0
        ][
            "retained_reads"
        ]
    )

    print()
    print("Frozen technical benchmark")
    print(
        "  pilot: EM36"
    )
    print(
        "  trimmed reads entering decoy stage:",
        trimmed_input_reads,
    )
    print(
        "  published reads mapping to PAO1 genes:",
        PUBLISHED_PAO1_GENE_READS,
    )
    print(
        "  published PAO1 genes with >=1 read:",
        PUBLISHED_GENES_DETECTED,
    )
    print(
        "  pre-frozen assigned-read tolerance:",
        "+/-",
        int(
            100
            * MAX_ASSIGNED_READ_RELATIVE_ERROR
        ),
        "%",
    )
    print(
        "  pre-frozen detected-gene tolerance:",
        "+/-",
        int(
            100
            * MAX_DETECTED_GENE_RELATIVE_ERROR
        ),
        "%",
    )

    # ----------------------------------------------------------
    # 1. Exact PAO1 annotation + SAF
    # ----------------------------------------------------------

    ensure_gff()

    saf = build_saf()

    print()
    print("PAO1 annotation provenance")
    print(
        "  GFF:",
        PAO1_GFF,
    )
    print(
        "  GFF MD5:",
        md5_file(
            PAO1_GFF
        ),
    )
    print(
        "  SAF:",
        PAO1_SAF,
    )
    print(
        "  SAF genes:",
        len(
            saf
        ),
    )

    # ----------------------------------------------------------
    # 2. Build exact Bowtie2 indices
    # ----------------------------------------------------------

    ensure_bowtie_index(
        DECOY_FASTA,
        DECOY_INDEX_PREFIX,
    )

    ensure_bowtie_index(
        PAO1_FASTA,
        PAO1_INDEX_PREFIX,
    )

    # ----------------------------------------------------------
    # 3. Decoy alignment
    #
    # Published method:
    # Bowtie2 default end-to-end against non-Pseudomonas
    # metagenome, then remove mapped reads with Seqtk.
    #
    # Bowtie2 --un-gz emits exactly the reads with zero reported
    # decoy alignments. This is the same retained read set while
    # avoiding a multi-GB intermediate SAM and name inversion.
    # No alignment sensitivity options are changed.
    # ----------------------------------------------------------

    decoy_log = (
        OUT
        / "EM36_decoy_Bowtie2.log"
    )

    if DECOY_UNMAPPED.exists():
        DECOY_UNMAPPED.unlink()

    print()
    print("Decoy alignment")

    run_command(
        [
            bowtie2,
            "-x",
            str(
                DECOY_INDEX_PREFIX
            ),
            "-U",
            str(
                TRIMMED_FASTQ
            ),
            "-p",
            str(
                THREADS
            ),
            "--un-gz",
            str(
                DECOY_UNMAPPED
            ),
            "-S",
            "/dev/null",
        ],
        stderr_path=decoy_log,
    )

    decoy_stats = parse_bowtie2_summary(
        decoy_log
    )

    after_decoy_reads = count_fastq_reads(
        DECOY_UNMAPPED
    )

    print()
    print("Decoy-filter summary")
    print(
        "  trimmed reads:",
        decoy_stats[
            "total_reads"
        ],
    )
    print(
        "  mapped to >=1 decoy:",
        decoy_stats[
            "aligned_at_least_once"
        ],
    )
    print(
        "  decoy-mapped fraction:",
        round(
            decoy_stats[
                "alignment_fraction"
            ],
            6,
        ),
    )
    print(
        "  reads retained after decoy filtering:",
        after_decoy_reads,
    )
    print(
        "  retained fraction:",
        round(
            after_decoy_reads
            / decoy_stats[
                "total_reads"
            ],
            6,
        ),
    )

    if (
        after_decoy_reads
        != decoy_stats[
            "aligned_zero"
        ]
    ):
        raise RuntimeError(
            "Bowtie2 --un-gz count does not equal aligned-zero count."
        )

    # ----------------------------------------------------------
    # 4. PAO1 alignment
    # ----------------------------------------------------------

    pao1_log = (
        OUT
        / "EM36_PAO1_Bowtie2.log"
    )

    if PAO1_SAM.exists():
        PAO1_SAM.unlink()

    print()
    print("PAO1 alignment")

    run_command(
        [
            bowtie2,
            "-x",
            str(
                PAO1_INDEX_PREFIX
            ),
            "-U",
            str(
                DECOY_UNMAPPED
            ),
            "-p",
            str(
                THREADS
            ),
            "-S",
            str(
                PAO1_SAM
            ),
        ],
        stderr_path=pao1_log,
    )

    pao1_stats = parse_bowtie2_summary(
        pao1_log
    )

    print()
    print("PAO1 Bowtie2 summary")
    print(
        "  reads entering PAO1 alignment:",
        pao1_stats[
            "total_reads"
        ],
    )
    print(
        "  reads aligned >=1 time:",
        pao1_stats[
            "aligned_at_least_once"
        ],
    )
    print(
        "  PAO1 alignment fraction:",
        round(
            pao1_stats[
                "alignment_fraction"
            ],
            6,
        ),
    )

    # ----------------------------------------------------------
    # 5. Published Rsubread counting
    # ----------------------------------------------------------

    print()
    print(
        "Running unstranded Rsubread featureCounts with frozen SAF"
    )

    count_csv, stat_csv = run_featurecounts()

    counts = pd.read_csv(
        count_csv
    )

    stats = pd.read_csv(
        stat_csv
    )

    assigned_rows = stats.loc[
        stats[
            "Status"
        ]
        .astype(str)
        .eq(
            "Assigned"
        )
    ]

    if len(
        assigned_rows
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify Rsubread Assigned count."
        )

    featurecounts_assigned = int(
        assigned_rows.iloc[
            0
        ][
            "count"
        ]
    )

    sum_gene_counts = int(
        counts[
            "count"
        ].sum()
    )

    detected_genes = int(
        (
            counts[
                "count"
            ]
            > 0
        ).sum()
    )

    if (
        featurecounts_assigned
        != sum_gene_counts
    ):
        raise RuntimeError(
            "featureCounts Assigned count != sum of gene counts."
        )

    # ----------------------------------------------------------
    # 6. Pre-frozen benchmark comparison
    # ----------------------------------------------------------

    assigned_relative_error = abs(
        featurecounts_assigned
        - PUBLISHED_PAO1_GENE_READS
    ) / PUBLISHED_PAO1_GENE_READS

    detected_relative_error = abs(
        detected_genes
        - PUBLISHED_GENES_DETECTED
    ) / PUBLISHED_GENES_DETECTED

    assigned_ratio = (
        featurecounts_assigned
        / PUBLISHED_PAO1_GENE_READS
    )

    detected_ratio = (
        detected_genes
        / PUBLISHED_GENES_DETECTED
    )

    print()
    print("Published EM36 benchmark comparison")
    print()
    print("  PAO1-gene reads")
    print(
        "    published:",
        PUBLISHED_PAO1_GENE_READS,
    )
    print(
        "    reconstructed Rsubread Assigned:",
        featurecounts_assigned,
    )
    print(
        "    reconstructed / published:",
        round(
            assigned_ratio,
            4,
        ),
    )
    print(
        "    relative error:",
        round(
            assigned_relative_error,
            4,
        ),
    )

    print()
    print("  detected PAO1 genes")
    print(
        "    published:",
        PUBLISHED_GENES_DETECTED,
    )
    print(
        "    reconstructed:",
        detected_genes,
    )
    print(
        "    reconstructed / published:",
        round(
            detected_ratio,
            4,
        ),
    )
    print(
        "    relative error:",
        round(
            detected_relative_error,
            4,
        ),
    )

    print()
    print(
        "  Bowtie2 PAO1-aligned reads:",
        pao1_stats[
            "aligned_at_least_once"
        ],
    )

    # ----------------------------------------------------------
    # 7. Save compact technical summary
    # ----------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "pilot":
                    "EM36",

                "trimmed_reads":
                    trimmed_input_reads,

                "decoy_mapped_reads":
                    decoy_stats[
                        "aligned_at_least_once"
                    ],

                "decoy_mapped_fraction":
                    decoy_stats[
                        "alignment_fraction"
                    ],

                "reads_after_decoy_filter":
                    after_decoy_reads,

                "PAO1_Bowtie2_mapped_reads":
                    pao1_stats[
                        "aligned_at_least_once"
                    ],

                "PAO1_Bowtie2_mapping_fraction":
                    pao1_stats[
                        "alignment_fraction"
                    ],

                "Rsubread_assigned_gene_reads":
                    featurecounts_assigned,

                "PAO1_genes_in_SAF":
                    len(
                        saf
                    ),

                "PAO1_genes_detected":
                    detected_genes,

                "published_PAO1_gene_reads":
                    PUBLISHED_PAO1_GENE_READS,

                "assigned_read_ratio_to_published":
                    assigned_ratio,

                "assigned_read_relative_error":
                    assigned_relative_error,

                "published_genes_detected":
                    PUBLISHED_GENES_DETECTED,

                "detected_gene_ratio_to_published":
                    detected_ratio,

                "detected_gene_relative_error":
                    detected_relative_error,
            }
        ]
    )

    summary.to_csv(
        OUT
        / "EM36_published_pipeline_replication_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 8. Formal audit checks
    #
    # Thresholds were fixed BEFORE observing this reconstruction:
    #
    # assigned gene reads: within +/-25% of publication
    # detected genes:      within +/-5% of publication
    #
    # If either fails, we diagnose annotation/tool/version
    # differences. We do NOT relax thresholds.
    # ----------------------------------------------------------

    checks = {
        "trimmed_input_matches_Audit04e":
            decoy_stats[
                "total_reads"
            ]
            == trimmed_input_reads,

        "decoy_filter_output_matches_Bowtie2_zero_alignment":
            after_decoy_reads
            == decoy_stats[
                "aligned_zero"
            ],

        "decoy_filter_retains_reads":
            after_decoy_reads
            > 0,

        "PAO1_alignment_input_matches_decoy_output":
            pao1_stats[
                "total_reads"
            ]
            == after_decoy_reads,

        "PAO1_alignment_nonzero":
            pao1_stats[
                "aligned_at_least_once"
            ]
            > 0,

        "SAF_has_more_than_5500_genes":
            len(
                saf
            )
            > 5500,

        "Rsubread_assigned_equals_sum_counts":
            featurecounts_assigned
            == sum_gene_counts,

        "Rsubread_assigned_nonzero":
            featurecounts_assigned
            > 0,

        "detected_genes_nonzero":
            detected_genes
            > 0,

        "assigned_reads_within_prefrozen_25pct":
            assigned_relative_error
            <= MAX_ASSIGNED_READ_RELATIVE_ERROR,

        "detected_genes_within_prefrozen_5pct":
            detected_relative_error
            <= MAX_DETECTED_GENE_RELATIVE_ERROR,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04f checks")

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
        "PRJNA576508_EM36_PUBLISHED_PIPELINE_TECHNICALLY_REPLICATED"
        if passed
        else "PRJNA576508_EM36_PUBLISHED_PIPELINE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: these technical acceptance thresholds were "
        "fixed before observing the reconstructed EM36 benchmark. "
        "They must not be relaxed if this audit fails."
    )

    print()
    print(
        "No CF-State/iModulon activity projection has been performed."
    )

    # ----------------------------------------------------------
    # 9. Save audit checks + provenance
    # ----------------------------------------------------------

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

    pd.DataFrame(
        [
            {
                "file":
                    "PAO1_GFF",

                "path":
                    str(
                        PAO1_GFF
                    ),

                "md5":
                    md5_file(
                        PAO1_GFF
                    ),
            },

            {
                "file":
                    "PAO1_SAF",

                "path":
                    str(
                        PAO1_SAF
                    ),

                "md5":
                    md5_file(
                        PAO1_SAF
                    ),
            },

            {
                "file":
                    "decoy_filtered_FASTQ",

                "path":
                    str(
                        DECOY_UNMAPPED
                    ),

                "md5":
                    md5_file(
                        DECOY_UNMAPPED
                    ),
            },
        ]
    ).to_csv(
        OUT
        / "technical_provenance.csv",
        index=False,
    )

    (
        OUT
        / "AUDIT_04F_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    # PAO1 SAM is large and is reproducible from frozen inputs.
    # Keep it if audit fails for diagnosis; delete it after a
    # successful audit to avoid unnecessary project storage.
    if passed:
        print()
        print(
            "Technical pilot passed; removing reproducible large "
            "PAO1 SAM intermediate."
        )

        PAO1_SAM.unlink(
            missing_ok=True
        )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed Audit 04f checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
