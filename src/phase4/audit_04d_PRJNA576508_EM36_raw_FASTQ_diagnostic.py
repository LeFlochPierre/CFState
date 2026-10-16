from pathlib import Path
from collections import Counter
import gzip
import hashlib
import math
import shutil
import subprocess

import pandas as pd


ROOT = Path.cwd()

PILOT = (
    ROOT
    / "results/phase4/audit_04c_PRJNA576508_reference_and_toolchain_freeze"
    / "FROZEN_Atlanta_technical_pilot.csv"
)

RAW_DIR = (
    ROOT
    / "data/raw/phase4/PRJNA576508/pilot_EM36"
)

OUT = (
    ROOT
    / "results/phase4/audit_04d_PRJNA576508_EM36_raw_FASTQ_diagnostic"
)

FASTQ = (
    RAW_DIR
    / "SRR10267761.fastq.gz"
)

SAMPLE_STRIDE = 100

EXPECTED_LABEL = "EM36"
EXPECTED_RUN = "SRR10267761"
EXPECTED_READS = 57609988

# These are NOT assumed to be the true adapters.
# They are diagnostic motifs only.
#
# Exact adapter identity was not reported in the publication.
DIAGNOSTIC_MOTIFS = {
    "TruSeq_R1_prefix12":
        "AGATCGGAAGAG",

    "TruSeq_R1_long":
        "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA",

    "Nextera_prefix12":
        "CTGTCTCTTATA",

    "Nextera_long":
        "CTGTCTCTTATACACATCT",
}


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


def normalize_download_url(value):
    text = str(
        value
    ).strip()

    if ";" in text:
        text = text.split(
            ";"
        )[
            0
        ].strip()

    if text.startswith(
        "ftp://"
    ):
        text = (
            "https://"
            + text[
                len(
                    "ftp://"
                ):
            ]
        )

    elif text.startswith(
        "ftp."
    ):
        text = (
            "https://"
            + text
        )

    return text


def normalize_md5(value):
    text = str(
        value
    ).strip()

    if ";" in text:
        text = text.split(
            ";"
        )[
            0
        ].strip()

    return text.lower()


def download_fastq(
    url,
    expected_md5,
):
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if FASTQ.exists():
        observed = md5_file(
            FASTQ
        )

        if observed == expected_md5:
            print()
            print(
                "Existing FASTQ MD5 is correct:",
                FASTQ,
            )

            return

        print()
        print(
            "Existing FASTQ MD5 mismatch; removing and downloading again."
        )

        FASTQ.unlink()

    curl = shutil.which(
        "curl"
    )

    if curl is None:
        raise RuntimeError(
            "curl is not available."
        )

    temporary = Path(
        str(
            FASTQ
        )
        + ".part"
    )

    temporary.unlink(
        missing_ok=True
    )

    print()
    print("Downloading frozen EM36 FASTQ")
    print(
        " ",
        url,
    )

    command = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "5",
        "--retry-delay",
        "3",
        "--continue-at",
        "-",
        "--output",
        str(
            temporary
        ),
        url,
    ]

    result = subprocess.run(
        command
    )

    if result.returncode != 0:
        raise RuntimeError(
            "FASTQ download failed."
        )

    observed = md5_file(
        temporary
    )

    print()
    print(
        "Downloaded MD5:",
        observed,
    )
    print(
        "Expected MD5:",
        expected_md5,
    )

    if observed != expected_md5:
        raise RuntimeError(
            "Downloaded FASTQ MD5 does not match ENA."
        )

    temporary.rename(
        FASTQ
    )


def phred_mean(values):
    if not values:
        return float(
            "nan"
        )

    return float(
        sum(
            values
        )
        / len(
            values
        )
    )


def shannon_entropy(counter):
    total = sum(
        counter.values()
    )

    if total <= 0:
        return float(
            "nan"
        )

    entropy = 0.0

    for count in counter.values():
        probability = (
            count
            / total
        )

        entropy -= (
            probability
            * math.log2(
                probability
            )
        )

    return entropy


def audit_fastq():
    read_count = 0
    malformed_records = 0
    sequence_quality_mismatches = 0

    length_counts = Counter()

    sampled_reads = 0

    suffix_counts = {
        8:
            Counter(),

        10:
            Counter(),

        12:
            Counter(),
    }

    # Position relative to 3' end:
    # -25 ... -1
    tail_base_counts = {
        position:
            Counter()
        for position in range(
            -25,
            0,
        )
    }

    motif_counts_anywhere = Counter()
    motif_counts_last30 = Counter()

    first10_quality_values = []
    last10_quality_values = []

    n_bases_sampled = 0
    total_bases_sampled = 0

    with gzip.open(
        FASTQ,
        "rt",
        encoding="utf-8",
        errors="strict",
        newline="",
    ) as handle:

        while True:
            header = handle.readline()

            if not header:
                break

            sequence = handle.readline()
            plus = handle.readline()
            quality = handle.readline()

            if (
                not sequence
                or not plus
                or not quality
            ):
                malformed_records += 1
                break

            read_count += 1

            header = header.rstrip(
                "\r\n"
            )

            sequence = sequence.rstrip(
                "\r\n"
            ).upper()

            plus = plus.rstrip(
                "\r\n"
            )

            quality = quality.rstrip(
                "\r\n"
            )

            if (
                not header.startswith(
                    "@"
                )
                or not plus.startswith(
                    "+"
                )
            ):
                malformed_records += 1

            if len(
                sequence
            ) != len(
                quality
            ):
                sequence_quality_mismatches += 1

            length_counts[
                len(
                    sequence
                )
            ] += 1

            # Evenly sample throughout the complete FASTQ.
            if (
                read_count
                % SAMPLE_STRIDE
                != 0
            ):
                continue

            sampled_reads += 1

            sequence_length = len(
                sequence
            )

            for k in suffix_counts:
                if sequence_length >= k:
                    suffix_counts[
                        k
                    ][
                        sequence[
                            -k:
                        ]
                    ] += 1

            for position in range(
                -25,
                0,
            ):
                if sequence_length >= abs(
                    position
                ):
                    base = sequence[
                        position
                    ]

                    tail_base_counts[
                        position
                    ][
                        base
                    ] += 1

            tail30 = sequence[
                -30:
            ]

            for motif_name, motif in DIAGNOSTIC_MOTIFS.items():
                if motif in sequence:
                    motif_counts_anywhere[
                        motif_name
                    ] += 1

                if motif in tail30:
                    motif_counts_last30[
                        motif_name
                    ] += 1

            quality_scores = [
                ord(
                    character
                )
                - 33
                for character in quality
            ]

            first10_quality_values.extend(
                quality_scores[
                    :10
                ]
            )

            last10_quality_values.extend(
                quality_scores[
                    -10:
                ]
            )

            total_bases_sampled += sequence_length

            n_bases_sampled += sequence.count(
                "N"
            )

    return {
        "read_count":
            read_count,

        "malformed_records":
            malformed_records,

        "sequence_quality_mismatches":
            sequence_quality_mismatches,

        "length_counts":
            length_counts,

        "sampled_reads":
            sampled_reads,

        "suffix_counts":
            suffix_counts,

        "tail_base_counts":
            tail_base_counts,

        "motif_counts_anywhere":
            motif_counts_anywhere,

        "motif_counts_last30":
            motif_counts_last30,

        "mean_first10_quality":
            phred_mean(
                first10_quality_values
            ),

        "mean_last10_quality":
            phred_mean(
                last10_quality_values
            ),

        "sampled_N_fraction":
            (
                n_bases_sampled
                / total_bases_sampled
                if total_bases_sampled
                > 0
                else float(
                    "nan"
                )
            ),
    }


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04d")
    print("Frozen EM36 raw FASTQ intake and adapter diagnostic")
    print("=" * 78)

    if not PILOT.exists():
        raise FileNotFoundError(
            str(
                PILOT
            )
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

    row = pilot.iloc[
        0
    ]

    pilot_label = str(
        row[
            "pilot_label"
        ]
    )

    run_accession = str(
        row[
            "run_accession"
        ]
    )

    expected_reads = int(
        row[
            "pilot_total_reads"
        ]
    )

    expected_md5 = normalize_md5(
        row[
            "fastq_md5"
        ]
    )

    download_url = normalize_download_url(
        row[
            "fastq_ftp"
        ]
    )

    print()
    print("Frozen pilot")
    print(
        "  label:",
        pilot_label,
    )
    print(
        "  run:",
        run_accession,
    )
    print(
        "  expected reads:",
        expected_reads,
    )
    print(
        "  expected compressed FASTQ MD5:",
        expected_md5,
    )
    print(
        "  source:",
        download_url,
    )

    # ----------------------------------------------------------
    # 1. Download + compressed-file integrity
    # ----------------------------------------------------------

    download_fastq(
        download_url,
        expected_md5,
    )

    observed_md5 = md5_file(
        FASTQ
    )

    print()
    print("Frozen FASTQ")
    print(
        "  path:",
        FASTQ,
    )
    print(
        "  compressed bytes:",
        FASTQ.stat().st_size,
    )
    print(
        "  observed MD5:",
        observed_md5,
    )

    # ----------------------------------------------------------
    # 2. Stream complete FASTQ once
    # ----------------------------------------------------------

    print()
    print(
        "Streaming complete FASTQ for structural QC and "
        "deterministic 1-in-"
        + str(
            SAMPLE_STRIDE
        )
        + " diagnostic sampling"
    )

    audit = audit_fastq()

    read_count = audit[
        "read_count"
    ]

    length_counts = audit[
        "length_counts"
    ]

    sampled_reads = audit[
        "sampled_reads"
    ]

    mode_length = None

    if length_counts:
        mode_length = length_counts.most_common(
            1
        )[
            0
        ][
            0
        ]

    minimum_length = min(
        length_counts
    )

    maximum_length = max(
        length_counts
    )

    reads_below_25 = sum(
        count
        for length, count
        in length_counts.items()
        if length
        < 25
    )

    print()
    print("Raw FASTQ structural QC")
    print(
        "  reads:",
        read_count,
    )
    print(
        "  expected reads:",
        expected_reads,
    )
    print(
        "  malformed records:",
        audit[
            "malformed_records"
        ],
    )
    print(
        "  sequence/quality length mismatches:",
        audit[
            "sequence_quality_mismatches"
        ],
    )
    print(
        "  minimum read length:",
        minimum_length,
    )
    print(
        "  modal read length:",
        mode_length,
    )
    print(
        "  maximum read length:",
        maximum_length,
    )
    print(
        "  reads shorter than 25 bp:",
        reads_below_25,
    )
    print(
        "  deterministic diagnostic reads:",
        sampled_reads,
    )
    print(
        "  sampled N fraction:",
        round(
            audit[
                "sampled_N_fraction"
            ],
            8,
        ),
    )
    print(
        "  mean Phred first 10 bases:",
        round(
            audit[
                "mean_first10_quality"
            ],
            3,
        ),
    )
    print(
        "  mean Phred last 10 bases:",
        round(
            audit[
                "mean_last10_quality"
            ],
            3,
        ),
    )

    # ----------------------------------------------------------
    # 3. Read-length distribution
    # ----------------------------------------------------------

    length_table = pd.DataFrame(
        [
            {
                "read_length":
                    length,

                "read_count":
                    count,

                "fraction":
                    (
                        count
                        / read_count
                    ),
            }
            for length, count
            in sorted(
                length_counts.items()
            )
        ]
    )

    print()
    print("Read-length distribution")
    print(
        length_table
        .round(
            {
                "fraction":
                    8
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. De novo exact terminal k-mer diagnostic
    #
    # No adapter sequence is assumed here.
    # ----------------------------------------------------------

    suffix_rows = []

    for k in [
        8,
        10,
        12,
    ]:
        counter = audit[
            "suffix_counts"
        ][
            k
        ]

        total = sum(
            counter.values()
        )

        entropy = shannon_entropy(
            counter
        )

        for rank, (
            sequence,
            count,
        ) in enumerate(
            counter.most_common(
                20
            ),
            start=1,
        ):
            suffix_rows.append(
                {
                    "k":
                        k,

                    "rank":
                        rank,

                    "suffix":
                        sequence,

                    "count":
                        count,

                    "fraction_of_sampled_reads":
                        (
                            count
                            / total
                            if total
                            > 0
                            else float(
                                "nan"
                            )
                        ),

                    "suffix_distribution_entropy_bits":
                        entropy,
                }
            )

    suffix_table = pd.DataFrame(
        suffix_rows
    )

    print()
    print("Most common exact 3-prime suffixes")
    print(
        suffix_table
        .loc[
            suffix_table[
                "rank"
            ]
            <= 10
        ]
        .round(
            {
                "fraction_of_sampled_reads":
                    8,

                "suffix_distribution_entropy_bits":
                    4,
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. 3' positional consensus
    # ----------------------------------------------------------

    tail_rows = []

    for position in range(
        -25,
        0,
    ):
        counter = audit[
            "tail_base_counts"
        ][
            position
        ]

        total = sum(
            counter.values()
        )

        if total == 0:
            continue

        most_common_base, most_common_count = counter.most_common(
            1
        )[
            0
        ]

        tail_rows.append(
            {
                "position_from_3prime_end":
                    position,

                "A_fraction":
                    counter[
                        "A"
                    ]
                    / total,

                "C_fraction":
                    counter[
                        "C"
                    ]
                    / total,

                "G_fraction":
                    counter[
                        "G"
                    ]
                    / total,

                "T_fraction":
                    counter[
                        "T"
                    ]
                    / total,

                "N_fraction":
                    counter[
                        "N"
                    ]
                    / total,

                "consensus_base":
                    most_common_base,

                "consensus_fraction":
                    most_common_count
                    / total,
            }
        )

    tail_table = pd.DataFrame(
        tail_rows
    )

    print()
    print("3-prime positional base composition")
    print(
        tail_table
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Common Illumina motifs — diagnostic ONLY
    #
    # These are not used to trim anything in this audit.
    # ----------------------------------------------------------

    motif_rows = []

    for motif_name, motif in DIAGNOSTIC_MOTIFS.items():
        anywhere = audit[
            "motif_counts_anywhere"
        ][
            motif_name
        ]

        last30 = audit[
            "motif_counts_last30"
        ][
            motif_name
        ]

        motif_rows.append(
            {
                "diagnostic_motif":
                    motif_name,

                "sequence":
                    motif,

                "sampled_reads":
                    sampled_reads,

                "reads_with_motif_anywhere":
                    anywhere,

                "fraction_anywhere":
                    (
                        anywhere
                        / sampled_reads
                        if sampled_reads
                        > 0
                        else float(
                            "nan"
                        )
                    ),

                "reads_with_motif_in_last30":
                    last30,

                "fraction_last30":
                    (
                        last30
                        / sampled_reads
                        if sampled_reads
                        > 0
                        else float(
                            "nan"
                        )
                    ),
            }
        )

    motif_table = pd.DataFrame(
        motif_rows
    )

    print()
    print("Common Illumina motif diagnostic")
    print(
        motif_table
        .round(
            {
                "fraction_anywhere":
                    8,

                "fraction_last30":
                    8,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT: the motifs above are diagnostic probes only. "
        "They have not been assumed to be the Cornforth adapter "
        "and no read has been trimmed."
    )

    # ----------------------------------------------------------
    # 7. Compact adapter-evidence summary
    # ----------------------------------------------------------

    top12 = suffix_table.loc[
        (
            suffix_table[
                "k"
            ]
            == 12
        )
        & (
            suffix_table[
                "rank"
            ]
            == 1
        )
    ]

    if len(
        top12
    ) != 1:
        raise RuntimeError(
            "Could not determine top terminal 12-mer."
        )

    top12_sequence = str(
        top12.iloc[
            0
        ][
            "suffix"
        ]
    )

    top12_fraction = float(
        top12.iloc[
            0
        ][
            "fraction_of_sampled_reads"
        ]
    )

    maximum_tail_consensus = float(
        tail_table[
            "consensus_fraction"
        ].max()
    )

    strongest_diagnostic_motif_fraction = float(
        motif_table[
            "fraction_last30"
        ].max()
    )

    strongest_diagnostic_motif_name = str(
        motif_table.sort_values(
            "fraction_last30",
            ascending=False,
        )
        .iloc[
            0
        ][
            "diagnostic_motif"
        ]
    )

    diagnostic_summary = pd.DataFrame(
        [
            {
                "pilot":
                    pilot_label,

                "sampled_reads":
                    sampled_reads,

                "top_terminal_12mer":
                    top12_sequence,

                "top_terminal_12mer_fraction":
                    top12_fraction,

                "maximum_single_base_fraction_in_last25":
                    maximum_tail_consensus,

                "strongest_common_Illumina_diagnostic":
                    strongest_diagnostic_motif_name,

                "strongest_common_Illumina_last30_fraction":
                    strongest_diagnostic_motif_fraction,

                "adapter_identity_status":
                    (
                        "UNRESOLVED_PENDING_DIAGNOSTIC_INTERPRETATION"
                    ),
            }
        ]
    )

    print()
    print("Adapter-evidence summary")
    print(
        diagnostic_summary
        .round(
            {
                "top_terminal_12mer_fraction":
                    8,

                "maximum_single_base_fraction_in_last25":
                    6,

                "strongest_common_Illumina_last30_fraction":
                    8,
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 8. Formal Audit 04d checks
    #
    # Adapter identity is deliberately NOT a PASS criterion.
    # This audit asks only whether the frozen raw pilot is intact
    # and suitable for the next preprocessing decision.
    # ----------------------------------------------------------

    checks = {
        "pilot_is_frozen_EM36":
            pilot_label
            == EXPECTED_LABEL,

        "run_is_frozen_SRR10267761":
            run_accession
            == EXPECTED_RUN,

        "compressed_FASTQ_MD5_matches_ENA":
            observed_md5
            == expected_md5,

        "FASTQ_read_count_matches_frozen_metadata":
            read_count
            == expected_reads
            == EXPECTED_READS,

        "no_malformed_FASTQ_records":
            audit[
                "malformed_records"
            ]
            == 0,

        "no_sequence_quality_length_mismatches":
            audit[
                "sequence_quality_mismatches"
            ]
            == 0,

        "modal_read_length_is_75bp":
            mode_length
            == 75,

        "diagnostic_sample_ge_100k_reads":
            sampled_reads
            >= 100000,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04d checks")

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
        "PRJNA576508_EM36_RAW_FASTQ_INTAKE_ACCEPTED"
        if passed
        else "PRJNA576508_EM36_RAW_FASTQ_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "No trimming, decoy alignment, PAO1 alignment, gene "
        "counting, or CF-State projection has been performed."
    )

    # ----------------------------------------------------------
    # 9. Save
    # ----------------------------------------------------------

    length_table.to_csv(
        OUT
        / "EM36_raw_read_length_distribution.csv",
        index=False,
    )

    suffix_table.to_csv(
        OUT
        / "EM36_terminal_kmer_diagnostic.csv",
        index=False,
    )

    tail_table.to_csv(
        OUT
        / "EM36_3prime_base_composition.csv",
        index=False,
    )

    motif_table.to_csv(
        OUT
        / "EM36_common_Illumina_motif_diagnostic.csv",
        index=False,
    )

    diagnostic_summary.to_csv(
        OUT
        / "EM36_adapter_evidence_summary.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "metric":
                    "compressed_fastq_md5",

                "value":
                    observed_md5,
            },

            {
                "metric":
                    "raw_read_count",

                "value":
                    read_count,
            },

            {
                "metric":
                    "minimum_read_length",

                "value":
                    minimum_length,
            },

            {
                "metric":
                    "modal_read_length",

                "value":
                    mode_length,
            },

            {
                "metric":
                    "maximum_read_length",

                "value":
                    maximum_length,
            },

            {
                "metric":
                    "reads_below_25bp",

                "value":
                    reads_below_25,
            },

            {
                "metric":
                    "mean_first10_phred",

                "value":
                    audit[
                        "mean_first10_quality"
                    ],
            },

            {
                "metric":
                    "mean_last10_phred",

                "value":
                    audit[
                        "mean_last10_quality"
                    ],
            },
        ]
    ).to_csv(
        OUT
        / "EM36_raw_FASTQ_QC_summary.csv",
        index=False,
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

    (
        OUT
        / "AUDIT_04D_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed Audit 04d checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
