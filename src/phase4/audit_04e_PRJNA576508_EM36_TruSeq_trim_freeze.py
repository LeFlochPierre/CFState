from pathlib import Path
import gzip
import hashlib
import re
import shutil
import subprocess

import pandas as pd


ROOT = Path.cwd()

RAW_FASTQ = (
    ROOT
    / "data/raw/phase4/PRJNA576508/pilot_EM36"
    / "SRR10267761.fastq.gz"
)

OUT_DIR = (
    ROOT
    / "data/processed/phase4/PRJNA576508/pilot_EM36"
)

TRIMMED_FASTQ = (
    OUT_DIR
    / "SRR10267761.TruSeq_trimmed.min25.fastq.gz"
)

AUDIT_OUT = (
    ROOT
    / "results/phase4/audit_04e_PRJNA576508_EM36_TruSeq_trim_freeze"
)

PILOT = (
    ROOT
    / "results/phase4/audit_04c_PRJNA576508_reference_and_toolchain_freeze"
    / "FROZEN_Atlanta_technical_pilot.csv"
)

CUTADAPT_REPORT = (
    AUDIT_OUT
    / "cutadapt_report.txt"
)

COMMAND_FILE = (
    AUDIT_OUT
    / "frozen_cutadapt_command.txt"
)

ADAPTER = (
    "AGATCGGAAGAGCACACGTCTGAACTCCAGTCAC"
)

MINIMUM_LENGTH = 25
ERROR_RATE = 0.1
MINIMUM_OVERLAP = 3


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


def count_fastq(path):
    reads = 0

    minimum_length = None
    maximum_length = None

    total_bases = 0
    reads_below_25 = 0

    length_counts = {}

    with gzip.open(
        path,
        "rt",
        encoding="utf-8",
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
                raise RuntimeError(
                    "Incomplete FASTQ record in "
                    + str(
                        path
                    )
                )

            sequence = sequence.rstrip(
                "\r\n"
            )

            quality = quality.rstrip(
                "\r\n"
            )

            if len(
                sequence
            ) != len(
                quality
            ):
                raise RuntimeError(
                    "Sequence/quality length mismatch."
                )

            length = len(
                sequence
            )

            reads += 1
            total_bases += length

            length_counts[
                length
            ] = (
                length_counts.get(
                    length,
                    0,
                )
                + 1
            )

            if (
                minimum_length is None
                or length
                < minimum_length
            ):
                minimum_length = length

            if (
                maximum_length is None
                or length
                > maximum_length
            ):
                maximum_length = length

            if length < 25:
                reads_below_25 += 1

    modal_length = max(
        length_counts,
        key=length_counts.get,
    )

    mean_length = (
        total_bases
        / reads
        if reads > 0
        else float(
            "nan"
        )
    )

    length_table = pd.DataFrame(
        [
            {
                "read_length":
                    length,

                "read_count":
                    count,

                "fraction":
                    count
                    / reads,
            }
            for length, count
            in sorted(
                length_counts.items()
            )
        ]
    )

    return {
        "reads":
            reads,

        "minimum_length":
            minimum_length,

        "maximum_length":
            maximum_length,

        "modal_length":
            modal_length,

        "mean_length":
            mean_length,

        "reads_below_25":
            reads_below_25,

        "length_table":
            length_table,
    }


def parse_cutadapt_report(text):
    values = {}

    patterns = {
        "total_reads_processed":
            (
                r"Total reads processed:\s*"
                r"([\d,]+)"
            ),

        "reads_with_adapters":
            (
                r"Reads with adapters:\s*"
                r"([\d,]+)"
                r"\s*"
                r"\(([\d.]+)%\)"
            ),

        "reads_too_short":
            (
                r"Reads that were too short:\s*"
                r"([\d,]+)"
                r"\s*"
                r"\(([\d.]+)%\)"
            ),

        "reads_written":
            (
                r"Reads written \(passing filters\):\s*"
                r"([\d,]+)"
                r"\s*"
                r"\(([\d.]+)%\)"
            ),
    }

    for key, pattern in patterns.items():
        match = re.search(
            pattern,
            text,
        )

        if match is None:
            continue

        if key == "total_reads_processed":
            values[
                key
            ] = int(
                match.group(
                    1
                ).replace(
                    ",",
                    "",
                )
            )

        else:
            values[
                key
            ] = int(
                match.group(
                    1
                ).replace(
                    ",",
                    "",
                )
            )

            values[
                key
                + "_percent"
            ] = float(
                match.group(
                    2
                )
            )

    return values


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04e")
    print("Freeze and apply TruSeq trimming rule to frozen EM36 pilot")
    print("=" * 78)

    for path in [
        RAW_FASTQ,
        PILOT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    cutadapt = shutil.which(
        "cutadapt"
    )

    if cutadapt is None:
        raise RuntimeError(
            "cutadapt is not available."
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_OUT.mkdir(
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

    pilot_label = str(
        pilot_row[
            "pilot_label"
        ]
    )

    run_accession = str(
        pilot_row[
            "run_accession"
        ]
    )

    expected_input_reads = int(
        pilot_row[
            "pilot_total_reads"
        ]
    )

    # ----------------------------------------------------------
    # 1. Freeze trimming rule
    #
    # Basis:
    # - Cornforth library = TruSeq stranded total RNA
    # - Cutadapt 1.13 documentation recommends this adapter for
    #   single-end TruSeq reads
    # - Audit 04d2 independently observed the corresponding
    #   TruSeq motif in the majority of EM36 reads
    #
    # We explicitly freeze Cutadapt-1.13 default matching values
    # to avoid depending silently on modern defaults.
    # ----------------------------------------------------------

    command = [
        cutadapt,
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
            TRIMMED_FASTQ
        ),
        str(
            RAW_FASTQ
        ),
    ]

    command_text = " ".join(
        command
    )

    COMMAND_FILE.write_text(
        command_text
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Frozen trimming rule")
    print(
        "  pilot:",
        pilot_label,
    )
    print(
        "  run:",
        run_accession,
    )
    print(
        "  adapter:",
        ADAPTER,
    )
    print(
        "  adapter type:",
        "3-prime (-a)",
    )
    print(
        "  minimum retained length:",
        MINIMUM_LENGTH,
    )
    print(
        "  maximum error rate:",
        ERROR_RATE,
    )
    print(
        "  minimum overlap:",
        MINIMUM_OVERLAP,
    )
    print(
        "  quality trimming:",
        "NONE",
    )

    # ----------------------------------------------------------
    # 2. Run Cutadapt once
    # ----------------------------------------------------------

    if TRIMMED_FASTQ.exists():
        print()
        print(
            "Existing trimmed FASTQ detected; removing so Audit "
            "04e is regenerated from the frozen command."
        )

        TRIMMED_FASTQ.unlink()

    print()
    print("Running frozen Cutadapt rule")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    report_text = (
        result.stdout
        + "\n"
        + result.stderr
    )

    CUTADAPT_REPORT.write_text(
        report_text,
        encoding="utf-8",
    )

    print()
    print(report_text)

    if result.returncode != 0:
        raise RuntimeError(
            "Cutadapt failed."
        )

    if not TRIMMED_FASTQ.exists():
        raise RuntimeError(
            "Cutadapt produced no output FASTQ."
        )

    # ----------------------------------------------------------
    # 3. Parse Cutadapt summary
    # ----------------------------------------------------------

    parsed = parse_cutadapt_report(
        report_text
    )

    required_report_fields = [
        "total_reads_processed",
        "reads_with_adapters",
        "reads_with_adapters_percent",
        "reads_too_short",
        "reads_too_short_percent",
        "reads_written",
        "reads_written_percent",
    ]

    missing_report_fields = [
        field
        for field in required_report_fields
        if field
        not in parsed
    ]

    if missing_report_fields:
        raise RuntimeError(
            "Could not parse Cutadapt fields: "
            + ", ".join(
                missing_report_fields
            )
        )

    # ----------------------------------------------------------
    # 4. Independently stream trimmed FASTQ
    # ----------------------------------------------------------

    print()
    print(
        "Streaming trimmed FASTQ for independent output QC"
    )

    trimmed = count_fastq(
        TRIMMED_FASTQ
    )

    input_reads = parsed[
        "total_reads_processed"
    ]

    output_reads = trimmed[
        "reads"
    ]

    retention_fraction = (
        output_reads
        / input_reads
    )

    removed_fraction = (
        1.0
        - retention_fraction
    )

    adapter_fraction = (
        parsed[
            "reads_with_adapters"
        ]
        / input_reads
    )

    print()
    print("Trimmed FASTQ summary")
    print(
        "  input reads:",
        input_reads,
    )
    print(
        "  reads with adapter:",
        parsed[
            "reads_with_adapters"
        ],
        "(",
        round(
            adapter_fraction,
            6,
        ),
        ")",
    )
    print(
        "  reads removed as <25 bp:",
        parsed[
            "reads_too_short"
        ],
    )
    print(
        "  retained reads:",
        output_reads,
    )
    print(
        "  retained fraction:",
        round(
            retention_fraction,
            6,
        ),
    )
    print(
        "  removed fraction:",
        round(
            removed_fraction,
            6,
        ),
    )
    print(
        "  minimum retained length:",
        trimmed[
            "minimum_length"
        ],
    )
    print(
        "  modal retained length:",
        trimmed[
            "modal_length"
        ],
    )
    print(
        "  maximum retained length:",
        trimmed[
            "maximum_length"
        ],
    )
    print(
        "  mean retained length:",
        round(
            trimmed[
                "mean_length"
            ],
            4,
        ),
    )
    print(
        "  reads below 25 bp:",
        trimmed[
            "reads_below_25"
        ],
    )
    print(
        "  trimmed FASTQ MD5:",
        md5_file(
            TRIMMED_FASTQ
        ),
    )

    # ----------------------------------------------------------
    # 5. Freeze provenance
    # ----------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "pilot":
                    pilot_label,

                "run_accession":
                    run_accession,

                "raw_FASTQ":
                    str(
                        RAW_FASTQ
                    ),

                "raw_FASTQ_MD5":
                    md5_file(
                        RAW_FASTQ
                    ),

                "trimmed_FASTQ":
                    str(
                        TRIMMED_FASTQ
                    ),

                "trimmed_FASTQ_MD5":
                    md5_file(
                        TRIMMED_FASTQ
                    ),

                "adapter":
                    ADAPTER,

                "adapter_type":
                    "3prime",

                "minimum_length":
                    MINIMUM_LENGTH,

                "error_rate":
                    ERROR_RATE,

                "minimum_overlap":
                    MINIMUM_OVERLAP,

                "input_reads":
                    input_reads,

                "adapter_reads":
                    parsed[
                        "reads_with_adapters"
                    ],

                "adapter_fraction":
                    adapter_fraction,

                "reads_too_short":
                    parsed[
                        "reads_too_short"
                    ],

                "retained_reads":
                    output_reads,

                "retained_fraction":
                    retention_fraction,

                "retained_min_length":
                    trimmed[
                        "minimum_length"
                    ],

                "retained_modal_length":
                    trimmed[
                        "modal_length"
                    ],

                "retained_max_length":
                    trimmed[
                        "maximum_length"
                    ],

                "retained_mean_length":
                    trimmed[
                        "mean_length"
                    ],
            }
        ]
    )

    # ----------------------------------------------------------
    # 6. Formal checks
    #
    # The >=80% adapter requirement is pre-specified from Audit
    # 04d2, which found the exact long TruSeq motif in ~86% of
    # deterministically sampled reads.
    #
    # No mapping benchmark is used here.
    # ----------------------------------------------------------

    checks = {
        "frozen_pilot_is_EM36":
            pilot_label
            == "EM36",

        "frozen_run_is_SRR10267761":
            run_accession
            == "SRR10267761",

        "input_read_count_matches_frozen_pilot":
            input_reads
            == expected_input_reads,

        "Cutadapt_adapter_fraction_ge_0_80":
            adapter_fraction
            >= 0.80,

        "trimmed_output_nonempty":
            output_reads
            > 0,

        "Cutadapt_report_output_matches_FASTQ_count":
            output_reads
            == parsed[
                "reads_written"
            ],

        "all_retained_reads_ge_25bp":
            trimmed[
                "reads_below_25"
            ]
            == 0
            and trimmed[
                "minimum_length"
            ]
            >= 25,

        "some_short_insert_reads_removed":
            parsed[
                "reads_too_short"
            ]
            > 0,

        "finite_retention_fraction":
            retention_fraction
            > 0
            and retention_fraction
            < 1,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04e checks")

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
        "PRJNA576508_EM36_TRUSEQ_TRIMMING_RULE_FROZEN_AND_APPLIED"
        if passed
        else "PRJNA576508_EM36_TRIMMING_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: the adapter sequence was not explicitly stated "
        "in the Cornforth paper. It is frozen here from the TruSeq "
        "library chemistry, Cutadapt 1.13 TruSeq recommendation, "
        "and independent Audit-04d2 read-structure evidence."
    )

    print()
    print(
        "Audit 04d remains formally REVIEW_REQUIRED for the "
        "publication-versus-archive 75/77-bp discrepancy. Audit "
        "04d2 established that the archived read structure itself "
        "is internally consistent."
    )

    print()
    print(
        "No decoy alignment, PAO1 alignment, gene counting, or "
        "CF-State projection has yet been performed."
    )

    # ----------------------------------------------------------
    # 7. Save
    # ----------------------------------------------------------

    summary.to_csv(
        AUDIT_OUT
        / "EM36_frozen_TruSeq_trimming_summary.csv",
        index=False,
    )

    trimmed[
        "length_table"
    ].to_csv(
        AUDIT_OUT
        / "EM36_trimmed_read_length_distribution.csv",
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
        AUDIT_OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    (
        AUDIT_OUT
        / "AUDIT_04E_STATUS.txt"
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
            "Failed Audit 04e checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
