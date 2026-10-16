from pathlib import Path
from collections import Counter
import gzip

import numpy as np
import pandas as pd


ROOT = Path.cwd()

FASTQ = (
    ROOT
    / "data/raw/phase4/PRJNA576508/pilot_EM36"
    / "SRR10267761.fastq.gz"
)

MANIFEST = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "PRJNA576508_Atlanta_sputum_manifest.csv"
)

PILOT = (
    ROOT
    / "results/phase4/audit_04c_PRJNA576508_reference_and_toolchain_freeze"
    / "FROZEN_Atlanta_technical_pilot.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_04d2_PRJNA576508_EM36_read_structure"
)

RUN = "SRR10267761"

SAMPLE_STRIDE = 100

TRUSEQ_PREFIX = "AGATCGGAAGAG"

TRUSEQ_LONG = (
    "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA"
)


def all_occurrences(sequence, motif):
    positions = []

    start = 0

    while True:
        index = sequence.find(
            motif,
            start,
        )

        if index < 0:
            break

        positions.append(
            index
        )

        start = (
            index
            + 1
        )

    return positions


def summarize_counter(counter):
    total = sum(
        counter.values()
    )

    rows = []

    for value, count in sorted(
        counter.items()
    ):
        rows.append(
            {
                "value":
                    value,

                "count":
                    count,

                "fraction":
                    (
                        count
                        / total
                        if total
                        > 0
                        else np.nan
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04d2")
    print("EM36 archived-read structure and TruSeq motif-position diagnostic")
    print("=" * 78)

    for path in [
        FASTQ,
        MANIFEST,
        PILOT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. ENA/SRA metadata consistency
    # ----------------------------------------------------------

    manifest = pd.read_csv(
        MANIFEST
    )

    row = manifest.loc[
        manifest[
            "run_accession"
        ]
        .astype(str)
        .eq(
            RUN
        )
    ]

    if len(
        row
    ) != 1:
        raise RuntimeError(
            "Could not uniquely resolve "
            + RUN
            + " in frozen ENA manifest."
        )

    row = row.iloc[
        0
    ]

    ena_read_count = int(
        row[
            "read_count"
        ]
    )

    ena_base_count = pd.to_numeric(
        pd.Series(
            [
                row[
                    "base_count"
                ]
            ]
        ),
        errors="coerce",
    ).iloc[
        0
    ]

    if pd.isna(
        ena_base_count
    ):
        ena_mean_length = np.nan

    else:
        ena_base_count = int(
            ena_base_count
        )

        ena_mean_length = (
            ena_base_count
            / ena_read_count
        )

    print()
    print("Frozen ENA metadata")
    print(
        "  read count:",
        ena_read_count,
    )
    print(
        "  base count:",
        ena_base_count,
    )
    print(
        "  ENA mean bases/read:",
        (
            round(
                ena_mean_length,
                4,
            )
            if np.isfinite(
                ena_mean_length
            )
            else "NA"
        ),
    )

    # ----------------------------------------------------------
    # 2. Deterministic FASTQ sampling
    # ----------------------------------------------------------

    read_count = 0
    sampled_reads = 0
    sampled_bases = 0

    length_counter = Counter()

    prefix_position_counter = Counter()
    long_position_counter = Counter()

    prefix_distance_from_end_counter = Counter()
    long_distance_from_end_counter = Counter()

    read_start_12 = Counter()
    read_start_20 = Counter()

    reads_with_prefix = 0
    reads_with_long = 0

    reads_starting_prefix = 0
    reads_starting_long = 0

    reads_ending_prefix = 0
    reads_ending_long = 0

    # Exact 3' adapter-trimming simulation:
    #
    # If a motif occurs, trim from the FIRST occurrence onward.
    # This is diagnostic only and is NOT used to generate data.
    prefix_trimmed_lengths = Counter()
    long_trimmed_lengths = Counter()

    prefix_kept_ge25 = 0
    long_kept_ge25 = 0

    prefix_found_lengths = []
    long_found_lengths = []

    with gzip.open(
        FASTQ,
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
                    "Incomplete FASTQ record."
                )

            read_count += 1

            if (
                read_count
                % SAMPLE_STRIDE
                != 0
            ):
                continue

            sequence = sequence.rstrip(
                "\r\n"
            ).upper()

            sampled_reads += 1
            sampled_bases += len(
                sequence
            )

            length_counter[
                len(
                    sequence
                )
            ] += 1

            if len(
                sequence
            ) >= 12:
                read_start_12[
                    sequence[
                        :12
                    ]
                ] += 1

            if len(
                sequence
            ) >= 20:
                read_start_20[
                    sequence[
                        :20
                    ]
                ] += 1

            prefix_positions = all_occurrences(
                sequence,
                TRUSEQ_PREFIX,
            )

            long_positions = all_occurrences(
                sequence,
                TRUSEQ_LONG,
            )

            if prefix_positions:
                reads_with_prefix += 1

                first = prefix_positions[
                    0
                ]

                prefix_found_lengths.append(
                    first
                )

                prefix_trimmed_lengths[
                    first
                ] += 1

                if first >= 25:
                    prefix_kept_ge25 += 1

                for position in prefix_positions:
                    prefix_position_counter[
                        position
                    ] += 1

                    distance = (
                        len(
                            sequence
                        )
                        - (
                            position
                            + len(
                                TRUSEQ_PREFIX
                            )
                        )
                    )

                    prefix_distance_from_end_counter[
                        distance
                    ] += 1

                if sequence.startswith(
                    TRUSEQ_PREFIX
                ):
                    reads_starting_prefix += 1

                if sequence.endswith(
                    TRUSEQ_PREFIX
                ):
                    reads_ending_prefix += 1

            if long_positions:
                reads_with_long += 1

                first = long_positions[
                    0
                ]

                long_found_lengths.append(
                    first
                )

                long_trimmed_lengths[
                    first
                ] += 1

                if first >= 25:
                    long_kept_ge25 += 1

                for position in long_positions:
                    long_position_counter[
                        position
                    ] += 1

                    distance = (
                        len(
                            sequence
                        )
                        - (
                            position
                            + len(
                                TRUSEQ_LONG
                            )
                        )
                    )

                    long_distance_from_end_counter[
                        distance
                    ] += 1

                if sequence.startswith(
                    TRUSEQ_LONG
                ):
                    reads_starting_long += 1

                if sequence.endswith(
                    TRUSEQ_LONG
                ):
                    reads_ending_long += 1

    observed_mean_length = (
        sampled_bases
        / sampled_reads
    )

    print()
    print("Deterministic sampled FASTQ structure")
    print(
        "  total FASTQ reads:",
        read_count,
    )
    print(
        "  sampled reads:",
        sampled_reads,
    )
    print(
        "  sampled mean read length:",
        round(
            observed_mean_length,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 3. Position summary
    # ----------------------------------------------------------

    prefix_fraction = (
        reads_with_prefix
        / sampled_reads
    )

    long_fraction = (
        reads_with_long
        / sampled_reads
    )

    print()
    print("TruSeq exact motif occurrence")
    print(
        "  prefix:",
        TRUSEQ_PREFIX,
    )
    print(
        "    reads containing motif:",
        reads_with_prefix,
        "/",
        sampled_reads,
        "=",
        round(
            prefix_fraction,
            6,
        ),
    )
    print(
        "    reads starting with motif:",
        reads_starting_prefix,
        "=",
        round(
            reads_starting_prefix
            / sampled_reads,
            6,
        ),
    )
    print(
        "    reads ending with motif:",
        reads_ending_prefix,
        "=",
        round(
            reads_ending_prefix
            / sampled_reads,
            6,
        ),
    )

    print()
    print(
        "  long motif:",
        TRUSEQ_LONG,
    )
    print(
        "    reads containing motif:",
        reads_with_long,
        "/",
        sampled_reads,
        "=",
        round(
            long_fraction,
            6,
        ),
    )
    print(
        "    reads starting with motif:",
        reads_starting_long,
        "=",
        round(
            reads_starting_long
            / sampled_reads,
            6,
        ),
    )
    print(
        "    reads ending with motif:",
        reads_ending_long,
        "=",
        round(
            reads_ending_long
            / sampled_reads,
            6,
        ),
    )

    # ----------------------------------------------------------
    # 4. Exact occurrence-position tables
    # ----------------------------------------------------------

    prefix_positions = summarize_counter(
        prefix_position_counter
    )

    long_positions = summarize_counter(
        long_position_counter
    )

    prefix_distance = summarize_counter(
        prefix_distance_from_end_counter
    )

    long_distance = summarize_counter(
        long_distance_from_end_counter
    )

    print()
    print("Most common TruSeq-prefix start positions (0-based)")
    print(
        prefix_positions
        .sort_values(
            "count",
            ascending=False,
        )
        .head(
            30
        )
        .round(
            {
                "fraction":
                    6
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Most common long-TruSeq start positions (0-based)")
    print(
        long_positions
        .sort_values(
            "count",
            ascending=False,
        )
        .head(
            30
        )
        .round(
            {
                "fraction":
                    6
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Most common 5' sequences
    # ----------------------------------------------------------

    start12_rows = []

    for rank, (
        sequence,
        count,
    ) in enumerate(
        read_start_12.most_common(
            20
        ),
        start=1,
    ):
        start12_rows.append(
            {
                "rank":
                    rank,

                "sequence":
                    sequence,

                "count":
                    count,

                "fraction":
                    count
                    / sampled_reads,
            }
        )

    start20_rows = []

    for rank, (
        sequence,
        count,
    ) in enumerate(
        read_start_20.most_common(
            20
        ),
        start=1,
    ):
        start20_rows.append(
            {
                "rank":
                    rank,

                "sequence":
                    sequence,

                "count":
                    count,

                "fraction":
                    count
                    / sampled_reads,
            }
        )

    start12 = pd.DataFrame(
        start12_rows
    )

    start20 = pd.DataFrame(
        start20_rows
    )

    print()
    print("Most common exact 5-prime 12-mers")
    print(
        start12
        .round(
            {
                "fraction":
                    6
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Most common exact 5-prime 20-mers")
    print(
        start20
        .round(
            {
                "fraction":
                    6
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Exact-motif trim simulation
    #
    # Again: DIAGNOSTIC ONLY.
    #
    # If Cutadapt were trimming a 3' adapter corresponding exactly
    # to this motif, the biological insert length would equal the
    # motif's first start position.
    # ----------------------------------------------------------

    prefix_trim = summarize_counter(
        prefix_trimmed_lengths
    ).rename(
        columns={
            "value":
                "simulated_insert_length"
        }
    )

    long_trim = summarize_counter(
        long_trimmed_lengths
    ).rename(
        columns={
            "value":
                "simulated_insert_length"
        }
    )

    median_prefix_insert = (
        float(
            np.median(
                prefix_found_lengths
            )
        )
        if prefix_found_lengths
        else np.nan
    )

    median_long_insert = (
        float(
            np.median(
                long_found_lengths
            )
        )
        if long_found_lengths
        else np.nan
    )

    print()
    print("Diagnostic exact-adapter trim simulation")
    print(
        "  prefix-motif median first occurrence position:",
        median_prefix_insert,
    )
    print(
        "  prefix-motif reads retaining >=25 bp before motif:",
        prefix_kept_ge25,
        "/",
        reads_with_prefix,
        "=",
        round(
            (
                prefix_kept_ge25
                / reads_with_prefix
                if reads_with_prefix
                else np.nan
            ),
            6,
        ),
    )

    print(
        "  long-motif median first occurrence position:",
        median_long_insert,
    )
    print(
        "  long-motif reads retaining >=25 bp before motif:",
        long_kept_ge25,
        "/",
        reads_with_long,
        "=",
        round(
            (
                long_kept_ge25
                / reads_with_long
                if reads_with_long
                else np.nan
            ),
            6,
        ),
    )

    # ----------------------------------------------------------
    # 7. Determine read-structure category
    #
    # This classification is descriptive only.
    # It does NOT trim or modify reads.
    # ----------------------------------------------------------

    prefix_start_fraction = (
        reads_starting_prefix
        / sampled_reads
    )

    long_start_fraction = (
        reads_starting_long
        / sampled_reads
    )

    if (
        long_start_fraction
        >= 0.50
    ):
        structure_class = (
            "TRUSEQ_SEQUENCE_PREDOMINANTLY_AT_5PRIME_START"
        )

    elif (
        long_fraction
        >= 0.50
        and median_long_insert
        >= 10
    ):
        structure_class = (
            "TRUSEQ_SEQUENCE_PREDOMINANTLY_INTERNAL_AFTER_SHORT_PREFIX"
        )

    elif (
        prefix_fraction
        >= 0.05
    ):
        structure_class = (
            "TRUSEQ_ADAPTER_LIKE_READTHROUGH_PRESENT"
        )

    else:
        structure_class = (
            "NO_STRONG_EXACT_TRUSEQ_SIGNATURE"
        )

    metadata_length_consistent = True

    if np.isfinite(
        ena_mean_length
    ):
        metadata_length_consistent = (
            abs(
                observed_mean_length
                - ena_mean_length
            )
            <= 0.05
        )

    summary = pd.DataFrame(
        [
            {
                "run":
                    RUN,

                "ENA_read_count":
                    ena_read_count,

                "ENA_base_count":
                    ena_base_count,

                "ENA_mean_length":
                    ena_mean_length,

                "sampled_mean_length":
                    observed_mean_length,

                "ENA_vs_FASTQ_mean_length_consistent":
                    metadata_length_consistent,

                "TruSeq_prefix_fraction":
                    prefix_fraction,

                "TruSeq_long_fraction":
                    long_fraction,

                "TruSeq_prefix_start_fraction":
                    prefix_start_fraction,

                "TruSeq_long_start_fraction":
                    long_start_fraction,

                "median_prefix_first_position":
                    median_prefix_insert,

                "median_long_first_position":
                    median_long_insert,

                "read_structure_classification":
                    structure_class,
            }
        ]
    )

    print()
    print("Read-structure diagnostic summary")
    print(
        summary
        .round(
            6
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 8. Formal checks
    #
    # We are NOT requiring 75 bp here.
    #
    # Audit 04d retains that formal failure.
    #
    # 04d2 asks whether the unexpected observed read length is
    # internally consistent with the archived ENA object and
    # whether the adapter structure can be characterized.
    # ----------------------------------------------------------

    checks = {
        "FASTQ_read_count_matches_ENA":
            read_count
            == ena_read_count,

        "ENA_and_FASTQ_mean_read_length_consistent":
            metadata_length_consistent,

        "diagnostic_sample_ge_500k":
            sampled_reads
            >= 500000,

        "TruSeq_prefix_detected_in_majority":
            prefix_fraction
            >= 0.50,

        "TruSeq_long_detected_in_majority":
            long_fraction
            >= 0.50,

        "read_structure_classified":
            structure_class
            != "NO_STRONG_EXACT_TRUSEQ_SIGNATURE",
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04d2 checks")

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
        "PRJNA576508_EM36_ARCHIVED_READ_STRUCTURE_CHARACTERIZED"
        if passed
        else "PRJNA576508_EM36_READ_STRUCTURE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: Audit 04d remains formally REVIEW_REQUIRED "
        "because the archived FASTQ is not modal 75 bp. Audit 04d2 "
        "does not alter that result."
    )

    print()
    print(
        "No read has been trimmed or aligned and no CF-State "
        "projection has been performed."
    )

    # ----------------------------------------------------------
    # 9. Save
    # ----------------------------------------------------------

    prefix_positions.to_csv(
        OUT
        / "TruSeq_prefix_start_positions.csv",
        index=False,
    )

    long_positions.to_csv(
        OUT
        / "TruSeq_long_start_positions.csv",
        index=False,
    )

    prefix_distance.to_csv(
        OUT
        / "TruSeq_prefix_distance_from_read_end.csv",
        index=False,
    )

    long_distance.to_csv(
        OUT
        / "TruSeq_long_distance_from_read_end.csv",
        index=False,
    )

    start12.to_csv(
        OUT
        / "top_5prime_12mers.csv",
        index=False,
    )

    start20.to_csv(
        OUT
        / "top_5prime_20mers.csv",
        index=False,
    )

    prefix_trim.to_csv(
        OUT
        / "diagnostic_prefix_exact_trim_lengths.csv",
        index=False,
    )

    long_trim.to_csv(
        OUT
        / "diagnostic_long_exact_trim_lengths.csv",
        index=False,
    )

    summary.to_csv(
        OUT
        / "EM36_read_structure_summary.csv",
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
            for key, value in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    (
        OUT
        / "AUDIT_04D2_STATUS.txt"
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
            "Failed Audit 04d2 checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
