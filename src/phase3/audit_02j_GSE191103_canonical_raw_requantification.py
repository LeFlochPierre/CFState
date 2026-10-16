from pathlib import Path
import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd


ROOT = Path.cwd()

MANIFEST = (
    ROOT
    / "results/phase3/audit_02i_GSE191103_raw_requantification_preflight"
    / "GSE191103_WT_PA14_raw_run_manifest.csv"
)

PA14_GB = (
    ROOT
    / "references/mappings/NC_008463.1.gb"
)

PROJECTION_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

WORK = (
    ROOT
    / "data/raw/phase3/GSE191103/canonical_requantification"
)

OUT = (
    ROOT
    / "results/phase3/audit_02j_GSE191103_canonical_raw_requantification"
)

THREADS = 8


def run(cmd, log_path=None):
    print(
        "  $",
        " ".join(
            str(x)
            for x in cmd
        ),
        flush=True,
    )

    if log_path is None:
        subprocess.run(
            cmd,
            check=True,
        )
        return

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        subprocess.run(
            cmd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def run_pipe(command, log_path):
    print(
        "  $",
        command,
        flush=True,
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        subprocess.run(
            [
                "bash",
                "-c",
                "set -o pipefail; " + command,
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )


def mean_pairwise_corr(matrix, samples):
    corr = matrix[
        samples
    ].corr()

    values = []

    for i in range(
        len(samples)
    ):
        for j in range(
            i + 1,
            len(samples),
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


def pairwise_corr_table(
    matrix,
    samples,
    condition,
):
    corr = matrix[
        samples
    ].corr()

    rows = []

    for i in range(
        len(samples)
    ):
        for j in range(
            i + 1,
            len(samples),
        ):
            rows.append(
                {
                    "condition":
                        condition,

                    "sample_1":
                        samples[
                            i
                        ],

                    "sample_2":
                        samples[
                            j
                        ],

                    "pearson_r":
                        float(
                            corr.iloc[
                                i,
                                j,
                            ]
                        ),
                }
            )

    return rows


def build_reference():
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython is required."
        )

    record = SeqIO.read(
        str(
            PA14_GB
        ),
        "genbank",
    )

    fasta = WORK / "NC_008463.1.fa"
    saf = WORK / "NC_008463.1_CDS.saf"

    if not fasta.exists():
        with fasta.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                ">"
                + str(
                    record.id
                )
                + "\n"
            )

            seq = str(
                record.seq
            )

            for i in range(
                0,
                len(
                    seq
                ),
                80,
            ):
                handle.write(
                    seq[
                        i:
                        i + 80
                    ]
                    + "\n"
                )

    rows = []

    for feature in record.features:
        if feature.type != "CDS":
            continue

        locus = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if not locus:
            continue

        gene_id = str(
            locus[
                0
            ]
        ).strip()

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
                    gene_id,

                "Chr":
                    str(
                        record.id
                    ),

                "Start":
                    start,

                "End":
                    end,

                "Strand":
                    strand,
            }
        )

    saf_table = pd.DataFrame(
        rows
    )

    if saf_table[
        "GeneID"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate CDS locus tags in PA14 SAF."
        )

    saf_table.to_csv(
        saf,
        sep="\t",
        index=False,
    )

    print()
    print("Canonical PA14 counting reference")
    print(
        "  accession:",
        record.id,
    )
    print(
        "  genome length:",
        len(
            record.seq
        ),
    )
    print(
        "  CDS loci:",
        len(
            saf_table
        ),
    )

    return (
        fasta,
        saf,
        saf_table,
    )


def parse_alignment_rate(log_path):
    text = log_path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    matches = re.findall(
        r"([\d.]+)% overall alignment rate",
        text,
    )

    if not matches:
        return np.nan

    return float(
        matches[
            -1
        ]
    ) / 100.0


def parse_fc_summary(path):
    summary = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
    )

    total = summary.sum(
        axis=0
    )

    assigned = summary.loc[
        "Assigned"
    ]

    fraction = (
        assigned
        / total
    )

    return fraction


def read_featurecounts(count_path, sample_lookup):
    table = pd.read_csv(
        count_path,
        sep="\t",
        comment="#",
    )

    if "Geneid" not in table.columns:
        raise RuntimeError(
            "featureCounts output lacks Geneid."
        )

    count_columns = list(
        table.columns[
            6:
        ]
    )

    rename = {}

    for column in count_columns:
        basename = Path(
            str(
                column
            )
        ).name

        matched = None

        for sample, bam in sample_lookup.items():
            if basename == Path(
                bam
            ).name:
                matched = sample
                break

        if matched is None:
            raise RuntimeError(
                "Could not map featureCounts column: "
                + str(
                    column
                )
            )

        rename[
            column
        ] = matched

    counts = (
        table.set_index(
            "Geneid"
        )[
            count_columns
        ]
        .rename(
            columns=rename
        )
    )

    counts.index = (
        counts.index
        .astype(str)
        .str.strip()
    )

    return counts


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02j")
    print("GSE191103 WT PA14 canonical raw-read requantification")
    print("=" * 78)

    for path in [
        MANIFEST,
        PA14_GB,
        PROJECTION_MAP,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    WORK.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    sra_dir = WORK / "sra"
    fastq_dir = WORK / "fastq"
    trim_dir = WORK / "trimmed"
    bam_dir = WORK / "bam"
    log_dir = WORK / "logs"
    fc_dir = WORK / "featurecounts"

    for directory in [
        sra_dir,
        fastq_dir,
        trim_dir,
        bam_dir,
        log_dir,
        fc_dir,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest = pd.read_csv(
        MANIFEST
    )

    expected_samples = [
        "PA14_BF_1",
        "PA14_BF_2",
        "PA14_BF_3",
        "PA14_PL_1",
        "PA14_PL_2",
        "PA14_PL_3",
    ]

    if set(
        manifest[
            "sample"
        ]
    ) != set(
        expected_samples
    ):
        raise RuntimeError(
            "Manifest does not contain the expected six samples."
        )

    if manifest[
        "srr"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate SRR accessions in manifest."
        )

    fasta, saf, saf_table = build_reference()

    # ----------------------------------------------------------
    # Bowtie2 index
    # ----------------------------------------------------------

    index_prefix = WORK / "NC_008463.1_bowtie2"

    index_marker = Path(
        str(
            index_prefix
        )
        + ".1.bt2"
    )

    index_marker_large = Path(
        str(
            index_prefix
        )
        + ".1.bt2l"
    )

    if (
        not index_marker.exists()
        and not index_marker_large.exists()
    ):
        print()
        print("Building Bowtie2 index")

        run(
            [
                "bowtie2-build",
                str(
                    fasta
                ),
                str(
                    index_prefix
                ),
            ],
            log_dir
            / "bowtie2_build.log",
        )

    # ----------------------------------------------------------
    # Download, trim and align
    # ----------------------------------------------------------

    bam_lookup = {}

    alignment_rows = []

    print()
    print("Raw-read processing")

    for _, row in manifest.iterrows():
        sample = str(
            row[
                "sample"
            ]
        )

        srr = str(
            row[
                "srr"
            ]
        )

        print()
        print(
            sample,
            srr,
        )

        sra_path = (
            sra_dir
            / srr
            / (
                srr
                + ".sra"
            )
        )

        if not sra_path.exists():
            run(
                [
                    "prefetch",
                    srr,
                    "--output-directory",
                    str(
                        sra_dir
                    ),
                ],
                log_dir
                / (
                    srr
                    + "_prefetch.log"
                ),
            )

        if not sra_path.exists():
            raise RuntimeError(
                "SRA prefetch did not create "
                + str(
                    sra_path
                )
            )

        raw1 = (
            fastq_dir
            / (
                srr
                + "_1.fastq"
            )
        )

        raw2 = (
            fastq_dir
            / (
                srr
                + "_2.fastq"
            )
        )

        trim1 = (
            trim_dir
            / (
                sample
                + "_R1.fastq.gz"
            )
        )

        trim2 = (
            trim_dir
            / (
                sample
                + "_R2.fastq.gz"
            )
        )

        if (
            not trim1.exists()
            or not trim2.exists()
        ):
            if (
                not raw1.exists()
                or not raw2.exists()
            ):
                run(
                    [
                        "fasterq-dump",
                        str(
                            sra_path
                        ),
                        "--split-files",
                        "-e",
                        str(
                            THREADS
                        ),
                        "-O",
                        str(
                            fastq_dir
                        ),
                    ],
                    log_dir
                    / (
                        srr
                        + "_fasterq.log"
                    ),
                )

            if (
                not raw1.exists()
                or not raw2.exists()
            ):
                raise RuntimeError(
                    "Paired FASTQ files missing for "
                    + srr
                )

            # These SRA libraries are approximately 2x47 bp.
            # Do not use the 75-bp minimum employed for the
            # longer GSE299861 reads.
            run(
                [
                    "cutadapt",
                    "-j",
                    str(
                        THREADS
                    ),
                    "-q",
                    "20,20",
                    "-m",
                    "20",
                    "--max-n",
                    "0.1",
                    "-a",
                    "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA",
                    "-A",
                    "AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT",
                    "-o",
                    str(
                        trim1
                    ),
                    "-p",
                    str(
                        trim2
                    ),
                    str(
                        raw1
                    ),
                    str(
                        raw2
                    ),
                ],
                log_dir
                / (
                    sample
                    + "_cutadapt.log"
                ),
            )

            # The SRA archive remains frozen locally, so temporary
            # uncompressed FASTQs can be removed after successful trim.
            if trim1.exists() and trim2.exists():
                raw1.unlink(
                    missing_ok=True
                )
                raw2.unlink(
                    missing_ok=True
                )

        bam = (
            bam_dir
            / (
                sample
                + ".sorted.bam"
            )
        )

        bowtie_log = (
            log_dir
            / (
                sample
                + "_bowtie2.log"
            )
        )

        if not bam.exists():
            command = (
                "bowtie2 "
                "--very-sensitive "
                "-p "
                + str(
                    THREADS
                )
                + " -x "
                + str(
                    index_prefix
                )
                + " -1 "
                + str(
                    trim1
                )
                + " -2 "
                + str(
                    trim2
                )
                + " 2> "
                + str(
                    bowtie_log
                )
                + " | samtools sort -@ "
                + str(
                    THREADS
                )
                + " -o "
                + str(
                    bam
                )
                + " -"
            )

            run_pipe(
                command,
                log_dir
                / (
                    sample
                    + "_alignment_pipeline.log"
                ),
            )

            run(
                [
                    "samtools",
                    "index",
                    str(
                        bam
                    ),
                ],
                log_dir
                / (
                    sample
                    + "_samtools_index.log"
                ),
            )

        bam_lookup[
            sample
        ] = str(
            bam
        )

        alignment_rate = parse_alignment_rate(
            bowtie_log
        )

        alignment_rows.append(
            {
                "sample":
                    sample,

                "srr":
                    srr,

                "alignment_rate":
                    alignment_rate,
            }
        )

        print(
            "  alignment rate:",
            (
                format(
                    alignment_rate,
                    ".2%"
                )
                if np.isfinite(
                    alignment_rate
                )
                else "NA"
            ),
        )

    alignment = pd.DataFrame(
        alignment_rows
    )

    alignment.to_csv(
        OUT
        / "alignment_rates.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Empirical strandedness
    # ----------------------------------------------------------

    print()
    print("Empirical strandedness inference")

    bams = [
        bam_lookup[
            sample
        ]
        for sample
        in expected_samples
    ]

    strand_rows = []

    strand_fractions = {}

    for strand in [
        0,
        1,
        2,
    ]:
        fc_out = (
            fc_dir
            / (
                "strand_"
                + str(
                    strand
                )
                + ".txt"
            )
        )

        run(
            [
                "featureCounts",
                "-F",
                "SAF",
                "-a",
                str(
                    saf
                ),
                "-o",
                str(
                    fc_out
                ),
                "-T",
                str(
                    THREADS
                ),
                "-p",
                "--countReadPairs",
                "-B",
                "-C",
                "-s",
                str(
                    strand
                ),
            ]
            + bams,
            log_dir
            / (
                "featureCounts_strand_"
                + str(
                    strand
                )
                + ".log"
            ),
        )

        fractions = parse_fc_summary(
            Path(
                str(
                    fc_out
                )
                + ".summary"
            )
        )

        values = []

        for sample, bam in zip(
            expected_samples,
            bams,
        ):
            candidates = [
                column
                for column
                in fractions.index
                if Path(
                    str(
                        column
                    )
                ).name
                == Path(
                    bam
                ).name
            ]

            if len(
                candidates
            ) != 1:
                raise RuntimeError(
                    "Could not identify featureCounts summary "
                    "column for "
                    + sample
                )

            fraction = float(
                fractions.loc[
                    candidates[
                        0
                    ]
                ]
            )

            values.append(
                fraction
            )

            strand_rows.append(
                {
                    "strand_setting":
                        strand,

                    "sample":
                        sample,

                    "assigned_fraction":
                        fraction,
                }
            )

        strand_fractions[
            strand
        ] = float(
            np.mean(
                values
            )
        )

        print(
            "  -s",
            strand,
            "mean assigned fraction:",
            format(
                strand_fractions[
                    strand
                ],
                ".2%",
            ),
        )

    unstranded = strand_fractions[
        0
    ]

    forward = strand_fractions[
        1
    ]

    reverse = strand_fractions[
        2
    ]

    stranded_best_setting = (
        1
        if forward >= reverse
        else 2
    )

    stranded_best = max(
        forward,
        reverse,
    )

    stranded_worst = min(
        forward,
        reverse,
    )

    if (
        stranded_best
        >= 0.70
        * unstranded
        and stranded_best
        >= 2.0
        * max(
            stranded_worst,
            1e-12,
        )
    ):
        selected_strand = (
            stranded_best_setting
        )
    else:
        selected_strand = 0

    print(
        "  selected featureCounts strand:",
        selected_strand,
    )

    pd.DataFrame(
        strand_rows
    ).to_csv(
        OUT
        / "strandedness_diagnostic.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Final canonical PA14 CDS counts
    # ----------------------------------------------------------

    print()
    print("Final canonical PA14 CDS quantification")

    final_fc = (
        fc_dir
        / "GSE191103_PA14_final_counts.txt"
    )

    run(
        [
            "featureCounts",
            "-F",
            "SAF",
            "-a",
            str(
                saf
            ),
            "-o",
            str(
                final_fc
            ),
            "-T",
            str(
                THREADS
            ),
            "-p",
            "--countReadPairs",
            "-B",
            "-C",
            "-s",
            str(
                selected_strand
            ),
        ]
        + bams,
        log_dir
        / "featureCounts_final.log",
    )

    canonical_counts = read_featurecounts(
        final_fc,
        bam_lookup,
    )

    canonical_counts = canonical_counts.reindex(
        saf_table[
            "GeneID"
        ]
    )

    canonical_counts = canonical_counts[
        expected_samples
    ]

    canonical_counts.to_csv(
        OUT
        / "GSE191103_canonical_PA14_CDS_counts.csv"
    )

    final_assigned = parse_fc_summary(
        Path(
            str(
                final_fc
            )
            + ".summary"
        )
    )

    assigned_rows = []

    for sample, bam in bam_lookup.items():
        candidate = [
            column
            for column
            in final_assigned.index
            if Path(
                str(
                    column
                )
            ).name
            == Path(
                bam
            ).name
        ]

        if len(
            candidate
        ) != 1:
            raise RuntimeError(
                "Could not identify final assigned fraction for "
                + sample
            )

        assigned_rows.append(
            {
                "sample":
                    sample,

                "assigned_CDS_fraction":
                    float(
                        final_assigned.loc[
                            candidate[
                                0
                            ]
                        ]
                    ),
            }
        )

    assigned_table = pd.DataFrame(
        assigned_rows
    )

    print()
    print("Final CDS assignment")
    print(
        assigned_table
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    assigned_table.to_csv(
        OUT
        / "final_CDS_assignment_fractions.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Frozen 5286-gene projection-compatible expression
    # ----------------------------------------------------------

    projection_map = pd.read_csv(
        PROJECTION_MAP
    )

    projection_map[
        "current_locus_tag"
    ] = (
        projection_map[
            "current_locus_tag"
        ]
        .astype(str)
        .str.strip()
    )

    projection_map[
        "pao1_gene"
    ] = (
        projection_map[
            "pao1_gene"
        ]
        .astype(str)
        .str.strip()
    )

    projection_map[
        "length_bp"
    ] = pd.to_numeric(
        projection_map[
            "length_bp"
        ],
        errors="raise",
    )

    if len(
        projection_map
    ) != 5286:
        raise RuntimeError(
            "Expected frozen 5286-gene projection map."
        )

    current_tags = list(
        projection_map[
            "current_locus_tag"
        ]
    )

    absent = [
        gene
        for gene in current_tags
        if gene not in canonical_counts.index
    ]

    if absent:
        raise RuntimeError(
            "Projection-compatible current PA14 loci missing "
            "from canonical count matrix: "
            + str(
                len(
                    absent
                )
            )
        )

    proj_counts = canonical_counts.reindex(
        current_tags
    ).copy()

    lengths = pd.Series(
        projection_map[
            "length_bp"
        ].to_numpy(
            dtype=float
        ),
        index=current_tags,
    )

    rpk = proj_counts.divide(
        lengths
        / 1000.0,
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

    # Move into PAO1 model-gene naming immediately.
    pao1_names = list(
        projection_map[
            "pao1_gene"
        ]
    )

    proj_counts.index = (
        pao1_names
    )

    tpm.index = (
        pao1_names
    )

    log_tpm.index = (
        pao1_names
    )

    proj_counts.to_csv(
        OUT
        / "GSE191103_canonical_projection_gene_counts_PAO1_space.csv"
    )

    tpm.to_csv(
        OUT
        / "GSE191103_canonical_projection_compatible_TPM_PAO1_space.csv"
    )

    log_tpm.to_csv(
        OUT
        / "GSE191103_canonical_projection_log2TPM_plus1_PAO1_space.csv"
    )

    print()
    print("Canonical projection-compatible expression")
    print(
        "  genes:",
        log_tpm.shape[
            0
        ],
    )
    print(
        "  samples:",
        log_tpm.shape[
            1
        ],
    )
    print(
        "  TPM sum min:",
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
        "  TPM sum max:",
        round(
            float(
                tpm.sum(
                    axis=0
                ).max()
            ),
            3,
        ),
    )

    biofilm = [
        "PA14_BF_1",
        "PA14_BF_2",
        "PA14_BF_3",
    ]

    planktonic = [
        "PA14_PL_1",
        "PA14_PL_2",
        "PA14_PL_3",
    ]

    biofilm_r = mean_pairwise_corr(
        log_tpm,
        biofilm,
    )

    planktonic_r = mean_pairwise_corr(
        log_tpm,
        planktonic,
    )

    print()
    print("Raw-reprocessed replicate QC")
    print(
        "  biofilm mean pairwise r:",
        round(
            biofilm_r,
            4,
        ),
    )
    print(
        "  planktonic mean pairwise r:",
        round(
            planktonic_r,
            4,
        ),
    )

    pairwise_rows = []

    pairwise_rows.extend(
        pairwise_corr_table(
            log_tpm,
            biofilm,
            "biofilm",
        )
    )

    pairwise_rows.extend(
        pairwise_corr_table(
            log_tpm,
            planktonic,
            "planktonic",
        )
    )

    pairwise = pd.DataFrame(
        pairwise_rows
    )

    print()
    print("Pairwise raw-reprocessed expression correlations")
    print(
        pairwise
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    pairwise.to_csv(
        OUT
        / "raw_reprocessed_pairwise_expression_correlations.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Audit checks
    # ----------------------------------------------------------

    finite_alignment_rates = alignment[
        "alignment_rate"
    ].dropna()

    checks = {
        "six_samples_quantified":
            canonical_counts.shape[
                1
            ]
            == 6,

        "canonical_CDS_genes_ge_5900":
            canonical_counts.shape[
                0
            ]
            >= 5900,

        "projection_gene_count_5286":
            log_tpm.shape[
                0
            ]
            == 5286,

        "all_alignment_rates_ge_80pct":
            (
                len(
                    finite_alignment_rates
                )
                == 6
                and bool(
                    (
                        finite_alignment_rates
                        >= 0.80
                    ).all()
                )
            ),

        "finite_expression":
            bool(
                np.isfinite(
                    log_tpm.to_numpy(
                        dtype=float
                    )
                ).all()
            ),

        "TPM_sums_equal_1e6":
            bool(
                np.allclose(
                    tpm.sum(
                        axis=0
                    ).to_numpy(
                        dtype=float
                    ),
                    1_000_000.0,
                    rtol=0,
                    atol=1e-3,
                )
            ),
    }

    # Important:
    # replicate correlations are reported here but are NOT yet
    # converted into a PASS/FAIL gate. The purpose of Audit 02j
    # is to determine whether canonical raw reprocessing improves
    # the failed processed-matrix QC without moving the old gate.
    passed = all(
        checks.values()
    )

    print()
    print("Technical requantification checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    print()
    print(
        "Audit 02j technical status:",
        "PASS"
        if passed
        else "FAIL",
    )

    print()
    print(
        "Replicate-QC decision metrics "
        "(not used to alter Audit 02g retroactively)"
    )
    print(
        "  previous processed biofilm r: 0.4917"
    )
    print(
        "  canonical raw biofilm r:",
        round(
            biofilm_r,
            4,
        ),
    )
    print(
        "  previous processed planktonic r: 0.6756"
    )
    print(
        "  canonical raw planktonic r:",
        round(
            planktonic_r,
            4,
        ),
    )

    if (
        biofilm_r >= 0.75
        and planktonic_r >= 0.75
    ):
        decision = (
            "RAW_REPROCESSING_RESCUES_REPLICATE_QC"
        )

    elif (
        biofilm_r > 0.4917
        or planktonic_r > 0.6756
    ):
        decision = (
            "RAW_REPROCESSING_IMPROVES_BUT_DOES_NOT_RESCUE_QC"
        )

    else:
        decision = (
            "RAW_REPROCESSING_DOES_NOT_RESCUE_QC"
        )

    print()
    print(
        "REPLICATE QC DECISION:",
        decision,
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

    summary = [
        "# CF-State Phase 3 — Audit 02j",
        "",
        "## GSE191103 canonical PA14 raw-read requantification",
        "",
        "Six WT PA14 libraries were reprocessed from raw SRA reads "
        "against the frozen NC_008463.1 reference.",
        "",
        "Pipeline:",
        "",
        "raw paired FASTQ → Cutadapt → Bowtie2 → empirical "
        "strandedness → featureCounts → frozen 5,286-gene "
        "PAO1↔PA14 projection-compatible universe → "
        "projection-compatible TPM.",
        "",
        "- Selected featureCounts strand: "
        + str(
            selected_strand
        ),
        "- Biofilm replicate mean r: "
        + format(
            biofilm_r,
            ".4f",
        ),
        "- Planktonic replicate mean r: "
        + format(
            planktonic_r,
            ".4f",
        ),
        "",
        "## Replicate-QC decision",
        "",
        decision,
        "",
        "Audit 02g remains unchanged; this raw-read audit tests "
        "whether its processed-expression QC failure was caused "
        "by the supplied processed matrix.",
    ]

    (
        OUT
        / "AUDIT_02J_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary
        )
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
            "Failed technical checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
