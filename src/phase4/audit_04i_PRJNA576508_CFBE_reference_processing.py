from pathlib import Path
import gzip
import hashlib
import importlib.util
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd


ROOT = Path.cwd()

PIPELINE_SCRIPT = (
    ROOT
    / "src/phase4/audit_04g_PRJNA576508_Atlanta_full_cohort_processing.py"
)

MANIFEST = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "PRJNA576508_full_ENA_manifest.csv"
)

REFERENCE_FREEZE = (
    ROOT
    / "results/phase4/audit_04h3_PRJNA576508_reference_plan_repair"
    / "FROZEN_four_CFBE_reference_runs.csv"
)

REFERENCE_PLAN = (
    ROOT
    / "results/phase4/audit_04h3_PRJNA576508_reference_plan_repair"
    / "FROZEN_corrected_external_reference_plan.csv"
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
    / "data/raw/phase4/PRJNA576508/CFBE_reference_4"
)

WORK_DIR = (
    ROOT
    / "data/processed/phase4/PRJNA576508/CFBE_reference_4"
)

OUT = (
    ROOT
    / "results/phase4/audit_04i_PRJNA576508_CFBE_reference_processing"
)

COUNT_DIR = (
    OUT
    / "per_sample_counts"
)

LOG_DIR = (
    OUT
    / "logs"
)

SUMMARY_DIR = (
    OUT
    / "per_sample_summaries"
)

THREADS = 8

MIN_PAIR_CORRELATION = 0.90
MIN_ADAPTER_FRACTION = 0.80
MIN_DETECTED_GENES = 4000


def load_frozen_pipeline():
    spec = importlib.util.spec_from_file_location(
        "audit04g_frozen",
        PIPELINE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not import frozen Audit 04g pipeline."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


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


def first_value(value):
    return str(
        value
    ).split(";")[0].strip()


def normalize_url(value):
    text = first_value(
        value
    )

    if text.startswith(
        "ftp://"
    ):
        return (
            "https://"
            + text[len("ftp://"):]
        )

    if text.startswith(
        "ftp."
    ):
        return (
            "https://"
            + text
        )

    return text


def valid_md5(
    path,
    expected_md5,
):
    if not path.exists():
        return False

    observed = md5_file(
        path
    )

    return (
        observed
        == expected_md5
    )


def try_curl(
    url,
    destination,
):
    curl = shutil.which(
        "curl"
    )

    if curl is None:
        return False

    temporary = Path(
        str(destination)
        + ".part"
    )

    command = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "20",
        "--retry-delay",
        "15",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(temporary),
        url,
    ]

    print()
    print(
        " ",
        " ".join(
            command
        )
    )

    result = subprocess.run(
        command
    )

    if result.returncode != 0:
        return False

    if not temporary.exists():
        return False

    temporary.rename(
        destination
    )

    return True


def try_ena_data_get(
    run,
    destination,
):
    executable = shutil.which(
        "enaDataGet"
    )

    if executable is None:
        return False

    with tempfile.TemporaryDirectory(
        prefix="ena_cfbe_"
    ) as temporary_directory:
        temporary_directory = Path(
            temporary_directory
        )

        command = [
            executable,
            "-f",
            "fastq",
            "-d",
            str(
                temporary_directory
            ),
            run,
        ]

        print()
        print(
            " ",
            " ".join(
                command
            )
        )

        result = subprocess.run(
            command
        )

        if result.returncode != 0:
            return False

        candidates = list(
            temporary_directory.rglob(
                run
                + ".fastq.gz"
            )
        )

        if len(
            candidates
        ) != 1:
            return False

        shutil.copy2(
            candidates[0],
            destination,
        )

    return True


def fetch_fastq(
    run,
    url,
    expected_md5,
    destination,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if valid_md5(
        destination,
        expected_md5,
    ):
        print(
            "  existing raw FASTQ MD5 PASS"
        )
        return

    if destination.exists():
        destination.unlink()

    partial = Path(
        str(destination)
        + ".part"
    )

    partial.unlink(
        missing_ok=True
    )

    print()
    print(
        "Attempt 1: ENA HTTPS"
    )

    success = try_curl(
        normalize_url(
            url
        ),
        destination,
    )

    if (
        success
        and valid_md5(
            destination,
            expected_md5,
        )
    ):
        print(
            "  HTTPS MD5 PASS"
        )
        return

    destination.unlink(
        missing_ok=True
    )

    partial.unlink(
        missing_ok=True
    )

    print()
    print(
        "Attempt 2: enaDataGet"
    )

    success = try_ena_data_get(
        run,
        destination,
    )

    if (
        success
        and valid_md5(
            destination,
            expected_md5,
        )
    ):
        print(
            "  enaDataGet MD5 PASS"
        )
        return

    destination.unlink(
        missing_ok=True
    )

    raise RuntimeError(
        "Could not retrieve exact FASTQ for "
        + run
    )


def process_reference(
    frozen_pipeline,
    row,
):
    run = str(
        row[
            "run_accession"
        ]
    )

    label = str(
        row[
            "pair_sample"
        ]
    )

    expected_md5 = first_value(
        row[
            "fastq_md5"
        ]
    ).lower()

    url = first_value(
        row[
            "fastq_ftp"
        ]
    )

    expected_raw_reads = int(
        row[
            "read_count"
        ]
    )

    raw = (
        RAW_DIR
        / (
            run
            + ".fastq.gz"
        )
    )

    sample_work = (
        WORK_DIR
        / label
    )

    sample_work.mkdir(
        parents=True,
        exist_ok=True,
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

    count_csv = (
        COUNT_DIR
        / (
            label
            + "_gene_counts.csv"
        )
    )

    stat_csv = (
        OUT
        / (
            label
            + "_Rsubread_stat.csv"
        )
    )

    summary_csv = (
        SUMMARY_DIR
        / (
            label
            + "_technical_summary.csv"
        )
    )

    cutadapt_log = (
        LOG_DIR
        / (
            label
            + "_cutadapt.log"
        )
    )

    decoy_log = (
        LOG_DIR
        / (
            label
            + "_decoy_bowtie2.log"
        )
    )

    pao1_log = (
        LOG_DIR
        / (
            label
            + "_PAO1_bowtie2.log"
        )
    )

    print()
    print("=" * 78)
    print(
        "PROCESSING",
        label,
        run,
    )
    print("=" * 78)

    fetch_fastq(
        run,
        url,
        expected_md5,
        raw,
    )

    raw_md5_pass = (
        md5_file(
            raw
        )
        == expected_md5
    )

    # ------------------------------------------------------
    # Exact Audit-04g trimming rule
    # ------------------------------------------------------

    trimmed.unlink(
        missing_ok=True
    )

    cutadapt_command = [
        shutil.which(
            "cutadapt"
        ),
        "-a",
        frozen_pipeline.ADAPTER,
        "-m",
        str(
            frozen_pipeline.MINIMUM_LENGTH
        ),
        "-e",
        str(
            frozen_pipeline.ERROR_RATE
        ),
        "-O",
        str(
            frozen_pipeline.MINIMUM_OVERLAP
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
            label
            + ": Cutadapt failed."
        )

    cutadapt = frozen_pipeline.parse_cutadapt_report(
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
    # Exact Audit-04g decoy alignment
    # ------------------------------------------------------

    decoy_unmapped.unlink(
        missing_ok=True
    )

    frozen_pipeline.run_command(
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

    decoy = frozen_pipeline.parse_bowtie_summary(
        decoy_log
    )

    after_decoy = frozen_pipeline.count_fastq_reads(
        decoy_unmapped
    )

    if (
        after_decoy
        != decoy[
            "zero"
        ]
    ):
        raise RuntimeError(
            label
            + ": decoy retained-read mismatch."
        )

    # ------------------------------------------------------
    # Exact Audit-04g PAO1 alignment
    # ------------------------------------------------------

    sam.unlink(
        missing_ok=True
    )

    frozen_pipeline.run_command(
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

    pao1 = frozen_pipeline.parse_bowtie_summary(
        pao1_log
    )

    # ------------------------------------------------------
    # Exact Audit-04g Rsubread rule
    # ------------------------------------------------------

    frozen_pipeline.run_featurecounts(
        label,
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
            label
            + ": Assigned status not unique."
        )

    assigned = int(
        assigned_row.iloc[
            0
        ][
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
            label
            + ": assigned count != sum of genes."
        )

    checks = {
        "raw_MD5_pass":
            raw_md5_pass,

        "raw_read_count_matches_ENA":
            cutadapt[
                "input_reads"
            ]
            == expected_raw_reads,

        "adapter_fraction_ge_0_80":
            adapter_fraction
            >= MIN_ADAPTER_FRACTION,

        "trimmed_reads_nonzero":
            cutadapt[
                "written_reads"
            ]
            > 0,

        "decoy_output_consistent":
            after_decoy
            == decoy[
                "zero"
            ],

        "PAO1_mapping_nonzero":
            pao1[
                "mapped"
            ]
            > 0,

        "Rsubread_assigned_nonzero":
            assigned
            > 0,

        "assigned_equals_sum_counts":
            assigned
            == sum_counts,

        "detected_genes_ge_4000":
            detected
            >= MIN_DETECTED_GENES,

        "count_table_has_5567_genes":
            len(
                counts
            )
            == 5567,
    }

    technical_pass = all(
        checks.values()
    )

    summary = {
        "pair_sample":
            label,

        "pair":
            str(
                row[
                    "pair"
                ]
            ),

        "replicate":
            int(
                row[
                    "replicate"
                ]
            ),

        "run_accession":
            run,

        "expected_raw_reads":
            expected_raw_reads,

        "raw_reads":
            cutadapt[
                "input_reads"
            ],

        "raw_MD5_pass":
            raw_md5_pass,

        "adapter_fraction":
            adapter_fraction,

        "trimmed_reads":
            cutadapt[
                "written_reads"
            ],

        "decoy_mapped_reads":
            decoy[
                "mapped"
            ],

        "decoy_mapped_fraction":
            decoy[
                "mapped_fraction"
            ],

        "reads_after_decoy":
            after_decoy,

        "PAO1_mapped_reads":
            pao1[
                "mapped"
            ],

        "PAO1_mapping_fraction":
            pao1[
                "mapped_fraction"
            ],

        "Rsubread_assigned_reads":
            assigned,

        "detected_genes":
            detected,

        "technical_pass":
            technical_pass,
    }

    for key, value in checks.items():
        summary[
            "check_"
            + key
        ] = value

    pd.DataFrame(
        [
            summary
        ]
    ).to_csv(
        summary_csv,
        index=False,
    )

    print()
    print(
        label,
        "technical summary"
    )
    print(
        "  raw reads:",
        cutadapt[
            "input_reads"
        ],
    )
    print(
        "  adapter fraction:",
        round(
            adapter_fraction,
            6,
        ),
    )
    print(
        "  retained after trimming:",
        cutadapt[
            "written_reads"
        ],
    )
    print(
        "  decoy-mapped fraction:",
        round(
            decoy[
                "mapped_fraction"
            ],
            6,
        ),
    )
    print(
        "  PAO1 mapped:",
        pao1[
            "mapped"
        ],
    )
    print(
        "  Rsubread assigned:",
        assigned,
    )
    print(
        "  detected genes:",
        detected,
    )
    print(
        "  technical status:",
        (
            "PASS"
            if technical_pass
            else "FAIL"
        ),
    )

    if technical_pass:
        sam.unlink(
            missing_ok=True
        )

        trimmed.unlink(
            missing_ok=True
        )

        decoy_unmapped.unlink(
            missing_ok=True
        )

    return summary


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04i")
    print("Process frozen four-run CFBE external reference")
    print("=" * 78)

    for path in [
        PIPELINE_SCRIPT,
        MANIFEST,
        REFERENCE_FREEZE,
        REFERENCE_PLAN,
        PAO1_SAF,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
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

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    frozen_pipeline = load_frozen_pipeline()

    # run_featurecounts writes its temporary R script under
    # the imported module's OUT directory.
    frozen_pipeline.OUT = OUT
    frozen_pipeline.PAO1_SAF = PAO1_SAF

    manifest = pd.read_csv(
        MANIFEST
    )

    frozen = pd.read_csv(
        REFERENCE_FREEZE
    )

    required_columns = [
        "run_accession",
        "pair_sample",
        "pair",
        "replicate",
    ]

    for column in required_columns:
        if column not in frozen.columns:
            raise RuntimeError(
                "Frozen reference table missing "
                + column
            )

    merged = frozen[
        required_columns
    ].merge(
        manifest,
        on="run_accession",
        how="left",
        validate="one_to_one",
    )

    if len(
        merged
    ) != 4:
        raise RuntimeError(
            "Expected exactly four reference runs."
        )

    required_manifest_fields = [
        "fastq_ftp",
        "fastq_md5",
        "read_count",
    ]

    for column in required_manifest_fields:
        if merged[
            column
        ].isna().any():
            raise RuntimeError(
                "Missing manifest field: "
                + column
            )

    merged = merged.sort_values(
        [
            "pair",
            "replicate",
        ]
    ).reset_index(
        drop=True
    )

    print()
    print("Frozen processing rules")
    print(
        "  adapter:",
        frozen_pipeline.ADAPTER,
    )
    print(
        "  minimum retained length:",
        frozen_pipeline.MINIMUM_LENGTH,
    )
    print(
        "  error rate:",
        frozen_pipeline.ERROR_RATE,
    )
    print(
        "  minimum overlap:",
        frozen_pipeline.MINIMUM_OVERLAP,
    )
    print(
        "  pair replicate correlation gate:",
        MIN_PAIR_CORRELATION,
    )
    print(
        "  CF candidate exposure:",
        "NONE",
    )

    summaries = []

    for _, row in merged.iterrows():
        summary = process_reference(
            frozen_pipeline,
            row,
        )

        summaries.append(
            summary
        )

    summary = pd.DataFrame(
        summaries
    ).sort_values(
        [
            "pair",
            "replicate",
        ]
    ).reset_index(
        drop=True
    )

    summary.to_csv(
        OUT
        / "CFBE_4_technical_processing_summary.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Assemble frozen 4-sample raw count matrix
    # ----------------------------------------------------------

    matrix = None

    for label in [
        "C1",
        "C2",
        "R1",
        "R2",
    ]:
        path = (
            COUNT_DIR
            / (
                label
                + "_gene_counts.csv"
            )
        )

        if not path.exists():
            raise RuntimeError(
                "Missing count file for "
                + label
            )

        table = pd.read_csv(
            path
        )

        if list(
            table.columns
        ) != [
            "GeneID",
            "count",
        ]:
            raise RuntimeError(
                "Unexpected count-table columns for "
                + label
            )

        table = table.rename(
            columns={
                "count":
                    label
            }
        )

        if matrix is None:
            matrix = table

        else:
            matrix = matrix.merge(
                table,
                on="GeneID",
                how="outer",
                validate="one_to_one",
            )

    matrix = matrix.fillna(
        0
    )

    for label in [
        "C1",
        "C2",
        "R1",
        "R2",
    ]:
        matrix[
            label
        ] = pd.to_numeric(
            matrix[
                label
            ],
            errors="raise",
        ).astype(
            int
        )

    count_matrix_path = (
        OUT
        / "FROZEN_CFBE_4_raw_PAO1_gene_counts.csv"
    )

    matrix.to_csv(
        count_matrix_path,
        index=False,
    )

    # ----------------------------------------------------------
    # Replicate coherence on log2(CPM + 1)
    #
    # This is independent of the CF-State candidate.
    # ----------------------------------------------------------

    expression = matrix.set_index(
        "GeneID"
    ).astype(
        float
    )

    library_sizes = expression.sum(
        axis=0
    )

    cpm = expression.divide(
        library_sizes,
        axis=1,
    ) * 1e6

    log_cpm = np.log2(
        cpm
        + 1.0
    )

    c_corr = float(
        log_cpm[
            "C1"
        ].corr(
            log_cpm[
                "C2"
            ],
            method="pearson",
        )
    )

    r_corr = float(
        log_cpm[
            "R1"
        ].corr(
            log_cpm[
                "R2"
            ],
            method="pearson",
        )
    )

    c_spearman = float(
        log_cpm[
            "C1"
        ].corr(
            log_cpm[
                "C2"
            ],
            method="spearman",
        )
    )

    r_spearman = float(
        log_cpm[
            "R1"
        ].corr(
            log_cpm[
                "R2"
            ],
            method="spearman",
        )
    )

    c_mean = log_cpm[
        [
            "C1",
            "C2",
        ]
    ].mean(
        axis=1
    )

    r_mean = log_cpm[
        [
            "R1",
            "R2",
        ]
    ].mean(
        axis=1
    )

    cr_mean_corr = float(
        c_mean.corr(
            r_mean,
            method="pearson",
        )
    )

    print()
    print("=" * 78)
    print("CFBE replicate coherence")
    print("=" * 78)

    print(
        "  C1 vs C2 Pearson:",
        round(
            c_corr,
            6,
        ),
    )
    print(
        "  C1 vs C2 Spearman:",
        round(
            c_spearman,
            6,
        ),
    )
    print(
        "  R1 vs R2 Pearson:",
        round(
            r_corr,
            6,
        ),
    )
    print(
        "  R1 vs R2 Spearman:",
        round(
            r_spearman,
            6,
        ),
    )
    print(
        "  C-pair mean vs R-pair mean Pearson:",
        round(
            cr_mean_corr,
            6,
        ),
    )

    coherence = pd.DataFrame(
        [
            {
                "comparison":
                    "C1_vs_C2",

                "pearson_log2CPM":
                    c_corr,

                "spearman_log2CPM":
                    c_spearman,
            },

            {
                "comparison":
                    "R1_vs_R2",

                "pearson_log2CPM":
                    r_corr,

                "spearman_log2CPM":
                    r_spearman,
            },

            {
                "comparison":
                    "C_mean_vs_R_mean",

                "pearson_log2CPM":
                    cr_mean_corr,

                "spearman_log2CPM":
                    float(
                        c_mean.corr(
                            r_mean,
                            method="spearman",
                        )
                    ),
            },
        ]
    )

    coherence.to_csv(
        OUT
        / "CFBE_reference_replicate_coherence.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Freeze primary + sensitivity reference profiles
    #
    # These are still gene-expression references only.
    # No iModulon projection is performed here.
    # ----------------------------------------------------------

    reference_profiles = pd.DataFrame(
        {
            "GeneID":
                log_cpm.index,

            "PRIMARY_CFBE_ALL4":
                log_cpm[
                    [
                        "C1",
                        "C2",
                        "R1",
                        "R2",
                    ]
                ].mean(
                    axis=1
                ).values,

            "SENSITIVITY_CFBE_C_PAIR":
                log_cpm[
                    [
                        "C1",
                        "C2",
                    ]
                ].mean(
                    axis=1
                ).values,

            "SENSITIVITY_CFBE_R_PAIR":
                log_cpm[
                    [
                        "R1",
                        "R2",
                    ]
                ].mean(
                    axis=1
                ).values,
        }
    )

    reference_profiles.to_csv(
        OUT
        / "FROZEN_CFBE_log2CPM_reference_profiles.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Formal checks
    # ----------------------------------------------------------

    checks = {
        "exactly_4_reference_runs_processed":
            len(
                summary
            )
            == 4,

        "all_4_raw_MD5_pass":
            bool(
                summary[
                    "raw_MD5_pass"
                ].all()
            ),

        "all_4_technical_processing_pass":
            bool(
                summary[
                    "technical_pass"
                ].all()
            ),

        "count_matrix_has_5567_genes":
            len(
                matrix
            )
            == 5567,

        "count_matrix_has_exact_C1_C2_R1_R2":
            set(
                matrix.columns
            )
            == {
                "GeneID",
                "C1",
                "C2",
                "R1",
                "R2",
            },

        "C_pair_Pearson_ge_prefrozen_0_90":
            c_corr
            >= MIN_PAIR_CORRELATION,

        "R_pair_Pearson_ge_prefrozen_0_90":
            r_corr
            >= MIN_PAIR_CORRELATION,

        "three_reference_profiles_frozen":
            set(
                reference_profiles.columns
            )
            == {
                "GeneID",
                "PRIMARY_CFBE_ALL4",
                "SENSITIVITY_CFBE_C_PAIR",
                "SENSITIVITY_CFBE_R_PAIR",
            },

        "no_CF_candidate_used":
            True,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04i checks")

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
        "PRJNA576508_CFBE_EXTERNAL_REFERENCE_EXPRESSION_FROZEN"
        if passed
        else "PRJNA576508_CFBE_EXTERNAL_REFERENCE_PROCESSING_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: all four CFBE samples were processed using "
        "the exact preprocessing/alignment/counting rules already "
        "validated prospectively in Audits 04f2 and 04g."
    )

    print()
    print(
        "The primary ALL4 reference and C/R pair sensitivity "
        "references are now frozen in gene-expression space."
    )

    print()
    print(
        "No iModulon projection, controlled-state residualisation, "
        "or comparison with the frozen Audit-03h CF-airway "
        "candidate has been performed."
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
        / "AUDIT_04I_STATUS.txt"
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
            "Failed Audit 04i checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
