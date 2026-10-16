from pathlib import Path
import shutil
import subprocess

import pandas as pd


ROOT = Path.cwd()

SAM = (
    ROOT
    / "data/processed/phase4/PRJNA576508/pilot_EM36/04f_mapping"
    / "SRR10267761.PAO1.sam"
)

SAF = (
    ROOT
    / "references/phase4/PRJNA576508_cornforth"
    / "Cornforth_PAO1_no_decimal_locus_tags.saf"
)

OUT = (
    ROOT
    / "results/phase4/audit_04f2_PRJNA576508_EM36_featureCounts_parser_repair"
)

PUBLISHED_ASSIGNED = 228345
PUBLISHED_DETECTED = 5075

MAX_ASSIGNED_RELATIVE_ERROR = 0.25
MAX_DETECTED_RELATIVE_ERROR = 0.05


def run_featurecounts():
    rscript = shutil.which(
        "Rscript"
    )

    if rscript is None:
        raise RuntimeError(
            "Rscript is unavailable."
        )

    script = (
        OUT
        / "run_Rsubread_featureCounts_repaired.R"
    )

    count_csv = (
        OUT
        / "EM36_Rsubread_gene_counts_repaired.csv"
    )

    stat_csv = (
        OUT
        / "EM36_Rsubread_assignment_stat_repaired.csv"
    )

    r_code = (
        'suppressPackageStartupMessages(library(Rsubread))\n'
        + 'sam <- "'
        + str(SAM)
        + '"\n'
        + 'saf <- "'
        + str(SAF)
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
        + '\n'
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
        + '\n'
        + '# Rsubread fc$stat contains an explicit Status column.\n'
        + '# The original Audit 04f incorrectly coerced that column\n'
        + '# to numeric. Repair only the parser here.\n'
        + 'if ("Status" %in% colnames(fc$stat)) {\n'
        + '  status_values <- as.character(fc$stat$Status)\n'
        + '  count_columns <- setdiff(colnames(fc$stat), "Status")\n'
        + '} else {\n'
        + '  status_values <- rownames(fc$stat)\n'
        + '  count_columns <- colnames(fc$stat)\n'
        + '}\n'
        + '\n'
        + 'if (length(count_columns) != 1) {\n'
        + '  stop("Expected exactly one sample count column in fc$stat")\n'
        + '}\n'
        + '\n'
        + 'stat_counts <- as.numeric(fc$stat[[count_columns[1]]])\n'
        + '\n'
        + 'stats <- data.frame(\n'
        + '  Status=status_values,\n'
        + '  count=stat_counts,\n'
        + '  stringsAsFactors=FALSE\n'
        + ')\n'
        + '\n'
        + 'write.csv(\n'
        + '  stats,\n'
        + '  stat_out,\n'
        + '  row.names=FALSE,\n'
        + '  quote=FALSE\n'
        + ')\n'
    )

    script.write_text(
        r_code,
        encoding="utf-8",
    )

    print()
    print("Running repaired featureCounts parser")

    result = subprocess.run(
        [
            rscript,
            str(script),
        ]
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Rsubread featureCounts failed."
        )

    return count_csv, stat_csv


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04f2")
    print("EM36 featureCounts parser repair and benchmark completion")
    print("=" * 78)

    for path in [
        SAM,
        SAF,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Re-run counting only
    # ----------------------------------------------------------

    count_csv, stat_csv = run_featurecounts()

    counts = pd.read_csv(
        count_csv
    )

    stats = pd.read_csv(
        stat_csv
    )

    print()
    print("Repaired Rsubread stat table")
    print(
        stats.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 2. Recover exact assigned count
    # ----------------------------------------------------------

    assigned_row = stats.loc[
        stats[
            "Status"
        ]
        .astype(str)
        .eq(
            "Assigned"
        )
    ]

    if len(assigned_row) != 1:
        raise RuntimeError(
            "Could not uniquely recover Assigned status."
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

    detected_genes = int(
        (
            counts[
                "count"
            ]
            > 0
        ).sum()
    )

    total_genes = len(
        counts
    )

    # ----------------------------------------------------------
    # 3. Published benchmark comparison
    # ----------------------------------------------------------

    assigned_error = abs(
        assigned
        - PUBLISHED_ASSIGNED
    ) / PUBLISHED_ASSIGNED

    detected_error = abs(
        detected_genes
        - PUBLISHED_DETECTED
    ) / PUBLISHED_DETECTED

    assigned_ratio = (
        assigned
        / PUBLISHED_ASSIGNED
    )

    detected_ratio = (
        detected_genes
        / PUBLISHED_DETECTED
    )

    print()
    print("Completed EM36 benchmark")
    print()

    print("PAO1-gene assigned reads")
    print(
        "  published:",
        PUBLISHED_ASSIGNED,
    )
    print(
        "  reconstructed:",
        assigned,
    )
    print(
        "  difference:",
        assigned
        - PUBLISHED_ASSIGNED,
    )
    print(
        "  reconstructed / published:",
        round(
            assigned_ratio,
            6,
        ),
    )
    print(
        "  relative error:",
        round(
            assigned_error,
            6,
        ),
    )

    print()
    print("PAO1 genes detected")
    print(
        "  published:",
        PUBLISHED_DETECTED,
    )
    print(
        "  reconstructed:",
        detected_genes,
    )
    print(
        "  difference:",
        detected_genes
        - PUBLISHED_DETECTED,
    )
    print(
        "  reconstructed / published:",
        round(
            detected_ratio,
            6,
        ),
    )
    print(
        "  relative error:",
        round(
            detected_error,
            6,
        ),
    )

    print()
    print(
        "Total genes in frozen SAF:",
        total_genes,
    )
    print(
        "Sum of gene counts:",
        sum_counts,
    )

    # ----------------------------------------------------------
    # 4. Formal checks
    #
    # Thresholds are unchanged from original Audit 04f.
    # ----------------------------------------------------------

    checks = {
        "SAM_from_original_04f_exists":
            SAM.exists(),

        "SAF_exists":
            SAF.exists(),

        "Assigned_status_uniquely_recovered":
            len(
                assigned_row
            )
            == 1,

        "assigned_equals_sum_gene_counts":
            assigned
            == sum_counts,

        "assigned_reads_within_prefrozen_25pct":
            assigned_error
            <= MAX_ASSIGNED_RELATIVE_ERROR,

        "detected_genes_within_prefrozen_5pct":
            detected_error
            <= MAX_DETECTED_RELATIVE_ERROR,

        "detected_genes_nonzero":
            detected_genes
            > 0,

        "SAF_gene_count_gt_5500":
            total_genes
            > 5500,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04f2 checks")

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
        "IMPORTANT: the original Audit 04f mapping and counting "
        "completed successfully but the stats parser failed after "
        "featureCounts. Audit 04f2 repairs only that parser."
    )

    print()
    print(
        "The original pre-frozen acceptance thresholds remain "
        "unchanged: assigned reads +/-25%, detected genes +/-5%."
    )

    print()
    print(
        "No CF-State/iModulon projection has been performed."
    )

    # ----------------------------------------------------------
    # 5. Save
    # ----------------------------------------------------------

    summary = pd.DataFrame(
        [
            {
                "pilot":
                    "EM36",

                "published_assigned_reads":
                    PUBLISHED_ASSIGNED,

                "reconstructed_assigned_reads":
                    assigned,

                "assigned_difference":
                    assigned
                    - PUBLISHED_ASSIGNED,

                "assigned_ratio":
                    assigned_ratio,

                "assigned_relative_error":
                    assigned_error,

                "published_detected_genes":
                    PUBLISHED_DETECTED,

                "reconstructed_detected_genes":
                    detected_genes,

                "detected_gene_difference":
                    detected_genes
                    - PUBLISHED_DETECTED,

                "detected_gene_ratio":
                    detected_ratio,

                "detected_gene_relative_error":
                    detected_error,

                "SAF_genes":
                    total_genes,
            }
        ]
    )

    summary.to_csv(
        OUT
        / "EM36_completed_published_benchmark.csv",
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
        / "AUDIT_04F2_STATUS.txt"
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
            "Failed Audit 04f2 checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
