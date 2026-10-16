from pathlib import Path
import subprocess

import numpy as np
import pandas as pd


ROOT = Path.cwd()

AUDIT12D = (
    ROOT
    / "results/phase2/audit_12d_GSE299861_canonical_requantification"
)

WORK12D = (
    ROOT
    / "data/processed/phase2/GSE299861_canonical_requantification"
)

GENBANK = (
    ROOT
    / "references/mappings/NC_002516.2.gb"
)

OUT = (
    ROOT
    / "results/phase2/audit_12e_GSE299861_assignment_diagnostic"
)

THREADS = 8


def run(command):
    print()
    print("$", command)

    subprocess.run(
        [
            "bash",
            "-c",
            "set -o pipefail; " + command,
        ],
        check=True,
    )


def build_all_feature_saf():
    from Bio import SeqIO

    record = SeqIO.read(
        str(GENBANK),
        "genbank",
    )

    # One counting interval per unique locus tag.
    # Prefer biologically specific feature types over generic "gene".
    priority = {
        "CDS": 10,
        "rRNA": 9,
        "tRNA": 9,
        "tmRNA": 9,
        "ncRNA": 9,
        "misc_RNA": 8,
        "gene": 1,
    }

    best = {}

    for feature in record.features:
        if feature.type not in priority:
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

        current = best.get(
            locus
        )

        rank = priority[
            feature.type
        ]

        if (
            current is not None
            and current["priority"] >= rank
        ):
            continue

        strand = (
            "+"
            if feature.location.strand == 1
            else "-"
        )

        best[
            locus
        ] = {
            "GeneID":
                locus,

            "Chr":
                record.id,

            "Start":
                int(
                    feature.location.start
                )
                + 1,

            "End":
                int(
                    feature.location.end
                ),

            "Strand":
                strand,

            "feature_type":
                feature.type,

            "priority":
                rank,
        }

    rows = []

    for locus, item in best.items():
        rows.append(
            {
                "GeneID":
                    item[
                        "GeneID"
                    ],

                "Chr":
                    item[
                        "Chr"
                    ],

                "Start":
                    item[
                        "Start"
                    ],

                "End":
                    item[
                        "End"
                    ],

                "Strand":
                    item[
                        "Strand"
                    ],
            }
        )

    saf = pd.DataFrame(
        rows
    )

    path = (
        OUT
        / "NC_002516.2_all_annotated_loci.saf"
    )

    saf.to_csv(
        path,
        sep="\t",
        index=False,
    )

    type_counts = pd.Series(
        [
            item[
                "feature_type"
            ]
            for item in best.values()
        ]
    ).value_counts()

    print()
    print("Broad canonical annotation")
    print(
        "  unique locus tags:",
        len(
            saf
        ),
    )

    print()
    print("Selected feature types")
    print(
        type_counts.to_string()
    )

    return path


def load_summary(path):
    table = pd.read_csv(
        path,
        sep="\t",
        index_col=0,
    )

    return table


def assignment_fraction(summary):
    assigned = summary.loc[
        "Assigned"
    ].astype(
        float
    )

    total = summary.sum(
        axis=0
    ).astype(
        float
    )

    return (
        assigned
        / total
    )


def sample_name(column, sample_qc):
    text = str(
        column
    )

    for _, row in sample_qc.iterrows():
        srr = str(
            row[
                "srr"
            ]
        )

        if srr in text:
            return (
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

    return Path(
        text
    ).name


def run_featurecounts(
    label,
    annotation,
    bams,
    strand,
    strict,
):
    output = (
        OUT
        / (
            label
            + ".txt"
        )
    )

    options = (
        "-p --countReadPairs "
    )

    if strict:
        options += (
            "-B -C "
        )

    command = (
        "featureCounts "
        "-T "
        + str(
            THREADS
        )
        + " "
        "-F SAF "
        "-a "
        + str(
            annotation
        )
        + " "
        "-o "
        + str(
            output
        )
        + " "
        + options
        + "-s "
        + str(
            strand
        )
        + " "
        + " ".join(
            str(
                bam
            )
            for bam in bams
        )
    )

    run(
        command
    )

    summary = load_summary(
        Path(
            str(
                output
            )
            + ".summary"
        )
    )

    return summary


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 12e")
    print("GSE299861 featureCounts assignment diagnostic")
    print("=" * 78)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    sample_qc_path = (
        AUDIT12D
        / "sample_qc.csv"
    )

    requant_path = (
        AUDIT12D
        / "requantification_summary.csv"
    )

    cds_saf = (
        WORK12D
        / "reference"
        / "NC_002516.2_model_CDS.saf"
    )

    original_summary_path = (
        AUDIT12D
        / "GSE299861_featureCounts.txt.summary"
    )

    for path in [
        sample_qc_path,
        requant_path,
        cds_saf,
        original_summary_path,
        GENBANK,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    sample_qc = pd.read_csv(
        sample_qc_path
    )

    requant = pd.read_csv(
        requant_path
    )

    strand = int(
        requant.loc[
            0,
            "featureCounts_strand_mode",
        ]
    )

    bams = [
        Path(
            value
        )
        for value in sample_qc[
            "bam"
        ]
    ]

    for bam in bams:
        if not bam.exists():
            raise FileNotFoundError(
                str(
                    bam
                )
            )

    print()
    print(
        "Frozen strand mode:",
        strand,
    )

    print(
        "BAMs:",
        len(
            bams
        ),
    )

    # ----------------------------------------------------------
    # Existing CDS-only strict count
    # ----------------------------------------------------------

    cds_strict = load_summary(
        original_summary_path
    )

    # ----------------------------------------------------------
    # Create broader annotation
    # ----------------------------------------------------------

    all_saf = build_all_feature_saf()

    # ----------------------------------------------------------
    # Three diagnostic alternatives
    # ----------------------------------------------------------

    cds_relaxed = run_featurecounts(
        label="CDS_relaxed",
        annotation=cds_saf,
        bams=bams,
        strand=strand,
        strict=False,
    )

    all_strict = run_featurecounts(
        label="all_loci_strict",
        annotation=all_saf,
        bams=bams,
        strand=strand,
        strict=True,
    )

    all_relaxed = run_featurecounts(
        label="all_loci_relaxed",
        annotation=all_saf,
        bams=bams,
        strand=strand,
        strict=False,
    )

    # ----------------------------------------------------------
    # Standardize column names
    # ----------------------------------------------------------

    summaries = {
        "CDS_strict":
            cds_strict,

        "CDS_relaxed":
            cds_relaxed,

        "all_loci_strict":
            all_strict,

        "all_loci_relaxed":
            all_relaxed,
    }

    fractions = {}

    for label, summary in summaries.items():
        values = assignment_fraction(
            summary
        )

        renamed = {}

        for column, value in values.items():
            name = sample_name(
                column,
                sample_qc,
            )

            renamed[
                name
            ] = float(
                value
            )

        fractions[
            label
        ] = renamed

    comparison = pd.DataFrame(
        fractions
    )

    comparison.index.name = (
        "sample"
    )

    comparison[
        "gain_relax_pair_filters"
    ] = (
        comparison[
            "CDS_relaxed"
        ]
        - comparison[
            "CDS_strict"
        ]
    )

    comparison[
        "gain_broader_annotation_strict"
    ] = (
        comparison[
            "all_loci_strict"
        ]
        - comparison[
            "CDS_strict"
        ]
    )

    comparison[
        "gain_both"
    ] = (
        comparison[
            "all_loci_relaxed"
        ]
        - comparison[
            "CDS_strict"
        ]
    )

    comparison.to_csv(
        OUT
        / "assignment_fraction_comparison.csv"
    )

    print()
    print("Assignment fractions")
    print(
        comparison
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # Detailed original failure categories
    # ----------------------------------------------------------

    detail_rows = []

    original = cds_strict.copy()

    for column in original.columns:
        name = sample_name(
            column,
            sample_qc,
        )

        total = float(
            original[
                column
            ].sum()
        )

        for category in original.index:
            count = float(
                original.loc[
                    category,
                    column,
                ]
            )

            detail_rows.append(
                {
                    "sample":
                        name,

                    "category":
                        category,

                    "count":
                        int(
                            count
                        ),

                    "fraction":
                        (
                            count
                            / total
                            if total > 0
                            else np.nan
                        ),
                }
            )

    details = pd.DataFrame(
        detail_rows
    )

    details.to_csv(
        OUT
        / "CDS_strict_assignment_categories.csv",
        index=False,
    )

    print()
    print(
        "Largest original featureCounts categories "
        "per sample"
    )

    for sample in comparison.index:
        subset = (
            details.loc[
                details[
                    "sample"
                ]
                == sample
            ]
            .sort_values(
                "fraction",
                ascending=False,
            )
            .head(
                6
            )
        )

        print()
        print(
            sample
        )

        print(
            subset[
                [
                    "category",
                    "count",
                    "fraction",
                ]
            ]
            .round(
                {
                    "fraction":
                        4,
                }
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # Diagnose dominant cause
    # ----------------------------------------------------------

    mean_pair_gain = float(
        comparison[
            "gain_relax_pair_filters"
        ].mean()
    )

    mean_annotation_gain = float(
        comparison[
            "gain_broader_annotation_strict"
        ].mean()
    )

    min_all_strict = float(
        comparison[
            "all_loci_strict"
        ].min()
    )

    min_cds_relaxed = float(
        comparison[
            "CDS_relaxed"
        ].min()
    )

    print()
    print("Diagnostic summary")
    print(
        "  mean gain from relaxing -B/-C:",
        round(
            mean_pair_gain,
            4,
        ),
    )

    print(
        "  mean gain from broad annotation:",
        round(
            mean_annotation_gain,
            4,
        ),
    )

    print(
        "  minimum CDS-relaxed assignment:",
        round(
            min_cds_relaxed,
            4,
        ),
    )

    print(
        "  minimum all-loci strict assignment:",
        round(
            min_all_strict,
            4,
        ),
    )

    if (
        mean_annotation_gain
        > mean_pair_gain
        and mean_annotation_gain
        >= 0.10
    ):
        diagnosis = (
            "ANNOTATION_EXCLUSION_DOMINANT"
        )

    elif (
        mean_pair_gain
        > mean_annotation_gain
        and mean_pair_gain
        >= 0.10
    ):
        diagnosis = (
            "PAIR_FILTERS_DOMINANT"
        )

    elif (
        mean_annotation_gain
        >= 0.05
        or mean_pair_gain
        >= 0.05
    ):
        diagnosis = (
            "MIXED_TECHNICAL_CAUSES"
        )

    else:
        diagnosis = (
            "LOW_ASSIGNMENT_NOT_EXPLAINED"
        )

    print()
    print(
        "DIAGNOSIS:",
        diagnosis,
    )

    (
        OUT
        / "AUDIT_12E_DIAGNOSIS.txt"
    ).write_text(
        diagnosis
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
