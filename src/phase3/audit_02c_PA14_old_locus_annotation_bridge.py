from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd


ROOT = Path.cwd()

PA14_GB = (
    ROOT
    / "references/mappings/NC_008463.1.gb"
)

EXPRESSION = (
    ROOT
    / "results/phase3/audit_02a_GSE191103_PA14_intake"
    / "GSE191103_WT_PA14_processed_expression.csv"
)

STRICT_MAP = (
    ROOT
    / "results/phase3/audit_01a_PA14_RBH_ambiguity_diagnostic"
    / "strict_unique_PAO1_PA14_map.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_02c_PA14_old_locus_annotation_bridge"
)


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02c")
    print("PA14 old-locus-tag annotation bridge")
    print("=" * 78)

    for path in [
        PA14_GB,
        EXPRESSION,
        STRICT_MAP,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython is required."
        )

    # ----------------------------------------------------------
    # Expression IDs
    # ----------------------------------------------------------

    expression = pd.read_csv(
        EXPRESSION,
        index_col=0,
    )

    expression.index = (
        expression.index
        .astype(str)
        .str.strip()
    )

    expression_ids = set(
        expression.index
    )

    print()
    print("GSE191103 expression")
    print(
        "  genes:",
        expression.shape[
            0
        ],
    )
    print(
        "  samples:",
        expression.shape[
            1
        ],
    )

    print(
        "  PA14_ IDs:",
        sum(
            gene.startswith(
                "PA14_"
            )
            for gene in expression_ids
        ),
    )

    # ----------------------------------------------------------
    # Parse current + old PA14 annotation
    # ----------------------------------------------------------

    record = SeqIO.read(
        str(PA14_GB),
        "genbank",
    )

    rows = []

    old_to_rows = defaultdict(
        list
    )

    current_to_rows = defaultdict(
        list
    )

    priority = {
        "CDS": 10,
        "rRNA": 9,
        "tRNA": 9,
        "tmRNA": 9,
        "ncRNA": 9,
        "misc_RNA": 8,
        "gene": 1,
    }

    for feature in record.features:
        if feature.type not in priority:
            continue

        locus_tags = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if not locus_tags:
            continue

        current = str(
            locus_tags[
                0
            ]
        ).strip()

        old_tags = [
            str(
                value
            ).strip()
            for value
            in feature.qualifiers.get(
                "old_locus_tag",
                [],
            )
            if str(
                value
            ).strip()
        ]

        gene_values = feature.qualifiers.get(
            "gene",
            [],
        )

        gene_symbol = (
            str(
                gene_values[
                    0
                ]
            ).strip()
            if gene_values
            else ""
        )

        length_bp = int(
            len(
                feature.extract(
                    record.seq
                )
            )
        )

        row = {
            "current_locus_tag":
                current,

            "old_locus_tags":
                ";".join(
                    old_tags
                ),

            "feature_type":
                feature.type,

            "gene_symbol":
                gene_symbol,

            "length_bp":
                length_bp,

            "start":
                int(
                    feature.location.start
                )
                + 1,

            "end":
                int(
                    feature.location.end
                ),

            "strand":
                feature.location.strand,

            "priority":
                priority[
                    feature.type
                ],
        }

        rows.append(
            row
        )

        current_to_rows[
            current
        ].append(
            row
        )

        for old in old_tags:
            old_to_rows[
                old
            ].append(
                row
            )

    annotation = pd.DataFrame(
        rows
    )

    print()
    print("NC_008463.1 annotation")
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
        "  feature rows with locus tags:",
        len(
            annotation
        ),
    )
    print(
        "  distinct current locus tags:",
        annotation[
            "current_locus_tag"
        ].nunique(),
    )
    print(
        "  distinct old locus tags:",
        len(
            old_to_rows
        ),
    )

    # ----------------------------------------------------------
    # Resolve one best feature per old PA14 locus tag
    # ----------------------------------------------------------

    bridge_rows = []

    ambiguous_old_tags = []

    for old_tag, candidates in old_to_rows.items():
        candidates = sorted(
            candidates,
            key=lambda row: row[
                "priority"
            ],
            reverse=True,
        )

        best_priority = candidates[
            0
        ][
            "priority"
        ]

        best = [
            row
            for row in candidates
            if row[
                "priority"
            ]
            == best_priority
        ]

        unique_current = sorted(
            set(
                row[
                    "current_locus_tag"
                ]
                for row in best
            )
        )

        unique_lengths = sorted(
            set(
                row[
                    "length_bp"
                ]
                for row in best
            )
        )

        if (
            len(
                unique_current
            )
            != 1
            or len(
                unique_lengths
            )
            != 1
        ):
            ambiguous_old_tags.append(
                {
                    "old_locus_tag":
                        old_tag,

                    "candidate_current_loci":
                        ";".join(
                            unique_current
                        ),

                    "candidate_lengths":
                        ";".join(
                            str(
                                x
                            )
                            for x
                            in unique_lengths
                        ),
                }
            )
            continue

        selected = best[
            0
        ]

        bridge_rows.append(
            {
                "old_locus_tag":
                    old_tag,

                "current_locus_tag":
                    selected[
                        "current_locus_tag"
                    ],

                "feature_type":
                    selected[
                        "feature_type"
                    ],

                "gene_symbol":
                    selected[
                        "gene_symbol"
                    ],

                "length_bp":
                    selected[
                        "length_bp"
                    ],

                "start":
                    selected[
                        "start"
                    ],

                "end":
                    selected[
                        "end"
                    ],

                "strand":
                    selected[
                        "strand"
                    ],
            }
        )

    bridge = pd.DataFrame(
        bridge_rows
    )

    ambiguous = pd.DataFrame(
        ambiguous_old_tags
    )

    bridge.to_csv(
        OUT
        / "PA14_old_to_current_locus_bridge.csv",
        index=False,
    )

    ambiguous.to_csv(
        OUT
        / "ambiguous_old_locus_tags.csv",
        index=False,
    )

    bridge_old_set = set(
        bridge[
            "old_locus_tag"
        ]
    )

    # ----------------------------------------------------------
    # Expression coverage through old_locus_tag
    # ----------------------------------------------------------

    shared_expression = (
        expression_ids
        & bridge_old_set
    )

    expression_coverage = (
        len(
            shared_expression
        )
        / len(
            expression_ids
        )
    )

    print()
    print("Old-locus annotation bridge")
    print(
        "  unambiguous old locus tags:",
        len(
            bridge
        ),
    )
    print(
        "  ambiguous old locus tags:",
        len(
            ambiguous
        ),
    )
    print(
        "  expression genes matched:",
        len(
            shared_expression
        ),
    )
    print(
        "  expression gene coverage:",
        format(
            expression_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # Strict RBH coverage
    # ----------------------------------------------------------

    strict = pd.read_csv(
        STRICT_MAP
    )

    strict[
        "pa14_gene"
    ] = (
        strict[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    strict_pa14 = set(
        strict[
            "pa14_gene"
        ]
    )

    strict_bridge = (
        strict_pa14
        & bridge_old_set
    )

    strict_bridge_coverage = (
        len(
            strict_bridge
        )
        / len(
            strict_pa14
        )
    )

    print()
    print("Strict PAO1<->PA14 RBH compatibility")
    print(
        "  strict PA14 RBH genes:",
        len(
            strict_pa14
        ),
    )
    print(
        "  strict PA14 genes with old-locus lengths:",
        len(
            strict_bridge
        ),
    )
    print(
        "  strict RBH length coverage:",
        format(
            strict_bridge_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # Create expression-compatible length table
    # ----------------------------------------------------------

    length_table = (
        bridge.loc[
            bridge[
                "old_locus_tag"
            ].isin(
                expression_ids
            ),
            [
                "old_locus_tag",
                "current_locus_tag",
                "feature_type",
                "gene_symbol",
                "length_bp",
                "start",
                "end",
                "strand",
            ],
        ]
        .copy()
    )

    length_table = length_table.sort_values(
        "old_locus_tag"
    )

    length_table.to_csv(
        OUT
        / "GSE191103_PA14_old_locus_gene_lengths.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Inspect unmatched expression IDs
    # ----------------------------------------------------------

    unmatched = sorted(
        expression_ids
        - bridge_old_set
    )

    print()
    print("First unmatched expression IDs")

    if unmatched:
        for gene in unmatched[
            :30
        ]:
            print(
                " ",
                gene,
            )
    else:
        print(
            "  none"
        )

    pd.DataFrame(
        {
            "unmatched_expression_gene":
                unmatched
        }
    ).to_csv(
        OUT
        / "unmatched_expression_ids.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Audit gates
    # ----------------------------------------------------------

    checks = {
        "old_locus_tags_present":
            len(
                bridge
            )
            > 5000,

        "expression_old_locus_coverage_ge_95pct":
            expression_coverage
            >= 0.95,

        "strict_RBH_length_coverage_ge_95pct":
            strict_bridge_coverage
            >= 0.95,

        "ambiguous_old_tags_lt_1pct":
            (
                len(
                    ambiguous
                )
                / max(
                    len(
                        bridge
                    )
                    + len(
                        ambiguous
                    ),
                    1,
                )
            )
            < 0.01,

        "all_lengths_positive":
            bool(
                (
                    length_table[
                        "length_bp"
                    ]
                    > 0
                ).all()
            ),
    }

    passed = all(
        checks.values()
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

    status = (
        "OLD_LOCUS_BRIDGE_ACCEPTED"
        if passed
        else "ANNOTATION_BRIDGE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_02C_STATUS.txt"
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
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
