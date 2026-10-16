from pathlib import Path
from collections import defaultdict
import re

import numpy as np
import pandas as pd
from Bio import SeqIO


ROOT = Path.cwd()

COUNTS = (
    ROOT
    / "results/phase3/audit_03a_GSE55197_stationary_intake"
    / "GSE55197_stationary_exponential_reads_per_gene.csv"
)

PROJECTION_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

PA14_GB = (
    ROOT
    / "references/mappings/NC_008463.1.gb"
)

OUT = (
    ROOT
    / "results/phase3/audit_03b_GSE55197_gene_ID_bridge"
)


def clean(value):
    return str(
        value
    ).strip()


def classify_id(value):
    text = clean(
        value
    )

    if re.fullmatch(
        r"PA14_\d+",
        text,
    ):
        return "PA14_numeric"

    if re.fullmatch(
        r"PA14_RS\d+",
        text,
    ):
        return "PA14_RS"

    if re.fullmatch(
        r"PA\d{4}",
        text,
    ):
        return "PAO1_locus"

    if re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_-]*",
        text,
    ):
        return "symbol_or_alias"

    return "other"


def padded_pa14(value):
    text = clean(
        value
    )

    match = re.fullmatch(
        r"PA14_(\d+)",
        text,
    )

    if match is None:
        return None

    number = match.group(
        1
    )

    if len(
        number
    ) >= 5:
        return text

    return (
        "PA14_"
        + number.zfill(
            5
        )
    )


def add_candidate(
    candidate_dict,
    observed,
    pao1_gene,
    pa14_gene,
    current_locus_tag,
    method,
):
    key = (
        clean(
            pao1_gene
        ),
        clean(
            pa14_gene
        ),
        clean(
            current_locus_tag
        ),
    )

    candidate_dict[
        observed
    ][
        key
    ].add(
        method
    )


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 03b")
    print("GSE55197 historical/alias gene-ID bridge diagnostic")
    print("=" * 78)

    for path in [
        COUNTS,
        PROJECTION_MAP,
        PA14_GB,
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

    counts = pd.read_csv(
        COUNTS,
        index_col=0,
    )

    counts.index = (
        counts.index
        .astype(str)
        .str.strip()
    )

    counts = counts.apply(
        pd.to_numeric,
        errors="raise",
    )

    projection = pd.read_csv(
        PROJECTION_MAP
    )

    required = [
        "pao1_gene",
        "pa14_gene",
        "current_locus_tag",
    ]

    for column in required:
        if column not in projection.columns:
            raise RuntimeError(
                "Projection map missing column: "
                + column
            )

        projection[
            column
        ] = (
            projection[
                column
            ]
            .astype(str)
            .str.strip()
        )

    if len(
        projection
    ) != 5286:
        raise RuntimeError(
            "Expected frozen 5286-gene projection map."
        )

    # ----------------------------------------------------------
    # 1. Inspect observed identifier universe
    # ----------------------------------------------------------

    id_table = pd.DataFrame(
        {
            "observed_id":
                counts.index,
        }
    )

    id_table[
        "id_class"
    ] = [
        classify_id(
            value
        )
        for value
        in id_table[
            "observed_id"
        ]
    ]

    id_table[
        "total_count"
    ] = counts.sum(
        axis=1
    ).reindex(
        id_table[
            "observed_id"
        ]
    ).to_numpy()

    class_summary = (
        id_table
        .groupby(
            "id_class"
        )
        .agg(
            n_ids=(
                "observed_id",
                "size",
            ),
            total_count=(
                "total_count",
                "sum",
            ),
        )
        .sort_values(
            "total_count",
            ascending=False,
        )
    )

    class_summary[
        "fraction_total_count_mass"
    ] = (
        class_summary[
            "total_count"
        ]
        / class_summary[
            "total_count"
        ].sum()
    )

    print()
    print("Observed identifier classes")
    print(
        class_summary
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 2. Frozen projection lookup dictionaries
    # ----------------------------------------------------------

    old_lookup = {}

    current_lookup = {}

    pao1_lookup = {}

    for _, row in projection.iterrows():
        record = {
            "pao1_gene":
                row[
                    "pao1_gene"
                ],
            "pa14_gene":
                row[
                    "pa14_gene"
                ],
            "current_locus_tag":
                row[
                    "current_locus_tag"
                ],
        }

        old_lookup[
            row[
                "pa14_gene"
            ]
        ] = record

        current_lookup[
            row[
                "current_locus_tag"
            ]
        ] = record

        pao1_lookup[
            row[
                "pao1_gene"
            ]
        ] = record

    # ----------------------------------------------------------
    # 3. Current RefSeq annotation aliases
    # ----------------------------------------------------------

    record = SeqIO.read(
        str(
            PA14_GB
        ),
        "genbank",
    )

    current_to_symbols = defaultdict(
        set
    )

    symbol_to_current = defaultdict(
        set
    )

    old_to_current_annotation = defaultdict(
        set
    )

    for feature in record.features:
        current_values = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if not current_values:
            continue

        current = clean(
            current_values[
                0
            ]
        )

        old_values = feature.qualifiers.get(
            "old_locus_tag",
            [],
        )

        gene_values = feature.qualifiers.get(
            "gene",
            [],
        )

        for old in old_values:
            old = clean(
                old
            )

            old_to_current_annotation[
                old
            ].add(
                current
            )

        for symbol in gene_values:
            symbol = clean(
                symbol
            )

            if not symbol:
                continue

            current_to_symbols[
                current
            ].add(
                symbol
            )

            symbol_to_current[
                symbol
            ].add(
                current
            )

    unique_symbol_to_current = {
        symbol:
            next(
                iter(
                    values
                )
            )
        for symbol, values
        in symbol_to_current.items()
        if len(
            values
        )
        == 1
    }

    print()
    print("Current PA14 annotation aliases")
    print(
        "  unique gene symbols:",
        len(
            unique_symbol_to_current
        ),
    )
    print(
        "  old locus tags represented:",
        len(
            old_to_current_annotation
        ),
    )

    # ----------------------------------------------------------
    # 4. Candidate bridge
    #
    # Priority is NOT used to force a mapping. We gather every
    # independently valid route and accept an observed ID only
    # when all routes resolve to one frozen projection gene.
    # ----------------------------------------------------------

    candidates = defaultdict(
        lambda: defaultdict(
            set
        )
    )

    for observed in counts.index:
        # A. Exact historical PA14 locus tag.
        if observed in old_lookup:
            row = old_lookup[
                observed
            ]

            add_candidate(
                candidates,
                observed,
                row[
                    "pao1_gene"
                ],
                row[
                    "pa14_gene"
                ],
                row[
                    "current_locus_tag"
                ],
                "EXACT_OLD_PA14",
            )

        # B. Exact current RefSeq locus tag.
        if observed in current_lookup:
            row = current_lookup[
                observed
            ]

            add_candidate(
                candidates,
                observed,
                row[
                    "pao1_gene"
                ],
                row[
                    "pa14_gene"
                ],
                row[
                    "current_locus_tag"
                ],
                "EXACT_CURRENT_REFSEQ",
            )

        # C. PAO1 orthologue ID appears directly in processed file.
        if observed in pao1_lookup:
            row = pao1_lookup[
                observed
            ]

            add_candidate(
                candidates,
                observed,
                row[
                    "pao1_gene"
                ],
                row[
                    "pa14_gene"
                ],
                row[
                    "current_locus_tag"
                ],
                "EXACT_PAO1_ORTHOLOG",
            )

        # D. Explicit formatting normalization:
        #    PA14_0010 -> PA14_00010, etc.
        normalized = padded_pa14(
            observed
        )

        if (
            normalized is not None
            and normalized != observed
            and normalized in old_lookup
        ):
            row = old_lookup[
                normalized
            ]

            add_candidate(
                candidates,
                observed,
                row[
                    "pao1_gene"
                ],
                row[
                    "pa14_gene"
                ],
                row[
                    "current_locus_tag"
                ],
                "PA14_ZERO_PAD",
            )

        # E. Unambiguous current RefSeq gene symbol.
        if observed in unique_symbol_to_current:
            current = unique_symbol_to_current[
                observed
            ]

            if current in current_lookup:
                row = current_lookup[
                    current
                ]

                add_candidate(
                    candidates,
                    observed,
                    row[
                        "pao1_gene"
                    ],
                    row[
                        "pa14_gene"
                    ],
                    row[
                        "current_locus_tag"
                    ],
                    "UNIQUE_GENE_SYMBOL",
                )

    # ----------------------------------------------------------
    # 5. Accept only uniquely resolved observed IDs
    # ----------------------------------------------------------

    bridge_rows = []

    ambiguous_rows = []

    unmatched_rows = []

    for observed in counts.index:
        options = candidates.get(
            observed,
            {},
        )

        if len(
            options
        ) == 0:
            unmatched_rows.append(
                {
                    "observed_id":
                        observed,

                    "id_class":
                        classify_id(
                            observed
                        ),

                    "total_count":
                        float(
                            counts.loc[
                                observed
                            ].sum()
                        ),
                }
            )

            continue

        if len(
            options
        ) > 1:
            for key, methods in options.items():
                ambiguous_rows.append(
                    {
                        "observed_id":
                            observed,

                        "pao1_gene":
                            key[
                                0
                            ],

                        "pa14_gene":
                            key[
                                1
                            ],

                        "current_locus_tag":
                            key[
                                2
                            ],

                        "methods":
                            ";".join(
                                sorted(
                                    methods
                                )
                            ),

                        "total_count":
                            float(
                                counts.loc[
                                    observed
                                ].sum()
                            ),
                    }
                )

            continue

        key = next(
            iter(
                options
            )
        )

        methods = options[
            key
        ]

        bridge_rows.append(
            {
                "observed_id":
                    observed,

                "pao1_gene":
                    key[
                        0
                    ],

                "pa14_gene":
                    key[
                        1
                    ],

                "current_locus_tag":
                    key[
                        2
                    ],

                "methods":
                    ";".join(
                        sorted(
                            methods
                        )
                    ),

                "total_count":
                    float(
                        counts.loc[
                            observed
                        ].sum()
                    ),
            }
        )

    bridge = pd.DataFrame(
        bridge_rows
    )

    ambiguous = pd.DataFrame(
        ambiguous_rows
    )

    unmatched = pd.DataFrame(
        unmatched_rows
    )

    # ----------------------------------------------------------
    # 6. Detect many-observed-ID -> one projection-gene collisions
    # ----------------------------------------------------------

    if not bridge.empty:
        target_counts = (
            bridge[
                "pao1_gene"
            ]
            .value_counts()
        )

        collision_targets = set(
            target_counts[
                target_counts
                > 1
            ].index
        )

        collisions = bridge.loc[
            bridge[
                "pao1_gene"
            ].isin(
                collision_targets
            )
        ].copy()

        clean_bridge = bridge.loc[
            ~bridge[
                "pao1_gene"
            ].isin(
                collision_targets
            )
        ].copy()

    else:
        collisions = pd.DataFrame()
        clean_bridge = bridge.copy()

    # ----------------------------------------------------------
    # 7. Coverage and count-mass accounting
    # ----------------------------------------------------------

    mapped_ids = set(
        clean_bridge[
            "observed_id"
        ]
    )

    mapped_projection_genes = set(
        clean_bridge[
            "pao1_gene"
        ]
    )

    gene_coverage = (
        len(
            mapped_projection_genes
        )
        / 5286.0
    )

    mapped_mass = counts.loc[
        counts.index.isin(
            mapped_ids
        )
    ].sum(
        axis=0
    )

    total_mass = counts.sum(
        axis=0
    )

    count_mass_fraction = (
        mapped_mass
        / total_mass
    )

    min_count_mass = float(
        count_mass_fraction.min()
    )

    print()
    print("Resolved frozen projection bridge")
    print(
        "  observed IDs resolved uniquely:",
        len(
            clean_bridge
        ),
    )
    print(
        "  frozen projection genes represented:",
        len(
            mapped_projection_genes
        ),
        "/ 5286",
    )
    print(
        "  frozen projection gene coverage:",
        format(
            gene_coverage,
            ".2%",
        ),
    )
    print(
        "  ambiguous observed IDs:",
        len(
            set(
                ambiguous[
                    "observed_id"
                ]
            )
        )
        if not ambiguous.empty
        else 0,
    )
    print(
        "  collision target genes:",
        len(
            set(
                collisions[
                    "pao1_gene"
                ]
            )
        )
        if not collisions.empty
        else 0,
    )
    print(
        "  unmatched observed IDs:",
        len(
            unmatched
        ),
    )

    print()
    print("Count mass represented by clean bridge")
    print(
        count_mass_fraction
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 8. Mapping-method contribution
    # ----------------------------------------------------------

    if not clean_bridge.empty:
        method_summary = (
            clean_bridge.assign(
                primary_method=clean_bridge[
                    "methods"
                ]
            )
            .groupby(
                "primary_method"
            )
            .agg(
                n_ids=(
                    "observed_id",
                    "size",
                ),
                total_count=(
                    "total_count",
                    "sum",
                ),
            )
            .sort_values(
                "n_ids",
                ascending=False,
            )
        )

        print()
        print("Resolved IDs by mapping route")
        print(
            method_summary
            .round(
                2
            )
            .to_string()
        )

    # ----------------------------------------------------------
    # 9. Largest still-unmatched IDs
    # ----------------------------------------------------------

    if not unmatched.empty:
        unmatched = unmatched.sort_values(
            "total_count",
            ascending=False,
        )

        print()
        print("Highest-count unmatched identifiers")
        print(
            unmatched
            .head(
                40
            )
            .to_string(
                index=False
            )
        )

    if not collisions.empty:
        print()
        print("Observed-ID collisions")
        print(
            collisions
            .sort_values(
                [
                    "pao1_gene",
                    "total_count",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 10. Save diagnostic outputs
    # ----------------------------------------------------------

    id_table.to_csv(
        OUT
        / "observed_identifier_classes.csv",
        index=False,
    )

    class_summary.to_csv(
        OUT
        / "identifier_class_summary.csv"
    )

    clean_bridge.to_csv(
        OUT
        / "GSE55197_unambiguous_projection_bridge.csv",
        index=False,
    )

    ambiguous.to_csv(
        OUT
        / "ambiguous_identifier_mappings.csv",
        index=False,
    )

    unmatched.to_csv(
        OUT
        / "unmatched_identifiers.csv",
        index=False,
    )

    collisions.to_csv(
        OUT
        / "projection_gene_collisions.csv",
        index=False,
    )

    count_mass_fraction.to_csv(
        OUT
        / "mapped_count_mass_fraction.csv",
        header=[
            "fraction"
        ],
    )

    # ----------------------------------------------------------
    # 11. Audit decision
    #
    # This is a namespace/alias bridge audit. We require almost
    # the entire frozen model universe and most expression mass.
    # No ambiguous mapping is ever guessed.
    # ----------------------------------------------------------

    n_ambiguous = (
        len(
            set(
                ambiguous[
                    "observed_id"
                ]
            )
        )
        if not ambiguous.empty
        else 0
    )

    n_collision_targets = (
        len(
            set(
                collisions[
                    "pao1_gene"
                ]
            )
        )
        if not collisions.empty
        else 0
    )

    checks = {
        "projection_genes_ge_5000":
            len(
                mapped_projection_genes
            )
            >= 5000,

        "projection_gene_coverage_ge_94pct":
            gene_coverage
            >= 0.94,

        "minimum_count_mass_ge_90pct":
            min_count_mass
            >= 0.90,

        "ambiguous_observed_IDs_le_10":
            n_ambiguous
            <= 10,

        "collision_targets_le_10":
            n_collision_targets
            <= 10,
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
        "GSE55197_IDENTIFIER_BRIDGE_ACCEPTED"
        if passed
        else "GSE55197_IDENTIFIER_BRIDGE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
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
        / "AUDIT_03B_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
