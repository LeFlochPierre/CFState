from pathlib import Path
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

WORKBOOK = (
    ROOT
    / "data/raw/phase3/GSE191103"
    / "GSE191103_ProcessedDataMatrix_biofilm_lasR.xlsx"
)

BRIDGE = (
    ROOT
    / "results/phase3/audit_02c_PA14_old_locus_annotation_bridge"
    / "GSE191103_PA14_old_locus_gene_lengths.csv"
)

STRICT_MAP = (
    ROOT
    / "results/phase3/audit_01a_PA14_RBH_ambiguity_diagnostic"
    / "strict_unique_PAO1_PA14_map.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_02d_PA14_missing_count_mass"
)

SAMPLES = [
    "PA14_BF_1",
    "PA14_BF_2",
    "PA14_BF_3",
    "PA14_PL_1",
    "PA14_PL_2",
    "PA14_PL_3",
]


def classify_id(value):
    text = str(
        value
    ).strip()

    if re.fullmatch(
        r"PA14_\d+",
        text,
    ):
        return "PA14_old_locus"

    if re.fullmatch(
        r"PA\d{4}\.\d+",
        text,
    ):
        return "PAO1_like_decimal"

    if re.fullmatch(
        r"PA\d{4}-PA\d{4}",
        text,
    ):
        return "PAO1_like_range"

    if text.startswith(
        "PA14_"
    ):
        return "PA14_other"

    if text.startswith(
        "PA"
    ):
        return "PA_other"

    return "other"


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02d")
    print("GSE191103 missing PA14 count-mass diagnostic")
    print("=" * 78)

    for path in [
        WORKBOOK,
        BRIDGE,
        STRICT_MAP,
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
    # 1. Find worksheet containing WT samples + Locus_ID
    # ----------------------------------------------------------

    workbook = pd.ExcelFile(
        WORKBOOK
    )

    selected_sheet = None
    table = None

    for sheet in workbook.sheet_names:
        candidate = pd.read_excel(
            WORKBOOK,
            sheet_name=sheet,
        )

        columns = set(
            str(
                x
            ).strip()
            for x in candidate.columns
        )

        if (
            "Locus_ID" in columns
            and all(
                sample in columns
                for sample in SAMPLES
            )
        ):
            selected_sheet = sheet
            table = candidate
            break

    if table is None:
        raise RuntimeError(
            "Could not find worksheet containing Locus_ID "
            "and all six WT PA14 samples."
        )

    print()
    print(
        "Selected worksheet:",
        selected_sheet,
    )
    print(
        "  shape:",
        table.shape,
    )

    print()
    print("Workbook columns")
    for column in table.columns:
        print(
            " ",
            repr(
                column
            ),
        )

    # ----------------------------------------------------------
    # 2. Clean expression matrix
    # ----------------------------------------------------------

    table[
        "Locus_ID"
    ] = (
        table[
            "Locus_ID"
        ]
        .astype(str)
        .str.strip()
    )

    counts = table[
        SAMPLES
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    counts.index = table[
        "Locus_ID"
    ]

    finite_mask = np.isfinite(
        counts.to_numpy(
            dtype=float
        )
    ).all(
        axis=1
    )

    counts = counts.loc[
        finite_mask
    ].copy()

    # ----------------------------------------------------------
    # 3. Frozen bridge
    # ----------------------------------------------------------

    bridge = pd.read_csv(
        BRIDGE
    )

    bridge_ids = set(
        bridge[
            "old_locus_tag"
        ]
        .astype(str)
        .str.strip()
    )

    matched_mask = counts.index.isin(
        bridge_ids
    )

    matched = counts.loc[
        matched_mask
    ]

    unmatched = counts.loc[
        ~matched_mask
    ].copy()

    print()
    print("Bridge coverage")
    print(
        "  total expression genes:",
        counts.shape[
            0
        ],
    )
    print(
        "  matched genes:",
        matched.shape[
            0
        ],
    )
    print(
        "  unmatched genes:",
        unmatched.shape[
            0
        ],
    )

    # ----------------------------------------------------------
    # 4. Count-mass accounting
    # ----------------------------------------------------------

    total_counts = counts.sum(
        axis=0
    )

    matched_counts = matched.sum(
        axis=0
    )

    unmatched_counts = unmatched.sum(
        axis=0
    )

    fractions = pd.DataFrame(
        {
            "total_counts":
                total_counts,

            "matched_counts":
                matched_counts,

            "unmatched_counts":
                unmatched_counts,

            "matched_fraction":
                matched_counts
                / total_counts,

            "unmatched_fraction":
                unmatched_counts
                / total_counts,
        }
    )

    print()
    print("Count-mass accounting")
    print(
        fractions
        .round(
            4
        )
        .to_string()
    )

    fractions.to_csv(
        OUT
        / "count_mass_accounting.csv"
    )

    # ----------------------------------------------------------
    # 5. Rank missing genes
    # ----------------------------------------------------------

    unmatched_summary = unmatched.copy()

    unmatched_summary[
        "total_count"
    ] = unmatched.sum(
        axis=1
    )

    unmatched_summary[
        "mean_count"
    ] = unmatched.mean(
        axis=1
    )

    unmatched_summary[
        "max_count"
    ] = unmatched.max(
        axis=1
    )

    unmatched_summary[
        "id_class"
    ] = [
        classify_id(
            gene
        )
        for gene in unmatched_summary.index
    ]

    unmatched_summary[
        "biofilm_total"
    ] = unmatched[
        [
            "PA14_BF_1",
            "PA14_BF_2",
            "PA14_BF_3",
        ]
    ].sum(
        axis=1
    )

    unmatched_summary[
        "planktonic_total"
    ] = unmatched[
        [
            "PA14_PL_1",
            "PA14_PL_2",
            "PA14_PL_3",
        ]
    ].sum(
        axis=1
    )

    unmatched_summary[
        "biofilm_minus_planktonic_total"
    ] = (
        unmatched_summary[
            "biofilm_total"
        ]
        - unmatched_summary[
            "planktonic_total"
        ]
    )

    # ----------------------------------------------------------
    # 6. Bring through workbook annotation columns
    # ----------------------------------------------------------

    metadata_columns = [
        column
        for column in table.columns
        if column not in SAMPLES
    ]

    metadata = table[
        metadata_columns
    ].copy()

    metadata = metadata.loc[
        finite_mask
    ]

    metadata = metadata.set_index(
        "Locus_ID"
    )

    metadata.index = (
        metadata.index
        .astype(str)
        .str.strip()
    )

    extra_metadata_columns = [
        column
        for column in metadata.columns
        if column not in unmatched_summary.columns
    ]

    unmatched_summary = unmatched_summary.join(
        metadata[
            extra_metadata_columns
        ],
        how="left",
    )

    unmatched_summary = unmatched_summary.sort_values(
        "total_count",
        ascending=False,
    )

    unmatched_summary.to_csv(
        OUT
        / "unmatched_genes_ranked_by_count.csv"
    )

    print()
    print("Highest-count unmatched genes")
    display_columns = [
        "total_count",
        "biofilm_total",
        "planktonic_total",
        "id_class",
    ]

    annotation_candidates = [
        column
        for column in extra_metadata_columns
        if any(
            token in str(
                column
            ).lower()
            for token in [
                "gene",
                "name",
                "description",
                "product",
                "annotation",
            ]
        )
    ]

    display_columns.extend(
        annotation_candidates[
            :4
        ]
    )

    print(
        unmatched_summary[
            display_columns
        ]
        .head(
            40
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 7. How concentrated is the missing count mass?
    # ----------------------------------------------------------

    total_unmatched_mass = float(
        unmatched_summary[
            "total_count"
        ].sum()
    )

    cumulative = (
        unmatched_summary[
            "total_count"
        ].cumsum()
    )

    if total_unmatched_mass > 0:
        cumulative_fraction = (
            cumulative
            / total_unmatched_mass
        )
    else:
        cumulative_fraction = (
            cumulative
            * np.nan
        )

    unmatched_summary[
        "cumulative_unmatched_mass_fraction"
    ] = cumulative_fraction

    def genes_for_fraction(
        target,
    ):
        if total_unmatched_mass <= 0:
            return 0

        values = cumulative_fraction.to_numpy()

        return int(
            np.searchsorted(
                values,
                target,
                side="left",
            )
            + 1
        )

    n50 = genes_for_fraction(
        0.50
    )

    n80 = genes_for_fraction(
        0.80
    )

    n90 = genes_for_fraction(
        0.90
    )

    print()
    print("Concentration of unmatched count mass")
    print(
        "  genes accounting for 50%:",
        n50,
    )
    print(
        "  genes accounting for 80%:",
        n80,
    )
    print(
        "  genes accounting for 90%:",
        n90,
    )

    # Save updated cumulative table.
    unmatched_summary.to_csv(
        OUT
        / "unmatched_genes_ranked_by_count.csv"
    )

    # ----------------------------------------------------------
    # 8. ID-class contribution
    # ----------------------------------------------------------

    class_summary = (
        unmatched_summary
        .groupby(
            "id_class"
        )
        .agg(
            n_genes=(
                "total_count",
                "size",
            ),
            total_count=(
                "total_count",
                "sum",
            ),
            biofilm_total=(
                "biofilm_total",
                "sum",
            ),
            planktonic_total=(
                "planktonic_total",
                "sum",
            ),
        )
        .sort_values(
            "total_count",
            ascending=False,
        )
    )

    class_summary[
        "fraction_unmatched_count_mass"
    ] = (
        class_summary[
            "total_count"
        ]
        / total_unmatched_mass
    )

    class_summary.to_csv(
        OUT
        / "unmatched_count_mass_by_ID_class.csv"
    )

    print()
    print("Unmatched count mass by ID class")
    print(
        class_summary
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 9. Which strict PA14 orthologues are missing lengths?
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

    strict[
        "pao1_gene"
    ] = (
        strict[
            "pao1_gene"
        ]
        .astype(str)
        .str.strip()
    )

    missing_strict = strict.loc[
        ~strict[
            "pa14_gene"
        ].isin(
            bridge_ids
        )
    ].copy()

    missing_strict[
        "present_in_expression"
    ] = missing_strict[
        "pa14_gene"
    ].isin(
        counts.index
    )

    missing_strict.to_csv(
        OUT
        / "strict_RBH_genes_missing_verified_lengths.csv",
        index=False,
    )

    print()
    print("Strict orthologues lacking verified old-locus lengths")
    print(
        "  total:",
        len(
            missing_strict
        ),
    )
    print(
        "  present in expression matrix:",
        int(
            missing_strict[
                "present_in_expression"
            ].sum()
        ),
    )

    if not missing_strict.empty:
        print()
        print(
            missing_strict[
                [
                    "pao1_gene",
                    "pa14_gene",
                    "present_in_expression",
                ]
            ]
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 10. Diagnostic status
    # ----------------------------------------------------------

    largest_gene_fraction = (
        float(
            unmatched_summary[
                "total_count"
            ].iloc[
                0
            ]
            / total_unmatched_mass
        )
        if total_unmatched_mass > 0
        else 0.0
    )

    print()
    print("Diagnostic metrics")
    print(
        "  unmatched genes:",
        len(
            unmatched_summary
        ),
    )
    print(
        "  largest single gene fraction of unmatched mass:",
        format(
            largest_gene_fraction,
            ".2%",
        ),
    )
    print(
        "  genes required for 80% of unmatched mass:",
        n80,
    )

    if n80 <= 20:
        status = (
            "MISSING_MASS_HIGHLY_CONCENTRATED"
        )

    elif n80 <= 60:
        status = (
            "MISSING_MASS_MODERATELY_CONCENTRATED"
        )

    else:
        status = (
            "MISSING_MASS_BROADLY_DISTRIBUTED"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_02D_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
