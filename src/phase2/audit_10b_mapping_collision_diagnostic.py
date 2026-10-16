from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

RAW = (
    ROOT
    / "data/raw/phase2/GSE152295/GSE152295_PSEUDO_processed.txt.gz"
)

MAPPING = (
    ROOT
    / "results/phase2/audit_10_multistress_imodulon_dictionary"
    / "gene_mapping.csv"
)

OUT = (
    ROOT
    / "results/phase2/audit_10b_mapping_collision_diagnostic"
)


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 10b")
    print("GSE152295 mapping-collision diagnostic")
    print("=" * 78)

    for path in [RAW, MAPPING]:
        if not path.exists():
            raise FileNotFoundError(str(path))

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = pd.read_csv(
        RAW,
        sep="\t",
        compression="gzip",
    )

    mapping = pd.read_csv(
        MAPPING,
    )

    if len(raw) != len(mapping):
        raise RuntimeError(
            "Raw and mapping row counts differ: "
            + str(len(raw))
            + " vs "
            + str(len(mapping))
        )

    expression_cols = [
        col
        for col in raw.columns
        if str(col).endswith(" - TPM")
    ]

    mapping = mapping.copy()
    mapping["source_row"] = np.arange(len(mapping))

    mapped = mapping.loc[
        mapping["mapped_locus"].notna()
    ].copy()

    duplicate_mask = mapped[
        "mapped_locus"
    ].duplicated(
        keep=False
    )

    collision_rows = mapped.loc[
        duplicate_mask
    ].copy()

    collision_loci = sorted(
        collision_rows[
            "mapped_locus"
        ].unique()
    )

    print()
    print("Mapping overview")
    print(
        "  source rows:",
        len(mapping),
    )
    print(
        "  mapped rows:",
        len(mapped),
    )
    print(
        "  unmapped rows:",
        int(
            mapping[
                "mapped_locus"
            ].isna().sum()
        ),
    )
    print(
        "  duplicate/collision rows:",
        len(collision_rows),
    )
    print(
        "  affected canonical loci:",
        len(collision_loci),
    )

    # ------------------------------------------------------------
    # Characterise each collision group
    # ------------------------------------------------------------

    summary_rows = []
    detail_rows = []

    for locus in collision_loci:
        group = collision_rows.loc[
            collision_rows[
                "mapped_locus"
            ]
            == locus
        ].copy()

        rows = group[
            "source_row"
        ].astype(int).tolist()

        names = group[
            "Name"
        ].fillna("").astype(str).tolist()

        methods = group[
            "mapping_method"
        ].fillna("").astype(str).tolist()

        regions = group[
            "Region"
        ].fillna("").astype(str).tolist()

        chromosomes = group[
            "Chromosome"
        ].fillna("").astype(str).tolist()

        unique_regions = sorted(
            set(regions)
        )

        unique_chromosomes = sorted(
            set(chromosomes)
        )

        expression_vectors = [
            raw.loc[
                row,
                expression_cols,
            ].astype(float).to_numpy()
            for row in rows
        ]

        correlations = []
        max_differences = []

        for i in range(len(expression_vectors)):
            for j in range(i + 1, len(expression_vectors)):
                x = expression_vectors[i]
                y = expression_vectors[j]

                if (
                    np.std(x) > 0
                    and np.std(y) > 0
                ):
                    correlations.append(
                        float(
                            np.corrcoef(
                                x,
                                y,
                            )[0, 1]
                        )
                    )

                max_differences.append(
                    float(
                        np.max(
                            np.abs(
                                x - y
                            )
                        )
                    )
                )

        min_corr = (
            min(correlations)
            if correlations
            else np.nan
        )

        max_abs_diff = (
            max(max_differences)
            if max_differences
            else np.nan
        )

        exact_expression_duplicate = bool(
            max_abs_diff == 0.0
        )

        same_region = (
            len(unique_regions) == 1
        )

        same_chromosome = (
            len(unique_chromosomes) == 1
        )

        method_pattern = ";".join(
            sorted(methods)
        )

        if same_region and exact_expression_duplicate:
            collision_type = (
                "EXACT_DUPLICATE_SAME_REGION"
            )

        elif same_region:
            collision_type = (
                "SAME_REGION_DIFFERENT_EXPRESSION"
            )

        else:
            collision_type = (
                "DIFFERENT_REGION_MAPPING_COLLISION"
            )

        summary_rows.append(
            {
                "mapped_locus": locus,
                "n_rows": len(group),
                "names": " | ".join(names),
                "methods": method_pattern,
                "regions": " | ".join(regions),
                "same_region": same_region,
                "same_chromosome": same_chromosome,
                "minimum_expression_correlation": min_corr,
                "maximum_absolute_TPM_difference": max_abs_diff,
                "exact_expression_duplicate": exact_expression_duplicate,
                "collision_type": collision_type,
            }
        )

        for _, row in group.iterrows():
            detail_rows.append(
                row.to_dict()
            )

    summary = pd.DataFrame(
        summary_rows
    )

    details = pd.DataFrame(
        detail_rows
    )

    summary.to_csv(
        OUT
        / "mapping_collision_summary.csv",
        index=False,
    )

    details.to_csv(
        OUT
        / "mapping_collision_rows.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Unmapped rows
    # ------------------------------------------------------------

    unmapped = mapping.loc[
        mapping[
            "mapped_locus"
        ].isna()
    ].copy()

    unmapped.to_csv(
        OUT
        / "unmapped_features.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Aggregate collision categories
    # ------------------------------------------------------------

    category = (
        summary[
            "collision_type"
        ]
        .value_counts()
        .rename_axis(
            "collision_type"
        )
        .reset_index(
            name="n_loci"
        )
    )

    category.to_csv(
        OUT
        / "collision_type_counts.csv",
        index=False,
    )

    print()
    print("Collision types")

    if category.empty:
        print("  none")
    else:
        print(
            category.to_string(
                index=False
            )
        )

    print()
    print("Collision examples")

    if summary.empty:
        print("  none")
    else:
        print(
            summary[
                [
                    "mapped_locus",
                    "n_rows",
                    "names",
                    "methods",
                    "regions",
                    "minimum_expression_correlation",
                    "maximum_absolute_TPM_difference",
                    "collision_type",
                ]
            ]
            .head(30)
            .round(
                {
                    "minimum_expression_correlation": 4,
                    "maximum_absolute_TPM_difference": 4,
                }
            )
            .to_string(
                index=False
            )
        )

    print()
    print("Five unmapped features")

    if unmapped.empty:
        print("  none")
    else:
        print(
            unmapped[
                [
                    "source_row",
                    "Name",
                    "Chromosome",
                    "Region",
                    "mapping_method",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------

    n_different_region = int(
        (
            summary[
                "collision_type"
            ]
            == "DIFFERENT_REGION_MAPPING_COLLISION"
        ).sum()
    )

    n_same_region_different_expression = int(
        (
            summary[
                "collision_type"
            ]
            == "SAME_REGION_DIFFERENT_EXPRESSION"
        ).sum()
    )

    print()
    print("Diagnostic decision")
    print(
        "  different-region collisions:",
        n_different_region,
    )
    print(
        "  same-region but different-expression collisions:",
        n_same_region_different_expression,
    )

    if (
        n_different_region == 0
        and n_same_region_different_expression == 0
    ):
        status = (
            "SAFE_EXACT_DUPLICATES"
        )

        print(
            "  STATUS:",
            status,
        )
        print(
            "  All collisions are exact duplicate rows from the same region."
        )
        print(
            "  We should keep one copy rather than SUM them."
        )

    else:
        status = (
            "REMAP_REQUIRED"
        )

        print(
            "  STATUS:",
            status,
        )
        print(
            "  At least one collision involves different regions "
            "or different expression vectors."
        )
        print(
            "  These rows should be resolved using genomic coordinates "
            "before the dictionary is frozen."
        )

    (
        OUT
        / "AUDIT_10B_STATUS.txt"
    ).write_text(
        status + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
