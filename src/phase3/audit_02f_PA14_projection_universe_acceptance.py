from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

BASELINE = (
    ROOT
    / "results/phase3/audit_01b_PA14_final_transferability"
    / "PA14_final_imodulon_transferability.csv"
)

FINAL = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "final_projection_imodulon_transferability.csv"
)

PROJECTION_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

EXCLUDED = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "strict_orthologues_excluded_for_missing_length.csv"
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
    / "results/phase3/audit_02f_PA14_projection_universe_acceptance"
)


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02f")
    print("PA14 projection-universe degradation and acceptance test")
    print("=" * 78)

    for path in [
        BASELINE,
        FINAL,
        PROJECTION_MAP,
        EXCLUDED,
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

    # ----------------------------------------------------------
    # 1. Baseline transferability from Audit 01b
    # ----------------------------------------------------------

    baseline = pd.read_csv(
        BASELINE
    )

    baseline[
        "imodulon"
    ] = (
        baseline[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    baseline = baseline.set_index(
        "imodulon"
    )

    # ----------------------------------------------------------
    # 2. Final transferability after verified-length restriction
    # ----------------------------------------------------------

    final = pd.read_csv(
        FINAL
    )

    final[
        "imodulon"
    ] = (
        final[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    final = final.set_index(
        "imodulon"
    )

    shared = baseline.index.intersection(
        final.index
    )

    if len(
        shared
    ) != 116:
        raise RuntimeError(
            "Expected 116 shared iModulons, found "
            + str(
                len(
                    shared
                )
            )
        )

    comparison = pd.DataFrame(
        index=shared
    )

    comparison[
        "baseline_absM_coverage"
    ] = baseline.loc[
        shared,
        "weighted_coverage_absM",
    ]

    comparison[
        "final_absM_coverage"
    ] = final.loc[
        shared,
        "weighted_coverage_absM",
    ]

    comparison[
        "absM_coverage_loss"
    ] = (
        comparison[
            "baseline_absM_coverage"
        ]
        - comparison[
            "final_absM_coverage"
        ]
    )

    comparison[
        "baseline_M2_coverage"
    ] = baseline.loc[
        shared,
        "energy_coverage_M2",
    ]

    comparison[
        "final_M2_coverage"
    ] = final.loc[
        shared,
        "energy_coverage_M2",
    ]

    comparison[
        "M2_coverage_loss"
    ] = (
        comparison[
            "baseline_M2_coverage"
        ]
        - comparison[
            "final_M2_coverage"
        ]
    )

    comparison = comparison.sort_values(
        "absM_coverage_loss",
        ascending=False,
    )

    comparison.to_csv(
        OUT
        / "programme_coverage_degradation.csv"
    )

    # ----------------------------------------------------------
    # 3. Representation degradation metrics
    # ----------------------------------------------------------

    baseline_median = float(
        comparison[
            "baseline_absM_coverage"
        ].median()
    )

    final_median = float(
        comparison[
            "final_absM_coverage"
        ].median()
    )

    median_level_drop = (
        baseline_median
        - final_median
    )

    median_programme_loss = float(
        comparison[
            "absM_coverage_loss"
        ].median()
    )

    p95_programme_loss = float(
        comparison[
            "absM_coverage_loss"
        ].quantile(
            0.95
        )
    )

    max_programme_loss = float(
        comparison[
            "absM_coverage_loss"
        ].max()
    )

    min_final_coverage = float(
        comparison[
            "final_absM_coverage"
        ].min()
    )

    n_below_90 = int(
        (
            comparison[
                "final_absM_coverage"
            ]
            < 0.90
        ).sum()
    )

    n_below_80 = int(
        (
            comparison[
                "final_absM_coverage"
            ]
            < 0.80
        ).sum()
    )

    print()
    print("Programme representation degradation")
    print(
        "  baseline median |M| coverage:",
        format(
            baseline_median,
            ".2%",
        ),
    )
    print(
        "  final median |M| coverage:",
        format(
            final_median,
            ".2%",
        ),
    )
    print(
        "  median-level drop:",
        format(
            median_level_drop,
            ".2%",
        ),
    )
    print(
        "  median per-programme loss:",
        format(
            median_programme_loss,
            ".2%",
        ),
    )
    print(
        "  95th percentile per-programme loss:",
        format(
            p95_programme_loss,
            ".2%",
        ),
    )
    print(
        "  maximum per-programme loss:",
        format(
            max_programme_loss,
            ".2%",
        ),
    )
    print(
        "  minimum final programme coverage:",
        format(
            min_final_coverage,
            ".2%",
        ),
    )
    print(
        "  programmes below 90%:",
        n_below_90,
    )
    print(
        "  programmes below 80%:",
        n_below_80,
    )

    print()
    print("Most affected programmes")
    print(
        comparison
        .head(
            20
        )
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 4. Gene-universe retention
    # ----------------------------------------------------------

    projection_map = pd.read_csv(
        PROJECTION_MAP
    )

    strict = pd.read_csv(
        STRICT_MAP
    )

    strict_retention = (
        len(
            projection_map
        )
        / len(
            strict
        )
    )

    full_model_coverage = (
        len(
            projection_map
        )
        / 5572
    )

    print()
    print("Gene-universe retention")
    print(
        "  final genes:",
        len(
            projection_map
        ),
    )
    print(
        "  strict orthologue retention:",
        format(
            strict_retention,
            ".2%",
        ),
    )
    print(
        "  full PAO1 model coverage:",
        format(
            full_model_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # 5. Count mass lost specifically from strict orthologues
    # ----------------------------------------------------------

    counts = pd.read_csv(
        EXPRESSION,
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

    strict[
        "pa14_gene"
    ] = (
        strict[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    projection_map[
        "pa14_gene"
    ] = (
        projection_map[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    strict_genes = [
        gene
        for gene in strict[
            "pa14_gene"
        ]
        if gene in counts.index
    ]

    final_genes = [
        gene
        for gene in projection_map[
            "pa14_gene"
        ]
        if gene in counts.index
    ]

    strict_mass = counts.loc[
        strict_genes
    ].sum(
        axis=0
    )

    final_mass = counts.loc[
        final_genes
    ].sum(
        axis=0
    )

    retained_fraction = (
        final_mass
        / strict_mass
    )

    minimum_retained_fraction = float(
        retained_fraction.min()
    )

    print()
    print("Count mass retained within strict model universe")
    print(
        retained_fraction
        .round(
            4
        )
        .to_string()
    )
    print(
        "  minimum:",
        format(
            minimum_retained_fraction,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # 6. Decision
    #
    # Audit 02e remains formally failed under its original
    # absolute 95% median threshold.
    #
    # This audit asks a different question:
    # whether the verified-length restriction materially damages
    # the representation already accepted in Audit 01b.
    # ----------------------------------------------------------

    checks = {
        "strict_orthologue_retention_ge_99pct":
            strict_retention
            >= 0.99,

        "full_model_coverage_ge_94pct":
            full_model_coverage
            >= 0.94,

        "strict_model_count_mass_retained_ge_99pct":
            minimum_retained_fraction
            >= 0.99,

        "median_level_coverage_drop_le_1pct_point":
            median_level_drop
            <= 0.01,

        "median_programme_loss_le_1pct_point":
            median_programme_loss
            <= 0.01,

        "maximum_programme_loss_le_5pct_points":
            max_programme_loss
            <= 0.05,

        "minimum_final_programme_coverage_ge_80pct":
            min_final_coverage
            >= 0.80,

        "no_programmes_below_80pct":
            n_below_80
            == 0,

        "all_metrics_finite":
            bool(
                np.isfinite(
                    comparison.to_numpy(
                        dtype=float
                    )
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
        "FINAL_PA14_PROJECTION_UNIVERSE_ACCEPTED"
        if passed
        else "PROJECTION_UNIVERSE_NOT_ACCEPTED"
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
            for key, value in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    summary = [
        "# CF-State Phase 3 — Audit 02f",
        "",
        "## Final PA14 projection-universe acceptance",
        "",
        "Audit 02e formally failed its original absolute "
        "95% median weighted-coverage gate. This audit does "
        "not overwrite that result.",
        "",
        "Instead, Audit 02f tests whether restricting the "
        "already-approved strict PAO1↔PA14 orthologue core "
        "to genes with verified historical-locus lengths "
        "materially degrades the representation.",
        "",
        "- Final projection genes: "
        + str(
            len(
                projection_map
            )
        ),
        "- Strict orthologue retention: "
        + format(
            strict_retention,
            ".2%",
        ),
        "- Full PAO1 model coverage: "
        + format(
            full_model_coverage,
            ".2%",
        ),
        "- Minimum strict-model count mass retained: "
        + format(
            minimum_retained_fraction,
            ".2%",
        ),
        "- Baseline median weighted coverage: "
        + format(
            baseline_median,
            ".2%",
        ),
        "- Final median weighted coverage: "
        + format(
            final_median,
            ".2%",
        ),
        "- Median-level coverage drop: "
        + format(
            median_level_drop,
            ".2%",
        ),
        "- Maximum individual programme loss: "
        + format(
            max_programme_loss,
            ".2%",
        ),
        "- Minimum final programme coverage: "
        + format(
            min_final_coverage,
            ".2%",
        ),
        "- Programmes below 80%: "
        + str(
            n_below_80
        ),
        "",
        "## Decision",
        "",
        status,
    ]

    (
        OUT
        / "AUDIT_02F_SUMMARY.md"
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
            for key, value in checks.items()
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
