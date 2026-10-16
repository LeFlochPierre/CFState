from pathlib import Path
from io import StringIO
import json

import numpy as np
import pandas as pd


ROOT = Path.cwd()

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

STRICT_MAP = (
    ROOT
    / "results/phase3/audit_01a_PA14_RBH_ambiguity_diagnostic"
    / "strict_unique_PAO1_PA14_map.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_01b_PA14_final_transferability"
)


def json_df(value):
    if isinstance(value, str):
        return pd.read_json(
            StringIO(value)
        )

    if isinstance(value, dict):
        return pd.DataFrame(
            value
        )

    raise RuntimeError(
        "Unexpected serialized DataFrame."
    )


def status(weighted):
    if weighted >= 0.95:
        return "HIGH"

    if weighted >= 0.80:
        return "MODERATE"

    return "LOW"


def top_n_coverage(
    weights,
    mapped_mask,
    n,
):
    order = np.argsort(
        -np.abs(
            weights
        )
    )

    selected = order[
        :min(
            n,
            len(
                order
            ),
        )
    ]

    return float(
        mapped_mask[
            selected
        ].mean()
    )


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 01b")
    print("Final strict PAO1 -> PA14 iModulon transferability")
    print("=" * 78)

    for path in [
        MODEL,
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
    # Frozen aeruPRECISE model
    # ----------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(
            handle
        )

    M = json_df(
        raw[
            "M"
        ]
    )

    M.index = (
        M.index
        .astype(str)
        .str.strip()
    )

    M.columns = (
        M.columns
        .astype(str)
        .str.strip()
    )

    if M.shape != (
        5572,
        116,
    ):
        raise RuntimeError(
            "Unexpected M shape: "
            + str(
                M.shape
            )
        )

    # ----------------------------------------------------------
    # Frozen strict 1:1 map
    # ----------------------------------------------------------

    mapping = pd.read_csv(
        STRICT_MAP
    )

    required = {
        "pao1_gene",
        "pa14_gene",
    }

    missing = (
        required
        - set(
            mapping.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Strict map missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    mapping[
        "pao1_gene"
    ] = (
        mapping[
            "pao1_gene"
        ]
        .astype(str)
        .str.strip()
    )

    mapping[
        "pa14_gene"
    ] = (
        mapping[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    if mapping[
        "pao1_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Strict map contains duplicate PAO1 genes."
        )

    if mapping[
        "pa14_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Strict map contains duplicate PA14 genes."
        )

    model_genes = set(
        M.index
    )

    mapping = mapping.loc[
        mapping[
            "pao1_gene"
        ].isin(
            model_genes
        )
    ].copy()

    mapped_pao1 = set(
        mapping[
            "pao1_gene"
        ]
    )

    global_coverage = (
        len(
            mapped_pao1
        )
        / M.shape[
            0
        ]
    )

    print()
    print("Frozen strict mapping")
    print(
        "  PAO1 genes mapped:",
        len(
            mapped_pao1
        ),
    )
    print(
        "  model genes:",
        M.shape[
            0
        ],
    )
    print(
        "  global gene coverage:",
        format(
            global_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # Programme-level transferability
    # ----------------------------------------------------------

    mapped_mask = np.array(
        [
            gene in mapped_pao1
            for gene in M.index
        ],
        dtype=bool,
    )

    rows = []

    for imodulon in M.columns:
        weights = (
            M[
                imodulon
            ]
            .to_numpy(
                dtype=float
            )
        )

        abs_weights = np.abs(
            weights
        )

        squared_weights = (
            weights
            ** 2
        )

        abs_total = float(
            abs_weights.sum()
        )

        sq_total = float(
            squared_weights.sum()
        )

        abs_coverage = (
            float(
                abs_weights[
                    mapped_mask
                ].sum()
            )
            / abs_total
        )

        energy_coverage = (
            float(
                squared_weights[
                    mapped_mask
                ].sum()
            )
            / sq_total
        )

        top10 = top_n_coverage(
            weights,
            mapped_mask,
            10,
        )

        top20 = top_n_coverage(
            weights,
            mapped_mask,
            20,
        )

        rows.append(
            {
                "imodulon":
                    imodulon,

                "weighted_coverage_absM":
                    abs_coverage,

                "energy_coverage_M2":
                    energy_coverage,

                "top10_gene_coverage":
                    top10,

                "top20_gene_coverage":
                    top20,

                "transfer_status":
                    status(
                        abs_coverage
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result = result.sort_values(
        "weighted_coverage_absM",
        ascending=True,
    )

    result.to_csv(
        OUT
        / "PA14_final_imodulon_transferability.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Status summary
    # ----------------------------------------------------------

    counts = (
        result[
            "transfer_status"
        ]
        .value_counts()
        .reindex(
            [
                "HIGH",
                "MODERATE",
                "LOW",
            ],
            fill_value=0,
        )
    )

    weighted = result[
        "weighted_coverage_absM"
    ]

    energy = result[
        "energy_coverage_M2"
    ]

    print()
    print("Final programme transferability")
    print(
        "  median |M|-weighted coverage:",
        format(
            float(
                weighted.median()
            ),
            ".2%",
        ),
    )
    print(
        "  minimum |M|-weighted coverage:",
        format(
            float(
                weighted.min()
            ),
            ".2%",
        ),
    )
    print(
        "  median M^2 energy coverage:",
        format(
            float(
                energy.median()
            ),
            ".2%",
        ),
    )
    print(
        "  minimum M^2 energy coverage:",
        format(
            float(
                energy.min()
            ),
            ".2%",
        ),
    )

    print()
    print("Transferability classes")
    print(
        "  HIGH:",
        int(
            counts[
                "HIGH"
            ]
        ),
    )
    print(
        "  MODERATE:",
        int(
            counts[
                "MODERATE"
            ]
        ),
    )
    print(
        "  LOW:",
        int(
            counts[
                "LOW"
            ]
        ),
    )

    print()
    print("Lowest-coverage programmes")
    print(
        result[
            [
                "imodulon",
                "weighted_coverage_absM",
                "energy_coverage_M2",
                "top10_gene_coverage",
                "top20_gene_coverage",
                "transfer_status",
            ]
        ]
        .head(
            20
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Programme-level genes lost in transfer
    # ----------------------------------------------------------

    missing_rows = []

    for imodulon in M.columns:
        weights = (
            M[
                imodulon
            ]
            .astype(
                float
            )
        )

        missing = weights.loc[
            ~weights.index.isin(
                mapped_pao1
            )
        ]

        total_abs = float(
            weights.abs().sum()
        )

        largest = (
            missing
            .abs()
            .sort_values(
                ascending=False
            )
            .head(
                15
            )
        )

        for rank, gene in enumerate(
            largest.index,
            start=1,
        ):
            original_weight = float(
                weights.loc[
                    gene
                ]
            )

            missing_rows.append(
                {
                    "imodulon":
                        imodulon,

                    "rank":
                        rank,

                    "missing_pao1_gene":
                        gene,

                    "M_weight":
                        original_weight,

                    "abs_M_weight":
                        abs(
                            original_weight
                        ),

                    "fraction_total_abs_weight":
                        abs(
                            original_weight
                        )
                        / total_abs,
                }
            )

    pd.DataFrame(
        missing_rows
    ).to_csv(
        OUT
        / "largest_missing_weights.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Audit decision
    # ----------------------------------------------------------

    checks = {
        "strict_mapping_one_to_one":
            (
                not mapping[
                    "pao1_gene"
                ].duplicated().any()
                and not mapping[
                    "pa14_gene"
                ].duplicated().any()
            ),

        "global_gene_coverage_ge_90pct":
            global_coverage
            >= 0.90,

        "median_weighted_coverage_ge_95pct":
            float(
                weighted.median()
            )
            >= 0.95,

        "minimum_weighted_coverage_ge_80pct":
            float(
                weighted.min()
            )
            >= 0.80,

        "no_low_transfer_programmes":
            int(
                counts[
                    "LOW"
                ]
            )
            == 0,

        "all_metrics_finite":
            bool(
                np.isfinite(
                    result[
                        [
                            "weighted_coverage_absM",
                            "energy_coverage_M2",
                        ]
                    ].to_numpy(
                        dtype=float
                    )
                ).all()
            ),
    }

    passed = all(
        checks.values()
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

    summary = [
        "# CF-State Phase 3 — Audit 01b",
        "",
        "## Final PAO1 → PA14 iModulon transferability",
        "",
        "The PAO1→PA14 mapping was restricted to the frozen "
        "strict one-to-one reciprocal-best-hit core defined in "
        "Audit 01a. Ambiguous inparalog mappings were excluded.",
        "",
        "- Strict mapped PAO1 genes: "
        + str(
            len(
                mapped_pao1
            )
        ),
        "- Global model-gene coverage: "
        + format(
            global_coverage,
            ".2%",
        ),
        "- Median |M|-weighted programme coverage: "
        + format(
            float(
                weighted.median()
            ),
            ".2%",
        ),
        "- Minimum |M|-weighted programme coverage: "
        + format(
            float(
                weighted.min()
            ),
            ".2%",
        ),
        "- HIGH programmes: "
        + str(
            int(
                counts[
                    "HIGH"
                ]
            )
        ),
        "- MODERATE programmes: "
        + str(
            int(
                counts[
                    "MODERATE"
                ]
            )
        ),
        "- LOW programmes: "
        + str(
            int(
                counts[
                    "LOW"
                ]
            )
        ),
        "",
        "## Interpretation",
        "",
        "This audit quantifies representational transferability only. "
        "It does not yet establish biological equivalence of PAO1 and "
        "PA14 programme activities.",
        "",
        "## Decision",
        "",
        "PASS: "
        + str(
            passed
        ),
    ]

    (
        OUT
        / "AUDIT_01B_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Phase 3 Audit 01b:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_01B_SUMMARY.md",
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
