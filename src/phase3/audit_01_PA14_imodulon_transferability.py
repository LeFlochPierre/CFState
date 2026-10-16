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

RBH = (
    ROOT
    / "references/mappings/pao1_pa14_rbbh.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_01_PA14_imodulon_transferability"
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
        "Unexpected serialized DataFrame type."
    )


def clean_series(series):
    return (
        series
        .astype(str)
        .str.strip()
        .replace(
            {
                "": np.nan,
                "nan": np.nan,
                "None": np.nan,
                "NA": np.nan,
            }
        )
    )


def detect_pao1_column(table, model_genes):
    scores = []

    model_set = set(
        model_genes
    )

    for column in table.columns:
        values = clean_series(
            table[
                column
            ]
        )

        matches = int(
            values.isin(
                model_set
            ).sum()
        )

        scores.append(
            {
                "column":
                    column,

                "model_gene_matches":
                    matches,
            }
        )

    score_table = (
        pd.DataFrame(
            scores
        )
        .sort_values(
            "model_gene_matches",
            ascending=False,
        )
    )

    best = score_table.iloc[
        0
    ]

    if int(
        best[
            "model_gene_matches"
        ]
    ) < 1000:
        raise RuntimeError(
            "Could not identify the PAO1 column. "
            "Best model-gene match count was "
            + str(
                int(
                    best[
                        "model_gene_matches"
                    ]
                )
            )
            + "."
        )

    return (
        best[
            "column"
        ],
        score_table,
    )


def detect_pa14_column(
    table,
    pao1_column,
):
    candidates = []

    for column in table.columns:
        if column == pao1_column:
            continue

        values = clean_series(
            table[
                column
            ]
        )

        nonmissing = int(
            values.notna().sum()
        )

        unique = int(
            values.dropna().nunique()
        )

        name = str(
            column
        ).lower()

        name_bonus = 0

        if "pa14" in name:
            name_bonus += 100000

        if "ortholog" in name:
            name_bonus += 10000

        if "hit" in name:
            name_bonus += 1000

        candidates.append(
            {
                "column":
                    column,

                "nonmissing":
                    nonmissing,

                "unique":
                    unique,

                "name_bonus":
                    name_bonus,

                "score":
                    name_bonus
                    + unique,
            }
        )

    candidate_table = (
        pd.DataFrame(
            candidates
        )
        .sort_values(
            [
                "score",
                "nonmissing",
            ],
            ascending=False,
        )
    )

    if candidate_table.empty:
        raise RuntimeError(
            "No candidate PA14 column found."
        )

    best = candidate_table.iloc[
        0
    ]

    if int(
        best[
            "nonmissing"
        ]
    ) < 1000:
        raise RuntimeError(
            "Could not identify a plausible PA14 column."
        )

    return (
        best[
            "column"
        ],
        candidate_table,
    )


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

    if len(
        selected
    ) == 0:
        return np.nan

    return float(
        mapped_mask[
            selected
        ].mean()
    )


def cumulative_support_metrics(
    weights,
    mapped_mask,
    target_fraction=0.90,
):
    abs_weights = np.abs(
        weights
    )

    total = float(
        abs_weights.sum()
    )

    if total == 0:
        return (
            0,
            np.nan,
        )

    order = np.argsort(
        -abs_weights
    )

    ordered_weights = abs_weights[
        order
    ]

    cumulative = np.cumsum(
        ordered_weights
    )

    threshold = (
        target_fraction
        * total
    )

    n_support = int(
        np.searchsorted(
            cumulative,
            threshold,
            side="left",
        )
        + 1
    )

    selected = order[
        :n_support
    ]

    mapped_fraction = float(
        mapped_mask[
            selected
        ].mean()
    )

    return (
        n_support,
        mapped_fraction,
    )


def transfer_status(
    weighted_coverage,
):
    if weighted_coverage >= 0.95:
        return "HIGH"

    if weighted_coverage >= 0.80:
        return "MODERATE"

    return "LOW"


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 01")
    print("PAO1 -> PA14 iModulon transferability")
    print("=" * 78)

    for path in [
        MODEL,
        RBH,
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

    model_genes = list(
        M.index
    )

    model_set = set(
        model_genes
    )

    print()
    print("Frozen aeruPRECISE representation")
    print(
        "  genes:",
        M.shape[
            0
        ],
    )
    print(
        "  iModulons:",
        M.shape[
            1
        ],
    )

    if M.shape != (
        5572,
        116,
    ):
        raise RuntimeError(
            "Unexpected aeruPRECISE M shape: "
            + str(
                M.shape
            )
        )

    # ----------------------------------------------------------
    # Load and diagnose RBH table
    # ----------------------------------------------------------

    rbh_raw = pd.read_csv(
        RBH
    )

    print()
    print("Raw PAO1 <-> PA14 RBH table")
    print(
        "  shape:",
        rbh_raw.shape,
    )
    print(
        "  columns:",
        list(
            rbh_raw.columns
        ),
    )

    pao1_col, pao1_scores = (
        detect_pao1_column(
            rbh_raw,
            model_genes,
        )
    )

    pa14_col, pa14_scores = (
        detect_pa14_column(
            rbh_raw,
            pao1_col,
        )
    )

    print()
    print("Detected mapping columns")
    print(
        "  PAO1:",
        pao1_col,
    )
    print(
        "  PA14:",
        pa14_col,
    )

    print()
    print("PAO1 column detection")
    print(
        pao1_scores
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print()
    print("PA14 column detection")
    print(
        pa14_scores
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    mapping = pd.DataFrame(
        {
            "pao1_gene":
                clean_series(
                    rbh_raw[
                        pao1_col
                    ]
                ),

            "pa14_gene":
                clean_series(
                    rbh_raw[
                        pa14_col
                    ]
                ),
        }
    )

    mapping = mapping.dropna(
        subset=[
            "pao1_gene",
            "pa14_gene",
        ]
    )

    mapping = mapping.loc[
        mapping[
            "pao1_gene"
        ].isin(
            model_set
        )
    ].copy()

    mapping = mapping.drop_duplicates()

    pao1_duplicates = mapping.loc[
        mapping[
            "pao1_gene"
        ].duplicated(
            keep=False
        )
    ]

    pa14_duplicates = mapping.loc[
        mapping[
            "pa14_gene"
        ].duplicated(
            keep=False
        )
    ]

    print()
    print("RBH integrity")
    print(
        "  model PAO1 genes represented:",
        mapping[
            "pao1_gene"
        ].nunique(),
    )
    print(
        "  unique PA14 genes:",
        mapping[
            "pa14_gene"
        ].nunique(),
    )
    print(
        "  PAO1 rows involved in duplicate mappings:",
        len(
            pao1_duplicates
        ),
    )
    print(
        "  PA14 rows involved in duplicate mappings:",
        len(
            pa14_duplicates
        ),
    )

    if not pao1_duplicates.empty:
        pao1_duplicates.to_csv(
            OUT
            / "duplicate_PAO1_mappings.csv",
            index=False,
        )

    if not pa14_duplicates.empty:
        pa14_duplicates.to_csv(
            OUT
            / "duplicate_PA14_mappings.csv",
            index=False,
        )

    if (
        not pao1_duplicates.empty
        or not pa14_duplicates.empty
    ):
        raise RuntimeError(
            "The supposed RBH map is not one-to-one. "
            "Inspect duplicate mapping outputs before continuing."
        )

    mapping = mapping.sort_values(
        "pao1_gene"
    )

    mapping.to_csv(
        OUT
        / "frozen_PAO1_PA14_RBH_model_genes.csv",
        index=False,
    )

    mapped_pao1 = set(
        mapping[
            "pao1_gene"
        ]
    )

    overall_gene_coverage = (
        len(
            mapped_pao1
        )
        / len(
            model_genes
        )
    )

    print()
    print("Global PAO1 -> PA14 model coverage")
    print(
        "  mapped model genes:",
        len(
            mapped_pao1
        ),
    )
    print(
        "  total model genes:",
        len(
            model_genes
        ),
    )
    print(
        "  unweighted coverage:",
        format(
            overall_gene_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # Per-iModulon weighted transferability
    # ----------------------------------------------------------

    mapped_mask = np.array(
        [
            gene in mapped_pao1
            for gene in model_genes
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

        squared_total = float(
            squared_weights.sum()
        )

        abs_mapped = float(
            abs_weights[
                mapped_mask
            ].sum()
        )

        squared_mapped = float(
            squared_weights[
                mapped_mask
            ].sum()
        )

        weighted_coverage = (
            abs_mapped
            / abs_total
            if abs_total > 0
            else np.nan
        )

        energy_coverage = (
            squared_mapped
            / squared_total
            if squared_total > 0
            else np.nan
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

        support90_n, support90_coverage = (
            cumulative_support_metrics(
                weights,
                mapped_mask,
                target_fraction=0.90,
            )
        )

        unmapped_abs_weight = (
            abs_total
            - abs_mapped
        )

        rows.append(
            {
                "imodulon":
                    imodulon,

                "weighted_coverage_absM":
                    weighted_coverage,

                "energy_coverage_M2":
                    energy_coverage,

                "top10_gene_coverage":
                    top10,

                "top20_gene_coverage":
                    top20,

                "support90_gene_count":
                    support90_n,

                "support90_gene_coverage":
                    support90_coverage,

                "unmapped_abs_weight":
                    unmapped_abs_weight,

                "transfer_status":
                    transfer_status(
                        weighted_coverage
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
        / "PA14_imodulon_transferability.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Summary statistics
    # ----------------------------------------------------------

    status_counts = (
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
    print("iModulon transferability summary")
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
        "  HIGH (>=95% weighted):",
        int(
            status_counts[
                "HIGH"
            ]
        ),
    )
    print(
        "  MODERATE (80-95% weighted):",
        int(
            status_counts[
                "MODERATE"
            ]
        ),
    )
    print(
        "  LOW (<80% weighted):",
        int(
            status_counts[
                "LOW"
            ]
        ),
    )

    print()
    print("Lowest-coverage iModulons")
    print(
        result[
            [
                "imodulon",
                "weighted_coverage_absM",
                "energy_coverage_M2",
                "top10_gene_coverage",
                "top20_gene_coverage",
                "support90_gene_count",
                "support90_gene_coverage",
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

    print()
    print("Highest-coverage iModulons")
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
        .tail(
            10
        )
        .sort_values(
            "weighted_coverage_absM",
            ascending=False,
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Identify influential missing genes
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

        if missing.empty:
            continue

        largest = (
            missing
            .abs()
            .sort_values(
                ascending=False
            )
            .head(
                20
            )
        )

        total_abs = float(
            weights.abs().sum()
        )

        for rank, gene in enumerate(
            largest.index,
            start=1,
        ):
            value = float(
                weights.loc[
                    gene
                ]
            )

            missing_rows.append(
                {
                    "imodulon":
                        imodulon,

                    "rank_within_missing":
                        rank,

                    "pao1_gene":
                        gene,

                    "M_weight":
                        value,

                    "abs_M_weight":
                        abs(
                            value
                        ),

                    "fraction_total_abs_weight":
                        (
                            abs(
                                value
                            )
                            / total_abs
                            if total_abs > 0
                            else np.nan
                        ),
                }
            )

    missing_table = pd.DataFrame(
        missing_rows
    )

    missing_table.to_csv(
        OUT
        / "largest_unmapped_gene_weights_by_imodulon.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Audit checks
    # ----------------------------------------------------------

    checks = {
        "model_shape_5572x116":
            M.shape
            == (
                5572,
                116,
            ),

        "rbh_one_to_one":
            (
                pao1_duplicates.empty
                and pa14_duplicates.empty
            ),

        "global_model_gene_coverage_ge_80pct":
            overall_gene_coverage
            >= 0.80,

        "all_programme_metrics_finite":
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

        "median_weighted_programme_coverage_ge_90pct":
            float(
                weighted.median()
            )
            >= 0.90,
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

    # ----------------------------------------------------------
    # Markdown summary
    # ----------------------------------------------------------

    summary = [
        "# CF-State Phase 3 — Audit 01",
        "",
        "## PAO1 → PA14 iModulon transferability",
        "",
        "### Objective",
        "",
        "Quantify how much of each frozen PAO1 aeruPRECISE "
        "iModulon can be represented in PA14 using the existing "
        "reciprocal-best-hit gene map.",
        "",
        "No PA14 expression data were projected in this audit.",
        "",
        "### Global mapping",
        "",
        "- PAO1 model genes: "
        + str(
            len(
                model_genes
            )
        ),
        "- PAO1 genes with PA14 RBHs: "
        + str(
            len(
                mapped_pao1
            )
        ),
        "- Global unweighted model-gene coverage: "
        + format(
            overall_gene_coverage,
            ".2%",
        ),
        "",
        "### Programme transferability",
        "",
        "- Median absolute-M weighted coverage: "
        + format(
            float(
                weighted.median()
            ),
            ".2%",
        ),
        "- Minimum absolute-M weighted coverage: "
        + format(
            float(
                weighted.min()
            ),
            ".2%",
        ),
        "- Median squared-M energy coverage: "
        + format(
            float(
                energy.median()
            ),
            ".2%",
        ),
        "- Minimum squared-M energy coverage: "
        + format(
            float(
                energy.min()
            ),
            ".2%",
        ),
        "",
        "Transferability classes:",
        "",
        "- HIGH (>=95%): "
        + str(
            int(
                status_counts[
                    "HIGH"
                ]
            )
        ),
        "- MODERATE (80-95%): "
        + str(
            int(
                status_counts[
                    "MODERATE"
                ]
            )
        ),
        "- LOW (<80%): "
        + str(
            int(
                status_counts[
                    "LOW"
                ]
            )
        ),
        "",
        "### Interpretation",
        "",
        "Absolute-M weighted coverage is the primary transferability "
        "measure because genes with larger iModulon weights contribute "
        "more strongly to the programme representation.",
        "",
        "Raw gene-count coverage alone must not be used to declare an "
        "iModulon transferable.",
        "",
        "This audit evaluates representational conservation only. "
        "It does not yet test whether PA14 expression projects "
        "biologically or quantitatively like PAO1 expression.",
        "",
        "### Decision",
        "",
        "PASS: "
        + str(
            passed
        ),
    ]

    (
        OUT
        / "AUDIT_01_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Phase 3 Audit 01:",
        "PASS"
        if passed
        else "FAIL",
    )
    print(
        "Summary:",
        OUT
        / "AUDIT_01_SUMMARY.md",
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
