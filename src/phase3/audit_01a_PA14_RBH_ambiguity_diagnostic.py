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
    / "results/phase3/audit_01a_PA14_RBH_ambiguity_diagnostic"
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


def clean(series):
    values = (
        series
        .astype(str)
        .str.strip()
    )

    values = values.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NA": np.nan,
            "--": np.nan,
        }
    )

    return values


def nonempty(value):
    if pd.isna(value):
        return False

    return bool(
        str(value).strip()
    )


def weighted_coverage_table(
    M,
    mapped_genes,
):
    mapped_set = set(
        mapped_genes
    )

    mask = np.array(
        [
            gene in mapped_set
            for gene in M.index
        ],
        dtype=bool,
    )

    rows = []

    for imodulon in M.columns:
        weights = M[
            imodulon
        ].to_numpy(
            dtype=float
        )

        abs_w = np.abs(
            weights
        )

        sq_w = weights ** 2

        abs_total = float(
            abs_w.sum()
        )

        sq_total = float(
            sq_w.sum()
        )

        abs_cov = (
            float(
                abs_w[
                    mask
                ].sum()
            )
            / abs_total
            if abs_total > 0
            else np.nan
        )

        sq_cov = (
            float(
                sq_w[
                    mask
                ].sum()
            )
            / sq_total
            if sq_total > 0
            else np.nan
        )

        rows.append(
            {
                "imodulon":
                    imodulon,

                "weighted_coverage_absM":
                    abs_cov,

                "energy_coverage_M2":
                    sq_cov,
            }
        )

    return pd.DataFrame(
        rows
    )


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 01a")
    print("PAO1 <-> PA14 RBH ambiguity diagnostic")
    print("=" * 78)

    for path in [
        MODEL,
        RBH,
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
    # Frozen model
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

    model_genes = set(
        M.index
    )

    print()
    print("Frozen model")
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

    # ----------------------------------------------------------
    # Load known RBH schema
    # ----------------------------------------------------------

    raw_map = pd.read_csv(
        RBH
    )

    required = [
        "Locus Tag (Query)",
        "Locus Tag (Hit)",
        "Inparalog (Query)",
        "Inparalog (Hit)",
        "Description (Query)",
        "Description (Hit)",
    ]

    missing = [
        col
        for col in required
        if col not in raw_map.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing RBH columns: "
            + ", ".join(
                missing
            )
        )

    table = pd.DataFrame(
        {
            "pao1_gene":
                clean(
                    raw_map[
                        "Locus Tag (Query)"
                    ]
                ),

            "pa14_gene":
                clean(
                    raw_map[
                        "Locus Tag (Hit)"
                    ]
                ),

            "pao1_description":
                clean(
                    raw_map[
                        "Description (Query)"
                    ]
                ),

            "pa14_description":
                clean(
                    raw_map[
                        "Description (Hit)"
                    ]
                ),

            "query_inparalog":
                clean(
                    raw_map[
                        "Inparalog (Query)"
                    ]
                ),

            "hit_inparalog":
                clean(
                    raw_map[
                        "Inparalog (Hit)"
                    ]
                ),
        }
    )

    table = table.dropna(
        subset=[
            "pao1_gene",
            "pa14_gene",
        ]
    )

    table = table.loc[
        table[
            "pao1_gene"
        ].isin(
            model_genes
        )
    ].copy()

    print()
    print("Model-restricted RBH table")
    print(
        "  rows:",
        len(
            table
        ),
    )
    print(
        "  unique PAO1 genes:",
        table[
            "pao1_gene"
        ].nunique(),
    )
    print(
        "  unique PA14 genes:",
        table[
            "pa14_gene"
        ].nunique(),
    )

    # ----------------------------------------------------------
    # Exact duplicate pairs
    # ----------------------------------------------------------

    exact_pair_duplicates = table.loc[
        table.duplicated(
            subset=[
                "pao1_gene",
                "pa14_gene",
            ],
            keep=False,
        )
    ].copy()

    unique_pairs = table.drop_duplicates(
        subset=[
            "pao1_gene",
            "pa14_gene",
        ]
    ).copy()

    print()
    print("Exact pair duplication")
    print(
        "  rows involved in repeated identical PAO1-PA14 pairs:",
        len(
            exact_pair_duplicates
        ),
    )
    print(
        "  unique PAO1-PA14 pairs:",
        len(
            unique_pairs
        ),
    )

    # ----------------------------------------------------------
    # Count multiplicity after exact-pair deduplication
    # ----------------------------------------------------------

    query_counts = (
        unique_pairs[
            "pao1_gene"
        ]
        .value_counts()
    )

    hit_counts = (
        unique_pairs[
            "pa14_gene"
        ]
        .value_counts()
    )

    unique_pairs[
        "n_hits_for_pao1"
    ] = (
        unique_pairs[
            "pao1_gene"
        ]
        .map(
            query_counts
        )
    )

    unique_pairs[
        "n_pao1_for_pa14"
    ] = (
        unique_pairs[
            "pa14_gene"
        ]
        .map(
            hit_counts
        )
    )

    unique_pairs[
        "query_inparalog_flag"
    ] = unique_pairs[
        "query_inparalog"
    ].map(
        nonempty
    )

    unique_pairs[
        "hit_inparalog_flag"
    ] = unique_pairs[
        "hit_inparalog"
    ].map(
        nonempty
    )

    unique_pairs[
        "any_inparalog_flag"
    ] = (
        unique_pairs[
            "query_inparalog_flag"
        ]
        | unique_pairs[
            "hit_inparalog_flag"
        ]
    )

    ambiguous = unique_pairs.loc[
        (
            unique_pairs[
                "n_hits_for_pao1"
            ]
            > 1
        )
        | (
            unique_pairs[
                "n_pao1_for_pa14"
            ]
            > 1
        )
    ].copy()

    strict_unique = unique_pairs.loc[
        (
            unique_pairs[
                "n_hits_for_pao1"
            ]
            == 1
        )
        & (
            unique_pairs[
                "n_pao1_for_pa14"
            ]
            == 1
        )
    ].copy()

    strict_no_inparalog = strict_unique.loc[
        ~strict_unique[
            "any_inparalog_flag"
        ]
    ].copy()

    print()
    print("Multiplicity structure")
    print(
        "  PAO1 genes with >1 PA14 hit:",
        int(
            (
                query_counts
                > 1
            ).sum()
        ),
    )
    print(
        "  PA14 genes hit by >1 PAO1 gene:",
        int(
            (
                hit_counts
                > 1
            ).sum()
        ),
    )
    print(
        "  ambiguous mapping rows:",
        len(
            ambiguous
        ),
    )
    print(
        "  strict one-to-one rows:",
        len(
            strict_unique
        ),
    )
    print(
        "  strict one-to-one rows with no inparalog flag:",
        len(
            strict_no_inparalog
        ),
    )

    # ----------------------------------------------------------
    # Do ambiguity and inparalog flags coincide?
    # ----------------------------------------------------------

    if not ambiguous.empty:
        ambiguous_with_flag = int(
            ambiguous[
                "any_inparalog_flag"
            ].sum()
        )

        print()
        print("Ambiguity vs inparalog annotation")
        print(
            "  ambiguous rows with an inparalog flag:",
            ambiguous_with_flag,
        )
        print(
            "  ambiguous rows without an inparalog flag:",
            len(
                ambiguous
            )
            - ambiguous_with_flag,
        )

        print()
        print("All ambiguous mappings")
        print(
            ambiguous[
                [
                    "pao1_gene",
                    "pa14_gene",
                    "pao1_description",
                    "pa14_description",
                    "query_inparalog",
                    "hit_inparalog",
                    "n_hits_for_pao1",
                    "n_pao1_for_pa14",
                ]
            ]
            .sort_values(
                [
                    "pao1_gene",
                    "pa14_gene",
                ]
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # Save diagnostic maps
    # ----------------------------------------------------------

    unique_pairs.to_csv(
        OUT
        / "all_unique_pairs_with_multiplicity.csv",
        index=False,
    )

    ambiguous.to_csv(
        OUT
        / "ambiguous_RBH_pairs.csv",
        index=False,
    )

    strict_unique.to_csv(
        OUT
        / "strict_unique_PAO1_PA14_map.csv",
        index=False,
    )

    strict_no_inparalog.to_csv(
        OUT
        / "strict_unique_no_inparalog_PAO1_PA14_map.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Quantify what we lose by being conservative
    # ----------------------------------------------------------

    maps = {
        "all_unique_PAO1_genes":
            set(
                unique_pairs[
                    "pao1_gene"
                ]
            ),

        "strict_one_to_one":
            set(
                strict_unique[
                    "pao1_gene"
                ]
            ),

        "strict_one_to_one_no_inparalog":
            set(
                strict_no_inparalog[
                    "pao1_gene"
                ]
            ),
    }

    summary_rows = []
    coverage_tables = {}

    for name, genes in maps.items():
        coverage = (
            len(
                genes
            )
            / M.shape[
                0
            ]
        )

        programme = weighted_coverage_table(
            M,
            genes,
        )

        coverage_tables[
            name
        ] = programme

        summary_rows.append(
            {
                "mapping_definition":
                    name,

                "mapped_PAO1_genes":
                    len(
                        genes
                    ),

                "global_gene_coverage":
                    coverage,

                "median_absM_weighted_coverage":
                    float(
                        programme[
                            "weighted_coverage_absM"
                        ].median()
                    ),

                "minimum_absM_weighted_coverage":
                    float(
                        programme[
                            "weighted_coverage_absM"
                        ].min()
                    ),

                "median_M2_energy_coverage":
                    float(
                        programme[
                            "energy_coverage_M2"
                        ].median()
                    ),

                "minimum_M2_energy_coverage":
                    float(
                        programme[
                            "energy_coverage_M2"
                        ].min()
                    ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "mapping_definition_coverage_comparison.csv",
        index=False,
    )

    print()
    print("Coverage under increasingly conservative maps")
    print(
        summary
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Identify programmes most affected by ambiguity removal
    # ----------------------------------------------------------

    all_cov = (
        coverage_tables[
            "all_unique_PAO1_genes"
        ]
        .set_index(
            "imodulon"
        )
    )

    strict_cov = (
        coverage_tables[
            "strict_one_to_one"
        ]
        .set_index(
            "imodulon"
        )
    )

    impact = pd.DataFrame(
        {
            "all_unique_absM_coverage":
                all_cov[
                    "weighted_coverage_absM"
                ],

            "strict_1to1_absM_coverage":
                strict_cov[
                    "weighted_coverage_absM"
                ],
        }
    )

    impact[
        "coverage_loss_from_ambiguity_removal"
    ] = (
        impact[
            "all_unique_absM_coverage"
        ]
        - impact[
            "strict_1to1_absM_coverage"
        ]
    )

    impact = impact.sort_values(
        "coverage_loss_from_ambiguity_removal",
        ascending=False,
    )

    impact.to_csv(
        OUT
        / "programme_impact_of_ambiguity_removal.csv"
    )

    print()
    print("Programmes most affected by removing ambiguous mappings")
    print(
        impact
        .head(
            20
        )
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # Diagnostic decision
    # ----------------------------------------------------------

    strict_summary = summary.loc[
        summary[
            "mapping_definition"
        ]
        == "strict_one_to_one"
    ].iloc[
        0
    ]

    strict_global = float(
        strict_summary[
            "global_gene_coverage"
        ]
    )

    strict_median = float(
        strict_summary[
            "median_absM_weighted_coverage"
        ]
    )

    max_programme_loss = float(
        impact[
            "coverage_loss_from_ambiguity_removal"
        ].max()
    )

    print()
    print("Diagnostic decision metrics")
    print(
        "  strict 1:1 global gene coverage:",
        format(
            strict_global,
            ".2%",
        ),
    )
    print(
        "  strict 1:1 median |M|-weighted programme coverage:",
        format(
            strict_median,
            ".2%",
        ),
    )
    print(
        "  maximum programme coverage loss from ambiguity removal:",
        format(
            max_programme_loss,
            ".2%",
        ),
    )

    if (
        strict_global >= 0.90
        and strict_median >= 0.95
        and max_programme_loss <= 0.05
    ):
        status = (
            "STRICT_ONE_TO_ONE_CORE_ACCEPTABLE"
        )

    elif (
        strict_global >= 0.85
        and strict_median >= 0.90
    ):
        status = (
            "STRICT_CORE_USABLE_BUT_PROGRAMME_REVIEW_REQUIRED"
        )

    else:
        status = (
            "AMBIGUITY_MATERIALLY_AFFECTS_TRANSFER"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_01A_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
