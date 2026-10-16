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

BRIDGE = (
    ROOT
    / "results/phase3/audit_02c_PA14_old_locus_annotation_bridge"
    / "GSE191103_PA14_old_locus_gene_lengths.csv"
)

AUDIT01B = (
    ROOT
    / "results/phase3/audit_01b_PA14_final_transferability"
    / "PA14_final_imodulon_transferability.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
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


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02e")
    print("Freeze final PA14 projection-compatible gene universe")
    print("=" * 78)

    for path in [
        MODEL,
        EXPRESSION,
        STRICT_MAP,
        BRIDGE,
        AUDIT01B,
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
    # Frozen aeruPRECISE M
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

    # ----------------------------------------------------------
    # Counts
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

    print()
    print("GSE191103 WT count matrix")
    print(
        "  genes:",
        counts.shape[
            0
        ],
    )
    print(
        "  samples:",
        counts.shape[
            1
        ],
    )

    # ----------------------------------------------------------
    # Strict one-to-one PAO1 <-> PA14 map
    # ----------------------------------------------------------

    strict = pd.read_csv(
        STRICT_MAP
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

    strict[
        "pa14_gene"
    ] = (
        strict[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    if strict[
        "pao1_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate PAO1 genes in strict map."
        )

    if strict[
        "pa14_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate PA14 genes in strict map."
        )

    print()
    print("Strict orthologue core")
    print(
        "  pairs:",
        len(
            strict
        ),
    )

    strict_present = strict.loc[
        strict[
            "pa14_gene"
        ].isin(
            counts.index
        )
    ].copy()

    print(
        "  pairs present in expression:",
        len(
            strict_present
        ),
    )

    # ----------------------------------------------------------
    # Verified historical-locus lengths
    # ----------------------------------------------------------

    bridge = pd.read_csv(
        BRIDGE
    )

    bridge[
        "old_locus_tag"
    ] = (
        bridge[
            "old_locus_tag"
        ]
        .astype(str)
        .str.strip()
    )

    bridge[
        "length_bp"
    ] = pd.to_numeric(
        bridge[
            "length_bp"
        ],
        errors="raise",
    )

    if bridge[
        "old_locus_tag"
    ].duplicated().any():
        raise RuntimeError(
            "Duplicate old locus tags in length bridge."
        )

    if (
        bridge[
            "length_bp"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "Non-positive gene length found."
        )

    projection_map = strict_present.merge(
        bridge[
            [
                "old_locus_tag",
                "current_locus_tag",
                "feature_type",
                "gene_symbol",
                "length_bp",
                "start",
                "end",
                "strand",
            ]
        ],
        left_on="pa14_gene",
        right_on="old_locus_tag",
        how="inner",
        validate="one_to_one",
    )

    projection_map = projection_map.drop(
        columns=[
            "old_locus_tag"
        ]
    )

    projection_map = projection_map.sort_values(
        "pao1_gene"
    )

    projection_map.to_csv(
        OUT
        / "frozen_PA14_projection_map.csv",
        index=False,
    )

    projection_pao1 = set(
        projection_map[
            "pao1_gene"
        ]
    )

    projection_pa14 = set(
        projection_map[
            "pa14_gene"
        ]
    )

    model_coverage = (
        len(
            projection_pao1
        )
        / M.shape[
            0
        ]
    )

    strict_retention = (
        len(
            projection_map
        )
        / len(
            strict
        )
    )

    print()
    print("Final projection-compatible universe")
    print(
        "  genes:",
        len(
            projection_map
        ),
    )
    print(
        "  fraction of strict 1:1 core retained:",
        format(
            strict_retention,
            ".2%",
        ),
    )
    print(
        "  fraction of full PAO1 model genes:",
        format(
            model_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # Count mass within the MODEL-COMPATIBLE orthologue universe
    #
    # This is deliberately NOT compared to all 6026 expression
    # features because Audit 02d showed that the excluded mass is
    # dominated by ncRNA/intergenic features outside aeruPRECISE.
    # ----------------------------------------------------------

    strict_pa14 = [
        gene
        for gene in strict[
            "pa14_gene"
        ]
        if gene in counts.index
    ]

    projection_pa14_ordered = [
        gene
        for gene in projection_map[
            "pa14_gene"
        ]
        if gene in counts.index
    ]

    strict_count_mass = counts.loc[
        strict_pa14
    ].sum(
        axis=0
    )

    projection_count_mass = counts.loc[
        projection_pa14_ordered
    ].sum(
        axis=0
    )

    retained_model_count_fraction = (
        projection_count_mass
        / strict_count_mass
    )

    minimum_model_count_fraction = float(
        retained_model_count_fraction.min()
    )

    print()
    print(
        "Count mass retained within strict orthologue universe"
    )
    print(
        retained_model_count_fraction
        .round(
            4
        )
        .to_string()
    )

    print(
        "  minimum:",
        format(
            minimum_model_count_fraction,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # The 34 strict genes lost because verified lengths are absent
    # ----------------------------------------------------------

    excluded_strict = strict_present.loc[
        ~strict_present[
            "pa14_gene"
        ].isin(
            projection_pa14
        )
    ].copy()

    excluded_counts = counts.reindex(
        excluded_strict[
            "pa14_gene"
        ]
    )

    excluded_counts[
        "total_count"
    ] = excluded_counts.sum(
        axis=1
    )

    excluded_counts[
        "pao1_gene"
    ] = excluded_counts.index.map(
        dict(
            zip(
                strict[
                    "pa14_gene"
                ],
                strict[
                    "pao1_gene"
                ],
            )
        )
    )

    excluded_counts = excluded_counts.sort_values(
        "total_count",
        ascending=False,
    )

    excluded_counts.to_csv(
        OUT
        / "strict_orthologues_excluded_for_missing_length.csv"
    )

    print()
    print(
        "Strict orthologues excluded because no verified length:",
        len(
            excluded_strict
        ),
    )

    print()
    print("Highest-count excluded strict orthologues")
    print(
        excluded_counts[
            [
                "pao1_gene",
                "total_count",
            ]
        ]
        .head(
            20
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # Programme transferability after final 5286-gene restriction
    # ----------------------------------------------------------

    mapped_mask = np.array(
        [
            gene in projection_pao1
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

        weighted_coverage = (
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

        if weighted_coverage >= 0.95:
            transfer_status = "HIGH"

        elif weighted_coverage >= 0.80:
            transfer_status = "MODERATE"

        else:
            transfer_status = "LOW"

        rows.append(
            {
                "imodulon":
                    imodulon,

                "weighted_coverage_absM":
                    weighted_coverage,

                "energy_coverage_M2":
                    energy_coverage,

                "transfer_status":
                    transfer_status,
            }
        )

    final_transfer = pd.DataFrame(
        rows
    )

    previous = pd.read_csv(
        AUDIT01B
    )

    previous = previous.set_index(
        "imodulon"
    )

    previous.index = (
        previous.index
        .astype(str)
        .str.strip()
    )

    final_transfer[
        "audit01b_weighted_coverage"
    ] = (
        previous.reindex(
            final_transfer[
                "imodulon"
            ]
        )[
            "weighted_coverage_absM"
        ]
        .to_numpy()
    )

    final_transfer[
        "additional_coverage_loss_from_length_filter"
    ] = (
        final_transfer[
            "audit01b_weighted_coverage"
        ]
        - final_transfer[
            "weighted_coverage_absM"
        ]
    )

    final_transfer = final_transfer.sort_values(
        "weighted_coverage_absM"
    )

    final_transfer.to_csv(
        OUT
        / "final_projection_imodulon_transferability.csv",
        index=False,
    )

    weighted = final_transfer[
        "weighted_coverage_absM"
    ]

    energy = final_transfer[
        "energy_coverage_M2"
    ]

    max_additional_loss = float(
        final_transfer[
            "additional_coverage_loss_from_length_filter"
        ].max()
    )

    counts_status = (
        final_transfer[
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

    print()
    print("Final projection programme coverage")
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
    print(
        "  maximum additional |M| coverage loss "
        "from missing-length restriction:",
        format(
            max_additional_loss,
            ".2%",
        ),
    )

    print()
    print("Transferability classes after final restriction")
    print(
        "  HIGH:",
        int(
            counts_status[
                "HIGH"
            ]
        ),
    )
    print(
        "  MODERATE:",
        int(
            counts_status[
                "MODERATE"
            ]
        ),
    )
    print(
        "  LOW:",
        int(
            counts_status[
                "LOW"
            ]
        ),
    )

    print()
    print("Lowest-coverage programmes")
    print(
        final_transfer[
            [
                "imodulon",
                "weighted_coverage_absM",
                "energy_coverage_M2",
                "audit01b_weighted_coverage",
                "additional_coverage_loss_from_length_filter",
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
    # Audit gates
    # ----------------------------------------------------------

    checks = {
        "projection_genes_ge_5200":
            len(
                projection_map
            )
            >= 5200,

        "full_model_gene_coverage_ge_94pct":
            model_coverage
            >= 0.94,

        "strict_orthologue_retention_ge_99pct":
            strict_retention
            >= 0.99,

        "strict_model_count_mass_retained_ge_98pct":
            minimum_model_count_fraction
            >= 0.98,

        "median_weighted_programme_coverage_ge_95pct":
            float(
                weighted.median()
            )
            >= 0.95,

        "minimum_weighted_programme_coverage_ge_80pct":
            float(
                weighted.min()
            )
            >= 0.80,

        "no_LOW_programmes":
            int(
                counts_status[
                    "LOW"
                ]
            )
            == 0,

        "max_additional_programme_loss_le_5pct":
            max_additional_loss
            <= 0.05,
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

    if passed:
        status = (
            "FINAL_PA14_PROJECTION_UNIVERSE_ACCEPTED"
        )
    else:
        status = (
            "PROJECTION_UNIVERSE_REVIEW_REQUIRED"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_02E_STATUS.txt"
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
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
