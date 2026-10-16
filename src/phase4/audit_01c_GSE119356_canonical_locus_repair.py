from pathlib import Path
from io import StringIO
import json

import numpy as np
import pandas as pd


ROOT = Path.cwd()

XLSX = (
    ROOT
    / "data/raw/phase4/GSE119356"
    / "GSE119356_ProcessedDataMatrix.xlsx"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411"
    / "paeru_116_iModulon.json"
)

OUT = (
    ROOT
    / "results/phase4/audit_01c_GSE119356_canonical_locus_repair"
)


SAMPLES = [
    "invivo_BR",
    "invivo_ULP",
    "invivo_ULC",
    "invivo_LLP",
    "invivo_LLC",
    "invitro_BR",
    "invitro_ULP",
    "invitro_ULC",
    "invitro_LLP",
    "invitro_LLC",
]


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


def canonical_locus(value):
    text = str(
        value
    ).strip()

    if "," in text:
        text = text.split(
            ",",
            1,
        )[0].strip()

    return text


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 01c")
    print("GSE119356 canonical PAO1 locus repair and intake")
    print("=" * 78)

    for path in [
        XLSX,
        MODEL,
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
    # 1. Published matrix
    # ----------------------------------------------------------

    table = pd.read_excel(
        XLSX,
        sheet_name="Tabelle1",
    )

    required_columns = [
        "locus"
    ] + SAMPLES

    missing_columns = [
        column
        for column in required_columns
        if column not in table.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "Missing required columns: "
            + ", ".join(
                missing_columns
            )
        )

    raw_locus = (
        table[
            "locus"
        ]
        .astype(str)
        .str.strip()
    )

    canonical = raw_locus.map(
        canonical_locus
    )

    counts = (
        table[
            SAMPLES
        ]
        .apply(
            pd.to_numeric,
            errors="raise",
        )
    )

    print()
    print("Published expression matrix")
    print(
        "  rows:",
        len(
            table
        ),
    )
    print(
        "  samples:",
        len(
            SAMPLES
        ),
    )
    print(
        "  unique raw locus strings:",
        raw_locus.nunique(),
    )
    print(
        "  unique canonical locus tokens:",
        canonical.nunique(),
    )

    # ----------------------------------------------------------
    # 2. Frozen aeruPRECISE model
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
            "Unexpected model shape: "
            + str(
                M.shape
            )
        )

    model_genes = set(
        M.index
    )

    # ----------------------------------------------------------
    # 3. Canonical-token collisions
    #
    # This is the key correction relative to Audit 01b.
    # Decimal loci remain decimal loci.
    # ----------------------------------------------------------

    mapping = pd.DataFrame(
        {
            "raw_locus":
                raw_locus,

            "canonical_locus":
                canonical,

            "in_model":
                canonical.isin(
                    model_genes
                ),

            "total_count":
                counts.sum(
                    axis=1
                ),
        }
    )

    duplicate_canonical = (
        mapping[
            "canonical_locus"
        ]
        .value_counts()
    )

    duplicate_canonical = duplicate_canonical.loc[
        duplicate_canonical > 1
    ]

    collision_tokens = set(
        duplicate_canonical.index
    )

    collisions = mapping.loc[
        mapping[
            "canonical_locus"
        ].isin(
            collision_tokens
        )
    ].copy()

    model_collisions = collisions.loc[
        collisions[
            "in_model"
        ]
    ].copy()

    print()
    print("Canonical locus-token audit")
    print(
        "  duplicated canonical tokens:",
        len(
            collision_tokens
        ),
    )
    print(
        "  duplicated canonical tokens involving model genes:",
        model_collisions[
            "canonical_locus"
        ].nunique()
        if not model_collisions.empty
        else 0,
    )

    if not collisions.empty:
        print()
        print("Canonical-token collisions")
        print(
            collisions
            .sort_values(
                [
                    "canonical_locus",
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
    # 4. Exact model matching after canonicalization
    # ----------------------------------------------------------

    represented = mapping.loc[
        mapping[
            "in_model"
        ]
    ].copy()

    represented_genes = set(
        represented[
            "canonical_locus"
        ]
    )

    model_coverage = (
        len(
            represented_genes
        )
        / len(
            model_genes
        )
    )

    missing_model = sorted(
        model_genes.difference(
            represented_genes
        )
    )

    nonmodel = mapping.loc[
        ~mapping[
            "in_model"
        ]
    ].copy()

    nonmodel = nonmodel.sort_values(
        "total_count",
        ascending=False,
    )

    print()
    print("Exact aeruPRECISE compatibility after canonical repair")
    print(
        "  model genes:",
        len(
            model_genes
        ),
    )
    print(
        "  represented model genes:",
        len(
            represented_genes
        ),
    )
    print(
        "  model gene coverage:",
        format(
            model_coverage,
            ".2%",
        ),
    )
    print(
        "  missing model genes:",
        len(
            missing_model
        ),
    )

    print()
    print("Highest-count rows outside aeruPRECISE model")
    print(
        nonmodel[
            [
                "raw_locus",
                "canonical_locus",
                "total_count",
            ]
        ]
        .head(
            40
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Construct exact model-compatible count matrix
    #
    # No rows are merged.
    # If a model locus appears more than once after canonicalizing,
    # we stop rather than summing it.
    # ----------------------------------------------------------

    represented_mapping = mapping.loc[
        mapping[
            "in_model"
        ]
    ].copy()

    represented_duplicate_mask = (
        represented_mapping[
            "canonical_locus"
        ]
        .duplicated(
            keep=False
        )
    )

    represented_duplicates = represented_mapping.loc[
        represented_duplicate_mask
    ].copy()

    if not represented_duplicates.empty:
        print()
        print(
            "ERROR: duplicate model loci remain after exact canonical parsing."
        )
        print(
            represented_duplicates
            .sort_values(
                "canonical_locus"
            )
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Exact canonical parsing produced duplicate model genes. "
            "Do not merge without separate diagnosis."
        )

    row_indices = represented_mapping.index

    model_counts = counts.loc[
        row_indices
    ].copy()

    model_counts.index = represented_mapping[
        "canonical_locus"
    ].to_numpy()

    model_counts = model_counts.reindex(
        M.index
    )

    # Missing model genes are retained as NaN here rather than
    # silently zero-completed. Projection compatibility will be
    # decided explicitly below.
    observed_mask = (
        model_counts.notna()
        .all(
            axis=1
        )
    )

    if int(
        observed_mask.sum()
    ) != len(
        represented_genes
    ):
        raise RuntimeError(
            "Unexpected mismatch in represented gene count."
        )

    observed_counts = model_counts.loc[
        observed_mask
    ].copy()

    # ----------------------------------------------------------
    # 6. Model-compatible sequencing depth
    #
    # Whole-matrix count fraction is descriptive only because
    # abundant ncRNA/rRNA/intergenic features may be outside M.
    # ----------------------------------------------------------

    total_depth = counts.sum(
        axis=0
    )

    model_depth = observed_counts.sum(
        axis=0
    )

    model_mass_fraction = (
        model_depth
        / total_depth
    )

    detected_model_genes = (
        observed_counts
        > 0
    ).sum(
        axis=0
    )

    sample_qc = pd.DataFrame(
        {
            "total_matrix_counts":
                total_depth,

            "model_compatible_counts":
                model_depth,

            "model_count_mass_fraction":
                model_mass_fraction,

            "detected_model_genes":
                detected_model_genes,
        }
    )

    print()
    print("Per-sample model-compatible QC")
    print(
        sample_qc
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 7. Exact iModulon weight coverage caused by missing genes
    # ----------------------------------------------------------

    gene_present = np.array(
        [
            gene in represented_genes
            for gene in M.index
        ],
        dtype=bool,
    )

    programme_rows = []

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

        squared_total = float(
            squared_weights.sum()
        )

        abs_coverage = float(
            abs_weights[
                gene_present
            ].sum()
            / abs_total
        )

        energy_coverage = float(
            squared_weights[
                gene_present
            ].sum()
            / squared_total
        )

        programme_rows.append(
            {
                "imodulon":
                    imodulon,

                "weighted_coverage_absM":
                    abs_coverage,

                "energy_coverage_M2":
                    energy_coverage,
            }
        )

    programme = pd.DataFrame(
        programme_rows
    )

    median_abs = float(
        programme[
            "weighted_coverage_absM"
        ].median()
    )

    minimum_abs = float(
        programme[
            "weighted_coverage_absM"
        ].min()
    )

    median_m2 = float(
        programme[
            "energy_coverage_M2"
        ].median()
    )

    minimum_m2 = float(
        programme[
            "energy_coverage_M2"
        ].min()
    )

    below_80_abs = int(
        (
            programme[
                "weighted_coverage_absM"
            ]
            < 0.80
        ).sum()
    )

    robust_programmes = int(
        (
            (
                programme[
                    "weighted_coverage_absM"
                ]
                >= 0.90
            )
            & (
                programme[
                    "energy_coverage_M2"
                ]
                >= 0.80
            )
        ).sum()
    )

    print()
    print("Exact iModulon coverage in GSE119356")
    print(
        "  median |M|-weighted coverage:",
        format(
            median_abs,
            ".2%",
        ),
    )
    print(
        "  minimum |M|-weighted coverage:",
        format(
            minimum_abs,
            ".2%",
        ),
    )
    print(
        "  median M^2 energy coverage:",
        format(
            median_m2,
            ".2%",
        ),
    )
    print(
        "  minimum M^2 energy coverage:",
        format(
            minimum_m2,
            ".2%",
        ),
    )
    print(
        "  programmes with |M| coverage <80%:",
        below_80_abs,
    )
    print(
        "  programmes meeting >=90% |M| and >=80% M^2:",
        robust_programmes,
        "/ 116",
    )

    print()
    print("Lowest-coverage programmes")
    print(
        programme
        .sort_values(
            "weighted_coverage_absM"
        )
        .head(
            15
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 8. Descriptive sample correlation
    #
    # Not a replicate QC gate.
    # Spatial heterogeneity is expected in Phase 4.
    # ----------------------------------------------------------

    log_cpm = np.log2(
        observed_counts.divide(
            observed_counts.sum(
                axis=0
            ),
            axis=1,
        )
        * 1_000_000.0
        + 1.0
    )

    correlation = log_cpm.corr()

    print()
    print("Descriptive log2(model-CPM+1) sample correlation")
    print(
        correlation
        .round(
            3
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 9. Final repaired intake decision
    #
    # We do NOT gate on whole-matrix model count fraction.
    # We care about:
    #   - exact PAO1 model compatibility,
    #   - adequate model-compatible depth,
    #   - broad iModulon loading coverage,
    #   - no forced gene merging.
    # ----------------------------------------------------------

    minimum_model_depth = float(
        model_depth.min()
    )

    minimum_detected_model = int(
        detected_model_genes.min()
    )

    checks = {
        "all_10_expected_samples_present":
            counts.shape[
                1
            ]
            == 10,

        "no_duplicate_model_loci_after_canonicalization":
            represented_duplicates.empty,

        "model_gene_coverage_ge_95pct":
            model_coverage
            >= 0.95,

        "represented_model_genes_ge_5300":
            len(
                represented_genes
            )
            >= 5300,

        "minimum_model_compatible_counts_ge_50k":
            minimum_model_depth
            >= 50000,

        "minimum_detected_model_genes_ge_3000":
            minimum_detected_model
            >= 3000,

        "median_imodulon_absM_coverage_ge_95pct":
            median_abs
            >= 0.95,

        "minimum_imodulon_absM_coverage_ge_85pct":
            minimum_abs
            >= 0.85,

        "no_programmes_below_80pct_absM":
            below_80_abs
            == 0,

        "robust_programmes_ge_110":
            robust_programmes
            >= 110,

        "finite_counts":
            bool(
                np.isfinite(
                    observed_counts.to_numpy(
                        dtype=float
                    )
                ).all()
            ),

        "nonnegative_counts":
            bool(
                (
                    observed_counts
                    >= 0
                ).all().all()
            ),
    }

    passed = all(
        checks.values()
    )

    print()
    print("Repaired Phase-4 intake checks")

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
        "GSE119356_CANONICAL_PAO1_INTAKE_ACCEPTED"
        if passed
        else "GSE119356_CANONICAL_PAO1_INTAKE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    # ----------------------------------------------------------
    # 10. Save frozen intake products
    # ----------------------------------------------------------

    mapping.to_csv(
        OUT
        / "GSE119356_canonical_locus_map.csv",
        index=False,
    )

    nonmodel.to_csv(
        OUT
        / "GSE119356_nonmodel_rows.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "missing_model_gene":
                missing_model
        }
    ).to_csv(
        OUT
        / "GSE119356_missing_aeruprecise_genes.csv",
        index=False,
    )

    observed_counts.to_csv(
        OUT
        / "GSE119356_model_compatible_raw_counts.csv"
    )

    sample_qc.to_csv(
        OUT
        / "GSE119356_model_compatible_sample_QC.csv"
    )

    programme.to_csv(
        OUT
        / "GSE119356_imodulon_coverage.csv",
        index=False,
    )

    correlation.to_csv(
        OUT
        / "GSE119356_descriptive_model_CPM_correlation.csv"
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
        / "AUDIT_01C_STATUS.txt"
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
            "Failed repaired intake checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
