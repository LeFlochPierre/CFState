from pathlib import Path
from io import StringIO
import json

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

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

PA14_GB = (
    ROOT
    / "references/mappings/NC_008463.1.gb"
)

OUT = (
    ROOT
    / "results/phase3/audit_03c_GSE55197_composite_ID_resolution"
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


def split_identifier(value):
    text = str(
        value
    ).strip()

    if "," in text:
        locus, annotation = text.split(
            ",",
            1,
        )

        return (
            locus.strip(),
            annotation.strip(),
        )

    return (
        text,
        "",
    )


def pairwise_r(
    matrix,
    a,
    b,
):
    return float(
        matrix[
            [
                a,
                b,
            ]
        ]
        .corr()
        .iloc[
            0,
            1,
        ]
    )


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 03c")
    print("GSE55197 composite identifier resolution")
    print("=" * 78)

    for path in [
        COUNTS,
        PROJECTION_MAP,
        MODEL,
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

    # ----------------------------------------------------------
    # 1. Original processed matrix
    # ----------------------------------------------------------

    original = pd.read_csv(
        COUNTS,
        index_col=0,
    )

    original.index = (
        original.index
        .astype(str)
        .str.strip()
    )

    original = original.apply(
        pd.to_numeric,
        errors="raise",
    )

    samples = list(
        original.columns
    )

    print()
    print("Original processed matrix")
    print(
        "  rows:",
        original.shape[
            0
        ],
    )
    print(
        "  samples:",
        original.shape[
            1
        ],
    )

    # ----------------------------------------------------------
    # 2. Split "locus_tag,gene_symbol"
    # ----------------------------------------------------------

    mapping_rows = []

    for observed in original.index:
        locus, annotation = split_identifier(
            observed
        )

        mapping_rows.append(
            {
                "observed_id":
                    observed,
                "normalized_locus":
                    locus,
                "annotation_suffix":
                    annotation,
            }
        )

    parsed = pd.DataFrame(
        mapping_rows
    )

    n_composite = int(
        (
            parsed[
                "annotation_suffix"
            ]
            != ""
        ).sum()
    )

    duplicated_loci = (
        parsed[
            "normalized_locus"
        ]
        .duplicated(
            keep=False
        )
    )

    collisions = parsed.loc[
        duplicated_loci
    ].copy()

    print()
    print("Composite-ID parsing")
    print(
        "  IDs containing comma annotation:",
        n_composite,
    )
    print(
        "  normalized locus collisions:",
        collisions[
            "normalized_locus"
        ].nunique()
        if not collisions.empty
        else 0,
    )

    if not collisions.empty:
        print()
        print("First normalized-locus collisions")
        print(
            collisions
            .head(
                40
            )
            .to_string(
                index=False
            )
        )

        collisions.to_csv(
            OUT
            / "normalized_locus_collisions.csv",
            index=False,
        )

        raise RuntimeError(
            "Composite-ID normalization created duplicate locus tags. "
            "Review before combining rows."
        )

    normalized = original.copy()

    normalized.index = parsed[
        "normalized_locus"
    ].to_numpy()

    if normalized.index.duplicated().any():
        raise RuntimeError(
            "Duplicate normalized locus tags remain."
        )

    normalized.to_csv(
        OUT
        / "GSE55197_stationary_exponential_normalized_locus_counts.csv"
    )

    parsed.to_csv(
        OUT
        / "GSE55197_identifier_normalization_map.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 3. Frozen PA14 -> PAO1 projection map
    # ----------------------------------------------------------

    projection = pd.read_csv(
        PROJECTION_MAP
    )

    for column in [
        "pa14_gene",
        "pao1_gene",
        "current_locus_tag",
    ]:
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

    projection_old = set(
        projection[
            "pa14_gene"
        ]
    )

    observed = set(
        normalized.index
    )

    represented = observed.intersection(
        projection_old
    )

    missing_projection = projection.loc[
        ~projection[
            "pa14_gene"
        ].isin(
            represented
        )
    ].copy()

    projection_coverage = (
        len(
            represented
        )
        / len(
            projection
        )
    )

    print()
    print("Frozen projection-universe recovery")
    print(
        "  frozen projection genes:",
        len(
            projection
        ),
    )
    print(
        "  represented after ID normalization:",
        len(
            represented
        ),
    )
    print(
        "  coverage:",
        format(
            projection_coverage,
            ".2%",
        ),
    )
    print(
        "  missing frozen projection genes:",
        len(
            missing_projection
        ),
    )

    missing_projection.to_csv(
        OUT
        / "missing_frozen_projection_genes.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 4. Current annotation compatibility diagnostic
    # ----------------------------------------------------------

    record = SeqIO.read(
        str(
            PA14_GB
        ),
        "genbank",
    )

    old_annotation_ids = set()

    old_cds_ids = set()

    for feature in record.features:
        old_values = feature.qualifiers.get(
            "old_locus_tag",
            [],
        )

        for value in old_values:
            value = str(
                value
            ).strip()

            old_annotation_ids.add(
                value
            )

            if feature.type == "CDS":
                old_cds_ids.add(
                    value
                )

    annotation_overlap = len(
        observed.intersection(
            old_annotation_ids
        )
    )

    cds_overlap = len(
        observed.intersection(
            old_cds_ids
        )
    )

    print()
    print("Current RefSeq historical-tag compatibility")
    print(
        "  normalized IDs matching any old locus tag:",
        annotation_overlap,
    )
    print(
        "  normalized IDs matching old CDS locus tags:",
        cds_overlap,
    )

    # ----------------------------------------------------------
    # 5. Replicate QC is recomputed after normalization.
    #
    # Identifier normalization should not change numerical
    # correlations because no rows were merged or discarded.
    # ----------------------------------------------------------

    library_sizes = normalized.sum(
        axis=0
    )

    cpm = normalized.divide(
        library_sizes,
        axis=1,
    ) * 1_000_000.0

    log_cpm = np.log2(
        cpm
        + 1.0
    )

    exp_r = pairwise_r(
        log_cpm,
        "PA14_EXP_1",
        "PA14_EXP_2",
    )

    stat_r = pairwise_r(
        log_cpm,
        "PA14_STAT_1",
        "PA14_STAT_2",
    )

    print()
    print("Replicate QC after identifier normalization")
    print(
        "  exponential replicate r:",
        round(
            exp_r,
            4,
        ),
    )
    print(
        "  stationary replicate r:",
        round(
            stat_r,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 6. Model-compatible count depth
    #
    # We deliberately do NOT divide by whole-transcriptome count
    # mass. GSE55197 contains extremely abundant rRNA/ncRNA rows
    # outside the coding-gene aeruPRECISE model.
    # ----------------------------------------------------------

    projection_rows = projection.loc[
        projection[
            "pa14_gene"
        ].isin(
            represented
        )
    ].copy()

    projection_counts = normalized.reindex(
        projection_rows[
            "pa14_gene"
        ]
    )

    model_depth = projection_counts.sum(
        axis=0
    )

    detected_projection_genes = (
        projection_counts
        > 0
    ).sum(
        axis=0
    )

    model_depth_table = pd.DataFrame(
        {
            "projection_gene_counts":
                model_depth,
            "detected_projection_genes":
                detected_projection_genes,
        }
    )

    print()
    print("Model-compatible count depth")
    print(
        model_depth_table.to_string()
    )

    model_depth_table.to_csv(
        OUT
        / "model_compatible_count_depth.csv"
    )

    # ----------------------------------------------------------
    # 7. Recalculate actual iModulon weight coverage for the
    #    exact GSE55197-compatible gene subset.
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

    represented_pao1 = set(
        projection_rows[
            "pao1_gene"
        ]
    )

    mask = np.array(
        [
            gene in represented_pao1
            for gene
            in M.index
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

        squared = (
            weights
            ** 2
        )

        abs_total = float(
            abs_weights.sum()
        )

        sq_total = float(
            squared.sum()
        )

        weighted_coverage = float(
            abs_weights[
                mask
            ].sum()
            / abs_total
        )

        energy_coverage = float(
            squared[
                mask
            ].sum()
            / sq_total
        )

        programme_rows.append(
            {
                "imodulon":
                    imodulon,
                "weighted_coverage_absM":
                    weighted_coverage,
                "energy_coverage_M2":
                    energy_coverage,
            }
        )

    programme = pd.DataFrame(
        programme_rows
    )

    median_weighted = float(
        programme[
            "weighted_coverage_absM"
        ].median()
    )

    minimum_weighted = float(
        programme[
            "weighted_coverage_absM"
        ].min()
    )

    n_below_80 = int(
        (
            programme[
                "weighted_coverage_absM"
            ]
            < 0.80
        ).sum()
    )

    print()
    print("Exact GSE55197 iModulon representation")
    print(
        "  median |M|-weighted coverage:",
        format(
            median_weighted,
            ".2%",
        ),
    )
    print(
        "  minimum |M|-weighted coverage:",
        format(
            minimum_weighted,
            ".2%",
        ),
    )
    print(
        "  programmes below 80%:",
        n_below_80,
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

    programme.to_csv(
        OUT
        / "GSE55197_exact_imodulon_transferability.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 8. Audit decision
    # ----------------------------------------------------------

    minimum_model_depth = float(
        model_depth.min()
    )

    checks = {
        "no_normalized_locus_collisions":
            collisions.empty,

        "projection_genes_ge_5000":
            len(
                represented
            )
            >= 5000,

        "projection_gene_coverage_ge_94pct":
            projection_coverage
            >= 0.94,

        "exponential_replicate_r_ge_090":
            exp_r
            >= 0.90,

        "stationary_replicate_r_ge_090":
            stat_r
            >= 0.90,

        "minimum_model_compatible_counts_ge_100k":
            minimum_model_depth
            >= 100000,

        "median_imodulon_weight_coverage_ge_90pct":
            median_weighted
            >= 0.90,

        "no_programmes_below_80pct":
            n_below_80
            == 0,
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
        "GSE55197_STATIONARY_PROJECTION_UNIVERSE_ACCEPTED"
        if passed
        else "GSE55197_IDENTIFIER_RESOLUTION_REVIEW_REQUIRED"
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
        / "AUDIT_03C_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
