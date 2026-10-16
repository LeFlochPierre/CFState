from pathlib import Path
from io import StringIO
from collections import defaultdict
import json

import numpy as np
import pandas as pd
from Bio import SeqIO


ROOT = Path.cwd()

COUNTS = (
    ROOT
    / "results/phase4/audit_01c_GSE119356_canonical_locus_repair"
    / "GSE119356_model_compatible_raw_counts.csv"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411"
    / "paeru_116_iModulon.json"
)

PHASE3_ROBUST = (
    ROOT
    / "results/phase3/audit_04_freeze_crossstrain_state_framework"
    / "phase3_robust_transferable_imodulons.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_02a_GSE119356_matched_projection_QC"
)


REGIONS = [
    "BR",
    "ULP",
    "ULC",
    "LLP",
    "LLC",
]

INVIVO = [
    "invivo_" + region
    for region in REGIONS
]

INVITRO = [
    "invitro_" + region
    for region in REGIONS
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


def vector_corr(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        np.std(a) == 0
        or np.std(b) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def cosine(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return np.nan

    return float(
        np.dot(
            a,
            b,
        )
        / denominator
    )


def find_pao1_genbank():
    search_roots = [
        ROOT / "references",
        ROOT / "data",
        ROOT / "external",
    ]

    candidates = []

    for base in search_roots:
        if not base.exists():
            continue

        for path in base.rglob(
            "*NC_002516.2*"
        ):
            if path.is_file():
                suffixes = "".join(
                    path.suffixes
                ).lower()

                if (
                    ".gb" in suffixes
                    or ".gbk" in suffixes
                    or ".genbank" in suffixes
                ):
                    candidates.append(
                        path
                    )

    candidates = sorted(
        set(
            candidates
        ),
        key=lambda x: len(
            str(
                x
            )
        ),
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not locate PAO1 NC_002516.2 GenBank file "
            "under references/, data/, or external/."
        )

    return candidates[0]


def build_length_map(genbank_path):
    record = SeqIO.read(
        str(
            genbank_path
        ),
        "genbank",
    )

    lengths = defaultdict(
        set
    )

    feature_types = defaultdict(
        set
    )

    for feature in record.features:
        tags = []

        for qualifier in [
            "locus_tag",
            "old_locus_tag",
        ]:
            for value in feature.qualifiers.get(
                qualifier,
                [],
            ):
                value = str(
                    value
                ).strip()

                if value:
                    tags.append(
                        value
                    )

        if not tags:
            continue

        length = int(
            len(
                feature.location
            )
        )

        if length <= 0:
            continue

        for tag in tags:
            lengths[
                tag
            ].add(
                length
            )

            feature_types[
                tag
            ].add(
                feature.type
            )

    unique_lengths = {}

    ambiguous = {}

    for tag, values in lengths.items():
        if len(
            values
        ) == 1:
            unique_lengths[
                tag
            ] = next(
                iter(
                    values
                )
            )
        else:
            ambiguous[
                tag
            ] = sorted(
                values
            )

    return (
        unique_lengths,
        ambiguous,
        feature_types,
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 02a")
    print("GSE119356 matched CF-lung contrast projection QC")
    print("=" * 78)

    for path in [
        COUNTS,
        MODEL,
        PHASE3_ROBUST,
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
    # 1. Frozen PAO1 model
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
            "Unexpected aeruPRECISE M shape: "
            + str(
                M.shape
            )
        )

    imodulons = list(
        M.columns
    )

    print()
    print("Frozen PAO1 representation")
    print(
        "  genes:",
        M.shape[0],
    )
    print(
        "  iModulons:",
        M.shape[1],
    )

    # ----------------------------------------------------------
    # 2. Phase-3 primary programme set
    # ----------------------------------------------------------

    robust_table = pd.read_csv(
        PHASE3_ROBUST
    )

    if "imodulon" not in robust_table.columns:
        raise RuntimeError(
            "Phase-3 robust programme file missing 'imodulon' column."
        )

    phase3_primary = set(
        robust_table[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    primary_in_model = [
        imodulon
        for imodulon in imodulons
        if imodulon in phase3_primary
    ]

    sensitivity_only = [
        imodulon
        for imodulon in imodulons
        if imodulon not in phase3_primary
    ]

    print()
    print("Frozen Phase-3 interpretation hierarchy")
    print(
        "  primary programmes:",
        len(
            primary_in_model
        ),
    )
    print(
        "  sensitivity-only programmes:",
        len(
            sensitivity_only
        ),
    )

    # ----------------------------------------------------------
    # 3. Canonical model-compatible counts
    # ----------------------------------------------------------

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

    expected_samples = (
        INVIVO
        + INVITRO
    )

    missing_samples = [
        sample
        for sample in expected_samples
        if sample not in counts.columns
    ]

    if missing_samples:
        raise RuntimeError(
            "Missing expected samples: "
            + ", ".join(
                missing_samples
            )
        )

    counts = counts[
        expected_samples
    ]

    print()
    print("Input count matrix")
    print(
        "  represented model genes:",
        counts.shape[0],
    )
    print(
        "  samples:",
        counts.shape[1],
    )

    # ----------------------------------------------------------
    # 4. PAO1 gene-length recovery
    # ----------------------------------------------------------

    genbank = find_pao1_genbank()

    print()
    print("PAO1 annotation")
    print(
        "  GenBank:",
        genbank,
    )

    (
        length_map,
        ambiguous_lengths,
        feature_types,
    ) = build_length_map(
        genbank
    )

    resolved_genes = [
        gene
        for gene in counts.index
        if gene in length_map
    ]

    missing_length_genes = [
        gene
        for gene in counts.index
        if gene not in length_map
    ]

    ambiguous_model_genes = [
        gene
        for gene in counts.index
        if gene in ambiguous_lengths
    ]

    length_coverage = (
        len(
            resolved_genes
        )
        / len(
            counts.index
        )
    )

    model_coverage_after_length = (
        len(
            resolved_genes
        )
        / len(
            M.index
        )
    )

    print()
    print("Gene-length recovery")
    print(
        "  count-matrix genes:",
        len(
            counts.index
        ),
    )
    print(
        "  genes with unique annotation length:",
        len(
            resolved_genes
        ),
    )
    print(
        "  count-matrix length coverage:",
        format(
            length_coverage,
            ".2%",
        ),
    )
    print(
        "  full-model coverage after length restriction:",
        format(
            model_coverage_after_length,
            ".2%",
        ),
    )
    print(
        "  missing length genes:",
        len(
            missing_length_genes
        ),
    )
    print(
        "  ambiguous length genes:",
        len(
            ambiguous_model_genes
        ),
    )

    if missing_length_genes:
        print()
        print(
            "First missing-length genes:",
            ", ".join(
                missing_length_genes[
                    :40
                ]
            ),
        )

    if ambiguous_model_genes:
        print()
        print(
            "Ambiguous model-gene lengths:",
            ", ".join(
                ambiguous_model_genes[
                    :40
                ]
            ),
        )

    # ----------------------------------------------------------
    # 5. Freeze projection-compatible gene subset
    #
    # We do NOT invent lengths for unresolved genes.
    # ----------------------------------------------------------

    projection_genes = [
        gene
        for gene in M.index
        if gene in counts.index
        and gene in length_map
    ]

    projection_counts = counts.reindex(
        projection_genes
    )

    lengths = pd.Series(
        [
            float(
                length_map[
                    gene
                ]
            )
            for gene in projection_genes
        ],
        index=projection_genes,
        name="length_bp",
    )

    # ----------------------------------------------------------
    # 6. TPM normalization
    # ----------------------------------------------------------

    rpk = projection_counts.divide(
        lengths
        / 1000.0,
        axis=0,
    )

    denominators = rpk.sum(
        axis=0
    )

    if (
        denominators <= 0
    ).any():
        raise RuntimeError(
            "Invalid RPK denominator."
        )

    tpm = rpk.divide(
        denominators,
        axis=1,
    ) * 1_000_000.0

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    tpm_sums = tpm.sum(
        axis=0
    )

    print()
    print("TPM normalization")
    print(
        "  minimum TPM sum:",
        round(
            float(
                tpm_sums.min()
            ),
            3,
        ),
    )
    print(
        "  maximum TPM sum:",
        round(
            float(
                tpm_sums.max()
            ),
            3,
        ),
    )

    # ----------------------------------------------------------
    # 7. Remaining programme loading coverage
    # ----------------------------------------------------------

    M_shared = M.reindex(
        projection_genes
    )

    present_mask = np.array(
        [
            gene in set(
                projection_genes
            )
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

        absolute = np.abs(
            weights
        )

        squared = (
            weights
            ** 2
        )

        abs_coverage = float(
            absolute[
                present_mask
            ].sum()
            / absolute.sum()
        )

        energy_coverage = float(
            squared[
                present_mask
            ].sum()
            / squared.sum()
        )

        programme_rows.append(
            {
                "imodulon":
                    imodulon,

                "weighted_coverage_absM":
                    abs_coverage,

                "energy_coverage_M2":
                    energy_coverage,

                "phase3_primary":
                    imodulon
                    in phase3_primary,
            }
        )

    programme_coverage = pd.DataFrame(
        programme_rows
    )

    median_abs = float(
        programme_coverage[
            "weighted_coverage_absM"
        ].median()
    )

    minimum_abs = float(
        programme_coverage[
            "weighted_coverage_absM"
        ].min()
    )

    robust_here = int(
        (
            (
                programme_coverage[
                    "weighted_coverage_absM"
                ]
                >= 0.90
            )
            & (
                programme_coverage[
                    "energy_coverage_M2"
                ]
                >= 0.80
            )
        ).sum()
    )

    print()
    print("Projection-compatible iModulon coverage")
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
        "  programmes meeting >=90% |M| and >=80% M^2:",
        robust_here,
        "/ 116",
    )

    # ----------------------------------------------------------
    # 8. Matched expression contrasts
    #
    # delta = direct CF-lung expression minus matched
    # culture-derived isolate-pool expression.
    # ----------------------------------------------------------

    M_array = M_shared.to_numpy(
        dtype=float
    )

    pinv_M = np.linalg.pinv(
        M_array
    )

    contrast_x = {}

    contrast_a = {}

    qc_rows = []

    paired_expression_rows = []

    for region in REGIONS:
        invivo = (
            "invivo_"
            + region
        )

        invitro = (
            "invitro_"
            + region
        )

        paired_r = float(
            log_tpm[
                [
                    invivo,
                    invitro,
                ]
            ]
            .corr()
            .iloc[
                0,
                1,
            ]
        )

        paired_expression_rows.append(
            {
                "region":
                    region,

                "invivo":
                    invivo,

                "invitro":
                    invitro,

                "expression_correlation":
                    paired_r,
            }
        )

        delta = (
            log_tpm[
                invivo
            ]
            - log_tpm[
                invitro
            ]
        )

        delta_array = delta.to_numpy(
            dtype=float
        )

        activity = (
            pinv_M
            @ delta_array
        )

        reconstruction = (
            M_array
            @ activity
        )

        projection_corr = vector_corr(
            delta_array,
            reconstruction,
        )

        relative_error = float(
            np.linalg.norm(
                delta_array
                - reconstruction
            )
            / np.linalg.norm(
                delta_array
            )
        )

        energy_fraction = float(
            np.linalg.norm(
                reconstruction
            )
            ** 2
            / (
                np.linalg.norm(
                    delta_array
                )
                ** 2
            )
        )

        contrast_x[
            region
        ] = delta

        contrast_a[
            region
        ] = pd.Series(
            activity,
            index=imodulons,
            name=region,
        )

        qc_rows.append(
            {
                "region":
                    region,

                "paired_expression_correlation":
                    paired_r,

                "projection_corr":
                    projection_corr,

                "relative_projection_error":
                    relative_error,

                "projection_energy_fraction":
                    energy_fraction,
            }
        )

    contrast_qc = pd.DataFrame(
        qc_rows
    )

    paired_expression = pd.DataFrame(
        paired_expression_rows
    )

    print()
    print("Matched regional contrast projection QC")
    print(
        contrast_qc
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Pooled in-vivo minus pooled in-vitro contrast
    #
    # This is not a replacement for the regional contrasts.
    # It gives one aggregate technical diagnostic.
    # ----------------------------------------------------------

    pooled_delta = (
        log_tpm[
            INVIVO
        ].mean(
            axis=1
        )
        - log_tpm[
            INVITRO
        ].mean(
            axis=1
        )
    )

    pooled_activity_array = (
        pinv_M
        @ pooled_delta.to_numpy(
            dtype=float
        )
    )

    pooled_reconstruction = (
        M_array
        @ pooled_activity_array
    )

    pooled_corr = vector_corr(
        pooled_delta,
        pooled_reconstruction,
    )

    pooled_error = float(
        np.linalg.norm(
            pooled_delta.to_numpy(
                dtype=float
            )
            - pooled_reconstruction
        )
        / np.linalg.norm(
            pooled_delta.to_numpy(
                dtype=float
            )
        )
    )

    pooled_energy = float(
        np.linalg.norm(
            pooled_reconstruction
        )
        ** 2
        / (
            np.linalg.norm(
                pooled_delta.to_numpy(
                    dtype=float
                )
            )
            ** 2
        )
    )

    pooled_activity = pd.Series(
        pooled_activity_array,
        index=imodulons,
        name="pooled_invivo_minus_invitro",
    )

    print()
    print("Pooled in-vivo minus in-vitro projection")
    print(
        "  projection correlation:",
        round(
            pooled_corr,
            4,
        ),
    )
    print(
        "  relative projection error:",
        round(
            pooled_error,
            4,
        ),
    )
    print(
        "  projection energy fraction:",
        round(
            pooled_energy,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 10. Algebra consistency
    #
    # Because projection is linear:
    #
    # projection(mean regional delta)
    #
    # must equal
    #
    # mean(projection(each regional delta)).
    # ----------------------------------------------------------

    activity_matrix = pd.DataFrame(
        contrast_a
    )

    mean_regional_activity = activity_matrix.mean(
        axis=1
    )

    algebra_difference = float(
        np.max(
            np.abs(
                mean_regional_activity.to_numpy(
                    dtype=float
                )
                - pooled_activity.to_numpy(
                    dtype=float
                )
            )
        )
    )

    print()
    print("Projection linearity check")
    print(
        "  max pooled-vs-mean-region activity difference:",
        format(
            algebra_difference,
            ".3e",
        ),
    )

    # ----------------------------------------------------------
    # 11. Regional heterogeneity
    #
    # Descriptive only.
    # There is deliberately NO similarity gate here.
    # ----------------------------------------------------------

    activity_correlation = (
        activity_matrix.corr()
    )

    expression_contrast_matrix = pd.DataFrame(
        contrast_x
    )

    expression_contrast_correlation = (
        expression_contrast_matrix.corr()
    )

    print()
    print("Regional contrast activity correlation")
    print(
        activity_correlation
        .round(
            3
        )
        .to_string()
    )

    print()
    print("Regional contrast expression correlation")
    print(
        expression_contrast_correlation
        .round(
            3
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 12. Descriptive activity magnitude
    # ----------------------------------------------------------

    activity_summary_rows = []

    for region in REGIONS:
        vector = activity_matrix[
            region
        ]

        activity_summary_rows.append(
            {
                "region":
                    region,

                "activity_L2_norm":
                    float(
                        np.linalg.norm(
                            vector.to_numpy(
                                dtype=float
                            )
                        )
                    ),

                "activity_RMS":
                    float(
                        np.sqrt(
                            np.mean(
                                vector.to_numpy(
                                    dtype=float
                                )
                                ** 2
                            )
                        )
                    ),
            }
        )

    activity_summary = pd.DataFrame(
        activity_summary_rows
    )

    print()
    print("Regional programme-shift magnitude")
    print(
        activity_summary
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 13. Save outputs
    # ----------------------------------------------------------

    length_table = pd.DataFrame(
        {
            "gene":
                projection_genes,

            "length_bp":
                [
                    length_map[
                        gene
                    ]
                    for gene in projection_genes
                ],
        }
    )

    length_table.to_csv(
        OUT
        / "GSE119356_projection_gene_lengths.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "missing_length_gene":
                missing_length_genes
        }
    ).to_csv(
        OUT
        / "GSE119356_missing_gene_lengths.csv",
        index=False,
    )

    projection_counts.to_csv(
        OUT
        / "GSE119356_projection_compatible_counts.csv"
    )

    tpm.to_csv(
        OUT
        / "GSE119356_projection_compatible_TPM.csv"
    )

    log_tpm.to_csv(
        OUT
        / "GSE119356_projection_log2TPM_plus1.csv"
    )

    programme_coverage.to_csv(
        OUT
        / "GSE119356_projection_imodulon_coverage.csv",
        index=False,
    )

    paired_expression.to_csv(
        OUT
        / "matched_region_expression_correlation.csv",
        index=False,
    )

    contrast_qc.to_csv(
        OUT
        / "matched_region_contrast_projection_QC.csv",
        index=False,
    )

    expression_contrast_matrix.to_csv(
        OUT
        / "matched_region_expression_contrasts.csv"
    )

    activity_matrix.to_csv(
        OUT
        / "matched_region_imodulon_contrasts.csv"
    )

    pooled_activity.to_csv(
        OUT
        / "pooled_invivo_minus_invitro_imodulon_contrast.csv",
        header=True,
    )

    activity_correlation.to_csv(
        OUT
        / "regional_contrast_activity_correlation.csv"
    )

    expression_contrast_correlation.to_csv(
        OUT
        / "regional_contrast_expression_correlation.csv"
    )

    activity_summary.to_csv(
        OUT
        / "regional_programme_shift_magnitude.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 14. Technical decision
    #
    # We deliberately do NOT require the five lung regions to
    # resemble one another.
    #
    # Spatial heterogeneity is a Phase-4 biological output.
    #
    # The pooled contrast is used only to establish that the
    # overall in-vivo/culture difference has substantial support
    # in the frozen iModulon basis.
    # ----------------------------------------------------------

    minimum_projection_depth = float(
        projection_counts.sum(
            axis=0
        ).min()
    )

    finite_tpm = bool(
        np.isfinite(
            tpm.to_numpy(
                dtype=float
            )
        ).all()
    )

    finite_activity = bool(
        np.isfinite(
            activity_matrix.to_numpy(
                dtype=float
            )
        ).all()
    )

    checks = {
        "gene_length_coverage_ge_98pct":
            length_coverage
            >= 0.98,

        "full_model_coverage_after_length_ge_95pct":
            model_coverage_after_length
            >= 0.95,

        "minimum_projection_counts_ge_50k":
            minimum_projection_depth
            >= 50000,

        "TPM_sums_equal_1e6":
            bool(
                np.allclose(
                    tpm_sums.to_numpy(
                        dtype=float
                    ),
                    1_000_000.0,
                    rtol=1e-8,
                    atol=1e-3,
                )
            ),

        "median_imodulon_absM_coverage_ge_95pct":
            median_abs
            >= 0.95,

        "minimum_imodulon_absM_coverage_ge_85pct":
            minimum_abs
            >= 0.85,

        "at_least_110_programmes_robust":
            robust_here
            >= 110,

        "pooled_projection_corr_ge_040":
            pooled_corr
            >= 0.40,

        "pooled_projection_energy_ge_015":
            pooled_energy
            >= 0.15,

        "projection_linearity":
            algebra_difference
            < 1e-8,

        "finite_TPM":
            finite_tpm,

        "finite_activities":
            finite_activity,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 02a checks")

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
        "GSE119356_MATCHED_CF_LUNG_PROJECTION_ACCEPTED"
        if passed
        else "GSE119356_MATCHED_CF_LUNG_PROJECTION_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "NOTE: no biological interpretation of programme identities "
        "is made in Audit 02a. Regional heterogeneity is retained "
        "rather than treated as replicate noise."
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
        / "AUDIT_02A_STATUS.txt"
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
            "Failed Audit 02a checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
