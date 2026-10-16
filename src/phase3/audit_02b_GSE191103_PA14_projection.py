from pathlib import Path
from io import StringIO
import json
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


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

TRANSFER = (
    ROOT
    / "results/phase3/audit_01b_PA14_final_transferability"
    / "PA14_final_imodulon_transferability.csv"
)

BRIDGE_LENGTHS = (
    ROOT
    / "results/phase3/audit_02c_PA14_old_locus_annotation_bridge"
    / "GSE191103_PA14_old_locus_gene_lengths.csv"
)

PA14_GB = (
    ROOT
    / "references/mappings/NC_008463.1.gb"
)

PA14_GB_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    "efetch.fcgi?db=nuccore&id=NC_008463.1"
    "&rettype=gbwithparts&retmode=text"
)

GSE203348 = (
    ROOT
    / "results/phase2/audit_03_external_pao1_activity_inference"
    / "GSE203348_activity_summary.csv"
)

GSE120602 = (
    ROOT
    / "results/phase2/audit_04_controlled_state_anchoring_GSE120602"
    / "controlled_contrast_vectors.csv"
)

GSE299861 = (
    ROOT
    / "results/phase2/audit_12f_GSE299861_canonical_imodulon_challenge"
    / "GSE299861_biofilm_programme_vector.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_02b_GSE191103_PA14_projection"
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


def download_reference():
    if PA14_GB.exists():
        print(
            "PA14 reference already present:"
        )
        print(
            " ",
            PA14_GB,
        )
        return

    PA14_GB.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Downloading PA14 NC_008463.1 GenBank reference ..."
    )

    request = urllib.request.Request(
        PA14_GB_URL,
        headers={
            "User-Agent":
                "CF-State Phase3 Audit02b"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        PA14_GB.write_bytes(
            response.read()
        )

    print(
        "Downloaded:",
        PA14_GB,
    )


def pa14_gene_lengths():
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython is required."
        )

    record = SeqIO.read(
        str(PA14_GB),
        "genbank",
    )

    lengths = {}

    feature_types = {}

    for feature in record.features:
        tags = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if not tags:
            continue

        locus = str(
            tags[0]
        ).strip()

        length = int(
            len(
                feature.extract(
                    record.seq
                )
            )
        )

        if length <= 0:
            continue

        priority = {
            "CDS": 10,
            "rRNA": 9,
            "tRNA": 9,
            "tmRNA": 9,
            "ncRNA": 9,
            "gene": 1,
        }

        rank = priority.get(
            feature.type,
            0,
        )

        previous_type = feature_types.get(
            locus
        )

        previous_rank = (
            priority.get(
                previous_type,
                0,
            )
            if previous_type is not None
            else -1
        )

        if rank > previous_rank:
            lengths[
                locus
            ] = length

            feature_types[
                locus
            ] = feature.type

    return (
        record,
        pd.Series(
            lengths,
            name="length_bp",
            dtype=float,
        ),
        feature_types,
    )


def mean_pairwise_corr(
    matrix,
    columns,
):
    corr = matrix[
        columns
    ].corr()

    values = []

    for i in range(
        len(columns)
    ):
        for j in range(
            i + 1,
            len(columns),
        ):
            values.append(
                float(
                    corr.iloc[
                        i,
                        j,
                    ]
                )
            )

    return float(
        np.mean(
            values
        )
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

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom == 0:
        return np.nan

    return float(
        np.dot(
            a,
            b,
        )
        / denom
    )


def sign_agreement(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    return float(
        np.mean(
            np.sign(a)
            == np.sign(b)
        )
    )


def read_indexed_vector(
    path,
    column,
):
    table = pd.read_csv(
        path
    )

    index_col = None

    for candidate in [
        "imodulon",
        "iModulon",
        "Unnamed: 0",
    ]:
        if candidate in table.columns:
            index_col = candidate
            break

    if index_col is None:
        raise RuntimeError(
            "Could not identify iModulon column in "
            + str(
                path
            )
        )

    table = table.set_index(
        index_col
    )

    if column not in table.columns:
        raise RuntimeError(
            str(
                path
            )
            + " missing expected column "
            + column
        )

    vector = pd.to_numeric(
        table[
            column
        ],
        errors="raise",
    )

    vector.index = (
        vector.index
        .astype(str)
        .str.strip()
    )

    return vector


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02b")
    print("GSE191103 WT PA14 -> PAO1 iModulon projection")
    print("=" * 78)

    for path in [
        MODEL,
        EXPRESSION,
        STRICT_MAP,
        TRANSFER,
        BRIDGE_LENGTHS,
        GSE203348,
        GSE120602,
        GSE299861,
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

    download_reference()

    # ----------------------------------------------------------
    # 1. Frozen PAO1 model
    # ----------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        model_raw = json.load(
            handle
        )

    M_full = json_df(
        model_raw[
            "M"
        ]
    )

    A_train = json_df(
        model_raw[
            "A"
        ]
    )

    M_full.index = (
        M_full.index
        .astype(str)
        .str.strip()
    )

    M_full.columns = (
        M_full.columns
        .astype(str)
        .str.strip()
    )

    A_train.index = (
        A_train.index
        .astype(str)
        .str.strip()
    )

    A_train = A_train.reindex(
        M_full.columns
    )

    if M_full.shape != (
        5572,
        116,
    ):
        raise RuntimeError(
            "Unexpected aeruPRECISE M shape."
        )

    print()
    print("Frozen PAO1 aeruPRECISE representation")
    print(
        "  genes:",
        M_full.shape[
            0
        ],
    )
    print(
        "  iModulons:",
        M_full.shape[
            1
        ],
    )

    # ----------------------------------------------------------
    # 2. PA14 raw unique-read count matrix
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

    if (
        counts.to_numpy(
            dtype=float
        )
        < 0
    ).any():
        raise RuntimeError(
            "Negative values found in raw count matrix."
        )

    biofilm_cols = [
        col
        for col in counts.columns
        if "PA14_BF_" in col
    ]

    planktonic_cols = [
        col
        for col in counts.columns
        if "PA14_PL_" in col
    ]

    if len(
        biofilm_cols
    ) != 3:
        raise RuntimeError(
            "Expected exactly three WT biofilm samples."
        )

    if len(
        planktonic_cols
    ) != 3:
        raise RuntimeError(
            "Expected exactly three WT planktonic samples."
        )

    print()
    print("Raw PA14 count matrix")
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
    # 3. PA14 old-locus gene lengths and TPM
    #
    # GSE191103 uses the historical PA14_##### locus-tag
    # namespace. Audit 02c established an unambiguous bridge
    # from these expression identifiers to current RefSeq
    # annotation and gene lengths.
    # ----------------------------------------------------------

    length_table = pd.read_csv(
        BRIDGE_LENGTHS
    )

    required_length_columns = {
        "old_locus_tag",
        "length_bp",
    }

    missing_length_columns = (
        required_length_columns
        - set(
            length_table.columns
        )
    )

    if missing_length_columns:
        raise RuntimeError(
            "PA14 length bridge missing columns: "
            + ", ".join(
                sorted(
                    missing_length_columns
                )
            )
        )

    length_table[
        "old_locus_tag"
    ] = (
        length_table[
            "old_locus_tag"
        ]
        .astype(str)
        .str.strip()
    )

    length_table[
        "length_bp"
    ] = pd.to_numeric(
        length_table[
            "length_bp"
        ],
        errors="raise",
    )

    if length_table[
        "old_locus_tag"
    ].duplicated().any():
        raise RuntimeError(
            "Old-locus length bridge contains duplicate PA14 IDs."
        )

    if (
        length_table[
            "length_bp"
        ]
        <= 0
    ).any():
        raise RuntimeError(
            "Old-locus length bridge contains non-positive lengths."
        )

    lengths = pd.Series(
        length_table[
            "length_bp"
        ].to_numpy(
            dtype=float
        ),
        index=length_table[
            "old_locus_tag"
        ],
        name="length_bp",
    )

    shared_length_genes = (
        counts.index.intersection(
            lengths.index
        )
    )

    length_coverage = (
        len(
            shared_length_genes
        )
        / counts.shape[
            0
        ]
    )

    # Also check how much of the actual sequencing count mass
    # is retained after excluding expression IDs for which the
    # historical/current annotation bridge cannot provide a
    # reliable length.
    total_counts = counts.sum(
        axis=0
    )

    retained_counts = counts.loc[
        shared_length_genes
    ].sum(
        axis=0
    )

    count_mass_retained = (
        retained_counts
        / total_counts
    )

    minimum_count_mass_retained = float(
        count_mass_retained.min()
    )

    print()
    print("PA14 historical-locus length bridge")
    print(
        "  expression genes:",
        counts.shape[
            0
        ],
    )
    print(
        "  expression genes with verified lengths:",
        len(
            shared_length_genes
        ),
    )
    print(
        "  expression gene-length coverage:",
        format(
            length_coverage,
            ".2%",
        ),
    )
    print(
        "  minimum count mass retained:",
        format(
            minimum_count_mass_retained,
            ".2%",
        ),
    )

    print()
    print("Count mass retained by sample")
    print(
        count_mass_retained
        .round(
            4
        )
        .to_string()
    )

    if length_coverage < 0.95:
        raise RuntimeError(
            "Insufficient PA14 old-locus gene-length coverage "
            "for reliable TPM calculation."
        )

    if minimum_count_mass_retained < 0.95:
        raise RuntimeError(
            "Genes lacking verified lengths account for more "
            "than 5% of sequencing count mass in at least one "
            "sample. Inspect before TPM normalization."
        )

    counts_for_tpm = counts.loc[
        shared_length_genes
    ].copy()

    lengths_for_tpm = lengths.reindex(
        shared_length_genes
    )

    length_kb = (
        lengths_for_tpm
        / 1000.0
    )

    rpk = counts_for_tpm.divide(
        length_kb,
        axis=0,
    )

    rpk_sums = rpk.sum(
        axis=0
    )

    if (
        rpk_sums
        <= 0
    ).any():
        raise RuntimeError(
            "Invalid RPK normalization denominator."
        )

    tpm = rpk.divide(
        rpk_sums,
        axis=1,
    ) * 1_000_000.0

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    print()
    print("PA14 normalization")
    print(
        "  TPM sum minimum:",
        round(
            float(
                tpm.sum(
                    axis=0
                ).min()
            ),
            3,
        ),
    )
    print(
        "  TPM sum maximum:",
        round(
            float(
                tpm.sum(
                    axis=0
                ).max()
            ),
            3,
        ),
    )

    expression_bio_r = mean_pairwise_corr(
        log_tpm,
        biofilm_cols,
    )

    expression_plank_r = mean_pairwise_corr(
        log_tpm,
        planktonic_cols,
    )

    print(
        "  planktonic log2(TPM+1) replicate mean r:",
        round(
            expression_plank_r,
            4,
        ),
    )
    print(
        "  biofilm log2(TPM+1) replicate mean r:",
        round(
            expression_bio_r,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 4. Strict PA14 -> PAO1 mapping
    # ----------------------------------------------------------

    mapping = pd.read_csv(
        STRICT_MAP
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
            "Strict map has duplicate PAO1 genes."
        )

    if mapping[
        "pa14_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Strict map has duplicate PA14 genes."
        )

    usable_map = mapping.loc[
        mapping[
            "pa14_gene"
        ].isin(
            log_tpm.index
        )
    ].copy()

    pa14_to_pao1 = dict(
        zip(
            usable_map[
                "pa14_gene"
            ],
            usable_map[
                "pao1_gene"
            ],
        )
    )

    projected_input = log_tpm.loc[
        usable_map[
            "pa14_gene"
        ]
    ].copy()

    projected_input.index = [
        pa14_to_pao1[
            gene
        ]
        for gene in projected_input.index
    ]

    projected_input = projected_input.reindex(
        [
            gene
            for gene in M_full.index
            if gene in projected_input.index
        ]
    )

    if projected_input.index.duplicated().any():
        raise RuntimeError(
            "Duplicate PAO1 genes after strict mapping."
        )

    mapped_gene_coverage = (
        projected_input.shape[
            0
        ]
        / M_full.shape[
            0
        ]
    )

    print()
    print("Strict cross-strain expression map")
    print(
        "  mapped expression genes:",
        projected_input.shape[
            0
        ],
    )
    print(
        "  PAO1 model gene coverage:",
        format(
            mapped_gene_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # 5. Within-experiment planktonic centering
    # ----------------------------------------------------------

    plank_reference = projected_input[
        planktonic_cols
    ].mean(
        axis=1
    )

    X_centered = projected_input.subtract(
        plank_reference,
        axis=0,
    )

    max_plank_gene_residual = float(
        X_centered[
            planktonic_cols
        ].mean(
            axis=1
        ).abs().max()
    )

    # ----------------------------------------------------------
    # 6. Projection using only conserved PAO1 model rows
    # ----------------------------------------------------------

    M_shared = M_full.reindex(
        projected_input.index
    )

    M_array = M_shared.to_numpy(
        dtype=float
    )

    X_array = X_centered.to_numpy(
        dtype=float
    )

    A_external_array = (
        np.linalg.pinv(
            M_array
        )
        @ X_array
    )

    A_external = pd.DataFrame(
        A_external_array,
        index=M_shared.columns,
        columns=projected_input.columns,
    )

    reconstructed = (
        M_array
        @ A_external_array
    )

    projection_corr = float(
        np.corrcoef(
            X_array.ravel(),
            reconstructed.ravel(),
        )[
            0,
            1
        ]
    )

    relative_error = float(
        np.linalg.norm(
            X_array
            - reconstructed
        )
        / np.linalg.norm(
            X_array
        )
    )

    training_sd = float(
        np.nanstd(
            A_train.to_numpy(
                dtype=float
            )
        )
    )

    external_sd = float(
        np.nanstd(
            A_external_array
        )
    )

    scale_ratio = (
        external_sd
        / training_sd
    )

    centered_plank_activity_r = (
        mean_pairwise_corr(
            A_external,
            planktonic_cols,
        )
    )

    biofilm_activity_r = (
        mean_pairwise_corr(
            A_external,
            biofilm_cols,
        )
    )

    max_mean_plank_activity = float(
        A_external[
            planktonic_cols
        ].mean(
            axis=1
        ).abs().max()
    )

    print()
    print("PA14 -> PAO1 iModulon projection QC")
    print(
        "  projection correlation:",
        round(
            projection_corr,
            4,
        ),
    )
    print(
        "  relative projection error:",
        round(
            relative_error,
            4,
        ),
    )
    print(
        "  external/training activity SD ratio:",
        round(
            scale_ratio,
            4,
        ),
    )
    print(
        "  centered planktonic activity r "
        "(diagnostic only):",
        round(
            centered_plank_activity_r,
            4,
        ),
    )
    print(
        "  biofilm activity replicate mean r:",
        round(
            biofilm_activity_r,
            4,
        ),
    )
    print(
        "  max |mean planktonic gene residual|:",
        format(
            max_plank_gene_residual,
            ".3e",
        ),
    )
    print(
        "  max |mean planktonic activity|:",
        format(
            max_mean_plank_activity,
            ".3e",
        ),
    )

    # ----------------------------------------------------------
    # 7. Biofilm programme response
    # ----------------------------------------------------------

    delta = (
        A_external[
            biofilm_cols
        ].mean(
            axis=1
        )
        - A_external[
            planktonic_cols
        ].mean(
            axis=1
        )
    )

    delta.name = (
        "delta_biofilm_minus_planktonic"
    )

    transfer = pd.read_csv(
        TRANSFER
    ).set_index(
        "imodulon"
    )

    transfer.index = (
        transfer.index
        .astype(str)
        .str.strip()
    )

    programme = pd.DataFrame(
        {
            "imodulon":
                delta.index,

            "delta_biofilm_minus_planktonic":
                delta.values,

            "weighted_transfer_coverage":
                transfer.reindex(
                    delta.index
                )[
                    "weighted_coverage_absM"
                ].values,

            "energy_transfer_coverage":
                transfer.reindex(
                    delta.index
                )[
                    "energy_coverage_M2"
                ].values,

            "transfer_status":
                transfer.reindex(
                    delta.index
                )[
                    "transfer_status"
                ].values,
        }
    )

    programme[
        "abs_delta"
    ] = (
        programme[
            "delta_biofilm_minus_planktonic"
        ].abs()
    )

    programme = programme.sort_values(
        "abs_delta",
        ascending=False,
    )

    print()
    print("Largest PA14 biofilm programme changes")
    print(
        programme[
            [
                "imodulon",
                "delta_biofilm_minus_planktonic",
                "weighted_transfer_coverage",
                "energy_transfer_coverage",
                "transfer_status",
            ]
        ]
        .head(
            25
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 8. Cross-study comparisons with PAO1 biofilms
    # ----------------------------------------------------------

    reference_vectors = {
        "GSE203348":
            read_indexed_vector(
                GSE203348,
                "delta_biofilm_minus_planktonic",
            ),

        "GSE120602":
            read_indexed_vector(
                GSE120602,
                "biofilm_vs_planktonic",
            ),

        "GSE299861":
            read_indexed_vector(
                GSE299861,
                "delta_biofilm_minus_planktonic",
            ),
    }

    comparison_rows = []

    for name, vector in reference_vectors.items():
        aligned = vector.reindex(
            delta.index
        )

        if aligned.isna().any():
            raise RuntimeError(
                name
                + " is missing iModulon values."
            )

        a = delta.to_numpy(
            dtype=float
        )

        b = aligned.to_numpy(
            dtype=float
        )

        comparison_rows.append(
            {
                "comparison":
                    "GSE191103_PA14_vs_"
                    + name,

                "pearson":
                    float(
                        pearsonr(
                            a,
                            b,
                        ).statistic
                    ),

                "spearman":
                    float(
                        spearmanr(
                            a,
                            b,
                        ).statistic
                    ),

                "cosine":
                    cosine(
                        a,
                        b,
                    ),

                "sign_agreement":
                    sign_agreement(
                        a,
                        b,
                    ),
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    print()
    print("Cross-study biofilm agreement")
    print(
        comparison
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Sensitivity: exclude MODERATE-transfer programmes
    # ----------------------------------------------------------

    high_programmes = (
        transfer.index[
            transfer[
                "transfer_status"
            ]
            == "HIGH"
        ]
        .intersection(
            delta.index
        )
    )

    high_comparison_rows = []

    for name, vector in reference_vectors.items():
        a = delta.reindex(
            high_programmes
        ).to_numpy(
            dtype=float
        )

        b = vector.reindex(
            high_programmes
        ).to_numpy(
            dtype=float
        )

        high_comparison_rows.append(
            {
                "comparison":
                    "GSE191103_PA14_vs_"
                    + name,

                "n_HIGH_programmes":
                    len(
                        high_programmes
                    ),

                "pearson_HIGH_only":
                    float(
                        pearsonr(
                            a,
                            b,
                        ).statistic
                    ),

                "spearman_HIGH_only":
                    float(
                        spearmanr(
                            a,
                            b,
                        ).statistic
                    ),

                "cosine_HIGH_only":
                    cosine(
                        a,
                        b,
                    ),

                "sign_agreement_HIGH_only":
                    sign_agreement(
                        a,
                        b,
                    ),
            }
        )

    high_comparison = pd.DataFrame(
        high_comparison_rows
    )

    print()
    print("Cross-study agreement using HIGH-transfer programmes only")
    print(
        high_comparison
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 10. Save outputs
    # ----------------------------------------------------------

    tpm.to_csv(
        OUT
        / "GSE191103_WT_PA14_TPM.csv"
    )

    log_tpm.to_csv(
        OUT
        / "GSE191103_WT_PA14_log2TPM_plus1.csv"
    )

    projected_input.to_csv(
        OUT
        / "GSE191103_WT_PA14_as_PAO1_expression.csv"
    )

    A_external.to_csv(
        OUT
        / "GSE191103_WT_PA14_imodulon_activities.csv"
    )

    programme.drop(
        columns=[
            "abs_delta"
        ]
    ).to_csv(
        OUT
        / "GSE191103_WT_PA14_biofilm_programme_vector.csv",
        index=False,
    )

    comparison.to_csv(
        OUT
        / "cross_study_biofilm_agreement.csv",
        index=False,
    )

    high_comparison.to_csv(
        OUT
        / "cross_study_biofilm_agreement_HIGH_transfer_only.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 11. Audit gates
    # ----------------------------------------------------------

    checks = {
        "six_WT_samples":
            counts.shape[
                1
            ]
            == 6,

        "PA14_length_coverage_ge_95pct":
            length_coverage
            >= 0.95,

        "PA14_count_mass_retained_ge_95pct":
            minimum_count_mass_retained
            >= 0.95,

        "strict_cross_strain_model_coverage_ge_90pct":
            mapped_gene_coverage
            >= 0.90,

        "planktonic_expression_replicate_r_ge_075":
            expression_plank_r
            >= 0.75,

        "biofilm_expression_replicate_r_ge_075":
            expression_bio_r
            >= 0.75,

        "projection_corr_ge_040":
            projection_corr
            >= 0.40,

        "biofilm_activity_replicate_r_ge_075":
            biofilm_activity_r
            >= 0.75,

        "planktonic_reference_centered":
            max_mean_plank_activity
            < 1e-8,

        "finite_projection":
            bool(
                np.isfinite(
                    A_external_array
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
        "# CF-State Phase 3 — Audit 02b",
        "",
        "## GSE191103 WT PA14 cross-strain iModulon projection",
        "",
        "Raw unique gene-read counts were normalized using "
        "PA14 NC_008463.1 feature lengths to TPM, transformed "
        "as log2(TPM+1), mapped through the frozen strict "
        "PAO1↔PA14 one-to-one orthologue core, centered to the "
        "within-experiment planktonic reference, and projected "
        "into the frozen PAO1 aeruPRECISE representation.",
        "",
        "- PA14 expression genes with verified old-locus lengths: "
        + format(
            length_coverage,
            ".2%",
        ),
        "- Minimum sequencing count mass retained: "
        + format(
            minimum_count_mass_retained,
            ".2%",
        ),
        "- PAO1 model gene coverage after strict mapping: "
        + format(
            mapped_gene_coverage,
            ".2%",
        ),
        "- Planktonic expression replicate correlation: "
        + format(
            expression_plank_r,
            ".4f",
        ),
        "- Biofilm expression replicate correlation: "
        + format(
            expression_bio_r,
            ".4f",
        ),
        "- Projection correlation: "
        + format(
            projection_corr,
            ".4f",
        ),
        "- Relative projection error: "
        + format(
            relative_error,
            ".4f",
        ),
        "- External/training activity SD ratio: "
        + format(
            scale_ratio,
            ".4f",
        ),
        "- Biofilm activity replicate correlation: "
        + format(
            biofilm_activity_r,
            ".4f",
        ),
        "",
        "Programme-specific PAO1→PA14 transferability scores "
        "are retained in the output and should be considered "
        "when interpreting individual iModulons.",
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
        / "AUDIT_02B_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Phase 3 Audit 02b:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_02B_SUMMARY.md",
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
