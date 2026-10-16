from pathlib import Path
from io import BytesIO, StringIO
import gzip
import json
import tarfile

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path.cwd()

ARCHIVE = (
    ROOT
    / "data/raw/phase3/GSE55197/GSE55197_RAW.tar"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

EXP_STAT_COUNTS = (
    ROOT
    / "results/phase3/audit_03c_GSE55197_composite_ID_resolution"
    / "GSE55197_stationary_exponential_normalized_locus_counts.csv"
)

PROJECTION_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

TRANSFER = (
    ROOT
    / "results/phase3/audit_03c_GSE55197_composite_ID_resolution"
    / "GSE55197_exact_imodulon_transferability.csv"
)

PA14_STATIONARY = (
    ROOT
    / "results/phase3/audit_03d_GSE55197_stationary_crossstrain_projection"
    / "GSE55197_PA14_stationary_programme_vector.csv"
)

PHASE2_ROOT = (
    ROOT
    / "results/phase2"
)

OUT = (
    ROOT
    / "results/phase3/audit_03e_GSE55197_anoxic_crossstrain_projection"
)


ANOXIC_GSM = {
    "GSM1331354": "PA14_ANOX_1",
    "GSM1331355": "PA14_ANOX_2",
}

EXPONENTIAL = [
    "PA14_EXP_1",
    "PA14_EXP_2",
]

ANOXIC = [
    "PA14_ANOX_1",
    "PA14_ANOX_2",
]

ALL_SAMPLES = (
    EXPONENTIAL
    + ANOXIC
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


def normalize_identifier(value):
    text = str(
        value
    ).strip()

    if "," in text:
        locus = text.split(
            ",",
            1,
        )[0].strip()

        return locus

    return text


def parse_processed_member(
    raw_bytes,
    filename,
):
    if filename.endswith(
        ".gz"
    ):
        raw_bytes = gzip.decompress(
            raw_bytes
        )

    text = raw_bytes.decode(
        "utf-8",
        errors="replace",
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.startswith(
            "#"
        )
    ]

    table = pd.read_csv(
        StringIO(
            "\n".join(
                lines
            )
        ),
        sep=r"\s+",
        header=None,
        engine="python",
    )

    if table.shape[1] < 2:
        raise RuntimeError(
            "Unexpected processed file format: "
            + filename
        )

    ids = (
        table.iloc[:, 0]
        .astype(str)
        .map(
            normalize_identifier
        )
    )

    values = pd.to_numeric(
        table.iloc[:, 1],
        errors="coerce",
    )

    valid = values.notna()

    result = pd.Series(
        values.loc[
            valid
        ].to_numpy(
            dtype=float
        ),
        index=ids.loc[
            valid
        ],
        name=filename,
    )

    if result.index.duplicated().any():
        raise RuntimeError(
            filename
            + " creates duplicate locus tags after "
            "composite-ID normalization."
        )

    if (
        result
        < 0
    ).any():
        raise RuntimeError(
            "Negative counts in "
            + filename
        )

    return result


def pair_r(
    matrix,
    sample_1,
    sample_2,
):
    return float(
        matrix[
            [
                sample_1,
                sample_2,
            ]
        ]
        .corr()
        .iloc[
            0,
            1,
        ]
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


def permutation_alignment_p(
    a,
    b,
    n_perm=10000,
    seed=20261007,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    observed = cosine(
        a,
        b,
    )

    rng = np.random.default_rng(
        seed
    )

    null = np.empty(
        n_perm,
        dtype=float,
    )

    for i in range(
        n_perm
    ):
        null[i] = cosine(
            a,
            rng.permutation(
                b
            ),
        )

    p = (
        1
        + int(
            np.sum(
                null
                >= observed
            )
        )
    ) / (
        n_perm
        + 1
    )

    return (
        observed,
        float(p),
    )


def read_vector(
    path,
    column,
    imodulons,
):
    table = pd.read_csv(
        path
    )

    index_column = None

    for candidate in [
        "imodulon",
        "iModulon",
        "Unnamed: 0",
    ]:
        if candidate in table.columns:
            index_column = candidate
            break

    if index_column is None:
        raise RuntimeError(
            "Could not identify iModulon column in "
            + str(path)
        )

    if column not in table.columns:
        raise RuntimeError(
            "Missing column "
            + column
            + " in "
            + str(path)
        )

    table[index_column] = (
        table[index_column]
        .astype(str)
        .str.strip()
    )

    vector = pd.to_numeric(
        table.set_index(
            index_column
        )[column],
        errors="raise",
    )

    vector = vector.reindex(
        imodulons
    )

    if vector.isna().any():
        raise RuntimeError(
            "Comparator does not contain all iModulons: "
            + str(path)
        )

    return vector


def discover_microaerophilic_axis(
    imodulons,
):
    """
    Search frozen Phase-2 outputs for the GSE152295
    microaerophilic controlled axis.
    """

    preferred_columns = [
        "microaerophilic",
        "microaerophilic_vs_ctrl",
        "microaerophilic_vs_control",
        "microaerophilic_vs_Ctrl",
        "microaerophilic_minus_ctrl",
        "microaerophilic_minus_control",
    ]

    candidates = []

    for path in PHASE2_ROOT.rglob(
        "*.csv"
    ):
        try:
            table = pd.read_csv(
                path
            )
        except Exception:
            continue

        if table.empty:
            continue

        index_column = None

        for possible in [
            "imodulon",
            "iModulon",
            "Unnamed: 0",
        ]:
            if possible in table.columns:
                observed = set(
                    table[possible]
                    .astype(str)
                    .str.strip()
                )

                overlap = len(
                    observed.intersection(
                        set(
                            imodulons
                        )
                    )
                )

                if overlap >= 110:
                    index_column = possible
                    break

        if index_column is None:
            continue

        for column in preferred_columns:
            if column in table.columns:
                score = 0

                path_lower = str(
                    path
                ).lower()

                if "gse152295" in path_lower:
                    score += 100

                if "10c" in path_lower:
                    score += 50

                if "axis" in path_lower:
                    score += 10

                if "dictionary" in path_lower:
                    score += 10

                candidates.append(
                    (
                        score,
                        path,
                        column,
                    )
                )

    if not candidates:
        return (
            None,
            None,
            None,
        )

    candidates = sorted(
        candidates,
        key=lambda x: (
            -x[0],
            len(
                str(
                    x[1]
                )
            ),
        ),
    )

    _, path, column = candidates[0]

    vector = read_vector(
        path,
        column,
        imodulons,
    )

    return (
        vector,
        path,
        column,
    )


def compare_vectors(
    target,
    reference,
    name,
    programme_set,
):
    a = target.reindex(
        programme_set
    ).to_numpy(
        dtype=float
    )

    b = reference.reindex(
        programme_set
    ).to_numpy(
        dtype=float
    )

    observed_cosine, p = (
        permutation_alignment_p(
            a,
            b,
        )
    )

    return {
        "comparison":
            name,

        "programme_set":
            programme_set.name,

        "n_programmes":
            len(
                programme_set
            ),

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
            observed_cosine,

        "sign_agreement":
            sign_agreement(
                a,
                b,
            ),

        "programme_label_permutation_p":
            p,
    }


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 03e")
    print("GSE55197 PA14 anoxic cross-strain oxygen-state projection")
    print("=" * 78)

    for path in [
        ARCHIVE,
        MODEL,
        EXP_STAT_COUNTS,
        PROJECTION_MAP,
        TRANSFER,
        PA14_STATIONARY,
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
    # 1. Frozen PAO1 representation
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

    A_train = json_df(
        raw[
            "A"
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

    A_train.index = (
        A_train.index
        .astype(str)
        .str.strip()
    )

    A_train = A_train.reindex(
        M.columns
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
    # 2. Existing normalized exponential samples
    # ----------------------------------------------------------

    exp_stat = pd.read_csv(
        EXP_STAT_COUNTS,
        index_col=0,
    )

    exp_stat.index = (
        exp_stat.index
        .astype(str)
        .str.strip()
    )

    exponential_counts = exp_stat[
        EXPONENTIAL
    ].copy()

    # ----------------------------------------------------------
    # 3. Extract + normalize anoxic samples from same archive
    # ----------------------------------------------------------

    anoxic_vectors = {}

    manifest_rows = []

    with tarfile.open(
        ARCHIVE,
        "r",
    ) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
        ]

        for gsm, sample in ANOXIC_GSM.items():
            matches = [
                member
                for member in members
                if Path(
                    member.name
                ).name.startswith(
                    gsm
                )
            ]

            if len(
                matches
            ) != 1:
                raise RuntimeError(
                    gsm
                    + ": expected exactly one archive member, found "
                    + str(
                        [
                            member.name
                            for member in matches
                        ]
                    )
                )

            member = matches[
                0
            ]

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                raise RuntimeError(
                    "Could not extract "
                    + member.name
                )

            vector = parse_processed_member(
                extracted.read(),
                member.name,
            )

            anoxic_vectors[
                sample
            ] = vector

            manifest_rows.append(
                {
                    "gsm":
                        gsm,

                    "sample":
                        sample,

                    "archive_file":
                        member.name,

                    "genes":
                        len(
                            vector
                        ),

                    "total_reads_per_gene":
                        float(
                            vector.sum()
                        ),

                    "detected_genes":
                        int(
                            (
                                vector
                                > 0
                            ).sum()
                        ),
                }
            )

    manifest = pd.DataFrame(
        manifest_rows
    )

    print()
    print("Anoxic samples")
    print(
        manifest.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. Check normalized gene universes
    # ----------------------------------------------------------

    gene_sets = [
        set(
            exponential_counts.index
        )
    ]

    for vector in anoxic_vectors.values():
        gene_sets.append(
            set(
                vector.index
            )
        )

    common_genes = set.intersection(
        *gene_sets
    )

    union_genes = set.union(
        *gene_sets
    )

    print()
    print("Normalized gene universe")
    print(
        "  common genes:",
        len(
            common_genes
        ),
    )
    print(
        "  union genes:",
        len(
            union_genes
        ),
    )

    # ----------------------------------------------------------
    # 5. Frozen 5286-gene projection universe
    # ----------------------------------------------------------

    projection = pd.read_csv(
        PROJECTION_MAP
    )

    for column in [
        "pa14_gene",
        "pao1_gene",
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

    projection[
        "length_bp"
    ] = pd.to_numeric(
        projection[
            "length_bp"
        ],
        errors="raise",
    )

    if len(
        projection
    ) != 5286:
        raise RuntimeError(
            "Expected frozen 5286-gene projection universe."
        )

    pa14_order = list(
        projection[
            "pa14_gene"
        ]
    )

    pao1_order = list(
        projection[
            "pao1_gene"
        ]
    )

    # All 5286 were present in the exponential data in Audit 03c.
    missing_exp = [
        gene
        for gene in pa14_order
        if gene not in exponential_counts.index
    ]

    if missing_exp:
        raise RuntimeError(
            "Exponential samples missing frozen projection genes: "
            + str(
                len(
                    missing_exp
                )
            )
        )

    for sample, vector in anoxic_vectors.items():
        missing = [
            gene
            for gene in pa14_order
            if gene not in vector.index
        ]

        if missing:
            raise RuntimeError(
                sample
                + " missing frozen projection genes: "
                + str(
                    len(
                        missing
                    )
                )
            )

    counts = pd.DataFrame(
        index=pa14_order
    )

    counts[
        "PA14_EXP_1"
    ] = exponential_counts.loc[
        pa14_order,
        "PA14_EXP_1",
    ]

    counts[
        "PA14_EXP_2"
    ] = exponential_counts.loc[
        pa14_order,
        "PA14_EXP_2",
    ]

    for sample in ANOXIC:
        counts[
            sample
        ] = anoxic_vectors[
            sample
        ].reindex(
            pa14_order
        )

    if counts.isna().any().any():
        raise RuntimeError(
            "Missing values in frozen projection count matrix."
        )

    print()
    print("Frozen cross-strain projection universe")
    print(
        "  genes:",
        counts.shape[
            0
        ],
    )

    print()
    print("Model-compatible count depth")
    print(
        counts.sum(
            axis=0
        ).to_string()
    )

    # ----------------------------------------------------------
    # 6. Projection-compatible TPM
    # ----------------------------------------------------------

    lengths = pd.Series(
        projection[
            "length_bp"
        ].to_numpy(
            dtype=float
        ),
        index=pa14_order,
    )

    rpk = counts.divide(
        lengths
        / 1000.0,
        axis=0,
    )

    denominator = rpk.sum(
        axis=0
    )

    if (
        denominator
        <= 0
    ).any():
        raise RuntimeError(
            "Invalid RPK denominator."
        )

    tpm = rpk.divide(
        denominator,
        axis=1,
    ) * 1_000_000.0

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    exp_expression_r = pair_r(
        log_tpm,
        "PA14_EXP_1",
        "PA14_EXP_2",
    )

    anox_expression_r = pair_r(
        log_tpm,
        "PA14_ANOX_1",
        "PA14_ANOX_2",
    )

    print()
    print("Projection-compatible expression")
    print(
        "  exponential replicate r:",
        round(
            exp_expression_r,
            4,
        ),
    )
    print(
        "  anoxic replicate r:",
        round(
            anox_expression_r,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 7. Rename into PAO1 model space + exponential centering
    # ----------------------------------------------------------

    counts.index = pao1_order
    tpm.index = pao1_order
    log_tpm.index = pao1_order

    M_shared = M.reindex(
        pao1_order
    )

    if M_shared.isna().any().any():
        raise RuntimeError(
            "Missing model values after PAO1 alignment."
        )

    exp_reference = log_tpm[
        EXPONENTIAL
    ].mean(
        axis=1
    )

    X_centered = log_tpm.subtract(
        exp_reference,
        axis=0,
    )

    # ----------------------------------------------------------
    # 8. iModulon projection
    # ----------------------------------------------------------

    M_array = M_shared.to_numpy(
        dtype=float
    )

    X_array = X_centered.to_numpy(
        dtype=float
    )

    pinv_M = np.linalg.pinv(
        M_array
    )

    A_array = (
        pinv_M
        @ X_array
    )

    activities = pd.DataFrame(
        A_array,
        index=M.columns,
        columns=ALL_SAMPLES,
    )

    reconstruction = (
        M_array
        @ A_array
    )

    full_projection_corr = vector_corr(
        X_array.ravel(),
        reconstruction.ravel(),
    )

    full_relative_error = float(
        np.linalg.norm(
            X_array
            - reconstruction
        )
        / np.linalg.norm(
            X_array
        )
    )

    anox_activity_r = pair_r(
        activities,
        "PA14_ANOX_1",
        "PA14_ANOX_2",
    )

    max_exp_activity_residual = float(
        activities[
            EXPONENTIAL
        ]
        .mean(
            axis=1
        )
        .abs()
        .max()
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
            A_array
        )
    )

    scale_ratio = (
        external_sd
        / training_sd
    )

    print()
    print("Projection QC")
    print(
        "  anoxic activity replicate r:",
        round(
            anox_activity_r,
            4,
        ),
    )
    print(
        "  full centered projection correlation:",
        round(
            full_projection_corr,
            4,
        ),
    )
    print(
        "  full relative projection error:",
        round(
            full_relative_error,
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
        "  max |mean exponential activity|:",
        format(
            max_exp_activity_residual,
            ".3e",
        ),
    )

    # ----------------------------------------------------------
    # 9. Anoxic minus exponential contrast
    # ----------------------------------------------------------

    delta_x = (
        log_tpm[
            ANOXIC
        ].mean(
            axis=1
        )
        - log_tpm[
            EXPONENTIAL
        ].mean(
            axis=1
        )
    )

    delta_a_array = (
        pinv_M
        @ delta_x.to_numpy(
            dtype=float
        )
    )

    anoxic_delta = pd.Series(
        delta_a_array,
        index=M.columns,
        name="anoxic_vs_exponential",
    )

    delta_reconstruction = (
        M_array
        @ delta_a_array
    )

    contrast_projection_corr = vector_corr(
        delta_x.to_numpy(
            dtype=float
        ),
        delta_reconstruction,
    )

    contrast_relative_error = float(
        np.linalg.norm(
            delta_x.to_numpy(
                dtype=float
            )
            - delta_reconstruction
        )
        / np.linalg.norm(
            delta_x.to_numpy(
                dtype=float
            )
        )
    )

    contrast_energy_fraction = float(
        np.linalg.norm(
            delta_reconstruction
        )
        ** 2
        / (
            np.linalg.norm(
                delta_x.to_numpy(
                    dtype=float
                )
            )
            ** 2
        )
    )

    print()
    print("Anoxic-minus-exponential contrast projection")
    print(
        "  corr(delta X, M @ delta A):",
        round(
            contrast_projection_corr,
            4,
        ),
    )
    print(
        "  relative projection error:",
        round(
            contrast_relative_error,
            4,
        ),
    )
    print(
        "  fraction contrast energy in PAO1 iModulon span:",
        round(
            contrast_energy_fraction,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 10. Cross-strain reliability programme set
    # ----------------------------------------------------------

    transfer = pd.read_csv(
        TRANSFER
    )

    transfer[
        "imodulon"
    ] = (
        transfer[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    transfer = transfer.set_index(
        "imodulon"
    ).reindex(
        imodulons
    )

    robust_mask = (
        (
            transfer[
                "weighted_coverage_absM"
            ]
            >= 0.90
        )
        & (
            transfer[
                "energy_coverage_M2"
            ]
            >= 0.80
        )
    )

    all_programmes = pd.Index(
        imodulons,
        name="ALL_116",
    )

    robust_programmes = pd.Index(
        transfer.index[
            robust_mask
        ],
        name="ROBUST_TRANSFER",
    )

    print()
    print("Programme reliability")
    print(
        "  all programmes:",
        len(
            all_programmes
        ),
    )
    print(
        "  robust programmes:",
        len(
            robust_programmes
        ),
    )

    # ----------------------------------------------------------
    # 11. Largest PA14 anoxic programme changes
    # ----------------------------------------------------------

    programme_table = pd.DataFrame(
        {
            "imodulon":
                imodulons,

            "anoxic_vs_exponential":
                anoxic_delta.reindex(
                    imodulons
                ).values,

            "weighted_transfer_coverage":
                transfer[
                    "weighted_coverage_absM"
                ].values,

            "energy_transfer_coverage":
                transfer[
                    "energy_coverage_M2"
                ].values,

            "robust_transfer":
                robust_mask.values,
        }
    )

    programme_table[
        "abs_delta"
    ] = programme_table[
        "anoxic_vs_exponential"
    ].abs()

    programme_table = programme_table.sort_values(
        "abs_delta",
        ascending=False,
    )

    print()
    print("Largest PA14 anoxic programme changes")
    print(
        programme_table[
            [
                "imodulon",
                "anoxic_vs_exponential",
                "weighted_transfer_coverage",
                "energy_transfer_coverage",
                "robust_transfer",
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
    # 12. Existing PA14 stationary vector
    # ----------------------------------------------------------

    pa14_stationary = read_vector(
        PA14_STATIONARY,
        "stationary_vs_exponential",
        imodulons,
    )

    # ----------------------------------------------------------
    # 13. Find independent PAO1 microaerophilic axis
    # ----------------------------------------------------------

    (
        microaerophilic,
        micro_path,
        micro_column,
    ) = discover_microaerophilic_axis(
        imodulons
    )

    print()
    print("PAO1 oxygen-limitation comparator")

    if microaerophilic is not None:
        print(
            "  file:",
            micro_path,
        )
        print(
            "  column:",
            micro_column,
        )
    else:
        print(
            "  WARNING: microaerophilic Phase-2 axis "
            "not automatically located."
        )

    # ----------------------------------------------------------
    # 14. How stationary-like is the PA14 anoxic state?
    # ----------------------------------------------------------

    comparison_rows = []

    for programme_set in [
        all_programmes,
        robust_programmes,
    ]:
        comparison_rows.append(
            compare_vectors(
                anoxic_delta,
                pa14_stationary,
                "PA14_anoxic_vs_PA14_stationary",
                programme_set,
            )
        )

    # ----------------------------------------------------------
    # 15. Raw anoxic vs PAO1 microaerophilic
    # ----------------------------------------------------------

    if microaerophilic is not None:
        for programme_set in [
            all_programmes,
            robust_programmes,
        ]:
            comparison_rows.append(
                compare_vectors(
                    anoxic_delta,
                    microaerophilic,
                    "PA14_anoxic_vs_PAO1_microaerophilic",
                    programme_set,
                )
            )

    # ----------------------------------------------------------
    # 16. Remove PA14 stationary-aligned component
    #
    # This is a GEOMETRIC sensitivity analysis, not a causal
    # correction for medium or growth phase.
    # ----------------------------------------------------------

    stat_array = pa14_stationary.reindex(
        imodulons
    ).to_numpy(
        dtype=float
    )

    anox_array = anoxic_delta.reindex(
        imodulons
    ).to_numpy(
        dtype=float
    )

    stat_denominator = float(
        np.dot(
            stat_array,
            stat_array,
        )
    )

    if stat_denominator <= 0:
        raise RuntimeError(
            "Stationary vector has zero norm."
        )

    stationary_beta = float(
        np.dot(
            anox_array,
            stat_array,
        )
        / stat_denominator
    )

    stationary_component = (
        stationary_beta
        * stat_array
    )

    anoxic_residual_array = (
        anox_array
        - stationary_component
    )

    anoxic_residual = pd.Series(
        anoxic_residual_array,
        index=imodulons,
        name="anoxic_residual_after_stationary",
    )

    stationary_energy_fraction = float(
        np.linalg.norm(
            stationary_component
        )
        ** 2
        / (
            np.linalg.norm(
                anox_array
            )
            ** 2
        )
    )

    residual_energy_fraction = float(
        np.linalg.norm(
            anoxic_residual_array
        )
        ** 2
        / (
            np.linalg.norm(
                anox_array
            )
            ** 2
        )
    )

    print()
    print("Stationary-component sensitivity analysis")
    print(
        "  stationary projection coefficient beta:",
        round(
            stationary_beta,
            4,
        ),
    )
    print(
        "  anoxic programme energy along stationary direction:",
        round(
            stationary_energy_fraction,
            4,
        ),
    )
    print(
        "  residual anoxic programme energy:",
        round(
            residual_energy_fraction,
            4,
        ),
    )

    if microaerophilic is not None:
        for programme_set in [
            all_programmes,
            robust_programmes,
        ]:
            comparison_rows.append(
                compare_vectors(
                    anoxic_residual,
                    microaerophilic,
                    "PA14_anoxic_residual_vs_PAO1_microaerophilic",
                    programme_set,
                )
            )

    comparisons = pd.DataFrame(
        comparison_rows
    )

    print()
    print("Cross-state / cross-strain programme agreement")
    print(
        comparisons
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 17. Save
    # ----------------------------------------------------------

    manifest.to_csv(
        OUT
        / "GSE55197_anoxic_sample_manifest.csv",
        index=False,
    )

    counts.to_csv(
        OUT
        / "GSE55197_anoxic_projection_gene_counts_PAO1_space.csv"
    )

    tpm.to_csv(
        OUT
        / "GSE55197_anoxic_projection_compatible_TPM_PAO1_space.csv"
    )

    log_tpm.to_csv(
        OUT
        / "GSE55197_anoxic_projection_log2TPM_plus1_PAO1_space.csv"
    )

    activities.to_csv(
        OUT
        / "GSE55197_anoxic_imodulon_activities.csv"
    )

    programme_table.drop(
        columns=[
            "abs_delta"
        ]
    ).to_csv(
        OUT
        / "GSE55197_PA14_anoxic_programme_vector.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "imodulon":
                imodulons,

            "anoxic_vs_exponential":
                anoxic_delta.reindex(
                    imodulons
                ).values,

            "stationary_component":
                stationary_component,

            "anoxic_residual_after_stationary":
                anoxic_residual_array,
        }
    ).to_csv(
        OUT
        / "GSE55197_anoxic_stationary_residual_decomposition.csv",
        index=False,
    )

    comparisons.to_csv(
        OUT
        / "GSE55197_anoxic_crossstrain_agreement.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 18. Technical acceptance only
    #
    # Neither agreement with stationary nor agreement with the
    # PAO1 oxygen axis is used as a technical PASS criterion.
    # ----------------------------------------------------------

    minimum_depth = float(
        counts.sum(
            axis=0
        ).min()
    )

    checks = {
        "projection_gene_count_5286":
            counts.shape[
                0
            ]
            == 5286,

        "minimum_model_count_depth_ge_100k":
            minimum_depth
            >= 100000,

        "exponential_expression_r_ge_090":
            exp_expression_r
            >= 0.90,

        "anoxic_expression_r_ge_090":
            anox_expression_r
            >= 0.90,

        "anoxic_activity_r_ge_075":
            anox_activity_r
            >= 0.75,

        "exponential_reference_centered":
            max_exp_activity_residual
            < 1e-8,

        "finite_projection":
            bool(
                np.isfinite(
                    A_array
                ).all()
            ),

        "contrast_projection_corr_ge_040":
            contrast_projection_corr
            >= 0.40,

        "contrast_energy_fraction_ge_015":
            contrast_energy_fraction
            >= 0.15,

        "robust_programmes_ge_90":
            len(
                robust_programmes
            )
            >= 90,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Technical audit checks")

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
        "GSE55197_ANOXIC_PROJECTION_ACCEPTED"
        if passed
        else "GSE55197_ANOXIC_PROJECTION_REVIEW_REQUIRED"
    )

    print()
    print(
        "TECHNICAL STATUS:",
        status,
    )

    print()
    print(
        "NOTE: the PA14 anoxic contrast is not a pure oxygen "
        "perturbation because medium and growth state differ from "
        "the exponential reference. Cross-strain oxygen agreement "
        "and stationary-residual agreement are scientific "
        "sensitivity analyses, not technical gates."
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

    summary = [
        "# CF-State Phase 3 — Audit 03e",
        "",
        "## GSE55197 PA14 anoxic cross-strain oxygen-state projection",
        "",
        "PA14 anoxic cultures were compared with the existing "
        "PA14 exponential reference and projected through the "
        "frozen 5,286-gene PA14↔PAO1 universe.",
        "",
        "Important design limitation: anoxic cultures differ from "
        "the exponential reference in oxygen availability, medium, "
        "and growth state. Therefore this is interpreted as an "
        "oxygen-associated state-family test, not a pure oxygen "
        "causal contrast.",
        "",
        "- Exponential expression replicate r: "
        + format(
            exp_expression_r,
            ".4f",
        ),
        "- Anoxic expression replicate r: "
        + format(
            anox_expression_r,
            ".4f",
        ),
        "- Anoxic activity replicate r: "
        + format(
            anox_activity_r,
            ".4f",
        ),
        "- Contrast projection correlation: "
        + format(
            contrast_projection_corr,
            ".4f",
        ),
        "- Contrast energy fraction: "
        + format(
            contrast_energy_fraction,
            ".4f",
        ),
        "- Stationary-direction energy fraction: "
        + format(
            stationary_energy_fraction,
            ".4f",
        ),
        "- Residual anoxic energy fraction: "
        + format(
            residual_energy_fraction,
            ".4f",
        ),
        "",
        "## Technical decision",
        "",
        status,
    ]

    (
        OUT
        / "AUDIT_03E_SUMMARY.md"
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
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed technical checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
