from pathlib import Path
from io import StringIO
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

PA14_ANOXIC_DECOMP = (
    ROOT
    / "results/phase3/audit_03e_GSE55197_anoxic_crossstrain_projection"
    / "GSE55197_anoxic_stationary_residual_decomposition.csv"
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
    / "results/phase3/audit_03g_GSE55197_PA14_biofilm_composition"
)


SAMPLES = {
    "GSM1331360": {
        "sample": "PA14_BF24_1",
        "condition": "biofilm_24h",
    },
    "GSM1331361": {
        "sample": "PA14_BF24_2",
        "condition": "biofilm_24h",
    },
    "GSM1331362": {
        "sample": "PA14_BF48_1",
        "condition": "biofilm_48h",
    },
    "GSM1331363": {
        "sample": "PA14_BF48_2",
        "condition": "biofilm_48h",
    },
}

EXP = [
    "PA14_EXP_1",
    "PA14_EXP_2",
]

BF24 = [
    "PA14_BF24_1",
    "PA14_BF24_2",
]

BF48 = [
    "PA14_BF48_1",
    "PA14_BF48_2",
]

ALL_SAMPLES = (
    EXP
    + BF24
    + BF48
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
        return text.split(
            ",",
            1,
        )[0].strip()

    return text


def parse_member(raw_bytes, filename):
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
            + " creates duplicate locus tags."
        )

    return result


def pair_r(matrix, a, b):
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


def permutation_p(
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
                null >= observed
            )
        )
    ) / (
        n_perm
        + 1
    )

    return observed, float(p)


def read_vector(
    path,
    value_column,
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
            "Cannot identify iModulon column in "
            + str(path)
        )

    if value_column not in table.columns:
        raise RuntimeError(
            str(path)
            + " missing column "
            + value_column
        )

    table[index_column] = (
        table[index_column]
        .astype(str)
        .str.strip()
    )

    vector = pd.to_numeric(
        table.set_index(
            index_column
        )[value_column],
        errors="raise",
    ).reindex(
        imodulons
    )

    if vector.isna().any():
        raise RuntimeError(
            "Missing iModulons in "
            + str(path)
        )

    return vector


def compare(
    target,
    reference,
    comparison_name,
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

    observed_cosine, p = permutation_p(
        a,
        b,
    )

    return {
        "comparison":
            comparison_name,

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

        "permutation_p":
            p,
    }


def decompose_two_anchor(
    target,
    stationary,
    oxygen,
):
    target_array = target.to_numpy(
        dtype=float
    )

    stationary_array = stationary.to_numpy(
        dtype=float
    )

    oxygen_array = oxygen.to_numpy(
        dtype=float
    )

    anchor_matrix = np.column_stack(
        [
            stationary_array,
            oxygen_array,
        ]
    )

    coefficients = (
        np.linalg.pinv(
            anchor_matrix
        )
        @ target_array
    )

    reconstruction = (
        anchor_matrix
        @ coefficients
    )

    residual = (
        target_array
        - reconstruction
    )

    target_energy = float(
        np.dot(
            target_array,
            target_array,
        )
    )

    reconstruction_energy = float(
        np.dot(
            reconstruction,
            reconstruction,
        )
    )

    stationary_component = (
        coefficients[0]
        * stationary_array
    )

    oxygen_component = (
        coefficients[1]
        * oxygen_array
    )

    stationary_energy = float(
        np.dot(
            stationary_component,
            stationary_component,
        )
        / target_energy
    )

    oxygen_energy = float(
        np.dot(
            oxygen_component,
            oxygen_component,
        )
        / target_energy
    )

    explained_energy = (
        reconstruction_energy
        / target_energy
    )

    residual_energy = float(
        np.dot(
            residual,
            residual,
        )
        / target_energy
    )

    return {
        "stationary_coefficient":
            float(
                coefficients[0]
            ),

        "oxygen_coefficient":
            float(
                coefficients[1]
            ),

        "stationary_energy_fraction":
            stationary_energy,

        "oxygen_energy_fraction":
            oxygen_energy,

        "anchor_explained_energy_fraction":
            explained_energy,

        "residual_energy_fraction":
            residual_energy,

        "reconstruction_cosine":
            cosine(
                target_array,
                reconstruction,
            ),

        "residual":
            residual,

        "reconstruction":
            reconstruction,
    }


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 03g")
    print("GSE55197 PA14 biofilm composition in conserved state space")
    print("=" * 78)

    for path in [
        ARCHIVE,
        MODEL,
        EXP_STAT_COUNTS,
        PROJECTION_MAP,
        TRANSFER,
        PA14_STATIONARY,
        PA14_ANOXIC_DECOMP,
        GSE203348,
        GSE120602,
        GSE299861,
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
        raw["M"]
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
    # 2. Exponential reference samples
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

    # ----------------------------------------------------------
    # 3. Extract 24h and 48h biofilm samples
    # ----------------------------------------------------------

    vectors = {}

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

        for gsm, metadata in SAMPLES.items():
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
                    + ": expected one archive file, found "
                    + str(
                        [
                            item.name
                            for item in matches
                        ]
                    )
                )

            member = matches[0]

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                raise RuntimeError(
                    "Unable to extract "
                    + member.name
                )

            vector = parse_member(
                extracted.read(),
                member.name,
            )

            sample = metadata[
                "sample"
            ]

            vectors[
                sample
            ] = vector

            manifest_rows.append(
                {
                    "gsm":
                        gsm,

                    "sample":
                        sample,

                    "condition":
                        metadata[
                            "condition"
                        ],

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
                                vector > 0
                            ).sum()
                        ),
                }
            )

    manifest = pd.DataFrame(
        manifest_rows
    )

    print()
    print("PA14 biofilm samples")
    print(
        manifest.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. Frozen projection universe
    # ----------------------------------------------------------

    projection = pd.read_csv(
        PROJECTION_MAP
    )

    for column in [
        "pa14_gene",
        "pao1_gene",
    ]:
        projection[column] = (
            projection[column]
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
            "Expected 5286 frozen projection genes."
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

    # Require the complete accepted universe.
    for sample, vector in vectors.items():
        missing = [
            gene
            for gene in pa14_order
            if gene not in vector.index
        ]

        if missing:
            raise RuntimeError(
                sample
                + " missing "
                + str(
                    len(
                        missing
                    )
                )
                + " frozen projection genes."
            )

    counts = pd.DataFrame(
        index=pa14_order
    )

    counts[
        "PA14_EXP_1"
    ] = exp_stat.loc[
        pa14_order,
        "PA14_EXP_1",
    ]

    counts[
        "PA14_EXP_2"
    ] = exp_stat.loc[
        pa14_order,
        "PA14_EXP_2",
    ]

    for sample in (
        BF24
        + BF48
    ):
        counts[
            sample
        ] = vectors[
            sample
        ].reindex(
            pa14_order
        )

    print()
    print("Model-compatible count depth")
    print(
        counts.sum(
            axis=0
        ).to_string()
    )

    # ----------------------------------------------------------
    # 5. Projection-compatible TPM
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

    tpm = rpk.divide(
        rpk.sum(
            axis=0
        ),
        axis=1,
    ) * 1_000_000.0

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    exp_r = pair_r(
        log_tpm,
        "PA14_EXP_1",
        "PA14_EXP_2",
    )

    bf24_r = pair_r(
        log_tpm,
        "PA14_BF24_1",
        "PA14_BF24_2",
    )

    bf48_r = pair_r(
        log_tpm,
        "PA14_BF48_1",
        "PA14_BF48_2",
    )

    print()
    print("Projection-compatible expression replicate QC")
    print(
        "  exponential r:",
        round(
            exp_r,
            4,
        ),
    )
    print(
        "  biofilm 24h r:",
        round(
            bf24_r,
            4,
        ),
    )
    print(
        "  biofilm 48h r:",
        round(
            bf48_r,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 6. Move into PAO1 model space and project
    # ----------------------------------------------------------

    counts.index = pao1_order
    tpm.index = pao1_order
    log_tpm.index = pao1_order

    M_shared = M.reindex(
        pao1_order
    )

    M_array = M_shared.to_numpy(
        dtype=float
    )

    pinv_M = np.linalg.pinv(
        M_array
    )

    exp_reference = log_tpm[
        EXP
    ].mean(
        axis=1
    )

    X_centered = log_tpm.subtract(
        exp_reference,
        axis=0,
    )

    A_array = (
        pinv_M
        @ X_centered.to_numpy(
            dtype=float
        )
    )

    activities = pd.DataFrame(
        A_array,
        index=imodulons,
        columns=ALL_SAMPLES,
    )

    bf24_activity_r = pair_r(
        activities,
        "PA14_BF24_1",
        "PA14_BF24_2",
    )

    bf48_activity_r = pair_r(
        activities,
        "PA14_BF48_1",
        "PA14_BF48_2",
    )

    max_reference_residual = float(
        activities[
            EXP
        ]
        .mean(
            axis=1
        )
        .abs()
        .max()
    )

    print()
    print("Programme-level replicate QC")
    print(
        "  biofilm 24h activity r:",
        round(
            bf24_activity_r,
            4,
        ),
    )
    print(
        "  biofilm 48h activity r:",
        round(
            bf48_activity_r,
            4,
        ),
    )

    # ----------------------------------------------------------
    # 7. Group-mean biofilm vectors
    # ----------------------------------------------------------

    bf24_x = (
        log_tpm[
            BF24
        ].mean(
            axis=1
        )
        - exp_reference
    )

    bf48_x = (
        log_tpm[
            BF48
        ].mean(
            axis=1
        )
        - exp_reference
    )

    bf24_a = pd.Series(
        pinv_M
        @ bf24_x.to_numpy(
            dtype=float
        ),
        index=imodulons,
        name="biofilm24_vs_exponential",
    )

    bf48_a = pd.Series(
        pinv_M
        @ bf48_x.to_numpy(
            dtype=float
        ),
        index=imodulons,
        name="biofilm48_vs_exponential",
    )

    maturation_a = (
        bf48_a
        - bf24_a
    )

    maturation_a.name = (
        "biofilm48_minus_biofilm24"
    )

    # ----------------------------------------------------------
    # 8. Projection fit of each biofilm state
    # ----------------------------------------------------------

    fit_rows = []

    for name, x, a in [
        (
            "biofilm_24h",
            bf24_x,
            bf24_a,
        ),
        (
            "biofilm_48h",
            bf48_x,
            bf48_a,
        ),
    ]:
        x_array = x.to_numpy(
            dtype=float
        )

        recon = (
            M_array
            @ a.to_numpy(
                dtype=float
            )
        )

        fit_rows.append(
            {
                "state":
                    name,

                "projection_corr":
                    vector_corr(
                        x_array,
                        recon,
                    ),

                "relative_error":
                    float(
                        np.linalg.norm(
                            x_array
                            - recon
                        )
                        / np.linalg.norm(
                            x_array
                        )
                    ),

                "projection_energy_fraction":
                    float(
                        np.linalg.norm(
                            recon
                        )
                        ** 2
                        / (
                            np.linalg.norm(
                                x_array
                            )
                            ** 2
                        )
                    ),
            }
        )

    fit_table = pd.DataFrame(
        fit_rows
    )

    print()
    print("Biofilm contrast projection fit")
    print(
        fit_table
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Frozen reliable programme set
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

    # ----------------------------------------------------------
    # 10. Conserved-state anchors
    #
    # stationary = strong cross-strain validation
    #
    # oxygen residual = PA14 anoxic response after removal of
    # stationary direction; independently validated against the
    # PAO1 microaerophilic programme family in Audit 03f.
    # ----------------------------------------------------------

    stationary = read_vector(
        PA14_STATIONARY,
        "stationary_vs_exponential",
        imodulons,
    )

    oxygen_residual = read_vector(
        PA14_ANOXIC_DECOMP,
        "anoxic_residual_after_stationary",
        imodulons,
    )

    anchor_dot = float(
        np.dot(
            stationary.to_numpy(
                dtype=float
            ),
            oxygen_residual.to_numpy(
                dtype=float
            ),
        )
    )

    anchor_norm_product = float(
        np.linalg.norm(
            stationary
        )
        * np.linalg.norm(
            oxygen_residual
        )
    )

    anchor_cosine = (
        anchor_dot
        / anchor_norm_product
    )

    print()
    print("Conserved-state anchor geometry")
    print(
        "  stationary vs oxygen-residual cosine:",
        format(
            anchor_cosine,
            ".4e",
        ),
    )

    # ----------------------------------------------------------
    # 11. Biofilm composition
    # ----------------------------------------------------------

    decomposition_rows = []

    residual_vectors = {}

    for name, vector in [
        (
            "biofilm_24h",
            bf24_a,
        ),
        (
            "biofilm_48h",
            bf48_a,
        ),
    ]:
        result = decompose_two_anchor(
            vector,
            stationary,
            oxygen_residual,
        )

        residual_vectors[
            name
        ] = pd.Series(
            result[
                "residual"
            ],
            index=imodulons,
            name=name + "_residual",
        )

        decomposition_rows.append(
            {
                "state":
                    name,

                "stationary_coefficient":
                    result[
                        "stationary_coefficient"
                    ],

                "oxygen_coefficient":
                    result[
                        "oxygen_coefficient"
                    ],

                "stationary_energy_fraction":
                    result[
                        "stationary_energy_fraction"
                    ],

                "oxygen_energy_fraction":
                    result[
                        "oxygen_energy_fraction"
                    ],

                "combined_anchor_energy_fraction":
                    result[
                        "anchor_explained_energy_fraction"
                    ],

                "residual_energy_fraction":
                    result[
                        "residual_energy_fraction"
                    ],

                "anchor_reconstruction_cosine":
                    result[
                        "reconstruction_cosine"
                    ],
            }
        )

    decomposition = pd.DataFrame(
        decomposition_rows
    )

    print()
    print("PA14 biofilm conserved-state composition")
    print(
        decomposition
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 12. Biofilm maturation
    # ----------------------------------------------------------

    maturation_similarity = {
        "bf24_vs_bf48_pearson":
            vector_corr(
                bf24_a,
                bf48_a,
            ),

        "bf24_vs_bf48_cosine":
            cosine(
                bf24_a,
                bf48_a,
            ),

        "maturation_vs_stationary_cosine":
            cosine(
                maturation_a,
                stationary,
            ),

        "maturation_vs_oxygen_residual_cosine":
            cosine(
                maturation_a,
                oxygen_residual,
            ),
    }

    print()
    print("Biofilm maturation geometry")

    for key, value in maturation_similarity.items():
        print(
            " ",
            key,
            ":",
            round(
                value,
                4,
            ),
        )

    # ----------------------------------------------------------
    # 13. Largest raw and residual programmes
    # ----------------------------------------------------------

    programme_table = pd.DataFrame(
        {
            "imodulon":
                imodulons,

            "biofilm24_vs_exponential":
                bf24_a.values,

            "biofilm48_vs_exponential":
                bf48_a.values,

            "biofilm48_minus_biofilm24":
                maturation_a.values,

            "biofilm24_residual":
                residual_vectors[
                    "biofilm_24h"
                ].values,

            "biofilm48_residual":
                residual_vectors[
                    "biofilm_48h"
                ].values,

            "robust_transfer":
                robust_mask.values,
        }
    )

    print()
    print("Largest 24h biofilm programmes")
    print(
        programme_table.assign(
            magnitude=programme_table[
                "biofilm24_vs_exponential"
            ].abs()
        )
        .sort_values(
            "magnitude",
            ascending=False,
        )[
            [
                "imodulon",
                "biofilm24_vs_exponential",
                "robust_transfer",
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
    print("Largest 48h biofilm programmes")
    print(
        programme_table.assign(
            magnitude=programme_table[
                "biofilm48_vs_exponential"
            ].abs()
        )
        .sort_values(
            "magnitude",
            ascending=False,
        )[
            [
                "imodulon",
                "biofilm48_vs_exponential",
                "robust_transfer",
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
    print("Largest 48h residual programmes after conserved anchors")
    print(
        programme_table.assign(
            magnitude=programme_table[
                "biofilm48_residual"
            ].abs()
        )
        .sort_values(
            "magnitude",
            ascending=False,
        )[
            [
                "imodulon",
                "biofilm48_residual",
                "robust_transfer",
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
    # 14. Compare with independent PAO1 biofilm vectors
    # ----------------------------------------------------------

    references = {
        "GSE203348_PAO1_biofilm":
            read_vector(
                GSE203348,
                "delta_biofilm_minus_planktonic",
                imodulons,
            ),

        "GSE120602_PAO1_biofilm":
            read_vector(
                GSE120602,
                "biofilm_vs_planktonic",
                imodulons,
            ),

        "GSE299861_PAO1_biofilm":
            read_vector(
                GSE299861,
                "delta_biofilm_minus_planktonic",
                imodulons,
            ),
    }

    comparison_rows = []

    targets = {
        "PA14_biofilm24_raw":
            bf24_a,

        "PA14_biofilm48_raw":
            bf48_a,

        "PA14_biofilm24_residual":
            residual_vectors[
                "biofilm_24h"
            ],

        "PA14_biofilm48_residual":
            residual_vectors[
                "biofilm_48h"
            ],
    }

    for target_name, target in targets.items():
        for reference_name, reference in references.items():
            for programme_set in [
                all_programmes,
                robust_programmes,
            ]:
                comparison_rows.append(
                    compare(
                        target,
                        reference,
                        target_name
                        + "_vs_"
                        + reference_name,
                        programme_set,
                    )
                )

    comparisons = pd.DataFrame(
        comparison_rows
    )

    print()
    print("PA14 vs independent PAO1 biofilm programme agreement")
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
    # 15. Technical gates
    #
    # Biological composition and agreement are outputs,
    # not acceptance gates.
    # ----------------------------------------------------------

    min_fit_corr = float(
        fit_table[
            "projection_corr"
        ].min()
    )

    min_fit_energy = float(
        fit_table[
            "projection_energy_fraction"
        ].min()
    )

    minimum_depth = float(
        counts.sum(
            axis=0
        ).min()
    )

    checks = {
        "projection_gene_count_5286":
            counts.shape[0]
            == 5286,

        "minimum_model_count_depth_ge_100k":
            minimum_depth
            >= 100000,

        "exponential_expression_r_ge_090":
            exp_r
            >= 0.90,

        "biofilm24_expression_r_ge_090":
            bf24_r
            >= 0.90,

        "biofilm48_expression_r_ge_090":
            bf48_r
            >= 0.90,

        "biofilm24_activity_r_ge_075":
            bf24_activity_r
            >= 0.75,

        "biofilm48_activity_r_ge_075":
            bf48_activity_r
            >= 0.75,

        "reference_centered":
            max_reference_residual
            < 1e-8,

        "each_biofilm_projection_corr_ge_040":
            min_fit_corr
            >= 0.40,

        "each_biofilm_projection_energy_ge_015":
            min_fit_energy
            >= 0.15,

        "anchors_nearly_orthogonal":
            abs(
                anchor_cosine
            )
            < 1e-8,

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
        "GSE55197_PA14_BIOFILM_COMPOSITION_ACCEPTED"
        if passed
        else "GSE55197_PA14_BIOFILM_REVIEW_REQUIRED"
    )

    print()
    print(
        "TECHNICAL STATUS:",
        status,
    )

    print()
    print(
        "NOTE: stationary/oxygen decomposition and PAO1 biofilm "
        "agreement are scientific outputs, not technical gates."
    )

    # ----------------------------------------------------------
    # 16. Save
    # ----------------------------------------------------------

    manifest.to_csv(
        OUT
        / "GSE55197_PA14_biofilm_sample_manifest.csv",
        index=False,
    )

    counts.to_csv(
        OUT
        / "GSE55197_PA14_biofilm_projection_counts_PAO1_space.csv"
    )

    tpm.to_csv(
        OUT
        / "GSE55197_PA14_biofilm_projection_TPM_PAO1_space.csv"
    )

    log_tpm.to_csv(
        OUT
        / "GSE55197_PA14_biofilm_projection_log2TPM_PAO1_space.csv"
    )

    activities.to_csv(
        OUT
        / "GSE55197_PA14_biofilm_imodulon_activities.csv"
    )

    fit_table.to_csv(
        OUT
        / "biofilm_projection_fit.csv",
        index=False,
    )

    decomposition.to_csv(
        OUT
        / "biofilm_conserved_state_decomposition.csv",
        index=False,
    )

    programme_table.to_csv(
        OUT
        / "GSE55197_PA14_biofilm_programme_vectors.csv",
        index=False,
    )

    comparisons.to_csv(
        OUT
        / "GSE55197_PA14_vs_PAO1_biofilm_agreement.csv",
        index=False,
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
        "# CF-State Phase 3 — Audit 03g",
        "",
        "## GSE55197 PA14 biofilm composition",
        "",
        "PA14 24 h and 48 h biofilm states were projected through "
        "the frozen 5,286-gene PA14↔PAO1 universe.",
        "",
        "The programme vectors were decomposed into two previously "
        "validated physiological components:",
        "",
        "1. PA14 stationary-phase direction",
        "2. PA14 oxygen-associated residual direction",
        "",
        "The remaining programme vector is retained as the "
        "biofilm residual rather than forced into either anchor.",
        "",
        "- 24 h expression replicate r: "
        + format(
            bf24_r,
            ".4f",
        ),
        "- 48 h expression replicate r: "
        + format(
            bf48_r,
            ".4f",
        ),
        "- 24 h activity replicate r: "
        + format(
            bf24_activity_r,
            ".4f",
        ),
        "- 48 h activity replicate r: "
        + format(
            bf48_activity_r,
            ".4f",
        ),
        "",
        "## Technical decision",
        "",
        status,
    ]

    (
        OUT
        / "AUDIT_03G_SUMMARY.md"
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
