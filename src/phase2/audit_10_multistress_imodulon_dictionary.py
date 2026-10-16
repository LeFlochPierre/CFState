from pathlib import Path
from io import StringIO
from collections import defaultdict
import json
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

DATA = ROOT / "data/raw/phase2/GSE152295/GSE152295_PSEUDO_processed.txt.gz"
GENBANK = ROOT / "references/mappings/NC_002516.2.gb"

MODEL = ROOT / "external/aeruprecise/precise411/paeru_116_iModulon.json"
CATALOGUE = ROOT / "external/aeruprecise/precise411/iM_table.csv"

IRON_ANCHOR = (
    ROOT
    / "results/phase2/audit_08_iron_limitation_anchor_GSE224821"
    / "iron_anchor_vector.csv"
)

CONTROLLED = (
    ROOT
    / "results/phase2/audit_04_controlled_state_anchoring_GSE120602"
    / "controlled_contrast_vectors.csv"
)

OUT = (
    ROOT
    / "results/phase2/audit_10_multistress_imodulon_dictionary"
)

N_PERM = 5000
SEED = 20261006


CONDITION_MAP = {
    "As": "acidic",
    "Bs": "bile",
    "Ctrl": "control",
    "Li": "low_iron",
    "Mig": "microaerophilic",
    "Nd": "nutritional_downshift",
    "Ns": "nitrosative",
    "Oss": "osmotic",
    "Oxs": "oxidative",
    "Sp": "stationary",
    "Tm": "temperature",
    "Vic": "virulence_inducing",
}


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
        "Unexpected serialized DataFrame type"
    )


def load_annotation_maps(path):
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython required. Install with:\n"
            "conda install -c conda-forge biopython"
        )

    record = SeqIO.read(
        str(path),
        "genbank",
    )

    loci = set()
    symbol_loci = defaultdict(set)

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

        if not re.fullmatch(
            r"PA\d{4}",
            locus,
        ):
            continue

        loci.add(
            locus
        )

        for symbol in feature.qualifiers.get(
            "gene",
            [],
        ):
            symbol = str(
                symbol
            ).strip().lower()

            if symbol:
                symbol_loci[
                    symbol
                ].add(
                    locus
                )

    unique_symbols = {
        symbol: next(
            iter(values)
        )
        for symbol, values
        in symbol_loci.items()
        if len(values) == 1
    }

    return loci, unique_symbols


def map_feature_name(
    name,
    loci,
    unique_symbols,
):
    text = str(
        name
    ).strip()

    # First choice: explicit PA#### token anywhere in the field.
    hits = re.findall(
        r"PA\d{4}",
        text,
        flags=re.IGNORECASE,
    )

    hits = [
        hit.upper()
        for hit in hits
        if hit.upper() in loci
    ]

    hits = sorted(
        set(
            hits
        )
    )

    if len(hits) == 1:
        return (
            hits[0],
            "embedded_locus_tag",
        )

    # Second choice: whole field itself is a canonical locus tag.
    upper = text.upper()

    if upper in loci:
        return (
            upper,
            "exact_locus_tag",
        )

    # Third choice: unique canonical gene-symbol mapping.
    candidates = [
        text.lower()
    ]

    # Sometimes fields contain an annotation after whitespace,
    # parentheses, slash, semicolon, comma, etc.
    candidates.extend(
        token.lower()
        for token in re.split(
            r"[\s;,/|()\[\]]+",
            text,
        )
        if token
    )

    for candidate in candidates:
        if candidate in unique_symbols:
            return (
                unique_symbols[
                    candidate
                ],
                "unique_gene_symbol",
            )

    return (
        None,
        "unmapped",
    )


def parse_sample(sample):
    match = re.search(
        r"PSEUDO_([A-Za-z]+)_(\d+)",
        str(sample),
    )

    if match is None:
        raise RuntimeError(
            "Could not parse sample name: "
            + str(sample)
        )

    code = match.group(
        1
    )

    replicate = int(
        match.group(
            2
        )
    )

    if code not in CONDITION_MAP:
        raise RuntimeError(
            "Unknown PATHOgenex condition code: "
            + code
        )

    return (
        code,
        CONDITION_MAP[
            code
        ],
        replicate,
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


def cosine(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    denom = (
        np.linalg.norm(
            x
        )
        *
        np.linalg.norm(
            y
        )
    )

    if denom == 0:
        return np.nan

    return float(
        np.dot(
            x,
            y,
        )
        / denom
    )


def global_corr(x, y):
    x = np.asarray(
        x,
        dtype=float,
    ).ravel()

    y = np.asarray(
        y,
        dtype=float,
    ).ravel()

    good = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )

    return float(
        np.corrcoef(
            x[
                good
            ],
            y[
                good
            ],
        )[0, 1]
    )


def vector_comparison(
    x,
    y,
    seed,
):
    x = pd.Series(
        x,
        dtype=float,
    )

    y = pd.Series(
        y,
        dtype=float,
    )

    pearson = float(
        x.corr(
            y,
            method="pearson",
        )
    )

    spearman = float(
        x.corr(
            y,
            method="spearman",
        )
    )

    observed_cosine = cosine(
        x.values,
        y.values,
    )

    sign_agreement = float(
        (
            np.sign(
                x.values
            )
            ==
            np.sign(
                y.values
            )
        ).mean()
    )

    rng = np.random.default_rng(
        seed
    )

    null = np.empty(
        N_PERM,
        dtype=float,
    )

    y_values = y.to_numpy()

    for i in range(
        N_PERM
    ):
        shuffled = y_values[
            rng.permutation(
                len(
                    y_values
                )
            )
        ]

        null[
            i
        ] = cosine(
            x.values,
            shuffled,
        )

    # Direction is pre-specified:
    # low iron vs control should agree with low iron vs high iron,
    # and stationary vs control should agree with stationary vs planktonic.
    p = (
        1
        + int(
            np.sum(
                null
                >= observed_cosine
            )
        )
    ) / (
        N_PERM
        + 1
    )

    return {
        "pearson":
            pearson,

        "spearman":
            spearman,

        "cosine":
            observed_cosine,

        "sign_agreement":
            sign_agreement,

        "permutation_p":
            float(
                p
            ),
    }


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 10")
    print("Independent PAO1 multi-stress iModulon dictionary")
    print("=" * 78)

    for path in [
        DATA,
        GENBANK,
        MODEL,
        CATALOGUE,
        IRON_ANCHOR,
        CONTROLLED,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ============================================================
    # 1-2. Load corrected model-aligned TPM matrix
    #
    # Audit 10c established a perfect 1:1 mapping:
    # 5572 source rows -> 5572 frozen aeruPRECISE genes,
    # with zero unmapped rows and zero collisions.
    # ============================================================

    aligned_path = (
        ROOT
        / "results/phase2/audit_10c_correct_full_locus_mapping"
        / "GSE152295_model_aligned_TPM.csv"
    )

    if not aligned_path.exists():
        raise FileNotFoundError(
            str(aligned_path)
        )

    expr = pd.read_csv(
        aligned_path,
        index_col=0,
    )

    expr.index = (
        expr.index
        .astype(str)
        .str.strip()
    )

    expr = expr.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if expr.isna().any().any():
        raise RuntimeError(
            "Corrected aligned TPM matrix contains non-numeric values."
        )

    if not expr.index.is_unique:
        raise RuntimeError(
            "Corrected aligned TPM matrix contains duplicate gene IDs."
        )

    expression_columns = list(
        expr.columns
    )

    if expr.shape != (5572, 36):
        raise RuntimeError(
            "Expected corrected matrix shape (5572, 36), found "
            + str(expr.shape)
        )

    if (expr < 0).any().any():
        raise RuntimeError(
            "Corrected TPM matrix contains negative values."
        )

    column_sums = expr.sum(
        axis=0
    )

    max_tpm_sum_deviation = float(
        (column_sums - 1_000_000.0)
        .abs()
        .max()
    )

    print()
    print("Corrected GSE152295 mapping")
    print(
        "  model-aligned TPM matrix:",
        expr.shape,
    )
    print(
        "  unique genes:",
        expr.index.is_unique,
    )
    print(
        "  max TPM-sum deviation from 1e6:",
        round(
            max_tpm_sum_deviation,
            6,
        ),
    )
    print(
        "  mapping provenance:",
        "Audit 10c PERFECT_1_TO_1",
    )

    # ============================================================
    # 3. Parse experimental conditions
    # ============================================================

    sample_rows = []

    groups = {
        condition: []
        for condition in CONDITION_MAP.values()
    }

    for sample in expression_columns:
        code, condition, replicate = parse_sample(
            sample
        )

        groups[
            condition
        ].append(
            sample
        )

        sample_rows.append(
            {
                "sample":
                    sample,

                "code":
                    code,

                "condition":
                    condition,

                "replicate":
                    replicate,
            }
        )

    sample_meta = pd.DataFrame(
        sample_rows
    )

    sample_meta.to_csv(
        OUT
        / "sample_metadata.csv",
        index=False,
    )

    print()
    print("Condition inventory")

    for condition in CONDITION_MAP.values():
        print(
            "  "
            + condition
            + ": "
            + str(
                len(
                    groups[
                        condition
                    ]
                )
            )
        )

        if len(
            groups[
                condition
            ]
        ) != 3:
            raise RuntimeError(
                "Expected 3 replicates for "
                + condition
            )

    # ============================================================
    # 4. TPM -> log2(TPM+1), common-control centring
    # ============================================================

    log_tpm = np.log2(
        expr
        + 1.0
    )

    control_columns = groups[
        "control"
    ]

    reference = (
        log_tpm[
            control_columns
        ]
        .mean(
            axis=1
        )
    )

    centered = log_tpm.subtract(
        reference,
        axis=0,
    )

    # ============================================================
    # 5. Frozen aeruPRECISE projection
    # ============================================================

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        model_raw = json.load(
            handle
        )

    M = (
        json_df(
            model_raw[
                "M"
            ]
        )
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    X_train = (
        json_df(
            model_raw[
                "X"
            ]
        )
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    M.index = (
        M.index
        .astype(str)
    )

    centered.index = (
        centered.index
        .astype(str)
    )

    shared = (
        M.index
        .intersection(
            centered.index
        )
    )

    coverage = (
        len(
            shared
        )
        / len(
            M.index
        )
    )

    M_shared = M.loc[
        shared
    ]

    X_new = centered.loc[
        shared,
        expression_columns,
    ]

    weighted_coverage = (
        M_shared
        .abs()
        .sum(
            axis=0
        )
        /
        M
        .abs()
        .sum(
            axis=0
        )
    )

    median_weighted = float(
        weighted_coverage.median()
    )

    minimum_weighted = float(
        weighted_coverage.min()
    )

    print()
    print("aeruPRECISE compatibility")
    print(
        "  model genes:",
        len(
            M
        ),
    )
    print(
        "  shared genes:",
        len(
            shared
        ),
    )
    print(
        "  gene coverage:",
        format(
            coverage,
            ".2%",
        ),
    )
    print(
        "  median weighted coverage:",
        round(
            median_weighted,
            4,
        ),
    )
    print(
        "  minimum weighted coverage:",
        round(
            minimum_weighted,
            4,
        ),
    )

    if coverage < 0.90:
        print()
        print("First 40 unmapped PATHOgenex features")
        print(
            unmapped.head(
                40
            ).to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Less than 90% of aeruPRECISE genes mapped. "
            "Stop before projection."
        )

    A_values = (
        np.linalg.pinv(
            M_shared.to_numpy(
                dtype=float
            )
        )
        @
        X_new.to_numpy(
            dtype=float
        )
    )

    activities = pd.DataFrame(
        A_values,
        index=M.columns.astype(
            str
        ),
        columns=expression_columns,
    )

    # ============================================================
    # 6. Projection and replicate QC
    # ============================================================

    replicate_rows = []

    for condition in CONDITION_MAP.values():
        replicate_rows.append(
            {
                "condition":
                    condition,

                "mean_pairwise_logTPM_r":
                    mean_pairwise_corr(
                        log_tpm.loc[
                            shared
                        ],
                        groups[
                            condition
                        ],
                    ),
            }
        )

    replicate_qc = pd.DataFrame(
        replicate_rows
    )

    max_reference_activity = float(
        activities[
            control_columns
        ]
        .mean(
            axis=1
        )
        .abs()
        .max()
    )

    reconstructed = (
        M_shared.to_numpy(
            dtype=float
        )
        @ A_values
    )

    observed = X_new.to_numpy(
        dtype=float
    )

    projection_corr = global_corr(
        observed,
        reconstructed,
    )

    projection_error = float(
        np.linalg.norm(
            observed
            - reconstructed
        )
        /
        np.linalg.norm(
            observed
        )
    )

    training_sd = float(
        np.nanstd(
            X_train.to_numpy(
                dtype=float
            )
        )
    )

    external_sd = float(
        np.nanstd(
            observed
        )
    )

    scale_ratio = (
        external_sd
        / training_sd
    )

    print()
    print("Replicate QC")
    print(
        replicate_qc
        .round(4)
        .to_string(
            index=False
        )
    )

    print()
    print("Projection QC")
    print(
        "  max |mean control activity|:",
        format(
            max_reference_activity,
            ".3e",
        ),
    )
    print(
        "  corr(centered X, M@A):",
        round(
            projection_corr,
            4,
        ),
    )
    print(
        "  relative projection error:",
        round(
            projection_error,
            4,
        ),
    )
    print(
        "  external/training SD ratio:",
        round(
            scale_ratio,
            4,
        ),
    )

    # ============================================================
    # 7. Eleven controlled stress vectors
    # ============================================================

    condition_means = {}

    for condition in CONDITION_MAP.values():
        condition_means[
            condition
        ] = (
            activities[
                groups[
                    condition
                ]
            ]
            .mean(
                axis=1
            )
        )

    condition_means = pd.DataFrame(
        condition_means
    )

    stresses = [
        condition
        for condition in CONDITION_MAP.values()
        if condition != "control"
    ]

    contrasts = pd.DataFrame(
        index=activities.index
    )

    for condition in stresses:
        contrasts[
            condition
        ] = (
            condition_means[
                condition
            ]
            -
            condition_means[
                "control"
            ]
        )

    # ============================================================
    # 8. Programme annotations / top responses
    # ============================================================

    catalogue = pd.read_csv(
        CATALOGUE
    )

    annotation = (
        catalogue[
            [
                "name",
                "function",
                "category",
            ]
        ]
        .rename(
            columns={
                "name":
                    "imodulon"
            }
        )
    )

    top_rows = []

    for condition in stresses:
        ranked = (
            contrasts[
                condition
            ]
            .abs()
            .sort_values(
                ascending=False
            )
            .head(
                15
            )
        )

        for rank, imodulon in enumerate(
            ranked.index,
            start=1,
        ):
            top_rows.append(
                {
                    "condition":
                        condition,

                    "rank":
                        rank,

                    "imodulon":
                        imodulon,

                    "delta_activity":
                        float(
                            contrasts.loc[
                                imodulon,
                                condition,
                            ]
                        ),
                }
            )

    top_programmes = pd.DataFrame(
        top_rows
    ).merge(
        annotation,
        on="imodulon",
        how="left",
    )

    # ============================================================
    # 9. Pairwise geometry between controlled stresses
    # ============================================================

    cosine_matrix = pd.DataFrame(
        index=stresses,
        columns=stresses,
        dtype=float,
    )

    spearman_matrix = pd.DataFrame(
        index=stresses,
        columns=stresses,
        dtype=float,
    )

    for c1 in stresses:
        for c2 in stresses:
            cosine_matrix.loc[
                c1,
                c2,
            ] = cosine(
                contrasts[
                    c1
                ],
                contrasts[
                    c2
                ],
            )

            spearman_matrix.loc[
                c1,
                c2,
            ] = float(
                contrasts[
                    c1
                ].corr(
                    contrasts[
                        c2
                    ],
                    method="spearman",
                )
            )

    # ============================================================
    # 10. Effective dimensionality of controlled state space
    # ============================================================

    unit_vectors = []

    for condition in stresses:
        vector = (
            contrasts[
                condition
            ]
            .to_numpy(
                dtype=float
            )
        )

        norm = np.linalg.norm(
            vector
        )

        if norm == 0:
            raise RuntimeError(
                "Zero-length stress vector: "
                + condition
            )

        unit_vectors.append(
            vector
            / norm
        )

    unit_matrix = np.column_stack(
        unit_vectors
    )

    _, singular_values, _ = np.linalg.svd(
        unit_matrix,
        full_matrices=False,
    )

    energy = (
        singular_values
        ** 2
    )

    energy = (
        energy
        / energy.sum()
    )

    effective_dimension = float(
        1.0
        / np.sum(
            energy
            ** 2
        )
    )

    svd_table = pd.DataFrame(
        {
            "component":
                np.arange(
                    1,
                    len(
                        singular_values
                    )
                    + 1,
                ),

            "singular_value":
                singular_values,

            "energy_fraction":
                energy,

            "cumulative_energy":
                np.cumsum(
                    energy
                ),
        }
    )

    print()
    print("Controlled-state dimensionality")
    print(
        "  number of stress vectors:",
        len(
            stresses
        ),
    )
    print(
        "  effective dimension:",
        round(
            effective_dimension,
            3,
        ),
    )

    print(
        "  first component energy:",
        format(
            energy[
                0
            ],
            ".2%",
        ),
    )

    print(
        "  first 3 components:",
        format(
            energy[
                :3
            ].sum(),
            ".2%",
        ),
    )

    print(
        "  first 5 components:",
        format(
            energy[
                :5
            ].sum(),
            ".2%",
        ),
    )

    # ============================================================
    # 11. Independent replication of previously constructed axes
    # ============================================================

    iron_table = pd.read_csv(
        IRON_ANCHOR
    ).set_index(
        "imodulon"
    )

    controlled = pd.read_csv(
        CONTROLLED,
        index_col=0,
    )

    common_iron = (
        contrasts.index
        .intersection(
            iron_table.index
        )
    )

    atlas_iron = (
        contrasts.loc[
            common_iron,
            "low_iron",
        ]
    )

    previous_iron = (
        iron_table.loc[
            common_iron,
            "iron_low_minus_high",
        ]
    )

    iron_validation = vector_comparison(
        atlas_iron.values,
        previous_iron.values,
        SEED,
    )

    common_stationary = (
        contrasts.index
        .intersection(
            controlled.index
        )
    )

    atlas_stationary = (
        contrasts.loc[
            common_stationary,
            "stationary",
        ]
    )

    previous_stationary = (
        controlled.loc[
            common_stationary,
            "stationary_vs_planktonic",
        ]
    )

    stationary_validation = vector_comparison(
        atlas_stationary.values,
        previous_stationary.values,
        SEED + 1,
    )

    validation = pd.DataFrame(
        [
            {
                "axis":
                    "iron",

                "atlas_contrast":
                    "GSE152295 low_iron vs control",

                "previous_contrast":
                    "GSE224821 low_iron vs high_iron",

                **iron_validation,
            },
            {
                "axis":
                    "stationary",

                "atlas_contrast":
                    "GSE152295 stationary vs control",

                "previous_contrast":
                    "GSE120602 stationary vs planktonic",

                **stationary_validation,
            },
        ]
    )

    print()
    print("Independent anchor replication")
    print(
        validation[
            [
                "axis",
                "pearson",
                "spearman",
                "cosine",
                "sign_agreement",
                "permutation_p",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ============================================================
    # 12. Print top programmes
    # ============================================================

    for condition in stresses:
        print()
        print(
            "--- "
            + condition
            + " ---"
        )

        show = (
            top_programmes.loc[
                top_programmes[
                    "condition"
                ]
                == condition,
                [
                    "rank",
                    "imodulon",
                    "delta_activity",
                    "function",
                ],
            ]
            .head(
                8
            )
        )

        print(
            show.round(
                {
                    "delta_activity":
                        3,
                }
            ).to_string(
                index=False
            )
        )

    # ============================================================
    # 13. Save everything
    # ============================================================

    log_tpm.to_csv(
        OUT
        / "GSE152295_mapped_log2TPM_plus1.csv"
    )

    centered.to_csv(
        OUT
        / "GSE152295_centered_logTPM.csv"
    )

    activities.to_csv(
        OUT
        / "GSE152295_imodulon_activities.csv"
    )

    condition_means.to_csv(
        OUT
        / "condition_mean_activities.csv"
    )

    contrasts.to_csv(
        OUT
        / "stress_contrast_vectors.csv"
    )

    top_programmes.to_csv(
        OUT
        / "top15_programmes_per_stress.csv",
        index=False,
    )

    cosine_matrix.to_csv(
        OUT
        / "stress_pairwise_cosine.csv"
    )

    spearman_matrix.to_csv(
        OUT
        / "stress_pairwise_spearman.csv"
    )

    svd_table.to_csv(
        OUT
        / "stress_dictionary_svd.csv",
        index=False,
    )

    validation.to_csv(
        OUT
        / "independent_anchor_replication.csv",
        index=False,
    )

    replicate_qc.to_csv(
        OUT
        / "replicate_qc.csv",
        index=False,
    )

    pd.DataFrame(
        [
            [
                "gene_coverage",
                coverage,
            ],
            [
                "median_weighted_coverage",
                median_weighted,
            ],
            [
                "minimum_weighted_coverage",
                minimum_weighted,
            ],
            [
                "max_abs_mean_control_activity",
                max_reference_activity,
            ],
            [
                "projection_correlation",
                projection_corr,
            ],
            [
                "relative_projection_error",
                projection_error,
            ],
            [
                "external_training_sd_ratio",
                scale_ratio,
            ],
            [
                "stress_dictionary_effective_dimension",
                effective_dimension,
            ],
        ],
        columns=[
            "metric",
            "value",
        ],
    ).to_csv(
        OUT
        / "projection_qc.csv",
        index=False,
    )

    # ============================================================
    # 14. Technical audit gate
    # ============================================================

    checks = {
        "36_samples":
            len(
                expression_columns
            )
            == 36,

        "12_conditions":
            len(
                groups
            )
            == 12,

        "3_replicates_each":
            all(
                len(
                    values
                )
                == 3
                for values in groups.values()
            ),

        "gene_coverage_ge_90pct":
            coverage
            >= 0.90,

        "median_weighted_ge_95pct":
            median_weighted
            >= 0.95,

        "minimum_weighted_ge_80pct":
            minimum_weighted
            >= 0.80,

        "control_centred":
            max_reference_activity
            < 1e-8,

        "finite_activities":
            np.isfinite(
                A_values
            ).all(),

        "11_nonzero_stress_vectors":
            len(
                stresses
            )
            == 11
            and all(
                np.linalg.norm(
                    contrasts[
                        condition
                    ]
                )
                > 0
                for condition in stresses
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

    # ============================================================
    # 15. Summary
    # ============================================================

    lines = [
        "# CF-State Phase 2 — Audit 10",
        "",
        "## Objective",
        "",
        "Build an independent controlled PAO1 physiological-state "
        "dictionary by projecting the GSE152295 multi-stress RNA atlas "
        "into the frozen aeruPRECISE-411 iModulon representation.",
        "",
        "## Experimental dictionary",
        "",
    ]

    for condition in stresses:
        lines.append(
            "- "
            + condition
            + " vs untreated exponential-phase control"
        )

    lines.extend(
        [
            "",
            "## Technical QC",
            "",
            f"- aeruPRECISE gene coverage: {coverage:.2%}",
            f"- median weighted programme coverage: {median_weighted:.4f}",
            f"- minimum weighted programme coverage: {minimum_weighted:.4f}",
            f"- projection correlation: {projection_corr:.4f}",
            f"- relative projection error: {projection_error:.4f}",
            f"- external/training SD ratio: {scale_ratio:.4f}",
            "",
            "## Controlled-state geometry",
            "",
            f"- 11 stress vectors",
            f"- effective dimension: {effective_dimension:.3f}",
            f"- first component energy: {energy[0]:.2%}",
            f"- first three components: {energy[:3].sum():.2%}",
            f"- first five components: {energy[:5].sum():.2%}",
            "",
            "## Independent anchor replication",
            "",
            f"- Iron cosine: {iron_validation['cosine']:.4f}",
            f"- Iron Spearman: {iron_validation['spearman']:.4f}",
            f"- Iron permutation p: {iron_validation['permutation_p']:.4g}",
            f"- Stationary cosine: {stationary_validation['cosine']:.4f}",
            f"- Stationary Spearman: {stationary_validation['spearman']:.4f}",
            f"- Stationary permutation p: "
            f"{stationary_validation['permutation_p']:.4g}",
            "",
            "## Interpretation boundary",
            "",
            "These are experimentally controlled transcriptional perturbation "
            "vectors in iModulon space. They are candidate physiological "
            "anchors, not mutually exclusive cell-state labels.",
            "",
            "No biofilm target was used to select or fit these axes.",
            "",
            "## Decision",
            "",
            "PASS: "
            + str(
                passed
            ),
        ]
    )

    (
        OUT
        / "AUDIT_10_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 10:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_10_SUMMARY.md",
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
