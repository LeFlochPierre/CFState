from pathlib import Path
from io import StringIO
from itertools import combinations
from math import factorial
import json

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
    / "results/phase2/audit_12d_GSE299861_canonical_requantification"
    / "GSE299861_canonical_log2TPM_plus1.csv"
)

DICTIONARY = (
    ROOT
    / "results/phase2/audit_10_multistress_imodulon_dictionary"
    / "stress_contrast_vectors.csv"
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

OUT = (
    ROOT
    / "results/phase2/audit_12f_GSE299861_canonical_imodulon_challenge"
)

N_PERM = 4999
SEED = 120299861


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


def read_indexed_csv(path):
    table = pd.read_csv(
        path
    )

    candidates = [
        "imodulon",
        "iModulon",
        "Imodulon",
        "Unnamed: 0",
    ]

    index_col = None

    for candidate in candidates:
        if candidate in table.columns:
            index_col = candidate
            break

    if index_col is not None:
        table = table.set_index(
            index_col
        )

    table.index = (
        table.index
        .astype(str)
        .str.strip()
    )

    return table


def read_vector(path, column):
    table = read_indexed_csv(
        path
    )

    if column not in table.columns:
        raise RuntimeError(
            str(path)
            + " does not contain column "
            + column
            + ". Available columns: "
            + ", ".join(
                table.columns.astype(str)
            )
        )

    vector = pd.to_numeric(
        table[
            column
        ],
        errors="raise",
    )

    vector.name = column

    return vector


def mean_pairwise_corr(matrix, columns):
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


def projection(target, dictionary):
    target = np.asarray(
        target,
        dtype=float,
    )

    dictionary = np.asarray(
        dictionary,
        dtype=float,
    )

    if dictionary.shape[1] == 0:
        reconstructed = np.zeros_like(
            target
        )

    else:
        coefficients = np.linalg.lstsq(
            dictionary,
            target,
            rcond=None,
        )[0]

        reconstructed = (
            dictionary
            @ coefficients
        )

    total_energy = float(
        np.dot(
            target,
            target,
        )
    )

    explained_energy = float(
        np.dot(
            reconstructed,
            reconstructed,
        )
    )

    fraction = (
        explained_energy
        / total_energy
        if total_energy > 0
        else np.nan
    )

    return (
        reconstructed,
        fraction,
    )


def exact_shapley(target, dictionary):
    axis_names = list(
        dictionary.columns
    )

    n = len(
        axis_names
    )

    if n > 15:
        raise RuntimeError(
            "Exact Shapley calculation unexpectedly large."
        )

    target_array = target.to_numpy(
        dtype=float
    )

    cache = {
        0: 0.0
    }

    def subset_value(mask):
        if mask in cache:
            return cache[
                mask
            ]

        selected = [
            i
            for i in range(n)
            if mask
            & (
                1 << i
            )
        ]

        D = dictionary.iloc[
            :,
            selected,
        ].to_numpy(
            dtype=float
        )

        _, value = projection(
            target_array,
            D,
        )

        cache[
            mask
        ] = float(
            value
        )

        return cache[
            mask
        ]

    shapley = np.zeros(
        n,
        dtype=float,
    )

    full_mask = (
        1 << n
    ) - 1

    for i in range(n):
        bit = (
            1 << i
        )

        for mask in range(
            1 << n
        ):
            if mask & bit:
                continue

            k = int(
                mask.bit_count()
            )

            weight = (
                factorial(k)
                * factorial(
                    n - k - 1
                )
                / factorial(n)
            )

            marginal = (
                subset_value(
                    mask | bit
                )
                - subset_value(
                    mask
                )
            )

            shapley[
                i
            ] += (
                weight
                * marginal
            )

    total = subset_value(
        full_mask
    )

    rows = []

    for i, axis in enumerate(
        axis_names
    ):
        axis_vector = dictionary[
            axis
        ].to_numpy(
            dtype=float
        )

        single_mask = (
            1 << i
        )

        without_mask = (
            full_mask
            & ~single_mask
        )

        single = subset_value(
            single_mask
        )

        unique = (
            total
            - subset_value(
                without_mask
            )
        )

        signed_cos = cosine(
            target_array,
            axis_vector,
        )

        rows.append(
            {
                "axis":
                    axis,

                "shapley_explained_energy":
                    shapley[
                        i
                    ],

                "fraction_of_dictionary_explanation":
                    (
                        shapley[
                            i
                        ]
                        / total
                        if total > 0
                        else np.nan
                    ),

                "single_axis_energy":
                    single,

                "unique_energy":
                    unique,

                "signed_cosine":
                    signed_cos,
            }
        )

    result = pd.DataFrame(
        rows
    ).sort_values(
        "shapley_explained_energy",
        ascending=False,
    )

    return (
        result,
        total,
    )


def permutation_test(
    target,
    dictionary,
    observed,
):
    rng = np.random.default_rng(
        SEED
    )

    target_array = target.to_numpy(
        dtype=float
    )

    D = dictionary.to_numpy(
        dtype=float
    )

    projection_matrix = (
        D
        @ np.linalg.pinv(
            D
        )
    )

    denom = float(
        np.dot(
            target_array,
            target_array,
        )
    )

    null = np.zeros(
        N_PERM,
        dtype=float,
    )

    for i in range(
        N_PERM
    ):
        permuted = rng.permutation(
            target_array
        )

        reconstructed = (
            projection_matrix
            @ permuted
        )

        null[
            i
        ] = (
            float(
                np.dot(
                    reconstructed,
                    reconstructed,
                )
            )
            / denom
        )

    p_value = (
        1
        + int(
            np.sum(
                null >= observed
            )
        )
    ) / (
        N_PERM
        + 1
    )

    return (
        float(
            p_value
        ),
        null,
    )


def load_optional_functions(raw, imodulons):
    candidate_keys = [
        "imodulon_table",
        "iModulon_table",
        "imodulonTable",
    ]

    for key in candidate_keys:
        if key not in raw:
            continue

        try:
            table = json_df(
                raw[
                    key
                ]
            )
        except Exception:
            continue

        table.index = (
            table.index
            .astype(str)
            .str.strip()
        )

        function_col = None

        for column in table.columns:
            lower = str(
                column
            ).lower()

            if lower in {
                "function",
                "functions",
            }:
                function_col = column
                break

        if function_col is None:
            continue

        values = table[
            function_col
        ].reindex(
            imodulons
        )

        return values

    return pd.Series(
        np.nan,
        index=imodulons,
        dtype=object,
    )


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 12f")
    print("GSE299861 canonical iModulon reversed-biofilm challenge")
    print("=" * 78)

    for path in [
        MODEL,
        EXPRESSION,
        DICTIONARY,
        GSE203348,
        GSE120602,
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

    if set(
        M.columns
    ) != set(
        A_train.index
    ):
        raise RuntimeError(
            "M/A iModulon identities are inconsistent."
        )

    A_train = A_train.reindex(
        M.columns
    )

    imodulons = list(
        M.columns
    )

    functions = load_optional_functions(
        raw,
        imodulons,
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
    # Canonical GSE299861 expression
    # ----------------------------------------------------------

    X = pd.read_csv(
        EXPRESSION,
        index_col=0,
    )

    X.index = (
        X.index
        .astype(str)
        .str.strip()
    )

    biofilm_cols = [
        col
        for col in X.columns
        if "_biofilm_" in col
    ]

    planktonic_cols = [
        col
        for col in X.columns
        if "_planktonic_" in col
    ]

    if len(
        biofilm_cols
    ) != 3:
        raise RuntimeError(
            "Expected exactly three biofilm samples."
        )

    if len(
        planktonic_cols
    ) != 3:
        raise RuntimeError(
            "Expected exactly three planktonic samples."
        )

    shared = [
        gene
        for gene in M.index
        if gene in X.index
    ]

    coverage = (
        len(
            shared
        )
        / len(
            M.index
        )
    )

    print()
    print("Canonical external expression")
    print(
        "  matrix:",
        X.shape,
    )
    print(
        "  shared model genes:",
        len(
            shared
        ),
    )
    print(
        "  model gene coverage:",
        format(
            coverage,
            ".2%",
        ),
    )

    if coverage < 0.99:
        raise RuntimeError(
            "Unexpectedly incomplete canonical expression matrix."
        )

    X = X.reindex(
        M.index
    )

    if X.isna().any().any():
        raise RuntimeError(
            "Missing values after canonical model alignment."
        )

    # Replicate reproducibility must be assessed BEFORE
    # within-planktonic reference centering. Once the three
    # planktonic samples are centered by their own mean, their
    # residual vectors sum to zero by construction and their
    # pairwise correlations are expected to become negative.
    plank_expression_r = mean_pairwise_corr(
        X,
        planktonic_cols,
    )

    biofilm_expression_r = mean_pairwise_corr(
        X,
        biofilm_cols,
    )

    # ----------------------------------------------------------
    # Planktonic reference centering
    # ----------------------------------------------------------

    plank_reference = X[
        planktonic_cols
    ].mean(
        axis=1
    )

    X_centered = X.subtract(
        plank_reference,
        axis=0,
    )

    max_mean_plank_gene = float(
        X_centered[
            planktonic_cols
        ].mean(
            axis=1
        ).abs().max()
    )

    # ----------------------------------------------------------
    # iModulon projection
    # ----------------------------------------------------------

    M_array = M.to_numpy(
        dtype=float
    )

    X_array = X_centered.to_numpy(
        dtype=float
    )

    M_pinv = np.linalg.pinv(
        M_array
    )

    A_external_array = (
        M_pinv
        @ X_array
    )

    A_external = pd.DataFrame(
        A_external_array,
        index=imodulons,
        columns=X.columns,
    )

    X_reconstructed = (
        M_array
        @ A_external_array
    )

    projection_corr = float(
        np.corrcoef(
            X_array.ravel(),
            X_reconstructed.ravel(),
        )[
            0,
            1
        ]
    )

    projection_error = float(
        np.linalg.norm(
            X_array
            - X_reconstructed
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

    plank_activity_mean = (
        A_external[
            planktonic_cols
        ].mean(
            axis=1
        )
    )

    biofilm_activity_mean = (
        A_external[
            biofilm_cols
        ].mean(
            axis=1
        )
    )

    max_mean_plank_activity = float(
        plank_activity_mean.abs().max()
    )

    delta = (
        biofilm_activity_mean
        - plank_activity_mean
    )

    delta.name = (
        "delta_biofilm_minus_planktonic"
    )

    centered_plank_activity_r = mean_pairwise_corr(
        A_external,
        planktonic_cols,
    )

    biofilm_activity_r = mean_pairwise_corr(
        A_external,
        biofilm_cols,
    )

    print()
    print("Projection QC")
    print(
        "  planktonic expression replicate mean r (pre-centering):",
        round(
            plank_expression_r,
            4,
        ),
    )
    print(
        "  biofilm expression replicate mean r (pre-centering):",
        round(
            biofilm_expression_r,
            4,
        ),
    )
    print(
        "  centered planktonic activity mean r (diagnostic only):",
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
            max_mean_plank_gene,
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
        "  external/training activity SD ratio:",
        round(
            scale_ratio,
            4,
        ),
    )

    # ----------------------------------------------------------
    # Cross-study comparisons
    # ----------------------------------------------------------

    other_vectors = {
        "GSE203348":
            read_vector(
                GSE203348,
                "delta_biofilm_minus_planktonic",
            ),

        "GSE120602":
            read_vector(
                GSE120602,
                "biofilm_vs_planktonic",
            ),
    }

    comparison_rows = []

    for name, vector in other_vectors.items():
        vector = vector.reindex(
            imodulons
        )

        if vector.isna().any():
            missing = vector.index[
                vector.isna()
            ].tolist()

            raise RuntimeError(
                name
                + " missing iModulons: "
                + ", ".join(
                    missing[:10]
                )
            )

        a = delta.to_numpy(
            dtype=float
        )

        b = vector.to_numpy(
            dtype=float
        )

        comparison_rows.append(
            {
                "comparison":
                    "GSE299861_vs_"
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

    cross_study = pd.DataFrame(
        comparison_rows
    )

    print()
    print("Cross-study biofilm agreement")
    print(
        cross_study
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Largest programme changes
    # ----------------------------------------------------------

    programme_table = pd.DataFrame(
        {
            "imodulon":
                imodulons,

            "delta_biofilm_minus_planktonic":
                delta.reindex(
                    imodulons
                ).to_numpy(),

            "function":
                functions.reindex(
                    imodulons
                ).to_numpy(),
        }
    )

    programme_table[
        "abs_delta"
    ] = (
        programme_table[
            "delta_biofilm_minus_planktonic"
        ].abs()
    )

    programme_table = (
        programme_table
        .sort_values(
            "abs_delta",
            ascending=False,
        )
    )

    print()
    print("Largest GSE299861 biofilm programme changes")
    print(
        programme_table[
            [
                "imodulon",
                "delta_biofilm_minus_planktonic",
                "function",
            ]
        ]
        .head(
            20
        )
        .round(
            {
                "delta_biofilm_minus_planktonic":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Frozen controlled-stress dictionary
    # ----------------------------------------------------------

    dictionary = read_indexed_csv(
        DICTIONARY
    )

    dictionary = dictionary.apply(
        pd.to_numeric,
        errors="raise",
    )

    dictionary = dictionary.reindex(
        imodulons
    )

    if dictionary.isna().any().any():
        raise RuntimeError(
            "Frozen stress dictionary does not align "
            "cleanly to the 116 iModulons."
        )

    shapley, dictionary_fraction = exact_shapley(
        delta.reindex(
            imodulons
        ),
        dictionary,
    )

    reconstructed_array, _ = projection(
        delta.reindex(
            imodulons
        ).to_numpy(
            dtype=float
        ),
        dictionary.to_numpy(
            dtype=float
        ),
    )

    reconstructed = pd.Series(
        reconstructed_array,
        index=imodulons,
        name="dictionary_reconstruction",
    )

    residual = (
        delta.reindex(
            imodulons
        )
        - reconstructed
    )

    reconstruction_cosine = cosine(
        delta.reindex(
            imodulons
        ).to_numpy(
            dtype=float
        ),
        reconstructed.to_numpy(
            dtype=float
        ),
    )

    residual_fraction = (
        1.0
        - dictionary_fraction
    )

    p_value, null = permutation_test(
        delta.reindex(
            imodulons
        ),
        dictionary,
        dictionary_fraction,
    )

    print()
    print("Frozen dictionary decomposition")
    print(
        "  dictionary fraction energy explained:",
        round(
            dictionary_fraction,
            4,
        ),
    )
    print(
        "  reconstruction cosine:",
        round(
            reconstruction_cosine,
            4,
        ),
    )
    print(
        "  residual energy fraction:",
        round(
            residual_fraction,
            4,
        ),
    )
    print(
        "  permutation p:",
        round(
            p_value,
            4,
        ),
    )

    print()
    print("Stress-axis attribution")
    print(
        shapley[
            [
                "axis",
                "shapley_explained_energy",
                "fraction_of_dictionary_explanation",
                "signed_cosine",
            ]
        ]
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Residual programmes
    # ----------------------------------------------------------

    residual_table = pd.DataFrame(
        {
            "imodulon":
                imodulons,

            "observed_delta":
                delta.reindex(
                    imodulons
                ).to_numpy(),

            "dictionary_reconstruction":
                reconstructed.reindex(
                    imodulons
                ).to_numpy(),

            "residual_delta":
                residual.reindex(
                    imodulons
                ).to_numpy(),

            "function":
                functions.reindex(
                    imodulons
                ).to_numpy(),
        }
    )

    residual_energy_total = float(
        np.sum(
            residual_table[
                "residual_delta"
            ].to_numpy(
                dtype=float
            )
            ** 2
        )
    )

    residual_table[
        "residual_energy_fraction"
    ] = (
        residual_table[
            "residual_delta"
        ]
        ** 2
    ) / residual_energy_total

    residual_table[
        "abs_residual"
    ] = (
        residual_table[
            "residual_delta"
        ].abs()
    )

    residual_table = residual_table.sort_values(
        "abs_residual",
        ascending=False,
    )

    print()
    print("Largest dictionary residuals")
    print(
        residual_table[
            [
                "imodulon",
                "observed_delta",
                "dictionary_reconstruction",
                "residual_delta",
                "residual_energy_fraction",
                "function",
            ]
        ]
        .head(
            20
        )
        .round(
            {
                "observed_delta":
                    3,

                "dictionary_reconstruction":
                    3,

                "residual_delta":
                    3,

                "residual_energy_fraction":
                    4,
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    A_external.to_csv(
        OUT
        / "GSE299861_canonical_imodulon_activities.csv"
    )

    programme_table.drop(
        columns=[
            "abs_delta"
        ]
    ).to_csv(
        OUT
        / "GSE299861_biofilm_programme_vector.csv",
        index=False,
    )

    cross_study.to_csv(
        OUT
        / "cross_study_biofilm_agreement.csv",
        index=False,
    )

    shapley.to_csv(
        OUT
        / "stress_dictionary_shapley.csv",
        index=False,
    )

    residual_table.drop(
        columns=[
            "abs_residual"
        ]
    ).to_csv(
        OUT
        / "dictionary_residual_programmes.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "null_dictionary_fraction":
                null
        }
    ).to_csv(
        OUT
        / "dictionary_permutation_null.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Technical gates
    # ----------------------------------------------------------

    checks = {
        "canonical_gene_coverage_ge_99pct":
            coverage >= 0.99,

        "planktonic_reference_centered":
            max_mean_plank_activity
            < 1e-8,

        "finite_projection":
            bool(
                np.isfinite(
                    A_external_array
                ).all()
            ),

        "projection_corr_ge_0_40":
            projection_corr >= 0.40,

        "planktonic_expression_replicate_r_ge_0_75":
            plank_expression_r >= 0.75,

        "biofilm_expression_replicate_r_ge_0_75":
            biofilm_expression_r >= 0.75,

        "biofilm_activity_replicate_r_ge_0_75":
            biofilm_activity_r >= 0.75,

        "dictionary_finite":
            bool(
                np.isfinite(
                    dictionary.to_numpy(
                        dtype=float
                    )
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

    summary_lines = [
        "# CF-State Phase 2 — Audit 12f",
        "",
        "## Objective",
        "",
        "Test the pre-specified GSE299861 reversed-biofilm "
        "challenge using canonical raw-read re-quantification, "
        "with no dependence on the original PAO1DT annotation.",
        "",
        "## Technical projection",
        "",
        "- Canonical gene coverage: "
        + format(
            coverage,
            ".2%",
        ),
        "- Projection correlation: "
        + format(
            projection_corr,
            ".4f",
        ),
        "- Relative projection error: "
        + format(
            projection_error,
            ".4f",
        ),
        "- External/training activity SD ratio: "
        + format(
            scale_ratio,
            ".4f",
        ),
        "- Planktonic expression replicate correlation "
        "(pre-centering): "
        + format(
            plank_expression_r,
            ".4f",
        ),
        "- Biofilm expression replicate correlation "
        "(pre-centering): "
        + format(
            biofilm_expression_r,
            ".4f",
        ),
        "- Centered planktonic activity correlation "
        "(diagnostic only): "
        + format(
            centered_plank_activity_r,
            ".4f",
        ),
        "- Biofilm activity replicate correlation: "
        + format(
            biofilm_activity_r,
            ".4f",
        ),
        "",
        "## Controlled dictionary",
        "",
        "- Fraction energy explained: "
        + format(
            dictionary_fraction,
            ".4f",
        ),
        "- Reconstruction cosine: "
        + format(
            reconstruction_cosine,
            ".4f",
        ),
        "- Permutation p-value: "
        + format(
            p_value,
            ".4f",
        ),
        "",
        "## Audit decision",
        "",
        "PASS: "
        + str(
            passed
        ),
        "",
        "Interpret cross-study direction and stress-axis "
        "attribution only after this technical audit passes.",
    ]

    (
        OUT
        / "AUDIT_12F_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary_lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 12f:",
        "PASS"
        if passed
        else "FAIL",
    )
    print(
        "Summary:",
        OUT
        / "AUDIT_12F_SUMMARY.md",
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
