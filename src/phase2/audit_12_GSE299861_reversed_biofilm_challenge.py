from pathlib import Path
from io import StringIO
from collections import defaultdict
import hashlib
import importlib.util
import json
import re
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path.cwd()

DATA_DIR = (
    ROOT
    / "data/raw/phase2/GSE299861"
)

FPKM = (
    DATA_DIR
    / "GSE299861_PAO1_all.fpkm.txt.gz"
)

URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE299nnn/GSE299861/suppl/"
    "GSE299861_PAO1_all.fpkm.txt.gz"
)

GENBANK = (
    ROOT
    / "references/mappings/NC_002516.2.gb"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

PRECISE = (
    ROOT
    / "external/aeruprecise/precise411/sample_table.csv"
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

CATALOGUE = (
    ROOT
    / "external/aeruprecise/precise411/iM_table.csv"
)

AUDIT11_SCRIPT = (
    ROOT
    / "src/phase2/audit_11_frozen_dictionary_biofilm_decomposition.py"
)

OUT = (
    ROOT
    / "results/phase2/audit_12_GSE299861_reversed_biofilm_challenge"
)

N_PERM = 5000
SEED = 20261006


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
        "Unexpected serialized dataframe type"
    )


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def download():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if FPKM.exists():
        print(
            "FPKM file already present:"
        )
        print(
            " ",
            FPKM,
        )
        return

    print(
        "Downloading GEO PAO1 FPKM matrix ..."
    )

    request = urllib.request.Request(
        URL,
        headers={
            "User-Agent":
                "CF-State Phase2 Audit12"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        FPKM.write_bytes(
            response.read()
        )

    print(
        "Downloaded:",
        FPKM,
    )


def read_table():
    attempts = [
        {
            "sep": "\t",
        },
        {
            "sep": ",",
        },
    ]

    for kwargs in attempts:
        try:
            df = pd.read_csv(
                FPKM,
                compression="gzip",
                **kwargs,
            )

            if df.shape[1] >= 3:
                return df

        except Exception:
            pass

    return pd.read_csv(
        FPKM,
        compression="gzip",
        sep=None,
        engine="python",
    )


def load_gene_symbols(path):
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython required."
        )

    record = SeqIO.read(
        str(path),
        "genbank",
    )

    symbol_to_loci = defaultdict(
        set
    )

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

        for symbol in feature.qualifiers.get(
            "gene",
            [],
        ):
            symbol = str(
                symbol
            ).strip().lower()

            if symbol:
                symbol_to_loci[
                    symbol
                ].add(
                    locus
                )

    return {
        symbol: next(
            iter(
                loci
            )
        )
        for symbol, loci
        in symbol_to_loci.items()
        if len(
            loci
        ) == 1
    }


def classify_sample(name):
    text = str(
        name
    ).lower()

    # Exact GEO sample IDs.
    biofilm_gsm = {
        "gsm9047975",
        "gsm9047976",
        "gsm9047977",
    }

    planktonic_gsm = {
        "gsm9047978",
        "gsm9047979",
        "gsm9047980",
    }

    if any(
        gsm in text
        for gsm in biofilm_gsm
    ):
        return "biofilm"

    if any(
        gsm in text
        for gsm in planktonic_gsm
    ):
        return "planktonic"

    if (
        "biofilm" in text
        or "bio" in text
        and "pao1" in text
    ):
        return "biofilm"

    if (
        "plank" in text
        or "day0" in text
        or "day_0" in text
    ):
        return "planktonic"

    return None


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


def vector_metrics(x, y):
    x = pd.Series(
        np.asarray(
            x,
            dtype=float,
        )
    )

    y = pd.Series(
        np.asarray(
            y,
            dtype=float,
        )
    )

    denom = (
        np.linalg.norm(
            x
        )
        * np.linalg.norm(
            y
        )
    )

    cosine = float(
        np.dot(
            x,
            y,
        )
        / denom
    )

    return {
        "pearson":
            float(
                x.corr(
                    y,
                    method="pearson",
                )
            ),

        "spearman":
            float(
                x.corr(
                    y,
                    method="spearman",
                )
            ),

        "cosine":
            cosine,

        "sign_agreement":
            float(
                (
                    np.sign(
                        x
                    )
                    ==
                    np.sign(
                        y
                    )
                ).mean()
            ),
    }


def load_audit11_functions():
    spec = importlib.util.spec_from_file_location(
        "audit11_functions",
        AUDIT11_SCRIPT,
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 12")
    print("GSE299861 reversed-biofilm challenge")
    print("=" * 78)

    for path in [
        GENBANK,
        MODEL,
        PRECISE,
        DICTIONARY,
        GSE203348,
        GSE120602,
        CATALOGUE,
        AUDIT11_SCRIPT,
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
    # 1. Freeze source
    # ============================================================

    download()

    print()
    print("Frozen source")
    print(
        "  path:",
        FPKM,
    )
    print(
        "  SHA256:",
        sha256(
            FPKM
        ),
    )

    # ============================================================
    # 2. Provenance
    # ============================================================

    precise = pd.read_csv(
        PRECISE,
        low_memory=False,
    )

    aliases = [
        "GSE299861",
        "PRJNA835970",
    ]

    overlap_rows = []

    for alias in aliases:
        matched = set()

        for col in precise.columns:
            mask = (
                precise[col]
                .fillna("")
                .astype(str)
                .str.contains(
                    alias,
                    regex=False,
                )
            )

            matched.update(
                precise.index[
                    mask
                ].tolist()
            )

        overlap_rows.append(
            {
                "alias":
                    alias,

                "n_matching_rows":
                    len(
                        matched
                    ),
            }
        )

    overlap = pd.DataFrame(
        overlap_rows
    )

    training_overlap = bool(
        (
            overlap[
                "n_matching_rows"
            ]
            > 0
        ).any()
    )

    print()
    print("aeruPRECISE provenance")
    print(
        overlap.to_string(
            index=False
        )
    )

    # ============================================================
    # 3. Load FPKM table
    # ============================================================

    raw = read_table()

    print()
    print("Raw FPKM table")
    print(
        "  shape:",
        raw.shape,
    )

    print()
    print("Columns")
    for col in raw.columns:
        print(
            " ",
            col,
        )

    # ============================================================
    # 4. Frozen model and gene mapping
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

    model_ids = list(
        M.index
    )

    model_set = set(
        model_ids
    )

    symbols = load_gene_symbols(
        GENBANK
    )

    # Find likely gene-ID column.
    candidate_scores = []

    for col in raw.columns:
        values = (
            raw[col]
            .astype(str)
            .str.strip()
        )

        exact_fraction = float(
            values.isin(
                model_set
            ).mean()
        )

        symbol_fraction = float(
            values.str.lower().isin(
                symbols
            ).mean()
        )

        candidate_scores.append(
            (
                exact_fraction
                + symbol_fraction,
                col,
            )
        )

    candidate_scores.sort(
        reverse=True
    )

    _, gene_col = (
        candidate_scores[0]
    )

    mapped = []

    mapping_rows = []

    for value in raw[
        gene_col
    ]:
        text = str(
            value
        ).strip()

        locus = None
        method = None

        if text in model_set:
            locus = text
            method = "exact_model_id"

        elif text.lower() in symbols:
            candidate = symbols[
                text.lower()
            ]

            if candidate in model_set:
                locus = candidate
                method = "unique_gene_symbol"

        if locus is None:
            method = "unmapped"

        mapped.append(
            locus
        )

        mapping_rows.append(
            {
                "source_id":
                    text,

                "mapped_model_gene":
                    locus,

                "mapping_method":
                    method,
            }
        )

    mapping = pd.DataFrame(
        mapping_rows
    )

    raw[
        "_mapped_gene"
    ] = mapped

    # ============================================================
    # 5. Identify sample columns
    # ============================================================

    # ------------------------------------------------------------
    # Use FPKM columns only.
    #
    # The GEO processed table contains paired Count and FPKM
    # columns for each biological replicate. Audit 12 uses the
    # published FPKM measurements and converts those to TPM, so
    # Count columns must not be treated as additional samples.
    # ------------------------------------------------------------

    numeric_columns = [
        col
        for col in raw.columns
        if str(col).endswith("_FPKM")
    ]

    if len(numeric_columns) != 6:
        raise RuntimeError(
            "Expected exactly 6 PAO1 FPKM columns, found "
            + str(len(numeric_columns))
            + ": "
            + str(numeric_columns)
        )

    groups = {
        "biofilm": [],
        "planktonic": [],
    }

    for col in numeric_columns:
        condition = classify_sample(
            col
        )

        if condition is not None:
            groups[
                condition
            ].append(
                col
            )

    print()
    print("Detected PAO1 samples")
    print(
        "  biofilm:",
        groups[
            "biofilm"
        ],
    )
    print(
        "  planktonic:",
        groups[
            "planktonic"
        ],
    )

    if (
        len(
            groups[
                "biofilm"
            ]
        )
        != 3
        or len(
            groups[
                "planktonic"
            ]
        )
        != 3
    ):
        raise RuntimeError(
            "Could not identify exactly 3 biofilm "
            "and 3 planktonic PAO1 columns."
        )

    selected = (
        groups[
            "planktonic"
        ]
        + groups[
            "biofilm"
        ]
    )

    # ============================================================
    # 6. Build mapped FPKM matrix
    # ============================================================

    usable = raw.loc[
        raw[
            "_mapped_gene"
        ].notna()
    ].copy()

    expr = (
        usable[
            selected
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    expr.index = (
        usable[
            "_mapped_gene"
        ]
        .astype(str)
        .values
    )

    if expr.index.duplicated().any():
        duplicate_count = int(
            expr.index
            .duplicated()
            .sum()
        )

        print(
            "  duplicate mapped genes collapsed:",
            duplicate_count,
        )

        expr = (
            expr.groupby(
                level=0
            )
            .sum()
        )

    shared = (
        M.index
        .intersection(
            expr.index
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

    print()
    print("Gene mapping")
    print(
        "  mapped unique genes:",
        len(
            expr
        ),
    )
    print(
        "  shared aeruPRECISE genes:",
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

    mapping.to_csv(
        OUT
        / "gene_mapping.csv",
        index=False,
    )

    # ============================================================
    # 7. FPKM -> TPM -> log2(TPM+1)
    # ============================================================

    # For a given sample:
    #
    # TPM_i = FPKM_i / sum(FPKM) * 1e6
    #
    # This avoids introducing a second gene-length calculation.
    fpkm_sums = expr.sum(
        axis=0
    )

    if (
        fpkm_sums
        <= 0
    ).any():
        raise RuntimeError(
            "Non-positive FPKM column sum."
        )

    tpm = expr.divide(
        fpkm_sums,
        axis=1,
    ) * 1_000_000.0

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    reference = (
        log_tpm[
            groups[
                "planktonic"
            ]
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
    # 8. Frozen projection
    # ============================================================

    M_shared = M.loc[
        shared
    ]

    X_new = centered.loc[
        shared,
        selected,
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
        columns=selected,
    )

    plank_mean = (
        activities[
            groups[
                "planktonic"
            ]
        ]
        .mean(
            axis=1
        )
    )

    biofilm_mean = (
        activities[
            groups[
                "biofilm"
            ]
        ]
        .mean(
            axis=1
        )
    )

    delta = (
        biofilm_mean
        - plank_mean
    )

    # ============================================================
    # 9. Technical QC
    # ============================================================

    plank_r = mean_pairwise_corr(
        log_tpm,
        groups[
            "planktonic"
        ],
    )

    biofilm_r = mean_pairwise_corr(
        log_tpm,
        groups[
            "biofilm"
        ],
    )

    max_reference_activity = float(
        plank_mean
        .abs()
        .max()
    )

    reconstruction = (
        M_shared.to_numpy(
            dtype=float
        )
        @ A_values
    )

    observed = X_new.to_numpy(
        dtype=float
    )

    flat_x = observed.ravel()
    flat_y = reconstruction.ravel()

    projection_corr = float(
        np.corrcoef(
            flat_x,
            flat_y,
        )[0, 1]
    )

    projection_error = float(
        np.linalg.norm(
            observed
            - reconstruction
        )
        /
        np.linalg.norm(
            observed
        )
    )

    external_sd = float(
        np.nanstd(
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

    scale_ratio = (
        external_sd
        / training_sd
    )

    print()
    print("Projection QC")
    print(
        "  planktonic replicate mean r:",
        round(
            plank_r,
            4,
        ),
    )
    print(
        "  biofilm replicate mean r:",
        round(
            biofilm_r,
            4,
        ),
    )
    print(
        "  weighted programme coverage median:",
        round(
            median_weighted,
            4,
        ),
    )
    print(
        "  weighted programme coverage minimum:",
        round(
            minimum_weighted,
            4,
        ),
    )
    print(
        "  max |mean planktonic activity|:",
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
    # 10. Programme-level comparison with prior biofilms
    # ============================================================

    previous_203 = (
        pd.read_csv(
            GSE203348
        )
        .set_index(
            "imodulon"
        )[
            "delta_biofilm_minus_planktonic"
        ]
    )

    previous_120 = (
        pd.read_csv(
            GSE120602,
            index_col=0,
        )[
            "biofilm_vs_planktonic"
        ]
    )

    comparisons = []

    for name, previous in [
        (
            "GSE203348",
            previous_203,
        ),
        (
            "GSE120602",
            previous_120,
        ),
    ]:
        common = delta.index.intersection(
            previous.index
        )

        metrics = vector_metrics(
            delta.loc[
                common
            ],
            previous.loc[
                common
            ],
        )

        comparisons.append(
            {
                "comparison":
                    "GSE299861_vs_"
                    + name,

                **metrics,
            }
        )

    comparisons = pd.DataFrame(
        comparisons
    )

    print()
    print("Cross-study biofilm agreement")
    print(
        comparisons
        .round(4)
        .to_string(
            index=False
        )
    )

    # ============================================================
    # 11. Top programme changes
    # ============================================================

    catalogue = (
        pd.read_csv(
            CATALOGUE
        )[
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

    programme = pd.DataFrame(
        {
            "imodulon":
                delta.index,

            "delta_biofilm_minus_planktonic":
                delta.values,
        }
    )

    programme[
        "abs_delta"
    ] = (
        programme[
            "delta_biofilm_minus_planktonic"
        ]
        .abs()
    )

    programme = programme.merge(
        catalogue,
        on="imodulon",
        how="left",
    )

    programme = programme.sort_values(
        "abs_delta",
        ascending=False,
    )

    print()
    print("Largest GSE299861 biofilm programme changes")
    print(
        programme[
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

    # ============================================================
    # 12. Frozen dictionary decomposition
    # ============================================================

    dictionary = pd.read_csv(
        DICTIONARY,
        index_col=0,
    )

    common = (
        dictionary.index
        .intersection(
            delta.index
        )
    )

    if len(
        common
    ) != 116:
        raise RuntimeError(
            "Expected 116 common iModulons for dictionary decomposition."
        )

    D = dictionary.loc[
        common
    ].to_numpy(
        dtype=float
    )

    target = delta.loc[
        common
    ].to_numpy(
        dtype=float
    )

    audit11 = load_audit11_functions()

    attribution, explained = audit11.exact_shapley(
        target,
        D,
        list(
            dictionary.columns
        ),
    )

    reconstructed = audit11.reconstruct(
        target,
        D,
    )

    reconstruction_cosine = audit11.cosine(
        target,
        reconstructed,
    )

    null, permutation_p = audit11.permutation_test(
        target=target,
        dictionary_matrix=D,
        observed=explained,
        n_perm=N_PERM,
        seed=SEED,
    )

    attribution = attribution.sort_values(
        "shapley_explained_energy",
        ascending=False,
    )

    residual = (
        target
        - reconstructed
    )

    residual_energy = float(
        np.square(
            residual
        ).sum()
    )

    residual_df = pd.DataFrame(
        {
            "imodulon":
                common,

            "observed_delta":
                target,

            "dictionary_reconstruction":
                reconstructed,

            "residual_delta":
                residual,
        }
    )

    residual_df[
        "abs_residual"
    ] = (
        residual_df[
            "residual_delta"
        ]
        .abs()
    )

    residual_df[
        "residual_energy_fraction"
    ] = (
        np.square(
            residual_df[
                "residual_delta"
            ]
        )
        / residual_energy
    )

    residual_df = residual_df.merge(
        catalogue,
        on="imodulon",
        how="left",
    )

    residual_df = residual_df.sort_values(
        "abs_residual",
        ascending=False,
    )

    print()
    print("Frozen dictionary decomposition")
    print(
        "  dictionary fraction energy explained:",
        round(
            explained,
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
            1.0
            - explained,
            4,
        ),
    )
    print(
        "  permutation p:",
        round(
            permutation_p,
            4,
        ),
    )

    print()
    print("Stress-axis attribution")
    print(
        attribution[
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

    print()
    print("Largest dictionary residuals")
    print(
        residual_df[
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
            15
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

    # ============================================================
    # 13. Save
    # ============================================================

    activities.to_csv(
        OUT
        / "GSE299861_imodulon_activities.csv"
    )

    programme.to_csv(
        OUT
        / "GSE299861_biofilm_programme_vector.csv",
        index=False,
    )

    comparisons.to_csv(
        OUT
        / "cross_study_biofilm_agreement.csv",
        index=False,
    )

    attribution.to_csv(
        OUT
        / "GSE299861_dictionary_shapley_attribution.csv",
        index=False,
    )

    residual_df.to_csv(
        OUT
        / "GSE299861_dictionary_residual.csv",
        index=False,
    )

    np.savetxt(
        OUT
        / "dictionary_permutation_null.csv",
        null,
        delimiter=",",
    )

    # ============================================================
    # 14. Audit gate
    # ============================================================

    checks = {
        "no_training_overlap":
            not training_overlap,

        "3_biofilm_3_planktonic":
            (
                len(
                    groups[
                        "biofilm"
                    ]
                )
                == 3
                and len(
                    groups[
                        "planktonic"
                    ]
                )
                == 3
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

        "planktonic_reference_centred":
            max_reference_activity
            < 1e-8,

        "finite_activities":
            np.isfinite(
                A_values
            ).all(),

        "dictionary_decomposition_finite":
            np.isfinite(
                explained
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

    summary = [
        "# CF-State Phase 2 — Audit 12",
        "",
        "## Objective",
        "",
        "Challenge the frozen physiological dictionary with the "
        "Phase-1 reversed PAO1 biofilm experiment GSE299861.",
        "",
        "## Technical QC",
        "",
        f"- Model gene coverage: {coverage:.2%}",
        f"- Median weighted coverage: {median_weighted:.4f}",
        f"- Minimum weighted coverage: {minimum_weighted:.4f}",
        f"- Planktonic replicate r: {plank_r:.4f}",
        f"- Biofilm replicate r: {biofilm_r:.4f}",
        "",
        "## Dictionary decomposition",
        "",
        f"- Energy explained: {explained:.2%}",
        f"- Reconstruction cosine: {reconstruction_cosine:.4f}",
        f"- Permutation p: {permutation_p:.4g}",
        "",
        "## Interpretation boundary",
        "",
        "GSE299861 was selected because Phase 1 identified it as a "
        "directionally unusual biofilm experiment. The frozen "
        "GSE152295 dictionary was constructed before this challenge.",
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
        / "AUDIT_12_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 12:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_12_SUMMARY.md",
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
