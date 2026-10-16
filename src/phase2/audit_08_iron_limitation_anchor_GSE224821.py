from pathlib import Path
from io import StringIO
import json
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

COUNTS = ROOT / "external/EPathDash/GEO_Datasets/Pseudomonas/Count_Tables/GSE224821_counts.csv"
DESIGN = ROOT / "external/EPathDash/GEO_Datasets/Pseudomonas/Design_Matrices/GSE224821_design.csv"
GENBANK = ROOT / "references/mappings/NC_002516.2.gb"

MODEL = ROOT / "external/aeruprecise/precise411/paeru_116_iModulon.json"
CATALOGUE = ROOT / "external/aeruprecise/precise411/iM_table.csv"
PRECISE_SAMPLES = ROOT / "external/aeruprecise/precise411/sample_table.csv"

AUDIT03 = (
    ROOT
    / "results/phase2/audit_03_external_pao1_activity_inference"
    / "GSE203348_activity_summary.csv"
)

AUDIT04 = (
    ROOT
    / "results/phase2/audit_04_controlled_state_anchoring_GSE120602"
    / "controlled_contrast_vectors.csv"
)

OUT = ROOT / "results/phase2/audit_08_iron_limitation_anchor_GSE224821"

N_PERM = 5000
SEED = 20261006

IRON_CORE = [
    "PvdS",
    "FpvR",
    "PchR",
    "FoxR",
    "Iron acquisition",
    "Fur",
]

IRON_EXTENDED = IRON_CORE + [
    "PrrF",
    "k-opioid",
]


def json_df(value):
    if isinstance(value, str):
        return pd.read_json(StringIO(value))

    if isinstance(value, dict):
        return pd.DataFrame(value)

    raise RuntimeError(
        "Unexpected serialized DataFrame type"
    )


def unit(x):
    x = np.asarray(x, dtype=float)

    n = np.linalg.norm(x)

    if n == 0:
        raise RuntimeError(
            "Zero-length vector"
        )

    return x / n


def cosine(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = (
        np.linalg.norm(x)
        * np.linalg.norm(y)
    )

    if denom == 0:
        return np.nan

    return float(
        np.dot(x, y)
        / denom
    )


def pa_fraction(values):
    values = (
        pd.Series(values)
        .astype(str)
        .str.strip()
    )

    return float(
        values.str.fullmatch(
            r"PA\d{4}"
        ).mean()
    )


def load_counts(path):
    df = pd.read_csv(path)

    best_col = None
    best_score = 0.0

    for col in df.columns:
        score = pa_fraction(
            df[col]
        )

        if score > best_score:
            best_score = score
            best_col = col

    if best_score > 0.5:
        genes = (
            df[best_col]
            .astype(str)
            .str.strip()
        )

        matrix = (
            df.drop(
                columns=[best_col]
            )
            .copy()
        )

        matrix.index = genes

    else:
        n_pa_cols = sum(
            bool(
                re.fullmatch(
                    r"PA\d{4}",
                    str(col),
                )
            )
            for col in df.columns
        )

        if n_pa_cols < 1000:
            raise RuntimeError(
                "Could not identify PAO1 locus tags "
                "or expression-matrix orientation."
            )

        first = df.columns[0]

        if not re.fullmatch(
            r"PA\d{4}",
            str(first),
        ):
            matrix = (
                df.drop(
                    columns=[first]
                )
                .copy()
            )

            matrix.index = (
                df[first]
                .astype(str)
            )

        else:
            matrix = df.copy()

        matrix = matrix.T

    matrix = matrix.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if matrix.isna().any().any():
        raise RuntimeError(
            "Count matrix contains non-numeric values."
        )

    matrix.index = (
        matrix.index
        .astype(str)
        .str.strip()
    )

    matrix = matrix.loc[
        matrix.index.str.fullmatch(
            r"PA\d{4}"
        )
    ]

    duplicates = int(
        matrix.index
        .duplicated()
        .sum()
    )

    if duplicates:
        matrix = (
            matrix.groupby(
                level=0
            )
            .sum()
        )

    return matrix, duplicates


def sample_context(sample, design):
    pieces = [
        str(sample)
    ]

    # Sample may itself be a column in the design matrix.
    if sample in design.columns:
        pieces.extend(
            design[sample]
            .fillna("")
            .astype(str)
            .tolist()
        )

    # Or sample may occur inside one of the design rows.
    for _, row in design.iterrows():
        values = (
            row.fillna("")
            .astype(str)
        )

        match = False

        for value in values:
            if value == sample:
                match = True
                break

            if (
                len(value) > 5
                and (
                    value in str(sample)
                    or str(sample) in value
                )
            ):
                match = True
                break

        if match:
            pieces.extend(
                values.tolist()
            )

    return " ".join(
        pieces
    ).lower()


def classify_sample(sample, design):
    text = sample_context(
        sample,
        design,
    )

    # We want fresh CAA only.
    fresh_caa = (
        (
            "in_caa" in text
            or "100% caa" in text
            or "fresh caa" in text
            or "caa" in text
        )
        and "pa_sn" not in text
        and "bc_sn" not in text
        and "supernat" not in text
        and "spent medium" not in text
    )

    if not fresh_caa:
        return None

    low = (
        "felow" in text
        or "fe_low" in text
        or "low iron" in text
        or "iron low" in text
        or "iron availability: low" in text
    )

    high = (
        "fehigh" in text
        or "fe_high" in text
        or "high iron" in text
        or "iron high" in text
        or "iron availability: high" in text
    )

    if low and not high:
        return "low_iron"

    if high and not low:
        return "high_iron"

    return None


def gene_lengths(path):
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

    out = {}

    for feature_type in [
        "CDS",
        "gene",
        "tRNA",
        "rRNA",
        "ncRNA",
    ]:
        for feature in record.features:
            if feature.type != feature_type:
                continue

            tags = feature.qualifiers.get(
                "locus_tag",
                [],
            )

            if not tags:
                continue

            gene = str(
                tags[0]
            )

            if not re.fullmatch(
                r"PA\d{4}",
                gene,
            ):
                continue

            if gene not in out:
                out[gene] = int(
                    len(
                        feature.location
                    )
                )

    result = pd.Series(
        out,
        dtype=float,
    )

    if len(result) < 5000:
        raise RuntimeError(
            "Too few PAO1 gene lengths: "
            + str(
                len(result)
            )
        )

    return result


def mean_pairwise_corr(df, columns):
    corr = df[columns].corr()

    vals = [
        float(
            corr.iloc[i, j]
        )
        for i in range(
            len(columns)
        )
        for j in range(
            i + 1,
            len(columns),
        )
    ]

    return float(
        np.mean(vals)
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

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    return float(
        np.corrcoef(
            x[mask],
            y[mask],
        )[0, 1]
    )


def random_set_test(
    abs_delta,
    programme_names,
    target_names,
    n_perm,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    target_names = [
        name
        for name in target_names
        if name in programme_names
    ]

    n_target = len(
        target_names
    )

    if n_target == 0:
        raise RuntimeError(
            "No iron programmes found"
        )

    target_indices = [
        programme_names.index(
            name
        )
        for name in target_names
    ]

    observed = float(
        np.mean(
            abs_delta[
                target_indices
            ]
        )
    )

    null = np.empty(
        n_perm,
        dtype=float,
    )

    all_indices = np.arange(
        len(
            programme_names
        )
    )

    for i in range(
        n_perm
    ):
        idx = rng.choice(
            all_indices,
            size=n_target,
            replace=False,
        )

        null[i] = float(
            np.mean(
                abs_delta[
                    idx
                ]
            )
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

    return {
        "n_programmes":
            n_target,

        "observed_mean_abs_delta":
            observed,

        "null_mean":
            float(
                np.mean(
                    null
                )
            ),

        "null_sd":
            float(
                np.std(
                    null
                )
            ),

        "permutation_p":
            float(
                p
            ),
    }


def make_existing_anchor_basis(
    contrasts,
    order,
):
    stationary = (
        contrasts.loc[
            order,
            "stationary_vs_planktonic",
        ]
        .to_numpy(
            dtype=float
        )
    )

    antibiotic = (
        contrasts.loc[
            order,
            "persister_vs_stationary",
        ]
        .to_numpy(
            dtype=float
        )
    )

    B = np.column_stack(
        [
            unit(
                stationary
            ),
            unit(
                antibiotic
            ),
        ]
    )

    Q, _ = np.linalg.qr(
        B
    )

    return Q[:, :2]


def fraction_energy(
    target,
    Q,
):
    target = np.asarray(
        target,
        dtype=float,
    )

    projected = (
        Q
        @ (
            Q.T
            @ target
        )
    )

    return float(
        np.dot(
            projected,
            projected,
        )
        / np.dot(
            target,
            target,
        )
    )


def iron_increment_permutation(
    target,
    iron,
    Q2,
    observed_increment,
    n_perm,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    target = np.asarray(
        target,
        dtype=float,
    )

    null = np.empty(
        n_perm,
        dtype=float,
    )

    denominator = float(
        np.dot(
            target,
            target,
        )
    )

    for i in range(
        n_perm
    ):
        shuffled = iron[
            rng.permutation(
                len(
                    iron
                )
            )
        ]

        orthogonal = (
            shuffled
            - Q2
            @ (
                Q2.T
                @ shuffled
            )
        )

        norm = np.linalg.norm(
            orthogonal
        )

        if norm < 1e-12:
            null[i] = 0.0
            continue

        u = (
            orthogonal
            / norm
        )

        null[i] = float(
            np.dot(
                target,
                u,
            )
            ** 2
            / denominator
        )

    p = (
        1
        + int(
            np.sum(
                null
                >= observed_increment
            )
        )
    ) / (
        n_perm
        + 1
    )

    return null, float(
        p
    )


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 08")
    print("Controlled iron-limitation iModulon anchor: GSE224821")
    print("=" * 78)

    required = [
        COUNTS,
        DESIGN,
        GENBANK,
        MODEL,
        CATALOGUE,
        PRECISE_SAMPLES,
        AUDIT03,
        AUDIT04,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # 1. Provenance
    # ------------------------------------------------------------

    precise = pd.read_csv(
        PRECISE_SAMPLES,
        low_memory=False,
    )

    aliases = [
        "GSE224821",
        "PRJNA932752",
    ]

    overlap_rows = []

    for alias in aliases:
        mask = pd.Series(
            False,
            index=precise.index,
        )

        for col in precise.columns:
            mask = (
                mask
                |
                precise[col]
                .fillna("")
                .astype(str)
                .str.contains(
                    alias,
                    regex=False,
                )
            )

        overlap_rows.append(
            {
                "alias":
                    alias,

                "n_matching_rows":
                    int(
                        mask.sum()
                    ),
            }
        )

    overlap = pd.DataFrame(
        overlap_rows
    )

    overlap.to_csv(
        OUT
        / "aeruprecise_overlap_screen.csv",
        index=False,
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
    print(
        "aeruPRECISE overlap:",
        training_overlap,
    )

    if training_overlap:
        raise RuntimeError(
            "GSE224821 unexpectedly overlaps "
            "the frozen aeruPRECISE compendium."
        )

    # ------------------------------------------------------------
    # 2. Count matrix and conditions
    # ------------------------------------------------------------

    counts, duplicates = load_counts(
        COUNTS
    )

    design = pd.read_csv(
        DESIGN
    )

    groups = {
        "low_iron": [],
        "high_iron": [],
    }

    contexts = []

    for sample in counts.columns:
        condition = classify_sample(
            sample,
            design,
        )

        contexts.append(
            {
                "sample":
                    sample,

                "classification":
                    condition,

                "context":
                    sample_context(
                        sample,
                        design,
                    ),
            }
        )

        if condition is not None:
            groups[
                condition
            ].append(
                sample
            )

    context_df = pd.DataFrame(
        contexts
    )

    context_df.to_csv(
        OUT
        / "sample_classification.csv",
        index=False,
    )

    print()
    print("Count matrix")
    print(
        "  genes:",
        counts.shape[0],
    )
    print(
        "  all samples:",
        counts.shape[1],
    )
    print(
        "  duplicate loci collapsed:",
        duplicates,
    )

    print()
    print("Fresh CAA iron conditions")
    print(
        "  low iron:",
        groups["low_iron"],
    )
    print(
        "  high iron:",
        groups["high_iron"],
    )

    if (
        len(
            groups["low_iron"]
        )
        != 3
        or len(
            groups["high_iron"]
        )
        != 3
    ):
        print()
        print(
            context_df[
                [
                    "sample",
                    "classification",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print("Design matrix")
        print(
            design.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Expected exactly 3 fresh-CAA low-iron "
            "and 3 fresh-CAA high-iron samples."
        )

    selected = (
        groups["high_iron"]
        + groups["low_iron"]
    )

    counts = counts[
        selected
    ]

    # ------------------------------------------------------------
    # 3. Counts -> TPM -> log2(TPM+1)
    # ------------------------------------------------------------

    lengths = gene_lengths(
        GENBANK
    )

    # ------------------------------------------------------------
    # GEO/publication filtering correction
    #
    # The GSE224821 study states that genes with no reads across
    # the compared conditions were excluded from downstream data.
    # Therefore genes absent from the processed count matrix but
    # present in the canonical PAO1 annotation are restored as
    # explicit zero-count genes before TPM calculation.
    #
    # We do NOT impute non-zero expression.
    # ------------------------------------------------------------

    original_count_genes = len(counts.index)

    retained_all_zero = int(
        (counts.sum(axis=1) == 0).sum()
    )

    canonical_missing = (
        lengths.index
        .difference(
            counts.index
        )
    )

    noncanonical_input = (
        counts.index
        .difference(
            lengths.index
        )
    )

    print()
    print("Processed-matrix filtering diagnostic")
    print(
        "  genes in processed count matrix:",
        original_count_genes,
    )
    print(
        "  retained genes with zero counts in all selected samples:",
        retained_all_zero,
    )
    print(
        "  canonical PAO1 genes absent from processed matrix:",
        len(canonical_missing),
    )
    print(
        "  input genes absent from canonical annotation:",
        len(noncanonical_input),
    )

    # Restore the canonical PAO1 universe.
    # Missing rows are filled with literal zero counts.
    counts = counts.reindex(
        lengths.index,
        fill_value=0.0,
    )

    print(
        "  canonical genes after zero completion:",
        counts.shape[0],
    )

    zero_completion = pd.DataFrame(
        {
            "gene": canonical_missing.astype(str),
            "restored_count_value": 0.0,
            "reason": (
                "absent_from_processed_GSE224821_matrix;"
                "publication_reports_zero-read genes_filtered"
            ),
        }
    )

    zero_completion.to_csv(
        OUT / "zero_completed_genes.csv",
        index=False,
    )

    rpk = counts.divide(
        lengths
        / 1000.0,
        axis=0,
    )

    tpm = rpk.divide(
        rpk.sum(
            axis=0
        )
        / 1_000_000.0,
        axis=1,
    )

    log_tpm = np.log2(
        tpm
        + 1.0
    )

    # High iron is reference.
    reference = (
        log_tpm[
            groups[
                "high_iron"
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

    # ------------------------------------------------------------
    # 4. Frozen iModulon projection
    # ------------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(
            handle
        )

    M = (
        json_df(
            raw["M"]
        )
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    X_train = (
        json_df(
            raw["X"]
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
        len(shared)
        / len(M.index)
    )

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

    print()
    print("aeruPRECISE compatibility")
    print(
        "  model genes:",
        len(M),
    )
    print(
        "  shared genes:",
        len(shared),
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

    # ------------------------------------------------------------
    # 5. Technical QC
    # ------------------------------------------------------------

    low_r = mean_pairwise_corr(
        log_tpm,
        groups["low_iron"],
    )

    high_r = mean_pairwise_corr(
        log_tpm,
        groups["high_iron"],
    )

    max_reference_activity = float(
        activities[
            groups[
                "high_iron"
            ]
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
    print("Projection QC")
    print(
        "  high-iron replicate mean r:",
        round(
            high_r,
            4,
        ),
    )
    print(
        "  low-iron replicate mean r:",
        round(
            low_r,
            4,
        ),
    )
    print(
        "  max |mean high-iron activity|:",
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

    # ------------------------------------------------------------
    # 6. Iron activity vector
    # ------------------------------------------------------------

    high_mean = (
        activities[
            groups[
                "high_iron"
            ]
        ]
        .mean(
            axis=1
        )
    )

    low_mean = (
        activities[
            groups[
                "low_iron"
            ]
        ]
        .mean(
            axis=1
        )
    )

    delta = (
        low_mean
        - high_mean
    )

    programme = pd.DataFrame(
        {
            "imodulon":
                activities.index,

            "high_iron_mean_activity":
                high_mean.values,

            "low_iron_mean_activity":
                low_mean.values,

            "delta_low_minus_high":
                delta.values,

            "weighted_gene_coverage":
                weighted_coverage
                .reindex(
                    activities.index
                )
                .values,
        }
    )

    programme[
        "abs_delta"
    ] = (
        programme[
            "delta_low_minus_high"
        ]
        .abs()
    )

    programme[
        "abs_rank"
    ] = (
        programme[
            "abs_delta"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    catalogue = pd.read_csv(
        CATALOGUE
    )

    annotation = catalogue[
        [
            "name",
            "regulator_readable",
            "function",
            "category",
        ]
    ].rename(
        columns={
            "name":
                "imodulon"
        }
    )

    programme = programme.merge(
        annotation,
        on="imodulon",
        how="left",
        validate="one_to_one",
    )

    programme[
        "iron_core"
    ] = (
        programme[
            "imodulon"
        ]
        .isin(
            IRON_CORE
        )
    )

    programme[
        "iron_extended"
    ] = (
        programme[
            "imodulon"
        ]
        .isin(
            IRON_EXTENDED
        )
    )

    programme = programme.sort_values(
        "abs_delta",
        ascending=False,
    )

    # ------------------------------------------------------------
    # 7. Is the published iron set unusually responsive?
    # ------------------------------------------------------------

    names = (
        activities.index
        .astype(str)
        .tolist()
    )

    abs_delta = (
        delta
        .reindex(
            names
        )
        .abs()
        .to_numpy(
            dtype=float
        )
    )

    core_test = random_set_test(
        abs_delta,
        names,
        IRON_CORE,
        N_PERM,
        SEED,
    )

    extended_test = random_set_test(
        abs_delta,
        names,
        IRON_EXTENDED,
        N_PERM,
        SEED + 1,
    )

    iron_set_tests = pd.DataFrame(
        [
            {
                "set":
                    "iron_core",
                **core_test,
            },
            {
                "set":
                    "iron_extended",
                **extended_test,
            },
        ]
    )

    print()
    print("Largest low-iron vs high-iron programme changes")
    print(
        programme[
            [
                "abs_rank",
                "imodulon",
                "delta_low_minus_high",
                "function",
                "category",
                "iron_extended",
            ]
        ]
        .head(20)
        .round(
            {
                "delta_low_minus_high":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Predefined iron programme responses")
    print(
        programme.loc[
            programme[
                "iron_extended"
            ],
            [
                "imodulon",
                "delta_low_minus_high",
                "abs_rank",
                "function",
            ],
        ]
        .sort_values(
            "abs_rank"
        )
        .round(
            {
                "delta_low_minus_high":
                    3,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Iron-set enrichment")
    print(
        iron_set_tests
        .round(4)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # 8. Does iron explain the prior biofilm residual?
    # ------------------------------------------------------------

    controlled = pd.read_csv(
        AUDIT04,
        index_col=0,
    )

    gse203348 = (
        pd.read_csv(
            AUDIT03
        )
        .set_index(
            "imodulon"
        )[
            "delta_biofilm_minus_planktonic"
        ]
    )

    common = (
        activities.index
        .intersection(
            controlled.index
        )
        .intersection(
            gse203348.index
        )
    )

    if len(common) != 116:
        raise RuntimeError(
            "Expected 116 common iModulons for anchor expansion, found "
            + str(
                len(common)
            )
        )

    order = list(
        common
    )

    Q2 = make_existing_anchor_basis(
        controlled,
        order,
    )

    iron = (
        delta
        .reindex(
            order
        )
        .to_numpy(
            dtype=float
        )
    )

    # Remove any part of the iron vector that is already
    # represented by stationary/persistence axes.
    iron_orth = (
        iron
        - Q2
        @ (
            Q2.T
            @ iron
        )
    )

    original_iron_norm = float(
        np.linalg.norm(
            iron
        )
    )

    orthogonal_iron_norm = float(
        np.linalg.norm(
            iron_orth
        )
    )

    retained_unique_fraction = (
        orthogonal_iron_norm
        ** 2
        /
        original_iron_norm
        ** 2
    )

    if orthogonal_iron_norm < 1e-8:
        raise RuntimeError(
            "Iron vector adds no dimension beyond existing anchors."
        )

    iron_axis = unit(
        iron_orth
    )

    Q3 = np.column_stack(
        [
            Q2,
            iron_axis,
        ]
    )

    biofilm_203 = (
        gse203348
        .reindex(
            order
        )
        .to_numpy(
            dtype=float
        )
    )

    biofilm_120 = (
        controlled.loc[
            order,
            "biofilm_vs_planktonic",
        ]
        .to_numpy(
            dtype=float
        )
    )

    targets = {
        "GSE203348_biofilm":
            biofilm_203,

        "GSE120602_biofilm":
            biofilm_120,
    }

    expansion_rows = []

    for i, (
        target_name,
        target,
    ) in enumerate(
        targets.items()
    ):
        f2 = fraction_energy(
            target,
            Q2,
        )

        f3 = fraction_energy(
            target,
            Q3,
        )

        increment = (
            f3
            - f2
        )

        residual2 = (
            target
            - Q2
            @ (
                Q2.T
                @ target
            )
        )

        residual_iron_cosine = cosine(
            residual2,
            iron_axis,
        )

        residual_iron_energy = float(
            np.dot(
                residual2,
                iron_axis,
            )
            ** 2
            /
            np.dot(
                residual2,
                residual2,
            )
        )

        _, p = iron_increment_permutation(
            target=target,
            iron=iron,
            Q2=Q2,
            observed_increment=increment,
            n_perm=N_PERM,
            seed=SEED + 10 + i,
        )

        expansion_rows.append(
            {
                "target":
                    target_name,

                "stationary_antibiotic_fraction":
                    f2,

                "stationary_antibiotic_iron_fraction":
                    f3,

                "incremental_energy_from_iron":
                    increment,

                "fraction_prior_residual_explained_by_iron":
                    residual_iron_energy,

                "residual_vs_unique_iron_cosine":
                    residual_iron_cosine,

                "iron_increment_permutation_p":
                    p,
            }
        )

    expansion = pd.DataFrame(
        expansion_rows
    )

    print()
    print(
        "Iron axis relative to existing anchors"
    )
    print(
        "  unique iron energy fraction:",
        round(
            retained_unique_fraction,
            4,
        ),
    )

    print()
    print(
        "Expanded controlled-anchor model"
    )
    print(
        expansion
        .round(4)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------

    tpm.to_csv(
        OUT
        / "GSE224821_freshCAA_TPM.csv"
    )

    log_tpm.to_csv(
        OUT
        / "GSE224821_freshCAA_log2TPM_plus1.csv"
    )

    centered.to_csv(
        OUT
        / "GSE224821_freshCAA_centered_logTPM.csv"
    )

    activities.to_csv(
        OUT
        / "GSE224821_freshCAA_imodulon_activities.csv"
    )

    programme.to_csv(
        OUT
        / "iron_programme_response.csv",
        index=False,
    )

    iron_set_tests.to_csv(
        OUT
        / "iron_programme_set_enrichment.csv",
        index=False,
    )

    expansion.to_csv(
        OUT
        / "expanded_controlled_anchor_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "imodulon":
                order,

            "iron_low_minus_high":
                iron,

            "iron_unique_after_existing_anchors":
                iron_orth,
        }
    ).to_csv(
        OUT
        / "iron_anchor_vector.csv",
        index=False,
    )

    qc = pd.DataFrame(
        [
            [
                "model_gene_coverage",
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
                "high_iron_replicate_r",
                high_r,
            ],
            [
                "low_iron_replicate_r",
                low_r,
            ],
            [
                "max_abs_reference_activity",
                max_reference_activity,
            ],
            [
                "projection_correlation",
                projection_corr,
            ],
            [
                "projection_error",
                projection_error,
            ],
            [
                "external_training_sd_ratio",
                scale_ratio,
            ],
            [
                "unique_iron_energy_fraction",
                retained_unique_fraction,
            ],
        ],
        columns=[
            "metric",
            "value",
        ],
    )

    qc.to_csv(
        OUT
        / "projection_qc.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 10. Technical pass gate
    # ------------------------------------------------------------

    checks = {
        "no_training_overlap":
            not training_overlap,

        "3_low_and_3_high_samples":
            (
                len(
                    groups[
                        "low_iron"
                    ]
                )
                == 3
                and len(
                    groups[
                        "high_iron"
                    ]
                )
                == 3
            ),

        "model_gene_coverage_ge_90pct":
            coverage >= 0.90,

        "median_weighted_coverage_ge_95pct":
            median_weighted >= 0.95,

        "minimum_weighted_coverage_ge_80pct":
            minimum_weighted >= 0.80,

        "reference_centred":
            max_reference_activity < 1e-8,

        "finite_activities":
            np.isfinite(
                A_values
            ).all(),

        "iron_adds_nonzero_dimension":
            orthogonal_iron_norm > 1e-8,
    }

    passed = all(
        checks.values()
    )

    pd.DataFrame(
        [
            {
                "check": key,
                "pass": value,
            }
            for key, value
            in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    g203 = expansion.loc[
        expansion[
            "target"
        ]
        == "GSE203348_biofilm"
    ].iloc[0]

    lines = [
        "# CF-State Phase 2 — Audit 08",
        "",
        "## Objective",
        "",
        "Construct a controlled PAO1 iron-limitation iModulon axis "
        "using fresh CAA low-iron versus high-iron GSE224821 samples, "
        "then test whether this axis explains independent biofilm "
        "programme structure left unexplained by stationary/persistence.",
        "",
        "## Contrast",
        "",
        "- Low iron minus high iron",
        "- 3 biological replicates per condition",
        "- Fresh CAA conditions only",
        "- Supernatant conditions excluded",
        "",
        "## Technical QC",
        "",
        f"- Gene coverage: {coverage:.2%}",
        f"- Median weighted coverage: {median_weighted:.4f}",
        f"- Minimum weighted coverage: {minimum_weighted:.4f}",
        f"- High-iron replicate r: {high_r:.4f}",
        f"- Low-iron replicate r: {low_r:.4f}",
        f"- Projection correlation: {projection_corr:.4f}",
        f"- Relative projection error: {projection_error:.4f}",
        "",
        "## Iron programme-set test",
        "",
        f"- Core iron-set permutation p: "
        f"{core_test['permutation_p']:.4g}",
        f"- Extended iron-set permutation p: "
        f"{extended_test['permutation_p']:.4g}",
        "",
        "## Independent GSE203348 biofilm test",
        "",
        f"- Previous stationary+antibiotic energy explained: "
        f"{g203['stationary_antibiotic_fraction']:.2%}",
        f"- With independent iron axis added: "
        f"{g203['stationary_antibiotic_iron_fraction']:.2%}",
        f"- Increment attributable to iron: "
        f"{g203['incremental_energy_from_iron']:.2%}",
        f"- Fraction of prior residual explained by iron: "
        f"{g203['fraction_prior_residual_explained_by_iron']:.2%}",
        f"- Permutation p for iron increment: "
        f"{g203['iron_increment_permutation_p']:.4g}",
        "",
        "## Interpretation boundary",
        "",
        "A significant result would support GSE224821 low-vs-high iron "
        "as an experimentally anchored iron axis in CF-State. "
        "It would not establish that all PvdS/FoxR/PchR activity in "
        "other experiments is caused exclusively by iron limitation.",
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
        / "AUDIT_08_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 08:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_08_SUMMARY.md",
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
