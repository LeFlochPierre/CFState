from pathlib import Path
from io import StringIO
import gzip, json, re
import numpy as np
import pandas as pd

ROOT = Path.cwd()

RAW = ROOT / "data/raw/expanded_biofilm/GSE120602/published_counts"
GB = ROOT / "references/mappings/NC_002516.2.gb"
MODEL = ROOT / "external/aeruprecise/precise411/paeru_116_iModulon.json"
PRECISE_SAMPLES = ROOT / "external/aeruprecise/precise411/sample_table.csv"

AUDIT03 = (
    ROOT
    / "results/phase2/audit_03_external_pao1_activity_inference"
    / "GSE203348_activity_summary.csv"
)

OUT = ROOT / "results/phase2/audit_04_controlled_state_anchoring_GSE120602"


def json_df(v):
    if isinstance(v, str):
        return pd.read_json(StringIO(v))
    if isinstance(v, dict):
        return pd.DataFrame(v)
    raise RuntimeError("Unexpected serialized dataframe type")


def condition_from_name(name):
    s = name.lower()

    if "plusciprolarge" in s:
        return "persister"

    if "minusciprolarge" in s:
        return "stationary"

    if "paplanktonic" in s:
        return "planktonic"

    if "pabiofilm" in s:
        return "biofilm"

    return None


def read_counts(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    vals = {}

    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = re.split(r"[\t,\s]+", line.strip())

            pos = next(
                (
                    i
                    for i, token in enumerate(parts)
                    if re.fullmatch(r"PA\d{4}", token)
                ),
                None,
            )

            if pos is None:
                continue

            nums = []

            for token in parts[pos + 1:]:
                try:
                    nums.append(float(token))
                except ValueError:
                    pass

            if nums:
                gene = parts[pos]
                vals[gene] = vals.get(gene, 0.0) + nums[-1]

    if len(vals) < 4000:
        raise RuntimeError(
            f"Only {len(vals)} PAO1 loci recovered from {path.name}"
        )

    return pd.Series(vals, dtype=float)


def gene_lengths(path):
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Install Biopython with: "
            "conda install -c conda-forge biopython"
        )

    record = SeqIO.read(str(path), "genbank")
    out = {}

    for feature_type in ["CDS", "gene", "tRNA", "rRNA", "ncRNA"]:
        for feature in record.features:
            if feature.type != feature_type:
                continue

            tags = feature.qualifiers.get("locus_tag", [])

            if not tags:
                continue

            gene = str(tags[0])

            if not re.fullmatch(r"PA\d{4}", gene):
                continue

            if gene not in out:
                out[gene] = int(len(feature.location))

    lengths = pd.Series(out, dtype=float)

    if len(lengths) < 5000:
        raise RuntimeError(
            f"Only {len(lengths)} PAO1 gene lengths recovered"
        )

    return lengths


def mean_pairwise_corr(df, cols):
    corr = df[cols].corr()

    vals = [
        corr.iloc[i, j]
        for i in range(len(cols))
        for j in range(i + 1, len(cols))
    ]

    return float(np.mean(vals))


def cosine(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = np.linalg.norm(x) * np.linalg.norm(y)

    if denom == 0:
        return np.nan

    return float(np.dot(x, y) / denom)


def global_corr(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    good = np.isfinite(x) & np.isfinite(y)

    return float(
        np.corrcoef(
            x[good],
            y[good],
        )[0, 1]
    )


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 04")
    print("Controlled-state iModulon anchoring: GSE120602")
    print("=" * 78)

    for path in [RAW, GB, MODEL, PRECISE_SAMPLES]:
        if not path.exists():
            raise FileNotFoundError(str(path))

    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Provenance check
    # ------------------------------------------------------------

    precise = pd.read_csv(
        PRECISE_SAMPLES,
        low_memory=False,
    )

    aliases = [
        "GSE120602",
        "PRJNA493733",
        "SRP162817",
        "GSM3405609",
        "GSM3405610",
        "GSM3405611",
        "GSM3405612",
        "GSM3405613",
        "GSM3405614",
        "GSM3405615",
        "GSM3405616",
    ]

    provenance = []

    for alias in aliases:
        mask = pd.Series(
            False,
            index=precise.index,
        )

        for column in precise.columns:
            values = (
                precise[column]
                .fillna("")
                .astype(str)
            )

            mask = mask | values.str.contains(
                alias,
                regex=False,
            )

        provenance.append(
            {
                "alias": alias,
                "n_matching_rows": int(mask.sum()),
            }
        )

    provenance = pd.DataFrame(provenance)

    provenance.to_csv(
        OUT / "aeruprecise_alias_overlap_screen.csv",
        index=False,
    )

    training_overlap = bool(
        (provenance["n_matching_rows"] > 0).any()
    )

    print()
    print("aeruPRECISE alias-level overlap:", training_overlap)

    if training_overlap:
        print(
            provenance.loc[
                provenance["n_matching_rows"] > 0
            ].to_string(index=False)
        )

    # ------------------------------------------------------------
    # Load eight published count files
    # ------------------------------------------------------------

    paths = sorted(
        RAW.glob("*_readcount.txt.gz")
    )

    if len(paths) != 8:
        raise RuntimeError(
            f"Expected 8 read-count files, found {len(paths)}"
        )

    series = []
    metadata_rows = []

    for path in paths:
        condition = condition_from_name(
            path.name
        )

        if condition is None:
            raise RuntimeError(
                f"Cannot classify {path.name}"
            )

        sample = (
            path.name
            .replace("_readcount.txt.gz", "")
        )

        s = read_counts(path)
        s.name = sample

        series.append(s)

        metadata_rows.append(
            {
                "sample": sample,
                "condition": condition,
                "source_file": str(
                    path.relative_to(ROOT)
                ),
            }
        )

    counts = (
        pd.concat(
            series,
            axis=1,
        )
        .fillna(0.0)
    )

    metadata = pd.DataFrame(
        metadata_rows
    )

    condition_order = [
        "planktonic",
        "biofilm",
        "stationary",
        "persister",
    ]

    groups = {
        condition: metadata.loc[
            metadata["condition"] == condition,
            "sample",
        ].tolist()

        for condition in condition_order
    }

    print()
    print("Samples")

    for condition in condition_order:
        print(
            f"  {condition}: "
            f"{groups[condition]}"
        )

        if len(groups[condition]) != 2:
            raise RuntimeError(
                f"Expected 2 {condition} replicates"
            )

    # ------------------------------------------------------------
    # Counts -> TPM -> log2(TPM+1)
    # ------------------------------------------------------------

    lengths = gene_lengths(GB)

    usable = counts.index.intersection(
        lengths.index
    )

    counts = counts.loc[usable]
    lengths = lengths.loc[usable]

    rpk = counts.divide(
        lengths / 1000.0,
        axis=0,
    )

    tpm = rpk.divide(
        rpk.sum(axis=0) / 1_000_000.0,
        axis=1,
    )

    log_tpm = np.log2(
        tpm + 1.0
    )

    planktonic = groups["planktonic"]

    reference = (
        log_tpm[
            planktonic
        ]
        .mean(axis=1)
    )

    centered = (
        log_tpm.subtract(
            reference,
            axis=0,
        )
    )

    # ------------------------------------------------------------
    # Frozen aeruPRECISE projection
    # ------------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)

    M = (
        json_df(raw["M"])
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    X_train = (
        json_df(raw["X"])
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    M.index = M.index.astype(str)
    centered.index = centered.index.astype(str)

    shared = M.index.intersection(
        centered.index
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
        metadata["sample"].tolist(),
    ]

    weighted_coverage = (
        M_shared.abs().sum(axis=0)
        / M.abs().sum(axis=0)
    )

    median_weighted = float(
        weighted_coverage.median()
    )

    minimum_weighted = float(
        weighted_coverage.min()
    )

    print()
    print("aeruPRECISE compatibility")
    print("  model genes:", len(M))
    print("  shared genes:", len(shared))
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
        raise RuntimeError(
            "Model gene coverage below 90%"
        )

    A_values = (
        np.linalg.pinv(
            M_shared.to_numpy(
                dtype=float
            )
        )
        @ X_new.to_numpy(
            dtype=float
        )
    )

    activities = pd.DataFrame(
        A_values,
        index=M.columns.astype(str),
        columns=X_new.columns,
    )

    # ------------------------------------------------------------
    # Technical QC
    # ------------------------------------------------------------

    replicate_qc = pd.DataFrame(
        [
            {
                "condition": condition,
                "mean_pairwise_logTPM_r":
                    mean_pairwise_corr(
                        log_tpm,
                        groups[condition],
                    ),
            }

            for condition
            in condition_order
        ]
    )

    max_reference_activity = float(
        activities[
            groups["planktonic"]
        ]
        .mean(axis=1)
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
        / np.linalg.norm(
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
        replicate_qc.to_string(
            index=False
        )
    )

    print()
    print("Projection QC")
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

    # ------------------------------------------------------------
    # Condition means
    # ------------------------------------------------------------

    means = {
        condition:
            activities[
                groups[condition]
            ].mean(axis=1)

        for condition
        in condition_order
    }

    condition_means = pd.DataFrame(
        means
    )

    # ------------------------------------------------------------
    # Controlled transition vectors
    # ------------------------------------------------------------

    contrasts = pd.DataFrame(
        index=activities.index
    )

    contrasts[
        "biofilm_vs_planktonic"
    ] = (
        means["biofilm"]
        - means["planktonic"]
    )

    contrasts[
        "stationary_vs_planktonic"
    ] = (
        means["stationary"]
        - means["planktonic"]
    )

    contrasts[
        "persister_vs_stationary"
    ] = (
        means["persister"]
        - means["stationary"]
    )

    contrasts[
        "persister_vs_planktonic"
    ] = (
        means["persister"]
        - means["planktonic"]
    )

    # ------------------------------------------------------------
    # Similarity among transitions
    # ------------------------------------------------------------

    similarity_rows = []

    for c1 in contrasts.columns:
        for c2 in contrasts.columns:
            similarity_rows.append(
                {
                    "contrast_1": c1,
                    "contrast_2": c2,
                    "pearson":
                        float(
                            contrasts[c1].corr(
                                contrasts[c2],
                                method="pearson",
                            )
                        ),
                    "spearman":
                        float(
                            contrasts[c1].corr(
                                contrasts[c2],
                                method="spearman",
                            )
                        ),
                    "cosine":
                        cosine(
                            contrasts[c1],
                            contrasts[c2],
                        ),
                }
            )

    similarity = pd.DataFrame(
        similarity_rows
    )

    # ------------------------------------------------------------
    # Top programme changes
    # ------------------------------------------------------------

    top_rows = []

    for contrast in contrasts.columns:
        ranked = (
            contrasts[
                contrast
            ]
            .abs()
            .sort_values(
                ascending=False
            )
            .head(20)
        )

        for rank, imodulon in enumerate(
            ranked.index,
            start=1,
        ):
            top_rows.append(
                {
                    "contrast":
                        contrast,

                    "rank":
                        rank,

                    "imodulon":
                        imodulon,

                    "delta_activity":
                        float(
                            contrasts.loc[
                                imodulon,
                                contrast,
                            ]
                        ),

                    "weighted_gene_coverage":
                        float(
                            weighted_coverage.loc[
                                imodulon
                            ]
                        ),
                }
            )

    top_programmes = pd.DataFrame(
        top_rows
    )

    # ------------------------------------------------------------
    # Compare biofilm transition to independent Audit 03 study
    # ------------------------------------------------------------

    cross_study = None

    if AUDIT03.exists():
        previous = (
            pd.read_csv(
                AUDIT03
            )
            .set_index(
                "imodulon"
            )[
                "delta_biofilm_minus_planktonic"
            ]
        )

        current = contrasts[
            "biofilm_vs_planktonic"
        ]

        common = previous.index.intersection(
            current.index
        )

        x = previous.loc[
            common
        ]

        y = current.loc[
            common
        ]

        cross_study = {
            "n_imodulons":
                len(common),

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
                cosine(
                    x,
                    y,
                ),

            "sign_agreement":
                float(
                    (
                        np.sign(x.values)
                        == np.sign(y.values)
                    ).mean()
                ),
        }

        pd.DataFrame(
            {
                "GSE203348":
                    x,

                "GSE120602":
                    y,
            }
        ).to_csv(
            OUT
            / "GSE203348_vs_GSE120602_biofilm_programmes.csv"
        )

        pd.DataFrame(
            [cross_study]
        ).to_csv(
            OUT
            / "cross_study_biofilm_summary.csv",
            index=False,
        )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    metadata.to_csv(
        OUT / "sample_metadata.csv",
        index=False,
    )

    tpm.to_csv(
        OUT / "GSE120602_TPM.csv"
    )

    log_tpm.to_csv(
        OUT / "GSE120602_log2TPM_plus1.csv"
    )

    centered.to_csv(
        OUT / "GSE120602_centered_logTPM.csv"
    )

    activities.to_csv(
        OUT / "GSE120602_imodulon_activities.csv"
    )

    condition_means.to_csv(
        OUT / "condition_mean_activities.csv"
    )

    contrasts.to_csv(
        OUT / "controlled_contrast_vectors.csv"
    )

    similarity.to_csv(
        OUT / "controlled_contrast_similarity.csv",
        index=False,
    )

    top_programmes.to_csv(
        OUT / "top20_programmes_by_contrast.csv",
        index=False,
    )

    replicate_qc.to_csv(
        OUT / "replicate_qc.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print()
    print(
        "Controlled-state similarity "
        "(Spearman)"
    )

    print(
        similarity.pivot(
            index="contrast_1",
            columns="contrast_2",
            values="spearman",
        )
        .round(3)
        .to_string()
    )

    for contrast in contrasts.columns:
        print()
        print(
            "--- "
            + contrast
            + " ---"
        )

        display = top_programmes.loc[
            top_programmes[
                "contrast"
            ]
            == contrast,
            [
                "rank",
                "imodulon",
                "delta_activity",
            ],
        ].head(10)

        print(
            display.to_string(
                index=False
            )
        )

    if cross_study is not None:
        print()
        print(
            "Cross-study biofilm agreement "
            "(GSE203348 vs GSE120602)"
        )

        print(
            "  Pearson:",
            round(
                cross_study["pearson"],
                4,
            ),
        )

        print(
            "  Spearman:",
            round(
                cross_study["spearman"],
                4,
            ),
        )

        print(
            "  cosine:",
            round(
                cross_study["cosine"],
                4,
            ),
        )

        print(
            "  sign agreement:",
            format(
                cross_study[
                    "sign_agreement"
                ],
                ".2%",
            ),
        )

    # ------------------------------------------------------------
    # Audit gate
    # ------------------------------------------------------------

    checks = {
        "8_samples":
            counts.shape[1] == 8,

        "2_replicates_each":
            all(
                len(groups[c]) == 2
                for c in condition_order
            ),

        "gene_coverage_ge_90pct":
            coverage >= 0.90,

        "median_weighted_ge_95pct":
            median_weighted >= 0.95,

        "minimum_weighted_ge_80pct":
            minimum_weighted >= 0.80,

        "reference_centred":
            max_reference_activity < 1e-8,

        "finite_activities":
            np.isfinite(
                A_values
            ).all(),
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
        OUT / "audit_pass_checks.csv",
        index=False,
    )

    lines = [
        "# CF-State Phase 2 — Audit 04",
        "",
        "## Objective",
        "",
        "Project GSE120602 planktonic, biofilm, stationary and "
        "ciprofloxacin-persister conditions into frozen aeruPRECISE-411.",
        "",
        "## Provenance",
        "",
        f"- aeruPRECISE alias-level overlap detected: {training_overlap}",
        "",
        "## Technical QC",
        "",
        f"- Gene coverage: {coverage:.2%}",
        f"- Median weighted coverage: {median_weighted:.4f}",
        f"- Minimum weighted coverage: {minimum_weighted:.4f}",
        f"- corr(centered X, M@A): {projection_corr:.4f}",
        f"- Relative projection error: {projection_error:.4f}",
        f"- External/training SD ratio: {scale_ratio:.4f}",
        "",
        "## Decision",
        "",
        f"- PASS: {passed}",
        "",
        "## Interpretation boundary",
        "",
        "Programme responses are candidate anchors only. "
        "They require reproduction in an independent experiment "
        "before being treated as validated physiological readouts.",
    ]

    if cross_study is not None:
        lines.extend(
            [
                "",
                "## Independent biofilm comparison",
                "",
                f"- Pearson: {cross_study['pearson']:.4f}",
                f"- Spearman: {cross_study['spearman']:.4f}",
                f"- Cosine: {cross_study['cosine']:.4f}",
                f"- Sign agreement: {cross_study['sign_agreement']:.2%}",
            ]
        )

    (
        OUT
        / "AUDIT_04_SUMMARY.md"
    ).write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 04:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_04_SUMMARY.md",
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
