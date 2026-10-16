from pathlib import Path
from io import StringIO
import json
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

COUNTS = ROOT / "external/EPathDash/GEO_Datasets/Pseudomonas/Count_Tables/GSE203348_counts.csv"
DESIGN = ROOT / "external/EPathDash/GEO_Datasets/Pseudomonas/Design_Matrices/GSE203348_design.csv"
GENBANK = ROOT / "references/mappings/NC_002516.2.gb"
MODEL = ROOT / "external/aeruprecise/precise411/paeru_116_iModulon.json"

OUT = ROOT / "results/phase2/audit_03_external_pao1_activity_inference"


def read_json_df(value):
    if isinstance(value, str):
        return pd.read_json(StringIO(value))
    if isinstance(value, dict):
        return pd.DataFrame(value)
    raise RuntimeError("Unexpected serialized DataFrame type")


def pa_fraction(values):
    values = pd.Series(values).astype(str).str.strip()
    return float(values.str.fullmatch(r"PA\d{4}").mean())


def load_counts(path):
    df = pd.read_csv(path)

    # Case 1: genes are rows and one column contains PA#### identifiers.
    best_col = None
    best_score = 0.0

    for col in df.columns:
        score = pa_fraction(df[col])
        if score > best_score:
            best_score = score
            best_col = col

    if best_score > 0.5:
        genes = df[best_col].astype(str).str.strip()
        matrix = df.drop(columns=[best_col]).copy()
        matrix.index = genes

    # Case 2: genes are columns, samples are rows.
    elif sum(bool(re.fullmatch(r"PA\d{4}", str(c))) for c in df.columns) > 1000:
        first = df.columns[0]

        if not re.fullmatch(r"PA\d{4}", str(first)):
            sample_names = df[first].astype(str)
            matrix = df.drop(columns=[first]).copy()
            matrix.index = sample_names
        else:
            matrix = df.copy()

        matrix = matrix.T
        matrix.index = matrix.index.astype(str)

    else:
        raise RuntimeError(
            "Could not determine orientation or PAO1 identifier column "
            "for GSE203348_counts.csv"
        )

    matrix = matrix.apply(pd.to_numeric, errors="coerce")

    if matrix.isna().any().any():
        bad = int(matrix.isna().sum().sum())
        raise RuntimeError("Count matrix contains non-numeric values: " + str(bad))

    matrix.index = matrix.index.astype(str).str.strip()
    matrix = matrix.loc[matrix.index.str.fullmatch(r"PA\d{4}")]

    duplicates = int(matrix.index.duplicated().sum())

    if duplicates:
        # Counts for repeated canonical loci should be additive.
        matrix = matrix.groupby(level=0).sum()

    return matrix, duplicates


def classify_samples(columns):
    bio = []
    plank = []

    for col in columns:
        s = str(col).lower()

        if "biofwt" in s or ("biofilm" in s and "wt" in s):
            bio.append(col)

        elif "plankwt" in s or ("plank" in s and "wt" in s):
            plank.append(col)

    return plank, bio


def load_gene_lengths(genbank_path):
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython is required to read the existing PAO1 GenBank file. "
            "Install with: conda install -c conda-forge biopython"
        )

    record = SeqIO.read(str(genbank_path), "genbank")

    lengths = {}

    # Prefer CDS/gene feature coordinates carrying canonical locus tags.
    preferred_types = ["CDS", "gene", "tRNA", "rRNA", "ncRNA"]

    for feature_type in preferred_types:
        for feature in record.features:
            if feature.type != feature_type:
                continue

            tags = feature.qualifiers.get("locus_tag", [])

            if not tags:
                continue

            gene = str(tags[0])

            if not re.fullmatch(r"PA\d{4}", gene):
                continue

            if gene not in lengths:
                lengths[gene] = int(len(feature.location))

    result = pd.Series(lengths, name="length_bp")

    if len(result) < 5000:
        raise RuntimeError(
            "Too few canonical PAO1 gene lengths recovered: " + str(len(result))
        )

    return result


def pairwise_mean_corr(df, columns):
    corr = df[columns].corr()

    vals = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            vals.append(float(corr.iloc[i, j]))

    return float(np.mean(vals))


def global_corr(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    mask = np.isfinite(x) & np.isfinite(y)

    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 03")
    print("External PAO1 iModulon projection: GSE203348")
    print("=" * 78)

    for path in [COUNTS, DESIGN, GENBANK, MODEL]:
        if not path.exists():
            raise FileNotFoundError(str(path))

    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Source data
    # ------------------------------------------------------------

    counts, duplicate_genes = load_counts(COUNTS)

    print()
    print("Count matrix")
    print("  genes:", counts.shape[0])
    print("  samples:", counts.shape[1])
    print("  duplicate loci collapsed:", duplicate_genes)

    print()
    print("Sample columns:")
    for sample in counts.columns:
        print(" ", sample)

    design = pd.read_csv(DESIGN)
    design.to_csv(OUT / "GSE203348_design_copy.csv", index=False)

    plank_cols, bio_cols = classify_samples(counts.columns)

    print()
    print("Detected conditions")
    print("  planktonic:", plank_cols)
    print("  biofilm:", bio_cols)

    if len(plank_cols) != 3 or len(bio_cols) != 3:
        print()
        print("Design matrix:")
        print(design.to_string(index=False))

        raise RuntimeError(
            "Expected exactly 3 WT planktonic and 3 WT biofilm samples."
        )

    selected = plank_cols + bio_cols
    counts = counts[selected]

    # ------------------------------------------------------------
    # 2. PAO1 gene lengths
    # ------------------------------------------------------------

    lengths = load_gene_lengths(GENBANK)

    usable = counts.index.intersection(lengths.index)

    print()
    print("Gene-length mapping")
    print("  count genes:", len(counts.index))
    print("  PAO1 lengths:", len(lengths.index))
    print("  usable:", len(usable))

    counts = counts.loc[usable]
    lengths = lengths.loc[usable]

    # ------------------------------------------------------------
    # 3. Counts -> TPM
    # ------------------------------------------------------------

    length_kb = lengths / 1000.0

    rpk = counts.divide(length_kb, axis=0)

    scaling = rpk.sum(axis=0) / 1_000_000.0

    if (scaling <= 0).any():
        raise RuntimeError("Invalid TPM scaling factor")

    tpm = rpk.divide(scaling, axis=1)

    # PyModulon accepts TPM/log-TPM.
    # We use log2(TPM + 1), then centre gene-wise on matched planktonic controls.
    log_tpm = np.log2(tpm + 1.0)

    reference = log_tpm[plank_cols].mean(axis=1)

    centered = log_tpm.subtract(reference, axis=0)

    # ------------------------------------------------------------
    # 4. Frozen aeruPRECISE M
    # ------------------------------------------------------------

    with MODEL.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    M = read_json_df(raw["M"])
    X_train = read_json_df(raw["X"])

    M = M.apply(pd.to_numeric, errors="coerce")
    X_train = X_train.apply(pd.to_numeric, errors="coerce")

    M.index = M.index.astype(str)
    centered.index = centered.index.astype(str)

    shared = M.index.intersection(centered.index)

    coverage = len(shared) / len(M.index)

    print()
    print("aeruPRECISE compatibility")
    print("  model genes:", len(M.index))
    print("  shared genes:", len(shared))
    print("  gene coverage:", format(coverage, ".2%"))

    if coverage < 0.90:
        raise RuntimeError(
            "Model gene coverage below 90%; projection stopped."
        )

    M_shared = M.loc[shared]
    X_new = centered.loc[shared, selected]

    # ------------------------------------------------------------
    # 5. Weighted programme coverage
    # ------------------------------------------------------------

    total_weight = M.abs().sum(axis=0)
    retained_weight = M_shared.abs().sum(axis=0)

    weighted_coverage = retained_weight / total_weight

    median_weighted = float(weighted_coverage.median())
    minimum_weighted = float(weighted_coverage.min())

    print("  median weighted programme coverage:", round(median_weighted, 4))
    print("  minimum weighted programme coverage:", round(minimum_weighted, 4))

    # ------------------------------------------------------------
    # 6. Frozen projection
    #
    # Same mathematical operation used by infer_activities:
    # A_new = pinv(M_shared) @ X_centered
    # ------------------------------------------------------------

    A_new_values = np.linalg.pinv(
        M_shared.to_numpy(dtype=float)
    ) @ X_new.to_numpy(dtype=float)

    A_new = pd.DataFrame(
        A_new_values,
        index=M.columns.astype(str),
        columns=selected,
    )

    # Since the reference samples were gene-wise centred,
    # their mean activity should also be numerically ~0.
    mean_reference_activity = A_new[plank_cols].mean(axis=1)
    max_reference_abs = float(mean_reference_activity.abs().max())

    # ------------------------------------------------------------
    # 7. QC
    # ------------------------------------------------------------

    plank_r = pairwise_mean_corr(log_tpm, plank_cols)
    bio_r = pairwise_mean_corr(log_tpm, bio_cols)

    reconstruction = M_shared.to_numpy(dtype=float) @ A_new_values

    projection_corr = global_corr(
        X_new.to_numpy(dtype=float),
        reconstruction,
    )

    denom = np.linalg.norm(X_new.to_numpy(dtype=float))

    projection_error = float(
        np.linalg.norm(
            X_new.to_numpy(dtype=float) - reconstruction
        ) / denom
    )

    # Compare basic scale with the original centred training X.
    training_sd = float(np.nanstd(X_train.to_numpy(dtype=float)))
    external_sd = float(np.nanstd(X_new.to_numpy(dtype=float)))
    scale_ratio = external_sd / training_sd

    print()
    print("Projection QC")
    print("  planktonic replicate mean r:", round(plank_r, 4))
    print("  biofilm replicate mean r:", round(bio_r, 4))
    print("  max |mean reference activity|:", format(max_reference_abs, ".3e"))
    print("  corr(centered X, M@A):", round(projection_corr, 4))
    print("  relative projection error:", round(projection_error, 4))
    print("  training X global SD:", round(training_sd, 4))
    print("  external centred X global SD:", round(external_sd, 4))
    print("  external/training SD ratio:", round(scale_ratio, 4))

    # ------------------------------------------------------------
    # 8. Condition-level activity changes
    # ------------------------------------------------------------

    plank_mean = A_new[plank_cols].mean(axis=1)
    bio_mean = A_new[bio_cols].mean(axis=1)

    delta = bio_mean - plank_mean

    activity_summary = pd.DataFrame(
        {
            "imodulon": A_new.index,
            "planktonic_mean_activity": plank_mean.values,
            "biofilm_mean_activity": bio_mean.values,
            "delta_biofilm_minus_planktonic": delta.values,
            "weighted_gene_coverage": weighted_coverage.reindex(A_new.index).values,
        }
    )

    activity_summary["abs_delta"] = (
        activity_summary["delta_biofilm_minus_planktonic"].abs()
    )

    activity_summary = activity_summary.sort_values(
        "abs_delta",
        ascending=False,
    )

    print()
    print("Largest activity changes")
    print("(DESCRIPTIVE ONLY — not yet biological anchors)")
    print(
        activity_summary.head(20)[
            [
                "imodulon",
                "delta_biofilm_minus_planktonic",
                "weighted_gene_coverage",
            ]
        ].to_string(index=False)
    )

    # ------------------------------------------------------------
    # 9. Save evidence
    # ------------------------------------------------------------

    tpm.to_csv(OUT / "GSE203348_TPM.csv")
    log_tpm.to_csv(OUT / "GSE203348_log2TPM_plus1.csv")
    centered.to_csv(OUT / "GSE203348_centered_logTPM.csv")
    A_new.to_csv(OUT / "GSE203348_imodulon_activities.csv")

    activity_summary.to_csv(
        OUT / "GSE203348_activity_summary.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "imodulon": M.columns.astype(str),
            "weighted_gene_coverage": weighted_coverage.values,
        }
    ).sort_values(
        "weighted_gene_coverage"
    ).to_csv(
        OUT / "imodulon_weighted_gene_coverage.csv",
        index=False,
    )

    qc = pd.DataFrame(
        [
            ["model_gene_coverage", coverage],
            ["median_weighted_programme_coverage", median_weighted],
            ["minimum_weighted_programme_coverage", minimum_weighted],
            ["planktonic_pairwise_logTPM_r", plank_r],
            ["biofilm_pairwise_logTPM_r", bio_r],
            ["max_abs_mean_reference_activity", max_reference_abs],
            ["corr_centered_expression_vs_MA", projection_corr],
            ["relative_projection_error", projection_error],
            ["training_X_global_sd", training_sd],
            ["external_centered_X_global_sd", external_sd],
            ["external_training_sd_ratio", scale_ratio],
        ],
        columns=["metric", "value"],
    )

    qc.to_csv(
        OUT / "projection_qc.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # 10. Audit gate
    # ------------------------------------------------------------

    checks = {
        "3_planktonic_and_3_biofilm": (
            len(plank_cols) == 3 and len(bio_cols) == 3
        ),
        "model_gene_coverage_ge_90pct": coverage >= 0.90,
        "median_weighted_coverage_ge_95pct": median_weighted >= 0.95,
        "minimum_weighted_coverage_ge_80pct": minimum_weighted >= 0.80,
        "reference_centred": max_reference_abs < 1e-8,
        "finite_activities": np.isfinite(A_new_values).all(),
    }

    passed = all(checks.values())

    pd.DataFrame(
        [{"check": k, "pass": v} for k, v in checks.items()]
    ).to_csv(
        OUT / "audit_pass_checks.csv",
        index=False,
    )

    summary = []

    summary.append("# CF-State Phase 2 — Audit 03")
    summary.append("")
    summary.append("## Objective")
    summary.append("")
    summary.append(
        "Test whether one external PAO1 RNA-seq experiment can be projected "
        "reproducibly into the frozen aeruPRECISE-411 iModulon representation."
    )
    summary.append("")
    summary.append("## Dataset")
    summary.append("")
    summary.append("- GSE203348")
    summary.append("- 3 WT planktonic replicates")
    summary.append("- 3 WT biofilm replicates")
    summary.append("- Existing EPathDash count table")
    summary.append("")
    summary.append("## Processing")
    summary.append("")
    summary.append("- Counts -> TPM using PAO1 NC_002516.2 gene lengths")
    summary.append("- log2(TPM + 1)")
    summary.append("- Gene-wise centring on mean WT planktonic reference")
    summary.append("- Frozen projection: pinv(M_shared) @ X_centered")
    summary.append("")
    summary.append("## QC")
    summary.append("")
    summary.append("Gene coverage: " + format(coverage, ".2%"))
    summary.append("Median weighted coverage: " + format(median_weighted, ".4f"))
    summary.append("Minimum weighted coverage: " + format(minimum_weighted, ".4f"))
    summary.append("Planktonic replicate r: " + format(plank_r, ".4f"))
    summary.append("Biofilm replicate r: " + format(bio_r, ".4f"))
    summary.append("Projection correlation: " + format(projection_corr, ".4f"))
    summary.append("Relative projection error: " + format(projection_error, ".4f"))
    summary.append("External/training SD ratio: " + format(scale_ratio, ".4f"))
    summary.append("")
    summary.append("## Decision")
    summary.append("")
    summary.append("PASS: " + str(passed))
    summary.append("")
    summary.append("## Interpretation boundary")
    summary.append("")
    summary.append(
        "The activity differences observed here are descriptive technical "
        "outputs only. They are not yet validated physiological-state anchors."
    )

    (OUT / "AUDIT_03_SUMMARY.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("Audit 03:", "PASS" if passed else "FAIL")
    print("Summary:", OUT / "AUDIT_03_SUMMARY.md")

    if not passed:
        failed = [k for k, v in checks.items() if not v]
        raise RuntimeError("Failed checks: " + ", ".join(failed))


if __name__ == "__main__":
    main()
