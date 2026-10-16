from pathlib import Path
import hashlib
import traceback

import numpy as np
import pandas as pd


ROOT = Path.cwd()

ATLANTA_DIR = ROOT / (
    "results/phase4/"
    "audit_04g_PRJNA576508_Atlanta_full_cohort_processing"
)

CFBE_DIR = ROOT / (
    "results/phase4/"
    "audit_04i3_PRJNA576508_CFBE_reference_metadata_repair"
)

OUT = ROOT / (
    "results/phase4/"
    "audit_04j_PRJNA576508_Atlanta_expression_normalisation"
)

ATLANTA = ATLANTA_DIR / (
    "FROZEN_Atlanta_13_raw_PAO1_gene_counts.csv"
)

REFERENCE = CFBE_DIR / (
    "FROZEN_CFBE_log2TPM_reference_profiles.csv"
)

CFBE_FOUR = CFBE_DIR / (
    "FROZEN_CFBE_four_log2TPM_profiles.csv"
)

SAF = ROOT / (
    "references/phase4/PRJNA576508_cornforth/"
    "Cornforth_PAO1_no_decimal_locus_tags.saf"
)

EXPECTED_SAMPLES = {
    "EM6", "EM13", "EM24", "EM30", "EM33",
    "EM35", "EM36", "EM37", "EM43",
    "EM48", "EM49", "EM53", "EM54"
}

CFBE_SAMPLES = ["C1", "C2", "R1", "R2"]

REFERENCE_COLUMNS = [
    "PRIMARY_CFBE_ALL4",
    "SENSITIVITY_CFBE_C_PAIR",
    "SENSITIVITY_CFBE_R_PAIR"
]

checks = {}


def check(name, condition):
    passed = bool(condition)
    checks[name] = passed

    print(
        name,
        "PASS" if passed else "FAIL",
        flush=True
    )

    if not passed:
        raise RuntimeError("Failed check: " + name)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def read_gene_table(path):
    table = pd.read_csv(path, dtype={"GeneID": str})

    check(
        path.name + "_has_GeneID",
        "GeneID" in table.columns
    )

    check(
        path.name + "_unique_GeneID",
        table["GeneID"].is_unique
        and table["GeneID"].notna().all()
    )

    return table.set_index("GeneID")


def align_table(table, genes, name):
    check(
        name + "_exact_gene_universe",
        len(table) == len(genes)
        and set(table.index) == set(genes)
    )

    return table.reindex(genes)


def save_expression(table, path):
    table.reset_index().to_csv(
        path,
        index=False,
        float_format="%.15g"
    )


def main():
    print("=" * 72)
    print("CF-State Phase 4 — Audit 04j")
    print("Atlanta TPM normalisation and external contrasts")
    print("=" * 72)

    status_04g = (
        ATLANTA_DIR / "AUDIT_04G_STATUS.txt"
    )

    status_04i3 = (
        CFBE_DIR / "AUDIT_04I3_STATUS.txt"
    )

    sources = {
        "Atlanta_counts": ATLANTA,
        "CFBE_reference": REFERENCE,
        "CFBE_four_profiles": CFBE_FOUR,
        "PAO1_SAF": SAF,
        "Audit_04g_status": status_04g,
        "Audit_04i3_status": status_04i3,
        "Audit_04j_script": Path(__file__)
    }

    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(str(path))

    check(
        "Audit_04g_frozen",
        status_04g.read_text().strip()
        == "PRJNA576508_ATLANTA_13_PUBLISHED_PIPELINE_TECHNICALLY_REPLICATED"
    )

    check(
        "Audit_04i3_frozen",
        status_04i3.read_text().strip()
        == "PRJNA576508_CFBE_ALL4_LOG2TPM_EXTERNAL_REFERENCE_FROZEN"
    )

    print()
    print("========== ATLANTA COUNTS ==========")

    counts = read_gene_table(ATLANTA)
    samples = list(counts.columns)

    check(
        "exact_13_Atlanta_samples",
        len(samples) == 13
        and set(samples) == EXPECTED_SAMPLES
    )

    check(
        "Atlanta_5567_genes",
        len(counts) == 5567
    )

    counts = counts.apply(
        pd.to_numeric,
        errors="coerce"
    )

    count_values = counts.to_numpy(dtype=float)

    check(
        "all_counts_finite",
        np.isfinite(count_values).all()
    )

    check(
        "all_counts_nonnegative_integers",
        (count_values >= 0).all()
        and (count_values == np.floor(count_values)).all()
    )

    check(
        "all_samples_have_assigned_reads",
        (counts.sum(axis=0) > 0).all()
    )

    genes = counts.index

    print()
    print("========== FROZEN GENE LENGTHS ==========")

    saf = read_gene_table(SAF)

    saf = align_table(
        saf,
        genes,
        "SAF"
    )

    starts = pd.to_numeric(
        saf["Start"], errors="raise"
    )

    ends = pd.to_numeric(
        saf["End"], errors="raise"
    )

    lengths = (ends - starts + 1).astype(float)

    check(
        "all_SAF_lengths_valid",
        np.isfinite(lengths.to_numpy()).all()
        and (lengths > 0).all()
    )

    print()
    print("========== TPM NORMALISATION ==========")

    # Identical gene-length convention to Audit 04i3.
    # Divide counts by length in kilobases.

    rates = counts.div(
        lengths / 1000.0,
        axis=0
    )

    denominators = rates.sum(axis=0)

    check(
        "all_TPM_denominators_positive",
        np.isfinite(denominators.to_numpy()).all()
        and (denominators > 0).all()
    )

    tpm = rates.div(
        denominators,
        axis=1
    ) * 1e6

    log_tpm = np.log2(tpm + 1.0)

    check(
        "TPM_finite_nonnegative",
        np.isfinite(tpm.to_numpy()).all()
        and (tpm.to_numpy() >= 0).all()
    )

    check(
        "TPM_sums_one_million",
        np.allclose(
            tpm.sum(axis=0).to_numpy(),
            1e6,
            rtol=0,
            atol=0.01
        )
    )

    check(
        "log2TPM_finite",
        np.isfinite(log_tpm.to_numpy()).all()
    )

    print()
    print("========== CFBE REFERENCE INTEGRITY ==========")

    reference = read_gene_table(REFERENCE)

    reference = align_table(
        reference, genes, "CFBE_reference"
    )

    check(
        "exact_frozen_reference_columns",
        list(reference.columns) == REFERENCE_COLUMNS
    )

    four = read_gene_table(CFBE_FOUR)

    four = align_table(
        four, genes, "CFBE_four_profiles"
    )

    check(
        "exact_four_CFBE_samples",
        list(four.columns) == CFBE_SAMPLES
    )

    check(
        "all_CFBE_expression_finite",
        np.isfinite(reference.to_numpy()).all()
        and np.isfinite(four.to_numpy()).all()
    )

    # Independently recover the three frozen references.
    reconstructed = pd.DataFrame(
        {
            "PRIMARY_CFBE_ALL4":
                four[CFBE_SAMPLES].mean(axis=1),

            "SENSITIVITY_CFBE_C_PAIR":
                four[["C1", "C2"]].mean(axis=1),

            "SENSITIVITY_CFBE_R_PAIR":
                four[["R1", "R2"]].mean(axis=1)
        },
        index=genes
    )

    max_reference_difference = float(
        np.max(
            np.abs(
                reconstructed.to_numpy()
                - reference.to_numpy()
            )
        )
    )

    print(
        "Maximum reference reconstruction difference:",
        max_reference_difference
    )

    check(
        "reference_means_match_frozen_04i3",
        max_reference_difference <= 1e-9
    )

    print()
    print("========== EXTERNAL CONTRASTS ==========")

    contrasts = {}

    for name in REFERENCE_COLUMNS:
        contrast = log_tpm.sub(
            reference[name],
            axis=0
        )

        check(
            name + "_contrast_finite",
            np.isfinite(contrast.to_numpy()).all()
        )

        restored = contrast.add(
            reference[name],
            axis=0
        )

        check(
            name + "_contrast_reconstructs_expression",
            np.allclose(
                restored.to_numpy(),
                log_tpm.to_numpy(),
                rtol=0,
                atol=1e-10
            )
        )

        contrasts[name] = contrast

    print()
    print("========== DESCRIPTIVE EXPRESSION QC ==========")

    correlations = log_tpm.corr()

    qc_rows = []

    for sample in samples:
        sample_counts = counts[sample]
        sample_tpm = tpm[sample]

        other_correlations = (
            correlations.loc[sample]
            .drop(sample)
        )

        qc_rows.append(
            {
                "sample": sample,
                "assigned_gene_counts":
                    int(sample_counts.sum()),
                "detected_genes":
                    int((sample_counts > 0).sum()),
                "zero_fraction":
                    float((sample_counts == 0).mean()),
                "genes_TPM_ge_1":
                    int((sample_tpm >= 1).sum()),
                "top_10_gene_TPM_fraction":
                    float(
                        sample_tpm.nlargest(10).sum()
                        / 1e6
                    ),
                "mean_other_Atlanta_Pearson":
                    float(other_correlations.mean())
            }
        )

    qc = pd.DataFrame(qc_rows)

    check(
        "expression_QC_complete",
        len(qc) == 13
        and qc.drop(columns=["sample"])
        .notna().all().all()
    )

    print(
        qc.to_string(
            index=False,
            float_format=lambda x: "%.5f" % x
        )
    )

    print()
    print("========== FREEZING OUTPUTS ==========")

    # No new biological exclusion thresholds are introduced.
    # All 13 Atlanta samples are retained.
    # No iModulon projection or Audit-03h comparison occurs.

    stage = OUT / "staging"
    stage.mkdir()

    save_expression(
        tpm,
        stage / "FROZEN_Atlanta_13_TPM_expression.csv"
    )

    save_expression(
        log_tpm,
        stage / "FROZEN_Atlanta_13_log2TPM_expression.csv"
    )

    for name, contrast in contrasts.items():
        save_expression(
            contrast,
            stage / (
                "FROZEN_Atlanta_13_minus_"
                + name
                + "_log2TPM.csv"
            )
        )

    qc.to_csv(
        stage / "Atlanta_13_expression_QC.csv",
        index=False
    )

    correlations.to_csv(
        stage / "Atlanta_13_expression_correlations.csv"
    )

    pd.DataFrame(
        [
            {"check": name, "pass": passed}
            for name, passed in checks.items()
        ]
    ).to_csv(
        stage / "audit_pass_checks.csv",
        index=False
    )

    pd.DataFrame(
        [
            {
                "source": name,
                "path": str(path),
                "sha256": sha256_file(path)
            }
            for name, path in sources.items()
        ]
    ).to_csv(
        stage / "input_provenance_sha256.csv",
        index=False
    )

    notes = (
        "Audit 04j: Atlanta expression normalisation\n"
        "All 13 frozen Atlanta samples retained.\n"
        "Gene universe: frozen 5567-gene PAO1 SAF.\n"
        "Gene length: End - Start + 1.\n"
        "TPM: length-normalised counts scaled to 1e6.\n"
        "log2TPM shorthand means log2(TPM + 1).\n"
        "Primary reference: mean of four CFBE log2(TPM+1) profiles.\n"
        "Sensitivities: C-pair and R-pair reference means.\n"
        "Contrast: Atlanta log2(TPM+1) minus frozen reference.\n"
        "Descriptive QC metrics are not sample-exclusion gates.\n"
        "No cohort-3 centring against other Atlanta samples.\n"
        "No Audit-03h candidate used.\n"
        "No iModulon projection performed.\n"
        "External CFBE culture is not a patient-matched baseline.\n"
    )

    (stage / "AUDIT_04J_METHODS.txt").write_text(
        notes,
        encoding="utf-8"
    )

    # Move completed outputs into their final location.
    for path in stage.iterdir():
        path.rename(OUT / path.name)

    stage.rmdir()

    status = (
        "PRJNA576508_ATLANTA_13_LOG2TPM_"
        "EXTERNAL_CONTRASTS_FROZEN"
    )

    (OUT / "AUDIT_04J_STATUS.txt").write_text(
        status + "\n",
        encoding="utf-8"
    )

    print()
    print("=" * 72)
    print("STATUS:", status)
    print("=" * 72)
    print("All 13 samples retained.")
    print("All three external contrasts frozen.")
    print("Frozen Audit-03h candidate untouched.")


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 04j output already exists. "
            "Refusing to overwrite audit history."
        )

    OUT.mkdir(parents=True)

    try:
        main()
    except Exception:
        (OUT / "AUDIT_04J_STATUS.txt").write_text(
            "PRJNA576508_ATLANTA_04J_REVIEW_REQUIRED\n"
        )

        (OUT / "AUDIT_04J_FAILURE.txt").write_text(
            traceback.format_exc()
        )

        pd.DataFrame(
            [
                {"check": name, "pass": value}
                for name, value in checks.items()
            ]
        ).to_csv(
            OUT / "audit_pass_checks.csv",
            index=False
        )

        raise
