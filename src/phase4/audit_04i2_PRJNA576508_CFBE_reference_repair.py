from pathlib import Path
import hashlib
import importlib.util
import inspect
import re
import shutil
import textwrap

import numpy as np
import pandas as pd


ROOT = Path.cwd()

SOURCE = ROOT / (
    "src/phase4/"
    "audit_04i_PRJNA576508_CFBE_reference_processing.py"
)

ORIGINAL = ROOT / (
    "results/phase4/"
    "audit_04i_PRJNA576508_CFBE_reference_processing"
)

OUT = ROOT / (
    "results/phase4/"
    "audit_04i2_PRJNA576508_CFBE_reference_repair"
)

FROZEN_RUNS = ROOT / (
    "results/phase4/"
    "audit_04h3_PRJNA576508_reference_plan_repair/"
    "FROZEN_four_CFBE_reference_runs.csv"
)

MANIFEST = ROOT / (
    "results/phase4/"
    "audit_04a_PRJNA576508_third_cohort_preflight/"
    "PRJNA576508_full_ENA_manifest.csv"
)

SAF = ROOT / (
    "references/phase4/PRJNA576508_cornforth/"
    "Cornforth_PAO1_no_decimal_locus_tags.saf"
)

COUNT_DIR = OUT / "per_sample_counts"
SUMMARY_DIR = OUT / "per_sample_summaries"
LOG_DIR = OUT / "logs"

LABELS = ["C1", "C2", "R1", "R2"]
MIN_GENES = 4000
MIN_CORRELATION = 0.90


def md5(path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def load_original():
    spec = importlib.util.spec_from_file_location(
        "audit04i_original", SOURCE
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_repaired_function(module):
    """
    Copy the original process_reference implementation
    into memory, changing only the incorrect adapter QC.

    Original source and historical outputs are untouched.
    """
    source = textwrap.dedent(
        inspect.getsource(module.process_reference)
    )

    pattern = (
        r'\s*"adapter_fraction_ge_0_80":'
        r'\s*adapter_fraction'
        r'\s*>= MIN_ADAPTER_FRACTION,'
    )

    repaired, count = re.subn(
        pattern, "", source
    )

    if count != 1:
        raise RuntimeError(
            "Could not uniquely locate original adapter gate. "
            "No processing has been started."
        )

    namespace = vars(module)
    exec(
        compile(repaired, str(SOURCE), "exec"),
        namespace
    )

    return hashlib.sha256(
        repaired.encode("utf-8")
    ).hexdigest()


def corrected_checks(summary):
    checks = {}

    for column in summary.index:
        if column.startswith("check_"):
            if column == "check_adapter_fraction_ge_0_80":
                continue

            value = summary[column]
            checks[column] = (
                str(value).strip().lower() == "true"
            )

    checks["check_detected_genes_ge_4000"] = (
        int(summary["detected_genes"]) >= MIN_GENES
    )

    if not checks:
        raise RuntimeError("Technical QC fields missing.")

    return checks


def validate_counts(path, label, summary, genes):
    table = pd.read_csv(path)

    if list(table.columns) != ["GeneID", "count"]:
        raise RuntimeError(
            label + ": unexpected count schema"
        )

    if table["GeneID"].duplicated().any():
        raise RuntimeError(
            label + ": duplicate count GeneIDs"
        )

    if len(table) != 5567:
        raise RuntimeError(
            label + ": expected 5567 genes"
        )

    if set(table["GeneID"]) != set(genes):
        raise RuntimeError(
            label + ": SAF/count gene mismatch"
        )

    counts = pd.to_numeric(
        table["count"], errors="raise"
    )

    if counts.isna().any():
        raise RuntimeError(label + ": missing counts")

    if not np.isfinite(counts).all():
        raise RuntimeError(label + ": nonfinite counts")

    if (counts < 0).any():
        raise RuntimeError(label + ": negative counts")

    if not np.all(counts == np.floor(counts)):
        raise RuntimeError(label + ": noninteger counts")

    assigned = int(counts.sum())
    detected = int((counts > 0).sum())

    if assigned != int(summary["Rsubread_assigned_reads"]):
        raise RuntimeError(
            label + ": assigned-read disagreement"
        )

    if detected != int(summary["detected_genes"]):
        raise RuntimeError(
            label + ": detected-gene disagreement"
        )

    return table.set_index("GeneID")["count"]


def main():
    print("=" * 78)
    print("CF-State Phase 4 — Audit 04i2")
    print("Independent CFBE reference-processing repair")
    print("=" * 78)

    required = [SOURCE, FROZEN_RUNS, MANIFEST, SAF]

    for path in required:
        if not path.is_file():
            raise FileNotFoundError(str(path))

    for path in [OUT, COUNT_DIR, SUMMARY_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    module = load_original()

    # Patch only the incorrectly imposed adapter gate.
    repaired_hash = prepare_repaired_function(module)

    # Redirect all future outputs to the new audit.
    module.OUT = OUT
    module.COUNT_DIR = COUNT_DIR
    module.LOG_DIR = LOG_DIR
    module.SUMMARY_DIR = SUMMARY_DIR

    # These paths retain the frozen original preprocessing.
    module.RAW_DIR.mkdir(parents=True, exist_ok=True)
    module.WORK_DIR.mkdir(parents=True, exist_ok=True)

    frozen_pipeline = module.load_frozen_pipeline()
    frozen_pipeline.OUT = OUT
    frozen_pipeline.PAO1_SAF = SAF

    frozen = pd.read_csv(FROZEN_RUNS)
    manifest = pd.read_csv(MANIFEST)

    if len(frozen) != 4:
        raise RuntimeError("Expected four frozen runs")

    if set(frozen["pair_sample"]) != set(LABELS):
        raise RuntimeError("Unexpected frozen run labels")

    if frozen["run_accession"].duplicated().any():
        raise RuntimeError("Duplicate frozen accessions")

    metadata = frozen[
        ["run_accession", "pair_sample", "pair", "replicate"]
    ].merge(
        manifest,
        on="run_accession",
        how="left",
        validate="one_to_one"
    )

    if len(metadata) != 4:
        raise RuntimeError("Incomplete reference manifest")

    metadata = metadata.set_index("pair_sample").loc[LABELS]

    if metadata[
        ["fastq_md5", "fastq_ftp", "read_count"]
    ].isna().any().any():
        raise RuntimeError("Missing ENA metadata")

    saf = pd.read_csv(SAF, sep="\t")

    required_saf = {"GeneID", "Start", "End"}
    if not required_saf.issubset(saf.columns):
        raise RuntimeError("SAF columns missing")

    if saf["GeneID"].duplicated().any():
        raise RuntimeError("Duplicate SAF GeneIDs")

    lengths = pd.Series(
        (
            saf["End"].astype(int)
            - saf["Start"].astype(int)
            + 1
        ).to_numpy(dtype=float),
        index=saf["GeneID"].astype(str),
        name="length_bp"
    )

    if len(lengths) != 5567:
        raise RuntimeError("Unexpected SAF gene universe")

    if (lengths <= 0).any():
        raise RuntimeError("Nonpositive SAF lengths")

    source_hashes = {
        "original_04i_sha256": hashlib.sha256(
            SOURCE.read_bytes()
        ).hexdigest(),
        "repaired_process_function_sha256": repaired_hash,
        "frozen_SAF_md5": md5(SAF),
        "frozen_manifest_md5": md5(FROZEN_RUNS),
    }

    pd.DataFrame(
        [{"item": k, "hash": v}
         for k, v in source_hashes.items()]
    ).to_csv(
        OUT / "audit_source_provenance.csv",
        index=False
    )

    summaries = []
    count_columns = []

    for label in LABELS:
        print()
        print("-" * 60)
        print("REFERENCE:", label)
        print("-" * 60)

        row = metadata.loc[label]

        destination_count = (
            COUNT_DIR / (label + "_gene_counts.csv")
        )
        destination_summary = (
            SUMMARY_DIR / (label + "_technical_summary.csv")
        )

        if label == "C1":
            original_count = (
                ORIGINAL / "per_sample_counts"
                / "C1_gene_counts.csv"
            )

            original_summary = (
                ORIGINAL / "per_sample_summaries"
                / "C1_technical_summary.csv"
            )

            original_stat = (
                ORIGINAL / "C1_Rsubread_stat.csv"
            )

            if not all(
                p.is_file()
                for p in [
                    original_count,
                    original_summary,
                    original_stat
                ]
            ):
                raise RuntimeError(
                    "Validated original C1 outputs missing"
                )

            raw = (
                module.RAW_DIR
                / (str(row["run_accession"]) + ".fastq.gz")
            )

            expected_md5 = (
                str(row["fastq_md5"])
                .split(";")[0].strip().lower()
            )

            if not raw.is_file():
                raise RuntimeError("C1 raw FASTQ missing")

            if md5(raw) != expected_md5:
                raise RuntimeError("C1 raw MD5 failed")

            shutil.copy2(
                original_count, destination_count
            )

            shutil.copy2(
                original_stat,
                OUT / "C1_Rsubread_stat.csv"
            )

            for suffix in [
                "_cutadapt.log",
                "_decoy_bowtie2.log",
                "_PAO1_bowtie2.log"
            ]:
                original_log = (
                    ORIGINAL / "logs" / ("C1" + suffix)
                )
                if not original_log.is_file():
                    raise RuntimeError(
                        "Missing original C1 log: " + suffix
                    )
                shutil.copy2(
                    original_log,
                    LOG_DIR / ("C1" + suffix)
                )

            summary = pd.read_csv(
                original_summary
            ).iloc[0].copy()

            checks = corrected_checks(summary)

            # Explicit verification of the legacy C1 failure.
            if (
                str(
                    summary["check_adapter_fraction_ge_0_80"]
                ).lower() != "false"
            ):
                raise RuntimeError(
                    "Original C1 adapter-gate failure not found"
                )

            if not all(checks.values()):
                raise RuntimeError(
                    "C1 has another technical QC failure: "
                    + str([k for k, v in checks.items() if not v])
                )

            print("Reusing original C1 counts and logs")

        else:
            summary_dict = module.process_reference(
                frozen_pipeline, row
            )

            summary = pd.Series(summary_dict)
            checks = corrected_checks(summary)

        passed = all(checks.values())

        summary["technical_pass"] = bool(passed)
        summary["qc_rule"] = (
            "04i2_adapter_descriptive_only"
        )

        summary["adapter_fraction_QC_role"] = (
            "descriptive_not_gate"
        )

        summary = summary.drop(
            labels=["check_adapter_fraction_ge_0_80"],
            errors="ignore"
        )

        pd.DataFrame([summary]).to_csv(
            destination_summary, index=False
        )

        if not passed:
            raise RuntimeError(
                label + " technical QC failed: "
                + str([k for k, v in checks.items() if not v])
            )

        counts = validate_counts(
            destination_count,
            label,
            summary,
            lengths.index
        ).rename(label)

        summaries.append(summary)
        count_columns.append(counts)

        print(
            label,
            "PASS | assigned:",
            int(summary["Rsubread_assigned_reads"]),
            "| detected:",
            int(summary["detected_genes"]),
            "| adapter fraction:",
            round(float(summary["adapter_fraction"]), 5)
        )

    matrix = pd.concat(count_columns, axis=1)
    matrix = matrix.reindex(lengths.index)

    if matrix.isna().any().any():
        raise RuntimeError("Missing merged count values")

    matrix.index.name = "GeneID"

    # Expression normalisation:
    # TPM_g = (counts_g / length_kb_g)
    #         / sum_h(counts_h / length_kb_h) * 1e6

    rate = matrix.div(lengths / 1000.0, axis=0)
    denominators = rate.sum(axis=0)

    if (
        (denominators <= 0).any()
        or not np.isfinite(denominators).all()
    ):
        raise RuntimeError("Invalid TPM denominators")

    tpm = rate.div(denominators, axis=1) * 1e6

    log_tpm = np.log2(tpm + 1.0)

    c_corr = float(
        log_tpm["C1"].corr(log_tpm["C2"])
    )

    r_corr = float(
        log_tpm["R1"].corr(log_tpm["R2"])
    )

    reference = pd.DataFrame(
        {
            "PRIMARY_CFBE_ALL4": log_tpm[LABELS].mean(axis=1),
            "SENSITIVITY_CFBE_C_PAIR":
                log_tpm[["C1", "C2"]].mean(axis=1),
            "SENSITIVITY_CFBE_R_PAIR":
                log_tpm[["R1", "R2"]].mean(axis=1),
        }
    )

    checks = {
        "exactly_four_samples":
            len(summaries) == 4,
        "exact_frozen_labels":
            list(matrix.columns) == LABELS,
        "exact_5567_genes":
            len(matrix) == 5567,
        "all_technical_QC_pass":
            all(bool(s["technical_pass"]) for s in summaries),
        "TPM_finite_nonnegative":
            bool(
                np.isfinite(tpm.to_numpy()).all()
                and (tpm.to_numpy() >= 0).all()
            ),
        "TPM_sums_one_million":
            bool(
                np.allclose(
                    tpm.sum(axis=0).to_numpy(),
                    1e6,
                    atol=0.01,
                    rtol=0
                )
            ),
        "C_pair_Pearson_ge_0_90":
            bool(
                np.isfinite(c_corr)
                and c_corr >= MIN_CORRELATION
            ),
        "R_pair_Pearson_ge_0_90":
            bool(
                np.isfinite(r_corr)
                and r_corr >= MIN_CORRELATION
            ),
        "exact_three_reference_profiles":
            list(reference.columns) == [
                "PRIMARY_CFBE_ALL4",
                "SENSITIVITY_CFBE_C_PAIR",
                "SENSITIVITY_CFBE_R_PAIR"
            ],
    }

    print()
    print("=" * 78)
    print("AUDIT 04i2 QC")
    print("=" * 78)
    print("C-pair Pearson:", round(c_corr, 6))
    print("R-pair Pearson:", round(r_corr, 6))

    for name, passed in checks.items():
        print(
            name, "PASS" if passed else "FAIL"
        )

    pd.DataFrame(
        [{"check": k, "pass": v}
         for k, v in checks.items()]
    ).to_csv(
        OUT / "audit_pass_checks.csv",
        index=False
    )

    pd.DataFrame(summaries).to_csv(
        OUT / "CFBE_four_sample_technical_summary.csv",
        index=False
    )

    pd.DataFrame(
        [
            {"pair": "C", "Pearson_log2TPM": c_corr},
            {"pair": "R", "Pearson_log2TPM": r_corr}
        ]
    ).to_csv(
        OUT / "CFBE_pair_correlations.csv",
        index=False
    )

    status_path = OUT / "AUDIT_04I2_STATUS.txt"

    if not all(checks.values()):
        status = (
            "PRJNA576508_CFBE_REFERENCE_REPAIR_"
            "REVIEW_REQUIRED"
        )
        status_path.write_text(status + "\n")

        raise RuntimeError(
            "Audit 04i2 failed. No reference frozen."
        )

    matrix.reset_index().to_csv(
        OUT / "FROZEN_CFBE_four_raw_gene_counts.csv",
        index=False
    )

    tpm.reset_index().to_csv(
        OUT / "FROZEN_CFBE_four_TPM_profiles.csv",
        index=False
    )

    log_tpm.reset_index().to_csv(
        OUT / "FROZEN_CFBE_four_log2TPM_profiles.csv",
        index=False
    )

    reference.reset_index().to_csv(
        OUT / "FROZEN_CFBE_log2TPM_reference_profiles.csv",
        index=False
    )

    status = (
        "PRJNA576508_CFBE_ALL4_LOG2TPM_"
        "EXTERNAL_REFERENCE_FROZEN"
    )

    status_path.write_text(status + "\n")

    print()
    print("STATUS:", status)
    print()
    print(
        "No Audit-03h candidate or CF-airway residual "
        "was accessed."
    )


if __name__ == "__main__":
    main()
