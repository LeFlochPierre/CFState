from pathlib import Path
from urllib.parse import urlsplit
import csv
import gzip
import hashlib
import shutil
import statistics
import traceback

import pandas as pd

ROOT = Path.cwd()
P5 = ROOT / "results/phase5"

FROZEN = P5 / "audit_05e_Cornforth2018_download_manifest_lock"
RAW = ROOT / "data/raw/cornforth2018_SRP135669"
OUT = P5 / "audit_05f_download_QC_and_pilot_readiness"

STATUS = (
    "CFSTATE_PHASE5_05F_DOWNLOAD_AND_PILOT_READINESS_VERIFIED"
)

PILOT = {
    "Human_Sputum_G": "SRR6833349",
    "Lubbock_CW_15": "SRR6833340"
}


def check(name, condition):
    passed = bool(condition)
    print(name, "PASS" if passed else "FAIL", flush=True)

    if not passed:
        raise RuntimeError("Failed: " + name)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1048576), b""
        ):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_fastq(path, limit=200):
    lengths = []
    gc_count = 0
    base_count = 0

    with gzip.open(path, "rt") as handle:
        for _ in range(limit):
            identifier = handle.readline()

            if not identifier:
                break

            sequence = handle.readline().strip()
            plus = handle.readline()
            quality = handle.readline().strip()

            if not identifier.startswith("@"):
                raise RuntimeError(
                    "Invalid FASTQ identifier: " + str(path)
                )

            if not plus.startswith("+"):
                raise RuntimeError(
                    "Invalid FASTQ separator: " + str(path)
                )

            if not sequence or len(sequence) != len(quality):
                raise RuntimeError(
                    "Sequence/quality length mismatch: " + str(path)
                )

            lengths.append(len(sequence))
            gc_count += (
                sequence.upper().count("G")
                + sequence.upper().count("C")
            )
            base_count += len(sequence)

    if not lengths:
        raise RuntimeError("Empty FASTQ: " + str(path))

    return {
        "reads_inspected": len(lengths),
        "read_length_min": min(lengths),
        "read_length_median": statistics.median(lengths),
        "read_length_max": max(lengths),
        "preview_GC_fraction": gc_count / base_count
    }


def main():
    print("=" * 74)
    print("CF-State Phase 5 — Audit 05f")
    print("Download verification and two-sample pilot readiness")
    print("=" * 74)

    inputs = {
        "05e_status": FROZEN / "AUDIT_05E_STATUS.txt",
        "FASTQ_manifest":
            FROZEN / "FROZEN_individual_FASTQ_download_manifest.csv",
        "clinical_manifest":
            FROZEN / "FROZEN_13_main_clinical_sample_manifest.csv",
        "verification":
            RAW / "download_verification.csv",
        "completion_marker":
            RAW / "DOWNLOAD_COMPLETE_MD5_VERIFIED.txt",
        "script": Path(__file__)
    }

    for name, path in inputs.items():
        check("exists_" + name, path.is_file())

    check(
        "05e_frozen",
        inputs["05e_status"].read_text().strip()
        == "CFSTATE_PHASE5_05E_CLINICAL_MAPPING_AND_DOWNLOAD_PLAN_LOCKED"
    )

    check(
        "download_completion_recorded",
        inputs["completion_marker"].read_text().strip()
        == "13_PRIMARY_CLINICAL_FASTQS_MD5_VERIFIED"
    )

    manifest = pd.read_csv(
        inputs["FASTQ_manifest"], dtype=str
    ).fillna("")

    manifest = manifest.loc[
        manifest["analysis_role"] == "main_clinical_13"
    ].copy()

    clinical = pd.read_csv(
        inputs["clinical_manifest"], dtype=str
    ).fillna("")

    verification = pd.read_csv(
        inputs["verification"], dtype=str
    ).fillna("")

    check(
        "exact_13_download_records",
        len(manifest) == 13
        and len(verification) == 13
        and len(clinical) == 13
    )

    check(
        "unique_run_identifiers",
        manifest["run_accession"].is_unique
        and verification["run_accession"].is_unique
        and clinical["run_accession"].is_unique
    )

    check(
        "matching_13_runs",
        set(manifest["run_accession"])
        == set(verification["run_accession"])
        == set(clinical["run_accession"])
    )

    check(
        "all_reported_MD5_pass",
        verification["verification_status"].eq("PASS").all()
    )

    verification = verification.set_index("run_accession")
    clinical = clinical.set_index("run_accession")

    qc = []

    print()
    print("========== VERIFY FILES AND FASTQ PREVIEWS ==========")

    for record in manifest.to_dict("records"):
        run = record["run_accession"]

        filename = Path(
            urlsplit(record["fastq_https"]).path
        ).name

        path = RAW / (run + "__" + filename)

        check(run + "_file_exists", path.is_file())

        size = path.stat().st_size
        expected_size = int(record["expected_bytes"])

        check(
            run + "_file_size",
            size == expected_size
        )

        logged = verification.loc[run]

        check(
            run + "_recorded_MD5_matches_archive",
            logged["verified_md5"].lower()
            == record["expected_md5"].lower()
        )

        check(
            run + "_recorded_size_matches_archive",
            int(logged["verified_bytes"]) == expected_size
        )

        check(
            run + "_recorded_path_matches",
            Path(logged["path"]).resolve()
            == path.resolve()
        )

        preview = inspect_fastq(path)

        qc.append({
            "published_sample":
                clinical.loc[run, "published_sample"],
            "group":
                clinical.loc[run, "group"],
            "run_accession": run,
            "FASTQ_path": str(path),
            "FASTQ_bytes": size,
            "archive_MD5_verified_by_downloader": True,
            **preview
        })

    qc = pd.DataFrame(qc)

    check(
        "all_13_FASTQ_previews_valid",
        len(qc) == 13
        and qc["reads_inspected"].gt(0).all()
    )

    print()
    print(
        qc[
            [
                "published_sample",
                "group",
                "read_length_min",
                "read_length_median",
                "read_length_max",
                "preview_GC_fraction"
            ]
        ].round(3).to_string(index=False)
    )

    print()
    print("========== PROCESSING TOOLS ==========")

    required_tools = [
        "cutadapt",
        "bowtie2",
        "bowtie2-build",
        "samtools",
        "Rscript"
    ]

    optional_tools = [
        "fastqc",
        "fastp",
        "seqtk",
        "featureCounts",
        "pigz"
    ]

    tool_records = []

    for name in required_tools + optional_tools:
        location = shutil.which(name)

        tool_records.append({
            "tool": name,
            "required_for_candidate_pipeline":
                name in required_tools,
            "available": location is not None,
            "executable_path": location or ""
        })

        print(
            name,
            "FOUND" if location else "NOT FOUND",
            location or ""
        )

    tools = pd.DataFrame(tool_records)

    print()
    print("========== EXISTING REFERENCE RESOURCES ==========")

    reference_dir = (
        ROOT / "references/phase4/PRJNA576508_cornforth"
    )

    saf = (
        reference_dir /
        "Cornforth_PAO1_no_decimal_locus_tags.saf"
    )

    print("PAO1 SAF exists:", saf.is_file())

    reference_files = []

    if reference_dir.is_dir():
        reference_files = sorted(
            path for path in reference_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {
                ".saf", ".fa", ".fna", ".fasta",
                ".gb", ".gbk", ".bt2", ".bt2l",
                ".fai"
            }
        )

    for path in reference_files[:30]:
        print(path.relative_to(ROOT))

    print()
    print("========== PREVIOUS PIPELINE IMPLEMENTATIONS ==========")

    previous_scripts = sorted(
        path for path in (
            ROOT / "src/phase4"
        ).glob("*04g*")
        if path.is_file()
    )

    for path in previous_scripts:
        print(path.relative_to(ROOT))

    print()
    print("========== DISK SPACE ==========")

    disk = shutil.disk_usage(RAW)

    print(
        "Available GB:",
        round(disk.free / 1e9, 2)
    )

    print(
        "Downloaded clinical FASTQ GB:",
        round(qc["FASTQ_bytes"].sum() / 1e9, 2)
    )

    print()
    print("========== TECHNICAL PILOT ==========")

    pilot_rows = []

    for sample, run in PILOT.items():
        selected = qc.loc[
            (qc["published_sample"] == sample)
            & (qc["run_accession"] == run)
        ]

        check(
            "pilot_" + sample + "_unique",
            len(selected) == 1
        )

        pilot_rows.append(selected.iloc[0].to_dict())

    pilot = pd.DataFrame(pilot_rows)

    print(
        pilot[
            [
                "published_sample",
                "group",
                "run_accession",
                "FASTQ_path",
                "read_length_median"
            ]
        ].to_string(index=False)
    )

    print()
    print("========== WRITE AUDIT 05f ==========")

    qc.to_csv(
        OUT / "FROZEN_13_download_and_FASTQ_preview_QC.csv",
        index=False
    )

    tools.to_csv(
        OUT / "FROZEN_processing_tool_inventory.csv",
        index=False
    )

    pilot.to_csv(
        OUT / "FROZEN_two_sample_technical_pilot.csv",
        index=False
    )

    pd.DataFrame([
        {
            "input": name,
            "path": str(path),
            "sha256": sha256(path)
        }
        for name, path in inputs.items()
    ]).to_csv(
        OUT / "input_provenance_sha256.csv",
        index=False
    )

    notes = (
        "Audit 05f: download and technical pilot readiness.\n"
        "Uses frozen 05e clinical sample design.\n"
        "Verifies size, downloader's MD5 evidence and FASTQ previews.\n"
        "Does not rehash the entire 31.8 GB archive.\n"
        "Does not perform full gzip integrity testing.\n"
        "Pilot pair fixed using published bacterial coverage.\n"
        "No expression outcomes or candidate values accessed.\n"
        "Cornforth 2018 originally used decoy filtering and a "
        "multi-strain pangenome.\n"
        "PAO1-only projection feasibility requires independent QC.\n"
        "All 13 primary specimens remain in the processing plan.\n"
    )

    (
        OUT / "AUDIT_05F_METHODS.txt"
    ).write_text(notes)

    (
        OUT / "AUDIT_05F_STATUS.txt"
    ).write_text(STATUS + "\n")

    print()
    print("=" * 74)
    print("STATUS:", STATUS)
    print("=" * 74)


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 05f already exists; refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()
    except Exception:
        (
            OUT / "AUDIT_05F_STATUS.txt"
        ).write_text(
            "CFSTATE_PHASE5_05F_REVIEW_REQUIRED\n"
        )
        (
            OUT / "AUDIT_05F_FAILURE.txt"
        ).write_text(traceback.format_exc())
        raise
