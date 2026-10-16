from pathlib import Path
from urllib.parse import urlsplit
import csv
import hashlib
import shutil
import subprocess

ROOT = Path.cwd()

MANIFEST = ROOT / (
    "results/phase5/"
    "audit_05e_Cornforth2018_download_manifest_lock/"
    "FROZEN_individual_FASTQ_download_manifest.csv"
)

STATUS = ROOT / (
    "results/phase5/"
    "audit_05e_Cornforth2018_download_manifest_lock/"
    "AUDIT_05E_STATUS.txt"
)

DEST = ROOT / "data/raw/cornforth2018_SRP135669"
DEST.mkdir(parents=True, exist_ok=True)


def md5sum(path):
    digest = hashlib.md5()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(8 * 1024 * 1024), b""
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    expected_status = (
        "CFSTATE_PHASE5_05E_"
        "CLINICAL_MAPPING_AND_DOWNLOAD_PLAN_LOCKED"
    )

    if STATUS.read_text().strip() != expected_status:
        raise RuntimeError("Audit 05e status is not locked")

    if shutil.which("curl") is None:
        raise RuntimeError("curl is not installed")

    with MANIFEST.open(newline="") as handle:
        rows = [
            row for row in csv.DictReader(handle)
            if row["analysis_role"] == "main_clinical_13"
        ]

    if len(rows) != 13:
        raise RuntimeError(
            "Expected exactly 13 primary FASTQ files"
        )

    jobs = []

    for row in rows:
        run = row["run_accession"]
        url = row["fastq_https"]
        size = int(row["expected_bytes"])
        md5 = row["expected_md5"].lower()

        parsed = urlsplit(url)
        filename = Path(parsed.path).name

        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or not filename.endswith(".fastq.gz")
            or size <= 0
            or len(md5) != 32
        ):
            raise RuntimeError(
                "Invalid FASTQ metadata for " + run
            )

        final = DEST / (run + "__" + filename)
        partial = Path(str(final) + ".part")

        if final.exists() and final.stat().st_size != size:
            raise RuntimeError(
                "Existing file has incorrect size: " + str(final)
            )

        if partial.exists() and partial.stat().st_size > size:
            raise RuntimeError(
                "Partial file exceeds expected size: " + str(partial)
            )

        jobs.append((run, url, size, md5, final, partial))

    if len({job[4] for job in jobs}) != len(jobs):
        raise RuntimeError("Duplicate destination filename")

    remaining = sum(
        max(
            0,
            size - (
                partial.stat().st_size
                if partial.exists() else 0
            )
        )
        for run, url, size, md5, final, partial in jobs
        if not final.exists()
    )

    free = shutil.disk_usage(DEST).free
    reserve = 8 * 1024**3

    print("=" * 70)
    print("Cornforth 2018 — Primary clinical FASTQ download")
    print("=" * 70)
    print("Files:", len(jobs))
    print("Remaining download GB:", round(remaining / 1e9, 2))
    print("Available disk GB:", round(free / 1e9, 2))
    print("Destination:", DEST)
    print(flush=True)

    if free < remaining + reserve:
        raise RuntimeError(
            "Insufficient disk space for downloads plus 8 GiB reserve"
        )

    verified = []

    for index, (run, url, size, md5, final, partial) in enumerate(
        jobs, start=1
    ):
        print()
        print(f"[{index}/{len(jobs)}] {run}", flush=True)

        if not final.exists():
            print("Downloading/resuming:", final.name, flush=True)

            command = [
                "curl",
                "--fail",
                "--location",
                "--retry", "8",
                "--retry-delay", "10",
                "--connect-timeout", "30",
                "--continue-at", "-",
                "--silent",
                "--show-error",
                "--output", str(partial),
                url
            ]

            subprocess.run(command, check=True)

            if partial.stat().st_size != size:
                raise RuntimeError(
                    "Downloaded size mismatch: " + run
                )

            print("Checking MD5...", flush=True)

            actual = md5sum(partial)

            if actual != md5:
                raise RuntimeError(
                    "MD5 mismatch for " + run
                    + ". Do not promote this partial file."
                )

            partial.rename(final)

        else:
            print("Existing file: checking MD5...", flush=True)

            if md5sum(final) != md5:
                raise RuntimeError(
                    "Existing file MD5 mismatch: " + run
                )

        verified.append({
            "run_accession": run,
            "path": str(final),
            "verified_bytes": size,
            "verified_md5": md5,
            "verification_status": "PASS"
        })

        print(run, "VERIFIED", flush=True)

    with (DEST / "download_verification.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(verified[0])
        )
        writer.writeheader()
        writer.writerows(verified)

    (DEST / "DOWNLOAD_COMPLETE_MD5_VERIFIED.txt").write_text(
        "13_PRIMARY_CLINICAL_FASTQS_MD5_VERIFIED\n"
    )

    print()
    print("=" * 70)
    print("ALL 13 PRIMARY CLINICAL FASTQS MD5 VERIFIED")
    print("=" * 70)


if __name__ == "__main__":
    main()
