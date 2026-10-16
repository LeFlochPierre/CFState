from pathlib import Path
import hashlib
import shutil
import subprocess
import tempfile

import pandas as pd


ROOT = Path.cwd()

MANIFEST = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "PRJNA576508_Atlanta_sputum_manifest.csv"
)

RAW_DIR = (
    ROOT
    / "data/raw/phase4/PRJNA576508/Atlanta_13"
)


def md5_file(path):
    digest = hashlib.md5()

    with path.open("rb") as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def first_value(value):
    return str(
        value
    ).split(";")[0].strip()


def https_url(value):
    text = first_value(
        value
    )

    if text.startswith(
        "ftp://"
    ):
        text = (
            "https://"
            + text[
                len("ftp://"):
            ]
        )

    elif text.startswith(
        "ftp."
    ):
        text = (
            "https://"
            + text
        )

    return text


def ftp_url(value):
    text = first_value(
        value
    )

    if text.startswith(
        "ftp://"
    ):
        return text

    if text.startswith(
        "https://"
    ):
        text = text[
            len("https://"):
        ]

    return (
        "ftp://"
        + text
    )


def run_quiet(command):
    print()
    print(
        " ",
        " ".join(
            str(x)
            for x in command
        )
    )

    result = subprocess.run(
        command
    )

    return (
        result.returncode
        == 0
    )


def valid_file(
    path,
    expected_md5,
):
    if not path.exists():
        return False

    observed = md5_file(
        path
    )

    print(
        "  MD5 observed:",
        observed,
    )
    print(
        "  MD5 expected:",
        expected_md5,
    )

    return (
        observed
        == expected_md5
    )


def try_curl(
    url,
    destination,
):
    curl = shutil.which(
        "curl"
    )

    if curl is None:
        return False

    temporary = Path(
        str(destination)
        + ".part"
    )

    command = [
        curl,
        "-L",
        "--fail",
        "--retry",
        "20",
        "--retry-delay",
        "15",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--continue-at",
        "-",
        "--output",
        str(temporary),
        url,
    ]

    return run_quiet(
        command
    )


def try_ena_data_get(
    run,
    destination,
):
    ena = shutil.which(
        "enaDataGet"
    )

    if ena is None:
        print(
            "  enaDataGet unavailable"
        )
        return False

    with tempfile.TemporaryDirectory(
        prefix="ena_"
    ) as temp:
        temp_path = Path(
            temp
        )

        command = [
            ena,
            "-f",
            "fastq",
            "-d",
            str(
                temp_path
            ),
            run,
        ]

        if not run_quiet(
            command
        ):
            return False

        candidates = list(
            temp_path.rglob(
                run
                + ".fastq.gz"
            )
        )

        if len(
            candidates
        ) != 1:
            print(
                "  enaDataGet candidate files:",
                [
                    str(x)
                    for x in candidates
                ],
            )

            return False

        shutil.copy2(
            candidates[
                0
            ],
            destination,
        )

    return True


def fetch_one(
    run,
    url_value,
    expected_md5,
):
    destination = (
        RAW_DIR
        / (
            run
            + ".fastq.gz"
        )
    )

    print()
    print("=" * 78)
    print(
        "FETCH",
        run,
    )
    print("=" * 78)

    if destination.exists():
        print(
            "Existing file detected"
        )

        if valid_file(
            destination,
            expected_md5,
        ):
            print(
                "STATUS: EXISTING_MD5_PASS"
            )
            return

        print(
            "Existing file has wrong MD5; removing."
        )

        destination.unlink()

    partial = Path(
        str(destination)
        + ".part"
    )

    # ------------------------------------------------------
    # Route 1: ENA HTTPS with aggressive transient retry
    # ------------------------------------------------------

    url_https = https_url(
        url_value
    )

    print()
    print(
        "Attempt 1: ENA HTTPS"
    )
    print(
        " ",
        url_https,
    )

    if try_curl(
        url_https,
        destination,
    ):
        if partial.exists():
            partial.rename(
                destination
            )

        if valid_file(
            destination,
            expected_md5,
        ):
            print(
                "STATUS: HTTPS_MD5_PASS"
            )
            return

        destination.unlink(
            missing_ok=True
        )

    partial.unlink(
        missing_ok=True
    )

    # ------------------------------------------------------
    # Route 2: official enaBrowserTools accession downloader
    # ------------------------------------------------------

    print()
    print(
        "Attempt 2: enaDataGet"
    )

    if try_ena_data_get(
        run,
        destination,
    ):
        if valid_file(
            destination,
            expected_md5,
        ):
            print(
                "STATUS: ENADATAGET_MD5_PASS"
            )
            return

        destination.unlink(
            missing_ok=True
        )

    # ------------------------------------------------------
    # Route 3: ENA FTP
    # ------------------------------------------------------

    url_ftp = ftp_url(
        url_value
    )

    print()
    print(
        "Attempt 3: ENA FTP"
    )
    print(
        " ",
        url_ftp,
    )

    if try_curl(
        url_ftp,
        destination,
    ):
        if partial.exists():
            partial.rename(
                destination
            )

        if valid_file(
            destination,
            expected_md5,
        ):
            print(
                "STATUS: FTP_MD5_PASS"
            )
            return

        destination.unlink(
            missing_ok=True
        )

    partial.unlink(
        missing_ok=True
    )

    raise RuntimeError(
        "All download routes failed for "
        + run
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4")
    print("Robust retrieval of frozen PRJNA576508 Atlanta 13 FASTQs")
    print("=" * 78)

    if not MANIFEST.exists():
        raise FileNotFoundError(
            str(
                MANIFEST
            )
        )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST
    )

    if len(
        manifest
    ) != 13:
        raise RuntimeError(
            "Expected exactly 13 frozen Atlanta runs."
        )

    results = []

    for _, row in (
        manifest
        .sort_values(
            "short_sample_label"
        )
        .iterrows()
    ):
        run = str(
            row[
                "run_accession"
            ]
        )

        label = str(
            row[
                "short_sample_label"
            ]
        )

        expected_md5 = first_value(
            row[
                "fastq_md5"
            ]
        ).lower()

        url = first_value(
            row[
                "fastq_ftp"
            ]
        )

        fetch_one(
            run,
            url,
            expected_md5,
        )

        path = (
            RAW_DIR
            / (
                run
                + ".fastq.gz"
            )
        )

        observed_md5 = md5_file(
            path
        )

        results.append(
            {
                "sample":
                    label,

                "run":
                    run,

                "path":
                    str(
                        path
                    ),

                "expected_md5":
                    expected_md5,

                "observed_md5":
                    observed_md5,

                "md5_pass":
                    observed_md5
                    == expected_md5,

                "bytes":
                    path.stat().st_size,
            }
        )

    result = pd.DataFrame(
        results
    )

    print()
    print("=" * 78)
    print("Atlanta FASTQ retrieval summary")
    print("=" * 78)

    print(
        result[
            [
                "sample",
                "run",
                "bytes",
                "md5_pass",
            ]
        ]
        .to_string(
            index=False
        )
    )

    all_pass = bool(
        result[
            "md5_pass"
        ].all()
    )

    print()
    print(
        "MD5 PASS:",
        int(
            result[
                "md5_pass"
            ].sum()
        ),
        "/ 13",
    )

    print()
    print(
        "STATUS:",
        (
            "PRJNA576508_ATLANTA13_FASTQS_RETRIEVED"
            if all_pass
            else "PRJNA576508_ATLANTA13_FASTQ_RETRIEVAL_REVIEW_REQUIRED"
        ),
    )

    if not all_pass:
        raise RuntimeError(
            "Not all frozen FASTQs passed MD5."
        )


if __name__ == "__main__":
    main()
