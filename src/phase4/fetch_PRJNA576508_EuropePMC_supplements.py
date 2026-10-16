from pathlib import Path
import subprocess
import zipfile
import shutil


ROOT = Path.cwd()

DATA_DIR = (
    ROOT
    / "data/raw/phase4/PRJNA576508/published_supplement"
)

ZIP_PATH = (
    DATA_DIR
    / "PMC6960289_supplementaryFiles.zip"
)

URL = (
    "https://www.ebi.ac.uk/europepmc/webservices/rest/"
    "PMC6960289/supplementaryFiles"
)

EXPECTED = [
    "mbio.03042-19-sd001.xlsx",
    "mbio.03042-19-sd002.xlsx",
]


def valid_xlsx(path):
    if not path.exists():
        return False

    if path.stat().st_size < 1000:
        return False

    return zipfile.is_zipfile(
        path
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4")
    print("Fetch Cornforth supplements through Europe PMC")
    print("=" * 78)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Europe PMC supplementary endpoint")
    print(
        " ",
        URL,
    )

    command = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "--retry-delay",
        "2",
        "-A",
        "CF-State supplementary retrieval",
        "-o",
        str(
            ZIP_PATH
        ),
        URL,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Europe PMC download failed:\n"
            + result.stderr
        )

    print()
    print(
        "Downloaded:",
        ZIP_PATH,
    )
    print(
        "  bytes:",
        ZIP_PATH.stat().st_size,
    )

    if not zipfile.is_zipfile(
        ZIP_PATH
    ):
        preview = ZIP_PATH.read_bytes()[
            :500
        ]

        raise RuntimeError(
            "Europe PMC response is not a ZIP archive.\n"
            + repr(
                preview
            )
        )

    with zipfile.ZipFile(
        ZIP_PATH,
        "r",
    ) as archive:

        members = archive.namelist()

        print()
        print(
            "Archive members:",
            len(
                members
            ),
        )

        for member in members:
            print(
                " ",
                member,
            )

        xlsx_members = [
            member
            for member in members
            if member.lower().endswith(
                ".xlsx"
            )
        ]

        print()
        print(
            "XLSX members:",
            len(
                xlsx_members
            ),
        )

        for member in xlsx_members:
            info = archive.getinfo(
                member
            )

            print(
                " ",
                member,
                info.file_size,
                "bytes",
            )

        if len(
            xlsx_members
        ) < 2:
            raise RuntimeError(
                "Expected at least two XLSX supplements."
            )

        selected = {}

        # First try exact / case-insensitive filename matches.
        for expected in EXPECTED:
            matches = [
                member
                for member in xlsx_members
                if Path(
                    member
                ).name.lower()
                == expected.lower()
            ]

            if len(
                matches
            ) == 1:
                selected[
                    expected
                ] = matches[
                    0
                ]

        # If Europe PMC renamed them, use published sizes:
        #
        # Data Set S1 ~125.6 KB
        # Data Set S2 ~756.2 KB
        #
        # Therefore smaller XLSX = S1,
        # larger XLSX = S2.
        if len(
            selected
        ) != 2:

            ordered = sorted(
                xlsx_members,
                key=lambda member:
                    archive.getinfo(
                        member
                    ).file_size,
            )

            selected = {
                EXPECTED[
                    0
                ]:
                    ordered[
                        0
                    ],

                EXPECTED[
                    1
                ]:
                    ordered[
                        -1
                    ],
            }

            print()
            print(
                "Using workbook-size fallback because archive "
                "filenames differ from ASM filenames."
            )

        for expected, source_member in selected.items():

            destination = (
                DATA_DIR
                / expected
            )

            with archive.open(
                source_member,
                "r",
            ) as source:
                with destination.open(
                    "wb"
                ) as target:
                    shutil.copyfileobj(
                        source,
                        target,
                    )

            if not valid_xlsx(
                destination
            ):
                raise RuntimeError(
                    "Extracted file is not a valid XLSX: "
                    + str(
                        destination
                    )
                )

            print()
            print(
                "Saved:",
                destination,
            )
            print(
                "  source member:",
                source_member,
            )
            print(
                "  bytes:",
                destination.stat().st_size,
            )

    print()
    print("Supplement recovery complete.")

    for expected in EXPECTED:
        destination = (
            DATA_DIR
            / expected
        )

        print(
            " ",
            expected,
            "OK"
            if valid_xlsx(
                destination
            )
            else "INVALID",
        )


if __name__ == "__main__":
    main()
