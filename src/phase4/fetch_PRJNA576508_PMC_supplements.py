from pathlib import Path
from urllib.request import Request, urlopen
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path.cwd()

DATA_DIR = (
    ROOT
    / "data/raw/phase4/PRJNA576508/published_supplement"
)

PACKAGE = (
    DATA_DIR
    / "PMC6960289_oa_package.tar.gz"
)

EXPECTED = [
    "mbio.03042-19-sd001.xlsx",
    "mbio.03042-19-sd002.xlsx",
]

OA_API_URLS = [
    "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC6960289",
    "https://pmc.ncbi.nlm.nih.gov/utils/oa/oa.fcgi?id=PMC6960289",
]


def fetch_bytes_urllib(url):
    request = Request(
        url,
        headers={
            "User-Agent":
                "Mozilla/5.0 CF-State supplementary fetch"
        },
    )

    with urlopen(
        request,
        timeout=90,
    ) as response:
        return response.read()


def fetch_bytes_curl(url):
    command = [
        "curl",
        "-L",
        "--fail",
        "--silent",
        "--show-error",
        "--retry",
        "3",
        "-A",
        "Mozilla/5.0 CF-State supplementary fetch",
        url,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.decode(
                "utf-8",
                errors="ignore",
            )
        )

    return result.stdout


def fetch_bytes(url):
    try:
        return fetch_bytes_urllib(
            url
        )

    except Exception:
        return fetch_bytes_curl(
            url
        )


def get_oa_package_url():
    errors = []

    for api_url in OA_API_URLS:
        print(
            "Trying PMC OA API:",
            api_url,
        )

        try:
            data = fetch_bytes(
                api_url
            )

            root = ET.fromstring(
                data
            )

            links = root.findall(
                ".//link"
            )

            for link in links:
                if (
                    str(
                        link.attrib.get(
                            "format",
                            ""
                        )
                    ).lower()
                    != "tgz"
                ):
                    continue

                href = link.attrib.get(
                    "href"
                )

                if not href:
                    continue

                if href.startswith(
                    "ftp://ftp.ncbi.nlm.nih.gov/"
                ):
                    href = (
                        "https://ftp.ncbi.nlm.nih.gov/"
                        + href[
                            len(
                                "ftp://ftp.ncbi.nlm.nih.gov/"
                            ):
                        ]
                    )

                return href

            errors.append(
                api_url
                + " -> no tgz link in OA response"
            )

        except Exception as error:
            errors.append(
                api_url
                + " -> "
                + type(
                    error
                ).__name__
                + ": "
                + str(
                    error
                )
            )

    raise RuntimeError(
        "Could not resolve PMC OA package.\n"
        + "\n".join(
            errors
        )
    )


def download_package(url):
    if PACKAGE.exists():
        try:
            with tarfile.open(
                PACKAGE,
                "r:gz",
            ):
                print(
                    "Existing PMC package is readable:",
                    PACKAGE,
                )

                return

        except Exception:
            PACKAGE.unlink()

    print()
    print(
        "Downloading PMC OA package:"
    )
    print(
        " ",
        url,
    )

    data = fetch_bytes(
        url
    )

    PACKAGE.write_bytes(
        data
    )

    print(
        "  downloaded bytes:",
        PACKAGE.stat().st_size,
    )

    with tarfile.open(
        PACKAGE,
        "r:gz",
    ):
        pass


def extract_xlsx_files():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = []

    with tarfile.open(
        PACKAGE,
        "r:gz",
    ) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            if not member.name.lower().endswith(
                ".xlsx"
            ):
                continue

            handle = archive.extractfile(
                member
            )

            if handle is None:
                continue

            data = handle.read()

            candidates.append(
                {
                    "member":
                        member.name,

                    "basename":
                        Path(
                            member.name
                        ).name,

                    "size":
                        len(
                            data
                        ),

                    "data":
                        data,
                }
            )

    print()
    print(
        "XLSX files found in PMC package:",
        len(
            candidates
        ),
    )

    for candidate in candidates:
        print(
            " ",
            candidate[
                "basename"
            ],
            candidate[
                "size"
            ],
            "bytes",
            "("
            + candidate[
                "member"
            ]
            + ")",
        )

    if len(
        candidates
    ) < 2:
        raise RuntimeError(
            "Expected at least two XLSX files in PMC package."
        )

    selected = {}

    # First try exact original filenames.
    for expected in EXPECTED:
        exact = [
            candidate
            for candidate in candidates
            if candidate[
                "basename"
            ].lower()
            == expected.lower()
        ]

        if len(
            exact
        ) == 1:
            selected[
                expected
            ] = exact[
                0
            ]

    # PMC sometimes renames supplementary assets.
    # If exact filenames are unavailable, Data Set S1 is the
    # smaller XLSX and Data Set S2 is the larger XLSX.
    if len(
        selected
    ) < 2:
        ordered = sorted(
            candidates,
            key=lambda item:
                item[
                    "size"
                ],
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
            "Original supplement filenames were not preserved "
            "by PMC; assigning by workbook size."
        )

    for expected, candidate in selected.items():
        destination = (
            DATA_DIR
            / expected
        )

        destination.write_bytes(
            candidate[
                "data"
            ]
        )

        if not zipfile.is_zipfile(
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
            candidate[
                "member"
            ],
        )
        print(
            "  bytes:",
            destination.stat().st_size,
        )

    for expected in EXPECTED:
        path = (
            DATA_DIR
            / expected
        )

        if not path.exists():
            raise RuntimeError(
                "Missing expected supplement after extraction: "
                + str(
                    path
                )
            )

    print()
    print(
        "PMC supplementary extraction complete."
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4")
    print("Recover Cornforth supplements through PMC Open Access package")
    print("=" * 78)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    package_url = get_oa_package_url()

    print()
    print(
        "Resolved OA package:"
    )
    print(
        " ",
        package_url,
    )

    download_package(
        package_url
    )

    extract_xlsx_files()


if __name__ == "__main__":
    main()
