from pathlib import Path
from urllib.request import Request, urlopen
import gzip
import json
import re
import shutil
import tarfile

import pandas as pd


ROOT = Path.cwd()

OUT = (
    ROOT
    / "results/phase4/audit_03c_PRJEB24688_published_analysis_archive"
)

DATA_DIR = (
    ROOT
    / "data/raw/phase4/PRJEB24688/published_analysis"
)

ARCHIVE = (
    DATA_DIR
    / "Rossi_et_al_2018_analysis.tar.gz"
)

ZENODO_API = (
    "https://zenodo.org/api/records/1162703"
)


TEXT_SUFFIXES = {
    ".txt",
    ".tsv",
    ".csv",
    ".r",
    ".R",
    ".py",
    ".sh",
    ".md",
    ".readme",
    ".log",
    ".yaml",
    ".yml",
}


SEARCH_TERMS = [
    "htseq",
    "count",
    "counts",
    "expression",
    "matrix",
    "deseq",
    "rlog",
    "soft-core",
    "softcore",
    "5102",
    "pa14",
    "nc_008463",
    "invivo",
    "in vivo",
    "sputum",
    "stationary",
    "exponential",
    "3h",
    "24h",
]


def fetch_json(url):
    request = Request(
        url,
        headers={
            "User-Agent":
                "CF-State Phase4 archive audit"
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as handle:
        return json.loads(
            handle.read().decode(
                "utf-8"
            )
        )


def download_file(url, destination):
    request = Request(
        url,
        headers={
            "User-Agent":
                "CF-State Phase4 archive audit"
        },
    )

    with urlopen(
        request,
        timeout=120,
    ) as response:
        with destination.open(
            "wb"
        ) as handle:
            shutil.copyfileobj(
                response,
                handle,
            )


def is_probably_text(path):
    suffix = path.suffix

    if suffix in TEXT_SUFFIXES:
        return True

    lower = path.name.lower()

    text_names = [
        "readme",
        "makefile",
        "snakefile",
    ]

    return any(
        value in lower
        for value in text_names
    )


def safe_decode(raw):
    for encoding in [
        "utf-8",
        "latin-1",
    ]:
        try:
            return raw.decode(
                encoding
            )
        except UnicodeDecodeError:
            pass

    return None


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 03c")
    print("PRJEB24688 published analysis archive inspection")
    print("=" * 78)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Resolve Zenodo archive URL
    # ----------------------------------------------------------

    print()
    print("Retrieving Zenodo record")

    record = fetch_json(
        ZENODO_API
    )

    files = record.get(
        "files",
        []
    )

    if len(
        files
    ) == 0:
        raise RuntimeError(
            "Zenodo record contains no files."
        )

    selected = None

    for item in files:
        key = str(
            item.get(
                "key",
                ""
            )
        )

        if key.endswith(
            ".tar.gz"
        ):
            selected = item
            break

    if selected is None:
        raise RuntimeError(
            "Could not identify published tar.gz analysis archive."
        )

    archive_name = str(
        selected.get(
            "key",
            ""
        )
    )

    archive_size = int(
        selected.get(
            "size",
            0
        )
    )

    links = selected.get(
        "links",
        {}
    )

    archive_url = (
        links.get(
            "self"
        )
        or links.get(
            "content"
        )
    )

    if not archive_url:
        raise RuntimeError(
            "Could not resolve archive download URL."
        )

    print(
        "  archive:",
        archive_name,
    )

    print(
        "  expected size:",
        archive_size,
        "bytes",
    )

    # ----------------------------------------------------------
    # 2. Download archive
    # ----------------------------------------------------------

    if not ARCHIVE.exists():
        print()
        print("Downloading analysis archive")

        download_file(
            archive_url,
            ARCHIVE,
        )

    actual_size = ARCHIVE.stat().st_size

    print()
    print("Local archive")
    print(
        "  file:",
        ARCHIVE,
    )
    print(
        "  size:",
        actual_size,
        "bytes",
    )

    # ----------------------------------------------------------
    # 3. Inspect archive members without blindly extracting
    # ----------------------------------------------------------

    member_rows = []

    with tarfile.open(
        ARCHIVE,
        "r:gz",
    ) as archive:
        members = archive.getmembers()

        for member in members:
            member_rows.append(
                {
                    "name":
                        member.name,

                    "size_bytes":
                        member.size,

                    "is_file":
                        member.isfile(),

                    "suffix":
                        Path(
                            member.name
                        ).suffix,
                }
            )

    members_table = pd.DataFrame(
        member_rows
    )

    print()
    print("Archive structure")
    print(
        "  members:",
        len(
            members_table
        ),
    )
    print(
        "  files:",
        int(
            members_table[
                "is_file"
            ].sum()
        ),
    )

    print()
    print("Archive file listing")

    files_only = members_table.loc[
        members_table[
            "is_file"
        ]
    ].copy()

    print(
        files_only[
            [
                "name",
                "size_bytes",
            ]
        ]
        .sort_values(
            "name"
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. Candidate data files
    # ----------------------------------------------------------

    candidate_patterns = [
        "count",
        "counts",
        "expression",
        "matrix",
        "htseq",
        "rlog",
        "normalized",
        "norm",
        "fpkm",
        "tpm",
        "gene",
        "softcore",
        "soft_core",
        "5102",
    ]

    def candidate_score(name):
        lower = str(
            name
        ).lower()

        score = 0

        for pattern in candidate_patterns:
            if pattern in lower:
                score += 1

        return score

    files_only[
        "candidate_score"
    ] = files_only[
        "name"
    ].map(
        candidate_score
    )

    candidate_files = (
        files_only.loc[
            files_only[
                "candidate_score"
            ]
            > 0
        ]
        .sort_values(
            [
                "candidate_score",
                "name",
            ],
            ascending=[
                False,
                True,
            ],
        )
    )

    print()
    print("Filename-based expression/count candidates")

    if candidate_files.empty:
        print(
            "  none"
        )
    else:
        print(
            candidate_files[
                [
                    "name",
                    "size_bytes",
                    "candidate_score",
                ]
            ].to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 5. Search text-like files inside archive
    #
    # Only inspect files <=5 MB.
    # ----------------------------------------------------------

    search_rows = []

    text_inventory_rows = []

    with tarfile.open(
        ARCHIVE,
        "r:gz",
    ) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            path = Path(
                member.name
            )

            if member.size > 5_000_000:
                continue

            if not is_probably_text(
                path
            ):
                continue

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                continue

            raw = extracted.read()

            text = safe_decode(
                raw
            )

            if text is None:
                continue

            lines = text.splitlines()

            text_inventory_rows.append(
                {
                    "file":
                        member.name,

                    "size_bytes":
                        member.size,

                    "n_lines":
                        len(
                            lines
                        ),
                }
            )

            lower_text = text.lower()

            terms_found = [
                term
                for term in SEARCH_TERMS
                if term.lower()
                in lower_text
            ]

            if not terms_found:
                continue

            for line_number, line in enumerate(
                lines,
                start=1,
            ):
                lower_line = line.lower()

                matched_terms = [
                    term
                    for term in SEARCH_TERMS
                    if term.lower()
                    in lower_line
                ]

                if not matched_terms:
                    continue

                cleaned_line = line.strip()

                if len(
                    cleaned_line
                ) > 500:
                    cleaned_line = (
                        cleaned_line[
                            :500
                        ]
                        + "..."
                    )

                search_rows.append(
                    {
                        "file":
                            member.name,

                        "line_number":
                            line_number,

                        "matched_terms":
                            ",".join(
                                matched_terms
                            ),

                        "text":
                            cleaned_line,
                    }
                )

    text_inventory = pd.DataFrame(
        text_inventory_rows
    )

    search_hits = pd.DataFrame(
        search_rows
    )

    print()
    print("Text-like files inspected")
    print(
        " ",
        len(
            text_inventory
        ),
    )

    if not search_hits.empty:
        print()
        print("Relevant processing/search hits")

        for _, row in search_hits.head(
            150
        ).iterrows():
            print(
                str(
                    row[
                        "file"
                    ]
                )
                + ":"
                + str(
                    row[
                        "line_number"
                    ]
                )
                + " ["
                + str(
                    row[
                        "matched_terms"
                    ]
                )
                + "] "
                + str(
                    row[
                        "text"
                    ]
                )
            )

    # ----------------------------------------------------------
    # 6. Identify likely tabular data files by extension
    # ----------------------------------------------------------

    tabular_extensions = {
        ".csv",
        ".tsv",
        ".txt",
    }

    tabular_rows = []

    with tarfile.open(
        ARCHIVE,
        "r:gz",
    ) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            path = Path(
                member.name
            )

            if path.suffix.lower() not in tabular_extensions:
                continue

            if member.size > 20_000_000:
                continue

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                continue

            raw = extracted.read()

            text = safe_decode(
                raw
            )

            if text is None:
                continue

            lines = [
                line
                for line in text.splitlines()
                if line.strip()
            ]

            if len(
                lines
            ) < 2:
                continue

            first_line = lines[
                0
            ]

            if "\t" in first_line:
                delimiter = "\t"
                delimiter_name = "TAB"

            elif "," in first_line:
                delimiter = ","
                delimiter_name = "COMMA"

            else:
                delimiter = None
                delimiter_name = "UNKNOWN"

            if delimiter is not None:
                n_fields = len(
                    first_line.split(
                        delimiter
                    )
                )
            else:
                n_fields = 1

            tabular_rows.append(
                {
                    "file":
                        member.name,

                    "size_bytes":
                        member.size,

                    "nonempty_lines":
                        len(
                            lines
                        ),

                    "delimiter_guess":
                        delimiter_name,

                    "first_line_fields":
                        n_fields,

                    "header_preview":
                        first_line[
                            :300
                        ],
                }
            )

    tabular = pd.DataFrame(
        tabular_rows
    )

    if not tabular.empty:
        tabular = tabular.sort_values(
            [
                "first_line_fields",
                "nonempty_lines",
            ],
            ascending=[
                False,
                False,
            ],
        )

    print()
    print("Potential tabular files")

    if tabular.empty:
        print(
            "  none"
        )
    else:
        print(
            tabular[
                [
                    "file",
                    "size_bytes",
                    "nonempty_lines",
                    "delimiter_guess",
                    "first_line_fields",
                    "header_preview",
                ]
            ]
            .head(
                80
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 7. Detect sample names/accessions inside archive text
    # ----------------------------------------------------------

    expected_tokens = [
        "P24M1",
        "P30M0",
        "P73F1",
        "P77F1",
        "INVIVO",
        "ERR2275090",
        "ERR2275091",
        "ERR2275097",
        "ERR2275100",
    ]

    token_presence_rows = []

    combined_text = ""

    with tarfile.open(
        ARCHIVE,
        "r:gz",
    ) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            if member.size > 5_000_000:
                continue

            path = Path(
                member.name
            )

            if not is_probably_text(
                path
            ):
                continue

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                continue

            raw = extracted.read()

            text = safe_decode(
                raw
            )

            if text is not None:
                combined_text += (
                    "\n"
                    + text
                )

    combined_upper = (
        combined_text.upper()
    )

    for token in expected_tokens:
        token_presence_rows.append(
            {
                "token":
                    token,

                "present":
                    token.upper()
                    in combined_upper,
            }
        )

    token_presence = pd.DataFrame(
        token_presence_rows
    )

    print()
    print("Study-token recovery inside archive")
    print(
        token_presence.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 8. Save diagnostics
    # ----------------------------------------------------------

    members_table.to_csv(
        OUT
        / "archive_member_manifest.csv",
        index=False,
    )

    candidate_files.to_csv(
        OUT
        / "candidate_expression_files.csv",
        index=False,
    )

    text_inventory.to_csv(
        OUT
        / "text_file_inventory.csv",
        index=False,
    )

    search_hits.to_csv(
        OUT
        / "processing_search_hits.csv",
        index=False,
    )

    tabular.to_csv(
        OUT
        / "potential_tabular_files.csv",
        index=False,
    )

    token_presence.to_csv(
        OUT
        / "study_token_recovery.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 9. Decision
    #
    # PASS here means the archive is successfully audited.
    # It does NOT require that processed counts exist.
    #
    # If a useful count matrix exists, the next audit should
    # evaluate it.
    #
    # Otherwise, we proceed to canonical raw reprocessing using
    # the frozen Audit-03b run design.
    # ----------------------------------------------------------

    archive_readable = (
        len(
            members_table
        )
        > 0
    )

    processing_details_found = (
        not search_hits.empty
    )

    possible_matrix_found = False

    if not tabular.empty:
        possible_matrix_found = bool(
            (
                (
                    tabular[
                        "nonempty_lines"
                    ]
                    >= 4000
                )
                & (
                    tabular[
                        "first_line_fields"
                    ]
                    >= 5
                )
            ).any()
        )

    checks = {
        "archive_downloaded":
            ARCHIVE.exists(),

        "archive_size_nonzero":
            actual_size
            > 0,

        "archive_readable":
            archive_readable,

        "text_files_inspected":
            len(
                text_inventory
            )
            > 0,

        "processing_details_found":
            processing_details_found,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 03c archive checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    print()
    print(
        "Possible full expression/count matrix found:",
        "YES"
        if possible_matrix_found
        else "NO",
    )

    if passed:
        if possible_matrix_found:
            status = (
                "PRJEB24688_PUBLISHED_ARCHIVE_AUDITED_MATRIX_CANDIDATE_FOUND"
            )
        else:
            status = (
                "PRJEB24688_PUBLISHED_ARCHIVE_AUDITED_RAW_REPROCESSING_REQUIRED"
            )
    else:
        status = (
            "PRJEB24688_PUBLISHED_ARCHIVE_REVIEW_REQUIRED"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_03C_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed Audit 03c checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
