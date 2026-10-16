from pathlib import Path
from urllib.request import Request, urlopen
import re
import subprocess

import pandas as pd


ROOT = Path.cwd()

DATA_DIR = (
    ROOT
    / "data/raw/phase4/PRJNA576508/published_supplement"
)

OUT = (
    ROOT
    / "results/phase4/audit_04b_PRJNA576508_supplement_and_decoy_freeze"
)

FROZEN_COHORT = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "FROZEN_PRJNA576508_independent_third_cohort.csv"
)

FILES = {
    "mbio.03042-19-sd001.xlsx": [
        (
            "https://journals.asm.org/doi/suppl/"
            "10.1128/mbio.03042-19/suppl_file/"
            "mbio.03042-19-sd001.xlsx"
        ),
        (
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC6960289/bin/mbio.03042-19-sd001.xlsx"
        ),
        (
            "https://www.ncbi.nlm.nih.gov/pmc/articles/"
            "PMC6960289/bin/mbio.03042-19-sd001.xlsx"
        ),
    ],

    "mbio.03042-19-sd002.xlsx": [
        (
            "https://journals.asm.org/doi/suppl/"
            "10.1128/mbio.03042-19/suppl_file/"
            "mbio.03042-19-sd002.xlsx"
        ),
        (
            "https://pmc.ncbi.nlm.nih.gov/articles/"
            "PMC6960289/bin/mbio.03042-19-sd002.xlsx"
        ),
        (
            "https://www.ncbi.nlm.nih.gov/pmc/articles/"
            "PMC6960289/bin/mbio.03042-19-sd002.xlsx"
        ),
    ],
}


def valid_xlsx_bytes(data):
    # XLSX is a ZIP archive and therefore begins with PK.
    return (
        data is not None
        and len(data) > 1000
        and data[:2] == b"PK"
    )


def urllib_download(url):
    request = Request(
        url,
        headers={
            "User-Agent":
                (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 "
                    "Chrome/153 Safari/537.36"
                ),

            "Accept":
                (
                    "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet,application/octet-stream,*/*"
                ),

            "Referer":
                "https://journals.asm.org/doi/full/10.1128/mbio.03042-19",
        },
    )

    with urlopen(
        request,
        timeout=60,
    ) as response:
        return response.read()


def curl_download(url, destination):
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
        (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 Chrome/153 Safari/537.36"
        ),
        "-e",
        "https://journals.asm.org/doi/full/10.1128/mbio.03042-19",
        "-o",
        str(destination),
        url,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        return False

    if not destination.exists():
        return False

    data = destination.read_bytes()

    if not valid_xlsx_bytes(
        data
    ):
        destination.unlink(
            missing_ok=True
        )
        return False

    return True


def download_with_fallback(
    filename,
    urls,
    destination,
):
    if destination.exists():
        existing = destination.read_bytes()

        if valid_xlsx_bytes(
            existing
        ):
            print(
                "  already present:",
                destination,
            )
            return "existing"

        destination.unlink()

    errors = []

    for url in urls:
        print(
            "  trying:",
            url,
        )

        try:
            data = urllib_download(
                url
            )

            if valid_xlsx_bytes(
                data
            ):
                destination.write_bytes(
                    data
                )

                return url

            errors.append(
                url
                + " -> response was not an XLSX file"
            )

        except Exception as error:
            errors.append(
                url
                + " -> "
                + type(error).__name__
                + ": "
                + str(error)
            )

        # Curl fallback for the same URL.
        try:
            if curl_download(
                url,
                destination,
            ):
                return (
                    url
                    + " [curl]"
                )

        except Exception as error:
            errors.append(
                url
                + " curl -> "
                + type(error).__name__
                + ": "
                + str(error)
            )

    raise RuntimeError(
        "Could not download "
        + filename
        + ". Attempts:\n"
        + "\n".join(
            errors
        )
    )


def normalized_em_label(value):
    text = str(
        value
    ).strip().upper()

    match = re.fullmatch(
        r"EM(\d+)([A-Z]?)",
        text,
    )

    if match is None:
        return None

    # EM35B etc. correspond to frozen 04a label EM35.
    return (
        "EM"
        + match.group(1)
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04b")
    print("PRJNA576508 published supplement and decoy-reference freeze")
    print("=" * 78)

    if not FROZEN_COHORT.exists():
        raise FileNotFoundError(
            str(
                FROZEN_COHORT
            )
        )

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Download supplements directly
    #
    # Deliberately do NOT scrape the ASM article page.
    # ----------------------------------------------------------

    downloaded = {}

    print()
    print("Retrieving published supplementary workbooks")

    for filename, urls in FILES.items():
        destination = (
            DATA_DIR
            / filename
        )

        source = download_with_fallback(
            filename,
            urls,
            destination,
        )

        downloaded[
            filename
        ] = destination

        print(
            "  OK:",
            filename,
        )
        print(
            "    bytes:",
            destination.stat().st_size,
        )
        print(
            "    source:",
            source,
        )

    s1_path = downloaded[
        "mbio.03042-19-sd001.xlsx"
    ]

    s2_path = downloaded[
        "mbio.03042-19-sd002.xlsx"
    ]

    # ----------------------------------------------------------
    # 2. Validate Excel workbooks
    # ----------------------------------------------------------

    try:
        s1_excel = pd.ExcelFile(
            s1_path
        )

        s2_excel = pd.ExcelFile(
            s2_path
        )

    except Exception as error:
        raise RuntimeError(
            "Downloaded supplement is not a valid Excel workbook: "
            + str(error)
        )

    print()
    print("Data Set S1 sheets")

    for index, sheet in enumerate(
        s1_excel.sheet_names,
        start=1,
    ):
        print(
            " ",
            index,
            repr(
                sheet
            ),
        )

    print()
    print("Data Set S2 sheets")

    for index, sheet in enumerate(
        s2_excel.sheet_names,
        start=1,
    ):
        print(
            " ",
            index,
            repr(
                sheet
            ),
        )

    if len(
        s1_excel.sheet_names
    ) < 3:
        raise RuntimeError(
            "Expected at least three sheets in Data Set S1."
        )

    # Publication defines:
    #
    # S1 Sheet 1 = human samples
    # S1 Sheet 2 = model samples
    # S1 Sheet 3 = non-Pseudomonas decoy genomes

    human = pd.read_excel(
        s1_path,
        sheet_name=s1_excel.sheet_names[
            0
        ],
    )

    models = pd.read_excel(
        s1_path,
        sheet_name=s1_excel.sheet_names[
            1
        ],
    )

    decoys = pd.read_excel(
        s1_path,
        sheet_name=s1_excel.sheet_names[
            2
        ],
    )

    print()
    print("Published Data Set S1 structure")
    print(
        "  human sample rows:",
        len(
            human
        ),
    )
    print(
        "  model sample rows:",
        len(
            models
        ),
    )
    print(
        "  decoy rows:",
        len(
            decoys
        ),
    )

    print()
    print("Human metadata columns")
    print(
        list(
            human.columns
        )
    )

    print()
    print("Human metadata preview")
    print(
        human.head(
            25
        ).to_string(
            index=False
        )
    )

    print()
    print("Decoy-reference columns")
    print(
        list(
            decoys.columns
        )
    )

    print()
    print("Decoy-reference preview")
    print(
        decoys.head(
            30
        ).to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 3. Recover the exact 13 frozen Atlanta samples from the
    #    publication's human metadata
    # ----------------------------------------------------------

    frozen = pd.read_csv(
        FROZEN_COHORT
    )

    frozen_labels = set(
        frozen[
            "short_sample_label"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    detected_labels = set()

    for column in human.columns:
        for value in (
            human[
                column
            ]
            .fillna("")
            .astype(str)
        ):
            matches = re.findall(
                r"\bEM\d+[A-Za-z]?\b",
                value,
                flags=re.IGNORECASE,
            )

            for match in matches:
                normalized = normalized_em_label(
                    match
                )

                if normalized is not None:
                    detected_labels.add(
                        normalized
                    )

    recovered = (
        frozen_labels
        .intersection(
            detected_labels
        )
    )

    missing = (
        frozen_labels
        - detected_labels
    )

    extra_em = (
        detected_labels
        - frozen_labels
    )

    print()
    print("Atlanta sample recovery")
    print(
        "  Audit-04a frozen:",
        len(
            frozen_labels
        ),
        sorted(
            frozen_labels
        ),
    )
    print(
        "  recovered from Data Set S1:",
        len(
            recovered
        ),
        sorted(
            recovered
        ),
    )
    print(
        "  frozen labels missing:",
        sorted(
            missing
        ),
    )
    print(
        "  other EM labels in human sheet:",
        sorted(
            extra_em
        ),
    )

    # ----------------------------------------------------------
    # 4. Audit exact decoy-reference inventory
    # ----------------------------------------------------------

    decoys = decoys.dropna(
        how="all"
    ).copy()

    accession_pattern = re.compile(
        (
            r"\b(?:"
            r"NC_|NZ_|CP|AE|AP|AM|FM|FN|FO|HG|LN|LT|LR|CM|AC|"
            r"GCF_|GCA_"
            r")[A-Z0-9_.]+\b"
        ),
        flags=re.IGNORECASE,
    )

    accession_tokens = []

    for column in decoys.columns:
        values = (
            decoys[
                column
            ]
            .fillna("")
            .astype(str)
        )

        for value in values:
            accession_tokens.extend(
                accession_pattern.findall(
                    value
                )
            )

    accession_tokens = sorted(
        {
            token.strip()
            for token in accession_tokens
            if token.strip()
        }
    )

    print()
    print("Published decoy-reference inventory")
    print(
        "  non-empty rows:",
        len(
            decoys
        ),
    )
    print(
        "  unique accession-like identifiers:",
        len(
            accession_tokens
        ),
    )

    if accession_tokens:
        print()
        print("First decoy accession-like identifiers")

        for token in accession_tokens[
            :120
        ]:
            print(
                " ",
                token,
            )

    # ----------------------------------------------------------
    # 5. Data Set S2 audit
    #
    # The publication describes S2 as DERIVED analyses:
    # z-scores, shared gene list, clinic DE, elusive genes.
    #
    # We inspect it but do not treat it as raw sample counts.
    # ----------------------------------------------------------

    s2_rows = []

    for index, sheet in enumerate(
        s2_excel.sheet_names,
        start=1,
    ):
        table = pd.read_excel(
            s2_path,
            sheet_name=sheet,
        )

        s2_rows.append(
            {
                "sheet_number":
                    index,

                "sheet_name":
                    sheet,

                "rows":
                    len(
                        table
                    ),

                "columns":
                    len(
                        table.columns
                    ),

                "column_preview":
                    " | ".join(
                        str(
                            column
                        )
                        for column in list(
                            table.columns
                        )[
                            :20
                        ]
                    ),
            }
        )

        print()
        print(
            "Data Set S2 sheet",
            index,
            repr(
                sheet
            ),
        )
        print(
            "  shape:",
            table.shape,
        )
        print(
            "  columns:",
            list(
                table.columns
            )[
                :30
            ],
        )

        print(
            table.head(
                8
            ).to_string(
                index=False
            )
        )

    s2_inventory = pd.DataFrame(
        s2_rows
    )

    # ----------------------------------------------------------
    # 6. Freeze outputs
    # ----------------------------------------------------------

    human.to_csv(
        OUT
        / "Cornforth_DataS1_human_samples.csv",
        index=False,
    )

    models.to_csv(
        OUT
        / "Cornforth_DataS1_model_samples.csv",
        index=False,
    )

    decoys.to_csv(
        OUT
        / "Cornforth_DataS1_decoy_reference_table.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "accession_like_token":
                accession_tokens
        }
    ).to_csv(
        OUT
        / "Cornforth_decoy_accession_tokens.csv",
        index=False,
    )

    s2_inventory.to_csv(
        OUT
        / "Cornforth_DataS2_sheet_inventory.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "frozen_Atlanta_label":
                sorted(
                    frozen_labels
                ),

            "recovered_in_DataS1":
                [
                    label
                    in recovered
                    for label in sorted(
                        frozen_labels
                    )
                ],
        }
    ).to_csv(
        OUT
        / "Atlanta_sample_metadata_recovery.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 7. Checks
    # ----------------------------------------------------------

    checks = {
        "DataS1_valid_XLSX":
            len(
                s1_excel.sheet_names
            )
            >= 3,

        "DataS2_valid_XLSX":
            len(
                s2_excel.sheet_names
            )
            >= 1,

        "all_13_frozen_Atlanta_samples_recovered":
            len(
                recovered
            )
            == 13,

        "human_metadata_nonempty":
            len(
                human
            )
            > 0,

        "model_metadata_nonempty":
            len(
                models
            )
            > 0,

        "decoy_reference_nonempty":
            len(
                decoys
            )
            > 0,

        "DataS2_has_expected_four_or_more_sheets":
            len(
                s2_excel.sheet_names
            )
            >= 4,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04b checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    status = (
        "PRJNA576508_SUPPLEMENT_AND_DECOY_REFERENCE_FROZEN"
        if passed
        else "PRJNA576508_SUPPLEMENT_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: Data Set S2 contains derived analyses, not a "
        "published per-sample raw count matrix suitable for replacing "
        "the original read-processing pipeline."
    )

    print()
    print(
        "No PRJNA576508 expression values have yet been compared "
        "with the frozen Audit-03h CF-airway candidate."
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
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    (
        OUT
        / "AUDIT_04B_STATUS.txt"
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
            "Failed Audit 04b checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
