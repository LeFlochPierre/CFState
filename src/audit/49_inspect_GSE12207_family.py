from pathlib import Path
import gzip
import subprocess


RAW_DIR = Path(
    "data/raw/expanded_biofilm/GSE12207"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE12207"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SOFT_FILE = (
    RAW_DIR
    / "GSE12207_family.soft.gz"
)

URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE12nnn/GSE12207/soft/"
    "GSE12207_family.soft.gz"
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 49: "
    "inspect GSE12207 family SOFT"
)
print("=" * 100)


# ============================================================
# Download
# ============================================================

if not SOFT_FILE.exists():

    print()
    print("Downloading GSE12207 family SOFT...")

    subprocess.run(
        [
            "curl",
            "-4",
            "--http1.1",
            "-L",
            "--fail",
            "--retry",
            "5",
            "--retry-delay",
            "2",
            URL,
            "-o",
            str(SOFT_FILE),
        ],
        check=True,
    )

else:

    print()
    print(
        "SOFT already present: "
        + str(SOFT_FILE)
    )


if not SOFT_FILE.exists():

    raise RuntimeError(
        "SOFT file was not downloaded."
    )


if SOFT_FILE.stat().st_size == 0:

    raise RuntimeError(
        "SOFT file is empty."
    )


# ============================================================
# Check gzip
# ============================================================

with gzip.open(
    SOFT_FILE,
    "rt",
    encoding="utf-8",
    errors="replace",
) as handle:

    first_line = handle.readline()


if not first_line:

    raise RuntimeError(
        "Could not read gzip SOFT file."
    )


print()
print(
    "SOFT size: "
    + str(
        SOFT_FILE.stat().st_size
    )
    + " bytes"
)


# ============================================================
# Parse basic sample metadata
# ============================================================

samples = []

current_sample = None

platform_id = None

platform_header = None

platform_preview = []

inside_platform_table = False


with gzip.open(
    SOFT_FILE,
    "rt",
    encoding="utf-8",
    errors="replace",
) as handle:

    for raw_line in handle:

        line = raw_line.rstrip(
            "\n"
        )


        # ----------------------------------------------------
        # Platform
        # ----------------------------------------------------

        if line.startswith(
            "^PLATFORM = "
        ):

            platform_id = (
                line.split(
                    "=",
                    1,
                )[1]
                .strip()
            )


        if line == "!platform_table_begin":

            inside_platform_table = True
            continue


        if line == "!platform_table_end":

            inside_platform_table = False
            continue


        if inside_platform_table:

            if platform_header is None:

                platform_header = line.split(
                    "\t"
                )

            elif len(
                platform_preview
            ) < 8:

                platform_preview.append(
                    line
                )


        # ----------------------------------------------------
        # Samples
        # ----------------------------------------------------

        if line.startswith(
            "^SAMPLE = "
        ):

            if current_sample is not None:

                samples.append(
                    current_sample
                )


            current_sample = {
                "gsm":
                    line.split(
                        "=",
                        1,
                    )[1].strip(),

                "title":
                    None,

                "platform":
                    None,

                "rows":
                    None,
            }

            continue


        if current_sample is None:
            continue


        if line.startswith(
            "!Sample_title = "
        ):

            current_sample[
                "title"
            ] = (
                line.split(
                    "=",
                    1,
                )[1]
                .strip()
            )


        elif line.startswith(
            "!Sample_platform_id = "
        ):

            current_sample[
                "platform"
            ] = (
                line.split(
                    "=",
                    1,
                )[1]
                .strip()
            )


        elif line.startswith(
            "!Sample_data_row_count = "
        ):

            current_sample[
                "rows"
            ] = (
                line.split(
                    "=",
                    1,
                )[1]
                .strip()
            )


if current_sample is not None:

    samples.append(
        current_sample
    )


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print("PLATFORM")
print("=" * 100)

print(
    "Platform ID: "
    + str(
        platform_id
    )
)


if platform_header is not None:

    print()

    print(
        "Platform columns:"
    )

    print(
        platform_header
    )


    print()

    print(
        "Platform preview:"
    )

    for row in platform_preview:

        print(
            row
        )


print()
print("=" * 100)
print("SAMPLES")
print("=" * 100)


for sample in samples:

    print(
        sample["gsm"]
        + "\t"
        + str(
            sample["title"]
        )
        + "\t"
        + str(
            sample["platform"]
        )
        + "\trows="
        + str(
            sample["rows"]
        )
    )


print()
print(
    "Number of samples: "
    + str(
        len(
            samples
        )
    )
)


# ============================================================
# Expected design check
# ============================================================

expected = {
    "GSM307112":
        "Planktonic cells exp phase, biological replicate 1",

    "GSM307113":
        "Planktonic cells exp phase, biological replicate 2",

    "GSM307114":
        "Planktonic cells exp phase, biological replicate 3",

    "GSM307115":
        "Planktonic cells stat phase, biological replicate 1",

    "GSM307116":
        "Planktonic cells stat phase, biological replicate 2",

    "GSM307117":
        "Planktonic cells stat phase, biological replicate 3",

    "GSM307118":
        "Colonies after 15 hrs, biological replicate 1",

    "GSM307119":
        "Colonies after 15 hrs, biological replicate 2",

    "GSM307120":
        "Colonies after 15 hrs, biological replicate 3",

    "GSM307121":
        "Colonies after 40 hrs, biological replicate 1",

    "GSM307122":
        "Colonies after 40 hrs, biological replicate 2",

    "GSM307123":
        "Colonies after 40 hrs, biological replicate 3",

    "GSM307124":
        "Biofilms after 3 days, biological replicate 1",

    "GSM307125":
        "Biofilms after 3 days, biological replicate 2",

    "GSM307126":
        "Biofilms after 3 days, biological replicate 3",
}


observed = {
    sample["gsm"]:
        sample["title"]
    for sample in samples
}


missing = sorted(
    set(
        expected
    )
    - set(
        observed
    )
)


unexpected = sorted(
    set(
        observed
    )
    - set(
        expected
    )
)


title_mismatches = []


for gsm, expected_title in expected.items():

    if gsm not in observed:
        continue

    if observed[
        gsm
    ] != expected_title:

        title_mismatches.append(
            (
                gsm,
                expected_title,
                observed[
                    gsm
                ],
            )
        )


print()
print("=" * 100)
print("DESIGN CHECK")
print("=" * 100)

print(
    "Expected samples: 15"
)

print(
    "Observed samples: "
    + str(
        len(
            samples
        )
    )
)

print(
    "Missing expected samples: "
    + str(
        missing
    )
)

print(
    "Unexpected samples: "
    + str(
        unexpected
    )
)

print(
    "Title mismatches: "
    + str(
        len(
            title_mismatches
        )
    )
)


if title_mismatches:

    for mismatch in title_mismatches:

        print(
            mismatch
        )


print()
print("=" * 100)
print("AUDIT 49 COMPLETE")
print("=" * 100)

print(
    "Next step: map GPL84 probe sets to PAO1 "
    "locus tags and extract the processed "
    "expression matrix."
)

print()
