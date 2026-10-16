from pathlib import Path
import tarfile
import urllib.request

import pandas as pd


RAW_DIR = Path(
    "data/raw/expanded_biofilm/GSE120602/published_counts"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE120602/published_counts"
)

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


TAR_FILE = RAW_DIR / "GSE120602_RAW.tar"

URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE120nnn/GSE120602/suppl/GSE120602_RAW.tar"
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 47: "
    "inspect GSE120602 published count files"
)
print("=" * 100)


# ============================================================
# Download GEO supplementary count archive
# ============================================================

if not TAR_FILE.exists():

    print()
    print("Downloading GSE120602_RAW.tar...")

    urllib.request.urlretrieve(
        URL,
        TAR_FILE,
    )

else:

    print()
    print(
        "Archive already exists: "
        + str(TAR_FILE)
    )


if not TAR_FILE.exists():

    raise RuntimeError(
        "GEO archive was not downloaded."
    )


if TAR_FILE.stat().st_size == 0:

    raise RuntimeError(
        "GEO archive is empty."
    )


print(
    "Archive size: "
    + str(
        TAR_FILE.stat().st_size
    )
    + " bytes"
)


# ============================================================
# Inspect archive
# ============================================================

with tarfile.open(
    TAR_FILE,
    "r",
) as tar:

    members = [
        member
        for member in tar.getmembers()
        if member.isfile()
    ]


print()
print("=" * 100)
print("FILES IN GEO ARCHIVE")
print("=" * 100)


for member in members:

    print(
        member.name
        + "\t"
        + str(member.size)
        + " bytes"
    )


print()

print(
    "Number of files: "
    + str(
        len(members)
    )
)


# ============================================================
# Extract safely
# ============================================================

with tarfile.open(
    TAR_FILE,
    "r",
) as tar:

    for member in members:

        filename = Path(
            member.name
        ).name

        destination = (
            RAW_DIR
            / filename
        )

        if destination.exists():
            continue

        handle = tar.extractfile(
            member
        )

        if handle is None:

            raise RuntimeError(
                "Could not extract "
                + member.name
            )

        with open(
            destination,
            "wb",
        ) as output_handle:

            output_handle.write(
                handle.read()
            )


# ============================================================
# Sample definitions
# ============================================================

sample_metadata = {
    "GSM3405609": {
        "condition": "persister_ciprofloxacin",
        "replicate": 1,
    },
    "GSM3405610": {
        "condition": "persister_ciprofloxacin",
        "replicate": 2,
    },
    "GSM3405611": {
        "condition": "stationary",
        "replicate": 1,
    },
    "GSM3405612": {
        "condition": "stationary",
        "replicate": 2,
    },
    "GSM3405613": {
        "condition": "planktonic",
        "replicate": 1,
    },
    "GSM3405614": {
        "condition": "planktonic",
        "replicate": 2,
    },
    "GSM3405615": {
        "condition": "biofilm",
        "replicate": 1,
    },
    "GSM3405616": {
        "condition": "biofilm",
        "replicate": 2,
    },
}


# ============================================================
# Locate sample files
# ============================================================

records = []


for gsm, metadata in sample_metadata.items():

    matches = sorted(
        RAW_DIR.glob(
            gsm + "*"
        )
    )

    matches = [
        path
        for path in matches
        if path.is_file()
        and path.name != TAR_FILE.name
    ]


    if len(matches) != 1:

        print()
        print(
            "Matches for "
            + gsm
            + ":"
        )

        for match in matches:

            print(
                "  "
                + str(match)
            )

        raise RuntimeError(
            "Expected exactly one published count file for "
            + gsm
        )


    records.append(
        {
            "gsm": gsm,
            "condition": metadata["condition"],
            "replicate": metadata["replicate"],
            "file": str(matches[0]),
            "filename": matches[0].name,
        }
    )


file_manifest = pd.DataFrame(
    records
)


print()
print("=" * 100)
print("CANONICAL SAMPLE FILES")
print("=" * 100)

print(
    file_manifest.to_string(
        index=False
    )
)


file_manifest.to_csv(
    OUT_DIR
    / "GSE120602_published_count_manifest.csv",
    index=False,
)


# ============================================================
# Inspect count tables
# ============================================================

inspection_records = []


for row in records:

    path = Path(
        row["file"]
    )

    table = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
    )


    print()
    print("=" * 100)
    print(
        row["gsm"]
        + " — "
        + row["condition"]
    )
    print("=" * 100)

    print(
        "File: "
        + path.name
    )

    print(
        "Shape: "
        + str(
            table.shape
        )
    )

    print(
        "Columns: "
        + str(
            list(
                table.columns
            )
        )
    )

    print()
    print(
        table.head(
            8
        ).to_string(
            index=False
        )
    )


    inspection_records.append(
        {
            "gsm": row["gsm"],
            "condition": row["condition"],
            "replicate": row["replicate"],
            "n_rows": int(
                table.shape[0]
            ),
            "n_columns": int(
                table.shape[1]
            ),
            "columns": " | ".join(
                [
                    str(column)
                    for column in table.columns
                ]
            ),
        }
    )


inspection = pd.DataFrame(
    inspection_records
)


print()
print("=" * 100)
print("COUNT FILE SUMMARY")
print("=" * 100)

print(
    inspection.to_string(
        index=False
    )
)


inspection.to_csv(
    OUT_DIR
    / "GSE120602_published_count_structure.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 47 COMPLETE")
print("=" * 100)

print(
    "Next decision: determine whether the published "
    "gene identifiers map cleanly to the frozen "
    "4,858-gene PAO1 universe."
)

print()
