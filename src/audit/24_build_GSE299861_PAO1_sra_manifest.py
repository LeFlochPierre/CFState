from pathlib import Path
import re
import io
import time

import pandas as pd
import requests


OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# The six PAO1 samples only
# ============================================================

SAMPLES = {
    "GSM9047975": {
        "sample": "biofilm_PAO1_1",
        "state": "biofilm",
    },
    "GSM9047976": {
        "sample": "biofilm_PAO1_2",
        "state": "biofilm",
    },
    "GSM9047977": {
        "sample": "biofilm_PAO1_3",
        "state": "biofilm",
    },
    "GSM9047978": {
        "sample": "planktonic_PAO1_1",
        "state": "planktonic",
    },
    "GSM9047979": {
        "sample": "planktonic_PAO1_2",
        "state": "planktonic",
    },
    "GSM9047980": {
        "sample": "planktonic_PAO1_3",
        "state": "planktonic",
    },
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 24: "
    "build GSE299861 PAO1 SRA manifest"
)
print("=" * 100)


# ============================================================
# Retrieve SRX accession from GEO SOFT
# ============================================================
# ============================================================
# NCBI request helper
#
# NCBI occasionally returns HTTP 429 when requests arrive
# too quickly. Retry with increasing delays.
# ============================================================

def ncbi_get(
    url,
    params=None,
    max_attempts=8,
):

    headers = {
        "User-Agent":
            "CF-State/1.0 academic-research"
    }

    for attempt in range(
        1,
        max_attempts + 1,
    ):

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=120,
        )

        if response.status_code == 429:

            wait_seconds = min(
                2 ** attempt,
                60,
            )

            print(
                f"  NCBI rate limit hit "
                f"(HTTP 429). "
                f"Waiting {wait_seconds}s..."
            )

            time.sleep(
                wait_seconds
            )

            continue

        response.raise_for_status()

        # Stay comfortably below the unauthenticated
        # NCBI request-rate limit.
        time.sleep(
            0.5
        )

        return response


    raise RuntimeError(
        "NCBI continued returning HTTP 429 "
        f"after {max_attempts} attempts."
    )

def get_srx(gsm):

    url = (
        "https://www.ncbi.nlm.nih.gov/geo/"
        "query/acc.cgi"
        f"?acc={gsm}"
        "&targ=self"
        "&form=text"
        "&view=full"
    )

    r = ncbi_get(
        url
    )

    r.raise_for_status()

    text = r.text

    matches = re.findall(
        r"SRX\d+",
        text,
    )

    matches = sorted(
        set(matches)
    )

    if not matches:

        raise RuntimeError(
            f"No SRX accession found for {gsm}"
        )

    if len(matches) > 1:

        print(
            f"WARNING: {gsm} has multiple SRX values: "
            f"{matches}"
        )

    return matches[0]


# ============================================================
# Retrieve SRA RunInfo from SRX
# ============================================================

def get_runinfo(srx):

    # Find SRA database UID
    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/esearch.fcgi"
    )

    params = {
        "db": "sra",
        "term": f"{srx}[ACCN]",
        "retmode": "json",
    }

    r = ncbi_get(
        search_url,
        params=params
    )

    r.raise_for_status()

    payload = r.json()

    ids = payload[
        "esearchresult"
    ][
        "idlist"
    ]

    if not ids:

        raise RuntimeError(
            f"No SRA database record found for {srx}"
        )

    uid = ids[0]


    # Download run information
    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/efetch.fcgi"
    )

    params = {
        "db": "sra",
        "id": uid,
        "rettype": "runinfo",
        "retmode": "text",
    }

    r = ncbi_get(
        fetch_url,
        params=params,
    )

    r.raise_for_status()

    runinfo = pd.read_csv(
        io.StringIO(
            r.text
        )
    )

    if runinfo.empty:

        raise RuntimeError(
            f"Empty RunInfo for {srx}"
        )

    return runinfo


# ============================================================
# Build manifest
# ============================================================

records = []


for gsm, metadata in SAMPLES.items():

    print()
    print(
        f"Resolving {gsm} "
        f"({metadata['sample']})..."
    )

    srx = get_srx(
        gsm
    )

    print(
        f"  Experiment: {srx}"
    )

    runinfo = get_runinfo(
        srx
    )


    for _, row in runinfo.iterrows():

        record = {
            "GSM":
                gsm,

            "sample":
                metadata[
                    "sample"
                ],

            "state":
                metadata[
                    "state"
                ],

            "SRX":
                srx,

            "SRR":
                row.get(
                    "Run",
                    "",
                ),

            "spots":
                row.get(
                    "spots",
                    pd.NA,
                ),

            "bases":
                row.get(
                    "bases",
                    pd.NA,
                ),

            "avg_length":
                row.get(
                    "avgLength",
                    pd.NA,
                ),

            "size_MB":
                row.get(
                    "size_MB",
                    pd.NA,
                ),

            "library_layout":
                row.get(
                    "LibraryLayout",
                    "",
                ),

            "platform":
                row.get(
                    "Platform",
                    "",
                ),

            "model":
                row.get(
                    "Model",
                    "",
                ),

            "bioproject":
                row.get(
                    "BioProject",
                    "",
                ),

            "biosample":
                row.get(
                    "BioSample",
                    "",
                ),
        }

        records.append(
            record
        )


manifest = pd.DataFrame(
    records
)


# ============================================================
# Save
# ============================================================

manifest.to_csv(
    OUT_DIR
    / "GSE299861_PAO1_sra_manifest.csv",
    index=False,
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print("SRA MANIFEST")
print("=" * 100)

display_cols = [
    "sample",
    "state",
    "GSM",
    "SRX",
    "SRR",
    "library_layout",
    "spots",
    "bases",
    "size_MB",
]

print(
    manifest[
        display_cols
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("TOTAL SIZE")
print("=" * 100)

size = pd.to_numeric(
    manifest[
        "size_MB"
    ],
    errors="coerce",
)

print(
    f"Total SRA size: "
    f"{size.sum():,.1f} MB"
)

print(
    f"≈ {size.sum() / 1024:.2f} GB"
)


print()
print("=" * 100)
print("AUDIT 24 COMPLETE")
print("=" * 100)

print(
    f"Saved manifest to: "
    f"{OUT_DIR / 'GSE299861_PAO1_sra_manifest.csv'}"
)

print()
