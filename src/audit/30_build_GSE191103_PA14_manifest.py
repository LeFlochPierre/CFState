from pathlib import Path
import io
import re
import time

import pandas as pd
import requests


OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE191103"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


SAMPLES = {
    "GSM5739075": {
        "sample": "biofilm_PA14_1",
        "state": "biofilm",
    },
    "GSM5739076": {
        "sample": "biofilm_PA14_2",
        "state": "biofilm",
    },
    "GSM5739077": {
        "sample": "biofilm_PA14_3",
        "state": "biofilm",
    },
    "GSM5739108": {
        "sample": "planktonic_PA14_1",
        "state": "planktonic",
    },
    "GSM5739109": {
        "sample": "planktonic_PA14_2",
        "state": "planktonic",
    },
    "GSM5739110": {
        "sample": "planktonic_PA14_3",
        "state": "planktonic",
    },
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 30: "
    "build GSE191103 PA14 raw-data manifest"
)
print("=" * 100)


def safe_get(
    url,
    params=None,
    max_attempts=8,
):

    headers = {
        "User-Agent":
            "CF-State academic research"
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

        if response.status_code in [
            429,
            500,
            502,
            503,
            504,
        ]:

            wait_seconds = min(
                2 ** attempt,
                60,
            )

            print(
                "Temporary server error "
                + str(response.status_code)
                + ". Waiting "
                + str(wait_seconds)
                + " seconds."
            )

            time.sleep(
                wait_seconds
            )

            continue

        response.raise_for_status()

        time.sleep(
            0.5
        )

        return response

    raise RuntimeError(
        "Repeated web request failure."
    )


def get_srx(
    gsm,
):

    url = (
        "https://www.ncbi.nlm.nih.gov/geo/"
        "query/acc.cgi"
    )

    params = {
        "acc": gsm,
        "targ": "self",
        "form": "text",
        "view": "full",
    }

    response = safe_get(
        url,
        params=params,
    )

    matches = sorted(
        set(
            re.findall(
                r"SRX\d+",
                response.text,
            )
        )
    )

    if len(matches) == 0:
        raise RuntimeError(
            "No SRX found for "
            + gsm
        )

    return matches[0]


def get_runinfo(
    srx,
):

    search_url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/esearch.fcgi"
    )

    search_params = {
        "db": "sra",
        "term": srx + "[ACCN]",
        "retmode": "json",
    }

    response = safe_get(
        search_url,
        params=search_params,
    )

    ids = response.json()[
        "esearchresult"
    ][
        "idlist"
    ]

    if len(ids) == 0:
        raise RuntimeError(
            "No SRA UID for "
            + srx
        )

    fetch_url = (
        "https://eutils.ncbi.nlm.nih.gov/"
        "entrez/eutils/efetch.fcgi"
    )

    fetch_params = {
        "db": "sra",
        "id": ids[0],
        "rettype": "runinfo",
        "retmode": "text",
    }

    response = safe_get(
        fetch_url,
        params=fetch_params,
    )

    runinfo = pd.read_csv(
        io.StringIO(
            response.text
        )
    )

    if runinfo.empty:
        raise RuntimeError(
            "Empty RunInfo for "
            + srx
        )

    return runinfo


records = []


for gsm, info in SAMPLES.items():

    print()
    print(
        "Resolving "
        + gsm
        + " ("
        + info["sample"]
        + ")"
    )

    srx = get_srx(
        gsm
    )

    print(
        "  Experiment: "
        + srx
    )

    runinfo = get_runinfo(
        srx
    )

    for _, row in runinfo.iterrows():

        records.append(
            {
                "GSM":
                    gsm,

                "sample":
                    info["sample"],

                "state":
                    info["state"],

                "SRX":
                    srx,

                "SRR":
                    row.get(
                        "Run",
                        "",
                    ),

                "library_layout":
                    row.get(
                        "LibraryLayout",
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
            }
        )


manifest = pd.DataFrame(
    records
)


manifest.to_csv(
    OUT_DIR
    / "GSE191103_PA14_sra_manifest.csv",
    index=False,
)


print()
print("=" * 100)
print("SRA MANIFEST")
print("=" * 100)

display_columns = [
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
        display_columns
    ].to_string(
        index=False
    )
)


sizes = pd.to_numeric(
    manifest["size_MB"],
    errors="coerce",
)

total_mb = float(
    sizes.sum()
)

total_gb = (
    total_mb
    / 1024.0
)


print()
print("=" * 100)
print("TOTAL SIZE")
print("=" * 100)

print(
    "Total SRA size: "
    + format(
        total_mb,
        ",.1f",
    )
    + " MB"
)

print(
    "Approximately "
    + format(
        total_gb,
        ".2f",
    )
    + " GB"
)


print()
print("=" * 100)
print("AUDIT 30 COMPLETE")
print("=" * 100)

print(
    "Saved to: "
    + str(
        OUT_DIR
        / "GSE191103_PA14_sra_manifest.csv"
    )
)
