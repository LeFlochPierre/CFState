from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from io import StringIO
import pandas as pd

out = Path("results/phase7/audit_07f_GSE139943_run_preflight")

if out.exists():
    raise RuntimeError("Audit 07f already exists. Refusing overwrite.")

experiments = {
    2: "SRX7101180",
    33: "SRX7101182",
    35: "SRX7101183",
    37: "SRX7101184",
    40: "SRX7101185",
}

fields = [
    "run_accession",
    "experiment_accession",
    "sample_accession",
    "library_strategy",
    "library_source",
    "library_layout",
    "read_count",
    "base_count",
    "fastq_ftp",
    "fastq_md5",
    "fastq_bytes",
]

all_rows = []

for day, experiment in experiments.items():
    params = urlencode({
        "accession": experiment,
        "result": "read_run",
        "fields": ",".join(fields),
        "format": "tsv",
    })

    url = (
        "https://www.ebi.ac.uk/ena/portal/api/filereport?"
        + params
    )

    print(f"Checking day {day}: {experiment}", flush=True)

    request = Request(
        url,
        headers={"User-Agent": "CFState-research/1.0"}
    )

    with urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")

    df = pd.read_csv(StringIO(body), sep="\t", dtype=str)
    df = df.fillna("")

    if df.empty:
        raise RuntimeError(f"No ENA runs for {experiment}")

    if set(df["experiment_accession"]) != {experiment}:
        raise RuntimeError(f"Experiment mismatch: {experiment}")

    df.insert(0, "CF146_day", day)
    all_rows.append(df)

result = pd.concat(all_rows, ignore_index=True)

if result["run_accession"].duplicated().any():
    raise RuntimeError("Duplicate run accessions")

def fastq_gb(value):
    if not value:
        return 0.0
    return sum(int(x) for x in value.split(";")) / 1e9

result["FASTQ_GB"] = result["fastq_bytes"].apply(fastq_gb)
result["FASTQ_available"] = result["fastq_ftp"].ne("")

out.mkdir(parents=True)

result.to_csv(
    out / "FROZEN_GSE139943_candidate_run_metadata.csv",
    index=False
)

print("\n=== GSE139943 RAW RNA-SEQ PREFLIGHT ===")

cols = [
    "CF146_day",
    "run_accession",
    "library_strategy",
    "library_source",
    "library_layout",
    "read_count",
    "FASTQ_GB",
    "FASTQ_available",
]

print(result[cols].to_string(index=False))

print("\nTotal compressed FASTQ GB:",
      round(result["FASTQ_GB"].sum(), 3))

status = "CFSTATE_PHASE7_07F_RUN_PREFLIGHT_COMPLETE_REVIEW_REQUIRED"

(out / "AUDIT_07F_STATUS.txt").write_text(status + "\n")
print("\nSTATUS:", status)
