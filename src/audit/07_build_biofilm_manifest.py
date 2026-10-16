from pathlib import Path
import pandas as pd


OUT = Path("data/metadata/biofilm_sample_manifest.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


records = []


def add_sample(
    study,
    experimental_family,
    sample,
    state,
    genotype="WT",
    include=True,
    exclusion_reason="",
):
    records.append(
        {
            "study": study,
            "experimental_family": experimental_family,
            "sample": sample,
            "state": state,
            "genotype": genotype,
            "include_primary_analysis": include,
            "exclusion_reason": exclusion_reason,
        }
    )


# ============================================================
# GSE136111
# Clean WT biofilm vs planktonic experiment
# ============================================================

for sample in ["B_1", "B_2", "B_3"]:
    add_sample(
        study="GSE136111",
        experimental_family="GSE136111",
        sample=sample,
        state="biofilm",
    )

for sample in ["P_1", "P_2", "P_3"]:
    add_sample(
        study="GSE136111",
        experimental_family="GSE136111",
        sample=sample,
        state="planktonic",
    )


# ============================================================
# GSE203348
# Canonical copy of WT samples also present in GSE223663
# ============================================================

for sample in [
    "BioFWTBR1_S10",
    "BioFWTBR2_S11",
    "BioFWTBR3_S12",
]:
    add_sample(
        study="GSE203348",
        experimental_family="GSE203348_GSE223663",
        sample=sample,
        state="biofilm",
    )

for sample in [
    "PlankWTBR1_S1",
    "PlankWTBR2_S2",
    "PlankWTBR3_S3",
]:
    add_sample(
        study="GSE203348",
        experimental_family="GSE203348_GSE223663",
        sample=sample,
        state="planktonic",
    )


# ============================================================
# GSE223663
# WT samples are exact duplicates of GSE203348
# Retain them in manifest, but exclude from primary analysis
# ============================================================

for sample in [
    "BioFWTBR1_S10",
    "BioFWTBR2_S11",
    "BioFWTBR3_S12",
]:
    add_sample(
        study="GSE223663",
        experimental_family="GSE203348_GSE223663",
        sample=sample,
        state="biofilm",
        include=False,
        exclusion_reason=(
            "Exact duplicate of corresponding GSE203348 WT sample"
        ),
    )

for sample in [
    "PlankWTBR1_S1",
    "PlankWTBR2_S2",
    "PlankWTBR3_S3",
]:
    add_sample(
        study="GSE223663",
        experimental_family="GSE203348_GSE223663",
        sample=sample,
        state="planktonic",
        include=False,
        exclusion_reason=(
            "Exact duplicate of corresponding GSE203348 WT sample"
        ),
    )


# ============================================================
# GSE87213
# Use WT samples only for primary biofilm-state analysis
# ============================================================

for sample in [
    "WTB3.raw",
    "WTB4.raw",
    "WTB6.raw",
]:
    add_sample(
        study="GSE87213",
        experimental_family="GSE87213",
        sample=sample,
        state="biofilm",
    )

for sample in [
    "WTP4.raw",
    "WTP5.raw",
    "WTP6.raw",
]:
    add_sample(
        study="GSE87213",
        experimental_family="GSE87213",
        sample=sample,
        state="planktonic",
    )


# ============================================================
# Save
# ============================================================

df = pd.DataFrame(records)

df.to_csv(
    OUT,
    index=False,
)


primary = df[
    df["include_primary_analysis"]
].copy()


print()
print("=" * 80)
print("CF-STATE — Biofilm sample manifest")
print("=" * 80)

print(f"Manifest rows:          {len(df)}")
print(f"Primary samples:        {len(primary)}")
print(
    f"Independent families:  "
    f"{primary['experimental_family'].nunique()}"
)

print()

print("Primary samples by state:")
print(
    primary["state"]
    .value_counts()
    .to_string()
)

print()

print("Primary samples by study:")
print(
    primary.groupby(
        ["study", "state"]
    ).size().to_string()
)

print()

print("Excluded samples:")
excluded = df[
    ~df["include_primary_analysis"]
]

if excluded.empty:
    print("None")
else:
    print(
        excluded[
            [
                "study",
                "sample",
                "exclusion_reason",
            ]
        ].to_string(index=False)
    )

print()
print(f"Saved: {OUT}")
