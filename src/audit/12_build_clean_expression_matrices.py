from pathlib import Path
import pandas as pd
import numpy as np


COUNT_DIR = Path(
    "external/EPathDash/GEO_Datasets/Pseudomonas/Count_Tables"
)

HARM_DIR = Path(
    "results/audit/gene_harmonisation"
)

QC_DIR = Path(
    "results/audit/expression_qc"
)

OUT_DIR = Path(
    "data/processed/biofilm"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


PRIMARY_SAMPLES = {
    "GSE136111": {
        "B_1": "biofilm",
        "B_2": "biofilm",
        "B_3": "biofilm",
        "P_1": "planktonic",
        "P_2": "planktonic",
        "P_3": "planktonic",
    },

    "GSE203348": {
        "BioFWTBR1_S10": "biofilm",
        "BioFWTBR2_S11": "biofilm",
        "BioFWTBR3_S12": "biofilm",
        "PlankWTBR1_S1": "planktonic",
        "PlankWTBR2_S2": "planktonic",
        "PlankWTBR3_S3": "planktonic",
    },

    "GSE87213": {
        "WTB3.raw": "biofilm",
        "WTB4.raw": "biofilm",
        "WTB6.raw": "biofilm",
        "WTP4.raw": "planktonic",
        "WTP5.raw": "planktonic",
        "WTP6.raw": "planktonic",
    },
}


print()
print("=" * 100)
print("CF-STATE — Audit 12: build clean harmonised expression matrices")
print("=" * 100)


# ============================================================
# Original common universe
# ============================================================

common = pd.read_csv(
    HARM_DIR / "common_pao1_gene_universe.csv"
)

original_universe = set(
    common["canonical_pao1"]
    .astype(str)
    .str.strip()
)

print()
print(
    f"Audit-10 common universe: "
    f"{len(original_universe)} genes"
)


# ============================================================
# Remove ambiguous duplicated mappings from GSE136111
# ============================================================

dup_qc = pd.read_csv(
    QC_DIR / "duplicate_canonical_mapping_qc.csv"
)

bad_duplicate_genes = set(
    dup_qc.loc[
        dup_qc["expression_identical"] == False,
        "canonical_pao1",
    ]
    .astype(str)
    .str.strip()
)

print()
print("Genes removed due to non-identical duplicate mapping:")

for x in sorted(bad_duplicate_genes):
    print(f"  {x}")


# ============================================================
# Remove genes with structured missingness in GSE87213
# ============================================================

missing_file = (
    QC_DIR
    / "GSE87213_common_genes_with_missing_values.csv"
)

missing_df = pd.read_csv(
    missing_file
)

bad_missing_genes = set(
    missing_df["canonical_pao1"]
    .astype(str)
    .str.strip()
)

print()
print("Genes removed due to structured GSE87213 missingness:")

for x in sorted(bad_missing_genes):
    print(f"  {x}")


# ============================================================
# Final clean universe
# ============================================================

excluded = (
    bad_duplicate_genes
    | bad_missing_genes
)

clean_universe = sorted(
    original_universe
    - excluded
)

print()
print("=" * 100)
print("CLEAN GENE UNIVERSE")
print("=" * 100)

print(
    f"Original:       {len(original_universe)}"
)

print(
    f"Excluded:       {len(excluded)}"
)

print(
    f"Final clean:    {len(clean_universe)}"
)

pd.DataFrame(
    {
        "canonical_pao1": clean_universe
    }
).to_csv(
    OUT_DIR / "clean_gene_universe.csv",
    index=False,
)


excluded_records = []

for gene in sorted(excluded):

    reasons = []

    if gene in bad_duplicate_genes:
        reasons.append(
            "non_identical_duplicate_mapping_GSE136111"
        )

    if gene in bad_missing_genes:
        reasons.append(
            "structured_missingness_GSE87213"
        )

    excluded_records.append(
        {
            "canonical_pao1": gene,
            "reason": "|".join(reasons),
        }
    )

pd.DataFrame(
    excluded_records
).to_csv(
    OUT_DIR / "excluded_genes.csv",
    index=False,
)


# ============================================================
# Build per-study canonical mappings
# ============================================================

def load_study(study):

    raw = pd.read_csv(
        COUNT_DIR / f"{study}_counts.csv",
        encoding="utf-8-sig",
    )

    raw.columns = [
        str(c)
        .replace("\ufeff", "")
        .strip()
        for c in raw.columns
    ]

    source_id = (
        raw.iloc[:, 0]
        .astype(str)
        .str.strip()
    )

    sample_cols = list(
        PRIMARY_SAMPLES[study].keys()
    )

    numeric = raw[
        sample_cols
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if study == "GSE136111":

        mapping = pd.read_csv(
            HARM_DIR
            / "GSE136111_mapping_audit.csv"
        )

        canonical = (
            mapping["canonical_pao1"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        mapping_source = (
            mapping["original_id"]
            .astype(str)
            .str.strip()
        )

        if not source_id.reset_index(
            drop=True
        ).equals(
            mapping_source.reset_index(
                drop=True
            )
        ):
            raise RuntimeError(
                "GSE136111 mapping order mismatch."
            )

    elif study == "GSE203348":

        canonical = source_id.copy()

    elif study == "GSE87213":

        mapping = pd.read_csv(
            HARM_DIR
            / "GSE87213_mapping_audit.csv"
        )

        canonical = (
            mapping["canonical_pao1"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        mapping_source = (
            mapping["original_id"]
            .astype(str)
            .str.strip()
        )

        if not source_id.reset_index(
            drop=True
        ).equals(
            mapping_source.reset_index(
                drop=True
            )
        ):
            raise RuntimeError(
                "GSE87213 mapping order mismatch."
            )

    else:
        raise ValueError(study)

    mapped = pd.DataFrame(
        {
            "source_id": source_id,
            "canonical_pao1": canonical,
        }
    )

    mapped = pd.concat(
        [
            mapped.reset_index(drop=True),
            numeric.reset_index(drop=True),
        ],
        axis=1,
    )

    mapped = mapped[
        mapped["canonical_pao1"].isin(
            clean_universe
        )
    ].copy()

    return mapped


# ============================================================
# Construct matrices
# ============================================================

metadata_records = []
combined_rows = []

for study in PRIMARY_SAMPLES:

    mapped = load_study(
        study
    )

    sample_cols = list(
        PRIMARY_SAMPLES[study].keys()
    )

    print()
    print("=" * 100)
    print(study)
    print("=" * 100)

    print(
        f"Rows after cleaning:       "
        f"{len(mapped)}"
    )

    print(
        f"Unique canonical genes:    "
        f"{mapped['canonical_pao1'].nunique()}"
    )

    if mapped["canonical_pao1"].duplicated().any():

        duplicates = mapped.loc[
            mapped["canonical_pao1"].duplicated(
                keep=False
            ),
            "canonical_pao1",
        ].unique()

        raise RuntimeError(
            f"{study}: duplicate genes remain: "
            f"{duplicates}"
        )

    if set(
        mapped["canonical_pao1"]
    ) != set(
        clean_universe
    ):

        missing = (
            set(clean_universe)
            - set(mapped["canonical_pao1"])
        )

        raise RuntimeError(
            f"{study}: missing clean genes: "
            f"{sorted(missing)[:20]}"
        )

    expression = (
        mapped
        .set_index(
            "canonical_pao1"
        )[sample_cols]
        .loc[
            clean_universe
        ]
    )

    # genes x samples
    gene_by_sample = expression.copy()

    gene_by_sample.to_csv(
        OUT_DIR
        / f"{study}_harmonised_unscaled_gene_by_sample.csv"
    )

    # samples x genes
    sample_by_gene = expression.T

    sample_by_gene.to_csv(
        OUT_DIR
        / f"{study}_harmonised_unscaled_sample_by_gene.csv"
    )

    n_missing = int(
        sample_by_gene
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Matrix shape:             "
        f"{sample_by_gene.shape}"
    )

    print(
        f"Missing values:           "
        f"{n_missing}"
    )

    if n_missing != 0:
        raise RuntimeError(
            f"{study}: missing values remain."
        )

    for sample in sample_by_gene.index:

        state = PRIMARY_SAMPLES[
            study
        ][sample]

        metadata_records.append(
            {
                "study": study,
                "sample": sample,
                "state": state,
            }
        )

        row = sample_by_gene.loc[
            sample
        ].to_dict()

        row.update(
            {
                "study": study,
                "sample": sample,
                "state": state,
            }
        )

        combined_rows.append(
            row
        )


# ============================================================
# Combined matrix
# ============================================================

metadata = pd.DataFrame(
    metadata_records
)

metadata.to_csv(
    OUT_DIR
    / "sample_metadata.csv",
    index=False,
)


combined = pd.DataFrame(
    combined_rows
)

combined = combined[
    [
        "study",
        "sample",
        "state",
    ]
    + clean_universe
]


combined.to_csv(
    OUT_DIR
    / "all_studies_harmonised_unscaled.csv",
    index=False,
)


print()
print("=" * 100)
print("FINAL DATASET")
print("=" * 100)

print(
    f"Samples:             "
    f"{len(metadata)}"
)

print(
    f"Genes:               "
    f"{len(clean_universe)}"
)

print()
print("Samples by study/state:")

print(
    metadata.groupby(
        [
            "study",
            "state",
        ]
    )
    .size()
    .to_string()
)

print()
print(
    f"Saved to: {OUT_DIR}"
)

print()
print(
    "IMPORTANT: these matrices are harmonised "
    "but NOT normalised across studies."
)
print()
