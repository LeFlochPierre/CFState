from pathlib import Path
import pandas as pd


COUNT_FILE = Path(
    "results/audit/expanded_biofilm/SRP009257/"
    "reprocessing/counts/counts_s1.txt"
)

MAPPING_FILE = Path(
    "results/audit/expanded_biofilm/GSE191103/"
    "PAO1_PA14_clean_1to1_mapping.csv"
)

FROZEN_FILE = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
)

SIGNATURE_FILE = Path(
    "results/audit/core_biofilm_signature/"
    "core_biofilm_signature_fdr05.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/SRP009257/"
    "finalization"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load
# ============================================================

counts = pd.read_csv(
    COUNT_FILE,
    sep="\t",
    comment="#",
)

mapping = pd.read_csv(
    MAPPING_FILE
)

frozen = pd.read_csv(
    FROZEN_FILE
)

signature = pd.read_csv(
    SIGNATURE_FILE
)


# ============================================================
# Checks
# ============================================================

if "Geneid" not in counts.columns:
    raise RuntimeError(
        "Geneid missing from featureCounts table."
    )


for column in [
    "pa14_locus_tag",
    "canonical_pao1",
]:
    if column not in mapping.columns:
        raise RuntimeError(
            "Missing mapping column: "
            + column
        )


if "canonical_pao1" not in signature.columns:
    raise RuntimeError(
        "Signature file missing canonical_pao1."
    )


# ============================================================
# Frozen gene column
# ============================================================

frozen_gene_col = None


for candidate in [
    "canonical_pao1",
    "GeneID",
    "gene",
    "locus_tag",
]:

    if candidate in frozen.columns:
        frozen_gene_col = candidate
        break


if frozen_gene_col is None:

    if frozen.shape[1] == 1:
        frozen_gene_col = frozen.columns[0]

    else:
        raise RuntimeError(
            "Could not determine frozen gene column."
        )


frozen_genes = (
    frozen[
        frozen_gene_col
    ]
    .astype(str)
    .tolist()
)


if len(frozen_genes) != 4858:
    raise RuntimeError(
        "Expected 4,858 frozen genes."
    )


# ============================================================
# Frozen PAO1 -> strict PA14 mapping
# ============================================================

if mapping[
    "canonical_pao1"
].duplicated().any():

    raise RuntimeError(
        "Duplicate PAO1 genes in strict mapping."
    )


mapping_by_pao1 = (
    mapping[
        [
            "canonical_pao1",
            "pa14_locus_tag",
        ]
    ]
    .copy()
)


frozen_map = pd.DataFrame(
    {
        "canonical_pao1":
            frozen_genes
    }
).merge(
    mapping_by_pao1,
    on="canonical_pao1",
    how="left",
    validate="one_to_one",
)


n_without_ortholog = int(
    frozen_map[
        "pa14_locus_tag"
    ]
    .isna()
    .sum()
)


# ============================================================
# Which PA14 tags actually exist in our featureCounts annotation?
# ============================================================

feature_ids = set(
    counts[
        "Geneid"
    ]
    .astype(str)
)


frozen_map[
    "present_in_featureCounts_annotation"
] = (
    frozen_map[
        "pa14_locus_tag"
    ]
    .astype(str)
    .isin(
        feature_ids
    )
)


missing = (
    frozen_map.loc[
        ~frozen_map[
            "present_in_featureCounts_annotation"
        ]
    ]
    .copy()
)


# ============================================================
# Check frozen 251 signature
# ============================================================

signature_genes = set(
    signature[
        "canonical_pao1"
    ]
    .astype(str)
)


missing[
    "in_frozen_251_signature"
] = (
    missing[
        "canonical_pao1"
    ]
    .isin(
        signature_genes
    )
)


signature_missing = (
    missing.loc[
        missing[
            "in_frozen_251_signature"
        ]
    ]
    .copy()
)


# ============================================================
# Print
# ============================================================

print()
print("=" * 100)
print(
    "CF-STATE — Audit 44a: "
    "diagnose structurally missing SRP009257 frozen genes"
)
print("=" * 100)


print(
    "Frozen genes: "
    + str(
        len(
            frozen_map
        )
    )
)

print(
    "Frozen genes without strict PA14 ortholog: "
    + str(
        n_without_ortholog
    )
)

print(
    "Frozen PA14 orthologs present in "
    "featureCounts annotation: "
    + str(
        int(
            frozen_map[
                "present_in_featureCounts_annotation"
            ].sum()
        )
    )
)

print(
    "Frozen PA14 orthologs absent from "
    "featureCounts annotation: "
    + str(
        len(
            missing
        )
    )
)

print(
    "Missing genes that belong to "
    "frozen 251 signature: "
    + str(
        len(
            signature_missing
        )
    )
)


print()
print("=" * 100)
print("MISSING FROZEN GENES")
print("=" * 100)


if missing.empty:

    print("None.")

else:

    print(
        missing.to_string(
            index=False
        )
    )


print()
print("=" * 100)
print("MISSING FROZEN 251-SIGNATURE GENES")
print("=" * 100)


if signature_missing.empty:

    print("None.")

else:

    print(
        signature_missing.to_string(
            index=False
        )
    )


# ============================================================
# Save
# ============================================================

missing.to_csv(
    OUT_DIR
    / "SRP009257_missing_frozen_genes.csv",
    index=False,
)


signature_missing.to_csv(
    OUT_DIR
    / "SRP009257_missing_frozen_signature_genes.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 44a COMPLETE")
print("=" * 100)
print()