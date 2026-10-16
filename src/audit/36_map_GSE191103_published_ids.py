from pathlib import Path
import re

import pandas as pd


# ============================================================
# Paths
# ============================================================

PUBLISHED_FILE = Path(
    "data/raw/expanded_biofilm/GSE191103/"
    "GSE191103_ProcessedDataMatrix_biofilm_lasR.xlsx"
)

ORTHOLOGY_FILE = Path(
    "results/audit/expanded_biofilm/GSE191103/"
    "PAO1_PA14_clean_1to1_mapping.csv"
)

FROZEN_FILE = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE191103/"
    "published_id_mapping"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print("CF-STATE — Audit 36: inspect GSE191103 published gene identifiers")
print("=" * 100)


# ============================================================
# Load data
# ============================================================

published = pd.read_excel(
    PUBLISHED_FILE,
    sheet_name="ProcessedDataMatrix",
)

orthology = pd.read_csv(
    ORTHOLOGY_FILE
)

frozen = pd.read_csv(
    FROZEN_FILE
)


if "Locus_ID" not in published.columns:
    raise RuntimeError(
        "Published matrix has no Locus_ID column."
    )


if "canonical_pao1" not in orthology.columns:
    raise RuntimeError(
        "Orthology file missing canonical_pao1."
    )


if "pa14_locus_tag" not in orthology.columns:
    raise RuntimeError(
        "Orthology file missing pa14_locus_tag."
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


frozen_genes = set(
    frozen[frozen_gene_col]
    .astype(str)
)


# ============================================================
# PA14 -> PAO1 lookup
# ============================================================

pa14_to_pao1 = dict(
    zip(
        orthology["pa14_locus_tag"].astype(str),
        orthology["canonical_pao1"].astype(str),
    )
)


# ============================================================
# Classify identifier formats
# ============================================================

def classify_id(value):

    value = str(value).strip()

    if re.fullmatch(r"PA\d{4}", value):
        return "exact_PAO1"

    if re.fullmatch(r"PA\d{4}\.\d+", value):
        return "PAO1_decimal_suffix"

    if re.fullmatch(r"PA\d{4}-PA\d{4}", value):
        return "PAO1_range"

    if re.fullmatch(r"PA14_\d+", value):
        return "PA14_old_locus"

    if re.fullmatch(r"PA14_RS\d+", value):
        return "PA14_current_locus"

    return "other"


published["id_type"] = (
    published["Locus_ID"]
    .astype(str)
    .map(classify_id)
)


type_counts = (
    published["id_type"]
    .value_counts()
    .rename_axis("id_type")
    .reset_index(name="n_rows")
)


print()
print("=" * 100)
print("IDENTIFIER TYPES")
print("=" * 100)

print(
    type_counts.to_string(
        index=False
    )
)


# ============================================================
# Show examples
# ============================================================

print()
print("=" * 100)
print("EXAMPLES")
print("=" * 100)


for id_type in type_counts["id_type"]:

    examples = (
        published.loc[
            published["id_type"] == id_type,
            "Locus_ID",
        ]
        .astype(str)
        .head(10)
        .tolist()
    )

    print()
    print(id_type + ":")

    for value in examples:
        print("  " + value)


# ============================================================
# Build only DEFENSIBLE mappings
#
# 1. Exact PAO1 IDs map directly.
# 2. Exact historical PA14 IDs map through strict orthology.
#
# We intentionally DO NOT collapse:
#   PAxxxx.1
#   PAxxxx-PAxxxx
# because these may represent distinct transcript features.
# ============================================================

mapping_records = []


for row_index, row in published.iterrows():

    locus_id = str(
        row["Locus_ID"]
    ).strip()

    id_type = str(
        row["id_type"]
    )

    canonical = None
    source = None


    if id_type == "exact_PAO1":

        canonical = locus_id
        source = "direct_PAO1"


    elif id_type == "PA14_old_locus":

        if locus_id in pa14_to_pao1:

            canonical = pa14_to_pao1[
                locus_id
            ]

            source = "strict_PA14_to_PAO1"


    if canonical is not None:

        mapping_records.append(
            {
                "published_row":
                    int(row_index),

                "Locus_ID":
                    locus_id,

                "id_type":
                    id_type,

                "canonical_pao1":
                    canonical,

                "mapping_source":
                    source,

                "is_frozen_gene":
                    canonical in frozen_genes,
            }
        )


mapped = pd.DataFrame(
    mapping_records
)


if mapped.empty:
    raise RuntimeError(
        "No defensible published identifiers could be mapped."
    )


# ============================================================
# Check duplicate mappings
# ============================================================

duplicate_mask = mapped[
    "canonical_pao1"
].duplicated(
    keep=False
)


duplicate_mappings = (
    mapped.loc[
        duplicate_mask
    ]
    .sort_values(
        "canonical_pao1"
    )
)


mapped_unique = (
    mapped.loc[
        ~duplicate_mask
    ]
    .copy()
)


mapped_frozen = set(
    mapped_unique.loc[
        mapped_unique[
            "is_frozen_gene"
        ],
        "canonical_pao1",
    ]
)


missing_frozen = (
    frozen_genes
    - mapped_frozen
)


# ============================================================
# Summary
# ============================================================

print()
print("=" * 100)
print("DEFENSIBLE MAPPING SUMMARY")
print("=" * 100)

print(
    "Published rows: "
    + str(len(published))
)

print(
    "Rows with defensible PAO1 mapping: "
    + str(len(mapped))
)

print(
    "Rows involved in duplicate canonical mappings: "
    + str(len(duplicate_mappings))
)

print(
    "Unique mapped canonical genes: "
    + str(
        mapped_unique[
            "canonical_pao1"
        ].nunique()
    )
)

print()

print(
    "Frozen genes expected: "
    + str(len(frozen_genes))
)

print(
    "Frozen genes mapped unambiguously: "
    + str(len(mapped_frozen))
)

print(
    "Frozen genes missing: "
    + str(len(missing_frozen))
)


# ============================================================
# Save
# ============================================================

type_counts.to_csv(
    OUT_DIR
    / "published_identifier_type_counts.csv",
    index=False,
)


mapped.to_csv(
    OUT_DIR
    / "published_to_PAO1_all_defensible_mappings.csv",
    index=False,
)


duplicate_mappings.to_csv(
    OUT_DIR
    / "duplicate_canonical_mappings.csv",
    index=False,
)


mapped_unique.to_csv(
    OUT_DIR
    / "published_to_PAO1_unique_mappings.csv",
    index=False,
)


with open(
    OUT_DIR
    / "missing_frozen_genes.txt",
    "w",
) as handle:

    for gene in sorted(
        missing_frozen
    ):
        handle.write(
            gene + "\n"
        )


print()
print("=" * 100)
print("AUDIT 36 COMPLETE")
print("=" * 100)

print(
    "Outputs: "
    + str(OUT_DIR)
)

print()
