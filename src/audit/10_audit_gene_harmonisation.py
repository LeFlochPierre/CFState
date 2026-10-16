from pathlib import Path
from collections import defaultdict
import re

import pandas as pd
from Bio import SeqIO


ROOT = Path("external/EPathDash/GEO_Datasets/Pseudomonas")
COUNT_DIR = ROOT / "Count_Tables"

MAP_DIR = Path("references/mappings")

ORTHOLOG_FILE = MAP_DIR / "pao1_pa14_rbbh.csv"
PAO1_GENBANK = MAP_DIR / "NC_002516.2.gb"

OUT_DIR = Path("results/audit/gene_harmonisation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Load PAO1 annotation
# ============================================================

print()
print("=" * 100)
print("CF-STATE — Gene harmonisation audit")
print("=" * 100)

record = SeqIO.read(PAO1_GENBANK, "genbank")

symbol_to_loci = defaultdict(set)
pao1_loci_from_reference = set()

for feature in record.features:

    qualifiers = feature.qualifiers

    locus_tags = qualifiers.get("locus_tag", [])

    if not locus_tags:
        continue

    locus = locus_tags[0].strip()

    if re.fullmatch(r"PA\d{4}", locus):
        pao1_loci_from_reference.add(locus)

    for gene_name in qualifiers.get("gene", []):
        gene_name = gene_name.strip()

        if gene_name:
            symbol_to_loci[gene_name].add(locus)


unique_symbol_map = {
    symbol: next(iter(loci))
    for symbol, loci in symbol_to_loci.items()
    if len(loci) == 1
}

ambiguous_symbol_map = {
    symbol: loci
    for symbol, loci in symbol_to_loci.items()
    if len(loci) > 1
}


print()
print("PAO1 REFERENCE ANNOTATION")
print("-" * 100)

print(
    f"PAO1 locus tags in GenBank:       "
    f"{len(pao1_loci_from_reference)}"
)

print(
    f"Unique gene-symbol mappings:      "
    f"{len(unique_symbol_map)}"
)

print(
    f"Ambiguous gene symbols:           "
    f"{len(ambiguous_symbol_map)}"
)


# ============================================================
# 2. Load PAO1 <-> PA14 ortholog table
# ============================================================

orth = pd.read_csv(ORTHOLOG_FILE)

pao1_col = "Locus Tag (Query)"
pa14_col = "Locus Tag (Hit)"

orth[pao1_col] = (
    orth[pao1_col]
    .astype(str)
    .str.strip()
)

orth[pa14_col] = (
    orth[pa14_col]
    .astype(str)
    .str.strip()
)


print()
print("=" * 100)
print("PAO1 ↔ PA14 ORTHOLOG TABLE")
print("=" * 100)

print(f"Rows:                          {len(orth)}")
print(
    f"Unique PAO1 loci:              "
    f"{orth[pao1_col].nunique()}"
)
print(
    f"Unique PA14 loci:              "
    f"{orth[pa14_col].nunique()}"
)


pao1_counts = orth[pao1_col].value_counts()
pa14_counts = orth[pa14_col].value_counts()

repeated_pao1 = set(
    pao1_counts[pao1_counts > 1].index
)

repeated_pa14 = set(
    pa14_counts[pa14_counts > 1].index
)

print(
    f"PAO1 loci occurring >1 time:   "
    f"{len(repeated_pao1)}"
)

print(
    f"PA14 loci occurring >1 time:   "
    f"{len(repeated_pa14)}"
)


# Conservative one-to-one mapping:
# PAO1 locus appears once AND PA14 locus appears once.

orth["clean_1to1"] = (
    ~orth[pao1_col].isin(repeated_pao1)
    &
    ~orth[pa14_col].isin(repeated_pa14)
)

clean_orth = orth[
    orth["clean_1to1"]
].copy()

pa14_to_pao1 = dict(
    zip(
        clean_orth[pa14_col],
        clean_orth[pao1_col],
    )
)

print(
    f"Clean 1-to-1 ortholog pairs:   "
    f"{len(clean_orth)}"
)


# Save ortholog audit
orth.to_csv(
    OUT_DIR / "pao1_pa14_ortholog_audit.csv",
    index=False,
)


# ============================================================
# 3. GSE136111
# ============================================================

g136 = pd.read_csv(
    COUNT_DIR / "GSE136111_counts.csv",
    encoding="utf-8-sig",
)

ids136 = (
    g136.iloc[:, 0]
    .astype(str)
    .str.strip()
)

records136 = []

for identifier in ids136:

    # UTR rows
    if re.fullmatch(
        r"[35]'UTR_\d+",
        identifier
    ):
        status = "excluded_UTR"
        pao1_id = ""

    # Already PAO1 locus tag
    elif re.fullmatch(
        r"PA\d{4}",
        identifier
    ):

        if identifier in pao1_loci_from_reference:
            status = "direct_PAO1"
            pao1_id = identifier
        else:
            status = "PAO1_not_in_reference"
            pao1_id = ""

    # Gene symbol with unique reference mapping
    elif identifier in unique_symbol_map:
        status = "symbol_mapped"
        pao1_id = unique_symbol_map[
            identifier
        ]

    # Ambiguous symbol
    elif identifier in ambiguous_symbol_map:
        status = "ambiguous_symbol"
        pao1_id = ""

    else:
        status = "unmapped"
        pao1_id = ""

    records136.append(
        {
            "original_id": identifier,
            "status": status,
            "canonical_pao1": pao1_id,
        }
    )


map136 = pd.DataFrame(records136)

map136.to_csv(
    OUT_DIR / "GSE136111_mapping_audit.csv",
    index=False,
)


print()
print("=" * 100)
print("GSE136111")
print("=" * 100)

print(
    map136["status"]
    .value_counts()
    .to_string()
)

mapped136 = set(
    map136.loc[
        map136["canonical_pao1"] != "",
        "canonical_pao1",
    ]
)


# Check whether multiple original features collapse
# onto the same canonical PAO1 gene.

mapped136_nonempty = map136[
    map136["canonical_pao1"] != ""
]

collapse_counts = (
    mapped136_nonempty[
        "canonical_pao1"
    ]
    .value_counts()
)

collapsed136 = collapse_counts[
    collapse_counts > 1
]

print()
print(
    f"Unique canonical PAO1 genes:   "
    f"{len(mapped136)}"
)

print(
    f"PAO1 genes receiving >1 row:   "
    f"{len(collapsed136)}"
)

if len(collapsed136):

    print()
    print("Examples of duplicate mappings:")

    for locus in collapsed136.head(20).index:

        source_ids = map136.loc[
            map136["canonical_pao1"] == locus,
            "original_id",
        ].tolist()

        print(
            f"  {locus}: "
            + ", ".join(source_ids)
        )


# ============================================================
# 4. GSE203348
# ============================================================

g203 = pd.read_csv(
    COUNT_DIR / "GSE203348_counts.csv",
    encoding="utf-8-sig",
)

ids203 = (
    g203.iloc[:, 0]
    .astype(str)
    .str.strip()
)

valid203 = ids203[
    ids203.str.fullmatch(
        r"PA\d{4}",
        case=False,
    )
]

mapped203 = set(
    valid203[
        valid203.isin(
            pao1_loci_from_reference
        )
    ]
)


print()
print("=" * 100)
print("GSE203348")
print("=" * 100)

print(
    f"Feature rows:                 "
    f"{len(ids203)}"
)

print(
    f"PAO1-formatted IDs:           "
    f"{len(valid203)}"
)

print(
    f"Present in PAO1 reference:    "
    f"{len(mapped203)}"
)

other203 = ids203[
    ~ids203.isin(mapped203)
]

print(
    f"Other/unmapped IDs:           "
    f"{len(other203)}"
)

if len(other203):

    print()
    print("Other IDs:")

    print(
        other203.to_string(index=False)
    )


# ============================================================
# 5. GSE87213
# ============================================================

g872 = pd.read_csv(
    COUNT_DIR / "GSE87213_counts.csv",
    encoding="utf-8-sig",
)

ids872 = (
    g872.iloc[:, 0]
    .astype(str)
    .str.strip()
)

records872 = []

for identifier in ids872:

    if re.fullmatch(
        r"PA14_\d+",
        identifier
    ):

        if identifier in pa14_to_pao1:

            status = "clean_1to1_ortholog"
            pao1_id = pa14_to_pao1[
                identifier
            ]

        else:

            status = "PA14_without_clean_1to1"
            pao1_id = ""

    elif identifier.startswith("EBG"):

        status = "excluded_EBG"
        pao1_id = ""

    else:

        status = "other_unmapped"
        pao1_id = ""

    records872.append(
        {
            "original_id": identifier,
            "status": status,
            "canonical_pao1": pao1_id,
        }
    )


map872 = pd.DataFrame(records872)

map872.to_csv(
    OUT_DIR / "GSE87213_mapping_audit.csv",
    index=False,
)


print()
print("=" * 100)
print("GSE87213")
print("=" * 100)

print(
    map872["status"]
    .value_counts()
    .to_string()
)

mapped872 = set(
    map872.loc[
        map872["canonical_pao1"] != "",
        "canonical_pao1",
    ]
)

print()
print(
    f"Unique canonical PAO1 genes:  "
    f"{len(mapped872)}"
)


# ============================================================
# 6. Common universe
# ============================================================

common = (
    mapped136
    &
    mapped203
    &
    mapped872
)


print()
print("=" * 100)
print("COMMON CANONICAL PAO1 GENE UNIVERSE")
print("=" * 100)

print(
    f"GSE136111 mapped PAO1 genes:  "
    f"{len(mapped136)}"
)

print(
    f"GSE203348 mapped PAO1 genes:  "
    f"{len(mapped203)}"
)

print(
    f"GSE87213 mapped PAO1 genes:   "
    f"{len(mapped872)}"
)

print()
print(
    f"COMMON ACROSS ALL 3:          "
    f"{len(common)}"
)


common_df = pd.DataFrame(
    {
        "canonical_pao1": sorted(common)
    }
)

common_df.to_csv(
    OUT_DIR / "common_pao1_gene_universe.csv",
    index=False,
)


# ============================================================
# 7. Pairwise overlap
# ============================================================

print()
print("=" * 100)
print("PAIRWISE OVERLAPS")
print("=" * 100)

print(
    f"GSE136111 ∩ GSE203348: "
    f"{len(mapped136 & mapped203)}"
)

print(
    f"GSE136111 ∩ GSE87213:  "
    f"{len(mapped136 & mapped872)}"
)

print(
    f"GSE203348 ∩ GSE87213:  "
    f"{len(mapped203 & mapped872)}"
)


print()
print("=" * 100)
print("Saved audit files to:")
print(OUT_DIR)
print("=" * 100)
