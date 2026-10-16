from pathlib import Path

import pandas as pd
from Bio import SeqIO


INPUT_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "robust_consensus_reversal/"
    "Audit60_gene_consensus_and_GSE299861_effect.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "reversal_biology"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 61: "
    "annotate biological programmes driving GSE299861 reversal"
)
print("=" * 100)


# ============================================================
# Find canonical PAO1 GenBank file
# ============================================================

candidates = []


for pattern in [
    "*.gb",
    "*.gbk",
    "*.gbff",
]:

    candidates.extend(
        Path("references").rglob(
            pattern
        )
    )


preferred = [
    path
    for path in candidates
    if "NC_002516.2" in path.name
]


if preferred:

    genbank_file = sorted(
        preferred
    )[0]

elif candidates:

    genbank_file = sorted(
        candidates
    )[0]

else:

    raise RuntimeError(
        "No GenBank file found under references/."
    )


print()
print(
    "Canonical annotation:"
)

print(
    genbank_file
)


# ============================================================
# Parse canonical PAO1 annotation
# ============================================================

record = SeqIO.read(
    genbank_file,
    "genbank",
)


annotation_records = []


for feature in record.features:

    if feature.type != "gene":
        continue


    locus_tags = feature.qualifiers.get(
        "locus_tag",
        []
    )


    if not locus_tags:
        continue


    locus_tag = str(
        locus_tags[0]
    )


    if not (
        len(locus_tag) == 6
        and locus_tag.startswith("PA")
        and locus_tag[2:].isdigit()
    ):

        continue


    start = int(
        feature.location.start
    ) + 1

    end = int(
        feature.location.end
    )


    strand = feature.location.strand


    annotation_records.append(
        {
            "canonical_pao1":
                locus_tag,

            "start":
                start,

            "end":
                end,

            "strand":
                strand,
        }
    )


gene_annotation = pd.DataFrame(
    annotation_records
)


if gene_annotation[
    "canonical_pao1"
].duplicated().any():

    raise RuntimeError(
        "Duplicate locus tags in gene annotation."
    )


# ============================================================
# Add gene/product annotations from CDS features
# ============================================================

cds_records = []


for feature in record.features:

    if feature.type != "CDS":
        continue


    locus_tags = feature.qualifiers.get(
        "locus_tag",
        []
    )


    if not locus_tags:
        continue


    locus_tag = str(
        locus_tags[0]
    )


    if not (
        len(locus_tag) == 6
        and locus_tag.startswith("PA")
        and locus_tag[2:].isdigit()
    ):

        continue


    genes = feature.qualifiers.get(
        "gene",
        []
    )


    products = feature.qualifiers.get(
        "product",
        []
    )


    gene_name = (
        str(genes[0])
        if genes
        else ""
    )


    product = (
        str(products[0])
        if products
        else ""
    )


    cds_records.append(
        {
            "canonical_pao1":
                locus_tag,

            "gene_name":
                gene_name,

            "product":
                product,
        }
    )


cds_annotation = pd.DataFrame(
    cds_records
)


cds_annotation = (
    cds_annotation
    .drop_duplicates(
        subset=[
            "canonical_pao1"
        ]
    )
)


annotation = gene_annotation.merge(
    cds_annotation,
    on="canonical_pao1",
    how="left",
    validate="one_to_one",
)


annotation[
    "gene_name"
] = annotation[
    "gene_name"
].fillna(
    ""
)


annotation[
    "product"
] = annotation[
    "product"
].fillna(
    ""
)


print()
print(
    "Canonical genes annotated: "
    + str(
        len(annotation)
    )
)


# ============================================================
# Load Audit 60 results
# ============================================================

results = pd.read_csv(
    INPUT_FILE
)


required_columns = [
    "canonical_pao1",
    "n_positive_other6",
    "n_negative_other6",
    "majority_count_other6",
    "consensus_sign",
    "median_effect_other6",
    "GSE299861_published_effect",
    "GSE299861_sign",
    "opposes_consensus",
]


for column in required_columns:

    if column not in results.columns:

        raise RuntimeError(
            "Missing Audit 60 column: "
            + column
        )


annotated = results.merge(
    annotation,
    on="canonical_pao1",
    how="left",
    validate="one_to_one",
)


annotated[
    "opposition_strength"
] = (
    -annotated[
        "median_effect_other6"
    ]
    *
    annotated[
        "GSE299861_published_effect"
    ]
)


annotated.to_csv(
    OUT_DIR
    / "Audit61_all_genes_annotated.csv",
    index=False,
)


# ============================================================
# Robust reversal set
#
# Require:
#   >=5/6 other studies agree
#   median effect magnitude >= 0.05
#   GSE299861 points in opposite direction
# ============================================================

robust = annotated.loc[
    (
        annotated[
            "majority_count_other6"
        ]
        >= 5
    )
    &
    (
        annotated[
            "median_effect_other6"
        ].abs()
        >= 0.05
    )
    &
    (
        annotated[
            "opposes_consensus"
        ]
        == True
    )
].copy()


robust = robust.sort_values(
    "opposition_strength",
    ascending=False,
)


robust.to_csv(
    OUT_DIR
    / "Audit61_robust_reversed_genes.csv",
    index=False,
)


print()
print("=" * 100)
print("ROBUST REVERSED GENE SET")
print("=" * 100)

print(
    "Criteria:"
)

print(
    "  >=5/6 other studies agree in direction"
)

print(
    "  |median other-six effect| >= 0.05"
)

print(
    "  GSE299861 effect is opposite"
)

print()

print(
    "Robust reversed genes: "
    + str(
        len(
            robust
        )
    )
)


# ============================================================
# Split reversal directions
# ============================================================

normally_up_but_down = robust.loc[
    (
        robust[
            "consensus_sign"
        ]
        > 0
    )
    &
    (
        robust[
            "GSE299861_sign"
        ]
        < 0
    )
].copy()


normally_down_but_up = robust.loc[
    (
        robust[
            "consensus_sign"
        ]
        < 0
    )
    &
    (
        robust[
            "GSE299861_sign"
        ]
        > 0
    )
].copy()


print()
print(
    "Normally biofilm-higher, but LOWER in GSE299861: "
    + str(
        len(
            normally_up_but_down
        )
    )
)

print(
    "Normally biofilm-lower, but HIGHER in GSE299861: "
    + str(
        len(
            normally_down_but_up
        )
    )
)


# ============================================================
# Top annotated genes
# ============================================================

display_columns = [
    "canonical_pao1",
    "gene_name",
    "product",
    "n_positive_other6",
    "n_negative_other6",
    "median_effect_other6",
    "GSE299861_published_effect",
    "opposition_strength",
]


print()
print("=" * 100)
print("TOP 60 ROBUST REVERSED GENES")
print("=" * 100)

print(
    robust[
        display_columns
    ]
    .head(
        60
    )
    .to_string(
        index=False
    )
)


robust[
    display_columns
].head(
    100
).to_csv(
    OUT_DIR
    / "Audit61_top100_robust_reversed_genes.csv",
    index=False,
)


# ============================================================
# Genomic clustering
#
# Same reversal direction and <=10 kb between selected genes.
# This mirrors our earlier genomic-neighbourhood logic:
# these are genomic modules, not claimed operons.
# ============================================================

module_input = robust.loc[
    robust[
        "start"
    ].notna()
].copy()


module_input[
    "start"
] = module_input[
    "start"
].astype(
    int
)


module_input[
    "end"
] = module_input[
    "end"
].astype(
    int
)


module_input = module_input.sort_values(
    [
        "start",
        "end",
    ]
).reset_index(
    drop=True
)


module_ids = []

current_module = 0

previous_end = None
previous_consensus_sign = None


for _, row in module_input.iterrows():

    start = int(
        row[
            "start"
        ]
    )

    end = int(
        row[
            "end"
        ]
    )

    consensus_sign = int(
        row[
            "consensus_sign"
        ]
    )


    start_new = False


    if previous_end is None:

        start_new = True

    else:

        gap = (
            start
            - previous_end
        )


        if gap > 10000:

            start_new = True


        if consensus_sign != previous_consensus_sign:

            start_new = True


    if start_new:

        current_module += 1


    module_ids.append(
        current_module
    )


    previous_end = end
    previous_consensus_sign = consensus_sign


module_input[
    "module_id"
] = module_ids


# ============================================================
# Summarise modules
# ============================================================

module_records = []


for module_id, group in module_input.groupby(
    "module_id"
):

    genes = group[
        "canonical_pao1"
    ].tolist()


    gene_names = [
        value
        for value in group[
            "gene_name"
        ].astype(str)
        if value
        and value != "nan"
    ]


    products = [
        value
        for value in group[
            "product"
        ].astype(str)
        if value
        and value != "nan"
    ]


    module_start = int(
        group[
            "start"
        ].min()
    )


    module_end = int(
        group[
            "end"
        ].max()
    )


    module_records.append(
        {
            "module_id":
                int(
                    module_id
                ),

            "n_genes":
                len(
                    group
                ),

            "start":
                module_start,

            "end":
                module_end,

            "span_bp":
                (
                    module_end
                    - module_start
                    + 1
                ),

            "consensus_direction":
                (
                    "biofilm_higher_elsewhere"
                    if int(
                        group[
                            "consensus_sign"
                        ].iloc[
                            0
                        ]
                    ) > 0
                    else "biofilm_lower_elsewhere"
                ),

            "genes":
                ";".join(
                    genes
                ),

            "gene_names":
                ";".join(
                    gene_names
                ),

            "products":
                ";".join(
                    products
                ),

            "sum_opposition_strength":
                float(
                    group[
                        "opposition_strength"
                    ].sum()
                ),

            "mean_opposition_strength":
                float(
                    group[
                        "opposition_strength"
                    ].mean()
                ),
        }
    )


modules = pd.DataFrame(
    module_records
)


modules = modules.sort_values(
    [
        "n_genes",
        "sum_opposition_strength",
    ],
    ascending=[
        False,
        False,
    ],
)


modules.to_csv(
    OUT_DIR
    / "Audit61_genomic_modules_all.csv",
    index=False,
)


multi_gene_modules = modules.loc[
    modules[
        "n_genes"
    ]
    >= 2
].copy()


multi_gene_modules.to_csv(
    OUT_DIR
    / "Audit61_genomic_modules_multigene.csv",
    index=False,
)


print()
print("=" * 100)
print("GENOMIC REVERSAL MODULES")
print("=" * 100)

print(
    "All modules: "
    + str(
        len(
            modules
        )
    )
)

print(
    "Multi-gene modules: "
    + str(
        len(
            multi_gene_modules
        )
    )
)

print(
    "Genes in multi-gene modules: "
    + str(
        int(
            multi_gene_modules[
                "n_genes"
            ].sum()
        )
    )
)


print()
print(
    "Top multi-gene modules:"
)


module_display = [
    "module_id",
    "n_genes",
    "span_bp",
    "consensus_direction",
    "genes",
    "gene_names",
    "sum_opposition_strength",
]


print(
    multi_gene_modules[
        module_display
    ]
    .head(
        30
    )
    .to_string(
        index=False
    )
)


# ============================================================
# Product keyword inventory
#
# This is descriptive only, not formal enrichment.
# ============================================================

keywords = [
    "phage",
    "pilus",
    "pili",
    "flagell",
    "chemotaxis",
    "secretion",
    "transport",
    "efflux",
    "iron",
    "siderophore",
    "pyoverdine",
    "pyochelin",
    "quorum",
    "alginate",
    "polysaccharide",
    "biofilm",
    "oxidase",
    "reductase",
    "respiration",
    "nitrate",
    "nitrite",
    "denitrification",
    "stress",
    "ribosomal",
    "ribosome",
]


keyword_records = []


product_text = (
    robust[
        "gene_name"
    ].fillna(
        ""
    )
    .astype(str)
    + " "
    + robust[
        "product"
    ].fillna(
        ""
    )
    .astype(str)
).str.lower()


for keyword in keywords:

    mask = product_text.str.contains(
        keyword,
        regex=False,
    )


    matching = robust.loc[
        mask
    ]


    keyword_records.append(
        {
            "keyword":
                keyword,

            "n_genes":
                len(
                    matching
                ),

            "genes":
                ";".join(
                    matching[
                        "canonical_pao1"
                    ].tolist()
                ),
        }
    )


keyword_table = pd.DataFrame(
    keyword_records
)


keyword_table.to_csv(
    OUT_DIR
    / "Audit61_product_keyword_inventory.csv",
    index=False,
)


print()
print("=" * 100)
print("DESCRIPTIVE PRODUCT KEYWORDS")
print("=" * 100)

print(
    keyword_table.loc[
        keyword_table[
            "n_genes"
        ]
        > 0
    ]
    .sort_values(
        "n_genes",
        ascending=False,
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 61 COMPLETE")
print("=" * 100)

print(
    "Outputs saved to:"
)

print(
    OUT_DIR
)

print()

print(
    "Important: genomic modules are proximity-based "
    "neighbourhoods, not confirmed operons."
)

print(
    "Keyword counts are descriptive only; formal pathway "
    "enrichment should be done separately."
)

print()
