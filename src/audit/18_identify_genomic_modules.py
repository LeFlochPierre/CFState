from pathlib import Path
import re

import pandas as pd
from Bio import SeqIO


SIGNATURE_FILE = Path(
    "results/audit/core_biofilm_signature/"
    "core_biofilm_signature_fdr05.csv"
)

GENBANK_FILE = Path(
    "references/mappings/NC_002516.2.gb"
)

OUT_DIR = Path(
    "results/audit/core_biofilm_modules"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load signature
# ============================================================

sig = pd.read_csv(
    SIGNATURE_FILE
)

print()
print("=" * 100)
print("CF-STATE — Audit 18: genomic modules in core biofilm signature")
print("=" * 100)

print(f"Core genes: {len(sig)}")


# ============================================================
# Parse PAO1 genome annotation
# ============================================================

record = SeqIO.read(
    GENBANK_FILE,
    "genbank",
)

annotations = {}

for feature in record.features:

    if feature.type != "CDS":
        continue

    locus_tags = feature.qualifiers.get(
        "locus_tag",
        [],
    )

    if not locus_tags:
        continue

    locus = locus_tags[0]

    if not re.fullmatch(
        r"PA\d{4}",
        locus,
    ):
        continue

    start = int(
        feature.location.start
    )

    end = int(
        feature.location.end
    )

    strand = feature.location.strand

    gene = feature.qualifiers.get(
        "gene",
        [""],
    )[0]

    product = feature.qualifiers.get(
        "product",
        [""],
    )[0]

    annotations[locus] = {
        "start": start,
        "end": end,
        "strand": strand,
        "gene_name_ref": gene,
        "product_ref": product,
    }


# ============================================================
# Attach genomic positions
# ============================================================

sig["start"] = sig[
    "canonical_pao1"
].map(
    lambda x:
        annotations.get(
            x,
            {},
        ).get(
            "start",
            pd.NA,
        )
)

sig["end"] = sig[
    "canonical_pao1"
].map(
    lambda x:
        annotations.get(
            x,
            {},
        ).get(
            "end",
            pd.NA,
        )
)

sig["strand"] = sig[
    "canonical_pao1"
].map(
    lambda x:
        annotations.get(
            x,
            {},
        ).get(
            "strand",
            pd.NA,
        )
)


sig = sig.dropna(
    subset=["start", "end"]
).copy()

sig["start"] = sig[
    "start"
].astype(int)

sig["end"] = sig[
    "end"
].astype(int)

sig = sig.sort_values(
    "start"
).reset_index(
    drop=True
)


# ============================================================
# Identify same-direction local modules
#
# Rule:
# consecutive significant genes belong to the same module if:
#
# 1. they change in the same direction
# 2. genomic distance between them <= 10 kb
#
# This is deliberately an "operon-like / genomic module"
# definition, NOT a claim that every cluster is one operon.
# ============================================================

MAX_GAP_BP = 10_000

module_ids = []

current_module = 1

previous_end = None
previous_direction = None


for _, row in sig.iterrows():

    direction = row[
        "direction"
    ]

    start = row[
        "start"
    ]

    if previous_end is None:

        module_ids.append(
            current_module
        )

    else:

        gap = (
            start
            - previous_end
        )

        same_direction = (
            direction
            == previous_direction
        )

        close_enough = (
            gap <= MAX_GAP_BP
        )

        if (
            same_direction
            and close_enough
        ):

            module_ids.append(
                current_module
            )

        else:

            current_module += 1

            module_ids.append(
                current_module
            )

    previous_end = row[
        "end"
    ]

    previous_direction = direction


sig[
    "module_id"
] = module_ids


# ============================================================
# Summarise modules
# ============================================================

module_records = []

for module_id, block in sig.groupby(
    "module_id"
):

    block = block.sort_values(
        "start"
    )

    genes = block[
        "canonical_pao1"
    ].tolist()

    names = (
        block[
            "gene_name"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    products = (
        block[
            "product"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    start = int(
        block["start"].min()
    )

    end = int(
        block["end"].max()
    )

    direction = block[
        "direction"
    ].iloc[0]

    module_records.append(
        {
            "module_id":
                module_id,

            "direction":
                direction,

            "n_genes":
                len(block),

            "start":
                start,

            "end":
                end,

            "span_bp":
                end - start,

            "genes":
                ";".join(
                    genes
                ),

            "gene_names":
                ";".join(
                    names
                ),

            "products":
                ";".join(
                    products
                ),

            "median_consistency_score":
                block[
                    "consistency_score"
                ].median(),

            "max_consistency_score":
                block[
                    "consistency_score"
                ].max(),
        }
    )


modules = pd.DataFrame(
    module_records
)


# ============================================================
# Keep biologically interesting multi-gene modules
# ============================================================

multi = modules[
    modules[
        "n_genes"
    ] >= 2
].copy()


multi = multi.sort_values(
    [
        "n_genes",
        "median_consistency_score",
    ],
    ascending=[
        False,
        False,
    ],
)


# ============================================================
# Save
# ============================================================

sig.to_csv(
    OUT_DIR
    / "core_signature_with_genomic_modules.csv",
    index=False,
)

modules.to_csv(
    OUT_DIR
    / "all_modules.csv",
    index=False,
)

multi.to_csv(
    OUT_DIR
    / "multi_gene_modules.csv",
    index=False,
)


# ============================================================
# Output
# ============================================================

print()
print("=" * 100)
print("MODULE SUMMARY")
print("=" * 100)

print(
    f"All modules:             "
    f"{len(modules)}"
)

print(
    f"Multi-gene modules:      "
    f"{len(multi)}"
)

print(
    f"Genes in multi-gene "
    f"modules: "
    f"{multi['n_genes'].sum()}"
)

print()


for direction in [
    "higher_in_biofilm",
    "lower_in_biofilm",
]:

    subset = multi[
        multi["direction"]
        == direction
    ]

    print()
    print("=" * 100)
    print(
        direction.upper()
    )
    print("=" * 100)

    if subset.empty:

        print(
            "No multi-gene modules."
        )

        continue

    print(
        subset[
            [
                "module_id",
                "n_genes",
                "span_bp",
                "genes",
                "gene_names",
                "median_consistency_score",
            ]
        ]
        .head(30)
        .to_string(
            index=False
        )
    )


print()
print("=" * 100)
print("AUDIT 18 COMPLETE")
print("=" * 100)

print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)

print()
print(
    "NOTE: these are genomic neighbourhoods, "
    "not automatically confirmed operons."
)
print()
