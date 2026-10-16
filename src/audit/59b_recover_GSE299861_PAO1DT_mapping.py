from pathlib import Path
from difflib import SequenceMatcher

import pandas as pd


PUBLISHED_FILE = Path(
    "data/raw/expanded_biofilm/GSE299861/published_expression/"
    "GSE299861_PAO1_all.fpkm.txt.gz"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "published_expression_mapping"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 59b: "
    "recover GSE299861 PAO1DT -> canonical PAO1 mapping"
)
print("=" * 100)


# ============================================================
# Load authors' table
# ============================================================

published = pd.read_csv(
    PUBLISHED_FILE,
    sep="\t",
    compression="gzip",
)


required_columns = [
    "GeneId",
    "Exonic.gene.sizes",
]


for column in required_columns:

    if column not in published.columns:

        raise RuntimeError(
            "Missing published-table column: "
            + column
        )


published[
    "GeneId"
] = (
    published[
        "GeneId"
    ]
    .astype(str)
)


published[
    "author_length"
] = pd.to_numeric(
    published[
        "Exonic.gene.sizes"
    ],
    errors="raise",
).astype(int)


print()
print("=" * 100)
print("AUTHORS' ANNOTATION")
print("=" * 100)

print(
    "Published feature rows: "
    + str(len(published))
)


id_pattern_ok = (
    published[
        "GeneId"
    ]
    .str.fullmatch(
        r"PAO1DT_\d+"
    )
)


print(
    "PAO1DT-style IDs: "
    + str(
        int(
            id_pattern_ok.sum()
        )
    )
    + " / "
    + str(len(published))
)


# ============================================================
# Check whether PAO1DT numbering follows table order
# ============================================================

author_numbers = (
    published[
        "GeneId"
    ]
    .str.extract(
        r"PAO1DT_(\d+)",
        expand=False,
    )
)


author_numbers = pd.to_numeric(
    author_numbers,
    errors="coerce",
)


numbering_complete = (
    author_numbers.notna().all()
)


numbering_sequential = False


if numbering_complete:

    expected_numbers = pd.Series(
        range(
            1,
            len(published) + 1,
        ),
        index=published.index,
        dtype=float,
    )


    numbering_sequential = bool(
        (
            author_numbers
            == expected_numbers
        ).all()
    )


print(
    "PAO1DT numbering complete: "
    + str(numbering_complete)
)

print(
    "PAO1DT numbering exactly follows row order: "
    + str(numbering_sequential)
)


# ============================================================
# Discover canonical PAO1 SAF annotation
#
# We choose the SAF file containing the largest number
# of exact canonical PAxxxx IDs.
# ============================================================

saf_candidates = list(
    Path("references").rglob(
        "*.saf"
    )
)


candidate_records = []

candidate_tables = {}


for path in saf_candidates:

    try:

        table = pd.read_csv(
            path,
            sep="\t",
        )

    except Exception:

        continue


    if (
        "Start" not in table.columns
        or "End" not in table.columns
    ):

        continue


    best_id_column = None
    best_n = 0


    for column in table.columns:

        values = (
            table[
                column
            ]
            .astype(str)
        )


        n_canonical = int(
            values.str.fullmatch(
                r"PA\d{4}",
                na=False,
            ).sum()
        )


        if n_canonical > best_n:

            best_n = n_canonical
            best_id_column = column


    if best_id_column is None:

        continue


    candidate_records.append(
        {
            "path": str(path),
            "id_column": str(best_id_column),
            "canonical_PA_rows": best_n,
        }
    )


    candidate_tables[
        str(path)
    ] = (
        table,
        best_id_column,
    )


if not candidate_records:

    raise RuntimeError(
        "No SAF file with canonical PAxxxx IDs "
        "was found under references/."
    )


candidate_df = pd.DataFrame(
    candidate_records
).sort_values(
    "canonical_PA_rows",
    ascending=False,
)


print()
print("=" * 100)
print("CANONICAL SAF CANDIDATES")
print("=" * 100)

print(
    candidate_df.head(
        10
    ).to_string(
        index=False
    )
)


selected_path = str(
    candidate_df.iloc[
        0
    ][
        "path"
    ]
)


canonical_raw, canonical_id_column = (
    candidate_tables[
        selected_path
    ]
)


print()
print(
    "Selected canonical annotation:"
)

print(
    selected_path
)


# ============================================================
# Prepare canonical PAO1 genes in genomic order
# ============================================================

canonical = canonical_raw.copy()


canonical[
    "canonical_pao1"
] = (
    canonical[
        canonical_id_column
    ]
    .astype(str)
)


canonical = canonical.loc[
    canonical[
        "canonical_pao1"
    ]
    .str.fullmatch(
        r"PA\d{4}",
        na=False,
    )
].copy()


canonical[
    "Start"
] = pd.to_numeric(
    canonical[
        "Start"
    ],
    errors="raise",
).astype(int)


canonical[
    "End"
] = pd.to_numeric(
    canonical[
        "End"
    ],
    errors="raise",
).astype(int)


canonical[
    "canonical_length"
] = (
    canonical[
        "End"
    ]
    -
    canonical[
        "Start"
    ]
    + 1
)


canonical = (
    canonical.sort_values(
        [
            "Start",
            "End",
            "canonical_pao1",
        ]
    )
    .reset_index(
        drop=True
    )
)


if canonical[
    "canonical_pao1"
].duplicated().any():

    duplicates = (
        canonical.loc[
            canonical[
                "canonical_pao1"
            ].duplicated(
                keep=False
            ),
            "canonical_pao1",
        ]
        .unique()
        .tolist()
    )


    raise RuntimeError(
        "Duplicate canonical PA IDs in selected SAF: "
        + ", ".join(
            duplicates[:20]
        )
    )


print()
print("=" * 100)
print("CANONICAL PAO1 ANNOTATION")
print("=" * 100)

print(
    "Canonical PAxxxx genes: "
    + str(len(canonical))
)

print(
    "First canonical gene: "
    + str(
        canonical.iloc[
            0
        ][
            "canonical_pao1"
        ]
    )
)

print(
    "Last canonical gene: "
    + str(
        canonical.iloc[
            -1
        ][
            "canonical_pao1"
        ]
    )
)


# ============================================================
# Ordered gene-length alignment
#
# The authors used sequential PAO1DT IDs. We compare their
# ordered gene-length sequence with the canonical PAO1
# gene-length sequence.
#
# SequenceMatcher finds exact consecutive length blocks while
# allowing insertions/deletions between blocks.
# ============================================================

author_lengths = (
    published[
        "author_length"
    ]
    .astype(int)
    .tolist()
)


canonical_lengths = (
    canonical[
        "canonical_length"
    ]
    .astype(int)
    .tolist()
)


matcher = SequenceMatcher(
    None,
    author_lengths,
    canonical_lengths,
    autojunk=False,
)


blocks = [
    block
    for block in matcher.get_matching_blocks()
    if block.size > 0
]


block_records = []


for block_index, block in enumerate(
    blocks,
    start=1,
):

    block_records.append(
        {
            "block":
                block_index,

            "author_start_index":
                block.a,

            "canonical_start_index":
                block.b,

            "block_size":
                block.size,

            "author_start_id":
                published.iloc[
                    block.a
                ][
                    "GeneId"
                ],

            "canonical_start_id":
                canonical.iloc[
                    block.b
                ][
                    "canonical_pao1"
                ],
        }
    )


block_df = pd.DataFrame(
    block_records
)


block_df.to_csv(
    OUT_DIR
    / "Audit59b_exact_length_matching_blocks.csv",
    index=False,
)


matched_all = int(
    sum(
        block.size
        for block in blocks
    )
)


matched_block3 = int(
    sum(
        block.size
        for block in blocks
        if block.size >= 3
    )
)


matched_block5 = int(
    sum(
        block.size
        for block in blocks
        if block.size >= 5
    )
)


largest_block = int(
    max(
        block.size
        for block in blocks
    )
) if blocks else 0


print()
print("=" * 100)
print("ORDERED GENE-LENGTH ALIGNMENT")
print("=" * 100)

print(
    "Exact aligned rows in all matching blocks: "
    + str(matched_all)
)

print(
    "Rows in matching blocks >=3 genes: "
    + str(matched_block3)
)

print(
    "Rows in matching blocks >=5 genes: "
    + str(matched_block5)
)

print(
    "Largest exact consecutive block: "
    + str(largest_block)
)

print(
    "Canonical genes covered by >=3-gene blocks: "
    + format(
        100.0
        * matched_block3
        / len(canonical),
        ".2f",
    )
    + "%"
)


print()
print(
    "Largest matching blocks:"
)

print(
    block_df.sort_values(
        "block_size",
        ascending=False,
    )
    .head(
        20
    )
    .to_string(
        index=False
    )
)


# ============================================================
# Build conservative high-confidence crosswalk
#
# Only exact length matches inside consecutive blocks of
# at least 3 genes are retained.
# ============================================================

mapping_records = []


for block_index, block in enumerate(
    blocks,
    start=1,
):

    if block.size < 3:
        continue


    for offset in range(
        block.size
    ):

        author_index = (
            block.a
            + offset
        )

        canonical_index = (
            block.b
            + offset
        )


        author_row = published.iloc[
            author_index
        ]


        canonical_row = canonical.iloc[
            canonical_index
        ]


        mapping_records.append(
            {
                "PAO1DT_ID":
                    author_row[
                        "GeneId"
                    ],

                "canonical_pao1":
                    canonical_row[
                        "canonical_pao1"
                    ],

                "author_length":
                    int(
                        author_row[
                            "author_length"
                        ]
                    ),

                "canonical_length":
                    int(
                        canonical_row[
                            "canonical_length"
                        ]
                    ),

                "canonical_start":
                    int(
                        canonical_row[
                            "Start"
                        ]
                    ),

                "canonical_end":
                    int(
                        canonical_row[
                            "End"
                        ]
                    ),

                "matching_block":
                    block_index,

                "matching_block_size":
                    block.size,
            }
        )


mapping = pd.DataFrame(
    mapping_records
)


if not mapping.empty:

    if mapping[
        "PAO1DT_ID"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate PAO1DT mappings detected."
        )


    if mapping[
        "canonical_pao1"
    ].duplicated().any():

        raise RuntimeError(
            "Duplicate canonical mappings detected."
        )


    if not (
        mapping[
            "author_length"
        ]
        ==
        mapping[
            "canonical_length"
        ]
    ).all():

        raise RuntimeError(
            "Length mismatch inside exact mapping."
        )


mapping.to_csv(
    OUT_DIR
    / "Audit59b_high_confidence_PAO1DT_to_PA_mapping.csv",
    index=False,
)


# ============================================================
# Frozen 4853 coverage of recovered mapping
# ============================================================

COMMON_FILE = Path(
    "results/audit/expanded_biofilm/GSE12207/"
    "finalization/"
    "GSE12207_cross_platform_gene_universe_4853.csv"
)


common = pd.read_csv(
    COMMON_FILE
)


common_genes = set(
    common[
        "canonical_pao1"
    ].astype(str)
)


mapped_canonical = set(
    mapping[
        "canonical_pao1"
    ].astype(str)
) if not mapping.empty else set()


common_mapped = (
    common_genes
    & mapped_canonical
)


print()
print("=" * 100)
print("HIGH-CONFIDENCE CROSSWALK")
print("=" * 100)

print(
    "High-confidence PAO1DT -> PA mappings: "
    + str(len(mapping))
)

print(
    "Common 4,853 modelling genes mapped: "
    + str(len(common_mapped))
)

print(
    "Common-gene mapping coverage: "
    + format(
        100.0
        * len(common_mapped)
        / 4853.0,
        ".2f",
    )
    + "%"
)


if not mapping.empty:

    print()
    print(
        "First 20 high-confidence mappings:"
    )

    print(
        mapping.head(
            20
        ).to_string(
            index=False
        )
    )


    print()
    print(
        "Last 20 high-confidence mappings:"
    )

    print(
        mapping.tail(
            20
        ).to_string(
            index=False
        )
    )


print()
print("=" * 100)
print("AUDIT 59b COMPLETE")
print("=" * 100)

print(
    "No expression values were compared yet."
)

print(
    "The recovered crosswalk uses only exact gene-length "
    "runs of at least 3 consecutive genes."
)

print()
