from pathlib import Path
from io import StringIO
from collections import defaultdict
import json
import re

import pandas as pd


ROOT = Path.cwd()

RAW = (
    ROOT
    / "data/raw/phase2/GSE152295/GSE152295_PSEUDO_processed.txt.gz"
)

GENBANK = (
    ROOT
    / "references/mappings/NC_002516.2.gb"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

OUT = (
    ROOT
    / "results/phase2/audit_10c_correct_full_locus_mapping"
)


def json_df(value):
    if isinstance(value, str):
        return pd.read_json(
            StringIO(value)
        )

    if isinstance(value, dict):
        return pd.DataFrame(
            value
        )

    raise RuntimeError(
        "Unexpected serialized dataframe type"
    )


def parse_region(region):
    text = str(region)

    match = re.search(
        r"(\d+)\.\.(\d+)",
        text,
    )

    if match is None:
        return None

    start = int(
        match.group(1)
    )

    end = int(
        match.group(2)
    )

    return (
        start,
        end,
    )


def load_genbank_maps(path):
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython required. Install with:\n"
            "conda install -c conda-forge biopython"
        )

    record = SeqIO.read(
        str(path),
        "genbank",
    )

    locus_tags = set()

    symbol_to_loci = defaultdict(
        set
    )

    interval_to_loci = defaultdict(
        set
    )

    locus_rows = []

    # Prefer features that actually describe genes.
    accepted_types = {
        "CDS",
        "gene",
        "tRNA",
        "rRNA",
        "ncRNA",
    }

    for feature in record.features:
        if feature.type not in accepted_types:
            continue

        tags = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if not tags:
            continue

        locus = str(
            tags[0]
        ).strip()

        # IMPORTANT:
        # Do not force locus tags into PA####.
        # Valid PAO1 tags include PA1019a, PA1112b,
        # PA0457.1, etc.
        if not locus.upper().startswith(
            "PA"
        ):
            continue

        locus_tags.add(
            locus
        )

        start = int(
            feature.location.start
        ) + 1

        end = int(
            feature.location.end
        )

        interval_to_loci[
            (
                start,
                end,
            )
        ].add(
            locus
        )

        symbols = []

        for symbol in feature.qualifiers.get(
            "gene",
            [],
        ):
            clean = str(
                symbol
            ).strip()

            if clean:
                symbols.append(
                    clean
                )

                symbol_to_loci[
                    clean.lower()
                ].add(
                    locus
                )

        locus_rows.append(
            {
                "locus_tag":
                    locus,

                "feature_type":
                    feature.type,

                "start":
                    start,

                "end":
                    end,

                "gene_symbols":
                    ";".join(
                        symbols
                    ),
            }
        )

    unique_symbols = {
        symbol: next(
            iter(
                loci
            )
        )
        for symbol, loci
        in symbol_to_loci.items()
        if len(
            loci
        ) == 1
    }

    return (
        locus_tags,
        unique_symbols,
        interval_to_loci,
        pd.DataFrame(
            locus_rows
        ),
    )


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 10c")
    print("Correct full PAO1 locus-tag mapping")
    print("=" * 78)

    for path in [
        RAW,
        GENBANK,
        MODEL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Source matrix
    # ------------------------------------------------------------

    raw = pd.read_csv(
        RAW,
        sep="\t",
        compression="gzip",
    )

    print()
    print(
        "Source rows:",
        len(raw),
    )

    # ------------------------------------------------------------
    # Frozen model gene universe
    # ------------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        model_raw = json.load(
            handle
        )

    M = json_df(
        model_raw["M"]
    )

    model_ids = [
        str(x).strip()
        for x in M.index
    ]

    model_set = set(
        model_ids
    )

    print(
        "Frozen model genes:",
        len(
            model_ids
        ),
    )

    suffix_model_ids = [
        gene
        for gene in model_ids
        if re.fullmatch(
            r"PA\d{4}(?:[A-Za-z]+|\.\d+)",
            gene,
        )
    ]

    print(
        "Model locus tags with suffix/decimal:",
        len(
            suffix_model_ids
        ),
    )

    if suffix_model_ids:
        print(
            "Examples:",
            suffix_model_ids[
                :20
            ],
        )

    # ------------------------------------------------------------
    # Full GenBank annotation
    # ------------------------------------------------------------

    (
        genbank_loci,
        unique_symbols,
        interval_to_loci,
        locus_table,
    ) = load_genbank_maps(
        GENBANK
    )

    locus_table.to_csv(
        OUT
        / "genbank_locus_inventory.csv",
        index=False,
    )

    print()
    print(
        "Distinct GenBank locus tags:",
        len(
            genbank_loci
        ),
    )

    print(
        "Unique gene symbols:",
        len(
            unique_symbols
        ),
    )

    # ------------------------------------------------------------
    # Correct mapping
    # ------------------------------------------------------------

    rows = []

    for source_row, row in raw.iterrows():
        name = str(
            row["Name"]
        ).strip()

        region = str(
            row["Region"]
        ).strip()

        mapped = None
        method = None

        # --------------------------------------------------------
        # 1. Exact match to the frozen model universe.
        #
        # This is the critical fix:
        # PA0032a remains PA0032a.
        # --------------------------------------------------------

        if name in model_set:
            mapped = name
            method = "exact_model_id"

        # --------------------------------------------------------
        # 2. Case-insensitive exact model match.
        # --------------------------------------------------------

        if mapped is None:
            matching_model = [
                gene
                for gene in model_ids
                if gene.lower()
                == name.lower()
            ]

            if len(
                matching_model
            ) == 1:
                mapped = matching_model[
                    0
                ]

                method = (
                    "case_insensitive_exact_model_id"
                )

        # --------------------------------------------------------
        # 3. Exact GenBank locus tag, provided it exists in M.
        # --------------------------------------------------------

        if (
            mapped is None
            and name in genbank_loci
            and name in model_set
        ):
            mapped = name
            method = "exact_genbank_locus"

        # --------------------------------------------------------
        # 4. Unique gene symbol -> locus.
        # --------------------------------------------------------

        if mapped is None:
            symbol = name.lower()

            if symbol in unique_symbols:
                locus = unique_symbols[
                    symbol
                ]

                if locus in model_set:
                    mapped = locus
                    method = "unique_gene_symbol"

        # --------------------------------------------------------
        # 5. Exact genomic-coordinate match.
        #
        # This is only a fallback.
        # No nearest-gene mapping and no overlap guessing.
        # --------------------------------------------------------

        if mapped is None:
            parsed = parse_region(
                region
            )

            if parsed is not None:
                candidates = [
                    locus
                    for locus in interval_to_loci.get(
                        parsed,
                        set(),
                    )
                    if locus in model_set
                ]

                candidates = sorted(
                    set(
                        candidates
                    )
                )

                if len(
                    candidates
                ) == 1:
                    mapped = candidates[
                        0
                    ]

                    method = (
                        "exact_coordinate_match"
                    )

        if mapped is None:
            method = "unmapped"

        rows.append(
            {
                "source_row":
                    source_row,

                "Name":
                    name,

                "Chromosome":
                    row.get(
                        "Chromosome",
                        "",
                    ),

                "Region":
                    region,

                "mapped_model_gene":
                    mapped,

                "mapping_method":
                    method,
            }
        )

    mapping = pd.DataFrame(
        rows
    )

    mapping.to_csv(
        OUT
        / "corrected_gene_mapping.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Mapping QC
    # ------------------------------------------------------------

    mapped = mapping.loc[
        mapping[
            "mapped_model_gene"
        ].notna()
    ].copy()

    unmapped = mapping.loc[
        mapping[
            "mapped_model_gene"
        ].isna()
    ].copy()

    duplicated = mapped.loc[
        mapped[
            "mapped_model_gene"
        ].duplicated(
            keep=False
        )
    ].copy()

    mapped_unique = set(
        mapped[
            "mapped_model_gene"
        ]
    )

    unused_model = sorted(
        model_set
        - mapped_unique
    )

    method_counts = (
        mapping[
            "mapping_method"
        ]
        .value_counts()
        .rename_axis(
            "mapping_method"
        )
        .reset_index(
            name="n_rows"
        )
    )

    method_counts.to_csv(
        OUT
        / "mapping_method_counts.csv",
        index=False,
    )

    unmapped.to_csv(
        OUT
        / "unmapped_features.csv",
        index=False,
    )

    duplicated.to_csv(
        OUT
        / "remaining_mapping_collisions.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "unused_model_gene":
                unused_model
        }
    ).to_csv(
        OUT
        / "unused_model_genes.csv",
        index=False,
    )

    print()
    print("Corrected mapping methods")
    print(
        method_counts.to_string(
            index=False
        )
    )

    print()
    print("Corrected mapping QC")
    print(
        "  mapped rows:",
        len(
            mapped
        ),
    )
    print(
        "  unique mapped model genes:",
        len(
            mapped_unique
        ),
    )
    print(
        "  unmapped source rows:",
        len(
            unmapped
        ),
    )
    print(
        "  rows involved in remaining collisions:",
        len(
            duplicated
        ),
    )
    print(
        "  unused model genes:",
        len(
            unused_model
        ),
    )

    if not unmapped.empty:
        print()
        print("Unmapped source rows")
        print(
            unmapped[
                [
                    "source_row",
                    "Name",
                    "Region",
                ]
            ].to_string(
                index=False
            )
        )

    if not duplicated.empty:
        print()
        print("Remaining collisions")
        print(
            duplicated[
                [
                    "source_row",
                    "Name",
                    "Region",
                    "mapped_model_gene",
                    "mapping_method",
                ]
            ]
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

    if unused_model:
        print()
        print("First unused frozen-model genes")
        for gene in unused_model[
            :30
        ]:
            print(
                " ",
                gene,
            )

    # ------------------------------------------------------------
    # Build model-aligned TPM matrix
    # ------------------------------------------------------------

    expression_cols = [
        column
        for column in raw.columns
        if str(
            column
        ).endswith(
            " - TPM"
        )
    ]

    usable_mapping = mapping.loc[
        mapping[
            "mapped_model_gene"
        ].notna()
    ]

    if (
        usable_mapping[
            "mapped_model_gene"
        ].duplicated()
        .any()
    ):
        print()
        print(
            "Model-aligned matrix NOT written "
            "because mapping is still non-unique."
        )

    else:
        source_indices = (
            usable_mapping[
                "source_row"
            ]
            .astype(int)
            .tolist()
        )

        aligned = (
            raw.loc[
                source_indices,
                expression_cols,
            ]
            .copy()
        )

        aligned.index = (
            usable_mapping[
                "mapped_model_gene"
            ]
            .astype(str)
            .values
        )

        aligned = aligned.reindex(
            [
                gene
                for gene in model_ids
                if gene in aligned.index
            ]
        )

        aligned.to_csv(
            OUT
            / "GSE152295_model_aligned_TPM.csv"
        )

        print()
        print(
            "Model-aligned TPM matrix:",
            aligned.shape,
        )

    # ------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------

    n_unique = len(
        mapped_unique
    )

    coverage = (
        n_unique
        / len(
            model_ids
        )
    )

    perfect = (
        len(
            unmapped
        )
        == 0
        and len(
            duplicated
        )
        == 0
        and len(
            unused_model
        )
        == 0
        and n_unique
        == len(
            model_ids
        )
    )

    acceptable = (
        len(
            duplicated
        )
        == 0
        and coverage
        >= 0.99
    )

    print()
    print("Final mapping coverage")
    print(
        " ",
        str(
            n_unique
        )
        + "/"
        + str(
            len(
                model_ids
            )
        ),
        "("
        + format(
            coverage,
            ".2%",
        )
        + ")",
    )

    if perfect:
        status = "PERFECT_1_TO_1"

    elif acceptable:
        status = "HIGH_COVERAGE_NO_COLLISIONS"

    else:
        status = "FURTHER_REVIEW_REQUIRED"

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_10C_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
