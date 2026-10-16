from pathlib import Path
from io import StringIO
from collections import defaultdict
import json
import re

import pandas as pd


ROOT = Path.cwd()

FPKM = (
    ROOT
    / "data/raw/phase2/GSE299861"
    / "GSE299861_PAO1_all.fpkm.txt.gz"
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
    / "results/phase2/audit_12a_GSE299861_gene_mapping"
)


def json_df(value):
    if isinstance(value, str):
        return pd.read_json(
            StringIO(value)
        )

    if isinstance(value, dict):
        return pd.DataFrame(value)

    raise RuntimeError(
        "Unexpected serialized DataFrame type"
    )


def clean_geneid(value):
    if pd.isna(value):
        return None

    text = str(value).strip()

    if not text:
        return None

    text = re.sub(
        r"^GeneID:",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Handle values read as floating point, e.g. "878321.0".
    if re.fullmatch(
        r"\d+\.0",
        text,
    ):
        text = text[:-2]

    match = re.search(
        r"\b(\d{4,})\b",
        text,
    )

    if match:
        return match.group(1)

    return None


def load_genbank_maps(path):
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython is required."
        )

    record = SeqIO.read(
        str(path),
        "genbank",
    )

    geneid_to_loci = defaultdict(set)
    symbol_to_loci = defaultdict(set)
    old_locus_to_loci = defaultdict(set)
    locus_tags = set()

    for feature in record.features:
        tags = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if not tags:
            continue

        locus = str(
            tags[0]
        ).strip()

        if not locus:
            continue

        locus_tags.add(
            locus
        )

        for xref in feature.qualifiers.get(
            "db_xref",
            [],
        ):
            text = str(
                xref
            ).strip()

            if text.startswith(
                "GeneID:"
            ):
                geneid = text.split(
                    ":",
                    1,
                )[1]

                geneid_to_loci[
                    geneid
                ].add(
                    locus
                )

        for symbol in feature.qualifiers.get(
            "gene",
            [],
        ):
            symbol = str(
                symbol
            ).strip().lower()

            if symbol:
                symbol_to_loci[
                    symbol
                ].add(
                    locus
                )

        for old in feature.qualifiers.get(
            "old_locus_tag",
            [],
        ):
            old = str(
                old
            ).strip().lower()

            if old:
                old_locus_to_loci[
                    old
                ].add(
                    locus
                )

    unique_geneid = {
        key: next(
            iter(values)
        )
        for key, values
        in geneid_to_loci.items()
        if len(values) == 1
    }

    unique_symbol = {
        key: next(
            iter(values)
        )
        for key, values
        in symbol_to_loci.items()
        if len(values) == 1
    }

    unique_old_locus = {
        key: next(
            iter(values)
        )
        for key, values
        in old_locus_to_loci.items()
        if len(values) == 1
    }

    return (
        locus_tags,
        unique_geneid,
        unique_symbol,
        unique_old_locus,
    )


def possible_exact_locus(text, model_set):
    if text is None:
        return None

    value = str(
        text
    ).strip()

    if value in model_set:
        return value

    matches = re.findall(
        r"(?<![A-Za-z0-9])"
        r"(PA\d{4}(?:[A-Za-z]+|\.\d+)?)"
        r"(?![A-Za-z0-9])",
        value,
        flags=re.IGNORECASE,
    )

    valid = []

    for match in matches:
        candidates = [
            gene
            for gene in model_set
            if gene.lower()
            == match.lower()
        ]

        valid.extend(
            candidates
        )

    valid = sorted(
        set(valid)
    )

    if len(valid) == 1:
        return valid[0]

    return None


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 12a")
    print("GSE299861 PAO1 identifier mapping repair")
    print("=" * 78)

    for path in [
        FPKM,
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

    raw = pd.read_csv(
        FPKM,
        sep="\t",
        compression="gzip",
    )

    print()
    print("Source table")
    print(
        "  shape:",
        raw.shape,
    )

    required = {
        "GeneId",
        "GeneName",
    }

    missing = (
        required
        - set(raw.columns)
    )

    if missing:
        raise RuntimeError(
            "Missing expected columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    print()
    print("First 20 identifiers")
    print(
        raw[
            [
                "GeneId",
                "GeneName",
            ]
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Frozen model universe
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

    # ------------------------------------------------------------
    # GenBank identifier maps
    # ------------------------------------------------------------

    (
        genbank_loci,
        geneid_map,
        symbol_map,
        old_locus_map,
    ) = load_genbank_maps(
        GENBANK
    )

    print()
    print("Frozen annotation maps")
    print(
        "  model genes:",
        len(model_ids),
    )
    print(
        "  GenBank locus tags:",
        len(genbank_loci),
    )
    print(
        "  unique NCBI GeneIDs:",
        len(geneid_map),
    )
    print(
        "  unique gene symbols:",
        len(symbol_map),
    )
    print(
        "  unique old locus tags:",
        len(old_locus_map),
    )

    # ------------------------------------------------------------
    # Row-wise robust mapping
    # ------------------------------------------------------------

    rows = []

    for source_row, row in raw.iterrows():
        gene_id_raw = row[
            "GeneId"
        ]

        gene_name_raw = row[
            "GeneName"
        ]

        gene_id_text = str(
            gene_id_raw
        ).strip()

        gene_name_text = str(
            gene_name_raw
        ).strip()

        candidates = []

        # 1. Exact or embedded PA locus tag in GeneId.
        locus = possible_exact_locus(
            gene_id_text,
            model_set,
        )

        if locus is not None:
            candidates.append(
                (
                    locus,
                    "GeneId_exact_or_embedded_locus",
                )
            )

        # 2. Exact or embedded PA locus tag in GeneName.
        locus = possible_exact_locus(
            gene_name_text,
            model_set,
        )

        if locus is not None:
            candidates.append(
                (
                    locus,
                    "GeneName_exact_or_embedded_locus",
                )
            )

        # 3. NCBI numeric GeneID.
        geneid = clean_geneid(
            gene_id_raw
        )

        if (
            geneid is not None
            and geneid in geneid_map
        ):
            locus = geneid_map[
                geneid
            ]

            if locus in model_set:
                candidates.append(
                    (
                        locus,
                        "NCBI_GeneID",
                    )
                )

        # 4. GeneName as unique symbol.
        symbol = gene_name_text.lower()

        if symbol in symbol_map:
            locus = symbol_map[
                symbol
            ]

            if locus in model_set:
                candidates.append(
                    (
                        locus,
                        "GeneName_unique_symbol",
                    )
                )

        # 5. GeneId as unique symbol, just in case.
        symbol = gene_id_text.lower()

        if symbol in symbol_map:
            locus = symbol_map[
                symbol
            ]

            if locus in model_set:
                candidates.append(
                    (
                        locus,
                        "GeneId_unique_symbol",
                    )
                )

        # 6. Old locus tags.
        for value, source in [
            (
                gene_id_text,
                "GeneId",
            ),
            (
                gene_name_text,
                "GeneName",
            ),
        ]:
            key = value.lower()

            if key in old_locus_map:
                locus = old_locus_map[
                    key
                ]

                if locus in model_set:
                    candidates.append(
                        (
                            locus,
                            source
                            + "_old_locus_tag",
                        )
                    )

        loci = sorted(
            set(
                locus
                for locus, _
                in candidates
            )
        )

        if len(loci) == 1:
            mapped = loci[0]

            methods = sorted(
                set(
                    method
                    for locus, method
                    in candidates
                    if locus == mapped
                )
            )

            method = ";".join(
                methods
            )

            conflict = False

        elif len(loci) == 0:
            mapped = None
            method = "unmapped"
            conflict = False

        else:
            mapped = None
            method = "conflicting_identifiers"
            conflict = True

        rows.append(
            {
                "source_row":
                    source_row,

                "GeneId":
                    gene_id_raw,

                "GeneName":
                    gene_name_raw,

                "parsed_NCBI_GeneID":
                    geneid,

                "mapped_model_gene":
                    mapped,

                "mapping_method":
                    method,

                "candidate_loci":
                    ";".join(
                        loci
                    ),

                "identifier_conflict":
                    conflict,
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
    # QC
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

    conflicts = mapping.loc[
        mapping[
            "identifier_conflict"
        ]
    ].copy()

    duplicate_rows = mapped.loc[
        mapped[
            "mapped_model_gene"
        ].duplicated(
            keep=False
        )
    ].copy()

    unique_model_genes = set(
        mapped[
            "mapped_model_gene"
        ]
    )

    missing_model_genes = sorted(
        model_set
        - unique_model_genes
    )

    coverage = (
        len(unique_model_genes)
        / len(model_ids)
    )

    print()
    print("Corrected mapping QC")
    print(
        "  source rows:",
        len(mapping),
    )
    print(
        "  mapped rows:",
        len(mapped),
    )
    print(
        "  unique mapped model genes:",
        len(unique_model_genes),
    )
    print(
        "  model coverage:",
        format(
            coverage,
            ".2%",
        ),
    )
    print(
        "  unmapped rows:",
        len(unmapped),
    )
    print(
        "  conflicting rows:",
        len(conflicts),
    )
    print(
        "  rows involved in duplicate model mappings:",
        len(duplicate_rows),
    )
    print(
        "  model genes absent from source:",
        len(missing_model_genes),
    )

    print()
    print("Mapping methods")
    print(
        mapping[
            "mapping_method"
        ]
        .value_counts()
        .head(20)
        .to_string()
    )

    if not conflicts.empty:
        print()
        print("Identifier conflicts")
        print(
            conflicts[
                [
                    "source_row",
                    "GeneId",
                    "GeneName",
                    "candidate_loci",
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    if not unmapped.empty:
        print()
        print("First 30 unmapped rows")
        print(
            unmapped[
                [
                    "source_row",
                    "GeneId",
                    "GeneName",
                    "mapping_method",
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    if not duplicate_rows.empty:
        print()
        print("First 30 duplicate model mappings")
        print(
            duplicate_rows[
                [
                    "source_row",
                    "GeneId",
                    "GeneName",
                    "mapped_model_gene",
                    "mapping_method",
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Build aligned FPKM matrix only when mapping is unambiguous.
    # ------------------------------------------------------------

    fpkm_columns = [
        col
        for col in raw.columns
        if str(col).endswith(
            "_FPKM"
        )
    ]

    if len(fpkm_columns) != 6:
        raise RuntimeError(
            "Expected six FPKM columns."
        )

    if (
        len(conflicts) == 0
        and len(duplicate_rows) == 0
    ):
        source_rows = (
            mapped[
                "source_row"
            ]
            .astype(int)
            .tolist()
        )

        aligned = (
            raw.loc[
                source_rows,
                fpkm_columns,
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        aligned.index = (
            mapped[
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
            / "GSE299861_model_aligned_FPKM.csv"
        )

        print()
        print(
            "Model-aligned FPKM matrix:",
            aligned.shape,
        )

    else:
        print()
        print(
            "Aligned FPKM matrix NOT written because "
            "mapping remains ambiguous."
        )

    # ------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------

    if (
        coverage >= 0.95
        and len(conflicts) == 0
        and len(duplicate_rows) == 0
    ):
        status = "HIGH_COVERAGE_UNAMBIGUOUS"

    elif (
        coverage >= 0.90
        and len(conflicts) == 0
    ):
        status = "USABLE_BUT_DUPLICATES_REQUIRE_REVIEW"

    else:
        status = "FURTHER_MAPPING_REVIEW_REQUIRED"

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_12A_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
