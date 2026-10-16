from pathlib import Path
from io import StringIO
from collections import defaultdict
import hashlib
import json
import urllib.request

import pandas as pd


ROOT = Path.cwd()

FPKM = (
    ROOT
    / "data/raw/phase2/GSE299861"
    / "GSE299861_PAO1_all.fpkm.txt.gz"
)

SOURCE_GB = (
    ROOT
    / "data/raw/phase2/GSE299861"
    / "CP085082.gb"
)

SOURCE_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    "efetch.fcgi?db=nuccore&id=CP085082"
    "&rettype=gbwithparts&retmode=text"
)

CANONICAL_GB = (
    ROOT
    / "references/mappings/NC_002516.2.gb"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)

OUT = (
    ROOT
    / "results/phase2/audit_12b_PAO1DT_to_canonical_crosswalk"
)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def download_source_genome():
    if SOURCE_GB.exists():
        print(
            "CP085082 already present:"
        )
        print(
            " ",
            SOURCE_GB,
        )
        return

    print(
        "Downloading CP085082 GenBank annotation ..."
    )

    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "User-Agent":
                "CF-State Phase2 Audit12b"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        SOURCE_GB.write_bytes(
            response.read()
        )

    print(
        "Downloaded:",
        SOURCE_GB,
    )


def model_gene_ids(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(handle)

    value = raw["M"]

    if isinstance(value, str):
        df = pd.read_json(StringIO(value))
    else:
        df = pd.DataFrame(
            value
        )

    return [
        str(x).strip()
        for x in df.index
    ]


def feature_records(
    genbank_path,
):
    try:
        from Bio import SeqIO
    except ImportError:
        raise RuntimeError(
            "Biopython is required."
        )

    record = SeqIO.read(
        str(genbank_path),
        "genbank",
    )

    # Keep one best representation per locus.
    priority = {
        "CDS": 5,
        "tRNA": 4,
        "rRNA": 4,
        "ncRNA": 4,
        "gene": 1,
    }

    records = {}

    for feature in record.features:
        if feature.type not in priority:
            continue

        locus_values = feature.qualifiers.get(
            "locus_tag",
            [],
        )

        if not locus_values:
            continue

        locus = str(
            locus_values[0]
        ).strip()

        current = records.get(
            locus
        )

        if (
            current is not None
            and current["priority"]
            >= priority[
                feature.type
            ]
        ):
            continue

        gene_values = feature.qualifiers.get(
            "gene",
            [],
        )

        gene = (
            str(
                gene_values[0]
            ).strip()
            if gene_values
            else ""
        )

        translation_values = (
            feature.qualifiers.get(
                "translation",
                [],
            )
        )

        protein = (
            str(
                translation_values[0]
            )
            .replace(
                " ",
                "",
            )
            .replace(
                "\n",
                "",
            )
            if translation_values
            else ""
        )

        nucleotide = str(
            feature.extract(
                record.seq
            )
        ).upper()

        records[
            locus
        ] = {
            "locus":
                locus,

            "feature_type":
                feature.type,

            "gene":
                gene,

            "protein":
                protein,

            "nucleotide":
                nucleotide,

            "start":
                int(
                    feature.location.start
                )
                + 1,

            "end":
                int(
                    feature.location.end
                ),

            "strand":
                feature.location.strand,

            "priority":
                priority[
                    feature.type
                ],
        }

    return (
        record,
        records,
    )


def unique_lookup(
    records,
    key,
    allowed=None,
):
    index = defaultdict(
        set
    )

    for locus, row in records.items():
        if (
            allowed is not None
            and locus not in allowed
        ):
            continue

        value = row[
            key
        ]

        if not value:
            continue

        if key == "gene":
            value = value.lower()

        index[
            value
        ].add(
            locus
        )

    return {
        value: next(
            iter(
                loci
            )
        )
        for value, loci
        in index.items()
        if len(
            loci
        ) == 1
    }


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 12b")
    print("PAO1DT-to-canonical PAO1 cross-annotation mapping")
    print("=" * 78)

    for path in [
        FPKM,
        CANONICAL_GB,
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

    download_source_genome()

    print()
    print("Frozen CP085082 source")
    print(
        "  size:",
        SOURCE_GB.stat().st_size,
    )
    print(
        "  SHA256:",
        sha256(
            SOURCE_GB
        ),
    )

    # ------------------------------------------------------------
    # Source expression IDs
    # ------------------------------------------------------------

    fpkm = pd.read_csv(
        FPKM,
        sep="\t",
        compression="gzip",
    )

    source_ids = (
        fpkm[
            "GeneId"
        ]
        .astype(str)
        .str.strip()
    )

    print()
    print("GSE299861 expression identifiers")
    print(
        "  rows:",
        len(
            source_ids
        ),
    )
    print(
        "  unique GeneIds:",
        source_ids.nunique(),
    )
    print(
        "  PAO1DT prefix:",
        int(
            source_ids.str.startswith(
                "PAO1DT_"
            ).sum()
        ),
    )

    # ------------------------------------------------------------
    # Load both genome annotations
    # ------------------------------------------------------------

    (
        source_record,
        source_features,
    ) = feature_records(
        SOURCE_GB
    )

    (
        canonical_record,
        canonical_features,
    ) = feature_records(
        CANONICAL_GB
    )

    model_ids = model_gene_ids(
        MODEL
    )

    model_set = set(
        model_ids
    )

    source_loci = set(
        source_features
    )

    print()
    print("CP085082 annotation")
    print(
        "  accession:",
        source_record.id,
    )
    print(
        "  genome length:",
        len(
            source_record.seq
        ),
    )
    print(
        "  distinct locus tags:",
        len(
            source_loci
        ),
    )
    print(
        "  PAO1DT locus tags:",
        sum(
            locus.startswith(
                "PAO1DT_"
            )
            for locus in source_loci
        ),
    )

    expression_in_annotation = (
        source_ids.isin(
            source_loci
        )
    )

    print(
        "  expression GeneIds found exactly in CP085082:",
        str(
            int(
                expression_in_annotation.sum()
            )
        )
        + "/"
        + str(
            len(
                source_ids
            )
        ),
    )

    if (
        expression_in_annotation.mean()
        < 0.90
    ):
        print()
        print(
            "First expression IDs absent from CP085082:"
        )

        for value in source_ids.loc[
            ~expression_in_annotation
        ].head(
            30
        ):
            print(
                " ",
                value,
            )

        raise RuntimeError(
            "CP085082 does not explain at least 90% "
            "of the PAO1DT expression identifiers."
        )

    # ------------------------------------------------------------
    # Canonical lookup tables
    # ------------------------------------------------------------

    canonical_protein = unique_lookup(
        canonical_features,
        "protein",
        allowed=model_set,
    )

    canonical_nucleotide = unique_lookup(
        canonical_features,
        "nucleotide",
        allowed=model_set,
    )

    canonical_gene = unique_lookup(
        canonical_features,
        "gene",
        allowed=model_set,
    )

    # ------------------------------------------------------------
    # Crosswalk each PAO1DT locus
    # ------------------------------------------------------------

    rows = []

    for source_id in source_ids:
        source = source_features.get(
            source_id
        )

        if source is None:
            rows.append(
                {
                    "source_locus":
                        source_id,

                    "source_gene":
                        "",

                    "mapped_model_gene":
                        None,

                    "mapping_method":
                        "source_locus_missing_from_CP085082",

                    "candidate_targets":
                        "",
                }
            )

            continue

        candidates = []

        protein = source[
            "protein"
        ]

        nucleotide = source[
            "nucleotide"
        ]

        gene = source[
            "gene"
        ]

        if (
            protein
            and protein
            in canonical_protein
        ):
            candidates.append(
                (
                    canonical_protein[
                        protein
                    ],
                    "unique_exact_protein",
                )
            )

        if (
            nucleotide
            and nucleotide
            in canonical_nucleotide
        ):
            candidates.append(
                (
                    canonical_nucleotide[
                        nucleotide
                    ],
                    "unique_exact_nucleotide",
                )
            )

        if (
            gene
            and gene.lower()
            in canonical_gene
        ):
            candidates.append(
                (
                    canonical_gene[
                        gene.lower()
                    ],
                    "unique_gene_symbol",
                )
            )

        targets = sorted(
            set(
                target
                for target, method
                in candidates
            )
        )

        if len(
            targets
        ) == 1:
            mapped = targets[
                0
            ]

            methods = sorted(
                set(
                    method
                    for target, method
                    in candidates
                    if target == mapped
                )
            )

            method = ";".join(
                methods
            )

        elif len(
            targets
        ) == 0:
            mapped = None
            method = "no_exact_cross_annotation_match"

        else:
            mapped = None
            method = "conflicting_cross_annotation_matches"

        rows.append(
            {
                "source_locus":
                    source_id,

                "source_gene":
                    gene,

                "source_feature_type":
                    source[
                        "feature_type"
                    ],

                "source_start":
                    source[
                        "start"
                    ],

                "source_end":
                    source[
                        "end"
                    ],

                "mapped_model_gene":
                    mapped,

                "mapping_method":
                    method,

                "candidate_targets":
                    ";".join(
                        targets
                    ),
            }
        )

    crosswalk = pd.DataFrame(
        rows
    )

    crosswalk.to_csv(
        OUT
        / "PAO1DT_to_canonical_crosswalk.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # QC
    # ------------------------------------------------------------

    mapped = crosswalk.loc[
        crosswalk[
            "mapped_model_gene"
        ].notna()
    ].copy()

    unresolved = crosswalk.loc[
        crosswalk[
            "mapped_model_gene"
        ].isna()
    ].copy()

    duplicate_targets = mapped.loc[
        mapped[
            "mapped_model_gene"
        ].duplicated(
            keep=False
        )
    ].copy()

    unique_targets = set(
        mapped[
            "mapped_model_gene"
        ]
    )

    coverage = (
        len(
            unique_targets
        )
        / len(
            model_ids
        )
    )

    print()
    print("Exact cross-annotation mapping")
    print(
        "  mapped expression rows:",
        len(
            mapped
        ),
    )
    print(
        "  unique canonical model genes:",
        len(
            unique_targets
        ),
    )
    print(
        "  aeruPRECISE model coverage:",
        format(
            coverage,
            ".2%",
        ),
    )
    print(
        "  unresolved expression rows:",
        len(
            unresolved
        ),
    )
    print(
        "  rows involved in duplicate canonical targets:",
        len(
            duplicate_targets
        ),
    )

    print()
    print("Mapping methods")
    print(
        crosswalk[
            "mapping_method"
        ]
        .value_counts()
        .head(
            20
        )
        .to_string()
    )

    if not unresolved.empty:
        print()
        print("First unresolved PAO1DT loci")
        print(
            unresolved[
                [
                    "source_locus",
                    "source_gene",
                    "source_feature_type",
                    "source_start",
                    "source_end",
                    "mapping_method",
                ]
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    if not duplicate_targets.empty:
        print()
        print("First duplicate canonical targets")
        print(
            duplicate_targets[
                [
                    "source_locus",
                    "source_gene",
                    "mapped_model_gene",
                    "mapping_method",
                ]
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Build aligned FPKM only if mapping is sufficiently clean
    # ------------------------------------------------------------

    fpkm_columns = [
        col
        for col in fpkm.columns
        if str(
            col
        ).endswith(
            "_FPKM"
        )
    ]

    if len(
        fpkm_columns
    ) != 6:
        raise RuntimeError(
            "Expected six FPKM columns."
        )

    if (
        coverage >= 0.90
        and len(
            duplicate_targets
        )
        == 0
    ):
        source_to_target = (
            mapped.set_index(
                "source_locus"
            )[
                "mapped_model_gene"
            ]
            .to_dict()
        )

        usable = fpkm.loc[
            fpkm[
                "GeneId"
            ]
            .astype(str)
            .isin(
                source_to_target
            )
        ].copy()

        aligned = (
            usable[
                fpkm_columns
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        aligned.index = (
            usable[
                "GeneId"
            ]
            .astype(str)
            .map(
                source_to_target
            )
            .values
        )

        if aligned.index.duplicated().any():
            raise RuntimeError(
                "Unexpected duplicate canonical genes "
                "after crosswalk."
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
            "Aligned FPKM matrix NOT written."
        )

    # ------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------

    if (
        coverage >= 0.95
        and len(
            duplicate_targets
        )
        == 0
    ):
        status = (
            "HIGH_COVERAGE_EXACT_CROSSWALK"
        )

    elif (
        coverage >= 0.90
        and len(
            duplicate_targets
        )
        == 0
    ):
        status = (
            "USABLE_EXACT_CROSSWALK"
        )

    else:
        status = (
            "APPROXIMATE_ORTHOLOGY_REQUIRED"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_12B_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
