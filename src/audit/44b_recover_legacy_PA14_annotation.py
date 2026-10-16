from pathlib import Path
import hashlib
import subprocess

import pandas as pd

try:
    from Bio import SeqIO
except ImportError as exc:
    raise RuntimeError(
        "Biopython is required. Install with: "
        "conda install -c conda-forge biopython"
    ) from exc


LEGACY_DIR = Path(
    "references/mappings/SRP009257/PA14_legacy"
)

LEGACY_GB = (
    LEGACY_DIR
    / "CP000438.1.gb"
)

LEGACY_SAF = (
    LEGACY_DIR
    / "CP000438.1_legacy_PA14_locus_tags.saf"
)

CURRENT_FASTA = Path(
    "references/mappings/GSE191103/PA14/"
    "NC_008463.1.fasta"
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

MISSING_FILE = Path(
    "results/audit/expanded_biofilm/SRP009257/"
    "finalization/SRP009257_missing_frozen_genes.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/SRP009257/"
    "legacy_annotation_rescue"
)

LEGACY_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 44b: "
    "recover historical PA14 annotation from CP000438.1"
)
print("=" * 100)


# ============================================================
# Download original PA14 GenBank annotation
# ============================================================

if not LEGACY_GB.exists():

    url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        "efetch.fcgi?db=nuccore&id=CP000438.1"
        "&rettype=gbwithparts&retmode=text"
    )

    print()
    print("Downloading CP000438.1 GenBank record...")

    subprocess.run(
        [
            "curl",
            "-4",
            "-L",
            "--fail",
            "--retry",
            "3",
            url,
            "-o",
            str(LEGACY_GB),
        ],
        check=True,
    )

else:

    print()
    print(
        "Legacy GenBank already present: "
        + str(LEGACY_GB)
    )


if not LEGACY_GB.exists():
    raise RuntimeError(
        "Legacy GenBank download failed."
    )

if LEGACY_GB.stat().st_size == 0:
    raise RuntimeError(
        "Legacy GenBank file is empty."
    )


# ============================================================
# Verify legacy and current PA14 sequences are identical
# ============================================================

legacy_record = SeqIO.read(
    LEGACY_GB,
    "genbank",
)

current_record = SeqIO.read(
    CURRENT_FASTA,
    "fasta",
)

legacy_sequence = str(
    legacy_record.seq
).upper()

current_sequence = str(
    current_record.seq
).upper()

legacy_md5 = hashlib.md5(
    legacy_sequence.encode()
).hexdigest()

current_md5 = hashlib.md5(
    current_sequence.encode()
).hexdigest()

same_sequence = (
    legacy_sequence
    == current_sequence
)


print()
print("=" * 100)
print("SEQUENCE COMPATIBILITY")
print("=" * 100)

print(
    "Legacy accession: "
    + str(legacy_record.id)
)

print(
    "Legacy length: "
    + format(
        len(legacy_sequence),
        ",",
    )
)

print(
    "Current accession: "
    + str(current_record.id)
)

print(
    "Current length: "
    + format(
        len(current_sequence),
        ",",
    )
)

print(
    "Exact same nucleotide sequence: "
    + str(same_sequence)
)

print(
    "Legacy MD5: "
    + legacy_md5
)

print(
    "Current MD5: "
    + current_md5
)


if not same_sequence:
    raise RuntimeError(
        "Legacy and current PA14 sequences are not identical. "
        "Do not reuse existing BAM coordinates."
    )


# ============================================================
# Recover historical PA14 locus tags
# ============================================================

feature_priority = {
    "gene": 0,
    "CDS": 1,
    "rRNA": 1,
    "tRNA": 1,
    "ncRNA": 1,
    "tmRNA": 1,
}

features_by_locus = {}


for feature in legacy_record.features:

    if feature.type not in feature_priority:
        continue

    locus_values = feature.qualifiers.get(
        "locus_tag",
        [],
    )

    if len(locus_values) != 1:
        continue

    locus_tag = str(
        locus_values[0]
    ).strip()

    if not locus_tag.startswith(
        "PA14_"
    ):
        continue

    priority = feature_priority[
        feature.type
    ]

    previous = features_by_locus.get(
        locus_tag
    )

    if previous is not None:

        if priority >= previous[
            "priority"
        ]:
            continue

    start = int(
        feature.location.start
    ) + 1

    end = int(
        feature.location.end
    )

    strand_value = feature.location.strand

    if strand_value == 1:
        strand = "+"

    elif strand_value == -1:
        strand = "-"

    else:
        strand = "."

    gene_values = feature.qualifiers.get(
        "gene",
        [],
    )

    product_values = feature.qualifiers.get(
        "product",
        [],
    )

    gene_name = (
        str(gene_values[0])
        if gene_values
        else ""
    )

    product = (
        str(product_values[0])
        if product_values
        else ""
    )

    features_by_locus[
        locus_tag
    ] = {
        "priority": priority,
        "feature_type": feature.type,
        "start": start,
        "end": end,
        "strand": strand,
        "gene": gene_name,
        "product": product,
    }


records = []


for locus_tag, info in features_by_locus.items():

    records.append(
        {
            "pa14_locus_tag": locus_tag,
            "feature_type": info["feature_type"],
            "start": info["start"],
            "end": info["end"],
            "strand": info["strand"],
            "gene": info["gene"],
            "product": info["product"],
        }
    )


legacy_features = pd.DataFrame(
    records
).sort_values(
    [
        "start",
        "end",
        "pa14_locus_tag",
    ]
).reset_index(
    drop=True
)


print()
print("=" * 100)
print("LEGACY ANNOTATION")
print("=" * 100)

print(
    "Unique historical PA14 locus tags recovered: "
    + str(
        len(legacy_features)
    )
)

print()
print(
    legacy_features[
        "feature_type"
    ]
    .value_counts()
    .to_string()
)


# ============================================================
# Build legacy SAF
#
# Coordinates come from CP000438.1.
# Chromosome name must match existing NC_008463.1 BAMs.
# ============================================================

saf = pd.DataFrame(
    {
        "GeneID":
            legacy_features[
                "pa14_locus_tag"
            ],

        "Chr":
            current_record.id,

        "Start":
            legacy_features[
                "start"
            ],

        "End":
            legacy_features[
                "end"
            ],

        "Strand":
            legacy_features[
                "strand"
            ],
    }
)


saf.to_csv(
    LEGACY_SAF,
    sep="\t",
    index=False,
)


legacy_features.to_csv(
    OUT_DIR
    / "CP000438.1_legacy_feature_table.csv",
    index=False,
)


# ============================================================
# Frozen gene coverage
# ============================================================

mapping = pd.read_csv(
    MAPPING_FILE
)

frozen = pd.read_csv(
    FROZEN_FILE
)

signature = pd.read_csv(
    SIGNATURE_FILE
)


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


frozen_map = pd.DataFrame(
    {
        "canonical_pao1":
            frozen_genes
    }
).merge(
    mapping[
        [
            "canonical_pao1",
            "pa14_locus_tag",
        ]
    ],
    on="canonical_pao1",
    how="left",
    validate="one_to_one",
)


legacy_tags = set(
    legacy_features[
        "pa14_locus_tag"
    ]
)


frozen_map[
    "present_in_legacy_annotation"
] = (
    frozen_map[
        "pa14_locus_tag"
    ]
    .isin(
        legacy_tags
    )
)


missing_legacy = (
    frozen_map.loc[
        ~frozen_map[
            "present_in_legacy_annotation"
        ]
    ]
    .copy()
)


signature_genes = set(
    signature[
        "canonical_pao1"
    ]
    .astype(str)
)


missing_signature = (
    missing_legacy.loc[
        missing_legacy[
            "canonical_pao1"
        ]
        .isin(
            signature_genes
        )
    ]
    .copy()
)


print()
print("=" * 100)
print("FROZEN COVERAGE USING LEGACY ANNOTATION")
print("=" * 100)

print(
    "Frozen genes expected: "
    + str(
        len(
            frozen_genes
        )
    )
)

print(
    "Frozen genes present in legacy annotation: "
    + str(
        int(
            frozen_map[
                "present_in_legacy_annotation"
            ].sum()
        )
    )
)

print(
    "Frozen genes still missing: "
    + str(
        len(
            missing_legacy
        )
    )
)

print(
    "Frozen 251-signature genes still missing: "
    + str(
        len(
            missing_signature
        )
    )
)


# ============================================================
# Specifically test rescue of previous 36
# ============================================================

if MISSING_FILE.exists():

    previous_missing = pd.read_csv(
        MISSING_FILE
    )

    rescue = previous_missing.merge(
        legacy_features,
        on="pa14_locus_tag",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    rescue[
        "recovered_in_legacy_annotation"
    ] = (
        rescue["_merge"]
        == "both"
    )

    rescue = rescue.drop(
        columns=[
            "_merge"
        ]
    )

    rescue.to_csv(
        OUT_DIR
        / "current_missing_36_legacy_rescue.csv",
        index=False,
    )

    recovered = int(
        rescue[
            "recovered_in_legacy_annotation"
        ].sum()
    )

    print()
    print("=" * 100)
    print(
        "RESCUE OF THE 36 CURRENT-ANNOTATION "
        "MISSING GENES"
    )
    print("=" * 100)

    print(
        "Recovered from legacy annotation: "
        + str(recovered)
        + " / "
        + str(
            len(
                rescue
            )
        )
    )

    print()

    print(
        rescue[
            [
                "canonical_pao1",
                "pa14_locus_tag",
                "in_frozen_251_signature",
                "recovered_in_legacy_annotation",
                "feature_type",
                "start",
                "end",
                "strand",
                "gene",
                "product",
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# Save diagnostics
# ============================================================

frozen_map.to_csv(
    OUT_DIR
    / "frozen_gene_legacy_annotation_coverage.csv",
    index=False,
)


missing_legacy.to_csv(
    OUT_DIR
    / "frozen_genes_missing_from_legacy_annotation.csv",
    index=False,
)


missing_signature.to_csv(
    OUT_DIR
    / "signature_genes_missing_from_legacy_annotation.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 44b COMPLETE")
print("=" * 100)

print(
    "Legacy SAF: "
    + str(
        LEGACY_SAF
    )
)

print()
