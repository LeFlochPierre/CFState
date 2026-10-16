from pathlib import Path
from collections import Counter
import requests
import pandas as pd

from Bio import SeqIO


RAW_DIR = Path(
    "data/raw/expanded_biofilm/GSE299861"
)

REF_DIR = Path(
    "references/mappings/GSE299861"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861"
)

REF_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Exact genomes sequenced by the study.
#
# CP085082 = PAO1 subline
# CP079712 = MPAO1 subline

GENOMES = {
    "PAO1": "CP085082",
    "MPAO1": "CP079712",
}

EXPRESSION_FILES = {
    "PAO1":
        "GSE299861_PAO1_all.fpkm.txt.gz",

    "MPAO1":
        "GSE299861_MPAO1_all.fpkm.txt.gz",
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 23: "
    "recover GSE299861 source-gene annotations"
)
print("=" * 100)


# ============================================================
# Download exact GenBank records
# ============================================================

for strain, accession in GENOMES.items():

    outfile = (
        REF_DIR
        / f"{accession}.gb"
    )

    if not outfile.exists():

        url = (
            "https://eutils.ncbi.nlm.nih.gov/"
            "entrez/eutils/efetch.fcgi"
            f"?db=nuccore&id={accession}"
            "&rettype=gbwithparts"
            "&retmode=text"
        )

        print()
        print(
            f"Downloading {strain} genome "
            f"{accession}"
        )

        response = requests.get(
            url,
            timeout=120,
        )

        response.raise_for_status()

        outfile.write_text(
            response.text
        )

        print(
            f"Saved: {outfile}"
        )

    else:

        print()
        print(
            f"Already present: {outfile}"
        )


# ============================================================
# Helper: flatten GenBank qualifier values
# ============================================================

def qualifier_values(feature):

    records = []

    for key, values in (
        feature.qualifiers.items()
    ):

        for value in values:

            records.append(
                (
                    key,
                    str(value).strip(),
                )
            )

    return records


# ============================================================
# Inspect each strain
# ============================================================

overall_records = []


for strain, accession in GENOMES.items():

    print()
    print("=" * 100)
    print(
        f"{strain} — {accession}"
    )
    print("=" * 100)


    # --------------------------------------------------------
    # Expression matrix
    # --------------------------------------------------------

    expression_file = (
        RAW_DIR
        / EXPRESSION_FILES[
            strain
        ]
    )

    expr = pd.read_csv(
        expression_file,
        sep="\t",
        compression="gzip",
    )

    source_gene_ids = (
        expr["GeneId"]
        .astype(str)
        .str.strip()
    )

    source_gene_names = (
        expr["GeneName"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    source_id_set = set(
        source_gene_ids
    )

    source_name_set = set(
        source_gene_names
    )


    print(
        f"Expression rows:           "
        f"{len(expr)}"
    )

    print(
        f"Unique GeneId values:      "
        f"{source_gene_ids.nunique()}"
    )

    print(
        f"Unique GeneName values:    "
        f"{source_gene_names.nunique()}"
    )


    # --------------------------------------------------------
    # GenBank
    # --------------------------------------------------------

    gb_file = (
        REF_DIR
        / f"{accession}.gb"
    )

    record = SeqIO.read(
        gb_file,
        "genbank",
    )


    print()
    print(
        f"Genome length:             "
        f"{len(record.seq):,} bp"
    )


    feature_type_counts = Counter(
        feature.type
        for feature in record.features
    )

    print()
    print(
        "Feature type counts:"
    )

    for feature_type, count in (
        feature_type_counts.most_common()
    ):

        print(
            f"  {feature_type:15s} "
            f"{count}"
        )


    # --------------------------------------------------------
    # Build annotation table
    # --------------------------------------------------------

    annotation_records = []

    all_qualifier_values = set()

    qualifier_key_counts = Counter()


    for feature_index, feature in enumerate(
        record.features
    ):

        if feature.type not in {
            "gene",
            "CDS",
            "rRNA",
            "tRNA",
            "ncRNA",
            "tmRNA",
        }:
            continue


        qualifiers = feature.qualifiers


        locus_tag = (
            qualifiers.get(
                "locus_tag",
                [""],
            )[0]
        )

        gene = (
            qualifiers.get(
                "gene",
                [""],
            )[0]
        )

        old_locus_tag = (
            qualifiers.get(
                "old_locus_tag",
                [""],
            )[0]
        )

        protein_id = (
            qualifiers.get(
                "protein_id",
                [""],
            )[0]
        )

        product = (
            qualifiers.get(
                "product",
                [""],
            )[0]
        )


        q_values = (
            qualifier_values(
                feature
            )
        )

        for key, value in q_values:

            qualifier_key_counts[
                key
            ] += 1

            all_qualifier_values.add(
                value
            )


        annotation_records.append(
            {
                "feature_index":
                    feature_index,

                "feature_type":
                    feature.type,

                "start":
                    int(
                        feature.location.start
                    ),

                "end":
                    int(
                        feature.location.end
                    ),

                "strand":
                    feature.location.strand,

                "locus_tag":
                    locus_tag,

                "gene":
                    gene,

                "old_locus_tag":
                    old_locus_tag,

                "protein_id":
                    protein_id,

                "product":
                    product,
            }
        )


    annotation = pd.DataFrame(
        annotation_records
    )


    annotation.to_csv(
        OUT_DIR
        / f"{strain}_study_genome_annotation.csv",
        index=False,
    )


    # --------------------------------------------------------
    # Identifier overlaps
    # --------------------------------------------------------

    locus_tags = set(
        annotation[
            "locus_tag"
        ].astype(str)
    )

    old_locus_tags = set(
        annotation[
            "old_locus_tag"
        ].astype(str)
    )

    genes = set(
        annotation[
            "gene"
        ].astype(str)
    )

    protein_ids = set(
        annotation[
            "protein_id"
        ].astype(str)
    )


    direct_locus_overlap = (
        source_id_set
        & locus_tags
    )

    old_locus_overlap = (
        source_id_set
        & old_locus_tags
    )

    gene_overlap = (
        source_id_set
        & genes
    )

    protein_overlap = (
        source_id_set
        & protein_ids
    )

    any_qualifier_overlap = (
        source_id_set
        & all_qualifier_values
    )


    print()
    print(
        "GeneId overlap with GenBank annotation:"
    )

    print(
        f"  locus_tag:          "
        f"{len(direct_locus_overlap)}"
    )

    print(
        f"  old_locus_tag:      "
        f"{len(old_locus_overlap)}"
    )

    print(
        f"  gene name:          "
        f"{len(gene_overlap)}"
    )

    print(
        f"  protein_id:         "
        f"{len(protein_overlap)}"
    )

    print(
        f"  ANY qualifier:      "
        f"{len(any_qualifier_overlap)}"
    )


    # --------------------------------------------------------
    # GeneName overlap
    # --------------------------------------------------------

    name_locus_overlap = (
        source_name_set
        & locus_tags
    )

    name_gene_overlap = (
        source_name_set
        & genes
    )

    name_any_overlap = (
        source_name_set
        & all_qualifier_values
    )


    print()
    print(
        "GeneName overlap with annotation:"
    )

    print(
        f"  locus_tag:          "
        f"{len(name_locus_overlap)}"
    )

    print(
        f"  gene name:          "
        f"{len(name_gene_overlap)}"
    )

    print(
        f"  ANY qualifier:      "
        f"{len(name_any_overlap)}"
    )


    # --------------------------------------------------------
    # Show locus-tag patterns
    # --------------------------------------------------------

    nonblank_locus = (
        annotation[
            "locus_tag"
        ]
        .fillna("")
        .astype(str)
    )

    nonblank_locus = (
        nonblank_locus[
            nonblank_locus
            != ""
        ]
    )


    print()
    print(
        "First 30 GenBank locus tags:"
    )

    print(
        nonblank_locus
        .drop_duplicates()
        .head(30)
        .to_string(
            index=False
        )
    )


    print()
    print(
        "Most common qualifier keys:"
    )

    for key, count in (
        qualifier_key_counts
        .most_common(20)
    ):

        print(
            f"  {key:20s} "
            f"{count}"
        )


    # --------------------------------------------------------
    # Save expression IDs with mapping status
    # --------------------------------------------------------

    id_audit = pd.DataFrame(
        {
            "GeneId":
                source_gene_ids,

            "GeneName":
                source_gene_names,
        }
    )


    id_audit[
        "matches_locus_tag"
    ] = id_audit[
        "GeneId"
    ].isin(
        locus_tags
    )

    id_audit[
        "matches_old_locus_tag"
    ] = id_audit[
        "GeneId"
    ].isin(
        old_locus_tags
    )

    id_audit[
        "matches_gene_name"
    ] = id_audit[
        "GeneId"
    ].isin(
        genes
    )

    id_audit[
        "matches_protein_id"
    ] = id_audit[
        "GeneId"
    ].isin(
        protein_ids
    )

    id_audit[
        "matches_any_qualifier"
    ] = id_audit[
        "GeneId"
    ].isin(
        all_qualifier_values
    )


    id_audit.to_csv(
        OUT_DIR
        / f"{strain}_source_id_mapping_audit.csv",
        index=False,
    )


    unmatched = id_audit[
        ~id_audit[
            "matches_any_qualifier"
        ]
    ]


    print()
    print(
        f"Unmatched GeneIds after "
        f"all qualifier checks: "
        f"{len(unmatched)}"
    )

    if len(unmatched):

        print()
        print(
            "First 30 unmatched IDs:"
        )

        print(
            unmatched[
                [
                    "GeneId",
                    "GeneName",
                ]
            ]
            .head(30)
            .to_string(
                index=False
            )
        )


    overall_records.append(
        {
            "strain":
                strain,

            "accession":
                accession,

            "expression_rows":
                len(expr),

            "annotation_rows":
                len(annotation),

            "geneid_locus_matches":
                len(
                    direct_locus_overlap
                ),

            "geneid_old_locus_matches":
                len(
                    old_locus_overlap
                ),

            "geneid_any_qualifier_matches":
                len(
                    any_qualifier_overlap
                ),

            "unmatched_geneids":
                len(
                    unmatched
                ),
        }
    )


# ============================================================
# Final summary
# ============================================================

overall = pd.DataFrame(
    overall_records
)

overall.to_csv(
    OUT_DIR
    / "audit23_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 23 SUMMARY")
print("=" * 100)

print(
    overall.to_string(
        index=False
    )
)

print()
print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)
print()
