from pathlib import Path
import re
import subprocess
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import hypergeom


INPUT_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "reversal_biology/"
    "Audit61_all_genes_annotated.csv"
)

ROBUST_FILE = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "reversal_biology/"
    "Audit61_robust_reversed_genes.csv"
)

KEGG_DIR = Path(
    "references/mappings/kegg"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE299861/"
    "reversal_biology/KEGG_enrichment"
)

KEGG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


LINK_FILE = (
    KEGG_DIR
    / "pae_gene_to_pathway.tsv"
)

PATHWAY_FILE = (
    KEGG_DIR
    / "pae_pathways.tsv"
)


LINK_URL = (
    "https://rest.kegg.jp/link/pathway/pae"
)

PATHWAY_URL = (
    "https://rest.kegg.jp/list/pathway/pae"
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 62: "
    "formal KEGG enrichment of GSE299861 reversal"
)
print("=" * 100)


# ============================================================
# Download helper
#
# Cache the KEGG responses locally.
# Only two requests are made if files are absent.
# ============================================================

def download_text(
    url,
    destination,
):

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):

        print(
            "Using cached KEGG file: "
            + str(destination)
        )

        return


    print(
        "Downloading:"
    )

    print(
        url
    )


    request = urllib.request.Request(
        url,
        headers={
            "User-Agent":
                "CF-State academic analysis"
        },
    )


    try:

        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:

            content = response.read()


        if not content:

            raise RuntimeError(
                "Empty KEGG response."
            )


        destination.write_bytes(
            content
        )


    except Exception as first_error:

        print(
            "urllib download failed; trying curl."
        )

        print(
            str(first_error)
        )


        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-delay",
                "2",
                url,
                "-o",
                str(destination),
            ],
            check=True,
        )


    if (
        not destination.exists()
        or destination.stat().st_size == 0
    ):

        raise RuntimeError(
            "Failed to download "
            + url
        )


download_text(
    LINK_URL,
    LINK_FILE,
)

download_text(
    PATHWAY_URL,
    PATHWAY_FILE,
)


# ============================================================
# Parse KEGG PAO1 gene -> pathway links
# ============================================================

link_records = []


with open(
    LINK_FILE,
    "r",
    encoding="utf-8",
) as handle:

    for raw_line in handle:

        line = raw_line.strip()

        if not line:
            continue


        fields = line.split(
            "\t"
        )


        if len(fields) != 2:
            continue


        left = fields[0]
        right = fields[1]


        gene = None
        pathway = None


        if (
            left.startswith("pae:")
            and right.startswith("path:")
        ):

            gene = left.split(
                ":",
                1,
            )[1]

            pathway = right.split(
                ":",
                1,
            )[1]


        elif (
            right.startswith("pae:")
            and left.startswith("path:")
        ):

            gene = right.split(
                ":",
                1,
            )[1]

            pathway = left.split(
                ":",
                1,
            )[1]


        if gene is None:
            continue


        if not re.fullmatch(
            r"PA\d{4}",
            gene,
        ):

            continue


        link_records.append(
            {
                "canonical_pao1":
                    gene,

                "pathway":
                    pathway,
            }
        )


links = pd.DataFrame(
    link_records
)


if links.empty:

    raise RuntimeError(
        "No PAxxxx KEGG links parsed."
    )


links = links.drop_duplicates()


# ============================================================
# Parse pathway names
# ============================================================

pathway_records = []


with open(
    PATHWAY_FILE,
    "r",
    encoding="utf-8",
) as handle:

    for raw_line in handle:

        line = raw_line.strip()

        if not line:
            continue


        fields = line.split(
            "\t",
            1,
        )


        if len(fields) != 2:
            continue


        pathway_id = fields[0]


        if pathway_id.startswith(
            "path:"
        ):

            pathway_id = pathway_id.split(
                ":",
                1,
            )[1]


        title = fields[1]


        suffix = (
            " - Pseudomonas aeruginosa PAO1"
        )


        if title.endswith(
            suffix
        ):

            title = title[
                :-len(suffix)
            ]


        pathway_records.append(
            {
                "pathway":
                    pathway_id,

                "pathway_name":
                    title,
            }
        )


pathway_names = (
    pd.DataFrame(
        pathway_records
    )
    .drop_duplicates(
        subset=[
            "pathway"
        ]
    )
)


print()
print("=" * 100)
print("KEGG ANNOTATION")
print("=" * 100)

print(
    "PAO1 gene-pathway links: "
    + str(
        len(links)
    )
)

print(
    "Unique PAO1 genes linked to >=1 pathway: "
    + str(
        links[
            "canonical_pao1"
        ].nunique()
    )
)

print(
    "PAO1 pathways: "
    + str(
        pathway_names[
            "pathway"
        ].nunique()
    )
)


# ============================================================
# Load Audit 61 universe and robust reversal set
# ============================================================

background = pd.read_csv(
    INPUT_FILE
)

robust = pd.read_csv(
    ROBUST_FILE
)


required_columns = [
    "canonical_pao1",
    "consensus_sign",
    "GSE299861_sign",
    "majority_count_other6",
    "median_effect_other6",
    "GSE299861_published_effect",
]


for column in required_columns:

    if column not in background.columns:

        raise RuntimeError(
            "Missing background column: "
            + column
        )


for column in required_columns:

    if column not in robust.columns:

        raise RuntimeError(
            "Missing robust-set column: "
            + column
        )


background[
    "canonical_pao1"
] = (
    background[
        "canonical_pao1"
    ]
    .astype(str)
)


robust[
    "canonical_pao1"
] = (
    robust[
        "canonical_pao1"
    ]
    .astype(str)
)


background_genes = set(
    background[
        "canonical_pao1"
    ]
)


if len(background_genes) != 4164:

    raise RuntimeError(
        "Expected 4,164 comparison-background genes, found "
        + str(
            len(
                background_genes
            )
        )
    )


if len(robust) != 484:

    raise RuntimeError(
        "Expected 484 robust reversed genes, found "
        + str(
            len(
                robust
            )
        )
    )


# ============================================================
# Define direction-specific reversal sets
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


gene_sets = {
    "ALL_ROBUST_REVERSED":
        set(
            robust[
                "canonical_pao1"
            ]
        ),

    "NORMALLY_BIOFILM_HIGHER_BUT_LOWER_IN_GSE299861":
        set(
            normally_up_but_down[
                "canonical_pao1"
            ]
        ),

    "NORMALLY_BIOFILM_LOWER_BUT_HIGHER_IN_GSE299861":
        set(
            normally_down_but_up[
                "canonical_pao1"
            ]
        ),
}


print()
print("=" * 100)
print("ENRICHMENT GENE SETS")
print("=" * 100)

print(
    "Comparison background: "
    + str(
        len(
            background_genes
        )
    )
)

for name, genes in gene_sets.items():

    print(
        name
        + ": "
        + str(
            len(
                genes
            )
        )
    )


# ============================================================
# Restrict KEGG links to the valid 4,164-gene universe
# ============================================================

background_links = links.loc[
    links[
        "canonical_pao1"
    ].isin(
        background_genes
    )
].copy()


kegg_background_genes = set(
    background_links[
        "canonical_pao1"
    ]
)


print()
print(
    "Background genes with >=1 KEGG pathway: "
    + str(
        len(
            kegg_background_genes
        )
    )
)

print(
    "KEGG annotation coverage of comparison universe: "
    + format(
        100.0
        * len(
            kegg_background_genes
        )
        / len(
            background_genes
        ),
        ".2f",
    )
    + "%"
)


# ============================================================
# Benjamini-Hochberg
# ============================================================

def benjamini_hochberg(
    p_values,
):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )


    n = len(
        p_values
    )


    order = np.argsort(
        p_values
    )


    ranked = p_values[
        order
    ]


    adjusted = (
        ranked
        * n
        / np.arange(
            1,
            n + 1,
        )
    )


    adjusted = np.minimum.accumulate(
        adjusted[
            ::-1
        ]
    )[
        ::-1
    ]


    adjusted = np.clip(
        adjusted,
        0.0,
        1.0,
    )


    result = np.empty(
        n,
        dtype=float,
    )


    result[
        order
    ] = adjusted


    return result


# ============================================================
# Enrichment function
#
# Primary universe:
#   all 4,164 comparable genes.
#
# Sensitivity universe:
#   only background genes linked to >=1 KEGG pathway.
# ============================================================

def run_enrichment(
    set_name,
    selected_genes,
    universe_mode,
):

    if universe_mode == "FULL_4164":

        universe = set(
            background_genes
        )

    elif universe_mode == "KEGG_ANNOTATED_ONLY":

        universe = set(
            kegg_background_genes
        )

    else:

        raise RuntimeError(
            "Unknown universe mode."
        )


    selected = (
        set(
            selected_genes
        )
        & universe
    )


    M = len(
        universe
    )

    N = len(
        selected
    )


    records = []


    for pathway in sorted(
        background_links[
            "pathway"
        ].unique()
    ):

        pathway_genes = set(
            background_links.loc[
                background_links[
                    "pathway"
                ]
                == pathway,
                "canonical_pao1",
            ]
        )


        pathway_genes = (
            pathway_genes
            & universe
        )


        K = len(
            pathway_genes
        )


        if K < 5:
            continue


        overlap = (
            selected
            & pathway_genes
        )


        k = len(
            overlap
        )


        p_value = float(
            hypergeom.sf(
                k - 1,
                M,
                K,
                N,
            )
        )


        expected = (
            N
            * K
            / M
        )


        if expected > 0:

            enrichment_ratio = (
                k
                / expected
            )

        else:

            enrichment_ratio = np.nan


        records.append(
            {
                "gene_set":
                    set_name,

                "universe":
                    universe_mode,

                "pathway":
                    pathway,

                "background_size":
                    M,

                "selected_size":
                    N,

                "pathway_background_genes":
                    K,

                "overlap_genes_n":
                    k,

                "expected_overlap":
                    expected,

                "enrichment_ratio":
                    enrichment_ratio,

                "p_value":
                    p_value,

                "overlap_genes":
                    ";".join(
                        sorted(
                            overlap
                        )
                    ),
            }
        )


    result = pd.DataFrame(
        records
    )


    if result.empty:

        return result


    result[
        "fdr"
    ] = benjamini_hochberg(
        result[
            "p_value"
        ].to_numpy()
    )


    result = result.merge(
        pathway_names,
        on="pathway",
        how="left",
        validate="many_to_one",
    )


    result = result.sort_values(
        [
            "fdr",
            "p_value",
            "enrichment_ratio",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


    return result


# ============================================================
# Run all analyses
# ============================================================

all_results = []


for set_name, selected_genes in gene_sets.items():

    for universe_mode in [
        "FULL_4164",
        "KEGG_ANNOTATED_ONLY",
    ]:

        result = run_enrichment(
            set_name,
            selected_genes,
            universe_mode,
        )


        if not result.empty:

            all_results.append(
                result
            )


if not all_results:

    raise RuntimeError(
        "No KEGG enrichment results produced."
    )


enrichment = pd.concat(
    all_results,
    ignore_index=True,
)


enrichment.to_csv(
    OUT_DIR
    / "Audit62_all_KEGG_enrichment_results.csv",
    index=False,
)


# ============================================================
# Primary results
# ============================================================

primary = enrichment.loc[
    enrichment[
        "universe"
    ]
    == "FULL_4164"
].copy()


primary.to_csv(
    OUT_DIR
    / "Audit62_primary_FULL4164_results.csv",
    index=False,
)


# ============================================================
# Print direction-specific top pathways
# ============================================================

print()
print("=" * 100)
print("PRIMARY KEGG ENRICHMENT — FULL 4,164-GENE BACKGROUND")
print("=" * 100)


display_columns = [
    "pathway",
    "pathway_name",
    "selected_size",
    "pathway_background_genes",
    "overlap_genes_n",
    "expected_overlap",
    "enrichment_ratio",
    "p_value",
    "fdr",
    "overlap_genes",
]


for set_name in gene_sets:

    subset = primary.loc[
        primary[
            "gene_set"
        ]
        == set_name
    ].copy()


    print()
    print("-" * 100)
    print(
        set_name
    )
    print("-" * 100)


    if subset.empty:

        print(
            "No pathways tested."
        )

        continue


    print(
        subset[
            display_columns
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )


# ============================================================
# Significant pathways
# ============================================================

significant = primary.loc[
    (
        primary[
            "fdr"
        ]
        <= 0.05
    )
    &
    (
        primary[
            "overlap_genes_n"
        ]
        >= 2
    )
].copy()


significant.to_csv(
    OUT_DIR
    / "Audit62_significant_KEGG_FDR05.csv",
    index=False,
)


print()
print("=" * 100)
print("SIGNIFICANT KEGG PATHWAYS — FDR <= 0.05")
print("=" * 100)


if significant.empty:

    print(
        "None."
    )

else:

    print(
        significant[
            [
                "gene_set",
                "pathway",
                "pathway_name",
                "overlap_genes_n",
                "enrichment_ratio",
                "p_value",
                "fdr",
                "overlap_genes",
            ]
        ]
        .to_string(
            index=False
        )
    )


# ============================================================
# Sensitivity comparison
# ============================================================

comparison_records = []


for set_name in gene_sets:

    full = enrichment.loc[
        (
            enrichment[
                "gene_set"
            ]
            == set_name
        )
        &
        (
            enrichment[
                "universe"
            ]
            == "FULL_4164"
        )
    ].copy()


    annotated = enrichment.loc[
        (
            enrichment[
                "gene_set"
            ]
            == set_name
        )
        &
        (
            enrichment[
                "universe"
            ]
            == "KEGG_ANNOTATED_ONLY"
        )
    ].copy()


    full_sig = set(
        full.loc[
            full[
                "fdr"
            ]
            <= 0.05,
            "pathway",
        ]
    )


    annotated_sig = set(
        annotated.loc[
            annotated[
                "fdr"
            ]
            <= 0.05,
            "pathway",
        ]
    )


    comparison_records.append(
        {
            "gene_set":
                set_name,

            "significant_FULL4164":
                len(
                    full_sig
                ),

            "significant_KEGG_ANNOTATED_ONLY":
                len(
                    annotated_sig
                ),

            "significant_in_both":
                len(
                    full_sig
                    & annotated_sig
                ),

            "FULL4164_only":
                ";".join(
                    sorted(
                        full_sig
                        - annotated_sig
                    )
                ),

            "ANNOTATED_ONLY_only":
                ";".join(
                    sorted(
                        annotated_sig
                        - full_sig
                    )
                ),
        }
    )


sensitivity = pd.DataFrame(
    comparison_records
)


sensitivity.to_csv(
    OUT_DIR
    / "Audit62_background_sensitivity_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("BACKGROUND SENSITIVITY")
print("=" * 100)

print(
    sensitivity.to_string(
        index=False
    )
)


# ============================================================
# Save KEGG annotation restricted to comparison universe
# ============================================================

background_links = background_links.merge(
    pathway_names,
    on="pathway",
    how="left",
    validate="many_to_one",
)


background_links.to_csv(
    OUT_DIR
    / "Audit62_KEGG_links_4164_background.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 62 COMPLETE")
print("=" * 100)

print(
    "Primary enrichment universe: "
    "4,164 high-confidence comparable genes."
)

print(
    "A KEGG-annotated-only universe was also analysed "
    "as a sensitivity check."
)

print()

print(
    "Outputs saved to:"
)

print(
    OUT_DIR
)

print()
