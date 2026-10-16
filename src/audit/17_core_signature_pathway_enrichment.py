from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import requests

from scipy.stats import fisher_exact


SIGNATURE_FILE = Path(
    "results/audit/core_biofilm_signature/"
    "core_biofilm_signature_fdr05.csv"
)

BACKGROUND_FILE = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
)

OUT_DIR = Path(
    "results/audit/core_biofilm_biology"
)

CACHE_DIR = OUT_DIR / "kegg_cache"

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Helpers
# ============================================================

def fetch_or_cache(url, filename):

    path = CACHE_DIR / filename

    if path.exists():
        return path.read_text()

    print(f"Downloading: {url}")

    response = requests.get(
        url,
        timeout=60,
    )

    response.raise_for_status()

    path.write_text(
        response.text
    )

    return response.text


def bh_fdr(p_values):

    p_values = np.asarray(
        p_values,
        dtype=float,
    )

    n = len(p_values)

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
        adjusted[::-1]
    )[::-1]

    adjusted = np.clip(
        adjusted,
        0,
        1,
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
# Load CF-State results
# ============================================================

signature = pd.read_csv(
    SIGNATURE_FILE
)

background_df = pd.read_csv(
    BACKGROUND_FILE
)

background = set(
    background_df[
        "canonical_pao1"
    ]
    .astype(str)
    .str.strip()
)


higher = set(
    signature.loc[
        signature["direction"]
        == "higher_in_biofilm",
        "canonical_pao1",
    ]
    .astype(str)
    .str.strip()
)


lower = set(
    signature.loc[
        signature["direction"]
        == "lower_in_biofilm",
        "canonical_pao1",
    ]
    .astype(str)
    .str.strip()
)


all_signature = (
    higher
    | lower
)


print()
print("=" * 100)
print("CF-STATE — Audit 17: biological pathway enrichment")
print("=" * 100)

print(
    f"Background genes:              "
    f"{len(background)}"
)

print(
    f"Core biofilm genes:            "
    f"{len(all_signature)}"
)

print(
    f"Higher in biofilm:             "
    f"{len(higher)}"
)

print(
    f"Lower in biofilm:              "
    f"{len(lower)}"
)


# ============================================================
# Download PAO1 KEGG pathway mappings
# ============================================================

gene_pathway_text = fetch_or_cache(
    "https://rest.kegg.jp/link/pathway/pae",
    "pae_gene_pathway.tsv",
)

pathway_name_text = fetch_or_cache(
    "https://rest.kegg.jp/list/pathway/pae",
    "pae_pathway_names.tsv",
)


# ============================================================
# Parse pathway names
# ============================================================

pathway_names = {}

for line in pathway_name_text.splitlines():

    if not line.strip():
        continue

    parts = line.split(
        "\t",
        1,
    )

    if len(parts) != 2:
        continue

    pathway_id = (
        parts[0]
        .replace(
            "path:",
            "",
        )
        .strip()
    )

    pathway_name = (
        parts[1]
        .replace(
            " - Pseudomonas aeruginosa PAO1",
            "",
        )
        .strip()
    )

    pathway_names[
        pathway_id
    ] = pathway_name


# ============================================================
# Parse gene ↔ pathway mapping
# ============================================================

pathway_to_genes = defaultdict(
    set
)

for line in gene_pathway_text.splitlines():

    if not line.strip():
        continue

    parts = line.split(
        "\t"
    )

    if len(parts) != 2:
        continue

    left, right = parts

    # Normally:
    # pae:PA0001    path:paeXXXXX

    if left.startswith("pae:"):
        gene = left.replace(
            "pae:",
            "",
        )

        pathway = right.replace(
            "path:",
            "",
        )

    elif right.startswith("pae:"):
        gene = right.replace(
            "pae:",
            "",
        )

        pathway = left.replace(
            "path:",
            "",
        )

    else:
        continue

    pathway_to_genes[
        pathway
    ].add(
        gene
    )


print()
print(
    f"KEGG pathways retrieved:       "
    f"{len(pathway_to_genes)}"
)


mapped_background = set()

for genes in pathway_to_genes.values():

    mapped_background |= (
        genes
        & background
    )


print(
    f"Background genes with >=1 "
    f"KEGG pathway: "
    f"{len(mapped_background)}"
)


# ============================================================
# Annotation lookup for output
# ============================================================

gene_name_map = dict(
    zip(
        signature[
            "canonical_pao1"
        ],
        signature[
            "gene_name"
        ].fillna(""),
    )
)


# ============================================================
# Enrichment
# ============================================================

def run_enrichment(
    selected_genes,
    label,
):

    selected_genes = (
        set(selected_genes)
        & background
    )

    N = len(background)
    n = len(selected_genes)

    records = []

    for pathway, pathway_genes in (
        pathway_to_genes.items()
    ):

        pathway_background = (
            pathway_genes
            & background
        )

        K = len(
            pathway_background
        )

        # Avoid extremely tiny pathways
        if K < 5:
            continue

        overlap = (
            selected_genes
            & pathway_background
        )

        a = len(overlap)

        if a == 0:
            continue

        # 2 x 2 table:
        #
        #                   pathway   not pathway
        # selected              a          b
        # not selected          c          d

        b = n - a
        c = K - a
        d = N - a - b - c

        odds_ratio, p_value = (
            fisher_exact(
                [
                    [a, b],
                    [c, d],
                ],
                alternative="greater",
            )
        )

        expected = (
            n
            * K
            / N
        )

        fold_enrichment = (
            a / expected
            if expected > 0
            else np.nan
        )

        overlap_sorted = sorted(
            overlap
        )

        annotated_overlap = []

        for gene in overlap_sorted:

            gene_name = (
                gene_name_map.get(
                    gene,
                    "",
                )
            )

            if gene_name:
                annotated_overlap.append(
                    f"{gene}({gene_name})"
                )
            else:
                annotated_overlap.append(
                    gene
                )

        records.append(
            {
                "signature":
                    label,

                "pathway_id":
                    pathway,

                "pathway_name":
                    pathway_names.get(
                        pathway,
                        pathway,
                    ),

                "signature_size":
                    n,

                "pathway_background_size":
                    K,

                "overlap":
                    a,

                "expected_overlap":
                    expected,

                "fold_enrichment":
                    fold_enrichment,

                "odds_ratio":
                    odds_ratio,

                "p_value":
                    p_value,

                "genes":
                    ";".join(
                        annotated_overlap
                    ),
            }
        )

    result = pd.DataFrame(
        records
    )

    if result.empty:
        return result

    result[
        "fdr_q_value"
    ] = bh_fdr(
        result[
            "p_value"
        ].to_numpy()
    )

    result = result.sort_values(
        [
            "fdr_q_value",
            "p_value",
            "fold_enrichment",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    return result


# ============================================================
# Run separately
# ============================================================

results = {}

for label, genes in [
    (
        "higher_in_biofilm",
        higher,
    ),
    (
        "lower_in_biofilm",
        lower,
    ),
    (
        "all_core_genes",
        all_signature,
    ),
]:

    result = run_enrichment(
        genes,
        label,
    )

    results[
        label
    ] = result

    result.to_csv(
        OUT_DIR
        / f"kegg_enrichment_{label}.csv",
        index=False,
    )


# ============================================================
# Print useful results
# ============================================================

for label in [
    "higher_in_biofilm",
    "lower_in_biofilm",
]:

    result = results[
        label
    ]

    print()
    print("=" * 100)
    print(
        label.upper()
    )
    print("=" * 100)

    if result.empty:

        print(
            "No mapped pathways."
        )

        continue

    significant = result[
        result["fdr_q_value"]
        <= 0.05
    ]

    print(
        f"Pathways at FDR <= 0.05: "
        f"{len(significant)}"
    )

    print()

    display = (
        significant
        if len(significant)
        else result.head(15)
    )

    cols = [
        "pathway_id",
        "pathway_name",
        "overlap",
        "pathway_background_size",
        "fold_enrichment",
        "p_value",
        "fdr_q_value",
        "genes",
    ]

    print(
        display[
            cols
        ]
        .head(25)
        .to_string(
            index=False
        )
    )


# ============================================================
# Save annotated signature
# ============================================================

signature.to_csv(
    OUT_DIR
    / "core_signature_251_genes.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 17 COMPLETE")
print("=" * 100)

print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)

print()
print(
    "IMPORTANT:"
)

print(
    "The enrichment background is the 4,858 genes "
    "that were actually tested in CF-State, "
    "not the full PAO1 genome."
)
print()
