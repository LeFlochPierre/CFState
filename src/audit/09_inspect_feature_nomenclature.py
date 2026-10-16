from pathlib import Path
from collections import Counter
import re
import pandas as pd


ROOT = Path("external/EPathDash/GEO_Datasets/Pseudomonas")
COUNT_DIR = ROOT / "Count_Tables"

OUT = Path("results/audit/feature_nomenclature")
OUT.mkdir(parents=True, exist_ok=True)


def structure_signature(x):
    """
    Reduce an identifier to a rough character structure.

    Examples:
      PA0001       -> AA0000
      PA14_22460   -> AA00_00000
      dnaA         -> AAAA
    """
    result = []

    for ch in str(x):
        if ch.isalpha():
            result.append("A")
        elif ch.isdigit():
            result.append("0")
        else:
            result.append(ch)

    return "".join(result)


# ============================================================
# GSE136111
# ============================================================

path = COUNT_DIR / "GSE136111_counts.csv"

df = pd.read_csv(
    path,
    encoding="utf-8-sig",
)

gene_col = df.columns[0]

ids = (
    df[gene_col]
    .fillna("")
    .astype(str)
    .str.strip()
)


def broad_class(x):

    if re.fullmatch(r"PA\d{4}", x, re.I):
        return "PAO1_locus"

    if re.fullmatch(
        r"[A-Za-z][A-Za-z0-9_.-]*",
        x
    ):
        return "simple_name"

    return "complex_or_other"


classes = ids.map(broad_class)

audit = pd.DataFrame(
    {
        "identifier": ids,
        "class": classes,
        "signature": ids.map(
            structure_signature
        ),
        "length": ids.str.len(),
    }
)

audit.to_csv(
    OUT / "GSE136111_feature_nomenclature.csv",
    index=False,
)


print()
print("=" * 100)
print("GSE136111 FEATURE NOMENCLATURE")
print("=" * 100)

print()
print("Broad classes:")
print(
    audit["class"]
    .value_counts()
    .to_string()
)


# ------------------------------------------------------------
# Complex identifiers
# ------------------------------------------------------------

complex_ids = audit[
    audit["class"] == "complex_or_other"
]

print()
print("=" * 100)
print("FIRST 150 COMPLEX / OTHER IDENTIFIERS")
print("=" * 100)

print(
    complex_ids[
        ["identifier", "signature"]
    ]
    .head(150)
    .to_string(index=False)
)


# ------------------------------------------------------------
# Most common identifier structures
# ------------------------------------------------------------

print()
print("=" * 100)
print("MOST COMMON IDENTIFIER STRUCTURES")
print("=" * 100)

print(
    audit["signature"]
    .value_counts()
    .head(50)
    .to_string()
)


# ------------------------------------------------------------
# Special characters
# ------------------------------------------------------------

characters = [
    ":",
    "|",
    ";",
    "_",
    "-",
    ".",
    "/",
    "(",
    ")",
    " ",
]

print()
print("=" * 100)
print("SPECIAL CHARACTER COUNTS")
print("=" * 100)

for char in characters:

    n = ids.str.contains(
        re.escape(char),
        regex=True,
    ).sum()

    print(
        f"{repr(char):6s}: {n}"
    )


# ------------------------------------------------------------
# Longest identifiers
# ------------------------------------------------------------

print()
print("=" * 100)
print("LONGEST IDENTIFIERS")
print("=" * 100)

print(
    audit.sort_values(
        "length",
        ascending=False
    )
    .head(100)[
        [
            "identifier",
            "class",
            "signature",
            "length",
        ]
    ]
    .to_string(index=False)
)


# ------------------------------------------------------------
# IDs beginning with useful prefixes
# ------------------------------------------------------------

prefix_patterns = [
    "utr",
    "sRNA",
    "ncRNA",
    "tRNA",
    "rRNA",
    "IGR",
    "intergenic",
    "transcript",
]

print()
print("=" * 100)
print("BIOLOGICALLY INTERESTING PREFIX/TEXT MATCHES")
print("=" * 100)

for pattern in prefix_patterns:

    subset = ids[
        ids.str.contains(
            pattern,
            case=False,
            regex=False,
        )
    ]

    print()
    print(
        f"{pattern}: {len(subset)}"
    )

    if len(subset):
        print(
            subset.head(30).to_string(
                index=False
            )
        )


# ============================================================
# GSE87213 missing features
# ============================================================

path = COUNT_DIR / "GSE87213_counts.csv"

g87 = pd.read_csv(
    path,
    encoding="utf-8-sig",
)

feature_col = g87.columns[0]
sample_cols = g87.columns[1:]

numeric = g87[
    sample_cols
].apply(
    pd.to_numeric,
    errors="coerce",
)

missing = g87.loc[
    numeric.isna().any(axis=1)
].copy()

missing.to_csv(
    OUT / "GSE87213_missing_feature_rows.csv",
    index=False,
)

print()
print("=" * 100)
print("GSE87213 MISSING FEATURES")
print("=" * 100)

print(
    missing.to_string(index=False)
)

print()
print(
    f"Saved audit files to: {OUT}"
)
