from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact, hypergeom


INPUT_FILE = Path(
    "results/audit/external_validation/"
    "cross_external_comparison/"
    "all_251_cross_external_comparison.csv"
)

OUT_DIR = Path(
    "results/audit/external_validation/"
    "cross_external_comparison"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 40: "
    "test overlap between external validations"
)
print("=" * 100)


df = pd.read_csv(
    INPUT_FILE
)


required_columns = [
    "same_direction_299861",
    "same_direction_191103",
    "strong_299861",
    "strong_191103",
]


for column in required_columns:

    if column not in df.columns:
        raise RuntimeError(
            "Missing column: " + column
        )


if len(df) != 251:
    raise RuntimeError(
        "Expected 251 genes."
    )


def as_bool(series):

    if series.dtype == bool:
        return series

    values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    unexpected = (
        set(values.unique())
        - {"true", "false"}
    )

    if unexpected:
        raise RuntimeError(
            "Unexpected boolean values: "
            + str(sorted(unexpected))
        )

    return values.map(
        {
            "true": True,
            "false": False,
        }
    )


same_a = as_bool(
    df["same_direction_299861"]
)

same_b = as_bool(
    df["same_direction_191103"]
)

strong_a = as_bool(
    df["strong_299861"]
)

strong_b = as_bool(
    df["strong_191103"]
)


def analyse_overlap(
    a,
    b,
    label,
):

    n = len(a)

    n_a = int(a.sum())
    n_b = int(b.sum())

    n_both = int(
        (a & b).sum()
    )

    n_a_only = int(
        (a & ~b).sum()
    )

    n_b_only = int(
        (~a & b).sum()
    )

    n_neither = int(
        (~a & ~b).sum()
    )

    expected_both = (
        n_a
        * n_b
        / n
    )

    table = [
        [
            n_both,
            n_a_only,
        ],
        [
            n_b_only,
            n_neither,
        ],
    ]

    odds_ratio, fisher_p = fisher_exact(
        table,
        alternative="greater",
    )

    hypergeom_p = hypergeom.sf(
        n_both - 1,
        n,
        n_a,
        n_b,
    )

    observed_agreement = (
        n_both
        + n_neither
    ) / n

    expected_agreement = (
        (n_a / n)
        * (n_b / n)
        +
        (1.0 - n_a / n)
        * (1.0 - n_b / n)
    )

    if expected_agreement == 1.0:
        kappa = float("nan")
    else:
        kappa = (
            observed_agreement
            - expected_agreement
        ) / (
            1.0
            - expected_agreement
        )

    return {
        "analysis":
            label,

        "n_genes":
            n,

        "positive_299861":
            n_a,

        "positive_191103":
            n_b,

        "observed_overlap":
            n_both,

        "expected_overlap_independence":
            expected_both,

        "overlap_minus_expected":
            n_both
            - expected_both,

        "a_only":
            n_a_only,

        "b_only":
            n_b_only,

        "neither":
            n_neither,

        "odds_ratio":
            odds_ratio,

        "fisher_greater_p":
            fisher_p,

        "hypergeometric_greater_p":
            hypergeom_p,

        "cohen_kappa":
            kappa,
    }


same_result = analyse_overlap(
    same_a,
    same_b,
    "same_direction",
)

strong_result = analyse_overlap(
    strong_a,
    strong_b,
    "strong_same_direction",
)


results = pd.DataFrame(
    [
        same_result,
        strong_result,
    ]
)


results.to_csv(
    OUT_DIR
    / "external_overlap_enrichment.csv",
    index=False,
)


print()
print("=" * 100)
print("OVERLAP TEST")
print("=" * 100)


for _, row in results.iterrows():

    print()
    print(
        str(row["analysis"])
    )

    print(
        "  GSE299861 positive: "
        + str(
            int(
                row[
                    "positive_299861"
                ]
            )
        )
    )

    print(
        "  GSE191103 positive: "
        + str(
            int(
                row[
                    "positive_191103"
                ]
            )
        )
    )

    print(
        "  Observed overlap: "
        + str(
            int(
                row[
                    "observed_overlap"
                ]
            )
        )
    )

    print(
        "  Expected overlap under independence: "
        + format(
            row[
                "expected_overlap_independence"
            ],
            ".2f",
        )
    )

    print(
        "  Overlap minus expected: "
        + format(
            row[
                "overlap_minus_expected"
            ],
            ".2f",
        )
    )

    print(
        "  Fisher exact enrichment p: "
        + format(
            row[
                "fisher_greater_p"
            ],
            ".4f",
        )
    )

    print(
        "  Hypergeometric enrichment p: "
        + format(
            row[
                "hypergeometric_greater_p"
            ],
            ".4f",
        )
    )

    print(
        "  Cohen kappa: "
        + format(
            row[
                "cohen_kappa"
            ],
            ".4f",
        )
    )


print()
print("=" * 100)
print("AUDIT 40 COMPLETE")
print("=" * 100)

print(
    "Output: "
    + str(
        OUT_DIR
        / "external_overlap_enrichment.csv"
    )
)

print()
