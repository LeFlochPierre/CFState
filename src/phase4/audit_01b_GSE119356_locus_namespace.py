from pathlib import Path
from io import StringIO
from collections import Counter
import json
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

XLSX = (
    ROOT
    / "data/raw/phase4/GSE119356"
    / "GSE119356_ProcessedDataMatrix.xlsx"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411"
    / "paeru_116_iModulon.json"
)

OUT = (
    ROOT
    / "results/phase4/audit_01b_GSE119356_locus_namespace"
)


EXPECTED_SAMPLES = [
    "invivo_BR",
    "invivo_ULP",
    "invivo_ULC",
    "invivo_LLP",
    "invivo_LLC",
    "invitro_BR",
    "invitro_ULP",
    "invitro_ULC",
    "invitro_LLP",
    "invitro_LLC",
]


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
        "Unexpected serialized DataFrame."
    )


def unique_embedded_pa_tag(text):
    text = str(
        text
    ).strip()

    hits = re.findall(
        r"(?<![A-Za-z0-9])PA\d{4}(?!\d)",
        text,
    )

    hits = sorted(
        set(
            hits
        )
    )

    if len(
        hits
    ) == 1:
        return hits[0]

    return None


def classify_identifier(text):
    text = str(
        text
    ).strip()

    if re.fullmatch(
        r"PA\d{4}",
        text,
    ):
        return "exact_PA####"

    if re.fullmatch(
        r"PA\d{4}\.\d+",
        text,
    ):
        return "PAO1_decimal"

    if re.fullmatch(
        r"PA\d{4}-PA\d{4}",
        text,
    ):
        return "PAO1_range"

    hits = re.findall(
        r"PA\d{4}",
        text,
    )

    unique_hits = set(
        hits
    )

    if len(
        unique_hits
    ) == 1:
        return "composite_single_PA####"

    if len(
        unique_hits
    ) > 1:
        return "composite_multiple_PA####"

    if "," in text:
        return "comma_no_PA####"

    if ";" in text:
        return "semicolon_no_PA####"

    if text.lower() in {
        "",
        "nan",
        "none",
        "na",
    }:
        return "missing"

    return "other"


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 01b")
    print("GSE119356 PAO1 locus-namespace diagnostic")
    print("=" * 78)

    for path in [
        XLSX,
        MODEL,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Read published expression sheet
    # ----------------------------------------------------------

    table = pd.read_excel(
        XLSX,
        sheet_name="Tabelle1",
    )

    if "locus" not in table.columns:
        raise RuntimeError(
            "Expected 'locus' column not found."
        )

    missing_samples = [
        sample
        for sample in EXPECTED_SAMPLES
        if sample not in table.columns
    ]

    if missing_samples:
        raise RuntimeError(
            "Missing expected samples: "
            + ", ".join(
                missing_samples
            )
        )

    locus = (
        table[
            "locus"
        ]
        .astype(str)
        .str.strip()
    )

    counts = (
        table[
            EXPECTED_SAMPLES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    if counts.isna().any().any():
        raise RuntimeError(
            "Non-numeric values found in expected sample columns."
        )

    total_count = counts.sum(
        axis=1
    )

    diagnostic = pd.DataFrame(
        {
            "raw_locus":
                locus,

            "identifier_class":
                [
                    classify_identifier(
                        value
                    )
                    for value in locus
                ],

            "unique_embedded_PA_tag":
                [
                    unique_embedded_pa_tag(
                        value
                    )
                    for value in locus
                ],

            "total_count":
                total_count,
        }
    )

    # ----------------------------------------------------------
    # 2. Frozen aeruPRECISE gene universe
    # ----------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw = json.load(
            handle
        )

    M = json_df(
        raw[
            "M"
        ]
    )

    M.index = (
        M.index
        .astype(str)
        .str.strip()
    )

    model_genes = set(
        M.index
    )

    if len(
        model_genes
    ) != 5572:
        raise RuntimeError(
            "Expected 5572 aeruPRECISE genes."
        )

    # ----------------------------------------------------------
    # 3. Exact raw overlap
    # ----------------------------------------------------------

    raw_ids = set(
        locus
    )

    exact_overlap = (
        raw_ids
        .intersection(
            model_genes
        )
    )

    exact_coverage = (
        len(
            exact_overlap
        )
        / len(
            model_genes
        )
    )

    print()
    print("Raw locus namespace")
    print(
        "  rows:",
        len(
            locus
        ),
    )
    print(
        "  unique raw locus strings:",
        locus.nunique(),
    )
    print(
        "  exact aeruPRECISE matches:",
        len(
            exact_overlap
        ),
        "/ 5572",
    )
    print(
        "  exact model coverage:",
        format(
            exact_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # 4. Identifier classes
    # ----------------------------------------------------------

    class_summary = (
        diagnostic
        .groupby(
            "identifier_class"
        )
        .agg(
            n_rows=(
                "raw_locus",
                "size",
            ),
            total_count=(
                "total_count",
                "sum",
            ),
        )
        .sort_values(
            "total_count",
            ascending=False,
        )
    )

    class_summary[
        "fraction_total_count"
    ] = (
        class_summary[
            "total_count"
        ]
        / class_summary[
            "total_count"
        ].sum()
    )

    print()
    print("Identifier classes")
    print(
        class_summary
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 5. Show examples from every non-trivial class
    # ----------------------------------------------------------

    print()
    print("Identifier examples")

    for identifier_class in class_summary.index:
        subset = diagnostic.loc[
            diagnostic[
                "identifier_class"
            ]
            == identifier_class
        ]

        examples = (
            subset[
                "raw_locus"
            ]
            .drop_duplicates()
            .head(
                15
            )
            .tolist()
        )

        print()
        print(
            " ",
            identifier_class,
            "(",
            len(
                subset
            ),
            "rows )",
        )

        for example in examples:
            print(
                "   ",
                repr(
                    example
                )
            )

    # ----------------------------------------------------------
    # 6. Conservative single-tag normalization
    #
    # Accept:
    #   PA0001
    #   "PA0001,..."
    #   "... PA0001 ..."
    #
    # only when exactly one PA#### locus tag is present.
    #
    # Never guess when multiple tags occur.
    # ----------------------------------------------------------

    normalized = []

    methods = []

    for raw_locus, embedded in zip(
        diagnostic[
            "raw_locus"
        ],
        diagnostic[
            "unique_embedded_PA_tag"
        ],
    ):
        if re.fullmatch(
            r"PA\d{4}",
            str(
                raw_locus
            ),
        ):
            normalized.append(
                str(
                    raw_locus
                )
            )

            methods.append(
                "EXACT_PA_TAG"
            )

        elif (
            embedded is not None
            and embedded in model_genes
        ):
            normalized.append(
                embedded
            )

            methods.append(
                "UNIQUE_EMBEDDED_PA_TAG"
            )

        else:
            normalized.append(
                None
            )

            methods.append(
                "UNRESOLVED"
            )

    diagnostic[
        "normalized_model_gene"
    ] = normalized

    diagnostic[
        "normalization_method"
    ] = methods

    resolved = diagnostic.loc[
        diagnostic[
            "normalized_model_gene"
        ].notna()
    ].copy()

    # ----------------------------------------------------------
    # 7. Check collisions
    # ----------------------------------------------------------

    target_counts = (
        resolved[
            "normalized_model_gene"
        ]
        .value_counts()
    )

    collision_genes = set(
        target_counts.loc[
            target_counts > 1
        ].index
    )

    collisions = resolved.loc[
        resolved[
            "normalized_model_gene"
        ].isin(
            collision_genes
        )
    ].copy()

    clean = resolved.loc[
        ~resolved[
            "normalized_model_gene"
        ].isin(
            collision_genes
        )
    ].copy()

    represented_model_genes = set(
        clean[
            "normalized_model_gene"
        ]
    )

    normalized_coverage = (
        len(
            represented_model_genes
        )
        / len(
            model_genes
        )
    )

    print()
    print("Conservative normalization")
    print(
        "  uniquely resolved rows:",
        len(
            resolved
        ),
    )
    print(
        "  collision target genes:",
        len(
            collision_genes
        ),
    )
    print(
        "  clean model genes represented:",
        len(
            represented_model_genes
        ),
        "/ 5572",
    )
    print(
        "  normalized model coverage:",
        format(
            normalized_coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # 8. Mapping methods
    # ----------------------------------------------------------

    method_summary = (
        diagnostic
        .groupby(
            "normalization_method"
        )
        .agg(
            n_rows=(
                "raw_locus",
                "size",
            ),
            total_count=(
                "total_count",
                "sum",
            ),
        )
        .sort_values(
            "n_rows",
            ascending=False,
        )
    )

    print()
    print("Normalization routes")
    print(
        method_summary
        .round(
            2
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 9. Model-compatible count mass
    # ----------------------------------------------------------

    clean_rows = set(
        clean.index
    )

    model_mass = counts.loc[
        counts.index.isin(
            clean_rows
        )
    ].sum(
        axis=0
    )

    total_mass = counts.sum(
        axis=0
    )

    model_mass_fraction = (
        model_mass
        / total_mass
    )

    print()
    print("Model-compatible count mass")
    print(
        model_mass_fraction
        .round(
            4
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # 10. Highest-count unresolved identifiers
    # ----------------------------------------------------------

    unresolved = diagnostic.loc[
        diagnostic[
            "normalized_model_gene"
        ].isna()
    ].copy()

    unresolved = unresolved.sort_values(
        "total_count",
        ascending=False,
    )

    print()
    print("Highest-count unresolved locus strings")
    print(
        unresolved[
            [
                "raw_locus",
                "identifier_class",
                "total_count",
            ]
        ]
        .head(
            50
        )
        .to_string(
            index=False
        )
    )

    if not collisions.empty:
        print()
        print("Normalization collisions")
        print(
            collisions[
                [
                    "raw_locus",
                    "normalized_model_gene",
                    "total_count",
                ]
            ]
            .sort_values(
                [
                    "normalized_model_gene",
                    "total_count",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .head(
                50
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 11. Missing model genes
    # ----------------------------------------------------------

    missing_model = sorted(
        model_genes.difference(
            represented_model_genes
        )
    )

    print()
    print("Model genes still absent after conservative normalization")
    print(
        "  n:",
        len(
            missing_model
        ),
    )

    print(
        "  first 40:",
        ", ".join(
            missing_model[
                :40
            ]
        ),
    )

    # ----------------------------------------------------------
    # 12. Save
    # ----------------------------------------------------------

    diagnostic.to_csv(
        OUT
        / "GSE119356_locus_namespace_diagnostic.csv",
        index=False,
    )

    class_summary.to_csv(
        OUT
        / "identifier_class_summary.csv"
    )

    method_summary.to_csv(
        OUT
        / "normalization_method_summary.csv"
    )

    collisions.to_csv(
        OUT
        / "normalization_collisions.csv",
        index=False,
    )

    unresolved.to_csv(
        OUT
        / "unresolved_locus_strings.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "missing_model_gene":
                missing_model
        }
    ).to_csv(
        OUT
        / "missing_model_genes.csv",
        index=False,
    )

    model_mass_fraction.to_csv(
        OUT
        / "model_compatible_count_mass_fraction.csv",
        header=[
            "fraction"
        ],
    )

    # ----------------------------------------------------------
    # 13. Diagnostic decision
    #
    # This is not yet the final intake audit.
    # We only ask whether a conservative namespace resolution
    # is sufficient for the next repaired intake step.
    # ----------------------------------------------------------

    checks = {
        "normalized_model_coverage_ge_90pct":
            normalized_coverage
            >= 0.90,

        "collision_targets_le_10":
            len(
                collision_genes
            )
            <= 10,

        "resolved_model_genes_ge_5000":
            len(
                represented_model_genes
            )
            >= 5000,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Namespace diagnostic checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    status = (
        "GSE119356_LOCUS_NAMESPACE_CONSERVATIVELY_RESOLVED"
        if passed
        else "GSE119356_LOCUS_NAMESPACE_NEEDS_DEEPER_BRIDGE"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_01B_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
