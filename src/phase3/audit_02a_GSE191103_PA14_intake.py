from pathlib import Path
import hashlib
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path.cwd()

RAW_DIR = (
    ROOT
    / "data/raw/phase3/GSE191103"
)

SOURCE = (
    RAW_DIR
    / "GSE191103_ProcessedDataMatrix_biofilm_lasR.xlsx"
)

SOURCE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE191nnn/GSE191103/suppl/"
    "GSE191103_ProcessedDataMatrix_biofilm_lasR.xlsx"
)

STRICT_MAP = (
    ROOT
    / "results/phase3/audit_01a_PA14_RBH_ambiguity_diagnostic"
    / "strict_unique_PAO1_PA14_map.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_02a_GSE191103_PA14_intake"
)


BIOFILM_EXPECTED = [
    "PA14_BF_1",
    "PA14_BF_2",
    "PA14_BF_3",
]

PLANKTONIC_EXPECTED = [
    "PA14_PL_1",
    "PA14_PL_2",
    "PA14_PL_3",
]


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(
                block
            )

    return h.hexdigest()


def download():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SOURCE.exists():
        print(
            "Processed workbook already present:"
        )
        print(
            " ",
            SOURCE,
        )
        return

    print(
        "Downloading GSE191103 processed workbook ..."
    )

    request = urllib.request.Request(
        SOURCE_URL,
        headers={
            "User-Agent":
                "CF-State Phase3 Audit02a"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        SOURCE.write_bytes(
            response.read()
        )

    print(
        "Downloaded:",
        SOURCE,
    )


def normalize_name(value):
    return (
        str(
            value
        )
        .strip()
        .replace(
            " ",
            "_",
        )
    )


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 02a")
    print("GSE191103 WT PA14 biofilm/planktonic intake")
    print("=" * 78)

    if not STRICT_MAP.exists():
        raise FileNotFoundError(
            str(
                STRICT_MAP
            )
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    download()

    print()
    print("Frozen source")
    print(
        "  path:",
        SOURCE,
    )
    print(
        "  size:",
        SOURCE.stat().st_size,
    )
    print(
        "  SHA256:",
        sha256(
            SOURCE
        ),
    )

    # ----------------------------------------------------------
    # Workbook structure
    # ----------------------------------------------------------

    workbook = pd.ExcelFile(
        SOURCE
    )

    print()
    print("Workbook sheets")

    for sheet in workbook.sheet_names:
        print(
            " ",
            sheet,
        )

    sheet_summaries = []

    candidate_tables = []

    for sheet in workbook.sheet_names:
        table = pd.read_excel(
            SOURCE,
            sheet_name=sheet,
        )

        normalized_columns = {
            column:
                normalize_name(
                    column
                )
            for column in table.columns
        }

        normalized_values = set(
            normalized_columns.values()
        )

        n_expected = sum(
            sample in normalized_values
            for sample in (
                BIOFILM_EXPECTED
                + PLANKTONIC_EXPECTED
            )
        )

        sheet_summaries.append(
            {
                "sheet":
                    sheet,

                "rows":
                    table.shape[
                        0
                    ],

                "columns":
                    table.shape[
                        1
                    ],

                "expected_WT_sample_columns_found":
                    n_expected,
            }
        )

        if n_expected > 0:
            candidate_tables.append(
                (
                    sheet,
                    table,
                    normalized_columns,
                    n_expected,
                )
            )

    summary = pd.DataFrame(
        sheet_summaries
    )

    print()
    print("Sheet summary")
    print(
        summary.to_string(
            index=False
        )
    )

    if not candidate_tables:
        raise RuntimeError(
            "No worksheet contains recognizable WT PA14 sample columns."
        )

    candidate_tables.sort(
        key=lambda x: x[
            3
        ],
        reverse=True,
    )

    sheet, table, normalized_columns, n_expected = (
        candidate_tables[
            0
        ]
    )

    print()
    print(
        "Selected worksheet:",
        sheet,
    )
    print(
        "  shape:",
        table.shape,
    )
    print(
        "  WT sample columns recognized:",
        n_expected,
    )

    print()
    print("Columns")
    for column in table.columns:
        print(
            " ",
            repr(
                column
            ),
        )

    # ----------------------------------------------------------
    # Identify the six WT samples
    # ----------------------------------------------------------

    inverse = {
        normalized:
            original
        for original, normalized
        in normalized_columns.items()
    }

    missing_samples = [
        sample
        for sample in (
            BIOFILM_EXPECTED
            + PLANKTONIC_EXPECTED
        )
        if sample not in inverse
    ]

    if missing_samples:
        raise RuntimeError(
            "Missing expected WT samples: "
            + ", ".join(
                missing_samples
            )
        )

    biofilm_cols = [
        inverse[
            sample
        ]
        for sample
        in BIOFILM_EXPECTED
    ]

    planktonic_cols = [
        inverse[
            sample
        ]
        for sample
        in PLANKTONIC_EXPECTED
    ]

    print()
    print("WT PA14 samples")
    print(
        "  biofilm:",
        biofilm_cols,
    )
    print(
        "  planktonic:",
        planktonic_cols,
    )

    # ----------------------------------------------------------
    # Detect likely PA14 gene-ID column
    # ----------------------------------------------------------

    strict_map = pd.read_csv(
        STRICT_MAP
    )

    pa14_genes = set(
        strict_map[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    id_scores = []

    for column in table.columns:
        values = (
            table[
                column
            ]
            .astype(str)
            .str.strip()
        )

        matches = int(
            values.isin(
                pa14_genes
            ).sum()
        )

        id_scores.append(
            {
                "column":
                    column,

                "strict_PA14_RBH_matches":
                    matches,
            }
        )

    id_scores = (
        pd.DataFrame(
            id_scores
        )
        .sort_values(
            "strict_PA14_RBH_matches",
            ascending=False,
        )
    )

    print()
    print("Candidate PA14 gene-ID columns")
    print(
        id_scores
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    best_id_col = id_scores.iloc[
        0
    ][
        "column"
    ]

    best_matches = int(
        id_scores.iloc[
            0
        ][
            "strict_PA14_RBH_matches"
        ]
    )

    print()
    print(
        "Selected gene-ID column:",
        best_id_col,
    )
    print(
        "  strict RBH matches:",
        best_matches,
    )

    if best_matches < 4000:
        raise RuntimeError(
            "Processed matrix does not contain enough recognizable "
            "PA14 locus tags for safe cross-strain projection."
        )

    # ----------------------------------------------------------
    # Extract WT expression matrix
    # ----------------------------------------------------------

    selected_columns = (
        biofilm_cols
        + planktonic_cols
    )

    expression = table[
        selected_columns
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    gene_ids = (
        table[
            best_id_col
        ]
        .astype(str)
        .str.strip()
    )

    expression.index = (
        gene_ids
    )

    expression = expression.loc[
        ~expression.index.duplicated(
            keep=False
        )
    ].copy()

    finite_rows = np.isfinite(
        expression.to_numpy(
            dtype=float
        )
    ).all(
        axis=1
    )

    expression = expression.loc[
        finite_rows
    ].copy()

    # ----------------------------------------------------------
    # Strict cross-strain coverage
    # ----------------------------------------------------------

    observed_pa14 = set(
        expression.index
    )

    strict_pa14 = set(
        strict_map[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    shared = (
        observed_pa14
        & strict_pa14
    )

    coverage = (
        len(
            shared
        )
        / len(
            strict_pa14
        )
    )

    print()
    print("WT PA14 expression matrix")
    print(
        "  finite unique genes:",
        expression.shape[
            0
        ],
    )
    print(
        "  samples:",
        expression.shape[
            1
        ],
    )
    print(
        "  strict RBH genes represented:",
        len(
            shared
        ),
    )
    print(
        "  strict RBH expression coverage:",
        format(
            coverage,
            ".2%",
        ),
    )

    # ----------------------------------------------------------
    # Raw expression diagnostics
    # ----------------------------------------------------------

    print()
    print("Expression value diagnostics")
    print(
        "  minimum:",
        float(
            expression.min().min()
        ),
    )
    print(
        "  maximum:",
        float(
            expression.max().max()
        ),
    )
    print(
        "  median:",
        float(
            np.nanmedian(
                expression.to_numpy(
                    dtype=float
                )
            )
        ),
    )

    column_sums = expression.sum(
        axis=0
    )

    print()
    print("Column sums")
    print(
        column_sums
        .round(
            3
        )
        .to_string()
    )

    # ----------------------------------------------------------
    # Save frozen intake
    # ----------------------------------------------------------

    expression.to_csv(
        OUT
        / "GSE191103_WT_PA14_processed_expression.csv"
    )

    strict_map.loc[
        strict_map[
            "pa14_gene"
        ]
        .astype(str)
        .isin(
            shared
        )
    ].to_csv(
        OUT
        / "GSE191103_observed_strict_RBH_map.csv",
        index=False,
    )

    summary.to_csv(
        OUT
        / "workbook_sheet_summary.csv",
        index=False,
    )

    id_scores.to_csv(
        OUT
        / "gene_id_column_scores.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # Intake decision
    # ----------------------------------------------------------

    checks = {
        "three_WT_biofilm_samples":
            len(
                biofilm_cols
            )
            == 3,

        "three_WT_planktonic_samples":
            len(
                planktonic_cols
            )
            == 3,

        "gene_ID_matches_ge_4000":
            best_matches
            >= 4000,

        "strict_RBH_expression_coverage_ge_90pct":
            coverage
            >= 0.90,

        "finite_expression_matrix":
            bool(
                np.isfinite(
                    expression.to_numpy(
                        dtype=float
                    )
                ).all()
            ),
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit checks")

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
        "READY_FOR_PA14_PROJECTION"
        if passed
        else "INTAKE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    (
        OUT
        / "AUDIT_02A_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
