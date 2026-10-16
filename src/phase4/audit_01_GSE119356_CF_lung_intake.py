from pathlib import Path
from urllib.request import urlretrieve
from io import StringIO
import json
import re

import numpy as np
import pandas as pd


ROOT = Path.cwd()

DATA_DIR = (
    ROOT
    / "data/raw/phase4/GSE119356"
)

OUT = (
    ROOT
    / "results/phase4/audit_01_GSE119356_CF_lung_intake"
)

XLSX = (
    DATA_DIR
    / "GSE119356_ProcessedDataMatrix.xlsx"
)

URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE119nnn/GSE119356/suppl/"
    "GSE119356_ProcessedDataMatrix.xlsx"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411/paeru_116_iModulon.json"
)


EXPECTED = [
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


REGIONS = [
    "BR",
    "ULP",
    "ULC",
    "LLP",
    "LLC",
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
        "Unexpected serialized model DataFrame."
    )


def pao1_fraction(values):
    values = pd.Series(
        values
    ).astype(str).str.strip()

    if len(values) == 0:
        return 0.0

    matched = values.str.fullmatch(
        r"PA\d{4}"
    )

    return float(
        matched.mean()
    )


def choose_gene_column(table):
    candidates = []

    for column in table.columns:
        fraction = pao1_fraction(
            table[column]
        )

        candidates.append(
            (
                fraction,
                column,
            )
        )

    candidates = sorted(
        candidates,
        reverse=True,
        key=lambda item: item[0],
    )

    best_fraction, best_column = candidates[0]

    print()
    print("Gene-ID column candidates")

    for fraction, column in candidates[:10]:
        print(
            " ",
            repr(
                column
            ),
            "PAO1-ID fraction:",
            round(
                fraction,
                4,
            ),
        )

    if best_fraction < 0.80:
        raise RuntimeError(
            "Could not identify a convincing PAO1 locus-tag column."
        )

    return best_column


def normalize_sample_name(value):
    text = str(
        value
    ).strip()

    text = re.sub(
        r"\s+",
        "_",
        text,
    )

    return text


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 01")
    print("GSE119356 direct CF-lung transcriptome intake")
    print("=" * 78)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Download published processed matrix
    # ----------------------------------------------------------

    if not XLSX.exists():
        print()
        print("Downloading published GSE119356 processed matrix")
        print(
            "  destination:",
            XLSX,
        )

        urlretrieve(
            URL,
            XLSX,
        )

    print()
    print("Published processed matrix")
    print(
        "  file:",
        XLSX,
    )
    print(
        "  size:",
        XLSX.stat().st_size,
        "bytes",
    )

    # ----------------------------------------------------------
    # 2. Inspect workbook structure
    # ----------------------------------------------------------

    workbook = pd.ExcelFile(
        XLSX
    )

    print()
    print("Workbook sheets")

    for sheet in workbook.sheet_names:
        print(
            " ",
            sheet,
        )

    sheet_summaries = []

    selected = None
    selected_sheet = None

    for sheet in workbook.sheet_names:
        table = pd.read_excel(
            XLSX,
            sheet_name=sheet,
        )

        sheet_summaries.append(
            {
                "sheet":
                    sheet,
                "rows":
                    table.shape[0],
                "columns":
                    table.shape[1],
            }
        )

        if table.shape[0] >= 4000:
            if selected is None:
                selected = table
                selected_sheet = sheet

    if selected is None:
        raise RuntimeError(
            "No expression-sized worksheet found."
        )

    pd.DataFrame(
        sheet_summaries
    ).to_csv(
        OUT
        / "workbook_sheet_summary.csv",
        index=False,
    )

    print()
    print("Selected expression sheet")
    print(
        "  sheet:",
        selected_sheet,
    )
    print(
        "  dimensions:",
        selected.shape,
    )

    print()
    print("Columns")
    for column in selected.columns:
        print(
            " ",
            repr(
                column
            )
        )

    # ----------------------------------------------------------
    # 3. Resolve gene-ID column
    # ----------------------------------------------------------

    gene_column = choose_gene_column(
        selected
    )

    genes = (
        selected[
            gene_column
        ]
        .astype(str)
        .str.strip()
    )

    duplicate_genes = int(
        genes.duplicated().sum()
    )

    print()
    print("Gene namespace")
    print(
        "  selected ID column:",
        gene_column,
    )
    print(
        "  rows:",
        len(
            genes
        ),
    )
    print(
        "  unique IDs:",
        genes.nunique(),
    )
    print(
        "  duplicated IDs:",
        duplicate_genes,
    )
    print(
        "  PA#### fraction:",
        round(
            pao1_fraction(
                genes
            ),
            4,
        ),
    )

    # ----------------------------------------------------------
    # 4. Resolve numeric sample columns
    # ----------------------------------------------------------

    expression = selected.drop(
        columns=[
            gene_column
        ]
    ).copy()

    expression.columns = [
        normalize_sample_name(
            column
        )
        for column in expression.columns
    ]

    numeric_columns = []

    numeric_data = {}

    for column in expression.columns:
        values = pd.to_numeric(
            expression[
                column
            ],
            errors="coerce",
        )

        finite_fraction = float(
            values.notna().mean()
        )

        if finite_fraction >= 0.95:
            numeric_columns.append(
                column
            )

            numeric_data[
                column
            ] = values

    matrix = pd.DataFrame(
        numeric_data,
        index=genes,
    )

    matrix = matrix.loc[
        ~matrix.index.duplicated(
            keep=False
        )
    ].copy()

    print()
    print("Numeric expression columns")

    for column in numeric_columns:
        print(
            " ",
            column,
        )

    # ----------------------------------------------------------
    # 5. Match expected sample labels flexibly
    # ----------------------------------------------------------

    resolved = {}

    for expected in EXPECTED:
        exact = [
            column
            for column in matrix.columns
            if column.lower()
            == expected.lower()
        ]

        if len(
            exact
        ) == 1:
            resolved[
                expected
            ] = exact[0]
            continue

        compact_expected = re.sub(
            r"[^a-z0-9]",
            "",
            expected.lower(),
        )

        fuzzy = []

        for column in matrix.columns:
            compact_column = re.sub(
                r"[^a-z0-9]",
                "",
                column.lower(),
            )

            if compact_expected in compact_column:
                fuzzy.append(
                    column
                )

        if len(
            fuzzy
        ) == 1:
            resolved[
                expected
            ] = fuzzy[0]

    print()
    print("Expected-sample resolution")

    for expected in EXPECTED:
        print(
            " ",
            expected,
            "->",
            resolved.get(
                expected,
                "NOT_FOUND",
            ),
        )

    all_expected_resolved = (
        len(
            resolved
        )
        == len(
            EXPECTED
        )
    )

    if all_expected_resolved:
        final = pd.DataFrame(
            {
                expected:
                    matrix[
                        actual
                    ]
                for expected, actual
                in resolved.items()
            }
        )
    else:
        final = matrix.copy()

    # ----------------------------------------------------------
    # 6. Count-matrix QC
    # ----------------------------------------------------------

    final = final.apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite_values = bool(
        np.isfinite(
            final.to_numpy(
                dtype=float
            )
        ).all()
    )

    nonnegative_values = bool(
        (
            final
            >= 0
        ).all().all()
    )

    sample_qc_rows = []

    for sample in final.columns:
        values = final[
            sample
        ]

        sample_qc_rows.append(
            {
                "sample":
                    sample,
                "total_unique_gene_reads":
                    float(
                        values.sum()
                    ),
                "detected_genes":
                    int(
                        (
                            values
                            > 0
                        ).sum()
                    ),
                "detected_fraction":
                    float(
                        (
                            values
                            > 0
                        ).mean()
                    ),
            }
        )

    sample_qc = pd.DataFrame(
        sample_qc_rows
    )

    print()
    print("Per-sample processed-count QC")
    print(
        sample_qc
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Frozen PAO1 model overlap
    # ----------------------------------------------------------

    if not MODEL.exists():
        raise FileNotFoundError(
            str(
                MODEL
            )
        )

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

    M.columns = (
        M.columns
        .astype(str)
        .str.strip()
    )

    model_genes = set(
        M.index
    )

    observed_genes = set(
        final.index
    )

    overlap = sorted(
        model_genes.intersection(
            observed_genes
        )
    )

    missing_model = sorted(
        model_genes.difference(
            observed_genes
        )
    )

    model_coverage = (
        len(
            overlap
        )
        / len(
            model_genes
        )
    )

    print()
    print("aeruPRECISE model compatibility")
    print(
        "  model genes:",
        len(
            model_genes
        ),
    )
    print(
        "  observed expression genes:",
        len(
            observed_genes
        ),
    )
    print(
        "  overlapping model genes:",
        len(
            overlap
        ),
    )
    print(
        "  model gene coverage:",
        format(
            model_coverage,
            ".2%",
        ),
    )
    print(
        "  missing model genes:",
        len(
            missing_model
        ),
    )

    # ----------------------------------------------------------
    # 8. Matched spatial design
    # ----------------------------------------------------------

    pair_rows = []

    for region in REGIONS:
        invivo = (
            "invivo_"
            + region
        )

        invitro = (
            "invitro_"
            + region
        )

        pair_rows.append(
            {
                "region":
                    region,
                "invivo_sample":
                    invivo,
                "invitro_sample":
                    invitro,
                "invivo_present":
                    invivo
                    in final.columns,
                "invitro_present":
                    invitro
                    in final.columns,
                "matched_pair_complete":
                    (
                        invivo
                        in final.columns
                    )
                    and (
                        invitro
                        in final.columns
                    ),
            }
        )

    pairs = pd.DataFrame(
        pair_rows
    )

    print()
    print("Matched CF-lung regional design")
    print(
        pairs.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Simple log-count correlation diagnostic
    #
    # This is descriptive only. These are not biological
    # replicates; each in-vivo sample represents a different
    # spatial region.
    # ----------------------------------------------------------

    log_counts = np.log2(
        final
        + 1.0
    )

    correlation = log_counts.corr()

    if all_expected_resolved:
        print()
        print("All-sample log2(count+1) correlation")
        print(
            correlation
            .round(
                3
            )
            .to_string()
        )

    # ----------------------------------------------------------
    # 10. Save intake outputs
    # ----------------------------------------------------------

    final.to_csv(
        OUT
        / "GSE119356_raw_unique_gene_reads.csv"
    )

    sample_qc.to_csv(
        OUT
        / "sample_count_QC.csv",
        index=False,
    )

    pairs.to_csv(
        OUT
        / "matched_region_design.csv",
        index=False,
    )

    correlation.to_csv(
        OUT
        / "log_count_sample_correlation.csv"
    )

    pd.DataFrame(
        {
            "missing_model_gene":
                missing_model
        }
    ).to_csv(
        OUT
        / "missing_aeruprecise_model_genes.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 11. Intake decision
    #
    # We do NOT require similarity between spatial regions.
    # Biological heterogeneity is precisely what Phase 4 aims
    # to measure.
    # ----------------------------------------------------------

    minimum_depth = float(
        sample_qc[
            "total_unique_gene_reads"
        ].min()
    )

    minimum_detected = int(
        sample_qc[
            "detected_genes"
        ].min()
    )

    complete_pairs = int(
        pairs[
            "matched_pair_complete"
        ].sum()
    )

    checks = {
        "expression_matrix_ge_5000_genes":
            final.shape[
                0
            ]
            >= 5000,

        "unique_gene_identifiers":
            duplicate_genes
            == 0,

        "PAO1_namespace_ge_95pct":
            pao1_fraction(
                final.index
            )
            >= 0.95,

        "all_10_expected_samples_resolved":
            all_expected_resolved,

        "five_complete_spatial_pairs":
            complete_pairs
            == 5,

        "model_gene_coverage_ge_90pct":
            model_coverage
            >= 0.90,

        "minimum_detected_genes_ge_3000":
            minimum_detected
            >= 3000,

        "minimum_unique_gene_reads_ge_50k":
            minimum_depth
            >= 50000,

        "finite_values":
            finite_values,

        "nonnegative_values":
            nonnegative_values,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Phase-4 intake checks")

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
        "GSE119356_CF_LUNG_READY_FOR_PROJECTION_AUDIT"
        if passed
        else "GSE119356_CF_LUNG_INTAKE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    pd.DataFrame(
        [
            {
                "check":
                    key,
                "pass":
                    value,
            }
            for key, value
            in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    (
        OUT
        / "AUDIT_01_STATUS.txt"
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
            "Failed Phase-4 intake checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
