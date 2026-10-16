from pathlib import Path
import gzip
import hashlib
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path.cwd()

DATA_DIR = (
    ROOT
    / "data/raw/phase2/GSE152295"
)

DATA = (
    DATA_DIR
    / "GSE152295_PSEUDO_processed.txt.gz"
)

PRECISE = (
    ROOT
    / "external/aeruprecise/precise411/sample_table.csv"
)

OUT = (
    ROOT
    / "results/phase2/audit_09_GSE152295_stress_atlas_intake"
)

URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE152nnn/GSE152295/suppl/"
    "GSE152295_PSEUDO_processed.txt.gz"
)

ALIASES = [
    "GSE152295",
    "PRJNA638918",
    "SRP266954",
]


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            h.update(block)

    return h.hexdigest()


def download():
    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DATA.exists():
        print(
            "Processed Pseudomonas file already present:"
        )
        print(
            " ",
            DATA,
        )
        return

    print(
        "Downloading GEO processed Pseudomonas TPM file ..."
    )

    request = urllib.request.Request(
        URL,
        headers={
            "User-Agent":
                "CF-State Phase2 Audit09"
        },
    )

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        DATA.write_bytes(
            response.read()
        )

    print(
        "Downloaded:",
        DATA,
    )


def exact_alias_screen():
    precise = pd.read_csv(
        PRECISE,
        low_memory=False,
    )

    rows = []

    for alias in ALIASES:
        matched_indices = set()
        fields = set()

        for column in precise.columns:
            values = (
                precise[column]
                .fillna("")
                .astype(str)
            )

            mask = values.str.contains(
                alias,
                regex=False,
            )

            if mask.any():
                fields.add(
                    column
                )

                matched_indices.update(
                    precise.index[
                        mask
                    ].tolist()
                )

        rows.append(
            {
                "alias":
                    alias,

                "n_matching_rows":
                    len(
                        matched_indices
                    ),

                "matched_fields":
                    ";".join(
                        sorted(
                            fields
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def read_table():
    attempts = [
        {
            "sep": "\t",
        },
        {
            "sep": ",",
        },
    ]

    for kwargs in attempts:
        try:
            df = pd.read_csv(
                DATA,
                compression="gzip",
                **kwargs,
            )

            if df.shape[1] > 2:
                return df

        except Exception:
            pass

    return pd.read_csv(
        DATA,
        compression="gzip",
        sep=None,
        engine="python",
    )


def pa_fraction(series):
    values = (
        series
        .astype(str)
        .str.strip()
    )

    return float(
        values.str.fullmatch(
            r"PA\d{4}"
        ).mean()
    )


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 09")
    print("GSE152295 PAO1 multi-stress atlas intake")
    print("=" * 78)

    if not PRECISE.exists():
        raise FileNotFoundError(
            str(PRECISE)
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Freeze local copy
    # ----------------------------------------------------------

    download()

    print()
    print("Frozen source")
    print(
        "  path:",
        DATA,
    )
    print(
        "  size:",
        DATA.stat().st_size,
        "bytes",
    )
    print(
        "  SHA256:",
        sha256(DATA),
    )

    # ----------------------------------------------------------
    # 2. aeruPRECISE provenance screen
    # ----------------------------------------------------------

    overlap = exact_alias_screen()

    overlap.to_csv(
        OUT
        / "aeruprecise_overlap_screen.csv",
        index=False,
    )

    print()
    print("aeruPRECISE project-level provenance screen")
    print(
        overlap.to_string(
            index=False
        )
    )

    training_alias_overlap = bool(
        (
            overlap[
                "n_matching_rows"
            ]
            > 0
        ).any()
    )

    # ----------------------------------------------------------
    # 3. Inspect processed matrix
    # ----------------------------------------------------------

    df = read_table()

    print()
    print("Processed matrix")
    print(
        "  shape:",
        df.shape,
    )

    print()
    print("Columns")
    for i, column in enumerate(
        df.columns,
        start=1,
    ):
        print(
            f"  {i:2d}. {column}"
        )

    # ----------------------------------------------------------
    # 4. Identify PAO1 gene-ID column
    # ----------------------------------------------------------

    scores = []

    for column in df.columns:
        try:
            score = pa_fraction(
                df[column]
            )
        except Exception:
            score = 0.0

        scores.append(
            (
                score,
                column,
            )
        )

    scores.sort(
        reverse=True
    )

    best_score, gene_col = scores[0]

    print()
    print(
        "Best PAO1 locus-tag column:",
        gene_col,
    )
    print(
        "PA#### fraction:",
        round(
            best_score,
            4,
        ),
    )

    if best_score < 0.50:
        print()
        print("First 10 rows")
        print(
            df.head(10).to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Could not confidently identify "
            "canonical PAO1 locus tags."
        )

    # ----------------------------------------------------------
    # 5. Numeric/sample columns
    # ----------------------------------------------------------

    numeric = []

    for column in df.columns:
        if column == gene_col:
            continue

        converted = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        fraction_numeric = float(
            converted.notna().mean()
        )

        if fraction_numeric >= 0.95:
            numeric.append(
                column
            )

    if not numeric:
        raise RuntimeError(
            "No numeric expression columns found."
        )

    expression = (
        df[
            numeric
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    negative_fraction = float(
        (
            expression
            < 0
        )
        .sum()
        .sum()
        /
        expression.size
    )

    column_sums = expression.sum(
        axis=0
    )

    stats = pd.DataFrame(
        {
            "sample":
                numeric,

            "sum":
                column_sums.values,

            "min":
                expression.min(
                    axis=0
                ).values,

            "median":
                expression.median(
                    axis=0
                ).values,

            "max":
                expression.max(
                    axis=0
                ).values,

            "zero_fraction":
                (
                    expression
                    == 0
                )
                .mean(
                    axis=0
                )
                .values,
        }
    )

    stats.to_csv(
        OUT
        / "expression_column_qc.csv",
        index=False,
    )

    # TPM should normally have approximately 1e6 total.
    tpm_like = (
        (
            column_sums
            > 900000
        )
        &
        (
            column_sums
            < 1100000
        )
    )

    n_tpm_like = int(
        tpm_like.sum()
    )

    print()
    print("Expression-value diagnostics")
    print(
        "  numeric columns:",
        len(numeric),
    )
    print(
        "  negative-value fraction:",
        round(
            negative_fraction,
            8,
        ),
    )
    print(
        "  columns summing ~1e6:",
        str(n_tpm_like)
        + "/"
        + str(
            len(numeric)
        ),
    )

    print()
    print("Expression column summary")
    print(
        stats.round(3).to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 6. Identify stress-condition names
    # ----------------------------------------------------------

    keywords = {
        "control": [
            "ctrl",
            "control",
        ],

        "low_iron": [
            "lowiron",
            "low_iron",
            "iron",
            "_li",
            "li_",
        ],

        "hypoxia": [
            "hyp",
            "hypoxia",
        ],

        "stationary": [
            "stationary",
            "_sp",
            "sp_",
        ],

        "nutritional_downshift": [
            "nutr",
            "_nd",
            "nd_",
        ],

        "oxidative": [
            "oxidative",
            "_oxs",
            "oxs_",
        ],

        "nitrosative": [
            "nitros",
            "_ns",
            "ns_",
        ],

        "osmotic": [
            "osmotic",
            "_oss",
            "oss_",
        ],

        "temperature": [
            "temperature",
            "_tm",
            "tm_",
        ],

        "acid": [
            "acid",
            "_as",
            "as_",
        ],

        "bile": [
            "bile",
            "_bs",
            "bs_",
        ],

        "virulence_condition": [
            "virulence",
            "_vic",
            "vic_",
        ],
    }

    condition_rows = []

    for sample in numeric:
        s = str(
            sample
        ).lower()

        matches = []

        for condition, terms in keywords.items():
            if any(
                term in s
                for term in terms
            ):
                matches.append(
                    condition
                )

        condition_rows.append(
            {
                "sample":
                    sample,

                "candidate_condition":
                    ";".join(
                        matches
                    ),
            }
        )

    condition_table = pd.DataFrame(
        condition_rows
    )

    condition_table.to_csv(
        OUT
        / "candidate_sample_conditions.csv",
        index=False,
    )

    print()
    print("Candidate sample-condition parsing")
    print(
        condition_table.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Save matrix schema
    # ----------------------------------------------------------

    schema = pd.DataFrame(
        [
            {
                "n_rows":
                    df.shape[0],

                "n_columns":
                    df.shape[1],

                "gene_id_column":
                    gene_col,

                "canonical_PA_fraction":
                    best_score,

                "n_numeric_expression_columns":
                    len(
                        numeric
                    ),

                "negative_fraction":
                    negative_fraction,

                "n_columns_summing_approximately_1e6":
                    n_tpm_like,

                "project_alias_overlap_with_precise411":
                    training_alias_overlap,

                "source_sha256":
                    sha256(
                        DATA
                    ),
            }
        ]
    )

    schema.to_csv(
        OUT
        / "matrix_schema.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 8. Audit decision
    # ----------------------------------------------------------

    checks = {
        "source_downloaded":
            DATA.exists(),

        "canonical_gene_ids_detected":
            best_score >= 0.50,

        "expression_columns_detected":
            len(numeric) >= 6,

        "expression_nonnegative":
            negative_fraction == 0.0,

        "matrix_nonempty":
            df.shape[0] > 1000,
    }

    passed = all(
        checks.values()
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

    lines = [
        "# CF-State Phase 2 — Audit 09",
        "",
        "## Objective",
        "",
        "Freeze and inspect the Pseudomonas aeruginosa portion of "
        "GSE152295 before any iModulon projection.",
        "",
        "## Dataset",
        "",
        "- PAO1 multi-stress RNA atlas",
        "- Processed Pseudomonas expression matrix from GEO",
        "- GEO publication describes 11 infection-relevant stresses "
        "plus untreated exponential-phase control",
        "",
        "## Provenance",
        "",
        "- GSE152295 exact/project alias overlap with aeruPRECISE-411: "
        + str(
            training_alias_overlap
        ),
        "",
        "Absence of these aliases is not yet treated as exhaustive "
        "sample-level proof of independence.",
        "",
        "## Matrix",
        "",
        "- Rows: "
        + str(
            df.shape[0]
        ),
        "- Columns: "
        + str(
            df.shape[1]
        ),
        "- Gene-ID column: "
        + str(
            gene_col
        ),
        "- Numeric expression columns: "
        + str(
            len(numeric)
        ),
        "- Columns summing approximately to 1e6: "
        + str(
            n_tpm_like
        ),
        "/"
        + str(
            len(numeric)
        ),
        "",
        "## Decision",
        "",
        "PASS: "
        + str(
            passed
        ),
        "",
        "Audit 10 should only be written after the sample naming "
        "and expression scale observed here have been verified.",
    ]

    (
        OUT
        / "AUDIT_09_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 09:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_09_SUMMARY.md",
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
