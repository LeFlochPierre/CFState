from pathlib import Path

import pandas as pd


SEARCH_ROOTS = [
    Path("data/processed/biofilm"),
    Path("data/processed/expanded_biofilm"),
]

STUDIES = [
    "GSE136111",
    "GSE203348",
    "GSE87213",
    "GSE299861",
    "GSE191103",
    "GSE120602",
    "GSE12207",
]

OUT_DIR = Path(
    "results/audit/cross_study_modeling"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 53: "
    "inventory cross-study modelling matrices"
)
print("=" * 100)


# ============================================================
# Find CSV files
# ============================================================

all_csvs = []


for root in SEARCH_ROOTS:

    if not root.exists():
        continue

    for path in root.rglob(
        "*.csv"
    ):

        all_csvs.append(
            path
        )


records = []


# ============================================================
# Inspect candidates study by study
# ============================================================

for study in STUDIES:

    print()
    print("=" * 100)
    print(study)
    print("=" * 100)

    candidates = [
        path
        for path in all_csvs
        if study.lower()
        in str(path).lower()
    ]


    if not candidates:

        print(
            "NO CSV CANDIDATES FOUND"
        )

        records.append(
            {
                "study":
                    study,

                "path":
                    "",

                "rows":
                    None,

                "columns":
                    None,

                "orientation_guess":
                    "missing",

                "metadata_columns":
                    "",

                "gene_like_columns":
                    None,

                "status":
                    "no_candidate",
            }
        )

        continue


    for path in sorted(
        candidates
    ):

        try:

            table = pd.read_csv(
                path
            )

        except Exception as exc:

            print()
            print(
                "Could not read:"
            )

            print(
                str(path)
            )

            print(
                str(exc)
            )

            continue


        columns = [
            str(column)
            for column in table.columns
        ]


        metadata_candidates = [
            "sample",
            "study",
            "gsm",
            "state",
            "condition",
            "replicate",
            "label",
            "group",
        ]


        metadata_present = [
            column
            for column in metadata_candidates
            if column in columns
        ]


        gene_like_columns = [
            column
            for column in columns
            if (
                len(column) == 6
                and column.startswith("PA")
                and column[2:].isdigit()
            )
        ]


        first_column = (
            columns[0]
            if columns
            else ""
        )


        first_values = []

        if len(table) > 0:

            first_values = (
                table.iloc[
                    :10,
                    0,
                ]
                .astype(str)
                .tolist()
            )


        first_value_gene_count = sum(
            (
                len(value) == 6
                and value.startswith("PA")
                and value[2:].isdigit()
            )
            for value in first_values
        )


        if len(
            gene_like_columns
        ) >= 4000:

            orientation = (
                "samples_x_genes"
            )

        elif (
            first_value_gene_count
            >= 5
            and table.shape[0]
            >= 4000
        ):

            orientation = (
                "genes_x_samples"
            )

        else:

            orientation = (
                "other_or_unclear"
            )


        print()

        print(
            "Path:"
        )

        print(
            str(path)
        )

        print(
            "Shape: "
            + str(
                table.shape
            )
        )

        print(
            "Orientation guess: "
            + orientation
        )

        print(
            "Metadata columns: "
            + str(
                metadata_present
            )
        )

        print(
            "PAxxxx gene columns: "
            + str(
                len(
                    gene_like_columns
                )
            )
        )


        if "condition" in table.columns:

            print(
                "Condition counts:"
            )

            print(
                table[
                    "condition"
                ]
                .value_counts(
                    dropna=False
                )
                .to_string()
            )


        elif "state" in table.columns:

            print(
                "State counts:"
            )

            print(
                table[
                    "state"
                ]
                .value_counts(
                    dropna=False
                )
                .to_string()
            )


        records.append(
            {
                "study":
                    study,

                "path":
                    str(path),

                "rows":
                    int(
                        table.shape[0]
                    ),

                "columns":
                    int(
                        table.shape[1]
                    ),

                "orientation_guess":
                    orientation,

                "metadata_columns":
                    "|".join(
                        metadata_present
                    ),

                "gene_like_columns":
                    len(
                        gene_like_columns
                    ),

                "status":
                    "read_ok",
            }
        )


# ============================================================
# Summary
# ============================================================

inventory = pd.DataFrame(
    records
)


inventory.to_csv(
    OUT_DIR
    / "Audit53_cross_study_matrix_inventory.csv",
    index=False,
)


print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)


for study in STUDIES:

    subset = inventory.loc[
        inventory[
            "study"
        ]
        == study
    ]


    usable = subset.loc[
        subset[
            "orientation_guess"
        ]
        .isin(
            [
                "samples_x_genes",
                "genes_x_samples",
            ]
        )
    ]


    print()

    print(
        study
    )

    if usable.empty:

        print(
            "  No obvious expression matrix found."
        )

    else:

        for _, row in usable.iterrows():

            print(
                "  "
                + str(
                    row[
                        "orientation_guess"
                    ]
                )
                + " | "
                + str(
                    row[
                        "rows"
                    ]
                )
                + " x "
                + str(
                    row[
                        "columns"
                    ]
                )
            )

            print(
                "  "
                + str(
                    row[
                        "path"
                    ]
                )
            )


print()
print("=" * 100)
print("AUDIT 53 COMPLETE")
print("=" * 100)

print(
    "Inventory saved to:"
)

print(
    OUT_DIR
    / "Audit53_cross_study_matrix_inventory.csv"
)

print()
