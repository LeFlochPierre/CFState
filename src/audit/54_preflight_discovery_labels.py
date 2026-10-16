from pathlib import Path

import pandas as pd


FILES = {
    "GSE136111": Path(
        "data/processed/biofilm/"
        "GSE136111_harmonised_unscaled_sample_by_gene.csv"
    ),

    "GSE203348": Path(
        "data/processed/biofilm/"
        "GSE203348_harmonised_unscaled_sample_by_gene.csv"
    ),

    "GSE87213": Path(
        "data/processed/biofilm/"
        "GSE87213_harmonised_unscaled_sample_by_gene.csv"
    ),
}


SEARCH_ROOTS = [
    Path("data/metadata"),
    Path("results/audit"),
    Path("data/processed/biofilm"),
]


print()
print("=" * 100)
print(
    "CF-STATE — Audit 54: "
    "preflight labels for original discovery studies"
)
print("=" * 100)


for study, path in FILES.items():

    print()
    print("=" * 100)
    print(study)
    print("=" * 100)

    if not path.exists():

        raise RuntimeError(
            "Missing matrix: "
            + str(path)
        )


    table = pd.read_csv(
        path
    )


    print(
        "Matrix:"
    )

    print(
        str(path)
    )

    print(
        "Shape: "
        + str(table.shape)
    )

    print(
        "First column name: "
        + str(table.columns[0])
    )


    print()
    print("ROW / SAMPLE IDENTIFIERS")
    print("-" * 100)


    first_column = (
        table.iloc[
            :,
            0
        ]
        .astype(str)
    )


    for index, value in enumerate(
        first_column,
        start=1,
    ):

        print(
            str(index)
            + "\t"
            + value
        )


    # ========================================================
    # Basic expression sanity check
    # ========================================================

    gene_columns = [
        column
        for column in table.columns
        if (
            len(
                str(column)
            )
            == 6
            and str(column).startswith(
                "PA"
            )
            and str(column)[
                2:
            ].isdigit()
        )
    ]


    print()
    print(
        "PAxxxx gene columns: "
        + str(
            len(
                gene_columns
            )
        )
    )


    # ========================================================
    # Search repository for files mentioning this study
    # ========================================================

    print()
    print("POSSIBLE METADATA / MANIFEST FILES")
    print("-" * 100)


    candidates = []


    for root in SEARCH_ROOTS:

        if not root.exists():
            continue

        for candidate in root.rglob(
            "*"
        ):

            if not candidate.is_file():
                continue

            name = candidate.name.lower()

            if study.lower() not in name:
                continue

            if candidate == path:
                continue

            if candidate.suffix.lower() not in {
                ".csv",
                ".tsv",
                ".txt",
            }:
                continue

            candidates.append(
                candidate
            )


    if not candidates:

        print(
            "None found by filename."
        )

    else:

        for candidate in sorted(
            candidates
        ):

            print(
                str(candidate)
            )


            try:

                if candidate.suffix.lower() == ".tsv":

                    meta = pd.read_csv(
                        candidate,
                        sep="\t",
                    )

                else:

                    meta = pd.read_csv(
                        candidate,
                    )

            except Exception:

                continue


            useful_columns = [
                column
                for column in meta.columns
                if any(
                    word in str(
                        column
                    ).lower()
                    for word in [
                        "sample",
                        "state",
                        "condition",
                        "group",
                        "biofilm",
                        "plank",
                        "label",
                        "gsm",
                    ]
                )
            ]


            if useful_columns:

                print(
                    "  shape: "
                    + str(
                        meta.shape
                    )
                )

                print(
                    "  useful columns: "
                    + str(
                        useful_columns
                    )
                )


                preview_columns = (
                    useful_columns[
                        :6
                    ]
                )


                print(
                    meta[
                        preview_columns
                    ]
                    .head(
                        12
                    )
                    .to_string(
                        index=False
                    )
                )


                print()


print()
print("=" * 100)
print("AUDIT 54 COMPLETE")
print("=" * 100)

print(
    "Do not assign discovery labels from row order yet."
)

print(
    "Use the printed sample identifiers / metadata "
    "to define explicit biofilm and planktonic labels."
)

print()
