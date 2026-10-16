from pathlib import Path
import re

import pandas as pd


ROOT = Path.cwd()

S1 = (
    ROOT
    / "data/raw/phase4/PRJNA576508/published_supplement"
    / "mbio.03042-19-sd001.xlsx"
)

S2 = (
    ROOT
    / "data/raw/phase4/PRJNA576508/published_supplement"
    / "mbio.03042-19-sd002.xlsx"
)

FROZEN_COHORT = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "FROZEN_PRJNA576508_independent_third_cohort.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_04b2_PRJNA576508_supplement_parser_repair"
)


def extract_em_label(value):
    """
    Recover EM identifier anywhere inside strings such as:

        HSP_EM13
        HSP_EM35B

    The trailing letter identifies the published sample variant,
    while Audit 04a froze the corresponding ENA labels as EM35,
    EM37, EM48, EM53.

    Therefore the frozen comparison coordinate is EM + digits.
    """

    text = str(
        value
    ).strip().upper()

    match = re.search(
        r"EM(\d+)[A-Z]?",
        text,
    )

    if match is None:
        return None

    return (
        "EM"
        + match.group(1)
    )


def is_ncbi_sequence_accession(value):
    """
    Conservative full-string genomic accession parser.

    Examples accepted:
        NC_005966.1
        NZ_CP011007.1
        NZ_CM000955.1
        AE010301.2
        CP012718.1

    Requiring digits + version suffix prevents words such as
    Acinetobacter, APS, acnes, etc. from being misclassified.
    """

    text = str(
        value
    ).strip().upper()

    pattern = (
        r"[A-Z]{1,4}"
        r"(?:_[A-Z]{0,4})?"
        r"\d+"
        r"\.\d+"
    )

    return (
        re.fullmatch(
            pattern,
            text,
        )
        is not None
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04b2")
    print("PRJNA576508 supplement parser repair and exact decoy freeze")
    print("=" * 78)

    for path in [
        S1,
        S2,
        FROZEN_COHORT,
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
    # 1. Read published supplement
    # ----------------------------------------------------------

    excel = pd.ExcelFile(
        S1
    )

    if len(
        excel.sheet_names
    ) < 3:
        raise RuntimeError(
            "Data Set S1 has fewer than three sheets."
        )

    human = pd.read_excel(
        S1,
        sheet_name=excel.sheet_names[
            0
        ],
    )

    models = pd.read_excel(
        S1,
        sheet_name=excel.sheet_names[
            1
        ],
    )

    decoys = pd.read_excel(
        S1,
        sheet_name=excel.sheet_names[
            2
        ],
    )

    frozen = pd.read_csv(
        FROZEN_COHORT
    )

    # ----------------------------------------------------------
    # 2. Repair Atlanta sample parser
    # ----------------------------------------------------------

    if "Sample" not in human.columns:
        raise RuntimeError(
            "Published human sheet has no Sample column."
        )

    human[
        "normalized_EM_label"
    ] = human[
        "Sample"
    ].map(
        extract_em_label
    )

    frozen_labels = sorted(
        set(
            frozen[
                "short_sample_label"
            ]
            .astype(str)
            .str.strip()
            .str.upper()
        )
    )

    published_atlanta = human.loc[
        human[
            "Clinic location"
        ]
        .astype(str)
        .str.strip()
        .str.upper()
        .eq(
            "USA"
        )
    ].copy()

    published_atlanta_labels = sorted(
        set(
            published_atlanta[
                "normalized_EM_label"
            ]
            .dropna()
            .astype(str)
        )
    )

    recovered = sorted(
        set(
            frozen_labels
        )
        .intersection(
            published_atlanta_labels
        )
    )

    missing = sorted(
        set(
            frozen_labels
        )
        - set(
            published_atlanta_labels
        )
    )

    extra = sorted(
        set(
            published_atlanta_labels
        )
        - set(
            frozen_labels
        )
    )

    print()
    print("Atlanta sample parser repair")
    print(
        "  frozen Audit-04a labels:",
        len(
            frozen_labels
        ),
        frozen_labels,
    )
    print(
        "  published USA rows:",
        len(
            published_atlanta
        ),
    )
    print(
        "  published normalized labels:",
        len(
            published_atlanta_labels
        ),
        published_atlanta_labels,
    )
    print(
        "  recovered:",
        len(
            recovered
        ),
        recovered,
    )
    print(
        "  missing:",
        missing,
    )
    print(
        "  extra:",
        extra,
    )

    print()
    print("Published Atlanta metadata")
    print(
        published_atlanta[
            [
                "Sample",
                "normalized_EM_label",
                "Clinic location",
                "Sex",
                "total reads",
                "reads mapping \nto PAO1 genes",
                "number of PAO1\n genes with at least\none read mapping",
                "SRA accession",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 3. Repair decoy accession parser
    # ----------------------------------------------------------

    decoys = decoys.dropna(
        how="all"
    ).copy()

    sequence_column = decoys.columns[
        0
    ]

    species_column = decoys.columns[
        1
    ]

    decoys[
        "sequence_raw"
    ] = (
        decoys[
            sequence_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    decoys[
        "species_strain_raw"
    ] = (
        decoys[
            species_column
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    decoys[
        "is_exact_sequence_accession"
    ] = decoys[
        "sequence_raw"
    ].map(
        is_ncbi_sequence_accession
    )

    accession_rows = decoys.loc[
        decoys[
            "is_exact_sequence_accession"
        ]
    ].copy()

    invalid_nonempty = decoys.loc[
        (
            decoys[
                "sequence_raw"
            ]
            .ne("")
        )
        & (
            ~decoys[
                "is_exact_sequence_accession"
            ]
        )
    ].copy()

    accessions = (
        accession_rows[
            "sequence_raw"
        ]
        .str.upper()
        .tolist()
    )

    unique_accessions = sorted(
        set(
            accessions
        )
    )

    duplicated_accessions = sorted(
        accession_rows.loc[
            accession_rows[
                "sequence_raw"
            ]
            .str.upper()
            .duplicated(
                keep=False
            ),
            "sequence_raw",
        ]
        .str.upper()
        .unique()
        .tolist()
    )

    print()
    print("Exact decoy sequence-accession recovery")
    print(
        "  published non-empty rows:",
        len(
            decoys
        ),
    )
    print(
        "  exact accession rows:",
        len(
            accession_rows
        ),
    )
    print(
        "  unique accessions:",
        len(
            unique_accessions
        ),
    )
    print(
        "  duplicated accessions:",
        len(
            duplicated_accessions
        ),
    )

    if duplicated_accessions:
        print(
            "  duplicates:",
            duplicated_accessions,
        )

    print()
    print("Non-accession entries in sequence column")

    if invalid_nonempty.empty:
        print(
            "  none"
        )
    else:
        print(
            invalid_nonempty[
                [
                    "sequence_raw",
                    "species_strain_raw",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print()
    print("Exact published decoy sequence inventory")
    print(
        accession_rows[
            [
                "sequence_raw",
                "species_strain_raw",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. Published PAO1 mapping-depth sanity check
    #
    # Descriptive only. These are the authors' published results,
    # not our own future processing QC.
    # ----------------------------------------------------------

    pao1_reads_col = (
        "reads mapping \nto PAO1 genes"
    )

    pao1_genes_col = (
        "number of PAO1\n genes with at least\none read mapping"
    )

    published_atlanta[
        pao1_reads_col
    ] = pd.to_numeric(
        published_atlanta[
            pao1_reads_col
        ],
        errors="raise",
    )

    published_atlanta[
        pao1_genes_col
    ] = pd.to_numeric(
        published_atlanta[
            pao1_genes_col
        ],
        errors="raise",
    )

    minimum_pao1_reads = int(
        published_atlanta[
            pao1_reads_col
        ].min()
    )

    minimum_pao1_genes = int(
        published_atlanta[
            pao1_genes_col
        ].min()
    )

    median_pao1_genes = float(
        published_atlanta[
            pao1_genes_col
        ].median()
    )

    print()
    print("Published Atlanta PAO1 mapping summary")
    print(
        "  minimum reads mapping to PAO1 genes:",
        minimum_pao1_reads,
    )
    print(
        "  minimum PAO1 genes with >=1 read:",
        minimum_pao1_genes,
    )
    print(
        "  median PAO1 genes with >=1 read:",
        round(
            median_pao1_genes,
            1,
        ),
    )

    # ----------------------------------------------------------
    # 5. Freeze repaired outputs
    # ----------------------------------------------------------

    published_atlanta.to_csv(
        OUT
        / "FROZEN_Cornforth_13_Atlanta_sample_metadata.csv",
        index=False,
    )

    accession_rows[
        [
            "sequence_raw",
            "species_strain_raw",
        ]
    ].rename(
        columns={
            "sequence_raw":
                "sequence_accession",

            "species_strain_raw":
                "species_strain",
        }
    ).to_csv(
        OUT
        / "FROZEN_Cornforth_decoy_sequence_inventory.csv",
        index=False,
    )

    invalid_nonempty[
        [
            "sequence_raw",
            "species_strain_raw",
        ]
    ].to_csv(
        OUT
        / "decoy_non_accession_rows_for_review.csv",
        index=False,
    )

    human.to_csv(
        OUT
        / "Cornforth_human_metadata_with_repaired_labels.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 6. Checks
    #
    # We do not force "101" accession rows here because the paper
    # describes 101 GENOMES whereas the supplement may list
    # multiple chromosome/sequence accessions for some genomes.
    # We freeze the exact published sequence inventory instead.
    # ----------------------------------------------------------

    checks = {
        "13_USA_rows_in_published_metadata":
            len(
                published_atlanta
            )
            == 13,

        "all_13_Audit04a_labels_recovered":
            len(
                recovered
            )
            == 13,

        "no_frozen_Atlanta_labels_missing":
            len(
                missing
            )
            == 0,

        "no_extra_USA_EM_labels":
            len(
                extra
            )
            == 0,

        "at_least_100_exact_decoy_sequence_accessions":
            len(
                unique_accessions
            )
            >= 100,

        "no_duplicate_decoy_accessions":
            len(
                duplicated_accessions
            )
            == 0,

        "all_13_published_samples_have_positive_PAO1_reads":
            bool(
                (
                    published_atlanta[
                        pao1_reads_col
                    ]
                    > 0
                ).all()
            ),

        "all_13_published_samples_cover_at_least_4000_PAO1_genes":
            bool(
                (
                    published_atlanta[
                        pao1_genes_col
                    ]
                    >= 4000
                ).all()
            ),

        "DataS2_present":
            S2.exists(),
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04b2 checks")

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
        "PRJNA576508_SUPPLEMENT_AND_EXACT_DECOY_INVENTORY_FROZEN"
        if passed
        else "PRJNA576508_SUPPLEMENT_PARSER_REPAIR_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: Audit 04b remains formally REVIEW_REQUIRED "
        "because of the original parser bug. Audit 04b2 documents "
        "and repairs that failure without changing the biological "
        "cohort or any CF-airway candidate."
    )

    print()
    print(
        "No PRJNA576508 expression values have yet been compared "
        "with the frozen Audit-03h CF-airway candidate."
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
        / "AUDIT_04B2_STATUS.txt"
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
            "Failed Audit 04b2 checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
