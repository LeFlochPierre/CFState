from pathlib import Path
import pandas as pd


ROOT = Path.cwd()

MANIFEST = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "PRJNA576508_full_ENA_manifest.csv"
)

SUPPLEMENT = (
    ROOT
    / "data/raw/phase4/PRJNA576508/published_supplement"
    / "mbio.03042-19-sd001.xlsx"
)

OUT = (
    ROOT
    / "results/phase4/audit_04h3_PRJNA576508_reference_plan_repair"
)

REFERENCE_RUNS = {
    "SRR10267767": "R2",
    "SRR10267768": "R1",
    "SRR10267775": "C2",
    "SRR10267776": "C1",
}


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04h3")
    print("Repair and freeze Atlanta CFBE external-reference plan")
    print("=" * 78)

    for path in [
        MANIFEST,
        SUPPLEMENT,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST
    )

    runs = manifest.loc[
        manifest[
            "run_accession"
        ]
        .astype(str)
        .isin(
            REFERENCE_RUNS.keys()
        )
    ].copy()

    if len(runs) != 4:
        raise RuntimeError(
            "Expected exactly four frozen CFBE reference runs."
        )

    runs[
        "pair_sample"
    ] = runs[
        "run_accession"
    ].map(
        REFERENCE_RUNS
    )

    runs[
        "pair"
    ] = runs[
        "pair_sample"
    ].str[
        0
    ]

    runs[
        "replicate"
    ] = runs[
        "pair_sample"
    ].str[
        1
    ].astype(
        int
    )

    runs = runs.sort_values(
        [
            "pair",
            "replicate",
        ]
    ).reset_index(
        drop=True
    )

    # ----------------------------------------------------------
    # 1. Explicit deposited-title interpretation
    # ----------------------------------------------------------

    runs[
        "explicit_without_RSV"
    ] = (
        runs[
            "sample_title"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            "without rsv infection",
            regex=False,
        )
    )

    runs[
        "contains_RSV_infected_phrase"
    ] = (
        runs[
            "sample_title"
        ]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(
            "rsv-infected",
            regex=False,
        )
    )

    runs[
        "conservative_condition"
    ] = "CFBE_NO_RSV"

    print()
    print("Frozen four-run metadata")
    print(
        runs[
            [
                "run_accession",
                "pair_sample",
                "pair",
                "replicate",
                "sample_accession",
                "sample_title",
                "explicit_without_RSV",
                "contains_RSV_infected_phrase",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 2. Print exact supplement headers + C/R rows
    # ----------------------------------------------------------

    sheet = "laboratory model information"

    supplement = pd.read_excel(
        SUPPLEMENT,
        sheet_name=sheet,
        dtype=str,
    ).fillna(
        ""
    )

    print()
    print("Original supplement column headers")
    print(
        list(
            supplement.columns
        )
    )

    first_column = supplement.columns[
        0
    ]

    supplement[
        "_row_label"
    ] = (
        supplement[
            first_column
        ]
        .astype(str)
        .str.strip()
    )

    model_rows = supplement.loc[
        supplement[
            "_row_label"
        ].isin(
            [
                "C1",
                "C2",
                "R1",
                "R2",
            ]
        )
    ].copy()

    print()
    print("Exact C1/C2/R1/R2 supplement rows")

    if model_rows.empty:
        print(
            "  none recovered"
        )

    else:
        print(
            model_rows.to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 3. Freeze corrected reference plan
    #
    # IMPORTANT:
    #
    # All deposited samples explicitly say "without RSV infection".
    #
    # Therefore C/R is retained only as pair provenance.
    # It is NOT interpreted as RSV status.
    #
    # Primary:
    #   equal mean of all four samples.
    #
    # Sensitivity:
    #   C pair only
    #   R pair only
    #
    # No expression information enters this choice.
    # ----------------------------------------------------------

    plan_rows = []

    for _, row in runs.iterrows():
        plan_rows.append(
            {
                "reference_name":
                    "PRIMARY_CFBE_ALL4",

                "role":
                    "PRIMARY",

                "run_accession":
                    row[
                        "run_accession"
                    ],

                "pair_sample":
                    row[
                        "pair_sample"
                    ],

                "pair":
                    row[
                        "pair"
                    ],

                "condition":
                    "CFBE_NO_RSV",

                "weighting":
                    "equal replicate mean",

                "rationale":
                    (
                        "All four same-study PAO1 CF airway "
                        "epithelial samples explicitly deposited "
                        "as without RSV infection."
                    ),
            }
        )

    for _, row in runs.loc[
        runs[
            "pair"
        ]
        == "C"
    ].iterrows():
        plan_rows.append(
            {
                "reference_name":
                    "SENSITIVITY_CFBE_C_PAIR",

                "role":
                    "SENSITIVITY",

                "run_accession":
                    row[
                        "run_accession"
                    ],

                "pair_sample":
                    row[
                        "pair_sample"
                    ],

                "pair":
                    "C",

                "condition":
                    "CFBE_NO_RSV",

                "weighting":
                    "equal replicate mean",

                "rationale":
                    (
                        "C1/C2 experimental pair retained as "
                        "pre-specified provenance sensitivity."
                    ),
            }
        )

    for _, row in runs.loc[
        runs[
            "pair"
        ]
        == "R"
    ].iterrows():
        plan_rows.append(
            {
                "reference_name":
                    "SENSITIVITY_CFBE_R_PAIR",

                "role":
                    "SENSITIVITY",

                "run_accession":
                    row[
                        "run_accession"
                    ],

                "pair_sample":
                    row[
                        "pair_sample"
                    ],

                "pair":
                    "R",

                "condition":
                    "CFBE_NO_RSV",

                "weighting":
                    "equal replicate mean",

                "rationale":
                    (
                        "R1/R2 experimental pair retained as "
                        "pre-specified provenance sensitivity; "
                        "R is not interpreted as RSV-positive."
                    ),
            }
        )

    plan = pd.DataFrame(
        plan_rows
    )

    print()
    print("Corrected frozen external-reference plan")
    print(
        plan.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. Freeze downstream rules BEFORE expression processing
    # ----------------------------------------------------------

    validation_rules = pd.DataFrame(
        [
            {
                "rule":
                    "primary_reference",

                "value":
                    "PRIMARY_CFBE_ALL4",

                "reason":
                    (
                        "Use all four virus-free same-study "
                        "CFBE samples with equal weighting."
                    ),
            },

            {
                "rule":
                    "sensitivity_reference_C",

                "value":
                    "SENSITIVITY_CFBE_C_PAIR",

                "reason":
                    (
                        "Check robustness to C-pair provenance."
                    ),
            },

            {
                "rule":
                    "sensitivity_reference_R",

                "value":
                    "SENSITIVITY_CFBE_R_PAIR",

                "reason":
                    (
                        "Check robustness to R-pair provenance."
                    ),
            },

            {
                "rule":
                    "C_R_interpretation",

                "value":
                    (
                        "experimental provenance only; "
                        "not RSV status"
                    ),

                "reason":
                    (
                        "All four deposited sample titles explicitly "
                        "state without RSV infection."
                    ),
            },

            {
                "rule":
                    "expression_reference_summary",

                "value":
                    (
                        "equal mean of replicate log2(TPM+1) "
                        "profiles within each frozen reference"
                    ),

                "reason":
                    (
                        "Fixed before any Atlanta CF-State result."
                    ),
            },

            {
                "rule":
                    "Atlanta_contrast",

                "value":
                    (
                        "each Atlanta sputum log2(TPM+1) profile "
                        "minus frozen external-reference mean"
                    ),

                "reason":
                    (
                        "Avoid within-Atlanta centering, which would "
                        "remove a common airway programme."
                    ),
            },

            {
                "rule":
                    "candidate_blinding",

                "value":
                    (
                        "Audit-03h candidate remains untouched until "
                        "reference processing and projection QC pass"
                    ),

                "reason":
                    (
                        "Preserve prospective third-cohort validation."
                    ),
            },
        ]
    )

    print()
    print("Frozen downstream rules")
    print(
        validation_rules.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Checks
    # ----------------------------------------------------------

    all_four_without_rsv = bool(
        runs[
            "explicit_without_RSV"
        ].all()
    )

    c_runs = set(
        runs.loc[
            runs[
                "pair"
            ]
            == "C",
            "run_accession",
        ]
    )

    r_runs = set(
        runs.loc[
            runs[
                "pair"
            ]
            == "R",
            "run_accession",
        ]
    )

    checks = {
        "exactly_4_reference_runs":
            len(
                runs
            )
            == 4,

        "all_4_titles_explicitly_without_RSV":
            all_four_without_rsv,

        "C_pair_exactly_2_runs":
            len(
                c_runs
            )
            == 2,

        "R_pair_exactly_2_runs":
            len(
                r_runs
            )
            == 2,

        "C_pair_is_SRR10267775_76":
            c_runs
            == {
                "SRR10267775",
                "SRR10267776",
            },

        "R_pair_is_SRR10267767_68":
            r_runs
            == {
                "SRR10267767",
                "SRR10267768",
            },

        "primary_reference_has_all_4_runs":
            int(
                (
                    plan[
                        "reference_name"
                    ]
                    == "PRIMARY_CFBE_ALL4"
                ).sum()
            )
            == 4,

        "C_sensitivity_has_2_runs":
            int(
                (
                    plan[
                        "reference_name"
                    ]
                    == "SENSITIVITY_CFBE_C_PAIR"
                ).sum()
            )
            == 2,

        "R_sensitivity_has_2_runs":
            int(
                (
                    plan[
                        "reference_name"
                    ]
                    == "SENSITIVITY_CFBE_R_PAIR"
                ).sum()
            )
            == 2,

        "no_expression_used":
            True,

        "no_Audit03h_candidate_used":
            True,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04h3 checks")

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
        "PRJNA576508_CFBE_ALL4_EXTERNAL_REFERENCE_PLAN_FROZEN"
        if passed
        else "PRJNA576508_CFBE_REFERENCE_PLAN_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: Audit 04h remains on record as a mechanically "
        "passing metadata freeze, but its RSV/no-RSV interpretation "
        "is superseded by Audits 04h2 and 04h3."
    )

    print()
    print(
        "C/R labels are retained only as experimental-pair "
        "provenance. No deposited sample is treated as RSV-positive."
    )

    print()
    print(
        "No expression values, iModulon activities, controlled-state "
        "decomposition, or Audit-03h candidate values were used."
    )

    # ----------------------------------------------------------
    # 6. Save
    # ----------------------------------------------------------

    runs.to_csv(
        OUT
        / "FROZEN_four_CFBE_reference_runs.csv",
        index=False,
    )

    model_rows.to_csv(
        OUT
        / "Cornforth_DataS1_exact_C1_C2_R1_R2_rows.csv",
        index=False,
    )

    plan.to_csv(
        OUT
        / "FROZEN_corrected_external_reference_plan.csv",
        index=False,
    )

    validation_rules.to_csv(
        OUT
        / "FROZEN_downstream_validation_rules.csv",
        index=False,
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
        / "AUDIT_04H3_STATUS.txt"
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
            "Failed Audit 04h3 checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
