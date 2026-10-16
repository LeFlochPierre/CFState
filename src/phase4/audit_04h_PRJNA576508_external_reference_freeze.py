from pathlib import Path

import pandas as pd


ROOT = Path.cwd()

FULL_MANIFEST = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "PRJNA576508_full_ENA_manifest.csv"
)

MODEL_METADATA = (
    ROOT
    / "results/phase4/audit_04b_PRJNA576508_supplement_and_decoy_freeze"
    / "Cornforth_DataS1_model_samples.csv"
)

ATLANTA_COUNTS = (
    ROOT
    / "results/phase4/audit_04g_PRJNA576508_Atlanta_full_cohort_processing"
    / "FROZEN_Atlanta_13_raw_PAO1_gene_counts.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_04h_PRJNA576508_external_reference_freeze"
)


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04h")
    print("Freeze external reference strategy before Atlanta projection")
    print("=" * 78)

    for path in [
        FULL_MANIFEST,
        MODEL_METADATA,
        ATLANTA_COUNTS,
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
        FULL_MANIFEST
    )

    model_metadata = pd.read_csv(
        MODEL_METADATA
    )

    # ----------------------------------------------------------
    # 1. Identify same-study PAO1 airway epithelial model runs
    # ----------------------------------------------------------

    title = (
        manifest[
            "sample_title"
        ]
        .fillna("")
        .astype(str)
    )

    organism = (
        manifest[
            "scientific_name"
        ]
        .fillna("")
        .astype(str)
    )

    reference_runs = manifest.loc[
        (
            organism.str.contains(
                "PAO1",
                case=False,
                regex=False,
            )
        )
        & (
            title.str.contains(
                "airway epithelial",
                case=False,
                regex=False,
            )
        )
    ].copy()

    reference_runs = reference_runs.sort_values(
        "run_accession"
    ).reset_index(
        drop=True
    )

    print()
    print("Same-study PAO1 airway epithelial runs")
    print(
        reference_runs[
            [
                "run_accession",
                "sample_accession",
                "sample_title",
                "scientific_name",
                "library_strategy",
                "library_layout",
                "instrument_model",
                "read_count",
                "fastq_md5",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 2. Freeze biological condition classification
    #
    # ENA titles contain two groups:
    #
    # - P. aeruginosa on CF airway epithelial cells
    # - P. aeruginosa on RSV-infected CF airway epithelial cells
    #
    # The second group contains the literal phrase "RSV-infected".
    #
    # We do NOT use any expression information here.
    # ----------------------------------------------------------

    reference_runs[
        "reference_condition"
    ] = [
        (
            "CFBE_RSV"
            if "rsv-infected"
            in str(sample_title).lower()
            else "CFBE_NO_RSV"
        )
        for sample_title
        in reference_runs[
            "sample_title"
        ]
    ]

    no_rsv = reference_runs.loc[
        reference_runs[
            "reference_condition"
        ]
        == "CFBE_NO_RSV"
    ].copy()

    rsv = reference_runs.loc[
        reference_runs[
            "reference_condition"
        ]
        == "CFBE_RSV"
    ].copy()

    print()
    print("Frozen reference-condition classification")
    print(
        reference_runs[
            [
                "run_accession",
                "reference_condition",
                "sample_title",
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 3. Inspect corresponding published model metadata
    #
    # Descriptive only.
    #
    # Search all cells for airway / epithelial / RSV terms so that
    # the published model-sheet context is frozen alongside ENA.
    # ----------------------------------------------------------

    relevant_mask = pd.Series(
        False,
        index=model_metadata.index,
    )

    for column in model_metadata.columns:
        text_column = (
            model_metadata[
                column
            ]
            .fillna("")
            .astype(str)
        )

        relevant_mask = (
            relevant_mask
            | text_column.str.contains(
                "epithelial",
                case=False,
                regex=False,
            )
            | text_column.str.contains(
                "RSV",
                case=False,
                regex=False,
            )
        )

    relevant_model_metadata = model_metadata.loc[
        relevant_mask
    ].copy()

    print()
    print("Relevant rows from Cornforth model metadata")

    if relevant_model_metadata.empty:
        print(
            "  none detected"
        )
    else:
        print(
            relevant_model_metadata.to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 4. Freeze reference strategy
    #
    # PRIMARY:
    #   uninfected CF airway epithelial model only.
    #
    # Rationale:
    #   same project / PAO1 / airway-associated environment /
    #   no added viral perturbation.
    #
    # Sensitivities are frozen NOW, before any CF-State result:
    #
    #   all four epithelial samples
    #   RSV-infected epithelial samples only
    #
    # ----------------------------------------------------------

    plan_rows = []

    for _, row in no_rsv.iterrows():
        plan_rows.append(
            {
                "reference_name":
                    "PRIMARY_CFBE_NO_RSV",

                "role":
                    "PRIMARY",

                "run_accession":
                    row[
                        "run_accession"
                    ],

                "reference_condition":
                    "CFBE_NO_RSV",

                "weighting":
                    "equal replicate mean",

                "rationale":
                    (
                        "Same-study PAO1 CF airway epithelial model "
                        "without added RSV perturbation."
                    ),
            }
        )

    for _, row in reference_runs.iterrows():
        plan_rows.append(
            {
                "reference_name":
                    "SENSITIVITY_CFBE_ALL4",

                "role":
                    "SENSITIVITY",

                "run_accession":
                    row[
                        "run_accession"
                    ],

                "reference_condition":
                    row[
                        "reference_condition"
                    ],

                "weighting":
                    "equal replicate mean",

                "rationale":
                    (
                        "All same-study PAO1 CF airway epithelial "
                        "samples pooled independently of RSV state."
                    ),
            }
        )

    for _, row in rsv.iterrows():
        plan_rows.append(
            {
                "reference_name":
                    "SENSITIVITY_CFBE_RSV",

                "role":
                    "SENSITIVITY",

                "run_accession":
                    row[
                        "run_accession"
                    ],

                "reference_condition":
                    "CFBE_RSV",

                "weighting":
                    "equal replicate mean",

                "rationale":
                    (
                        "Same-study PAO1 CF airway epithelial model "
                        "with RSV perturbation."
                    ),
            }
        )

    plan = pd.DataFrame(
        plan_rows
    )

    print()
    print("Frozen external-reference plan")
    print(
        plan.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Freeze later validation rules conceptually
    #
    # These do not calculate anything yet.
    # ----------------------------------------------------------

    validation_plan = pd.DataFrame(
        [
            {
                "rule":
                    "primary_reference",

                "value":
                    "PRIMARY_CFBE_NO_RSV",

                "status":
                    "FROZEN_BEFORE_PROJECTION",
            },

            {
                "rule":
                    "sensitivity_reference_1",

                "value":
                    "SENSITIVITY_CFBE_ALL4",

                "status":
                    "FROZEN_BEFORE_PROJECTION",
            },

            {
                "rule":
                    "sensitivity_reference_2",

                "value":
                    "SENSITIVITY_CFBE_RSV",

                "status":
                    "FROZEN_BEFORE_PROJECTION",
            },

            {
                "rule":
                    "reference_expression_summary",

                "value":
                    (
                        "equal mean of replicate log2(TPM+1) "
                        "profiles within each frozen reference"
                    ),

                "status":
                    "FROZEN_BEFORE_PROJECTION",
            },

            {
                "rule":
                    "Atlanta_contrast",

                "value":
                    (
                        "each sputum log2(TPM+1) profile minus "
                        "frozen external-reference mean"
                    ),

                "status":
                    "FROZEN_BEFORE_PROJECTION",
            },

            {
                "rule":
                    "CF_candidate_usage",

                "value":
                    (
                        "Audit-03h vector remains untouched until "
                        "reference samples are processed and Atlanta "
                        "projection QC is complete"
                    ),

                "status":
                    "FROZEN_BEFORE_PROJECTION",
            },
        ]
    )

    # ----------------------------------------------------------
    # 6. Checks
    # ----------------------------------------------------------

    all_rnaseq = bool(
        reference_runs[
            "library_strategy"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "RNA-SEQ"
        )
        .all()
    )

    all_single = bool(
        reference_runs[
            "library_layout"
        ]
        .astype(str)
        .str.upper()
        .eq(
            "SINGLE"
        )
        .all()
    )

    all_nextseq = bool(
        reference_runs[
            "instrument_model"
        ]
        .astype(str)
        .str.contains(
            "NextSeq",
            case=False,
            regex=False,
        )
        .all()
    )

    all_md5 = bool(
        reference_runs[
            "fastq_md5"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(
            0
        )
        .all()
    )

    all_fastq = bool(
        reference_runs[
            "fastq_ftp"
        ]
        .fillna("")
        .astype(str)
        .str.len()
        .gt(
            0
        )
        .all()
    )

    checks = {
        "exactly_4_same_study_PAO1_epithelial_runs":
            len(
                reference_runs
            )
            == 4,

        "exactly_2_no_RSV_reference_runs":
            len(
                no_rsv
            )
            == 2,

        "exactly_2_RSV_reference_runs":
            len(
                rsv
            )
            == 2,

        "all_reference_runs_RNAseq":
            all_rnaseq,

        "all_reference_runs_single_end":
            all_single,

        "all_reference_runs_NextSeq":
            all_nextseq,

        "all_reference_FASTQ_links_present":
            all_fastq,

        "all_reference_MD5s_present":
            all_md5,

        "Atlanta_13_count_matrix_already_frozen":
            ATLANTA_COUNTS.exists(),

        "primary_reference_has_2_runs":
            int(
                (
                    plan[
                        "reference_name"
                    ]
                    == "PRIMARY_CFBE_NO_RSV"
                ).sum()
            )
            == 2,

        "all4_sensitivity_has_4_runs":
            int(
                (
                    plan[
                        "reference_name"
                    ]
                    == "SENSITIVITY_CFBE_ALL4"
                ).sum()
            )
            == 4,

        "RSV_sensitivity_has_2_runs":
            int(
                (
                    plan[
                        "reference_name"
                    ]
                    == "SENSITIVITY_CFBE_RSV"
                ).sum()
            )
            == 2,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04h checks")

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
        "PRJNA576508_EXTERNAL_REFERENCE_STRATEGY_FROZEN_BEFORE_PROJECTION"
        if passed
        else "PRJNA576508_EXTERNAL_REFERENCE_STRATEGY_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: this audit uses only study design and metadata. "
        "No reference expression has been processed and no Atlanta "
        "sample has been compared with the frozen Audit-03h "
        "CF-airway programme."
    )

    # ----------------------------------------------------------
    # 7. Save
    # ----------------------------------------------------------

    reference_runs.to_csv(
        OUT
        / "FROZEN_same_study_PAO1_epithelial_reference_runs.csv",
        index=False,
    )

    plan.to_csv(
        OUT
        / "FROZEN_Atlanta_external_reference_plan.csv",
        index=False,
    )

    validation_plan.to_csv(
        OUT
        / "FROZEN_future_validation_rules.csv",
        index=False,
    )

    relevant_model_metadata.to_csv(
        OUT
        / "Cornforth_model_metadata_relevant_to_epithelial_reference.csv",
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
        / "AUDIT_04H_STATUS.txt"
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
            "Failed Audit 04h checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
