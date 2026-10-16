from pathlib import Path
import re

import pandas as pd


ROOT = Path.cwd()

MANIFEST = (
    ROOT
    / "results/phase4/audit_03a_PRJEB24688_replication_preflight"
    / "PRJEB24688_ENA_run_manifest.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_03b_PRJEB24688_freeze_study_design"
)


PILOT_SAMPLES = {
    "P11F2_S1",
    "P24M1_S1",
    "P30M0_S1",
}


def parse_invivo_title(title):
    text = str(title).strip().upper()

    match = re.fullmatch(
        r"(P\d+[MF]\d+)_INVIVO_(S\d+)(?:T(\d+))?",
        text,
    )

    if match is None:
        return None

    patient = match.group(1)
    sample = match.group(2)
    technical = match.group(3)

    biological_sample = (
        patient
        + "_"
        + sample
    )

    technical_label = (
        "T" + technical
        if technical is not None
        else "SINGLE"
    )

    return {
        "patient":
            patient,

        "biological_sample":
            biological_sample,

        "technical_replicate":
            technical_label,
    }


def parse_invitro_reference(title):
    text = str(title).strip()
    upper = text.upper()

    patient = None
    isolate = None

    if upper.startswith("PA14_"):
        patient = "LAB_PA14"
        isolate = "PA14"

    elif "P24M1A" in upper:
        patient = "P24M1"
        isolate = "P24M1A"

    elif "P24M1B" in upper:
        patient = "P24M1"
        isolate = "P24M1B"

    elif "P30M0" in upper:
        patient = "P30M0"
        isolate = "P30M0"

    elif "P73F1" in upper:
        patient = "P73F1"
        isolate = "P73F1"

    elif "P77F1" in upper:
        patient = "P77F1"
        isolate = "P77F1"

    condition = None

    if "_3H_" in upper:
        condition = "3H"

    elif "_24H_" in upper:
        condition = "24H"

    elif "_EXP_" in upper:
        condition = "EXP"

    elif "_STAT_" in upper:
        condition = "STAT"

    replicate = None

    match = re.search(
        r"_REP(\d+)$",
        upper,
    )

    if match is not None:
        replicate = (
            "rep"
            + match.group(1)
        )

    else:
        match = re.search(
            r"_R(\d+)$",
            upper,
        )

        if match is not None:
            replicate = (
                "R"
                + match.group(1)
            )

    return {
        "patient":
            patient,

        "isolate":
            isolate,

        "condition_label":
            condition,

        "replicate":
            replicate,
    }


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 03b")
    print("PRJEB24688 exact study-design freeze")
    print("=" * 78)

    if not MANIFEST.exists():
        raise FileNotFoundError(
            str(MANIFEST)
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = pd.read_csv(
        MANIFEST
    )

    manifest[
        "sample_title"
    ] = (
        manifest[
            "sample_title"
        ]
        .fillna("")
        .astype(str)
    )

    manifest[
        "library_selection"
    ] = (
        manifest[
            "library_selection"
        ]
        .fillna("")
        .astype(str)
    )

    # ----------------------------------------------------------
    # 1. Exact top-level run classification
    # ----------------------------------------------------------

    title_upper = (
        manifest[
            "sample_title"
        ]
        .str.upper()
    )

    selection_upper = (
        manifest[
            "library_selection"
        ]
        .str.upper()
    )

    manifest[
        "is_WGS"
    ] = (
        selection_upper.eq(
            "RANDOM"
        )
        | title_upper.str.contains(
            "WHOLE GENOME SEQUENCING",
            regex=False,
        )
    )

    manifest[
        "is_RNAseq"
    ] = (
        selection_upper.str.contains(
            "RRNA",
            regex=False,
        )
        & ~manifest[
            "is_WGS"
        ]
    )

    manifest[
        "is_invivo"
    ] = (
        manifest[
            "is_RNAseq"
        ]
        & title_upper.str.contains(
            "_INVIVO_",
            regex=False,
        )
    )

    manifest[
        "is_invitro"
    ] = (
        manifest[
            "is_RNAseq"
        ]
        & ~manifest[
            "is_invivo"
        ]
    )

    print()
    print("Top-level ENA classification")
    print(
        "  total runs:",
        len(
            manifest
        ),
    )
    print(
        "  WGS runs:",
        int(
            manifest[
                "is_WGS"
            ].sum()
        ),
    )
    print(
        "  RNA-seq runs:",
        int(
            manifest[
                "is_RNAseq"
            ].sum()
        ),
    )
    print(
        "  in-vivo RNA runs:",
        int(
            manifest[
                "is_invivo"
            ].sum()
        ),
    )
    print(
        "  in-vitro RNA runs:",
        int(
            manifest[
                "is_invitro"
            ].sum()
        ),
    )

    # ----------------------------------------------------------
    # 2. Exact in-vivo structure
    # ----------------------------------------------------------

    invivo = manifest.loc[
        manifest[
            "is_invivo"
        ]
    ].copy()

    parsed_rows = []

    for index, row in invivo.iterrows():
        parsed = parse_invivo_title(
            row[
                "sample_title"
            ]
        )

        if parsed is None:
            parsed_rows.append(
                {
                    "index":
                        index,

                    "patient":
                        None,

                    "biological_sample":
                        None,

                    "technical_replicate":
                        None,
                }
            )

        else:
            parsed_rows.append(
                {
                    "index":
                        index,

                    **parsed,
                }
            )

    parsed_table = (
        pd.DataFrame(
            parsed_rows
        )
        .set_index(
            "index"
        )
    )

    invivo = invivo.join(
        parsed_table
    )

    invivo[
        "pilot_non_rRNA_depleted"
    ] = (
        invivo[
            "biological_sample"
        ]
        .isin(
            PILOT_SAMPLES
        )
    )

    invivo[
        "primary_high_resolution"
    ] = (
        ~invivo[
            "pilot_non_rRNA_depleted"
        ]
    )

    unparsed_invivo = invivo.loc[
        invivo[
            "biological_sample"
        ].isna()
    ]

    if not unparsed_invivo.empty:
        print()
        print("Unparsed in-vivo titles")
        print(
            unparsed_invivo[
                [
                    "run_accession",
                    "sample_title",
                ]
            ].to_string(
                index=False
            )
        )

    biological_summary = (
        invivo
        .groupby(
            [
                "patient",
                "biological_sample",
                "pilot_non_rRNA_depleted",
                "primary_high_resolution",
            ],
            dropna=False,
        )
        .agg(
            n_ENA_runs=(
                "run_accession",
                "size",
            ),
            runs=(
                "run_accession",
                lambda values:
                    ",".join(
                        values.astype(str)
                    ),
            ),
            total_read_count=(
                "read_count",
                "sum",
            ),
            total_fastq_GB=(
                "total_fastq_GB",
                "sum",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "patient",
                "biological_sample",
            ]
        )
    )

    print()
    print("Frozen biological sputum samples")
    print(
        biological_summary
        .round(
            {
                "total_fastq_GB":
                    3
            }
        )
        .to_string(
            index=False
        )
    )

    total_biological_samples = int(
        biological_summary[
            "biological_sample"
        ].nunique()
    )

    total_patients = int(
        biological_summary[
            "patient"
        ].nunique()
    )

    pilot_biological = (
        biological_summary.loc[
            biological_summary[
                "pilot_non_rRNA_depleted"
            ]
        ]
    )

    primary_biological = (
        biological_summary.loc[
            biological_summary[
                "primary_high_resolution"
            ]
        ]
    )

    primary_invivo_runs = invivo.loc[
        invivo[
            "primary_high_resolution"
        ]
    ].copy()

    primary_patients = sorted(
        primary_biological[
            "patient"
        ].unique()
    )

    technical_groups = (
        primary_biological.loc[
            primary_biological[
                "n_ENA_runs"
            ]
            > 1
        ]
    )

    print()
    print("Primary high-resolution sputum cohort")
    print(
        "  biological samples:",
        len(
            primary_biological
        ),
    )
    print(
        "  ENA runs:",
        len(
            primary_invivo_runs
        ),
    )
    print(
        "  patients:",
        len(
            primary_patients
        ),
        primary_patients,
    )
    print(
        "  technical-replicate biological samples:",
        len(
            technical_groups
        ),
    )

    if not technical_groups.empty:
        print()
        print("Technical-replicate groups")
        print(
            technical_groups[
                [
                    "patient",
                    "biological_sample",
                    "n_ENA_runs",
                    "runs",
                ]
            ].to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 3. Exact in-vitro reference structure
    # ----------------------------------------------------------

    invitro = manifest.loc[
        manifest[
            "is_invitro"
        ]
    ].copy()

    invitro_rows = []

    for index, row in invitro.iterrows():
        parsed = parse_invitro_reference(
            row[
                "sample_title"
            ]
        )

        invitro_rows.append(
            {
                "index":
                    index,

                **parsed,
            }
        )

    invitro_parsed = (
        pd.DataFrame(
            invitro_rows
        )
        .set_index(
            "index"
        )
    )

    invitro = invitro.join(
        invitro_parsed
    )

    invitro[
        "matched_to_primary_patient"
    ] = (
        invitro[
            "patient"
        ]
        .isin(
            primary_patients
        )
    )

    unresolved_invitro = invitro.loc[
        invitro[
            "patient"
        ].isna()
        | invitro[
            "condition_label"
        ].isna()
    ]

    print()
    print("Frozen in-vitro RNA references")
    print(
        invitro[
            [
                "run_accession",
                "sample_title",
                "patient",
                "isolate",
                "condition_label",
                "replicate",
                "matched_to_primary_patient",
            ]
        ]
        .sort_values(
            [
                "patient",
                "isolate",
                "condition_label",
                "sample_title",
            ],
            na_position="last",
        )
        .to_string(
            index=False
        )
    )

    matched_invitro = invitro.loc[
        invitro[
            "matched_to_primary_patient"
        ]
    ].copy()

    lab_pa14 = invitro.loc[
        invitro[
            "patient"
        ]
        == "LAB_PA14"
    ].copy()

    patient_reference_summary = (
        matched_invitro
        .groupby(
            "patient"
        )
        .agg(
            n_runs=(
                "run_accession",
                "size",
            ),
            n_isolates=(
                "isolate",
                "nunique",
            ),
            conditions=(
                "condition_label",
                lambda values:
                    ",".join(
                        sorted(
                            set(
                                values.astype(str)
                            )
                        )
                    ),
            ),
            fastq_GB=(
                "total_fastq_GB",
                "sum",
            ),
        )
        .reset_index()
    )

    print()
    print("Primary-patient in-vitro reference coverage")
    print(
        patient_reference_summary
        .round(
            {
                "fastq_GB":
                    3
            }
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 4. Confirm each primary patient has at least two distinct
    #    laboratory growth-condition labels.
    #
    # We freeze the labels exactly as ENA reports them:
    # 3H / 24H / EXP / STAT.
    #
    # We do not yet force 3H==EXP or 24H==STAT numerically.
    # ----------------------------------------------------------

    patient_condition_counts = (
        matched_invitro
        .groupby(
            "patient"
        )[
            "condition_label"
        ]
        .nunique()
    )

    missing_reference_patients = [
        patient
        for patient in primary_patients
        if patient
        not in set(
            matched_invitro[
                "patient"
            ]
        )
    ]

    insufficient_condition_patients = [
        patient
        for patient in primary_patients
        if patient
        not in patient_condition_counts.index
        or int(
            patient_condition_counts.loc[
                patient
            ]
        )
        < 2
    ]

    # ----------------------------------------------------------
    # 5. Primary FASTQ workload
    # ----------------------------------------------------------

    primary_fastq_GB = float(
        primary_invivo_runs[
            "total_fastq_GB"
        ].sum()
    )

    matched_invitro_fastq_GB = float(
        matched_invitro[
            "total_fastq_GB"
        ].sum()
    )

    print()
    print("Frozen primary processing workload")
    print(
        "  high-resolution in-vivo FASTQ:",
        round(
            primary_fastq_GB,
            3,
        ),
        "GB",
    )
    print(
        "  matched patient in-vitro FASTQ:",
        round(
            matched_invitro_fastq_GB,
            3,
        ),
        "GB",
    )
    print(
        "  PA14 laboratory-reference FASTQ:",
        round(
            float(
                lab_pa14[
                    "total_fastq_GB"
                ].sum()
            ),
            3,
        ),
        "GB",
    )

    # ----------------------------------------------------------
    # 6. Freeze exact primary analysis design
    # ----------------------------------------------------------

    primary_design_rows = []

    for _, row in primary_biological.iterrows():
        patient = row[
            "patient"
        ]

        reference_rows = matched_invitro.loc[
            matched_invitro[
                "patient"
            ]
            == patient
        ]

        primary_design_rows.append(
            {
                "patient":
                    patient,

                "invivo_biological_sample":
                    row[
                        "biological_sample"
                    ],

                "invivo_ENA_runs":
                    row[
                        "runs"
                    ],

                "n_invivo_runs":
                    row[
                        "n_ENA_runs"
                    ],

                "reference_isolates":
                    ",".join(
                        sorted(
                            set(
                                reference_rows[
                                    "isolate"
                                ].astype(str)
                            )
                        )
                    ),

                "reference_condition_labels":
                    ",".join(
                        sorted(
                            set(
                                reference_rows[
                                    "condition_label"
                                ].astype(str)
                            )
                        )
                    ),

                "n_reference_runs":
                    len(
                        reference_rows
                    ),
            }
        )

    primary_design = pd.DataFrame(
        primary_design_rows
    )

    print()
    print("Frozen Phase-4 replication design")
    print(
        primary_design.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Audit gates
    #
    # These expectations come from the actual study structure:
    #
    # 12 biological sputum samples / 5 patients total
    # 3 pilot samples excluded from main analysis
    # 9 high-resolution biological sputum samples
    # 4 patients in the primary high-resolution cohort
    #
    # ENA has 15 in-vivo runs because some biological samples
    # were split into technical sequencing/processing replicates.
    # ----------------------------------------------------------

    checks = {
        "43_total_ENA_runs":
            len(
                manifest
            )
            == 43,

        "5_WGS_runs":
            int(
                manifest[
                    "is_WGS"
                ].sum()
            )
            == 5,

        "38_RNAseq_runs":
            int(
                manifest[
                    "is_RNAseq"
                ].sum()
            )
            == 38,

        "15_invivo_ENA_runs":
            len(
                invivo
            )
            == 15,

        "23_invitro_RNA_runs":
            len(
                invitro
            )
            == 23,

        "all_invivo_titles_parsed":
            unparsed_invivo.empty,

        "12_biological_sputum_samples":
            total_biological_samples
            == 12,

        "5_total_sputum_patients":
            total_patients
            == 5,

        "3_known_pilot_biological_samples":
            len(
                pilot_biological
            )
            == 3
            and set(
                pilot_biological[
                    "biological_sample"
                ]
            )
            == PILOT_SAMPLES,

        "9_primary_high_resolution_samples":
            len(
                primary_biological
            )
            == 9,

        "12_primary_high_resolution_ENA_runs":
            len(
                primary_invivo_runs
            )
            == 12,

        "4_primary_high_resolution_patients":
            len(
                primary_patients
            )
            == 4,

        "3_technical_replicate_groups":
            len(
                technical_groups
            )
            == 3,

        "all_invitro_runs_resolved":
            unresolved_invitro.empty,

        "19_patient_matched_invitro_runs":
            len(
                matched_invitro
            )
            == 19,

        "4_PA14_reference_runs":
            len(
                lab_pa14
            )
            == 4,

        "all_primary_patients_have_lab_reference":
            len(
                missing_reference_patients
            )
            == 0,

        "all_primary_patients_have_two_growth_conditions":
            len(
                insufficient_condition_patients
            )
            == 0,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 03b study-design checks")

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
        "PRJEB24688_STUDY_DESIGN_FROZEN_FOR_EXPRESSION_AUDIT"
        if passed
        else "PRJEB24688_STUDY_DESIGN_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: the three non-rRNA-depleted pilot sputum samples "
        "are excluded from the PRIMARY replication cohort. They may "
        "later be used only as a sensitivity analysis."
    )

    print()
    print(
        "The primary replication target is therefore 9 biological "
        "sputum samples from 4 patients, with technical runs merged "
        "only after their concordance has been checked."
    )

    # ----------------------------------------------------------
    # 8. Save frozen design
    # ----------------------------------------------------------

    manifest.to_csv(
        OUT
        / "PRJEB24688_exact_run_classification.csv",
        index=False,
    )

    invivo.to_csv(
        OUT
        / "PRJEB24688_invivo_run_manifest.csv",
        index=False,
    )

    biological_summary.to_csv(
        OUT
        / "PRJEB24688_biological_sputum_samples.csv",
        index=False,
    )

    primary_invivo_runs.to_csv(
        OUT
        / "PRJEB24688_PRIMARY_highres_invivo_runs.csv",
        index=False,
    )

    primary_biological.to_csv(
        OUT
        / "PRJEB24688_PRIMARY_highres_biological_samples.csv",
        index=False,
    )

    invitro.to_csv(
        OUT
        / "PRJEB24688_invitro_reference_runs.csv",
        index=False,
    )

    matched_invitro.to_csv(
        OUT
        / "PRJEB24688_PRIMARY_patient_matched_invitro_runs.csv",
        index=False,
    )

    lab_pa14.to_csv(
        OUT
        / "PRJEB24688_PA14_reference_runs.csv",
        index=False,
    )

    primary_design.to_csv(
        OUT
        / "PRJEB24688_frozen_primary_replication_design.csv",
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
        / "AUDIT_03B_STATUS.txt"
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
            "Failed Audit 03b checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
