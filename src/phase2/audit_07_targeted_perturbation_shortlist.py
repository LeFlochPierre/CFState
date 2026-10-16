from pathlib import Path

import pandas as pd


ROOT = Path.cwd()

PRECISE = (
    ROOT
    / "external/aeruprecise/precise411/sample_table.csv"
)

OUT = (
    ROOT
    / "results/phase2/audit_07_targeted_perturbation_shortlist"
)


CANDIDATES = [
    {
        "accession": "GSE224821",
        "anchor_family": "iron_limitation",
        "strain": "PAO1",
        "platform": "RNA-seq",
        "primary_contrast": "fresh CAA low iron vs fresh CAA high iron",
        "replicates_per_condition": 3,
        "directness": "DIRECT_ENVIRONMENTAL",
        "temporal_external": True,
        "aliases": [],
        "priority_reason": (
            "PvdS is a recurrent same-direction residual in both biofilm "
            "experiments; clean iron-rich/iron-limited PAO1 subset exists."
        ),
    },
    {
        "accession": "GSE246747",
        "anchor_family": "quorum_sensing",
        "strain": "PAO1",
        "platform": "RNA-seq",
        "primary_contrast": "WT PAO1 vs PAO1 delta-lasR",
        "replicates_per_condition": 3,
        "directness": "DIRECT_GENETIC",
        "temporal_external": True,
        "aliases": ["PRJNA1034550"],
        "priority_reason": (
            "QscR-2 and Quorum sensing are large independent residuals; "
            "simple 3-vs-3 LasR perturbation."
        ),
    },
    {
        "accession": "GSE261453",
        "anchor_family": "quorum_sensing",
        "strain": "PAO1",
        "platform": "RNA-seq",
        "primary_contrast": (
            "AHL-active vs AiiA-degraded conditions with pqsR/pqsE/pqsA mutants"
        ),
        "replicates_per_condition": 2,
        "directness": "FACTORIAL_GENETIC_ENVIRONMENTAL",
        "temporal_external": True,
        "aliases": ["PRJNA1086972"],
        "priority_reason": (
            "Rich secondary QS experiment allowing AHL/PQS decomposition."
        ),
    },
    {
        "accession": "GSE128430",
        "anchor_family": "secretion_virulence",
        "strain": "PAO1",
        "platform": "RNA-seq",
        "primary_contrast": (
            "ExsA and other virulence regulators versus PAO1 controls"
        ),
        "replicates_per_condition": 2,
        "directness": "DIRECT_GENETIC",
        "temporal_external": False,
        "aliases": ["PRJNA527727", "SRP188684"],
        "priority_reason": (
            "ExsB is the single largest unexplained GSE203348 programme; "
            "dataset includes ExsA, LasR, QscR and other regulators."
        ),
    },
    {
        "accession": "GSE129571",
        "anchor_family": "oxygen_denitrification",
        "strain": "PAO1",
        "platform": "RNA-seq",
        "primary_contrast": "WT PAO1 anaerobic vs WT PAO1 aerobic",
        "replicates_per_condition": 2,
        "directness": "DIRECT_ENVIRONMENTAL",
        "temporal_external": False,
        "aliases": ["PRJNA531832", "SRP191772"],
        "priority_reason": (
            "Direct aerobic/anaerobic WT comparison; useful for resolving "
            "context-dependent Anr/NarL directions."
        ),
    },
    {
        "accession": "GSE68534",
        "anchor_family": "oxygen_denitrification",
        "strain": "PAO1",
        "platform": "RNA-seq",
        "primary_contrast": "WT PAO1 vs delta-anr at 1% oxygen colony biofilm",
        "replicates_per_condition": 2,
        "directness": "REGULATOR_VALIDATION",
        "temporal_external": False,
        "aliases": [],
        "priority_reason": (
            "Good Anr-regulon validation but does not itself isolate "
            "oxygen concentration because all samples are low-oxygen biofilms."
        ),
    },
    {
        "accession": "GSE55197",
        "anchor_family": "iron_limitation",
        "strain": "PA14",
        "platform": "RNA-seq",
        "primary_contrast": "iron-starved PA14 conditions",
        "replicates_per_condition": 1,
        "directness": "CROSS_STRAIN_ENVIRONMENTAL",
        "temporal_external": False,
        "aliases": [],
        "priority_reason": (
            "Useful later for PAO1-to-PA14 portability, not a primary "
            "PAO1 anchoring dataset."
        ),
    },
]


def find_local_files(accession):
    roots = [
        ROOT / "external/EPathDash/GEO_Datasets/Pseudomonas",
        ROOT / "data/raw",
        ROOT / "data/processed",
    ]

    hits = []

    for base in roots:
        if not base.exists():
            continue

        for path in base.rglob(f"*{accession}*"):
            hits.append(
                str(path.relative_to(ROOT))
            )

    return sorted(set(hits))


def precise_alias_matches(table, aliases):
    matches = []

    for alias in aliases:
        mask = pd.Series(
            False,
            index=table.index,
        )

        for col in table.columns:
            values = (
                table[col]
                .fillna("")
                .astype(str)
            )

            mask = mask | values.str.contains(
                alias,
                regex=False,
            )

        n = int(mask.sum())

        if n > 0:
            matches.append(
                f"{alias}:{n}"
            )

    return matches


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 07")
    print("Targeted controlled-perturbation shortlist")
    print("=" * 78)

    if not PRECISE.exists():
        raise FileNotFoundError(
            str(PRECISE)
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    precise = pd.read_csv(
        PRECISE,
        low_memory=False,
    )

    rows = []

    for candidate in CANDIDATES:
        accession = candidate["accession"]

        aliases = [
            accession,
            *candidate["aliases"],
        ]

        overlap = precise_alias_matches(
            precise,
            aliases,
        )

        local_files = find_local_files(
            accession
        )

        count_files = [
            x
            for x in local_files
            if "count" in x.lower()
        ]

        design_files = [
            x
            for x in local_files
            if "design" in x.lower()
        ]

        metadata_files = [
            x
            for x in local_files
            if "metadata" in x.lower()
        ]

        training_overlap = (
            len(overlap) > 0
        )

        if training_overlap:
            status = "TRAINING_OVERLAP_INTERNAL_ONLY"

        elif candidate["strain"] != "PAO1":
            status = "CROSS_STRAIN_LATER"

        elif (
            count_files
            and candidate["directness"]
            == "DIRECT_ENVIRONMENTAL"
        ):
            status = "READY_NOW"

        elif candidate["temporal_external"]:
            status = "ACQUIRE_NEXT"

        else:
            status = "CANDIDATE_NEEDS_FULL_PROVENANCE"

        row = {
            **candidate,

            "aliases":
                ";".join(
                    candidate["aliases"]
                ),

            "precise_overlap":
                training_overlap,

            "precise_overlap_details":
                ";".join(
                    overlap
                ),

            "n_local_files":
                len(local_files),

            "n_local_count_files":
                len(count_files),

            "n_local_design_files":
                len(design_files),

            "n_local_metadata_files":
                len(metadata_files),

            "local_files":
                ";".join(
                    local_files
                ),

            "phase2_status":
                status,
        }

        rows.append(
            row
        )

    df = pd.DataFrame(
        rows
    )

    # ------------------------------------------------------------
    # Transparent selection hierarchy
    # ------------------------------------------------------------

    status_order = {
        "READY_NOW": 0,
        "ACQUIRE_NEXT": 1,
        "CANDIDATE_NEEDS_FULL_PROVENANCE": 2,
        "CROSS_STRAIN_LATER": 3,
        "TRAINING_OVERLAP_INTERNAL_ONLY": 4,
    }

    family_order = {
        # Recurrent same-direction residual and clean local dataset.
        "iron_limitation": 0,

        # QscR-2 / QS recurrent and same-direction.
        "quorum_sensing": 1,

        # ExsB is largest residual, but candidate data need more curation.
        "secretion_virulence": 2,

        # Important but Anr-2 changed sign between biofilm experiments.
        "oxygen_denitrification": 3,
    }

    df["status_rank"] = (
        df["phase2_status"]
        .map(status_order)
        .fillna(99)
    )

    df["family_rank"] = (
        df["anchor_family"]
        .map(family_order)
        .fillna(99)
    )

    df = df.sort_values(
        [
            "status_rank",
            "family_rank",
            "replicates_per_condition",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    df["selection_rank"] = (
        range(
            1,
            len(df) + 1,
        )
    )

    # ------------------------------------------------------------
    # Select next audit dataset
    # ------------------------------------------------------------

    ready = df.loc[
        df["phase2_status"]
        == "READY_NOW"
    ]

    if ready.empty:
        selected = None
    else:
        selected = ready.iloc[0]

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    df.to_csv(
        OUT
        / "targeted_perturbation_registry.csv",
        index=False,
    )

    print()
    print("Candidate registry")
    print(
        df[
            [
                "selection_rank",
                "accession",
                "anchor_family",
                "strain",
                "replicates_per_condition",
                "precise_overlap",
                "n_local_count_files",
                "phase2_status",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Local files for READY_NOW candidates")

    if ready.empty:
        print("  none")

    else:
        for _, row in ready.iterrows():
            print()
            print(
                row["accession"],
                "-"
                ,
                row["anchor_family"],
            )

            for path in str(
                row["local_files"]
            ).split(";"):
                if path:
                    print(
                        " ",
                        path,
                    )

    print()

    if selected is None:
        print(
            "No dataset is READY_NOW."
        )

    else:
        print(
            "Selected next controlled anchor:"
        )
        print(
            " ",
            selected["accession"],
            "-"
            ,
            selected["anchor_family"],
        )
        print(
            "  contrast:",
            selected["primary_contrast"],
        )

    # ------------------------------------------------------------
    # Audit checks
    # ------------------------------------------------------------

    checks = {
        "candidate_registry_nonempty":
            len(df) > 0,

        "no_ready_candidate_has_precise_overlap":
            bool(
                (
                    ready[
                        "precise_overlap"
                    ]
                    == False
                ).all()
            ),

        "at_least_one_primary_PAO1_candidate":
            bool(
                (
                    df["strain"]
                    == "PAO1"
                ).any()
            ),

        "at_least_one_ready_dataset":
            selected is not None,
    }

    passed = all(
        checks.values()
    )

    pd.DataFrame(
        [
            {
                "check": k,
                "pass": v,
            }
            for k, v
            in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    lines = [
        "# CF-State Phase 2 — Audit 07",
        "",
        "## Objective",
        "",
        "Freeze a targeted controlled-perturbation shortlist based on "
        "the unexplained aeruPRECISE programme structure from Audit 06.",
        "",
        "## Selection rules",
        "",
        "- Prefer direct PAO1 perturbations.",
        "- Prefer RNA-seq with biological replication.",
        "- Exclude aeruPRECISE training-overlap studies from external validation.",
        "- Prefer datasets already present locally before acquiring new data.",
        "- Use PA14 datasets later for cross-strain portability rather than "
        "primary PAO1 anchoring.",
        "",
    ]

    if selected is not None:
        lines.extend(
            [
                "## Selected next anchor",
                "",
                "- Accession: "
                + str(
                    selected["accession"]
                ),
                "- Family: "
                + str(
                    selected["anchor_family"]
                ),
                "- Contrast: "
                + str(
                    selected["primary_contrast"]
                ),
                "",
                "This selection is a workflow decision, not a biological result.",
                "",
            ]
        )

    lines.extend(
        [
            "## Decision",
            "",
            "PASS: "
            + str(
                passed
            ),
        ]
    )

    (
        OUT
        / "AUDIT_07_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 07:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_07_SUMMARY.md",
    )

    if not passed:
        failed = [
            k
            for k, v
            in checks.items()
            if not v
        ]

        raise RuntimeError(
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
