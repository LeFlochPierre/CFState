from pathlib import Path
import hashlib
import re
import traceback

import numpy as np
import pandas as pd


ROOT = Path.cwd()

PREVIOUS = ROOT / (
    "results/phase5/"
    "audit_05c2_Cornforth2018_metadata_fields_repair"
)

OUT = ROOT / (
    "results/phase5/"
    "audit_05d_Cornforth2018_sample_mapping"
)

MANIFEST = PREVIOUS / "SOURCE_ENA_SRP135669_read_run.tsv"

PLAN = PREVIOUS / "FROZEN_published_2018_sample_plan.csv"

STATUS = PREVIOUS / "AUDIT_05C2_STATUS.txt"

PUBLISHED_COUNTS = {
    "Human_Sputum_A": 492733,
    "Human_Sputum_B": 2047185,
    "Human_Sputum_C": 1958877,
    "Human_Sputum_D": 414990,
    "Human_Sputum_E": 453389,
    "Human_Sputum_F": 184178,
    "Human_Sputum_G": 10213756,
    "Lubbock_CW_15": 2251929,
    "Lubbock_CW_16": 191274,
    "Lubbock_CW_18": 713108,
    "Lubbock_CW_21": 30836,
    "Lubbock_Burn_2_3": 682982,
    "Lubbock_Burn_2_4": 105278,
    "Danish_Chronic_Wound_1": 5112,
    "Danish_Chronic_Wound_2": 10177
}


def verify(label, condition):
    success = bool(condition)
    print(label, "PASS" if success else "FAIL", flush=True)

    if not success:
        raise RuntimeError("Failed check: " + label)


def clean(value):
    return re.sub(
        r"[^a-z0-9]", "",
        str(value).lower()
    )


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def total_fastq_bytes(value):
    if not value or not str(value).strip():
        return np.nan

    parts = str(value).split(";")

    try:
        numbers = [int(p.strip()) for p in parts]
    except ValueError:
        return np.nan

    if any(n < 0 for n in numbers):
        return np.nan

    return float(sum(numbers))


def main():
    print("=" * 76)
    print("CF-State Phase 5 — Audit 05d")
    print("Cornforth 2018 sample-to-run mapping and feasibility")
    print("=" * 76)

    inputs = {
        "05c2_status": STATUS,
        "published_sample_plan": PLAN,
        "ENA_run_metadata": MANIFEST,
        "script": Path(__file__)
    }

    for name, path in inputs.items():
        verify("input_" + name, path.is_file())

    verify(
        "05c2_metadata_preflight_complete",
        STATUS.read_text().strip()
        == (
            "CFSTATE_PHASE5_05C2_"
            "NEGATIVE_CONTROL_METADATA_PREFLIGHT_COMPLETE"
        )
    )

    published = pd.read_csv(
        PLAN, dtype=str
    ).fillna("")

    ena = pd.read_csv(
        MANIFEST, sep="\t", dtype=str
    ).fillna("")

    print()
    print("========== PUBLICATION AND ENA INTEGRITY ==========")

    verify(
        "published_sample_fields",
        {
            "published_sample",
            "infection_context",
            "published_analysis_role"
        }.issubset(published.columns)
    )

    verify(
        "published_15_unique",
        len(published) == 15
        and published["published_sample"].is_unique
        and set(published["published_sample"])
        == set(PUBLISHED_COUNTS)
    )

    verify(
        "ENA_required_identifiers",
        {
            "run_accession",
            "sample_accession"
        }.issubset(ena.columns)
    )

    verify(
        "ENA_nonempty_unique_runs",
        len(ena) > 0
        and ena["run_accession"].ne("").all()
        and ena["run_accession"].is_unique
    )

    verify(
        "ENA_all_runs_have_sample_accession",
        ena["sample_accession"].ne("").all()
    )

    print("Published specimens:", len(published))
    print("ENA run records:", len(ena))
    print("Unique ENA samples:", ena["sample_accession"].nunique())

    print()
    print("========== AVAILABLE ANNOTATION FIELDS ==========")

    annotation_fields = [
        name for name in [
            "sample_alias",
            "sample_title",
            "experiment_title",
            "library_name",
            "sample_description",
            "sample_name"
        ]
        if name in ena.columns
    ]

    print(annotation_fields)

    verify(
        "has_searchable_sample_metadata",
        len(annotation_fields) > 0
    )

    print()
    print("========== PROVISIONAL NAME MATCHING ==========")

    evidence_rows = []

    for record in ena.to_dict("records"):
        for published_row in published.to_dict("records"):

            name = published_row["published_sample"]
            target = clean(name)

            exact_fields = []
            containing_fields = []

            for field in annotation_fields:
                observed = clean(record.get(field, ""))

                if not observed:
                    continue

                if observed == target:
                    exact_fields.append(field)

                elif target in observed:
                    containing_fields.append(field)

            if not exact_fields and not containing_fields:
                continue

            match_type = (
                "EXACT_NORMALISED_NAME"
                if exact_fields
                else "NAME_CONTAINED_IN_METADATA"
            )

            evidence_rows.append({
                "published_sample": name,
                "infection_context":
                    published_row["infection_context"],
                "run_accession": record["run_accession"],
                "sample_accession":
                    record["sample_accession"],
                "match_type": match_type,
                "evidence_fields": ";".join(
                    exact_fields + containing_fields
                ),
                "mapping_verification":
                    "PROVISIONAL_NOT_BIOLOGICALLY_VERIFIED"
            })

    evidence = pd.DataFrame(
        evidence_rows,
        columns=[
            "published_sample",
            "infection_context",
            "run_accession",
            "sample_accession",
            "match_type",
            "evidence_fields",
            "mapping_verification"
        ]
    )

    evidence.to_csv(
        OUT / "PROVISIONAL_sample_run_mapping_evidence.csv",
        index=False
    )

    ambiguous_runs = set()

    if not evidence.empty:
        n_names = evidence.groupby(
            "run_accession"
        )["published_sample"].nunique()

        ambiguous_runs = set(
            n_names[n_names > 1].index
        )

    print("Provisional mapping records:", len(evidence))
    print("Runs mapping to multiple published names:", len(ambiguous_runs))

    print()
    print("========== PUBLISHED SPECIMEN COVERAGE ==========")

    summary_rows = []

    for row in published.to_dict("records"):
        name = row["published_sample"]

        matched = evidence.loc[
            evidence["published_sample"] == name
        ]

        runs = sorted(
            set(matched["run_accession"])
        )

        samples = sorted(
            set(matched["sample_accession"])
        )

        if len(runs) == 0:
            status = "UNRESOLVED_NO_NAME_MATCH"

        elif any(r in ambiguous_runs for r in runs):
            status = "AMBIGUOUS_RUN_MATCH"

        elif len(samples) != 1:
            status = "AMBIGUOUS_MULTIPLE_SAMPLE_ACCESSIONS"

        else:
            status = "PROVISIONAL_UNIQUE_SAMPLE_ACCESSION"

        matched_runs = ena.loc[
            ena["run_accession"].isin(runs)
        ]

        bytes_total = np.nan

        if "fastq_bytes" in matched_runs.columns:
            sizes = matched_runs[
                "fastq_bytes"
            ].map(total_fastq_bytes)

            if len(sizes) and sizes.notna().all():
                bytes_total = float(sizes.sum())

        donor_note = row.get(
            "donor_dependency_note", ""
        )

        summary_rows.append({
            "published_sample": name,
            "infection_context":
                row["infection_context"],
            "analysis_role":
                row["published_analysis_role"],
            "published_PAO1_ortholog_mapped_reads":
                PUBLISHED_COUNTS[name],
            "n_matching_ENA_runs": len(runs),
            "n_distinct_ENA_sample_accessions":
                len(samples),
            "candidate_runs": ";".join(runs),
            "candidate_sample_accessions":
                ";".join(samples),
            "estimated_FASTQ_GB":
                bytes_total / 1e9,
            "mapping_status": status,
            "published_donor_note": donor_note,
            "donor_independence_verified": False
        })

    summary = pd.DataFrame(summary_rows)

    verify(
        "all_15_published_specimens_accounted_for",
        len(summary) == 15
        and summary["published_sample"].is_unique
    )

    print(
        summary[
            [
                "published_sample",
                "infection_context",
                "n_matching_ENA_runs",
                "n_distinct_ENA_sample_accessions",
                "mapping_status",
                "estimated_FASTQ_GB"
            ]
        ].round(3).to_string(index=False)
    )

    summary.to_csv(
        OUT / "FROZEN_15_published_specimen_mapping_audit.csv",
        index=False
    )

    print()
    print("========== UNRESOLVED / AMBIGUOUS ==========")

    needs_review = summary.loc[
        summary["mapping_status"]
        != "PROVISIONAL_UNIQUE_SAMPLE_ACCESSION"
    ]

    if needs_review.empty:
        print("All 15 have provisional unique sample accessions.")
    else:
        print(
            needs_review[
                ["published_sample", "mapping_status"]
            ].to_string(index=False)
        )

    print()
    print("========== UNMATCHED ENA RUNS ==========")

    assigned = set(evidence["run_accession"])

    unmatched = ena.loc[
        ~ena["run_accession"].isin(assigned)
    ]

    preview = [
        field for field in [
            "run_accession",
            "sample_accession",
            "sample_alias",
            "sample_title",
            "experiment_title",
            "library_name",
            "read_count",
            "fastq_bytes"
        ]
        if field in ena.columns
    ]

    print("Unmatched ENA runs:", len(unmatched))

    if len(unmatched):
        print(
            unmatched[preview]
            .head(35)
            .to_string(
                index=False,
                max_colwidth=55
            )
        )

    unmatched.to_csv(
        OUT / "UNMATCHED_ENA_runs_for_review.csv",
        index=False
    )

    print()
    print("========== DONOR AND BIOLOGICAL DESIGN ==========")

    print("Known published dependency: two burn specimens, same patient.")
    print("Other donor relationships require source verification.")
    print("ENA accession uniqueness is NOT proof of donor independence.")
    print("No expression outcomes or frozen CF candidate accessed.")

    print()
    print("========== PROVENANCE ==========")

    pd.DataFrame([
        {
            "source": name,
            "path": str(path),
            "sha256": sha256(path)
        }
        for name, path in inputs.items()
    ]).to_csv(
        OUT / "input_sha256_provenance.csv",
        index=False
    )

    (
        OUT / "AUDIT_05D_DESIGN_NOTES.txt"
    ).write_text(
        "Cornforth et al. 2018 PNAS, "
        "DOI 10.1073/pnas.1717525115.\n"
        "Source of published specimen names and mapped read counts: "
        "publication Table 1.\n"
        "Sequence archive: SRP135669.\n"
        "Mapping is based on ENA specimen-name concordance only.\n"
        "Even unique accessions are not biologically verified here.\n"
        "Seven published CF sputum and six US non-CF wound/burn "
        "specimens belong to the main human comparison.\n"
        "Two Danish wound specimens were handled separately "
        "due to very low bacterial coverage.\n"
        "Two US burn samples derive from one patient.\n"
        "Published bacterial mapped counts are NOT ENA read counts.\n"
        "No retroactive QC thresholds or cohort exclusions.\n"
        "No raw sequencing files downloaded.\n"
        "No conclusions about CF specificity.\n"
    )

    status = (
        "CFSTATE_PHASE5_05D_"
        "SAMPLE_MAPPING_DIAGNOSTIC_FROZEN"
    )

    (
        OUT / "AUDIT_05D_STATUS.txt"
    ).write_text(status + "\n")

    print()
    print("=" * 76)
    print("STATUS:", status)
    print("=" * 76)
    print("Sample mapping remains provisional pending verification.")


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 05d output exists; refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()

    except Exception:
        (
            OUT / "AUDIT_05D_STATUS.txt"
        ).write_text(
            "CFSTATE_PHASE5_05D_REVIEW_REQUIRED\n"
        )

        (
            OUT / "AUDIT_05D_FAILURE.txt"
        ).write_text(traceback.format_exc())

        raise
