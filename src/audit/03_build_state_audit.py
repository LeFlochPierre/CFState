from pathlib import Path
import pandas as pd
import re

INFILE = Path(
    "data/metadata/epathdash_pa_study_inventory_raw.csv"
)

OUTFILE = Path(
    "data/metadata/epathdash_pa_state_audit.csv"
)

df = pd.read_csv(INFILE).fillna("")


def classify_candidate_state(row):
    """
    Conservative candidate-state assignment.

    This is ONLY for triage.
    Final inclusion/exclusion must be manually reviewed.
    """

    text = " | ".join(
        [
            str(row.get("title", "")),
            str(row.get("treatment", "")),
            str(row.get("genotype", "")),
            str(row.get("design_levels", "")),
        ]
    ).lower()

    states = []

    # Biofilm
    if re.search(r"\bbiofilm\b", text) and re.search(
        r"\bplanktonic\b", text
    ):
        states.append("biofilm")

    # Oxygen limitation
    if any(
        term in text
        for term in [
            "microaerobic",
            "anaerobic",
            "oxygen limitation",
            "oxygen-limited",
            "hypoxia",
        ]
    ):
        states.append("oxygen_limitation")

    # Iron limitation
    if any(
        term in text
        for term in [
            "iron-low",
            "iron low",
            "iron limitation",
            "iron-limited",
            "iron depleted",
            "iron-depleted",
        ]
    ):
        states.append("iron_limitation")

    # Antibiotic / antimicrobial treatment
    antibiotic_terms = [
        "ciprofloxacin",
        "levofloxacin",
        "lomefloxacin",
        "novobiocin",
        "coumermycin",
        "meropenem",
        "colistin",
        "tobramycin",
        "eravacycline",
        "azithromycin",
    ]

    if any(term in text for term in antibiotic_terms):
        states.append("antibiotic_stress")

    # Quorum sensing
    if any(
        term in text
        for term in [
            "quorum",
            "rhlr",
            "pqse",
        ]
    ):
        states.append("quorum_sensing_related")

    # Interspecies / host-like environment
    if any(
        term in text
        for term in [
            "candida",
            "burkholderia",
            "co-culture",
            "coculture",
            "mucin",
            "glycan",
        ]
    ):
        states.append("host_or_interspecies")

    return " | ".join(states)


df["candidate_state"] = df.apply(
    classify_candidate_state,
    axis=1,
)

# Columns for manual curation
df["usable_for_state_learning"] = ""
df["direct_perturbation"] = ""
df["control_group"] = ""
df["perturbed_group"] = ""
df["independent_study_id"] = df["geo_accession"]
df["exclusion_reason"] = ""
df["curation_notes"] = ""

columns = [
    "source",
    "geo_accession",
    "title",
    "strain",
    "medium",
    "treatment",
    "genotype",
    "n_samples",
    "design_levels",
    "candidate_state",
    "usable_for_state_learning",
    "direct_perturbation",
    "control_group",
    "perturbed_group",
    "independent_study_id",
    "exclusion_reason",
    "curation_notes",
    "description",
    "link",
]

# Keep only columns actually present
columns = [c for c in columns if c in df.columns]

df = df[columns]

OUTFILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

df.to_csv(
    OUTFILE,
    index=False,
)

print()
print("=" * 80)
print("CF-STATE candidate-state audit")
print("=" * 80)
print(f"Studies: {len(df)}")
print()

print("Candidate-state counts:")
print(
    df["candidate_state"]
    .replace("", "UNCLASSIFIED")
    .value_counts()
    .to_string()
)

print()
print(f"Saved: {OUTFILE}")
print()

print(
    df[
        [
            "geo_accession",
            "n_samples",
            "candidate_state",
            "design_levels",
        ]
    ].to_string(index=False)
)

