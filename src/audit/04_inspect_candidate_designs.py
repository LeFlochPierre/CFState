from pathlib import Path
import pandas as pd

ROOT = Path("external/EPathDash/GEO_Datasets/Pseudomonas")
DESIGN_DIR = ROOT / "Design_Matrices"

AUDIT_FILE = Path(
    "data/metadata/epathdash_pa_state_audit.csv"
)

OUT_DIR = Path("results/audit/candidate_designs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


audit = pd.read_csv(AUDIT_FILE).fillna("")

# Only studies assigned at least one candidate state
candidates = audit[
    audit["candidate_state"].astype(str).str.strip() != ""
].copy()

print()
print("=" * 100)
print("CF-STATE — candidate study design inspection")
print("=" * 100)
print(f"Candidate studies: {len(candidates)}")
print()


for _, row in candidates.iterrows():

    accession = row["geo_accession"]
    state = row["candidate_state"]

    design_file = DESIGN_DIR / f"{accession}_design.csv"

    print()
    print("=" * 100)
    print(f"{accession}")
    print(f"Candidate state: {state}")
    print("=" * 100)

    if not design_file.exists():
        print("DESIGN FILE MISSING")
        continue

    design = pd.read_csv(
        design_file,
        encoding="utf-8-sig"
    )

    design.columns = [
        str(c).replace("\ufeff", "").strip()
        for c in design.columns
    ]

    print(f"Samples: {len(design)}")
    print(f"Columns: {list(design.columns)}")
    print()

    print(design.to_string(index=False))

    # Save an easy-to-open copy for each study
    out_file = OUT_DIR / f"{accession}_design.csv"

    design.to_csv(
        out_file,
        index=False
    )


print()
print("=" * 100)
print("Saved individual candidate designs to:")
print(OUT_DIR)
print("=" * 100)

