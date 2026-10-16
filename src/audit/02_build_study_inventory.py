from pathlib import Path
import pandas as pd

ROOT = Path("external/EPathDash/GEO_Datasets/Pseudomonas")

META_DIR = ROOT / "Metadata"
DESIGN_DIR = ROOT / "Design_Matrices"
COUNT_DIR = ROOT / "Count_Tables"

OUT = Path("data/metadata/epathdash_pa_study_inventory_raw.csv")
OUT.parent.mkdir(parents=True, exist_ok=True)


def clean_value(x):
    if pd.isna(x):
        return ""
    x = str(x).strip()
    if x.lower() in {"na", "nan", "none"}:
        return ""
    return x


records = []

metadata_files = sorted(META_DIR.glob("*_metadata.csv"))

for meta_file in metadata_files:

    accession = meta_file.name.replace("_metadata.csv", "")

    design_file = DESIGN_DIR / f"{accession}_design.csv"
    count_file = COUNT_DIR / f"{accession}_counts.csv"

    # ---------- metadata ----------
    meta = pd.read_csv(meta_file, encoding="utf-8-sig")

    if len(meta) != 1:
        print(
            f"WARNING: {accession} metadata contains "
            f"{len(meta)} rows instead of 1"
        )

    row = meta.iloc[0]

    # ---------- design matrix ----------
    n_samples = None
    design_columns = ""
    design_levels = ""

    if design_file.exists():

        design = pd.read_csv(design_file, encoding="utf-8-sig")

        # Remove possible BOM / spaces
        design.columns = [
            str(c).replace("\ufeff", "").strip()
            for c in design.columns
        ]

        n_samples = len(design)

        design_columns = " | ".join(design.columns)

        level_strings = []

        for col in design.columns:

            values = (
                design[col]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            level_strings.append(
                f"{col}: {', '.join(values)}"
            )

        design_levels = " || ".join(level_strings)

    # ---------- metadata fields ----------
    title = clean_value(row.get("Title", ""))
    description = clean_value(row.get("Description", ""))
    strain = clean_value(row.get("Strain", ""))
    medium = clean_value(row.get("Medium", ""))
    treatment = clean_value(row.get("Treatment", ""))
    genotype = clean_value(row.get("Genotype", ""))
    date = clean_value(row.get("Date", ""))
    link = clean_value(row.get("Link", ""))

    # Core text = experimental manipulation information.
    # We deliberately exclude the Description here because descriptions
    # often mention biological consequences that were NOT the perturbation.
    core_text = " | ".join(
        x for x in [
            title,
            treatment,
            genotype,
            medium,
            design_levels
        ]
        if x
    )

    record = {
        "source": "EPathDash",
        "geo_accession": accession,
        "date": date,
        "title": title,
        "strain": strain,
        "medium": medium,
        "treatment": treatment,
        "genotype": genotype,
        "n_samples": n_samples,
        "design_columns": design_columns,
        "design_levels": design_levels,
        "count_table_available": count_file.exists(),
        "design_matrix_available": design_file.exists(),
        "core_experimental_text": core_text,
        "description": description,
        "link": link,
    }

    records.append(record)


df = pd.DataFrame(records)

df = df.sort_values("geo_accession").reset_index(drop=True)

df.to_csv(OUT, index=False)

print()
print("=" * 80)
print("CF-STATE P. aeruginosa EPathDash inventory")
print("=" * 80)
print(f"Studies: {len(df)}")
print(f"Total samples across design matrices: {df['n_samples'].sum()}")
print(f"Studies with count tables: {df['count_table_available'].sum()}")
print(f"Studies with design matrices: {df['design_matrix_available'].sum()}")
print()
print(f"Saved:")
print(OUT)
print()

print(
    df[
        [
            "geo_accession",
            "title",
            "strain",
            "medium",
            "treatment",
            "genotype",
            "n_samples",
            "design_levels",
        ]
    ].to_string(index=False)
)
