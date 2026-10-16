from pathlib import Path
import pandas as pd

ROOT = Path("external/EPathDash/GEO_Datasets/Pseudomonas")

folders = {
    "metadata": ROOT / "Metadata",
    "design": ROOT / "Design_Matrices",
    "counts": ROOT / "Count_Tables",
}

print("\nCF-STATE — E.PathDash Pseudomonas inspection")
print("=" * 70)

for name, folder in folders.items():
    files = sorted(folder.glob("*.csv"))

    print(f"\n{name.upper()}")
    print("-" * 70)
    print(f"Number of files: {len(files)}")

    for f in files[:5]:
        print(f"  {f.name}")

print("\n\nMETADATA CONTENT")
print("=" * 70)

metadata_files = sorted(folders["metadata"].glob("*.csv"))

for f in metadata_files:
    print(f"\n### {f.name}")

    try:
        df = pd.read_csv(f)

        print(f"shape: {df.shape}")
        print("columns:")
        print(list(df.columns))

        print("\nfirst rows:")
        print(df.head(5).to_string(index=False))

    except Exception as e:
        print(f"ERROR reading {f}: {e}")

    print("-" * 70)
