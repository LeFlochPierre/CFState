from pathlib import Path
from io import StringIO
import hashlib
import json

import numpy as np
import pandas as pd


ROOT = Path.cwd()
P4 = ROOT / "results/phase4"

OUT = P4 / "audit_04k2_PRJNA576508_gene_universe_diagnostic"

MODEL = ROOT / (
    "external/aeruprecise/precise411/"
    "paeru_116_iModulon.json"
)

MASK = P4 / (
    "audit_03d_PRJEB24688_published_count_intake/"
    "PRJEB24688_frozen_replication_gene_map.csv"
)

ATL = P4 / (
    "audit_04g_PRJNA576508_Atlanta_full_cohort_processing/"
    "FROZEN_Atlanta_13_raw_PAO1_gene_counts.csv"
)

CFBE = P4 / (
    "audit_04i3_PRJNA576508_CFBE_reference_metadata_repair/"
    "FROZEN_CFBE_four_raw_gene_counts.csv"
)

ATL_EXPRESSION = P4 / (
    "audit_04j2_PRJNA576508_Atlanta_expression_normalisation/"
    "FROZEN_Atlanta_13_log2TPM_expression.csv"
)

CFBE_REFERENCE = P4 / (
    "audit_04i3_PRJNA576508_CFBE_reference_metadata_repair/"
    "FROZEN_CFBE_log2TPM_reference_profiles.csv"
)

SAF = ROOT / (
    "references/phase4/PRJNA576508_cornforth/"
    "Cornforth_PAO1_no_decimal_locus_tags.saf"
)

STABLE = P4 / (
    "audit_03e_softcore_projection_fidelity/"
    "PRJEB24688_empirically_stable_primary_programmes.csv"
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(4 * 1024 * 1024), b""
        ):
            h.update(block)
    return h.hexdigest()


def gene_ids(path, separator=","):
    df = pd.read_csv(
        path,
        sep=separator,
        usecols=["GeneID"],
        dtype=str
    )

    ids = df["GeneID"].str.strip()

    if ids.isna().any() or ids.duplicated().any():
        raise RuntimeError(
            "Missing or duplicate GeneID: " + str(path)
        )

    return set(ids)


def load_model():
    with MODEL.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    value = payload["M"]

    if isinstance(value, str):
        M = pd.read_json(StringIO(value))
    elif isinstance(value, dict):
        M = pd.DataFrame(value)
    else:
        raise RuntimeError("Unexpected model M format")

    M.index = M.index.astype(str).str.strip()
    M.columns = M.columns.astype(str).str.strip()

    if M.shape != (5572, 116):
        raise RuntimeError("Unexpected model dimensions")

    if not M.index.is_unique or not M.columns.is_unique:
        raise RuntimeError("Duplicate model identifiers")

    M = M.apply(pd.to_numeric, errors="raise")

    if not np.isfinite(M.to_numpy()).all():
        raise RuntimeError("Nonfinite model weights")

    return M


def main():
    print("=" * 76)
    print("CF-State Phase 4 — Audit 04k2")
    print("Blinded gene-universe compatibility diagnostic")
    print("=" * 76)

    inputs = {
        "model": MODEL,
        "mask": MASK,
        "Atlanta_counts": ATL,
        "CFBE_counts": CFBE,
        "Atlanta_expression": ATL_EXPRESSION,
        "CFBE_reference": CFBE_REFERENCE,
        "SAF": SAF,
        "stable_programmes": STABLE
    }

    for name, path in inputs.items():
        if not path.is_file():
            raise FileNotFoundError(
                name + ": " + str(path)
            )

    M = load_model()

    mask_table = pd.read_csv(
        MASK, dtype={"pao1_gene": str}
    )

    if "pao1_gene" not in mask_table.columns:
        raise RuntimeError("Missing pao1_gene field")

    mask_list = (
        mask_table["pao1_gene"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(mask_list) != 4925:
        raise RuntimeError("Expected 4925 mask genes")

    if len(set(mask_list)) != 4925:
        raise RuntimeError("Mask contains duplicates")

    mask = set(mask_list)
    model = set(M.index)

    atl = gene_ids(ATL)
    cfbe = gene_ids(CFBE)
    atl_expression = gene_ids(ATL_EXPRESSION)
    cfbe_reference = gene_ids(CFBE_REFERENCE)
    saf = gene_ids(SAF, separator="\t")

    universes = {
        "MODEL": model,
        "MASK": mask,
        "ATLANTA_COUNTS": atl,
        "CFBE_COUNTS": cfbe,
        "ATLANTA_EXPRESSION": atl_expression,
        "CFBE_REFERENCE": cfbe_reference,
        "SAF": saf
    }

    print()
    print("========== UNIVERSE SIZES ==========")

    for name, genes in universes.items():
        print(name, len(genes))

    print()
    print("========== MASK COMPATIBILITY ==========")

    missing_rows = []

    for name, genes in universes.items():
        if name == "MASK":
            continue

        absent = sorted(mask - genes)

        print()
        print(name)
        print("  mask genes present:", len(mask & genes))
        print("  mask genes missing:", len(absent))

        for gene in absent[:25]:
            print("   ", gene)

        for gene in absent:
            missing_rows.append({
                "universe": name,
                "GeneID": gene
            })

    pd.DataFrame(
        missing_rows,
        columns=["universe", "GeneID"]
    ).to_csv(
        OUT / "missing_mask_genes_by_universe.csv",
        index=False
    )

    common = set.intersection(
        model,
        mask,
        atl,
        cfbe,
        atl_expression,
        cfbe_reference,
        saf
    )

    print()
    print("========== COMMON PROJECTION UNIVERSE ==========")
    print("Frozen original mask:", len(mask))
    print("Common available genes:", len(common))
    print("Unavailable mask genes:", len(mask - common))

    if not common:
        raise RuntimeError("No shared genes")

    missing = sorted(mask - common)

    ordered_mask = [
        gene for gene in M.index if gene in mask
    ]

    ordered_common = [
        gene for gene in M.index if gene in common
    ]

    pd.DataFrame({
        "GeneID": ordered_common
    }).to_csv(
        OUT / "DIAGNOSTIC_common_gene_order.csv",
        index=False
    )

    pd.DataFrame({
        "GeneID": missing
    }).to_csv(
        OUT / "DIAGNOSTIC_missing_original_mask_genes.csv",
        index=False
    )

    if not mask.issubset(model):
        raise RuntimeError(
            "Frozen mask has genes absent from model"
        )

    original = M.loc[ordered_mask]
    reduced = M.loc[ordered_common]

    original_matrix = original.to_numpy(dtype=float)
    reduced_matrix = reduced.to_numpy(dtype=float)

    print()
    print("========== MATRIX RANK ==========")

    original_rank = int(
        np.linalg.matrix_rank(original_matrix)
    )

    reduced_rank = int(
        np.linalg.matrix_rank(reduced_matrix)
    )

    print("Original 4925-gene rank:", original_rank)
    print("Available common-gene rank:", reduced_rank)

    print()
    print("========== WEIGHTED PROGRAMME COVERAGE ==========")

    original_abs = original.abs().sum(axis=0)
    retained_abs = reduced.abs().sum(axis=0)

    original_sq = original.pow(2).sum(axis=0)
    retained_sq = reduced.pow(2).sum(axis=0)

    if (original_abs <= 0).any() or (original_sq <= 0).any():
        raise RuntimeError("Zero-weight model programme")

    coverage_abs = retained_abs / original_abs
    coverage_energy = retained_sq / original_sq

    stable = pd.read_csv(STABLE)

    if "imodulon" not in stable.columns:
        raise RuntimeError("Stable programme names missing")

    selected = set(
        stable["imodulon"].astype(str).str.strip()
    )

    if len(selected) != 108:
        raise RuntimeError("Expected 108 stable programmes")

    if not selected.issubset(set(M.columns)):
        raise RuntimeError("Stable programmes absent from M")

    report = pd.DataFrame({
        "imodulon": M.columns,
        "previously_stable_108": [
            item in selected for item in M.columns
        ],
        "original_mask_abs_weight":
            original_abs.to_numpy(),
        "retained_abs_weight_fraction":
            coverage_abs.to_numpy(),
        "retained_squared_weight_fraction":
            coverage_energy.to_numpy()
    })

    report.to_csv(
        OUT / "programme_weight_retention.csv",
        index=False
    )

    print()
    print("All 116 programmes:")
    print(
        report[
            [
                "retained_abs_weight_fraction",
                "retained_squared_weight_fraction"
            ]
        ].describe().round(6).to_string()
    )

    stable_report = report.loc[
        report["previously_stable_108"]
    ]

    print()
    print("Frozen 108 stable programmes:")
    print(
        stable_report[
            [
                "retained_abs_weight_fraction",
                "retained_squared_weight_fraction"
            ]
        ].describe().round(6).to_string()
    )

    print()
    print("Ten programmes with greatest weight loss:")

    lowest = report.nsmallest(
        10, "retained_squared_weight_fraction"
    )

    print(
        lowest[
            [
                "imodulon",
                "previously_stable_108",
                "retained_abs_weight_fraction",
                "retained_squared_weight_fraction"
            ]
        ].round(6).to_string(index=False)
    )

    print()
    print("========== MISSING GENE CONTRIBUTIONS ==========")

    missing_in_model = [
        gene for gene in ordered_mask
        if gene not in common
    ]

    gene_rows = []

    for gene in missing_in_model:
        weights = M.loc[gene].to_numpy(dtype=float)

        gene_rows.append({
            "GeneID": gene,
            "sum_abs_weights": float(
                np.abs(weights).sum()
            ),
            "sum_squared_weights": float(
                np.square(weights).sum()
            ),
            "nonzero_programme_weights": int(
                np.count_nonzero(weights)
            )
        })

    missing_weights = pd.DataFrame(
        gene_rows,
        columns=[
            "GeneID",
            "sum_abs_weights",
            "sum_squared_weights",
            "nonzero_programme_weights"
        ]
    )

    missing_weights.to_csv(
        OUT / "missing_gene_imodulon_weight_contribution.csv",
        index=False
    )

    if len(missing_weights):
        print(
            missing_weights.sort_values(
                "sum_squared_weights",
                ascending=False
            ).head(20).to_string(index=False)
        )
    else:
        print("No missing model-weight genes")

    pd.DataFrame([
        {
            "source": name,
            "sha256": sha256(path)
        }
        for name, path in inputs.items()
    ]).to_csv(
        OUT / "input_sha256_provenance.csv",
        index=False
    )

    summary = {
        "original_mask_genes": len(mask),
        "available_common_genes": len(common),
        "missing_mask_genes": len(missing),
        "original_matrix_rank": original_rank,
        "available_matrix_rank": reduced_rank,
        "stable_programmes": len(selected),
        "min_108_abs_weight_retention": float(
            stable_report[
                "retained_abs_weight_fraction"
            ].min()
        ),
        "min_108_squared_weight_retention": float(
            stable_report[
                "retained_squared_weight_fraction"
            ].min()
        )
    }

    pd.DataFrame([summary]).to_csv(
        OUT / "audit_04k2_diagnostic_summary.csv",
        index=False
    )

    status = "PRJNA576508_04K2_GENE_UNIVERSE_DIAGNOSTIC_COMPLETE"

    (OUT / "AUDIT_04K2_STATUS.txt").write_text(
        status + "\n"
    )

    print()
    print("=" * 76)
    print("STATUS:", status)
    print("=" * 76)
    print("No projection was performed.")
    print("No programme calibration was modified.")
    print("No Audit-03h candidate was accessed.")
    print("This is a diagnostic, not a projection QC PASS.")


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 04k2 already exists. Refusing overwrite."
        )

    OUT.mkdir(parents=True)
    main()
