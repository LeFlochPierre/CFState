from pathlib import Path
from io import StringIO
import json
import platform

import numpy as np
import pandas as pd


ROOT = Path.cwd()

MODEL = (
    ROOT
    / "external"
    / "aeruprecise"
    / "precise411"
    / "paeru_116_iModulon.json"
)

OUT = (
    ROOT
    / "results"
    / "phase2"
    / "audit_02_aeruprecise_object_integrity"
)


def read_dataframe(value, name):
    if value is None:
        return None

    if isinstance(value, str):
        try:
            return pd.read_json(StringIO(value))
        except Exception as exc:
            raise RuntimeError(
                f"Could not deserialize {name}: {exc}"
            )

    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except Exception as exc:
            raise RuntimeError(
                f"Could not convert {name} dictionary: {exc}"
            )

    raise RuntimeError(
        f"Unexpected type for {name}: {type(value).__name__}"
    )


def correlation(x, y):
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]

    if len(x) < 3:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 02")
    print("Frozen aeruPRECISE-411 object integrity")
    print("=" * 78)

    if not MODEL.exists():
        raise FileNotFoundError(
            f"Frozen model not found:\n{MODEL}"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Python: {platform.python_version()}")
    print(f"pandas: {pd.__version__}")
    print(f"Model:  {MODEL}")
    print()

    print("Reading published JSON directly ...")

    with MODEL.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    print(f"Top-level objects: {len(raw)}")

    inventory = []

    for key, value in raw.items():
        try:
            length = len(value)
        except Exception:
            length = np.nan

        inventory.append(
            {
                "key": key,
                "type": type(value).__name__,
                "length": length,
            }
        )

    pd.DataFrame(inventory).to_csv(
        OUT / "json_inventory.csv",
        index=False,
    )

    for required in ["M", "A"]:
        if required not in raw:
            raise RuntimeError(
                f"Published JSON is missing required object: {required}"
            )

    M = read_dataframe(raw["M"], "M")
    A = read_dataframe(raw["A"], "A")

    X = read_dataframe(raw.get("X"), "X")
    log_tpm = read_dataframe(
        raw.get("log_tpm"),
        "log_tpm",
    )

    gene_table = read_dataframe(
        raw.get("gene_table"),
        "gene_table",
    )

    sample_table = read_dataframe(
        raw.get("sample_table"),
        "sample_table",
    )

    imodulon_table = read_dataframe(
        raw.get("imodulon_table"),
        "imodulon_table",
    )

    print()
    print("Loaded matrices:")
    print(f"  M       {M.shape}")
    print(f"  A       {A.shape}")
    print(f"  X       {None if X is None else X.shape}")
    print(
        f"  log_tpm {None if log_tpm is None else log_tpm.shape}"
    )

    if M.shape[1] != 116:
        raise RuntimeError(
            f"Expected 116 iModulons, found {M.shape[1]}"
        )

    m_names = [str(x) for x in M.columns]
    a_names = [str(x) for x in A.index]

    same_components = m_names == a_names

    print()
    print("Core integrity:")
    print(f"  116 iModulons:              {M.shape[1] == 116}")
    print(f"  M columns == A rows:        {same_components}")
    print(f"  M gene IDs unique:          {M.index.is_unique}")
    print(f"  A sample IDs unique:        {A.columns.is_unique}")

    if not same_components:
        m_only = sorted(set(m_names) - set(a_names))
        a_only = sorted(set(a_names) - set(m_names))

        print(f"  Only in M: {m_only[:10]}")
        print(f"  Only in A: {a_only[:10]}")

        raise RuntimeError(
            "M and A component identities do not match."
        )

    checks = [
        {
            "check": "116_imodulons",
            "pass": M.shape[1] == 116,
            "value": M.shape[1],
        },
        {
            "check": "M_columns_equal_A_rows",
            "pass": same_components,
            "value": len(m_names),
        },
        {
            "check": "M_gene_ids_unique",
            "pass": M.index.is_unique,
            "value": len(M.index),
        },
        {
            "check": "A_sample_ids_unique",
            "pass": A.columns.is_unique,
            "value": len(A.columns),
        },
    ]

    if X is not None:
        x_genes_match = (
            [str(x) for x in X.index]
            == [str(x) for x in M.index]
        )

        x_samples_match = (
            [str(x) for x in X.columns]
            == [str(x) for x in A.columns]
        )

        print(f"  X genes == M genes:         {x_genes_match}")
        print(f"  X samples == A samples:     {x_samples_match}")

        checks.extend(
            [
                {
                    "check": "X_genes_equal_M_genes",
                    "pass": x_genes_match,
                    "value": X.shape[0],
                },
                {
                    "check": "X_samples_equal_A_samples",
                    "pass": x_samples_match,
                    "value": X.shape[1],
                },
            ]
        )

    pd.DataFrame(checks).to_csv(
        OUT / "integrity_checks.csv",
        index=False,
    )

    matrix_manifest = pd.DataFrame(
        [
            {
                "matrix": "M",
                "rows": M.shape[0],
                "cols": M.shape[1],
            },
            {
                "matrix": "A",
                "rows": A.shape[0],
                "cols": A.shape[1],
            },
            {
                "matrix": "X",
                "rows": None if X is None else X.shape[0],
                "cols": None if X is None else X.shape[1],
            },
            {
                "matrix": "log_tpm",
                "rows": None if log_tpm is None else log_tpm.shape[0],
                "cols": None if log_tpm is None else log_tpm.shape[1],
            },
        ]
    )

    matrix_manifest.to_csv(
        OUT / "matrix_manifest.csv",
        index=False,
    )

    # --------------------------------------------------
    # Export metadata
    # --------------------------------------------------

    if gene_table is not None:
        gene_table.to_csv(
            OUT / "gene_table.csv"
        )

    if sample_table is not None:
        sample_table.to_csv(
            OUT / "sample_table.csv"
        )

    if imodulon_table is not None:
        imodulon_table.to_csv(
            OUT / "imodulon_table.csv"
        )

    # --------------------------------------------------
    # Programme names
    # --------------------------------------------------

    programmes = pd.DataFrame(
        {
            "position": range(1, len(M.columns) + 1),
            "imodulon": M.columns.astype(str),
        }
    )

    programmes.to_csv(
        OUT / "programme_names.csv",
        index=False,
    )

    # --------------------------------------------------
    # Top-weight genes
    # --------------------------------------------------

    top_rows = []

    for component in M.columns:
        values = pd.to_numeric(
            M[component],
            errors="coerce",
        ).dropna()

        ranked = (
            values.abs()
            .sort_values(ascending=False)
            .head(20)
        )

        for rank, gene in enumerate(
            ranked.index,
            start=1,
        ):
            weight = float(values.loc[gene])

            top_rows.append(
                {
                    "imodulon": str(component),
                    "rank": rank,
                    "gene": str(gene),
                    "weight": weight,
                    "abs_weight": abs(weight),
                }
            )

    pd.DataFrame(top_rows).to_csv(
        OUT / "top20_genes_per_imodulon.csv",
        index=False,
    )

    # --------------------------------------------------
    # Weight norms
    # --------------------------------------------------

    norm_rows = []

    for component in M.columns:
        values = pd.to_numeric(
            M[component],
            errors="coerce",
        ).fillna(0.0)

        norm_rows.append(
            {
                "imodulon": str(component),
                "l1_weight_sum": float(
                    np.abs(values).sum()
                ),
                "l2_weight_norm": float(
                    np.sqrt(np.square(values).sum())
                ),
                "max_abs_weight": float(
                    np.abs(values).max()
                ),
            }
        )

    pd.DataFrame(norm_rows).to_csv(
        OUT / "imodulon_weight_norms.csv",
        index=False,
    )

    # --------------------------------------------------
    # Reconstruction X ~ M A
    # --------------------------------------------------

    reconstruction_summary = None

    if X is not None:
        common_genes = M.index.intersection(X.index)
        common_samples = A.columns.intersection(X.columns)

        M2 = M.loc[common_genes].apply(
            pd.to_numeric,
            errors="coerce",
        )

        A2 = A.loc[:, common_samples].apply(
            pd.to_numeric,
            errors="coerce",
        )

        X2 = X.loc[
            common_genes,
            common_samples,
        ].apply(
            pd.to_numeric,
            errors="coerce",
        )

        MA = M2.to_numpy() @ A2.to_numpy()
        XV = X2.to_numpy()

        residual = XV - MA

        corr = correlation(XV, MA)

        denominator = np.linalg.norm(XV)

        relative_error = (
            float(np.linalg.norm(residual) / denominator)
            if denominator > 0
            else np.nan
        )

        reconstruction_summary = {
            "genes": len(common_genes),
            "samples": len(common_samples),
            "correlation_X_MA": corr,
            "relative_frobenius_error": relative_error,
        }

        pd.DataFrame(
            [reconstruction_summary]
        ).to_csv(
            OUT / "reconstruction_summary.csv",
            index=False,
        )

        per_sample = []

        for j, sample in enumerate(common_samples):
            x = XV[:, j]
            y = MA[:, j]

            per_sample.append(
                {
                    "sample": str(sample),
                    "correlation": correlation(x, y),
                    "rmse": float(
                        np.sqrt(
                            np.mean(
                                np.square(x - y)
                            )
                        )
                    ),
                }
            )

        pd.DataFrame(per_sample).to_csv(
            OUT / "reconstruction_by_sample.csv",
            index=False,
        )

        print()
        print("Reconstruction:")
        print(f"  genes:                    {len(common_genes)}")
        print(f"  samples:                  {len(common_samples)}")
        print(f"  corr(X, M@A):             {corr:.4f}")
        print(f"  relative Frobenius error: {relative_error:.4f}")

    # --------------------------------------------------
    # Training-overlap confirmation from model metadata
    # --------------------------------------------------

    n_gse136111 = None

    if sample_table is not None:
        found = pd.Series(
            False,
            index=sample_table.index,
        )

        for column in sample_table.columns:
            values = (
                sample_table[column]
                .fillna("")
                .astype(str)
            )

            found = found | values.str.contains(
                "GSE136111",
                regex=False,
            )

        model_overlap = sample_table.loc[found]

        n_gse136111 = len(model_overlap)

        model_overlap.to_csv(
            OUT / "GSE136111_model_rows.csv"
        )

        print()
        print(
            "GSE136111 rows in serialized model metadata: "
            f"{n_gse136111}"
        )

    # --------------------------------------------------
    # Human-readable summary
    # --------------------------------------------------

    lines = [
        "# CF-State Phase 2 — Audit 02",
        "",
        "## Aim",
        "",
        "Verify the internal structure of the frozen "
        "aeruPRECISE-411 representation before projecting "
        "any CF-State transcriptomes.",
        "",
        "## Results",
        "",
        f"- M: {M.shape[0]} genes × {M.shape[1]} iModulons",
        f"- A: {A.shape[0]} iModulons × {A.shape[1]} samples",
        f"- X present: {X is not None}",
        f"- log-TPM present: {log_tpm is not None}",
        f"- M/A programme identities match: {same_components}",
        f"- M gene identifiers unique: {M.index.is_unique}",
        f"- A sample identifiers unique: {A.columns.is_unique}",
    ]

    if reconstruction_summary is not None:
        lines.extend(
            [
                "",
                "## ICA reconstruction",
                "",
                f"- corr(X, M@A): "
                f"{reconstruction_summary['correlation_X_MA']:.6f}",
                f"- relative Frobenius error: "
                f"{reconstruction_summary['relative_frobenius_error']:.6f}",
            ]
        )

    if n_gse136111 is not None:
        lines.extend(
            [
                "",
                "## Provenance",
                "",
                f"- GSE136111 rows found in serialized model metadata: "
                f"{n_gse136111}",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This audit tests representation integrity only.",
            "It does not infer CF-State activities or assign biological states.",
            "",
            "If all structural checks pass, the frozen representation is "
            "ready for Audit 03: reproduction of activity inference on one "
            "external PAO1 RNA-seq experiment.",
        ]
    )

    (
        OUT / "AUDIT_02_SUMMARY.md"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("First 20 programme names:")
    print(
        programmes.head(20).to_string(
            index=False
        )
    )

    print()
    print("Audit 02: PASS")
    print()
    print(
        f"Summary: {OUT / 'AUDIT_02_SUMMARY.md'}"
    )


if __name__ == "__main__":
    main()
