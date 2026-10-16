from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

AUDIT03 = (
    ROOT
    / "results/phase2/audit_03_external_pao1_activity_inference"
    / "GSE203348_activity_summary.csv"
)

AUDIT04 = (
    ROOT
    / "results/phase2/audit_04_controlled_state_anchoring_GSE120602"
    / "controlled_contrast_vectors.csv"
)

OUT = (
    ROOT
    / "results/phase2/audit_05_controlled_anchor_subspace"
)

N_PERM = 5000
SEED = 20261006


def unit(x):
    x = np.asarray(x, dtype=float)

    norm = np.linalg.norm(x)

    if norm == 0:
        raise RuntimeError("Zero-length vector encountered")

    return x / norm


def cosine(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    denom = np.linalg.norm(x) * np.linalg.norm(y)

    if denom == 0:
        return np.nan

    return float(
        np.dot(x, y) / denom
    )


def vector_metrics(x, y):
    sx = pd.Series(
        np.asarray(x, dtype=float)
    )

    sy = pd.Series(
        np.asarray(y, dtype=float)
    )

    return {
        "pearson": float(
            sx.corr(
                sy,
                method="pearson",
            )
        ),
        "spearman": float(
            sx.corr(
                sy,
                method="spearman",
            )
        ),
        "cosine": cosine(
            sx.values,
            sy.values,
        ),
        "sign_agreement": float(
            (
                np.sign(sx.values)
                == np.sign(sy.values)
            ).mean()
        ),
    }


def make_subspace(stationary, antibiotic):
    # Normalize each biological axis so its arbitrary
    # activity magnitude does not dominate the geometry.
    B = np.column_stack(
        [
            unit(stationary),
            unit(antibiotic),
        ]
    )

    rank = np.linalg.matrix_rank(B)

    if rank != 2:
        raise RuntimeError(
            "Controlled anchor matrix is not rank 2"
        )

    # QR gives an orthonormal basis for the exact same
    # 2D span. This avoids the non-orthogonality of the
    # biological axes affecting variance calculations.
    Q, _ = np.linalg.qr(B)

    return B, Q[:, :2]


def project_target(target, B, Q):
    target = np.asarray(
        target,
        dtype=float,
    )

    reconstructed = (
        Q
        @ (
            Q.T
            @ target
        )
    )

    residual = (
        target
        - reconstructed
    )

    target_energy = float(
        np.dot(
            target,
            target,
        )
    )

    reconstructed_energy = float(
        np.dot(
            reconstructed,
            reconstructed,
        )
    )

    fraction_explained = (
        reconstructed_energy
        / target_energy
    )

    reconstruction_cosine = cosine(
        target,
        reconstructed,
    )

    # Coefficients use unit-normalized biological anchor
    # vectors. They are descriptive because the axes are
    # not themselves orthogonal.
    coefficients = np.linalg.lstsq(
        B,
        unit(target),
        rcond=None,
    )[0]

    return {
        "reconstructed":
            reconstructed,

        "residual":
            residual,

        "fraction_energy_explained":
            fraction_explained,

        "reconstruction_cosine":
            reconstruction_cosine,

        "stationary_coefficient":
            float(
                coefficients[0]
            ),

        "antibiotic_specific_coefficient":
            float(
                coefficients[1]
            ),
    }


def permutation_test(
    target,
    stationary,
    antibiotic,
    observed,
    n_perm,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    n = len(target)

    null = np.empty(
        n_perm,
        dtype=float,
    )

    # Shuffle programme identities jointly for both
    # anchor vectors. This preserves their relationship
    # with each other but destroys correspondence with
    # the target biofilm programmes.
    for i in range(n_perm):
        order = rng.permutation(
            n
        )

        shuffled_s = (
            stationary[
                order
            ]
        )

        shuffled_a = (
            antibiotic[
                order
            ]
        )

        _, Q = make_subspace(
            shuffled_s,
            shuffled_a,
        )

        projection = (
            Q
            @ (
                Q.T
                @ target
            )
        )

        null[i] = (
            np.dot(
                projection,
                projection,
            )
            / np.dot(
                target,
                target,
            )
        )

    p = (
        1
        + int(
            np.sum(
                null
                >= observed
            )
        )
    ) / (
        n_perm
        + 1
    )

    return null, float(p)


def top_residual_table(
    names,
    target,
    reconstructed,
    residual,
    n=20,
):
    df = pd.DataFrame(
        {
            "imodulon": names,
            "observed_delta": target,
            "anchor_reconstruction": reconstructed,
            "residual_delta": residual,
        }
    )

    df["abs_residual"] = (
        df[
            "residual_delta"
        ]
        .abs()
    )

    return (
        df.sort_values(
            "abs_residual",
            ascending=False,
        )
        .head(n)
        .reset_index(
            drop=True
        )
    )


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 05")
    print("Controlled-anchor subspace decomposition")
    print("=" * 78)

    for path in [
        AUDIT03,
        AUDIT04,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(path)
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Load Audit 03 external biofilm vector
    # ------------------------------------------------------------

    a3 = (
        pd.read_csv(
            AUDIT03
        )
        .set_index(
            "imodulon"
        )
    )

    gse203348 = (
        a3[
            "delta_biofilm_minus_planktonic"
        ]
        .astype(float)
    )

    # ------------------------------------------------------------
    # Load Audit 04 controlled vectors
    # ------------------------------------------------------------

    a4 = pd.read_csv(
        AUDIT04,
        index_col=0,
    )

    required = [
        "biofilm_vs_planktonic",
        "stationary_vs_planktonic",
        "persister_vs_stationary",
        "persister_vs_planktonic",
    ]

    missing = [
        col
        for col in required
        if col not in a4.columns
    ]

    if missing:
        raise RuntimeError(
            "Audit 04 missing columns: "
            + ", ".join(
                missing
            )
        )

    common = (
        gse203348.index
        .intersection(
            a4.index
        )
    )

    if len(common) != 116:
        raise RuntimeError(
            "Expected 116 common iModulons, found "
            + str(
                len(common)
            )
        )

    names = common.to_numpy()

    biofilm_203 = (
        gse203348.loc[
            common
        ]
        .to_numpy(
            dtype=float
        )
    )

    biofilm_120 = (
        a4.loc[
            common,
            "biofilm_vs_planktonic",
        ]
        .to_numpy(
            dtype=float
        )
    )

    stationary = (
        a4.loc[
            common,
            "stationary_vs_planktonic",
        ]
        .to_numpy(
            dtype=float
        )
    )

    antibiotic = (
        a4.loc[
            common,
            "persister_vs_stationary",
        ]
        .to_numpy(
            dtype=float
        )
    )

    persister_total = (
        a4.loc[
            common,
            "persister_vs_planktonic",
        ]
        .to_numpy(
            dtype=float
        )
    )

    # ------------------------------------------------------------
    # Basic geometry
    # ------------------------------------------------------------

    vectors = {
        "stationary_vs_planktonic":
            stationary,

        "persister_vs_stationary":
            antibiotic,

        "persister_vs_planktonic":
            persister_total,

        "GSE120602_biofilm":
            biofilm_120,

        "GSE203348_biofilm":
            biofilm_203,
    }

    geometry_rows = []

    for name1, vector1 in vectors.items():
        for name2, vector2 in vectors.items():
            metrics = vector_metrics(
                vector1,
                vector2,
            )

            geometry_rows.append(
                {
                    "vector_1": name1,
                    "vector_2": name2,
                    **metrics,
                }
            )

    geometry = pd.DataFrame(
        geometry_rows
    )

    geometry.to_csv(
        OUT
        / "programme_vector_geometry.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Build 2D controlled anchor subspace
    # ------------------------------------------------------------

    B, Q = make_subspace(
        stationary,
        antibiotic,
    )

    anchor_metrics = vector_metrics(
        stationary,
        antibiotic,
    )

    print()
    print("Controlled anchor relationship")
    print(
        "  stationary vs antibiotic-specific Pearson:",
        round(
            anchor_metrics["pearson"],
            4,
        ),
    )
    print(
        "  stationary vs antibiotic-specific Spearman:",
        round(
            anchor_metrics["spearman"],
            4,
        ),
    )
    print(
        "  stationary vs antibiotic-specific cosine:",
        round(
            anchor_metrics["cosine"],
            4,
        ),
    )

    # ------------------------------------------------------------
    # Project biofilm vectors onto controlled subspace
    # ------------------------------------------------------------

    targets = {
        "GSE120602_biofilm":
            biofilm_120,

        "GSE203348_biofilm":
            biofilm_203,
    }

    summary_rows = []
    residual_tables = {}

    for i, (
        target_name,
        target,
    ) in enumerate(
        targets.items()
    ):
        result = project_target(
            target,
            B,
            Q,
        )

        stationary_metrics = (
            vector_metrics(
                target,
                stationary,
            )
        )

        antibiotic_metrics = (
            vector_metrics(
                target,
                antibiotic,
            )
        )

        null, p = permutation_test(
            target=target,
            stationary=stationary,
            antibiotic=antibiotic,
            observed=result[
                "fraction_energy_explained"
            ],
            n_perm=N_PERM,
            seed=SEED + i,
        )

        np.savetxt(
            OUT
            / (
                target_name
                + "_permutation_null.csv"
            ),
            null,
            delimiter=",",
        )

        summary_rows.append(
            {
                "target":
                    target_name,

                "stationary_pearson":
                    stationary_metrics[
                        "pearson"
                    ],

                "stationary_spearman":
                    stationary_metrics[
                        "spearman"
                    ],

                "stationary_cosine":
                    stationary_metrics[
                        "cosine"
                    ],

                "antibiotic_pearson":
                    antibiotic_metrics[
                        "pearson"
                    ],

                "antibiotic_spearman":
                    antibiotic_metrics[
                        "spearman"
                    ],

                "antibiotic_cosine":
                    antibiotic_metrics[
                        "cosine"
                    ],

                "controlled_subspace_fraction_energy":
                    result[
                        "fraction_energy_explained"
                    ],

                "controlled_subspace_reconstruction_cosine":
                    result[
                        "reconstruction_cosine"
                    ],

                "permutation_p":
                    p,

                "stationary_coefficient":
                    result[
                        "stationary_coefficient"
                    ],

                "antibiotic_specific_coefficient":
                    result[
                        "antibiotic_specific_coefficient"
                    ],
            }
        )

        residual = top_residual_table(
            names=names,
            target=target,
            reconstructed=result[
                "reconstructed"
            ],
            residual=result[
                "residual"
            ],
            n=20,
        )

        residual_tables[
            target_name
        ] = residual

        residual.to_csv(
            OUT
            / (
                target_name
                + "_top20_unexplained_programmes.csv"
            ),
            index=False,
        )

        full = pd.DataFrame(
            {
                "imodulon":
                    names,

                "observed_delta":
                    target,

                "anchor_reconstruction":
                    result[
                        "reconstructed"
                    ],

                "residual_delta":
                    result[
                        "residual"
                    ],
            }
        )

        full.to_csv(
            OUT
            / (
                target_name
                + "_anchor_decomposition.csv"
            ),
            index=False,
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        OUT
        / "controlled_subspace_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------

    print()
    print("Controlled-subspace decomposition")
    print(
        summary[
            [
                "target",
                "stationary_spearman",
                "antibiotic_spearman",
                "controlled_subspace_fraction_energy",
                "controlled_subspace_reconstruction_cosine",
                "permutation_p",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    for target_name in targets:
        print()
        print(
            "Top programmes NOT explained by "
            "stationary + antibiotic-specific anchors:"
        )
        print(
            target_name
        )

        print(
            residual_tables[
                target_name
            ][
                [
                    "imodulon",
                    "observed_delta",
                    "anchor_reconstruction",
                    "residual_delta",
                ]
            ]
            .head(12)
            .round(3)
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Audit gate
    # ------------------------------------------------------------

    finite = np.isfinite(
        summary.select_dtypes(
            include=[
                np.number
            ]
        ).to_numpy()
    ).all()

    checks = {
        "116_common_imodulons":
            len(common) == 116,

        "anchor_rank_2":
            np.linalg.matrix_rank(
                B
            ) == 2,

        "finite_results":
            bool(
                finite
            ),

        "two_targets_decomposed":
            len(summary) == 2,
    }

    passed = all(
        checks.values()
    )

    pd.DataFrame(
        [
            {
                "check": key,
                "pass": value,
            }
            for key, value
            in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Human-readable summary
    # ------------------------------------------------------------

    lines = [
        "# CF-State Phase 2 — Audit 05",
        "",
        "## Objective",
        "",
        "Test how much of independent biofilm-associated iModulon "
        "activity can be represented by two controlled physiological "
        "axes from GSE120602:",
        "",
        "1. stationary minus planktonic",
        "2. persister minus stationary",
        "",
        "The persister-minus-planktonic vector was deliberately not "
        "used as a third anchor because it is algebraically composed "
        "of the two axes above.",
        "",
        "## Interpretation boundary",
        "",
        "The GSE120602 biofilm decomposition is internal/descriptive "
        "because its controlled axes come from the same experiment.",
        "",
        "GSE203348 is the independent test of how well this controlled "
        "two-axis physiological subspace transfers to another "
        "biofilm experiment.",
        "",
        "Residual programmes are candidates for additional biological "
        "anchoring; they are not automatically CF-specific or "
        "biofilm-specific.",
        "",
        "## Decision",
        "",
        "PASS: " + str(
            passed
        ),
    ]

    (
        OUT
        / "AUDIT_05_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 05:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_05_SUMMARY.md",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
