from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/processed/biofilm")
OUT_DIR = Path("results/audit/expression_state_qc")

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


INPUT_FILE = (
    DATA_DIR
    / "all_studies_harmonised_unscaled.csv"
)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(INPUT_FILE)

meta_cols = [
    "study",
    "sample",
    "state",
]

gene_cols = [
    x
    for x in df.columns
    if x not in meta_cols
]

metadata = df[
    meta_cols
].copy()

X = df[
    gene_cols
].astype(float)


print()
print("=" * 100)
print("CF-STATE — Audit 13: expression transformation and state QC")
print("=" * 100)

print(f"Samples: {X.shape[0]}")
print(f"Genes:   {X.shape[1]}")

print()
print("Studies:")
print(
    metadata["study"]
    .value_counts()
    .to_string()
)

print()
print("States:")
print(
    metadata["state"]
    .value_counts()
    .to_string()
)


# ============================================================
# Representation A
# log2(x + 1)
# ============================================================

X_log = np.log2(
    X + 1.0
)

X_log = pd.DataFrame(
    X_log,
    index=X.index,
    columns=X.columns,
)

X_log.to_csv(
    OUT_DIR / "expression_log2p1.csv",
    index=False,
)


# ============================================================
# Representation B
# within-sample percentile ranks
#
# Every sample gets values from approximately 0 -> 1.
# This removes absolute sequencing-depth scale.
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)

X_rank.to_csv(
    OUT_DIR / "expression_sample_rank_percentile.csv",
    index=False,
)


# ============================================================
# Representation C
# within-study gene-centred log expression
#
# IMPORTANT:
# exploratory QC only.
#
# For every study and every gene:
#
#     sample log expression
#       -
#     study median expression of that gene
#
# This removes study-specific gene baselines.
# ============================================================

X_centered = pd.DataFrame(
    index=X.index,
    columns=X.columns,
    dtype=float,
)

for study in metadata["study"].unique():

    idx = metadata.index[
        metadata["study"] == study
    ]

    block = X_log.loc[
        idx
    ]

    gene_medians = block.median(
        axis=0
    )

    X_centered.loc[
        idx
    ] = (
        block
        - gene_medians
    )


X_centered.to_csv(
    OUT_DIR
    / "expression_within_study_gene_centered.csv",
    index=False,
)


# ============================================================
# Correlation QC
# ============================================================

def correlation_summary(
    expression,
    name,
):

    corr = expression.T.corr(
        method="pearson"
    )

    corr.to_csv(
        OUT_DIR
        / f"{name}_sample_correlations.csv"
    )

    records = []

    for i in range(len(corr)):

        for j in range(
            i + 1,
            len(corr),
        ):

            same_study = (
                metadata.loc[i, "study"]
                ==
                metadata.loc[j, "study"]
            )

            same_state = (
                metadata.loc[i, "state"]
                ==
                metadata.loc[j, "state"]
            )

            records.append(
                {
                    "representation": name,

                    "sample_1":
                        metadata.loc[
                            i,
                            "sample",
                        ],

                    "sample_2":
                        metadata.loc[
                            j,
                            "sample",
                        ],

                    "study_1":
                        metadata.loc[
                            i,
                            "study",
                        ],

                    "study_2":
                        metadata.loc[
                            j,
                            "study",
                        ],

                    "state_1":
                        metadata.loc[
                            i,
                            "state",
                        ],

                    "state_2":
                        metadata.loc[
                            j,
                            "state",
                        ],

                    "same_study":
                        same_study,

                    "same_state":
                        same_state,

                    "correlation":
                        corr.iloc[
                            i,
                            j,
                        ],
                }
            )

    return pd.DataFrame(
        records
    )


all_corr = []

for name, expression in [
    ("log2p1", X_log),
    ("rank", X_rank),
    (
        "within_study_centered",
        X_centered,
    ),
]:

    temp = correlation_summary(
        expression,
        name,
    )

    all_corr.append(
        temp
    )


corr_long = pd.concat(
    all_corr,
    ignore_index=True,
)

corr_long.to_csv(
    OUT_DIR
    / "pairwise_sample_correlations.csv",
    index=False,
)


# ============================================================
# Correlation summaries
# ============================================================

summary = (
    corr_long
    .groupby(
        [
            "representation",
            "same_study",
            "same_state",
        ]
    )["correlation"]
    .agg(
        [
            "count",
            "mean",
            "median",
            "min",
            "max",
        ]
    )
    .reset_index()
)

summary.to_csv(
    OUT_DIR
    / "correlation_summary.csv",
    index=False,
)

print()
print("=" * 100)
print("CORRELATION SUMMARY")
print("=" * 100)

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# PCA helper
# ============================================================

def run_pca(
    expression,
    representation,
):

    # Centre each gene for PCA.
    # We do not variance-scale genes here because
    # extremely low-variance genes could become
    # disproportionately important.

    arr = expression.to_numpy(
        dtype=float
    )

    arr = arr - np.mean(
        arr,
        axis=0,
        keepdims=True,
    )

    pca = PCA(
        n_components=min(
            5,
            arr.shape[0] - 1,
        )
    )

    coords = pca.fit_transform(
        arr
    )

    explained = (
        pca.explained_variance_ratio_
    )

    out = metadata.copy()

    for i in range(
        coords.shape[1]
    ):

        out[
            f"PC{i + 1}"
        ] = coords[:, i]

    out.to_csv(
        OUT_DIR
        / f"{representation}_pca_coordinates.csv",
        index=False,
    )

    print()
    print(
        f"{representation} PCA variance:"
    )

    for i, value in enumerate(
        explained
    ):

        print(
            f"  PC{i + 1}: "
            f"{value * 100:.2f}%"
        )

    # ----------------------------------------
    # Plot PC1 vs PC2
    # ----------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    studies = list(
        metadata["study"].unique()
    )

    states = [
        "biofilm",
        "planktonic",
    ]

    markers = {
        "biofilm": "o",
        "planktonic": "^",
    }

    # Use matplotlib default colours.
    colour_cycle = (
        plt.rcParams[
            "axes.prop_cycle"
        ]
        .by_key()["color"]
    )

    study_colours = {
        study:
            colour_cycle[
                i % len(colour_cycle)
            ]
        for i, study
        in enumerate(studies)
    }

    for study in studies:

        for state in states:

            mask = (
                (metadata["study"] == study)
                &
                (metadata["state"] == state)
            )

            if not mask.any():
                continue

            ax.scatter(
                out.loc[
                    mask,
                    "PC1",
                ],
                out.loc[
                    mask,
                    "PC2",
                ],
                marker=markers[
                    state
                ],
                s=90,
                label=(
                    f"{study} | "
                    f"{state}"
                ),
                color=study_colours[
                    study
                ],
            )

    # Label samples
    for i in range(
        len(out)
    ):

        ax.annotate(
            out.loc[
                i,
                "sample",
            ],
            (
                out.loc[
                    i,
                    "PC1",
                ],
                out.loc[
                    i,
                    "PC2",
                ],
            ),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel(
        f"PC1 "
        f"({explained[0] * 100:.1f}%)"
    )

    ax.set_ylabel(
        f"PC2 "
        f"({explained[1] * 100:.1f}%)"
    )

    ax.set_title(
        f"CF-State biofilm QC — "
        f"{representation}"
    )

    ax.legend(
        fontsize=7,
        bbox_to_anchor=(
            1.02,
            1,
        ),
        loc="upper left",
    )

    fig.tight_layout()

    fig.savefig(
        OUT_DIR
        / f"{representation}_pca.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    return out


# ============================================================
# Run PCA
# ============================================================

pca_outputs = {}

for name, expression in [
    ("log2p1", X_log),
    ("rank", X_rank),
    (
        "within_study_centered",
        X_centered,
    ),
]:

    pca_outputs[name] = run_pca(
        expression,
        name,
    )


# ============================================================
# Study/state centroids
# ============================================================

print()
print("=" * 100)
print("WITHIN-STUDY STATE CENTROID DISTANCES")
print("=" * 100)

centroid_records = []

for name, pca_df in pca_outputs.items():

    pc_cols = [
        x
        for x in pca_df.columns
        if x.startswith("PC")
    ]

    for study in metadata[
        "study"
    ].unique():

        block = pca_df[
            pca_df["study"]
            == study
        ]

        biofilm = block[
            block["state"]
            == "biofilm"
        ][pc_cols]

        planktonic = block[
            block["state"]
            == "planktonic"
        ][pc_cols]

        biofilm_centroid = (
            biofilm.mean(
                axis=0
            )
        )

        plank_centroid = (
            planktonic.mean(
                axis=0
            )
        )

        distance = float(
            np.linalg.norm(
                biofilm_centroid
                - plank_centroid
            )
        )

        centroid_records.append(
            {
                "representation": name,
                "study": study,
                "centroid_distance":
                    distance,
            }
        )


centroid_df = pd.DataFrame(
    centroid_records
)

centroid_df.to_csv(
    OUT_DIR
    / "state_centroid_distances.csv",
    index=False,
)

print(
    centroid_df.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 13 COMPLETE")
print("=" * 100)

print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)

print()
print(
    "Interpretation reminder:"
)

print(
    "  log2p1 = mostly raw study structure"
)

print(
    "  rank = scale-robust sample representation"
)

print(
    "  within_study_centered = exploratory "
    "state-vs-study QC only"
)

print()
