from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
)


DATA_FILE = Path(
    "data/processed/biofilm/"
    "all_studies_harmonised_unscaled.csv"
)

OUT_DIR = Path(
    "results/audit/loso_rank_pair_classifier"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Settings
# ============================================================

# Candidate genes must change in the same direction
# in BOTH training studies.

MIN_GENE_EFFECT = 0.05

# Take strongest genes from each side before constructing pairs.
N_HIGH_CANDIDATES = 50
N_LOW_CANDIDATES = 50

# Final compact classifier.
N_PAIRS = 15

# Prevent one gene from dominating all selected pairs.
MAX_GENE_USES = 2


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(
    DATA_FILE
)

META = [
    "study",
    "sample",
    "state",
]

GENES = [
    c
    for c in df.columns
    if c not in META
]

metadata = df[
    META
].copy()

X = df[
    GENES
].astype(float)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 20: "
    "leak-free rank-pair classifier"
)
print("=" * 100)

print(
    f"Samples: {len(df)}"
)

print(
    f"Genes:   {len(GENES)}"
)


# ============================================================
# Convert each sample independently to percentile ranks
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


# ============================================================
# Learn candidate biofilm-high and biofilm-low genes
# from TRAINING STUDIES ONLY
# ============================================================

def learn_candidate_genes(
    train_meta,
    train_rank,
):

    studies = list(
        train_meta[
            "study"
        ].unique()
    )

    if len(studies) != 2:

        raise RuntimeError(
            "Expected exactly two training studies."
        )


    effects = {}

    for study in studies:

        study_mask = (
            train_meta["study"]
            == study
        )

        bio_mask = (
            study_mask
            &
            (
                train_meta["state"]
                == "biofilm"
            )
        )

        plank_mask = (
            study_mask
            &
            (
                train_meta["state"]
                == "planktonic"
            )
        )

        bio_mean = (
            train_rank.loc[
                bio_mask
            ]
            .mean(
                axis=0
            )
        )

        plank_mean = (
            train_rank.loc[
                plank_mask
            ]
            .mean(
                axis=0
            )
        )

        effects[
            study
        ] = (
            bio_mean
            - plank_mean
        )


    e1 = effects[
        studies[0]
    ]

    e2 = effects[
        studies[1]
    ]


    # ----------------------------------------
    # Consistently higher in biofilm
    # ----------------------------------------

    high_mask = (
        (e1 > 0)
        &
        (e2 > 0)
    )

    high_strength = pd.Series(
        np.minimum(
            np.abs(e1),
            np.abs(e2),
        ),
        index=GENES,
    )

    high = pd.DataFrame(
        {
            "gene": GENES,
            "effect_1": e1.values,
            "effect_2": e2.values,
            "strength":
                high_strength.values,
        }
    )

    high = high[
        high_mask.values
    ]

    high = high[
        high["strength"]
        >= MIN_GENE_EFFECT
    ]

    high = high.sort_values(
        "strength",
        ascending=False,
    ).head(
        N_HIGH_CANDIDATES
    )


    # ----------------------------------------
    # Consistently lower in biofilm
    # ----------------------------------------

    low_mask = (
        (e1 < 0)
        &
        (e2 < 0)
    )

    low_strength = pd.Series(
        np.minimum(
            np.abs(e1),
            np.abs(e2),
        ),
        index=GENES,
    )

    low = pd.DataFrame(
        {
            "gene": GENES,
            "effect_1": e1.values,
            "effect_2": e2.values,
            "strength":
                low_strength.values,
        }
    )

    low = low[
        low_mask.values
    ]

    low = low[
        low["strength"]
        >= MIN_GENE_EFFECT
    ]

    low = low.sort_values(
        "strength",
        ascending=False,
    ).head(
        N_LOW_CANDIDATES
    )


    return (
        high,
        low,
        studies,
    )


# ============================================================
# Pair score helper
#
# Pair:
#   high_gene > low_gene
#
# should occur more often in biofilm than planktonic.
#
# Each study contributes:
#
# P(high > low | biofilm)
# -
# P(high > low | planktonic)
# ============================================================

def pair_effect_in_study(
    gene_high,
    gene_low,
    study,
    train_meta,
    train_rank,
):

    study_mask = (
        train_meta["study"]
        == study
    )

    bio_idx = train_meta.index[
        study_mask
        &
        (
            train_meta["state"]
            == "biofilm"
        )
    ]

    plank_idx = train_meta.index[
        study_mask
        &
        (
            train_meta["state"]
            == "planktonic"
        )
    ]


    def proportion_high_gt_low(
        idx,
    ):

        high_values = (
            train_rank.loc[
                idx,
                gene_high,
            ]
            .to_numpy()
        )

        low_values = (
            train_rank.loc[
                idx,
                gene_low,
            ]
            .to_numpy()
        )

        greater = (
            high_values
            > low_values
        ).astype(float)

        ties = (
            high_values
            == low_values
        ).astype(float)

        # Tie contributes 0.5 vote.
        return float(
            np.mean(
                greater
                + 0.5 * ties
            )
        )


    bio_prop = (
        proportion_high_gt_low(
            bio_idx
        )
    )

    plank_prop = (
        proportion_high_gt_low(
            plank_idx
        )
    )

    return (
        bio_prop
        - plank_prop
    )


# ============================================================
# Construct pair classifier using TRAINING data only
# ============================================================

def learn_pairs(
    high,
    low,
    studies,
    train_meta,
    train_rank,
):

    records = []


    for high_gene in high[
        "gene"
    ]:

        for low_gene in low[
            "gene"
        ]:

            effect_1 = (
                pair_effect_in_study(
                    high_gene,
                    low_gene,
                    studies[0],
                    train_meta,
                    train_rank,
                )
            )

            effect_2 = (
                pair_effect_in_study(
                    high_gene,
                    low_gene,
                    studies[1],
                    train_meta,
                    train_rank,
                )
            )


            # Pair must work in the expected
            # direction in BOTH training studies.

            if (
                effect_1 <= 0
                or
                effect_2 <= 0
            ):
                continue


            minimum_effect = min(
                effect_1,
                effect_2,
            )

            mean_effect = (
                effect_1
                + effect_2
            ) / 2.0


            records.append(
                {
                    "high_gene":
                        high_gene,

                    "low_gene":
                        low_gene,

                    studies[0]:
                        effect_1,

                    studies[1]:
                        effect_2,

                    "minimum_pair_effect":
                        minimum_effect,

                    "mean_pair_effect":
                        mean_effect,
                }
            )


    pairs = pd.DataFrame(
        records
    )


    if pairs.empty:

        raise RuntimeError(
            "No valid gene pairs found."
        )


    pairs = pairs.sort_values(
        [
            "minimum_pair_effect",
            "mean_pair_effect",
        ],
        ascending=[
            False,
            False,
        ],
    )


    # ----------------------------------------
    # Select diverse pairs
    #
    # Each gene may occur at most
    # MAX_GENE_USES times.
    # ----------------------------------------

    gene_usage = defaultdict(
        int
    )

    selected = []


    for _, row in pairs.iterrows():

        high_gene = row[
            "high_gene"
        ]

        low_gene = row[
            "low_gene"
        ]


        if (
            gene_usage[
                high_gene
            ]
            >= MAX_GENE_USES
        ):
            continue


        if (
            gene_usage[
                low_gene
            ]
            >= MAX_GENE_USES
        ):
            continue


        selected.append(
            row
        )

        gene_usage[
            high_gene
        ] += 1

        gene_usage[
            low_gene
        ] += 1


        if len(
            selected
        ) >= N_PAIRS:

            break


    selected = pd.DataFrame(
        selected
    )


    if selected.empty:

        raise RuntimeError(
            "No pairs survived selection."
        )


    return (
        pairs,
        selected,
    )


# ============================================================
# Score samples
#
# Every selected pair votes:
#
# high_gene > low_gene:
#     1 = biofilm vote
#
# high_gene < low_gene:
#     0 = planktonic vote
#
# tie:
#     0.5
#
# Final score is fraction of biofilm votes.
# Natural classification boundary = 0.5
# ============================================================

def score_samples(
    rank_matrix,
    selected_pairs,
):

    scores = []


    for idx in rank_matrix.index:

        votes = []


        for _, pair in (
            selected_pairs.iterrows()
        ):

            high_gene = pair[
                "high_gene"
            ]

            low_gene = pair[
                "low_gene"
            ]


            high_value = (
                rank_matrix.loc[
                    idx,
                    high_gene,
                ]
            )

            low_value = (
                rank_matrix.loc[
                    idx,
                    low_gene,
                ]
            )


            if (
                high_value
                > low_value
            ):

                vote = 1.0

            elif (
                high_value
                < low_value
            ):

                vote = 0.0

            else:

                vote = 0.5


            votes.append(
                vote
            )


        scores.append(
            np.mean(
                votes
            )
        )


    return pd.Series(
        scores,
        index=rank_matrix.index,
    )


# ============================================================
# Leave-one-study-out validation
# ============================================================

studies_all = list(
    metadata[
        "study"
    ].unique()
)

prediction_records = []
fold_records = []


for held_out in studies_all:

    print()
    print("=" * 100)
    print(
        f"HELD-OUT STUDY: "
        f"{held_out}"
    )
    print("=" * 100)


    train_mask = (
        metadata["study"]
        != held_out
    )

    test_mask = (
        metadata["study"]
        == held_out
    )


    train_meta = (
        metadata.loc[
            train_mask
        ]
        .copy()
    )

    test_meta = (
        metadata.loc[
            test_mask
        ]
        .copy()
    )

    train_rank = (
        X_rank.loc[
            train_mask
        ]
        .copy()
    )

    test_rank = (
        X_rank.loc[
            test_mask
        ]
        .copy()
    )


    # ----------------------------------------
    # Learn candidate genes
    # ----------------------------------------

    high, low, train_studies = (
        learn_candidate_genes(
            train_meta,
            train_rank,
        )
    )


    print(
        f"Training studies: "
        f"{', '.join(train_studies)}"
    )

    print(
        f"Candidate biofilm-high genes: "
        f"{len(high)}"
    )

    print(
        f"Candidate biofilm-low genes:  "
        f"{len(low)}"
    )


    # ----------------------------------------
    # Learn pair classifier
    # ----------------------------------------

    all_pairs, selected_pairs = (
        learn_pairs(
            high,
            low,
            train_studies,
            train_meta,
            train_rank,
        )
    )


    print(
        f"Valid training pairs: "
        f"{len(all_pairs)}"
    )

    print(
        f"Selected classifier pairs: "
        f"{len(selected_pairs)}"
    )


    selected_pairs.to_csv(
        OUT_DIR
        / (
            f"selected_pairs_without_"
            f"{held_out}.csv"
        ),
        index=False,
    )


    # ----------------------------------------
    # Training performance
    # ----------------------------------------

    train_scores = score_samples(
        train_rank,
        selected_pairs,
    )


    train_true = (
        train_meta[
            "state"
        ]
        == "biofilm"
    ).astype(int)


    train_pred = (
        train_scores
        > 0.5
    ).astype(int)


    train_accuracy = (
        accuracy_score(
            train_true,
            train_pred,
        )
    )


    # ----------------------------------------
    # Held-out prediction
    # ----------------------------------------

    test_scores = score_samples(
        test_rank,
        selected_pairs,
    )


    test_true = (
        test_meta[
            "state"
        ]
        == "biofilm"
    ).astype(int)


    test_pred = (
        test_scores
        > 0.5
    ).astype(int)


    test_accuracy = (
        accuracy_score(
            test_true,
            test_pred,
        )
    )


    test_balanced_accuracy = (
        balanced_accuracy_score(
            test_true,
            test_pred,
        )
    )


    try:

        test_auc = (
            roc_auc_score(
                test_true,
                test_scores,
            )
        )

    except ValueError:

        test_auc = np.nan


    cm = confusion_matrix(
        test_true,
        test_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )


    result = (
        test_meta.copy()
    )

    result[
        "biofilm_vote_fraction"
    ] = test_scores

    result[
        "predicted_state"
    ] = np.where(
        test_pred == 1,
        "biofilm",
        "planktonic",
    )

    result[
        "correct"
    ] = (
        result[
            "predicted_state"
        ]
        ==
        result[
            "state"
        ]
    )


    print()
    print("Held-out predictions:")

    print()

    print(
        result[
            [
                "sample",
                "state",
                "biofilm_vote_fraction",
                "predicted_state",
                "correct",
            ]
        ]
        .to_string(
            index=False
        )
    )


    print()

    print(
        f"Training accuracy:         "
        f"{train_accuracy:.3f}"
    )

    print(
        f"Held-out accuracy:         "
        f"{test_accuracy:.3f}"
    )

    print(
        f"Held-out balanced accuracy:"
        f" {test_balanced_accuracy:.3f}"
    )

    print(
        f"Held-out ROC AUC:          "
        f"{test_auc:.3f}"
    )

    print(
        f"Confusion matrix: "
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )


    fold_records.append(
        {
            "held_out_study":
                held_out,

            "n_high_candidates":
                len(high),

            "n_low_candidates":
                len(low),

            "n_valid_pairs":
                len(all_pairs),

            "n_selected_pairs":
                len(selected_pairs),

            "training_accuracy":
                train_accuracy,

            "held_out_accuracy":
                test_accuracy,

            "held_out_balanced_accuracy":
                test_balanced_accuracy,

            "held_out_auc":
                test_auc,

            "TN":
                tn,

            "FP":
                fp,

            "FN":
                fn,

            "TP":
                tp,
        }
    )


    for idx in result.index:

        prediction_records.append(
            {
                "held_out_study":
                    held_out,

                "sample":
                    result.loc[
                        idx,
                        "sample",
                    ],

                "state":
                    result.loc[
                        idx,
                        "state",
                    ],

                "biofilm_vote_fraction":
                    result.loc[
                        idx,
                        "biofilm_vote_fraction",
                    ],

                "predicted_state":
                    result.loc[
                        idx,
                        "predicted_state",
                    ],

                "correct":
                    result.loc[
                        idx,
                        "correct",
                    ],
            }
        )


# ============================================================
# Overall held-out performance
# ============================================================

predictions = pd.DataFrame(
    prediction_records
)

fold_summary = pd.DataFrame(
    fold_records
)


predictions.to_csv(
    OUT_DIR
    / "held_out_predictions.csv",
    index=False,
)

fold_summary.to_csv(
    OUT_DIR
    / "fold_summary.csv",
    index=False,
)


overall_true = (
    predictions["state"]
    == "biofilm"
).astype(int)


overall_pred = (
    predictions[
        "predicted_state"
    ]
    == "biofilm"
).astype(int)


overall_accuracy = (
    accuracy_score(
        overall_true,
        overall_pred,
    )
)


overall_balanced_accuracy = (
    balanced_accuracy_score(
        overall_true,
        overall_pred,
    )
)


overall_auc = (
    roc_auc_score(
        overall_true,
        predictions[
            "biofilm_vote_fraction"
        ],
    )
)


print()
print("=" * 100)
print(
    "AUDIT 20 — "
    "LEAVE-ONE-STUDY-OUT SUMMARY"
)
print("=" * 100)

print(
    fold_summary[
        [
            "held_out_study",
            "n_selected_pairs",
            "held_out_accuracy",
            "held_out_balanced_accuracy",
            "held_out_auc",
        ]
    ].to_string(
        index=False
    )
)


print()

print(
    f"Overall held-out accuracy:          "
    f"{overall_accuracy:.3f}"
)

print(
    f"Overall held-out balanced accuracy: "
    f"{overall_balanced_accuracy:.3f}"
)

print(
    f"Overall held-out ROC AUC:           "
    f"{overall_auc:.3f}"
)


print()
print("=" * 100)
print("AUDIT 20 COMPLETE")
print("=" * 100)

print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)

print()
print(
    "Every gene and gene pair used for a "
    "held-out sample was selected using "
    "the other two studies only."
)

print()
