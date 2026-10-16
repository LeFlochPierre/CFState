from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)


DATA_FILE = Path(
    "data/processed/biofilm/"
    "all_studies_harmonised_unscaled.csv"
)

OUT_DIR = Path(
    "results/audit/loso_biofilm_state_score"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load
# ============================================================

df = pd.read_csv(DATA_FILE)

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

metadata = df[META].copy()

X = df[GENES].astype(float)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 19: "
    "leak-free leave-one-study-out biofilm score"
)
print("=" * 100)

print(f"Samples: {len(df)}")
print(f"Genes:   {len(GENES)}")


# ============================================================
# Single-sample percentile ranks
#
# Important:
# each sample is transformed independently.
# Therefore an unseen sample does not need information
# from the training cohort.
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


# ============================================================
# Helper: learn signature using TRAINING STUDIES ONLY
# ============================================================

def learn_signature(
    train_meta,
    train_rank,
):

    train_studies = list(
        train_meta[
            "study"
        ].unique()
    )

    if len(train_studies) != 2:
        raise RuntimeError(
            "Expected exactly two training studies."
        )

    study_effects = []

    for study in train_studies:

        mask_study = (
            train_meta["study"]
            == study
        )

        bio_mask = (
            mask_study
            &
            (
                train_meta["state"]
                == "biofilm"
            )
        )

        plank_mask = (
            mask_study
            &
            (
                train_meta["state"]
                == "planktonic"
            )
        )

        bio_mean = (
            train_rank.loc[
                bio_mask
            ].mean(axis=0)
        )

        plank_mean = (
            train_rank.loc[
                plank_mask
            ].mean(axis=0)
        )

        effect = (
            bio_mean
            - plank_mean
        )

        study_effects.append(
            effect
        )

    effect_1 = study_effects[0]
    effect_2 = study_effects[1]

    same_positive = (
        (effect_1 > 0)
        &
        (effect_2 > 0)
    )

    same_negative = (
        (effect_1 < 0)
        &
        (effect_2 < 0)
    )

    consistent = (
        same_positive
        |
        same_negative
    )

    # Conservative strength:
    # use the weaker effect of the two studies.
    strength = np.minimum(
        np.abs(effect_1),
        np.abs(effect_2),
    )

    direction = np.where(
        same_positive,
        1.0,
        np.where(
            same_negative,
            -1.0,
            0.0,
        ),
    )

    weights = (
        direction
        * strength.to_numpy()
    )

    weights = pd.Series(
        weights,
        index=GENES,
    )

    # Mixed-direction genes receive weight zero.
    weights.loc[
        ~consistent
    ] = 0.0

    signature = pd.DataFrame(
        {
            "canonical_pao1":
                GENES,

            train_studies[0]:
                effect_1.values,

            train_studies[1]:
                effect_2.values,

            "direction":
                direction,

            "weight":
                weights.values,

            "abs_weight":
                np.abs(
                    weights.values
                ),
        }
    )

    signature = signature.sort_values(
        "abs_weight",
        ascending=False,
    )

    return weights, signature


# ============================================================
# Helper: calculate a SINGLE-SAMPLE score
#
# rank = 0 ... 1
#
# subtract 0.5 so:
# highly expressed = positive
# lowly expressed  = negative
#
# positive weights correspond to biofilm-high genes
# negative weights correspond to biofilm-low genes.
#
# Divide by total absolute weight to keep the scale stable.
# ============================================================

def score_samples(
    rank_matrix,
    weights,
):

    total_weight = np.sum(
        np.abs(
            weights.values
        )
    )

    if total_weight == 0:
        raise RuntimeError(
            "No signature genes learned."
        )

    centred_rank = (
        rank_matrix
        - 0.5
    )

    scores = (
        centred_rank
        .mul(
            weights,
            axis=1,
        )
        .sum(axis=1)
        / total_weight
    )

    return scores


# ============================================================
# Leave-one-study-out
# ============================================================

studies = list(
    metadata["study"].unique()
)

prediction_records = []
fold_records = []


for held_out in studies:

    print()
    print("=" * 100)
    print(
        f"HELD-OUT STUDY: {held_out}"
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
    # Learn signature ONLY from training data
    # ----------------------------------------

    weights, signature = (
        learn_signature(
            train_meta,
            train_rank,
        )
    )


    nonzero = (
        weights != 0
    )

    n_signature = int(
        nonzero.sum()
    )

    n_up = int(
        (
            weights > 0
        ).sum()
    )

    n_down = int(
        (
            weights < 0
        ).sum()
    )


    print(
        f"Training studies: "
        f"{', '.join(sorted(train_meta['study'].unique()))}"
    )

    print(
        f"Fold-specific signature genes: "
        f"{n_signature}"
    )

    print(
        f"  biofilm-high: {n_up}"
    )

    print(
        f"  biofilm-low:  {n_down}"
    )


    signature.to_csv(
        OUT_DIR
        / f"signature_training_without_{held_out}.csv",
        index=False,
    )


    # ----------------------------------------
    # Training scores
    # ----------------------------------------

    train_scores = score_samples(
        train_rank,
        weights,
    )


    # Learn classification threshold
    # using training data ONLY.
    #
    # Midpoint between mean class scores.
    # ----------------------------------------

    train_bio_mean = float(
        train_scores[
            train_meta["state"]
            == "biofilm"
        ].mean()
    )

    train_plank_mean = float(
        train_scores[
            train_meta["state"]
            == "planktonic"
        ].mean()
    )


    # Orient score so larger always means
    # "more biofilm-like".
    orientation = 1.0

    if (
        train_bio_mean
        < train_plank_mean
    ):

        orientation = -1.0

        train_scores = (
            train_scores
            * orientation
        )

        train_bio_mean = float(
            train_scores[
                train_meta["state"]
                == "biofilm"
            ].mean()
        )

        train_plank_mean = float(
            train_scores[
                train_meta["state"]
                == "planktonic"
            ].mean()
        )


    threshold = (
        train_bio_mean
        + train_plank_mean
    ) / 2.0


    # ----------------------------------------
    # Completely unseen held-out study
    # ----------------------------------------

    test_scores = score_samples(
        test_rank,
        weights,
    )

    test_scores = (
        test_scores
        * orientation
    )


    y_true = (
        test_meta["state"]
        == "biofilm"
    ).astype(int)


    y_pred = (
        test_scores
        >= threshold
    ).astype(int)


    auc = roc_auc_score(
        y_true,
        test_scores,
    )

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )


    print()
    print(
        f"Training biofilm mean score:    "
        f"{train_bio_mean:.6f}"
    )

    print(
        f"Training planktonic mean score: "
        f"{train_plank_mean:.6f}"
    )

    print(
        f"Training-derived threshold:     "
        f"{threshold:.6f}"
    )

    print()

    test_output = test_meta.copy()

    test_output[
        "biofilm_score"
    ] = test_scores.values

    test_output[
        "predicted_state"
    ] = np.where(
        y_pred == 1,
        "biofilm",
        "planktonic",
    )

    test_output[
        "correct"
    ] = (
        test_output[
            "predicted_state"
        ]
        ==
        test_output[
            "state"
        ]
    )


    print(
        test_output[
            [
                "sample",
                "state",
                "biofilm_score",
                "predicted_state",
                "correct",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        f"Held-out ROC AUC:  "
        f"{auc:.3f}"
    )

    print(
        f"Held-out accuracy: "
        f"{accuracy:.3f}"
    )


    fold_records.append(
        {
            "held_out_study":
                held_out,

            "n_signature_genes":
                n_signature,

            "n_biofilm_high":
                n_up,

            "n_biofilm_low":
                n_down,

            "training_biofilm_mean":
                train_bio_mean,

            "training_planktonic_mean":
                train_plank_mean,

            "threshold":
                threshold,

            "held_out_auc":
                auc,

            "held_out_accuracy":
                accuracy,
        }
    )


    for idx in test_output.index:

        prediction_records.append(
            {
                "held_out_study":
                    held_out,

                "study":
                    test_output.loc[
                        idx,
                        "study",
                    ],

                "sample":
                    test_output.loc[
                        idx,
                        "sample",
                    ],

                "state":
                    test_output.loc[
                        idx,
                        "state",
                    ],

                "biofilm_score":
                    test_output.loc[
                        idx,
                        "biofilm_score",
                    ],

                "predicted_state":
                    test_output.loc[
                        idx,
                        "predicted_state",
                    ],

                "correct":
                    test_output.loc[
                        idx,
                        "correct",
                    ],
            }
        )


# ============================================================
# Overall results
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
    predictions["predicted_state"]
    == "biofilm"
).astype(int)


overall_accuracy = accuracy_score(
    overall_true,
    overall_pred,
)


print()
print("=" * 100)
print("LEAVE-ONE-STUDY-OUT SUMMARY")
print("=" * 100)

print(
    fold_summary[
        [
            "held_out_study",
            "n_signature_genes",
            "held_out_auc",
            "held_out_accuracy",
        ]
    ].to_string(
        index=False
    )
)

print()

print(
    f"Overall held-out accuracy: "
    f"{overall_accuracy:.3f}"
)

print()

print(
    "NOTE:"
)

print(
    "Every held-out prediction was generated "
    "using a signature learned without that study."
)

print()
