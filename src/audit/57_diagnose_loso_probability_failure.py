
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


PREDICTION_FILE = Path(
    "results/audit/cross_study_modeling/"
    "Audit56_nested_loso/"
    "Audit56_outer_loso_predictions.csv"
)

OUT_DIR = Path(
    "results/audit/cross_study_modeling/"
    "Audit57_probability_diagnosis"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


DISCOVERY_STUDIES = {
    "GSE136111",
    "GSE203348",
    "GSE87213",
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 57: "
    "diagnose out-of-study discrimination vs calibration"
)
print("=" * 100)


# ============================================================
# Load predictions
# ============================================================

pred = pd.read_csv(
    PREDICTION_FILE
)


required_columns = [
    "study",
    "sample",
    "state",
    "label",
    "model",
    "held_out_study",
    "selected_C",
    "probability_biofilm",
    "predicted_label",
]


for column in required_columns:

    if column not in pred.columns:

        raise RuntimeError(
            "Missing prediction column: "
            + column
        )


if len(pred) != 80:

    raise RuntimeError(
        "Expected 80 rows: 40 samples x 2 models."
    )


if set(
    pred[
        "model"
    ].unique()
) != {
    "Frozen250",
    "Ridge4853",
}:

    raise RuntimeError(
        "Unexpected model names."
    )


if not (
    pred[
        "study"
    ]
    ==
    pred[
        "held_out_study"
    ]
).all():

    raise RuntimeError(
        "Study and held_out_study do not match."
    )


# ============================================================
# Evaluation-status annotation
# ============================================================

def evaluation_status(
    model,
    study,
):

    if model == "Ridge4853":

        return "UNBIASED_OUTER_LOSO"

    if study in DISCOVERY_STUDIES:

        return "SIGNATURE_DISCOVERY_LEAKAGE"

    return "EXTERNAL_SIGNATURE_TEST"


pred[
    "evaluation_status"
] = [
    evaluation_status(
        model,
        study,
    )
    for model, study in zip(
        pred[
            "model"
        ],
        pred[
            "study"
        ],
    )
]


# ============================================================
# Study-level probability diagnosis
# ============================================================

study_records = []


for model_name in [
    "Frozen250",
    "Ridge4853",
]:

    model_df = pred.loc[
        pred[
            "model"
        ]
        == model_name
    ].copy()


    for study in sorted(
        model_df[
            "study"
        ].unique()
    ):

        subset = model_df.loc[
            model_df[
                "study"
            ]
            == study
        ].copy()


        plank = subset.loc[
            subset[
                "label"
            ]
            == 0,
            "probability_biofilm",
        ].to_numpy(
            dtype=float
        )


        bio = subset.loc[
            subset[
                "label"
            ]
            == 1,
            "probability_biofilm",
        ].to_numpy(
            dtype=float
        )


        if len(plank) == 0 or len(bio) == 0:

            raise RuntimeError(
                "Missing one class in "
                + study
                + " / "
                + model_name
            )


        y = subset[
            "label"
        ].to_numpy(
            dtype=int
        )


        p = subset[
            "probability_biofilm"
        ].to_numpy(
            dtype=float
        )


        auc = float(
            roc_auc_score(
                y,
                p,
            )
        )


        prediction = (
            p
            >= 0.5
        ).astype(
            int
        )


        accuracy = float(
            accuracy_score(
                y,
                prediction,
            )
        )


        plank_mean = float(
            np.mean(
                plank
            )
        )

        plank_min = float(
            np.min(
                plank
            )
        )

        plank_max = float(
            np.max(
                plank
            )
        )


        bio_mean = float(
            np.mean(
                bio
            )
        )

        bio_min = float(
            np.min(
                bio
            )
        )

        bio_max = float(
            np.max(
                bio
            )
        )


        difference = (
            bio_mean
            - plank_mean
        )


        study_mean_probability = float(
            np.mean(
                p
            )
        )


        if plank_max < bio_min:

            separation_type = (
                "perfect_correct_direction"
            )

        elif bio_max < plank_min:

            separation_type = (
                "perfect_reversed_direction"
            )

        else:

            separation_type = (
                "overlapping_ranges"
            )


        if (
            np.all(
                p
                < 0.5
            )
        ):

            threshold_pattern = (
                "all_below_0.5"
            )

        elif (
            np.all(
                p
                >= 0.5
            )
        ):

            threshold_pattern = (
                "all_at_or_above_0.5"
            )

        else:

            threshold_pattern = (
                "straddles_0.5"
            )


        study_records.append(
            {
                "model":
                    model_name,

                "study":
                    study,

                "evaluation_status":
                    evaluation_status(
                        model_name,
                        study,
                    ),

                "n_samples":
                    len(
                        subset
                    ),

                "n_planktonic":
                    len(
                        plank
                    ),

                "n_biofilm":
                    len(
                        bio
                    ),

                "study_mean_probability":
                    study_mean_probability,

                "planktonic_mean_probability":
                    plank_mean,

                "planktonic_min_probability":
                    plank_min,

                "planktonic_max_probability":
                    plank_max,

                "biofilm_mean_probability":
                    bio_mean,

                "biofilm_min_probability":
                    bio_min,

                "biofilm_max_probability":
                    bio_max,

                "bio_minus_planktonic_mean":
                    difference,

                "auc":
                    auc,

                "accuracy_at_0.5":
                    accuracy,

                "separation_type":
                    separation_type,

                "threshold_pattern":
                    threshold_pattern,
            }
        )


study_diag = pd.DataFrame(
    study_records
)


study_diag.to_csv(
    OUT_DIR
    / "Audit57_study_probability_diagnostics.csv",
    index=False,
)


# ============================================================
# Print actual probabilities
# ============================================================

print()
print("=" * 100)
print("RIDGE4853 — ACTUAL HELD-OUT PROBABILITIES")
print("=" * 100)


ridge = pred.loc[
    pred[
        "model"
    ]
    == "Ridge4853"
].copy()


for study in sorted(
    ridge[
        "study"
    ].unique()
):

    subset = ridge.loc[
        ridge[
            "study"
        ]
        == study
    ].copy()


    subset = subset.sort_values(
        [
            "label",
            "probability_biofilm",
        ]
    )


    print()
    print(
        study
    )

    print(
        "-" * 80
    )


    for _, row in subset.iterrows():

        print(
            str(
                row[
                    "state"
                ]
            ).ljust(
                12
            )
            + "  "
            + str(
                row[
                    "sample"
                ]
            ).ljust(
                25
            )
            + "  P(biofilm)="
            + format(
                float(
                    row[
                        "probability_biofilm"
                    ]
                ),
                ".6f",
            )
        )


# ============================================================
# Ridge study-level diagnosis
# ============================================================

ridge_diag = study_diag.loc[
    study_diag[
        "model"
    ]
    == "Ridge4853"
].copy()


print()
print("=" * 100)
print("RIDGE4853 — STUDY-LEVEL DIAGNOSIS")
print("=" * 100)


ridge_display_columns = [
    "study",
    "study_mean_probability",
    "planktonic_mean_probability",
    "biofilm_mean_probability",
    "bio_minus_planktonic_mean",
    "auc",
    "accuracy_at_0.5",
    "separation_type",
    "threshold_pattern",
]


print(
    ridge_diag[
        ridge_display_columns
    ]
    .to_string(
        index=False
    )
)


# ============================================================
# Quantify study-offset variation
# ============================================================

offset_records = []


for model_name in [
    "Frozen250",
    "Ridge4853",
]:

    subset = study_diag.loc[
        study_diag[
            "model"
        ]
        == model_name
    ].copy()


    offset_records.append(
        {
            "model":
                model_name,

            "mean_study_probability":
                float(
                    subset[
                        "study_mean_probability"
                    ].mean()
                ),

            "sd_study_mean_probability":
                float(
                    subset[
                        "study_mean_probability"
                    ].std(
                        ddof=1
                    )
                ),

            "min_study_mean_probability":
                float(
                    subset[
                        "study_mean_probability"
                    ].min()
                ),

            "max_study_mean_probability":
                float(
                    subset[
                        "study_mean_probability"
                    ].max()
                ),

            "mean_bio_minus_planktonic":
                float(
                    subset[
                        "bio_minus_planktonic_mean"
                    ].mean()
                ),

            "n_correct_direction":
                int(
                    (
                        subset[
                            "bio_minus_planktonic_mean"
                        ]
                        > 0
                    ).sum()
                ),

            "n_reversed_direction":
                int(
                    (
                        subset[
                            "bio_minus_planktonic_mean"
                        ]
                        < 0
                    ).sum()
                ),
        }
    )


offset_summary = pd.DataFrame(
    offset_records
)


offset_summary.to_csv(
    OUT_DIR
    / "Audit57_model_offset_summary.csv",
    index=False,
)


print()
print("=" * 100)
print("BETWEEN-STUDY PROBABILITY OFFSET SUMMARY")
print("=" * 100)

print(
    offset_summary.to_string(
        index=False
    )
)


# ============================================================
# Pairwise concordance:
#
# How often does a biofilm sample receive a higher probability
# than a planktonic sample?
#
# We calculate this separately for:
#   1. pairs from the same held-out study
#   2. pairs from different studies
#
# This is particularly useful for understanding why the
# within-study AUC can be high while pooled AUC is poor.
# ============================================================

def pairwise_concordance(
    frame,
    same_study,
):

    positives = frame.loc[
        frame[
            "label"
        ]
        == 1
    ]


    negatives = frame.loc[
        frame[
            "label"
        ]
        == 0
    ]


    n_pairs = 0
    concordant = 0.0


    for _, positive in positives.iterrows():

        for _, negative in negatives.iterrows():

            pair_is_same_study = (
                positive[
                    "study"
                ]
                ==
                negative[
                    "study"
                ]
            )


            if pair_is_same_study != same_study:

                continue


            n_pairs += 1


            p_pos = float(
                positive[
                    "probability_biofilm"
                ]
            )

            p_neg = float(
                negative[
                    "probability_biofilm"
                ]
            )


            if p_pos > p_neg:

                concordant += 1.0

            elif p_pos == p_neg:

                concordant += 0.5


    if n_pairs == 0:

        return np.nan, 0


    return (
        float(
            concordant
            / n_pairs
        ),
        n_pairs,
    )


pair_records = []


for model_name in [
    "Frozen250",
    "Ridge4853",
]:

    subset = pred.loc[
        pred[
            "model"
        ]
        == model_name
    ].copy()


    within_value, within_n = pairwise_concordance(
        subset,
        same_study=True,
    )


    cross_value, cross_n = pairwise_concordance(
        subset,
        same_study=False,
    )


    pooled_auc = float(
        roc_auc_score(
            subset[
                "label"
            ],
            subset[
                "probability_biofilm"
            ],
        )
    )


    pair_records.append(
        {
            "model":
                model_name,

            "within_study_concordance":
                within_value,

            "within_study_pairs":
                within_n,

            "cross_study_concordance":
                cross_value,

            "cross_study_pairs":
                cross_n,

            "pooled_auc":
                pooled_auc,
        }
    )


pair_df = pd.DataFrame(
    pair_records
)


pair_df.to_csv(
    OUT_DIR
    / "Audit57_within_vs_cross_study_concordance.csv",
    index=False,
)


print()
print("=" * 100)
print("WITHIN-STUDY VS CROSS-STUDY CONCORDANCE")
print("=" * 100)

print(
    pair_df.to_string(
        index=False
    )
)


# ============================================================
# UNLABELLED STUDY-CENTERING DIAGNOSTIC
#
# IMPORTANT:
#
# This is NOT a deployable single-sample method.
#
# For each held-out study we subtract the mean logit of all
# samples in that held-out study. No labels are used, but the
# procedure requires a batch of samples from the new study.
#
# The purpose is only to test whether study-specific offsets
# explain the calibration failure.
# ============================================================

centered_records = []


for model_name in [
    "Frozen250",
    "Ridge4853",
]:

    model_df = pred.loc[
        pred[
            "model"
        ]
        == model_name
    ].copy()


    centered_parts = []


    for study in sorted(
        model_df[
            "study"
        ].unique()
    ):

        subset = model_df.loc[
            model_df[
                "study"
            ]
            == study
        ].copy()


        probability = (
            subset[
                "probability_biofilm"
            ]
            .to_numpy(
                dtype=float
            )
        )


        probability = np.clip(
            probability,
            1e-6,
            1.0 - 1e-6,
        )


        logits = np.log(
            probability
            / (
                1.0
                - probability
            )
        )


        mean_logit = float(
            np.mean(
                logits
            )
        )


        centered_logit = (
            logits
            - mean_logit
        )


        centered_probability = (
            1.0
            / (
                1.0
                + np.exp(
                    -centered_logit
                )
            )
        )


        subset[
            "original_probability"
        ] = probability


        subset[
            "study_mean_logit"
        ] = mean_logit


        subset[
            "centered_logit"
        ] = centered_logit


        subset[
            "centered_probability"
        ] = centered_probability


        subset[
            "centered_prediction"
        ] = (
            centered_probability
            >= 0.5
        ).astype(
            int
        )


        centered_parts.append(
            subset
        )


    centered = pd.concat(
        centered_parts,
        ignore_index=True,
    )


    y = centered[
        "label"
    ].to_numpy(
        dtype=int
    )


    p = centered[
        "centered_probability"
    ].to_numpy(
        dtype=float
    )


    prediction = (
        p
        >= 0.5
    ).astype(
        int
    )


    metrics = {
        "model":
            model_name,

        "auc_after_unlabelled_study_centering":
            float(
                roc_auc_score(
                    y,
                    p,
                )
            ),

        "accuracy_after_unlabelled_study_centering":
            float(
                accuracy_score(
                    y,
                    prediction,
                )
            ),

        "brier_after_unlabelled_study_centering":
            float(
                brier_score_loss(
                    y,
                    p,
                )
            ),

        "log_loss_after_unlabelled_study_centering":
            float(
                log_loss(
                    y,
                    p,
                    labels=[
                        0,
                        1,
                    ],
                )
            ),
    }


    centered_records.append(
        metrics
    )


    centered.to_csv(
        OUT_DIR
        / (
            "Audit57_"
            + model_name
            + "_study_centered_DIAGNOSTIC_ONLY.csv"
        ),
        index=False,
    )


centered_summary = pd.DataFrame(
    centered_records
)


centered_summary.to_csv(
    OUT_DIR
    / "Audit57_study_centering_summary_DIAGNOSTIC_ONLY.csv",
    index=False,
)


print()
print("=" * 100)
print("UNLABELLED STUDY-CENTERING DIAGNOSTIC ONLY")
print("=" * 100)

print(
    centered_summary.to_string(
        index=False
    )
)


print()
print(
    "IMPORTANT:"
)

print(
    "Study-centering is NOT a valid single-sample "
    "prediction method."
)

print(
    "It is shown only to test whether study-specific "
    "probability offsets explain the failure."
)


# ============================================================
# External-only frozen-signature summary
#
# The original 3 discovery studies must not be treated as
# independent validation for Frozen250.
# ============================================================

external_frozen = pred.loc[
    (
        pred[
            "model"
        ]
        == "Frozen250"
    )
    &
    (
        ~pred[
            "study"
        ].isin(
            DISCOVERY_STUDIES
        )
    )
].copy()


external_auc = float(
    roc_auc_score(
        external_frozen[
            "label"
        ],
        external_frozen[
            "probability_biofilm"
        ],
    )
)


external_brier = float(
    brier_score_loss(
        external_frozen[
            "label"
        ],
        external_frozen[
            "probability_biofilm"
        ],
    )
)


external_log_loss = float(
    log_loss(
        external_frozen[
            "label"
        ],
        external_frozen[
            "probability_biofilm"
        ],
        labels=[
            0,
            1,
        ],
    )
)


print()
print("=" * 100)
print("FROZEN250 — FAIR EXTERNAL STUDIES ONLY")
print("=" * 100)

print(
    "Studies: "
    + ", ".join(
        sorted(
            external_frozen[
                "study"
            ].unique()
        )
    )
)

print(
    "Samples: "
    + str(
        len(
            external_frozen
        )
    )
)

print(
    "Pooled external AUC: "
    + format(
        external_auc,
        ".6f",
    )
)

print(
    "Pooled external Brier: "
    + format(
        external_brier,
        ".6f",
    )
)

print(
    "Pooled external log loss: "
    + format(
        external_log_loss,
        ".6f",
    )
)


# ============================================================
# Sorted probability table
# ============================================================

sorted_predictions = pred.sort_values(
    [
        "model",
        "study",
        "label",
        "probability_biofilm",
    ]
)


sorted_predictions.to_csv(
    OUT_DIR
    / "Audit57_all_sorted_predictions.csv",
    index=False,
)


print()
print("=" * 100)
print("AUDIT 57 COMPLETE")
print("=" * 100)

print(
    "No models were refitted or retuned."
)

print(
    "This audit only diagnoses the out-of-study "
    "predictions generated in Audit 56."
)

print()
