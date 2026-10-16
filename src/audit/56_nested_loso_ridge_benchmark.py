from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)


DATA_FILE = Path(
    "data/processed/cross_study_modeling/"
    "CFSTATE_binary_7studies_40samples_4853genes_ranked.csv"
)

SIGNATURE_FILE = Path(
    "results/audit/core_biofilm_signature/"
    "core_biofilm_signature_fdr05.csv"
)

OUT_DIR = Path(
    "results/audit/cross_study_modeling/"
    "Audit56_nested_loso"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


C_GRID = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]


META_COLUMNS = [
    "study",
    "sample",
    "state",
    "label",
    "source_condition",
]


print()
print("=" * 100)
print(
    "CF-STATE — Audit 56: "
    "nested leave-one-study-out probability benchmark"
)
print("=" * 100)


# ============================================================
# Load data
# ============================================================

data = pd.read_csv(
    DATA_FILE
)


for column in META_COLUMNS:

    if column not in data.columns:

        raise RuntimeError(
            "Missing metadata column: "
            + column
        )


gene_columns = [
    column
    for column in data.columns
    if column not in META_COLUMNS
]


if len(gene_columns) != 4853:

    raise RuntimeError(
        "Expected 4,853 genes, found "
        + str(len(gene_columns))
    )


if len(data) != 40:

    raise RuntimeError(
        "Expected 40 binary samples."
    )


if data["study"].nunique() != 7:

    raise RuntimeError(
        "Expected 7 study families."
    )


if int(data["label"].sum()) != 20:

    raise RuntimeError(
        "Expected 20 biofilm samples."
    )


# ============================================================
# Frozen signature baseline
# ============================================================

signature = pd.read_csv(
    SIGNATURE_FILE
)


if "canonical_pao1" not in signature.columns:

    raise RuntimeError(
        "Signature file missing canonical_pao1."
    )


if "median_effect" not in signature.columns:

    raise RuntimeError(
        "Signature file missing median_effect."
    )


signature = signature[
    [
        "canonical_pao1",
        "median_effect",
    ]
].copy()


signature[
    "canonical_pao1"
] = (
    signature[
        "canonical_pao1"
    ]
    .astype(str)
)


available_signature = (
    signature.loc[
        signature[
            "canonical_pao1"
        ].isin(
            gene_columns
        )
    ]
    .copy()
)


missing_signature = sorted(
    set(
        signature[
            "canonical_pao1"
        ]
    )
    - set(
        gene_columns
    )
)


print()
print("=" * 100)
print("MODEL INPUT")
print("=" * 100)

print(
    "Samples: "
    + str(len(data))
)

print(
    "Studies: "
    + str(
        data["study"].nunique()
    )
)

print(
    "Whole-transcriptome genes: "
    + str(len(gene_columns))
)

print(
    "Available frozen-signature genes: "
    + str(
        len(available_signature)
    )
    + " / "
    + str(len(signature))
)

print(
    "Unavailable signature genes: "
    + (
        ", ".join(missing_signature)
        if missing_signature
        else "None"
    )
)


up_genes = (
    available_signature.loc[
        available_signature[
            "median_effect"
        ]
        > 0,
        "canonical_pao1",
    ]
    .tolist()
)


down_genes = (
    available_signature.loc[
        available_signature[
            "median_effect"
        ]
        < 0,
        "canonical_pao1",
    ]
    .tolist()
)


data[
    "frozen_score"
] = (
    data[
        up_genes
    ]
    .mean(
        axis=1
    )
    -
    data[
        down_genes
    ]
    .mean(
        axis=1
    )
)


# ============================================================
# Helpers
# ============================================================

def fit_logistic(
    X,
    y,
    C,
):

    model = LogisticRegression(
        penalty="l2",
        C=C,
        solver="liblinear",
        max_iter=5000,
        random_state=1,
    )

    model.fit(
        X,
        y,
    )

    return model


def choose_c_nested(
    train_df,
    feature_columns,
):

    training_studies = sorted(
        train_df[
            "study"
        ].unique()
    )


    records = []


    for C in C_GRID:

        fold_losses = []


        for inner_test_study in training_studies:

            inner_train = train_df.loc[
                train_df[
                    "study"
                ]
                != inner_test_study
            ]


            inner_test = train_df.loc[
                train_df[
                    "study"
                ]
                == inner_test_study
            ]


            X_train = (
                inner_train[
                    feature_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )


            y_train = (
                inner_train[
                    "label"
                ]
                .to_numpy(
                    dtype=int
                )
            )


            X_test = (
                inner_test[
                    feature_columns
                ]
                .to_numpy(
                    dtype=float
                )
            )


            y_test = (
                inner_test[
                    "label"
                ]
                .to_numpy(
                    dtype=int
                )
            )


            model = fit_logistic(
                X_train,
                y_train,
                C,
            )


            probability = (
                model.predict_proba(
                    X_test
                )[
                    :,
                    1
                ]
            )


            loss = log_loss(
                y_test,
                probability,
                labels=[
                    0,
                    1,
                ],
            )


            fold_losses.append(
                float(loss)
            )


        mean_loss = float(
            np.mean(
                fold_losses
            )
        )


        records.append(
            {
                "C":
                    C,

                "mean_inner_study_log_loss":
                    mean_loss,

                "min_inner_study_log_loss":
                    float(
                        np.min(
                            fold_losses
                        )
                    ),

                "max_inner_study_log_loss":
                    float(
                        np.max(
                            fold_losses
                        )
                    ),
            }
        )


    result = pd.DataFrame(
        records
    )


    result = result.sort_values(
        [
            "mean_inner_study_log_loss",
            "C",
        ],
        ascending=[
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )


    best_c = float(
        result.loc[
            0,
            "C",
        ]
    )


    return best_c, result


def calculate_metrics(
    y_true,
    probability,
):

    prediction = (
        probability
        >= 0.5
    ).astype(int)


    auc = float(
        roc_auc_score(
            y_true,
            probability,
        )
    )


    accuracy = float(
        accuracy_score(
            y_true,
            prediction,
        )
    )


    brier = float(
        brier_score_loss(
            y_true,
            probability,
        )
    )


    loss = float(
        log_loss(
            y_true,
            probability,
            labels=[
                0,
                1,
            ],
        )
    )


    positives = (
        y_true
        == 1
    )


    negatives = (
        y_true
        == 0
    )


    sensitivity = float(
        prediction[
            positives
        ].mean()
    )


    specificity = float(
        (
            prediction[
                negatives
            ]
            == 0
        ).mean()
    )


    return {
        "auc":
            auc,

        "accuracy":
            accuracy,

        "sensitivity":
            sensitivity,

        "specificity":
            specificity,

        "brier":
            brier,

        "log_loss":
            loss,
    }


# ============================================================
# Outer leave-one-study-out evaluation
# ============================================================

studies = sorted(
    data[
        "study"
    ].unique()
)


prediction_records = []
metric_records = []
coefficient_records = []


models = {
    "Frozen250":
        [
            "frozen_score"
        ],

    "Ridge4853":
        gene_columns,
}


for outer_test_study in studies:

    print()
    print("=" * 100)
    print(
        "OUTER HELD-OUT STUDY: "
        + outer_test_study
    )
    print("=" * 100)


    outer_train = data.loc[
        data[
            "study"
        ]
        != outer_test_study
    ].copy()


    outer_test = data.loc[
        data[
            "study"
        ]
        == outer_test_study
    ].copy()


    print(
        "Training samples: "
        + str(len(outer_train))
    )

    print(
        "Training studies: "
        + str(
            outer_train[
                "study"
            ].nunique()
        )
    )

    print(
        "Held-out samples: "
        + str(len(outer_test))
    )


    for model_name, features in models.items():

        print()
        print(
            model_name
        )


        best_c, tuning = choose_c_nested(
            outer_train,
            features,
        )


        tuning[
            "outer_test_study"
        ] = outer_test_study


        tuning[
            "model"
        ] = model_name


        tuning.to_csv(
            OUT_DIR
            / (
                "inner_tuning_"
                + model_name
                + "_heldout_"
                + outer_test_study
                + ".csv"
            ),
            index=False,
        )


        print(
            "Selected C: "
            + str(best_c)
        )


        X_train = (
            outer_train[
                features
            ]
            .to_numpy(
                dtype=float
            )
        )


        y_train = (
            outer_train[
                "label"
            ]
            .to_numpy(
                dtype=int
            )
        )


        X_test = (
            outer_test[
                features
            ]
            .to_numpy(
                dtype=float
            )
        )


        y_test = (
            outer_test[
                "label"
            ]
            .to_numpy(
                dtype=int
            )
        )


        model = fit_logistic(
            X_train,
            y_train,
            best_c,
        )


        probability = (
            model.predict_proba(
                X_test
            )[
                :,
                1
            ]
        )


        metrics = calculate_metrics(
            y_test,
            probability,
        )


        print(
            "AUC: "
            + format(
                metrics[
                    "auc"
                ],
                ".3f",
            )
        )

        print(
            "Accuracy: "
            + format(
                metrics[
                    "accuracy"
                ],
                ".3f",
            )
        )

        print(
            "Brier: "
            + format(
                metrics[
                    "brier"
                ],
                ".4f",
            )
        )

        print(
            "Log loss: "
            + format(
                metrics[
                    "log_loss"
                ],
                ".4f",
            )
        )


        metric_record = {
            "held_out_study":
                outer_test_study,

            "model":
                model_name,

            "selected_C":
                best_c,
        }


        metric_record.update(
            metrics
        )


        metric_records.append(
            metric_record
        )


        for row_index, probability_value in zip(
            outer_test.index,
            probability,
        ):

            prediction_records.append(
                {
                    "study":
                        data.loc[
                            row_index,
                            "study",
                        ],

                    "sample":
                        data.loc[
                            row_index,
                            "sample",
                        ],

                    "state":
                        data.loc[
                            row_index,
                            "state",
                        ],

                    "label":
                        int(
                            data.loc[
                                row_index,
                                "label",
                            ]
                        ),

                    "model":
                        model_name,

                    "held_out_study":
                        outer_test_study,

                    "selected_C":
                        best_c,

                    "probability_biofilm":
                        float(
                            probability_value
                        ),

                    "predicted_label":
                        int(
                            probability_value
                            >= 0.5
                        ),
                }
            )


        coefficients = (
            model.coef_[
                0
            ]
        )


        for feature, coefficient in zip(
            features,
            coefficients,
        ):

            coefficient_records.append(
                {
                    "held_out_study":
                        outer_test_study,

                    "model":
                        model_name,

                    "feature":
                        feature,

                    "coefficient":
                        float(
                            coefficient
                        ),
                }
            )


# ============================================================
# Save outer predictions
# ============================================================

predictions = pd.DataFrame(
    prediction_records
)


metrics = pd.DataFrame(
    metric_records
)


coefficients = pd.DataFrame(
    coefficient_records
)


predictions.to_csv(
    OUT_DIR
    / "Audit56_outer_loso_predictions.csv",
    index=False,
)


metrics.to_csv(
    OUT_DIR
    / "Audit56_per_study_metrics.csv",
    index=False,
)


coefficients.to_csv(
    OUT_DIR
    / "Audit56_outer_model_coefficients.csv",
    index=False,
)


# ============================================================
# Pooled out-of-study metrics
# ============================================================

pooled_records = []


for model_name in models:

    subset = predictions.loc[
        predictions[
            "model"
        ]
        == model_name
    ].copy()


    if len(subset) != 40:

        raise RuntimeError(
            model_name
            + " does not have 40 outer predictions."
        )


    y_true = (
        subset[
            "label"
        ]
        .to_numpy(
            dtype=int
        )
    )


    probability = (
        subset[
            "probability_biofilm"
        ]
        .to_numpy(
            dtype=float
        )
    )


    pooled_metrics = calculate_metrics(
        y_true,
        probability,
    )


    pooled_record = {
        "model":
            model_name,
    }


    pooled_record.update(
        pooled_metrics
    )


    pooled_records.append(
        pooled_record
    )


pooled = pd.DataFrame(
    pooled_records
)


pooled.to_csv(
    OUT_DIR
    / "Audit56_pooled_outer_metrics.csv",
    index=False,
)


# ============================================================
# Study-macro metrics
# ============================================================

macro_records = []


for model_name in models:

    subset = metrics.loc[
        metrics[
            "model"
        ]
        == model_name
    ]


    macro_records.append(
        {
            "model":
                model_name,

            "mean_study_auc":
                float(
                    subset[
                        "auc"
                    ].mean()
                ),

            "median_study_auc":
                float(
                    subset[
                        "auc"
                    ].median()
                ),

            "mean_study_accuracy":
                float(
                    subset[
                        "accuracy"
                    ].mean()
                ),

            "mean_study_brier":
                float(
                    subset[
                        "brier"
                    ].mean()
                ),

            "mean_study_log_loss":
                float(
                    subset[
                        "log_loss"
                    ].mean()
                ),

            "worst_study_auc":
                float(
                    subset[
                        "auc"
                    ].min()
                ),

            "worst_study_log_loss":
                float(
                    subset[
                        "log_loss"
                    ].max()
                ),
        }
    )


macro = pd.DataFrame(
    macro_records
)


macro.to_csv(
    OUT_DIR
    / "Audit56_macro_study_metrics.csv",
    index=False,
)


# ============================================================
# Constant 0.5 probability reference
# ============================================================

constant_probability = np.full(
    len(data),
    0.5,
)


constant_metrics = calculate_metrics(
    data[
        "label"
    ].to_numpy(
        dtype=int
    ),
    constant_probability,
)


# ============================================================
# Print final results
# ============================================================

print()
print("=" * 100)
print("PER-STUDY OUTER RESULTS")
print("=" * 100)

display_columns = [
    "held_out_study",
    "model",
    "selected_C",
    "auc",
    "accuracy",
    "sensitivity",
    "specificity",
    "brier",
    "log_loss",
]


print(
    metrics[
        display_columns
    ]
    .sort_values(
        [
            "held_out_study",
            "model",
        ]
    )
    .to_string(
        index=False
    )
)


print()
print("=" * 100)
print("POOLED OUT-OF-STUDY RESULTS")
print("=" * 100)

print(
    pooled.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("STUDY-MACRO RESULTS")
print("=" * 100)

print(
    macro.to_string(
        index=False
    )
)


print()
print("=" * 100)
print("CONSTANT 0.5 REFERENCE")
print("=" * 100)

for key, value in constant_metrics.items():

    print(
        key
        + ": "
        + format(
            value,
            ".6f",
        )
    )


print()
print("=" * 100)
print("AUDIT 56 COMPLETE")
print("=" * 100)

print(
    "Every prediction above was generated by a model "
    "that never saw any sample from that study."
)

print()

print(
    "Hyperparameter C was also selected using only "
    "whole-study holdouts inside the training studies."
)

print()
