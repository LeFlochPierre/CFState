from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DATA_FILE = Path(
    "data/processed/biofilm/"
    "all_studies_harmonised_unscaled.csv"
)

PAIR_DIR = Path(
    "results/audit/loso_rank_pair_classifier"
)

OUT_DIR = Path(
    "results/audit/signature_transferability"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Settings
# ============================================================

# Same threshold used when selecting candidate genes
# in Audit 20.
MIN_TRAIN_EFFECT = 0.05

# Effect considered substantial in the held-out study.
MIN_TEST_EFFECT = 0.05

TOP_N = 30


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


# ============================================================
# Within-sample percentile ranks
# ============================================================

X_rank = X.rank(
    axis=1,
    method="average",
    pct=True,
)


print()
print("=" * 100)
print(
    "CF-STATE — Audit 21: "
    "cross-study signature transferability"
)
print("=" * 100)

print(f"Samples: {len(df)}")
print(f"Genes:   {len(GENES)}")


# ============================================================
# Helper: calculate biofilm - planktonic gene effects
# for one study
# ============================================================

def calculate_gene_effect(
    study,
):

    study_mask = (
        metadata["study"]
        == study
    )

    bio_mask = (
        study_mask
        &
        (
            metadata["state"]
            == "biofilm"
        )
    )

    plank_mask = (
        study_mask
        &
        (
            metadata["state"]
            == "planktonic"
        )
    )

    bio_mean = (
        X_rank.loc[
            bio_mask
        ]
        .mean(axis=0)
    )

    plank_mean = (
        X_rank.loc[
            plank_mask
        ]
        .mean(axis=0)
    )

    return (
        bio_mean
        - plank_mean
    )


# Precompute study effects
studies = list(
    metadata["study"].unique()
)

study_effects = {
    study:
        calculate_gene_effect(study)
    for study in studies
}


# ============================================================
# Helper: Audit-20 pair effect in held-out study
#
# Effect =
# P(high_gene > low_gene | biofilm)
# -
# P(high_gene > low_gene | planktonic)
#
# Positive = pair transfers correctly
# Zero     = pair carries no state information
# Negative = pair reverses
# ============================================================

def pair_effect(
    study,
    high_gene,
    low_gene,
):

    study_mask = (
        metadata["study"]
        == study
    )

    bio_idx = metadata.index[
        study_mask
        &
        (
            metadata["state"]
            == "biofilm"
        )
    ]

    plank_idx = metadata.index[
        study_mask
        &
        (
            metadata["state"]
            == "planktonic"
        )
    ]


    def proportion(
        idx,
    ):

        high = (
            X_rank.loc[
                idx,
                high_gene,
            ]
            .to_numpy()
        )

        low = (
            X_rank.loc[
                idx,
                low_gene,
            ]
            .to_numpy()
        )

        greater = (
            high > low
        ).astype(float)

        ties = (
            high == low
        ).astype(float)

        return float(
            np.mean(
                greater
                + 0.5 * ties
            )
        )


    return (
        proportion(bio_idx)
        -
        proportion(plank_idx)
    )


# ============================================================
# Fold-by-fold analysis
# ============================================================

fold_summary_records = []
gene_transfer_all = []
pair_transfer_all = []


for held_out in studies:

    train_studies = [
        study
        for study in studies
        if study != held_out
    ]

    t1 = train_studies[0]
    t2 = train_studies[1]


    print()
    print("=" * 100)
    print(
        f"HELD-OUT STUDY: {held_out}"
    )
    print("=" * 100)

    print(
        f"Training studies: "
        f"{t1}, {t2}"
    )


    # ========================================================
    # Training consensus gene effects
    # ========================================================

    e1 = study_effects[t1]
    e2 = study_effects[t2]

    test_effect = study_effects[
        held_out
    ]


    same_positive = (
        (e1 > 0)
        &
        (e2 > 0)
    )

    same_negative = (
        (e1 < 0)
        &
        (e2 < 0)
    )

    same_direction = (
        same_positive
        |
        same_negative
    )


    min_train_abs = np.minimum(
        np.abs(e1),
        np.abs(e2),
    )


    candidate_mask = (
        same_direction
        &
        (
            min_train_abs
            >= MIN_TRAIN_EFFECT
        )
    )


    training_direction = np.where(
        same_positive,
        1,
        np.where(
            same_negative,
            -1,
            0,
        ),
    )


    consensus_effect = (
        training_direction
        * min_train_abs
    )


    gene_table = pd.DataFrame(
        {
            "canonical_pao1":
                GENES,

            f"{t1}_effect":
                e1.values,

            f"{t2}_effect":
                e2.values,

            "training_consensus_effect":
                consensus_effect,

            "training_min_abs_effect":
                min_train_abs,

            "training_candidate":
                candidate_mask.values,

            f"{held_out}_effect":
                test_effect.values,
        }
    )


    gene_table[
        "training_direction"
    ] = np.where(
        gene_table[
            "training_consensus_effect"
        ] > 0,
        "higher_in_biofilm",
        np.where(
            gene_table[
                "training_consensus_effect"
            ] < 0,
            "lower_in_biofilm",
            "mixed",
        )
    )


    # ========================================================
    # Classify held-out behaviour
    # ========================================================

    def classify_transfer(
        row,
    ):

        if not row[
            "training_candidate"
        ]:
            return "not_candidate"


        train_effect = row[
            "training_consensus_effect"
        ]

        test = row[
            f"{held_out}_effect"
        ]


        if np.isclose(
            test,
            0.0,
        ):
            return "heldout_zero"


        same_sign = (
            np.sign(train_effect)
            ==
            np.sign(test)
        )

        strong = (
            abs(test)
            >= MIN_TEST_EFFECT
        )


        if same_sign and strong:
            return "same_direction_strong"

        if same_sign and not strong:
            return "same_direction_weak"

        if (
            not same_sign
            and strong
        ):
            return "reversed_strong"

        return "reversed_weak"


    gene_table[
        "transfer_category"
    ] = gene_table.apply(
        classify_transfer,
        axis=1,
    )


    candidates = gene_table[
        gene_table[
            "training_candidate"
        ]
    ].copy()


    # ========================================================
    # Gene-level transfer statistics
    # ========================================================

    counts = (
        candidates[
            "transfer_category"
        ]
        .value_counts()
    )


    n_candidates = len(
        candidates
    )

    n_same_strong = int(
        counts.get(
            "same_direction_strong",
            0,
        )
    )

    n_same_weak = int(
        counts.get(
            "same_direction_weak",
            0,
        )
    )

    n_reverse_strong = int(
        counts.get(
            "reversed_strong",
            0,
        )
    )

    n_reverse_weak = int(
        counts.get(
            "reversed_weak",
            0,
        )
    )

    n_zero = int(
        counts.get(
            "heldout_zero",
            0,
        )
    )


    same_direction_total = (
        n_same_strong
        + n_same_weak
    )

    reversed_total = (
        n_reverse_strong
        + n_reverse_weak
    )


    same_direction_fraction = (
        same_direction_total
        / n_candidates
        if n_candidates
        else np.nan
    )


    strong_transfer_fraction = (
        n_same_strong
        / n_candidates
        if n_candidates
        else np.nan
    )


    # Correlation across all genes
    rho_all, p_all = spearmanr(
        consensus_effect,
        test_effect,
    )


    # Correlation among training candidates only
    rho_candidates, p_candidates = (
        spearmanr(
            candidates[
                "training_consensus_effect"
            ],
            candidates[
                f"{held_out}_effect"
            ],
        )
    )


    print()
    print("GENE-LEVEL TRANSFER")
    print("-" * 100)

    print(
        f"Training-consistent genes "
        f"with |effect| >= "
        f"{MIN_TRAIN_EFFECT:.2f}: "
        f"{n_candidates}"
    )

    print()

    print(
        f"Same direction + strong held-out effect: "
        f"{n_same_strong}"
    )

    print(
        f"Same direction + weak held-out effect:   "
        f"{n_same_weak}"
    )

    print(
        f"Reversed + strong held-out effect:       "
        f"{n_reverse_strong}"
    )

    print(
        f"Reversed + weak held-out effect:         "
        f"{n_reverse_weak}"
    )

    print(
        f"Held-out effect exactly zero:            "
        f"{n_zero}"
    )

    print()

    print(
        f"Same-direction transfer fraction:        "
        f"{same_direction_fraction:.3f}"
    )

    print(
        f"Strong same-direction transfer fraction: "
        f"{strong_transfer_fraction:.3f}"
    )

    print()

    print(
        f"All-gene training-vs-held-out "
        f"Spearman rho: "
        f"{rho_all:.4f} "
        f"(p={p_all:.3e})"
    )

    print(
        f"Candidate-only Spearman rho: "
        f"{rho_candidates:.4f} "
        f"(p={p_candidates:.3e})"
    )


    # ========================================================
    # Strongest training genes
    # ========================================================

    top = candidates.copy()

    top[
        "training_strength"
    ] = np.abs(
        top[
            "training_consensus_effect"
        ]
    )

    top = top.sort_values(
        "training_strength",
        ascending=False,
    ).head(
        TOP_N
    )


    print()
    print(
        f"TOP {TOP_N} TRAINING SIGNATURE GENES"
    )
    print("-" * 100)

    print(
        top[
            [
                "canonical_pao1",
                "training_consensus_effect",
                f"{held_out}_effect",
                "transfer_category",
            ]
        ].to_string(
            index=False
        )
    )


    # ========================================================
    # Audit-20 selected pair transfer
    # ========================================================

    pair_file = (
        PAIR_DIR
        / f"selected_pairs_without_{held_out}.csv"
    )

    pair_records = []


    if pair_file.exists():

        selected_pairs = pd.read_csv(
            pair_file
        )


        for _, pair in (
            selected_pairs.iterrows()
        ):

            high_gene = pair[
                "high_gene"
            ]

            low_gene = pair[
                "low_gene"
            ]


            held_effect = pair_effect(
                held_out,
                high_gene,
                low_gene,
            )


            if held_effect > 0:
                category = (
                    "transfers_correctly"
                )

            elif held_effect < 0:
                category = (
                    "reversed"
                )

            else:
                category = (
                    "no_discrimination"
                )


            pair_records.append(
                {
                    "held_out_study":
                        held_out,

                    "high_gene":
                        high_gene,

                    "low_gene":
                        low_gene,

                    "training_min_pair_effect":
                        pair[
                            "minimum_pair_effect"
                        ],

                    "heldout_pair_effect":
                        held_effect,

                    "transfer_category":
                        category,
                }
            )


        pair_table = pd.DataFrame(
            pair_records
        )


        n_pairs_correct = int(
            (
                pair_table[
                    "transfer_category"
                ]
                ==
                "transfers_correctly"
            ).sum()
        )

        n_pairs_reversed = int(
            (
                pair_table[
                    "transfer_category"
                ]
                ==
                "reversed"
            ).sum()
        )

        n_pairs_zero = int(
            (
                pair_table[
                    "transfer_category"
                ]
                ==
                "no_discrimination"
            ).sum()
        )


        print()
        print("AUDIT-20 PAIR TRANSFER")
        print("-" * 100)

        print(
            f"Pairs transferring correctly: "
            f"{n_pairs_correct} / "
            f"{len(pair_table)}"
        )

        print(
            f"Pairs reversing in held-out study: "
            f"{n_pairs_reversed} / "
            f"{len(pair_table)}"
        )

        print(
            f"Pairs with no discrimination: "
            f"{n_pairs_zero} / "
            f"{len(pair_table)}"
        )

        print()

        print(
            pair_table[
                [
                    "high_gene",
                    "low_gene",
                    "training_min_pair_effect",
                    "heldout_pair_effect",
                    "transfer_category",
                ]
            ].to_string(
                index=False
            )
        )


    else:

        pair_table = pd.DataFrame()

        n_pairs_correct = np.nan
        n_pairs_reversed = np.nan
        n_pairs_zero = np.nan

        print()
        print(
            "Audit-20 selected-pair file "
            f"not found for {held_out}"
        )


    # ========================================================
    # Save fold outputs
    # ========================================================

    gene_table[
        "held_out_study"
    ] = held_out

    gene_transfer_all.append(
        gene_table
    )


    if not pair_table.empty:

        pair_transfer_all.append(
            pair_table
        )


    fold_summary_records.append(
        {
            "held_out_study":
                held_out,

            "training_study_1":
                t1,

            "training_study_2":
                t2,

            "n_training_candidates":
                n_candidates,

            "n_same_direction_strong":
                n_same_strong,

            "n_same_direction_weak":
                n_same_weak,

            "n_reversed_strong":
                n_reverse_strong,

            "n_reversed_weak":
                n_reverse_weak,

            "n_heldout_zero":
                n_zero,

            "same_direction_fraction":
                same_direction_fraction,

            "strong_transfer_fraction":
                strong_transfer_fraction,

            "all_gene_spearman":
                rho_all,

            "candidate_spearman":
                rho_candidates,

            "audit20_pairs_correct":
                n_pairs_correct,

            "audit20_pairs_reversed":
                n_pairs_reversed,

            "audit20_pairs_zero":
                n_pairs_zero,
        }
    )


# ============================================================
# Combined outputs
# ============================================================

fold_summary = pd.DataFrame(
    fold_summary_records
)

gene_transfer = pd.concat(
    gene_transfer_all,
    ignore_index=True,
)

gene_transfer.to_csv(
    OUT_DIR
    / "gene_transferability_all_folds.csv",
    index=False,
)


if pair_transfer_all:

    pair_transfer = pd.concat(
        pair_transfer_all,
        ignore_index=True,
    )

    pair_transfer.to_csv(
        OUT_DIR
        / "audit20_pair_transferability.csv",
        index=False,
    )


fold_summary.to_csv(
    OUT_DIR
    / "fold_transferability_summary.csv",
    index=False,
)


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 100)
print("AUDIT 21 — SUMMARY")
print("=" * 100)

print(
    fold_summary[
        [
            "held_out_study",
            "n_training_candidates",
            "same_direction_fraction",
            "strong_transfer_fraction",
            "candidate_spearman",
            "audit20_pairs_correct",
            "audit20_pairs_reversed",
            "audit20_pairs_zero",
        ]
    ].to_string(
        index=False
    )
)


print()
print("=" * 100)
print("AUDIT 21 COMPLETE")
print("=" * 100)

print(
    f"Outputs saved to: "
    f"{OUT_DIR}"
)

print()

print(
    "Interpretation:"
)

print(
    "High same-direction transfer means a "
    "training-derived biofilm signature survives "
    "in the unseen experiment."
)

print(
    "Large reversed fractions or negative held-out "
    "pair effects indicate experiment-specific "
    "biology and explain poor cross-study prediction."
)

print()
