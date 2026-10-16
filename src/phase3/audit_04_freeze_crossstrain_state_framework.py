from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

FROZEN_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

TRANSFER = (
    ROOT
    / "results/phase3/audit_03c_GSE55197_composite_ID_resolution"
    / "GSE55197_exact_imodulon_transferability.csv"
)

STATIONARY_AGREEMENT = (
    ROOT
    / "results/phase3/audit_03d_GSE55197_stationary_crossstrain_projection"
    / "GSE55197_crossstrain_stationary_agreement.csv"
)

STATIONARY_SUMMARY = (
    ROOT
    / "results/phase3/audit_03d_GSE55197_stationary_crossstrain_projection"
    / "AUDIT_03D_SUMMARY.md"
)

OXYGEN_AGREEMENT = (
    ROOT
    / "results/phase3/audit_03f_GSE55197_anoxic_programme_stability"
    / "individual_replicate_crossstrain_oxygen_agreement.csv"
)

OXYGEN_SUMMARY = (
    ROOT
    / "results/phase3/audit_03f_GSE55197_anoxic_programme_stability"
    / "AUDIT_03F_SUMMARY.md"
)

BIOFILM_DECOMP = (
    ROOT
    / "results/phase3/audit_03h_GSE55197_biofilm_programme_stability"
    / "individual_biofilm_conserved_state_decomposition.csv"
)

BIOFILM_AGREEMENT = (
    ROOT
    / "results/phase3/audit_03h_GSE55197_biofilm_programme_stability"
    / "individual_biofilm_crossstudy_agreement.csv"
)

BIOFILM_STATUS = (
    ROOT
    / "results/phase3/audit_03h_GSE55197_biofilm_programme_stability"
    / "AUDIT_03H_STATUS.txt"
)

OUT = (
    ROOT
    / "results/phase3/audit_04_freeze_crossstrain_state_framework"
)


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            str(path)
        )


def finite(value):
    return bool(
        np.isfinite(
            float(value)
        )
    )


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 04")
    print("Freeze cross-strain physiological-state framework")
    print("=" * 78)

    required = [
        FROZEN_MAP,
        TRANSFER,
        STATIONARY_AGREEMENT,
        STATIONARY_SUMMARY,
        OXYGEN_AGREEMENT,
        OXYGEN_SUMMARY,
        BIOFILM_DECOMP,
        BIOFILM_AGREEMENT,
        BIOFILM_STATUS,
    ]

    for path in required:
        require_file(
            path
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Frozen cross-strain gene/programme universe
    # ----------------------------------------------------------

    frozen_map = pd.read_csv(
        FROZEN_MAP
    )

    transfer = pd.read_csv(
        TRANSFER
    )

    transfer["imodulon"] = (
        transfer["imodulon"]
        .astype(str)
        .str.strip()
    )

    robust_mask = (
        (
            transfer[
                "weighted_coverage_absM"
            ]
            >= 0.90
        )
        & (
            transfer[
                "energy_coverage_M2"
            ]
            >= 0.80
        )
    )

    robust = transfer.loc[
        robust_mask
    ].copy()

    weak = transfer.loc[
        ~robust_mask
    ].copy()

    print()
    print("Frozen transfer universe")
    print(
        "  PA14<->PAO1 genes:",
        len(
            frozen_map
        ),
    )
    print(
        "  total iModulons:",
        len(
            transfer
        ),
    )
    print(
        "  robust transferable iModulons:",
        len(
            robust
        ),
    )
    print(
        "  sensitivity-only iModulons:",
        len(
            weak
        ),
    )

    if not weak.empty:
        print()
        print("Sensitivity-only programmes")
        print(
            weak[
                [
                    "imodulon",
                    "weighted_coverage_absM",
                    "energy_coverage_M2",
                ]
            ]
            .round(
                4
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 2. Stationary state
    # ----------------------------------------------------------

    stationary = pd.read_csv(
        STATIONARY_AGREEMENT
    )

    stationary_all = stationary.loc[
        (
            stationary["reference"]
            == "GSE120602_stationary_vs_planktonic"
        )
        & (
            stationary["programme_set"]
            == "ALL_116"
        )
    ]

    stationary_robust = stationary.loc[
        (
            stationary["reference"]
            == "GSE120602_stationary_vs_planktonic"
        )
        & (
            stationary["programme_set"]
            == "ROBUST_TRANSFER"
        )
    ]

    if len(
        stationary_all
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify ALL_116 stationary comparison."
        )

    if len(
        stationary_robust
    ) != 1:
        raise RuntimeError(
            "Could not uniquely identify robust stationary comparison."
        )

    stationary_all = stationary_all.iloc[0]
    stationary_robust = stationary_robust.iloc[0]

    stationary_summary_text = STATIONARY_SUMMARY.read_text(
        encoding="utf-8"
    )

    stationary_status_ok = (
        "GSE55197_STATIONARY_CROSSSTRAIN_PROJECTION_ACCEPTED"
        in stationary_summary_text
    )

    # ----------------------------------------------------------
    # 3. Oxygen-associated state family
    # ----------------------------------------------------------

    oxygen = pd.read_csv(
        OXYGEN_AGREEMENT
    )

    oxygen_all = oxygen.loc[
        oxygen[
            "programme_set"
        ]
        == "ALL_116"
    ].copy()

    oxygen_raw = oxygen_all.loc[
        oxygen_all[
            "comparison"
        ]
        == "raw_anoxic_vs_PAO1_microaerophilic"
    ].copy()

    oxygen_residual = oxygen_all.loc[
        oxygen_all[
            "comparison"
        ]
        == "stationary_residual_vs_PAO1_microaerophilic"
    ].copy()

    if len(
        oxygen_raw
    ) != 2:
        raise RuntimeError(
            "Expected two individual raw anoxic comparisons."
        )

    if len(
        oxygen_residual
    ) != 2:
        raise RuntimeError(
            "Expected two stationary-residual oxygen comparisons."
        )

    oxygen_summary_text = OXYGEN_SUMMARY.read_text(
        encoding="utf-8"
    )

    oxygen_status_ok = (
        "GSE55197_ANOXIC_PROGRAMME_SIGNAL_ACCEPTED_WITH_EXPRESSION_QC_CAVEAT"
        in oxygen_summary_text
    )

    # ----------------------------------------------------------
    # 4. Biofilm composition
    # ----------------------------------------------------------

    biofilm_decomp = pd.read_csv(
        BIOFILM_DECOMP
    )

    expected_samples = {
        "PA14_BF24_1",
        "PA14_BF24_2",
        "PA14_BF48_1",
        "PA14_BF48_2",
    }

    observed_samples = set(
        biofilm_decomp[
            "sample"
        ]
    )

    if observed_samples != expected_samples:
        raise RuntimeError(
            "Unexpected biofilm replicate set."
        )

    biofilm_status_text = BIOFILM_STATUS.read_text(
        encoding="utf-8"
    ).strip()

    biofilm_status_ok = (
        biofilm_status_text
        == "GSE55197_PA14_BIOFILM_PROGRAMME_SIGNAL_ACCEPTED_WITH_EXPRESSION_QC_CAVEAT"
    )

    bf24 = biofilm_decomp.loc[
        biofilm_decomp[
            "timepoint"
        ].astype(str)
        == "24h"
    ].copy()

    bf48 = biofilm_decomp.loc[
        biofilm_decomp[
            "timepoint"
        ].astype(str)
        == "48h"
    ].copy()

    if len(
        bf24
    ) != 2:
        raise RuntimeError(
            "Expected two 24h biofilm replicates."
        )

    if len(
        bf48
    ) != 2:
        raise RuntimeError(
            "Expected two 48h biofilm replicates."
        )

    bf24_stationary_min = float(
        bf24[
            "stationary_energy_fraction"
        ].min()
    )

    bf24_stationary_max = float(
        bf24[
            "stationary_energy_fraction"
        ].max()
    )

    bf48_stationary_min = float(
        bf48[
            "stationary_energy_fraction"
        ].min()
    )

    bf48_stationary_max = float(
        bf48[
            "stationary_energy_fraction"
        ].max()
    )

    bf24_oxygen_max = float(
        bf24[
            "oxygen_energy_fraction"
        ].max()
    )

    bf48_oxygen_max = float(
        bf48[
            "oxygen_energy_fraction"
        ].max()
    )

    # ----------------------------------------------------------
    # 5. Cross-study biofilm interpretation check
    # ----------------------------------------------------------

    biofilm_agreement = pd.read_csv(
        BIOFILM_AGREEMENT
    )

    gse120602 = biofilm_agreement.loc[
        (
            biofilm_agreement[
                "reference"
            ]
            == "GSE120602"
        )
        & (
            biofilm_agreement[
                "programme_set"
            ]
            == "ALL_116"
        )
    ].copy()

    raw_120602 = gse120602.loc[
        gse120602[
            "vector"
        ]
        == "raw"
    ].copy()

    residual_120602 = gse120602.loc[
        gse120602[
            "vector"
        ]
        == "anchor_residual"
    ].copy()

    if len(
        raw_120602
    ) != 4:
        raise RuntimeError(
            "Expected four raw GSE120602 biofilm comparisons."
        )

    if len(
        residual_120602
    ) != 4:
        raise RuntimeError(
            "Expected four residual GSE120602 comparisons."
        )

    raw_120602_mean_pearson = float(
        raw_120602[
            "pearson"
        ].mean()
    )

    residual_120602_mean_pearson = float(
        residual_120602[
            "pearson"
        ].mean()
    )

    # ----------------------------------------------------------
    # 6. Freeze evidence table
    # ----------------------------------------------------------

    state_rows = [
        {
            "state_family":
                "stationary",

            "evidence_tier":
                "STRONG",

            "PA14_test":
                "GSE55197 stationary vs exponential",

            "PAO1_reference":
                "GSE120602 stationary vs planktonic",

            "technical_status":
                "ACCEPTED",

            "expression_QC_caveat":
                False,

            "programme_level_validated":
                True,

            "crossstrain_pearson":
                float(
                    stationary_all[
                        "pearson"
                    ]
                ),

            "crossstrain_cosine":
                float(
                    stationary_all[
                        "cosine"
                    ]
                ),

            "crossstrain_permutation_p":
                float(
                    stationary_all[
                        "programme_label_permutation_p"
                    ]
                ),

            "interpretation":
                (
                    "Conserved stationary-phase programme transfers "
                    "strongly from PAO1 to PA14."
                ),
        },

        {
            "state_family":
                "oxygen_associated",

            "evidence_tier":
                "SUPPORTING_WITH_QC_CAVEAT",

            "PA14_test":
                "GSE55197 anoxic vs exponential",

            "PAO1_reference":
                "GSE152295 microaerophilic",

            "technical_status":
                "PROGRAMME_SIGNAL_ACCEPTED",

            "expression_QC_caveat":
                True,

            "programme_level_validated":
                True,

            "crossstrain_pearson":
                float(
                    oxygen_raw[
                        "pearson"
                    ].mean()
                ),

            "crossstrain_cosine":
                float(
                    oxygen_raw[
                        "cosine"
                    ].mean()
                ),

            "crossstrain_permutation_p":
                float(
                    oxygen_raw[
                        "permutation_p"
                    ].max()
                ),

            "interpretation":
                (
                    "A reproducible oxygen-associated programme family "
                    "transfers across strains. Signal persists after "
                    "removing the stationary-aligned component, but the "
                    "PA14 anoxic experiment has a gene-level replicate "
                    "QC caveat and is not a pure oxygen-only contrast."
                ),
        },

        {
            "state_family":
                "biofilm",

            "evidence_tier":
                "COMPOSITE_STATE_WITH_QC_CAVEAT",

            "PA14_test":
                "GSE55197 24h and 48h biofilm vs exponential",

            "PAO1_reference":
                "GSE120602/GSE203348/GSE299861 biofilm contrasts",

            "technical_status":
                "PROGRAMME_SIGNAL_ACCEPTED",

            "expression_QC_caveat":
                True,

            "programme_level_validated":
                True,

            "crossstrain_pearson":
                np.nan,

            "crossstrain_cosine":
                np.nan,

            "crossstrain_permutation_p":
                np.nan,

            "interpretation":
                (
                    "Biofilm is not frozen as a single transferable axis. "
                    "PA14 biofilm is interpreted compositionally: a large "
                    "and increasing stationary-like component plus a "
                    "context-specific residual."
                ),
        },
    ]

    state_framework = pd.DataFrame(
        state_rows
    )

    # ----------------------------------------------------------
    # 7. Freeze quantitative biofilm summary
    # ----------------------------------------------------------

    biofilm_summary = pd.DataFrame(
        [
            {
                "timepoint":
                    "24h",

                "stationary_fraction_min":
                    bf24_stationary_min,

                "stationary_fraction_max":
                    bf24_stationary_max,

                "oxygen_fraction_max":
                    bf24_oxygen_max,
            },

            {
                "timepoint":
                    "48h",

                "stationary_fraction_min":
                    bf48_stationary_min,

                "stationary_fraction_max":
                    bf48_stationary_max,

                "oxygen_fraction_max":
                    bf48_oxygen_max,
            },
        ]
    )

    # ----------------------------------------------------------
    # 8. Final consistency checks
    # ----------------------------------------------------------

    checks = {
        "frozen_projection_genes_5286":
            len(
                frozen_map
            )
            == 5286,

        "transfer_programmes_116":
            len(
                transfer
            )
            == 116,

        "robust_programmes_111":
            len(
                robust
            )
            == 111,

        "stationary_status_accepted":
            stationary_status_ok,

        "oxygen_programme_status_accepted":
            oxygen_status_ok,

        "biofilm_programme_status_accepted":
            biofilm_status_ok,

        "stationary_metrics_finite":
            all(
                [
                    finite(
                        stationary_all[
                            "pearson"
                        ]
                    ),
                    finite(
                        stationary_all[
                            "cosine"
                        ]
                    ),
                    finite(
                        stationary_all[
                            "programme_label_permutation_p"
                        ]
                    ),
                ]
            ),

        "two_oxygen_replicates_raw":
            len(
                oxygen_raw
            )
            == 2,

        "two_oxygen_replicates_residual":
            len(
                oxygen_residual
            )
            == 2,

        "four_biofilm_replicates":
            len(
                biofilm_decomp
            )
            == 4,

        "biofilm_stationary_fraction_increases":
            bf48_stationary_min
            > bf24_stationary_max,

        "biofilm_oxygen_component_small":
            max(
                bf24_oxygen_max,
                bf48_oxygen_max,
            )
            < 0.05,

        "GSE120602_similarity_reduced_after_anchor_removal":
            residual_120602_mean_pearson
            < raw_120602_mean_pearson,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Frozen state framework")
    print(
        state_framework[
            [
                "state_family",
                "evidence_tier",
                "technical_status",
                "expression_QC_caveat",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print("Frozen biofilm composition")
    print(
        biofilm_summary
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    print()
    print("Cross-study biofilm decomposition diagnostic")
    print(
        "  mean raw PA14 vs GSE120602 Pearson:",
        round(
            raw_120602_mean_pearson,
            4,
        ),
    )
    print(
        "  mean residual PA14 vs GSE120602 Pearson:",
        round(
            residual_120602_mean_pearson,
            4,
        ),
    )

    print()
    print("Phase-3 freeze checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    status = (
        "PHASE3_CROSSSTRAIN_STATE_FRAMEWORK_FROZEN"
        if passed
        else "PHASE3_FRAMEWORK_FREEZE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    # ----------------------------------------------------------
    # 9. Save frozen framework
    # ----------------------------------------------------------

    state_framework.to_csv(
        OUT
        / "phase3_transferable_state_framework.csv",
        index=False,
    )

    robust.to_csv(
        OUT
        / "phase3_robust_transferable_imodulons.csv",
        index=False,
    )

    weak.to_csv(
        OUT
        / "phase3_sensitivity_only_imodulons.csv",
        index=False,
    )

    biofilm_summary.to_csv(
        OUT
        / "phase3_biofilm_composition_summary.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "check":
                    key,
                "pass":
                    value,
            }
            for key, value
            in checks.items()
        ]
    ).to_csv(
        OUT
        / "audit_pass_checks.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 10. Human-readable freeze record
    # ----------------------------------------------------------

    lines = [
        "# CF-State Phase 3 — Frozen cross-strain state framework",
        "",
        "## Frozen representation",
        "",
        "- PA14↔PAO1 projection universe: "
        + str(
            len(
                frozen_map
            )
        )
        + " genes",
        "- PAO1 iModulons: "
        + str(
            len(
                transfer
            )
        ),
        "- Robust transferable iModulons: "
        + str(
            len(
                robust
            )
        ),
        "",
        "## State evidence",
        "",
        "### Stationary — STRONG",
        "",
        "- Independent PA14 stationary response agrees strongly with "
        "the PAO1 stationary reference.",
        "- Pearson: "
        + format(
            float(
                stationary_all[
                    "pearson"
                ]
            ),
            ".4f",
        ),
        "- Cosine: "
        + format(
            float(
                stationary_all[
                    "cosine"
                ]
            ),
            ".4f",
        ),
        "- Permutation p: "
        + format(
            float(
                stationary_all[
                    "programme_label_permutation_p"
                ]
            ),
            ".4g",
        ),
        "",
        "### Oxygen-associated physiology — SUPPORTING",
        "",
        "- Programme-level signal is reproducible across both PA14 "
        "anoxic replicates.",
        "- Mean raw cross-strain Pearson: "
        + format(
            float(
                oxygen_raw[
                    "pearson"
                ].mean()
            ),
            ".4f",
        ),
        "- Mean stationary-residual cross-strain Pearson: "
        + format(
            float(
                oxygen_residual[
                    "pearson"
                ].mean()
            ),
            ".4f",
        ),
        "- Caveat: gene-level anoxic replicate QC missed the original "
        "0.90 expression threshold and the experiment is not a pure "
        "oxygen-only perturbation.",
        "",
        "### Biofilm — COMPOSITE STATE",
        "",
        "- Biofilm is not frozen as a universal transferable axis.",
        "- 24h stationary-like programme energy: "
        + format(
            bf24_stationary_min,
            ".4f",
        )
        + " to "
        + format(
            bf24_stationary_max,
            ".4f",
        ),
        "- 48h stationary-like programme energy: "
        + format(
            bf48_stationary_min,
            ".4f",
        )
        + " to "
        + format(
            bf48_stationary_max,
            ".4f",
        ),
        "- Oxygen-residual contribution remains small.",
        "- Agreement with GSE120602 PAO1 biofilm is strongly reduced "
        "after removing conserved physiological anchors.",
        "",
        "## Interpretation",
        "",
        "Phase 3 supports transfer of conserved physiological programmes "
        "across PAO1 and PA14. It does not support treating 'biofilm' "
        "as a single universal transcriptional state. Complex states "
        "should therefore be represented as mixtures of transferable "
        "physiological programmes plus context-specific residuals.",
        "",
        "## Phase-3 decision",
        "",
        status,
        "",
        "No additional PA14 perturbation datasets are required before "
        "moving to CF-airway transcriptome projection.",
    ]

    (
        OUT
        / "PHASE3_FRAMEWORK_FREEZE.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    (
        OUT
        / "AUDIT_04_STATUS.txt"
    ).write_text(
        status
        + "\n",
        encoding="utf-8",
    )

    if not passed:
        failed = [
            key
            for key, value
            in checks.items()
            if not value
        ]

        raise RuntimeError(
            "Failed Phase-3 freeze checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
