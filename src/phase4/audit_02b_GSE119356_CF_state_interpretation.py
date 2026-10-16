from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


ROOT = Path.cwd()

CF_ACTIVITY = (
    ROOT
    / "results/phase4/audit_02a_GSE119356_matched_projection_QC"
    / "matched_region_imodulon_contrasts.csv"
)

CF_POOLED = (
    ROOT
    / "results/phase4/audit_02a_GSE119356_matched_projection_QC"
    / "pooled_invivo_minus_invitro_imodulon_contrast.csv"
)

ROBUST = (
    ROOT
    / "results/phase3/audit_04_freeze_crossstrain_state_framework"
    / "phase3_robust_transferable_imodulons.csv"
)

STATIONARY = (
    ROOT
    / "results/phase3/audit_03d_GSE55197_stationary_crossstrain_projection"
    / "GSE55197_PA14_stationary_programme_vector.csv"
)

OXYGEN = (
    ROOT
    / "results/phase3/audit_03e_GSE55197_anoxic_crossstrain_projection"
    / "GSE55197_anoxic_stationary_residual_decomposition.csv"
)

MULTISTRESS = (
    ROOT
    / "results/phase2/audit_10_multistress_imodulon_dictionary"
    / "stress_contrast_vectors.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_02b_GSE119356_CF_state_interpretation"
)


REGIONS = [
    "BR",
    "ULP",
    "ULC",
    "LLP",
    "LLC",
]


def cosine(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return np.nan

    return float(
        np.dot(
            a,
            b,
        )
        / denominator
    )


def sign_agreement(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    return float(
        np.mean(
            np.sign(a)
            == np.sign(b)
        )
    )


def permutation_p(
    a,
    b,
    n_perm=10000,
    seed=20261007,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    observed = cosine(
        a,
        b,
    )

    rng = np.random.default_rng(
        seed
    )

    null = np.empty(
        n_perm,
        dtype=float,
    )

    for i in range(
        n_perm
    ):
        null[i] = cosine(
            a,
            rng.permutation(
                b
            ),
        )

    p = (
        1
        + int(
            np.sum(
                null >= observed
            )
        )
    ) / (
        n_perm
        + 1
    )

    return observed, float(p)


def find_index_column(table):
    for candidate in [
        "imodulon",
        "iModulon",
        "Unnamed: 0",
    ]:
        if candidate in table.columns:
            return candidate

    return None


def read_vector(
    path,
    column,
    imodulons,
):
    table = pd.read_csv(
        path
    )

    index_column = find_index_column(
        table
    )

    if index_column is None:
        raise RuntimeError(
            "Could not identify iModulon column in "
            + str(path)
        )

    if column not in table.columns:
        raise RuntimeError(
            str(path)
            + " missing column "
            + column
            + ". Available columns: "
            + ", ".join(
                str(value)
                for value in table.columns
            )
        )

    table[index_column] = (
        table[index_column]
        .astype(str)
        .str.strip()
    )

    vector = pd.to_numeric(
        table.set_index(
            index_column
        )[column],
        errors="raise",
    )

    vector = vector.reindex(
        imodulons
    )

    if vector.isna().any():
        raise RuntimeError(
            "Reference vector is missing iModulons: "
            + str(path)
        )

    return vector


def compare(
    target,
    reference,
    programme_set,
):
    a = target.reindex(
        programme_set
    ).to_numpy(
        dtype=float
    )

    b = reference.reindex(
        programme_set
    ).to_numpy(
        dtype=float
    )

    observed_cosine, p = permutation_p(
        a,
        b,
    )

    return {
        "pearson":
            float(
                pearsonr(
                    a,
                    b,
                ).statistic
            ),

        "spearman":
            float(
                spearmanr(
                    a,
                    b,
                ).statistic
            ),

        "cosine":
            observed_cosine,

        "sign_agreement":
            sign_agreement(
                a,
                b,
            ),

        "permutation_p":
            p,
    }


def load_multistress_dictionary(
    imodulons,
):
    table = pd.read_csv(
        MULTISTRESS
    )

    index_column = find_index_column(
        table
    )

    if index_column is None:
        raise RuntimeError(
            "Could not identify iModulon column in multistress dictionary."
        )

    table[index_column] = (
        table[index_column]
        .astype(str)
        .str.strip()
    )

    table = table.set_index(
        index_column
    )

    overlap = len(
        set(
            table.index
        ).intersection(
            set(
                imodulons
            )
        )
    )

    if overlap < 110:
        raise RuntimeError(
            "Multistress dictionary does not contain the expected "
            "iModulon universe."
        )

    table = table.reindex(
        imodulons
    )

    numeric_columns = []

    for column in table.columns:
        values = pd.to_numeric(
            table[
                column
            ],
            errors="coerce",
        )

        if values.notna().all():
            numeric_columns.append(
                column
            )

    if not numeric_columns:
        raise RuntimeError(
            "No numeric controlled-state axes found."
        )

    dictionary = table[
        numeric_columns
    ].apply(
        pd.to_numeric,
        errors="raise",
    )

    return dictionary


def normalized_subspace_fit(
    target,
    reference_matrix,
    programme_set,
):
    """
    Project target onto the span of L2-normalized controlled-state
    reference vectors.

    Coefficients are intentionally NOT interpreted as unique causal
    contributions because the state axes are correlated.

    We report only how much of the CF programme vector lies within
    the controlled-state span.
    """

    y = target.reindex(
        programme_set
    ).to_numpy(
        dtype=float
    )

    B = reference_matrix.reindex(
        programme_set
    ).to_numpy(
        dtype=float
    )

    norms = np.linalg.norm(
        B,
        axis=0,
    )

    valid = norms > 0

    B = B[
        :,
        valid
    ]

    norms = norms[
        valid
    ]

    B_normalized = B / norms

    coefficients = (
        np.linalg.pinv(
            B_normalized
        )
        @ y
    )

    reconstruction = (
        B_normalized
        @ coefficients
    )

    energy_fraction = float(
        np.linalg.norm(
            reconstruction
        )
        ** 2
        / (
            np.linalg.norm(
                y
            )
            ** 2
        )
    )

    reconstruction_cosine = cosine(
        y,
        reconstruction,
    )

    relative_error = float(
        np.linalg.norm(
            y
            - reconstruction
        )
        / np.linalg.norm(
            y
        )
    )

    return {
        "controlled_state_span_energy_fraction":
            energy_fraction,

        "controlled_state_span_cosine":
            reconstruction_cosine,

        "controlled_state_span_relative_error":
            relative_error,
    }


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 02b")
    print("GSE119356 first CF-lung physiological-state interpretation")
    print("=" * 78)

    for path in [
        CF_ACTIVITY,
        CF_POOLED,
        ROBUST,
        STATIONARY,
        OXYGEN,
        MULTISTRESS,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. CF-lung matched programme shifts
    # ----------------------------------------------------------

    regional = pd.read_csv(
        CF_ACTIVITY,
        index_col=0,
    )

    regional.index = (
        regional.index
        .astype(str)
        .str.strip()
    )

    regional = regional.apply(
        pd.to_numeric,
        errors="raise",
    )

    for region in REGIONS:
        if region not in regional.columns:
            raise RuntimeError(
                "Missing region: "
                + region
            )

    imodulons = list(
        regional.index
    )

    if len(
        imodulons
    ) != 116:
        raise RuntimeError(
            "Expected 116 iModulons, found "
            + str(
                len(
                    imodulons
                )
            )
        )

    pooled_table = pd.read_csv(
        CF_POOLED,
        index_col=0,
    )

    if pooled_table.shape[1] != 1:
        raise RuntimeError(
            "Expected one pooled activity column."
        )

    pooled = pd.to_numeric(
        pooled_table.iloc[
            :,
            0
        ],
        errors="raise",
    )

    pooled.index = (
        pooled.index
        .astype(str)
        .str.strip()
    )

    pooled = pooled.reindex(
        imodulons
    )

    if pooled.isna().any():
        raise RuntimeError(
            "Pooled activity missing iModulons."
        )

    print()
    print("CF-lung activity vectors")
    print(
        "  regional vectors:",
        regional.shape[1],
    )
    print(
        "  programmes:",
        regional.shape[0],
    )

    # ----------------------------------------------------------
    # 2. Frozen primary programme set
    # ----------------------------------------------------------

    robust_table = pd.read_csv(
        ROBUST
    )

    if "imodulon" not in robust_table.columns:
        raise RuntimeError(
            "Robust programme file missing imodulon column."
        )

    primary_programmes = [
        imodulon
        for imodulon in imodulons
        if imodulon
        in set(
            robust_table[
                "imodulon"
            ]
            .astype(str)
            .str.strip()
        )
    ]

    primary = pd.Index(
        primary_programmes,
        name="PRIMARY_111",
    )

    all_programmes = pd.Index(
        imodulons,
        name="ALL_116",
    )

    print()
    print("Interpretation programme sets")
    print(
        "  primary:",
        len(
            primary
        ),
    )
    print(
        "  all:",
        len(
            all_programmes
        ),
    )

    # ----------------------------------------------------------
    # 3. Strong/supporting Phase-3 anchors
    # ----------------------------------------------------------

    stationary = read_vector(
        STATIONARY,
        "stationary_vs_exponential",
        imodulons,
    )

    oxygen = read_vector(
        OXYGEN,
        "anoxic_residual_after_stationary",
        imodulons,
    )

    validated_references = {
        "stationary_STRONG":
            stationary,

        "oxygen_associated_SUPPORTING":
            oxygen,
    }

    # ----------------------------------------------------------
    # 4. Phase-2 controlled perturbation dictionary
    # ----------------------------------------------------------

    dictionary = load_multistress_dictionary(
        imodulons
    )

    print()
    print("Phase-2 controlled-state dictionary axes")

    for column in dictionary.columns:
        print(
            " ",
            column,
        )

    # Do not duplicate our frozen Phase-3 stationary and oxygen
    # references in the primary ranking.
    excluded_dictionary_names = {
        "ctrl",
        "control",
        "stationary",
        "microaerophilic",
    }

    exploratory_columns = []

    for column in dictionary.columns:
        normalized = (
            str(
                column
            )
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        if normalized in excluded_dictionary_names:
            continue

        exploratory_columns.append(
            column
        )

    exploratory_dictionary = dictionary[
        exploratory_columns
    ].copy()

    print()
    print(
        "Exploratory controlled-state axes retained:",
        len(
            exploratory_columns
        ),
    )

    # ----------------------------------------------------------
    # 5. Assemble target vectors
    # ----------------------------------------------------------

    targets = {
        region:
            regional[
                region
            ]
        for region in REGIONS
    }

    targets[
        "POOLED"
    ] = pooled

    # ----------------------------------------------------------
    # 6. Validated-state agreement
    # ----------------------------------------------------------

    validated_rows = []

    for target_name, target in targets.items():
        for reference_name, reference in validated_references.items():
            for programme_set in [
                primary,
                all_programmes,
            ]:
                stats = compare(
                    target,
                    reference,
                    programme_set,
                )

                validated_rows.append(
                    {
                        "target":
                            target_name,

                        "reference":
                            reference_name,

                        "programme_set":
                            programme_set.name,

                        "n_programmes":
                            len(
                                programme_set
                            ),

                        **stats,
                    }
                )

    validated = pd.DataFrame(
        validated_rows
    )

    print()
    print("Validated Phase-3 state alignment")
    print(
        validated
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Exploratory controlled-state ranking
    # ----------------------------------------------------------

    exploratory_rows = []

    for target_name, target in targets.items():
        for axis in exploratory_dictionary.columns:
            reference = exploratory_dictionary[
                axis
            ]

            stats = compare(
                target,
                reference,
                primary,
            )

            exploratory_rows.append(
                {
                    "target":
                        target_name,

                    "controlled_state":
                        axis,

                    "programme_set":
                        primary.name,

                    **stats,
                }
            )

    exploratory = pd.DataFrame(
        exploratory_rows
    )

    # Positive alignment is the interpretable direction here.
    ranked = exploratory.sort_values(
        [
            "target",
            "cosine",
        ],
        ascending=[
            True,
            False,
        ],
    )

    print()
    print("Top controlled-state similarities by CF-lung target")

    for target_name in targets:
        subset = ranked.loc[
            ranked[
                "target"
            ]
            == target_name
        ].head(
            6
        )

        print()
        print(
            target_name
        )

        print(
            subset[
                [
                    "controlled_state",
                    "pearson",
                    "spearman",
                    "cosine",
                    "sign_agreement",
                    "permutation_p",
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
    # 8. Build broad controlled-state span
    #
    # Include the validated stationary and oxygen anchors plus
    # all non-duplicate Phase-2 controlled axes.
    #
    # This asks:
    # "How much of the CF-lung programme shift can be represented
    #  by known controlled physiological directions?"
    #
    # It does NOT assign causal percentages to individual axes.
    # ----------------------------------------------------------

    span_dictionary = pd.DataFrame(
        index=imodulons
    )

    span_dictionary[
        "stationary_STRONG"
    ] = stationary

    span_dictionary[
        "oxygen_associated_SUPPORTING"
    ] = oxygen

    for column in exploratory_dictionary.columns:
        span_dictionary[
            "phase2_"
            + str(
                column
            )
        ] = exploratory_dictionary[
            column
        ]

    # Remove any exactly duplicated or zero columns.
    keep_columns = []

    fingerprints = []

    for column in span_dictionary.columns:
        vector = span_dictionary[
            column
        ].to_numpy(
            dtype=float
        )

        norm = np.linalg.norm(
            vector
        )

        if norm == 0:
            continue

        normalized_vector = (
            vector
            / norm
        )

        duplicate = False

        for existing in fingerprints:
            if np.allclose(
                normalized_vector,
                existing,
                rtol=1e-10,
                atol=1e-10,
            ):
                duplicate = True
                break

            if np.allclose(
                normalized_vector,
                -existing,
                rtol=1e-10,
                atol=1e-10,
            ):
                duplicate = True
                break

        if duplicate:
            continue

        keep_columns.append(
            column
        )

        fingerprints.append(
            normalized_vector
        )

    span_dictionary = span_dictionary[
        keep_columns
    ]

    print()
    print("Controlled-state span")
    print(
        "  unique reference axes:",
        span_dictionary.shape[1],
    )

    span_rows = []

    for target_name, target in targets.items():
        primary_fit = normalized_subspace_fit(
            target,
            span_dictionary,
            primary,
        )

        all_fit = normalized_subspace_fit(
            target,
            span_dictionary,
            all_programmes,
        )

        span_rows.append(
            {
                "target":
                    target_name,

                "programme_set":
                    primary.name,

                **primary_fit,
            }
        )

        span_rows.append(
            {
                "target":
                    target_name,

                "programme_set":
                    all_programmes.name,

                **all_fit,
            }
        )

    span_fit = pd.DataFrame(
        span_rows
    )

    print()
    print("Controlled-state subspace representation")
    print(
        span_fit
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Largest CF-lung programme shifts
    #
    # These are direct iModulon changes in vivo relative to the
    # matched cultured isolate pool.
    # ----------------------------------------------------------

    programme_rows = []

    for target_name, target in targets.items():
        for imodulon in imodulons:
            programme_rows.append(
                {
                    "target":
                        target_name,

                    "imodulon":
                        imodulon,

                    "activity_shift":
                        float(
                            target.loc[
                                imodulon
                            ]
                        ),

                    "abs_activity_shift":
                        abs(
                            float(
                                target.loc[
                                    imodulon
                                ]
                            )
                        ),

                    "phase3_primary":
                        imodulon
                        in set(
                            primary
                        ),
                }
            )

    programme_shifts = pd.DataFrame(
        programme_rows
    )

    print()
    print("Largest pooled CF-lung programme shifts")

    pooled_ranked = (
        programme_shifts.loc[
            programme_shifts[
                "target"
            ]
            == "POOLED"
        ]
        .sort_values(
            "abs_activity_shift",
            ascending=False,
        )
    )

    print(
        pooled_ranked[
            [
                "imodulon",
                "activity_shift",
                "phase3_primary",
            ]
        ]
        .head(
            25
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 10. Regional programme heterogeneity
    #
    # Identify programmes that vary most across the five spatial
    # regions after matching each region to its own isolate pool.
    # ----------------------------------------------------------

    regional_primary = regional.reindex(
        primary
    )

    regional_variability = pd.DataFrame(
        {
            "imodulon":
                primary,

            "regional_mean":
                regional_primary.mean(
                    axis=1
                ).to_numpy(),

            "regional_sd":
                regional_primary.std(
                    axis=1,
                    ddof=1,
                ).to_numpy(),

            "regional_range":
                (
                    regional_primary.max(
                        axis=1
                    )
                    - regional_primary.min(
                        axis=1
                    )
                ).to_numpy(),
        }
    )

    regional_variability[
        "abs_regional_mean"
    ] = regional_variability[
        "regional_mean"
    ].abs()

    regional_variability = (
        regional_variability
        .sort_values(
            "regional_sd",
            ascending=False,
        )
    )

    print()
    print("Most spatially heterogeneous primary programmes")
    print(
        regional_variability[
            [
                "imodulon",
                "regional_mean",
                "regional_sd",
                "regional_range",
            ]
        ]
        .head(
            20
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 11. Consensus-direction diagnostic
    #
    # Which programme shifts have the same sign in all 5 regions?
    # ----------------------------------------------------------

    consensus_rows = []

    for imodulon in primary:
        values = regional.loc[
            imodulon,
            REGIONS,
        ].to_numpy(
            dtype=float
        )

        signs = np.sign(
            values
        )

        all_positive = bool(
            np.all(
                signs > 0
            )
        )

        all_negative = bool(
            np.all(
                signs < 0
            )
        )

        consensus = (
            all_positive
            or all_negative
        )

        consensus_rows.append(
            {
                "imodulon":
                    imodulon,

                "consensus_direction":
                    consensus,

                "direction":
                    (
                        "positive"
                        if all_positive
                        else (
                            "negative"
                            if all_negative
                            else "mixed"
                        )
                    ),

                "regional_mean":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "minimum_abs_shift":
                    float(
                        np.min(
                            np.abs(
                                values
                            )
                        )
                    ),

                "maximum_abs_shift":
                    float(
                        np.max(
                            np.abs(
                                values
                            )
                        )
                    ),
            }
        )

    consensus = pd.DataFrame(
        consensus_rows
    )

    consensus_ranked = (
        consensus.loc[
            consensus[
                "consensus_direction"
            ]
        ]
        .sort_values(
            "minimum_abs_shift",
            ascending=False,
        )
    )

    print()
    print("Strongest programme shifts with the same direction in all 5 regions")
    print(
        consensus_ranked[
            [
                "imodulon",
                "direction",
                "regional_mean",
                "minimum_abs_shift",
                "maximum_abs_shift",
            ]
        ]
        .head(
            20
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 12. Internal consistency / audit decision
    #
    # No particular biological state is required to be present.
    # These checks only establish that the interpretation products
    # were generated consistently from the frozen framework.
    # ----------------------------------------------------------

    pooled_from_regions = regional.mean(
        axis=1
    )

    pooled_difference = float(
        np.max(
            np.abs(
                pooled_from_regions.to_numpy(
                    dtype=float
                )
                - pooled.to_numpy(
                    dtype=float
                )
            )
        )
    )

    finite_outputs = bool(
        np.isfinite(
            validated[
                [
                    "pearson",
                    "spearman",
                    "cosine",
                    "sign_agreement",
                    "permutation_p",
                ]
            ].to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            exploratory[
                [
                    "pearson",
                    "spearman",
                    "cosine",
                    "sign_agreement",
                    "permutation_p",
                ]
            ].to_numpy(
                dtype=float
            )
        ).all()
        and np.isfinite(
            span_fit[
                [
                    "controlled_state_span_energy_fraction",
                    "controlled_state_span_cosine",
                    "controlled_state_span_relative_error",
                ]
            ].to_numpy(
                dtype=float
            )
        ).all()
    )

    checks = {
        "five_regional_CF_vectors":
            regional.shape[1]
            == 5,

        "116_total_programmes":
            regional.shape[0]
            == 116,

        "111_primary_programmes":
            len(
                primary
            )
            == 111,

        "stationary_reference_loaded":
            bool(
                np.isfinite(
                    stationary.to_numpy(
                        dtype=float
                    )
                ).all()
            ),

        "oxygen_reference_loaded":
            bool(
                np.isfinite(
                    oxygen.to_numpy(
                        dtype=float
                    )
                ).all()
            ),

        "multistress_dictionary_ge_8_axes":
            exploratory_dictionary.shape[1]
            >= 8,

        "pooled_equals_mean_regional_activity":
            pooled_difference
            < 1e-8,

        "finite_interpretation_outputs":
            finite_outputs,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 02b consistency checks")

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
        "GSE119356_CF_STATE_INTERPRETATION_GENERATED"
        if passed
        else "GSE119356_CF_STATE_INTERPRETATION_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: these vectors represent in-vivo CF-lung "
        "transcription relative to matched in-vitro isolate pools. "
        "They are not yet absolute physiological-state coordinates."
    )

    print()
    print(
        "Controlled-state similarities are physiological analogies, "
        "not proof that the literal experimental perturbation occurred "
        "inside the lung."
    )

    # ----------------------------------------------------------
    # 13. Save
    # ----------------------------------------------------------

    validated.to_csv(
        OUT
        / "validated_state_alignment.csv",
        index=False,
    )

    exploratory.to_csv(
        OUT
        / "controlled_state_similarity_all.csv",
        index=False,
    )

    ranked.to_csv(
        OUT
        / "controlled_state_similarity_ranked.csv",
        index=False,
    )

    span_dictionary.to_csv(
        OUT
        / "controlled_state_reference_library.csv"
    )

    span_fit.to_csv(
        OUT
        / "controlled_state_subspace_fit.csv",
        index=False,
    )

    programme_shifts.to_csv(
        OUT
        / "CF_lung_programme_shifts.csv",
        index=False,
    )

    regional_variability.to_csv(
        OUT
        / "regional_programme_heterogeneity.csv",
        index=False,
    )

    consensus.to_csv(
        OUT
        / "regional_programme_direction_consensus.csv",
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

    (
        OUT
        / "AUDIT_02B_STATUS.txt"
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
            "Failed Audit 02b consistency checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
