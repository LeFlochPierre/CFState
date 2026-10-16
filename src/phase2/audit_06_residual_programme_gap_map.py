from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

CATALOGUE = (
    ROOT
    / "external/aeruprecise/precise411/iM_table.csv"
)

A05 = (
    ROOT
    / "results/phase2/audit_05_controlled_anchor_subspace"
)

EXT_RESIDUAL = (
    A05
    / "GSE203348_biofilm_anchor_decomposition.csv"
)

INTERNAL_RESIDUAL = (
    A05
    / "GSE120602_biofilm_anchor_decomposition.csv"
)

OUT = (
    ROOT
    / "results/phase2/audit_06_residual_programme_gap_map"
)


def norm_text(value):
    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def assign_anchor_family(row):
    """
    Heuristic biological grouping based only on the published
    aeruPRECISE iModulon name/function/category.

    These are planning labels for controlled-perturbation acquisition,
    not new biological annotations of the iModulons.
    """

    name = norm_text(
        row.get("name", "")
    )

    function = norm_text(
        row.get("function", "")
    )

    category = norm_text(
        row.get("category", "")
    )

    text = " ".join(
        [
            name,
            function,
            category,
        ]
    )

    # --------------------------------------------------
    # Quorum sensing / signalling
    # --------------------------------------------------

    qs_names = {
        "qscR-1".lower(),
        "qscR-2".lower(),
        "lasr",
        "mvfr",
        "ahl",
        "quorum sensing",
    }

    if (
        name in qs_names
        or "quorum" in text
        or "ahl" in text
        or "pqs signalling" in text
        or "phenazine" in text
        or "pyocyanin" in text
    ):
        return (
            "quorum_sensing",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Iron acquisition / siderophores
    # --------------------------------------------------

    iron_names = {
        "pvds",
        "fpvr",
        "pchr",
        "foxr",
        "fur",
        "iron acquisition",
        "k-opioid",
    }

    if (
        name in iron_names
        or "pyoverdine" in text
        or "pyochelin" in text
        or "iron acquisition" in text
        or "iron chelator" in text
        or (
            "iron" in text
            and "homeostasis" in text
        )
    ):
        return (
            "iron_limitation",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Oxygen / respiration / denitrification
    # --------------------------------------------------

    if (
        name.startswith("anr-")
        or name == "narl"
        or name == "roxr"
        or "denitrification" in text
        or "nitrate reduct" in text
        or "nitrite reduct" in text
        or "cytochrome oxidase" in text
        or "molybdenum cofactor" in text
    ):
        return (
            "oxygen_denitrification",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Secretion / virulence
    # --------------------------------------------------

    secretion_names = {
        "exsb",
        "tagr1",
        "sfa2",
        "bexr",
    }

    if (
        name in secretion_names
        or "t3ss" in text
        or "t6ss" in text
        or "secretion" in text
        or "virulence" in text
    ):
        return (
            "secretion_virulence",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Zinc / divalent metal
    # --------------------------------------------------

    if (
        name.startswith("zur-")
        or name == "czcr"
        or "zinc" in text
        or "divalent metal" in text
    ):
        return (
            "zinc_metal_homeostasis",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Sulphur
    # --------------------------------------------------

    if (
        name.startswith("cysb-")
        or name == "atsr"
        or "sulphur" in text
        or "sulfonate" in text
    ):
        return (
            "sulphur_metabolism",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Copper
    # --------------------------------------------------

    if (
        name == "cuer"
        or "copper" in text
    ):
        return (
            "copper_homeostasis",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Surface / attachment / matrix / motility
    # --------------------------------------------------

    if (
        name in {
            "pprb",
            "amrz",
            "flen-1",
            "flen-2",
        }
        or "chemotaxis" in text
        or "exopolysaccharide" in text
        or "pilus" in text
    ):
        return (
            "surface_motility_matrix",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Stationary / general growth physiology
    #
    # Already partly anchored by GSE120602.
    # --------------------------------------------------

    if (
        name.startswith("rpos-")
        or name.startswith("translational-")
        or name == "cell division"
        or "general stress response" in text
        or "cell wall biosynthesis" in text
    ):
        return (
            "growth_stationary",
            "EXISTING_ANCHOR",
        )

    # --------------------------------------------------
    # Antibiotic / generic stress
    #
    # Ciprofloxacin provides one anchor, but this family
    # remains only partially characterised.
    # --------------------------------------------------

    antibiotic_names = {
        "lexa",
        "soxr",
        "rpoh",
        "cpxr",
        "pmra",
        "ampc",
        "mexs-1",
        "mexs-2",
        "mexz",
        "esrc",
    }

    if (
        name in antibiotic_names
        or "antibiotic resistance" in category
        or "efflux" in text
        or "heat shock" in text
        or "sos response" in text
    ):
        return (
            "antibiotic_stress",
            "PARTIAL_ANCHOR",
        )

    # --------------------------------------------------
    # Prophages
    # --------------------------------------------------

    if (
        "prophage" in text
        or name.startswith("xisf")
        or name == "sigma-70"
    ):
        return (
            "prophage",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Carbon / amino acid / nutrient utilisation
    # --------------------------------------------------

    if (
        "carbon source" in category
        or "amino acid" in category
        or "metabolism" in category
        or "biosynthesis" in category
        or "catabolism" in text
        or "utilization" in text
    ):
        return (
            "nutrient_metabolism",
            "TARGET_GAP",
        )

    # --------------------------------------------------
    # Uncharacterized / singlets
    # --------------------------------------------------

    if (
        "uncharacterized" in name
        or "uncharacterized" in category
    ):
        return (
            "uncharacterized",
            "DISCOVERY_GAP",
        )

    if (
        "singlet" in name
        or "singlet" in category
    ):
        return (
            "singlet",
            "LOW_PRIORITY",
        )

    return (
        "other",
        "REVIEW",
    )


def prepare_residual(path, suffix):
    df = pd.read_csv(
        path
    )

    needed = {
        "imodulon",
        "observed_delta",
        "anchor_reconstruction",
        "residual_delta",
    }

    missing = (
        needed
        - set(
            df.columns
        )
    )

    if missing:
        raise RuntimeError(
            f"{path.name} missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    df = df[
        [
            "imodulon",
            "observed_delta",
            "anchor_reconstruction",
            "residual_delta",
        ]
    ].copy()

    df = df.rename(
        columns={
            "observed_delta":
                f"observed_delta_{suffix}",

            "anchor_reconstruction":
                f"anchor_reconstruction_{suffix}",

            "residual_delta":
                f"residual_delta_{suffix}",
        }
    )

    return df


def main():
    print("=" * 78)
    print("CF-State Phase 2 - Audit 06")
    print("Residual programme anchoring-gap map")
    print("=" * 78)

    for path in [
        CATALOGUE,
        EXT_RESIDUAL,
        INTERNAL_RESIDUAL,
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
    # Published programme catalogue
    # ------------------------------------------------------------

    catalogue = pd.read_csv(
        CATALOGUE
    )

    required_catalogue = {
        "name",
        "regulator_readable",
        "function",
        "category",
    }

    missing = (
        required_catalogue
        - set(
            catalogue.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Programme catalogue missing: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if len(catalogue) != 116:
        raise RuntimeError(
            "Expected 116 published programmes, found "
            + str(
                len(catalogue)
            )
        )

    # ------------------------------------------------------------
    # Residuals
    # ------------------------------------------------------------

    external = prepare_residual(
        EXT_RESIDUAL,
        "GSE203348",
    )

    internal = prepare_residual(
        INTERNAL_RESIDUAL,
        "GSE120602",
    )

    residuals = external.merge(
        internal,
        on="imodulon",
        how="inner",
        validate="one_to_one",
    )

    if len(residuals) != 116:
        raise RuntimeError(
            "Expected 116 shared residual programmes, found "
            + str(
                len(residuals)
            )
        )

    # ------------------------------------------------------------
    # Merge frozen annotations
    # ------------------------------------------------------------

    annotation = catalogue[
        [
            col
            for col in [
                "name",
                "regulator_readable",
                "function",
                "category",
                "n_genes",
                "exp_var",
                "precision",
                "recall",
            ]
            if col in catalogue.columns
        ]
    ].copy()

    annotation = annotation.rename(
        columns={
            "name":
                "imodulon"
        }
    )

    df = residuals.merge(
        annotation,
        on="imodulon",
        how="left",
        validate="one_to_one",
    )

    if df["function"].isna().all():
        raise RuntimeError(
            "Published programme annotation merge failed"
        )

    # ------------------------------------------------------------
    # Assign perturbation families
    # ------------------------------------------------------------

    families = df.apply(
        assign_anchor_family,
        axis=1,
        result_type="expand",
    )

    families.columns = [
        "anchor_family",
        "anchor_status",
    ]

    df = pd.concat(
        [
            df,
            families,
        ],
        axis=1,
    )

    # ------------------------------------------------------------
    # Independent residual energy
    # ------------------------------------------------------------

    residual = (
        df[
            "residual_delta_GSE203348"
        ]
        .to_numpy(
            dtype=float
        )
    )

    total_energy = float(
        np.square(
            residual
        ).sum()
    )

    if total_energy <= 0:
        raise RuntimeError(
            "Independent residual has zero energy"
        )

    df[
        "independent_residual_energy"
    ] = np.square(
        df[
            "residual_delta_GSE203348"
        ]
    )

    df[
        "independent_residual_energy_fraction"
    ] = (
        df[
            "independent_residual_energy"
        ]
        / total_energy
    )

    df[
        "abs_residual_GSE203348"
    ] = (
        df[
            "residual_delta_GSE203348"
        ]
        .abs()
    )

    df[
        "abs_residual_GSE120602"
    ] = (
        df[
            "residual_delta_GSE120602"
        ]
        .abs()
    )

    # Rank both residual sets independently.
    df[
        "rank_GSE203348"
    ] = (
        df[
            "abs_residual_GSE203348"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    df[
        "rank_GSE120602"
    ] = (
        df[
            "abs_residual_GSE120602"
        ]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    df[
        "top20_both_experiments"
    ] = (
        (
            df[
                "rank_GSE203348"
            ]
            <= 20
        )
        &
        (
            df[
                "rank_GSE120602"
            ]
            <= 20
        )
    )

    df[
        "residual_same_direction"
    ] = (
        np.sign(
            df[
                "residual_delta_GSE203348"
            ]
        )
        ==
        np.sign(
            df[
                "residual_delta_GSE120602"
            ]
        )
    )

    # ------------------------------------------------------------
    # Rank independent programmes
    # ------------------------------------------------------------

    ranked = df.sort_values(
        "independent_residual_energy_fraction",
        ascending=False,
    ).copy()

    ranked[
        "cumulative_independent_residual_energy"
    ] = (
        ranked[
            "independent_residual_energy_fraction"
        ]
        .cumsum()
    )

    ranked.to_csv(
        OUT
        / "annotated_independent_residual_programmes.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Summarise by anchor family
    # ------------------------------------------------------------

    family = (
        df.groupby(
            [
                "anchor_family",
                "anchor_status",
            ],
            dropna=False,
        )
        .agg(
            n_programmes=(
                "imodulon",
                "size",
            ),
            residual_energy=(
                "independent_residual_energy",
                "sum",
            ),
            mean_abs_residual=(
                "abs_residual_GSE203348",
                "mean",
            ),
            max_abs_residual=(
                "abs_residual_GSE203348",
                "max",
            ),
            recurrent_top20=(
                "top20_both_experiments",
                "sum",
            ),
        )
        .reset_index()
    )

    family[
        "residual_energy_fraction"
    ] = (
        family[
            "residual_energy"
        ]
        / total_energy
    )

    # Add top individual programmes for each family.
    top_names = {}

    for family_name, sub in df.groupby(
        "anchor_family"
    ):
        names = (
            sub.sort_values(
                "independent_residual_energy_fraction",
                ascending=False,
            )
            .head(5)[
                "imodulon"
            ]
            .tolist()
        )

        top_names[
            family_name
        ] = "; ".join(
            names
        )

    family[
        "top_programmes"
    ] = family[
        "anchor_family"
    ].map(
        top_names
    )

    # Priority applies only to gaps, not already-anchored families.
    priority_status = {
        "TARGET_GAP": 3,
        "DISCOVERY_GAP": 2,
        "PARTIAL_ANCHOR": 1,
        "REVIEW": 1,
        "EXISTING_ANCHOR": 0,
        "LOW_PRIORITY": 0,
    }

    family[
        "gap_weight"
    ] = (
        family[
            "anchor_status"
        ]
        .map(
            priority_status
        )
        .fillna(0)
    )

    family[
        "priority_score"
    ] = (
        family[
            "residual_energy_fraction"
        ]
        * family[
            "gap_weight"
        ]
    )

    family = family.sort_values(
        [
            "priority_score",
            "residual_energy_fraction",
        ],
        ascending=False,
    )

    family.to_csv(
        OUT
        / "controlled_perturbation_priority.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Published functional-category summary
    # ------------------------------------------------------------

    category = (
        df.groupby(
            "category",
            dropna=False,
        )
        .agg(
            n_programmes=(
                "imodulon",
                "size",
            ),
            residual_energy=(
                "independent_residual_energy",
                "sum",
            ),
        )
        .reset_index()
    )

    category[
        "residual_energy_fraction"
    ] = (
        category[
            "residual_energy"
        ]
        / total_energy
    )

    category = category.sort_values(
        "residual_energy_fraction",
        ascending=False,
    )

    category.to_csv(
        OUT
        / "published_category_residual_energy.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Recurrently unexplained programmes
    # ------------------------------------------------------------

    recurrent = (
        df.loc[
            df[
                "top20_both_experiments"
            ]
        ]
        .sort_values(
            "independent_residual_energy_fraction",
            ascending=False,
        )
    )

    recurrent.to_csv(
        OUT
        / "recurrent_unexplained_programmes.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Print key results
    # ------------------------------------------------------------

    print()
    print(
        "Top independent unexplained programmes"
    )

    display_cols = [
        "rank_GSE203348",
        "imodulon",
        "residual_delta_GSE203348",
        "independent_residual_energy_fraction",
        "function",
        "category",
        "anchor_family",
        "anchor_status",
    ]

    print(
        ranked[
            display_cols
        ]
        .head(20)
        .round(
            {
                "residual_delta_GSE203348": 3,
                "independent_residual_energy_fraction": 4,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Residual energy by proposed anchoring family"
    )

    print(
        family[
            [
                "anchor_family",
                "anchor_status",
                "n_programmes",
                "residual_energy_fraction",
                "recurrent_top20",
                "top_programmes",
            ]
        ]
        .round(
            {
                "residual_energy_fraction": 4,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Published aeruPRECISE category residual energy"
    )

    print(
        category[
            [
                "category",
                "n_programmes",
                "residual_energy_fraction",
            ]
        ]
        .head(15)
        .round(
            {
                "residual_energy_fraction": 4,
            }
        )
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Recurrently unexplained top-20 programmes "
        "in both biofilm experiments"
    )

    if recurrent.empty:
        print(
            "  none"
        )

    else:
        print(
            recurrent[
                [
                    "imodulon",
                    "residual_delta_GSE203348",
                    "residual_delta_GSE120602",
                    "residual_same_direction",
                    "anchor_family",
                    "function",
                ]
            ]
            .round(
                {
                    "residual_delta_GSE203348": 3,
                    "residual_delta_GSE120602": 3,
                }
            )
            .to_string(
                index=False
            )
        )

    # ------------------------------------------------------------
    # Simple audit checks
    # ------------------------------------------------------------

    energy_sum = float(
        ranked[
            "independent_residual_energy_fraction"
        ]
        .sum()
    )

    family_energy_sum = float(
        family[
            "residual_energy_fraction"
        ]
        .sum()
    )

    checks = {
        "116_programmes":
            len(df) == 116,

        "all_programmes_annotated":
            df[
                "category"
            ].notna().all(),

        "programme_energy_sums_to_one":
            abs(
                energy_sum
                - 1.0
            )
            < 1e-10,

        "family_energy_sums_to_one":
            abs(
                family_energy_sum
                - 1.0
            )
            < 1e-10,

        "every_programme_has_anchor_family":
            df[
                "anchor_family"
            ].notna().all(),
    }

    passed = all(
        checks.values()
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

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    top_gap_families = (
        family.loc[
            family[
                "anchor_status"
            ].isin(
                [
                    "TARGET_GAP",
                    "DISCOVERY_GAP",
                ]
            )
        ]
        .head(6)
    )

    lines = [
        "# CF-State Phase 2 — Audit 06",
        "",
        "## Objective",
        "",
        "Translate the independent GSE203348 residual left after "
        "stationary/persistence anchoring into a programme-level "
        "controlled-perturbation gap map.",
        "",
        "## Important interpretation rule",
        "",
        "The proposed anchor families are planning labels based on the "
        "published aeruPRECISE iModulon names/functions/categories. "
        "They are not new annotations and do not establish causality.",
        "",
        "## Highest-priority missing controlled families",
        "",
    ]

    for _, row in top_gap_families.iterrows():
        lines.append(
            "- "
            + str(
                row[
                    "anchor_family"
                ]
            )
            + ": "
            + format(
                row[
                    "residual_energy_fraction"
                ],
                ".2%",
            )
            + " of independent residual energy; top programmes: "
            + str(
                row[
                    "top_programmes"
                ]
            )
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "PASS: "
            + str(
                passed
            ),
            "",
            "Audit 07 should search specifically for independent, "
            "well-controlled perturbation datasets targeting the "
            "highest-priority missing families rather than simply "
            "adding more biofilm studies.",
        ]
    )

    (
        OUT
        / "AUDIT_06_SUMMARY.md"
    ).write_text(
        "\n".join(
            lines
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "Audit 06:",
        "PASS"
        if passed
        else "FAIL",
    )

    print(
        "Summary:",
        OUT
        / "AUDIT_06_SUMMARY.md",
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
