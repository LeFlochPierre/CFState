from pathlib import Path
import hashlib
import traceback

import numpy as np
import pandas as pd


ROOT = Path.cwd()
P4 = ROOT / "results/phase4"

H = P4 / "audit_03h_freeze_replicated_CF_airway_candidate"
M = P4 / "audit_04m2_PRJNA576508_blinded_residual_freeze"
N = P4 / "audit_04n_PRJNA576508_independent_03h_comparison"

OUT = ROOT / "results/phase5/audit_05a_frozen_alignment_attribution"

SAMPLES = [
    "EM6", "EM13", "EM24", "EM30", "EM33",
    "EM35", "EM36", "EM37", "EM43",
    "EM48", "EM49", "EM53", "EM54"
]

NORMS = ["FROZEN_5567_TPM", "SOFTCORE_4923_TPM"]

REFS = [
    "PRIMARY_CFBE_ALL4",
    "SENSITIVITY_CFBE_C_PAIR",
    "SENSITIVITY_CFBE_R_PAIR"
]


def verify(label, condition):
    if not bool(condition):
        raise RuntimeError("FAILED: " + label)

    print(label, "PASS", flush=True)


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def main():
    print("=" * 72)
    print("CF-State Phase 5 — Audit 05a")
    print("Frozen 108-programme consensus alignment attribution")
    print("=" * 72)

    files = {
        "03h_status": H / "AUDIT_03H_STATUS.txt",
        "04m2_status": M / "AUDIT_04M2_STATUS.txt",
        "04n_status": N / "AUDIT_04N_STATUS.txt",
        "candidate": H / "FROZEN_CF_airway_candidate_vectors.csv",
        "frozen_18": H / "conservatively_replicated_programmes.csv",
        "comparisons": N / "FROZEN_78_all_candidate_comparisons.csv",
        "direction_observations":
            N / "FROZEN_1404_programme_direction_observations.csv",
        "script": Path(__file__)
    }

    for key, path in files.items():
        verify("input_" + key, path.is_file())

    statuses = {
        "03h_status":
            "REPLICATED_CF_AIRWAY_PROGRAMME_CANDIDATE_FROZEN",
        "04m2_status":
            "PRJNA576508_ATLANTA_13_BLINDED_CONTROLLED_STATE_RESIDUALS_FROZEN",
        "04n_status":
            "PRJNA576508_INDEPENDENT_03H_DESCRIPTIVE_COMPARISON_FROZEN"
    }

    for key, expected in statuses.items():
        verify(
            key + "_frozen",
            files[key].read_text().strip() == expected
        )

    print()
    print("========== ORIGINAL FROZEN CANDIDATE ==========")

    candidate = pd.read_csv(
        files["candidate"],
        index_col=0
    )

    candidate.index = (
        candidate.index.astype(str).str.strip()
    )

    verify(
        "candidate_108_unique",
        len(candidate) == 108
        and candidate.index.is_unique
    )

    v = pd.to_numeric(
        candidate["two_cohort_consensus_unit"],
        errors="raise"
    ).to_numpy(dtype=float)

    norm_v = float(np.linalg.norm(v))

    verify(
        "consensus_is_unit_vector",
        np.isfinite(v).all()
        and abs(norm_v - 1) < 1e-6
    )

    programmes = list(candidate.index)

    frozen18 = pd.read_csv(files["frozen_18"])

    names18 = (
        frozen18["imodulon"]
        .astype(str).str.strip().tolist()
    )

    verify(
        "original_18_unchanged",
        len(names18) == 18
        and len(set(names18)) == 18
        and set(names18).issubset(set(programmes))
    )

    selected18 = set(names18)

    is18 = np.array([
        programme in selected18
        for programme in programmes
    ])

    saved = pd.read_csv(files["comparisons"])

    key_cols = [
        "normalisation",
        "reference",
        "sample"
    ]

    verify(
        "78_saved_comparisons",
        len(saved) == 78
        and not saved.duplicated(key_cols).any()
    )

    saved = saved.set_index(key_cols)

    directions = pd.read_csv(
        files["direction_observations"]
    )

    verify(
        "1404_frozen_direction_observations",
        len(directions) == 1404
    )

    contribution_rows = []
    sample_rows = []
    loo_rows = []

    print()
    print("========== REPRODUCE ALL 78 FROZEN COSINES ==========")

    for norm in NORMS:
        for ref in REFS:
            path = M / (
                "FROZEN_108_residuals__"
                + norm + "__" + ref + ".csv"
            )

            verify(
                "residual_file_" + norm + "_" + ref,
                path.is_file()
            )

            files["residual_" + norm + "_" + ref] = path

            table = pd.read_csv(path).set_index("imodulon")

            verify(
                "residual_structure_" + norm + "_" + ref,
                table.index.is_unique
                and set(table.index) == set(programmes)
                and set(table.columns) == set(SAMPLES)
            )

            y = table.reindex(
                index=programmes,
                columns=SAMPLES
            ).to_numpy(dtype=float)

            norms_y = np.linalg.norm(y, axis=0)

            verify(
                "finite_nonzero_residuals_" + norm + "_" + ref,
                np.isfinite(y).all()
                and (norms_y > 1e-12).all()
            )

            contribution = (
                v[:, None] * y
                / (norm_v * norms_y[None, :])
            )

            reconstructed = contribution.sum(axis=0)

            recorded = np.array([
                float(
                    saved.loc[
                        (norm, ref, sample),
                        "cosine_to_two_cohort_consensus_unit"
                    ]
                )
                for sample in SAMPLES
            ])

            verify(
                "cosine_reconstruction_" + norm + "_" + ref,
                np.allclose(
                    reconstructed,
                    recorded,
                    atol=1e-10,
                    rtol=0
                )
            )

            for j, sample in enumerate(SAMPLES):
                full = float(reconstructed[j])

                contribution18 = float(
                    contribution[is18, j].sum()
                )

                contribution90 = float(
                    contribution[~is18, j].sum()
                )

                sample_rows.append({
                    "normalisation": norm,
                    "reference": ref,
                    "sample": sample,
                    "full_108_cosine": full,
                    "signed_contribution_frozen_18":
                        contribution18,
                    "signed_contribution_other_90":
                        contribution90
                })

                for i, programme in enumerate(programmes):
                    contribution_rows.append({
                        "normalisation": norm,
                        "reference": ref,
                        "sample": sample,
                        "imodulon": programme,
                        "frozen_18": bool(is18[i]),
                        "consensus_loading": float(v[i]),
                        "residual_activity": float(y[i, j]),
                        "signed_cosine_contribution":
                            float(contribution[i, j])
                    })

                    if norm != NORMS[0] or ref != REFS[0]:
                        continue

                    # Descriptive leave-one-coordinate-out
                    # calculation. No programme is reselected.
                    v2 = float(
                        np.dot(v, v) - v[i] ** 2
                    )

                    y2 = float(
                        norms_y[j] ** 2 - y[i, j] ** 2
                    )

                    if v2 <= 1e-12 or y2 <= 1e-12:
                        raise RuntimeError(
                            "Unstable leave-one-coordinate comparison"
                        )

                    numerator = (
                        full * norm_v * norms_y[j]
                        - v[i] * y[i, j]
                    )

                    cosine_without = float(
                        numerator / np.sqrt(v2 * y2)
                    )

                    loo_rows.append({
                        "sample": sample,
                        "imodulon": programme,
                        "frozen_18": bool(is18[i]),
                        "full_cosine": full,
                        "cosine_without_coordinate":
                            cosine_without,
                        "change_full_minus_without":
                            full - cosine_without
                    })

            print(norm, ref, "REPRODUCED", flush=True)

    contributions = pd.DataFrame(contribution_rows)
    group = pd.DataFrame(sample_rows)
    loo = pd.DataFrame(loo_rows)

    verify(
        "expected_8424_programme_contributions",
        len(contributions) == 78 * 108
    )

    verify(
        "expected_78_group_decompositions",
        len(group) == 78
    )

    verify(
        "expected_1404_primary_coordinate_influences",
        len(loo) == 13 * 108
    )

    verify(
        "group_decomposition_exact",
        np.allclose(
            group["signed_contribution_frozen_18"]
            + group["signed_contribution_other_90"],
            group["full_108_cosine"],
            atol=1e-12
        )
    )

    print()
    print("========== PRIMARY PROGRAMME ATTRIBUTION ==========")

    primary = contributions.loc[
        (contributions["normalisation"] == NORMS[0])
        & (contributions["reference"] == REFS[0])
    ]

    primary_summary = (
        primary.groupby("imodulon", sort=False)
        .agg(
            frozen_18=("frozen_18", "first"),
            consensus_loading=("consensus_loading", "first"),
            median_signed_contribution=(
                "signed_cosine_contribution", "median"
            ),
            mean_signed_contribution=(
                "signed_cosine_contribution", "mean"
            ),
            min_signed_contribution=(
                "signed_cosine_contribution", "min"
            ),
            max_signed_contribution=(
                "signed_cosine_contribution", "max"
            ),
            positive_contribution_samples=(
                "signed_cosine_contribution",
                lambda x: int((x > 0).sum())
            )
        )
        .reset_index()
    )

    loo_summary = (
        loo.groupby("imodulon", sort=False)[
            "change_full_minus_without"
        ]
        .median()
        .rename("median_leave_one_out_support")
    )

    primary_summary = primary_summary.join(
        loo_summary,
        on="imodulon"
    )

    verify(
        "108_programme_summary",
        len(primary_summary) == 108
    )

    print()
    print("========== VERIFY ORIGINAL 18 DIRECTIONS ==========")

    primary_direction = directions.loc[
        (directions["normalisation"] == NORMS[0])
        & (directions["reference"] == REFS[0])
    ]

    verify(
        "primary_234_direction_observations",
        len(primary_direction) == 13 * 18
    )

    logged = primary_direction.set_index(
        ["sample", "imodulon"]
    )

    for row in primary.loc[
        primary["frozen_18"]
    ].itertuples(index=False):

        recorded = float(
            logged.loc[
                (row.sample, row.imodulon),
                "observed_residual_activity"
            ]
        )

        if abs(recorded - row.residual_activity) > 1e-9:
            raise RuntimeError(
                "Directional observation mismatch: "
                + row.imodulon
            )

    print("Original 18 direction activities reproduced PASS")

    print()
    print("========== TOP SUPPORTING PROGRAMMES ==========")

    print(
        primary_summary.nlargest(
            10, "median_signed_contribution"
        )[
            [
                "imodulon",
                "frozen_18",
                "median_signed_contribution",
                "median_leave_one_out_support"
            ]
        ].round(5).to_string(index=False)
    )

    print()
    print("========== TOP OPPOSING PROGRAMMES ==========")

    print(
        primary_summary.nsmallest(
            10, "median_signed_contribution"
        )[
            [
                "imodulon",
                "frozen_18",
                "median_signed_contribution",
                "median_leave_one_out_support"
            ]
        ].round(5).to_string(index=False)
    )

    print()
    print("========== 18 VERSUS 90 DECOMPOSITION ==========")

    primary_group = group.loc[
        (group["normalisation"] == NORMS[0])
        & (group["reference"] == REFS[0])
    ]

    print(
        primary_group[
            [
                "sample",
                "full_108_cosine",
                "signed_contribution_frozen_18",
                "signed_contribution_other_90"
            ]
        ].round(5).to_string(index=False)
    )

    print()
    print("========== ALL SIX VARIANTS ==========")

    print(
        group.groupby(
            ["normalisation", "reference"],
            sort=False
        )[
            [
                "full_108_cosine",
                "signed_contribution_frozen_18",
                "signed_contribution_other_90"
            ]
        ].median().round(5).to_string()
    )

    print()
    print("========== FREEZE AUDIT 05a ==========")

    contributions.to_csv(
        OUT / "FROZEN_8424_programme_signed_contributions.csv",
        index=False
    )

    group.to_csv(
        OUT / "FROZEN_78_signed_18_vs_90_decomposition.csv",
        index=False
    )

    loo.to_csv(
        OUT / "FROZEN_1404_primary_leave_one_coordinate_out.csv",
        index=False
    )

    primary_summary.to_csv(
        OUT / "FROZEN_108_primary_programme_attribution.csv",
        index=False
    )

    primary_summary.loc[
        primary_summary["frozen_18"]
    ].to_csv(
        OUT / "FROZEN_18_original_programme_attribution.csv",
        index=False
    )

    pd.DataFrame([
        {
            "input": key,
            "path": str(path),
            "sha256": sha256(path)
        }
        for key, path in files.items()
    ]).to_csv(
        OUT / "input_provenance_sha256.csv",
        index=False
    )

    notes = (
        "Descriptive post-unblinding decomposition of frozen Audit 04n cosine.\n"
        "Signed coordinate contributions sum exactly to full 108-dimensional cosine.\n"
        "The 18 versus 90 decomposition is signed, not a percentage breakdown.\n"
        "Leave-one-coordinate comparisons are descriptive, not causal effects.\n"
        "No target changes, refitting, significance tests or sample exclusions.\n"
        "Shared CFBE reference and unresolved donor independence remain limitations.\n"
    )

    (
        OUT / "AUDIT_05A_METHODS.txt"
    ).write_text(notes)

    status = (
        "CFSTATE_PHASE5_FROZEN_"
        "CONSENSUS_ALIGNMENT_ATTRIBUTION_COMPLETE"
    )

    (
        OUT / "AUDIT_05A_STATUS.txt"
    ).write_text(status + "\n")

    print()
    print("STATUS:", status)


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 05a output exists. Refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()
    except Exception:
        (
            OUT / "AUDIT_05A_STATUS.txt"
        ).write_text(
            "CFSTATE_PHASE5_AUDIT_05A_REVIEW_REQUIRED\n"
        )

        (
            OUT / "AUDIT_05A_FAILURE.txt"
        ).write_text(traceback.format_exc())

        raise
