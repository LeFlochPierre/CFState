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

OUT = ROOT / "results/phase5/audit_05b_Zur2_and_shared_reference_diagnostic"

NORMS = ["FROZEN_5567_TPM", "SOFTCORE_4923_TPM"]

REFS = [
    "PRIMARY_CFBE_ALL4",
    "SENSITIVITY_CFBE_C_PAIR",
    "SENSITIVITY_CFBE_R_PAIR"
]

SAMPLES = [
    "EM6", "EM13", "EM24", "EM30", "EM33",
    "EM35", "EM36", "EM37", "EM43",
    "EM48", "EM49", "EM53", "EM54"
]


def require(name, condition):
    success = bool(condition)
    print(name, "PASS" if success else "FAIL", flush=True)

    if not success:
        raise RuntimeError("Failed check: " + name)


def cosine(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator <= 1e-12:
        raise RuntimeError("Zero-norm cosine comparison")

    return float(np.dot(a, b) / denominator)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)

    return h.hexdigest()


def main():
    print("=" * 76)
    print("CF-State Phase 5 — Audit 05b")
    print("Zur-2 dependence and shared-reference diagnostic")
    print("=" * 76)

    inputs = {
        "03h_status": H / "AUDIT_03H_STATUS.txt",
        "04m2_status": M / "AUDIT_04M2_STATUS.txt",
        "04n_status": N / "AUDIT_04N_STATUS.txt",
        "candidate": H / "FROZEN_CF_airway_candidate_vectors.csv",
        "frozen18": H / "conservatively_replicated_programmes.csv",
        "previous_comparisons":
            N / "FROZEN_78_all_candidate_comparisons.csv",
        "script": Path(__file__)
    }

    for name, path in inputs.items():
        require("input_" + name, path.is_file())

    statuses = {
        "03h_status":
            "REPLICATED_CF_AIRWAY_PROGRAMME_CANDIDATE_FROZEN",
        "04m2_status":
            "PRJNA576508_ATLANTA_13_BLINDED_CONTROLLED_STATE_RESIDUALS_FROZEN",
        "04n_status":
            "PRJNA576508_INDEPENDENT_03H_DESCRIPTIVE_COMPARISON_FROZEN"
    }

    for name, expected in statuses.items():
        require(
            name + "_frozen",
            inputs[name].read_text().strip() == expected
        )

    candidate = pd.read_csv(
        inputs["candidate"], index_col=0
    )

    candidate.index = candidate.index.astype(str).str.strip()

    require(
        "candidate_108_unique",
        len(candidate) == 108 and candidate.index.is_unique
    )

    programmes = list(candidate.index)

    v = pd.to_numeric(
        candidate["two_cohort_consensus_unit"],
        errors="raise"
    ).to_numpy(dtype=float)

    require(
        "candidate_unit_norm",
        np.isfinite(v).all()
        and np.isclose(np.linalg.norm(v), 1, atol=1e-6)
    )

    registry = pd.read_csv(inputs["frozen18"])

    selected18 = set(
        registry["imodulon"].astype(str).str.strip()
    )

    require(
        "frozen_18_unchanged",
        len(registry) == 18
        and len(selected18) == 18
        and selected18.issubset(set(programmes))
    )

    require(
        "Zur2_not_in_frozen_18",
        "Zur-2" in programmes and "Zur-2" not in selected18
    )

    masks = {
        "without_Zur2":
            np.array([p != "Zur-2" for p in programmes]),
        "frozen_18":
            np.array([p in selected18 for p in programmes]),
        "other_90":
            np.array([p not in selected18 for p in programmes])
    }

    saved = pd.read_csv(
        inputs["previous_comparisons"]
    ).set_index(
        ["normalisation", "reference", "sample"]
    )

    require(
        "saved_78_unique",
        len(saved) == 78 and saved.index.is_unique
    )

    rows = []
    shared_rows = []

    print()
    print("========== POST HOC ROBUSTNESS ==========")

    for norm in NORMS:
        for ref in REFS:
            path = M / (
                "FROZEN_108_residuals__"
                + norm + "__" + ref + ".csv"
            )

            require(
                "residual_exists_" + norm + "_" + ref,
                path.is_file()
            )

            inputs["residual_" + norm + "_" + ref] = path

            table = pd.read_csv(path).set_index("imodulon")

            require(
                "residual_structure_" + norm + "_" + ref,
                table.shape == (108, 13)
                and table.index.is_unique
                and set(table.index) == set(programmes)
                and set(table.columns) == set(SAMPLES)
            )

            matrix = table.reindex(
                index=programmes,
                columns=SAMPLES
            ).to_numpy(dtype=float)

            require(
                "residual_values_finite_" + norm + "_" + ref,
                np.isfinite(matrix).all()
            )

            mean_residual = matrix.mean(axis=1)
            shared_direction = cosine(mean_residual, v)
            shared_dot = float(np.dot(mean_residual, v))

            centered = matrix - mean_residual[:, None]

            for j, sample in enumerate(SAMPLES):
                r = matrix[:, j]
                full = cosine(r, v)

                recorded = float(
                    saved.loc[
                        (norm, ref, sample),
                        "cosine_to_two_cohort_consensus_unit"
                    ]
                )

                if abs(full - recorded) > 1e-10:
                    raise RuntimeError(
                        "Frozen cosine mismatch: " + sample
                    )

                results = {
                    key: cosine(r[mask], v[mask])
                    for key, mask in masks.items()
                }

                rows.append({
                    "normalisation": norm,
                    "reference": ref,
                    "sample": sample,
                    "frozen_full_108_cosine": full,
                    "posthoc_without_Zur2_cosine":
                        results["without_Zur2"],
                    "posthoc_frozen18_only_cosine":
                        results["frozen_18"],
                    "posthoc_other90_only_cosine":
                        results["other_90"],
                    "Zur2_signed_coordinate_contribution":
                        float(
                            r[programmes.index("Zur-2")]
                            * v[programmes.index("Zur-2")]
                            / np.linalg.norm(r)
                        )
                })

                shared_rows.append({
                    "normalisation": norm,
                    "reference": ref,
                    "sample": sample,
                    "candidate_dot_full_residual":
                        float(np.dot(r, v)),
                    "candidate_dot_shared_mean":
                        shared_dot,
                    "candidate_dot_centered_residual":
                        float(np.dot(centered[:, j], v)),
                    "cosine_shared_mean_to_candidate":
                        shared_direction
                })

            print(norm, ref, "DONE", flush=True)

    result = pd.DataFrame(rows)
    shared = pd.DataFrame(shared_rows)

    require(
        "78_comparisons_complete",
        len(result) == 78 and len(shared) == 78
    )

    require(
        "shared_decomposition_reconstructs",
        np.allclose(
            shared["candidate_dot_shared_mean"]
            + shared["candidate_dot_centered_residual"],
            shared["candidate_dot_full_residual"],
            atol=1e-9,
            rtol=0
        )
    )

    summary = (
        result.groupby(
            ["normalisation", "reference"],
            sort=False
        )
        .agg(
            median_full_cosine=(
                "frozen_full_108_cosine", "median"
            ),
            positive_without_Zur2=(
                "posthoc_without_Zur2_cosine",
                lambda x: int((x > 0).sum())
            ),
            median_without_Zur2=(
                "posthoc_without_Zur2_cosine",
                "median"
            ),
            positive_frozen18_only=(
                "posthoc_frozen18_only_cosine",
                lambda x: int((x > 0).sum())
            ),
            median_frozen18_only=(
                "posthoc_frozen18_only_cosine",
                "median"
            ),
            positive_other90_only=(
                "posthoc_other90_only_cosine",
                lambda x: int((x > 0).sum())
            ),
            median_other90_only=(
                "posthoc_other90_only_cosine",
                "median"
            )
        )
        .reset_index()
    )

    print()
    print("========== PRIMARY SAMPLE COMPARISON ==========")

    primary = result.loc[
        (result["normalisation"] == NORMS[0])
        & (result["reference"] == REFS[0])
    ]

    print(
        primary[
            [
                "sample",
                "frozen_full_108_cosine",
                "posthoc_without_Zur2_cosine",
                "posthoc_frozen18_only_cosine",
                "posthoc_other90_only_cosine"
            ]
        ].round(5).to_string(index=False)
    )

    print()
    print("========== ALL SIX VARIANTS ==========")
    print(summary.round(5).to_string(index=False))

    print()
    print("========== SHARED COMPONENT ==========")

    primary_shared = shared.loc[
        (shared["normalisation"] == NORMS[0])
        & (shared["reference"] == REFS[0])
    ]

    print(
        primary_shared[
            [
                "sample",
                "candidate_dot_full_residual",
                "candidate_dot_shared_mean",
                "candidate_dot_centered_residual"
            ]
        ].round(5).to_string(index=False)
    )

    print()
    print("Primary mean-residual cosine to candidate:",
          round(float(
              primary_shared["cosine_shared_mean_to_candidate"].iloc[0]
          ), 6))

    print()
    print("========== FREEZE POST HOC DIAGNOSTIC ==========")

    result.to_csv(
        OUT / "POSTHOC_78_Zur2_and_subset_robustness.csv",
        index=False
    )

    shared.to_csv(
        OUT / "POSTHOC_78_shared_mean_decomposition.csv",
        index=False
    )

    summary.to_csv(
        OUT / "POSTHOC_six_variant_summary.csv",
        index=False
    )

    pd.DataFrame([
        {
            "input": key,
            "path": str(path),
            "sha256": sha256(path)
        }
        for key, path in inputs.items()
    ]).to_csv(
        OUT / "input_provenance_sha256.csv",
        index=False
    )

    (OUT / "AUDIT_05B_METHODS.txt").write_text(
        "Post hoc, descriptive robustness only.\n"
        "The frozen 108-programme primary comparison is unchanged.\n"
        "Recomputed cosines after removing Zur-2 and within "
        "the original 18 and other 90 programme coordinates.\n"
        "The Atlanta cohort-mean residual is decomposed from "
        "within-cohort deviations.\n"
        "Removing the cohort mean removes both any reference-"
        "derived common offset AND true shared infection biology.\n"
        "These diagnostics cannot establish CF specificity.\n"
        "No new candidate or programme subset is promoted.\n"
        "No patient-level statistical tests performed.\n"
    )

    status = (
        "CFSTATE_PHASE5_05B_REFERENCE_AND_ZUR2_"
        "ROBUSTNESS_DIAGNOSTIC_COMPLETE"
    )

    (OUT / "AUDIT_05B_STATUS.txt").write_text(status + "\n")

    print()
    print("STATUS:", status)


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 05b exists. Refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()
    except Exception:
        (OUT / "AUDIT_05B_STATUS.txt").write_text(
            "CFSTATE_PHASE5_05B_REVIEW_REQUIRED\n"
        )
        (OUT / "AUDIT_05B_FAILURE.txt").write_text(
            traceback.format_exc()
        )
        raise
