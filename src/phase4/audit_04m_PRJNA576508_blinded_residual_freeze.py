from pathlib import Path
import hashlib
import traceback

import numpy as np
import pandas as pd


ROOT = Path.cwd()
P4 = ROOT / "results/phase4"

K4 = P4 / "audit_04k4_PRJNA576508_blinded_softcore_projection"
L = P4 / "audit_04l_PRJNA576508_blinded_projection_QC_lock"

LIBRARY = P4 / (
    "audit_02b_GSE119356_CF_state_interpretation/"
    "controlled_state_reference_library.csv"
)

STABLE = P4 / (
    "audit_03e_softcore_projection_fidelity/"
    "PRJEB24688_empirically_stable_primary_programmes.csv"
)

MANIFEST = P4 / (
    "audit_04a_PRJNA576508_third_cohort_preflight/"
    "PRJNA576508_Atlanta_sputum_manifest.csv"
)

OUT = P4 / "audit_04m_PRJNA576508_blinded_residual_freeze"

SAMPLES = [
    "EM6", "EM13", "EM24", "EM30", "EM33",
    "EM35", "EM36", "EM37", "EM43",
    "EM48", "EM49", "EM53", "EM54"
]

NORMS = [
    "FROZEN_5567_TPM",
    "SOFTCORE_4923_TPM"
]

REFS = [
    "PRIMARY_CFBE_ALL4",
    "SENSITIVITY_CFBE_C_PAIR",
    "SENSITIVITY_CFBE_R_PAIR"
]

CHECKS = []


def check(name, value):
    okay = bool(value)
    CHECKS.append({"check": name, "pass": okay})
    print(name, "PASS" if okay else "FAIL", flush=True)

    if not okay:
        raise RuntimeError("Failed check: " + name)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(4 * 1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


def load_numeric(path, id_column):
    df = pd.read_csv(path)

    check(
        path.name + "_identifier",
        id_column in df.columns
    )

    df[id_column] = (
        df[id_column].astype(str).str.strip()
    )

    check(
        path.name + "_unique_identifiers",
        df[id_column].is_unique
        and df[id_column].notna().all()
    )

    df = df.set_index(id_column)
    df = df.apply(pd.to_numeric, errors="raise")

    check(
        path.name + "_finite",
        np.isfinite(df.to_numpy(dtype=float)).all()
    )

    return df


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na <= 1e-12 or nb <= 1e-12:
        return float("nan")

    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 76)
    print("CF-State Phase 4 — Audit 04m")
    print("Blinded controlled-state residualisation freeze")
    print("=" * 76)

    status_path = L / "AUDIT_04L_STATUS.txt"

    inputs = {
        "04l_status": status_path,
        "controlled_state_library": LIBRARY,
        "stable_programmes": STABLE,
        "Atlanta_manifest": MANIFEST,
        "script": Path(__file__)
    }

    for label, path in inputs.items():
        check(
            "input_" + label + "_exists",
            path.is_file()
        )

    check(
        "04l_design_frozen",
        status_path.read_text().strip()
        == (
            "PRJNA576508_ATLANTA_BLINDED_"
            "PROJECTION_QC_AND_DESIGN_LOCK_FROZEN"
        )
    )

    print()
    print("========== 108-PROGRAMME SPACE ==========")

    stable_table = pd.read_csv(STABLE)

    check(
        "stable_imodulon_column",
        "imodulon" in stable_table.columns
    )

    programmes = (
        stable_table["imodulon"]
        .astype(str).str.strip().tolist()
    )

    check(
        "exact_108_stable_programmes",
        len(programmes) == 108
        and len(set(programmes)) == 108
    )

    print()
    print("========== CONTROLLED-STATE LIBRARY ==========")

    library = load_numeric(LIBRARY, "imodulon")

    check(
        "library_contains_all_108",
        set(programmes).issubset(set(library.index))
    )

    check(
        "exact_11_controlled_axes",
        library.shape[1] == 11
        and library.columns.is_unique
    )

    original_basis = (
        library.reindex(programmes)
        .to_numpy(dtype=float)
    )

    norms = np.linalg.norm(
        original_basis, axis=0
    )

    check(
        "all_11_axes_have_nonzero_norm",
        np.isfinite(norms).all()
        and (norms > 0).all()
    )

    # Exact Audit 03f approach: normalize columns,
    # then least-squares project each activity vector.
    B = original_basis / norms

    rank = int(np.linalg.matrix_rank(B))

    print("Controlled-state dimensions:", B.shape)
    print("Controlled-state basis rank:", rank)

    check(
        "controlled_basis_nonzero_rank",
        1 <= rank <= 11
    )

    print()
    print("========== ATLANTA SAMPLE MANIFEST ==========")

    metadata = pd.read_csv(MANIFEST, dtype=str)

    check(
        "manifest_sample_label",
        "short_sample_label" in metadata.columns
    )

    metadata["short_sample_label"] = (
        metadata["short_sample_label"]
        .astype(str).str.strip()
    )

    selected_metadata = metadata.loc[
        metadata["short_sample_label"].isin(SAMPLES)
    ].copy()

    check(
        "manifest_exact_13",
        len(selected_metadata) == 13
        and set(selected_metadata["short_sample_label"])
        == set(SAMPLES)
    )

    check(
        "manifest_unique_sample_labels",
        selected_metadata["short_sample_label"].is_unique
    )

    check(
        "manifest_unique_sample_accessions",
        "sample_accession" in selected_metadata.columns
        and selected_metadata["sample_accession"].is_unique
    )

    donor_columns = [
        name for name in selected_metadata.columns
        if any(
            word in name.lower()
            for word in [
                "patient",
                "donor",
                "participant",
                "individual",
                "subject"
            ]
        )
    ]

    print("Explicit donor columns:", donor_columns)

    # Accessions identify samples, not necessarily donors.
    # Even if a donor-like column exists, independence
    # requires checking its documented provenance.
    donor_status = "NOT_VERIFIED"

    print("Donor independence:", donor_status)

    print()
    print("========== RESIDUALISE SIX FROZEN VARIANTS ==========")

    residuals = {}
    activities = {}
    fits = []
    outputs = {}

    for norm in NORMS:
        for reference in REFS:
            key = (norm, reference)

            path = K4 / (
                "FROZEN_108_activities__"
                + norm + "__" + reference + ".csv"
            )

            check(
                "activity_file_" + norm + "_" + reference,
                path.is_file()
            )

            inputs[
                "activities_" + norm + "_" + reference
            ] = path

            table = load_numeric(path, "imodulon")

            check(
                "activity_108_" + norm + "_" + reference,
                table.shape == (108, 13)
                and set(table.index) == set(programmes)
                and set(table.columns) == set(SAMPLES)
            )

            y = (
                table.reindex(
                    index=programmes,
                    columns=SAMPLES
                ).to_numpy(dtype=float)
            )

            beta = np.linalg.lstsq(
                B, y, rcond=None
            )[0]

            fitted = B @ beta
            residual = y - fitted

            check(
                "residual_finite_" + norm + "_" + reference,
                np.isfinite(residual).all()
                and np.isfinite(fitted).all()
                and np.isfinite(beta).all()
            )

            check(
                "activity_reconstructs_" + norm + "_" + reference,
                np.allclose(
                    fitted + residual,
                    y,
                    rtol=1e-10,
                    atol=1e-9
                )
            )

            # Residual must be orthogonal to the
            # 11-dimensional reference basis.
            orthogonality = (
                np.linalg.norm(B.T @ residual, axis=0)
                / np.maximum(
                    np.linalg.norm(y, axis=0),
                    1e-12
                )
            )

            check(
                "residual_orthogonal_" + norm + "_" + reference,
                (orthogonality <= 1e-8).all()
            )

            # Independent pseudoinverse reproduction.
            independent_residual = (
                y - B @ (np.linalg.pinv(B) @ y)
            )

            check(
                "pseudoinverse_agrees_" + norm + "_" + reference,
                np.allclose(
                    residual,
                    independent_residual,
                    atol=1e-8,
                    rtol=1e-10
                )
            )

            total = np.square(y).sum(axis=0)
            known = np.square(fitted).sum(axis=0)
            unknown = np.square(residual).sum(axis=0)

            check(
                "all_activity_energies_positive_"
                + norm + "_" + reference,
                (total > 1e-12).all()
            )

            known_fraction = known / total
            residual_fraction = unknown / total

            check(
                "known_plus_residual_energy_"
                + norm + "_" + reference,
                np.allclose(
                    known_fraction + residual_fraction,
                    1.0,
                    atol=1e-8,
                    rtol=0
                )
            )

            for j, sample in enumerate(SAMPLES):
                fits.append({
                    "normalisation": norm,
                    "reference": reference,
                    "sample": sample,
                    "known_state_energy_fraction_108":
                        float(known_fraction[j]),
                    "residual_energy_fraction_108":
                        float(residual_fraction[j]),
                    "activity_l2":
                        float(np.linalg.norm(y[:, j])),
                    "residual_l2":
                        float(np.linalg.norm(residual[:, j])),
                    "relative_residual_orthogonality":
                        float(orthogonality[j])
                })

            residuals[key] = residual
            activities[key] = y

            outputs[key] = {
                "residual": pd.DataFrame(
                    residual,
                    index=programmes,
                    columns=SAMPLES
                ),
                "coefficients": pd.DataFrame(
                    beta,
                    index=library.columns,
                    columns=SAMPLES
                )
            }

            print(norm, reference, "DONE", flush=True)

    check(
        "exact_78_residual_fits",
        len(fits) == 78
    )

    print()
    print("========== BLINDED RESIDUAL SENSITIVITY ==========")

    comparisons = []

    def compare(sample, kind, norm, alternative, first, second):
        comparisons.append({
            "sample": sample,
            "comparison": kind,
            "normalisation": norm,
            "alternative": alternative,
            "residual_cosine_108": cosine(first, second),
            "residual_relative_change_108": float(
                np.linalg.norm(second - first)
                / max(np.linalg.norm(first), 1e-12)
            )
        })

    for j, sample in enumerate(SAMPLES):
        for norm in NORMS:
            primary = residuals[
                (norm, REFS[0])
            ][:, j]

            for reference in REFS[1:]:
                other = residuals[
                    (norm, reference)
                ][:, j]

                compare(
                    sample,
                    "reference",
                    norm,
                    reference,
                    primary,
                    other
                )

        for reference in REFS:
            primary = residuals[
                (NORMS[0], reference)
            ][:, j]

            other = residuals[
                (NORMS[1], reference)
            ][:, j]

            compare(
                sample,
                "normalisation",
                NORMS[0],
                NORMS[1] + "__" + reference,
                primary,
                other
            )

    comparisons = pd.DataFrame(comparisons)

    check(
        "exact_91_residual_comparisons",
        len(comparisons) == 91
    )

    check(
        "all_residual_sensitivity_metrics_finite",
        np.isfinite(
            comparisons[
                [
                    "residual_cosine_108",
                    "residual_relative_change_108"
                ]
            ].to_numpy(dtype=float)
        ).all()
    )

    fit_table = pd.DataFrame(fits)

    print()
    print("========== PRIMARY RESIDUAL QC ==========")

    primary = fit_table.loc[
        (fit_table["normalisation"] == NORMS[0])
        & (fit_table["reference"] == REFS[0])
    ]

    print(
        primary[
            [
                "sample",
                "known_state_energy_fraction_108",
                "residual_energy_fraction_108",
                "residual_l2"
            ]
        ].round(5).to_string(index=False)
    )

    print()
    print("========== RESIDUAL SENSITIVITY SUMMARY ==========")

    print(
        comparisons.groupby("comparison")[
            [
                "residual_cosine_108",
                "residual_relative_change_108"
            ]
        ].agg(
            ["min", "median", "max"]
        ).round(5).to_string()
    )

    print()
    print("========== FREEZE OUTPUTS ==========")

    for (norm, reference), values in outputs.items():
        suffix = "__" + norm + "__" + reference + ".csv"

        values["residual"].rename_axis(
            "imodulon"
        ).reset_index().to_csv(
            OUT / ("FROZEN_108_residuals" + suffix),
            index=False,
            float_format="%.16g"
        )

        values["coefficients"].rename_axis(
            "controlled_state"
        ).reset_index().to_csv(
            OUT / ("FROZEN_11_state_coefficients" + suffix),
            index=False,
            float_format="%.16g"
        )

    fit_table.to_csv(
        OUT / "FROZEN_78_residual_energy_QC.csv",
        index=False
    )

    comparisons.to_csv(
        OUT / "FROZEN_91_residual_sensitivity_comparisons.csv",
        index=False
    )

    selected_metadata.to_csv(
        OUT / "FROZEN_Atlanta_13_sample_provenance.csv",
        index=False
    )

    pd.DataFrame(CHECKS).to_csv(
        OUT / "audit_pass_checks.csv",
        index=False
    )

    pd.DataFrame([
        {
            "input": name,
            "path": str(path),
            "sha256": sha256(path)
        }
        for name, path in inputs.items()
    ]).to_csv(
        OUT / "input_sha256_provenance.csv",
        index=False
    )

    protocol = (
        "AUDIT 04m: BLINDED RESIDUAL AND EVALUATION LOCK\n"
        "\n"
        "Original Audit 03f method: 11 unit-norm "
        "controlled-state axes followed by ordinary "
        "least-squares reconstruction and residualisation.\n"
        "Programme space: original 108 empirically "
        "stable programmes, original calibration.\n"
        "Cohort: all 13 frozen Atlanta sputum samples.\n"
        "Primary: FROZEN_5567_TPM, PRIMARY_CFBE_ALL4.\n"
        "Sensitivity: CFBE_C_PAIR, CFBE_R_PAIR, "
        "and SOFTCORE_4923_TPM.\n"
        "No sample exclusions or programme selection "
        "based on candidate agreement.\n"
        "Prospective candidate comparison: signed cosine "
        "similarity in the aligned 108-dimensional "
        "controlled-state residual space.\n"
        "Report all 13 individual sample cosines and "
        "cohort-level descriptive summaries.\n"
        "Separately report directional concordance for "
        "the original frozen 18 candidate programmes, "
        "without modifying their selection or directions.\n"
        "Repeat both comparisons for all five "
        "predefined sensitivity variants.\n"
        "Do not tune thresholds after unblinding.\n"
        "Donor independence: NOT VERIFIED.\n"
        "Until patient linkage is established, treat "
        "sample observations as potentially clustered "
        "and do not report a patient-level inferential p-value.\n"
        "Published source: Cornforth et al. 2020, "
        "DOI 10.1128/mbio.03042-19, Data Set S1.\n"
        "The paper reports 20 sputum samples from 19 "
        "patients; the repeated-donor pair needs "
        "mapping to our 13 Atlanta samples.\n"
        "External CFBE reference is not patient matched.\n"
        "Limited projection energy remains an "
        "interpretation constraint.\n"
        "No Audit 03h candidate accessed.\n"
    )

    (
        OUT / "FROZEN_PREUNBLINDING_EVALUATION_PROTOCOL.txt"
    ).write_text(protocol, encoding="utf-8")

    (
        OUT / "DONOR_PROVENANCE_STATUS.txt"
    ).write_text(
        "DONOR_INDEPENDENCE_NOT_VERIFIED\n"
    )

    status = (
        "PRJNA576508_ATLANTA_13_BLINDED_"
        "CONTROLLED_STATE_RESIDUALS_FROZEN"
    )

    (
        OUT / "AUDIT_04M_STATUS.txt"
    ).write_text(status + "\n")

    print()
    print("=" * 76)
    print("STATUS:", status)
    print("=" * 76)
    print("All 13 samples retained.")
    print("Candidate 03h not accessed.")
    print("Donor independence still needs verification.")


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 04m output exists; refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()

    except Exception:
        (
            OUT / "AUDIT_04M_STATUS.txt"
        ).write_text(
            "PRJNA576508_AUDIT_04M_REVIEW_REQUIRED\n"
        )

        (
            OUT / "AUDIT_04M_FAILURE.txt"
        ).write_text(
            traceback.format_exc()
        )

        pd.DataFrame(CHECKS).to_csv(
            OUT / "audit_pass_checks.csv",
            index=False
        )

        raise
