from pathlib import Path
import hashlib
import traceback

import numpy as np
import pandas as pd

ROOT = Path.cwd()
P4 = ROOT / "results/phase4"

H = P4 / "audit_03h_freeze_replicated_CF_airway_candidate"
L = P4 / "audit_04l_PRJNA576508_blinded_projection_QC_lock"
M = P4 / "audit_04m2_PRJNA576508_blinded_residual_freeze"

OUT = P4 / "audit_04n_PRJNA576508_independent_03h_comparison"

VECTORS = H / "FROZEN_CF_airway_candidate_vectors.csv"
REGISTRY = H / "conservatively_replicated_programmes.csv"
PROGRAMMES = H / "replicated_programme_registry_108.csv"

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

CANDIDATES = [
    "two_cohort_consensus_unit",
    "GSE119356_discovery_unit",
    "PRJEB24688_replication_unit"
]

CHECKS = []


def check(name, condition):
    okay = bool(condition)
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


def cosine(a, b):
    denominator = float(
        np.linalg.norm(a) * np.linalg.norm(b)
    )

    if denominator < 1e-12:
        raise RuntimeError("Zero-norm comparison vector")

    return float(np.dot(a, b) / denominator)


def pearson(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        raise RuntimeError("Zero-variance comparison vector")

    return float(np.corrcoef(a, b)[0, 1])


def read_indexed(path, index_name=None):
    df = pd.read_csv(path)

    first = df.columns[0]

    if index_name is not None:
        check(
            path.name + "_index_column",
            index_name in df.columns
        )
        first = index_name

    ids = df[first].astype(str).str.strip()

    check(
        path.name + "_unique_ids",
        ids.notna().all()
        and ids.ne("").all()
        and ids.is_unique
    )

    df = df.drop(columns=[first])
    df.index = ids.to_numpy()

    return df


def direction_sign(value):
    label = (
        str(value).strip().lower()
        .replace(" ", "_")
    )

    positive = {
        "+", "+1", "1", "1.0", "positive",
        "pos", "up", "increased",
        "positive_direction", "upregulated"
    }

    negative = {
        "-", "-1", "-1.0", "negative",
        "neg", "down", "decreased",
        "negative_direction", "downregulated"
    }

    if label in positive:
        return 1

    if label in negative:
        return -1

    if label.startswith("positive") or label.startswith("up_"):
        return 1

    if label.startswith("negative") or label.startswith("down_"):
        return -1

    try:
        number = float(label)
    except ValueError:
        raise RuntimeError(
            "Unrecognised frozen direction: " + repr(value)
        )

    if not np.isfinite(number) or number == 0:
        raise RuntimeError(
            "Invalid frozen direction: " + repr(value)
        )

    return int(np.sign(number))


def main():
    print("=" * 76)
    print("CF-State Phase 4 — Audit 04n")
    print("Independent Atlanta versus frozen 03h candidate")
    print("=" * 76)

    sources = {
        "03h_status": H / "AUDIT_03H_STATUS.txt",
        "04l_status": L / "AUDIT_04L_STATUS.txt",
        "04m2_status": M / "AUDIT_04M2_STATUS.txt",
        "donor_status": M / "DONOR_PROVENANCE_STATUS.txt",
        "preunblinding_protocol":
            M / "FROZEN_PREUNBLINDING_EVALUATION_PROTOCOL.txt",
        "frozen_108_candidates": VECTORS,
        "frozen_18_directions": REGISTRY,
        "frozen_108_registry": PROGRAMMES,
        "script": Path(__file__)
    }

    for name, path in sources.items():
        check("input_" + name + "_exists", path.is_file())

    expected = {
        "03h_status":
            "REPLICATED_CF_AIRWAY_PROGRAMME_CANDIDATE_FROZEN",
        "04l_status":
            "PRJNA576508_ATLANTA_BLINDED_"
            "PROJECTION_QC_AND_DESIGN_LOCK_FROZEN",
        "04m2_status":
            "PRJNA576508_ATLANTA_13_BLINDED_"
            "CONTROLLED_STATE_RESIDUALS_FROZEN",
        "donor_status":
            "DONOR_INDEPENDENCE_NOT_VERIFIED"
    }

    for name, status in expected.items():
        check(
            name + "_matches_frozen_status",
            sources[name].read_text().strip() == status
        )

    print()
    print("========== UNBLIND FROZEN 03h CANDIDATE ==========")

    frozen = read_indexed(VECTORS)

    check(
        "candidate_exact_108",
        frozen.shape[0] == 108
    )

    check(
        "candidate_columns",
        set(CANDIDATES).issubset(frozen.columns)
    )

    candidate = frozen[CANDIDATES].apply(
        pd.to_numeric,
        errors="raise"
    )

    check(
        "candidate_values_finite",
        np.isfinite(candidate.to_numpy()).all()
    )

    check(
        "three_reference_vectors_unit_norm",
        np.allclose(
            np.linalg.norm(candidate.to_numpy(), axis=0),
            1.0,
            rtol=0,
            atol=1e-6
        )
    )

    registry108 = pd.read_csv(PROGRAMMES)

    check(
        "registry_108_programmes",
        "imodulon" in registry108.columns
        and len(registry108) == 108
    )

    names = (
        registry108["imodulon"]
        .astype(str).str.strip()
    )

    check(
        "registry_exact_candidate_space",
        names.is_unique
        and set(names) == set(candidate.index)
    )

    order = list(candidate.index)

    print()
    print("========== FROZEN 18 PROGRAMME DIRECTIONS ==========")

    reg18 = pd.read_csv(REGISTRY)

    required = {
        "imodulon",
        "direction",
        "two_cohort_consensus_loading"
    }

    check(
        "frozen_18_columns",
        required.issubset(reg18.columns)
    )

    check(
        "exact_18_frozen_programmes",
        len(reg18) == 18
    )

    reg18["imodulon"] = (
        reg18["imodulon"]
        .astype(str).str.strip()
    )

    check(
        "18_unique_and_in_108",
        reg18["imodulon"].is_unique
        and set(reg18["imodulon"]).issubset(set(order))
    )

    reg18["expected_sign"] = (
        reg18["direction"].map(direction_sign)
    )

    loadings = pd.to_numeric(
        reg18["two_cohort_consensus_loading"],
        errors="raise"
    )

    check(
        "18_loading_signs_match_directions",
        np.isfinite(loadings).all()
        and (
            np.sign(loadings)
            == reg18["expected_sign"]
        ).all()
    )

    check(
        "18_signs_match_consensus_vector",
        all(
            int(np.sign(
                candidate.loc[
                    row.imodulon,
                    "two_cohort_consensus_unit"
                ]
            )) == int(row.expected_sign)
            for row in reg18.itertuples(index=False)
        )
    )

    expected_signs = (
        reg18.set_index("imodulon")["expected_sign"]
    )

    print("Frozen candidate programmes:", len(order))
    print("Frozen directional subset:", len(reg18))

    print()
    print("========== INDEPENDENT THIRD-COHORT COMPARISON ==========")

    results = []
    directions = []

    for norm in NORMS:
        for reference in REFS:
            path = M / (
                "FROZEN_108_residuals__"
                + norm + "__" + reference + ".csv"
            )

            check(
                "input_residual_" + norm + "_" + reference,
                path.is_file()
            )

            sources[
                "residual_" + norm + "_" + reference
            ] = path

            residual = read_indexed(
                path,
                index_name="imodulon"
            )

            check(
                "residual_108_by_13_" + norm + "_" + reference,
                residual.shape == (108, 13)
                and set(residual.index) == set(order)
                and set(residual.columns) == set(SAMPLES)
            )

            residual = (
                residual.reindex(
                    index=order,
                    columns=SAMPLES
                )
                .apply(pd.to_numeric, errors="raise")
            )

            check(
                "residual_finite_" + norm + "_" + reference,
                np.isfinite(residual.to_numpy()).all()
            )

            for sample in SAMPLES:
                y = residual[sample].to_numpy(dtype=float)

                record = {
                    "sample": sample,
                    "normalisation": norm,
                    "reference": reference,
                    "residual_l2":
                        float(np.linalg.norm(y))
                }

                for column in CANDIDATES:
                    v = candidate[column].to_numpy(dtype=float)

                    record[
                        "cosine_to_" + column
                    ] = cosine(y, v)

                    record[
                        "pearson_to_" + column
                    ] = pearson(y, v)

                subset = list(expected_signs.index)

                signs = np.sign(
                    residual.loc[subset, sample].to_numpy()
                )

                matches = (
                    signs == expected_signs.to_numpy(dtype=int)
                )

                record["frozen_18_direction_matches"] = int(
                    matches.sum()
                )

                record["frozen_18_direction_fraction"] = float(
                    matches.mean()
                )

                record["frozen_18_zero_activities"] = int(
                    (signs == 0).sum()
                )

                results.append(record)

                for i, programme in enumerate(subset):
                    directions.append({
                        "normalisation": norm,
                        "reference": reference,
                        "sample": sample,
                        "imodulon": programme,
                        "frozen_expected_sign":
                            int(expected_signs.iloc[i]),
                        "observed_residual_activity":
                            float(residual.loc[programme, sample]),
                        "observed_sign":
                            int(signs[i]),
                        "direction_match":
                            bool(matches[i])
                    })

            print(norm, reference, "COMPUTED", flush=True)

    results = pd.DataFrame(results)
    directions = pd.DataFrame(directions)

    check(
        "all_78_predefined_comparisons",
        len(results) == 78
        and not results.duplicated(
            ["normalisation", "reference", "sample"]
        ).any()
    )

    check(
        "all_1404_direction_observations",
        len(directions) == 78 * 18
    )

    print()
    print("========== PRIMARY COMPARISON ==========")

    primary = results.loc[
        (results["normalisation"] == NORMS[0])
        & (results["reference"] == REFS[0])
    ].copy()

    check(
        "primary_all_13",
        len(primary) == 13
        and set(primary["sample"]) == set(SAMPLES)
    )

    primary = (
        primary.set_index("sample")
        .reindex(SAMPLES)
        .reset_index()
    )

    cols = [
        "sample",
        "cosine_to_two_cohort_consensus_unit",
        "cosine_to_GSE119356_discovery_unit",
        "cosine_to_PRJEB24688_replication_unit",
        "frozen_18_direction_matches",
        "frozen_18_direction_fraction"
    ]

    print(primary[cols].round(5).to_string(index=False))

    print()
    print("========== PRIMARY DESCRIPTIVE SUMMARY ==========")

    cos = primary[
        "cosine_to_two_cohort_consensus_unit"
    ]

    print(
        "Consensus cosine: min / median / max",
        round(float(cos.min()), 6),
        round(float(cos.median()), 6),
        round(float(cos.max()), 6)
    )

    print(
        "Positive consensus cosines:",
        int((cos > 0).sum()),
        "/ 13"
    )

    print(
        "Median frozen-direction matches:",
        float(
            primary[
                "frozen_18_direction_matches"
            ].median()
        ),
        "/ 18"
    )

    print()
    print("========== ALL SIX VARIANTS ==========")

    variants = (
        results.groupby(
            ["normalisation", "reference"],
            sort=False
        )
        .agg(
            n_samples=("sample", "size"),
            median_consensus_cosine=(
                "cosine_to_two_cohort_consensus_unit",
                "median"
            ),
            min_consensus_cosine=(
                "cosine_to_two_cohort_consensus_unit",
                "min"
            ),
            max_consensus_cosine=(
                "cosine_to_two_cohort_consensus_unit",
                "max"
            ),
            n_positive_consensus_cosine=(
                "cosine_to_two_cohort_consensus_unit",
                lambda x: int((x > 0).sum())
            ),
            median_18_direction_fraction=(
                "frozen_18_direction_fraction",
                "median"
            )
        )
        .reset_index()
    )

    print(variants.round(5).to_string(index=False))

    print()
    print("========== PER-PROGRAMME DIRECTIONAL SUMMARY ==========")

    primary_directions = directions.loc[
        (directions["normalisation"] == NORMS[0])
        & (directions["reference"] == REFS[0])
    ]

    by_programme = (
        primary_directions.groupby(
            "imodulon",
            sort=False
        )
        .agg(
            frozen_expected_sign=(
                "frozen_expected_sign", "first"
            ),
            n_sample_directions_matching=(
                "direction_match", "sum"
            ),
            n_samples=(
                "direction_match", "size"
            ),
            mean_residual_activity=(
                "observed_residual_activity", "mean"
            ),
            median_residual_activity=(
                "observed_residual_activity", "median"
            )
        )
        .reset_index()
    )

    by_programme["fraction_samples_matching"] = (
        by_programme["n_sample_directions_matching"]
        / by_programme["n_samples"]
    )

    print(
        by_programme[
            [
                "imodulon",
                "frozen_expected_sign",
                "n_sample_directions_matching",
                "fraction_samples_matching"
            ]
        ].round(4).to_string(index=False)
    )

    print()
    print("========== SENSITIVITY VERSUS PRIMARY ==========")

    baseline = primary.set_index("sample")[
        "cosine_to_two_cohort_consensus_unit"
    ]

    sensitivities = results.loc[
        ~(
            (results["normalisation"] == NORMS[0])
            & (results["reference"] == REFS[0])
        )
    ].copy()

    sensitivities[
        "delta_vs_primary_consensus_cosine"
    ] = [
        float(
            row.cosine_to_two_cohort_consensus_unit
            - baseline[row.sample]
        )
        for row in sensitivities.itertuples(index=False)
    ]

    check(
        "all_65_sensitivity_comparisons",
        len(sensitivities) == 65
    )

    print(
        sensitivities.groupby(
            ["normalisation", "reference"]
        )["delta_vs_primary_consensus_cosine"]
        .agg(["min", "median", "max"])
        .round(5)
        .to_string()
    )

    print()
    print("========== FREEZE INDEPENDENT COMPARISON ==========")

    results.to_csv(
        OUT / "FROZEN_78_all_candidate_comparisons.csv",
        index=False
    )

    primary.to_csv(
        OUT / "FROZEN_13_primary_candidate_comparisons.csv",
        index=False
    )

    directions.to_csv(
        OUT / "FROZEN_1404_programme_direction_observations.csv",
        index=False
    )

    by_programme.to_csv(
        OUT / "FROZEN_18_directional_programme_summary.csv",
        index=False
    )

    variants.to_csv(
        OUT / "FROZEN_six_variant_descriptive_summary.csv",
        index=False
    )

    sensitivities.to_csv(
        OUT / "FROZEN_65_sensitivity_vs_primary.csv",
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
        for name, path in sources.items()
    ]).to_csv(
        OUT / "input_sha256_provenance.csv",
        index=False
    )

    methods = (
        "CF-State Audit 04n: independent third-cohort comparison.\n"
        "Primary: 13 Atlanta samples, frozen 5567-gene TPM, CFBE ALL4.\n"
        "Five predefined sensitivity variants.\n"
        "108 aligned controlled-state residual programmes.\n"
        "Frozen 03h two-cohort consensus, discovery and replication vectors.\n"
        "Original 18 programme directions, without reselection.\n"
        "Signed cosine, Pearson, and directional concordance.\n"
        "No new sample exclusion, threshold tuning, or refitting.\n"
        "DONOR INDEPENDENCE NOT VERIFIED.\n"
        "No patient-level significance tests or confidence intervals.\n"
        "CFBE culture reference is not patient-matched.\n"
        "Limited gene-space projection energy constrains interpretation.\n"
        "Technical completion does not establish biological replication.\n"
    )

    (
        OUT / "AUDIT_04N_METHODS_AND_LIMITATIONS.txt"
    ).write_text(methods, encoding="utf-8")

    status = (
        "PRJNA576508_INDEPENDENT_03H_"
        "DESCRIPTIVE_COMPARISON_FROZEN"
    )

    (
        OUT / "AUDIT_04N_STATUS.txt"
    ).write_text(status + "\n")

    print()
    print("=" * 76)
    print("STATUS:", status)
    print("=" * 76)
    print("No patient-level inference calculated.")
    print("Original Audit 03h candidate unchanged.")


if __name__ == "__main__":
    if OUT.exists():
        raise RuntimeError(
            "Audit 04n already exists; refusing overwrite."
        )

    OUT.mkdir(parents=True)

    try:
        main()

    except Exception:
        (
            OUT / "AUDIT_04N_STATUS.txt"
        ).write_text(
            "PRJNA576508_AUDIT_04N_REVIEW_REQUIRED\n"
        )

        (
            OUT / "AUDIT_04N_FAILURE.txt"
        ).write_text(traceback.format_exc())

        pd.DataFrame(CHECKS).to_csv(
            OUT / "audit_pass_checks.csv",
            index=False
        )

        raise
