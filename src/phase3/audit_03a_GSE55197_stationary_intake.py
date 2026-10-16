from pathlib import Path
from io import BytesIO, StringIO
import gzip
import tarfile
import time
import urllib.error
import urllib.request

import numpy as np
import pandas as pd


ROOT = Path.cwd()

DATA_DIR = (
    ROOT
    / "data/raw/phase3/GSE55197"
)

ARCHIVE = (
    DATA_DIR
    / "GSE55197_RAW.tar"
)

PROJECTION_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

OUT = (
    ROOT
    / "results/phase3/audit_03a_GSE55197_stationary_intake"
)

URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/"
    "GSE55nnn/GSE55197/suppl/GSE55197_RAW.tar"
)

SAMPLES = {
    "GSM1331366": {
        "name": "PA14_EXP_1",
        "condition": "exponential",
    },
    "GSM1331367": {
        "name": "PA14_EXP_2",
        "condition": "exponential",
    },
    "GSM1331392": {
        "name": "PA14_STAT_1",
        "condition": "stationary",
    },
    "GSM1331393": {
        "name": "PA14_STAT_2",
        "condition": "stationary",
    },
}


def download_with_retry(
    url,
    destination,
):
    if destination.exists():
        print(
            "Processed archive already present:"
        )
        print(
            " ",
            destination,
        )
        return

    max_attempts = 6

    for attempt in range(
        1,
        max_attempts + 1,
    ):
        try:
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent":
                        "CF-State Phase3 Audit03a"
                },
            )

            with urllib.request.urlopen(
                request,
                timeout=120,
            ) as response:
                with destination.open(
                    "wb"
                ) as handle:
                    while True:
                        chunk = response.read(
                            1024 * 1024
                        )

                        if not chunk:
                            break

                        handle.write(
                            chunk
                        )

            return

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            if attempt == max_attempts:
                raise

            wait = 2 ** attempt

            print(
                "Download retry in",
                wait,
                "seconds..."
            )

            time.sleep(
                wait
            )


def parse_processed_file(
    raw_bytes,
    filename,
):
    if filename.endswith(
        ".gz"
    ):
        raw_bytes = gzip.decompress(
            raw_bytes
        )

    text = raw_bytes.decode(
        "utf-8",
        errors="replace",
    )

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.startswith(
            "#"
        )
    ]

    if not lines:
        raise RuntimeError(
            "Empty processed file: "
            + filename
        )

    table = pd.read_csv(
        StringIO(
            "\n".join(
                lines
            )
        ),
        sep=r"\s+",
        header=None,
        engine="python",
    )

    if table.shape[
        1
    ] < 2:
        raise RuntimeError(
            "Expected at least two columns in "
            + filename
        )

    genes = (
        table.iloc[
            :,
            0,
        ]
        .astype(str)
        .str.strip()
    )

    values = pd.to_numeric(
        table.iloc[
            :,
            1,
        ],
        errors="coerce",
    )

    valid = values.notna()

    genes = genes.loc[
        valid
    ]

    values = values.loc[
        valid
    ]

    result = pd.Series(
        values.to_numpy(
            dtype=float
        ),
        index=genes,
        name=filename,
    )

    if result.index.duplicated().any():
        duplicated = int(
            result.index.duplicated().sum()
        )

        raise RuntimeError(
            filename
            + " contains "
            + str(
                duplicated
            )
            + " duplicate gene identifiers."
        )

    if (
        result
        < 0
    ).any():
        raise RuntimeError(
            "Negative read counts in "
            + filename
        )

    return result


def pairwise_r(
    matrix,
    sample_1,
    sample_2,
):
    return float(
        matrix[
            [
                sample_1,
                sample_2,
            ]
        ]
        .corr()
        .iloc[
            0,
            1,
        ]
    )


def main():
    print("=" * 78)
    print("CF-State Phase 3 - Audit 03a")
    print("GSE55197 WT PA14 stationary/exponential intake and replicate QC")
    print("=" * 78)

    DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not PROJECTION_MAP.exists():
        raise FileNotFoundError(
            str(
                PROJECTION_MAP
            )
        )

    # ----------------------------------------------------------
    # 1. Download published Reads-per-Gene archive
    # ----------------------------------------------------------

    print()
    print("Published processed data")

    download_with_retry(
        URL,
        ARCHIVE,
    )

    print(
        "  archive:",
        ARCHIVE,
    )
    print(
        "  size:",
        ARCHIVE.stat().st_size,
        "bytes",
    )

    # ----------------------------------------------------------
    # 2. Identify the four explicit WT samples
    # ----------------------------------------------------------

    sample_vectors = {}

    manifest_rows = []

    with tarfile.open(
        ARCHIVE,
        "r",
    ) as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile()
        ]

        for gsm, metadata in SAMPLES.items():
            matches = [
                member
                for member in members
                if Path(
                    member.name
                ).name.startswith(
                    gsm
                )
            ]

            if len(
                matches
            ) != 1:
                raise RuntimeError(
                    gsm
                    + ": expected one processed file, found "
                    + str(
                        [
                            item.name
                            for item in matches
                        ]
                    )
                )

            member = matches[
                0
            ]

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                raise RuntimeError(
                    "Could not read "
                    + member.name
                )

            raw_bytes = extracted.read()

            vector = parse_processed_file(
                raw_bytes,
                member.name,
            )

            name = metadata[
                "name"
            ]

            sample_vectors[
                name
            ] = vector

            manifest_rows.append(
                {
                    "gsm":
                        gsm,
                    "sample":
                        name,
                    "condition":
                        metadata[
                            "condition"
                        ],
                    "archive_file":
                        member.name,
                    "n_genes":
                        len(
                            vector
                        ),
                    "total_reads_per_gene":
                        float(
                            vector.sum()
                        ),
                    "detected_genes":
                        int(
                            (
                                vector
                                > 0
                            ).sum()
                        ),
                }
            )

    manifest = pd.DataFrame(
        manifest_rows
    )

    print()
    print("Selected GSE55197 samples")
    print(
        manifest.to_string(
            index=False
        )
    )

    manifest.to_csv(
        OUT
        / "GSE55197_stationary_sample_manifest.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 3. Gene-set compatibility across samples
    # ----------------------------------------------------------

    gene_sets = {
        sample:
            set(
                vector.index
            )
        for sample, vector
        in sample_vectors.items()
    }

    common_genes = set.intersection(
        *gene_sets.values()
    )

    union_genes = set.union(
        *gene_sets.values()
    )

    identical_gene_sets = all(
        genes
        == common_genes
        for genes
        in gene_sets.values()
    )

    print()
    print("Processed gene universe")
    print(
        "  common genes:",
        len(
            common_genes
        ),
    )
    print(
        "  union genes:",
        len(
            union_genes
        ),
    )
    print(
        "  identical gene sets:",
        identical_gene_sets,
    )

    if len(
        common_genes
    ) < 5000:
        raise RuntimeError(
            "Unexpectedly small common gene universe."
        )

    # Keep the common universe only for QC.
    ordered_genes = sorted(
        common_genes
    )

    counts = pd.DataFrame(
        {
            sample:
                vector.reindex(
                    ordered_genes
                )
            for sample, vector
            in sample_vectors.items()
        }
    )

    if counts.isna().any().any():
        raise RuntimeError(
            "Missing values after common-gene alignment."
        )

    counts.to_csv(
        OUT
        / "GSE55197_stationary_exponential_reads_per_gene.csv"
    )

    # ----------------------------------------------------------
    # 4. Determine gene-ID namespace against frozen PA14 map
    # ----------------------------------------------------------

    projection_map = pd.read_csv(
        PROJECTION_MAP
    )

    projection_map[
        "pa14_gene"
    ] = (
        projection_map[
            "pa14_gene"
        ]
        .astype(str)
        .str.strip()
    )

    projection_map[
        "current_locus_tag"
    ] = (
        projection_map[
            "current_locus_tag"
        ]
        .astype(str)
        .str.strip()
    )

    old_ids = set(
        projection_map[
            "pa14_gene"
        ]
    )

    current_ids = set(
        projection_map[
            "current_locus_tag"
        ]
    )

    observed_ids = set(
        counts.index
    )

    old_matches = len(
        observed_ids.intersection(
            old_ids
        )
    )

    current_matches = len(
        observed_ids.intersection(
            current_ids
        )
    )

    if old_matches > current_matches:
        namespace = "OLD_PA14_LOCUS"
        projection_ids = old_ids
    elif current_matches > old_matches:
        namespace = "CURRENT_REFSEQ_LOCUS"
        projection_ids = current_ids
    else:
        namespace = "AMBIGUOUS"
        projection_ids = set()

    projection_matches = len(
        observed_ids.intersection(
            projection_ids
        )
    )

    print()
    print("Gene-ID namespace")
    print(
        "  observed genes:",
        len(
            observed_ids
        ),
    )
    print(
        "  matches frozen old PA14 tags:",
        old_matches,
    )
    print(
        "  matches current RefSeq tags:",
        current_matches,
    )
    print(
        "  selected namespace:",
        namespace,
    )
    print(
        "  frozen 5286 projection genes represented:",
        projection_matches,
    )

    # ----------------------------------------------------------
    # 5. Count-depth and detection QC
    # ----------------------------------------------------------

    sample_qc = pd.DataFrame(
        {
            "total_reads_per_gene":
                counts.sum(
                    axis=0
                ),

            "detected_genes":
                (
                    counts
                    > 0
                ).sum(
                    axis=0
                ),
        }
    )

    sample_qc[
        "detected_fraction"
    ] = (
        sample_qc[
            "detected_genes"
        ]
        / counts.shape[
            0
        ]
    )

    print()
    print("Per-sample count QC")
    print(
        sample_qc
        .round(
            4
        )
        .to_string()
    )

    sample_qc.to_csv(
        OUT
        / "sample_count_QC.csv"
    )

    # ----------------------------------------------------------
    # 6. Replicate QC using log2(CPM + 1)
    #
    # Counts are Reads-per-Gene. For within-study replicate
    # correlation, library-size CPM is sufficient because gene
    # lengths are constant across samples.
    # ----------------------------------------------------------

    library_sizes = counts.sum(
        axis=0
    )

    if (
        library_sizes
        <= 0
    ).any():
        raise RuntimeError(
            "Invalid library size."
        )

    cpm = counts.divide(
        library_sizes,
        axis=1,
    ) * 1_000_000.0

    log_cpm = np.log2(
        cpm
        + 1.0
    )

    exp_r = pairwise_r(
        log_cpm,
        "PA14_EXP_1",
        "PA14_EXP_2",
    )

    stat_r = pairwise_r(
        log_cpm,
        "PA14_STAT_1",
        "PA14_STAT_2",
    )

    all_corr = log_cpm.corr()

    print()
    print("log2(CPM+1) replicate QC")
    print(
        "  exponential replicate r:",
        round(
            exp_r,
            4,
        ),
    )
    print(
        "  stationary replicate r:",
        round(
            stat_r,
            4,
        ),
    )

    print()
    print("All-sample correlation matrix")
    print(
        all_corr
        .round(
            4
        )
        .to_string()
    )

    all_corr.to_csv(
        OUT
        / "sample_logCPM_correlation_matrix.csv"
    )

    # ----------------------------------------------------------
    # 7. Does abundance filtering change replicate agreement?
    # ----------------------------------------------------------

    mean_cpm = cpm.mean(
        axis=1
    )

    filter_rows = []

    for threshold in [
        0,
        0.1,
        1,
        5,
        10,
        25,
    ]:
        keep = mean_cpm[
            mean_cpm
            >= threshold
        ].index

        subset = log_cpm.loc[
            keep
        ]

        filter_rows.append(
            {
                "mean_CPM_threshold":
                    threshold,
                "n_genes":
                    len(
                        keep
                    ),
                "exp_replicate_r":
                    pairwise_r(
                        subset,
                        "PA14_EXP_1",
                        "PA14_EXP_2",
                    ),
                "stationary_replicate_r":
                    pairwise_r(
                        subset,
                        "PA14_STAT_1",
                        "PA14_STAT_2",
                    ),
            }
        )

    filtered_qc = pd.DataFrame(
        filter_rows
    )

    print()
    print("Replicate QC after abundance filtering")
    print(
        filtered_qc
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    filtered_qc.to_csv(
        OUT
        / "abundance_filter_replicate_QC.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # 8. Projection-universe count-mass coverage
    # ----------------------------------------------------------

    if namespace != "AMBIGUOUS":
        projected_observed = sorted(
            observed_ids.intersection(
                projection_ids
            )
        )

        projection_mass = counts.loc[
            projected_observed
        ].sum(
            axis=0
        )

        total_mass = counts.sum(
            axis=0
        )

        projection_mass_fraction = (
            projection_mass
            / total_mass
        )

        print()
        print("Frozen projection-universe count mass")
        print(
            projection_mass_fraction
            .round(
                4
            )
            .to_string()
        )

        min_projection_mass = float(
            projection_mass_fraction.min()
        )

    else:
        projection_mass_fraction = pd.Series(
            np.nan,
            index=counts.columns,
        )

        min_projection_mass = np.nan

    projection_mass_fraction.to_csv(
        OUT
        / "projection_universe_count_mass_fraction.csv",
        header=[
            "fraction"
        ],
    )

    # ----------------------------------------------------------
    # 9. Intake gates
    # ----------------------------------------------------------

    min_total_reads = float(
        sample_qc[
            "total_reads_per_gene"
        ].min()
    )

    checks = {
        "four_expected_samples":
            counts.shape[
                1
            ]
            == 4,

        "common_genes_ge_5800":
            len(
                common_genes
            )
            >= 5800,

        "identical_gene_sets":
            identical_gene_sets,

        "namespace_resolved":
            namespace
            != "AMBIGUOUS",

        "projection_genes_ge_5200":
            projection_matches
            >= 5200,

        "minimum_total_reads_ge_100k":
            min_total_reads
            >= 100000,

        "exponential_replicate_r_ge_075":
            exp_r
            >= 0.75,

        "stationary_replicate_r_ge_075":
            stat_r
            >= 0.75,

        "projection_count_mass_ge_90pct":
            (
                np.isfinite(
                    min_projection_mass
                )
                and min_projection_mass
                >= 0.90
            ),

        "finite_values":
            bool(
                np.isfinite(
                    counts.to_numpy(
                        dtype=float
                    )
                ).all()
            ),
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit checks")

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
        "GSE55197_STATIONARY_READY_FOR_PROJECTION"
        if passed
        else "GSE55197_STATIONARY_INTAKE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
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

    summary = [
        "# CF-State Phase 3 — Audit 03a",
        "",
        "## GSE55197 WT PA14 stationary vs exponential intake",
        "",
        "- Exponential replicates: GSM1331366, GSM1331367",
        "- Stationary replicates: GSM1331392, GSM1331393",
        "- Common processed genes: "
        + str(
            len(
                common_genes
            )
        ),
        "- Gene namespace: "
        + namespace,
        "- Frozen projection genes represented: "
        + str(
            projection_matches
        ),
        "- Exponential replicate r: "
        + format(
            exp_r,
            ".4f",
        ),
        "- Stationary replicate r: "
        + format(
            stat_r,
            ".4f",
        ),
        "",
        "## Decision",
        "",
        status,
    ]

    (
        OUT
        / "AUDIT_03A_SUMMARY.md"
    ).write_text(
        "\n".join(
            summary
        )
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
            "Failed checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
