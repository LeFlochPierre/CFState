from pathlib import Path
from io import BytesIO, StringIO
import json
import re
import tarfile

import numpy as np
import pandas as pd


ROOT = Path.cwd()

ARCHIVE = (
    ROOT
    / "data/raw/phase4/PRJEB24688/published_analysis"
    / "Rossi_et_al_2018_analysis.tar.gz"
)

FROZEN_MAP = (
    ROOT
    / "results/phase3/audit_02e_PA14_projection_universe"
    / "frozen_PA14_projection_map.csv"
)

MODEL = (
    ROOT
    / "external/aeruprecise/precise411"
    / "paeru_116_iModulon.json"
)

PRIMARY_PROGRAMMES = (
    ROOT
    / "results/phase3/audit_04_freeze_crossstrain_state_framework"
    / "phase3_robust_transferable_imodulons.csv"
)

OUT = (
    ROOT
    / "results/phase4/audit_03d_PRJEB24688_published_count_intake"
)


EXPECTED_INVIVO_RUNS = [
    "P24M1_S2",
    "P24M1_S3",
    "P30M0_S2",
    "P30M0_S3R1",
    "P30M0_S3R2",
    "P30M0_S4R1",
    "P30M0_S4R2",
    "P73F1_S1R1",
    "P73F1_S1R2",
    "P77F1_S1",
    "P77F1_S2",
    "P77F1_S3",
]


EXPECTED_MATCHED_INVITRO = [
    "PA-P24M1A_24H_R1",
    "PA-P24M1A_3H_R1",
    "PA-P24M1A_3H_R2",
    "PA-P24M1B_24H_R1",
    "PA-P24M1B_24H_R2",
    "PA-P24M1B_3H_R1",
    "PA-P24M1B_3H_R2",
    "PA-P30M0_24H_R1",
    "PA-P30M0_24H_R2",
    "PA-P30M0_3H_R1",
    "PA-P30M0_3H_R2",
    "PA-P73F1_24H_R1",
    "PA-P73F1_24H_R2",
    "PA-P73F1_3H_R1",
    "PA-P73F1_3H_R2",
    "PA-P77F1_24H_R1",
    "PA-P77F1_24H_R2",
    "PA-P77F1_3H_R1",
    "PA-P77F1_3H_R2",
]


EXPECTED_PA14 = [
    "PA-PA14_24H_R1",
    "PA-PA14_24H_R2",
    "PA-PA14_3H_R1",
    "PA-PA14_3H_R2",
]


TECHNICAL_GROUPS = {
    "P30M0_S3": [
        "P30M0_S3R1",
        "P30M0_S3R2",
    ],
    "P30M0_S4": [
        "P30M0_S4R1",
        "P30M0_S4R2",
    ],
    "P73F1_S1": [
        "P73F1_S1R1",
        "P73F1_S1R2",
    ],
}


SINGLE_RUN_BIOLOGICAL = {
    "P24M1_S2":
        "P24M1_S2",

    "P24M1_S3":
        "P24M1_S3",

    "P30M0_S2":
        "P30M0_S2",

    "P77F1_S1":
        "P77F1_S1",

    "P77F1_S2":
        "P77F1_S2",

    "P77F1_S3":
        "P77F1_S3",
}


def json_df(value):
    if isinstance(value, str):
        return pd.read_json(
            StringIO(value)
        )

    if isinstance(value, dict):
        return pd.DataFrame(
            value
        )

    raise RuntimeError(
        "Unexpected serialized model DataFrame."
    )


def normalize_member_name(name):
    text = str(
        name
    )

    while text.startswith(
        "./"
    ):
        text = text[
            2:
        ]

    return text


def sample_name_from_count_file(name):
    filename = Path(
        name
    ).name

    suffixes = [
        ".PA14.clean.reads_count",
        ".clean.reads_count",
    ]

    for suffix in suffixes:
        if filename.endswith(
            suffix
        ):
            return filename[
                :-len(
                    suffix
                )
            ]

    return None


def parse_htseq_count(raw_bytes):
    text = raw_bytes.decode(
        "utf-8"
    )

    rows = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = line.strip()

        if not line:
            continue

        fields = line.split(
            "\t"
        )

        if len(
            fields
        ) != 2:
            fields = line.split()

        if len(
            fields
        ) != 2:
            raise RuntimeError(
                "Unexpected HTSeq-count row at line "
                + str(
                    line_number
                )
                + ": "
                + line[
                    :200
                ]
            )

        gene = str(
            fields[0]
        ).strip()

        value = float(
            fields[1]
        )

        if not value.is_integer():
            raise RuntimeError(
                "Non-integer HTSeq count for "
                + gene
            )

        rows.append(
            (
                gene,
                int(
                    value
                ),
            )
        )

    table = pd.DataFrame(
        rows,
        columns=[
            "gene",
            "count",
        ],
    )

    if table[
        "gene"
    ].duplicated().any():
        duplicated = (
            table.loc[
                table[
                    "gene"
                ].duplicated(
                    keep=False
                ),
                "gene",
            ]
            .unique()
            .tolist()
        )

        raise RuntimeError(
            "Duplicate HTSeq identifiers: "
            + ", ".join(
                duplicated[
                    :20
                ]
            )
        )

    return table.set_index(
        "gene"
    )[
        "count"
    ]


def read_archive_tsv(
    archive,
    ending,
):
    matches = []

    for member in archive.getmembers():
        if not member.isfile():
            continue

        name = normalize_member_name(
            member.name
        )

        if name.startswith(
            "._"
        ):
            continue

        if name.endswith(
            ending
        ):
            matches.append(
                member
            )

    if len(
        matches
    ) != 1:
        raise RuntimeError(
            "Expected one archive member ending "
            + ending
            + ", found "
            + str(
                len(
                    matches
                )
            )
        )

    handle = archive.extractfile(
        matches[0]
    )

    if handle is None:
        raise RuntimeError(
            "Could not read archive member "
            + matches[
                0
            ].name
        )

    return pd.read_csv(
        BytesIO(
            handle.read()
        ),
        sep="\t",
    )


def correlation(a, b):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    if (
        np.std(
            a
        )
        == 0
        or np.std(
            b
        )
        == 0
    ):
        return np.nan

    return float(
        np.corrcoef(
            a,
            b,
        )[0, 1]
    )


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 03d")
    print("PRJEB24688 published HTSeq counts and multi-strain universe")
    print("=" * 78)

    for path in [
        ARCHIVE,
        FROZEN_MAP,
        MODEL,
        PRIMARY_PROGRAMMES,
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
    # 1. Frozen CF-State PAO1<->PA14 map
    # ----------------------------------------------------------

    frozen = pd.read_csv(
        FROZEN_MAP
    )

    required_map_columns = [
        "pao1_gene",
        "pa14_gene",
        "current_locus_tag",
    ]

    missing_map_columns = [
        column
        for column in required_map_columns
        if column not in frozen.columns
    ]

    if missing_map_columns:
        raise RuntimeError(
            "Frozen map missing columns: "
            + ", ".join(
                missing_map_columns
            )
        )

    for column in required_map_columns:
        frozen[
            column
        ] = (
            frozen[
                column
            ]
            .astype(str)
            .str.strip()
        )

    if len(
        frozen
    ) != 5286:
        raise RuntimeError(
            "Expected frozen 5286-gene projection map."
        )

    if frozen[
        "pao1_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Frozen PAO1 genes are not unique."
        )

    if frozen[
        "pa14_gene"
    ].duplicated().any():
        raise RuntimeError(
            "Frozen PA14 genes are not unique."
        )

    print()
    print("Frozen cross-strain universe")
    print(
        "  genes:",
        len(
            frozen
        ),
    )

    # ----------------------------------------------------------
    # 2. Read published Rossi soft-core universe
    # ----------------------------------------------------------

    with tarfile.open(
        ARCHIVE,
        "r:gz",
    ) as archive:
        softcore = read_archive_tsv(
            archive,
            "data/pangenome/PAO1_PA14_soft_coregenome.tsv",
        )

    if softcore.shape[
        1
    ] != 2:
        raise RuntimeError(
            "Expected two-column Rossi soft-core table."
        )

    softcore.columns = [
        "rossi_pao1_gene",
        "rossi_pa14_gene",
    ]

    for column in softcore.columns:
        softcore[
            column
        ] = (
            softcore[
                column
            ]
            .astype(str)
            .str.strip()
        )

    print()
    print("Published Rossi multi-strain soft-core")
    print(
        "  rows:",
        len(
            softcore
        ),
    )
    print(
        "  unique PAO1 genes:",
        softcore[
            "rossi_pao1_gene"
        ].nunique(),
    )
    print(
        "  unique PA14 genes:",
        softcore[
            "rossi_pa14_gene"
        ].nunique(),
    )

    # ----------------------------------------------------------
    # 3. Intersect Rossi soft-core with frozen CF-State map
    #
    # Prefer PAO1 identity because it gives an independent
    # coordinate-system bridge.
    # Also verify PA14 agreement wherever possible.
    # ----------------------------------------------------------

    intersection = frozen.merge(
        softcore,
        left_on="pao1_gene",
        right_on="rossi_pao1_gene",
        how="inner",
    )

    pa14_direct_match = (
        intersection[
            "pa14_gene"
        ]
        == intersection[
            "rossi_pa14_gene"
        ]
    )

    pa14_current_match = (
        intersection[
            "current_locus_tag"
        ]
        == intersection[
            "rossi_pa14_gene"
        ]
    )

    intersection[
        "rossi_PA14_matches_frozen"
    ] = (
        pa14_direct_match
        | pa14_current_match
    )

    disagreement = intersection.loc[
        ~intersection[
            "rossi_PA14_matches_frozen"
        ]
    ].copy()

    print()
    print("Rossi soft-core x frozen CF-State intersection")
    print(
        "  overlapping genes:",
        len(
            intersection
        ),
    )
    print(
        "  fraction of frozen 5286:",
        format(
            len(
                intersection
            )
            / 5286,
            ".2%",
        ),
    )
    print(
        "  fraction of Rossi soft-core:",
        format(
            len(
                intersection
            )
            / len(
                softcore
            ),
            ".2%",
        ),
    )
    print(
        "  PA14 mapping disagreements:",
        len(
            disagreement
        ),
    )

    if not disagreement.empty:
        print()
        print("First PA14 mapping disagreements")
        print(
            disagreement[
                [
                    "pao1_gene",
                    "pa14_gene",
                    "current_locus_tag",
                    "rossi_pa14_gene",
                ]
            ]
            .head(
                30
            )
            .to_string(
                index=False
            )
        )

    # Conservative replication universe:
    # require PAO1 soft-core membership AND consistent PA14 mapping.
    replication_map = intersection.loc[
        intersection[
            "rossi_PA14_matches_frozen"
        ]
    ].copy()

    replication_genes = list(
        replication_map[
            "pao1_gene"
        ]
    )

    print()
    print("Frozen PRJEB24688 replication gene universe")
    print(
        "  genes:",
        len(
            replication_map
        ),
    )

    # ----------------------------------------------------------
    # 4. Construct PA14 alias -> PAO1 map
    # ----------------------------------------------------------

    alias_rows = []

    for _, row in replication_map.iterrows():
        pao1 = row[
            "pao1_gene"
        ]

        aliases = {
            row[
                "pa14_gene"
            ],
            row[
                "current_locus_tag"
            ],
            row[
                "rossi_pa14_gene"
            ],
        }

        for alias in aliases:
            alias = str(
                alias
            ).strip()

            if (
                not alias
                or alias.lower()
                in {
                    "nan",
                    "none",
                }
            ):
                continue

            alias_rows.append(
                {
                    "pa14_alias":
                        alias,

                    "pao1_gene":
                        pao1,
                }
            )

    aliases = (
        pd.DataFrame(
            alias_rows
        )
        .drop_duplicates()
    )

    ambiguous_alias = (
        aliases
        .groupby(
            "pa14_alias"
        )[
            "pao1_gene"
        ]
        .nunique()
    )

    ambiguous_alias = set(
        ambiguous_alias.loc[
            ambiguous_alias > 1
        ].index
    )

    if ambiguous_alias:
        raise RuntimeError(
            "Ambiguous PA14 aliases in replication universe: "
            + ", ".join(
                sorted(
                    ambiguous_alias
                )[
                    :20
                ]
            )
        )

    alias_to_pao1 = dict(
        zip(
            aliases[
                "pa14_alias"
            ],
            aliases[
                "pao1_gene"
            ],
        )
    )

    # ----------------------------------------------------------
    # 5. Read every published HTSeq count file
    # ----------------------------------------------------------

    count_vectors = {}

    count_file_rows = []

    with tarfile.open(
        ARCHIVE,
        "r:gz",
    ) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue

            name = normalize_member_name(
                member.name
            )

            basename = Path(
                name
            ).name

            if basename.startswith(
                "._"
            ):
                continue

            if not name.startswith(
                "raw/reads_counts/bacteria/"
            ):
                continue

            if not name.endswith(
                ".reads_count"
            ):
                continue

            sample = sample_name_from_count_file(
                name
            )

            if sample is None:
                continue

            handle = archive.extractfile(
                member
            )

            if handle is None:
                raise RuntimeError(
                    "Could not read count file "
                    + name
                )

            vector = parse_htseq_count(
                handle.read()
            )

            count_vectors[
                sample
            ] = vector

            special_mask = (
                vector.index
                .astype(str)
                .str.startswith(
                    "__"
                )
            )

            gene_vector = vector.loc[
                ~special_mask
            ]

            special_vector = vector.loc[
                special_mask
            ]

            count_file_rows.append(
                {
                    "sample":
                        sample,

                    "archive_file":
                        name,

                    "gene_rows":
                        len(
                            gene_vector
                        ),

                    "special_HTSeq_rows":
                        len(
                            special_vector
                        ),

                    "assigned_gene_counts":
                        int(
                            gene_vector.sum()
                        ),

                    "special_counts":
                        int(
                            special_vector.sum()
                        ),

                    "total_HTSeq_tallied":
                        int(
                            vector.sum()
                        ),
                }
            )

    count_files = pd.DataFrame(
        count_file_rows
    )

    print()
    print("Published HTSeq files recovered")
    print(
        "  files:",
        len(
            count_vectors
        ),
    )

    expected_all = (
        EXPECTED_INVIVO_RUNS
        + EXPECTED_MATCHED_INVITRO
        + EXPECTED_PA14
    )

    missing_expected = [
        sample
        for sample in expected_all
        if sample not in count_vectors
    ]

    unexpected = [
        sample
        for sample in count_vectors
        if sample not in expected_all
    ]

    print(
        "  expected files:",
        len(
            expected_all
        ),
    )
    print(
        "  missing expected:",
        len(
            missing_expected
        ),
    )
    print(
        "  unexpected:",
        len(
            unexpected
        ),
    )

    if missing_expected:
        print(
            "  missing:",
            ", ".join(
                missing_expected
            ),
        )

    if unexpected:
        print(
            "  unexpected:",
            ", ".join(
                sorted(
                    unexpected
                )
            ),
        )

    # ----------------------------------------------------------
    # 6. Audit gene namespaces across files
    # ----------------------------------------------------------

    namespace_rows = []

    mapped_vectors = {}

    duplicate_mapping_samples = []

    for sample, vector in count_vectors.items():
        special_mask = (
            vector.index
            .astype(str)
            .str.startswith(
                "__"
            )
        )

        genes = vector.loc[
            ~special_mask
        ]

        mapped_rows = []

        for source_gene, value in genes.items():
            source_gene = str(
                source_gene
            ).strip()

            if source_gene not in alias_to_pao1:
                continue

            mapped_rows.append(
                (
                    source_gene,
                    alias_to_pao1[
                        source_gene
                    ],
                    int(
                        value
                    ),
                )
            )

        mapped = pd.DataFrame(
            mapped_rows,
            columns=[
                "source_gene",
                "pao1_gene",
                "count",
            ],
        )

        if mapped[
            "pao1_gene"
        ].duplicated().any():
            duplicate_mapping_samples.append(
                sample
            )

        mapped_series = (
            mapped
            .set_index(
                "pao1_gene"
            )[
                "count"
            ]
        )

        mapped_series = mapped_series.reindex(
            replication_genes
        )

        mapped_vectors[
            sample
        ] = mapped_series

        represented = int(
            mapped_series.notna().sum()
        )

        namespace_rows.append(
            {
                "sample":
                    sample,

                "source_gene_rows":
                    len(
                        genes
                    ),

                "replication_genes_represented":
                    represented,

                "replication_gene_coverage":
                    represented
                    / len(
                        replication_genes
                    ),

                "mapped_assigned_counts":
                    float(
                        mapped_series.fillna(
                            0
                        ).sum()
                    ),

                "detected_replication_genes":
                    int(
                        (
                            mapped_series.fillna(
                                0
                            )
                            > 0
                        ).sum()
                    ),
            }
        )

    namespace_qc = pd.DataFrame(
        namespace_rows
    )

    print()
    print("Published-count replication-universe QC")
    print(
        namespace_qc
        .sort_values(
            "sample"
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Build mapped PAO1 count matrix
    # ----------------------------------------------------------

    mapped_counts = pd.DataFrame(
        mapped_vectors
    )

    mapped_counts = mapped_counts.reindex(
        replication_genes
    )

    # Missing rows are structural absence from a file, not zero
    # expression. We require essentially identical gene universes
    # before filling anything.
    missing_fraction = (
        mapped_counts.isna().mean(
            axis=0
        )
    )

    print()
    print("Mapped matrix missing-row fraction")
    print(
        missing_fraction
        .round(
            6
        )
        .to_string()
    )

    # Only zero-complete if every sample contains the full frozen
    # replication gene universe. Otherwise stop for diagnosis.
    full_gene_universe_all_samples = bool(
        (
            missing_fraction
            == 0
        ).all()
    )

    if full_gene_universe_all_samples:
        mapped_counts = mapped_counts.fillna(
            0
        )

    # ----------------------------------------------------------
    # 8. iModulon loading coverage of replication universe
    # ----------------------------------------------------------

    with MODEL.open(
        "r",
        encoding="utf-8",
    ) as handle:
        raw_model = json.load(
            handle
        )

    M = json_df(
        raw_model[
            "M"
        ]
    )

    M.index = (
        M.index
        .astype(str)
        .str.strip()
    )

    M.columns = (
        M.columns
        .astype(str)
        .str.strip()
    )

    present_set = set(
        replication_genes
    )

    present_mask = np.array(
        [
            gene in present_set
            for gene in M.index
        ],
        dtype=bool,
    )

    coverage_rows = []

    for imodulon in M.columns:
        weights = M[
            imodulon
        ].to_numpy(
            dtype=float
        )

        absolute = np.abs(
            weights
        )

        squared = weights ** 2

        coverage_rows.append(
            {
                "imodulon":
                    imodulon,

                "weighted_coverage_absM":
                    float(
                        absolute[
                            present_mask
                        ].sum()
                        / absolute.sum()
                    ),

                "energy_coverage_M2":
                    float(
                        squared[
                            present_mask
                        ].sum()
                        / squared.sum()
                    ),
            }
        )

    coverage = pd.DataFrame(
        coverage_rows
    )

    primary_table = pd.read_csv(
        PRIMARY_PROGRAMMES
    )

    primary_set = set(
        primary_table[
            "imodulon"
        ]
        .astype(str)
        .str.strip()
    )

    coverage[
        "phase3_primary"
    ] = coverage[
        "imodulon"
    ].isin(
        primary_set
    )

    median_abs = float(
        coverage[
            "weighted_coverage_absM"
        ].median()
    )

    minimum_abs = float(
        coverage[
            "weighted_coverage_absM"
        ].min()
    )

    median_m2 = float(
        coverage[
            "energy_coverage_M2"
        ].median()
    )

    minimum_m2 = float(
        coverage[
            "energy_coverage_M2"
        ].min()
    )

    robust_programmes = int(
        (
            (
                coverage[
                    "weighted_coverage_absM"
                ]
                >= 0.90
            )
            & (
                coverage[
                    "energy_coverage_M2"
                ]
                >= 0.80
            )
        ).sum()
    )

    primary_robust = int(
        (
            coverage[
                "phase3_primary"
            ]
            & (
                coverage[
                    "weighted_coverage_absM"
                ]
                >= 0.90
            )
            & (
                coverage[
                    "energy_coverage_M2"
                ]
                >= 0.80
            )
        ).sum()
    )

    print()
    print("Replication-universe iModulon coverage")
    print(
        "  median |M| coverage:",
        format(
            median_abs,
            ".2%",
        ),
    )
    print(
        "  minimum |M| coverage:",
        format(
            minimum_abs,
            ".2%",
        ),
    )
    print(
        "  median M2 coverage:",
        format(
            median_m2,
            ".2%",
        ),
    )
    print(
        "  minimum M2 coverage:",
        format(
            minimum_m2,
            ".2%",
        ),
    )
    print(
        "  robust programmes:",
        robust_programmes,
        "/ 116",
    )
    print(
        "  robust Phase-3 primary programmes:",
        primary_robust,
        "/ 111",
    )

    print()
    print("Lowest-coverage programmes")
    print(
        coverage
        .sort_values(
            "weighted_coverage_absM"
        )
        .head(
            15
        )
        .round(
            4
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 9. Technical-replicate QC
    #
    # Use log2(CPM+1) across the frozen replication universe.
    # Technical replicates must be highly concordant before
    # merging.
    # ----------------------------------------------------------

    if not full_gene_universe_all_samples:
        tech_qc = pd.DataFrame()

    else:
        cpm = mapped_counts.divide(
            mapped_counts.sum(
                axis=0
            ),
            axis=1,
        ) * 1_000_000.0

        log_cpm = np.log2(
            cpm
            + 1.0
        )

        tech_rows = []

        for biological_sample, runs in TECHNICAL_GROUPS.items():
            a = runs[
                0
            ]

            b = runs[
                1
            ]

            r = correlation(
                log_cpm[
                    a
                ],
                log_cpm[
                    b
                ],
            )

            tech_rows.append(
                {
                    "biological_sample":
                        biological_sample,

                    "run_1":
                        a,

                    "run_2":
                        b,

                    "log2_CPM_correlation":
                        r,

                    "counts_run_1":
                        float(
                            mapped_counts[
                                a
                            ].sum()
                        ),

                    "counts_run_2":
                        float(
                            mapped_counts[
                                b
                            ].sum()
                        ),
                }
            )

        tech_qc = pd.DataFrame(
            tech_rows
        )

        print()
        print("Technical-replicate QC")
        print(
            tech_qc
            .round(
                4
            )
            .to_string(
                index=False
            )
        )

    # ----------------------------------------------------------
    # 10. Merge technical replicates only after QC
    # ----------------------------------------------------------

    biological_counts = None

    if (
        full_gene_universe_all_samples
        and not tech_qc.empty
        and bool(
            (
                tech_qc[
                    "log2_CPM_correlation"
                ]
                >= 0.90
            ).all()
        )
    ):
        biological_vectors = {}

        for biological_sample, source_run in SINGLE_RUN_BIOLOGICAL.items():
            biological_vectors[
                biological_sample
            ] = mapped_counts[
                source_run
            ]

        for biological_sample, runs in TECHNICAL_GROUPS.items():
            biological_vectors[
                biological_sample
            ] = mapped_counts[
                runs
            ].sum(
                axis=1
            )

        biological_counts = pd.DataFrame(
            biological_vectors
        )

        print()
        print("Merged primary biological sputum matrix")
        print(
            "  genes:",
            biological_counts.shape[
                0
            ],
        )
        print(
            "  biological samples:",
            biological_counts.shape[
                1
            ],
        )

        print()
        print("Merged model-compatible counts")
        print(
            biological_counts.sum(
                axis=0
            )
            .astype(
                int
            )
            .to_string()
        )

    # ----------------------------------------------------------
    # 11. Published-count depth summary
    # ----------------------------------------------------------

    count_file_summary = count_files.copy()

    count_file_summary[
        "sample_class"
    ] = "OTHER"

    count_file_summary.loc[
        count_file_summary[
            "sample"
        ].isin(
            EXPECTED_INVIVO_RUNS
        ),
        "sample_class",
    ] = "PRIMARY_INVIVO_RUN"

    count_file_summary.loc[
        count_file_summary[
            "sample"
        ].isin(
            EXPECTED_MATCHED_INVITRO
        ),
        "sample_class",
    ] = "MATCHED_INVITRO"

    count_file_summary.loc[
        count_file_summary[
            "sample"
        ].isin(
            EXPECTED_PA14
        ),
        "sample_class",
    ] = "PA14_REFERENCE"

    print()
    print("Published HTSeq assigned-count summary")
    print(
        count_file_summary[
            [
                "sample",
                "sample_class",
                "gene_rows",
                "assigned_gene_counts",
                "special_counts",
            ]
        ]
        .sort_values(
            [
                "sample_class",
                "sample",
            ]
        )
        .to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 12. Audit checks
    #
    # These are technical intake checks only.
    # No CF residual replication is tested yet.
    # ----------------------------------------------------------

    min_primary_invivo_mapped_counts = np.nan

    if full_gene_universe_all_samples:
        min_primary_invivo_mapped_counts = float(
            mapped_counts[
                EXPECTED_INVIVO_RUNS
            ].sum(
                axis=0
            ).min()
        )

    all_technical_pairs_pass = (
        not tech_qc.empty
        and bool(
            (
                tech_qc[
                    "log2_CPM_correlation"
                ]
                >= 0.90
            ).all()
        )
    )

    checks = {
        "5102_Rossi_softcore_genes":
            len(
                softcore
            )
            == 5102,

        "replication_universe_ge_4800_genes":
            len(
                replication_map
            )
            >= 4800,

        "PA14_mapping_disagreement_le_25":
            len(
                disagreement
            )
            <= 25,

        "35_expected_HTSeq_files":
            len(
                count_vectors
            )
            == 35
            and len(
                missing_expected
            )
            == 0
            and len(
                unexpected
            )
            == 0,

        "no_duplicate_PA14_to_PAO1_mapping":
            len(
                duplicate_mapping_samples
            )
            == 0,

        "full_replication_gene_universe_in_all_files":
            full_gene_universe_all_samples,

        "median_imodulon_absM_coverage_ge_90pct":
            median_abs
            >= 0.90,

        "minimum_imodulon_absM_coverage_ge_80pct":
            minimum_abs
            >= 0.80,

        "at_least_105_programmes_robust":
            robust_programmes
            >= 105,

        "at_least_100_primary_programmes_robust":
            primary_robust
            >= 100,

        "all_three_technical_pairs_ge_090":
            all_technical_pairs_pass,

        "minimum_primary_invivo_mapped_counts_ge_50k":
            (
                np.isfinite(
                    min_primary_invivo_mapped_counts
                )
                and min_primary_invivo_mapped_counts
                >= 50000
            ),

        "nine_merged_primary_biological_samples":
            (
                biological_counts is not None
                and biological_counts.shape[
                    1
                ]
                == 9
            ),
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 03d checks")

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
        "PRJEB24688_PUBLISHED_COUNTS_ACCEPTED_FOR_CFSTATE_PROJECTION"
        if passed
        else "PRJEB24688_PUBLISHED_COUNT_INTAKE_REVIEW_REQUIRED"
    )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: the primary multi-patient replication universe "
        "is restricted to genes that are both in the frozen CF-State "
        "PAO1<->PA14 projection map and in the published Rossi "
        "multi-strain soft-core genome. This avoids interpreting "
        "strain-specific gene absence as transcriptional repression."
    )

    # ----------------------------------------------------------
    # 13. Save
    # ----------------------------------------------------------

    softcore.to_csv(
        OUT
        / "Rossi_PAO1_PA14_soft_coregenome.csv",
        index=False,
    )

    disagreement.to_csv(
        OUT
        / "Rossi_vs_CFSTATE_PA14_mapping_disagreements.csv",
        index=False,
    )

    replication_map.to_csv(
        OUT
        / "PRJEB24688_frozen_replication_gene_map.csv",
        index=False,
    )

    aliases.to_csv(
        OUT
        / "PRJEB24688_PA14_alias_to_PAO1_map.csv",
        index=False,
    )

    count_files.to_csv(
        OUT
        / "published_HTSeq_file_QC.csv",
        index=False,
    )

    namespace_qc.to_csv(
        OUT
        / "published_count_namespace_QC.csv",
        index=False,
    )

    coverage.to_csv(
        OUT
        / "replication_universe_imodulon_coverage.csv",
        index=False,
    )

    mapped_counts.to_csv(
        OUT
        / "PRJEB24688_run_level_PAO1_mapped_counts.csv"
    )

    tech_qc.to_csv(
        OUT
        / "primary_invivo_technical_replicate_QC.csv",
        index=False,
    )

    if biological_counts is not None:
        biological_counts.to_csv(
            OUT
            / "PRJEB24688_PRIMARY_9_biological_sputum_counts.csv"
        )

    matched_invitro_counts = mapped_counts.reindex(
        columns=EXPECTED_MATCHED_INVITRO
    )

    matched_invitro_counts.to_csv(
        OUT
        / "PRJEB24688_patient_matched_invitro_counts.csv"
    )

    pa14_reference_counts = mapped_counts.reindex(
        columns=EXPECTED_PA14
    )

    pa14_reference_counts.to_csv(
        OUT
        / "PRJEB24688_PA14_reference_counts.csv"
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
        / "AUDIT_03D_STATUS.txt"
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
            "Failed Audit 03d checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
