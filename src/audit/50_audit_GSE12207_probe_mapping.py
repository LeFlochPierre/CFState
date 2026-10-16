from pathlib import Path 
import gzip 
import re
import numpy as np
import pandas as pd


SOFT_FILE = Path(
    "data/raw/expanded_biofilm/GSE12207/"
    "GSE12207_family.soft.gz"
)

FROZEN_FILE = Path(
    "data/processed/biofilm/"
    "clean_gene_universe.csv"
)

OUT_DIR = Path(
    "results/audit/expanded_biofilm/GSE12207/"
    "probe_mapping"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


EXPECTED_SAMPLES = {
    "GSM307112": {
        "condition": "planktonic_exponential",
        "replicate": 1,
    },
    "GSM307113": {
        "condition": "planktonic_exponential",
        "replicate": 2,
    },
    "GSM307114": {
        "condition": "planktonic_exponential",
        "replicate": 3,
    },
    "GSM307115": {
        "condition": "planktonic_stationary",
        "replicate": 1,
    },
    "GSM307116": {
        "condition": "planktonic_stationary",
        "replicate": 2,
    },
    "GSM307117": {
        "condition": "planktonic_stationary",
        "replicate": 3,
    },
    "GSM307118": {
        "condition": "colony_15h",
        "replicate": 1,
    },
    "GSM307119": {
        "condition": "colony_15h",
        "replicate": 2,
    },
    "GSM307120": {
        "condition": "colony_15h",
        "replicate": 3,
    },
    "GSM307121": {
        "condition": "colony_40h",
        "replicate": 1,
    },
    "GSM307122": {
        "condition": "colony_40h",
        "replicate": 2,
    },
    "GSM307123": {
        "condition": "colony_40h",
        "replicate": 3,
    },
    "GSM307124": {
        "condition": "biofilm_3d",
        "replicate": 1,
    },
    "GSM307125": {
        "condition": "biofilm_3d",
        "replicate": 2,
    },
    "GSM307126": {
        "condition": "biofilm_3d",
        "replicate": 3,
    },
}


print()
print("=" * 100)
print(
    "CF-STATE — Audit 50: "
    "GSE12207 GPL84 probe mapping and expression audit"
)
print("=" * 100)


# ============================================================
# Frozen genes
# ============================================================

frozen = pd.read_csv(
    FROZEN_FILE
)

frozen_gene_col = None


for candidate in [
    "canonical_pao1",
    "GeneID",
    "gene",
    "locus_tag",
]:

    if candidate in frozen.columns:

        frozen_gene_col = candidate
        break


if frozen_gene_col is None:

    if frozen.shape[1] == 1:

        frozen_gene_col = frozen.columns[0]

    else:

        raise RuntimeError(
            "Could not identify frozen gene column."
        )


frozen_genes = (
    frozen[
        frozen_gene_col
    ]
    .astype(str)
    .tolist()
)


if len(frozen_genes) != 4858:

    raise RuntimeError(
        "Expected 4,858 frozen genes, found "
        + str(
            len(
                frozen_genes
            )
        )
    )


frozen_set = set(
    frozen_genes
)


# ============================================================
# Parse SOFT tables
# ============================================================

platform_header = None
platform_rows = []

sample_tables = {}
sample_headers = {}

current_sample = None

inside_platform = False
inside_sample = False


with gzip.open(
    SOFT_FILE,
    "rt",
    encoding="utf-8",
    errors="replace",
) as handle:

    for raw_line in handle:

        line = raw_line.rstrip(
            "\n"
        )


        if line.startswith(
            "^SAMPLE = "
        ):

            current_sample = (
                line.split(
                    "=",
                    1,
                )[1]
                .strip()
            )

            continue


        if line == "!platform_table_begin":

            inside_platform = True
            platform_header = None
            continue


        if line == "!platform_table_end":

            inside_platform = False
            continue


        if line == "!sample_table_begin":

            if current_sample is None:

                raise RuntimeError(
                    "Sample table encountered "
                    "without a current GSM."
                )

            inside_sample = True

            sample_headers[
                current_sample
            ] = None

            sample_tables[
                current_sample
            ] = []

            continue


        if line == "!sample_table_end":

            inside_sample = False
            continue


        if inside_platform:

            fields = line.split(
                "\t"
            )

            if platform_header is None:

                platform_header = fields

            else:

                platform_rows.append(
                    fields
                )

            continue


        if inside_sample:

            fields = line.split(
                "\t"
            )

            if sample_headers[
                current_sample
            ] is None:

                sample_headers[
                    current_sample
                ] = fields

            else:

                sample_tables[
                    current_sample
                ].append(
                    fields
                )


# ============================================================
# Platform dataframe
# ============================================================

if platform_header is None:

    raise RuntimeError(
        "GPL84 platform table was not found."
    )


platform = pd.DataFrame(
    platform_rows,
    columns=platform_header,
)


required_platform_columns = [
    "ID",
    "ORF",
]


for column in required_platform_columns:

    if column not in platform.columns:

        raise RuntimeError(
            "Missing GPL84 platform column: "
            + column
        )


if platform[
    "ID"
].duplicated().any():

    raise RuntimeError(
        "Duplicate GPL84 probe IDs detected."
    )


platform[
    "ORF"
] = (
    platform[
        "ORF"
    ]
    .astype(str)
    .str.strip()
)


platform[
    "canonical_pao1"
] = pd.NA


canonical_mask = (
    platform[
        "ORF"
    ]
    .str.fullmatch(
        r"PA\d{4}",
        na=False,
    )
)


platform.loc[
    canonical_mask,
    "canonical_pao1",
] = platform.loc[
    canonical_mask,
    "ORF",
]


n_platform = len(
    platform
)

n_exact_pao1 = int(
    canonical_mask.sum()
)

n_noncanonical = (
    n_platform
    - n_exact_pao1
)


canonical_platform = (
    platform.loc[
        canonical_mask
    ]
    .copy()
)


gene_probe_counts = (
    canonical_platform[
        "canonical_pao1"
    ]
    .value_counts()
)


unique_canonical_genes = int(
    gene_probe_counts.shape[0]
)


genes_with_multiple_probes = (
    gene_probe_counts.loc[
        gene_probe_counts
        > 1
    ]
)


n_multi_probe_genes = int(
    len(
        genes_with_multiple_probes
    )
)


max_probes_per_gene = int(
    gene_probe_counts.max()
)


platform_frozen_genes = (
    set(
        canonical_platform[
            "canonical_pao1"
        ]
    )
    & frozen_set
)


missing_frozen = sorted(
    frozen_set
    - platform_frozen_genes
)


print()
print("=" * 100)
print("GPL84 PROBE → PAO1 MAPPING")
print("=" * 100)

print(
    "GPL84 probe rows: "
    + str(
        n_platform
    )
)

print(
    "Exact PAxxxx probe rows: "
    + str(
        n_exact_pao1
    )
)

print(
    "Non-PAxxxx/control rows: "
    + str(
        n_noncanonical
    )
)

print(
    "Unique canonical PAO1 genes: "
    + str(
        unique_canonical_genes
    )
)

print(
    "Genes represented by >1 probe: "
    + str(
        n_multi_probe_genes
    )
)

print(
    "Maximum probes for one gene: "
    + str(
        max_probes_per_gene
    )
)


print()
print("=" * 100)
print("FROZEN GENE COVERAGE")
print("=" * 100)

print(
    "Frozen genes expected: 4858"
)

print(
    "Frozen genes covered by GPL84: "
    + str(
        len(
            platform_frozen_genes
        )
    )
)

print(
    "Frozen genes missing from GPL84: "
    + str(
        len(
            missing_frozen
        )
    )
)


if missing_frozen:

    print()

    print(
        "Missing frozen genes:"
    )

    print(
        ", ".join(
            missing_frozen
        )
    )


print()
print("=" * 100)
print("MULTI-PROBE GENES")
print("=" * 100)


if genes_with_multiple_probes.empty:

    print(
        "None."
    )

else:

    print(
        genes_with_multiple_probes
        .sort_values(
            ascending=False
        )
        .to_string()
    )


platform.to_csv(
    OUT_DIR
    / "GPL84_platform_with_PA_mapping.csv",
    index=False,
)


genes_with_multiple_probes.rename(
    "n_probes"
).to_csv(
    OUT_DIR
    / "GPL84_multi_probe_genes.csv"
)


pd.DataFrame(
    {
        "canonical_pao1":
            missing_frozen
    }
).to_csv(
    OUT_DIR
    / "GPL84_missing_frozen_genes.csv",
    index=False,
)


# ============================================================
# Sample table audit
# ============================================================

observed_samples = set(
    sample_tables
)


expected_samples = set(
    EXPECTED_SAMPLES
)


if observed_samples != expected_samples:

    raise RuntimeError(
        "Unexpected sample-table set. "
        "Observed: "
        + str(
            sorted(
                observed_samples
            )
        )
    )


sample_records = []

probe_series = {}

reference_probe_set = None


for gsm in sorted(
    EXPECTED_SAMPLES
):

    header = sample_headers[
        gsm
    ]


    rows = sample_tables[
        gsm
    ]


    table = pd.DataFrame(
        rows,
        columns=header,
    )


    if "ID_REF" not in table.columns:

        raise RuntimeError(
            gsm
            + " does not contain ID_REF."
        )


    if "VALUE" not in table.columns:

        raise RuntimeError(
            gsm
            + " does not contain VALUE."
        )


    if table[
        "ID_REF"
    ].duplicated().any():

        raise RuntimeError(
            gsm
            + " contains duplicate ID_REF values."
        )


    values = pd.to_numeric(
        table[
            "VALUE"
        ],
        errors="raise",
    )


    probe_ids = (
        table[
            "ID_REF"
        ]
        .astype(str)
    )


    probe_set = set(
        probe_ids
    )


    if reference_probe_set is None:

        reference_probe_set = probe_set


    same_probe_set = (
        probe_set
        == reference_probe_set
    )


    probe_series[
        gsm
    ] = pd.Series(
        values.to_numpy(),
        index=probe_ids.to_numpy(),
        name=gsm,
    )


    sample_records.append(
        {
            "gsm":
                gsm,

            "condition":
                EXPECTED_SAMPLES[
                    gsm
                ][
                    "condition"
                ],

            "replicate":
                EXPECTED_SAMPLES[
                    gsm
                ][
                    "replicate"
                ],

            "rows":
                len(
                    table
                ),

            "same_probe_set":
                same_probe_set,

            "min":
                float(
                    values.min()
                ),

            "q25":
                float(
                    values.quantile(
                        0.25
                    )
                ),

            "median":
                float(
                    values.median()
                ),

            "q75":
                float(
                    values.quantile(
                        0.75
                    )
                ),

            "max":
                float(
                    values.max()
                ),
        }
    )


sample_summary = pd.DataFrame(
    sample_records
)


print()
print("=" * 100)
print("SAMPLE EXPRESSION TABLES")
print("=" * 100)

print(
    sample_summary.to_string(
        index=False
    )
)


sample_summary.to_csv(
    OUT_DIR
    / "GSE12207_sample_expression_summary.csv",
    index=False,
)


if not sample_summary[
    "same_probe_set"
].all():

    raise RuntimeError(
        "Not all samples contain the same probe set."
    )


# ============================================================
# Probe-level matrix
# ============================================================

reference_probe_order = (
    platform[
        "ID"
    ]
    .astype(str)
    .tolist()
)


missing_platform_probes = (
    set(
        reference_probe_order
    )
    - reference_probe_set
)


extra_sample_probes = (
    reference_probe_set
    - set(
        reference_probe_order
    )
)


print()
print("=" * 100)
print("PLATFORM / SAMPLE PROBE CONSISTENCY")
print("=" * 100)

print(
    "Platform probes: "
    + str(
        len(
            reference_probe_order
        )
    )
)

print(
    "Sample probes: "
    + str(
        len(
            reference_probe_set
        )
    )
)

print(
    "Platform probes absent from samples: "
    + str(
        len(
            missing_platform_probes
        )
    )
)

print(
    "Sample probes absent from platform: "
    + str(
        len(
            extra_sample_probes
        )
    )
)


probe_matrix = pd.DataFrame(
    {
        gsm:
            probe_series[
                gsm
            ].reindex(
                reference_probe_order
            )

        for gsm in sorted(
            EXPECTED_SAMPLES
        )
    },
    index=reference_probe_order,
)


if probe_matrix.isna().any().any():

    raise RuntimeError(
        "Missing values after probe-matrix assembly."
    )


probe_matrix.index.name = (
    "probe_id"
)


probe_matrix.reset_index().to_csv(
    OUT_DIR
    / "GSE12207_probe_expression_matrix.csv",
    index=False,
)


# ============================================================
# Canonical-PA probe-level replicate QC
#
# This is still PROBE-level. We are not collapsing duplicate
# probes to genes yet.
# ============================================================

canonical_probe_ids = (
    canonical_platform[
        "ID"
    ]
    .astype(str)
    .tolist()
)


canonical_probe_matrix = (
    probe_matrix.loc[
        canonical_probe_ids
    ]
)


sample_rank = (
    canonical_probe_matrix.T
    .rank(
        axis=1,
        method="average",
        pct=True,
    )
)


correlation = (
    sample_rank.T
    .corr(
        method="spearman"
    )
)


condition_samples = {}


for gsm, info in EXPECTED_SAMPLES.items():

    condition = info[
        "condition"
    ]

    if condition not in condition_samples:

        condition_samples[
            condition
        ] = []

    condition_samples[
        condition
    ].append(
        gsm
    )


replicate_records = []


for condition, gsms in condition_samples.items():

    pair_values = []


    for i in range(
        len(
            gsms
        )
    ):

        for j in range(
            i + 1,
            len(
                gsms
            )
        ):

            rho = float(
                correlation.loc[
                    gsms[i],
                    gsms[j],
                ]
            )

            pair_values.append(
                rho
            )


    replicate_records.append(
        {
            "condition":
                condition,

            "n_samples":
                len(
                    gsms
                ),

            "median_pairwise_spearman":
                float(
                    np.median(
                        pair_values
                    )
                ),

            "min_pairwise_spearman":
                float(
                    np.min(
                        pair_values
                    )
                ),

            "max_pairwise_spearman":
                float(
                    np.max(
                        pair_values
                    )
                ),
        }
    )


replicate_df = pd.DataFrame(
    replicate_records
)


print()
print("=" * 100)
print("WITHIN-CONDITION PROBE-LEVEL REPLICATE QC")
print("=" * 100)

print(
    replicate_df.to_string(
        index=False
    )
)


replicate_df.to_csv(
    OUT_DIR
    / "GSE12207_probe_level_replicate_QC.csv",
    index=False,
)


correlation.to_csv(
    OUT_DIR
    / "GSE12207_probe_level_sample_correlations.csv"
)


print()
print("=" * 100)
print("AUDIT 50 COMPLETE")
print("=" * 100)

print(
    "No duplicate probes have been collapsed yet."
)

print(
    "Next step depends on GPL84 frozen-gene coverage "
    "and the number of multi-probe genes."
)

print()
