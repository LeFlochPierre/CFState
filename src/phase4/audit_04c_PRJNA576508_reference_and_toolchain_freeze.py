from pathlib import Path
from urllib.request import Request, urlopen
from urllib.parse import urlencode
import hashlib
import shutil
import subprocess
import time

import pandas as pd


ROOT = Path.cwd()

DECOY_TABLE = (
    ROOT
    / "results/phase4/audit_04b2_PRJNA576508_supplement_parser_repair"
    / "FROZEN_Cornforth_decoy_sequence_inventory.csv"
)

ATLANTA_METADATA = (
    ROOT
    / "results/phase4/audit_04b2_PRJNA576508_supplement_parser_repair"
    / "FROZEN_Cornforth_13_Atlanta_sample_metadata.csv"
)

ENA_MANIFEST = (
    ROOT
    / "results/phase4/audit_04a_PRJNA576508_third_cohort_preflight"
    / "PRJNA576508_Atlanta_sputum_manifest.csv"
)

REFERENCE_DIR = (
    ROOT
    / "references/phase4/PRJNA576508_cornforth"
)

DECOY_FASTA = (
    REFERENCE_DIR
    / "Cornforth_exact_104_decoy_sequences.fasta"
)

PAO1_FASTA = (
    REFERENCE_DIR
    / "PAO1_NC_002516.2.fasta"
)

OUT = (
    ROOT
    / "results/phase4/audit_04c_PRJNA576508_reference_and_toolchain_freeze"
)

EFETCH = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)

PAO1_ACCESSION = "NC_002516.2"


def md5_file(path):
    digest = hashlib.md5()

    with path.open(
        "rb"
    ) as handle:
        while True:
            block = handle.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def parse_fasta_records(text):
    records = []

    header = None
    sequence_parts = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line:
            continue

        if line.startswith(
            ">"
        ):
            if header is not None:
                records.append(
                    (
                        header,
                        "".join(
                            sequence_parts
                        ),
                    )
                )

            header = line[
                1:
            ]

            sequence_parts = []

        else:
            if header is None:
                raise RuntimeError(
                    "FASTA sequence encountered before header."
                )

            sequence_parts.append(
                line
            )

    if header is not None:
        records.append(
            (
                header,
                "".join(
                    sequence_parts
                ),
            )
        )

    return records


def fetch_fasta(accessions):
    payload = urlencode(
        {
            "db":
                "nuccore",

            "id":
                ",".join(
                    accessions
                ),

            "rettype":
                "fasta",

            "retmode":
                "text",
        }
    ).encode(
        "utf-8"
    )

    request = Request(
        EFETCH,
        data=payload,
        headers={
            "User-Agent":
                "CF-State exact reference retrieval",

            "Content-Type":
                "application/x-www-form-urlencoded",
        },
        method="POST",
    )

    with urlopen(
        request,
        timeout=120,
    ) as response:
        text = response.read().decode(
            "utf-8"
        )

    if not text.lstrip().startswith(
        ">"
    ):
        raise RuntimeError(
            "NCBI EFetch response was not FASTA:\n"
            + text[
                :500
            ]
        )

    return text


def accession_from_header(header):
    first = str(
        header
    ).split()[
        0
    ]

    # Handles simple headers and older gi|...|ref|ACC| style.
    if "|" in first:
        fields = [
            value
            for value in first.split(
                "|"
            )
            if value
        ]

        for value in reversed(
            fields
        ):
            if "." in value:
                return value

    return first


def write_wrapped_sequence(
    handle,
    header,
    sequence,
    width=80,
):
    handle.write(
        ">"
        + header
        + "\n"
    )

    for start in range(
        0,
        len(
            sequence
        ),
        width,
    ):
        handle.write(
            sequence[
                start:
                start
                + width
            ]
            + "\n"
        )


def download_exact_decoys(accessions):
    if DECOY_FASTA.exists():
        existing_text = DECOY_FASTA.read_text(
            encoding="utf-8"
        )

        existing_records = parse_fasta_records(
            existing_text
        )

        existing_accessions = {
            accession_from_header(
                header
            )
            for header, sequence
            in existing_records
        }

        if (
            len(
                existing_records
            )
            == len(
                accessions
            )
            and set(
                accessions
            )
            == existing_accessions
        ):
            print()
            print(
                "Existing exact decoy FASTA is complete:",
                DECOY_FASTA,
            )

            return existing_records

        print()
        print(
            "Existing decoy FASTA is incomplete or mismatched; "
            "rebuilding."
        )

        DECOY_FASTA.unlink()

    records_by_accession = {}

    batch_size = 20

    for start in range(
        0,
        len(
            accessions
        ),
        batch_size,
    ):
        batch = accessions[
            start:
            start
            + batch_size
        ]

        print(
            "  fetching decoy sequences",
            start + 1,
            "to",
            min(
                start
                + batch_size,
                len(
                    accessions
                ),
            ),
            "of",
            len(
                accessions
            ),
        )

        text = fetch_fasta(
            batch
        )

        records = parse_fasta_records(
            text
        )

        for header, sequence in records:
            accession = accession_from_header(
                header
            )

            if accession in records_by_accession:
                raise RuntimeError(
                    "Duplicate accession returned by NCBI: "
                    + accession
                )

            records_by_accession[
                accession
            ] = (
                header,
                sequence,
            )

        time.sleep(
            0.5
        )

    missing = [
        accession
        for accession in accessions
        if accession
        not in records_by_accession
    ]

    unexpected = sorted(
        set(
            records_by_accession
        )
        - set(
            accessions
        )
    )

    if missing:
        raise RuntimeError(
            "NCBI did not return exact sequence accessions: "
            + ", ".join(
                missing
            )
        )

    if unexpected:
        raise RuntimeError(
            "NCBI returned unexpected accessions: "
            + ", ".join(
                unexpected
            )
        )

    with DECOY_FASTA.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for accession in accessions:
            header, sequence = records_by_accession[
                accession
            ]

            write_wrapped_sequence(
                handle,
                header,
                sequence,
            )

    return [
        records_by_accession[
            accession
        ]
        for accession in accessions
    ]


def download_pao1():
    if PAO1_FASTA.exists():
        records = parse_fasta_records(
            PAO1_FASTA.read_text(
                encoding="utf-8"
            )
        )

        if (
            len(
                records
            )
            == 1
            and accession_from_header(
                records[
                    0
                ][
                    0
                ]
            )
            == PAO1_ACCESSION
        ):
            print()
            print(
                "Existing PAO1 FASTA is complete:",
                PAO1_FASTA,
            )

            return records[
                0
            ]

        PAO1_FASTA.unlink()

    print()
    print(
        "Fetching exact PAO1 reference:",
        PAO1_ACCESSION,
    )

    text = fetch_fasta(
        [
            PAO1_ACCESSION
        ]
    )

    records = parse_fasta_records(
        text
    )

    if len(
        records
    ) != 1:
        raise RuntimeError(
            "Expected one PAO1 FASTA record."
        )

    header, sequence = records[
        0
    ]

    returned_accession = accession_from_header(
        header
    )

    if returned_accession != PAO1_ACCESSION:
        raise RuntimeError(
            "PAO1 accession mismatch: "
            + returned_accession
        )

    with PAO1_FASTA.open(
        "w",
        encoding="utf-8",
    ) as handle:
        write_wrapped_sequence(
            handle,
            header,
            sequence,
        )

    return (
        header,
        sequence,
    )


def executable_info(command, args):
    path = shutil.which(
        command
    )

    if path is None:
        return {
            "tool":
                command,

            "available":
                False,

            "path":
                "",

            "version":
                "",
        }

    version = ""

    try:
        result = subprocess.run(
            [
                path,
                *args,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        output = (
            result.stdout
            + "\n"
            + result.stderr
        ).strip()

        if output:
            version = output.splitlines()[
                0
            ][
                :300
            ]

    except Exception as error:
        version = (
            type(
                error
            ).__name__
            + ": "
            + str(
                error
            )
        )

    return {
        "tool":
            command,

        "available":
            True,

        "path":
            path,

        "version":
            version,
    }


def rsubread_info():
    rscript = shutil.which(
        "Rscript"
    )

    if rscript is None:
        return {
            "tool":
                "Rsubread",

            "available":
                False,

            "path":
                "",

            "version":
                "",
        }

    code = (
        'if (requireNamespace("Rsubread", quietly=TRUE)) {'
        'cat(as.character(packageVersion("Rsubread")))'
        '} else {'
        'quit(status=2)'
        '}'
    )

    result = subprocess.run(
        [
            rscript,
            "-e",
            code,
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    if result.returncode != 0:
        return {
            "tool":
                "Rsubread",

            "available":
                False,

            "path":
                rscript,

            "version":
                "",
        }

    return {
        "tool":
            "Rsubread",

        "available":
            True,

        "path":
            rscript,

        "version":
            result.stdout.strip(),
    }


def main():
    print("=" * 78)
    print("CF-State Phase 4 - Audit 04c")
    print("Cornforth exact reference and published-pipeline toolchain freeze")
    print("=" * 78)

    for path in [
        DECOY_TABLE,
        ATLANTA_METADATA,
        ENA_MANIFEST,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                str(
                    path
                )
            )

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ----------------------------------------------------------
    # 1. Exact frozen decoy inventory
    # ----------------------------------------------------------

    decoys = pd.read_csv(
        DECOY_TABLE
    )

    if "sequence_accession" not in decoys.columns:
        raise RuntimeError(
            "Decoy inventory missing sequence_accession."
        )

    accessions = (
        decoys[
            "sequence_accession"
        ]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if len(
        accessions
    ) != 104:
        raise RuntimeError(
            "Expected 104 exact published sequence accessions, got "
            + str(
                len(
                    accessions
                )
            )
        )

    if len(
        set(
            accessions
        )
    ) != 104:
        raise RuntimeError(
            "Decoy inventory contains duplicate accessions."
        )

    print()
    print("Frozen Cornforth decoy inventory")
    print(
        "  exact sequence accessions:",
        len(
            accessions
        ),
    )

    # ----------------------------------------------------------
    # 2. Retrieve exact NCBI sequences
    # ----------------------------------------------------------

    decoy_records = download_exact_decoys(
        accessions
    )

    decoy_lengths = []

    returned_accessions = []

    for header, sequence in decoy_records:
        accession = accession_from_header(
            header
        )

        returned_accessions.append(
            accession
        )

        decoy_lengths.append(
            {
                "sequence_accession":
                    accession,

                "sequence_length_bp":
                    len(
                        sequence
                    ),
            }
        )

    decoy_lengths = pd.DataFrame(
        decoy_lengths
    )

    print()
    print("Exact decoy FASTA")
    print(
        "  records:",
        len(
            decoy_records
        ),
    )
    print(
        "  total bp:",
        int(
            decoy_lengths[
                "sequence_length_bp"
            ].sum()
        ),
    )
    print(
        "  minimum sequence bp:",
        int(
            decoy_lengths[
                "sequence_length_bp"
            ].min()
        ),
    )
    print(
        "  maximum sequence bp:",
        int(
            decoy_lengths[
                "sequence_length_bp"
            ].max()
        ),
    )
    print(
        "  MD5:",
        md5_file(
            DECOY_FASTA
        ),
    )

    # ----------------------------------------------------------
    # 3. Exact PAO1 reference
    # ----------------------------------------------------------

    pao1_header, pao1_sequence = download_pao1()

    print()
    print("Exact PAO1 reference")
    print(
        "  accession:",
        accession_from_header(
            pao1_header
        ),
    )
    print(
        "  length bp:",
        len(
            pao1_sequence
        ),
    )
    print(
        "  MD5:",
        md5_file(
            PAO1_FASTA
        ),
    )

    # ----------------------------------------------------------
    # 4. Freeze published processing method
    # ----------------------------------------------------------

    method = pd.DataFrame(
        [
            {
                "step":
                    1,

                "operation":
                    "adapter_trim",

                "published_software":
                    "Cutadapt",

                "published_version":
                    "1.13",

                "published_parameters":
                    "minimum retained read length = 25 bp",

                "important_note":
                    (
                        "Paper does not state the adapter sequence "
                        "in the methods; do not guess it before pilot "
                        "inspection."
                    ),
            },

            {
                "step":
                    2,

                "operation":
                    "decoy_alignment",

                "published_software":
                    "Bowtie2",

                "published_version":
                    "2.2.6",

                "published_parameters":
                    "default parameters; end-to-end alignment",

                "important_note":
                    (
                        "Reference is the exact non-P. aeruginosa "
                        "metagenome frozen from Data Set S1."
                    ),
            },

            {
                "step":
                    3,

                "operation":
                    "remove_decoy_mapped_reads",

                "published_software":
                    "Seqtk",

                "published_version":
                    "not stated",

                "published_parameters":
                    "remove reads mapping to decoy metagenome",

                "important_note":
                    "Keep only reads not mapped to decoys.",
            },

            {
                "step":
                    4,

                "operation":
                    "PAO1_alignment",

                "published_software":
                    "Bowtie2",

                "published_version":
                    "2.2.6",

                "published_parameters":
                    "default parameters; end-to-end alignment",

                "important_note":
                    "Reference NC_002516.2 / GCF_000006765.1.",
            },

            {
                "step":
                    5,

                "operation":
                    "gene_counting",

                "published_software":
                    "Rsubread",

                "published_version":
                    "1.26.1",

                "published_parameters":
                    (
                        "default options; SAF annotation composed "
                        "of genes without decimal points in locus tags"
                    ),

                "important_note":
                    "Comparisons performed without strand specificity.",
            },
        ]
    )

    print()
    print("Frozen published processing method")
    print(
        method.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 5. Local toolchain
    # ----------------------------------------------------------

    tool_rows = [
        executable_info(
            "cutadapt",
            [
                "--version"
            ],
        ),

        executable_info(
            "bowtie2",
            [
                "--version"
            ],
        ),

        executable_info(
            "bowtie2-build",
            [
                "--version"
            ],
        ),

        executable_info(
            "seqtk",
            [],
        ),

        executable_info(
            "Rscript",
            [
                "--version"
            ],
        ),
    ]

    tool_rows.append(
        rsubread_info()
    )

    tools = pd.DataFrame(
        tool_rows
    )

    print()
    print("Local toolchain")
    print(
        tools.to_string(
            index=False
        )
    )

    required_tools = {
        "cutadapt",
        "bowtie2",
        "bowtie2-build",
        "seqtk",
        "Rscript",
        "Rsubread",
    }

    available_tools = set(
        tools.loc[
            tools[
                "available"
            ],
            "tool",
        ]
    )

    missing_tools = sorted(
        required_tools
        - available_tools
    )

    print()
    print(
        "Missing required tools:",
        missing_tools
        if missing_tools
        else "none",
    )

    # ----------------------------------------------------------
    # 6. Deterministically freeze pilot sample
    #
    # Pilot selection uses raw sequencing depth only.
    # It does NOT use CF-State similarity or any future outcome.
    # ----------------------------------------------------------

    atlanta = pd.read_csv(
        ATLANTA_METADATA
    )

    manifest = pd.read_csv(
        ENA_MANIFEST
    )

    atlanta[
        "total reads"
    ] = pd.to_numeric(
        atlanta[
            "total reads"
        ],
        errors="raise",
    )

    median_reads = float(
        atlanta[
            "total reads"
        ].median()
    )

    atlanta[
        "distance_from_median_total_reads"
    ] = (
        atlanta[
            "total reads"
        ]
        - median_reads
    ).abs()

    pilot_row = (
        atlanta.sort_values(
            [
                "distance_from_median_total_reads",
                "normalized_EM_label",
            ]
        )
        .iloc[
            0
        ]
    )

    pilot_label = str(
        pilot_row[
            "normalized_EM_label"
        ]
    )

    manifest_match = manifest.loc[
        manifest[
            "short_sample_label"
        ]
        .astype(str)
        .eq(
            pilot_label
        )
    ]

    if len(
        manifest_match
    ) != 1:
        raise RuntimeError(
            "Could not uniquely map pilot "
            + pilot_label
            + " to Audit-04a ENA manifest."
        )

    manifest_row = manifest_match.iloc[
        0
    ]

    pilot = pd.DataFrame(
        [
            {
                "pilot_label":
                    pilot_label,

                "selection_rule":
                    (
                        "Atlanta sample with raw total read count "
                        "closest to cohort median; tie broken by label"
                    ),

                "cohort_median_total_reads":
                    median_reads,

                "pilot_total_reads":
                    int(
                        pilot_row[
                            "total reads"
                        ]
                    ),

                "published_PAO1_mapping_reads":
                    int(
                        pilot_row[
                            "reads mapping \nto PAO1 genes"
                        ]
                    ),

                "published_PAO1_genes_detected":
                    int(
                        pilot_row[
                            "number of PAO1\n genes with at least\none read mapping"
                        ]
                    ),

                "run_accession":
                    manifest_row[
                        "run_accession"
                    ],

                "sample_accession":
                    manifest_row[
                        "sample_accession"
                    ],

                "fastq_ftp":
                    manifest_row[
                        "fastq_ftp"
                    ],

                "fastq_md5":
                    manifest_row[
                        "fastq_md5"
                    ],

                "fastq_bytes":
                    manifest_row[
                        "fastq_bytes"
                    ],
            }
        ]
    )

    print()
    print("Pre-frozen technical pilot")
    print(
        pilot.to_string(
            index=False
        )
    )

    # ----------------------------------------------------------
    # 7. Freeze reference provenance
    # ----------------------------------------------------------

    reference_manifest = pd.DataFrame(
        [
            {
                "reference":
                    "Cornforth_non_Pa_decoy_metagenome",

                "records":
                    len(
                        decoy_records
                    ),

                "path":
                    str(
                        DECOY_FASTA
                    ),

                "md5":
                    md5_file(
                        DECOY_FASTA
                    ),

                "total_bp":
                    int(
                        decoy_lengths[
                            "sequence_length_bp"
                        ].sum()
                    ),
            },

            {
                "reference":
                    "PAO1_NC_002516.2",

                "records":
                    1,

                "path":
                    str(
                        PAO1_FASTA
                    ),

                "md5":
                    md5_file(
                        PAO1_FASTA
                    ),

                "total_bp":
                    len(
                        pao1_sequence
                    ),
            },
        ]
    )

    # ----------------------------------------------------------
    # 8. Audit checks
    # ----------------------------------------------------------

    exact_return = (
        returned_accessions
        == accessions
    )

    checks = {
        "104_frozen_decoy_accessions":
            len(
                accessions
            )
            == 104,

        "104_exact_decoy_sequences_retrieved":
            len(
                decoy_records
            )
            == 104,

        "NCBI_decoy_accessions_match_exact_frozen_order":
            exact_return,

        "all_decoy_sequences_nonempty":
            bool(
                (
                    decoy_lengths[
                        "sequence_length_bp"
                    ]
                    > 0
                ).all()
            ),

        "exact_PAO1_NC_002516_2_retrieved":
            (
                accession_from_header(
                    pao1_header
                )
                == PAO1_ACCESSION
                and len(
                    pao1_sequence
                )
                > 0
            ),

        "published_method_frozen":
            len(
                method
            )
            == 5,

        "pilot_uniquely_frozen":
            len(
                pilot
            )
            == 1,

        "all_required_tools_available":
            len(
                missing_tools
            )
            == 0,
    }

    passed = all(
        checks.values()
    )

    print()
    print("Audit 04c checks")

    for key, value in checks.items():
        print(
            " ",
            key,
            ":",
            "PASS"
            if value
            else "FAIL",
        )

    if passed:
        status = (
            "PRJNA576508_EXACT_REFERENCE_AND_PILOT_PIPELINE_FROZEN"
        )
    else:
        status = (
            "PRJNA576508_REFERENCE_OR_TOOLCHAIN_REVIEW_REQUIRED"
        )

    print()
    print(
        "STATUS:",
        status,
    )

    print()
    print(
        "IMPORTANT: no PRJNA576508 FASTQ has yet been processed "
        "and no Atlanta expression vector has been compared with "
        "the frozen Audit-03h CF-airway candidate."
    )

    print()
    print(
        "Adapter identity remains intentionally unresolved because "
        "the publication specifies Cutadapt and minimum length 25 "
        "but does not state the adapter sequence in the methods."
    )

    # ----------------------------------------------------------
    # 9. Save
    # ----------------------------------------------------------

    decoy_lengths.to_csv(
        OUT
        / "exact_decoy_sequence_lengths.csv",
        index=False,
    )

    reference_manifest.to_csv(
        OUT
        / "frozen_reference_manifest.csv",
        index=False,
    )

    method.to_csv(
        OUT
        / "frozen_published_processing_method.csv",
        index=False,
    )

    tools.to_csv(
        OUT
        / "local_toolchain.csv",
        index=False,
    )

    pilot.to_csv(
        OUT
        / "FROZEN_Atlanta_technical_pilot.csv",
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
        / "AUDIT_04C_STATUS.txt"
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
            "Failed Audit 04c checks: "
            + ", ".join(
                failed
            )
        )


if __name__ == "__main__":
    main()
