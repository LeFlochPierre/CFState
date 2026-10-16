from __future__ import annotations

import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd


AUDIT_NAME = "audit_01_aeruprecise_resource_freeze"
ROOT = Path.cwd()
OUT = ROOT / "results" / "phase2" / AUDIT_NAME
EXT = ROOT / "external" / "aeruprecise"

MODEL_REPO = "https://github.com/akanksha-r/modulome_paeru2.0.git"
DB_REPO = "https://github.com/SBRG/modulytics.git"

CFSTATE_STUDIES = [
    ("GSE136111", "binary_discovery"),
    ("GSE203348", "binary_discovery"),
    ("GSE87213", "binary_discovery"),
    ("GSE299861", "binary_later"),
    ("GSE191103", "binary_later"),
    ("GSE120602", "binary_later"),
    ("GSE12207", "binary_later"),
    ("GSE223663", "related_duplicate_excluded"),
    ("SRP009257", "specificity_control"),
]


def run(cmd: list[str]) -> str:
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def git_head(repo: str, branch_ref: str) -> str:
    out = run(["git", "ls-remote", repo, branch_ref])
    if not out:
        raise RuntimeError(f"Could not resolve {branch_ref} for {repo}")
    return out.split()[0]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    print(f"Downloading {path.name} ...")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CF-State Phase2 Audit01"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with tmp.open("wb") as handle:
            shutil.copyfileobj(response, handle)

    tmp.replace(path)


def extract_zip(zip_path: Path, dest: Path) -> list[str]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        zf.extractall(dest)
    return names


def main() -> None:
    if not (ROOT / "data").exists():
        raise RuntimeError(
            "Run this script from the CF-State repository root "
            "(expected to find ./data)."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    EXT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("CF-State Phase 2 - Audit 01")
    print("aeruPRECISE resource freeze and provenance")
    print("=" * 78)

    model_sha = git_head(MODEL_REPO, "refs/heads/main")
    db_sha = git_head(DB_REPO, "refs/heads/master")

    print(f"modulome_paeru2.0 commit: {model_sha}")
    print(f"modulytics commit:       {db_sha}")

    urls = {
        "precise411_model_json": (
            f"https://raw.githubusercontent.com/akanksha-r/"
            f"modulome_paeru2.0/{model_sha}/paeru_116_iModulon.json"
        ),
        "precise411_names": (
            f"https://raw.githubusercontent.com/akanksha-r/"
            f"modulome_paeru2.0/{model_sha}/iMod_116_name.csv"
        ),
        "precise411_imodulon_table": (
            f"https://raw.githubusercontent.com/SBRG/modulytics/{db_sha}/"
            "organisms/p_aeruginosa/precise411/data_files/iM_table.csv"
        ),
        "precise411_sample_table": (
            f"https://raw.githubusercontent.com/SBRG/modulytics/{db_sha}/"
            "organisms/p_aeruginosa/precise411/data_files/sample_table.csv"
        ),
        "precise411_dataset_meta": (
            f"https://raw.githubusercontent.com/SBRG/modulytics/{db_sha}/"
            "organisms/p_aeruginosa/precise411/dataset_meta.csv"
        ),
        "precise411_data_zip": (
            f"https://raw.githubusercontent.com/SBRG/modulytics/{db_sha}/"
            "organisms/p_aeruginosa/precise411/data_files.zip"
        ),
        "precise364_imodulon_table": (
            f"https://raw.githubusercontent.com/SBRG/modulytics/{db_sha}/"
            "organisms/p_aeruginosa/precise364/data_files/iM_table.csv"
        ),
        "precise364_sample_table": (
            f"https://raw.githubusercontent.com/SBRG/modulytics/{db_sha}/"
            "organisms/p_aeruginosa/precise364/data_files/sample_table.csv"
        ),
        "precise364_dataset_meta": (
            f"https://raw.githubusercontent.com/SBRG/modulytics/{db_sha}/"
            "organisms/p_aeruginosa/precise364/dataset_meta.csv"
        ),
        "precise364_data_zip": (
            f"https://raw.githubusercontent.com/SBRG/modulytics/{db_sha}/"
            "organisms/p_aeruginosa/precise364/data_files.zip"
        ),
    }

    paths = {
        "precise411_model_json": EXT / "precise411" / "paeru_116_iModulon.json",
        "precise411_names": EXT / "precise411" / "iMod_116_name.csv",
        "precise411_imodulon_table": EXT / "precise411" / "iM_table.csv",
        "precise411_sample_table": EXT / "precise411" / "sample_table.csv",
        "precise411_dataset_meta": EXT / "precise411" / "dataset_meta.csv",
        "precise411_data_zip": EXT / "precise411" / "data_files.zip",
        "precise364_imodulon_table": EXT / "precise364" / "iM_table.csv",
        "precise364_sample_table": EXT / "precise364" / "sample_table.csv",
        "precise364_dataset_meta": EXT / "precise364" / "dataset_meta.csv",
        "precise364_data_zip": EXT / "precise364" / "data_files.zip",
    }

    download_errors = []
    for key, url in urls.items():
        try:
            download(url, paths[key])
        except Exception as exc:
            download_errors.append((key, str(exc)))
            print(f"WARNING: failed to download {key}: {exc}")

    required = [
        "precise411_model_json",
        "precise411_imodulon_table",
        "precise411_sample_table",
        "precise411_dataset_meta",
        "precise364_imodulon_table",
        "precise364_sample_table",
        "precise364_dataset_meta",
    ]
    missing_required = [key for key in required if not paths[key].exists()]
    if missing_required:
        raise RuntimeError(
            "Required aeruPRECISE resources could not be frozen: "
            + ", ".join(missing_required)
        )

    zip_inventory_rows = []
    for model in ["precise411", "precise364"]:
        zip_key = f"{model}_data_zip"
        if not paths[zip_key].exists():
            continue

        extract_dir = EXT / model / "data_files"
        try:
            names = extract_zip(paths[zip_key], extract_dir)
            for name in names:
                zip_inventory_rows.append(
                    {"model": model, "archive_member": name}
                )
        except Exception as exc:
            print(f"WARNING: could not extract {zip_key}: {exc}")

    pd.DataFrame(zip_inventory_rows).to_csv(
        OUT / "archive_inventory.csv", index=False
    )

    manifest_rows = []
    for key, path in paths.items():
        if path.exists():
            manifest_rows.append(
                {
                    "resource": key,
                    "local_path": str(path.relative_to(ROOT)),
                    "url": urls[key],
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )

    pd.DataFrame(manifest_rows).to_csv(
        OUT / "resource_manifest.csv", index=False
    )

    revisions = pd.DataFrame(
        [
            {
                "resource": "modulome_paeru2.0",
                "repository": MODEL_REPO,
                "branch": "main",
                "commit": model_sha,
            },
            {
                "resource": "modulytics",
                "repository": DB_REPO,
                "branch": "master",
                "commit": db_sha,
            },
        ]
    )
    revisions.to_csv(OUT / "repository_revisions.csv", index=False)

    imod411 = pd.read_csv(paths["precise411_imodulon_table"])
    imod364 = pd.read_csv(paths["precise364_imodulon_table"])

    meta411 = pd.read_csv(
        paths["precise411_dataset_meta"], index_col=0
    ).iloc[:, 0]
    meta364 = pd.read_csv(
        paths["precise364_dataset_meta"], index_col=0
    ).iloc[:, 0]

    sample411 = pd.read_csv(
        paths["precise411_sample_table"],
        index_col=0,
        low_memory=False,
    )
    sample364 = pd.read_csv(
        paths["precise364_sample_table"],
        index_col=0,
        low_memory=False,
    )

    if len(imod411) != 116:
        raise RuntimeError(
            f"Expected 116 precise411 iModulons, found {len(imod411)}"
        )
    if len(imod364) != 104:
        raise RuntimeError(
            f"Expected 104 precise364 iModulons, found {len(imod364)}"
        )

    imod411.to_csv(
        OUT / "aeruprecise411_programme_catalogue.csv", index=False
    )
    imod364.to_csv(
        OUT / "aeruprecise364_programme_catalogue.csv", index=False
    )

    category_summary = (
        imod411.groupby("category", dropna=False)
        .agg(
            n_imodulons=("k", "size"),
            total_member_genes=("n_genes", "sum"),
            total_explained_variance=("exp_var", "sum"),
        )
        .reset_index()
        .sort_values(
            ["n_imodulons", "total_explained_variance"],
            ascending=[False, False],
        )
    )
    category_summary.to_csv(
        OUT / "aeruprecise411_category_summary.csv", index=False
    )

    overlap_rows = []
    overlap_details = []

    for accession, cfstate_role in CFSTATE_STUDIES:
        if accession.startswith("GSE") and "GEO Series" in sample411.columns:
            mask = sample411["GEO Series"].astype(str).eq(accession)
            matched_field = "GEO Series"
        elif accession.startswith("SRP") and "SRAStudy" in sample411.columns:
            mask = sample411["SRAStudy"].astype(str).eq(accession)
            matched_field = "SRAStudy"
        else:
            table_as_text = sample411.astype(str)
            mask = table_as_text.apply(
                lambda col: col.str.contains(accession, regex=False)
            ).any(axis=1)
            matched_field = "any_field"

        matches = sample411.loc[mask].copy()
        n_matches = len(matches)
        training_overlap = n_matches > 0

        if cfstate_role == "related_duplicate_excluded":
            phase2_status = "NOT_INDEPENDENT_CFSTATE_DUPLICATE"
        elif training_overlap:
            phase2_status = "TRAINING_OVERLAP_INTERNAL_ONLY"
        elif cfstate_role == "specificity_control":
            phase2_status = "CANDIDATE_EXTERNAL_CONTROL"
        else:
            phase2_status = "CANDIDATE_EXTERNAL"

        overlap_rows.append(
            {
                "accession": accession,
                "cfstate_role": cfstate_role,
                "precise411_accession_match": training_overlap,
                "n_matching_rows": n_matches,
                "matched_field": matched_field,
                "phase2_status": phase2_status,
                "caveat": (
                    "Confirmed overlap with published sample metadata"
                    if training_overlap
                    else
                    "No exact accession match; BioProject/run-level alias check "
                    "still recommended before final external-validation claims"
                ),
            }
        )

        if n_matches:
            keep_cols = [
                col
                for col in [
                    "Run",
                    "SRAStudy",
                    "BioProject",
                    "GEO Series",
                    "GEO Sample",
                    "BioSample",
                    "biosample_strain",
                    "strain_final",
                    "biosample_genotype",
                    "project",
                    "condition",
                    "refrence",
                    "sample",
                ]
                if col in matches.columns
            ]
            detail = matches[keep_cols].copy()
            detail.insert(0, "queried_accession", accession)
            overlap_details.append(detail.reset_index(drop=True))

    overlap = pd.DataFrame(overlap_rows)
    overlap.to_csv(
        OUT / "cfstate_aeruprecise411_overlap.csv", index=False
    )

    if overlap_details:
        pd.concat(overlap_details, ignore_index=True).to_csv(
            OUT / "cfstate_aeruprecise411_overlap_details.csv",
            index=False,
        )

    def meta_value(meta, key):
        return str(meta.get(key, "")).strip()

    model_summary = pd.DataFrame(
        [
            {
                "model": "aeruPRECISE411",
                "declared_samples": meta_value(meta411, "Number of Samples"),
                "sample_table_rows": len(sample411),
                "declared_conditions": meta_value(
                    meta411, "Number of Unique Conditions"
                ),
                "declared_genes": meta_value(meta411, "Number of Genes"),
                "declared_imodulons": meta_value(
                    meta411, "Number of iModulons"
                ),
                "programme_table_rows": len(imod411),
                "phase2_use": "PRIMARY_FROZEN_REPRESENTATION",
            },
            {
                "model": "aeruPRECISE364",
                "declared_samples": meta_value(meta364, "Number of Samples"),
                "sample_table_rows": len(sample364),
                "declared_conditions": meta_value(
                    meta364, "Number of Unique Conditions"
                ),
                "declared_genes": meta_value(meta364, "Number of Genes"),
                "declared_imodulons": meta_value(
                    meta364, "Number of iModulons"
                ),
                "programme_table_rows": len(imod364),
                "phase2_use": "SENSITIVITY_REPRESENTATION",
            },
        ]
    )
    model_summary.to_csv(OUT / "model_summary.csv", index=False)

    confirmed_overlap = overlap.loc[
        overlap["precise411_accession_match"], "accession"
    ].tolist()

    candidate_external = overlap.loc[
        overlap["phase2_status"].str.startswith("CANDIDATE_EXTERNAL"),
        "accession",
    ].tolist()

    summary_lines = [
        "# CF-State Phase 2 - Audit 01",
        "",
        "## Aim",
        "Freeze the published aeruPRECISE resources and establish "
        "training-overlap provenance before any CF-State activity inference.",
        "",
        "## Frozen representation",
        f"- Primary: aeruPRECISE411 ({len(imod411)} iModulons).",
        f"- Sensitivity: aeruPRECISE364 ({len(imod364)} iModulons).",
        f"- modulome_paeru2.0 commit: `{model_sha}`",
        f"- modulytics commit: `{db_sha}`",
        "",
        "## aeruPRECISE411 declared metadata",
        f"- Samples: {meta_value(meta411, 'Number of Samples')}",
        f"- Unique conditions: "
        f"{meta_value(meta411, 'Number of Unique Conditions')}",
        f"- Genes: {meta_value(meta411, 'Number of Genes')}",
        f"- iModulons: {meta_value(meta411, 'Number of iModulons')}",
        f"- Downloaded sample-table rows: {len(sample411)}",
        "",
        "## CF-State accession-overlap screen",
        f"- Confirmed exact accession overlap: "
        f"{', '.join(confirmed_overlap) if confirmed_overlap else 'none'}",
        f"- Candidate external accessions after exact-accession screen: "
        f"{', '.join(candidate_external) if candidate_external else 'none'}",
        "- IMPORTANT: absence of an exact accession match is not yet a "
        "complete alias/run/BioProject-level proof of independence.",
        "",
        "## Decision",
        "- Use aeruPRECISE411 as the frozen primary Phase-2 representation.",
        "- Use aeruPRECISE364 only as a later representation-sensitivity check.",
        "- Any CF-State study present in the 411 training metadata is "
        "internal/training-overlap evidence, not external validation.",
        "- Do not infer activities yet. Audit 02 will reproduce and inspect "
        "the frozen M/A/X representation before biological interpretation.",
    ]

    if download_errors:
        summary_lines.extend(["", "## Download warnings"])
        for key, err in download_errors:
            summary_lines.append(f"- {key}: {err}")

    (OUT / "AUDIT_01_SUMMARY.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print()
    print("Audit 01 complete.")
    print(f"Primary model: aeruPRECISE411 ({len(imod411)} iModulons)")
    print(f"Sensitivity model: aeruPRECISE364 ({len(imod364)} iModulons)")
    print(
        "Confirmed exact accession overlap: "
        + (", ".join(confirmed_overlap) if confirmed_overlap else "none")
    )
    print()
    print("Key outputs:")
    for filename in [
        "AUDIT_01_SUMMARY.md",
        "resource_manifest.csv",
        "repository_revisions.csv",
        "model_summary.csv",
        "aeruprecise411_programme_catalogue.csv",
        "aeruprecise411_category_summary.csv",
        "cfstate_aeruprecise411_overlap.csv",
        "cfstate_aeruprecise411_overlap_details.csv",
        "archive_inventory.csv",
    ]:
        print(f"  {OUT / filename}")


if __name__ == "__main__":
    main()
