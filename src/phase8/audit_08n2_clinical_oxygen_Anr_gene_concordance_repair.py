#!/usr/bin/env python3
"""Audit 08n2 — predeclared Anr/oxygen gene-response concordance in Cornforth clinical donors.

This is a diagnostic, read-only analysis of frozen inputs. It does NOT refit an
existing iModulon representation, calibrate array effects as RNA-seq activities,
or identify a CF-specific state. The two new CF cases are deliberately not
pooled with the 13 Cornforth samples for gene-level between-group inference.

Required: numpy, pandas, scipy. Commands:
  python src/phase8/audit_08n2_clinical_oxygen_Anr_gene_concordance_repair.py --self-test
  python -u src/phase8/audit_08n2_clinical_oxygen_Anr_gene_concordance_repair.py
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
OUT_REL = Path('results/phase8/audit_08n2_clinical_oxygen_Anr_gene_concordance_repair')
FILES = {
    'mask': Path('results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv'),
    'counts': Path('results/phase5/audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv'),
    'manifest': Path('results/phase5/audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv'),
    'array': Path('results/phase8/audit_08m_GPL84_oxygen_CFBE_Anr_gene_response/DESCRIPTIVE_all_five_array_gene_contrasts.csv'),
    'members': Path('results/phase8/audit_08m_GPL84_oxygen_CFBE_Anr_gene_response/FROZEN_Anr_and_control_top40_gene_membership.csv'),
    'markers': Path('results/phase8/audit_08m_GPL84_oxygen_CFBE_Anr_gene_response/DESCRIPTIVE_predefined_Anr_marker_cross_experiment.csv'),
    'previous_108': Path('results/phase7/audit_07i_shared_CFBE_reference_negative_controls/DESCRIPTIVE_reference_invariant_group_contrasts.csv'),
    '08j_markers': Path('results/phase8/audit_08j_GSE68534_frozen_Anr_regulator_response/DESCRIPTIVE_predefined_Anr_related_gene_markers.csv'),
}
PRIMARY_CONTRASTS = (
    'O2_2_minus_20',
    'O2_0p4_minus_20',
    'anoxic_nitrate_minus_20',
    'delta_anr_minus_WT_CFBE',
)
BOOT = 300
NULL_DRAWS = 1500
SEED = 20261015


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError('AUDIT_08N2_CHECK_FAILED: ' + message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float('nan')
    a, b = x - x.mean(), y - y.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def cosine(x: np.ndarray, y: np.ndarray) -> float:
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator > 1e-12 else float('nan')


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
    pvals = np.asarray(pvals, dtype=float)
    out = np.full(pvals.shape, np.nan)
    keep = np.flatnonzero(np.isfinite(pvals))
    if not len(keep):
        return out
    order = keep[np.argsort(pvals[keep])]
    tmp = pvals[order] * len(order) / np.arange(1, len(order) + 1)
    tmp = np.minimum.accumulate(tmp[::-1])[::-1]
    out[order] = np.clip(tmp, 0.0, 1.0)
    return out


def mean_log2cpm(counts: pd.DataFrame, pseudocount: float) -> pd.DataFrame:
    require(pseudocount > 0, 'CPM pseudocount must be positive')
    library = counts.sum(axis=0)
    require((library > 0).all(), 'zero total count library')
    # Full 5,567-gene assigned count universe, not mask-only size factors.
    cpm = counts.div(library, axis='columns') * 1_000_000.0
    return np.log2(cpm + pseudocount)


def load_clinical(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    mask_df = pd.read_csv(root / FILES['mask'], dtype={'GeneID': str})
    require('GeneID' in mask_df and len(mask_df) == 4923, 'frozen 4,923 mask')
    mask = mask_df['GeneID'].astype(str).tolist()
    require(len(set(mask)) == len(mask), 'duplicate mask gene identifiers')
    raw = pd.read_csv(root / FILES['counts'], dtype={'Geneid': str})
    require('Geneid' in raw, 'FROZEN 05h count Geneid column')
    require(raw['Geneid'].is_unique and len(raw) == 5567, '5,567 uniquely indexed genes')
    require(set(mask).issubset(set(raw.Geneid)), 'mask genes absent in 05h counts')
    raw = raw.set_index('Geneid')
    counts = raw.apply(pd.to_numeric, errors='raise')
    require(np.isfinite(counts.to_numpy()).all() and (counts.to_numpy() >= 0).all(), 'finite nonnegative raw counts')
    manifest = pd.read_csv(root / FILES['manifest'], dtype=str).fillna('')
    needed = {'published_sample', 'group', 'published_donor_group'}
    require(needed.issubset(manifest.columns), 'frozen donor-manifest columns')
    require(len(manifest) == 13 and manifest.published_sample.is_unique, '13 unique Cornforth specimens')
    require(len(counts.columns) == 13 and set(counts.columns) == set(manifest.published_sample), 'count columns match manifest')
    cf = manifest.loc[manifest['group'] == 'CF_sputum']
    noncf = manifest.loc[manifest['group'] != 'CF_sputum']
    require(len(cf) == 7 and len(noncf) == 6, 'Cornforth 7 CF and 6 wound/burn specimens')
    require(cf.published_donor_group.nunique() == 7 and noncf.published_donor_group.nunique() == 5,
            '7 independent CF and 5 non-CF donors, not 13 independent specimens')
    require('' not in set(manifest.published_donor_group), 'missing donor group')
    require(noncf['group'].astype(str).str.contains('wound|burn|non.?CF', case=False, regex=True).all(),
            'non-CF samples must be wound/burn cases, not unknown clinical groups')
    return counts, manifest, mask_df, mask, sorted(cf.published_donor_group.unique().tolist())


def grouped_expression(counts: pd.DataFrame, manifest: pd.DataFrame, pseudocount: float):
    expr = mean_log2cpm(counts, pseudocount)
    # Aggregate any repeated specimens WITHIN each donor before group means.
    metadata = manifest.set_index('published_sample').loc[expr.columns]
    series = {}
    roles = {}
    for donor, parts in metadata.groupby('published_donor_group', sort=True):
        samples = parts.index.tolist()
        groups = parts['group'].unique().tolist()
        require(len(groups) == 1, 'mixed clinical group within donor ' + donor)
        role = 'CF' if groups[0] == 'CF_sputum' else 'NONCF_WOUND_BURN'
        roles[donor] = role
        series[donor] = expr[samples].mean(axis=1)
    donor_mat = pd.DataFrame(series).loc[counts.index]
    cf = sorted(d for d, role in roles.items() if role == 'CF')
    noncf = sorted(d for d, role in roles.items() if role == 'NONCF_WOUND_BURN')
    require(len(cf) == 7 and len(noncf) == 5, 'expected 7 versus 5 independent donors')
    contrast = donor_mat[cf].mean(axis=1) - donor_mat[noncf].mean(axis=1)
    return donor_mat, contrast, cf, noncf, roles


def load_experiment(root: Path, mask: list[str]):
    data = pd.read_csv(root / FILES['array'], dtype={'GeneID': str})
    expected_cols = {'dataset', 'contrast', 'GeneID', 'gene_signal_change', 'n_treatment', 'n_reference'}
    require(expected_cols.issubset(data.columns), '08m gene table schema')
    require(~data.duplicated(['dataset', 'contrast', 'GeneID']).any(), 'duplicate experiment GeneID/contrast')
    data['gene_signal_change'] = pd.to_numeric(data['gene_signal_change'], errors='raise')
    require(np.isfinite(data['gene_signal_change']).all(), 'finite array changes')
    require(set(data.GeneID).issubset(set(mask)), 'experiment contains genes outside frozen mask')
    patterns = {
        'O2_2_minus_20': ('GSE6741', 2, 2),
        'O2_0p4_minus_20': ('GSE6741', 2, 2),
        'anoxic_nitrate_minus_20': ('GSE6741', 2, 2),
        'delta_anr_minus_WT_CFBE': ('GSE67006', 3, 3),
    }
    experiment = {}
    for name, (ds, a, b) in patterns.items():
        rows = data.loc[(data.contrast == name) & (data.dataset == ds)].copy()
        require(len(rows) == 4920, f'{name} must include 4,920 genes (observed {len(rows)})')
        require((pd.to_numeric(rows.n_treatment) == a).all() and
                (pd.to_numeric(rows.n_reference) == b).all(), f'{name} replicate design')
        experiment[name] = rows.set_index('GeneID')['gene_signal_change']
    shared = sorted(set.intersection(*(set(x.index) for x in experiment.values())))
    require(len(shared) == 4920, 'same 4,920 mapped genes across all four primary contrasts')
    require(set(mask) - set(shared) == {'PA1899', 'PA4022', 'PA4277'},
            'unexpected microarray-frozen-mask non-overlap')
    # Anr-intact direction is the SIGN-REVERSED CFBE delta_anr minus WT array effect.
    # Keep original raw direction as a separate output for transparent provenance.
    lab = pd.DataFrame({k: v.loc[shared] for k, v in experiment.items()}, index=shared)
    lab['WT_minus_delta_anr_CFBE'] = -lab['delta_anr_minus_WT_CFBE']
    return lab, data


def load_panels(root: Path, valid_genes: set[str]):
    df = pd.read_csv(root / FILES['members'], dtype={'GeneID': str})
    require({'imodulon', 'rank_abs_loading', 'GeneID', 'frozen_loading'}.issubset(df.columns),
            'frozen Anr/control top40 schema')
    df['rank_abs_loading'] = pd.to_numeric(df.rank_abs_loading, errors='raise')
    df['frozen_loading'] = pd.to_numeric(df.frozen_loading, errors='raise')
    require(not df.duplicated(['imodulon', 'GeneID']).any(), 'duplicate genes within a programme')
    panels = {}
    for im, rows in df.groupby('imodulon', sort=True):
        if im.startswith('Anr-'):
            require(len(rows) == 40, f'{im} expected frozen top40 genes')
        rows = rows.sort_values('rank_abs_loading')
        rows = rows.loc[rows.GeneID.isin(valid_genes)]
        if len(rows) >= 10:
            panels['TOP40_' + im] = list(zip(rows.GeneID.tolist(), rows.frozen_loading.tolist()))
    require(all('TOP40_Anr-' + str(i) in panels for i in (1, 2, 3, 4)),
            'four frozen Anr loading sets required')
    pre = pd.read_csv(root / FILES['markers'], dtype={'GeneID': str})
    require({'gene_symbol', 'GeneID'}.issubset(pre.columns), 'predeclared marker table schema')
    require(pre.GeneID.is_unique, '08m duplicated marker GeneID')
    respiratory = pre.loc[pre.gene_symbol != 'anr', ['GeneID', 'gene_symbol']]
    require(len(respiratory) == 7 and set(respiratory.GeneID).issubset(valid_genes),
            'seven predeclared respiratory markers available')
    panels['PREDECLARED_SEVEN_RESPIRATORY_MARKERS'] = [(g, 1.0) for g in respiratory.GeneID]
    return panels, pre


def sign_agreement(x: np.ndarray, y: np.ndarray, min_abs: float = 0.0):
    good = (np.abs(x) > max(min_abs, 1e-9)) & (np.abs(y) > max(min_abs, 1e-9)) & np.isfinite(x) & np.isfinite(y)
    n = int(good.sum())
    return (float((np.sign(x[good]) == np.sign(y[good])).mean()) if n else float('nan'), n)


def matched_bins(baseline: pd.Series, genes: list[str]) -> np.ndarray:
    # Only clinical mean expression is available for matching (not microarray baseline intensity).
    rank = baseline.reindex(genes).rank(method='first').to_numpy()
    require(np.isfinite(rank).all(), 'baseline clinical expression finite')
    return np.minimum((rank * 10 / (len(rank) + 1)).astype(int), 9)


def panel_nulls(panels, lab: pd.DataFrame, cl: pd.Series, baseline: pd.Series,
                rng: np.random.Generator, n_null: int = NULL_DRAWS):
    genes = lab.index.to_list()
    pos = {g: i for i, g in enumerate(genes)}
    strata = matched_bins(baseline, genes)
    bin_members = {i: np.flatnonzero(strata == i) for i in range(10)}
    c = cl.reindex(genes).to_numpy(float)
    records = []
    lab_keys = list(PRIMARY_CONTRASTS)  # No direction selection after observing results.
    # Anr-intact is fixed in advance as sign reversal for KO, for interpretability.
    lab_keys = [k if k != 'delta_anr_minus_WT_CFBE' else 'WT_minus_delta_anr_CFBE' for k in lab_keys]
    for panel, items in sorted(panels.items()):
        indices = np.array([pos[g] for g, _ in items], dtype=int)
        weights = np.array([w for _, w in items], dtype=float)
        valid = np.isfinite(weights) & (np.abs(weights) > 0)
        indices, weights = indices[valid], weights[valid]
        require(len(indices) >= 5, 'too few genes in panel ' + panel)
        # Match each panel gene to the same clinical-expression decile.
        rand = np.empty((n_null, len(indices)), dtype=np.int32)
        for stratum in np.unique(strata[indices]):
            panel_columns = np.flatnonzero(strata[indices] == stratum)
            pool = bin_members[int(stratum)]
            require(len(pool) >= len(panel_columns), 'stratum too small for matched gene-set draw')
            for j in range(n_null):
                rand[j, panel_columns] = rng.choice(pool, size=len(panel_columns), replace=False)
        for contrast in lab_keys:
            e = lab[contrast].to_numpy(float)
            observed, n = sign_agreement(c[indices], e[indices])
            # Exclude near-zero responses from BOTH observed and null denominators.
            # Otherwise zero-response genes could bias the enrichment p value.
            null_good = (np.abs(c[rand]) > 1e-9) & (np.abs(e[rand]) > 1e-9)
            null_den = null_good.sum(axis=1)
            null = np.divide(
                ((np.sign(c[rand]) == np.sign(e[rand])) & null_good).sum(axis=1),
                null_den, out=np.full(n_null, np.nan), where=(null_den > 0))
            finite_null = null[np.isfinite(null)]
            require(len(finite_null) >= n_null * 0.9, 'too many zero-only matched null sets')
            require(np.isfinite(observed), 'all-zero clinical/experimental responses in panel ' + panel)
            empirical_greater = (1.0 + int(np.sum(finite_null >= observed - 1e-12))) / (len(finite_null) + 1.0)
            observed_pearson = pearson(c[indices], e[indices])
            proj_c = float(np.dot(weights, c[indices]) / np.abs(weights).sum())
            proj_e = float(np.dot(weights, e[indices]) / np.abs(weights).sum())
            records.append({
                'panel': panel,
                'category': 'ANR_TOP40' if panel.startswith('TOP40_Anr-') else
                            ('PREDECLARED_RESPIRATORY' if panel.startswith('PREDECLARED_') else 'COMPARISON_TOP40'),
                'experimental_contrast_oriented': contrast,
                'n_genes': n,
                'fraction_gene_signs_matching': observed,
                'matched_random_median_sign_match': float(np.median(finite_null)),
                'matched_random_q025': float(np.quantile(finite_null, 0.025)),
                'matched_random_q975': float(np.quantile(finite_null, 0.975)),
                'empirical_one_sided_p_descriptive': empirical_greater,
                'within_panel_pearson_gene_effects': observed_pearson,
                'clinical_loading_weighted_change': proj_c if panel.startswith('TOP40_') else float('nan'),
                'experimental_loading_weighted_change': proj_e if panel.startswith('TOP40_') else float('nan'),
                'loading_signed_changes_same_sign': bool(proj_c * proj_e > 0) if panel.startswith('TOP40_') else None,
                'matching_covariate': 'Cornforth clinical mean log2(CPM+1) decile only',
                'interpretation': 'exploratory matched-gene-set descriptive null; no CF specificity inference',
            })
    ans = pd.DataFrame(records)
    ans['BH_q_all_reported_panels_contrasts'] = fdr_bh(ans['empirical_one_sided_p_descriptive'].to_numpy(float))
    return ans


def bootstrap_correlations(donor: pd.DataFrame, cf: list[str], noncf: list[str],
                           lab: pd.DataFrame, rng: np.random.Generator, reps: int = BOOT):
    indexed = donor.loc[lab.index]
    x = indexed[cf].to_numpy(float)
    y = indexed[noncf].to_numpy(float)
    boots = np.empty((reps, x.shape[0]), dtype=float)
    for b in range(reps):
        # Resample independent donors rather than 13 specimens.
        i = rng.integers(0, len(cf), len(cf))
        j = rng.integers(0, len(noncf), len(noncf))
        boots[b] = x[:, i].mean(axis=1) - y[:, j].mean(axis=1)
    rows = []
    for contrast in PRIMARY_CONTRASTS:
        name = 'WT_minus_delta_anr_CFBE' if contrast == 'delta_anr_minus_WT_CFBE' else contrast
        e = lab[name].to_numpy(float)
        vals = np.array([spearman(boots[b], e) for b in range(reps)])
        rows.append({
            'experimental_contrast_oriented': name,
            'donor_bootstrap_reps': reps,
            'bootstrap_spearman_q025': float(np.quantile(vals, .025)),
            'bootstrap_spearman_median': float(np.median(vals)),
            'bootstrap_spearman_q975': float(np.quantile(vals, .975)),
            'note': 'descriptive donor resampling, clinical groups not exchangeable disease controls',
        })
    return pd.DataFrame(rows)


def whole_genome(cl: pd.Series, lab: pd.DataFrame):
    c = cl.loc[lab.index].to_numpy(float)
    rows = []
    for contrast in PRIMARY_CONTRASTS:
        name = 'WT_minus_delta_anr_CFBE' if contrast == 'delta_anr_minus_WT_CFBE' else contrast
        e = lab[name].to_numpy(float)
        sign, n = sign_agreement(c, e)
        # Strong-effect agreement guards against nearly-zero values driving signs.
        strong_sign, strong_n = sign_agreement(c, e, min_abs=0.5)
        rows.append({
            'clinical_contrast': 'CORNFORTH_CF_minus_wound_burn_7v5_log2CPM',
            'experimental_contrast_oriented': name,
            'n_genes': len(c),
            'pearson_effect_concordance': pearson(c, e),
            'spearman_effect_concordance': spearman(c, e),
            'cosine_effect_concordance': cosine(c, e),
            'fraction_effect_signs_matching': sign,
            'n_sign_compared': n,
            'fraction_both_abs_changes_gt_0p5_sign_matching': strong_sign,
            'n_both_abs_changes_gt_0p5': strong_n,
            'units': 'clinical log2(CPM+1); microarray within-study transformed log signal difference',
            'note': 'effect-direction comparison only; not calibrated amplitudes or CF-specificity',
        })
    return pd.DataFrame(rows)


def sensitivity(counts: pd.DataFrame, manifest: pd.DataFrame, lab: pd.DataFrame, mask: list[str]):
    out = []
    for pseudo in (0.25, 1.0, 2.0):
        donor, contrast, cf, noncf, roles = grouped_expression(counts, manifest, pseudo)
        c = contrast.loc[lab.index].to_numpy(float)
        for exp in PRIMARY_CONTRASTS:
            name = 'WT_minus_delta_anr_CFBE' if exp == 'delta_anr_minus_WT_CFBE' else exp
            e = lab[name].to_numpy(float)
            out.append({'log2CPM_pseudocount': pseudo, 'experimental_contrast_oriented': name,
                        'spearman_genome': spearman(c, e), 'cosine_genome': cosine(c, e),
                        'description': 'gene-level sensitivity, NOT projected iModulon scores'})
    return pd.DataFrame(out)


def run(root: Path, out: Path, boot: int, null_draws: int, seed: int) -> None:
    require(not out.exists(), f'output directory already exists; DO NOT OVERWRITE: {out}')
    missing = [str(path) for path in FILES.values() if not (root / path).is_file()]
    require(not missing, 'frozen input file(s) missing: ' + '; '.join(missing))
    counts, manifest, mask_df, mask, cf_list = load_clinical(root)
    donor, cl, cf, noncf, roles = grouped_expression(counts, manifest, 1.0)
    lab, source_lab = load_experiment(root, mask)
    panels, markers = load_panels(root, set(lab.index))
    prior = pd.read_csv(root / FILES['previous_108'])
    require({'contrast', 'signed_projection'}.issubset(prior.columns), '07i 108-D contrast schema')
    check = prior.loc[prior.contrast == 'Cornforth_CF_minus_nonCF_7v5', 'signed_projection']
    require(len(check) == 1 and abs(float(check.iloc[0]) - 12.395840842375051) < 0.02,
            'freeze 07i CF clinical contrast diagnostic must match previous audit')
    # The 108-D signed score is a provenance guard, NOT used as a gene-level target.
    rng = np.random.default_rng(seed)
    print('AUDIT 08N2 | Cornforth primary clinical gene-level (7 CF donors v 5 wound/burn donors)')
    print('Loaded clinical genes:', len(mask), '| shared laboratory genes:', len(lab))
    print('Clinical comparison: donor-balanced log2(CPM+1), 7 CF vs 5 wound/burn (13 samples).')
    print('Cross-study NEW CF (2) vs Cornforth controls is NOT being treated as a within-study gene contrast.')
    main = whole_genome(cl, lab)
    boots = bootstrap_correlations(donor, cf, noncf, lab, rng, boot)
    mean_expr = donor.mean(axis=1)
    matched = panel_nulls(panels, lab, cl, mean_expr, rng, null_draws)
    sens = sensitivity(counts, manifest, lab, mask)
    # 08j independent RNA-seq knockout markers: retain both strains and both
    # native delta_anr-minus-WT and oriented intact-Anr sign conventions.
    earlier_all = pd.read_csv(root / FILES['08j_markers'], dtype={'PAO1_locus': str, 'strain': str})
    expected_08j = {'gene_symbol', 'PAO1_locus', 'strain', 'delta_anr_minus_WT_log2norm'}
    require(expected_08j.issubset(earlier_all.columns), '08j marker schema')
    # 08j may include additional rows/labels. Do not infer biological backgrounds
    # from those rows or discard them silently. Verify precisely the two contrasts
    # already reproduced by the frozen 08m marker cross-experiment table.
    original_strains = earlier_all['strain'].fillna('<MISSING>').astype(str).str.strip()
    selected = original_strains.isin(['PAO1', 'J215'])
    unexpected = original_strains.loc[~selected].value_counts(dropna=False).to_dict()
    print('08j marker rows: total=', len(earlier_all), '| PAO1/J215 rows=', int(selected.sum()),
          '| other/blank label counts=', unexpected)
    earlier = earlier_all.loc[selected].copy()
    earlier['strain'] = original_strains.loc[selected].to_numpy()
    require(set(earlier.strain.unique()) == {'PAO1', 'J215'},
            '08j must include both PAO1 and J215 knockout backgrounds')
    require(not earlier.duplicated(['PAO1_locus', 'strain']).any(),
            '08j duplicate marker gene within PAO1/J215 background')
    # Each of eight predeclared markers must have exactly one finite effect
    # for each of the two named strains. Other labels remain unused and reported.
    marker_ids = set(markers.GeneID.astype(str))
    earlier = earlier.loc[earlier['PAO1_locus'].isin(marker_ids)].copy()
    earlier['delta_anr_minus_WT_log2norm'] = pd.to_numeric(
        earlier['delta_anr_minus_WT_log2norm'], errors='raise')
    require(np.isfinite(earlier['delta_anr_minus_WT_log2norm'].to_numpy(float)).all(),
            '08j selected marker effects must be finite')
    require(len(earlier) == 2 * len(marker_ids),
            '08j requires 8 predeclared markers with one PAO1 and one J215 effect each')
    earlier_piv = earlier.pivot(index='PAO1_locus', columns='strain', values='delta_anr_minus_WT_log2norm')
    require(set(earlier_piv.index) == marker_ids and set(earlier_piv.columns) == {'PAO1', 'J215'},
            '08j complete marker-by-two-background coverage')
    # Strong provenance check against the independent 08m extraction (same
    # original 08j effects, already used to establish 7/7 marker concordance).
    for strain in ('PAO1', 'J215'):
        col = '08j_' + strain + '_delta_anr_minus_WT'
        require(col in markers.columns, '08m marker cross-experiment missing ' + col)
        before = pd.to_numeric(markers.set_index('GeneID').loc[earlier_piv.index, col], errors='raise')
        now = earlier_piv[strain].astype(float)
        require(np.isfinite(before.to_numpy()).all() and
                np.allclose(now.to_numpy(), before.to_numpy(), rtol=1e-6, atol=1e-6),
                '08j-vs-08m frozen knockout marker effect mismatch: ' + strain)
    mark_rows = []
    for row in markers.itertuples(index=False):
        gene = str(row.GeneID)
        if gene not in lab.index:
            continue
        data = {'gene_symbol': str(row.gene_symbol), 'GeneID': gene,
                'clinical_CF_minus_wound_burn_log2CPM': float(cl.loc[gene])}
        for name in list(PRIMARY_CONTRASTS) + ['WT_minus_delta_anr_CFBE']:
            data[name] = float(lab.loc[gene, name])
        for strain in ('PAO1', 'J215'):
            value = float(earlier_piv.loc[gene, strain])
            require(np.isfinite(value), '08j marker finite: ' + gene + '/' + strain)
            data['08j_' + strain + '_delta_anr_minus_WT'] = value
            data['08j_' + strain + '_WT_minus_delta_anr'] = -value
        mark_rows.append(data)
    # Reproduce the previously reported 7/7 respiratory-marker KO sign replication.
    mark_check = pd.DataFrame(mark_rows)
    resp_check = mark_check.loc[mark_check.gene_symbol != 'anr']
    for strain in ('PAO1', 'J215'):
        a = resp_check['delta_anr_minus_WT_CFBE'].to_numpy(float)
        b = resp_check['08j_' + strain + '_delta_anr_minus_WT'].to_numpy(float)
        n_match = int(np.sum(np.sign(a) == np.sign(b)))
        require(n_match == 7, 'reproduce 08m 7/7 respiratory-marker KO sign concordance: ' + strain)
    specimen_qc = manifest[['published_sample', 'group', 'published_donor_group']].copy()
    specimen_qc['assigned_counts_over_5567_genes'] = specimen_qc.published_sample.map(
        counts.sum(axis=0).to_dict())
    specimen_qc['frozen_mask_genes_detected_ge1'] = specimen_qc.published_sample.map(
        (counts.loc[mask] >= 1).sum(axis=0).to_dict())
    specimen_qc['frozen_mask_genes_detected_ge10'] = specimen_qc.published_sample.map(
        (counts.loc[mask] >= 10).sum(axis=0).to_dict())
    donor_rows = []
    for d, role in sorted(roles.items()):
        n_specimens = int((manifest.published_donor_group == d).sum())
        donor_rows.append({'donor': d, 'group': role, 'n_specimens_averaged': n_specimens})
    clinical_alone = cl.loc[lab.index]
    # Include clinical expression estimates and exactly the original 08m gene effects.
    vectors = lab.copy()
    vectors.insert(0, 'clinical_CF_minus_wound_burn_log2CPM', clinical_alone)
    vectors.index.name = 'GeneID'
    provenance = []
    for label, relative in FILES.items():
        provenance.append({'source': label, 'path': str(relative), 'sha256': sha(root / relative)})
    script = Path(__file__).resolve()
    provenance.append({'source': 'audit_08n2_script', 'path': str(script), 'sha256': sha(script)})
    limitations = '''AUDIT 08N2 — INTERPRETATION AND METHODS\n\nPrimary clinical data: 13 Cornforth bacterial RNA-seq specimens, 7 CF sputum donors and 5 non-CF wound/burn donors (two burn specimens are averaged within one donor). Counts converted to log2(CPM+1), with size factors computed over the COMPLETE 5,567-gene count matrix and independent donor averages computed before groups. This is a new descriptive gene-level contrast; it is NOT the frozen TPM or iModulon projection and is NOT a new trained model. CPM pseudocount sensitivity: 0.25, 1, 2.\n\nPerturbations: frozen 08m gene_signal_change within-study transformed microarray signal, never pooled at absolute-expression level with clinical RNA-seq. Oxygen effects are 2%-20%, 0.4%-20%, anoxia PLUS 100mM NITRATE versus 20% (nitrate confounded). CFBE WT-minus-delta_anr is predeclared sign reversal of observed delta_anr-minus-WT: anr deletion is NOT an oxygen dose. Microarray and clinical units are different. Effect Spearman, cosine and sign agreement describe transfer of relative GENE RESPONSE DIRECTIONS, NOT calibrated iModulon activities. No expression scale matching, score fractions or explained-variance claims.\n\nPredeclared seven respiratory markers; pre-existing 08j RNA-seq knockout markers (PAO1, J215, both 2v2) cross-checked numerically against the frozen 08m extraction and used to reproduce 7/7 knockout-effect signs separately in each strain; any other 08j row labels are reported and excluded from this two-background comparison; and reference-loading top40 Anr-1 through Anr-4; other pre-existing top40 sets are comparison pathways NOT necessarily null/negative controls. Loading weighted metrics use original loadings on raw per-gene effects, not calibrated model projection. Empirical null: 1,500 random unique gene sets matched only on rank-decile of mean clinical log2(CPM+1), not matched on microarray baseline expression, gene detectability, length or network dependence. Empirical one-sided p and BH q (across all displayed panel-perturbation comparisons) are EXPLORATORY, are not biological replicate p-values, and do not establish CF specificity. Strong effects require both |changes| >0.5 on their own scales and are descriptive.\n\nBootstrap intervals resample independent 7 CF and 5 non-CF donors with replacement, not individual specimens. Wound/burn and CF sputum differ in infection site, strains, geography, patient population and possibly processing; non-CF wound is not a matched non-CF AIRWAY control. New CF01 and CF146 cases were separate studies and are not pooled in 08n2 gene-level disease contrasts; 07i frozen 108-D contrasts remain supporting context only, and both 108-D positive comparisons SUBTRACT THE SAME Cornforth non-CF group. The 13 Atlanta samples do not imply 13 independent donors. No unmeasured metal availability/oxygen concentration/CF-specificity inference. FDR is descriptive conditional on gene-panel selection and correlated genes. Anr programme contributions should not be conflated with O2 concentrations.\n\nDecision gate: Look for coherent oxygen-only AND independent regulator-direction concordance in respiratory markers and Anr sets, robust to donors and CPM pseudocount, exceeding matched-gene null with limited control-pathway similarity. If absent or mixed, report honestly: no evidence of respiratory programme transfer to the clinical CF-associated contrast, and do not add latent axes based on an unexplained residual. If present, interpret as shared regulatory transcriptional patterns across settings, never CF specificity.\n'''
    # Create output dir only after all computations have successfully completed.
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(provenance).to_csv(out / 'INPUT_SHA256_PROVENANCE.csv', index=False)
    pd.DataFrame(donor_rows).to_csv(out / 'FROZEN_Cornforth_12_donor_roles_and_specimen_counts.csv', index=False)
    pd.DataFrame(
        [{'raw_strain_label': str(k), 'n_rows': int(v),
          'handling': 'excluded from specified PAO1/J215 comparisons'} for k, v in unexpected.items()],
        columns=['raw_strain_label', 'n_rows', 'handling']).to_csv(
            out / 'DIAGNOSTIC_08j_unselected_strain_label_counts.csv', index=False)
    specimen_qc.to_csv(out / 'DIAGNOSTIC_13_Cornforth_specimen_count_depth_and_detection.csv', index=False)
    vectors.to_csv(out / 'DESCRIPTIVE_4920_shared_clinical_and_experimental_gene_effects.csv')
    main.to_csv(out / 'DESCRIPTIVE_whole_genome_gene_response_concordance.csv', index=False)
    boots.to_csv(out / 'DESCRIPTIVE_donor_bootstrap_whole_genome_concordance.csv', index=False)
    matched.to_csv(out / 'EXPLORATORY_expression_matched_panel_concordance.csv', index=False)
    pd.DataFrame(mark_rows).to_csv(out / 'DESCRIPTIVE_predeclared_Anr_respiratory_marker_clinical_changes.csv', index=False)
    sens.to_csv(out / 'SENSITIVITY_3_log2CPM_pseudocounts.csv', index=False)
    (out / 'INTERPRETATION_LIMITS.txt').write_text(limitations)
    summary_lines = [
        'AUDIT 08N2 | Clinically associated oxygen/Anr gene-response concordance',
        'Primary: 7 Cornforth CF sputum donors vs 5 wound/burn patients.',
        'NO new model fit; NO CF-versus-non-CF-airway specificity test.',
        '',
        main[['experimental_contrast_oriented', 'spearman_effect_concordance',
              'fraction_effect_signs_matching']].to_string(index=False),
        '', 'EXPLORATORY Anr top40 and predeclared marker panels:',
        matched.loc[matched.category.isin(['ANR_TOP40', 'PREDECLARED_RESPIRATORY']),
                    ['panel', 'experimental_contrast_oriented', 'fraction_gene_signs_matching',
                     'BH_q_all_reported_panels_contrasts']].to_string(index=False),
        '', 'These are descriptive results; see INTERPRETATION_LIMITS.txt before interpretation.',
    ]
    (out / 'SUMMARY_FINDINGS.txt').write_text('\n'.join(summary_lines) + '\n')
    (out / 'AUDIT_08N2_STATUS.txt').write_text('CFSTATE_PHASE8_08N2_CLINICAL_ANR_OXYGEN_GENE_CONCORDANCE_DESCRIPTIVE_COMPLETE\n')
    print('\nPrimary whole-genome response concordance:')
    print(main[['experimental_contrast_oriented', 'spearman_effect_concordance',
                'fraction_effect_signs_matching']].to_string(index=False))
    main_anr = matched[matched.category == 'ANR_TOP40']
    print('\nAnr panels:')
    print(main_anr[['panel', 'experimental_contrast_oriented', 'fraction_gene_signs_matching',
                    'BH_q_all_reported_panels_contrasts']].to_string(index=False))
    print('\nCOMPLETED:', out)


def self_test() -> None:
    require(np.allclose(fdr_bh(np.array([.01, .02, .5])), [.03, .03, .5]), 'BH self test')
    require(np.isnan(pearson(np.array([1., 1., 1.]), np.array([2., 3., 4.]))), 'constant Pearson guard')
    require(np.isclose(spearman(np.array([1., 3., 2.]), np.array([3., 1., 2.])), -1.), 'Spearman test')
    require(sign_agreement(np.array([1., -1., 1.]), np.array([2., 2., 2.])) == (2. / 3., 3),
            'sign test')
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        mask = [f'PA{i:04d}' for i in range(1, 4924)]
        absent = {'PA1899', 'PA4022', 'PA4277'}
        labs = [g for g in mask if g not in absent]
        for path in FILES.values():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'GeneID': mask}).to_csv(root / FILES['mask'], index=False)
        genes = [f'PA{i:04d}' for i in range(1, 5568)]
        samples = [f'Human_Sputum_{chr(65 + i)}' for i in range(7)]
        samples += ['Lubbock_CW_15', 'Lubbock_CW_16', 'Lubbock_CW_18', 'Lubbock_CW_21',
                    'Lubbock_Burn_2_3', 'Lubbock_Burn_2_4']
        rng = np.random.default_rng(4)
        count_data = rng.poisson(200.0, size=(len(genes), len(samples)))
        count_data[:80, :7] += 150
        counts = pd.DataFrame(count_data, columns=samples)
        counts.insert(0, 'Geneid', genes)
        counts.to_csv(root / FILES['counts'], index=False)
        grp = ['CF_sputum'] * 7 + ['nonCF_chronic_wound'] * 4 + ['nonCF_burn'] * 2
        donors = ['distinct_patient_' + s for s in samples[:11]] + ['shared_Lubbock_burn_patient'] * 2
        manifest = pd.DataFrame({'published_sample': samples, 'group': grp, 'published_donor_group': donors})
        manifest.to_csv(root / FILES['manifest'], index=False)
        rows = []
        for name in PRIMARY_CONTRASTS:
            ds = 'GSE67006' if name == 'delta_anr_minus_WT_CFBE' else 'GSE6741'
            n = 3 if ds == 'GSE67006' else 2
            for g in labs:
                v = float(rng.normal())
                if g in mask[:80]:
                    v = -1. if ds == 'GSE67006' else 1.
                rows.append({'dataset': ds, 'contrast': name, 'GeneID': g, 'gene_signal_change': v,
                             'n_treatment': n, 'n_reference': n})
        pd.DataFrame(rows).to_csv(root / FILES['array'], index=False)
        panels = []
        for name in ['Anr-1', 'Anr-2', 'Anr-3', 'Anr-4', 'Zur-2']:
            for idx, g in enumerate(labs[:40]):
                panels.append({'imodulon': name, 'rank_abs_loading': idx + 1, 'GeneID': g,
                               'frozen_loading': 1. / (idx + 1)})
        pd.DataFrame(panels).to_csv(root / FILES['members'], index=False)
        marker_syms = ['anr', 'narG', 'narH', 'nirS', 'nirM', 'norB', 'nosZ', 'hemN']
        pd.DataFrame({'gene_symbol': marker_syms, 'GeneID': labs[:8],
                      '08j_PAO1_delta_anr_minus_WT': [-1.0] * 8,
                      '08j_J215_delta_anr_minus_WT': [-1.0] * 8}).to_csv(
                          root / FILES['markers'], index=False)
        pd.DataFrame([{'contrast': 'Cornforth_CF_minus_nonCF_7v5',
                       'signed_projection': 12.395840842375051}]).to_csv(root / FILES['previous_108'], index=False)
        jrows = []
        for strain in ('PAO1', 'J215'):
            for k, sym in enumerate(marker_syms):
                jrows.append({'gene_symbol': sym, 'PAO1_locus': labs[k], 'strain': strain,
                              'delta_anr_minus_WT_log2norm': -1.0})
        jrows.append({'gene_symbol': 'other', 'PAO1_locus': 'PA5000', 'strain': 'UNASSIGNED',
                      'delta_anr_minus_WT_log2norm': float('nan')})
        jrows.append({'gene_symbol': 'other', 'PAO1_locus': 'PA5001', 'strain': '',
                      'delta_anr_minus_WT_log2norm': float('nan')})
        pd.DataFrame(jrows).to_csv(root / FILES['08j_markers'], index=False)
        out = root / OUT_REL
        run(root, out, boot=8, null_draws=30, seed=SEED)
        require((out / 'AUDIT_08N2_STATUS.txt').is_file(), 'synthetic run completion')
        mismatch = pd.read_csv(root / FILES['markers'])
        mismatch.loc[0, '08j_PAO1_delta_anr_minus_WT'] = -3.0
        mismatch.to_csv(root / FILES['markers'], index=False)
        try:
            run(root, root / 'results/phase8/bad_provenance_test', boot=8, null_draws=30, seed=SEED)
        except RuntimeError as exc:
            require('08j-vs-08m frozen knockout marker effect mismatch' in str(exc),
                    'cross-audit mismatch must be rejected')
        else:
            raise AssertionError('mismatching 08j and 08m marker effects accepted')
        require(not (root / 'results/phase8/bad_provenance_test').exists(),
                'failed provenance test must not create output directory')
        try:
            run(root, out, boot=8, null_draws=30, seed=SEED)
        except RuntimeError as exc:
            require('DO NOT OVERWRITE' in str(exc), 'non-overwrite guard')
        else:
            raise AssertionError('existing audit was overwritten')
    print('SELF_TEST_PASSED: syntax-level math, synthetic files, donor balancing and no-overwrite')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--bootstrap', type=int, default=BOOT)
    p.add_argument('--null-draws', type=int, default=NULL_DRAWS)
    p.add_argument('--seed', type=int, default=SEED)
    args = p.parse_args()
    require(args.bootstrap >= 5 and args.null_draws >= 10, 'at least 5 bootstraps and 10 null draws')
    if args.self_test:
        self_test()
        return
    root = args.root.expanduser().resolve()
    run(root, root / OUT_REL, args.bootstrap, args.null_draws, args.seed)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('AUDIT 08N2 ERROR:', str(exc), file=sys.stderr)
        raise
