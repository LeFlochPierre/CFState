#!/usr/bin/env python3
"""CF-State 09h: signed regulon-shared donor response model, rigorous placebo.

Prediction target: 09b paired Zn+Fe+Mn minus untreated gene-response per donor.
Predictors: ONLY donor's untreated library-masked CPM, 09e4 literature priors.
Experimental unit: CF sputum donor. 17 leave-one-donor-out folds.

Improves on 09g's independent gene-by-gene ridge by regularising BOTH the
response intercept and the baseline-expression response coefficient towards
common signed regulator-linked parameters. Regulator effects are learned from
TRAINING donor responses only and are not claims of TF binding or causality.

A complete, degree-preserving, signed target-link shuffle is applied for each
null; regulator activities AND linked coefficients are recomputed. The same
model and hyperparameters are fit for the real and null networks. We report
single-null distributions and, separately, an averaged-null ensemble.

The primary alpha=16 and network lambda=8 are fixed prior to examining 09h
real results (alpha inherited from 09g; lambda a modest pre-specified value).
No hyperparameter is selected using held-out outcomes. Sensitivity to lambda
is a later, separately declared experiment rather than retrospective tuning.

Usage:
 python src/phase9/model_09h_signed_regulon_multitask_LODO.py --self-test
 python -u src/phase9/model_09h_signed_regulon_multitask_LODO.py | tee logs/model_09h_signed_regulon_multitask_LODO.log

No input overwritten. All scientific results printed to terminal and saved.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import audit_09g_regulatory_response_model_LODO as prev

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path('results/phase9/model_09h_signed_regulon_multitask_LODO')
ALPHA = 16.0
GRAPH_LAMBDA = 8.0
THETA_RIDGE = 1.0
N_NULLS = 40
N_BOOT = 2000
SEED = 20261015
PANELS = ['ALL_4923', 'CURATED_TARGETS', 'TRAIN16_STRONG', 'CURATED_AND_TRAIN16_STRONG']
PRIMARY_PANEL = 'CURATED_TARGETS'
CORE_METHODS = ['population_mean', 'own_baseline_ridge', 'curated_signed_multitask',
                'curated_unsigned_multitask', 'degree_matched_null_ensemble']


def need(condition, message):
    if not condition:
        raise RuntimeError('MODEL_09H_FAILED: ' + str(message))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def edges_from_rmap(genes, regs, rmap):
    # Row-indexed regulator/target mapping must be unique (one signed edge/pair).
    edges = []
    for r_idx, regulator in enumerate(regs):
        for gi, s in sorted(rmap[regulator].items()):
            need(s in (-1., 1.), 'unsigned or contradictory edge')
            need(0 <= gi < len(genes), 'target outside gene order')
            edges.append((int(r_idx), int(gi), float(s)))
    arr = np.asarray(edges, dtype=float)
    need(arr.ndim == 2 and arr.shape[1] == 3, 'no regulator-target edges')
    return arr[:, 0].astype(int), arr[:, 1].astype(int), arr[:, 2]


def network_target_mask(ng, gi):
    mask = np.zeros(ng, dtype=bool)
    mask[gi] = True
    return mask


def degree_matched_rewires(ri, gi, signs, ng, n_regs, n_nulls, seed):
    """Shuffle complete signed regulator-target neighborhoods among same-degree genes.

    This preserves each target gene's degree, regulator in-/out-degree,
    each regulator's edge signs, and the exact target panel. Nontrivial target
    permutations carry edges and signs; all activities are recomputed.
    """
    rng = np.random.default_rng(seed)
    deg = np.bincount(gi, minlength=ng)
    eligible = np.flatnonzero(deg > 0)
    groups = [np.flatnonzero(deg == k) for k in np.unique(deg[eligible])]
    groups = [v for v in groups if len(v) > 1]
    need(len(groups) >= 1, 'cannot generate valid degree-preserving rewires')
    rewires = []
    for _ in range(n_nulls):
        permutation = np.arange(ng)
        for group in groups:
            p = rng.permutation(group)
            permutation[group] = p
        if np.array_equal(permutation[eligible], eligible):
            # This is virtually impossible with real inputs; fail closed.
            raise RuntimeError('MODEL_09H_FAILED: identity placebo mapping')
        rew_gi = permutation[gi]
        need(np.array_equal(np.bincount(rew_gi, minlength=ng), deg),
             'degree-preserving shuffle changed target degrees')
        need(np.array_equal(np.bincount(ri, minlength=n_regs),
                            np.bincount(ri, minlength=n_regs)), 'regulator counts changed')
        rewires.append((ri.copy(), rew_gi, signs.copy()))
    return rewires


def baseline_train_predict(xtr, xt, ytr, alpha):
    # Standardisation is fold-local. Fold test sample NEVER affects standardisation.
    ztr, zte, _, _ = prev.standardize(xtr, xt)
    ymean = ytr.mean(axis=0)
    centered = ytr - ymean
    numer = (ztr * centered).sum(axis=0)
    den = (ztr * ztr).sum(axis=0) + alpha
    slopes = numer / den
    return ymean, slopes, numer, den, zte[0]


def multitask_prediction(ymean, numer, den, xtz, ri, gi, signs,
                         n_train, ng, n_regs, lam, theta_ridge,
                         max_iter=250, convergence=1e-10):
    """Joint convex signed multi-task shrinkage, fitted WITHOUT held-out y.

    For gene g, regulator r, literature sign s_rg and slopes beta_g:
      min sum_{i,g}(y_ig - intercept_g - z_ig * beta_g)^2
        + alpha * sum beta_g^2
        + lam * sum_(r,g) ((intercept_g - s_rg theta_r)^2
                          + (beta_g - s_rg phi_r)^2)
        + lam * theta_ridge * sum_r (theta_r^2 + phi_r^2).

    Coordinate descent obtains regulator-shared intercept theta_r and response
    slope phi_r, using only training responses and standardised untreated x.
    There is no extra fitted donor-specific latent variable at test time.
    """
    if lam == 0.0:
        return ymean + xtz * (numer / den)
    degree = np.bincount(gi, minlength=ng).astype(float)
    rdegree = np.bincount(ri, minlength=n_regs).astype(float)
    need(np.all(rdegree > 0), 'regulator without targets')
    intercept = ymean.copy()
    slope = numer / den
    for _ in range(max_iter):
        theta = np.bincount(ri, weights=signs * intercept[gi],
                            minlength=n_regs) / (rdegree + theta_ridge)
        phi = np.bincount(ri, weights=signs * slope[gi],
                          minlength=n_regs) / (rdegree + theta_ridge)
        sum_mu = np.bincount(gi, weights=signs * theta[ri], minlength=ng)
        sum_b = np.bincount(gi, weights=signs * phi[ri], minlength=ng)
        mu2 = (n_train * ymean + lam * sum_mu) / (n_train + lam * degree)
        beta2 = (numer + lam * sum_b) / (den + lam * degree)
        diff = max(float(np.max(np.abs(mu2 - intercept))),
                   float(np.max(np.abs(beta2 - slope))))
        intercept, slope = mu2, beta2
        if diff < convergence:
            break
    else:
        raise RuntimeError('MODEL_09H_FAILED: signed graph coordinate descent did not converge')
    pred = intercept + xtz * slope
    need(np.isfinite(pred).all(), 'nonfinite prediction')
    return pred


def score(y, p, mask):
    truth = y[mask]
    pred = p[mask]
    need(len(truth) > 0, 'empty evaluated panel')
    return (float(np.mean(np.abs(truth - pred))),
            float(np.sqrt(np.mean((truth - pred)**2))),
            float(np.mean(np.sign(truth) == np.sign(pred))), len(truth))


def paired_bootstrap(deltas, nboot, rng):
    v = np.asarray(deltas, dtype=float)
    indexes = rng.integers(0, len(v), size=(nboot, len(v)))
    means = v[indexes].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate(genes, donors, x, y, regs, rmap, alpha=ALPHA,
             lam=GRAPH_LAMBDA, n_nulls=N_NULLS, seed=SEED,
             nboot=N_BOOT):
    ng = len(genes)
    n = len(donors)
    need(x.shape == y.shape == (n, ng), 'input shape mismatch')
    need(n >= 6, 'need multiple donor holdouts')
    ri, gi, s = edges_from_rmap(genes, regs, rmap)
    cover = network_target_mask(ng, gi)
    rewires = degree_matched_rewires(ri, gi, s, ng, len(regs), n_nulls, seed)
    rng = np.random.default_rng(seed + 300)
    metric_rows = []
    null_rows = []
    pred_rows = []
    sel_rows = []
    for test_i, donor in enumerate(donors):
        train = np.arange(n) != test_i
        ytr = y[train]
        actual = y[test_i]
        mu, beta, numer, den, ztest = baseline_train_predict(x[train], x[~train], ytr, alpha)
        pred_mean = mu
        pred_own = mu + ztest * beta
        pred_signed = multitask_prediction(mu, numer, den, ztest, ri, gi, s,
                                           n-1, ng, len(regs), lam, THETA_RIDGE)
        pred_unsigned = multitask_prediction(mu, numer, den, ztest, ri, gi,
                                             np.ones_like(s), n-1, ng,
                                             len(regs), lam, THETA_RIDGE)
        null_predicts = [multitask_prediction(mu, numer, den, ztest,
                                             a, b, t, n-1, ng, len(regs),
                                             lam, THETA_RIDGE)
                         for (a, b, t) in rewires]
        pred_null_ensemble = np.mean(null_predicts, axis=0)
        votes = np.maximum((ytr > 0).sum(axis=0), (ytr < 0).sum(axis=0))
        min_vote = int(np.ceil(13 * (n-1) / 16))
        strong = (votes >= min_vote) & (np.abs(mu) >= 1.0)
        panels = {'ALL_4923': np.ones(ng, dtype=bool),
                  'CURATED_TARGETS': cover,
                  'TRAIN16_STRONG': strong,
                  'CURATED_AND_TRAIN16_STRONG': cover & strong}
        sel_rows.append({'donor_id': str(donor), 'n_strong_train_only': int(strong.sum()),
                         'n_covered_targets': int(cover.sum()),
                         'n_strong_and_covered': int((cover & strong).sum())})
        method_pred = dict(zip(CORE_METHODS,
                               [pred_mean, pred_own, pred_signed,
                                pred_unsigned, pred_null_ensemble]))
        for panel, mask in panels.items():
            if not np.any(mask):
                continue
            for method, prediction in method_pred.items():
                mae, rmse, signacc, n_genes = score(actual, prediction, mask)
                metric_rows.append({'donor_id':str(donor), 'panel':panel,
                                    'method':method, 'MAE':mae, 'RMSE':rmse,
                                    'sign_accuracy':signacc, 'n_genes':n_genes})
            for j, prediction in enumerate(null_predicts):
                mae, rmse, signacc, n_genes = score(actual, prediction, mask)
                null_rows.append({'donor_id':str(donor), 'panel':panel, 'null_id':j,
                                  'MAE':mae, 'RMSE':rmse,
                                  'sign_accuracy':signacc, 'n_genes':n_genes})
        pred_rows.append(pd.DataFrame({'donor_id':str(donor), 'GeneID':genes,
                                       'observed':actual, 'population_mean':pred_mean,
                                       'own_baseline_ridge':pred_own,
                                       'curated_signed_multitask':pred_signed,
                                       'curated_unsigned_multitask':pred_unsigned,
                                       'degree_matched_null_ensemble':pred_null_ensemble,
                                       'curated_target':cover, 'train16_strong':strong}))
    metrics = pd.DataFrame(metric_rows)
    null_metrics = pd.DataFrame(null_rows)
    predictions = pd.concat(pred_rows, ignore_index=True)
    selections = pd.DataFrame(sel_rows)
    main = metrics[metrics.panel == PRIMARY_PANEL].pivot(index='donor_id', columns='method', values='MAE')
    bynull = (null_metrics[null_metrics.panel == PRIMARY_PANEL]
              .groupby('null_id', as_index=False).MAE.mean())
    real_avg = float(main.curated_signed_multitask.mean())
    baseline_avg = float(main.own_baseline_ridge.mean())
    null_avgs = bynull.MAE.to_numpy()
    diff = main.curated_signed_multitask - main.own_baseline_ridge
    stats = {
        'primary_curated_targets':int(cover.sum()),
        'n_literature_edges_selected':int(len(ri)),
        'n_regulators':int(len(regs)),
        'n_donors':n,
        'curated_mean_MAE':real_avg,
        'own_mean_MAE':baseline_avg,
        'curated_minus_own_MAE':float(diff.mean()),
        'curated_better_than_own_donors':int((diff < 0).sum()),
        'curated_minus_own_donor_bootstrap_CI95':paired_bootstrap(diff, nboot, rng),
        'single_null_MAE_mean':float(null_avgs.mean()),
        'single_null_MAE_range':[float(null_avgs.min()), float(null_avgs.max())],
        'single_null_MAE_q025_q975':[float(q) for q in np.quantile(null_avgs,[.025,.975])],
        'single_nulls_beaten_by_curated':int((real_avg < null_avgs).sum()),
        'n_nulls':n_nulls,
        'null_ensemble_mean_MAE':float(main.degree_matched_null_ensemble.mean()),
        'unsigned_mean_MAE':float(main.curated_unsigned_multitask.mean()),
        'curated_better_than_unsigned_donors':int((main.curated_signed_multitask < main.curated_unsigned_multitask).sum()),
    }
    return metrics, null_metrics, predictions, selections, stats


def report(metrics, null_metrics, donors, regs, rmap, stats, alpha, lam):
    lines = []
    def say(t=''):
        t = str(t)
        print(t, flush=True)
        lines.append(t)
    say('\n' + '='*82)
    say('CF-STATE MODEL 09H | SIGNED REGULON MULTI-TASK RESPONSE PREDICTION')
    say('='*82)
    say('TASK: 17-donor LODO prediction, Zn+Fe+Mn combined intervention (same study)')
    say('INPUT: untreated-only expression, frozen paired responses, external literature links')
    say(f'Fixed ridge alpha={alpha:g}, graph lambda={lam:g}, TF shrinkage={THETA_RIDGE:g}, degree-preserving nulls={stats["n_nulls"]}')
    say(f'CURATED REGULATORS={len(regs)}, signed literature links={stats["n_literature_edges_selected"]}, covered genes={stats["primary_curated_targets"]}')
    say('Graph model: TRAIN-only gene effects jointly shrunk toward signed regulator-linked effects.')
    say('Each null rewires FULL signed network within identical target-degree classes,')
    say('preserving target degree, regulator degree, edge signs and covered gene panel.')
    say('\nREGULATORS (observed signed literature targets):')
    for r in regs:
        say(f'  {r}: {len(rmap[r])}')
    summary = (metrics.groupby(['panel','method'], as_index=False)
               .agg(mean_MAE=('MAE','mean'), mean_RMSE=('RMSE','mean'),
                    mean_sign_accuracy=('sign_accuracy','mean'),
                    mean_n_genes=('n_genes','mean')))
    say('\nHELD-OUT SUMMARY (UNWEIGHTED MEAN ACROSS DONORS):')
    for panel in PANELS:
        say('  ' + panel)
        df = summary[summary.panel == panel]
        for method in CORE_METHODS:
            t = df[df.method == method]
            if len(t):
                q = t.iloc[0]
                say(f'    {method:<31} MAE={q.mean_MAE:.4f} RMSE={q.mean_RMSE:.4f} '
                    f'sign={q.mean_sign_accuracy:.3f} n={q.mean_n_genes:.1f}')
    say('\nPRIMARY CURATED-TARGET PANEL / FAIR NULL CONTROLS:')
    say(f'  Real signed multi-task MAE       = {stats["curated_mean_MAE"]:.5f}')
    say(f'  Own-gene baseline ridge MAE     = {stats["own_mean_MAE"]:.5f}')
    say(f'  Real minus own MAE              = {stats["curated_minus_own_MAE"]:+.5f}')
    say(f'  Donor improvement count         = {stats["curated_better_than_own_donors"]}/{stats["n_donors"]}')
    say(f'  Descriptive 95% donor boot CI   = {stats["curated_minus_own_donor_bootstrap_CI95"]}')
    say(f'  Unsigned real-link MAE          = {stats["unsigned_mean_MAE"]:.5f}')
    say(f'  Single rewired MAE mean         = {stats["single_null_MAE_mean"]:.5f}')
    say(f'  Single rewired MAE 2.5-97.5%   = {stats["single_null_MAE_q025_q975"]}')
    say(f'  Real beats individual rewires   = {stats["single_nulls_beaten_by_curated"]}/{stats["n_nulls"]}')
    say(f'  Separate null ENSEMBLE MAE     = {stats["null_ensemble_mean_MAE"]:.5f}')
    say('\nPER-DONOR MAE, IDENTICAL CURATED TARGETS:')
    pv = metrics[metrics.panel == PRIMARY_PANEL].pivot(index='donor_id', columns='method', values='MAE')
    for donor in donors:
        r = pv.loc[str(donor)]
        say(f'  {donor:>4} own={r.own_baseline_ridge:.4f} signed={r.curated_signed_multitask:.4f} '
            f'unsigned={r.curated_unsigned_multitask:.4f} null_ensemble={r.degree_matched_null_ensemble:.4f}')
    say('\nLIMITS: within SAME 17-donor experiment, not independent intervention or clinical')
    say('transfer. Graph penalty assumes reported literature edge signs constrain response')
    say('similarity; signs do not prove physical TF binding or metal causality.')
    say('No outcome information from the held-out donor enters training, selection,')
    say('normalisation or placebo rewiring. All hyperparameters are fixed, no tuning.')
    say('Single-null quantiles are descriptive and are NOT inferential biological p-values.')
    say('Strong gene selection is TRAIN16 only. Corrected 09e4 source files unchanged.')
    return '\n'.join(lines) + '\n', summary


def run(root, out, n_nulls=N_NULLS):
    need(not out.exists(), 'output already exists, will not overwrite: ' + str(out))
    genes, donors, x, y, regs, rmap, links, cover, conflicts, paths = prev.load(root)
    need(np.array_equal(cover, np.array([bool(l) for l in links])),
         '09g regulator gene mask mismatch')
    metrics, nulls, predictions, selection, stats = evaluate(
        genes, donors, x, y, regs, rmap, n_nulls=n_nulls)
    txt, summary = report(metrics, nulls, donors, regs, rmap, stats, ALPHA, GRAPH_LAMBDA)
    out.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(out/'HELDOUT_MODEL_RESULTS_ALL_PANELS.csv', index=False)
    nulls.to_csv(out/'HELDOUT_DEGREE_MATCHED_SINGLE_NULLS.csv', index=False)
    predictions.to_csv(out/'HELDOUT_GENE_PREDICTIONS.csv', index=False)
    selection.to_csv(out/'HELDOUT_TRAIN_SELECTION_QC.csv', index=False)
    summary.to_csv(out/'SUMMARY_METHODS_BY_PANEL.csv', index=False)
    pd.DataFrame([{'source_path':str(p.resolve()), 'sha256':sha256(p)} for p in paths]).to_csv(
        out/'INPUT_SHA256_PROVENANCE.csv', index=False)
    status = {'status':'09H_TRAINED_SIGNED_MULTITASK_WITHIN_STUDY_NO_INDEPENDENT_VALIDATION',
              'fixed_alpha':ALPHA, 'fixed_graph_lambda':GRAPH_LAMBDA,
              'fixed_regulator_ridge':THETA_RIDGE,
              'source_study_donors':len(donors),
              'corrected_external_evidence_used':True,
              'heldout_treatment_side_unused_as_input':True,
              'all_results_descriptive':True,
              **stats}
    (out/'QC_AND_DECISION_GATE.json').write_text(json.dumps(status, indent=2) + '\n')
    (out/'SCIENTIFIC_TERMINAL_REPORT.txt').write_text(txt)
    (out/'STATUS.txt').write_text(status['status'] + '\n')
    print('\nSAVED WITHOUT OVERWRITE:', out, flush=True)


def self_test():
    rng = np.random.default_rng(109)
    n, ng = 17, 120
    genes = [f'PA{i:04d}' for i in range(1, ng+1)]
    regs = ['PA9000', 'PA9001', 'PA9002']
    rmap = {
        regs[0]: {g:1. for g in range(0, 30)},
        regs[1]: {g:-1. for g in range(30, 60)},
        regs[2]: {g:1. for g in range(60, 85)},
    }
    x = 5 + rng.normal(size=(n, ng))
    xz = (x - x.mean(0)) / x.std(0)
    # True shared regulator effects, plus gene-wise noise. The imposed true
    # regulatory structure must improve prediction with limited donors.
    mu = rng.normal(scale=.15, size=ng)
    beta = rng.normal(scale=.12, size=ng)
    for j, r in enumerate(regs):
        for gi, sign in rmap[r].items():
            mu[gi] = (1.8 + j*.45) * sign + rng.normal(scale=.10)
            beta[gi] = (.55 + j*.12) * sign + rng.normal(scale=.10)
    y = mu[None,:] + beta[None,:]*xz + rng.normal(scale=1.1, size=(n,ng))
    a = evaluate(genes, [str(i) for i in range(n)], x,y,regs,rmap,
                 n_nulls=8,seed=101,nboot=150)
    metrics, nulls, pred, selection, stats = a
    need(len(metrics)==n*len(PANELS)*len(CORE_METHODS), 'method x fold shape invalid')
    need(len(nulls)==n*len(PANELS)*8, 'degree-matched control output malformed')
    need(len(pred)==n*ng, 'holdout output size invalid')
    need(np.isfinite(metrics.MAE).all(), 'nonfinite predicted MAE')
    need(stats['curated_mean_MAE'] < stats['own_mean_MAE'],
         'true synthetic model does not improve own baseline')
    need(stats['curated_mean_MAE'] < stats['single_null_MAE_mean'],
         'true synthetic model does not improve degree-matched rewiring')
    ri,gi,ss=edges_from_rmap(genes,regs,rmap)
    rewires=degree_matched_rewires(ri,gi,ss,ng,len(regs),5,777)
    for r2,g2,s2 in rewires:
        need(np.array_equal(np.bincount(gi,minlength=ng),
                            np.bincount(g2,minlength=ng)), 'target degrees changed')
        need(np.array_equal(np.bincount(ri,minlength=len(regs)),
                            np.bincount(r2,minlength=len(regs))), 'regulator degrees changed')
        need(np.array_equal(s2,ss),'signed edges not preserved')
    print('SELF_TEST_PASSED: 17-donor synthetic LODO, 3 regulators, sign-aware')
    print('  joint multi-task intercept and slope learning; 8 matched rewired nulls;')
    print('  same-gene-degree, regulator-degree and signed-edge preservation;')
    print('  training-only scaling/selection, no held-out treatment predictor.')
    print('Synthetic true signed curated MAE =',round(stats['curated_mean_MAE'],4))
    print('Synthetic own-ridge MAE          =',round(stats['own_mean_MAE'],4))
    print('Synthetic rewired single mean MAE=',round(stats['single_null_MAE_mean'],4))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--nulls', type=int, default=N_NULLS,
                    help='Number of full degree-preserving randomized signed prior networks')
    args = ap.parse_args()
    need(1 <= args.nulls <= 500, '--nulls outside 1...500')
    if args.self_test:
        self_test()
    else:
        root = args.root.resolve()
        run(root, root / OUTPUT, n_nulls=args.nulls)


if __name__ == '__main__':
    main()
