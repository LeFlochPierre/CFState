#!/usr/bin/env python3
"""CF-State 09i: fixed-strength decomposition of 09h graph penalty.

DESCRIPTIVE DEVELOPMENT DIAGNOSTIC, NOT NEW INDEPENDENT VALIDATION.
Uses the same 17-donor Zn+Fe+Mn experiment as 09g/09h, tests the same
covered 328-gene panel without using held-out donor's treatment as input.
Dissects whether intercept sharing or slope sharing accounts for 09h's
failure. No graph strengths are tuned here: lambda=8 as in frozen 09h.

For gene g, regulator r, literature sign s_rg, penalties are
  lambda_mu * (mu_g - s_rg * theta_r)^2
  lambda_beta * (beta_g - s_rg * phi_r)^2
and separate squared penalties on the regulator-level theta and phi.

Do not interpret literal sign constraints as biological causal response signs.
No output directory is overwritten. All principal results print to terminal.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import audit_09g_regulatory_response_model_LODO as m9g
import model_09h_signed_regulon_multitask_LODO as m9h

ROOT = Path(__file__).resolve().parents[2]
OUT_REL = Path('results/phase9/model_09i_graph_penalty_component_ablation')
LAMBDA = m9h.GRAPH_LAMBDA
ALPHA = m9h.ALPHA
THETA_RIDGE = m9h.THETA_RIDGE
N_NULLS = 40
SEED = 20261015 + 91
METHODS = [
    'gene_wise_ridge',
    'signed_intercept_only',
    'signed_slope_only',
    'signed_joint_09h_replication',
    'unsigned_intercept_only',
    'unsigned_slope_only',
]
PANELS = ['ALL_4923', 'CURATED_TARGETS', 'TRAIN16_STRONG', 'CURATED_AND_TRAIN16_STRONG']


def need(ok, msg):
    if not ok:
        raise RuntimeError('MODEL_09I_FAILED: ' + str(msg))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def predict_components(ymean, numer, den, ztest, ri, gi, signs, ntrain,
                       ng, nr, lam_mu=0., lam_beta=0., theta_ridge=THETA_RIDGE,
                       max_iter=300, tolerance=1e-11):
    """Convex alternating weighted least-squares updates, fold TRAIN only.

    With independent intercept and slope penalties, each component can be
    shrunk independently. At lam_mu=lam_beta=0 exactly replicates ridge.
    """
    need(lam_mu >= 0 and lam_beta >= 0, 'negative graph penalty')
    degree = np.bincount(gi, minlength=ng).astype(float)
    rdeg = np.bincount(ri, minlength=nr).astype(float)
    need(np.all(rdeg > 0), 'empty regulator')
    mu = ymean.copy()
    beta = numer / den
    if lam_mu > 0:
        for iteration in range(max_iter):
            shared = np.bincount(ri, weights=signs * mu[gi], minlength=nr) / (rdeg + theta_ridge)
            target = np.bincount(gi, weights=signs * shared[ri], minlength=ng)
            new = (ntrain * ymean + lam_mu * target) / (ntrain + lam_mu * degree)
            if np.max(np.abs(new - mu)) < tolerance:
                mu = new
                break
            mu = new
        else:
            raise RuntimeError('MODEL_09I_FAILED: intercept graph iterations did not converge')
    if lam_beta > 0:
        for iteration in range(max_iter):
            shared = np.bincount(ri, weights=signs * beta[gi], minlength=nr) / (rdeg + theta_ridge)
            target = np.bincount(gi, weights=signs * shared[ri], minlength=ng)
            new = (numer + lam_beta * target) / (den + lam_beta * degree)
            if np.max(np.abs(new - beta)) < tolerance:
                beta = new
                break
            beta = new
        else:
            raise RuntimeError('MODEL_09I_FAILED: slope graph iterations did not converge')
    pred = mu + ztest * beta
    need(np.isfinite(pred).all(), 'invalid prediction')
    return pred


def run_evaluation(genes, donors, x, y, regs, rmap, n_nulls=N_NULLS):
    n, ng = y.shape
    need(x.shape == y.shape, 'X/Y dimensional mismatch')
    need(n >= 6, 'too few donors')
    ri, gi, signs = m9h.edges_from_rmap(genes, regs, rmap)
    covered = m9h.network_target_mask(ng, gi)
    rewires = m9h.degree_matched_rewires(ri, gi, signs, ng, len(regs), n_nulls, SEED)
    rowdata = []
    nullrows = []
    predrows = []
    cross_check_max = 0.
    for index, donor in enumerate(donors):
        train = np.arange(n) != index
        mu, beta, numer, den, ztest = m9h.baseline_train_predict(x[train], x[~train], y[train], ALPHA)
        signed_intercept = predict_components(mu, numer, den, ztest, ri, gi, signs,
                                               n-1, ng, len(regs), LAMBDA, 0)
        signed_slope = predict_components(mu, numer, den, ztest, ri, gi, signs,
                                           n-1, ng, len(regs), 0, LAMBDA)
        signed_joint = predict_components(mu, numer, den, ztest, ri, gi, signs,
                                           n-1, ng, len(regs), LAMBDA, LAMBDA)
        unsigned_intercept = predict_components(mu, numer, den, ztest, ri, gi,
                                                 np.ones_like(signs), n-1, ng,
                                                 len(regs), LAMBDA, 0)
        unsigned_slope = predict_components(mu, numer, den, ztest, ri, gi,
                                             np.ones_like(signs), n-1, ng,
                                             len(regs), 0, LAMBDA)
        own = mu + ztest * beta
        # Cross-check exact reproduction of the 09h computational implementation.
        legacy = m9h.multitask_prediction(mu, numer, den, ztest, ri, gi, signs,
                                           n-1, ng, len(regs), LAMBDA, THETA_RIDGE)
        maxerr = float(np.max(np.abs(legacy - signed_joint)))
        cross_check_max = max(cross_check_max, maxerr)
        need(maxerr < 1e-7, 'joint model failed replication of frozen 09h')
        models = dict(zip(METHODS, [own, signed_intercept, signed_slope, signed_joint,
                                   unsigned_intercept, unsigned_slope]))
        votes = np.maximum((y[train] > 0).sum(axis=0), (y[train] < 0).sum(axis=0))
        threshold = int(np.ceil(13 * (n-1) / 16))
        strong = (votes >= threshold) & (np.abs(mu) >= 1.)
        panels = {'ALL_4923': np.ones(ng, dtype=bool), 'CURATED_TARGETS': covered,
                  'TRAIN16_STRONG': strong, 'CURATED_AND_TRAIN16_STRONG': covered & strong}
        truth = y[index]
        for panel, mask in panels.items():
            if not mask.any():
                continue
            for method, prediction in models.items():
                mae, rmse, accuracy, count = m9h.score(truth, prediction, mask)
                rowdata.append({'donor_id':str(donor),'panel':panel,'method':method,
                                'MAE':mae,'RMSE':rmse,'direction_accuracy':accuracy,
                                'n_genes':count})
        for method, prediction in models.items():
            predrows.append(pd.DataFrame({'donor_id':str(donor),'GeneID':genes,
                                          'method':method,'observed':truth,
                                          'predicted':prediction}))
        # Fair null comparison on the FIXED panel and identical penalty, separately
        # for intercept-only and slope-only; no ensemble advantage.
        for null_id, (rnull, gnull, snull) in enumerate(rewires):
            for component, lmu, lbeta in [('intercept_only', LAMBDA, 0.),
                                          ('slope_only', 0., LAMBDA)]:
                pp = predict_components(mu, numer, den, ztest, rnull, gnull,
                                         snull, n-1, ng, len(regs), lmu, lbeta)
                mae, rmse, acc, count = m9h.score(truth, pp, covered)
                nullrows.append({'donor_id':str(donor),'null_id':null_id,
                                 'component':component,'MAE':mae,'RMSE':rmse,
                                 'direction_accuracy':acc,'n_genes':count})
    return (pd.DataFrame(rowdata),pd.DataFrame(nullrows),
            pd.concat(predrows, ignore_index=True),cross_check_max)


def report(metrics, null_metrics, cross_check_max, n_donors, n_regs, covered_count):
    lines = []
    def say(s=''):
        print(s,flush=True)
        lines.append(str(s))
    say('\n' + '='*80)
    say('CF-STATE MODEL 09I | FIXED GRAPH PENALTY COMPONENT ABLATION')
    say('='*80)
    say(f'Donors: {n_donors} | curated covered genes: {covered_count} | selected regulators: {n_regs}')
    say(f'Fixed inherited hyperparameters: alpha={ALPHA:g}, graph strength={LAMBDA:g}, TF shrinkage={THETA_RIDGE:g}')
    say(f'Exact numerical replication of frozen 09h signed joint prediction: max difference={cross_check_max:.3g}')
    say('Development diagnostic ONLY: same 17 donors, post-09h structural hypotheses, no new holdout.')
    say('\nHELD-OUT-DONOR METRICS (mean over donors; identical genes for all methods):')
    summary = metrics.groupby(['panel','method'],as_index=False).agg(
        MAE=('MAE','mean'),RMSE=('RMSE','mean'),
        direction_accuracy=('direction_accuracy','mean'),n_genes=('n_genes','mean'))
    for panel in PANELS:
        say('\n  ' + panel)
        sub = summary[summary.panel == panel].set_index('method')
        for method in METHODS:
            if method in sub.index:
                r = sub.loc[method]
                say(f'    {method:<32} MAE={r.MAE:.4f} RMSE={r.RMSE:.4f} '
                    f'sign={r.direction_accuracy:.3f} n={r.n_genes:.1f}')
    say('\nPRIMARY CURATED TARGETS | NETWORK-WIRING SPECIFICITY:')
    pv = metrics[metrics.panel == 'CURATED_TARGETS'].pivot(index='donor_id',columns='method',values='MAE')
    base = pv.gene_wise_ridge
    for method, comp in [('signed_intercept_only','intercept_only'),
                         ('signed_slope_only','slope_only')]:
        dif = pv[method] - base
        null_avgs = (null_metrics[null_metrics.component == comp]
                     .groupby('null_id').MAE.mean())
        real = float(pv[method].mean())
        say(f'  {method}: MAE={real:.5f}; minus gene-wise={dif.mean():+.5f}; '
            f'gene-wise beaten in {(dif < 0).sum()}/{len(dif)} donors')
        say(f'     single degree-matched rewires: mean MAE={null_avgs.mean():.5f}; '
            f'quantiles2.5-97.5={np.quantile(null_avgs,[.025,.975]).round(5).tolist()}; '
            f'curated better than {(real < null_avgs).sum()}/{len(null_avgs)}')
    say('\nPER-DONOR DIAGNOSTIC, SAME 328-GENE PANEL:')
    for donor, r in pv.iterrows():
        say(f'  {donor:>4} own={r.gene_wise_ridge:.4f} '
            f'intercept={r.signed_intercept_only:.4f} slope={r.signed_slope_only:.4f} '
            f'joint={r.signed_joint_09h_replication:.4f}')
    say('\nINTERPRETATION: This isolates graph-penalty components, not a new validation.')
    say('Post-hoc selection of the best-performing component would reuse held-out donors.')
    say('Signed TF-to-target regulation does NOT imply a signed response to metal addition.')
    say('No treatment expression was used as a prediction feature. Do not infer CF specificity.')
    return '\n'.join(lines) + '\n', summary


def execute(root, out, n_nulls=N_NULLS, data=None):
    need(not out.exists(), 'will not overwrite previous output: ' + str(out))
    if data is None:
        genes, donors, x, y, regs, rmap, _, covered, _, paths = m9g.load(root)
    else:
        genes, donors, x, y, regs, rmap, paths = data
        _, gi, _ = m9h.edges_from_rmap(genes, regs, rmap)
        covered = m9h.network_target_mask(len(genes), gi)
    metrics, nulls, predictions, deviation = run_evaluation(genes, donors, x,y,regs,rmap,n_nulls)
    stdout, summary = report(metrics,nulls,deviation,len(donors),len(regs),int(covered.sum()))
    out.mkdir(parents=True,exist_ok=False)
    metrics.to_csv(out/'LODO_FIXED_GRAPH_COMPONENT_ABLATION.csv',index=False)
    nulls.to_csv(out/'DEGREE_MATCHED_SINGLE_NULL_COMPARISONS.csv',index=False)
    predictions.to_csv(out/'HELDOUT_GENE_PREDICTIONS.csv',index=False)
    summary.to_csv(out/'SUMMARY_PANEL_METHOD.csv',index=False)
    (out/'TERMINAL_SCIENTIFIC_REPORT.txt').write_text(stdout)
    pd.DataFrame([{'source_path':str(Path(p).resolve()),'sha256':sha256(p)}
                  for p in paths]).to_csv(out/'INPUT_SHA256_PROVENANCE.csv',index=False)
    (out/'STATUS.json').write_text(json.dumps({
        'status':'09I_DIAGNOSTIC_COMPONENT_ABLATION_NO_NEW_INDEPENDENT_VALIDATION',
        'n_donors':len(donors), 'n_genes':len(genes), 'n_nulls':n_nulls,
        'n_curated_targets':int(covered.sum()),
        'fixed_graph_strength':LAMBDA,'exact_09h_replication_max_deviation':deviation,
        'claims_of_independent_external_generalisation':False},indent=2)+'\n')
    print('\nSAVED WITHOUT OVERWRITE:',out,flush=True)


def self_test():
    rng=np.random.default_rng(20261015)
    n,ng=9,68
    genes=[f'PA{i:04d}' for i in range(1,ng+1)]
    donors=[str(i) for i in range(n)]
    x=rng.normal(5,1,(n,ng))
    y= rng.normal(0,.5,(n,ng))
    regs=['PA8000','PA8001','PA8002']
    rmap={regs[0]:{i:1. for i in range(0,20)},
          regs[1]:{i:-1. for i in range(20,40)},
          regs[2]:{i:1. for i in range(40,60)}}
    args = (genes,donors,x,y,regs,rmap)
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        marker=root/'synthetic_input.txt'
        marker.write_text('source intentionally synthetic\n')
        out=root/'results'/'phase9'/'test_09i'
        execute(root,out,n_nulls=5,data=(*args,[marker]))
        assert (out/'STATUS.json').is_file()
        try:
            execute(root,out,n_nulls=5,data=(*args,[marker]))
        except RuntimeError as exc:
            need('overwrite' in str(exc), 'expected fail-closed on rerun')
        else:
            raise RuntimeError('MODEL_09I_FAILED: overwrite not prevented')
    # L=0 graph exactly equals independent gene-wise baseline.
    ii,jj,ss=m9h.edges_from_rmap(genes,regs,rmap)
    ztr=np.asarray(x[:-1]);ztest=x[-1:];yt=y[:-1]
    mu,beta,nu,den,zt=m9h.baseline_train_predict(ztr,ztest,yt,ALPHA)
    plain=mu+zt*beta
    check=predict_components(mu,nu,den,zt,ii,jj,ss,len(yt),ng,len(regs),0.,0.)
    need(np.allclose(plain,check),'lambda=0 does not reproduce gene-wise ridge')
    print('SELF_TEST_PASSED: exact 09h replication; component-specific graph penalties; '
          'degree-preserving null; no-overwrite; terminal report and provenance.',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--n-nulls',type=int,default=N_NULLS)
    opts=parser.parse_args()
    if opts.self_test:
        self_test()
        return
    need(opts.n_nulls == N_NULLS,'real-data comparison fixed at 40 degree-matched rewires')
    root=opts.root.resolve()
    out=(opts.output if opts.output else root/OUT_REL).resolve()
    execute(root,out,n_nulls=N_NULLS)

if __name__=='__main__':
    main()
