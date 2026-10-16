#!/usr/bin/env python3
"""CF-STATE 09g: first FITTED prior-constrained metal response model, donor-held-out.

Inputs: frozen 09b donor-paired metal effects and untreated samples; corrected
09e4 literature regulator/target evidence. A model predicts the metal-minus-
untreated gene response from a donor's *untreated* expression only.

Tests an actual learnable hypothesis (whether prior-guided regulator features
improve response prediction), NOT transfer to a new perturbation or clinical
patients. Keep all findings descriptive given 17 donors and shared study.

Commands:
  python audit_09g_regulatory_response_model_LODO.py --self-test
  python audit_09g_regulatory_response_model_LODO.py

No results are overwritten. Prints complete scientific summary directly to stdout.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import audit_09b_Neff2024_paired_gene_response_freeze as b
import audit_09c_LODO_metal_gene_response_generalisation as c

ROOT = Path(__file__).resolve().parents[2]
PRIOR = Path('results/phase9/audit_09e4_PAO1_suffix_locus_integrity_repair/FROZEN_CURATED_REGULATORY_ASSOCIATIONS_09E4.csv')
OUTPUT = Path('results/phase9/audit_09g_regulatory_response_model_LODO')
ALPHA = 16.0  # deliberately fixed, no tuning on held-out test donors
N_TOP_REGULATORS = 12
N_PLACEBO = 5
SEED = 20261015
PANELS = ['ALL_4923', 'CURATED_TARGETS', 'TRAIN16_STRONG', 'CURATED_AND_TRAIN16_STRONG']
METHODS = ['population_mean', 'own_baseline_ridge', 'curated_regulatory_ridge', 'shuffled_prior_ensemble']


def require(check, message):
    if not check:
        raise RuntimeError('AUDIT_09G_FAILED: ' + str(message))


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load(root):
    genes, donors, response, _, sources = c.load(root)
    qc = b.load_pairing(root)
    mask = b.read_mask(root)
    require(mask == genes, '09b and 09c frozen gene orders differ')
    counts, _, _ = b.load_counts(root, genes, qc)
    baseline = []
    for donor in donors:
        sub = qc[(qc.donor_id == donor) & (qc.condition == 'UNTREATED')]
        require(len(sub) == 1, 'missing or duplicated untreated donor sample ' + donor)
        sample = sub.iloc[0]
        key = sample['sample_match_key']
        untreated_raw = counts[key].to_numpy(float)
        total = float(untreated_raw.sum())
        require(total > 0, 'zero untreated library for donor ' + donor)
        # Untreated-only model input: library-masked CPM, independent of the
        # frozen cross-condition size factors used to define the response.
        baseline.append(np.log2(1.0 + untreated_raw / total * 1e6))
    x = np.stack(baseline)
    require(x.shape == response.shape == (17, 4923), 'unexpected baseline and effect arrays')
    require(np.isfinite(x).all() and np.isfinite(response).all(), 'invalid baseline or effects')
    # Repeat one important source-consistency check independently.
    z = pd.read_csv(root / b.OUTPUT / 'FROZEN_17_donors_x_4923_genes_paired_effects.csv')
    require(np.allclose(response, z.iloc[:, 1:].to_numpy(float).T, atol=1e-10),
            'frozen 09b gene effects mismatch')
    path = root / PRIOR
    require(path.is_file(), 'missing frozen corrected 09e4 evidence: ' + str(path))
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    mandatory = ['regulator_GeneID', 'target_GeneID', 'raw_mode_of_regulation',
                 'original_strain_includes_PAO1', 'row_status']
    require(set(mandatory).issubset(df.columns), 'unexpected 09e4 regulatory evidence columns')
    require((df.row_status == 'CURATED_MAPPED').all(), 'unexpected source row statuses')
    df = df[(df.original_strain_includes_PAO1.str.lower() == 'true') &
            (df.raw_mode_of_regulation.isin(['+', '-'])) &
            df.regulator_GeneID.str.fullmatch(r'PA\d{4}') &
            df.target_GeneID.str.fullmatch(r'PA\d{4}')].copy()
    gene_to_ix = {g: i for i, g in enumerate(genes)}
    df = df[df.target_GeneID.isin(gene_to_ix)]
    require(len(df) > 0, 'no independent curated signed PAO1 target edges')
    # A pair reported both ways is excluded, never assigned a convenient sign.
    df['sign'] = df.raw_mode_of_regulation.map({'+': 1.0, '-': -1.0})
    conflicts = df.groupby(['regulator_GeneID','target_GeneID'])['sign'].nunique()
    conflicts = set(conflicts[conflicts > 1].index)
    df = df[[tuple(v) not in conflicts for v in df[['regulator_GeneID','target_GeneID']].itertuples(index=False, name=None)]]
    df = df.drop_duplicates(['regulator_GeneID','target_GeneID'])
    grouped = {reg: {gene_to_ix[g]: float(s) for g, s in d[['target_GeneID','sign']].itertuples(index=False, name=None)}
               for reg, d in df.groupby('regulator_GeneID')}
    selected = sorted([reg for reg in grouped if len(grouped[reg]) >= 5],
                      key=lambda reg: (-len(grouped[reg]), reg))[:N_TOP_REGULATORS]
    require(len(selected) >= 2, 'fewer than two multi-target regulators available')
    rmap = {r: grouped[r] for r in selected}
    per_gene = [[] for _ in genes]
    for j, r in enumerate(selected):
        for gi in rmap[r]:
            per_gene[gi].append(j)
    targets = np.array([bool(v) for v in per_gene], dtype=bool)
    require(targets.sum() >= 15, 'fewer than 15 regulator-covered measured genes')
    paths = [sources['09b_effects'], root / b.COUNTS, root / b.PAIR,
             root / PRIOR, root / b.MASK]
    return genes, donors, x, response, selected, rmap, per_gene, targets, len(conflicts), paths


def standardize(train_x, test_x):
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd = np.where(sd >= 0.2, sd, 1.0)
    return (train_x - mu) / sd, (test_x - mu) / sd, mu, sd


def features_for_gene(ztrain, ztest, gi, regs, rmap, chosen):
    """Leave-target-out regulator activities; never reuse the target gene as its own regulon."""
    tr = [ztrain[:, gi]]
    te = [ztest[:, gi]]
    for ri in chosen:
        r = regs[ri]
        pairs = [(idx, sgn) for idx, sgn in rmap[r].items() if idx != gi]
        if len(pairs) < 2:
            continue
        idx = [q[0] for q in pairs]
        signs = np.asarray([q[1] for q in pairs])
        atr = (ztrain[:, idx] @ signs) / len(idx)
        ate = (ztest[:, idx] @ signs) / len(idx)
        scale = max(float(atr.std()), 0.25)
        tr.append((atr - atr.mean()) / scale)
        te.append((ate - atr.mean()) / scale)
    return np.column_stack(tr), np.column_stack(te)


def fit_ridge_predict(train_design, test_design, y, alpha):
    ymean = y.mean()
    centered = y - ymean
    p = train_design.shape[1]
    beta = np.linalg.solve(train_design.T @ train_design + alpha * np.eye(p),
                           train_design.T @ centered)
    return ymean + test_design @ beta, beta


def placebo_assignments(per_gene, genes_mask, n, seed):
    """Fixed placebo: shuffle regulator assignments among the SAME covered gene panel."""
    indices = np.flatnonzero(genes_mask)
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(n):
        perm = rng.permutation(len(indices))
        require(not np.array_equal(perm, np.arange(len(indices))), 'unexpected identity shuffle')
        d = [list(v) for v in per_gene]
        for gi, src in zip(indices, indices[perm]):
            d[gi] = list(per_gene[src])
        result.append(d)
    return result


def fold_predictions(xtr, xt, ytr, regs, rmap, links, target_mask, placebo, alpha):
    ztrain, ztest, _, _ = standardize(xtr, xt)
    mean = ytr.mean(axis=0)
    own = mean.copy()
    graph = mean.copy()
    nulls = np.broadcast_to(mean, (len(placebo), len(mean))).copy()
    # Own-gene untreated baseline is known for every gene; fit an identical
    # ridge model whether or not literature regulatory coverage exists.
    for gi in range(xtr.shape[1]):
        xx = ztrain[:, gi]
        yy = ytr[:, gi] - mean[gi]
        beta = float((xx @ yy) / ((xx @ xx) + alpha))
        own[gi] = mean[gi] + ztest[0, gi] * beta
        graph[gi] = own[gi]
        nulls[:, gi] = own[gi]
    for gi in np.flatnonzero(target_mask):
        tr, te = features_for_gene(ztrain, ztest, gi, regs, rmap, links[gi])
        graph[gi] = fit_ridge_predict(tr, te, ytr[:, gi], alpha)[0][0]
        for k, plinks in enumerate(placebo):
            tr0, te0 = features_for_gene(ztrain, ztest, gi, regs, rmap, plinks[gi])
            nulls[k, gi] = fit_ridge_predict(tr0, te0, ytr[:, gi], alpha)[0][0]
    return {'population_mean': mean, 'own_baseline_ridge': own,
            'curated_regulatory_ridge': graph, 'shuffled_prior_ensemble': nulls.mean(axis=0)}


def metrics(y, pred, mask):
    if not np.any(mask):
        return {'n': 0, 'mae': np.nan, 'rmse': np.nan, 'direction_accuracy': np.nan}
    actual = y[mask]
    predicted = pred[mask]
    return {'n': int(mask.sum()), 'mae': float(np.abs(actual-predicted).mean()),
            'rmse': float(np.sqrt(np.mean((actual-predicted)**2))),
            'direction_accuracy': float((np.sign(actual) == np.sign(predicted)).mean())}


def test_model(genes, donors, x, response, regs, rmap, per_gene, covered,
               alpha=ALPHA, n_placebo=N_PLACEBO, seed=SEED):
    require(x.shape == response.shape, 'baseline/response dimensions differ')
    require(x.shape[0] == len(donors) and x.shape[1] == len(genes), 'dimensional mismatch')
    placebo = placebo_assignments(per_gene, covered, n_placebo, seed)
    results, predictions, top = [], [], []
    for ti, donor in enumerate(donors):
        tr = np.arange(len(donors)) != ti
        train_y = response[tr]
        holdout_y = response[ti]
        mean = train_y.mean(axis=0)
        votes = np.maximum((train_y > 0).sum(axis=0), (train_y < 0).sum(axis=0))
        # Selection uses ONLY the other donors, never the held-out response.
        min_votes = max(1, int(np.ceil(13 * tr.sum() / 16)))
        strong = (votes >= min_votes) & (np.abs(mean) >= 1)
        panels = {'ALL_4923': np.ones(len(genes), bool),
                  'CURATED_TARGETS': covered,
                  'TRAIN16_STRONG': strong,
                  'CURATED_AND_TRAIN16_STRONG': covered & strong}
        preds = fold_predictions(x[tr], x[~tr], train_y, regs, rmap,
                                 per_gene, covered, placebo, alpha)
        for pname, mask in panels.items():
            for model, prediction in preds.items():
                results.append({'donor_id': donor, 'panel': pname, 'method': model,
                                **metrics(holdout_y, prediction, mask)})
        for model, prediction in preds.items():
            predictions.append(pd.DataFrame({'donor_id': donor, 'GeneID': genes,
                                             'method': model, 'predicted': prediction,
                                             'observed': holdout_y}))
        top.append({'donor_id': donor,
                    'n_strong_train_only': int(strong.sum()),
                    'n_strong_and_regulated': int((strong & covered).sum()),
                    'n_literature_regulated_genes': int(covered.sum())})
    return pd.DataFrame(results), pd.concat(predictions, ignore_index=True), pd.DataFrame(top)


def final_fit(x, y, genes, regs, rmap, links, covered, alpha):
    ztrain, _, mu, sd = standardize(x, x)
    rows = []
    for gi, gid in enumerate(genes):
        own = np.column_stack([ztrain[:, gi]])
        base_beta = fit_ridge_predict(own, own, y[:, gi], alpha)[1][0]
        reg_coefs = {r: 0. for r in regs}
        if covered[gi]:
            train, _ = features_for_gene(ztrain, ztrain, gi, regs, rmap, links[gi])
            prediction, beta = fit_ridge_predict(train, train, y[:, gi], alpha)
            # Re-derive valid feature identifiers in the same order.
            kept = []
            for ri in links[gi]:
                if len(rmap[regs[ri]]) - (gi in rmap[regs[ri]]) >= 2:
                    kept.append(regs[ri])
            require(len(kept) == len(beta)-1, 'feature and regulator coefficient mismatch')
            for r, value in zip(kept, beta[1:]):
                reg_coefs[r] = float(value)
            own_coef = float(beta[0])
        else:
            own_coef = float(base_beta)
        row = {'GeneID':gid, 'mean_training_response':float(y[:, gi].mean()),
               'untreated_gene_mean':float(mu[gi]), 'untreated_gene_scale':float(sd[gi]),
               'beta_own_baseline':own_coef,
               'literature_regulatory_target_covered':bool(covered[gi]),
               'n_regulators_in_model':len(links[gi])}
        row.update({'beta_' + r:reg_coefs[r] for r in regs})
        rows.append(row)
    return pd.DataFrame(rows)


def report(results, donors_qc, regs, rmap, covered, n_conflict, alpha, nplacebo):
    lines = []
    def say(v=''):
        v = str(v)
        print(v)
        lines.append(v)
    say('\n' + '='*78)
    say('CF-STATE 09G | FIRST TRAINED REGULATORY RESPONSE MODEL | 17 DONOR LODO')
    say('='*78)
    say('TASK: predict metal-minus-untreated responses using UNTREATED donor expression only')
    say('Input: log2(untreated-only masked-counts CPM+1), no treatment-side factor used')
    say('Models: 16-donor training, 1 held-out donor. No outcome-based test selection.')
    say(f'Primary ridge alpha={alpha:g} FIXED; curated regulators={len(regs)}; placebo maps={nplacebo}')
    say(f'Gene targets with >=1 chosen curated signed regulator: {covered.sum()}')
    say(f'Opposing-literature-sign pairs excluded: {n_conflict}')
    say('\nSELECTED LITERATURE PRIOR FEATURES (PAO1 including mixed, >=5 observed targets):')
    for r in regs:
        say(f'  {r}: {len(rmap[r])} gene targets in frozen response mask')
    say('\nHELD-OUT DONOR PREDICTION (UNWEIGHTED MEAN ACROSS 17 DONORS):')
    summary = results.groupby(['panel','method'], as_index=False).agg(
        mean_MAE=('mae','mean'), mean_RMSE=('rmse','mean'),
        mean_direction_accuracy=('direction_accuracy','mean'),
        mean_n_evaluated=('n','mean'))
    for name in PANELS:
        say('\n  Panel: ' + name)
        sub = summary[summary.panel == name]
        for meth in METHODS:
            row = sub[sub.method == meth]
            if len(row):
                r = row.iloc[0]
                say(f'    {meth:<27} MAE={r.mean_MAE:>7.4f}  RMSE={r.mean_RMSE:>7.4f}  '
                    f'sign={r.mean_direction_accuracy:>6.3f}  mean genes={r.mean_n_evaluated:.1f}')
    for panel in ['ALL_4923','CURATED_TARGETS','CURATED_AND_TRAIN16_STRONG']:
        p = results[results.panel == panel].pivot(index='donor_id', columns='method', values='mae')
        diff_own = p.curated_regulatory_ridge - p.own_baseline_ridge
        diff_perm = p.curated_regulatory_ridge - p.shuffled_prior_ensemble
        say('\n  MAE DIFFERENCES ON ' + panel + ' (negative = curated prior improves):')
        say(f'    vs own-baseline: mean={diff_own.mean():+.5f};  donors improving={(diff_own < 0).sum()}/17')
        say(f'    vs shuffled-prior: mean={diff_perm.mean():+.5f};  donors improving={(diff_perm < 0).sum()}/17')
    say('\nPER-DONOR CURATED-TARGET MAE:')
    q = results[results.panel == 'CURATED_TARGETS'].pivot(index='donor_id', columns='method', values='mae')
    for donor, row in q.iterrows():
        say(f'  {donor:>4} | mean={row.population_mean:.4f} own={row.own_baseline_ridge:.4f} '
            f'curated={row.curated_regulatory_ridge:.4f} shuffled={row.shuffled_prior_ensemble:.4f}')
    say('\nCAUTION: same ex-vivo 17-donor metal experiment. This is MODEL learning and within-study')
    say('donor holdout, NOT an independent perturbation test. Regulatory mode is literature-reported,')
    say('not proof of direct binding. Other-strain and dotted-locus evidence excluded.')
    say('Strong-gene panels are selected within train16 only. No metadata leakage from metal outcomes.')
    say('The shuffled-prior ensemble averages five fixed incorrect-link model predictions;')
    say('this is a competitive ablation, not a permutation p-value or a full null distribution.')
    return '\n'.join(lines)+'\n', summary


def run(root, out):
    require(not out.exists(), 'output exists; never overwrite: ' + str(out))
    genes, donors, x, y, regs, rmap, links, covered, n_conflict, paths = load(root)
    data, pred, donor_qc = test_model(genes, donors, x, y, regs, rmap, links, covered)
    fitted = final_fit(x, y, genes, regs, rmap, links, covered, ALPHA)
    txt, summary = report(data, donor_qc, regs, rmap, covered, n_conflict, ALPHA, N_PLACEBO)
    output = out
    output.mkdir(parents=True, exist_ok=False)
    data.to_csv(output/'HELDOUT_17_DONOR_MODEL_METRICS.csv', index=False)
    pred.to_csv(output/'HELDOUT_ALL_GENE_PREDICTIONS.csv', index=False)
    donor_qc.to_csv(output/'HELDOUT_TRAINING_SELECTION_QC.csv', index=False)
    summary.to_csv(output/'SUMMARY_MODEL_METHODS_BY_PANEL.csv', index=False)
    fitted.to_csv(output/'DESCRIPTIVE_FINAL_ALL17_FITTED_MODEL_COEFFICIENTS.csv', index=False)
    pd.DataFrame([{'regulator_GeneID':r, 'n_measured_PA_gene_targets':len(rmap[r]),
                   'target_loci_and_signs':'|'.join(f'{genes[i]}:{int(s):+d}' for i,s in sorted(rmap[r].items()))}
                  for r in regs]).to_csv(output/'MODEL_FIXED_REGULATOR_FEATURE_DEFINITIONS.csv', index=False)
    pd.DataFrame([{'path':str(p.resolve()),'sha256':checksum(p)} for p in paths]).to_csv(
        output/'INPUT_SHA256_PROVENANCE.csv',index=False)
    qc = {'status':'09G_MODEL_FITTED_WITHIN_STUDY_DONOR_HOLDOUT_NOT_EXTERNAL_VALIDATION',
          'donors':17,'genes':len(genes), 'curated_regulators':regs,
          'covered_targets':int(covered.sum()), 'opposing_pair_exclusions':n_conflict,
          'alpha_a_priori':ALPHA, 'fixed_placebo_maps':N_PLACEBO,
          'regulatory_edges_not_newly_inferred':True,
          'predictor_uses_untreated_only_library_counts':True,
          'interpretation':'exploratory within-study donor response prediction from untreated expression'}
    (output/'QC_09G.json').write_text(json.dumps(qc,indent=2)+'\n')
    (output/'SCIENTIFIC_TERMINAL_REPORT.txt').write_text(txt)
    (output/'STATUS.txt').write_text(qc['status']+'\n')
    print('\nFiles saved (without modifying previous audits):', output)


def selftest():
    rng=np.random.default_rng(111)
    n,g = 17,120
    genes=[f'PA{i:04d}' for i in range(1,g+1)]
    x=rng.normal(size=(n,g))*1.2 + 5
    # The reference prior defines two regulators with signed targets.
    regs=['PA9000','PA9001']
    rmap={regs[0]:{i:1. for i in range(20)},
          regs[1]:{i:-1. for i in range(25,45)}}
    links=[[] for _ in genes]
    for ri,r in enumerate(regs):
        for gi in rmap[r]: links[gi].append(ri)
    covered=np.array([bool(v) for v in links])
    zx=(x-x.mean(0))/np.maximum(x.std(0),0.2)
    m1=zx[:,:20].mean(1)
    m2=-zx[:,25:45].mean(1)
    y=rng.normal(scale=.25,size=(n,g))
    y[:,:20]+=2.0+2*m1[:,None]
    y[:,25:45]+=-2.0+2*m2[:,None]
    tests,pred,qc=test_model(genes,[str(i) for i in range(n)],x,y,regs,rmap,links,covered,
                             alpha=ALPHA,n_placebo=2)
    require(len(tests) == 17*4*4, 'incorrect cross-validation row count')
    require(np.isfinite(tests.loc[tests.n.gt(0),'mae']).all(), 'nonfinite metrics on populated synthetic panels')
    tbl=tests[tests.panel=='CURATED_TARGETS'].groupby('method').mae.mean()
    require(tbl.curated_regulatory_ridge < tbl.own_baseline_ridge,
            'synthetic regulator features failed to improve held-out MAE')
    require(tbl.curated_regulatory_ridge < tbl.shuffled_prior_ensemble,
            'synthetic correctly connected prior should beat wrong prior')
    params=final_fit(x,y,genes,regs,rmap,links,covered,ALPHA)
    require(len(params)==g and params.GeneID.is_unique, 'invalid final fit')
    print('SELF_TEST_PASSED: 17 donor LODO, TRAIN-ONLY feature scaling, no metal-outcome')
    print('  leakage, sign-typed regulator prior, leave-target-out scores, fixed placebo,')
    print('  alpha-regularised fitting and learned final coefficients.')
    print('Synthetic curated MAE:',round(float(tbl.curated_regulatory_ridge),4))
    print('Synthetic own-baseline MAE:',round(float(tbl.own_baseline_ridge),4))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--root',type=Path,default=ROOT)
    args=ap.parse_args()
    if args.self_test:
        selftest()
    else:
        run(args.root.resolve(), args.root.resolve()/OUTPUT)

if __name__=='__main__':
    main()
