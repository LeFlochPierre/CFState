#!/usr/bin/env python3
"""CF-State Phase 09c: train-16/test-1 donor reproducibility of measured metal gene responses.

Inputs are exactly the FROZEN gene-level responses from Audit 09b. In each fold,
select candidate genes and their effect directions using the OTHER 16 donors,
then score sign predictions in the untouched seventeenth donor. Outputs are
within-study, donor-held-out validation, NOT independent clinical validation.

No iModulon projection, no refitting of the reference model, and no inferred
transcription-factor/regulator edges. All outputs are non-overwriting.

Run:
  python src/phase9/audit_09c_LODO_metal_gene_response_generalisation.py --self-test
  python src/phase9/audit_09c_LODO_metal_gene_response_generalisation.py
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

ROOT = Path(__file__).resolve().parents[2]
B = Path('results/phase9/audit_09b_Neff2024_paired_gene_response_freeze')
OUT = Path('results/phase9/audit_09c_LODO_metal_gene_response_generalisation')
MASK = Path('results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv')
EXPECTED_B = 'PHASE09B_MEASURED_GENE_RESPONSE_FREEZE_COMPLETE_NOT_NETWORK_TRAINED'

SPEC = [
    ('PRIMARY_13of16_absmeanGE1', 13, 1.0),
    ('SENSITIVITY_12of16_absmeanGE0p5', 12, 0.5),
    ('SENSITIVITY_14of16_absmeanGE1', 14, 1.0),
]


def check(condition, msg):
    if not condition:
        raise RuntimeError('AUDIT_09C_CHECK_FAILED: ' + msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load(root):
    base = root / B
    inputs = {
        '09b_status': base / 'AUDIT_09B_STATUS.txt',
        '09b_effects': base / 'FROZEN_17_donors_x_4923_genes_paired_effects.csv',
        '09b_gene_summary': base / 'FROZEN_4923_gene_combined_metals_response_summary.csv',
        '09b_pairing': base / 'FROZEN_17_donor_pairing_and_sizefactors.csv',
        '09b_provenance': base / 'INPUT_SHA256_provenance.csv',
        'frozen_mask': root / MASK,
    }
    for tag, path in inputs.items():
        check(path.is_file(), f'missing {tag}: {path}')
    check(inputs['09b_status'].read_text().strip() == EXPECTED_B,
          '09b run not recorded as successfully completed')

    m = pd.read_csv(inputs['frozen_mask'], dtype=str)
    check('GeneID' in m and len(m) == 4923 and m['GeneID'].is_unique,
          'frozen mask must contain 4923 unique genes')
    genes = m['GeneID'].astype(str).tolist()
    check(all(re.fullmatch(r'PA\d{4}', gid) for gid in genes), 'bad mask IDs')
    z = pd.read_csv(inputs['09b_effects'])
    summary = pd.read_csv(inputs['09b_gene_summary'])
    check(z.shape == (4923, 18), f'wrong gene effects shape {z.shape}')
    check(len(summary) == 4923, 'wrong summary row count')
    check(z.columns[0] == 'GeneID' and summary.columns[0] == 'GeneID',
          'unrecognised gene ID column')
    check(z.GeneID.tolist() == genes and summary.GeneID.tolist() == genes,
          '09b outputs not in original frozen gene order')
    columns = list(z.columns[1:])
    ids = []
    for col in columns:
        match = re.fullmatch(r'donor_(.+)_METALS_minus_UNTREATED', str(col))
        check(match is not None, f'unknown donor effect column: {col}')
        ids.append(match.group(1))
    check(len(set(ids)) == 17 and '105' not in ids,
          '17 independent donor labels expected, excluding donor 105')
    pairing = pd.read_csv(inputs['09b_pairing'], dtype={'donor_id': str})
    check(len(pairing) == 17 and 'donor_id' in pairing and
          set(ids) == set(pairing['donor_id'].str.strip()) and pairing['donor_id'].is_unique,
          'donor IDs do not agree with frozen 09b pairing')
    x = z[columns].to_numpy(dtype=float).T  # donor x gene
    check(x.shape == (17, 4923) and np.isfinite(x).all(), 'nonfinite 09b effects')
    need_cols = ['mean_paired_log2_normalized_count_plus1_change',
                 'n_donors_increased', 'n_donors_decreased',
                 'ci_excludes_zero_descriptive',
                 'median_untreated_frozen_sizefactor_normalized_count']
    check(set(need_cols).issubset(summary.columns), '09b summary columns differ from expected')
    mean = x.mean(axis=0)
    check(np.allclose(mean, summary[need_cols[0]].to_numpy(float), atol=2e-8),
          '09b gene mean response does not reproduce original 17 effect columns')
    check(np.array_equal((x > 0).sum(axis=0), summary['n_donors_increased'].to_numpy(int)) and
          np.array_equal((x < 0).sum(axis=0), summary['n_donors_decreased'].to_numpy(int)),
          '09b gene sign counts do not reproduce donor effect matrix')
    recorded = pd.read_csv(inputs['09b_provenance'])
    check(set(['role', 'path', 'sha256']).issubset(recorded.columns),
          '09b SHA provenance missing expected columns')
    for row in recorded.itertuples(index=False):
        p = Path(str(row.path))
        if not p.is_absolute():
            p = root / p
        check(p.is_file() and sha256(p) == str(row.sha256),
              f'09b upstream input changed since 09b: {row.role}')
    return genes, ids, x, summary, inputs


def selected_from_training(train, votes, min_abs_mean):
    train_mean = train.mean(axis=0)
    nup = (train > 0).sum(axis=0)
    ndown = (train < 0).sum(axis=0)
    selection = ((np.maximum(nup, ndown) >= votes) &
                 (np.abs(train_mean) >= min_abs_mean))
    return selection, np.sign(train_mean), train_mean


def cosine(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else float('nan')


def evaluate(genes, donors, effects, summary, n_shuffles=500, seed=20261015,
             spec=SPEC):
    """Evaluate each held-out donor; no gene selection uses the held-out vector."""
    check(len(donors) == 17 and effects.shape == (17, 4923), 'expected 17 x 4923 effects')
    check(n_shuffles >= 20, 'at least 20 gene-shuffle control draws needed')
    rng = np.random.default_rng(seed)
    results = []
    freq = np.zeros(len(genes), dtype=int)
    full_gene_mean = effects.mean(axis=0)
    for i, donor in enumerate(donors):
        train = np.delete(effects, i, axis=0)
        heldout = effects[i]
        check(train.shape[0] == 16, 'training fold must contain 16 donors')
        all_mean = train.mean(axis=0)
        all_signs = np.sign(all_mean)
        fraction_genome = float(np.mean((all_signs == np.sign(heldout)) & (all_signs != 0)))
        for label, votes, threshold in spec:
            take, predicted, train_mean = selected_from_training(train, votes, threshold)
            ntake = int(take.sum())
            check(ntake > 0, f'no selected genes in donor {donor}, rule {label}')
            correct = (predicted[take] == np.sign(heldout[take]))
            observed = float(correct.mean())
            if label.startswith('PRIMARY'):
                freq += take.astype(int)
            # Permute held-out gene identities, NOT donor IDs, to preserve its
            # global response/sign composition. This is a descriptive diagnostic;
            # correlated genes make these shuffles non-independent.
            null = np.empty(n_shuffles, dtype=float)
            selected_signs = predicted[take]
            for k in range(n_shuffles):
                randomized_heldout_sign = np.sign(heldout[rng.permutation(len(genes))])[take]
                null[k] = np.mean(selected_signs == randomized_heldout_sign)
            selected_prediction = train_mean[take]
            selected_actual = heldout[take]
            record = {
                'donor_id': donor,
                'rule': label,
                'training_donors': 16,
                'heldout_donors': 1,
                'minimum_training_sign_votes': votes,
                'minimum_abs_training_mean_log2effect': threshold,
                'selected_genes_train16_only': ntake,
                'heldout_genes_zero_effect': int((selected_actual == 0).sum()),
                'heldout_genes_correct_sign': int(correct.sum()),
                'heldout_fraction_sign_correct': observed,
                'all4923_genes_fraction_sign_correct': fraction_genome,
                'heldout_selected_cosine': cosine(selected_prediction, selected_actual),
                'heldout_selected_MAE_log2effect': float(np.mean(np.abs(selected_prediction - selected_actual))),
                'shuffled_gene_label_null_median': float(np.median(null)),
                'shuffled_gene_label_null_q025': float(np.quantile(null, 0.025)),
                'shuffled_gene_label_null_q975': float(np.quantile(null, 0.975)),
                'observed_minus_shuffled_gene_null_median': observed - float(np.median(null)),
                'shuffled_gene_one_sided_empirical_p_DESCRIPTIVE': float((1+np.count_nonzero(null >= observed)) / (1 + n_shuffles)),
                'null_draws': n_shuffles,
                'selection_uses_heldout_donor': False,
                'experiment': 'PRJNA1006084_PA01_spikein_17_CF_sputum_donors_ZnFeMn',
            }
            results.append(record)
    fold = pd.DataFrame(results)
    gene_full_select = (np.maximum((effects > 0).sum(axis=0),
                                   (effects < 0).sum(axis=0)) >= 14) & (np.abs(full_gene_mean) >= 1)
    edges = pd.DataFrame({
        'GeneID': genes,
        'full17_mean_effect_log2norm_countplus1': full_gene_mean,
        'full17_mean_effect_sign': np.sign(full_gene_mean).astype(int),
        'full17_n_increased': (effects > 0).sum(axis=0),
        'full17_n_decreased': (effects < 0).sum(axis=0),
        'full17_n_matching_strongest_sign': np.maximum((effects > 0).sum(axis=0),
                                                        (effects < 0).sum(axis=0)),
        'full17_descriptive_bootstrap_CI_excludes_zero': summary['ci_excludes_zero_descriptive'].astype(bool).to_numpy(),
        'median_untreated_frozen_sizefactor_normalized_count': summary['median_untreated_frozen_sizefactor_normalized_count'],
        'full17_descriptive_14of17_and_absmeanGE1': gene_full_select,
        'n_LODO_training16_folds_selected_PRIMARY': freq,
        'n_folds_possible': 17,
        'edge_type': 'combined_Zn_Fe_Mn_perturbation_TO_gene_expression_response',
        'direct_regulatory_edge': False,
        'CF_specific_edge': False,
    })
    macro = []
    for rule, g in fold.groupby('rule', sort=False):
        donor_rng = np.random.default_rng(seed + sum(ord(c) for c in rule))
        # Uncertainty is across held-out *donors*, not independent genes.
        draws = donor_rng.integers(0, len(g), size=(5000, len(g)))
        acc = g.heldout_fraction_sign_correct.to_numpy(float)
        above_null = g.observed_minus_shuffled_gene_null_median.to_numpy(float)
        bs_acc = acc[draws].mean(axis=1)
        bs_diff = above_null[draws].mean(axis=1)
        macro.append({
            'rule': rule,
            'n_heldout_donors': len(g),
            'n_selected_train16_min': int(g.selected_genes_train16_only.min()),
            'n_selected_train16_median': float(g.selected_genes_train16_only.median()),
            'n_selected_train16_max': int(g.selected_genes_train16_only.max()),
            'median_heldout_sign_accuracy': float(g.heldout_fraction_sign_correct.median()),
            'mean_heldout_sign_accuracy': float(g.heldout_fraction_sign_correct.mean()),
            'donor_bootstrap_mean_accuracy_q025': float(np.quantile(bs_acc, .025)),
            'donor_bootstrap_mean_accuracy_q975': float(np.quantile(bs_acc, .975)),
            'min_heldout_sign_accuracy': float(g.heldout_fraction_sign_correct.min()),
            'max_heldout_sign_accuracy': float(g.heldout_fraction_sign_correct.max()),
            'mean_gene_shuffled_baseline': float(g.shuffled_gene_label_null_median.mean()),
            'mean_observed_minus_gene_shuffled': float(g.observed_minus_shuffled_gene_null_median.mean()),
            'donor_bootstrap_mean_minus_null_q025': float(np.quantile(bs_diff, .025)),
            'donor_bootstrap_mean_minus_null_q975': float(np.quantile(bs_diff, .975)),
            'n_donors_observed_above_shuffled_975': int((g.heldout_fraction_sign_correct > g.shuffled_gene_label_null_q975).sum()),
            'mean_selected_effect_cosine': float(g.heldout_selected_cosine.mean()),
        })
    return fold, pd.DataFrame(macro), edges


def output_report(summary, folds, edges, n_shuffles):
    a = summary.loc[summary.rule == 'PRIMARY_13of16_absmeanGE1'].iloc[0]
    worst = folds[folds.rule == 'PRIMARY_13of16_absmeanGE1'].sort_values('heldout_fraction_sign_correct').iloc[0]
    return f'''# CF-State Phase 09c: held-out donor response reproducibility

Status: **descriptive within-study leave-one-donor-out complete**. No clinical
CF specificity, direct regulator-gene interaction or independent clinical
transfer was established.

## Fixed training-only rule for this audit

For each held-out donor: use the OTHER 16 donors to select genes supported by
at least 13/16 effect signs and |mean training log2(normalised count + 1)
response| >= 1.0. Direction is sign of TRAINING mean. The held-out donor
is used **only** for evaluation. This threshold is a provisional rule
chosen before inspecting these held-out gene-selection results, not a
statistical discovery threshold or an independently preregistered rule.

The data-dependent 1,080 gene overlap observed in Audit 09b is *not* used as
an eligible feature list for the held-out validation (would leak held-out data).
The 09b full17 median baseline expression count is recorded descriptively but
not used for train16 selection. No expression-count threshold is claimed as
cross-validated; if required, recalculate it inside training folds.

- Held-out donors: {int(a.n_heldout_donors)}
- Genes selected per 16-donor training fold: median {a.n_selected_train16_median:.0f},
  range {int(a.n_selected_train16_min)}–{int(a.n_selected_train16_max)}
- Mean held-out gene-direction prediction accuracy: {a.mean_heldout_sign_accuracy:.4f}
  (descriptive donor-bootstrap interval: {a.donor_bootstrap_mean_accuracy_q025:.4f}–{a.donor_bootstrap_mean_accuracy_q975:.4f})
- Median held-out gene-direction prediction accuracy: {a.median_heldout_sign_accuracy:.4f}
- Mean shuffled held-out gene-label accuracy: {a.mean_gene_shuffled_baseline:.4f}
- Mean observed minus shuffled median accuracy: {a.mean_observed_minus_gene_shuffled:.4f}
  (descriptive donor-bootstrap interval: {a.donor_bootstrap_mean_minus_null_q025:.4f}–{a.donor_bootstrap_mean_minus_null_q975:.4f})
- Individual donors exceeding the gene-label shuffle upper 97.5% bound:
  {int(a.n_donors_observed_above_shuffled_975)}/{int(a.n_heldout_donors)}
- Weakest donor by held-out sign accuracy: {worst.donor_id} ({worst.heldout_fraction_sign_correct:.4f})
- Genes meeting FULL17 descriptive 14/17 and |mean| >=1: {int(edges.full17_descriptive_14of17_and_absmeanGE1.sum())}
- Null label shuffles/donor/rule: {n_shuffles}

## Interpretation and restrictions

1. **This is within-study donor holdout, not independent intervention validation.**
   All seventeen donors share one ex-vivo study, laboratory PAO1 strain,
   processed count pipeline and Zn+Fe+Mn combined perturbation.
2. 09b effects are log2(estimated Salmon count / frozen size factor +1)
   differences; they are not differential-expression shrinkage estimates.
3. Baseline is a permutation of held-out *gene labels*. It preserves overall
   signed effect composition but disrupts gene identities and correlated
   pathways. The empirical shuffle p-values are **diagnostic only**, not
   donor-independent biological p-values or gene-level FDR tests.
4. The full17 candidate list is descriptive and cannot be re-used as a
   preselected validation target for these same 17 donors.
5. A selected perturbation-to-gene effect is an experimentally observed
   expression-response link, **not** direct transcriptional regulation.
6. One intervention adds **three metals simultaneously**. This data cannot
   disentangle Zn, Fe and Mn. Organism studied is laboratory PAO1 spiked into
   CF sputum, not endogenous clinical bacterial transcriptomes.
7. Donor IDs are independent units, but there is one pair per donor, no
   separate biological repeat per intervention level within that donor.
8. Technical normalisation, correlated operons, gene-expression levels and
   study-specific effects may cause strong agreement. Independent experiments
   and clinical validation remain necessary before calling this CF-tailored.
9. The 09c edge inventory is intentionally gene-response evidence only.
   To build a TF regulatory network later, use separate curated evidence
   and clearly distinguish binding, signed regulation and indirect responses.

## Decision gate

If held-out sign prediction is clearly above shuffled-gene baseline across
most donors, the metal response is reproducible enough **within this study**
to form a provisional response layer. If not, revise the feature-selection
rule using training data only and record the limitation. Do not add extra
network complexity until an independent perturbation holdout is identified.
'''


def run(root, out, n_shuffles=500, seed=20261015):
    check(not out.exists(), 'non-overwrite: output directory already exists: ' + str(out))
    genes, donors, x, original, paths = load(root)
    folds, macro, edges = evaluate(genes, donors, x, original,
                                   n_shuffles=n_shuffles, seed=seed)
    pri = macro[macro.rule == 'PRIMARY_13of16_absmeanGE1'].iloc[0]
    report = output_report(macro, folds, edges, n_shuffles)
    provenance = pd.DataFrame([
        {'role': role, 'path': str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
         'sha256': sha256(path)} for role, path in paths.items()
    ])
    out.mkdir(parents=True, exist_ok=False)
    folds.to_csv(out/'DESCRIPTIVE_17_donor_train16_heldout1_gene_predictions.csv', index=False)
    macro.to_csv(out/'DESCRIPTIVE_LODO_rule_level_summary.csv', index=False)
    edges.to_csv(out/'DESCRIPTIVE_full17_candidate_perturbation_gene_edges_NOT_REGULATORY.csv', index=False)
    provenance.to_csv(out/'INPUT_SHA256_provenance.csv', index=False)
    (out/'SUMMARY_FINDINGS.md').write_text(report)
    (out/'AUDIT_09C_STATUS.txt').write_text('PHASE09C_HELDOUT_DONOR_GENE_RESPONSE_CHECK_COMPLETE_NOT_NETWORK_TRAINED\n')
    meta = {'n_donors':17, 'n_genes':4923, 'design':'train16/test1 leave-one-donor-out',
            'rules':SPEC, 'null_shuffles':n_shuffles, 'seed':seed, 'full17_bootstrap_overlap_not_used_in_selection':True}
    (out/'QC_DESIGN.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('AUDIT 09C completed:', out)
    print('Primary train16 genes selected per fold (min/median/max):',
          int(pri.n_selected_train16_min), int(pri.n_selected_train16_median), int(pri.n_selected_train16_max))
    print('Primary mean held-out gene-direction accuracy:', f'{pri.mean_heldout_sign_accuracy:.4f}')
    print('Mean gene-shuffled null accuracy:', f'{pri.mean_gene_shuffled_baseline:.4f}')
    print('Held-out donors above shuffled 97.5% bound:',
          f'{int(pri.n_donors_observed_above_shuffled_975)}/17')
    print('Full17 provisional response edges (descriptive only):',
          int(edges.full17_descriptive_14of17_and_absmeanGE1.sum()))
    print('No CF-specificity, independent external replication or regulatory edges inferred.')


def self_test():
    # Tests the *real end-to-end file reader and evaluator*, with deterministic
    # synthetic 4923-gene responses and deliberately weaker donors.
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        d = root/B
        d.mkdir(parents=True)
        (root/MASK).parent.mkdir(parents=True,exist_ok=True)
        genes = [f'PA{k:04d}' for k in range(1,4924)]
        pd.DataFrame({'GeneID':genes}).to_csv(root/MASK,index=False)
        donors = [str(k) for k in range(101,119) if k != 105]
        rng = np.random.default_rng(3)
        fx = rng.normal(0, 0.22, (17,4923))
        fx[:,:150] = 2.2 + rng.normal(0,.25,(17,150))
        fx[:,150:300] = -2.1 + rng.normal(0,.25,(17,150))
        fx[0,:150] *= -1  # one weaker/outlier donor: no refitting
        cols = {'GeneID':genes}
        for i, donor in enumerate(donors):
            cols[f'donor_{donor}_METALS_minus_UNTREATED'] = fx[i]
        pd.DataFrame(cols).to_csv(d/'FROZEN_17_donors_x_4923_genes_paired_effects.csv',index=False)
        summ = pd.DataFrame({
            'GeneID':genes,
            'mean_paired_log2_normalized_count_plus1_change':fx.mean(axis=0),
            'n_donors_increased':(fx > 0).sum(axis=0),
            'n_donors_decreased':(fx < 0).sum(axis=0),
            'ci_excludes_zero_descriptive':np.abs(fx.mean(axis=0))>1,
            'median_untreated_frozen_sizefactor_normalized_count':np.full(4923,100.),
        })
        summ.to_csv(d/'FROZEN_4923_gene_combined_metals_response_summary.csv',index=False)
        pd.DataFrame({'donor_id':donors}).to_csv(d/'FROZEN_17_donor_pairing_and_sizefactors.csv',index=False)
        (d/'AUDIT_09B_STATUS.txt').write_text(EXPECTED_B+'\n')
        pd.DataFrame([{'role':'mask', 'path':str(MASK), 'sha256':sha256(root/MASK)}]).to_csv(
            d/'INPUT_SHA256_provenance.csv',index=False)
        run(root, root/OUT, n_shuffles=40, seed=4)
        t = pd.read_csv(root/OUT/'DESCRIPTIVE_17_donor_train16_heldout1_gene_predictions.csv')
        p = t[t.rule=='PRIMARY_13of16_absmeanGE1']
        check(len(p)==17 and p.selected_genes_train16_only.min()>=150,
              'unexpected synthetic fold recovery')
        check(p.heldout_fraction_sign_correct.mean() > p.shuffled_gene_label_null_median.mean()+0.2,
              'synthetic donor validation did not exceed gene-shuffle baseline')
        check(not p['selection_uses_heldout_donor'].astype(bool).any(), 'holdout leakage flag')
        try:
            run(root, root/OUT, n_shuffles=20)
        except RuntimeError as err:
            check('non-overwrite' in str(err), 'repeat run wrong failure')
        else:
            raise AssertionError('failed to reject overwritten output')
        # Tamper with 09b gene-level effects: means must not silently go stale.
        zz = pd.read_csv(d/'FROZEN_17_donors_x_4923_genes_paired_effects.csv')
        zz.iloc[0,1] = 1000.
        zz.to_csv(d/'FROZEN_17_donors_x_4923_genes_paired_effects.csv',index=False)
        try:
            load(root)
        except RuntimeError as err:
            check('gene mean response' in str(err), 'failed for unexpected reason')
        else:
            raise AssertionError('tampered 09b effects must be rejected')
    print('SELF_TEST_PASSED: 17 donors x 4923 genes, train16-only selection,')
    print('  predictive donor holdout, gene-label-shuffled control, SHA, nonoverwrite, tamper check')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--null-shuffles', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261015)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        check(20<=args.null_shuffles<=20000,'null shuffles must be 20..20000')
        root = args.root.resolve()
        run(root, root/OUT, n_shuffles=args.null_shuffles, seed=args.seed)

if __name__=='__main__':
    main()
