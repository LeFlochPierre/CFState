#!/usr/bin/env python3
"""CF-State Audit 09d: construct a provenance-typed response evidence graph v0.

FROZEN inputs are Audits 09a and 09c, and the 4923 PAO1 gene mask.
- Build candidate metal perturbation->gene expression response edges selected from
  all 17 donors *after* held-out donor model evaluation is finished.
- Keep 4 x 4920 oxygen/Anr microarray changes as separate UNQUALIFIED evidence.
- Compare measured signs across contexts without pooling cross-platform effects.
- No regulatory interaction, causal mechanism, CF specificity, network learning,
  or external validation is inferred.
- Fail closed on unexpected input schemas; never overwrite existing outputs.

Run: python src/phase9/audit_09d_CFstate_evidence_network_v0.py --self-test
     python src/phase9/audit_09d_CFstate_evidence_network_v0.py
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
MASK = Path('results/phase4/audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_softcore_gene_order_4923.csv')
A = Path('results/phase9/audit_09a_perturbation_evidence_preflight')
C = Path('results/phase9/audit_09c_LODO_metal_gene_response_generalisation')
OUT = Path('results/phase9/audit_09d_CFstate_evidence_network_v0')
EXPECTED_A = 'PHASE09A_EVIDENCE_PREFLIGHT_COMPLETE_NOT_MODEL_FITTED'
EXPECTED_C = 'PHASE09C_HELDOUT_DONOR_GENE_RESPONSE_CHECK_COMPLETE_NOT_NETWORK_TRAINED'
O2 = ['O2_2_minus_20', 'O2_0p4_minus_20', 'anoxic_nitrate_minus_20']
ANR = 'delta_anr_minus_WT_CFBE'
CONTRASTS = O2 + [ANR]
CONTEXT = {
    'O2_2_minus_20': '2% O2 minus 20% O2; PAO1; GPL84 array',
    'O2_0p4_minus_20': '0.4% O2 minus 20% O2; PAO1; GPL84 array',
    'anoxic_nitrate_minus_20': 'anoxia + nitrate minus 20% O2; PAO1; GPL84 array; nitrate confounded',
    'delta_anr_minus_WT_CFBE': 'delta-anr minus WT on CFBE; PAO1; GPL84 array; knockout indirect effects',
}

def need(ok, msg):
    if not ok:
        raise RuntimeError('AUDIT_09D_FAILED: ' + str(msg))

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def input_paths(root):
    return {
        'frozen_gene_mask': root / MASK,
        '09a_status': root / A / 'AUDIT_09A_STATUS.txt',
        '09a_native_arrays': root / A / 'FROZEN_08m_native_array_gene_response_evidence.csv',
        '09c_status': root / C / 'AUDIT_09C_STATUS.txt',
        '09c_candidate_edges': root / C / 'DESCRIPTIVE_full17_candidate_perturbation_gene_edges_NOT_REGULATORY.csv',
        '09c_donor_evaluation': root / C / 'DESCRIPTIVE_17_donor_train16_heldout1_gene_predictions.csv',
        '09c_summary': root / C / 'DESCRIPTIVE_LODO_rule_level_summary.csv',
    }

def load(root):
    paths = input_paths(root)
    for name, path in paths.items():
        need(path.is_file(), f'missing {name}: {path}')
    need(paths['09a_status'].read_text().strip() == EXPECTED_A, 'wrong 09a completion status')
    need(paths['09c_status'].read_text().strip() == EXPECTED_C, 'wrong 09c completion status')
    mask = pd.read_csv(paths['frozen_gene_mask'], dtype=str)
    need('GeneID' in mask.columns and len(mask) == 4923 and mask.GeneID.is_unique, 'frozen mask differs')
    genes = mask.GeneID.tolist()
    need(all(re.fullmatch(r'PA\d{4}', g) for g in genes), 'unexpected GeneID in mask')

    metal = pd.read_csv(paths['09c_candidate_edges'], dtype={'GeneID': str})
    expected_m = ['GeneID', 'full17_mean_effect_log2norm_countplus1', 'full17_mean_effect_sign',
                  'full17_n_increased', 'full17_n_decreased', 'full17_n_matching_strongest_sign',
                  'full17_descriptive_14of17_and_absmeanGE1', 'n_LODO_training16_folds_selected_PRIMARY',
                  'direct_regulatory_edge', 'CF_specific_edge']
    need(set(expected_m).issubset(metal.columns), '09c candidate-edge schema changed')
    need(metal.GeneID.tolist() == genes, '09c edges not aligned to mask')
    for name in ['direct_regulatory_edge', 'CF_specific_edge']:
        need((metal[name].astype(str).str.lower() == 'false').all(), f'09c flag {name} unexpected')
    wanted = metal['full17_descriptive_14of17_and_absmeanGE1'].astype(str).str.lower() == 'true'
    sel = metal.loc[wanted].copy()
    need(len(sel) > 0 and len(sel) < 4923, 'unexpected no/all selected metal genes')
    need((sel.full17_n_matching_strongest_sign >= 14).all() and
         (sel.full17_mean_effect_log2norm_countplus1.abs() >= 1 - 1e-8).all(),
         '09c 14/17 and absmean >=1 frozen selection not reproduced')
    need(sel.full17_mean_effect_log2norm_countplus1.notna().all(), 'missing metal response')
    need(np.array_equal(np.sign(sel.full17_mean_effect_log2norm_countplus1).astype(int),
                        sel.full17_mean_effect_sign.astype(int)), 'metal signed effects inconsistent')
    need(sel.n_LODO_training16_folds_selected_PRIMARY.between(0, 17).all(),
         'invalid gene fold-selection frequencies')

    eva = pd.read_csv(paths['09c_donor_evaluation'], dtype={'donor_id': str})
    pri = eva.loc[eva.rule == 'PRIMARY_13of16_absmeanGE1'].copy()
    need(len(pri) == 17 and pri.donor_id.is_unique and
         not pri.selection_uses_heldout_donor.astype(bool).any(), '09c donor holdout verification failed')
    mac = pd.read_csv(paths['09c_summary'])
    mpri = mac.loc[mac.rule == 'PRIMARY_13of16_absmeanGE1']
    need(len(mpri) == 1 and np.isclose(pri.heldout_fraction_sign_correct.mean(),
                                      mpri.iloc[0].mean_heldout_sign_accuracy, atol=1e-8),
         '09c donor summary does not agree with primary fold evaluation')

    arr = pd.read_csv(paths['09a_native_arrays'], dtype={'GeneID': str})
    expected_a = ['dataset_id', 'contrast_id', 'GeneID', 'effect_value_native', 'effect_unit',
                  'effect_direction', 'n_biological_treatment', 'n_biological_reference',
                  'evidence_type', 'direct_regulatory_edge_proven']
    need(set(expected_a).issubset(arr.columns), '09a array schema changed')
    need(len(arr) == 19680 and not arr.duplicated(['contrast_id','GeneID']).any(), '09a 4x4920 genes missing/duplicate')
    need(set(arr.contrast_id) == set(CONTRASTS), 'unexpected contrasts in 09a')
    need(arr.GeneID.isin(genes).all(), '09a gene outside mask')
    need((arr.direct_regulatory_edge_proven.astype(str).str.lower() == 'false').all(),
         '09a native array direct regulatory flag unexpected')
    need(np.isfinite(arr.effect_value_native.to_numpy(float)).all(), '09a non-finite native effects')
    for key in CONTRASTS:
        frame = arr.loc[arr.contrast_id == key]
        need(len(frame) == 4920 and frame.GeneID.is_unique, '09a array genes differ for ' + key)
        expect = np.where(frame.effect_value_native.to_numpy(float) > 0, 'UP',
                          np.where(frame.effect_value_native.to_numpy(float) < 0, 'DOWN','ZERO'))
        need(np.array_equal(expect, frame.effect_direction.to_numpy(str)),
             '09a direction inconsistent for ' + key)
        ds = 'GSE67006' if key == ANR else 'GSE6741'
        need((frame.dataset_id == ds).all(), 'wrong array study for ' + key)
    return sel, arr, paths, pri

def prepare(sel, arr):
    # A genuine, one-perturbation gene-response graph at this stage.
    # Array effects are retained separately: neither top-40 markers nor array magnitude
    # cutoffs were independently validated for network inclusion.
    edges = pd.DataFrame({
        'source_node': 'intervention:PRJNA1006084_ZnFeMn_METALS_minus_UNTREATED',
        'target_node': 'gene:' + sel.GeneID,
        'source_type': 'combined_environmental_perturbation',
        'target_type': 'PAO1_locus',
        'edge_type': 'experimentally_measured_indirect_expression_response',
        'effect_direction': np.where(sel.full17_mean_effect_sign > 0, 'UP', 'DOWN'),
        'within_study_mean_effect': sel.full17_mean_effect_log2norm_countplus1.astype(float),
        'effect_unit': 'paired delta log2(Salmon_estimated_count/frozen_size_factor+1)',
        'n_independent_donors': 17,
        'n_donors_agreeing_with_majority_sign': sel.full17_n_matching_strongest_sign.astype(int),
        'n_LODO_train16_folds_selected': sel.n_LODO_training16_folds_selected_PRIMARY.astype(int),
        'selection_provenance': 'full17_descriptive_14of17_and_absmeanGE1; NOT heldout validation feature list',
        'evidence_tier': 'PROVISIONAL_WITHIN_STUDY_REPRODUCIBLE_RESPONSE',
        'study_id': 'PRJNA1006084',
        'direct_TF_regulation': False,
        'CF_specificity_demonstrated': False,
        'independently_replicated': False,
    })
    nodes = [{'node_id': edges.source_node.iloc[0], 'node_type': 'intervention',
              'label': 'Zn+Fe+Mn (combined) added to CF donor sputum; PAO1 spike-in'}]
    nodes += [{'node_id': 'gene:' + g, 'node_type': 'PAO1_locus', 'label': g}
              for g in sel.GeneID.tolist()]
    nodes = pd.DataFrame(nodes)
    check_unique = edges[['source_node','target_node']].duplicated().any()
    need(not check_unique and nodes.node_id.is_unique, 'duplicate response edges or nodes')
    # Measured array effects are a separate ledger, NOT network edges.
    array = arr.copy()
    array['evidence_tier'] = 'NATIVE_EXPERIMENTAL_RESPONSE_UNQUALIFIED_FOR_NETWORK_EDGE'
    array['not_in_provisional_graph'] = True
    array['context_warning'] = array.contrast_id.map(CONTEXT)
    # Only compare signs across *overlapping gene identifiers*, not effect units.
    aa = array.pivot(index='GeneID', columns='contrast_id', values='effect_value_native')
    cc = sel.set_index('GeneID').loc[:, ['full17_mean_effect_log2norm_countplus1',
                                        'full17_n_matching_strongest_sign',
                                        'n_LODO_training16_folds_selected_PRIMARY']]
    overlap = cc.join(aa, how='left')
    for key in CONTRASTS:
        overlap[key + '_matches_metal_effect_sign'] = np.where(
            overlap[key].notna(), np.sign(overlap[key]) == np.sign(overlap.full17_mean_effect_log2norm_countplus1),
            np.nan)
    diag = []
    for key in CONTRASTS:
        pair = overlap[['full17_mean_effect_log2norm_countplus1', key]].dropna()
        n = len(pair)
        both = (np.sign(pair.iloc[:,0]) != 0) & (np.sign(pair.iloc[:,1]) != 0)
        n_nonzero = int(both.sum())
        frac = float(np.mean(np.sign(pair.iloc[:,0].to_numpy()[both]) ==
                             np.sign(pair.iloc[:,1].to_numpy()[both]))) if n_nonzero else np.nan
        diag.append({'comparison_to_combined_metals': key,
                     'n_shared_selected_metal_genes': n,
                     'n_nonzero_both': n_nonzero,
                     'fraction_same_effect_sign_descriptive': frac,
                     'platforms': 'Salmon estimated RNA-seq counts versus GPL84 microarray',
                     'inference': 'NONINDEPENDENT_POSTSELECTION_DESCRIPTIVE; not cross-platform effect calibration',
                     'study_context': CONTEXT[key]})
    return nodes, edges, array, overlap.reset_index(), pd.DataFrame(diag)

def narrative(nedges, summary):
    lines = [
        '# CF-State Phase 09d: response evidence network v0', '',
        '**Status:** first provenance-typed response graph assembled; no network model trained.', '',
        f'- Qualified, provisional combined-metal intervention-to-gene response edges: **{nedges}**.',
        '- 17 CF sputum donors; one combined Zn+Fe+Mn addition; laboratory PAO1 spike-in.',
        '- Edges use full17 14/17 directional agreement and |mean response| >=1.0, AFTER earlier LODO evaluation.',
        '- The same 17 donors MUST NOT be used to claim an independent validation of this final full17 edge list.',
        '- Four additional frozen GPL84 array contrasts, 4,920 genes each, are in a *separate unqualified evidence ledger*.',
        '- No array contrast has been promoted into a qualified network edge here.',
        '- No ChIP/RegulomePA regulatory edges have been loaded: binding, causal regulation and indirect response remain distinct.',
        '', '## Cross-context sign comparisons (descriptive only)', '',
        'These assess sign overlap for genes selected from the metal study. The selection conditions on metal effects;',
        'no permutation p-values, formal validation, platform-scale equivalence, or causal interpretation is claimed.', ''
    ]
    for r in summary.itertuples(index=False):
        lines.append(f'- {r.comparison_to_combined_metals}: {r.n_shared_selected_metal_genes} shared metal genes; '
                     f'{r.fraction_same_effect_sign_descriptive:.3f} signed agreement among nonzero pairs.')
    lines += ['', '## Critical next steps', '',
              '1. Obtain and freeze an **externally curated** regulator-target network with evidence type, PubMed source, sign when supported, and exact PAO1 locus IDs. Keep binding-only edges separate from functional expression-response edges.',
              '2. Predeclare and implement a study-held-out biological validation task; a leave-one-donor-out result from the same ex-vivo study is not independent intervention validation.',
              '3. Build a transparent gene-signature baseline and original iModulon baseline before adding learned graph parameters.',
              '4. Evaluate clinical transfer separately with covariate and matched control limitations; do not call an association CF-specific without matched non-CF airway controls.',
              '', '## Explicit limitations', '',
              '- One central intervention plus gene nodes forms a response **graph component**, not a reconstructed regulatory network.',
              '- Gene-expression response is not proof that a TF physically binds, activates, or represses the target.',
              '- The 0% oxygen experiment also uses 100 mM nitrate, and Δanr has indirect effects.',
              '- GPL84 log-scale changes and log2 size-factor-normalised Salmon counts are not comparable in magnitude.',
              '- No individual metal attribution, patient phenotype estimation, infection status classification, or causal oxygen concentration is established.',
              '']
    return '\n'.join(lines)

def run(root, out):
    need(not out.exists(), 'non-overwrite: directory already exists: ' + str(out))
    sel, arr, paths, pri = load(root)
    nodes, edges, array, overlay, diag = prepare(sel, arr)
    out.mkdir(parents=True, exist_ok=False)
    nodes.to_csv(out/'NETWORK_V0_PROVISIONAL_METAL_NODES.csv', index=False)
    edges.to_csv(out/'NETWORK_V0_PROVISIONAL_METAL_RESPONSE_EDGES_NOT_REGULATORY.csv', index=False)
    array.to_csv(out/'FROZEN_19680_UNQUALIFIED_GPL84_GENE_RESPONSES_NOT_NETWORK_EDGES.csv', index=False)
    overlay.to_csv(out/'DESCRIPTIVE_selected_metal_genes_cross_context_08m_sign_overlay.csv', index=False)
    diag.to_csv(out/'DESCRIPTIVE_selected_metal_gene_sign_agreement_with_arrays.csv', index=False)
    (out/'NETWORK_SPECIFICATION_AND_DECISION_GATES.md').write_text(narrative(len(edges),diag), encoding='utf-8')
    pd.DataFrame([{'role': name, 'file': str(path.relative_to(root)), 'sha256': sha(path)}
                  for name, path in paths.items()]).to_csv(out/'INPUT_SHA256_PROVENANCE.csv', index=False)
    (out/'AUDIT_09D_STATUS.txt').write_text('PHASE09D_PROVENANCE_TYPED_METAL_RESPONSE_GRAPH_V0_NO_TF_NETWORK\n')
    print('AUDIT 09D complete:', out)
    print('Provisional response-network edges:', len(edges))
    print('Network nodes:', len(nodes))
    print('Native GPL84 measurements retained separately:', len(array))
    print('Metal-selected genes available for cross-context sign check:', len(overlay))
    print('No TF binding/regulatory edges, independent study validation or CF specificity inferred.')

def self_test():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        paths = input_paths(root)
        for p in paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        genes = [f'PA{i:04d}' for i in range(1,4924)]
        pd.DataFrame({'GeneID': genes}).to_csv(paths['frozen_gene_mask'], index=False)
        paths['09a_status'].write_text(EXPECTED_A+'\n')
        paths['09c_status'].write_text(EXPECTED_C+'\n')
        nsel=249
        edges = pd.DataFrame({'GeneID':genes,
                'full17_mean_effect_log2norm_countplus1': [2.1]*120+[-2.3]*(nsel-120)+[.02]*(4923-nsel),
                'full17_mean_effect_sign':[1]*120+[-1]*(nsel-120)+[1]*(4923-nsel),
                'full17_n_increased':[17]*120+[0]*(nsel-120)+[9]*(4923-nsel),
                'full17_n_decreased':[0]*120+[17]*(nsel-120)+[8]*(4923-nsel),
                'full17_n_matching_strongest_sign':[17]*nsel+[9]*(4923-nsel),
                'full17_descriptive_14of17_and_absmeanGE1':[True]*nsel+[False]*(4923-nsel),
                'n_LODO_training16_folds_selected_PRIMARY':[17]*nsel+[0]*(4923-nsel),
                'direct_regulatory_edge':[False]*4923,
                'CF_specific_edge':[False]*4923})
        edges.to_csv(paths['09c_candidate_edges'], index=False)
        ev = pd.DataFrame([{'donor_id':str(i),'rule':'PRIMARY_13of16_absmeanGE1',
                    'selection_uses_heldout_donor':False,'heldout_fraction_sign_correct':.96}
                    for i in range(1,18)])
        ev.to_csv(paths['09c_donor_evaluation'], index=False)
        pd.DataFrame([{'rule':'PRIMARY_13of16_absmeanGE1','mean_heldout_sign_accuracy':.96}]).to_csv(
            paths['09c_summary'],index=False)
        a = []
        for key in CONTRASTS:
            for i,gene in enumerate(genes[:4920]):
                eff = (i%9-4)/2
                a.append({'dataset_id':'GSE67006' if key==ANR else 'GSE6741',
                      'contrast_id':key,'GeneID':gene,'effect_value_native':eff,
                      'effect_unit':'transformed GPL84',
                      'effect_direction':'UP' if eff>0 else ('DOWN' if eff<0 else 'ZERO'),
                      'n_biological_treatment':3 if key==ANR else 2,
                      'n_biological_reference':3 if key==ANR else 2,
                      'evidence_type':'measured_expression_response_NOT_direct_TF_binding',
                      'direct_regulatory_edge_proven':False})
        pd.DataFrame(a).to_csv(paths['09a_native_arrays'],index=False)
        run(root,root/OUT)
        graph=pd.read_csv(root/OUT/'NETWORK_V0_PROVISIONAL_METAL_RESPONSE_EDGES_NOT_REGULATORY.csv')
        need(len(graph)==249 and (graph.direct_TF_regulation.astype(str).str.lower()=='false').all(),
             'synthetic network size/regulation flags wrong')
        need(len(pd.read_csv(root/OUT/'FROZEN_19680_UNQUALIFIED_GPL84_GENE_RESPONSES_NOT_NETWORK_EDGES.csv'))==19680,
             'synthetic evidence count wrong')
        try:
            run(root,root/OUT)
        except RuntimeError as e:
            need('non-overwrite' in str(e), 'existing output wrong rejection')
        else:
            raise AssertionError('output directory overwrite was not prevented')
    print('SELF_TEST_PASSED: strict 09a/09c schema, 249 metal response edges, 19680 separate array evidence rows, no-overwrite')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path, default=ROOT)
    ap.add_argument('--self-test', action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test()
    else:
        root=args.root.expanduser().resolve()
        run(root,root/OUT)

if __name__=='__main__':
    main()
