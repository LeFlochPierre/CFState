#!/usr/bin/env python3
"""CF-State Audit 08h: post-hoc shared orthogonal-programme leads.

Input: frozen descriptive programme decomposition from completed audit 08g.
Does NOT reproject samples, refit models, use independent controls, or validate
CF specificity. Both compared clinical contrasts share the SAME 5 Cornforth
non-CF wound/burn donor group; concordance may reflect that shared group.

Run from the CFstate project root:
  python src/phase8/audit_08h_orthogonal_programme_concordance.py --self-test
  python -u src/phase8/audit_08h_orthogonal_programme_concordance.py
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd().resolve()
SRC = ROOT / 'results/phase8/audit_08g_perturbation_anchored_clinical_decomposition'
OUT = ROOT / 'results/phase8/audit_08h_orthogonal_programme_concordance'
PROGRAMMES = SRC / 'DESCRIPTIVE_324_programme_contributions_to_clinical_decomposition.csv'
CONTRASTS = SRC / 'EXPLORATORY_3_clinical_contrast_parallel_orthogonal.csv'
GENES = SRC / 'EXPLORATORY_reference_genes_for_top_orthogonal_programmes.csv'
STATUS_FILE = SRC / 'AUDIT_08G_STATUS.txt'
STATUS = 'CFSTATE_PHASE8_08H_ORTHOGONAL_PROGRAMME_CONCORDANCE_DESCRIPTIVE_COMPLETE'
OLD = 'Cornforth_CF_minus_nonCF_7v5'
NEW = 'New_CF_minus_Cornforth_nonCF_2v5'
NEW_VS_OLD = 'New_CF_minus_Cornforth_CF_2v7'
COL = 'signed_CF_score_orthogonal_programme_contribution'
VEC = 'clinical_metal_orthogonal_programme_delta'


def require(label: str, cond: bool) -> None:
    print(f'{label} {"PASS" if cond else "FAIL"}', flush=True)
    if not cond:
        raise RuntimeError('CHECK FAILED: ' + label)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def combine(table: pd.DataFrame) -> pd.DataFrame:
    a = table.loc[table['contrast'].eq(OLD)].copy().set_index('imodulon')
    b = table.loc[table['contrast'].eq(NEW)].copy().set_index('imodulon')
    if len(a) != 108 or len(b) != 108 or not a.index.is_unique or set(a.index) != set(b.index):
        raise RuntimeError('Expected exactly the same unique 108 programme IDs for both contrasts')
    b = b.loc[a.index]
    for key in ['frozen_CF_signature_weight', 'mean_experimental_metal_change']:
        if not np.allclose(a[key].to_numpy(float), b[key].to_numpy(float), atol=1e-8):
            raise RuntimeError(f'Frozen input mismatch for {key}')
    old = a[COL].to_numpy(float)
    new = b[COL].to_numpy(float)
    consistent = (old * new > 0)
    same_positive = (old > 0) & (new > 0)
    same_negative = (old < 0) & (new < 0)
    r = pd.DataFrame({
        'imodulon': a.index.tolist(),
        'original18': a['selected_in_original18'].to_numpy(bool),
        'frozen_signature_weight': a['frozen_CF_signature_weight'].to_numpy(float),
        'mean_ZnFeMn_change': a['mean_experimental_metal_change'].to_numpy(float),
        'oldCF_minus_nonCF_orthogonal_programme_delta': a[VEC].to_numpy(float),
        'newCF_minus_nonCF_orthogonal_programme_delta': b[VEC].to_numpy(float),
        'oldCF_minus_nonCF_signed_contribution': old,
        'newCF_minus_nonCF_signed_contribution': new,
        'contribution_sign_agrees': consistent,
        'both_contributions_positive': same_positive,
        'both_contributions_negative': same_negative,
        'minimum_shared_positive_contribution': np.where(same_positive, np.minimum(old, new), 0.0),
        'minimum_shared_negative_magnitude': np.where(same_negative, np.minimum(-old, -new), 0.0),
        'minimum_shared_absolute_contribution_if_concordant': np.where(consistent, np.minimum(np.abs(old), np.abs(new)), 0.0),
        'difference_newCF_minus_oldCF_signed_contribution': new-old,
    })
    if not np.isfinite(r.select_dtypes('number').to_numpy(dtype=float)).all():
        raise RuntimeError('Nonfinite programme values')
    return r


def self_test() -> None:
    rows = []
    for contrast, c in [(OLD,[2.,-4.,-1.]),(NEW,[1.,-2.,3.])]:
        for i, v in enumerate(c):
            rows.append({'contrast':contrast,'imodulon':f'M{i}',
                         'selected_in_original18':i<2, 'frozen_CF_signature_weight':float(i+1)/10,
                         'mean_experimental_metal_change':float(i),
                         VEC:v*2,COL:v})
    # Pure helper test here uses three programmes; temporarily expect 3 for synthetic test.
    t = pd.DataFrame(rows)
    t = pd.concat([t] * 36, ignore_index=True)
    t['imodulon'] = [f'{row.imodulon}_{j}' for j in range(36) for row in pd.DataFrame(rows).itertuples(index=False)]
    out = combine(t)
    assert len(out) == 108
    assert np.isclose(out.loc[out.imodulon=='M0_0','minimum_shared_positive_contribution'].iloc[0],1.)
    assert np.isclose(out.loc[out.imodulon=='M1_0','minimum_shared_negative_magnitude'].iloc[0],2.)
    assert not out.loc[out.imodulon=='M2_0','contribution_sign_agrees'].iloc[0]
    print('AUDIT_08H_SELF_TEST_PASS', flush=True)


def main() -> None:
    print('='*78)
    print('CF-State Phase 8 — Audit 08h: orthogonal programme concordance')
    print('POST-HOC / SHARED NON-CF COMPARATOR / NOT INDEPENDENT VALIDATION')
    print('='*78, flush=True)
    require('project_root_has_results', (ROOT/'results/phase8').is_dir())
    require('new_output_directory', not OUT.exists())
    require('08g_completed', STATUS_FILE.is_file() and STATUS_FILE.read_text().strip() ==
            'CFSTATE_PHASE8_08G_PERTURBATION_ANCHORED_DECOMPOSITION_EXPLORATORY_COMPLETE')
    for name, path in [('108_programmes', PROGRAMMES), ('three_contrasts', CONTRASTS), ('reference_gene_hints', GENES)]:
        require('input_'+name, path.is_file())

    p = pd.read_csv(PROGRAMMES)
    q = pd.read_csv(CONTRASTS).set_index('contrast')
    require('three_times_108_input', len(p)==324 and set(p.contrast)=={OLD,NEW,NEW_VS_OLD})
    require('all_108_unique_each', all(len(x)==108 and x.imodulon.is_unique for _,x in p.groupby('contrast')))
    for name in (OLD,NEW,NEW_VS_OLD):
        expected = float(q.loc[name,'clinical_orthogonal_signed_CF_score'])
        actual = float(p.loc[p.contrast.eq(name),COL].sum())
        require('08g_orthogonal_score_reproduced_'+name, np.isclose(actual,expected,atol=1e-4))

    r = combine(p)
    require('same_shared_nonCF_algebra', np.allclose(
        r['difference_newCF_minus_oldCF_signed_contribution'].to_numpy(float),
        p.loc[p.contrast.eq(NEW_VS_OLD)].set_index('imodulon').loc[r.imodulon,COL].to_numpy(float), atol=1e-8))

    selected = r.sort_values('minimum_shared_absolute_contribution_if_concordant',ascending=False)
    pos = r.loc[r.both_contributions_positive].sort_values('minimum_shared_positive_contribution',ascending=False)
    neg = r.loc[r.both_contributions_negative].sort_values('minimum_shared_negative_magnitude',ascending=False)
    require('enough_concordant_programmes_to_describe', len(pos)+len(neg)>0)
    # The reference genes are selected from the top 12 for each of three contrasts in 08g.
    genes = pd.read_csv(GENES)
    require('reference_gene_headers', {'imodulon','PAO1_GeneID','reference_iModulon_loading'}.issubset(genes.columns))
    candidates = selected.loc[selected.minimum_shared_absolute_contribution_if_concordant>0,'imodulon'].head(12)
    gh = genes.loc[genes.imodulon.isin(candidates)].copy()
    missing = sorted(set(candidates) - set(gh.imodulon))

    print('\n========== SHARED-REFERENCE CAVEAT ==========',flush=True)
    print('Both positive clinical comparisons subtract the SAME five Cornforth non-CF donors.')
    print('Their programme concordance is NOT independent CF replication.')
    print('Orthogonal means only perpendicular to the experimental Zn+Fe+Mn mean vector.')
    print('The candidate is frozen; these rankings are retrospective descriptive results.')
    print('\n========== CROSS-COMPARISON PROGRAMME COUNTS ==========',flush=True)
    print('108 total; same-sign contributions:',int(r.contribution_sign_agrees.sum()),
          '; both positive:',len(pos),'; both negative:',len(neg),flush=True)
    cols = ['imodulon','original18','oldCF_minus_nonCF_signed_contribution',
            'newCF_minus_nonCF_signed_contribution',
            'minimum_shared_positive_contribution','mean_ZnFeMn_change']
    print('\n========== BOTH CONTRASTS: POSITIVE SIGNED CONTRIBUTIONS ==========',flush=True)
    print(pos[cols].head(15).round(5).to_string(index=False),flush=True)
    cols2 = ['imodulon','original18','oldCF_minus_nonCF_signed_contribution',
             'newCF_minus_nonCF_signed_contribution',
             'minimum_shared_negative_magnitude','mean_ZnFeMn_change']
    print('\n========== BOTH CONTRASTS: NEGATIVE SIGNED CONTRIBUTIONS ==========',flush=True)
    print(neg[cols2].head(15).round(5).to_string(index=False),flush=True)
    print('\n========== GENE MEMBERSHIP HINTS ==========',flush=True)
    print('Candidate programmes with top reference gene hints:',len(set(gh.imodulon)),
          'of',len(candidates),'; missing from 08g selected top-gene table:',missing,flush=True)

    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'AUDIT_08H_STATUS.txt').write_text('INCOMPLETE_PENDING_EXPORT\n')
    r.to_csv(OUT/'DESCRIPTIVE_108_shared_orthogonal_programme_directions.csv',index=False)
    pos.to_csv(OUT/'DESCRIPTIVE_shared_positive_programme_contributions.csv',index=False)
    neg.to_csv(OUT/'DESCRIPTIVE_shared_negative_programme_contributions.csv',index=False)
    gh.to_csv(OUT/'EXPLORATORY_top_reference_gene_membership_hints.csv',index=False)
    pd.DataFrame([{'path':str(x),'sha256':sha256(x)} for x in
                  [PROGRAMMES,CONTRASTS,GENES,STATUS_FILE,Path(__file__).resolve()]]).to_csv(
                      OUT/'INPUT_SHA256_PROVENANCE.csv',index=False)
    (OUT/'INTERPRETATION_LIMITS.txt').write_text(
        'Post-hoc descriptive agreement only; the two CF-versus-nonCF comparisons share\n'
        'exactly the same non-CF donor control group (n=5). Concordance may reflect it.\n'
        'CF lung versus wound/burn tissue, geography and processing are confounded.\n'
        'Original and new CF donors are not homogeneous and have different study histories.\n'
        'The mean metal axis is from PAO1 spike-ins given combined zinc, iron, manganese;\n'
        'orthogonal to this axis does not mean biologically non-metal.\n'
        'iModulon names are clues, not established programme-level biology without loadings.\n'
        'Reference gene members are not differential-expression validation.\n'
        'Candidate rankings used the clinical comparisons, so they are not held-out evidence.\n'
        'No p-values: sample-level independent replication is insufficient.\n'
    )
    (OUT/'AUDIT_08H_STATUS.txt').write_text(STATUS+'\n')
    print('\nSTATUS:',STATUS,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:main()
