#!/usr/bin/env python3
"""CF-State 09m. Read-only input census and REVIEW-ONLY contrast proposals.

This is NOT a training contrast freeze, differential expression analysis, or a
sample-to-count alignment check. No automatic treated-v-control labels.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

DESIGN_DIR = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Design_Matrices')
META_DIR = Path('external/EPathDash/GEO_Datasets/Pseudomonas/Metadata')
PRECISE = Path('external/aeruprecise/precise411/data_files/sample_table.csv')
OUT = Path('results/phase9/audit_09m_multi_perturbation_contrast_census')
IGNORE_COLUMN = re.compile(r'^(replicate|rep|repeat|sample|sample_id|run|gsm|srr|id|batch|library|donor|patient|subject|index|unnamed.*)$', re.I)
FILENAME_EXCLUDE = re.compile(r'design|metadata|sample_table|preview|diagnostic|summary|audit|qc_|\.git/', re.I)
EXP_EXT = ('.csv', '.tsv', '.txt', '.csv.gz', '.tsv.gz', '.txt.gz', '.parquet', '.xlsx', '.xlsx.gz', '.rds', '.rdata')
PRIOR_NAMES = {'GSE136111', 'GSE203348', 'GSE87213', 'GSE120602', 'GSE12207', 'GSE191103', 'GSE299861', 'GSE223663', 'GSE224821', 'GSE67006', 'GSE68534', 'GSE6741'}


def ensure(ok, why):
    if not ok:
        raise RuntimeError('AUDIT_09M_FAILED: '+str(why))


def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def safe_str(v):
    if pd.isna(v):
        return 'MISSING'
    x = str(v).strip()
    return re.sub(r'\s+', ' ', x)[:200] if x else 'MISSING'


def load_design(p):
    df = pd.read_csv(p, dtype=str, encoding='utf-8-sig', keep_default_na=False)
    df.columns = [safe_str(k) for k in df.columns]
    ensure(len(df.columns) == len(set(df.columns)), 'duplicate columns in '+str(p))
    for col in df.columns:
        df[col] = df[col].map(safe_str)
    return df


def kind_of_variable(col):
    col = col.lower()
    if re.search(r'strain|isolate|lineage|background', col):
        return 'STRAIN_COMPARISON_NOT_PERTURBATION'
    if re.search(r'genotype|mutant|mutation|knockout|deletion', col):
        return 'GENETIC_INTERVENTION_CANDIDATE'
    if re.search(r'treatment|expos|drug|antibiotic|oxygen|iron|zinc|metal|stress|dose|carbon|nutrient|condition', col):
        return 'ENVIRONMENT_OR_TREATMENT_CANDIDATE'
    if re.search(r'growth|state|biofilm|planktonic|phase', col):
        return 'PHYSIOLOGICAL_STATE_COMPARISON'
    if re.search(r'time|hour|minute|day|duration', col):
        return 'TIMEPOINT_COMPARISON_NOT_STANDALONE_PERTURBATION'
    if re.search(r'media|medium|culture', col):
        return 'CULTURE_CONTEXT_COMPARISON'
    return 'UNCLASSIFIED_GROUP_COMPARISON'


def infer_reference(levels):
    """Heuristic CONTROL-like value for prioritization ONLY, never assigned as truth."""
    def mark(s):
        s = s.lower().strip()
        if re.fullmatch(r'(wt|wild[ -]?type|control|ctrl|untreated|unexposed|uninduced|vehicle|mock|none|no treatment|0|0\.0|parental)', s):
            return True
        if re.fullmatch(r'no[ -].+|without[ -].+', s):
            return True
        return False
    return [v for v in levels if mark(v)]


def discover_expression_files(root, studies):
    """Names only: finding a candidate file DOES NOT verify column/sample alignment."""
    targets = [root / 'external/EPathDash/GEO_Datasets/Pseudomonas', root / 'data/processed', root / 'data/raw']
    found = {s: [] for s in studies}
    patt = re.compile('|'.join(re.escape(x) for x in sorted(studies, key=len, reverse=True)), re.I)
    for base in targets:
        if not base.is_dir():
            continue
        for p in base.rglob('*'):
            if not p.is_file():
                continue
            name = p.name
            if not name.lower().endswith(EXP_EXT) or FILENAME_EXCLUDE.search(name):
                continue
            matches = set(patt.findall(name))
            for sample_id in matches:
                s = sample_id.upper()
                found[s].append(str(p.relative_to(root)))
    for s in found:
        found[s] = sorted(set(found[s]))
    return found


def propose_contrasts(df, study, meta_strain):
    factors = [c for c in df.columns if not IGNORE_COLUMN.fullmatch(c.strip()) and 2 <= df[c].nunique() <= 12]
    proposals = []
    for variable in factors:
        other = [c for c in factors if c != variable]
        kind = kind_of_variable(variable)
        # Holding all observed other factors constant gives unconfounded design proposals
        # WITH RESPECT TO RECORDED columns only. Unrecorded confounding remains possible.
        chunks = [('', df)] if not other else list(df.groupby(other, dropna=False, sort=True))
        for strata, chunk in chunks:
            grouped = {str(k): len(z) for k, z in chunk.groupby(variable, dropna=False)}
            viable = sorted(k for k, n in grouped.items() if n >= 2 and k != 'MISSING')
            if len(viable) < 2:
                continue
            control_like = set(infer_reference(viable))
            for left, right in itertools.combinations(viable, 2):
                orientation = 'REVIEW_BOTH_DIRECTIONS'
                if left in control_like and right not in control_like:
                    orientation = f'POSSIBLE_CONTROL={left}'
                if right in control_like and left not in control_like:
                    orientation = f'POSSIBLE_CONTROL={right}'
                if isinstance(strata, tuple):
                    stratum = ' | '.join(f'{col}={val}' for col, val in zip(other, strata))
                elif other:
                    stratum = f'{other[0]}={strata}'
                else:
                    stratum = 'NONE'
                candidate = dict(
                    study=study, source='EPathDash_design',
                    changing_factor=variable, comparison_type=kind,
                    level_a=left, n_a=grouped[left], level_b=right, n_b=grouped[right],
                    controlled_recorded_factors=stratum, control_hint=orientation,
                    strain_metadata=meta_strain,
                    review_status='NOT_YET_VALIDATED',
                    sample_to_counts_mapping='NOT_CHECKED',
                    count_normalisation='NOT_CHECKED',
                    biological_replicates='NOT_VERIFIED',
                    independent_study='NOT_VERIFIED',
                    contrast_direction='NOT_FROZEN')
                priority = 10
                if kind == 'GENETIC_INTERVENTION_CANDIDATE': priority += 35
                if kind == 'ENVIRONMENT_OR_TREATMENT_CANDIDATE': priority += 30
                if kind == 'CULTURE_CONTEXT_COMPARISON': priority += 15
                if kind == 'PHYSIOLOGICAL_STATE_COMPARISON': priority += 12
                if kind.startswith('STRAIN_') or kind.startswith('TIMEPOINT_'): priority -= 25
                if orientation != 'REVIEW_BOTH_DIRECTIONS': priority += 12
                if min(grouped[left], grouped[right]) >= 3: priority += 8
                if 'PAO1' in meta_strain.upper(): priority += 5
                if study in PRIOR_NAMES: priority -= 3  # flag possible prior exposure, not exclusion
                candidate['design_priority'] = priority
                proposals.append(candidate)
    return proposals


def precise_summary(p):
    if not p.is_file():
        return {}, pd.DataFrame()
    try:
        df = pd.read_csv(p, dtype=str, encoding='utf-8-sig')
    except Exception as exc:
        return {'read_error': str(exc)}, pd.DataFrame()
    cols = list(df.columns)
    project = next((c for c in cols if re.search(r'(^project$|^study$|project[_ ]name|study[_ ]name)', c, re.I)), None)
    info = {'rows': len(df), 'columns': cols[:30], 'project_column': project, 'n_projects': None}
    summary = pd.DataFrame()
    if project:
        summary = df[project].fillna('MISSING').value_counts().rename_axis('project').reset_index(name='n_samples')
        info['n_projects'] = len(summary)
    return info, summary


def run(root, output, write=True):
    directory = root / DESIGN_DIR
    ensure(directory.is_dir(), 'missing '+str(directory))
    files = sorted(directory.glob('GSE*_design.csv'))
    ensure(files, 'no EPathDash design matrices')
    if write:
        ensure(not output.exists(), 'refusing to overwrite '+str(output))
    studies = [p.name.split('_design.csv')[0] for p in files]
    expression = discover_expression_files(root, studies)
    all_props, inventory, expr_rows, provenance = [], [], [], []
    for path in files:
        study = path.name.split('_design.csv')[0]
        meta_p = root / META_DIR / f'{study}_metadata.csv'
        metadata = pd.read_csv(meta_p, dtype=str, encoding='utf-8-sig') if meta_p.is_file() else pd.DataFrame()
        strain = 'UNKNOWN'
        title = ''
        if not metadata.empty:
            strain_col = next((c for c in metadata.columns if c.lower().strip()=='strain'), None)
            title_col = next((c for c in metadata.columns if c.lower().strip()=='title'), None)
            if strain_col: strain = safe_str(metadata[strain_col].iloc[0])
            if title_col: title = safe_str(metadata[title_col].iloc[0])
        try:
            design = load_design(path)
            props = propose_contrasts(design, study, strain)
            issue = ''
        except Exception as exc:
            design, props, issue = pd.DataFrame(), [], str(exc)
        for prop in props:
            prop['local_expression_candidate_count'] = len(expression[study])
            prop['prior_CFstate_study_seen'] = study in PRIOR_NAMES
        all_props.extend(props)
        inventory.append(dict(study=study, n_design_samples=len(design), n_design_columns=len(design.columns),
                              design_factors=' | '.join(map(str,design.columns)),
                              n_comparison_candidates=len(props), strain=strain,title=title,
                              local_expression_candidate_count=len(expression[study]),
                              previously_processed_or_audited=study in PRIOR_NAMES,
                              read_warning=issue))
        provenance.append(dict(source=str(path.relative_to(root)), sha256=sha256(path)))
        if meta_p.is_file(): provenance.append(dict(source=str(meta_p.relative_to(root)), sha256=sha256(meta_p)))
        for f in expression[study]:
            expr_rows.append(dict(study=study, candidate_expression_path=f, verified_sample_alignment=False))

    candidates = pd.DataFrame(all_props)
    inv = pd.DataFrame(inventory)
    e = pd.DataFrame(expr_rows,columns=['study','candidate_expression_path','verified_sample_alignment'])
    if not candidates.empty:
        candidates = candidates.sort_values(['design_priority','study','changing_factor'], ascending=[False,True,True]).reset_index(drop=True)
    precise_info, precise_projects = precise_summary(root/PRECISE)
    if (root/PRECISE).is_file(): provenance.append(dict(source=str(PRECISE),sha256=sha256(root/PRECISE)))

    print('\n'+'='*89)
    print('CF-STATE 09M | MULTI-PERTURBATION CONTRAST OPPORTUNITY CENSUS (NOT A MODEL)')
    print('='*89)
    print('EPathDash Pseudomonas studies with design files:', len(inv))
    print('Study design rows (NOT distinct experimental conditions):',int(inv.n_design_samples.sum()))
    print('Study files with at least one REVIEW-ONLY candidate pair:',int((inv.n_comparison_candidates>0).sum()))
    print('Candidate contrasts, including all eligible strata:',len(candidates))
    print('Studies with local expression filename leads:',int((inv.local_expression_candidate_count>0).sum()))
    print('Known historical CF-State study accessions:',len(set(studies)&PRIOR_NAMES))
    print('IMPORTANT: candidate raw-count files NOT sample-aligned, comparisons NOT validated.')
    print('Pairs compare ONE recorded variable within other matched design-factor strata.')
    print('Min 2 rows/group != 2 independent biological replicates. Treat/control not assigned.')
    print('\nCANDIDATE TYPES:')
    if not candidates.empty:
        print(candidates.comparison_type.value_counts().to_string())
    print('\nTOP 35 CANDIDATE GROUP COMPARISONS (REVIEW ONLY):')
    if not candidates.empty:
        cols=['study','strain_metadata','changing_factor','level_a','n_a','level_b','n_b','controlled_recorded_factors','comparison_type','local_expression_candidate_count']
        print(candidates.head(35)[cols].to_string(index=False,max_colwidth=45))
    print('\nDATASETS WITH MANY REVIEWABLE CONTRAST CANDIDATES:')
    print(inv.sort_values(['n_comparison_candidates','study'],ascending=[False,True]).head(20)[['study','n_design_samples','n_comparison_candidates','strain','local_expression_candidate_count','previously_processed_or_audited']].to_string(index=False))
    print('\nLOCALLY AVAILABLE EXPRESSION FILE LEADS (first 25; check structure before use):')
    if not e.empty:
        print(e.head(25).to_string(index=False,max_colwidth=100))
    else:
        print('None found by matching GSE accession in candidate expression filenames.')
        print('This does NOT mean expression is absent; inspect raw-count folder structure.')
    print('\nAERUPRECISE 411 SAMPLE TABLE (INDEPENDENT LAYER; NOT AUTOMATIC CONTRASTS):')
    print('sample_table:',str(PRECISE),'exists=',(root/PRECISE).is_file())
    print('rows:',precise_info.get('rows','NA'),'project column:',precise_info.get('project_column'),'projects:',precise_info.get('n_projects'))
    print('columns (first 30):', precise_info.get('columns',[]))
    if not precise_projects.empty:
        print(precise_projects.head(20).to_string(index=False))
    print('\nPRIMARY DECISION: confirm study sample-to-count mapping, treatment/control interpretation,')
    print('biological replication, platform/strain ID mapping and overlap with aeruPRECISE.')
    print('Do NOT fit model 2.0 to this unvalidated candidate list.')

    if write:
        output.mkdir(parents=True,exist_ok=False)
        inv.to_csv(output/'STUDY_DESIGN_INVENTORY.csv',index=False)
        candidates.to_csv(output/'CONTRAST_CANDIDATES_REVIEW_ONLY.csv',index=False)
        e.to_csv(output/'LOCAL_EXPRESSION_FILENAME_LEADS.csv',index=False)
        precise_projects.to_csv(output/'PRECISE411_PROJECT_COUNTS.csv',index=False)
        pd.DataFrame(provenance).to_csv(output/'SOURCE_PROVENANCE_SHA256.csv',index=False)
        status = dict(status='DESIGN_CANDIDATES_NOT_VALIDATED',n_studies=len(inv),n_candidates=len(candidates),
                      n_with_expression_name_leads=int((inv.local_expression_candidate_count>0).sum()),
                      precise411=precise_info,training_contrasts_frozen=0,method='single-factor difference with other recorded factors fixed; n>=2 samples/level')
        (output/'QC_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
        print('\nSAVED NON-DESTRUCTIVELY:',output)
    return inv, candidates


def self_test():
    with tempfile.TemporaryDirectory() as d:
        root=Path(d)
        p=root/DESIGN_DIR
        p.mkdir(parents=True)
        md=root/META_DIR
        md.mkdir(parents=True)
        (root/PRECISE).parent.mkdir(parents=True)
        pd.DataFrame({'project':['a','a','b'],'condition':['ctrl','ko','ctrl']}).to_csv(root/PRECISE,index=False)
        design=pd.DataFrame({'Replicate':['1','2','3','1','2','3','1','2','3','1','2','3'],
                             'Medium':['A']*6+['B']*6,
                             'Treatment':['WT']*3+['delta_anr']*3+['WT']*3+['delta_anr']*3})
        design.to_csv(p/'GSE000001_design.csv',index=False)
        pd.DataFrame({'Strain':['PAO1'],'Title':['KO demonstration']}).to_csv(md/'GSE000001_metadata.csv',index=False)
        pd.DataFrame({'Replicate':[1,2,1,2], 'Strain':['PAO1','PAO1','PA14','PA14']}).to_csv(p/'GSE000002_design.csv',index=False)
        (root/'data/processed').mkdir(parents=True)
        (root/'data/processed/GSE000001_counts.csv').write_text('Gene,WT_1\nPA0001,10\n')
        inv, candidates=run(root,root/'results/test09m',write=True)
        t=candidates[(candidates.study=='GSE000001') & (candidates.changing_factor=='Treatment')]
        ensure(len(t)==2,'two correctly stratified KO comparisons expected')
        ensure(set(t.n_a)=={3} and set(t.n_b)=={3},'min two replicates')
        ensure((t.comparison_type=='ENVIRONMENT_OR_TREATMENT_CANDIDATE').all(),'kind check')
        ensure(set(candidates[candidates.study=='GSE000002'].comparison_type)=={'STRAIN_COMPARISON_NOT_PERTURBATION'},'strain is not perturbation')
        ensure(int(inv[inv.study=='GSE000001'].local_expression_candidate_count.iloc[0])==1,'expression file index')
        try:
            run(root,root/'results/test09m',write=True)
        except RuntimeError as exc:
            ensure('overwrite' in str(exc),'non-overwrite')
        else:
            raise AssertionError('did not reject overwrite')
        print('SELF_TEST_PASSED: matched-factor contrast groups, candidate types, expression filename leads, PRECISE project index, source hashes, non-overwrite.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=Path.cwd())
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test()
    else:
        root=args.root.resolve()
        run(root,root/OUT,write=True)

if __name__=='__main__':
    main()
