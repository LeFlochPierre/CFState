#!/usr/bin/env python3
"""CF-State Phase 6 / Audit 06d: descriptive gene-loading triage only.

Reads frozen 06c outputs and technical coverage data. Does not discover,
fit or label CF-specific physiological programmes.
"""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path.cwd()
p4 = root / 'results/phase4'
p5 = root / 'results/phase5'
p6 = root / 'results/phase6'
source = p6 / 'audit_06c_residual_component_stability_diagnostics'
out = p6 / 'audit_06d_component_gene_annotation_triage'

status = source / 'AUDIT_06C_STATUS.txt'
assert status.read_text().strip() == (
    'CFSTATE_PHASE6_06C_RESIDUAL_TRANSFER_STABILITY_DIAGNOSTICS_COMPLETE'
)
if out.exists():
    raise RuntimeError('Audit 06d already exists; refusing to overwrite.')

weight_file = source / 'EXPLORATORY_NOT_VALIDATED_Atlanta_residual_PC1_PC2_PC3_gene_weights.csv'
training_file = source / 'FROZEN_Atlanta_leave_one_specimen_out_retraining.csv'
donor_file = source / 'FROZEN_external_leave_one_donor_out_checks.csv'
atlanta_file = p4 / 'audit_04k4_PRJNA576508_blinded_softcore_projection/FROZEN_Atlanta_softcore_4923_log2TPM.csv'
counts_file = p5 / 'audit_05h_Cornforth2018_full_depth_baseline/FROZEN_13_raw_PAO1_gene_counts_TECHNICAL_BASELINE.csv'
manifest_file = p5 / 'audit_05e_Cornforth2018_download_manifest_lock/FROZEN_13_main_clinical_sample_manifest.csv'
saf_file = root / 'references/phase4/PRJNA576508_cornforth/Cornforth_PAO1_no_decimal_locus_tags.saf'

w = pd.read_csv(weight_file).set_index('GeneID')
atlanta = pd.read_csv(atlanta_file).set_index('GeneID')
counts = pd.read_csv(counts_file).set_index('Geneid')
manifest = pd.read_csv(manifest_file)
saf = pd.read_csv(saf_file, sep='\t').set_index('GeneID')
training = pd.read_csv(training_file)
donors = pd.read_csv(donor_file)

axes = ['exploratory_PC1_weight', 'exploratory_PC2_weight', 'exploratory_PC3_weight']
assert len(w) == 4923 and w.index.is_unique
assert set(w.index) == set(atlanta.index) and set(w.index).issubset(counts.index)
assert all(x in w.columns for x in axes)
assert atlanta.shape == (4923, 13)
assert counts.shape == (5567, 13)
assert set(manifest['published_sample']) == set(counts.columns)
assert set(w.index).issubset(saf.index)
assert np.isfinite(w[axes].to_numpy()).all()
assert np.isfinite(atlanta.to_numpy()).all()
assert np.isfinite(counts.to_numpy()).all()
assert (counts.to_numpy() >= 0).all()

w = w[axes].copy()
genes = list(w.index)
cf_names = manifest.loc[manifest.group == 'CF_sputum', 'published_sample'].tolist()
non_names = manifest.loc[manifest.group != 'CF_sputum', 'published_sample'].tolist()
assert len(cf_names) == 7 and len(non_names) == 6

info = pd.DataFrame(index=genes)
info['Atlanta_mean_log2TPM'] = atlanta.loc[genes].mean(axis=1)
info['Atlanta_sd_log2TPM'] = atlanta.loc[genes].std(axis=1, ddof=1)
info['Cornforth_detected_frac_all13'] = (counts.loc[genes] > 0).mean(axis=1)
info['Cornforth_detected_frac_CF7'] = (counts.loc[genes, cf_names] > 0).mean(axis=1)
info['Cornforth_detected_frac_nonCF6'] = (counts.loc[genes, non_names] > 0).mean(axis=1)
info['PAO1_gene_length_bp'] = (saf.loc[genes, 'End'] - saf.loc[genes, 'Start'] + 1).astype(int)

all_genes = w.join(info)
rows = []
for component, col in enumerate(axes, start=1):
    for direction, ascending in [('positive', False), ('negative', True)]:
        ranked = all_genes.sort_values(col, ascending=ascending, kind='stable').head(20)
        for rank, (gene, row) in enumerate(ranked.iterrows(), start=1):
            rows.append({
                'component': f'PC{component}',
                'loading_direction': direction,
                'rank_within_direction': rank,
                'GeneID': gene,
                'signed_gene_weight': row[col],
                'Atlanta_mean_log2TPM': row['Atlanta_mean_log2TPM'],
                'Atlanta_sd_log2TPM': row['Atlanta_sd_log2TPM'],
                'Cornforth_detected_frac_all13': row['Cornforth_detected_frac_all13'],
                'Cornforth_detected_frac_CF7': row['Cornforth_detected_frac_CF7'],
                'Cornforth_detected_frac_nonCF6': row['Cornforth_detected_frac_nonCF6'],
                'PAO1_gene_length_bp': row['PAO1_gene_length_bp'],
            })
shortlist = pd.DataFrame(rows)

check_train = training.loc[
    (training.normalisation == 'FROZEN_5567_TPM')
    & (training.external_group == 'CF')
].sort_values('PC_subspace_overlap_with_full_Atlanta', ascending=True)
assert len(check_train) == 13
check_donors = donors.loc[donors.normalisation == 'FROZEN_5567_TPM']
assert len(check_donors) == 12

out.mkdir(parents=True, exist_ok=False)
all_genes.to_csv(out / 'EXPLORATORY_ALL_4923_PC_LOADINGS_AND_COVERAGE.csv', index_label='GeneID')
shortlist.to_csv(out / 'EXPLORATORY_TOP20_POSITIVE_NEGATIVE_PER_PC.csv', index=False)
check_train.to_csv(out / 'DESCRIPTIVE_Atlanta_training_influence_ranked.csv', index=False)
check_donors.to_csv(out / 'DESCRIPTIVE_external_donor_influence.csv', index=False)
(out / 'DESIGN_AND_LIMITATIONS.txt').write_text(
    'Gene-loading and technical-coverage triage from frozen 06c axes.\n'
    'PC signs are mathematically arbitrary; do not call a sign beneficial or pathogenic.\n'
    'Gene-loading magnitudes are not differential-expression tests or GO enrichment.\n'
    'Detected gene fractions are sensitive to sequencing depth and PAO1 mapping.\n'
    'Atlanta donors are not established independent; no CF-specific programme validated.\n'
    'No new latent model was fitted or selected by this audit.\n'
)
(out / 'AUDIT_06D_STATUS.txt').write_text(
    'CFSTATE_PHASE6_06D_EXPLORATORY_GENE_TRIAGE_COMPLETE\n'
)

print('=== LEAST STABLE ATLANTA OMITTED SPECIMENS ===')
print(check_train[
    ['omitted_Atlanta_specimen', 'PC_subspace_overlap_with_full_Atlanta',
     'external_fraction_with_3_retrained_PCs']
].head(5).round(5).to_string(index=False))
print('\n=== PC2 AND PC3: TOP 12 GENES BY ABSOLUTE LOADING ===')
for pc in (2, 3):
    col = f'exploratory_PC{pc}_weight'
    ranking = all_genes.assign(abs_weight=all_genes[col].abs()).sort_values('abs_weight', ascending=False).head(12)
    print(f'\nPC{pc}:')
    print(ranking[[col, 'Atlanta_mean_log2TPM', 'Atlanta_sd_log2TPM',
                   'Cornforth_detected_frac_all13']].round(4).to_string())
print('\nSTATUS: CFSTATE_PHASE6_06D_EXPLORATORY_GENE_TRIAGE_COMPLETE')
