#!/usr/bin/env bash
set -Eeuo pipefail

cd ~/projects/CFstate

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate CFSTATE_ENV

echo "Active environment: $CONDA_DEFAULT_ENV"

bash src/phase5/audit_05h_Cornforth2018_full_depth_baseline.sh \
    2>&1 | tee logs/audit_05h_Cornforth2018_full_depth_baseline.log
