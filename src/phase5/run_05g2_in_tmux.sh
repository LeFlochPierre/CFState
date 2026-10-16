#!/usr/bin/env bash
set -Eeuo pipefail

cd ~/projects/CFstate

{
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate CFSTATE_ENV

    echo "Active environment: $CONDA_DEFAULT_ENV"
    echo "Starting Audit 05g2"

    bash src/phase5/audit_05g2_Cornforth2018_technical_pilot.sh

} 2>&1 | tee logs/audit_05g2_Cornforth2018_technical_pilot.log
