#!/usr/bin/env bash
# Reproduce the tables and the figure (requires the LSEG panel, see README)
set -euo pipefail
cd "$(dirname "$0")"
for s in 01_panels 02_descriptives 03_lagged_scores 04_mandate 05_figure 06_robustness 07_rationales 08_correlations; do
    echo "== $s"
    python3 "$s.py"
done
