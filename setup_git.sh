#!/usr/bin/env bash
# setup_git.sh — initializes a fresh git repo in this folder and makes
# the first commit. Run from Git Bash on Windows (or Linux/macOS):
#
#     cd "/path/to/variational-smoother-matlab"
#     bash setup_git.sh
#
set -euo pipefail

# Clean up any stale .git
if [ -d ".git" ]; then
    echo "Removing stale .git directory..."
    rm -rf .git
fi

echo "Initializing git repository..."
git init -b main

git config user.name  "Ruwanthi Abeysekara"
git config user.email "ruwanthi822@gmail.com"

echo "Staging files..."
git add .

echo "Creating first commit..."
git commit -m "Initial commit: organized adaptive variational covariance smoother

- +config: single default() returning cfg struct (replaces hard-coded constants)
- +sim:    step and linear ground truth + Wishart sampler
- +estimators: forward filter, backward sampler, VI first / second order
- +adaptive: Newton-Raphson forward, RTS backward, M-step, run_em orchestrator
- +metrics: mse_db (total and off-diagonal, in dB)
- +plotting: entry_with_ci, lambda_trajectory, mse_summary
- +utils: sq, nearest_psd, cov_to_prec
- tests/smoke_test.m: small-T sanity check
- main.m: end-to-end pipeline"

echo
echo "Done. To push to GitHub:"
echo "  1. Create an empty repo at https://github.com/<your-username>/variational-smoother-matlab"
echo "  2. Run:"
echo "       git remote add origin https://github.com/<your-username>/variational-smoother-matlab.git"
echo "       git push -u origin main"
