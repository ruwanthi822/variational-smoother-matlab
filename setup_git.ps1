# setup_git.ps1 — initializes a fresh git repo in this folder and makes
# the first commit. Run from a PowerShell prompt in this folder:
#
#     cd "$Env:USERPROFILE\OneDrive - University of Maryland\Desktop\variational-smoother-matlab"
#     .\setup_git.ps1
#
# Requires git for Windows (https://git-scm.com/download/win).

$ErrorActionPreference = "Stop"

# Remove any stale .git left behind by a failed previous attempt
if (Test-Path .git) {
    Write-Host "Removing stale .git directory..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force .git
}

Write-Host "Initializing git repository..." -ForegroundColor Cyan
git init -b main

git config user.name  "Ruwanthi Abeysekara"
git config user.email "ruwanthi822@gmail.com"

Write-Host "Staging files..." -ForegroundColor Cyan
git add .

Write-Host "Creating first commit..." -ForegroundColor Cyan
git commit -m @"
Initial commit: organized adaptive variational covariance smoother

- +config: single default() returning cfg struct (replaces hard-coded constants)
- +sim:    step and linear ground truth + Wishart sampler
- +estimators: forward filter, backward sampler, VI first / second order
- +adaptive: Newton-Raphson forward, RTS backward, M-step, run_em orchestrator
- +metrics: mse_db (total and off-diagonal, in dB)
- +plotting: entry_with_ci, lambda_trajectory, mse_summary
- +utils: sq, nearest_psd, cov_to_prec
- tests/smoke_test.m: small-T sanity check
- main.m: end-to-end pipeline
"@

Write-Host ""
Write-Host "Done. To push to GitHub:" -ForegroundColor Green
Write-Host "  1. Create an empty repo at https://github.com/<your-username>/variational-smoother-matlab"
Write-Host "  2. Run:"
Write-Host "       git remote add origin https://github.com/<your-username>/variational-smoother-matlab.git"
Write-Host "       git push -u origin main"
