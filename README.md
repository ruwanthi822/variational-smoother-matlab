# Adaptive Variational Covariance Smoother (MATLAB)

Clean, organized MATLAB implementation of the adaptive variational
covariance smoother — dynamic functional connectivity estimation from
Wishart-distributed sample-covariance observations with a time-varying
smoothing dial λ(t) learned by EM.

## Layout

```
variational-smoother-matlab/
├── main.m                      Top-level end-to-end script
├── data/
│   └── C.mat                   Bundled ground-truth trajectory (600 x 20 x 20)
├── +config/
│   └── default.m               Single source of truth for every tunable constant
├── +sim/
│   ├── step_truth.m            Ground-truth C(t) with a brief off-diagonal pulse
│   ├── linear_truth.m          Ground-truth C(t) with a linear transition
│   ├── load_mat.m              Load a pre-computed C trajectory from a .mat file
│   └── sample_wishart.m        Draws Y_t ~ Wishart(k, C_t), computes X_true
├── +estimators/
│   ├── forward_filter.m        Sigma_t forward recursion (fixed or time-varying lambda)
│   ├── backward_sampler.m      L-sample Wishart Monte-Carlo backward smoother
│   ├── vi_first_order.m        VI^(1) backward pass   (Eq. 21)
│   ├── vi_second_order.m       VI^(2) backward pass   (Eqs. 21, 22, 35, 46)
│   └── compute_am_bm.m         Closed-form Wishart-moment coefficients
├── +adaptive/
│   ├── forward_filter.m        Newton-Raphson forward filter for z_t
│   ├── backward_smoother.m     RTS backward pass for z_t
│   ├── m_step.m                M-step updates for (rho, sigma2_eps, lambda_t, Sigma_t)
│   └── run_em.m                Full EM loop orchestrator
├── +metrics/
│   └── mse_db.m                Relative MSE in dB (total + off-diagonal)
├── +plotting/
│   ├── entry_with_ci.m         (i,j)-entry trace with shaded CI
│   ├── lambda_trajectory.m     lambda_t + omega_t panels
│   └── mse_summary.m           Bar chart across methods
├── +utils/
│   ├── sq.m                    squeeze shorthand
│   ├── nearest_psd.m           Symmetrize + PSD clip
│   └── cov_to_prec.m           MVSV-scaled covariance->precision conversion
├── tests/
│   └── smoke_test.m            Small-T sanity test (no plots)
└── docs/                       (kept for notes / figures)
```

## Quick start

Clone the repo, open MATLAB in the repo root, then

```matlab
>> main
```

produces four figures (precision trace, covariance trace, lambda
trajectory, MSE bars) and writes a timestamped `results_YYYYMMDD_HHMMSS.mat`.

To sanity-check before running the full pipeline:

```matlab
>> addpath(pwd);  cd tests;  smoke_test;  cd ..
```

## Changing parameters

Every knob lives in `+config/default.m`. A typical experiment looks like:

```matlab
cfg = config.default();
cfg.sim.T             = 1200;    % longer trial
cfg.sim.truth_type    = 'linear';% try the smooth-transition truth
cfg.em.max_iter       = 50;
cfg.vi.m_multiplier   = 10;      % larger m in VI
% ...then either pass cfg into each function yourself, or edit main.m
```

## Pipeline

The pipeline produces four estimates of the precision X_t and covariance C_t:

1. **Forward filter** (constant lambda) — fastest, biased by the fixed dial.
2. **Backward sampler** — L Monte Carlo Wishart samples drawn backwards through time.
3. **VI first order** — closed-form backward recursion for V^(1), uses adaptive lambda.
4. **VI second order** — adds V^(2) correction, uses adaptive lambda.

Adaptive lambda is estimated by an EM loop over an AR(1) latent z_t with
lambda_t = lambda_base * sigmoid(z_t). The E-step is a Newton-Raphson
forward filter + RTS backward smoother for z_t; the M-step updates rho and
sigma2_eps in closed form, then recomputes lambda_t and Sigma_t.

## MSE metric

`metrics.mse_db(X_true, X_est)` returns the relative MSE in dB (and a
variant that only looks at off-diagonal entries):

```
MSE_dB = 10 * log10( sum_t ||X_true(t) - X_est(t)||_F^2
                     / sum_t ||X_true(t)||_F^2 )
```

Lower is better.

## Push to GitHub

The repository is already initialized with git. To publish:

```bash
cd "variational-smoother-matlab"
# create an empty repo at https://github.com/<you>/variational-smoother-matlab
git remote add origin https://github.com/<you>/variational-smoother-matlab.git
git branch -M main
git push -u origin main
```

## License

MIT — see LICENSE.

## Citation

If this code is useful for your work, please cite the accompanying paper
"An Adaptive Variational Covariance Smoother with Application to Dynamic
Functional Connectivity Analysis" (manuscript in preparation).
