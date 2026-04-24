function cfg = default()
%DEFAULT  Default configuration struct for the variational covariance smoother.
%
%   cfg = config.default()
%
%   Returns a nested struct with every tunable constant in one place so
%   nothing is hard-coded inside the algorithm files.

    % ---------- simulation / data ----------
    cfg.sim.p            = 20;         % number of variables / channels
    cfg.sim.k            = 50;         % Wishart degrees of freedom (window)
    cfg.sim.T            = 600;        % number of time points
    cfg.sim.T_tr         = 60;         % length of transition region
    cfg.sim.perc         = 50;         % % of off-diagonal entries that are non-zero
    cfg.sim.seed         = 0;          % RNG seed (0 = use default)
    cfg.sim.truth_type   = 'step';     % 'step' | 'linear'

    % ---------- base smoothing dial ----------
    cfg.smoother.lambda_base = 0.5;    % base lambda in lambda_t = base*sigmoid(z_t)

    % ---------- backward Monte Carlo sampler ----------
    cfg.bsampler.L = 10;               % number of backward samples

    % ---------- variational inference ----------
    cfg.vi.VI_iter       = 2;          % number of backward passes
    cfg.vi.m_multiplier  = 5;          % m = m_multiplier * k

    % ---------- adaptive-lambda EM (inner EM over z_t) ----------
    cfg.em.max_iter            = 20;
    cfg.em.tol                 = 1e-2;
    cfg.em.rho_init            = 0.95;
    cfg.em.sigma2_eps_init     = 0.1;
    cfg.em.rho_bounds          = [0.001, 0.999];
    cfg.em.sigma2_eps_bounds   = [1e-6,  1.0];
    cfg.em.convergence_on      = 'lambda';   % 'lambda' | 'Sigma'
    cfg.em.verbose             = true;

    % ---------- Newton-Raphson (inside EM forward filter) ----------
    cfg.nr.max_iter      = 15;
    cfg.nr.tol           = 1e-6;
    cfg.nr.dz_clamp      = 2;
    cfg.nr.z_clamp       = 5;
    cfg.nr.eig_floor     = 1e-6;
    cfg.nr.hess_damping  = 0.1;
    cfg.nr.var_floor     = 1e-4;

    % ---------- which (i,j) entry to plot ----------
    cfg.plot.idx = [1 2];
end
