function cfg = default()
%DEFAULT  Default configuration struct for the variational covariance smoother.
%
%   cfg = config.default()
%
%   Returns a nested struct with every tunable constant in one place so
%   nothing is hard-coded inside the algorithm files.

    % ---------- simulation / data ----------
    %   Matches Master_updated.m defaults: k = 80, lambda_base = 0.45.
    cfg.sim.p            = 20;         % number of variables / channels
                                       %   ignored when truth_type='load_mat' (taken from file)
    cfg.sim.k            = 80;         % Wishart degrees of freedom (window)
    cfg.sim.T            = 600;        % number of time points
                                       %   ignored when truth_type='load_mat' (taken from file)
    cfg.sim.T_tr         = 60;         % length of transition region
    cfg.sim.perc         = 50;         % % of off-diagonal entries that are non-zero
    cfg.sim.seed         = 0;          % RNG seed (0 = use default)
    cfg.sim.truth_type   = 'load_mat'; % 'step' | 'linear' | 'load_mat'
    %   Default points at the bundled data/C.mat (shape 600 x 20 x 20),
    %   which is the same ground-truth trajectory used in the original
    %   paper-producing simulations.
    cfg.sim.load_mat     = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'data', 'C.mat');
    cfg.sim.load_varname = 'C';

    % ---------- base smoothing dial ----------
    cfg.smoother.lambda_base = 0.45;   % base lambda in lambda_t = base*sigmoid(z_t)

    % ---------- backward Monte Carlo sampler ----------
    cfg.bsampler.L = 10;               % number of backward samples

    % ---------- variational inference ----------
    cfg.vi.VI_iter       = 2;          % number of backward passes
    cfg.vi.m_multiplier  = 5;          % m = m_multiplier * k

    % ---------- adaptive-lambda EM (inner EM over z_t) ----------
    %   Matches Master_updated.m:
    %     max_iter 400, tol on relative sigma2_eps = 1e-4,
    %     rho init 0.8, rho fixed at 0.85, sigma2_eps init 10,
    %     rho bounds [0.5, 0.9999], sigma2_eps floor 1e-10.
    cfg.em.max_iter            = 400;
    cfg.em.tol                 = 1e-4;           % relative delta of sigma2_eps
    cfg.em.rho_init            = 0.8;
    cfg.em.rho_fixed           = 0.85;           % set to [] to let rho float
    cfg.em.sigma2_eps_init     = 10;
    cfg.em.rho_bounds          = [0.5, 0.9999];
    cfg.em.sigma2_eps_floor    = 1e-10;
    cfg.em.z_filt_init         = 1;              % z(1); sigmoid(1) ~ 0.73
    cfg.em.verbose             = true;

    % ---------- Newton-Raphson (inside EM forward filter) ----------
    %   Matches Master_updated.m constants:
    %     max_iter = 5, tol = 1e-2, dz_clamp = 20, hess_damping = 0.1
    cfg.nr.max_iter      = 5;
    cfg.nr.tol           = 1e-2;
    cfg.nr.dz_clamp      = 20;
    cfg.nr.z_clamp       = 20;
    cfg.nr.eig_floor     = 1e-6;
    cfg.nr.hess_damping  = 0.1;
    cfg.nr.var_floor     = 1e-6;

    % ---------- which (i,j) entry to plot ----------
    cfg.plot.idx = [1 2];
end
