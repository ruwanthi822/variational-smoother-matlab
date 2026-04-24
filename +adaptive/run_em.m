function [Sigma_t, lambda_t, z_smooth, rho, sigma2_eps, history] = ...
         run_em(Y, Sigma_init, cfg)
%RUN_EM  Adaptive-lambda EM over the latent AR(1) state z_t.
%
%   Alternates the E-step (adaptive.forward_filter + adaptive.backward_smoother)
%   with the M-step (adaptive.m_step) until lambda_t (or Sigma_t) converges.
%
%   Inputs
%       Y          : [T x p x p] Wishart observations
%       Sigma_init : [T x p x p] initial Sigma_t (e.g., from a constant-lambda
%                                forward filter)
%       cfg        : config struct
%
%   Outputs
%       Sigma_t    : final Sigma_t, [T x p x p]
%       lambda_t   : final lambda_t, [1 x T]
%       z_smooth   : final smoothed latent z_t, [1 x T]
%       rho        : final AR(1) coefficient
%       sigma2_eps : final AR(1) innovation variance
%       history    : struct with .lambda, .rho, .sigma2_eps, .delta per iter

    T = size(Y, 1);
    rho        = cfg.em.rho_init;
    sigma2_eps = cfg.em.sigma2_eps_init;
    Sigma_t    = Sigma_init;

    lambda_t_prev = zeros(1, T);
    Sigma_prev    = Sigma_t;

    history.lambda     = zeros(cfg.em.max_iter, T);
    history.rho        = zeros(1, cfg.em.max_iter);
    history.sigma2_eps = zeros(1, cfg.em.max_iter);
    history.delta      = zeros(1, cfg.em.max_iter);

    for iter = 1:cfg.em.max_iter
        % E-step
        [z_filt, z_pred, s2_filt, s2_pred] = ...
            adaptive.forward_filter(Y, Sigma_t, rho, sigma2_eps, cfg);
        [z_smooth, s2_smooth, A] = ...
            adaptive.backward_smoother(z_filt, z_pred, s2_filt, s2_pred, rho, cfg);

        % M-step
        [rho, sigma2_eps, lambda_t, Sigma_t] = ...
            adaptive.m_step(z_smooth, s2_smooth, A, Y, rho, cfg);

        % convergence
        d_lam = max(abs(lambda_t - lambda_t_prev));
        d_Sig = max(abs(Sigma_t(:) - Sigma_prev(:)));
        if strcmpi(cfg.em.convergence_on, 'sigma')
            delta = d_Sig;
        else
            delta = d_lam;
        end

        history.lambda(iter,:)     = lambda_t;
        history.rho(iter)          = rho;
        history.sigma2_eps(iter)   = sigma2_eps;
        history.delta(iter)        = delta;

        if cfg.em.verbose
            fprintf('  EM iter %2d  Dlam=%.4f  DSig=%.4f  rho=%.4f  sig2eps=%.4g\n', ...
                    iter, d_lam, d_Sig, rho, sigma2_eps);
        end

        if delta < cfg.em.tol
            if cfg.em.verbose
                fprintf('  EM converged at iter %d (delta=%.4g)\n', iter, delta);
            end
            history.lambda     = history.lambda(1:iter, :);
            history.rho        = history.rho(1:iter);
            history.sigma2_eps = history.sigma2_eps(1:iter);
            history.delta      = history.delta(1:iter);
            return;
        end

        lambda_t_prev = lambda_t;
        Sigma_prev    = Sigma_t;
    end

    if cfg.em.verbose
        fprintf('  EM reached max_iter=%d without converging (final delta=%.4g)\n', ...
                cfg.em.max_iter, delta);
    end
end
