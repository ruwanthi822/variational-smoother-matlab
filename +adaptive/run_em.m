function [Sigma_t, lambda_t, z_smooth, rho, sigma2_eps, history] = ...
         run_em(Y, Sigma_init, cfg)
%RUN_EM  Adaptive-lambda EM over the latent AR(1) state z_t.
%        Matches Master_updated.m convergence / fixed-rho behaviour.
%
%   Alternates the E-step (adaptive.forward_filter + adaptive.backward_smoother)
%   with the M-step (adaptive.m_step). Optionally overrides rho with
%   cfg.em.rho_fixed (a stabilizing trick used in Master_updated).
%
%   Convergence criterion: relative delta of sigma2_eps below cfg.em.tol
%   after the first two iterations.
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
    lambda_t   = cfg.smoother.lambda_base * ones(1, T);

    history.lambda      = zeros(cfg.em.max_iter, T);
    history.rho         = zeros(1, cfg.em.max_iter);
    history.sigma2_eps  = zeros(1, cfg.em.max_iter);
    history.delta_s2    = zeros(1, cfg.em.max_iter);

    sigma2_eps_prev = sigma2_eps;

    for iter = 1:cfg.em.max_iter
        % E-step
        [z_filt, z_pred, s2_filt, s2_pred] = ...
            adaptive.forward_filter(Y, Sigma_t, rho, sigma2_eps, cfg);
        [z_smooth, s2_smooth, A] = ...
            adaptive.backward_smoother(z_filt, z_pred, s2_filt, s2_pred, rho, cfg);

        % M-step
        [rho, sigma2_eps, lambda_t, Sigma_t] = ...
            adaptive.m_step(z_smooth, s2_smooth, A, Y, rho, cfg);

        % optional rho override (Master_updated pins rho at rho_fixed)
        if isfield(cfg.em, 'rho_fixed') && ~isempty(cfg.em.rho_fixed)
            rho = cfg.em.rho_fixed;
        end

        % relative change of sigma2_eps (skip iter 1)
        if iter > 1
            delta_s2 = abs(sigma2_eps - sigma2_eps_prev) / max(abs(sigma2_eps_prev), eps);
        else
            delta_s2 = Inf;
        end

        history.lambda(iter,:)     = lambda_t;
        history.rho(iter)          = rho;
        history.sigma2_eps(iter)   = sigma2_eps;
        history.delta_s2(iter)     = delta_s2;

        if cfg.em.verbose
            fprintf('  EM iter %3d  Ds2/s2=%.2e  rho=%.4f  sig2eps=%.4g  ', ...
                    iter, delta_s2, rho, sigma2_eps);
            fprintf('lam[min,mean,max]=[%.3f, %.3f, %.3f]\n', ...
                    min(lambda_t), mean(lambda_t), max(lambda_t));
        end

        if iter > 2 && delta_s2 < cfg.em.tol
            if cfg.em.verbose
                fprintf('  EM converged at iter %d (Ds2/s2=%.2e)\n', iter, delta_s2);
            end
            history.lambda      = history.lambda(1:iter, :);
            history.rho         = history.rho(1:iter);
            history.sigma2_eps  = history.sigma2_eps(1:iter);
            history.delta_s2    = history.delta_s2(1:iter);
            return;
        end

        sigma2_eps_prev = sigma2_eps;
    end

    if cfg.em.verbose
        fprintf('  EM reached max_iter=%d without converging (Ds2/s2=%.2e)\n', ...
                cfg.em.max_iter, delta_s2);
    end
end
