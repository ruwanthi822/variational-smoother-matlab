function [rho_new, sigma2_eps_new, lambda_t, Sigma_t] = ...
         m_step(z_smooth, sigma2_smooth, A, Y, rho_old, cfg)
%M_STEP  Update AR(1) parameters (rho, sigma2_eps) from smoothed
%        estimates, then recompute lambda_t and Sigma_t.
%
%   lambda_t = lambda_base * sigmoid(z_smooth_t)
%   Sigma_t  = lambda_t * Sigma_{t-1} + (1 - lambda_t) * Y_t
%
%   Inputs
%       z_smooth, sigma2_smooth, A : outputs of backward_smoother
%       Y                          : [T x p x p] Wishart observations
%       rho_old                    : previous rho (fallback if denom ~ 0)
%       cfg                        : config struct
%
%   Outputs
%       rho_new       : scalar, clipped to cfg.em.rho_bounds
%       sigma2_eps_new: scalar, clipped to cfg.em.sigma2_eps_bounds
%       lambda_t      : [1 x T]
%       Sigma_t       : [T x p x p]

    T  = numel(z_smooth);
    p  = cfg.sim.p;
    lb = cfg.smoother.lambda_base;

    % -------- update sigma2_eps --------
    sum_s2 = 0;
    for t = 2:T
        term_a = z_smooth(t)^2 + sigma2_smooth(t);
        term_b = rho_old^2 * (z_smooth(t-1)^2 + sigma2_smooth(t-1));
        term_c = -2 * rho_old * (A(t-1) * sigma2_smooth(t) + z_smooth(t) * z_smooth(t-1));
        sum_s2 = sum_s2 + term_a + term_b + term_c;
    end
    s2_raw = sum_s2 / (T - 1);
    sigma2_eps_new = max(min(s2_raw, cfg.em.sigma2_eps_bounds(2)), cfg.em.sigma2_eps_bounds(1));

    % -------- update rho --------
    num = 0; den = 0;
    for t = 2:T
        num = num + z_smooth(t) * z_smooth(t-1) + A(t-1) * sigma2_smooth(t);
        den = den + z_smooth(t-1)^2 + sigma2_smooth(t-1);
    end
    if den > 1e-10
        r = num / den;
        rho_new = max(min(r, cfg.em.rho_bounds(2)), cfg.em.rho_bounds(1));
    else
        rho_new = rho_old;
    end

    % -------- recompute lambda_t and Sigma_t --------
    omega    = exp(z_smooth) ./ (1 + exp(z_smooth));
    lambda_t = lb * omega;

    Sigma_t = zeros(T, p, p);
    Sigma_t(1,:,:) = (1 - lambda_t(1)) * squeeze(Y(1,:,:));
    for t = 2:T
        Sigma_t(t,:,:) = lambda_t(t) * squeeze(Sigma_t(t-1,:,:)) ...
                       + (1 - lambda_t(t)) * squeeze(Y(t,:,:));
    end
end
