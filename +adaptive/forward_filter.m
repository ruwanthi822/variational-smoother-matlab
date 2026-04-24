function [z_filt, z_pred, sigma2_filt, sigma2_pred] = ...
         forward_filter(Y, Sigma_em, rho, sigma2_eps, cfg)
%FORWARD_FILTER  Newton-Raphson forward filter for the latent AR(1) state z_t
%                driving the dynamic smoothing dial lambda_t.
%                Matches Master_updated.m line-for-line.
%
%   lambda_t = lambda_base * sigmoid(z_t)
%
%   Inside the NR loop we update ONLY z (gradient-scaled step); the Hessian
%   is computed ONCE after z_curr converges, and sigma2_filt = 1 / hess
%   with sign convention where hess > 0 at the posterior mode.
%
%   Inputs
%       Y          : [T x p x p]  Wishart observations
%       Sigma_em   : [T x p x p]  current EM iterate of Sigma_t
%       rho        : scalar
%       sigma2_eps : scalar
%       cfg        : config struct (uses cfg.sim.k, cfg.sim.p,
%                                       cfg.smoother.lambda_base, cfg.nr.*)
%
%   Outputs
%       z_filt, z_pred, sigma2_filt, sigma2_pred : [1 x T]

    T  = size(Y, 1);
    p  = cfg.sim.p;
    k  = cfg.sim.k;
    lb = cfg.smoother.lambda_base;
    n_param = 1 + p + k * lb / (1 - lb);   % harmonic-mean parameter (base lambda)

    z_filt       = zeros(1, T);
    z_pred       = zeros(1, T);
    sigma2_filt  = ones(1, T);
    sigma2_pred  = ones(1, T);

    z_filt(1)      = cfg.em.z_filt_init;
    sigma2_filt(1) = sigma2_eps / (1 - rho^2 + 1e-10);

    for t = 2:T
        % ---- one-step prediction ----
        z_pred(t)      = rho * z_filt(t-1);
        sigma2_pred(t) = rho^2 * sigma2_filt(t-1) + sigma2_eps;

        Sigma_tm1 = squeeze(Sigma_em(t-1,:,:));
        Y_t       = squeeze(Y(t,:,:));

        z_curr = z_pred(t);

        % ---- NR inner loop: update z only ----
        for nr_iter = 1:cfg.nr.max_iter
            exp_z       = exp(z_curr);
            omega_curr  = exp_z / (1 + exp_z);
            lambda_curr = lb * omega_curr;
            lambda_curr = max(min(lambda_curr, 1 - 1e-8), 1e-8);

            mixed_mat = lambda_curr * Sigma_tm1 + (1 - lambda_curr) * Y_t;
            mixed_inv = mixed_mat \ eye(p);
            diff_mat  = Sigma_tm1 - Y_t;

            grad_1 = (n_param / 2) * (1 - omega_curr);
            grad_2 = -((k - p + 1) / 2) * lambda_curr / (1 - lambda_curr) / (1 + exp_z);
            grad_3 = -((n_param + k) / 2) * lambda_curr * trace(mixed_inv * diff_mat) / (1 + exp_z);
            grad   = grad_1 + grad_2 + grad_3;

            delta_z = grad * sigma2_pred(t);
            delta_z = max(min(delta_z, cfg.nr.dz_clamp), -cfg.nr.dz_clamp);

            z_new = z_pred(t) + delta_z;

            if abs(z_new - z_curr) < cfg.nr.tol
                z_curr = z_new;
                break;
            end
            z_curr = z_new;
        end

        % ---- Hessian once, after z_curr is found ----
        exp_z       = exp(z_curr);
        omega_curr  = exp_z / (1 + exp_z);
        lambda_curr = lb * omega_curr;
        lambda_curr = max(min(lambda_curr, 1 - 1e-8), 1e-8);
        mixed_mat   = lambda_curr * Sigma_tm1 + (1 - lambda_curr) * Y_t;
        mixed_inv   = mixed_mat \ eye(p);
        diff_mat    = Sigma_tm1 - Y_t;

        hess_1 =  (n_param / 2) / (1 + exp_z)^2;
        hess_2 =  ((k - p + 1) / 2) * lambda_curr * (1 - lambda_curr - exp_z) ...
                 / (1 - lambda_curr)^2 / (1 + exp_z)^2;
        hess_3 =  ((n_param + k) / 2) * lambda_curr * (1 - exp_z) / (1 + exp_z)^2 ...
                 * trace(mixed_inv * diff_mat);
        hess_4 = -((n_param + k) / 2) * lambda_curr / (1 + exp_z)^2 ...
                 * trace(mixed_inv * diff_mat * mixed_inv * diff_mat);

        hess = hess_1 + hess_2 + hess_3 + hess_4 + 1 / sigma2_pred(t);

        if hess <= -1e-6
            hess = 1 / sigma2_pred(t) + cfg.nr.hess_damping;
        end

        z_filt(t)      = z_curr;
        sigma2_filt(t) = max(1 / hess, cfg.nr.var_floor);
    end
end
