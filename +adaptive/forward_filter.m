function [z_filt, z_pred, sigma2_filt, sigma2_pred] = ...
         forward_filter(Y, Sigma_em, rho, sigma2_eps, cfg)
%FORWARD_FILTER  Newton-Raphson forward filter for the latent AR(1) state z_t
%                driving the dynamic smoothing dial lambda_t.
%
%   lambda_t = cfg.smoother.lambda_base * sigmoid(z_t)
%
%   At each t we find the posterior mode z_{t|t} of
%       p(z_t | Y_{1:t})  ~  p(Y_t | z_t, Sigma_{t-1}) * N(z_t | rho * z_{t-1|t-1}, sigma2_pred(t))
%   via Newton-Raphson, and take the inverse curvature as sigma2_{t|t}.
%
%   Inputs
%       Y          : [T x p x p]  Wishart observations
%       Sigma_em   : [T x p x p]  current EM iterate of Sigma_t
%       rho        : scalar       AR(1) coefficient
%       sigma2_eps : scalar       AR(1) innovation variance
%       cfg        : config struct (uses cfg.sim.k, cfg.sim.p, cfg.smoother.lambda_base,
%                                       cfg.nr.*, cfg.plot.idx)
%
%   Outputs
%       z_filt       : [1 x T]
%       z_pred       : [1 x T]
%       sigma2_filt  : [1 x T]
%       sigma2_pred  : [1 x T]

    T  = size(Y, 1);
    p  = cfg.sim.p;
    k  = cfg.sim.k;
    lb = cfg.smoother.lambda_base;
    n_param = 1 + p + k * lb / (1 - lb);      % harmonic-mean parameter (using base lambda)

    z_filt       = zeros(1, T);
    z_pred       = zeros(1, T);
    sigma2_filt  = ones(1, T);
    sigma2_pred  = ones(1, T);

    z_filt(1)      = 0;
    sigma2_filt(1) = sigma2_eps / (1 - rho^2 + 1e-6);

    for t = 2:T
        % one-step prediction
        z_pred(t)      = rho * z_filt(t-1);
        sigma2_pred(t) = rho^2 * sigma2_filt(t-1) + sigma2_eps;

        Sigma_tm1 = squeeze(Sigma_em(t-1,:,:));
        Y_t       = squeeze(Y(t,:,:));

        % Newton-Raphson
        z_curr = z_pred(t);
        hess   = -1 / sigma2_pred(t) - cfg.nr.hess_damping;

        for it = 1:cfg.nr.max_iter
            exp_z   = exp(z_curr);
            omega   = exp_z / (1 + exp_z);
            lam     = lb * omega;
            lam     = max(min(lam, 1 - 1e-6), 1e-6);

            mixed = lam * Sigma_tm1 + (1 - lam) * Y_t;
            me    = min(eig(mixed));
            if me < cfg.nr.eig_floor
                mixed = mixed + (cfg.nr.eig_floor - me) * eye(p);
            end
            mixed_inv = mixed \ eye(p);
            diff_mat  = Sigma_tm1 - Y_t;

            d_omega = omega * (1 - omega);

            grad_1 = (n_param / 2) * (1 - omega);
            grad_2 = -((k - p + 1) / 2) * lb * d_omega / (1 - lam);
            grad_3 = -((n_param + k) / 2) * lb * d_omega * trace(mixed_inv * diff_mat);
            grad_4 = -(z_curr - z_pred(t)) / sigma2_pred(t);
            grad   = grad_1 + grad_2 + grad_3 + grad_4;

            hess_1 = (n_param / 2) * (-d_omega);
            hess_4 = -1 / sigma2_pred(t);
            hess   = hess_1 + hess_4 - cfg.nr.hess_damping;
            if hess >= 0
                hess = -1 / sigma2_pred(t) - cfg.nr.hess_damping;
            end

            delta_z = -grad / hess;
            delta_z = max(min(delta_z, cfg.nr.dz_clamp), -cfg.nr.dz_clamp);
            z_new   = z_curr + delta_z;
            z_new   = max(min(z_new, cfg.nr.z_clamp), -cfg.nr.z_clamp);

            if abs(z_new - z_curr) < cfg.nr.tol
                z_curr = z_new;
                break;
            end
            z_curr = z_new;
        end

        z_filt(t)      = z_curr;
        sigma2_filt(t) = max(-1 / hess, cfg.nr.var_floor);
    end
end
