function [z_smooth, sigma2_smooth, A] = ...
         backward_smoother(z_filt, z_pred, sigma2_filt, sigma2_pred, rho, cfg)
%BACKWARD_SMOOTHER  RTS (Rauch-Tung-Striebel) backward pass for the latent AR(1)
%                   state z_t once the forward filter has run.
%
%   Also returns the lag-one smoothing gains A(t) used by the M-step.
%
%   Inputs
%       z_filt, z_pred           : [1 x T]  filter mean / prediction mean
%       sigma2_filt, sigma2_pred : [1 x T]  filter / prediction variances
%       rho                      : scalar   AR(1) coefficient
%       cfg                      : config struct (uses cfg.nr.var_floor)
%
%   Outputs
%       z_smooth      : [1 x T]
%       sigma2_smooth : [1 x T]
%       A             : [1 x T]  RTS gains (A(t) = rho * sigma2_filt(t)/sigma2_pred(t+1))

    T = numel(z_filt);
    z_smooth      = zeros(1, T);
    sigma2_smooth = zeros(1, T);
    A             = zeros(1, T);

    z_smooth(T)      = z_filt(T);
    sigma2_smooth(T) = sigma2_filt(T);

    for t = T-1:-1:1
        if sigma2_pred(t+1) > 1e-10
            A(t) = rho * sigma2_filt(t) / sigma2_pred(t+1);
        end
        z_smooth(t)      = z_filt(t) + A(t) * (z_smooth(t+1) - z_pred(t+1));
        sigma2_smooth(t) = sigma2_filt(t) ...
                         + A(t)^2 * (sigma2_smooth(t+1) - sigma2_pred(t+1));
        sigma2_smooth(t) = max(sigma2_smooth(t), cfg.nr.var_floor);
    end
end
