function [Sigma, X_f, C_f, STD_pf, STD_cf, T_f] = forward_filter(Y, lambda, cfg)
%FORWARD_FILTER  One-pass forward recursion for the state covariance Sigma_t
%                given Wishart observations Y_t and a (possibly time-varying)
%                smoothing parameter lambda_t.
%
%   Sigma_t = lambda(t) * Sigma_{t-1} + (1 - lambda(t)) * Y_t
%
%   Inputs
%       Y      : [T x p x p]  Wishart observations
%       lambda : scalar or [1 x T] or [T x 1] vector
%       cfg    : config struct (uses cfg.sim.k)
%
%   Outputs
%       Sigma  : [T x p x p]   filtered state covariance
%       X_f    : [T x p x p]   filtered precision   ((n_t + k)/k) * inv(Sigma)
%       C_f    : [T x p x p]   filtered covariance  k/(n+k-p-1) * Sigma / (k*(1-lambda))
%       STD_pf : [1 x T]       std of X_f at (i,j) entry
%       STD_cf : [1 x T]       std of C_f at (i,j) entry
%       T_f    : scalar        wall-clock cputime for the forward pass

    T = size(Y, 1);
    p = size(Y, 2);
    k = cfg.sim.k;
    i = cfg.plot.idx(1);
    j = cfg.plot.idx(2);

    if isscalar(lambda), lambda = lambda * ones(1, T); end
    lambda = lambda(:)';

    n = 1 + p + k .* lambda ./ (1 - lambda);   % harmonic-mean parameter

    t0 = cputime;

    % forward recursion
    Sigma = zeros(T, p, p);
    Sigma(1,:,:) = (1 - lambda(1)) * squeeze(Y(1,:,:));
    for t = 2:T
        Sigma(t,:,:) = lambda(t) * squeeze(Sigma(t-1,:,:)) ...
                     + (1 - lambda(t)) * squeeze(Y(t,:,:));
    end

    % filtered precision and covariance
    X_f = zeros(T, p, p);
    C_f = zeros(T, p, p);
    for t = 1:T
        Sigma_t  = squeeze(Sigma(t,:,:));
        inv_Sig  = Sigma_t \ eye(p);
        X_f(t,:,:) = ((n(t) + k) / k) * inv_Sig;
        C_f(t,:,:) = (k / (n(t) + k - p - 1)) * Sigma_t / (k * (1 - lambda(t)));
    end

    T_f = cputime - t0;

    % STD of (i,j) entry
    STD_pf = zeros(1, T);
    STD_cf = zeros(1, T);
    nu = n + k;
    for t = 1:T
        S0 = squeeze(Sigma(t,:,:)) \ eye(p);
        STD_pf(t) = sqrt((n(t) + k) / k / k) * ...
                    sqrt(S0(i,j)^2 + S0(i,i) * S0(j,j));

        S0c = k * squeeze(Sigma(t,:,:));
        if nu(t) > p + 3
            var_cf = ((nu(t) - p + 1) * S0c(i,j)^2 + (nu(t) - p - 1) * S0c(i,i) * S0c(j,j)) ...
                     / (nu(t) - p) / (nu(t) - p - 1)^2 / (nu(t) - p - 3);
            STD_cf(t) = sqrt(var_cf) / (1 - lambda(t)) / k;
        end
    end
end
