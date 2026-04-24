function [Y, X_true] = sample_wishart(C, cfg, lambda)
%SAMPLE_WISHART  Draw Wishart observations Y_t and compute ground-truth
%                precision X_t from a ground-truth covariance trajectory C_t.
%
%   [Y, X_true] = sim.sample_wishart(C, cfg)                 % uses constant lambda_base
%   [Y, X_true] = sim.sample_wishart(C, cfg, lambda)         % lambda scalar or Tx1
%
%   X_t = inv(C_t) / (k * (1 - lambda(t)))
%   Y_t ~ Wishart_p(k, C_t)

    T = size(C, 1);
    p = size(C, 2);
    k = cfg.sim.k;

    if nargin < 3, lambda = cfg.smoother.lambda_base; end
    if isscalar(lambda), lambda = lambda * ones(1, T); end

    if cfg.sim.seed > 0
        rng(cfg.sim.seed + 1);
    end

    Y      = zeros(T, p, p);
    X_true = zeros(T, p, p);
    for t = 1:T
        Ct = squeeze(C(t,:,:));
        X_true(t,:,:) = (Ct \ eye(p)) / (k * (1 - lambda(t)));
        Y(t,:,:)      = wishrnd(Ct, k);
    end
end
