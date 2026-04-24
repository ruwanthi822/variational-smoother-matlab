function [X_b, C_b, STD_pb, STD_cb, T_b] = backward_sampler(Sigma, lambda, cfg, T_f)
%BACKWARD_SAMPLER  L-sample Wishart backward Monte-Carlo smoother.
%
%   Draws S(l, T, :, :) ~ Wishart( inv(Sigma_T)/k, n_T + k ), then
%   recurses backward:
%       S(l, t, :, :) = lambda(t) * S(l, t+1, :, :)  +  Wishart( inv(Sigma_t)/k, k )
%
%   Returns L-sample mean precision X_b and covariance C_b with their
%   per-(i,j) sample standard deviations.
%
%   Inputs
%       Sigma  : [T x p x p]  forward-filtered state covariance
%       lambda : scalar or [T x 1] smoothing parameter
%       cfg    : config struct (uses cfg.sim.k, cfg.bsampler.L, cfg.plot.idx)
%       T_f    : elapsed time of the forward pass (cumulates into T_b)
%
%   Outputs
%       X_b    : [T x p x p]  smoothed precision (sample mean over L)
%       C_b    : [T x p x p]  smoothed covariance (sample mean over L)
%       STD_pb : [1 x T]      sample std of X_b(i,j)
%       STD_cb : [1 x T]      sample std of C_b(i,j)
%       T_b    : scalar       T_f + backward-sampler cputime

    if nargin < 4, T_f = 0; end

    T = size(Sigma, 1);
    p = size(Sigma, 2);
    k = cfg.sim.k;
    L = cfg.bsampler.L;
    i = cfg.plot.idx(1);
    j = cfg.plot.idx(2);

    if isscalar(lambda), lambda = lambda * ones(1, T); end
    lambda = lambda(:)';

    n = 1 + p + k .* lambda ./ (1 - lambda);

    t0 = cputime;

    % draw L backward trajectories
    S = zeros(L, T, p, p);
    for l = 1:L
        S(l, T, :, :) = wishrnd( (squeeze(Sigma(T,:,:)) \ eye(p)) / k, n(T) + k );
        for t = T-1:-1:1
            S(l, t, :, :) = lambda(t) * squeeze(S(l, t+1, :, :)) ...
                          + wishrnd( (squeeze(Sigma(t,:,:)) \ eye(p)) / k, k );
        end
    end

    % sample-mean precision and covariance
    X_b = zeros(T, p, p);
    C_b = zeros(T, p, p);
    for t = 1:T
        X_b(t,:,:) = squeeze(mean(S(:, t, :, :), 1));
        C_sum = zeros(p);
        for l = 1:L
            C_sum = C_sum + (squeeze(S(l,t,:,:)) \ eye(p)) / k / (1 - lambda(t));
        end
        C_b(t,:,:) = C_sum / L;
    end

    T_b = T_f + cputime - t0;

    % sample std of (i,j) entry
    STD_pb = zeros(1, T);
    STD_cb = zeros(1, T);
    for t = 1:T
        STD_pb(t) = std(S(:, t, i, j));
        s0 = zeros(L, 1);
        for l = 1:L
            S0 = (squeeze(S(l,t,:,:)) \ eye(p)) / k / (1 - lambda(t));
            s0(l) = S0(i, j);
        end
        STD_cb(t) = std(s0);
    end
end
