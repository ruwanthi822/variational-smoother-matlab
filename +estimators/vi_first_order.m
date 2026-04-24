function [V0, V1, X_vi1, C_vi1, STD_pvi, STD_cvi, T_vi] = ...
         vi_first_order(Sigma, lambda, cfg, T_f)
%VI_FIRST_ORDER  First-order variational-inference backward pass.
%
%   Implements Eq. 21 backward recursion for V^(1)_t, then returns
%   estimates  X_vi1 = m * (V^(0) + lambda * V^(1))
%              C_vi1 = inv(X_vi1) * m / (m - p - 1) / k
%
%   Inputs
%       Sigma  : [T x p x p]
%       lambda : scalar or [1 x T] smoothing dial
%       cfg    : config struct
%       T_f    : elapsed time of forward pass (cumulates into T_vi)
%
%   Outputs
%       V0, V1  : [T x p x p]  VI building blocks
%       X_vi1   : [T x p x p]  first-order precision estimate
%       C_vi1   : [T x p x p]  first-order covariance estimate
%       STD_pvi : [1 x T]      std of X_vi1(i,j)
%       STD_cvi : [1 x T]      std of C_vi1(i,j)
%       T_vi    : scalar       T_f + VI cputime

    if nargin < 4, T_f = 0; end

    T = size(Sigma, 1);
    p = size(Sigma, 2);
    k = cfg.sim.k;
    m = cfg.vi.m_multiplier * k;
    i = cfg.plot.idx(1);
    j = cfg.plot.idx(2);

    if isscalar(lambda), lambda = lambda * ones(1, T); end
    lambda = lambda(:)';

    n = 1 + p + k .* lambda ./ (1 - lambda);
    c_m = (k - p - 1) / (m - p - 1);

    t0 = cputime;

    % V0: zeroth-order, V0_t = (1/m) * Sigma_t^{-1}
    V0 = zeros(T, p, p);
    for t = 1:T
        V0(t,:,:) = (squeeze(Sigma(t,:,:)) \ eye(p)) / m;
    end

    % V1: initialize as V0; fix terminal condition at T
    V1 = V0;
    V1(T,:,:) = (squeeze(Sigma(T,:,:)) \ eye(p)) * (n(T) + k) / k / m;

    % backward sweep, repeated VI_iter times
    for iter = 1:cfg.vi.VI_iter
        for t = T-1:-1:2
            St      = squeeze(Sigma(t,:,:));
            St_inv  = St \ eye(p);
            St_prev = squeeze(Sigma(t-1,:,:));
            Vprev   = squeeze(V0(t-1,:,:) + lambda(t) * V1(t-1,:,:));
            Vprev_inv = Vprev \ eye(p);
            Vnext   = squeeze(V0(t+1,:,:) + lambda(t) * V1(t+1,:,:));

            term1 = (m * c_m / k) * Vnext;
            term2 = -(c_m / k / m) * (St_inv * Vprev_inv * St_inv);
            term3 = (St_inv * St_prev * St_inv) / m;

            V1_t = term1 + term2 + term3;
            V1_t = (V1_t + V1_t') / 2;      % symmetrize
            V1(t,:,:) = V1_t;
        end
    end

    % estimates
    X_vi1 = zeros(T, p, p);
    C_vi1 = zeros(T, p, p);
    for t = 1:T
        X_vi1(t,:,:) = m * squeeze(V0(t,:,:)) + m * lambda(t) * squeeze(V1(t,:,:));
        C_vi1(t,:,:) = (squeeze(X_vi1(t,:,:)) \ eye(p)) * m / (m - p - 1) / k;
    end

    T_vi = T_f + cputime - t0;

    % STD for the (i,j) entry
    STD_pvi = zeros(1, T);
    STD_cvi = zeros(1, T);
    nu = m;
    for t = 1:T
        S0 = squeeze(V0(t,:,:)) + lambda(t) * squeeze(V1(t,:,:));
        STD_pvi(t) = sqrt( m * (S0(i,j)^2 + S0(i,i) * S0(j,j)) );

        Xinv = squeeze(X_vi1(t,:,:)) \ eye(p);
        if nu > p + 3
            cov_var = ((nu - p + 1) * Xinv(i,j)^2 + (nu - p - 1) * Xinv(i,i) * Xinv(j,j)) ...
                      / (nu - p) / (nu - p - 1)^2 / (nu - p - 3);
            STD_cvi(t) = m * sqrt(cov_var) / k;
        end
    end
end
