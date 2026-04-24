function [V0, V1, X_vi1, C_vi1, STD_pvi, STD_cvi, T_vi] = ...
         vi_first_order(Sigma, lambda, cfg, T_f)
%VI_FIRST_ORDER  First-order variational-inference backward pass.
%
%   Matches the paper-producing block in Master_updated.m.
%
%   Backward recursion for V^(1)_t (t = T-1 down to 2):
%       V1_t = omega_t * m * c1 * V_{t+1}
%              - omega_{t-1} * c2 * Sigma_t^{-1} * V_{t-1}^{-1} * Sigma_t^{-1}
%              + ((2 - omega_t) / m) * Sigma_t^{-1} * Sigma_{t-1} * Sigma_t^{-1}
%
%   Estimates:
%       X_vi1 = m * (V^(0) + lambda_t * V^(1))
%       C_vi1 = inv(X_vi1) / (k * (1 - lambda_t))
%
%   Inputs
%       Sigma  : [T x p x p]
%       lambda : scalar or [1 x T] smoothing dial
%       cfg    : config struct (uses cfg.sim.k, cfg.sim.p, cfg.vi.m_multiplier,
%                               cfg.smoother.lambda_base, cfg.vi.VI_iter, cfg.plot.idx)
%       T_f    : elapsed time of the forward pass (cumulates into T_vi)
%
%   Outputs
%       V0, V1   : [T x p x p]
%       X_vi1    : [T x p x p]
%       C_vi1    : [T x p x p]
%       STD_pvi  : [1 x T]
%       STD_cvi  : [1 x T]
%       T_vi     : scalar

    if nargin < 4, T_f = 0; end

    T = size(Sigma, 1);
    p = size(Sigma, 2);
    k = cfg.sim.k;
    m = cfg.vi.m_multiplier * k;
    lb = cfg.smoother.lambda_base;
    i  = cfg.plot.idx(1);
    j  = cfg.plot.idx(2);

    if isscalar(lambda), lambda = lambda * ones(1, T); end
    lambda = lambda(:)';
    n      = 1 + p + k .* lambda ./ (1 - lambda);   % harmonic-mean param

    c1 = (k - p - 1) / (k * (m - p - 1));
    c2 = (k - p - 1) / (k * m * (m - p - 1));

    omega = lambda / lb;

    t0 = cputime;

    % V0_t = (1/m) * Sigma_t^{-1}
    V0 = zeros(T, p, p);
    for t = 1:T
        V0(t,:,:) = (squeeze(Sigma(t,:,:)) \ eye(p)) / m;
    end

    % V1 and full V_dyn; initialize V1 at V0
    V1    = V0;
    V_dyn = zeros(T, p, p);
    Sigma_T = squeeze(Sigma(T,:,:));
    V_dyn(T,:,:) = (n(T) + k) / (k * m) * (Sigma_T \ eye(p));
    V1(T,:,:)    = squeeze(V_dyn(T,:,:));

    % backward sweep, repeated VI_iter times
    for iter = 1:cfg.vi.VI_iter
        for t = T-1:-1:2
            Sigma_tm1   = squeeze(Sigma(t-1,:,:));
            Sigma_t     = squeeze(Sigma(t,:,:));
            Sigma_t_inv = Sigma_t \ eye(p);

            V_tp1     = squeeze(V_dyn(t+1,:,:));
            V_tm1     = squeeze(V0(t-1,:,:));
            V_tm1_inv = V_tm1 \ eye(p);

            om_t   = omega(t);
            om_tm1 = omega(t-1);

            term1 =  om_t   * m * c1 * V_tp1;
            term2 = -om_tm1 * c2 * (Sigma_t_inv * V_tm1_inv * Sigma_t_inv);
            term3 = ((2 - om_t) / m) * (Sigma_t_inv * Sigma_tm1 * Sigma_t_inv);

            V1_t = term1 + term2 + term3;
            V1_t = (V1_t + V1_t') / 2;
            V1(t,:,:) = V1_t;

            % full V_dyn (used by the next backward step's V_tp1)
            V_dyn(t,:,:) = squeeze(V0(t,:,:)) + lambda(t) * V1_t;
        end
    end

    % boundary
    V1(1,:,:)    = squeeze(V0(1,:,:));
    V_dyn(1,:,:) = squeeze(V0(1,:,:));

    % ---------- estimates ----------
    % Symmetrize + scale-relative eigenvalue-lift, plus a fallback to V0
    % if V1 ever becomes non-finite. Keeps X_vi1 safely invertible without
    % changing the estimate when it's already positive definite.
    abs_floor  = 1e-8;
    rel_floor  = 1e-8;
    n_fallback = 0;

    X_vi1    = zeros(T, p, p);
    C_vi1    = zeros(T, p, p);
    Xinv_all = zeros(T, p, p);

    for t = 1:T
        V_1st = squeeze(V0(t,:,:)) + lambda(t) * squeeze(V1(t,:,:));
        X_t   = m * V_1st;

        if ~all(isfinite(X_t), 'all')
            X_t = m * squeeze(V0(t,:,:));
            n_fallback = n_fallback + 1;
        end

        X_t = (X_t + X_t') / 2;
        floor_t = max(abs_floor, rel_floor * norm(X_t, 'fro'));
        me = min(eig(X_t));
        if me < floor_t
            X_t = X_t + (floor_t - me) * eye(p);
        end
        X_vi1(t,:,:) = X_t;

        Xinv = X_t \ eye(p);
        Xinv_all(t,:,:) = Xinv;
        C_vi1(t,:,:) = Xinv / (k * (1 - lambda(t)));
    end

    if n_fallback > 0
        warning('vi_first_order:fallback', ...
                'Fell back to zeroth-order precision at %d of %d time points (non-finite V1).', ...
                n_fallback, T);
    end

    T_vi = T_f + cputime - t0;

    % ---------- STD of the (i,j) entry ----------
    STD_pvi = zeros(1, T);
    STD_cvi = zeros(1, T);
    nu = m;
    for t = 1:T
        V_t = squeeze(V0(t,:,:)) + lambda(t) * squeeze(V1(t,:,:));
        STD_pvi(t) = sqrt( m * (V_t(i,j)^2 + V_t(i,i) * V_t(j,j)) );

        Xinv = squeeze(Xinv_all(t,:,:));
        scale = 1 / (k * (1 - lambda(t)));
        if nu > p + 3
            cov_var = ((nu - p + 1) * Xinv(i,j)^2 + (nu - p - 1) * Xinv(i,i) * Xinv(j,j)) ...
                      / (nu - p) / (nu - p - 1)^2 / (nu - p - 3);
            STD_cvi(t) = scale * sqrt(cov_var);
        end
    end
end
