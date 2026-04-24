function [V0, V1, V2, X_vi2, C_vi2, STD_pvi2, STD_cvi2, T_vi2] = ...
         vi_second_order(Sigma, lambda, cfg, T_f)
%VI_SECOND_ORDER  Second-order variational-inference backward pass.
%
%   Implements Eqs. 21, 22, 35, 46 for V^(1), V^(2) and the resulting
%   second-order precision/covariance estimates
%       X_vi2 = m * (V^(0) + lambda * V^(1) + lambda^2 * V^(2))
%       C_vi2 = inv(X_vi2) * m / (m - p - 1) / k
%
%   Inputs
%       Sigma, lambda, cfg, T_f  (same meaning as in vi_first_order)
%
%   Outputs
%       V0, V1, V2 : [T x p x p]
%       X_vi2      : [T x p x p]
%       C_vi2      : [T x p x p]
%       STD_pvi2   : [1 x T]
%       STD_cvi2   : [1 x T]
%       T_vi2      : scalar

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

    [a_m, b_m] = estimators.compute_am_bm(m, p);
    c1 = (k - p - 1) / (k * (m - p - 1));
    c2 = (k - p - 1) / (k * m * (m - p - 1));
    c3 = (k - p - 1) / k;

    omega = lambda / cfg.smoother.lambda_base;   % omega_t (sigmoid component)

    t0 = cputime;

    % V0
    V0 = zeros(T, p, p);
    for t = 1:T
        V0(t,:,:) = (squeeze(Sigma(t,:,:)) \ eye(p)) / m;
    end

    V1   = V0;
    V2   = zeros(T, p, p);
    V_full = zeros(T, p, p);

    % terminal
    Sigma_T = squeeze(Sigma(T,:,:));
    V_full(T,:,:) = (n(T) + k) / (k * m) * (Sigma_T \ eye(p));
    V1(T,:,:)     = squeeze(V_full(T,:,:));

    for iter = 1:cfg.vi.VI_iter
        for t = T-1:-1:2

            Sigma_tm1   = squeeze(Sigma(t-1,:,:));
            Sigma_t     = squeeze(Sigma(t,:,:));
            Sigma_t_inv = Sigma_t \ eye(p);

            V_tp1 = squeeze(V_full(t+1,:,:));
            V_tm1 = squeeze(V0(t-1,:,:));
            V_tm1_inv = V_tm1 \ eye(p);

            om_t   = omega(t);
            om_tm1 = omega(t-1);

            % ---- Eq. 21  V^(1) ----
            term1_V1 = om_t * m * c1 * V_tp1;
            term2_V1 = -om_tm1 * c2 * (Sigma_t_inv * V_tm1_inv * Sigma_t_inv);
            term3_V1 = ((2 - om_t) / m) * (Sigma_t_inv * Sigma_tm1 * Sigma_t_inv);

            V1_t = term1_V1 + term2_V1 + term3_V1;
            V1_t = (V1_t + V1_t') / 2;
            V1(t,:,:) = V1_t;

            % ---- Eq. 22  V^(2) ----
            term1_V2 = om_t * V1_t * Sigma_tm1 * Sigma_t_inv;
            term2_V2 = om_t * Sigma_t_inv * Sigma_tm1 * V1_t;
            term3_V2 = -m * V1_t * Sigma_t * V1_t;
            term4_V2 = -om_tm1 * c1 * ( Sigma_t_inv * V_tm1_inv * V1_t ...
                                      + V1_t * V_tm1_inv * Sigma_t_inv );

            V_tp1_Sigma_t = V_tp1 * Sigma_t;
            tr_V_Sigma = trace(V_tp1_Sigma_t);
            part_A = m * om_t * (a_m * V_tp1_Sigma_t * V_tp1 + b_m * tr_V_Sigma * V_tp1);

            St_Vtm1_St = Sigma_t_inv * V_tm1_inv * Sigma_t_inv;
            tr_Vtm1_St = trace(V_tm1_inv * Sigma_t_inv);
            part_B = (om_tm1 / m^3) * ( a_m * St_Vtm1_St * V_tm1_inv * Sigma_t_inv ...
                                      + b_m * tr_Vtm1_St * St_Vtm1_St );
            term5_V2 = c3 * (part_A - part_B);

            V2_t = term1_V2 + term2_V2 + term3_V2 + term4_V2 + term5_V2;
            V2_t = (V2_t + V2_t') / 2;
            V2(t,:,:) = V2_t;

            % combine
            V_full(t,:,:) = squeeze(V0(t,:,:)) + lambda(t) * V1_t + lambda(t)^2 * V2_t;
        end
    end

    % boundary
    V1(1,:,:)     = squeeze(V0(1,:,:));
    V2(1,:,:)     = zeros(p);
    V_full(1,:,:) = squeeze(V0(1,:,:));

    % estimates
    X_vi2 = zeros(T, p, p);
    C_vi2 = zeros(T, p, p);
    for t = 1:T
        V_2nd = squeeze(V0(t,:,:)) + lambda(t) * squeeze(V1(t,:,:)) ...
                + lambda(t)^2 * squeeze(V2(t,:,:));
        X_vi2(t,:,:) = m * V_2nd;
        C_vi2(t,:,:) = (squeeze(X_vi2(t,:,:)) \ eye(p)) * m / (m - p - 1) / k;
    end

    T_vi2 = T_f + cputime - t0;

    % STD for the (i,j) entry
    STD_pvi2 = zeros(1, T);
    STD_cvi2 = zeros(1, T);
    nu = m;
    for t = 1:T
        V_t = squeeze(V0(t,:,:)) + lambda(t) * squeeze(V1(t,:,:)) ...
              + lambda(t)^2 * squeeze(V2(t,:,:));
        STD_pvi2(t) = sqrt( m * (V_t(i,j)^2 + V_t(i,i) * V_t(j,j)) );

        Xinv = squeeze(X_vi2(t,:,:)) \ eye(p);
        if nu > p + 3
            cov_var = ((nu - p + 1) * Xinv(i,j)^2 + (nu - p - 1) * Xinv(i,i) * Xinv(j,j)) ...
                      / (nu - p) / (nu - p - 1)^2 / (nu - p - 3);
            STD_cvi2(t) = m * sqrt(cov_var) / k;
        end
    end
end
