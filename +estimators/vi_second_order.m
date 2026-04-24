function [V0, V1, V2, X_vi2, C_vi2, STD_pvi2, STD_cvi2, T_vi2] = ...
         vi_second_order(Sigma, lambda, cfg, T_f)
%VI_SECOND_ORDER  Second-order variational-inference backward pass.
%
%   Matches the paper-producing block in Master_updated.m.
%
%   Returns V0, V1, V2 and the second-order precision/covariance
%       X_vi2 = m * (V^(0) + lambda_t * V^(1) + lambda_t^2 * V^(2))
%       C_vi2 = inv(X_vi2) / (k * (1 - lambda_t))

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
    n      = 1 + p + k .* lambda ./ (1 - lambda);

    [a_m, b_m] = estimators.compute_am_bm(m, p);
    c1 = (k - p - 1) / (k * (m - p - 1));
    c2 = (k - p - 1) / (k * m * (m - p - 1));
    c3 = (k - p - 1) / k;

    omega = lambda / lb;

    t0 = cputime;

    V0    = zeros(T, p, p);
    V1    = zeros(T, p, p);
    V2    = zeros(T, p, p);
    V_dyn = zeros(T, p, p);

    for t = 1:T
        V0(t,:,:) = (squeeze(Sigma(t,:,:)) \ eye(p)) / m;
    end

    Sigma_T = squeeze(Sigma(T,:,:));
    V_dyn(T,:,:) = (n(T) + k) / (k * m) * (Sigma_T \ eye(p));
    V1(T,:,:)    = squeeze(V_dyn(T,:,:));
    V2(T,:,:)    = zeros(p);

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

            % ----- V^(1) update -----
            term1_V1 =  om_t   * m * c1 * V_tp1;
            term2_V1 = -om_tm1 * c2 * (Sigma_t_inv * V_tm1_inv * Sigma_t_inv);
            term3_V1 = ((2 - om_t) / m) * (Sigma_t_inv * Sigma_tm1 * Sigma_t_inv);

            V1_t = term1_V1 + term2_V1 + term3_V1;
            V1_t = (V1_t + V1_t') / 2;
            V1(t,:,:) = V1_t;

            % ----- V^(2) update -----
            term1_V2 =  om_t   * V1_t * Sigma_tm1 * Sigma_t_inv;
            term2_V2 =  om_t   * Sigma_t_inv * Sigma_tm1 * V1_t;
            term3_V2 = -m      * V1_t * Sigma_t * V1_t;
            term4_V2 = -om_tm1 * c1 * (Sigma_t_inv * V_tm1_inv * V1_t ...
                                      + V1_t * V_tm1_inv * Sigma_t_inv);

            V_tp1_Sigma_t = V_tp1 * Sigma_t;
            tr_V_Sigma    = trace(V_tp1_Sigma_t);
            part_A = m * om_t * (a_m * V_tp1_Sigma_t * V_tp1 + b_m * tr_V_Sigma * V_tp1);

            St_Vtm1_St = Sigma_t_inv * V_tm1_inv * Sigma_t_inv;
            tr_Vtm1_St = trace(V_tm1_inv * Sigma_t_inv);
            part_B = (om_tm1 / m^3) * ( a_m * St_Vtm1_St * V_tm1_inv * Sigma_t_inv ...
                                      + b_m * tr_Vtm1_St * St_Vtm1_St );
            term5_V2 = c3 * (part_A - part_B);

            V2_t = term1_V2 + term2_V2 + term3_V2 + term4_V2 + term5_V2;
            V2_t = (V2_t + V2_t') / 2;
            V2(t,:,:) = V2_t;

            V_dyn(t,:,:) = squeeze(V0(t,:,:)) + lambda(t) * V1_t + lambda(t)^2 * V2_t;
        end
    end

    % boundary
    V1(1,:,:)    = squeeze(V0(1,:,:));
    V2(1,:,:)    = zeros(p);
    V_dyn(1,:,:) = squeeze(V0(1,:,:));

    % ---------- estimates ----------
    %
    % The second-order correction V2 can push X_vi2 into non-finite /
    % singular territory at some time points (triple products and /m^3
    % terms in the backward pass overflow when V1 or Sigma^{-1} is large).
    % We harden the inversion with three layers:
    %   1) if X_t has any NaN/Inf entries, fall back to the zeroth-order
    %      precision m*V0(t) (well-defined by construction)
    %   2) symmetrize
    %   3) lift eigenvalues to max(rel_floor * ||X_t||_F, abs_floor)
    %
    % n_fallback counts how many time points needed the fallback; printed
    % once as a diagnostic.
    abs_floor   = 1e-8;
    rel_floor   = 1e-8;
    n_fallback  = 0;

    X_vi2    = zeros(T, p, p);
    C_vi2    = zeros(T, p, p);
    Xinv_all = zeros(T, p, p);

    for t = 1:T
        V_2nd = squeeze(V0(t,:,:)) + lambda(t) * squeeze(V1(t,:,:)) ...
                + lambda(t)^2 * squeeze(V2(t,:,:));
        X_t = m * V_2nd;

        % fallback if X_t has any non-finite entries
        if ~all(isfinite(X_t), 'all')
            X_t = m * squeeze(V0(t,:,:));         % zeroth-order only
            n_fallback = n_fallback + 1;
        end

        % symmetrize + eigenvalue lift (scale-relative + absolute)
        X_t = (X_t + X_t') / 2;
        floor_t = max(abs_floor, rel_floor * norm(X_t, 'fro'));
        me = min(eig(X_t));
        if me < floor_t
            X_t = X_t + (floor_t - me) * eye(p);
        end
        X_vi2(t,:,:) = X_t;

        Xinv = X_t \ eye(p);
        Xinv_all(t,:,:) = Xinv;
        C_vi2(t,:,:) = Xinv / (k * (1 - lambda(t)));
    end

    if n_fallback > 0
        warning('vi_second_order:fallback', ...
                'Fell back to zeroth-order precision at %d of %d time points (non-finite X_vi2). Consider reducing VI_iter or using first-order.', ...
                n_fallback, T);
    end

    T_vi2 = T_f + cputime - t0;

    % ---------- STD of the (i,j) entry ----------
    STD_pvi2 = zeros(1, T);
    STD_cvi2 = zeros(1, T);
    nu = m;
    for t = 1:T
        V_t = squeeze(V0(t,:,:)) + lambda(t) * squeeze(V1(t,:,:)) ...
              + lambda(t)^2 * squeeze(V2(t,:,:));
        if ~all(isfinite(V_t), 'all')
            V_t = squeeze(V0(t,:,:));       % fallback
        end
        STD_pvi2(t) = sqrt( max(m * (V_t(i,j)^2 + V_t(i,i) * V_t(j,j)), 0) );

        Xinv = squeeze(Xinv_all(t,:,:));
        scale = 1 / (k * (1 - lambda(t)));
        if nu > p + 3
            cov_var = ((nu - p + 1) * Xinv(i,j)^2 + (nu - p - 1) * Xinv(i,i) * Xinv(j,j)) ...
                      / (nu - p) / (nu - p - 1)^2 / (nu - p - 3);
            STD_cvi2(t) = scale * sqrt(max(cov_var, 0));
        end
    end
end
