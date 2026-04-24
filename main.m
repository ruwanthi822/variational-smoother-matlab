%% MAIN  Adaptive Variational Covariance Smoother — end-to-end pipeline
%
%   1. Build ground-truth covariance trajectory C_t and draw Wishart obs Y_t.
%   2. Constant-lambda forward filter          -> Sigma_init, X_f, C_f.
%   3. L-sample backward Monte-Carlo smoother  -> X_b, C_b (with constant lambda).
%   4. Adaptive-lambda EM over z_t             -> lambda_t, Sigma_t.
%   5. VI first-order backward pass            -> X_vi1, C_vi1.
%   6. VI second-order backward pass           -> X_vi2, C_vi2.
%   7. Compare methods by MSE (total and off-diagonal, in dB).
%   8. Plot the (i,j) entry of every estimate against ground truth.
%
%   Run this file in MATLAB from the repository root:
%       >> main

clear; clc; close all;

%% ---------- 1. config and simulated data ----------
cfg = config.default();
fprintf('== Building ground-truth covariance (type=%s) ==\n', cfg.sim.truth_type);
switch lower(cfg.sim.truth_type)
    case 'step',   C_true = sim.step_truth(cfg);
    case 'linear', C_true = sim.linear_truth(cfg);
    otherwise,     error('Unknown truth_type: %s', cfg.sim.truth_type);
end

[Y, X_true] = sim.sample_wishart(C_true, cfg);
T = cfg.sim.T;
t_axis = 1:T;

%% ---------- 2. forward filter (constant lambda) ----------
fprintf('== Forward filter (constant lambda=%.2f) ==\n', cfg.smoother.lambda_base);
lambda_const = cfg.smoother.lambda_base * ones(1, T);
[Sigma_f, X_f, C_f, STD_pf, STD_cf, T_f] = ...
    estimators.forward_filter(Y, lambda_const, cfg);

%% ---------- 3. backward sampler ----------
fprintf('== Backward sampler (L=%d) ==\n', cfg.bsampler.L);
[X_b, C_b, STD_pb, STD_cb, T_b] = ...
    estimators.backward_sampler(Sigma_f, lambda_const, cfg, T_f);

%% ---------- 4. adaptive-lambda EM ----------
fprintf('== Adaptive-lambda EM ==\n');
[Sigma_em, lambda_t, z_smooth, rho, sigma2_eps, em_hist] = ...
    adaptive.run_em(Y, Sigma_f, cfg);

%% ---------- 5 & 6. first and second order VI (adaptive lambda) ----------
fprintf('== VI first-order backward pass ==\n');
[V0, V1, X_vi1, C_vi1, STD_pvi, STD_cvi, T_vi] = ...
    estimators.vi_first_order(Sigma_em, lambda_t, cfg, T_f);

fprintf('== VI second-order backward pass ==\n');
[~, ~, V2, X_vi2, C_vi2, STD_pvi2, STD_cvi2, T_vi2] = ...
    estimators.vi_second_order(Sigma_em, lambda_t, cfg, T_f);

%% ---------- 7. MSE comparison ----------
fprintf('\n== MSE in dB (lower is better) ==\n');
methods = {
    'forward filter',    X_f,   C_f;
    'backward sampler',  X_b,   C_b;
    'VI first order',    X_vi1, C_vi1;
    'VI second order',   X_vi2, C_vi2;
};

mse_table_prec = struct([]);
mse_table_cov  = struct([]);
for m = 1:size(methods, 1)
    name = methods{m, 1};
    Xe   = methods{m, 2};
    Ce   = methods{m, 3};

    [mse_p_tot, mse_p_off] = metrics.mse_db(X_true, Xe);
    [mse_c_tot, mse_c_off] = metrics.mse_db(C_true, Ce);

    fprintf('%-18s   X: total=%7.2f dB  off=%7.2f dB    C: total=%7.2f dB  off=%7.2f dB\n', ...
            name, mse_p_tot, mse_p_off, mse_c_tot, mse_c_off);

    mse_table_prec(end+1).name    = name; %#ok<*SAGROW>
    mse_table_prec(end  ).mse_tot = mse_p_tot;
    mse_table_prec(end  ).mse_off = mse_p_off;

    mse_table_cov (end+1).name    = name;
    mse_table_cov (end  ).mse_tot = mse_c_tot;
    mse_table_cov (end  ).mse_off = mse_c_off;
end

%% ---------- 8. plots ----------
prec_estimates(1) = struct('name','forward',   'X', X_f,   'STD', STD_pf);
prec_estimates(2) = struct('name','backward',  'X', X_b,   'STD', STD_pb);
prec_estimates(3) = struct('name','VI 1st',    'X', X_vi1, 'STD', STD_pvi);
prec_estimates(4) = struct('name','VI 2nd',    'X', X_vi2, 'STD', STD_pvi2);

cov_estimates(1)  = struct('name','forward',   'X', C_f,   'STD', STD_cf);
cov_estimates(2)  = struct('name','backward',  'X', C_b,   'STD', STD_cb);
cov_estimates(3)  = struct('name','VI 1st',    'X', C_vi1, 'STD', STD_cvi);
cov_estimates(4)  = struct('name','VI 2nd',    'X', C_vi2, 'STD', STD_cvi2);

plotting.entry_with_ci(t_axis, X_true, prec_estimates, cfg, 'precision');
plotting.entry_with_ci(t_axis, C_true, cov_estimates,  cfg, 'covariance');
plotting.lambda_trajectory(lambda_t, cfg);
plotting.mse_summary(mse_table_prec);   title('MSE — precision');
plotting.mse_summary(mse_table_cov);    title('MSE — covariance');

%% ---------- save a summary .mat for later comparison ----------
ts = datestr(now, 'yyyymmdd_HHMMSS');
save(sprintf('results_%s.mat', ts), ...
     'cfg', 'C_true', 'X_true', 'Y', ...
     'Sigma_f', 'X_f', 'C_f', ...
     'X_b', 'C_b', ...
     'Sigma_em', 'lambda_t', 'z_smooth', 'rho', 'sigma2_eps', 'em_hist', ...
     'V0', 'V1', 'V2', ...
     'X_vi1', 'C_vi1', 'X_vi2', 'C_vi2', ...
     'mse_table_prec', 'mse_table_cov');

fprintf('\nResults saved to results_%s.mat\n', ts);
