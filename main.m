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
    case 'step',     C_true = sim.step_truth(cfg);
    case 'linear',   C_true = sim.linear_truth(cfg);
    case 'load_mat', C_true = sim.load_mat(cfg);
    otherwise,       error('Unknown truth_type: %s', cfg.sim.truth_type);
end

% sync T and p with whatever the ground truth actually is
cfg.sim.T = size(C_true, 1);
cfg.sim.p = size(C_true, 2);

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
t_em_start = cputime;
[Sigma_em, lambda_t, z_smooth, rho, sigma2_eps, em_hist] = ...
    adaptive.run_em(Y, Sigma_f, cfg);
T_em = cputime - t_em_start;

%% ---------- 5 & 6. first and second order VI (adaptive lambda) ----------
fprintf('== VI first-order backward pass ==\n');
[V0, V1, X_vi1, C_vi1, STD_pvi, STD_cvi, T_vi] = ...
    estimators.vi_first_order(Sigma_em, lambda_t, cfg, T_f);

fprintf('== VI second-order backward pass ==\n');
[~, ~, V2, X_vi2, C_vi2, STD_pvi2, STD_cvi2, T_vi2] = ...
    estimators.vi_second_order(Sigma_em, lambda_t, cfg, T_f);

%% ---------- 7. MSE + timing comparison ----------
%   Timings follow the convention T_* returned by each estimator:
%     T_f, T_b are for const-lambda FF / BS.
%     T_em    is wall-clock of the adaptive-lambda EM (outer loop).
%     T_vi and T_vi2 are FF + VI (const-lambda), not inclusive of EM cost;
%     we add T_em to get the "total cost when running on adaptive-lambda".
T_timing = struct();
T_timing.forward_filter   = T_f;
T_timing.backward_sampler = T_b;
T_timing.em               = T_em;
T_timing.vi_first_total   = T_em + T_vi;
T_timing.vi_second_total  = T_em + T_vi2;

fprintf('\n== MSE in dB (lower is better)   and   wall-clock time (s) ==\n');
methods = {
    'forward filter',    X_f,   C_f,   T_timing.forward_filter;
    'backward sampler',  X_b,   C_b,   T_timing.backward_sampler;
    'VI first order',    X_vi1, C_vi1, T_timing.vi_first_total;
    'VI second order',   X_vi2, C_vi2, T_timing.vi_second_total;
};

mse_table_prec = struct([]);
mse_table_cov  = struct([]);
fprintf('%-18s  %-38s  %-38s  %-8s\n', '', 'precision  [total / off-diag] dB', ...
        'covariance [total / off-diag] dB', 'time s');
for m = 1:size(methods, 1)
    name = methods{m, 1};
    Xe   = methods{m, 2};
    Ce   = methods{m, 3};
    tm   = methods{m, 4};

    [mse_p_tot, mse_p_off] = metrics.mse_db(X_true, Xe);
    [mse_c_tot, mse_c_off] = metrics.mse_db(C_true, Ce);

    fprintf('%-18s  %8.2f / %8.2f                %8.2f / %8.2f                %7.3f\n', ...
            name, mse_p_tot, mse_p_off, mse_c_tot, mse_c_off, tm);

    mse_table_prec(end+1).name    = name; %#ok<*SAGROW>
    mse_table_prec(end  ).mse_tot = mse_p_tot;
    mse_table_prec(end  ).mse_off = mse_p_off;
    mse_table_prec(end  ).time_s  = tm;

    mse_table_cov (end+1).name    = name;
    mse_table_cov (end  ).mse_tot = mse_c_tot;
    mse_table_cov (end  ).mse_off = mse_c_off;
    mse_table_cov (end  ).time_s  = tm;
end
fprintf('(EM wall-clock alone = %.3f s, included in VI rows above)\n', T_em);

%% ---------- 8. plots: everything in one tiled figure ----------
methods_struct(1) = struct('name','Forward Filter',   'X_p', X_f,   'STD_p', STD_pf, ...
                           'X_c', C_f,   'STD_c', STD_cf);
methods_struct(2) = struct('name','Backward Sampler', 'X_p', X_b,   'STD_p', STD_pb, ...
                           'X_c', C_b,   'STD_c', STD_cb);
methods_struct(3) = struct('name','VI 1st order',     'X_p', X_vi1, 'STD_p', STD_pvi, ...
                           'X_c', C_vi1, 'STD_c', STD_cvi);
methods_struct(4) = struct('name','VI 2nd order',     'X_p', X_vi2, 'STD_p', STD_pvi2, ...
                           'X_c', C_vi2, 'STD_c', STD_cvi2);

plotting.grid_all(t_axis, X_true, C_true, methods_struct, ...
                  lambda_t, mse_table_prec, mse_table_cov, cfg);

%% ---------- save a summary .mat for later comparison ----------
ts = datestr(now, 'yyyymmdd_HHMMSS');
save(sprintf('results_%s.mat', ts), ...
     'cfg', 'C_true', 'X_true', 'Y', ...
     'Sigma_f', 'X_f', 'C_f', 'STD_pf', 'STD_cf', ...
     'X_b', 'C_b', 'STD_pb', 'STD_cb', ...
     'Sigma_em', 'lambda_t', 'z_smooth', 'rho', 'sigma2_eps', 'em_hist', ...
     'V0', 'V1', 'V2', ...
     'X_vi1', 'C_vi1', 'STD_pvi',  'STD_cvi', ...
     'X_vi2', 'C_vi2', 'STD_pvi2', 'STD_cvi2', ...
     'mse_table_prec', 'mse_table_cov', 'T_timing');

fprintf('\nResults saved to results_%s.mat\n', ts);
