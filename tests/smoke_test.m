function smoke_test()
%SMOKE_TEST  Minimal sanity test — small T, small p, no plots. Verifies
%            the pipeline runs end-to-end and produces finite MSE.

    fprintf('-- smoke_test: running mini pipeline --\n');

    cfg = config.default();
    cfg.sim.p    = 5;
    cfg.sim.T    = 120;
    cfg.sim.seed = 1;
    cfg.em.max_iter    = 5;
    cfg.vi.VI_iter     = 1;
    cfg.em.verbose     = false;
    cfg.bsampler.L     = 3;

    C_true = sim.step_truth(cfg);
    [Y, X_true] = sim.sample_wishart(C_true, cfg);

    lam_const = cfg.smoother.lambda_base * ones(1, cfg.sim.T);
    [Sigma_f, X_f, C_f] = estimators.forward_filter(Y, lam_const, cfg);
    [X_b, C_b]          = estimators.backward_sampler(Sigma_f, lam_const, cfg, 0);

    [Sigma_em, lambda_t] = adaptive.run_em(Y, Sigma_f, cfg);
    [~, ~, X_vi1, C_vi1] = estimators.vi_first_order(Sigma_em, lambda_t, cfg, 0);
    [~, ~, ~, X_vi2, C_vi2] = estimators.vi_second_order(Sigma_em, lambda_t, cfg, 0);

    methods_X = {X_f, X_b, X_vi1, X_vi2};
    methods_C = {C_f, C_b, C_vi1, C_vi2};
    names     = {'FF', 'BS', 'VI1', 'VI2'};

    all_ok = true;
    for m = 1:numel(methods_X)
        [mse_p, ~] = metrics.mse_db(X_true, methods_X{m});
        [mse_c, ~] = metrics.mse_db(C_true, methods_C{m});
        ok = isfinite(mse_p) && isfinite(mse_c);
        all_ok = all_ok && ok;
        fprintf('  %3s:  MSE_X=%7.2f dB   MSE_C=%7.2f dB   %s\n', ...
                names{m}, mse_p, mse_c, tern(ok, 'OK', 'FAIL'));
    end

    if all_ok
        fprintf('-- smoke_test: PASS --\n');
    else
        error('smoke_test: FAIL');
    end
end

function s = tern(cond, a, b)
    if cond, s = a; else, s = b; end
end
