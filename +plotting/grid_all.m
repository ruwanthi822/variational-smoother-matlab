function grid_all(t, X_true, C_true, methods, lambda_t, mse_prec, mse_cov, cfg)
%GRID_ALL  Render every diagnostic on a single figure as a tiled layout.
%
%   Rows 1..N_methods:   precision (left) and covariance (right) with +/-1
%                        STD confidence bands for each estimator.
%   Row N+1:             adaptive lambda_t trajectory (spans both columns).
%   Row N+2:             MSE bar charts (precision left, covariance right).
%
%   Inputs
%       t        : [1 x T]
%       X_true   : [T x p x p] ground-truth precision
%       C_true   : [T x p x p] ground-truth covariance
%       methods  : 1 x N struct with fields .name .X_p .STD_p .X_c .STD_c
%       lambda_t : [1 x T]
%       mse_prec : struct array (.name .mse_tot .mse_off)
%       mse_cov  : struct array (.name .mse_tot .mse_off)
%       cfg      : config struct (uses cfg.plot.idx, cfg.smoother.lambda_base)

    i = cfg.plot.idx(1);
    j = cfg.plot.idx(2);
    N = numel(methods);

    % total rows: N (methods) + 1 (lambda) + 1 (MSE bars)
    n_rows = N + 2;
    figure('Color','w', 'Name','Variational smoother — diagnostics', ...
           'Position',[80 60 1200 1000]);
    tl = tiledlayout(n_rows, 2, 'TileSpacing','compact', 'Padding','compact');

    % make shaded bands clearly visible:
    %   FaceAlpha = 0.35, estimate line in a darker tint, no band edges.
    prec_color = [0.10 0.35 0.80];
    cov_color  = [0.80 0.30 0.15];
    band_alpha = 0.35;

    % ---- rows 1..N: precision and covariance per method ----
    for m = 1:N
        mt = methods(m);

        % precision (left column)
        nexttile;
        mu = squeeze(mt.X_p(:, i, j));
        sd = mt.STD_p(:);
        hold on;
        if numel(sd) == numel(mu)
            fill([t(:); flipud(t(:))], [mu - sd; flipud(mu + sd)], prec_color, ...
                 'FaceAlpha', band_alpha, 'EdgeColor','none', ...
                 'DisplayName','\pm 1 STD');
        end
        plot(t, squeeze(X_true(:, i, j)), 'k-', 'LineWidth', 1.6, 'DisplayName','truth');
        plot(t, mu, '-', 'Color', prec_color * 0.7, 'LineWidth', 1.7, ...
             'DisplayName','estimate');
        ylabel(sprintf('X_{%d,%d}(t)', i, j));
        if m == 1, title('precision'); end
        if m == N, xlabel('time t'); end
        grid on;
        text(0.02, 0.95, mt.name, 'Units','normalized', ...
             'FontWeight','bold', 'FontSize', 11, ...
             'VerticalAlignment','top', ...
             'BackgroundColor',[1 1 1 0.7]);
        if m == 1, legend('Location','best'); end

        % covariance (right column)
        nexttile;
        mu = squeeze(mt.X_c(:, i, j));
        sd = mt.STD_c(:);
        hold on;
        if numel(sd) == numel(mu)
            fill([t(:); flipud(t(:))], [mu - sd; flipud(mu + sd)], cov_color, ...
                 'FaceAlpha', band_alpha, 'EdgeColor','none', ...
                 'DisplayName','\pm 1 STD');
        end
        plot(t, squeeze(C_true(:, i, j)), 'k-', 'LineWidth', 1.6, 'DisplayName','truth');
        plot(t, mu, '-', 'Color', cov_color * 0.7, 'LineWidth', 1.7, ...
             'DisplayName','estimate');
        ylabel(sprintf('C_{%d,%d}(t)', i, j));
        if m == 1, title('covariance'); end
        if m == N, xlabel('time t'); end
        grid on;
        text(0.02, 0.95, mt.name, 'Units','normalized', ...
             'FontWeight','bold', 'FontSize', 11, ...
             'VerticalAlignment','top', ...
             'BackgroundColor',[1 1 1 0.7]);
        if m == 1, legend('Location','best'); end
    end

    % ---- row N+1: lambda trajectory (spans both columns) ----
    ax_lam = nexttile([1 2]);
    T = numel(lambda_t);
    plot(ax_lam, 1:T, lambda_t, 'b-', 'LineWidth', 1.6); hold(ax_lam, 'on');
    plot(ax_lam, 1:T, cfg.smoother.lambda_base * ones(1, T), ...
         'r--', 'LineWidth', 1);
    xlabel(ax_lam, 'time t'); ylabel(ax_lam, '\lambda_t');
    title(ax_lam, 'adaptive smoothing dial \lambda_t');
    legend(ax_lam, '\lambda_t (adaptive)', ...
           sprintf('\\lambda_{base}=%.2f', cfg.smoother.lambda_base), ...
           'Location','best');
    grid(ax_lam, 'on');

    % ---- row N+2: MSE bars (precision left, covariance right) ----
    labels = {mse_prec.name};
    mse_p  = [[mse_prec.mse_tot]', [mse_prec.mse_off]'];
    mse_c  = [[mse_cov .mse_tot]', [mse_cov .mse_off]'];

    nexttile;
    bar(mse_p);
    set(gca, 'XTickLabel', labels, 'XTick', 1:numel(labels));
    ylabel('MSE (dB)');
    title('precision MSE');
    legend('total','off-diagonal','Location','best');
    grid on;

    nexttile;
    bar(mse_c);
    set(gca, 'XTickLabel', labels, 'XTick', 1:numel(labels));
    ylabel('MSE (dB)');
    title('covariance MSE');
    legend('total','off-diagonal','Location','best');
    grid on;

    sgtitle('Variational smoother — full diagnostics');
end
