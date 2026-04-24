function entry_single(t, X_true, C_true, est_prec, est_cov, cfg, name, extra_title)
%ENTRY_SINGLE  Two-panel figure showing a single method's precision and
%              covariance estimates for the (i,j) entry, with ground truth
%              and a shaded +/-1 STD confidence band.
%
%   t         : [1 x T]
%   X_true    : [T x p x p] ground-truth precision
%   C_true    : [T x p x p] ground-truth covariance
%   est_prec  : struct with fields .X  [T x p x p], .STD [1 x T]
%   est_cov   : struct with fields .X  [T x p x p], .STD [1 x T]
%   cfg       : config struct (uses cfg.plot.idx)
%   name      : char, method label (e.g. 'Forward Filter')
%   extra_title : (optional) extra text in the super-title

    if nargin < 8, extra_title = ''; end
    i = cfg.plot.idx(1);
    j = cfg.plot.idx(2);

    figure('Color','w','Name',name);
    tl = tiledlayout(2, 1, 'TileSpacing','compact', 'Padding','compact');

    % --- precision panel ---
    nexttile; hold on;
    mu = squeeze(est_prec.X(:, i, j));
    sd = est_prec.STD(:);
    if numel(sd) == numel(mu)
        fill([t(:); flipud(t(:))], [mu - sd; flipud(mu + sd)], ...
             [0.2 0.4 0.8], 'FaceAlpha', 0.18, 'EdgeColor','none', ...
             'DisplayName','\pm 1 STD');
    end
    plot(t, squeeze(X_true(:, i, j)), 'k-', 'LineWidth', 1.8, 'DisplayName','truth');
    plot(t, mu,                       '-',  'Color',[0.1 0.3 0.7], ...
         'LineWidth', 1.5, 'DisplayName','estimate');
    xlabel('time t'); ylabel(sprintf('X_{%d,%d}(t)', i, j));
    title('precision'); legend('Location','best'); grid on;

    % --- covariance panel ---
    nexttile; hold on;
    mu = squeeze(est_cov.X(:, i, j));
    sd = est_cov.STD(:);
    if numel(sd) == numel(mu)
        fill([t(:); flipud(t(:))], [mu - sd; flipud(mu + sd)], ...
             [0.8 0.3 0.2], 'FaceAlpha', 0.18, 'EdgeColor','none', ...
             'DisplayName','\pm 1 STD');
    end
    plot(t, squeeze(C_true(:, i, j)), 'k-', 'LineWidth', 1.8, 'DisplayName','truth');
    plot(t, mu,                       '-',  'Color',[0.7 0.2 0.1], ...
         'LineWidth', 1.5, 'DisplayName','estimate');
    xlabel('time t'); ylabel(sprintf('C_{%d,%d}(t)', i, j));
    title('covariance'); legend('Location','best'); grid on;

    if isempty(extra_title)
        sgtitle(name);
    else
        sgtitle(sprintf('%s   %s', name, extra_title));
    end
end
