function entry_with_ci(t, X_true, estimates, cfg, target)
%ENTRY_WITH_CI  Plot the (i,j) entry of a matrix trajectory over time for
%               multiple estimators, with shaded confidence bands.
%
%   estimates is a struct array with fields
%       .name   : legend label
%       .X      : [T x p x p]   estimate of precision or covariance (see target)
%       .STD    : [1 x T]       std of the (i,j) entry
%
%   target = 'precision' or 'covariance' (purely for axis labels/title)

    if nargin < 5, target = 'precision'; end
    i = cfg.plot.idx(1);
    j = cfg.plot.idx(2);

    figure('Color','w'); hold on;
    plot(t, squeeze(X_true(:, i, j)), 'k-', 'LineWidth', 2.0, 'DisplayName', 'ground truth');

    colors = lines(numel(estimates));
    for e = 1:numel(estimates)
        est = estimates(e);
        mu  = squeeze(est.X(:, i, j));
        sd  = est.STD(:);
        if numel(sd) == numel(mu)
            fill([t(:); flipud(t(:))], [mu - sd; flipud(mu + sd)], colors(e,:), ...
                 'FaceAlpha', 0.15, 'EdgeColor', 'none', 'HandleVisibility','off');
        end
        plot(t, mu, '-', 'Color', colors(e,:), 'LineWidth', 1.5, 'DisplayName', est.name);
    end

    xlabel('time t');
    ylabel(sprintf('%s_{%d,%d}(t)', target, i, j));
    title(sprintf('%s entry (%d,%d)  — truth vs. estimates', target, i, j));
    legend('Location','best'); grid on;
end
