function mse_summary(mse_table)
%MSE_SUMMARY  Bar chart of total and off-diagonal MSE (dB) across methods.
%
%   mse_table is a struct array with fields
%       .name    : method label
%       .mse_tot : total MSE in dB
%       .mse_off : off-diagonal MSE in dB

    n = numel(mse_table);
    labels = {mse_table.name};
    tot    = [mse_table.mse_tot];
    off    = [mse_table.mse_off];

    figure('Color','w');
    bar([tot(:) off(:)]);
    set(gca, 'XTickLabel', labels, 'XTick', 1:n);
    ylabel('MSE (dB, lower is better)');
    legend('total', 'off-diagonal', 'Location','best');
    title('MSE comparison across methods');
    grid on;
end
