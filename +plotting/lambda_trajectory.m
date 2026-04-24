function lambda_trajectory(lambda_t, cfg)
%LAMBDA_TRAJECTORY  Plot the estimated dynamic smoothing dial lambda_t.

    T = numel(lambda_t);
    t = 1:T;
    lb = cfg.smoother.lambda_base;

    figure('Color','w');
    subplot(2,1,1);
    plot(t, lambda_t, 'b-', 'LineWidth', 1.5); hold on;
    plot(t, lb * ones(1,T), 'r--', 'LineWidth', 1);
    xlabel('time t'); ylabel('\lambda_t');
    title('Dynamic smoothing dial');
    legend('\lambda_t (adaptive)', sprintf('\\lambda_{base}=%.2f', lb), 'Location','best');
    grid on;

    subplot(2,1,2);
    omega = lambda_t / lb;
    plot(t, omega, 'b-', 'LineWidth', 1.5); hold on;
    plot(t, ones(1,T), 'r--', 'LineWidth', 1);
    xlabel('time t'); ylabel('\omega_t');
    title('\omega_t = \lambda_t / \lambda_{base}');
    grid on;
end
