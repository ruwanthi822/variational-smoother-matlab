function C = linear_truth(cfg)
%LINEAR_TRUTH  Ground-truth covariance trajectory with a linear transition
%              between two random sparse matrices C1 -> C2 across [T/2].
%
%   C = sim.linear_truth(cfg)

    p    = cfg.sim.p;
    T    = cfg.sim.T;
    T_tr = cfg.sim.T_tr;
    perc = cfg.sim.perc;

    if cfg.sim.seed > 0
        rng(cfg.sim.seed);
    end

    C1 = zeros(p);
    C2 = zeros(p);
    for i = 2:p
        for j = 1:i-1
            f1 = binornd(1, perc*2 / p / (p-1));
            f2 = binornd(1, perc*2 / p / (p-1));
            if f1, v = 2*binornd(1,0.5) - 1; C1(i,j) = v; C1(j,i) = v; end
            if f2, v = 2*binornd(1,0.5) - 1; C2(i,j) = v; C2(j,i) = v; end
        end
    end

    % make positive definite
    C1 = C1 - 12*min(eig(C1))*eye(p) + 0.1*eye(p);
    C2 = C2 - 12*min(eig(C2))*eye(p) + 0.1*eye(p);

    C = zeros(T, p, p);
    half = (T - T_tr) / 2;

    for t = 1:half
        C(t,:,:)                = C1;
        C(t + half + T_tr, :, :) = C2;
    end
    for t = 1:T_tr
        alpha = t / T_tr;
        C(t + half, :, :) = alpha * C2 + (1 - alpha) * C1;
    end

    % pad to length T in case rounding left a last slice empty
    for t = 1:T
        if all(C(t,:,:) == 0, 'all')
            C(t,:,:) = C2;
        end
    end
end
