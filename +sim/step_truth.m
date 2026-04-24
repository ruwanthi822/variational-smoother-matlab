function C = step_truth(cfg)
%STEP_TRUTH  Build a step-change ground-truth covariance trajectory C(1:T, :, :).
%
%   C = sim.step_truth(cfg)
%
%   Produces a T x p x p trajectory with constant diagonals and a brief
%   off-diagonal deviation in the middle of the trial.

    p  = cfg.sim.p;
    T  = cfg.sim.T;
    i  = cfg.plot.idx(1);
    j  = cfg.plot.idx(2);

    C = zeros(T, p, p);
    for t = 1:T
        M = eye(p);
        for d = 1:p, M(d,d) = 8; end
        C(t,:,:) = M;
    end

    % brief pulse in the (i,j) off-diagonal
    pulse_start = round(T*0.475);
    pulse_end   = round(T*0.517);
    for t = pulse_start:pulse_end
        C(t, i, j) = 7;
        C(t, j, i) = 7;
    end

    % sanity: ensure all slices are PSD
    for t = 1:T
        Ct = squeeze(C(t,:,:));
        if min(eig(Ct)) <= 0
            C(t,:,:) = utils.nearest_psd(Ct, 1e-6);
        end
    end
end
