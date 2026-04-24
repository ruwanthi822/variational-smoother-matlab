function X = cov_to_prec(C, k, lambda)
%COV_TO_PREC  Convert a covariance estimate to a precision estimate with
%             MVSV (mean-value scaling) based on (k, lambda).
%
%   X = utils.cov_to_prec(C, k, lambda)
%
%   Inputs
%       C      : [p x p]  or  [T x p x p]
%       k      : scalar (Wishart d.o.f.)
%       lambda : scalar or length-T vector
%
%   Output
%       X : same shape as C, with X_t = (C_t^{-1}) / (k*(1-lambda(t)))

    nd = ndims(C);
    sz = size(C);

    if nd == 2
        p  = sz(1);
        if ~isscalar(lambda)
            error('cov_to_prec: 2D input needs scalar lambda.');
        end
        X = (C \ eye(p)) / (k * (1 - lambda));
        return;
    end

    if nd ~= 3 || sz(2) ~= sz(3)
        error('cov_to_prec: expected [p x p] or [T x p x p]. Got %s.', mat2str(sz));
    end

    T = sz(1);  p = sz(2);
    if isscalar(lambda)
        lam = lambda * ones(1, T);
    else
        lam = lambda(:)';
        if numel(lam) ~= T
            error('cov_to_prec: length(lambda)=%d does not match T=%d.', numel(lam), T);
        end
    end

    X = zeros(size(C));
    for t = 1:T
        Ct = squeeze(C(t,:,:));
        X(t,:,:) = (Ct \ eye(p)) / (k * (1 - lam(t)));
    end
end
