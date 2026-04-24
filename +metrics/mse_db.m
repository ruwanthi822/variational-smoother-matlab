function [Tot_MSE, offTot_MSE] = mse_db(X_true, X_est)
%MSE_DB  Relative MSE in dB between a ground-truth trajectory X_true and an
%        estimate X_est, reported both for the full matrix and for the
%        off-diagonal entries only.
%
%   Both inputs are [T x p x p]. Output is
%
%       10 * log10( sum_t || X_true(t) - X_est(t) ||_F^2 / sum_t || X_true(t) ||_F^2 )
%
%   (using the square of the entry-wise L2 norm, which matches Frobenius^2).

    T = size(X_true, 1);
    p = size(X_true, 3);

    num = 0; den = 0;
    num_off = 0; den_off = 0;
    off_mask = 1 - eye(p);
    for t = 1:T
        A  = squeeze(X_true(t,:,:));
        Ae = squeeze(X_est (t,:,:));
        num = num + sum((A - Ae).^2, 'all');
        den = den + sum(A.^2, 'all');

        Ao  = A  .* off_mask;
        Aoe = Ae .* off_mask;
        num_off = num_off + sum((Ao - Aoe).^2, 'all');
        den_off = den_off + sum(Ao.^2, 'all');
    end
    Tot_MSE    = 10 * log10(num / max(den, eps));
    offTot_MSE = 10 * log10(num_off / max(den_off, eps));
end
