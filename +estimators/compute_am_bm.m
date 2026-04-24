function [a_m, b_m] = compute_am_bm(m, p)
%COMPUTE_AM_BM  Second-order-VI scalar coefficients for a p-dimensional
%               inverse-Wishart with m degrees of freedom.
%
%   Derived from the Isserlis / Wishart moment identities used in the
%   second-order posterior correction. The closed-form values below
%   match the ones assumed by the VI second-order backward pass in
%   Master.m.
%
%   a_m = 1 / ((m - p - 1) * (m - p - 3))
%   b_m = 1 / ((m - p) * (m - p - 1) * (m - p - 3))
%
%   These require m > p + 3 for finite second moments.

    if m <= p + 3
        error('compute_am_bm: need m > p + 3 for finite second moments (m=%d, p=%d).', m, p);
    end
    a_m = 1 / ((m - p - 1) * (m - p - 3));
    b_m = 1 / ((m - p) * (m - p - 1) * (m - p - 3));
end
